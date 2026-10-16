"""Polynomial transfer maps from the S02 start to every (config, screen) of the plan.

The forward model of each measurement configuration is fixed, so tracking the reconstruction
sample through xsuite again at every stage is wasted work. Here every (config, screen) map
x(z0), y(z0) is fitted once as a polynomial of total order <= ORDER in the six start
coordinates, from one tracking of a training sample (plus an aperture-loss flag). After that
a "tracking" of N particles through all 68 screens is one matrix product,
monomials(N x terms) @ coefficients(terms x 136), about 1 s for 150k particles instead of
about 3 min on 4 cores.
"""

from __future__ import annotations

import itertools
import time

import common
import full6d
import ment6d
import numpy as np

ORDER = 4
# input scaling (also the training distribution: Gaussian with these rms, which covers the
# broad MENT prior)
SCALE = np.array(ment6d.BROAD_PRIOR_RMS)
OUT = common.HERE / "surrogate_maps.npz"


def exponents(order=ORDER, dim=6):
    return np.array([e for n in range(order + 1) for e in itertools.product(range(n + 1), repeat=dim)
                     if sum(e) == n], dtype=np.int8)


def monomials(z, exps, scale=SCALE):
    u = z / scale
    powers = [np.ones_like(u)]
    for _ in range(int(exps.max())):
        powers.append(powers[-1] * u)
    out = np.empty((len(z), len(exps)))
    for j, e in enumerate(exps):
        col = np.ones(len(z))
        for d in range(z.shape[1]):
            if e[d]:
                col = col * powers[e[d]][:, d]
        out[:, j] = col
    return out


def _track_coords(args):
    """Track the global sample through one configuration; (x, y, alive) per screen."""
    config = args
    line = ment6d._LINE
    sample = ment6d._SAMPLE
    full6d.apply_config(line, config)
    order = sorted(set(config.screens) | set(_EXTRA.get(config.name, ())), key=line.element_index.__getitem__)
    particles = ment6d.xsuite_model._particles(sample, line.converted.initial_energy_ev)
    out, start = {}, 0
    for screen in order:
        stop = line.element_index[screen] + 1
        line.line.track(particles, ele_start=start, ele_stop=stop)
        start = stop
        ids = particles.particle_id
        xy = np.zeros((len(sample), 2))
        alive = np.zeros(len(sample), bool)
        xy[ids, 0], xy[ids, 1] = np.asarray(particles.x), np.asarray(particles.y)
        alive[ids] = particles.state > 0
        out[screen] = (xy, alive)
    line.restore_nominal_quads()
    full6d.set_tdc_voltage(line, 0.0)
    return config.name, out


_EXTRA = {}  # config name -> elements to record besides its screens (aperture checks)
APERTURE_SUFFIXES = ("_pipe", "_col_in", "_col_out")
LOSS_FRACTION = 1e-3  # an aperture losing more of the training sample than this is modelled


def _exit_element(line, name):
    """The line element at the exit of lattice element `name` (a sliced cavity's pieces are
    named `<name>..<piece>`)."""
    if name in line.element_index:
        return name
    pieces = [n for n in line.line.element_names if n.startswith(name + "..")]
    return max(pieces, key=line.element_index.__getitem__)


def aperture_checks(aperture_line, configs, sample, workers=4):
    """{config name: [(element, kind, a, b)]}: the apertures that lose particles of `sample`
    in each configuration, by the element they sit at (coordinates at its exit), with the
    limit's half widths. kind is "rect" or "ellipse"."""
    import collections

    xl = aperture_line.line
    names = xl.element_names
    checks = {}
    for config in configs:
        full6d.apply_config(aperture_line, config)
        last = max(aperture_line.element_index[s] for s in config.screens)
        particles = ment6d.xsuite_model._particles(sample, aperture_line.converted.initial_energy_ev)
        xl.track(particles, ele_start=0, ele_stop=last + 1)
        aperture_line.restore_nominal_quads()
        full6d.set_tdc_voltage(aperture_line, 0.0)
        full6d.set_rf_phases(xl, {})
        lost = np.asarray(particles.state) <= 0
        counts = collections.Counter(names[i] for i in np.asarray(particles.at_element)[lost])
        out = []
        for name, n in counts.items():
            if n < LOSS_FRACTION * len(sample) or not name.endswith(APERTURE_SUFFIXES):
                continue
            element = _exit_element(aperture_line, name.rsplit("_", 2 if name.endswith(("_col_in", "_col_out")) else 1)[0])
            limit = xl[name]
            if hasattr(limit, "a_squ"):
                out.append((element, "ellipse", float(limit.a_squ) ** 0.5, float(limit.b_squ) ** 0.5))
            else:
                out.append((element, "rect", float(limit.max_x), float(limit.max_y)))
        checks[config.name] = sorted(set(out))
    return checks


def track_coords(line, sample, configs, workers=4):
    import multiprocessing as mp

    ment6d._LINE, ment6d._SAMPLE = line, sample
    with mp.get_context("fork").Pool(workers) as pool:
        results = pool.map(_track_coords, configs)
    return {(name, s): v for name, out in results for s, v in out.items()}


class Maps:
    """Fitted polynomial maps for a list of (config, screen) keys."""

    def __init__(self, keys, coef, exps, loss_radius, checks=None, order=None):
        self.checks = checks or {}  # config -> [(element, kind, a, b)], see aperture_checks
        self.order = order or {}  # element -> index along the line, for the checks
        self.keys = [tuple(k) for k in keys]  # every fitted map, screens and aperture checks
        at_checks = {(c, e) for c, v in self.checks.items() for e, *_ in v if "_dia_scr_" not in e}
        self.screen_keys = [k for k in self.keys if k not in at_checks]
        self.coef = coef  # terms x (2 * n_keys): x then y per key
        self.exps = exps
        self.loss_radius = loss_radius  # per key: start radius (in SCALE units) beyond which losses occur

    def coords(self, z, keys=None):
        """{key: (xy, alive)} for start coordinates z (N x 6)."""
        keys = self.screen_keys if keys is None else keys
        extra = sorted({(c, e) for c, _ in keys for e, *_ in self.checks.get(c, ())} - set(keys))
        wanted = list(keys) + extra
        cols = [self.keys.index(k) for k in wanted]
        sel = np.ravel([[2 * i, 2 * i + 1] for i in cols])
        xy = monomials(z, self.exps) @ self.coef[:, sel]
        r = np.sqrt(((z / SCALE) ** 2).sum(1))
        at = {k: xy[:, 2 * j: 2 * j + 2] for j, k in enumerate(wanted)}
        out = {}
        for i, k in zip(cols, keys):
            alive = r < self.loss_radius[i]
            for element, kind, a, b in self.checks.get(k[0], ()):
                if self.order[element] < self.order.get(k[1], np.inf):
                    u, v = at[(k[0], element)].T
                    alive &= ((u / a) ** 2 + (v / b) ** 2 <= 1) if kind == "ellipse" else ((np.abs(u) <= a) & (np.abs(v) <= b))
            out[k] = (at[k], alive)
        return out

    def indices(self, z, rng, keys=None):
        """Pixel index per key (as ment6d.track_all)."""
        keys = self.screen_keys if keys is None else keys
        out = {}
        for k, (xy, alive) in self.coords(z, keys).items():
            c = np.zeros((len(z), 4))
            c[:, 0], c[:, 2] = xy[:, 0], xy[:, 1]
            out[k] = ment6d.pixel_indices(c, alive, rng, k[1])
        return out

    def save(self, path=OUT):
        import json

        np.savez_compressed(path, keys=np.array([f"{c}|{s}" for c, s in self.keys]), coef=self.coef,
                            exps=self.exps, loss_radius=self.loss_radius,
                            checks=json.dumps(self.checks), order=json.dumps(self.order))

    @classmethod
    def load(cls, path=OUT):
        import json

        d = np.load(path)
        checks = json.loads(str(d["checks"])) if "checks" in d else {}
        order = json.loads(str(d["order"])) if "order" in d else {}
        return cls([tuple(k.split("|")) for k in d["keys"]], d["coef"], d["exps"], d["loss_radius"], checks, order)


def fit(line, configs, n_train=60_000, order=ORDER, seed=11, workers=4, log=print, cov=None,
        scale=1.0, aperture_line=None):
    """Fit on a Gaussian training sample: diagonal SCALE * scale, or covariance `cov` (e.g.
    1.5^2 times the current reconstruction's) when given; the maps are accurate where the
    training sample is, which matters on the dispersive FEA screens."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    if cov is None:
        z = rng.normal(0, 1, (n_train, 6)) * SCALE * scale
    else:
        z = rng.multivariate_normal(np.zeros(6), cov, n_train)
    checks = {}
    if aperture_line is not None:
        checks = aperture_checks(aperture_line, configs, z)
        log("aperture checks", {c: [e for e, *_ in v] for c, v in checks.items()})
    _EXTRA.clear()
    _EXTRA.update({c: [e for e, *_ in v] for c, v in checks.items()})
    try:
        tracked = track_coords(line, z, configs, workers)
    finally:
        _EXTRA.clear()
    log(f"tracked {n_train} training particles in {time.time() - t0:.0f} s")
    exps = exponents(order)
    phi = monomials(z, exps)
    r = np.sqrt(((z / SCALE) ** 2).sum(1))
    keys = sorted(tracked)
    coef = np.zeros((len(exps), 2 * len(keys)))
    loss_radius = np.zeros(len(keys))
    for i, k in enumerate(keys):
        xy, alive = tracked[k]
        loss_radius[i] = r[~alive].min() if (~alive).any() else np.inf
        good = alive & (r < loss_radius[i])
        coef[:, 2 * i: 2 * i + 2] = np.linalg.lstsq(phi[good], xy[good], rcond=None)[0]
    log(f"fitted {len(keys)} maps, order {order}, {len(exps)} terms, {time.time() - t0:.0f} s")
    order_index = {e: line.element_index[e] for v in checks.values() for e, *_ in v}
    order_index.update({k[1]: line.element_index[k[1]] for k in keys})
    return Maps(keys, coef, exps, loss_radius, checks, order_index), tracked, z


def validate(maps, line, configs, beam, log=print):
    """Compare surrogate and tracked coordinates on a test beam (inside-window particles)."""
    tracked = track_coords(line, beam, configs)
    pred = maps.coords(beam)
    rows = []
    for k in maps.keys:
        xy, alive = tracked[k]
        pxy, palive = pred[k]
        inside = alive & (np.abs(xy) < full6d.FIELD_OF_VIEW_M).all(1)
        err = (pxy - xy)[inside]
        rows.append(dict(key="|".join(k), inside=float(inside.mean()),
                         rms_err_um=float(np.sqrt((err**2).mean()) * 1e6) if inside.any() else 0.0,
                         p99_err_um=float(np.percentile(np.abs(err), 99) * 1e6) if inside.any() else 0.0,
                         alive_mismatch=float((alive != palive).mean())))
    worst = sorted(rows, key=lambda r: -r["rms_err_um"])[:8]
    for r in worst:
        log(r)
    return rows, tracked


class CombinedMaps:
    """Narrow, high-order maps inside their training domain (radius < `radius` in units of
    their training rms), broad maps outside it (early MENT stages sample the broad prior)."""

    def __init__(self, narrow, broad, narrow_scale, radius=4.0):
        self.narrow, self.broad = narrow, broad
        self.keys = narrow.screen_keys
        self.narrow_scale, self.radius = narrow_scale, radius

    def coords(self, z, keys=None):
        keys = self.keys if keys is None else keys
        inner = np.sqrt(((z / (SCALE * self.narrow_scale)) ** 2).sum(1)) < self.radius
        a = self.narrow.coords(z[inner], keys)
        b = self.broad.coords(z[~inner], keys)
        out = {}
        for k in keys:
            xy, alive = np.empty((len(z), 2)), np.empty(len(z), bool)
            xy[inner], alive[inner] = a[k]
            xy[~inner], alive[~inner] = b[k]
            out[k] = (xy, alive)
        return out

    def indices(self, z, rng, keys=None):
        keys = self.keys if keys is None else keys
        inner = np.sqrt(((z / (SCALE * self.narrow_scale)) ** 2).sum(1)) < self.radius
        a = self.narrow.indices(z[inner], rng, keys)
        b = self.broad.indices(z[~inner], rng, keys) if (~inner).any() else {}
        out = {}
        for k in keys:
            idx = np.empty(len(z), dtype=np.int32)
            idx[inner] = a[k]
            if (~inner).any():
                idx[~inner] = b[k]
            out[k] = idx
        return out


NARROW = dict(order=5, scale=0.5, path=common.HERE / "surrogate_maps_narrow.npz")


def load_combined(tag=""):
    narrow = NARROW["path"].with_name(NARROW["path"].stem + tag + ".npz")
    return CombinedMaps(Maps.load(narrow), Maps.load(OUT.with_name(OUT.stem + tag + ".npz")), NARROW["scale"])


def steering_estimate(path=common.HERE.parent / "steering" / "result_bpm_nominal_seed0.json"):
    """Quad offsets estimated by the BPM steering inference (steering_inference.py)."""
    r = common.read_json(path)
    return {q: tuple(np.array(v["estimate_um"]) * 1e-6) for q, v in r["quad_offsets"]["per_quad"].items()}


def fit_with_offsets(line, configs, quad_offsets, tag, log=print, aperture_line=None, kicks=None):
    """Broad and narrow maps of the model with the given quadrupole offsets (and corrector
    kicks); with `aperture_line` the maps also say which particles its apertures lose."""
    lines = [line] + ([aperture_line] if aperture_line is not None else [])
    for ln in lines:
        for q, (dx, dy) in quad_offsets.items():
            ln.line[q].shift_x, ln.line[q].shift_y = dx, dy
        for c, (h, v) in (kicks or {}).items():
            ln.set_corrector(c, h, v)
    try:
        broad, _, _ = fit(line, configs, log=log, aperture_line=aperture_line)
        broad.save(OUT.with_name(OUT.stem + tag + ".npz"))
        narrow, _, _ = fit(line, configs, order=NARROW["order"], scale=NARROW["scale"], log=log,
                           aperture_line=aperture_line)
        narrow.save(NARROW["path"].with_name(NARROW["path"].stem + tag + ".npz"))
    finally:
        for ln in lines:
            for q in quad_offsets:
                ln.line[q].shift_x, ln.line[q].shift_y = 0.0, 0.0
            for c in kicks or {}:
                ln.set_corrector(c, 0.0, 0.0)
    return load_combined(tag)
