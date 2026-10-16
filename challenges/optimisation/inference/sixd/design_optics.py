"""Quadrupole settings that make the screens after the TDC imageable (beam inside the
+-5 mm YAG window) for the 6D reconstruction at the S02 start.

On nominal optics only the S02 to S05 screens are imageable; from S06 on the beam is 10 to
40 mm rms in y. Two kinds of setting are designed here, one per target screen (each screen
is a separate destructive shot anyway):

- streak screens (non-dispersive, after the TDC): small beam with the TDC off and a large
  streak with the TDC on, so the y profile resolves tau;
- spectrometer screens (FEA dogleg, dispersive): the TDC streaks in y and the dispersion
  spreads delta in x, so the image is the (tau, delta) plane. Small betatron size in x for
  energy resolution; the delta extent may clip at the window edge (the reconstruction uses
  clipped images as they are).

Optimisation on linear maps from the S02 start (xsuite test particles, TDC included) with
the nominal design beam covariance, then checked by tracking. Currents within +-100 A.
"""

from __future__ import annotations

import sys

import common
import full6d
import numpy as np
from scipy.optimize import differential_evolution

cv = common.cv
OUT = common.HERE / "optics_design.json"
WINDOW_RMS_M = 1.6e-3  # 3 sigma inside the 5 mm half width

# target screen -> (kind, quads that may change); only magnets downstream of S05 are used,
# so the S02 to S05 screens are unaffected
UPSTREAM = ["cla_s06_mag_quad_01", "cla_s06_mag_quad_02", "cla_s07_mag_quad_01", "cla_s07_mag_quad_02"]
TARGETS = {
    "cla_s07_dia_scr_02": ("streak", UPSTREAM + ["cla_s07_mag_quad_03", "cla_s07_mag_quad_04"]),
    "cla_s07_dcp_scr_01": ("streak", UPSTREAM + [f"cla_s07_mag_quad_0{i}" for i in range(3, 7)]),
    "cla_s07_dia_scr_03": ("streak", UPSTREAM + [f"cla_s07_mag_quad_0{i}" for i in range(3, 8)]),
}
# designed after the others, starting from the cla_s07_dcp_scr_01 setting (beam small at
# 40 m) and changing only the magnets after it: a global search over all S06/S07 quads
# finds linear optima that blow up when tracked (chromatic, sigma_delta ~2 %)
SEQUENTIAL = {
    "cla_s07_dia_scr_04": ("streak", "cla_s07_dcp_scr_01", ["cla_s07_mag_quad_07", "cla_s07_mag_quad_08", "cla_s07_mag_quad_09"]),
    "cla_fea_dia_scr_01": (
        "spectrometer",
        "cla_s07_dcp_scr_01",
        ["cla_s07_mag_quad_07", "cla_s07_mag_quad_08", "cla_s07_mag_quad_09", "cla_s07_mag_quad_10",
         "cla_fea_mag_quad_01", "cla_fea_mag_quad_02"],
    ),
}


def beam_at(line, sigma0, screen, k1, voltage):
    full6d.apply_config(line, full6d.Config("design", k1, voltage, [screen]))
    m = line.linear_map([screen])[screen]
    return m, m @ sigma0 @ m.T


def tracked_sizes(line, sample, screen, k1, voltage):
    full6d.apply_config(line, full6d.Config("design", k1, voltage, [screen]))
    c = line.track_through(sample, [screen])[screen]
    return c[:, 0].std(), c[:, 2].std()


def score(line, sigma0, screen, kind, k1, sample=None):
    """Linear-map score (sample None) or tracked score (sample = design-beam particles;
    chromatic effects included, which matter here: sigma_delta is 1 to 2 %)."""
    v = full6d.TDC_VOLTAGE_V
    try:
        m0, s0 = beam_at(line, sigma0, screen, k1, 0.0)
        if sample is None:
            _, sv = beam_at(line, sigma0, screen, k1, v)
            sx0, sy0 = np.sqrt(s0[0, 0]), np.sqrt(s0[2, 2])
            sxv, syv = np.sqrt(sv[0, 0]), np.sqrt(sv[2, 2])
        else:
            sx0, sy0 = tracked_sizes(line, sample, screen, k1, 0.0)
            sxv, syv = tracked_sizes(line, sample, screen, k1, v)
    except RuntimeError:  # particles lost
        return 1e6, {}
    streak = np.sqrt(max(syv**2 - sy0**2, 1e-30))
    penalty = 0.0
    if kind == "streak":
        # time resolution: unstreaked y size over the streak; keep everything in the window
        cost = sy0 / streak
        for size in (sx0, sxv, syv):
            penalty += max(size / WINDOW_RMS_M - 1, 0) ** 2
    else:
        # energy resolution: transverse (betatron, chromatic included when tracked) x size
        # over the dispersion; x may clip up to 4 mm rms, y (streaked) must fit
        if sample is None:
            transverse = sigma0.copy()
            transverse[4:, :] = 0
            transverse[:, 4:] = 0
            beta_x = np.sqrt((m0 @ transverse @ m0.T)[0, 0])
        else:
            mono = sample.copy()
            mono[:, 5] = sample[:, 4] * sigma0[4, 5] / sigma0[4, 4]  # remove slice spread
            try:
                bx, _ = tracked_sizes(line, mono, screen, k1, 0.0)
            except RuntimeError:
                return 1e6, {}
            # x size of the chirp-only beam minus its dispersive part, in quadrature
            disp_part = abs(m0[0, 5]) * np.sqrt(sigma0[4, 5] ** 2 / sigma0[4, 4])
            beta_x = np.sqrt(max(bx**2 - disp_part**2, (0.05 * bx) ** 2))
        dispersion = abs(m0[0, 5])
        # the chirp is read from the tilt of the (tau, delta) image, so time resolution
        # comes first; the dispersion only has to stay visible (|R16| >= 0.01 m from S02)
        cost = sy0 / streak + 0.05 * beta_x / max(dispersion, 1e-9)
        penalty += max(0.01 / max(dispersion, 1e-9) - 1, 0) ** 2
        penalty += max(sxv / 4e-3 - 1, 0) ** 2 + max(syv / WINDOW_RMS_M - 1, 0) ** 2
    return cost + 100 * penalty, dict(
        sx0=float(sx0), sy0=float(sy0), sxv=float(sxv), syv=float(syv), streak=float(streak),
        r16=float(m0[0, 5]),
    )


def design(line, sigma0, screen, sample, seed=0, maxiter=40, base_k1=None):
    """Global search on linear maps, then a local search on tracked sizes. `base_k1`: fixed
    settings of other magnets (sequential designs)."""
    from scipy.optimize import minimize

    base_k1 = dict(base_k1 or {})
    if screen in TARGETS:
        kind, quads = TARGETS[screen]
    else:
        kind, _, quads = SEQUENTIAL[screen]
    bounds = [line.quad_k1_limits(q) for q in quads]
    bounds = [(min(b), max(b)) for b in bounds]
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])

    def f_linear(x):
        return score(line, sigma0, screen, kind, {**base_k1, **dict(zip(quads, x))})[0]

    def f_tracked(u):
        x = lo + (hi - lo) * np.clip(u, 0, 1)
        return score(line, sigma0, screen, kind, {**base_k1, **dict(zip(quads, x))}, sample)[0]

    x0 = np.clip(np.array([base_k1.get(q, line.nominal_k1[q]) for q in quads]), lo, hi)
    coarse = differential_evolution(
        f_linear, bounds, seed=seed, maxiter=maxiter, popsize=10, tol=1e-6, polish=False, x0=x0
    )
    # local tracked refinement from the best linear and the best of a few population members
    starts = [coarse.x] + [p for p in coarse.population[np.argsort(coarse.population_energies)[1:2]]]
    best = None
    for start in starts:
        res = minimize(f_tracked, (start - lo) / (hi - lo), method="Powell",
                       bounds=[(0, 1)] * len(quads), options={"maxfev": 300, "xtol": 1e-3})
        if best is None or res.fun < best.fun:
            best = res
    x = lo + (hi - lo) * np.clip(best.x, 0, 1)
    if best.fun >= 1e6 and coarse.fun < 1e6:
        x = coarse.x
    k1 = {**base_k1, **dict(zip(quads, x))}
    _, info = score(line, sigma0, screen, kind, k1, sample)
    currents = {q: line.quad_current_from_k1(q, k) for q, k in k1.items()}
    line.restore_nominal_quads()
    full6d.set_tdc_voltage(line, 0.0)
    return {"kind": kind, "k1": k1, "currents_a": currents, "cost": float(best.fun),
            "cost_linear": float(coarse.fun), **info}


ALL_AFTER_S05 = UPSTREAM + [f"cla_s07_mag_quad_{i:02d}" for i in range(3, 11)] + [
    "cla_fea_mag_quad_01", "cla_fea_mag_quad_02"]


def refine_all(line, sigma0, screen, sample, start_k1, quads=ALL_AFTER_S05, scale=0.05, step=0.03, seed=0,
               maxfev=1500):
    """Tracked local search over all quads after S05 from `start_k1`, starting from a random
    perturbation of relative size `scale` of each quad range. The sequential design keeps the
    magnets up to the TDC at the dcp setting, which fixes beta_y at the TDC and therefore the
    best possible time resolution; this lets them move."""
    from scipy.optimize import minimize

    kind = TARGETS[screen][0] if screen in TARGETS else SEQUENTIAL[screen][0]
    bounds = [line.quad_k1_limits(q) for q in quads]
    lo, hi = np.array([min(b) for b in bounds]), np.array([max(b) for b in bounds])
    x0 = np.array([start_k1.get(q, line.nominal_k1[q]) for q in quads])
    u0 = np.clip((x0 - lo) / (hi - lo) + scale * np.random.default_rng(seed).normal(size=len(quads)) * (seed > 0), 0, 1)

    def f(u):
        x = lo + (hi - lo) * np.clip(u, 0, 1)
        return score(line, sigma0, screen, kind, {**start_k1, **dict(zip(quads, x))}, sample)[0]

    # Nelder-Mead from a small simplex: Powell's line searches span the whole range, where
    # every point loses particles (flat 1e6 cost)
    simplex = np.vstack([u0] + [u0 + step * np.eye(len(quads))[i] for i in range(len(quads))])
    res = minimize(f, u0, method="Nelder-Mead",
                   options={"maxfev": maxfev, "initial_simplex": simplex, "xatol": 1e-4, "fatol": 1e-4})
    k1 = {**start_k1, **dict(zip(quads, lo + (hi - lo) * np.clip(res.x, 0, 1)))}
    _, info = score(line, sigma0, screen, kind, k1, sample)
    currents = {q: line.quad_current_from_k1(q, k) for q, k in k1.items()}
    line.restore_nominal_quads()
    full6d.set_tdc_voltage(line, 0.0)
    return {"kind": kind, "k1": k1, "currents_a": currents, "cost": float(res.fun), "seed": seed, **info}


def _refine_one(args):
    screen, start_k1, seed = args
    beam, _, _ = common.xsuite_model.load_distribution()
    res = refine_all(_LINE, np.cov(beam.T), screen, beam[::200][:1000], start_k1, seed=seed)
    res["tracked"] = check_tracked(_LINE, beam[:20000], screen, res["k1"])
    return res


def run_refine(line, screen, seeds=(0, 1, 2, 3), workers=4):
    """Refine one existing design over all quads after S05 (forked restarts); keep the best."""
    import multiprocessing as mp

    global _LINE
    _LINE = line
    results = common.read_json(OUT)
    start = results[screen]["k1"]
    with mp.get_context("fork").Pool(workers) as pool:
        runs = pool.map(_refine_one, [(screen, start, s) for s in seeds])
    for r in runs:
        if "sy0" not in r:
            print(r["seed"], "failed", flush=True)
            continue
        print(r["seed"], round(r["cost"], 4), {k: round(r[k] * 1e3, 3) for k in ("sy0", "streak", "sxv", "syv")},
              "r16", round(r["r16"], 4), "inside", r["tracked"].get("off", {}).get("inside"), flush=True)
    best = min(runs, key=lambda r: r["cost"])
    if best["cost"] < results[screen]["cost"]:
        results[screen] = best
        common.write_json(OUT, results, indent=1)
    report({screen: results[screen]})
    return runs


def check_tracked(line, beam, screen, k1):
    """rms sizes and window fraction from tracking the design beam, TDC off/+/-."""
    out = {}
    for label, v in (("off", 0.0), ("plus", full6d.TDC_VOLTAGE_V), ("minus", -full6d.TDC_VOLTAGE_V)):
        full6d.apply_config(line, full6d.Config("check", k1, v, [screen]))
        c = line.track_through(beam, [screen], allow_losses=True)[screen]
        out[label + "_alive"] = len(c) / len(beam)
        if len(c) == 0:
            continue
        inside = (np.abs(c[:, 0]) < full6d.FIELD_OF_VIEW_M) & (np.abs(c[:, 2]) < full6d.FIELD_OF_VIEW_M)
        out[label] = dict(sx=float(c[:, 0].std()), sy=float(c[:, 2].std()),
                          inside=float(inside.sum() / len(beam)))
    line.restore_nominal_quads()
    full6d.set_tdc_voltage(line, 0.0)
    return out


_LINE = None


def _design_one(args):
    screen, base_k1 = args
    beam, _, _ = common.xsuite_model.load_distribution()
    sigma0 = np.cov(beam.T)  # nominal design beam (the optics designer's input)
    res = design(_LINE, sigma0, screen, beam[::200][:1000], base_k1=base_k1)
    res["tracked"] = check_tracked(_LINE, beam[:20000], screen, res["k1"])
    return screen, res


def report(results):
    for screen, res in results.items():
        print(screen, {k: round(v, 5) for k, v in res.items() if isinstance(v, float)}, flush=True)
        print("   tracked", res["tracked"], flush=True)
        print("   currents", {q: round(i, 2) for q, i in res["currents_a"].items()}, flush=True)


def run(line, screens=None, workers=4, keep_existing=True):
    """Design the target screens (forked workers, in parallel), then the sequential ones."""
    import multiprocessing as mp

    global _LINE
    _LINE = line
    results = common.read_json(OUT) if keep_existing and OUT.exists() else {}
    wanted = list(screens or [*TARGETS, *SEQUENTIAL])
    ctx = mp.get_context("fork")
    first = [(s, None) for s in wanted if s in TARGETS]
    if first:
        with ctx.Pool(workers) as pool:
            results.update(dict(pool.map(_design_one, first)))
    second = [(s, results[SEQUENTIAL[s][1]]["k1"]) for s in wanted if s in SEQUENTIAL]
    if second:
        with ctx.Pool(workers) as pool:
            results.update(dict(pool.map(_design_one, second)))
    report({s: results[s] for s in wanted})
    common.write_json(OUT, results, indent=1)
    return results


if __name__ == "__main__":
    run(full6d.build_line(), sys.argv[1:] or None)
