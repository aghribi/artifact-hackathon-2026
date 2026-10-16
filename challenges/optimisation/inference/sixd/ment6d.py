"""Image-based 6D density reconstruction at the S02 start by sample reweighting (MENT).

The density at the start is written as rho(x) = g(x) prod_k h_k(P_k x), where g is a
Gaussian prior, P_k maps a start coordinate to its pixel on image k (xsuite tracking through
the whole line for the configuration of image k, plus the 20 um screen resolution), and h_k
is a per-pixel factor. This is the maximum-entropy density relative to g that reproduces
every image (MENT). With a sample drawn from g, rho is a set of per-particle weights
w_i = prod_k h_k(pixel_k(i)), so each configuration is tracked once per prior and the fit
needs no differentiable model. The h_k are found by multiplicative Gauss-Seidel sweeps over
the images.

Every image is normalised by the bunch charge (assumed known), and the part of the beam
outside the +-5 mm window is one extra bin, so clipped images are used as they are.

Prior: a Gaussian fitted to the image second moments with the model's linear maps,
inflated, then refined by refitting a Gaussian to the weighted sample and redrawing.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time

import common
import full6d
import numpy as np

cv = common.cv
xsuite_model = common.xsuite_model

IMAGES = common.HERE / "images.npz"
REBIN = 2  # 200 px of 50 um -> 100 px of 100 um
PIXELS = full6d.PIXELS // REBIN
PIXEL_M = 2 * full6d.FIELD_OF_VIEW_M / PIXELS
OUTSIDE = PIXELS * PIXELS  # index of the out-of-window bin
LABELS = common.LABELS

# set before forking the tracking workers
_LINE = None
_SAMPLE = None


# ---------------------------------------------------------------------------- data
def load_images(path=IMAGES):
    data = np.load(path, allow_pickle=True)
    keys = [tuple(str(k).split("|")) for k in data["keys"]]
    raw = np.load(path, allow_pickle=True)["images"] if data["images"].dtype == object else data["images"]
    images = [np.asarray(i, dtype=float) for i in raw] if raw.dtype == object else raw.astype(float)
    n_particles = int(data["n_particles"]) if "n_particles" in data else len(data["beam"])
    GEOMETRY.clear()
    if "geometry_screens" in data:  # SimEnv images: each screen's own chip
        for screen, g in zip(data["geometry_screens"], data["geometry"]):
            GEOMETRY[str(screen)] = (*map(float, g[:4]), int(g[4]), int(g[5]), float(g[6]))
    return keys, images, n_particles, data["beam"]


# Per-screen image geometry, for images that are not the +-5 mm, 200 x 200 virtual-machine
# ones (SimEnv uses each screen's real chip): screen -> (xmin, xmax, ymin, ymax, nx, ny,
# blur_m) of the reconstruction grid. Screens not listed use the default grid.
GEOMETRY: dict[str, tuple] = {}


def geometry(screen=None):
    if screen in GEOMETRY:
        return GEOMETRY[screen]
    f = full6d.FIELD_OF_VIEW_M
    return (-f, f, -f, f, PIXELS, PIXELS, full6d.SCREEN_RESOLUTION_M)


def measured_fractions(image: np.ndarray, n_particles: int, screen=None) -> np.ndarray:
    """Rebinned pixel fractions plus the out-of-window fraction (sums to 1)."""
    _, _, _, _, nx, ny, _ = geometry(screen)
    ry, rx = image.shape[0] // ny, image.shape[1] // nx
    coarse = image.reshape(ny, ry, nx, rx).sum((1, 3)).ravel()
    return np.append(coarse, max(n_particles - coarse.sum(), 0.0)) / n_particles


def image_moments(image: np.ndarray, n_particles: float | None = None) -> tuple[np.ndarray, float]:
    """Centroid-removed second moments (xx, yy, xy) of a full-resolution image (only the
    part inside the window: truncated moments), resolution removed, and the fraction of the
    bunch inside the window (1 if n_particles is None)."""
    centres = (np.arange(full6d.PIXELS) + 0.5) * (2 * full6d.FIELD_OF_VIEW_M / full6d.PIXELS) - full6d.FIELD_OF_VIEW_M
    total = image.sum()
    py, px = image.sum(1) / total, image.sum(0) / total
    mx, my = px @ centres, py @ centres
    xx = px @ (centres - mx) ** 2 - full6d.SCREEN_RESOLUTION_M**2
    yy = py @ (centres - my) ** 2 - full6d.SCREEN_RESOLUTION_M**2
    xy = ((centres - my)[:, None] * (centres - mx)[None, :] * image).sum() / total
    inside = 1.0 if n_particles is None else total / n_particles
    return np.array([xx, yy, xy]), inside


# ---------------------------------------------------------------------------- tracking
def pixel_indices(coords: np.ndarray, alive: np.ndarray, rng: np.random.Generator,
                  screen=None) -> np.ndarray:
    x0, x1, y0, y1, nx, ny, res = geometry(screen)
    blur = rng.normal(0, res, (len(coords), 2)) if res else np.zeros((len(coords), 2))
    ix = np.floor((coords[:, 0] + blur[:, 0] - x0) / ((x1 - x0) / nx))
    iy = np.floor((coords[:, 2] + blur[:, 1] - y0) / ((y1 - y0) / ny))
    inside = alive & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    index = np.full(len(coords), nx * ny, dtype=np.int32)
    index[inside] = (iy[inside] * nx + ix[inside]).astype(np.int32)
    return index


def track_config(args):
    """Track the global sample through one configuration; pixel index per screen."""
    config, seed = args
    line = _LINE
    full6d.apply_config(line, config)
    rng = np.random.default_rng(seed)
    order = sorted(config.screens, key=line.element_index.__getitem__)
    particles = xsuite_model._particles(_SAMPLE, line.converted.initial_energy_ev)
    out, start = {}, 0
    for screen in order:
        stop = line.element_index[screen] + 1
        line.line.track(particles, ele_start=start, ele_stop=stop)
        start = stop
        ids = particles.particle_id
        coords = np.zeros((len(_SAMPLE), 6))
        alive = np.zeros(len(_SAMPLE), bool)
        coords[ids] = _raw_coordinates(particles)
        alive[ids] = particles.state > 0
        out[screen] = pixel_indices(coords, alive, rng, screen)
    line.restore_nominal_quads()
    full6d.set_tdc_voltage(line, 0.0)
    return config.name, out


def _raw_coordinates(particles) -> np.ndarray:
    coords = np.column_stack([np.asarray(getattr(particles, k)) for k in xsuite_model.COORDINATES])
    coords[:, 4] *= -1
    return coords


def track_all(line, sample, configs, seed=0, workers=4):
    global _LINE, _SAMPLE
    _LINE, _SAMPLE = line, sample
    jobs = [(c, seed + i) for i, c in enumerate(configs)]
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.map(track_config, jobs)
    else:
        results = [track_config(j) for j in jobs]
    return {(name, s): idx for name, out in results for s, idx in out.items()}


# ---------------------------------------------------------------------------- prior
def linear_gaussian_fit(line, configs, keys, images, n_meas, min_inside=0.995, transverse_only=False):
    """Weighted least squares for the 21 covariance elements from the moments of images that
    are not clipped, with the model's linear maps; projected to positive definite. With
    `transverse_only`, only the 10 transverse elements from the S02 screens (no cavity
    before them, so the linear map is exact apart from chromatic terms)."""
    import observability as obs

    by_name = {c.name: c for c in configs}
    rows, values, sig = [], [], []
    for (name, screen), image in zip(keys, images):
        moments, inside = image_moments(image, n_meas)
        if inside < min_inside or (transverse_only and "_s02_" not in screen):
            continue
        config = by_name[name]
        full6d.apply_config(line, config)
        transfer = line.linear_map([screen])[screen]
        rows.append(obs.moment_rows(transfer))
        values.append(moments)
        sig.append([0.02 * moments[0], 0.02 * moments[1], 0.02 * np.sqrt(moments[0] * moments[1])])
    line.restore_nominal_quads()
    full6d.set_tdc_voltage(line, 0.0)
    a = np.vstack(rows)
    if transverse_only:
        a = a * np.array([i < 4 and j < 4 for i, j in common.PAIR_INDEX])
    b = np.concatenate(values)
    s = np.concatenate(sig)
    a, b = a / s[:, None], b / s
    # column scaling for conditioning
    scale = np.linalg.norm(a, axis=0)
    scale[scale == 0] = 1
    solution, *_ = np.linalg.lstsq(a / scale, b, rcond=1e-10)
    if transverse_only:
        solution[[c for c, (i, j) in enumerate(common.PAIR_INDEX) if not (i < 4 and j < 4)]] = 0
    sigma = common.unique_to_matrix(solution / scale)
    if transverse_only:
        out = np.zeros((6, 6))
        out[:4, :4] = nearest_positive(sigma[:4, :4])
        return out, len(values)
    return nearest_positive(sigma), len(values)


_WHITE = None


def _moments_job(args):
    """Second moments (xx, yy, xy) at the screens of one config for a Gaussian with
    covariance sigma, from the fixed whitened sample."""
    config, sigma = args
    line = _LINE
    full6d.apply_config(line, config)
    sample = _WHITE @ sqrt_psd(sigma).T
    tracked = line.track_through(sample, config.screens, allow_losses=True)
    out = {}
    rng = np.random.default_rng(11)
    half = full6d.FIELD_OF_VIEW_M
    for screen, c in tracked.items():
        x = c[:, 0] + rng.normal(0, full6d.SCREEN_RESOLUTION_M, len(c))
        y = c[:, 2] + rng.normal(0, full6d.SCREEN_RESOLUTION_M, len(c))
        keep = (np.abs(x) < half) & (np.abs(y) < half)
        x, y = x[keep], y[keep]
        if len(x) < 10:
            out[screen] = np.array([0.0, 0.0, 0.0, 0.0])
            continue
        res2 = full6d.SCREEN_RESOLUTION_M**2
        out[screen] = np.array([x.var() - res2, y.var() - res2,
                                np.mean((x - x.mean()) * (y - y.mean())), keep.sum() / len(_WHITE)])
    line.restore_nominal_quads()
    full6d.set_tdc_voltage(line, 0.0)
    return config.name, out


def sqrt_psd(sigma):
    values, vectors = np.linalg.eigh(sigma)
    return (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T


def gaussian_fit(line, configs, keys, images, n_meas, start, particles=3000,
                 iterations=8, workers=4, seed=5, log=print):
    """Nonlinear Gaussian fit of the 21 covariance elements to the second moments of the
    unclipped images, tracking a fixed whitened sample through the xsuite line (chromatic
    effects, cavities and the TDC included). Sigma = D L L^T D with D the start rms sizes.
    Levenberg-Marquardt with finite-difference Jacobians evaluated in parallel."""
    from scipy.optimize import least_squares

    global _LINE, _WHITE
    _LINE = line
    rng = np.random.default_rng(seed)
    white = rng.normal(size=(particles, 6))
    white -= white.mean(0)
    _WHITE = white @ np.linalg.inv(np.linalg.cholesky(np.cov(white.T))).T
    by_name = {c.name: c for c in configs}
    targets = {}
    for key, image in zip(keys, images):
        moments, inside = image_moments(image, n_meas)
        if inside > 0.05:
            targets[key] = np.append(moments, inside)
    names = sorted({k[0] for k in targets})
    cfgs = [by_name[n] for n in names]
    keys_fit = sorted(targets)
    meas = np.concatenate([targets[k] for k in keys_fit])
    scale = np.concatenate([[targets[k][0], targets[k][1], np.sqrt(targets[k][0] * targets[k][1]), 0.5]
                            for k in keys_fit])
    noise = 0.02 * scale  # 2 % on the moments, 1 % absolute on the inside fraction
    d = np.sqrt(np.diag(start))
    tril = np.tril_indices(6)
    l0 = np.linalg.cholesky(start / np.outer(d, d) + 1e-9 * np.eye(6))
    pool = mp.get_context("fork").Pool(workers)

    def sigma_of(p):
        lmat = np.zeros((6, 6))
        lmat[tril] = p
        return (lmat @ lmat.T) * np.outer(d, d)

    def predict_many(params):
        jobs = [(c, sigma_of(p)) for p in params for c in cfgs]
        results = pool.map(_moments_job, jobs)
        out = []
        for i in range(len(params)):
            chunk = dict(results[i * len(cfgs):(i + 1) * len(cfgs)])
            out.append(np.concatenate([chunk[c][s] for c, s in keys_fit]))
        return out

    def residual(p):
        return (predict_many([p])[0] - meas) / noise

    def jacobian(p):
        step = 1e-4
        params = [p] + [p + step * np.eye(len(p))[i] for i in range(len(p))]
        preds = predict_many(params)
        return np.column_stack([(preds[i + 1] - preds[0]) / noise / step for i in range(len(p))])

    try:
        result = least_squares(residual, l0[tril], jac=jacobian, method="lm", max_nfev=iterations)
    finally:
        pool.close()
    sigma = sigma_of(result.x)
    log(f"Gaussian fit on {len(keys_fit)} images (truncated moments): cost {result.cost:.3g}, "
        f"chi2/dof {2 * result.cost / max(len(meas) - 21, 1):.2f}, nfev {result.nfev}")
    return sigma, len(keys_fit)


def nearest_positive(sigma: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    d = np.sqrt(np.abs(np.diag(sigma)))
    corr = sigma / np.outer(d, d)
    values, vectors = np.linalg.eigh(corr)
    values = np.maximum(values, floor)
    corr = (vectors * values) @ vectors.T
    dd = np.sqrt(np.diag(corr))
    corr = corr / np.outer(dd, dd)
    return corr * np.outer(d, d)


def draw_gaussian(sigma, n, rng, mean=None):
    mean = np.zeros(6) if mean is None else mean
    return mean + rng.multivariate_normal(np.zeros(6), sigma, size=n, method="eigh")


# ---------------------------------------------------------------------------- MENT
def ment(indices: dict, data: dict, keys_used, sweeps=30, omega=0.6, eps=None, n_meas=1e5,
         log=None):
    """Gauss-Seidel MENT on per-particle weights. indices[key]: (N,) pixel index per image;
    data[key]: measured fractions (PIXELS^2 + 1). Returns normalised weights and history."""
    n = len(next(iter(indices.values())))
    eps = 0.5 / n_meas if eps is None else eps
    logw = np.zeros(n)
    history = []
    for sweep in range(sweeps):
        chi2, dof = 0.0, 0
        for key in keys_used:
            idx, meas = indices[key], data[key]
            w = np.exp(logw - logw.max())
            w /= w.sum()
            sim = np.bincount(idx, weights=w, minlength=len(meas))
            populated = sim > 0
            update = np.zeros_like(sim)
            update[populated] = omega * np.log((meas[populated] + eps) / (sim[populated] + eps))
            logw += update[idx]
            var = np.maximum(meas, sim) / n_meas + 1e-12
            used = (meas > 0) | (sim > 0)
            chi2 += float((((meas - sim) ** 2) / var)[used].sum())
            dof += int(used.sum())
        w = np.exp(logw - logw.max())
        w /= w.sum()
        ess = 1.0 / np.sum(w**2)
        history.append((sweep, chi2 / max(dof, 1), ess))
        if log:
            log(f"   sweep {sweep:2d} chi2/pixel {chi2 / max(dof, 1):.3f} ESS {ess:.0f}")
    return w, history


def predicted(indices, w, key):
    _, _, _, _, nx, ny, _ = geometry(key[1])
    return np.bincount(indices[key], weights=w, minlength=nx * ny + 1)


# ---------------------------------------------------------------------------- summaries
def weighted_cov(sample, w):
    mean = w @ sample
    d = sample - mean
    return (d * w[:, None]).T @ d, mean


def summary(sample, w=None) -> dict:
    w = np.full(len(sample), 1 / len(sample)) if w is None else w
    s, mean = weighted_cov(sample, w)
    sd = np.sqrt(np.diag(s))
    corr = s / np.outer(sd, sd)

    def eps(i):
        return float(np.sqrt(np.linalg.det(s[i : i + 2, i : i + 2])))

    tau, delta = sample[:, 4] - mean[4], sample[:, 5] - mean[5]
    chirp = s[4, 5] / s[4, 4]
    resid = delta - chirp * tau
    # slice emittance x in the central slice |tau| < 0.25 sigma_tau
    core = np.abs(tau) < 0.25 * sd[4]
    wc = w * core
    wc = wc / wc.sum()
    sc, _ = weighted_cov(sample, wc)
    return {
        "eps_x": eps(0),
        "eps_y": eps(2),
        "eps_z": eps(4),
        "eps_4d_ratio": float(np.sqrt(np.linalg.det(s[:4, :4])) / (eps(0) * eps(2))),
        "sigma_x": float(sd[0]),
        "sigma_y": float(sd[2]),
        "sigma_z": float(sd[4]),
        "sigma_delta": float(sd[5]),
        "chirp": float(chirp),
        "slice_sigma_delta": float(np.sqrt(w @ resid**2)),
        "corr_tau_delta": float(corr[4, 5]),
        "corr_x_y": float(corr[0, 2]),
        "corr_x_py": float(corr[0, 3]),
        "corr_px_y": float(corr[1, 2]),
        "corr_x_delta": float(corr[0, 5]),
        "corr_y_tau": float(corr[2, 4]),
        "slice_eps_x": float(np.sqrt(np.linalg.det(sc[0:2, 0:2]))),
        "slice_eps_y": float(np.sqrt(np.linalg.det(sc[2:4, 2:4]))),
        **{f"core_eps_{p}": trimmed_emittance(sample, w, i) for p, i in (("x", 0), ("y", 2), ("z", 4))},
        "cov": s.tolist(),
    }


CORE_FRACTION = 0.95


def trimmed_emittance(sample, w, i, fraction=CORE_FRACTION):
    """rms emittance of plane (i, i+1) after dropping the outer 1 - fraction of the charge
    (by Mahalanobis distance in that plane). The rms emittance of this beam is set by a
    ~1 % halo that lands outside the +-5 mm screens on most images, so the core value is the
    one images can measure."""
    u = sample[:, i : i + 2]
    s2, mean = weighted_cov(u, w)
    d = u - mean
    m = np.einsum("ij,jk,ik->i", d, np.linalg.inv(s2), d)
    order = np.argsort(m)
    cum = np.cumsum(w[order])
    keep = np.zeros(len(w), bool)
    keep[order[: np.searchsorted(cum, fraction * cum[-1]) + 1]] = True
    wk = w * keep
    wk = wk / wk.sum()
    sk, _ = weighted_cov(u, wk)
    return float(np.sqrt(np.linalg.det(sk)))


# ---------------------------------------------------------------------------- pipeline
# Broad, uncorrelated starting prior (rms of x, px, y, py, tau, delta). It only has to
# cover the beam; it is set from the order of magnitude of the S02 screen sizes, not from
# the true beam. A Gaussian fit to the image moments was tried as a start and fails: the
# beam has tails outside the +-5 mm window on most screens, which bias the moments.
BROAD_PRIOR_RMS = np.array([3e-3, 1.5e-3, 3e-3, 1.5e-3, 2e-3, 3e-2])


def reconstruct(line, configs, keys, images, n_meas, keys_used=None, n_sample=200_000,
                inflation=(1.0, 1.3, 1.15, 1.1, 1.1), sweeps=(8, 8, 10, 10, 15), seed=0,
                log=print, workers=4, truth_summary=None, refine_longitudinal=True,
                final_inflation=(1.15, 1.1), final_sweeps=(15, 30)):
    """Adaptive MENT: stage 0 uses the broad prior; every later stage redraws the sample
    from a Gaussian fitted to the previous weighted sample, inflated by `inflation`."""
    keys_used = keys if keys_used is None else keys_used
    data = {k: measured_fractions(img, n_meas, k[1]) for k, img in zip(keys, images)}
    names_used = {k[0] for k in keys_used}
    cfgs = [c for c in configs if c.name in names_used]
    sigma = np.diag(BROAD_PRIOR_RMS**2)
    rng = np.random.default_rng(seed)
    mean = np.zeros(6)
    stages = []
    for stage, (factor, n_sweeps) in enumerate(zip(inflation, sweeps)):
        prior = sigma * factor**2
        sample = draw_gaussian(prior, n_sample, rng, mean)
        t0 = time.time()
        indices = track_all(line, sample, cfgs, seed=seed + 100 * stage, workers=workers)
        log(f"stage {stage}: prior x{factor}, tracked {len(cfgs)} configs in {time.time() - t0:.0f} s")
        w, history = ment(indices, data, keys_used, sweeps=n_sweeps, n_meas=n_meas, log=log)
        sigma, mean = weighted_cov(sample, w)
        stages.append({"factor": factor, "history": history, "summary": summary(sample, w)})
        log(f"   sd {np.sqrt(np.diag(sigma))}")
        if truth_summary is not None:
            now = stages[-1]["summary"]
            log("   ratios " + " ".join(f"{k} {now[k] / truth_summary[k]:.3f}" for k in
                ("eps_x", "eps_y", "eps_z", "core_eps_x", "core_eps_y", "core_eps_z", "sigma_z", "sigma_delta", "chirp")))
    if refine_longitudinal:
        refined, res = longitudinal_fit(line, cfgs, keys_used, data, n_meas, sample, w,
                                        workers=workers, log=log)
        sigma, mean = np.cov(refined.T), refined.mean(0)
        for stage, (factor, n_sweeps) in enumerate(zip(final_inflation, final_sweeps), len(stages)):
            sample = draw_gaussian(sigma * factor**2, n_sample, rng, mean)
            indices = track_all(line, sample, cfgs, seed=seed + 100 * stage, workers=workers)
            w, history = ment(indices, data, keys_used, sweeps=n_sweeps, n_meas=n_meas, log=log)
            stages.append({"factor": factor, "history": history, "summary": summary(sample, w),
                           "longitudinal": True})
            sigma, mean = weighted_cov(sample, w)
            if truth_summary is not None:
                now = stages[-1]["summary"]
                log("   ratios " + " ".join(f"{k} {now[k] / truth_summary[k]:.3f}" for k in
                    ("eps_x", "eps_y", "eps_z", "core_eps_x", "core_eps_y", "core_eps_z", "sigma_z", "sigma_delta", "chirp")))
    return {"sample": sample, "weights": w, "indices": indices, "data": data,
            "stages": stages, "keys_used": keys_used}
    # (truth_summary is for monitoring only; it never enters the fit)


def image_chi2(indices, data, keys_used, n_meas, n_sample):
    """Pixel chi2 per used pixel of an unweighted sample against the images."""
    w = np.full(n_sample, 1.0 / n_sample)
    chi, used_total = 0.0, 0
    for key in keys_used:
        sim = predicted(indices, w, key)
        meas = data[key]
        var = meas / n_meas + sim / n_sample + 1e-12
        used = (meas > 0) | (sim > 0)
        chi += float((((meas - sim) ** 2) / var)[used].sum())
        used_total += int(used.sum())
    return chi / max(used_total, 1)


def longitudinal_fit(line, cfgs, keys_used, data, n_meas, sample, w, n_fit=40_000, seed=3,
                     workers=4, log=print):
    """Refine the energy coordinate: resample the MENT result, keep x, px, y, py, tau and
    replace delta by m + c t + q t^2 + s xi (t = tau - <tau>, xi standard normal), with
    (m, c, q, s) fitted to the images by Nelder-Mead on the pixel chi2. MENT alone builds the
    tau-delta correlation very slowly, because it enters every image only indirectly
    (chromatic effects, the dispersion of the FEA dogleg)."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    base = sample[rng.choice(len(sample), n_fit, p=w)]
    xi = rng.normal(size=n_fit)
    t = base[:, 4] - base[:, 4].mean()
    cov = np.cov(t, base[:, 5])
    c0 = cov[0, 1] / cov[0, 0]
    s0 = np.sqrt(max(cov[1, 1] - c0 * cov[0, 1], 1e-8))
    m0 = base[:, 5].mean()
    st = t.std()
    scale = np.array([1e-3, 10.0 * st, 1e-3, 1e-3])  # parameters in units of ~1e-3 in delta

    def build(p):
        m, c, q, sl = p * scale
        trial = base.copy()
        trial[:, 5] = m + c * t / st + q * (t / st) ** 2 + abs(sl) * xi
        return trial

    def f(p):
        trial = build(p)
        indices = track_all(line, trial, cfgs, seed=seed, workers=workers)
        value = image_chi2(indices, data, keys_used, n_meas, n_fit)
        log(f"   longitudinal {np.round(p * scale, 6)} chi2 {value:.4f}")
        return value

    p0 = np.array([m0, c0 * st, 0.0, s0]) / scale
    res = minimize(f, p0, method="Nelder-Mead",
                   options={"maxfev": 80, "xatol": 0.05, "fatol": 1e-3,
                            "initial_simplex": [p0 + d for d in (np.zeros(4), *np.eye(4) * 2.0)]})
    return build(res.x), res


def resample_stages(line, cfgs, keys_used, data, n_meas, sample, w, stages=6, n_sample=150_000,
                    bandwidth=0.28, sweeps=15, seed=21, workers=4, log=print, truth_summary=None):
    """Particle refinement after the Gaussian stages: draw the next sample from the current
    weighted sample with a Gaussian kernel (bandwidth x the weighted covariance), and run
    MENT relative to that proposal. A Gaussian proposal spreads the particles over a 6D
    ellipsoid of which this curved, strongly correlated beam fills a small part, and MENT
    stalls when the sample has no particles where the images need charge; resampling puts
    the particles where the charge is."""
    rng = np.random.default_rng(seed)
    history = []
    for stage in range(stages):
        cov, _ = weighted_cov(sample, w)
        parents = sample[rng.choice(len(sample), n_sample, p=w)]
        sample = parents + rng.multivariate_normal(np.zeros(6), bandwidth**2 * cov, n_sample, method="eigh")
        indices = track_all(line, sample, cfgs, seed=seed + 100 * stage, workers=workers)
        w, hist = ment(indices, data, keys_used, sweeps=sweeps, n_meas=n_meas)
        history.append(hist)
        log(f"resample stage {stage}: chi2/pixel {hist[-1][1]:.3f} ESS {hist[-1][2]:.0f}")
        if truth_summary is not None:
            now = summary(sample, w)
            log("   ratios " + " ".join(f"{k} {now[k] / truth_summary[k]:.3f}" for k in
                ("core_eps_x", "core_eps_y", "core_eps_z", "sigma_z", "sigma_delta", "chirp", "corr_x_y")))
    return sample, w, indices, history


def chirp_scan(line, cfgs, keys_used, data, n_meas, sample, w, chirps=(-10, -5, 0, 4, 8, 12, 16, 20, 24),
               slices=(0.5e-3, 1e-3, 2e-3, 4e-3), n_fit=40_000, seed=3, workers=4, log=print):
    """Global search of the S02 energy chirp: resample the current result, set
    delta = <delta> + c (tau - <tau>) + s xi and score every (c, s) of a grid by the image
    chi2. The chi2 has a second, shallower minimum at negative chirp, so a local search from
    a poor start can end there; with a good transverse beam the true chirp is a clear
    minimum. Returns the best sample (n_fit particles) and the grid."""
    rng = np.random.default_rng(seed)
    base = sample[rng.choice(len(sample), n_fit, p=w)]
    xi = rng.normal(size=n_fit)
    t = base[:, 4] - base[:, 4].mean()
    m = base[:, 5].mean()
    grid, best = [], None
    for c in chirps:
        for sl in slices:
            trial = base.copy()
            trial[:, 5] = m + c * t + sl * xi
            idx = track_all(line, trial, cfgs, seed=seed, workers=workers)
            value = image_chi2(idx, data, keys_used, n_meas, n_fit)
            grid.append((c, sl, value))
            log(f"   chirp {c:6.1f} slice {sl:.4f} chi2 {value:.4f}")
            if best is None or value < best[0]:
                best = (value, trial)
    return best[1], grid


def main() -> None:
    line = full6d.build_line()
    configs = full6d.make_plan(line)
    keys, images, n_meas, beam = load_images()
    result = reconstruct(line, configs, keys, images, n_meas)
    truth = summary(beam)
    reco = summary(result["sample"], result["weights"])
    for k in truth:
        if k != "cov":
            print(f"{k:18s} true {truth[k]: .4e} reco {reco[k]: .4e}")
    np.savez_compressed(common.HERE / "ment_result.npz", sample=result["sample"], weights=result["weights"])


if __name__ == "__main__":
    sys.exit(main())
