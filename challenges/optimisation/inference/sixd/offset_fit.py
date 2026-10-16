"""Quadrupole offsets and launch from the measurement plan's own centroids.

No extra measurement: every image of the plan comes with the BPM readings upstream of
the inserted screen, and the image itself has a centroid. Across the plan's quadrupole
and phase settings those centroids move in a way set by where the beam passes through
each quadrupole, so a least-squares fit of the quadrupole offsets and the launch
(x, px, y, py, delta) explains them. Each BPM's and screen's own centring is a free parameter
too, constant across settings, which is why the fit needs more than one setting.

Linear in the unknowns about the current estimate (finite differences on the model with
the steering kicks in), with Gaussian priors as regularisation, and relinearised a
couple of times.
"""

from __future__ import annotations

import numpy as np

QUAD_PRIOR = 50e-6  # rms survey error the fit is pulled towards zero with
LAUNCH_PRIOR = np.array([100e-6, 100e-6, 100e-6, 100e-6, 0.0, 1e-3])  # x px y py (zeta) delta
LAUNCHED = [0, 1, 2, 3, 5]
MIN_ON_CHIP = 0.97  # a clipped image's centroid depends on the beam's shape
DIAG_PRIOR = 30e-6
BPM_NOISE = 5e-6  # mean of 10 shots
SCREEN_NOISE = 10e-6
STEP = 20e-6
# SimEnv's beam pipe: a particle once outside it no longer counts at any diagnostic
# downstream. In the FEA optics a 0.25 % halo is scraped in S07, which moves the
# centroids there by 150 um; readings after any scraping are left out of the fit. On a
# model line built with apertures the line itself loses the particles instead.
PIPE_RADIUS = 50e-3
APERTURE_LOSS_LIMIT = 0.02  # with the apertures modelled, a small loss is predicted, not guessed


def _measurements(data, bpm_index):
    """Rows of (key index, diagnostic, plane, value, sigma)."""
    keys = [str(k) for k in data["keys"]]
    geometry = dict(zip([str(s) for s in data["geometry_screens"]], data["geometry"]))
    rows = []
    for i, key in enumerate(keys):
        screen = key.split("|")[1]
        for name, j in bpm_index.items():
            if data["bpm_valid"][i][j]:
                for plane in (0, 1):
                    rows.append((i, name, plane, float(data["bpm"][i][j][plane]), BPM_NOISE))
        img = np.asarray(data["images"][i], float)
        g = geometry[screen]
        xs = np.linspace(g[0], g[1], img.shape[1] + 1); xs = 0.5 * (xs[1:] + xs[:-1])
        ys = np.linspace(g[2], g[3], img.shape[0] + 1); ys = 0.5 * (ys[1:] + ys[:-1])
        s = img.sum()
        if s < MIN_ON_CHIP * float(data["n_particles"]):
            continue
        rows.append((i, screen, 0, float(img.sum(0) @ xs / s), SCREEN_NOISE))
        rows.append((i, screen, 1, float(img.sum(1) @ ys / s), SCREEN_NOISE))
    return keys, rows


def _predict(line, configs_by_name, keys, rows, beam, quads, params, geometry, aperture_line=False):
    """Model centroids for every measurement row at parameters (quad dx, dy..., launch)."""
    import full6d

    n = len(quads)
    for k, q in enumerate(quads):
        line.line[q].shift_x, line.line[q].shift_y = params[2 * k], params[2 * k + 1]
    launched = beam.copy()
    launched[:, LAUNCHED] += params[2 * n:2 * n + len(LAUNCHED)]
    out = np.zeros(len(rows))
    scraped = np.zeros(len(rows))
    try:
        by_key = {}
        for r, (i, name, plane, _, _) in enumerate(rows):
            by_key.setdefault(i, []).append((r, name, plane))
        for i, items in by_key.items():
            cname, screen = keys[i].split("|")
            full6d.apply_config(line, configs_by_name[cname])
            last = line.element_index[screen]
            stops = [d for d in line.line.element_names if ("_dia_bpm_" in d or "_dia_scr_" in d)
                     and line.element_index[d] <= last]
            tracked = line.track_through(launched, stops, allow_losses=True)
            line.restore_nominal_quads()
            full6d.set_rf_phases(line.line, {})
            for r, name, plane in items:
                c = tracked[name]
                if aperture_line:
                    scraped[r] = 1.0 - len(c) / len(launched)
                else:
                    keep = np.ones(len(c), bool)
                    for d in stops[:stops.index(name) + 1]:
                        keep &= np.hypot(tracked[d][:, 0], tracked[d][:, 2]) <= PIPE_RADIUS
                    scraped[r] = 1.0 - keep.mean()
                    c = c[keep]
                if name == screen:
                    g = geometry[screen]
                    c = c[(c[:, 0] > g[0]) & (c[:, 0] < g[1]) & (c[:, 2] > g[2]) & (c[:, 2] < g[3])]
                out[r] = c[:, 2 * plane].mean()
    finally:
        for q in quads:
            line.line[q].shift_x, line.line[q].shift_y = 0.0, 0.0
    return out, scraped


def fit(line, configs, data, beam, quads, kicks=None, iterations=1, n_particles=1000, log=print,
        aperture_line=False):
    """Fitted {quad: (dx, dy)}, launch (x, px, y, py, delta), diagnostic offsets {name: (dx, dy)}.

    `data` is an images file from simenv_acquire.acquire (BPM readings included), `beam`
    the model's beam (a subsample is used), `kicks` the steering {corrector: (h, v)}."""
    rng = np.random.default_rng(3)
    beam = beam[rng.choice(len(beam), min(n_particles, len(beam)), replace=False)]
    bpm_names = [str(b) for b in data["bpm_names"]]
    bpm_index = {b: j for j, b in enumerate(bpm_names) if b in line.element_index}
    keys, rows = _measurements(data, bpm_index)
    geometry = {str(s): g for s, g in zip(data["geometry_screens"], data["geometry"])}
    by_name = {c.name: c for c in configs}
    y = np.array([r[3] for r in rows])
    w = 1.0 / np.array([r[4] for r in rows])
    diags = sorted({(r[1], r[2]) for r in rows})
    D = np.zeros((len(rows), len(diags)))
    for r, row in enumerate(rows):
        D[r, diags.index((row[1], row[2]))] = 1.0
    n = len(quads)
    prior = np.concatenate([np.full(2 * n, QUAD_PRIOR), LAUNCH_PRIOR[LAUNCHED], np.full(len(diags), DIAG_PRIOR)])
    params = np.zeros(2 * n + len(LAUNCHED))
    kicks = kicks or {}
    for c, (h, v) in kicks.items():
        line.set_corrector(c, h, v)
    try:
        for it in range(iterations):
            f0, scraped = _predict(line, by_name, keys, rows, beam, quads, params, geometry, aperture_line)
            if it == 0:
                w = np.where(scraped > (APERTURE_LOSS_LIMIT if aperture_line else 0.0), 0.0, w)
                log(f"offset fit: {int((scraped > 0).sum())} of {len(rows)} readings follow scraping, left out")
            J = np.zeros((len(rows), len(params)))
            for p in range(len(params)):
                dp = params.copy(); dp[p] += STEP
                J[:, p] = (_predict(line, by_name, keys, rows, beam, quads, dp, geometry, aperture_line)[0] - f0) / STEP
            A = np.hstack([J, D])
            x0 = np.concatenate([params, np.zeros(len(diags))])
            r0 = y - f0 + J @ params  # linear model y = f0 + J (p - params) + D d
            M = np.vstack([A * w[:, None], np.diag(1.0 / prior)])
            b = np.concatenate([r0 * w, np.zeros(len(x0))])
            sol = np.linalg.lstsq(M, b, rcond=None)[0]
            params = sol[:len(params)]
            resid = (y - f0 - J @ (params - x0[:len(params)]) - D @ sol[len(params):]) * w
            log(f"offset fit iteration {it}: chi2/row {np.sum(resid ** 2) / max(np.count_nonzero(w), 1):.2f}, rows {np.count_nonzero(w)}")
    finally:
        for c in kicks:
            line.set_corrector(c, 0.0, 0.0)
    offsets = {q: (float(params[2 * k]), float(params[2 * k + 1])) for k, q in enumerate(quads)}
    diag = {}
    for (name, plane), v in zip(diags, sol[len(params):]):
        diag.setdefault(name, [0.0, 0.0])[plane] = float(v)
    return offsets, params[2 * n:], diag
