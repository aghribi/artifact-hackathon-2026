"""Image acquisition with shot-to-shot launch jitter.

Each image is one screen insertion: `shots` bunches at 10 Hz, each with its own launch
jitter on top of the fixed hidden launch error, summed into one image. The bunch is
`particles / shots` macroparticles per shot, so the counting statistics of the summed image
match the jitter-free images. Two images are kept per insertion: the plain sum, and the sum
after shifting every shot onto the mean centroid (the centroid is computed from the shot's own
image, as it would be online). Each screen is its own insertion, so jitter is independent
between screens.
"""

from __future__ import annotations

import multiprocessing as mp

import common
import full6d
import numpy as np

cv = common.cv
# project convention (steering/README.md): 20 um, 2 urad, 20 um, 2 urad, 0.02 %; plus an
# arrival-time jitter of 100 fs (30 um in tau), which matters for the TDC images
JITTER = np.array([20e-6, 2e-6, 20e-6, 2e-6, 30e-6, 2e-4])

_STATE = {}


def _edges(screen):
    g = _STATE.get("geometry", {}).get(screen)
    if g is None:
        e = np.linspace(-full6d.FIELD_OF_VIEW_M, full6d.FIELD_OF_VIEW_M, full6d.PIXELS + 1)
        return e, e
    return np.linspace(g[0], g[1], g[4] + 1), np.linspace(g[2], g[3], g[5] + 1)


def _histogram(coords, rng, screen):
    """Screen image with 20 um blur, rows = y; on the screen's chip when a geometry is
    given, else the +-5 mm virtual-machine window."""
    ex, ey = _edges(screen)
    blur = rng.normal(0, full6d.SCREEN_RESOLUTION_M, (len(coords), 2))
    image, _, _ = np.histogram2d(coords[:, 2] + blur[:, 1], coords[:, 0] + blur[:, 0], bins=[ey, ex])
    return image


def _centroid(image, screen=None):
    ex, ey = _edges(screen)
    cx, cy = 0.5 * (ex[1:] + ex[:-1]), 0.5 * (ey[1:] + ey[:-1])
    s = image.sum()
    return np.array([image.sum(0) @ cx, image.sum(1) @ cy]) / max(s, 1)


def _insertion(args):
    config, screen, seed = args
    line, beam, launch, jitter, shots = (_STATE[k] for k in ("line", "beam", "launch", "jitter", "shots"))
    rng = np.random.default_rng(seed)
    full6d.apply_config(line, config)
    per_shot = np.array_split(rng.permutation(len(beam)), shots)
    coords, cents = [], []
    for idx in per_shot:
        shot = beam[idx] + launch + rng.normal(0, 1, 6) * jitter
        c = line.track_through(shot, [screen], allow_losses=True)[screen]
        img = _histogram(c, rng, screen)
        coords.append(c)
        cents.append(_centroid(img, screen))
    line.restore_nominal_quads()
    full6d.set_tdc_voltage(line, 0.0)
    plain = sum(_histogram(c, rng, screen) for c in coords)
    mean = np.mean(cents, axis=0)
    aligned = 0
    for c, ce in zip(coords, cents):
        shifted = c.copy()
        shifted[:, 0] += mean[0] - ce[0]
        shifted[:, 2] += mean[1] - ce[1]
        aligned = aligned + _histogram(shifted, rng, screen)
    return (config.name, screen), plain, aligned, np.array(cents)


def acquire(line, beam, configs, keys, jitter=JITTER, shots=10, errors=None, seed=7, workers=4,
            out_prefix="images_jitter", geometry=None, rebin=2):
    """Images for the (config, screen) `keys`; writes <prefix>_plain.npz and _aligned.npz."""
    errors = errors or cv.HiddenErrors()
    for q, (dx, dy) in errors.quad_offsets.items():
        line.line[q].shift_x, line.line[q].shift_y = dx, dy
    _STATE.update(line=line, beam=beam, launch=errors.launch, jitter=np.asarray(jitter), shots=shots,
                  geometry=geometry or {})
    by_name = {c.name: c for c in configs}
    jobs = [(by_name[c], s, seed + i) for i, (c, s) in enumerate(keys)]
    try:
        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.map(_insertion, jobs)
    finally:
        for q in errors.quad_offsets:
            line.line[q].shift_x, line.line[q].shift_y = 0.0, 0.0
    names = np.array([f"{c}|{s}" for (c, s), *_ in results])
    order = np.argsort(names)
    extra = {}
    if geometry:
        screens = sorted({k.split("|")[1] for k in names})
        extra = dict(geometry_screens=np.array(screens),
                     geometry=np.array([(*geometry[s][:4], geometry[s][4] // rebin, geometry[s][5] // rebin,
                                         full6d.SCREEN_RESOLUTION_M) for s in screens]))
    for label, col in (("plain", 1), ("aligned", 2)):
        imgs = [results[i][col] for i in order]
        np.savez_compressed(common.HERE / f"{out_prefix}_{label}.npz", keys=names[order],
                            images=np.array(imgs, dtype=object) if geometry else np.array(imgs), beam=beam,
                            shots=len(results) * shots, jitter=np.asarray(jitter), **extra)
    return {names[i]: results[i][3] for i in order}
