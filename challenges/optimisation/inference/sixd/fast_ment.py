"""The run_ment.py reconstruction with every xsuite re-tracking replaced by the fitted
polynomial maps of surrogate.py (one matrix product per stage instead of about 3 min of
tracking). Same steps and outputs, written with a suffix."""

from __future__ import annotations

import sys
import time

import common
import full6d
import ment6d
import numpy as np
import run_ment
import surrogate


_ORIGINAL_MENT = ment6d.ment
CENTRES = (np.arange(ment6d.PIXELS) + 0.5) * ment6d.PIXEL_M - full6d.FIELD_OF_VIEW_M


def measured_centroid(fractions):
    img = fractions[:-1].reshape(ment6d.PIXELS, ment6d.PIXELS)  # rows are y
    total = img.sum()
    return np.array([img.sum(0) @ CENTRES, img.sum(1) @ CENTRES]) / max(total, 1e-12)


def use_maps(maps, register_data=None, damping=0.7):
    """Route ment6d.track_all through the maps.

    register: a free (dx, dy) translation per image, for an unknown orbit (quadrupole
    offsets, launch error): the model coordinates on each screen are shifted so that the
    in-window centroid of the reconstruction matches the measured one. Started at the
    measured centroid and updated after every MENT solve. Only the image positions are
    given up; sizes and shapes are still fitted."""
    register = register_data is not None
    # one offset per (quad setting, screen): the orbit does not change with the TDC state, and
    # the centroid moves between TDC off, +V and -V carry the time information
    group = lambda k: (k[0].rsplit("_", 1)[0] if k[0].endswith(("_off", "_plus", "_minus")) else k[0], k[1])
    offsets = {}
    for k, f in (register_data or {}).items():
        if k[0].endswith(("_plus", "_minus")):
            continue
        offsets[group(k)] = measured_centroid(f)
    last = {}

    def track_all(line, sample, configs, seed=0, workers=4):
        names = {c.name for c in configs}
        keys = [k for k in maps.keys if k[0] in names]
        if not register:
            return maps.indices(sample, np.random.default_rng(seed), keys)
        rng = np.random.default_rng(seed)
        out = {}
        for k, (xy, alive) in maps.coords(sample, keys).items():
            shifted = xy + offsets.get(group(k), 0.0)
            last[k] = (shifted, alive)
            c = np.zeros((len(sample), 4))
            c[:, 0], c[:, 2] = shifted[:, 0], shifted[:, 1]
            out[k] = ment6d.pixel_indices(c, alive, rng, k[1])
        return out

    def ment(indices, data, keys_used, *args, **kwargs):
        w, history = _ORIGINAL_MENT(indices, data, keys_used, *args, **kwargs)
        if register:
            residuals = {}
            for k in keys_used:
                if k not in last or len(last[k][0]) != len(w):
                    continue
                xy, alive = last[k]
                inside = alive & (np.abs(xy) < full6d.FIELD_OF_VIEW_M).all(1)
                if w[inside].sum() <= 0:
                    continue
                pred = (w[inside, None] * xy[inside]).sum(0) / w[inside].sum()
                residuals.setdefault(group(k), []).append(measured_centroid(data[k]) - pred)
            for g, r in residuals.items():
                offsets[g] = offsets.get(g, 0.0) + damping * np.mean(r, axis=0)
        return w, history

    ment6d.track_all = track_all
    ment6d.ment = ment if register else _ORIGINAL_MENT
    return offsets


UPSTREAM_SCREENS = ("s02_", "s03_", "s04_", "s05_")
FEW_SCREENS = ("cla_s02_dia_scr_01", "cla_s02_dia_scr_03", "cla_s03_dia_scr_01", "cla_s04_dia_scr_03",
               "cla_s05_dia_scr_01")


def lean_sets(keys):
    """Nested image sets, from the full plan down. Name -> list of (config, screen)."""
    up = lambda s: any(t in s for t in UPSTREAM_SCREENS)
    full = list(keys)
    s2 = [k for k in full if k[0] != "nominal_off" or up(k[1])]
    s3 = [k for k in s2 if k[0] != "s02b"]
    s4 = [k for k in s3 if not k[0].startswith(("s07_03", "s07_04"))]
    s5 = [k for k in s4 if not up(k[1]) or k[1] in FEW_SCREENS]
    s6 = [k for k in s5 if not k[0].startswith("s07_02")]
    s7 = [k for k in s6 if k[0] != "s02c"]
    # fewer screen insertions, more S02 settings (a magnet change is 30 s, a screen move 1 min)
    streak9 = [k for k in s4 if not up(k[1])]
    few = lambda screens, settings: [k for k in full if k[0] in settings and k[1] in screens]
    t1 = few(("cla_s02_dia_scr_03", "cla_s04_dia_scr_03"), ("nominal_off", "s02a", "s02b", "s02c")) + streak9
    t2 = few(("cla_s02_dia_scr_03", "cla_s03_dia_scr_01", "cla_s04_dia_scr_03"),
             ("nominal_off", "s02a", "s02b", "s02c")) + streak9
    no_tdc = [k for k in t2 if not k[0].endswith(("_plus", "_minus"))]
    return {"no TDC": no_tdc, "two screens four settings": t1, "three screens four settings": t2, "full": full, "upstream nominal only": s2, "two S02 settings": s3,
            "two streak screens": s4, "five transverse screens": s5, "one streak screen": s6,
            "one S02 setting": s7}


def run(line, images_path=common.HERE / "images.npz", set_name="full", suffix="_fast", log=print,
        register=False, maps=None):
    t0 = time.time()
    register_data = None
    if register:
        keys, images, n_meas, _ = ment6d.load_images(images_path)
        register_data = {k: ment6d.measured_fractions(img, n_meas, k[1]) for k, img in zip(keys, images)}
    offsets = use_maps(maps or surrogate.load_combined(), register_data)
    run_ment.OUT_JSON = common.HERE / f"ment_results{suffix}.json"
    run_ment.OUT_NPZ = common.HERE / f"ment_full{suffix}.npz"
    if set_name in run_ment.image_sets([]):
        results = run_ment.run(line, images_path, set_name, log=log)
    else:
        original = run_ment.image_sets
        run_ment.image_sets = lambda keys: {set_name: lean_sets(keys)[set_name]}
        try:
            results = run_ment.run(line, images_path, set_name, log=log)
        finally:
            run_ment.image_sets = original
    results["seconds"] = time.time() - t0
    if register:
        results["offsets_mm"] = {"|".join(k): [float(v) * 1e3 for v in o] for k, o in offsets.items()}
    common.write_json(run_ment.OUT_JSON, results, indent=1)
    log(f"reconstruction took {results['seconds']:.0f} s")
    return results


if __name__ == "__main__":
    run(full6d.build_line(), set_name=sys.argv[1] if len(sys.argv) > 1 else "full")
