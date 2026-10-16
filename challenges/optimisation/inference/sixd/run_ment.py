"""Image-based 6D reconstruction at the S02 start, end to end, scored against the true beam
(scoring only). Steps (ment6d.py):

1. adaptive MENT from a broad uncorrelated prior, Gaussian refit between stages
   (reconstruct), then a local Nelder-Mead refit of delta(tau) and more MENT stages;
2. particle resampling stages (resample_stages);
3. global chirp scan on a grid (chirp_scan), then resampling stages again.

Writes ment_full.npz (weighted sample) and ment_results.json. The numbers in README.md come
from these steps run one after the other (about 2.5 h on 4 cores).
"""

from __future__ import annotations

import sys

import common
import full6d
import ment6d
import numpy as np

OUT_JSON = common.HERE / "ment_results.json"
OUT_NPZ = common.HERE / "ment_full.npz"


def image_sets(keys):
    """Nested measurement sets: name -> list of (config, screen) keys."""
    def pick(predicate):
        return [k for k in keys if predicate(k[0])]

    nominal = pick(lambda c: c == "nominal_off")
    s02 = pick(lambda c: c.startswith("s02"))
    streak = pick(lambda c: c.startswith("s07"))
    spectrometer = pick(lambda c: c.startswith("fea"))
    return {
        "nominal": nominal,
        "+S02 quads": nominal + s02,
        "+streak": nominal + s02 + streak,
        "full": nominal + s02 + streak + spectrometer,
    }


def run(line, images_path, set_name="full", log=print):
    configs = full6d.make_plan(line)
    keys, images, n_meas, beam = ment6d.load_images(images_path)
    used = image_sets(keys)[set_name]
    data = {k: ment6d.measured_fractions(img, n_meas, k[1]) for k, img in zip(keys, images)}
    truth = ment6d.summary(beam)
    out = ment6d.reconstruct(
        line, configs, keys, images, n_meas, keys_used=used, n_sample=150_000, log=log,
        truth_summary=truth, refine_longitudinal=True,
        final_inflation=(1.1, 1.1, 1.05), final_sweeps=(20, 20, 40),
        inflation=(1.0, 1.3, 1.15, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1),
        sweeps=(8, 8, 10, 10, 15, 15, 15, 15, 20),
    )
    sample, w, _, _ = ment6d.resample_stages(line, configs, used, data, n_meas, out["sample"],
                                             out["weights"], stages=8, log=log, truth_summary=truth)
    best, grid = ment6d.chirp_scan(line, configs, used, data, n_meas, sample, w, log=log)
    sample, w, _, _ = ment6d.resample_stages(line, configs, used, data, n_meas, best,
                                             np.full(len(best), 1 / len(best)), stages=6, log=log,
                                             truth_summary=truth)
    np.savez_compressed(OUT_NPZ, sample=sample, weights=w)
    results = {"set": set_name, "n_images": len(used), "truth": truth,
               "reco": ment6d.summary(sample, w), "chirp_grid": grid}
    common.write_json(OUT_JSON, results, indent=1)
    return results


if __name__ == "__main__":
    run(full6d.build_line(), common.HERE / (sys.argv[1] if len(sys.argv) > 1 else "images.npz"))
