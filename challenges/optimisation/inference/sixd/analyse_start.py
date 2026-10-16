"""Fisher analysis at the S02 start from observe_start.npz (exact Jacobian, 1 % noise)."""

from __future__ import annotations

import common
import numpy as np
import observability as obs


def main() -> None:
    data = np.load(common.HERE / "observe_start.npz")
    jac, reference, labels, sigma0 = (data[k] for k in ("jac", "reference", "labels", "sigma0"))
    names = [str(label) for label in labels]
    config_of = np.repeat([n.split("|")[0] for n in names], 3)
    ref = reference.reshape(-1, 3)
    rms_x, rms_y = np.sqrt(ref[:, 0]), np.sqrt(np.maximum(ref[:, 1], 0))
    imageable_row = np.repeat((rms_x < 4e-3) & (rms_y < 4e-3), 3)
    noise = np.repeat(0.01 * np.column_stack([ref[:, 0], ref[:, 1], np.sqrt(np.abs(ref[:, 0] * ref[:, 1]))]), 1, axis=0).reshape(-1)
    scaled = jac / noise[:, None]

    def subset(config_names, only_imageable=False):
        mask = np.isin(config_of, config_names)
        if only_imageable:
            mask &= imageable_row
        return scaled[mask], int(mask.sum() // 3)

    s02 = ["s02a", "s02b", "s02c"]
    designed = sorted({c for c in config_of if c.endswith(("_off", "_plus", "_minus")) and c != "nominal_off"})
    streak = [c for c in designed if not c.startswith("fea")]
    spectrometer = [c for c in designed if c.startswith("fea")]
    no_tdc = [c for c in designed if c.endswith("_off")]
    sets = {
        "nominal, S02-S05 screens": subset(["nominal_off"], True),
        "+ 3 S02 quad settings": subset(["nominal_off", *s02], True),
        "+ designed S07 settings, TDC off": subset(["nominal_off", *s02, *no_tdc], True),
        "+ streak screens, TDC off/+/-": subset(["nominal_off", *s02, *streak], True),
        "+ spectrometer screen, TDC off/+/- (full plan)": subset(["nominal_off", *s02, *streak, *spectrometer], True),
        "full plan without S02 quad settings": subset(["nominal_off", *streak, *spectrometer], True),
    }
    names21 = [obs.name_of(c) for c in range(21)]
    out = {}
    for label, (system, n_screens) in sets.items():
        summary = obs.summarise(system)
        leak = obs.per_unknown_observable(system)
        derived = obs.derived_uncertainty(system, sigma0) if summary["rank"] == 21 else {}
        print(f"\n{label}: {n_screens} images, rank {summary['rank']}, cond {summary['condition']:.2e}")
        print("  null-space unknowns:", [names21[c] for c in range(21) if leak[c] > 1e-6])
        if derived:
            print("  derived rel. 1-sigma:", {k: round(v, 3) for k, v in derived.items()})
            print("  worst scaled unknowns:", [(names21[c], round(float(summary['uncertainty'][c]), 3)) for c in np.argsort(-summary['uncertainty'])[:6]])
        out[label] = {"rank": summary["rank"], "cond": summary["condition"], "n": n_screens,
                      "derived": derived, "unc": summary["uncertainty"].tolist()}
    common.write_json(common.HERE / "analyse_start.json", out, indent=1)


if __name__ == "__main__":
    main()
