"""Observability tables for several measurement sets. Saves results to observability.npz."""

from __future__ import annotations

import common
import numpy as np
import observability as obs

VOLTAGE = 5e6


def main() -> None:
    _, model, beam = common.make_pair(particles=20000)
    sigma_true = common.covariance(beam)
    screens = model.inventory.screens
    print("screens", screens)
    tdc_name = "cla_s07_dia_tdc_01"
    order = model.element_index
    after_tdc = [s for s in screens if order[s] > order[tdc_name]]
    before_tdc = [s for s in screens if order[s] < order[tdc_name]]
    dispersive = [s for s in screens if "_fea_" in s]
    non_dispersive = [s for s in screens if "_fea_" not in s]

    maps = {}
    for voltage in (-VOLTAGE, 0.0, VOLTAGE):
        maps[voltage] = obs.maps_for(model, screens, voltage)

    def collect(screen_list, voltages):
        out = []
        for s in screen_list:
            for v in voltages:
                if v != 0.0 and s in before_tdc:
                    continue
                if v == 0.0 or s in after_tdc:
                    out.append(maps[v][s])
        return out

    configs = {
        "A1 non-dispersive screens, TDC off": collect(non_dispersive, [0.0]),
        "A2 all screens, TDC off": collect(screens, [0.0]),
        "A3 non-dispersive screens, TDC off/+/-": collect(
            non_dispersive, [0.0, VOLTAGE, -VOLTAGE]
        ),
        "A4 all screens, TDC off/+/-  (baseline)": collect(
            screens, [0.0, VOLTAGE, -VOLTAGE]
        ),
        "A5 all screens, TDC off/+ only": collect(screens, [0.0, VOLTAGE]),
        "A6 FEA screens only, TDC off/+/-": collect(
            dispersive, [0.0, VOLTAGE, -VOLTAGE]
        ),
    }
    results = {}
    for name, map_list in configs.items():
        system = obs.scaled_system(map_list, sigma_true)
        summary = obs.summarise(system)
        leak = obs.per_unknown_observable(system)
        results[name] = (summary, leak, len(map_list))
        print(
            f"\n{name}: {len(map_list)} screens, rank {summary['rank']}/21, "
            f"cond {summary['condition']:.2e}"
        )
        unobservable = [obs.name_of(c) for c in range(21) if leak[c] > 1e-6]
        print("  unobservable-component unknowns:", unobservable)
        if summary["rank"] == 21:
            print("  1-sigma uncertainty of scaled unknowns:")
            for c in range(21):
                print(f"    {obs.name_of(c):14s} {summary['uncertainty'][c]:.3g}")
    np.savez(
        common.HERE / "observability.npz",
        **{
            f"sv_{key[:2]}": value[0]["singular_values"]
            for key, value in results.items()
        },
        **{f"unc_{key[:2]}": value[0]["uncertainty"] for key, value in results.items()},
        **{f"leak_{key[:2]}": value[1] for key, value in results.items()},
    )
    common.write_json(
        common.HERE / "observability_keys.json", {k[:2]: k for k in configs}
    )


if __name__ == "__main__":
    main()
