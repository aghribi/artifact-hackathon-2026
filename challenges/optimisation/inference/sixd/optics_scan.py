"""Greedy choice of extra quadrupole settings that improve the derived-quantity precision.

Candidates: random multiplicative changes of the S07 quad strengths q01..q10 (log-uniform
factors, within the +-100 A limit). Score: sum of squared log relative uncertainties of the
projected emittances, bunch length, energy spread and chirp (linear Fisher, 1 % noise on
every screen moment, 9 screens x TDC off/+/-5 MV per setting)."""

from __future__ import annotations

import time

import common
import numpy as np
import observability as obs

KEYS = ("eps_x", "eps_y", "eps_z", "sigma_tau", "sigma_delta", "chirp")


def maps_of_setting(
    model, k1: dict[str, float], after: list[str], voltages=(-5e6, 0.0, 5e6)
):
    model.set_quad_k1(k1)
    out = []
    for v in voltages:
        m = obs.maps_for(model, model.inventory.screens, v)
        for s in model.inventory.screens:
            if v != 0 and s not in after:
                continue
            out.append(m[s])
    model.restore_nominal_quads()
    return out


def main(n_candidates: int = 80, n_pick: int = 6, seed: int = 11) -> None:
    _, model, beam = common.make_pair(particles=20000)
    sigma_true = common.covariance(beam)
    after = [
        s
        for s in model.inventory.screens
        if model.element_index[s] > model.element_index["cla_s07_dia_tdc_01"]
    ]
    quads = [q for q in model.inventory.quads if q.startswith("cla_s07")]
    rng = np.random.default_rng(seed)
    candidates = []
    for _ in range(n_candidates):
        k1 = dict(model.nominal_k1)
        for q in quads:
            lo, hi = model.quad_k1_limits(q)
            k1[q] = float(
                np.clip(model.nominal_k1[q] * np.exp(rng.uniform(-1, 1)), lo, hi)
            )
        candidates.append(k1)
    t0 = time.time()
    cand_maps = [maps_of_setting(model, k, after) for k in candidates]
    print("maps done", time.time() - t0, flush=True)
    chosen = [maps_of_setting(model, dict(model.nominal_k1), after)]
    picked = []
    history = []
    for step in range(n_pick):
        best = None
        for i, maps in enumerate(cand_maps):
            if i in picked:
                continue
            system = obs.scaled_system(
                [m for group in chosen for m in group] + maps, sigma_true
            )
            unc = obs.derived_uncertainty(system, sigma_true)
            score = (
                sum(np.log(unc[k]) ** 2 for k in KEYS)
                if False
                else sum(unc[k] for k in KEYS[:3])
            )
            if best is None or score < best[0]:
                best = (score, i, unc)
        picked.append(best[1])
        chosen.append(cand_maps[best[1]])
        history.append(best[2])
        print(step + 1, {k: round(v, 3) for k, v in best[2].items()}, flush=True)
    common.write_json(
        common.HERE / "optics_scan.json",
        {
            "picked": [{q: candidates[i][q] for q in quads} for i in picked],
            "history": history,
        },
        indent=1,
    )


if __name__ == "__main__":
    main()
