"""Model-consistency test and the optional few-quad augmentation (S02 quadrupoles)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import analysis as an
import orbit_fit as of
import pipeline as pl
import screen_tomo as st

import clara_virtual as cv

xm = cv.xsuite_model
T0 = time.time()
FLOOR, DISP_GAIN = 0.02, 0.3
rng = np.random.default_rng(99)
machine, model, beam = cv.build_pair(beam_particles=20000)
long_sample, energy_ev, _ = xm.load_distribution()
beta_gamma = float(np.sqrt((energy_ev / xm.ELECTRON_REST_ENERGY_EV) ** 2 - 1))
screens, bpms = model.inventory.screens, model.inventory.bpms
nodes = st.delta_quadrature(long_sample[:, 4], long_sample[:, 5], 9)
truth = st.true_sigma(beam)
orbit = of.build_orbit_model(model, bpms + screens)
groups = pl.screen_groups(screens)
s02_screens = groups["S02"]
results: dict = {}


def fit_all(data, bpm_vec, label_screens=None, shots=3):
    readings, noise = pl.centroid_system(
        bpm_vec, data if label_screens is None else data_nominal_only(data), shots
    )
    return readings, noise


def data_nominal_only(data):
    keep = [i for i, s in enumerate(data.screens) if "@" not in s]
    return st.ScreenData(
        [data.screens[i] for i in keep],
        data.moments[keep],
        data.centroids[keep],
        data.shots,
    )


machine._beam = beam

# 2. augmentation of the S02 scan with a few extra settings of quad_01/quad_02
ex = pl.noise_free(machine, screens, bpms)
data0 = st.measure(machine, screens, shots=3)
bpm_vec = pl.read_bpm_vector(machine, bpms, 3)
readings, noise = pl.centroid_system(bpm_vec, data0, 3)
resp0, estimate = pl.orbit_aware_response(model, nodes, screens, orbit, readings, noise)
reference = st.fit_sigma(resp0, data0, None, FLOOR, dispersive_gain=DISP_GAIN).sigma
quad_names = ["cla_s02_mag_quad_01", "cla_s02_mag_quad_02"]
candidates = []
for index in range(120):
    amps = rng.uniform(-45, 45, 2)
    currents = {q: float(a) for q, a in zip(quad_names, amps, strict=True)}
    resp = pl.extra_setting_response(
        model, nodes, s02_screens, currents, f"c{index}", orbit, estimate
    )
    valid = []
    for s in s02_screens:
        a, c = st.moment_rows(resp, f"{s}@c{index}")
        pred = a @ st.sigma_to_vector(reference) + c
        if 0.1e-3 < np.sqrt(max(pred[:2].max(), 0)) < 3.5e-3 and pred[:2].min() > 0:
            valid.append(f"{s}@c{index}")
    if valid:
        candidates.append((index, currents, resp, valid))
print("valid candidates", len(candidates), flush=True)
assert candidates
chosen, merged = [], resp0
current_screens = list(screens)
for _ in range(4):
    best = None
    for index, currents, resp, valid in candidates:
        if index in [c[0] for c in chosen]:
            continue
        trial = pl.merge_responses(merged, resp)
        labels = current_screens + valid
        prec = an.precision(trial, labels, reference, beta_gamma, 0.02, n_samples=150)
        score = prec.get("d_eps_n_y_um", np.inf)
        if best is None or score < best[0]:
            best = (score, index, currents, resp, prec)
    if best is None:
        break
    score, index, currents, resp, prec = best
    valid = next(c[3] for c in candidates if c[0] == index)
    chosen.append((index, currents, valid))
    merged = pl.merge_responses(merged, resp)
    current_screens += valid
    print(
        "chosen",
        index,
        currents,
        "pred d_eps_y",
        score,
        "kappa",
        prec["kappa"],
        flush=True,
    )
    results.setdefault("augmentation_design", []).append(
        {
            "currents_A": currents,
            "predicted_rel_err_eps_y": score,
            "kappa": prec["kappa"],
            "n_screens": prec["n_screens"],
        }
    )
results["design_baseline"] = an.precision(
    resp0, screens, reference, beta_gamma, 0.02, n_samples=150
)

# real measurements of the chosen settings
data_all = data0
for index, currents, _ in chosen:
    data_all = pl.concatenate_data(
        data_all,
        pl.measure_extra_setting(machine, s02_screens, currents, f"c{index}", 3),
    )
rows = {}
for n_extra in range(len(chosen) + 1):
    labels = list(screens) + [s for _, _, v in chosen[:n_extra] for s in v]
    fit = st.fit_sigma(merged, data_all, labels, FLOOR, dispersive_gain=DISP_GAIN)
    row = pl.metric_row(fit.sigma, truth, beta_gamma)
    rows[f"nominal 23 + {n_extra} extra S02 settings"] = {
        k: row[k]
        for k in (
            "est_eps_n_x_um",
            "est_eps_n_y_um",
            "rel_eps_n_x_um",
            "rel_eps_n_y_um",
            "rel_beta_y_m",
            "abs_alpha_y",
            "abs_corr_xy",
            "sigma_error",
        )
    } | {"kappa": fit.condition_number, "chi2": fit.chi2_per_dof}
    s02_labels = list(s02_screens) + [s for _, _, v in chosen[:n_extra] for s in v]
    fit2 = (
        st.fit_sigma(merged, data_all, s02_labels, FLOOR)
        if len(s02_labels) >= 4
        else None
    )
    if fit2 is not None:
        row2 = pl.metric_row(fit2.sigma, truth, beta_gamma)
        rows[f"S02 screens only, {n_extra} extra settings"] = {
            k: row2[k]
            for k in (
                "est_eps_n_x_um",
                "est_eps_n_y_um",
                "rel_eps_n_x_um",
                "rel_eps_n_y_um",
                "rel_beta_y_m",
                "abs_alpha_y",
                "abs_corr_xy",
                "sigma_error",
            )
        } | {"kappa": fit2.condition_number, "chi2": fit2.chi2_per_dof}
results["augmentation_fits"] = rows
for k, v in rows.items():
    print(k, {a: round(b, 3) for a, b in v.items()}, flush=True)
with open(HERE / "results_augment_design.json", "w") as handle:
    json.dump(results, handle, indent=1, default=float)
print("saved", time.time() - T0)
