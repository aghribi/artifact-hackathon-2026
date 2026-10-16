"""Reconstruction at cla_s02_sim_aper_01 (the benchmark start) from all screens, nominal optics."""

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
import xsuite_model as xm

import clara_virtual as cv

N_NODES = 9
FLOOR = 0.02
DISP_GAIN = 0.3
USABLE_SIZE_MM = 4.0
RNG = np.random.default_rng(12345)


def log(message: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {message}", flush=True)


T0 = time.time()
machine, model, beam = cv.build_pair(beam_particles=20000)
long_sample, energy_ev, _ = xm.load_distribution()
beta_gamma = float(np.sqrt((energy_ev / xm.ELECTRON_REST_ENERGY_EV) ** 2 - 1))
screens = model.inventory.screens
bpms = model.inventory.bpms
groups = pl.screen_groups(screens)
truth = st.true_sigma(beam)
nodes = st.delta_quadrature(long_sample[:, 4], long_sample[:, 5], N_NODES)
orbit = of.build_orbit_model(model, bpms + screens)
log("built")
results: dict = {
    "truth": {k: float(v) for k, v in st.beam_metrics(truth, beta_gamma).items()}
}


def cumulative(names: list[str]) -> list[str]:
    return [s for g in names for s in groups[g]]


SUBSETS = {
    "S02 (3 screens)": cumulative(["S02"]),
    "S02+S03": cumulative(["S02", "S03"]),
    "S02-S05 (imageable)": cumulative(["S02", "S03", "S04", "S05"]),
    "S02-S06": cumulative(["S02", "S03", "S04", "S05", "VBC", "S06"]),
    "S02-S07": cumulative(["S02", "S03", "S04", "S05", "VBC", "S06", "S07"]),
    "all 23": list(screens),
}


def run_variants(data, bpm_vector, subset_screens, tag=""):
    """Fit the four forward-model variants with the same data."""
    readings, noise = pl.centroid_system(bpm_vector, data)
    out = {}
    resp_ach = st.slice_response(model, st.single_node(), screens)
    resp_chrom = st.slice_response(model, nodes, screens)
    resp_orbit, estimate = pl.orbit_aware_response(
        model, nodes, screens, orbit, readings, noise
    )
    variants = {
        "achromatic": (resp_ach, 0.0),
        "chromatic": (resp_chrom, 0.0),
        "chromatic+orbit": (resp_orbit, 0.0),
        "chromatic+orbit+weights": (resp_orbit, DISP_GAIN),
    }
    for name, (resp, gain) in variants.items():
        try:
            fit = st.fit_sigma(resp, data, subset_screens, FLOOR, dispersive_gain=gain)
        except np.linalg.LinAlgError:
            continue
        row = pl.metric_row(fit.sigma, truth, beta_gamma)
        row.update(kappa=fit.condition_number, chi2=fit.chi2_per_dof)
        out[name] = row
    return out, (resp_chrom, resp_orbit, estimate)


# --- 1. baseline with the real machine API (3 shots per screen) -----------------------
data = st.measure(machine, screens, shots=3)
bpm_vector = pl.read_bpm_vector(machine, bpms, 3)
sizes_mm = np.sqrt(data.moments[:, :2]) * 1e3
results["screens"] = {
    s: {
        "s": model.inventory.s_at[s],
        "energy_mev": model.inventory.momentum_mevc_at[s],
        "sigma_x_mm": float(sizes_mm[i, 0]),
        "sigma_y_mm": float(sizes_mm[i, 1]),
        "usable": bool(sizes_mm[i].max() < USABLE_SIZE_MM),
    }
    for i, s in enumerate(screens)
}
usable = [s for s in screens if results["screens"][s]["usable"]]
SUBSETS["usable only (sizes < 4 mm)"] = usable
results["baseline"] = {}
for name, subset in SUBSETS.items():
    out, (resp_chrom, resp_orbit, estimate) = run_variants(data, bpm_vector, subset)
    results["baseline"][name] = out
    log(
        f"baseline {name}: "
        + ", ".join(
            f"{k}: ey={v['est_eps_n_y_um']:.2f} ex={v['est_eps_n_x_um']:.2f}"
            for k, v in out.items()
        )
    )
results["orbit_estimate"] = {
    "launch_est": estimate[:5].tolist(),
    "launch_true": (machine._errors.launch[[0, 1, 2, 3, 5]]).tolist(),
    "s02_offsets_est_um": (estimate[5:10] * 1e6).tolist(),
    "s02_offsets_true_um": [
        machine._errors.quad_offsets[q][0] * 1e6 for q in orbit.quads[:5]
    ],
}

# closure of the forward model for the true Sigma (noise free)
exact = pl.noise_free(machine, screens, bpms)
closure = {}
for label, resp in (("chromatic", resp_chrom), ("chromatic+orbit", resp_orbit)):
    rows = []
    for i, s in enumerate(screens):
        a, c = st.moment_rows(resp, s)
        pred = a @ st.sigma_to_vector(truth) + c
        rows.append((pred[:2] / exact.moments[i, :2] - 1).tolist())
    closure[label] = rows
results["closure_true_sigma_xx_yy"] = closure
log("closure done")

# --- 2. conditioning and precision per subset of screens ------------------------------
reference = st.fit_sigma(resp_orbit, data, None, FLOOR, dispersive_gain=DISP_GAIN).sigma
cond = {}
cum_names = []
for g in groups:
    cum_names.append(g)
    sub = cumulative(cum_names)
    cond["+".join(cum_names[-1:]) + f" (n={len(sub)})"] = an.precision(
        resp_orbit, sub, reference, beta_gamma, 0.02
    )
    cond[list(cond)[-1]]["group"] = g
single = {
    g: an.precision(resp_orbit, groups[g], reference, beta_gamma, 0.02) for g in groups
}
loo = {
    s: an.precision(
        resp_orbit, [x for x in screens if x != s], reference, beta_gamma, 0.02
    )
    for s in screens
}
results["conditioning_cumulative"] = cond
results["conditioning_single_group"] = single
results["conditioning_leave_one_out"] = loo
results["conditioning_all"] = an.precision(
    resp_orbit, screens, reference, beta_gamma, 0.02
)
# marginal information: reduction of the predicted eps_y error when a group is removed from all
results["conditioning_drop_group"] = {
    g: an.precision(
        resp_orbit,
        [s for s in screens if s not in groups[g]],
        reference,
        beta_gamma,
        0.02,
    )
    for g in groups
}
log("conditioning done")

# --- 3. number of shots (noise replay, K runs each) ------------------------------------
shots_table = {}
for shots in (1, 3, 10, 30, 100):
    rows = []
    for _ in range(12):
        d, bpm_v = pl.replay(
            exact,
            screens,
            shots,
            machine.screen_resolution_m,
            machine.screen_resolution_m,
            RNG,
        )
        rd, nz = pl.centroid_system(bpm_v, d, shots)
        resp_o, _ = pl.orbit_aware_response(model, nodes, screens, orbit, rd, nz)
        fit = st.fit_sigma(resp_o, d, None, FLOOR, dispersive_gain=DISP_GAIN)
        rows.append(pl.metric_row(fit.sigma, truth, beta_gamma))
    shots_table[shots] = {
        key: [
            float(np.mean([r[key] for r in rows])),
            float(np.std([r[key] for r in rows])),
        ]
        for key in (
            "rel_eps_n_x_um",
            "rel_eps_n_y_um",
            "rel_beta_x_m",
            "rel_beta_y_m",
            "abs_alpha_x",
            "abs_alpha_y",
            "abs_corr_xy",
            "sigma_error",
        )
    }
    log(f"shots {shots}: {shots_table[shots]['rel_eps_n_y_um']}")
results["shots"] = shots_table


# --- 4. robustness: hidden errors and resolution --------------------------------------
def run_case(errors, resolution_m, assumed_m, orbit_aware=True, shots=3):
    st.apply_errors(machine, errors)
    machine.screen_resolution_m = resolution_m
    ex = pl.noise_free(machine, screens, bpms)
    d, bpm_v = pl.replay(ex, screens, shots, resolution_m, assumed_m, RNG)
    rd, nz = pl.centroid_system(bpm_v, d, shots)
    if orbit_aware:
        resp, _ = pl.orbit_aware_response(model, nodes, screens, orbit, rd, nz)
    else:
        resp = st.slice_response(model, nodes, screens)
    fit = st.fit_sigma(
        resp, d, None, FLOOR, dispersive_gain=DISP_GAIN if orbit_aware else 0.0
    )
    return pl.metric_row(fit.sigma, truth, beta_gamma)


keys = (
    "rel_eps_n_x_um",
    "rel_eps_n_y_um",
    "rel_beta_x_m",
    "rel_beta_y_m",
    "abs_corr_xy",
    "sigma_error",
)
hidden = {}
for offset_rms in (0.0, 1e-4, 3e-4):
    for launch_scale in (0.0, 1.0, 3.0):
        errs = st.scaled_errors(model.inventory, 0, launch_scale, offset_rms)
        for aware in (False, True):
            row = run_case(errs, 20e-6, 20e-6, aware)
            hidden[
                f"offset={offset_rms * 1e6:.0f}um launch x{launch_scale:g} orbit_aware={aware}"
            ] = {k: row[k] for k in keys}
log("hidden grid done")
seeds = {}
for aware in (False, True):
    rows = []
    for seed in range(1, 9):
        errs = st.scaled_errors(model.inventory, seed, 1.0, 1e-4)
        rows.append(run_case(errs, 20e-6, 20e-6, aware))
    seeds[f"orbit_aware={aware}"] = {
        k: [
            float(np.mean([r[k] for r in rows])),
            float(np.std([r[k] for r in rows])),
            float(np.max(np.abs([r[k] for r in rows]))),
        ]
        for k in keys
    }
results["hidden_grid"] = hidden
results["hidden_seeds"] = seeds
log("seeds done")

default_errors = st.scaled_errors(model.inventory, 0, 1.0, 1e-4)
resolution = {}
for res_um in (0, 20, 50, 100, 200):
    for assumed_label, assumed in (("known", res_um * 1e-6), ("assumed 20um", 20e-6)):
        row = run_case(default_errors, res_um * 1e-6, assumed)
        resolution[f"res={res_um}um {assumed_label}"] = {k: row[k] for k in keys}
results["resolution"] = resolution
log("resolution done")

# --- 5. energy spread sensitivity and identifiability -----------------------------------
st.apply_errors(machine, default_errors)
machine.screen_resolution_m = 20e-6
exact = pl.noise_free(machine, screens, bpms)
data_e, bpm_e = pl.replay(exact, screens, 10, 20e-6, 20e-6, RNG)
rd, nz = pl.centroid_system(bpm_e, data_e, 10)
energy = {}
scales = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7]
profile_subsets = {
    "all 23": list(screens),
    "S02-S05": SUBSETS["S02-S05 (imageable)"],
    "S02-S07": SUBSETS["S02-S07"],
    "S07+FEA+FEH": cumulative(["S07", "FEA", "FEH"]),
    "FEA+FEH (dispersive)": cumulative(["FEA", "FEH"]),
}
for scale in scales:
    scaled = nodes.scaled(scale)
    resp, _ = pl.orbit_aware_response(model, scaled, screens, orbit, rd, nz)
    entry = {}
    for name, sub in profile_subsets.items():
        fit = st.fit_sigma(resp, data_e, sub, FLOOR)
        row = pl.metric_row(fit.sigma, truth, beta_gamma)
        entry[name] = {
            "chi2_total": fit.chi2_per_dof * max(3 * len(sub) - 10, 1),
            "rows": 3 * len(sub),
            "rel_eps_n_x_um": row["rel_eps_n_x_um"],
            "rel_eps_n_y_um": row["rel_eps_n_y_um"],
            "sigma_error": row["sigma_error"],
        }
    energy[f"{scale:g}"] = entry
    log(f"energy scale {scale}")
results["energy_spread"] = {
    "sigma_delta_nominal": float(long_sample[:, 5].std()),
    "scan": energy,
}

with open(HERE / "results_s02.json", "w") as handle:
    json.dump(results, handle, indent=1, default=float)
log("saved")
