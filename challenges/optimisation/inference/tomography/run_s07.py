"""Reconstruction at cla_s07_sim_aper_01 from the S07 and FEA/FEH screens, nominal optics."""

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
FLOOR = 0.02
DISP_GAIN = 0.3
RNG = np.random.default_rng(777)


def log(message: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {message}", flush=True)


machine, model, beam, longitudinal, energy_ev = st.build_pair_at_s07()
beta_gamma = float(np.sqrt((energy_ev / xm.ELECTRON_REST_ENERGY_EV) ** 2 - 1))
screens = model.inventory.screens
bpms = model.inventory.bpms
groups = pl.screen_groups(screens)
truth = st.true_sigma(beam)
nodes = st.delta_quadrature(longitudinal[:, 0], longitudinal[:, 1], 9)
orbit = of.build_orbit_model(model, bpms + screens)
log(f"built, {len(screens)} screens, {len(bpms)} bpms")
results: dict = {
    "truth": {k: float(v) for k, v in st.beam_metrics(truth, beta_gamma).items()}
}

data = st.measure(machine, screens, shots=3)
bpm_vector = pl.read_bpm_vector(machine, bpms, 3)
readings, noise = pl.centroid_system(bpm_vector, data)
sizes_mm = np.sqrt(data.moments[:, :2]) * 1e3
results["screens"] = {
    s: {
        "s": model.inventory.s_at[s],
        "sigma_x_mm": float(sizes_mm[i, 0]),
        "sigma_y_mm": float(sizes_mm[i, 1]),
        "usable": bool(sizes_mm[i].max() < 4.0),
    }
    for i, s in enumerate(screens)
}
resp_ach = st.slice_response(model, st.single_node(), screens)
resp_chrom = st.slice_response(model, nodes, screens)
resp_orbit, estimate = pl.orbit_aware_response(
    model, nodes, screens, orbit, readings, noise
)
results["orbit_estimate"] = {
    "launch_est": estimate[:5].tolist(),
    "launch_true": machine._errors.launch[[0, 1, 2, 3, 5]].tolist(),
}
subsets = {
    "S07 screens (5)": groups["S07"],
    "FEA+FEH (5)": groups["FEA"] + groups["FEH"],
    "S07+FEA (9)": groups["S07"] + groups["FEA"],
    "all 10": list(screens),
}
base = {}
for name, sub in subsets.items():
    out = {}
    for vname, resp, gain in (
        ("achromatic", resp_ach, 0.0),
        ("chromatic", resp_chrom, 0.0),
        ("chromatic+orbit", resp_orbit, 0.0),
        ("chromatic+orbit+weights", resp_orbit, DISP_GAIN),
    ):
        fit = st.fit_sigma(resp, data, sub, FLOOR, dispersive_gain=gain)
        row = pl.metric_row(fit.sigma, truth, beta_gamma)
        row.update(kappa=fit.condition_number, chi2=fit.chi2_per_dof)
        out[vname] = row
    base[name] = out
    log(
        f"{name}: "
        + ", ".join(
            f"{k}: ex={v['est_eps_n_x_um']:.1f} ey={v['est_eps_n_y_um']:.1f}"
            for k, v in out.items()
        )
    )
results["baseline"] = base

exact = pl.noise_free(machine, screens, bpms)
results["closure_true_sigma_xx_yy"] = {
    label: pl.closure_xx_yy(resp, screens, truth, exact)
    for label, resp in (("chromatic", resp_chrom), ("chromatic+orbit", resp_orbit))
}
reference = st.fit_sigma(resp_orbit, data, None, FLOOR, dispersive_gain=DISP_GAIN).sigma
results["conditioning"] = {
    name: an.precision(resp_orbit, sub, reference, beta_gamma, 0.02)
    for name, sub in subsets.items()
}
results["conditioning_leave_one_out"] = {
    s: an.precision(
        resp_orbit, [x for x in screens if x != s], reference, beta_gamma, 0.02
    )
    for s in screens
}

shots = {}
for n in (1, 3, 10, 30):
    rows = []
    for _ in range(12):
        d, bpm_v = pl.replay(
            exact,
            screens,
            n,
            machine.screen_resolution_m,
            machine.screen_resolution_m,
            RNG,
        )
        rd, nz = pl.centroid_system(bpm_v, d, n)
        resp_o, _ = pl.orbit_aware_response(model, nodes, screens, orbit, rd, nz)
        rows.append(
            pl.metric_row(
                st.fit_sigma(resp_o, d, None, FLOOR, dispersive_gain=DISP_GAIN).sigma,
                truth,
                beta_gamma,
            )
        )
    shots[n] = {
        k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))]
        for k in (
            "rel_eps_n_x_um",
            "rel_eps_n_y_um",
            "rel_beta_x_m",
            "rel_beta_y_m",
            "abs_corr_xy",
            "sigma_error",
        )
    }
results["shots"] = shots
log("shots done")

energy = {}
d10, b10 = pl.replay(
    exact, screens, 10, machine.screen_resolution_m, machine.screen_resolution_m, RNG
)
rd, nz = pl.centroid_system(b10, d10, 10)
for scale in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7):
    resp, _ = pl.orbit_aware_response(
        model, nodes.scaled(scale), screens, orbit, rd, nz
    )
    entry = {}
    for name, sub in subsets.items():
        fit = st.fit_sigma(resp, d10, sub, FLOOR)
        row = pl.metric_row(fit.sigma, truth, beta_gamma)
        entry[name] = {
            "chi2_total": fit.chi2_per_dof * max(3 * len(sub) - 10, 1),
            "rows": 3 * len(sub),
            "rel_eps_n_x_um": row["rel_eps_n_x_um"],
            "rel_eps_n_y_um": row["rel_eps_n_y_um"],
            "sigma_error": row["sigma_error"],
        }
    energy[f"{scale:g}"] = entry
results["energy_spread"] = {
    "sigma_delta_nominal": float(longitudinal[:, 1].std()),
    "scan": energy,
}
log("energy done")

hidden = {}
for offset_rms in (0.0, 1e-4, 3e-4):
    errs = st.scaled_errors(model.inventory, 0, 1.0, offset_rms)
    st.apply_errors(machine, errs)
    ex = pl.noise_free(machine, screens, bpms)
    d, bv = pl.replay(ex, screens, 3, 20e-6, 20e-6, RNG)
    rd, nz = pl.centroid_system(bv, d, 3)
    for aware in (False, True):
        resp = (
            pl.orbit_aware_response(model, nodes, screens, orbit, rd, nz)[0]
            if aware
            else resp_chrom
        )
        row = pl.metric_row(
            st.fit_sigma(
                resp, d, None, FLOOR, dispersive_gain=DISP_GAIN if aware else 0
            ).sigma,
            truth,
            beta_gamma,
        )
        hidden[f"offset={offset_rms * 1e6:.0f}um orbit_aware={aware}"] = {
            k: row[k]
            for k in (
                "rel_eps_n_x_um",
                "rel_eps_n_y_um",
                "rel_beta_x_m",
                "rel_beta_y_m",
                "abs_corr_xy",
                "sigma_error",
            )
        }
results["hidden"] = hidden
st.apply_errors(machine, st.scaled_errors(model.inventory, 0, 1.0, 1e-4))
resolution = {}
for res_um in (0, 20, 50, 100):
    machine.screen_resolution_m = res_um * 1e-6
    ex = pl.noise_free(machine, screens, bpms)
    d, bv = pl.replay(ex, screens, 3, res_um * 1e-6, res_um * 1e-6, RNG)
    rd, nz = pl.centroid_system(bv, d, 3)
    resp = pl.orbit_aware_response(model, nodes, screens, orbit, rd, nz)[0]
    row = pl.metric_row(
        st.fit_sigma(resp, d, None, FLOOR, dispersive_gain=DISP_GAIN).sigma,
        truth,
        beta_gamma,
    )
    resolution[f"res={res_um}um"] = {
        k: row[k]
        for k in (
            "rel_eps_n_x_um",
            "rel_eps_n_y_um",
            "rel_beta_x_m",
            "rel_beta_y_m",
            "abs_corr_xy",
            "sigma_error",
        )
    }
results["resolution"] = resolution
with open(HERE / "results_s07.json", "w") as handle:
    json.dump(results, handle, indent=1, default=float)
log("saved")
