"""Independence of the transverse beam and delta assumed at S07 instead of the start."""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import orbit_fit as of
import pipeline as pl
import screen_tomo as st

import clara_virtual as cv

xm = cv.xsuite_model
PLANE = xm.RECONSTRUCTION_POINT
machine, model, beam = cv.build_pair(beam_particles=20000)
long_sample, energy_ev, _ = xm.load_distribution()
with np.load(HERE / "s07_beam.npz") as cache:
    long07 = cache["particles"][:, 4:6]
beta_gamma = float(np.sqrt((energy_ev / xm.ELECTRON_REST_ENERGY_EV) ** 2 - 1))
screens, bpms = model.inventory.screens, model.inventory.bpms
groups = pl.screen_groups(screens)
nodes_start = st.delta_quadrature(long_sample[:, 4], long_sample[:, 5], 9)
nodes_07 = st.delta_quadrature(long07[:, 0], long07[:, 1], 9)
truth = st.true_sigma(beam)
orbit = of.build_orbit_model(model, bpms + screens)
data0 = st.measure(machine, screens, shots=3)
bpm_vec = pl.read_bpm_vector(machine, bpms, 3)
readings, noise = pl.centroid_system(bpm_vec, data0, 3)
settings = [(-11.26, 26.30), (-10.93, 27.14), (-8.37, 18.02), (-9.16, 16.00)]
quads = ["cla_s02_mag_quad_01", "cla_s02_mag_quad_02"]
out = {}
for plane_name, plane, nodes in (
    ("start", None, nodes_start),
    ("S07", PLANE, nodes_07),
):
    resp, estimate = pl.orbit_aware_response(
        model, nodes, screens, orbit, readings, noise, plane=plane
    )
    data = data0
    labels_extra = []
    for j, (a, b) in enumerate(settings):
        cur = {quads[0]: a, quads[1]: b}
        d = pl.measure_extra_setting(machine, groups["S02"], cur, f"x{j}", 3)
        r = pl.extra_setting_response(
            model, nodes, groups["S02"], cur, f"x{j}", orbit, estimate, plane=plane
        )
        resp = pl.merge_responses(resp, r)
        data = pl.concatenate_data(data, d)
        labels_extra += d.screens
    for subset_name, sub in (
        ("23 nominal", list(screens)),
        ("S02 + 4 settings", groups["S02"] + labels_extra),
        ("all + 4 settings", list(screens) + labels_extra),
    ):
        fit = st.fit_sigma(resp, data, sub, 0.02)
        sigma_start = (
            fit.sigma
            if plane is None
            else st.transport_sigma(model, fit.sigma, long07, plane, None)
        )
        row = pl.metric_row(sigma_start, truth, beta_gamma)
        out[f"{plane_name}: {subset_name}"] = {
            k: round(row[k], 3)
            for k in (
                "est_eps_n_x_um",
                "est_eps_n_y_um",
                "est_beta_x_m",
                "est_beta_y_m",
                "est_alpha_x",
                "est_alpha_y",
                "est_corr_xy",
                "sigma_error",
            )
        } | {"chi2": round(fit.chi2_per_dof, 2), "kappa": round(fit.condition_number)}
        print(
            f"{plane_name}: {subset_name}",
            out[f"{plane_name}: {subset_name}"],
            flush=True,
        )
(HERE / "results_plane.json").write_text(json.dumps(out, indent=1))
