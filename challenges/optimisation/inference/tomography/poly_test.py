"""Energy-dependent Sigma(delta) model on the augmented S02 scan."""

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
machine, model, beam = cv.build_pair(beam_particles=20000)
long_sample, energy_ev, _ = xm.load_distribution()
beta_gamma = float(np.sqrt((energy_ev / xm.ELECTRON_REST_ENERGY_EV) ** 2 - 1))
screens, bpms = model.inventory.screens, model.inventory.bpms
groups = pl.screen_groups(screens)
nodes = st.delta_quadrature(long_sample[:, 4], long_sample[:, 5], 9)
truth = st.true_sigma(beam)
orbit = of.build_orbit_model(model, bpms + screens)
data0 = st.measure(machine, screens, shots=3)
bpm_vec = pl.read_bpm_vector(machine, bpms, 3)
readings, noise = pl.centroid_system(bpm_vec, data0, 3)
resp, estimate = pl.orbit_aware_response(model, nodes, screens, orbit, readings, noise)
quads = ["cla_s02_mag_quad_01", "cla_s02_mag_quad_02"]
settings = [(-11.26, 26.30), (-10.93, 27.14), (-8.37, 18.02), (-9.16, 16.00)]
data, extra = data0, []
for j, (a, b) in enumerate(settings):
    cur = {quads[0]: a, quads[1]: b}
    d = pl.measure_extra_setting(machine, groups["S02"], cur, f"x{j}", 3)
    resp = pl.merge_responses(
        resp,
        pl.extra_setting_response(
            model, nodes, groups["S02"], cur, f"x{j}", orbit, estimate
        ),
    )
    data = pl.concatenate_data(data, d)
    extra += d.screens
import json

rows = {}
sub = groups["S02"] + extra
for order_m, ridge in [(0, 1.0)] + [
    (o, r) for o in (1, 2) for r in (10.0, 30.0, 100.0, 1000.0)
]:
    reps = []
    for rep in range(6):
        d0 = st.measure(machine, groups["S02"], shots=3)
        dd = d0
        for j, (a, b) in enumerate(settings):
            cur = {quads[0]: a, quads[1]: b}
            dd = pl.concatenate_data(
                dd, pl.measure_extra_setting(machine, groups["S02"], cur, f"x{j}", 3)
            )
        projected, mats, chi2 = st.fit_polynomial_sigma(
            resp, dd, sub, order_m, 0.02, ridge
        )
        row = pl.metric_row(projected, truth, beta_gamma)
        reps.append(
            [
                row[k]
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
            ]
            + [chi2]
        )
    reps = np.array(reps)
    rows[f"order={order_m} ridge={ridge}"] = {
        "mean": reps.mean(0).round(3).tolist(),
        "std": reps.std(0).round(3).tolist(),
    }
    print(
        f"order={order_m} ridge={ridge}",
        "mean",
        reps.mean(0).round(3),
        "std",
        reps.std(0).round(3),
        flush=True,
    )
Path(HERE / "results_poly.json").write_text(json.dumps(rows, indent=1))
