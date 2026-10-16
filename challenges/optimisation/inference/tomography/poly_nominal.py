"""Energy-dependent Sigma(delta) model on the nominal-optics screens only."""

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
machine, model, beam = cv.build_pair(beam_particles=20000)
long_sample, energy_ev, _ = xm.load_distribution()
beta_gamma = float(np.sqrt((energy_ev / xm.ELECTRON_REST_ENERGY_EV) ** 2 - 1))
screens, bpms = model.inventory.screens, model.inventory.bpms
groups = pl.screen_groups(screens)
nodes = st.delta_quadrature(long_sample[:, 4], long_sample[:, 5], 9)
truth = st.true_sigma(beam)
orbit = of.build_orbit_model(model, bpms + screens)
exact = pl.noise_free(machine, screens, bpms)
rng = np.random.default_rng(5)
rows = {}
subsets = {
    "S02-S05 (imageable)": [s for g in ("S02", "S03", "S04", "S05") for s in groups[g]],
    "all 23": list(screens),
}
keys = (
    "est_eps_n_x_um",
    "est_eps_n_y_um",
    "est_beta_x_m",
    "est_beta_y_m",
    "est_alpha_x",
    "est_alpha_y",
    "est_corr_xy",
    "sigma_error",
)
for shots in (3, 30):
    reps = {}
    for rep in range(6):
        d, bv = pl.replay(exact, screens, shots, 20e-6, 20e-6, rng)
        rd, nz = pl.centroid_system(bv, d, shots)
        resp, _ = pl.orbit_aware_response(model, nodes, screens, orbit, rd, nz)
        for name, sub in subsets.items():
            for order, ridge in ((0, 1.0), (1, 30.0), (1, 100.0)):
                proj, _, chi2 = st.fit_polynomial_sigma(
                    resp, d, sub, order, 0.02, ridge
                )
                row = pl.metric_row(proj, truth, beta_gamma)
                reps.setdefault(f"{name} order={order} ridge={ridge}", []).append(
                    [row[k] for k in keys]
                )
    for label, vals in reps.items():
        arr = np.array(vals)
        rows[f"shots={shots} {label}"] = {
            "mean": arr.mean(0).round(3).tolist(),
            "std": arr.std(0).round(3).tolist(),
        }
        print(
            f"shots={shots} {label}",
            arr.mean(0).round(2),
            arr.std(0).round(2),
            flush=True,
        )
(HERE / "results_poly_nominal.json").write_text(json.dumps(rows, indent=1))
