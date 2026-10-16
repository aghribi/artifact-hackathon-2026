"""Where does the vertical-emittance bias of the augmented S02 scan come from?"""

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
rng = np.random.default_rng(3)
machine, model, beam = cv.build_pair(end="cla_s02_dia_scr_03", beam_particles=20000)
long_sample, energy_ev, _ = xm.load_distribution()
beta_gamma = float(np.sqrt((energy_ev / xm.ELECTRON_REST_ENERGY_EV) ** 2 - 1))
screens, bpms = model.inventory.screens, model.inventory.bpms
nodes = st.delta_quadrature(long_sample[:, 4], long_sample[:, 5], 9)
truth = st.true_sigma(beam)
orbit = of.build_orbit_model(model, bpms + screens)
quads = ["cla_s02_mag_quad_01", "cla_s02_mag_quad_02"]
settings = [(-11.26, 26.30), (-10.93, 27.14), (-8.37, 18.02), (-9.16, 16.00), (0, 0)]
synth = beam.copy()
synth[:, :4] = (np.linalg.cholesky(truth) @ rng.normal(size=(4, len(beam)))).T
zero = cv.HiddenErrors(quad_offsets={}, launch=np.zeros(6))
real = machine._errors
out = {}
for beam_name, hidden in (("true", beam), ("synth", synth)):
    for err_name, errs in (("zero", zero), ("real", real)):
        st.apply_errors(machine, errs)
        machine._beam = hidden
        data = None
        resp = None
        for j, (a, b) in enumerate(settings):
            cur = {quads[0]: a, quads[1]: b} if (a or b) else {}
            lab = f"s{j}"
            if cur:
                machine.set_quad_currents(cur)
            exact_tracked = machine._line.track_through(machine._shot_beam(), screens)
            machine.restore_nominal_quads()
            mom = np.array(
                [
                    [
                        exact_tracked[s][:, 0].var(),
                        exact_tracked[s][:, 2].var(),
                        np.cov(exact_tracked[s][:, 0], exact_tracked[s][:, 2])[0, 1],
                    ]
                    for s in screens
                ]
            )
            cen = np.array(
                [
                    [exact_tracked[s][:, 0].mean(), exact_tracked[s][:, 2].mean()]
                    for s in screens
                ]
            )
            mom = mom * (1 + rng.normal(0, 0.01 / np.sqrt(3), mom.shape))
            d = st.ScreenData([f"{s}@{lab}" for s in screens], mom, cen, 3)
            r = pl.extra_setting_response(
                model,
                nodes,
                screens,
                cur,
                lab,
                orbit,
                np.zeros(orbit.jacobian.shape[1]),
            )
            data = d if data is None else pl.concatenate_data(data, d)
            resp = r if resp is None else pl.merge_responses(resp, r)
        for n in (1, 3, 5):
            labels = (
                [
                    x
                    for x in data.screens
                    if int(x.split("@s")[1]) < n or x.endswith("@s4")
                ]
                if False
                else [x for x in data.screens if int(x.split("@s")[1]) < n]
            )
            fit = st.fit_sigma(resp, data, labels, 0.02)
            row = pl.metric_row(fit.sigma, truth, beta_gamma)
            out[f"{beam_name} beam, {err_name} errors, {n} settings"] = {
                k: round(row[k], 3)
                for k in (
                    "est_eps_n_x_um",
                    "est_eps_n_y_um",
                    "rel_beta_y_m",
                    "abs_alpha_y",
                    "sigma_error",
                )
            }
            print(
                f"{beam_name} {err_name} n={n}",
                out[f"{beam_name} beam, {err_name} errors, {n} settings"],
                flush=True,
            )
(HERE / "results_bias.json").write_text(json.dumps(out, indent=1))
