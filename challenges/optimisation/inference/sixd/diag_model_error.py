"""How well do the three predictors reproduce the machine moments when given the TRUE
covariance: (a) linear maps, (b) Gaussian sample through xsuite, (c) the true particles
through the error-free model (isolates the hidden errors)?"""

import common
import numpy as np
import observability as obs
import reconstruct as rec

machine, model, beam = common.make_pair(particles=20000)
sigma = common.covariance(beam)
settings = rec.quad_settings(model, "baseline")
shots = rec.acquire(machine, model, settings, model.inventory.screens)
fit = rec.NonlinearFit(model, shots, settings, sigma)
measured = rec.shot_vector(shots)
theta = fit.theta_from_covariance(sigma)
gaussian = fit.predict(theta)
# true particles through the model
tracked = {}
out = []
for v in (0.0, 5e6, -5e6):
    common.tdc.set_tdc(model.line, v)
    tracked[v] = model.track_through(beam, model.inventory.screens)
for s in shots:
    c = tracked[s["voltage"]][s["screen"]]
    x, y = c[:, 0], c[:, 2]
    out.append(
        [x.var() + 4e-10, y.var() + 4e-10, np.mean((x - x.mean()) * (y - y.mean()))]
    )
particles = np.array(out)
lin = []
cache = {}
for s in shots:
    if s["voltage"] not in cache:
        cache[s["voltage"]] = obs.maps_for(model, model.inventory.screens, s["voltage"])
    lin.append(
        obs.moment_rows(cache[s["voltage"]][s["screen"]])
        @ common.matrix_to_unique(sigma)
    )
lin = np.array(lin)
print(
    f"{'screen':22s} V  | rel. error vs machine: linear / gaussian / true-particles-in-model   (xx yy)"
)
for k, s in enumerate(shots):

    def f(p, k=k):
        return " ".join(
            f"{(p[k, i] - measured[k, i]) / measured[k, i]:+7.3f}" for i in (0, 1)
        )

    print(
        f"{s['screen']:22s} {s['voltage'] / 1e6:+.0f} | {f(lin)} | {f(gaussian)} | {f(particles)}"
    )
chi = lambda p: np.mean(((p - measured) / fit.noise) ** 2)
print(
    f"mean chi2 (1 pct noise): linear {chi(lin):.1f} gaussian {chi(gaussian):.1f} particles {chi(particles):.1f}"
)
