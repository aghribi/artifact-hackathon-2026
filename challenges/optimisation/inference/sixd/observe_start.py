"""Observability of the 21 second moments at the S02 start from screen moments, exact
(nonlinear) Jacobian: a fixed whitened sample is tracked through the whole xsuite line for
every measurement configuration, so chromatic effects, cavities and the TDC are included."""

from __future__ import annotations

import common
import full6d
import numpy as np
import observability as obs

cv = common.cv
xsuite_model = common.xsuite_model
STEP = 1e-6
PARTICLES = 3000


def sqrt_psd(sigma: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(sigma)
    return (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T


def main() -> None:
    line = full6d.build_line()
    beam, _, _ = xsuite_model.load_distribution()
    sigma0 = np.cov(beam.T)
    configs = full6d.make_plan(line)
    rng = np.random.default_rng(2)
    white = rng.normal(size=(PARTICLES, 6))
    white -= white.mean(0)
    white = white @ np.linalg.inv(np.linalg.cholesky(np.cov(white.T))).T

    def moments(sigma: np.ndarray) -> np.ndarray:
        sample = white @ sqrt_psd(sigma).T
        out = []
        for config in configs:
            full6d.apply_config(line, config)
            tracked = line.track_through(sample, config.screens)
            for s in config.screens:
                x, y = tracked[s][:, 0], tracked[s][:, 2]
                out += [x.var(), y.var(), np.mean((x - x.mean()) * (y - y.mean()))]
        line.restore_nominal_quads()
        return np.array(out)

    labels = [(c.name, s) for c in configs for s in c.screens]
    reference = moments(sigma0)
    scales = obs.column_scales(sigma0)
    jac = np.zeros((len(reference), 21))
    for c, (i, j) in enumerate(common.PAIR_INDEX):
        plus, minus = sigma0.copy(), sigma0.copy()
        delta = STEP * scales[c]
        plus[i, j] += delta
        minus[i, j] -= delta
        if i != j:
            plus[j, i] += delta
            minus[j, i] -= delta
        jac[:, c] = (moments(plus) - moments(minus)) / (2 * STEP)
        print("column", c, flush=True)
    np.savez(
        common.HERE / "observe_start.npz",
        jac=jac, reference=reference, labels=np.array([f"{c}|{s}" for c, s in labels]),
        sigma0=sigma0,
    )


if __name__ == "__main__":
    main()
