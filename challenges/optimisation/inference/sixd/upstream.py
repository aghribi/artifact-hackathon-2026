"""Can anything upstream of the TDC (S02 start, 35.8 MeV, no TDC, flat chicane) see the
longitudinal plane?  Two tests on the S02 beam:

1. linear: stack the 6x6 maps (about the on-axis reference) to every screen from S02 to S06
   and compute the rank of the moment operator.
2. nonlinear: Jacobian of the screen moments w.r.t. the 21 covariance elements, using a
   fixed whitened sample tracked through the full xsuite line (chromatic effects and the
   energy-dependent RF focusing included)."""

from __future__ import annotations

import common
import numpy as np
import observability as obs
import xsuite_model

END = "cla_s06_dia_scr_02"


def sqrt_psd(sigma: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(sigma)
    return (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T


def main() -> None:
    beam, energy_ev, _ = xsuite_model.load_distribution()
    line = common.cv.XsuiteLine(xsuite_model.build_line(energy_ev, end=END))
    screens = [s for s in line.inventory.screens]
    print("screens upstream:", screens)
    sigma0 = np.cov(beam.T)
    print(
        f"S02 sigma_tau mm {1e3 * np.sqrt(sigma0[4, 4]):.3f} sigma_delta % {100 * np.sqrt(sigma0[5, 5]):.3f}"
    )

    # 1. linear rank
    maps = line.linear_map(screens)
    system = obs.scaled_system([maps[s] for s in screens], sigma0)
    sv = np.linalg.svd(system, compute_uv=False)
    print(
        "linear: rank",
        int((sv > sv[0] * 1e-8).sum()),
        "of 21, sv",
        np.round(sv[:12] / sv[0], 8),
    )

    # 2. nonlinear Jacobian
    rng = np.random.default_rng(2)
    white = rng.normal(size=(4000, 6))
    white -= white.mean(0)
    white = white @ np.linalg.inv(np.linalg.cholesky(np.cov(white.T))).T

    def moments(sigma: np.ndarray) -> np.ndarray:
        sample = white @ sqrt_psd(sigma).T
        tracked = line.track_through(sample, screens)
        out = []
        for s in screens:
            x, y = tracked[s][:, 0], tracked[s][:, 2]
            out += [x.var(), y.var(), np.mean((x - x.mean()) * (y - y.mean()))]
        return np.array(out)

    reference = moments(sigma0)
    scales = obs.column_scales(sigma0)
    step = 1e-6
    jac = np.zeros((len(reference), 21))
    for c, (i, j) in enumerate(common.PAIR_INDEX):
        plus = sigma0.copy()
        minus = sigma0.copy()
        delta = step * scales[c]
        plus[i, j] += delta
        minus[i, j] -= delta
        if i != j:
            plus[j, i] += delta
            minus[j, i] -= delta
        jac[:, c] = (moments(plus) - moments(minus)) / (2 * step)
    noise = 0.01 * np.maximum(np.abs(reference), 1e-30)
    scaled = jac / noise[:, None]
    sv_nl = np.linalg.svd(scaled, compute_uv=False)
    print("nonlinear: singular values (info per unit scaled change, noise-normalised):")
    print(np.array2string(sv_nl, precision=3))
    fisher = scaled.T @ scaled
    cov = np.linalg.pinv(fisher, rcond=1e-12)
    sigma_unknown = np.sqrt(np.diag(cov))
    print("1-sigma scaled uncertainty (inf-like = unobservable):")
    for c in range(21):
        print(f"  {obs.name_of(c):14s} {sigma_unknown[c]:.3g}")
    np.savez(
        common.HERE / "upstream.npz",
        sv_linear=sv,
        sv_nonlinear=sv_nl,
        unc=sigma_unknown,
    )


if __name__ == "__main__":
    main()
