"""Run the reconstruction demo for the screen-only baseline and the baseline + extra setting."""

from __future__ import annotations

import sys
import time

import common
import numpy as np
import reconstruct as rec


def main() -> None:
    noerr = "--noerr" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    machine, model, beam = common.make_pair(particles=20000, hidden_errors=not noerr)
    sigma_true = common.covariance(beam)
    screens = model.inventory.screens
    results = {"true": sigma_true}
    for label in names or ["baseline", "q01q02", "scan"]:
        settings = rec.quad_settings(model, label)
        t0 = time.time()
        shots = rec.acquire(machine, model, settings, screens)
        sigma_lin = rec.linear_reconstruction(model, shots, settings)
        sigma_lin_psd = rec.project_psd(sigma_lin)
        fit = rec.NonlinearFit(model, shots, settings, sigma_lin_psd)
        sigma_nl, result = fit.solve(sigma_lin_psd)
        chi2 = float(np.mean(result.fun**2))
        print(
            f"\n=== {label}: {len(shots)} shots, chi2/dof-ish {chi2:.2f}, "
            f"nfev {result.nfev}, {time.time() - t0:.0f} s"
        )
        truth = rec.summary(sigma_true)
        lin = rec.summary(sigma_lin_psd)
        nl = rec.summary(sigma_nl)
        print(f"{'quantity':40s} {'true':>10s} {'linear':>10s} {'nonlinear':>10s}")
        for key in truth:
            print(f"{key:40s} {truth[key]:10.4g} {lin[key]:10.4g} {nl[key]:10.4g}")
        results[label] = {
            "linear": sigma_lin_psd,
            "nonlinear": sigma_nl,
            "shots": shots,
        }
        results[label]["chi2"] = chi2
    suffix = "_noerr" if noerr else ""
    common.write_pickle(
        common.HERE / f"demo_results_{'_'.join(names)}{suffix}.pkl", results
    )


if __name__ == "__main__":
    main()
