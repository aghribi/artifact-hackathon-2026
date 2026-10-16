"""What BPM centroids alone can and cannot give (no screen moments)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import orbit_fit as of
import pipeline as pl

import clara_virtual as cv

machine, model, beam = cv.build_pair(beam_particles=20000)
bpms = model.inventory.bpms
orbit = of.build_orbit_model(model, bpms)
errors = machine._errors
results: dict = {}

# 1. orbit fits from BPM readings only
readings = pl.read_bpm_vector(machine, bpms, 5)
noise = np.full(len(readings), pl.BPM_NOISE_M / np.sqrt(5))
launch_true = errors.launch[[0, 1, 2, 3, 5]]
true_offset_rms_um = float(
    np.sqrt(
        np.mean(np.array([v for q in errors.quad_offsets.values() for v in q]) ** 2)
    )
    * 1e6
)
for label, free in (
    ("launch+delta only", False),
    ("launch+delta+quad offsets (ridge)", True),
):
    estimate, covariance = of.solve_orbit(orbit, readings, noise, free_offsets=free)
    residual = orbit.jacobian @ estimate + orbit.baseline - readings
    results[label] = {
        "orbit_residual_rms_um": float(np.sqrt(np.mean(residual**2)) * 1e6),
        "launch_est": estimate[:5].tolist(),
        "launch_true": launch_true.tolist(),
        "launch_posterior_std": np.sqrt(np.diag(covariance)[:5]).tolist(),
        "quad_offset_rms_est_um": float(np.sqrt(np.mean(estimate[5:] ** 2)) * 1e6)
        if free
        else None,
        "quad_offset_rms_true_um": true_offset_rms_um,
    }
# dispersion pattern of the model at the BPMs (per unit delta) and launch/energy degeneracy
delta_column = orbit.jacobian[:, 4] * 1.0
results["dispersion_at_bpms_m"] = {
    b: [float(delta_column[2 * i]), float(delta_column[2 * i + 1])]
    for i, b in enumerate(bpms)
}
covariance_launch = of.solve_orbit(orbit, readings, noise, free_offsets=False)[1]
results["delta_mean_std_launch_only_fit"] = float(np.sqrt(covariance_launch[4, 4]))
results["delta_mean_true"] = float(errors.launch[5] + beam[:, 5].mean())

# 2. centroids do not depend on the second moments
wide = beam.copy()
wide[:, :4] = wide[:, :4] * np.array([1.5, 1.5, 0.5, 0.5])
exact_a = pl.noise_free(machine, [], bpms).bpm_centroids
machine._beam = wide
exact_b = pl.noise_free(machine, [], bpms).bpm_centroids
machine._beam = beam
results["centroid_change_when_sigma_changes_um_rms"] = float(
    np.sqrt(np.mean((exact_a - exact_b) ** 2)) * 1e6
)

# 3. corrector response check of the transfer map (model validation, beam independent)
corrector = "cla_s02_mag_hvcor_01"
kick = 2e-4
base = pl.read_bpm_vector(machine, bpms, 5)
machine.set_corrector(corrector, kick, kick)
shifted = pl.read_bpm_vector(machine, bpms, 5)
machine.set_corrector(corrector, 0.0, 0.0)
measured = shifted - base


def model_response(scale: float) -> np.ndarray:
    original = {q: model.line[q].k1 for q in model.inventory.quads}
    for q in model.inventory.quads[:5]:
        model.line[q].k1 = original[q] * scale
    reference = of._centroids(model, np.zeros(6), bpms)
    model.set_corrector(corrector, kick, kick)
    deflected = of._centroids(model, np.zeros(6), bpms)
    model.set_corrector(corrector, 0.0, 0.0)
    for q, k in original.items():
        model.line[q].k1 = k
    return deflected - reference


sensitivity = {}
for scale in (0.9, 0.97, 0.99, 1.0, 1.01, 1.03, 1.1):
    difference = model_response(scale) - measured
    sensitivity[f"S02 quads x{scale}"] = float(
        np.sqrt(np.mean(difference**2)) / np.sqrt(np.mean(measured**2))
    )
results["corrector_response_relative_rms_mismatch"] = sensitivity
results["corrector_response_noise_floor"] = float(
    pl.BPM_NOISE_M * np.sqrt(2 / 5) / np.sqrt(np.mean(measured**2))
)
with open(HERE / "results_bpm.json", "w") as handle:
    json.dump(results, handle, indent=1, default=float)
print(
    json.dumps(
        {k: v for k, v in results.items() if k != "dispersion_at_bpms_m"},
        indent=1,
        default=float,
    )
)
