"""Centroid-only inference: launch trajectory, mean energy offset and quadrupole offsets.

Centroids (BPMs, or screen centroids) are linear in the launch (x, px, y, py), in the mean
relative energy offset (through the dispersion) and in the quadrupole misalignments (each
one is a kick of -k L * offset). They carry no information on the second moments, so the
emittance, beta and alpha cannot come from them; they do pin the orbit about which the
chromatic slice maps must be taken.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import clara_virtual as cv

LAUNCH_PRIOR = np.array([3e-4, 3e-5, 3e-4, 3e-5, 2e-3])  # x, px, y, py, delta
OFFSET_PRIOR = 1e-4  # quadrupole offset rms, m
OFFSET_STEP = 5e-5


@dataclass
class OrbitModel:
    stops: list[str]
    quads: list[str]
    baseline: np.ndarray  # (2 * stops,) x, y at every stop for the on-axis reference
    jacobian: np.ndarray  # (2 * stops, 5 + 2 * quads)

    @property
    def parameter_names(self) -> list[str]:
        return (
            ["x0", "px0", "y0", "py0", "delta0"]
            + [f"{q}.dx" for q in self.quads]
            + [f"{q}.dy" for q in self.quads]
        )


def _centroids(model: cv.XsuiteLine, start: np.ndarray, stops: list[str]) -> np.ndarray:
    recorded = model.track_through(start[None, :], stops)
    return np.concatenate([recorded[s][0, [0, 2]] for s in stops])


def build_orbit_model(
    model: cv.XsuiteLine, stops: list[str], quads: list[str] | None = None
) -> OrbitModel:
    """Baseline orbit and Jacobian of the centroids at `stops` by finite differences of the
    xsuite line (the model is left unchanged)."""
    quads = model.inventory.quads if quads is None else quads
    zero = np.zeros(6)
    baseline = _centroids(model, zero, stops)
    columns = []
    for index, step in enumerate([1e-6, 1e-7, 1e-6, 1e-7, 1e-5]):
        start = zero.copy()
        start[[0, 1, 2, 3, 5][index]] = step
        columns.append((_centroids(model, start, stops) - baseline) / step)
    for attribute in ("shift_x", "shift_y"):
        for quad in quads:
            element = model.line[quad]
            original = getattr(element, attribute)
            setattr(element, attribute, original + OFFSET_STEP)
            columns.append((_centroids(model, zero, stops) - baseline) / OFFSET_STEP)
            setattr(element, attribute, original)
    return OrbitModel(list(stops), list(quads), baseline, np.column_stack(columns))


def prior_vector(orbit: OrbitModel, offset_prior: float = OFFSET_PRIOR) -> np.ndarray:
    return np.concatenate([LAUNCH_PRIOR, np.full(2 * len(orbit.quads), offset_prior)])


def solve_orbit(
    orbit: OrbitModel,
    readings: np.ndarray,
    noise_m: np.ndarray | float,
    offset_prior: float = OFFSET_PRIOR,
    free_offsets: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Ridge (Bayesian) estimate of the parameters from centroid readings; returns the
    estimate and its posterior covariance. `free_offsets=False` fits launch + energy only."""
    prior = prior_vector(orbit, offset_prior)
    keep = np.ones(len(prior), bool) if free_offsets else np.arange(len(prior)) < 5
    jac = orbit.jacobian[:, keep]
    sigma = np.broadcast_to(noise_m, readings.shape)
    weighted = np.vstack([jac / sigma[:, None], np.diag(1 / prior[keep])])
    target = np.concatenate([(readings - orbit.baseline) / sigma, np.zeros(keep.sum())])
    solution, *_ = np.linalg.lstsq(weighted, target, rcond=None)
    covariance = np.linalg.inv(weighted.T @ weighted)
    full = np.zeros(len(prior))
    full[keep] = solution
    full_cov = np.zeros((len(prior), len(prior)))
    full_cov[np.ix_(keep, keep)] = covariance
    return full, full_cov


def apply_offsets_to_model(
    model: cv.XsuiteLine, orbit: OrbitModel, estimate: np.ndarray
) -> None:
    """Install the estimated quadrupole offsets on the model line."""
    count = len(orbit.quads)
    for index, quad in enumerate(orbit.quads):
        model.line[quad].shift_x = float(estimate[5 + index])
        model.line[quad].shift_y = float(estimate[5 + count + index])


def clear_model_offsets(model: cv.XsuiteLine) -> None:
    for quad in model.inventory.quads:
        model.line[quad].shift_x = 0.0
        model.line[quad].shift_y = 0.0
