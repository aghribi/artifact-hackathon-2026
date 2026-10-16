"""Linear observability of the 6D second-moment matrix from screen moments.

Unknowns: the 21 independent elements of the 6x6 covariance at the reconstruction point,
scaled by sqrt(sigma_ii sigma_jj) of the true beam (so a diagonal unknown is a relative
variance and an off-diagonal one a correlation coefficient). Each inserted screen gives
<xx>, <yy>, <xy>, which are linear in the 21 unknowns through the 6x6 map M from the
reconstruction point to the screen:  <uv> = sum_ij M_ui M_vj sigma_ij.
"""

from __future__ import annotations

import common
import numpy as np
import tdc

RELATIVE_NOISE = 0.01  # the virtual screen puts 1 % on each moment


def moment_rows(transfer: np.ndarray) -> np.ndarray:
    """(3, 21) rows for <xx>, <yy>, <xy> at a screen with 6x6 map `transfer`."""
    rows = np.zeros((3, 21))
    for r, (u, v) in enumerate(((0, 0), (2, 2), (0, 2))):
        for c, (i, j) in enumerate(common.PAIR_INDEX):
            rows[r, c] = transfer[u, i] * transfer[v, j] + (
                transfer[u, j] * transfer[v, i] if i != j else 0.0
            )
    return rows


def column_scales(sigma: np.ndarray) -> np.ndarray:
    return np.array([np.sqrt(sigma[i, i] * sigma[j, j]) for i, j in common.PAIR_INDEX])


def scaled_system(maps_list, sigma_true: np.ndarray):
    """Stack scaled rows. maps_list: list of 6x6 maps (one per measurement). Returns the
    matrix A (rows divided by the 1 % noise of the predicted moment, columns by the unknown
    scale) so that the Fisher information is A^T A."""
    scales = column_scales(sigma_true)
    blocks = []
    for transfer in maps_list:
        rows = moment_rows(transfer)
        predicted = rows @ common.matrix_to_unique(sigma_true)
        noise = np.array(
            [
                RELATIVE_NOISE * abs(predicted[0]),
                RELATIVE_NOISE * abs(predicted[1]),
                RELATIVE_NOISE * np.sqrt(abs(predicted[0] * predicted[1])),
            ]
        )
        blocks.append((rows * scales) / noise[:, None])
    return np.vstack(blocks)


def summarise(system: np.ndarray) -> dict:
    singular = np.linalg.svd(system, compute_uv=False)
    rank_tolerance = singular[0] * 1e-8
    rank = int((singular > rank_tolerance).sum())
    fisher = system.T @ system
    uncertainty = np.full(21, np.inf)
    if rank == 21:
        uncertainty = np.sqrt(np.diag(np.linalg.inv(fisher)))
    return {
        "singular_values": singular,
        "rank": rank,
        "condition": singular[0] / singular[-1] if singular[-1] > 0 else np.inf,
        "uncertainty": uncertainty,
    }


def per_unknown_observable(system: np.ndarray) -> np.ndarray:
    """For each unknown, the uncertainty obtained when the others are marginalised, using the
    pseudo-inverse (finite only if the unknown is not in the null space): an unknown is
    individually observable if its unit vector has no component in the null space."""
    _, singular, vt = np.linalg.svd(system, full_matrices=True)
    rank = int((singular > singular[0] * 1e-8).sum())
    null = vt[rank:]
    leak = np.linalg.norm(null, axis=0)  # null-space weight on each unknown
    return leak


def maps_for(model, screens: list[str], voltage_v: float, phase_rad: float = 0.0):
    tdc.set_tdc(model.line, voltage_v, phase_rad)
    return model.linear_map(screens)


def name_of(index: int) -> str:
    i, j = common.PAIR_INDEX[index]
    return f"<{common.LABELS[i]} {common.LABELS[j]}>"


def derived_quantities(sigma: np.ndarray) -> dict[str, float]:
    """Quantities whose precision matters: projected emittances (geometric) and the
    longitudinal rms emittance, beam sizes, chirp."""
    return {
        "eps_x": float(np.sqrt(np.linalg.det(sigma[0:2, 0:2]))),
        "eps_y": float(np.sqrt(np.linalg.det(sigma[2:4, 2:4]))),
        "eps_z": float(np.sqrt(np.linalg.det(sigma[4:6, 4:6]))),
        "sigma_tau": float(np.sqrt(sigma[4, 4])),
        "sigma_delta": float(np.sqrt(sigma[5, 5])),
        "chirp": float(sigma[4, 5] / sigma[4, 4]),
    }


def derived_uncertainty(system: np.ndarray, sigma_true: np.ndarray) -> dict[str, float]:
    """Relative 1-sigma of derived quantities from the Fisher matrix (linear propagation).
    Needs a full-rank, well enough conditioned system."""
    scales = column_scales(sigma_true)
    covariance_theta = np.linalg.pinv(system.T @ system, rcond=1e-14)
    base = common.matrix_to_unique(sigma_true)
    reference = derived_quantities(sigma_true)
    out = {}
    for key, value in reference.items():
        gradient = np.zeros(21)
        for c in range(21):
            step = 1e-6 * scales[c]
            plus = base.copy()
            plus[c] += step
            minus = base.copy()
            minus[c] -= step
            gradient[c] = (
                derived_quantities(common.unique_to_matrix(plus))[key]
                - derived_quantities(common.unique_to_matrix(minus))[key]
            ) / (2 * step)
        gradient_scaled = gradient * scales
        out[key] = float(
            np.sqrt(gradient_scaled @ covariance_theta @ gradient_scaled) / abs(value)
        )
    return out
