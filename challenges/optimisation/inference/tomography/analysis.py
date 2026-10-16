"""Conditioning and precision studies for sets of screens (no machine access)."""

from __future__ import annotations

import numpy as np
import screen_tomo as st


def predicted_rows(
    response: st.Response, screens: list[str], sigma: np.ndarray, eta=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design, offsets and noise-free predicted moments for a reference Sigma."""
    design, offset = [], []
    for screen in screens:
        rows, constant = st.moment_rows(response, screen, eta)
        design.append(rows)
        offset.append(constant)
    design, offset = np.vstack(design), np.concatenate(offset)
    return design, offset, design @ st.sigma_to_vector(sigma) + offset


def row_uncertainty(predicted: np.ndarray, relative: float) -> np.ndarray:
    """Relative error on xx, yy and on xy (relative to sqrt(xx yy)), as in the fit."""
    matrix = predicted.reshape(-1, 3)
    magnitude = np.column_stack(
        [matrix[:, 0], matrix[:, 1], np.sqrt(np.abs(matrix[:, 0] * matrix[:, 1]))]
    )
    return (np.abs(magnitude) * relative).ravel()


def precision(
    response: st.Response,
    screens: list[str],
    sigma: np.ndarray,
    beta_gamma: float,
    relative: float = 0.02,
    n_samples: int = 400,
    seed: int = 0,
) -> dict[str, float]:
    """Linear error propagation: rms scatter of the reconstructed metrics when every
    moment carries an independent relative error `relative`; plus the condition number."""
    design, _, predicted = predicted_rows(response, screens, sigma)
    sigma_rows = row_uncertainty(predicted, relative)
    kappa, singular = st.condition_number(design, sigma_rows)
    weighted = design / sigma_rows[:, None]
    normal = weighted.T @ weighted
    if np.linalg.matrix_rank(weighted) < 10:
        return {"kappa": np.inf, "n_screens": len(screens)}
    covariance = np.linalg.inv(normal)
    rng = np.random.default_rng(seed)
    vector = st.sigma_to_vector(sigma)
    draws = rng.multivariate_normal(vector, covariance, n_samples)
    collected: dict[str, list[float]] = {}
    for draw in draws:
        candidate = st.vector_to_sigma(draw)
        if np.any(np.linalg.eigvalsh(candidate) <= 0):
            continue
        for key, value in st.beam_metrics(candidate, beta_gamma).items():
            collected.setdefault(key, []).append(value)
    reference = st.beam_metrics(sigma, beta_gamma)
    result = {"kappa": kappa, "n_screens": len(screens), "sv_min": singular[-1]}
    for key in (
        "eps_n_x_um",
        "eps_n_y_um",
        "beta_x_m",
        "beta_y_m",
        "alpha_x",
        "alpha_y",
        "corr_xy",
    ):
        values = np.array(collected.get(key, [np.nan]))
        denominator = (
            abs(reference[key]) if key not in ("alpha_x", "alpha_y", "corr_xy") else 1.0
        )
        result[f"d_{key}"] = float(np.std(values) / denominator)
    return result
