"""End-to-end reconstruction helpers shared by the run scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import orbit_fit as of
import screen_tomo as st

import clara_virtual as cv

SCREEN_NOISE = 0.01  # relative noise of VirtualMachine.screen_moments on xx, yy, xy
CENTROID_NOISE_SCREEN_M = 2e-6
BPM_NOISE_M = 10e-6


def screen_groups(screens: list[str]) -> dict[str, list[str]]:
    """Screens grouped by machine section, in line order."""
    labels = {
        "S02": "_s02_",
        "S03": "_s03_",
        "S04": "_s04_",
        "S05": "_s05_",
        "VBC": "_vbc_",
        "S06": "_s06_",
        "S07": "_s07_",
        "FEA": "_fea_",
        "FEH": "_feh_",
    }
    return {
        label: [s for s in screens if tag in s]
        for label, tag in labels.items()
        if any(tag in s for s in screens)
    }


@dataclass
class NoiseFree:
    """Exact beam statistics at the screens and BPMs, from tracking the hidden beam
    (simulation-side only, to replay the machine's noise law for many-shot studies)."""

    moments: np.ndarray  # (S, 3) variances without resolution
    centroids: np.ndarray  # (S, 2)
    bpm_centroids: np.ndarray  # (B, 2)


def noise_free(
    machine: cv.VirtualMachine, screens: list[str], bpms: list[str]
) -> NoiseFree:
    beam = machine._shot_beam()
    tracked = machine._line.track_through(beam, list(screens) + list(bpms))
    moments, centroids = [], []
    for screen in screens:
        x, y = tracked[screen][:, 0], tracked[screen][:, 2]
        moments.append([x.var(), y.var(), np.cov(x, y)[0, 1]])
        centroids.append([x.mean(), y.mean()])
    bpm_centroids = [tracked[b][:, [0, 2]].mean(axis=0) for b in bpms]
    return NoiseFree(np.array(moments), np.array(centroids), np.array(bpm_centroids))


def replay(
    exact: NoiseFree,
    screens: list[str],
    shots: int,
    resolution_m: float,
    assumed_resolution_m: float,
    rng: np.random.Generator,
) -> tuple[st.ScreenData, np.ndarray]:
    """Screen data and BPM centroids as the machine would deliver them (same noise law),
    averaged over `shots`."""
    moments = np.zeros_like(exact.moments)
    for _ in range(shots):
        noise = 1 + rng.normal(0, SCREEN_NOISE, exact.moments.shape)
        measured = exact.moments * noise
        measured[:, :2] = (exact.moments[:, :2] + resolution_m**2) * noise[:, :2]
        moments += measured / shots
    moments[:, :2] -= assumed_resolution_m**2
    centroids = exact.centroids + rng.normal(
        0, CENTROID_NOISE_SCREEN_M / np.sqrt(shots), exact.centroids.shape
    )
    bpm = exact.bpm_centroids + rng.normal(
        0, BPM_NOISE_M / np.sqrt(shots), exact.bpm_centroids.shape
    )
    return st.ScreenData(list(screens), moments, centroids, shots), bpm.ravel()


def read_bpm_vector(
    machine: cv.VirtualMachine, bpms: list[str], shots: int = 3
) -> np.ndarray:
    readings = [machine.read_bpms(bpms) for _ in range(shots)]
    return np.mean([np.concatenate([r[b] for b in bpms]) for r in readings], axis=0)


def centroid_system(
    bpm_vector: np.ndarray, data: st.ScreenData, shots_bpm: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    readings = np.concatenate([bpm_vector, data.centroids.ravel()])
    noise = np.concatenate(
        [
            np.full(len(bpm_vector), BPM_NOISE_M / np.sqrt(shots_bpm)),
            np.full(data.centroids.size, CENTROID_NOISE_SCREEN_M / np.sqrt(data.shots)),
        ]
    )
    return readings, noise


def orbit_aware_response(
    model: cv.XsuiteLine,
    nodes: st.SliceNodes,
    screens: list[str],
    orbit: of.OrbitModel,
    readings: np.ndarray,
    noise: np.ndarray,
    plane: str | None = None,
    offset_prior: float = of.OFFSET_PRIOR,
) -> tuple[st.Response, np.ndarray]:
    """Slice response about the centroid-fitted trajectory, with the fitted quadrupole
    offsets installed on the model while tracking."""
    estimate, _ = of.solve_orbit(orbit, readings, noise, offset_prior=offset_prior)
    of.apply_offsets_to_model(model, orbit, estimate)
    try:
        response = st.slice_response(
            model, nodes, screens, launch=estimate[:5], plane=plane
        )
    finally:
        of.clear_model_offsets(model)
    return response, estimate


def metric_row(
    sigma: np.ndarray, truth: np.ndarray, beta_gamma: float
) -> dict[str, float]:
    estimated = st.beam_metrics(sigma, beta_gamma)
    exact = st.beam_metrics(truth, beta_gamma)
    row = {f"est_{k}": float(v) for k, v in estimated.items()}
    row.update({f"true_{k}": float(v) for k, v in exact.items()})
    row["sigma_error"] = st.relative_sigma_error(sigma, truth)
    for key in ("eps_n_x_um", "eps_n_y_um", "beta_x_m", "beta_y_m"):
        row[f"rel_{key}"] = float(estimated[key] / exact[key] - 1)
    for key in ("alpha_x", "alpha_y", "corr_xy"):
        row[f"abs_{key}"] = float(estimated[key] - exact[key])
    return row


def closure_xx_yy(
    response: st.Response, screens: list[str], truth: np.ndarray, exact: NoiseFree
) -> list[list[float]]:
    """Relative error of the predicted (xx, yy) for the true Sigma vs the exact tracked
    moments, per screen: the model-error floor of the forward model."""
    rows = []
    for index, screen in enumerate(screens):
        design, offset = st.moment_rows(response, screen)
        predicted = design @ st.sigma_to_vector(truth) + offset
        rows.append((predicted[:2] / exact.moments[index, :2] - 1).tolist())
    return rows


def extra_setting_response(
    model: cv.XsuiteLine,
    nodes: st.SliceNodes,
    screens: list[str],
    currents: dict[str, float],
    label: str,
    orbit: of.OrbitModel,
    estimate: np.ndarray,
    plane: str | None = None,
) -> st.Response:
    """Slice response of `screens` with some quadrupoles set to `currents` (A); the keys are
    renamed `screen@label`. The fitted quadrupole offsets stay installed during tracking."""
    of.apply_offsets_to_model(model, orbit, estimate)
    model.set_quad_currents(currents)
    try:
        response = st.slice_response(
            model, nodes, screens, launch=estimate[:5], plane=plane
        )
    finally:
        model.restore_nominal_quads()
        of.clear_model_offsets(model)
    response.transfer = {f"{k}@{label}": v for k, v in response.transfer.items()}
    response.orbit = {f"{k}@{label}": v for k, v in response.orbit.items()}
    return response


def merge_responses(base: st.Response, *extra: st.Response) -> st.Response:
    transfer, orbit = dict(base.transfer), dict(base.orbit)
    for response in extra:
        transfer.update(response.transfer)
        orbit.update(response.orbit)
    return st.Response(base.nodes, transfer, orbit)


def measure_extra_setting(
    machine: cv.VirtualMachine,
    screens: list[str],
    currents: dict[str, float],
    label: str,
    shots: int,
) -> st.ScreenData:
    machine.set_quad_currents(currents)
    try:
        data = st.measure(machine, screens, shots=shots)
    finally:
        machine.restore_nominal_quads()
    data.screens = [f"{s}@{label}" for s in screens]
    return data


def concatenate_data(first: st.ScreenData, *others: st.ScreenData) -> st.ScreenData:
    parts = [first, *others]
    return st.ScreenData(
        [s for p in parts for s in p.screens],
        np.vstack([p.moments for p in parts]),
        np.vstack([p.centroids for p in parts]),
        first.shots,
    )
