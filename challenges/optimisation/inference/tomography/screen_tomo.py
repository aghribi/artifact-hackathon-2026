"""Screen-based 4D tomography with an energy-sliced xsuite forward model.

The nominal optics stay fixed. Every inserted YAG screen is one measurement of the
second moments (xx, yy, xy) of the beam at a different transport map from the
reconstruction point, so the set of screens is the "scan".

Forward model (all maps from xsuite tracking of the error-free design line):
the longitudinal distribution (tau, delta) at the reconstruction point is known from the
model. A Gauss quadrature in delta gives K slices; for every slice a reference particle
(tau_k, delta_k) is tracked through the line (cavities included) and the 4x4 transverse
map R_k and the off-momentum orbit D_k are taken at each screen by finite differences.
With the transverse beam independent of delta,

    S_screen = sum_k w_k (R_k Sigma R_k^T + D_k D_k^T) - m m^T,   m = sum_k w_k D_k,

which is linear in the ten unknowns of the 4x4 Sigma at the reconstruction point.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xsuite_model as xm

import clara_virtual as cv

PARAMETER_PAIRS = [(i, j) for i in range(4) for j in range(i, 4)]
PARAMETER_NAMES = [
    f"{['x', 'px', 'y', 'py'][i]}*{['x', 'px', 'y', 'py'][j]}"
    for i, j in PARAMETER_PAIRS
]
OUTPUT_PAIRS = [(0, 0), (2, 2), (0, 2)]  # xx, yy, xy as rows of (x, px, y, py)
OUTPUT_NAMES = ["xx", "yy", "xy"]
ELECTRON_MASS_EV = xm.ELECTRON_REST_ENERGY_EV
CENTROID_STEP = cv.TEST_AMPLITUDE


# ---------------------------------------------------------------------------
# sigma matrix helpers
# ---------------------------------------------------------------------------
def sigma_to_vector(sigma: np.ndarray) -> np.ndarray:
    return np.array([sigma[i, j] for i, j in PARAMETER_PAIRS])


def vector_to_sigma(vector: np.ndarray) -> np.ndarray:
    sigma = np.zeros((4, 4))
    for value, (i, j) in zip(vector, PARAMETER_PAIRS, strict=True):
        sigma[i, j] = sigma[j, i] = value
    return sigma


def beam_metrics(sigma: np.ndarray, beta_gamma: float) -> dict[str, float]:
    """Projected emittances, Twiss, coupling and intrinsic 4D emittances of a 4x4 Sigma."""
    metrics: dict[str, float] = {}
    for plane, offset in (("x", 0), ("y", 2)):
        block = sigma[offset : offset + 2, offset : offset + 2]
        emittance = np.sqrt(max(np.linalg.det(block), 0.0))
        metrics[f"eps_n_{plane}_um"] = emittance * beta_gamma * 1e6
        metrics[f"beta_{plane}_m"] = block[0, 0] / emittance
        metrics[f"alpha_{plane}"] = -block[0, 1] / emittance
        metrics[f"sigma_{plane}_mm"] = np.sqrt(block[0, 0]) * 1e3
    symplectic = np.zeros((4, 4))
    symplectic[0, 1] = symplectic[2, 3] = 1
    symplectic[1, 0] = symplectic[3, 2] = -1
    eigenvalues = np.linalg.eigvals(sigma @ symplectic)
    intrinsic = np.sort(np.abs(eigenvalues.imag))[::2]
    metrics["eps_n_intrinsic_1_um"] = intrinsic[0] * beta_gamma * 1e6
    metrics["eps_n_intrinsic_2_um"] = intrinsic[1] * beta_gamma * 1e6
    metrics["corr_xy"] = sigma[0, 2] / np.sqrt(sigma[0, 0] * sigma[2, 2])
    metrics["corr_x_py"] = sigma[0, 3] / np.sqrt(sigma[0, 0] * sigma[3, 3])
    metrics["corr_px_y"] = sigma[1, 2] / np.sqrt(sigma[1, 1] * sigma[2, 2])
    metrics["corr_px_py"] = sigma[1, 3] / np.sqrt(sigma[1, 1] * sigma[3, 3])
    return metrics


def relative_sigma_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Frobenius norm of the error in the units of the true rms (D^-1/2 E D^-1/2)."""
    scale = 1 / np.sqrt(np.diag(truth))
    return float(np.linalg.norm(scale[:, None] * (estimate - truth) * scale[None, :]))


# ---------------------------------------------------------------------------
# longitudinal quadrature
# ---------------------------------------------------------------------------
@dataclass
class SliceNodes:
    delta: np.ndarray
    tau: np.ndarray
    weight: np.ndarray

    def scaled(self, factor: float) -> SliceNodes:
        """Energy spread multiplied by `factor` about the mean (tau follows the chirp)."""
        mean = float(np.sum(self.weight * self.delta))
        mean_tau = float(np.sum(self.weight * self.tau))
        return SliceNodes(
            mean + factor * (self.delta - mean),
            mean_tau + factor * (self.tau - mean_tau),
            self.weight,
        )


def delta_quadrature(tau: np.ndarray, delta: np.ndarray, n_nodes: int) -> SliceNodes:
    """Gauss quadrature of the empirical delta distribution (Stieltjes / Lanczos), with
    tau of each node from a cubic fit of tau(delta) (the RF chirp)."""
    centred = delta - delta.mean()
    scale = centred.std()
    z = centred / scale
    n_samples = len(z)
    previous = np.zeros(n_samples)
    current = np.full(n_samples, 1 / np.sqrt(n_samples))
    alphas, betas = [], []
    beta = 0.0
    for _ in range(n_nodes):
        alpha = float(np.sum(z * current**2))
        alphas.append(alpha)
        nxt = z * current - alpha * current - beta * previous
        beta = float(np.linalg.norm(nxt))
        betas.append(beta)
        previous, current = current, nxt / beta
    jacobi = np.diag(alphas) + np.diag(betas[:-1], 1) + np.diag(betas[:-1], -1)
    values, vectors = np.linalg.eigh(jacobi)
    nodes_delta = delta.mean() + scale * values
    chirp = np.polyfit(delta, tau, 3)
    return SliceNodes(nodes_delta, np.polyval(chirp, nodes_delta), vectors[0] ** 2)


def single_node() -> SliceNodes:
    """Achromatic on-momentum model."""
    return SliceNodes(np.zeros(1), np.zeros(1), np.ones(1))


# ---------------------------------------------------------------------------
# forward model
# ---------------------------------------------------------------------------
@dataclass
class Response:
    """Per screen: R (K,4,4), D (K,4) of the reference particle of every slice."""

    nodes: SliceNodes
    transfer: dict[str, np.ndarray]
    orbit: dict[str, np.ndarray]


def track_from_plane(
    model: cv.XsuiteLine,
    coordinates: np.ndarray,
    plane: str | None,
    stops: list[str],
) -> dict[str, np.ndarray]:
    """Coordinates at the exit of every stop for particles given at the exit of `plane`
    (None = the line start). Stops downstream are tracked forward, upstream ones backtracked
    with the exact reverse xsuite map."""
    if plane is None:
        return model.track_through(coordinates, stops)
    plane_index = model.element_index[plane] + 1
    energy_ev = float(
        np.hypot(model.inventory.momentum_mevc_at[plane] * 1e6, ELECTRON_MASS_EV)
    )
    recorded = {}
    for stop in stops:
        stop_index = model.element_index[stop] + 1
        particles = xm._particles(coordinates, energy_ev)
        if stop_index >= plane_index:
            model.line.track(particles, ele_start=plane_index, ele_stop=stop_index)
        else:
            model.line.track(
                particles, ele_start=stop_index, ele_stop=plane_index, backtrack=True
            )
        recorded[stop] = xm._coordinates(particles)
    return recorded


def slice_response(
    model: cv.XsuiteLine,
    nodes: SliceNodes,
    stops: list[str],
    launch: np.ndarray | None = None,
    plane: str | None = None,
) -> Response:
    """Transverse maps and off-momentum orbits at `stops` for every slice reference,
    from the reconstruction `plane` (None = line start).
    `launch` = (x, px, y, py, delta shift) of the beam centroid at the reconstruction
    point; the maps are then taken about that trajectory (default: on axis)."""
    launch = np.zeros(5) if launch is None else np.asarray(launch, dtype=float)
    n_nodes = len(nodes.delta)
    rows = []
    for tau, delta in zip(nodes.tau, nodes.delta, strict=True):
        reference = np.array([*launch[:4], tau, delta + launch[4]], dtype=float)
        block = np.tile(reference, (9, 1))
        for column in range(4):
            block[1 + 2 * column, column] += CENTROID_STEP
            block[2 + 2 * column, column] -= CENTROID_STEP
        rows.append(block)
    recorded = track_from_plane(model, np.vstack(rows), plane, stops)
    transfer, orbit = {}, {}
    for name, state in recorded.items():
        state = state.reshape(n_nodes, 9, 6)
        orbit[name] = state[:, 0, :4]
        columns = [
            (state[:, 1 + 2 * c, :4] - state[:, 2 + 2 * c, :4]) / (2 * CENTROID_STEP)
            for c in range(4)
        ]
        transfer[name] = np.stack(columns, axis=-1)  # (K, out, in)
    return Response(nodes, transfer, orbit)


def transport_sigma(
    model: cv.XsuiteLine,
    sigma: np.ndarray,
    longitudinal: np.ndarray,
    plane: str | None,
    target: str | None,
    n_particles: int = 100000,
    seed: int = 0,
) -> np.ndarray:
    """4x4 covariance at `target` of a beam that is Gaussian with `sigma` at `plane`,
    independent of the known (tau, delta) sample, tracked through the xsuite line."""
    rng = np.random.default_rng(seed)
    choice = rng.choice(
        len(longitudinal), n_particles, replace=len(longitudinal) < n_particles
    )
    transverse = (np.linalg.cholesky(sigma) @ rng.normal(size=(4, n_particles))).T
    coordinates = np.column_stack([transverse, longitudinal[choice]])
    if target is None:
        coordinates_out = _track_to_start(model, coordinates, plane)
    else:
        coordinates_out = track_from_plane(model, coordinates, plane, [target])[target]
    return np.cov(coordinates_out[:, :4].T)


def _track_to_start(
    model: cv.XsuiteLine, coordinates: np.ndarray, plane: str
) -> np.ndarray:
    plane_index = model.element_index[plane] + 1
    energy_ev = float(
        np.hypot(model.inventory.momentum_mevc_at[plane] * 1e6, ELECTRON_MASS_EV)
    )
    particles = xm._particles(coordinates, energy_ev)
    model.line.track(particles, ele_start=0, ele_stop=plane_index, backtrack=True)
    return xm._coordinates(particles)


def effective_orbit(
    response: Response, screen: str, chromatic_offset: np.ndarray | None = None
) -> np.ndarray:
    """Slice orbits (K, 4) at `screen`. `chromatic_offset` = eta0 (x, px, y, py per unit
    delta) is a transverse-energy correlation at the reconstruction point (x0 = x_beta +
    eta0 * delta), which also stands in for kicks from hidden quadrupole misalignments
    close to the start: slice k starts displaced by eta0 * (delta_k - mean delta)."""
    orbit = response.orbit[screen]
    if chromatic_offset is None:
        return orbit
    nodes = response.nodes
    centred = nodes.delta - np.sum(nodes.weight * nodes.delta)
    shift = np.einsum("kij,j->ki", response.transfer[screen], chromatic_offset)
    return orbit + shift * centred[:, None]


def moment_rows(
    response: Response, screen: str, chromatic_offset: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """A (3, 10) and c (3): predicted (xx, yy, xy) = A @ sigma_vector + c."""
    weight = response.nodes.weight
    transfer = response.transfer[screen]
    orbit = effective_orbit(response, screen, chromatic_offset)
    mean = weight @ orbit
    design = np.zeros((3, len(PARAMETER_PAIRS)))
    offset = np.zeros(3)
    for row, (a, b) in enumerate(OUTPUT_PAIRS):
        for column, (i, j) in enumerate(PARAMETER_PAIRS):
            term = transfer[:, a, i] * transfer[:, b, j]
            if i != j:
                term = term + transfer[:, a, j] * transfer[:, b, i]
            design[row, column] = np.sum(weight * term)
        offset[row] = np.sum(weight * orbit[:, a] * orbit[:, b]) - mean[a] * mean[b]
    return design, offset


def predicted_centroid(
    response: Response, screen: str
) -> tuple[np.ndarray, np.ndarray]:
    """Linear map of the centroid (x, y) on the screen to the launch (x, px, y, py), and its
    offset from the energy spread: centroid = M @ launch4 + offset."""
    weight = response.nodes.weight
    transfer = np.einsum("k,kij->ij", weight, response.transfer[screen])
    offset = weight @ response.orbit[screen]
    return transfer[[0, 2]], offset[[0, 2]]


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------
@dataclass
class ScreenData:
    """Averaged screen moments (resolution already subtracted) and the shot count."""

    screens: list[str]
    moments: np.ndarray  # (S, 3) xx, yy, xy
    centroids: np.ndarray  # (S, 2)
    shots: int


def measure(
    machine: cv.VirtualMachine,
    screens: list[str],
    shots: int = 1,
    assumed_resolution_m: float | None = None,
) -> ScreenData:
    """Insert each screen in turn and average `shots` shots of the moments."""
    resolution = (
        machine.screen_resolution_m
        if assumed_resolution_m is None
        else assumed_resolution_m
    )
    moments, centroids = [], []
    for screen in screens:
        readings = [machine.screen_moments(screen) for _ in range(shots)]
        mean_reading = {k: np.mean([r[k] for r in readings]) for k in readings[0]}
        moments.append(
            [
                mean_reading["xx"] - resolution**2,
                mean_reading["yy"] - resolution**2,
                mean_reading["xy"],
            ]
        )
        centroids.append([mean_reading["mean_x"], mean_reading["mean_y"]])
    return ScreenData(list(screens), np.array(moments), np.array(centroids), shots)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------
def stack_system(
    response: Response,
    data: ScreenData,
    screens: list[str],
    chromatic_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design matrix, offsets and measured vector for the chosen screens."""
    designs, offsets, values = [], [], []
    for screen in screens:
        design, offset = moment_rows(response, screen, chromatic_offset)
        designs.append(design)
        offsets.append(offset)
        values.append(data.moments[data.screens.index(screen)])
    return np.vstack(designs), np.concatenate(offsets), np.concatenate(values)


@dataclass
class FitResult:
    sigma: np.ndarray
    sigma_linear: np.ndarray
    chi2_per_dof: float
    condition_number: float
    singular_values: np.ndarray
    used_screens: list[str]
    chromatic_offset: np.ndarray | None = None


def row_sigma(values: np.ndarray, shots: int, model_floor: float) -> np.ndarray:
    """Relative uncertainty per row: 1 % shot noise / sqrt(shots) plus a model floor.
    The xy row scales with sqrt(xx*yy) because xy can pass through zero."""
    matrix = values.reshape(-1, 3)
    magnitude = np.column_stack(
        [matrix[:, 0], matrix[:, 1], np.sqrt(np.abs(matrix[:, 0] * matrix[:, 1]))]
    )
    relative = np.hypot(0.01 / np.sqrt(shots), model_floor)
    return (np.abs(magnitude) * relative).ravel()


def condition_number(
    design: np.ndarray, sigma_rows: np.ndarray
) -> tuple[float, np.ndarray]:
    """Condition number of the weighted design with unit-norm columns (invariant to units)."""
    weighted = design / sigma_rows[:, None]
    weighted = weighted / np.linalg.norm(weighted, axis=0)
    singular = np.linalg.svd(weighted, compute_uv=False)
    return float(singular[0] / max(singular[-1], 1e-300)), singular


def _linear_solution(
    design: np.ndarray, target: np.ndarray, sigma_rows: np.ndarray
) -> np.ndarray:
    weighted = design / sigma_rows[:, None]
    column_norm = np.linalg.norm(weighted, axis=0)
    solution, *_ = np.linalg.lstsq(
        weighted / column_norm, target / sigma_rows, rcond=None
    )
    return solution / column_norm


def _psd_start(sigma: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((sigma + sigma.T) / 2)
    floor = 1e-4 * max(values.max(), 1e-30)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def dispersive_fraction(
    response: Response, screens: list[str], sigma: np.ndarray
) -> np.ndarray:
    """Per row (xx, yy, xy): share of the predicted second moment that comes from the
    energy spread through the off-momentum orbit D(delta), the part most sensitive to
    unmodelled steering."""
    fractions = []
    for screen in screens:
        design, offset = moment_rows(response, screen)
        prediction = design @ sigma_to_vector(sigma) + offset
        magnitude = np.array(
            [prediction[0], prediction[1], np.sqrt(abs(prediction[0] * prediction[1]))]
        )
        fractions.append(np.abs(offset) / np.maximum(np.abs(magnitude), 1e-30))
    return np.concatenate(fractions).clip(0, 5)


ETA_PRIOR = np.array([1e-2, 1e-3, 1e-2, 1e-3])  # m, rad per unit delta (weak prior)


def fit_sigma(
    response: Response,
    data: ScreenData,
    screens: list[str] | None = None,
    model_floor: float = 0.01,
    fit_chromatic_offset: bool = False,
    eta_prior: np.ndarray = ETA_PRIOR,
    robust_scale: float | None = None,
    dispersive_gain: float = 0.0,
) -> FitResult:
    """Weighted least squares for Sigma (variable projection over the optional eta0), then
    the positive-definite (Cholesky) refinement."""
    screens = screens or data.screens
    _, _, values = stack_system(response, data, screens)
    sigma_rows = row_sigma(values, data.shots, model_floor)
    if dispersive_gain > 0:
        first = fit_sigma(
            response, data, screens, model_floor, False, eta_prior, robust_scale
        )
        fraction = dispersive_fraction(response, screens, first.sigma)
        sigma_rows = sigma_rows * np.hypot(
            1.0,
            dispersive_gain
            * fraction
            / np.hypot(0.01 / np.sqrt(data.shots), model_floor),
        )

    def inner(eta: np.ndarray | None):
        design, offset, _ = stack_system(response, data, screens, eta)
        return design, offset, _linear_solution(design, values - offset, sigma_rows)

    eta = None
    if fit_chromatic_offset:

        def outer_residual(eta_vector: np.ndarray) -> np.ndarray:
            design, offset, vector = inner(eta_vector)
            data_residual = (design @ vector + offset - values) / sigma_rows
            return np.concatenate([data_residual, eta_vector / eta_prior])

        eta = least_squares(
            outer_residual, np.zeros(4), x_scale=eta_prior, method="trf"
        ).x
    design, offset, vector = inner(eta)
    sigma_linear = vector_to_sigma(vector)
    kappa, singular = condition_number(design, sigma_rows)

    start = _psd_start(sigma_linear)
    scale = np.sqrt(np.diag(start))
    unit_start = np.linalg.cholesky(start / np.outer(scale, scale))
    lower = np.tril_indices(4)

    def to_sigma(parameters: np.ndarray) -> np.ndarray:
        factor = np.zeros((4, 4))
        factor[lower] = parameters
        return np.outer(scale, scale) * (factor @ factor.T)

    def residual(parameters: np.ndarray) -> np.ndarray:
        prediction = design @ sigma_to_vector(to_sigma(parameters)) + offset
        return (prediction - values) / sigma_rows

    loss, f_scale = (
        ("linear", 1.0) if robust_scale is None else ("soft_l1", robust_scale)
    )
    fit = least_squares(
        residual,
        unit_start[lower],
        x_scale="jac",
        method="trf",
        loss=loss,
        f_scale=f_scale,
    )
    chi2 = float(np.sum(residual(fit.x) ** 2) / max(len(values) - 10, 1))
    return FitResult(
        to_sigma(fit.x), sigma_linear, chi2, kappa, singular, list(screens), eta
    )


# ---------------------------------------------------------------------------
# pair builders and hidden-error control
# ---------------------------------------------------------------------------
def build_pair_at_s07(
    end: str = xm.READOUT,
    seed: int = 0,
    errors: cv.HiddenErrors | None = None,
    beam_particles: int = 20000,
    **machine_kwargs,
):
    """(machine, model, true beam, (tau, delta) sample) on S07 aperture .. end, with the
    nominal beam tracked to S07 (cached by make_s07_beam.py)."""
    with np.load(Path(__file__).parent / "s07_beam.npz") as cache:
        full_beam, energy_ev = cache["particles"], float(cache["energy_ev"])
    rng = np.random.default_rng(seed)
    beam = full_beam[rng.choice(len(full_beam), beam_particles, replace=False)]
    machine_line = cv.XsuiteLine(
        xm.build_line(energy_ev, start=xm.RECONSTRUCTION_POINT, end=end)
    )
    model_line = cv.XsuiteLine(
        xm.build_line(energy_ev, start=xm.RECONSTRUCTION_POINT, end=end),
        machine_line.inventory,
    )
    errors = errors or cv.HiddenErrors.random(machine_line.inventory, seed=seed)
    machine = cv.VirtualMachine(
        machine_line, beam, errors, seed=seed + 1, **machine_kwargs
    )
    return machine, model_line, beam, full_beam[:, 4:6], energy_ev


def apply_errors(machine: cv.VirtualMachine, errors: cv.HiddenErrors) -> None:
    """Swap the hidden errors of an existing machine (scoring / robustness studies only)."""
    machine._errors = errors
    for quad in machine.inventory.quads:
        dx, dy = errors.quad_offsets.get(quad, (0.0, 0.0))
        machine._line.line[quad].shift_x = dx
        machine._line.line[quad].shift_y = dy


def scaled_errors(
    inventory: cv.LineInventory, seed: int, launch_scale: float, offset_rms: float
) -> cv.HiddenErrors:
    base = cv.HiddenErrors.random(inventory, seed=seed, quad_offset_rms=offset_rms)
    base.launch = base.launch * launch_scale
    return base


def true_sigma(beam: np.ndarray) -> np.ndarray:
    return np.cov(beam[:, :4].T)


# ---------------------------------------------------------------------------
# energy-dependent Sigma(delta) = sum_m (delta - mean)^m Sigma_m  (linear in all Sigma_m)
# ---------------------------------------------------------------------------
def moment_rows_polynomial(
    response: Response, screen: str, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Design (3, 10 (order + 1)) and offsets when the transverse covariance of a slice is
    a polynomial in its centred delta: the model stays linear in the coefficients."""
    nodes = response.nodes
    centred = nodes.delta - np.sum(nodes.weight * nodes.delta)
    transfer = response.transfer[screen]
    design = np.zeros((3, 10 * (order + 1)))
    _, offset = moment_rows(response, screen)
    for power in range(order + 1):
        weight = nodes.weight * centred**power
        for row, (a, b) in enumerate(OUTPUT_PAIRS):
            for column, (i, j) in enumerate(PARAMETER_PAIRS):
                term = transfer[:, a, i] * transfer[:, b, j]
                if i != j:
                    term = term + transfer[:, a, j] * transfer[:, b, i]
                design[row, 10 * power + column] = np.sum(weight * term)
    return design, offset


def fit_polynomial_sigma(
    response: Response,
    data: ScreenData,
    screens: list[str],
    order: int = 2,
    model_floor: float = 0.02,
    ridge: float = 3.0,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    """Projected Sigma (and the coefficient matrices) of the energy-dependent model.
    Higher-order coefficients carry a weak Gaussian prior (`ridge`, relative to Sigma_0)."""
    designs, offsets, values = [], [], []
    for screen in screens:
        design, offset = moment_rows_polynomial(response, screen, order)
        designs.append(design)
        offsets.append(offset)
        values.append(data.moments[data.screens.index(screen)])
    design, offset, values = (
        np.vstack(designs),
        np.concatenate(offsets),
        np.concatenate(values),
    )
    sigma_rows = row_sigma(values, data.shots, model_floor)
    weighted = design / sigma_rows[:, None]
    scale = np.linalg.norm(weighted, axis=0)
    solution, *_ = np.linalg.lstsq(
        np.vstack([weighted / scale, _prior_block(scale, order, ridge)]),
        np.concatenate([(values - offset) / sigma_rows, np.zeros(10 * order)]),
        rcond=None,
    )
    coefficients = solution / scale
    nodes = response.nodes
    centred = nodes.delta - np.sum(nodes.weight * nodes.delta)
    matrices = [
        vector_to_sigma(coefficients[10 * m : 10 * (m + 1)]) for m in range(order + 1)
    ]
    projected = sum(
        np.sum(nodes.weight * centred**m) * matrices[m] for m in range(order + 1)
    )
    residual = (design @ coefficients + offset - values) / sigma_rows
    chi2 = float(np.sum(residual**2) / max(len(values) - 10 * (order + 1), 1))
    return projected, matrices, chi2


def _prior_block(scale: np.ndarray, order: int, ridge: float) -> np.ndarray:
    block = np.zeros((10 * order, 10 * (order + 1)))
    for k in range(10 * order):
        block[k, 10 + k] = 1.0 / ridge
    return block
