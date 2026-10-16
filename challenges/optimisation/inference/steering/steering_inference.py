"""Steering inference on the virtual CLARA: launch error and quadrupole offsets from
centroid measurements, with the xsuite design model as the forward model.

Unknowns: the beam launch at the start (x, px, y, py, delta; tau has no first-order
effect on centroids) and the (dx, dy) offset of every quadrupole.

Forward model: the xsuite design line with the current estimate of the quad offsets;
predictions are centroids of a tracked model beam. The Jacobian is first order about the
estimated launch, averaged over Gauss-Hermite energy slices: a quadrupole shifted by d maps
x -> R_q (x - d) + d, so its offset adds (I - R_q) d at the quadrupole exit, carried to each
monitor by M_monitor M_q_exit^-1, with all maps from `XsuiteLine.linear_map`. The fit is a
Levenberg-Marquardt MAP fit, re-linearised every iteration, with shot-to-shot launch
jitter as a correlated noise covariance (see README.md for results).

Measurements:
- `bpm`: all screens out, beam reaches the end; every BPM reads (x, y) per shot.
- `screen`: no BPMs; one YAG screen inserted per shot (destructive), centroid only there.

Each configuration is the nominal optics plus one quadrupole changed by a current step
(beam-based alignment by quadrupole modulation), so offsets become observable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clara_virtual as cv

LAUNCH_COLUMNS = [0, 1, 2, 3, 5]  # x, px, y, py, delta
LAUNCH_LABELS = ["x", "px", "y", "py", "delta"]
MODULATION_STEP_A = 5.0
ENERGY_SLICES = 7


@dataclass
class Configuration:
    """A quadrupole setting and the monitors read in it."""

    label: str
    currents: dict[str, float]  # changes relative to nominal, in A (absolute currents)
    monitors: list[str]


def element_before(model: cv.XsuiteLine, name: str) -> str:
    return model.line.element_names[model.element_index[name] - 1]


def nominal_currents(model: cv.XsuiteLine) -> dict[str, float]:
    return {q: model.quad_current_from_k1(q, k1) for q, k1 in model.nominal_k1.items()}


def modulated_current(nominal_a: float) -> float:
    """Step the current by MODULATION_STEP_A towards zero current (small enough to keep the
    lattice-file optics transmitting the beam)."""
    step = MODULATION_STEP_A if nominal_a <= 0 else -MODULATION_STEP_A
    return float(
        np.clip(nominal_a + step, -cv.QUAD_CURRENT_LIMIT_A, cv.QUAD_CURRENT_LIMIT_A)
    )


def response_matrix(
    model: cv.XsuiteLine,
    configuration: Configuration,
    sigma_delta: float,
    launch: np.ndarray | None = None,
) -> np.ndarray:
    """(2 * monitors, 5 + 2 * quads) first-order response of monitor *beam centroids* to
    the unknowns, for the model set to this configuration.

    A single on-momentum particle is not enough: with a 1-2 % energy spread and large
    beta functions, the centroid of the beam follows the energy-averaged map. The
    response is therefore averaged over Gauss-Hermite energy slices of rms `sigma_delta`,
    each with its own xsuite map about an off-momentum reference particle."""
    model.restore_nominal_quads()
    model.set_quad_currents(configuration.currents)
    quads = model.inventory.quads
    entries = {q: element_before(model, q) for q in quads}
    stops = sorted(
        set(configuration.monitors) | set(quads) | set(entries.values()),
        key=model.element_index.__getitem__,
    )
    nodes, weights = np.polynomial.hermite_e.hermegauss(ENERGY_SLICES)
    weights = weights / weights.sum()
    response = 0.0
    for node, weight in zip(nodes, weights):
        reference = np.zeros(6) if launch is None else np.array(launch, float)
        reference[5] += node * sigma_delta
        maps = model.linear_map(stops, reference)
        rows = []
        for monitor in configuration.monitors:
            monitor_map = maps[monitor]
            block = np.zeros((2, len(LAUNCH_COLUMNS) + 2 * len(quads)))
            block[:, : len(LAUNCH_COLUMNS)] = monitor_map[np.ix_([0, 2], LAUNCH_COLUMNS)]
            for index, quad in enumerate(quads):
                if model.element_index[quad] >= model.element_index[monitor]:
                    continue
                exit_map = maps[quad]
                quad_map = exit_map @ np.linalg.inv(maps[entries[quad]])
                downstream = monitor_map @ np.linalg.inv(exit_map)
                # offset (dx, dy) -> (I - R_q) applied to (dx, 0, dy, 0, 0, 0)
                injected = (np.eye(6) - quad_map)[:, [0, 2]]
                columns = slice(
                    len(LAUNCH_COLUMNS) + 2 * index, len(LAUNCH_COLUMNS) + 2 * index + 2
                )
                block[:, columns] = (downstream @ injected)[[0, 2]]
            rows.append(block)
        response = response + weight * np.vstack(rows)
    model.restore_nominal_quads()
    return response


def configurations(
    model: cv.XsuiteLine, mode: str, screens_per_modulation: int = 2
) -> list[Configuration]:
    inventory = model.inventory
    nominal = nominal_currents(model)
    monitors_all = inventory.bpms if mode == "bpm" else inventory.screens
    result = [Configuration("nominal", {}, list(monitors_all))]
    for quad in inventory.quads:
        downstream = [
            m
            for m in monitors_all
            if model.element_index[m] > model.element_index[quad]
        ]
        if mode == "screen":
            downstream = downstream[:screens_per_modulation]
        if not downstream:
            continue
        result.append(
            Configuration(
                f"mod_{quad}", {quad: modulated_current(nominal[quad])}, downstream
            )
        )
    return result


def measure(
    machine: cv.VirtualMachine, configuration: Configuration, mode: str, shots: int
) -> np.ndarray:
    machine.restore_nominal_quads()
    machine.set_quad_currents(configuration.currents)
    if mode == "bpm":
        readings = [machine.read_bpms(configuration.monitors) for _ in range(shots)]
        values = np.mean(
            [np.concatenate([r[m] for m in configuration.monitors]) for r in readings],
            axis=0,
        )
    else:
        values = []
        for screen in configuration.monitors:
            moments = [machine.screen_moments(screen) for _ in range(shots)]
            values.append(
                [
                    np.mean([mo["mean_x"] for mo in moments]),
                    np.mean([mo["mean_y"] for mo in moments]),
                ]
            )
        values = np.concatenate(values)
    machine.restore_nominal_quads()
    return values


def apply_quad_offsets(line: cv.XsuiteLine, offsets: np.ndarray) -> None:
    for index, quad in enumerate(line.inventory.quads):
        line.line[quad].shift_x = float(offsets[2 * index])
        line.line[quad].shift_y = float(offsets[2 * index + 1])


def launch_vector(parameters: np.ndarray) -> np.ndarray:
    launch = np.zeros(6)
    launch[LAUNCH_COLUMNS] = parameters[: len(LAUNCH_COLUMNS)]
    return launch


def predict(model: cv.XsuiteLine, configuration: Configuration, model_beam: np.ndarray, launch: np.ndarray) -> np.ndarray:
    """Monitor beam centroids of the model (with its current quad offsets) for the model
    beam launched with `launch`."""
    model.restore_nominal_quads()
    model.set_quad_currents(configuration.currents)
    recorded = model.track_through(model_beam + launch, configuration.monitors, allow_losses=True)
    model.restore_nominal_quads()
    return np.concatenate([recorded[m][:, [0, 2]].mean(axis=0) for m in configuration.monitors])


def nuisance_columns(configs: list[Configuration], model: cv.XsuiteLine) -> np.ndarray:
    """One fixed (x, y) offset per monitor shared by all configurations: BPM/screen
    offsets plus the beam-shape part of the centroid the model beam cannot predict."""
    monitors = sorted({m for c in configs for m in c.monitors}, key=model.element_index.__getitem__)
    rows = sum(2 * len(c.monitors) for c in configs)
    nuisance = np.zeros((rows, 2 * len(monitors)))
    row = 0
    for configuration in configs:
        for monitor in configuration.monitors:
            column = 2 * monitors.index(monitor)
            nuisance[row : row + 2, column : column + 2] = np.eye(2)
            row += 2
    return nuisance


def run(
    mode: str,
    shots: int,
    seed: int,
    output: Path,
    monitor_offsets: bool = False,
    beam_knowledge: str = "exact",
    iterations: int = 4,
    beam_scale: float = 1.0,
    screens_per_modulation: int = 2,
) -> dict:
    """Gauss-Newton MAP fit of launch + quad offsets (+ optional monitor offsets).

    Each iteration re-linearises about the current estimate: the model gets the
    estimated quad offsets, predictions are centroids of the tracked model beam, and the
    Jacobian is the energy-sliced first-order response about the estimated launch.

    beam_knowledge: "exact" (the model beam is the machine's own distribution: tests the
    method) or "nominal" (an independent sample of the nominal beam file; with
    `--beam-file` style variants a different distribution can be swapped in)."""
    started = time.time()
    machine, model, _ = cv.build_pair(
        beam_particles=5000,
        seed=seed,
        shot_jitter=np.array([20e-6, 2e-6, 20e-6, 2e-6, 0, 2e-4]),
    )
    inventory = model.inventory
    start_beam, _, _ = cv.xsuite_model.load_distribution()
    if beam_knowledge == "exact":
        model_beam = machine._beam.copy()  # simulation shortcut: the beam itself is known
    else:
        model_beam = start_beam[np.random.default_rng(seed + 100).choice(len(start_beam), 5000, replace=False)]
    if beam_scale != 1.0:
        machine._beam = machine._beam * beam_scale
        model_beam = model_beam * beam_scale
        start_beam = start_beam * beam_scale
    sigma_delta = float(model_beam[:, 5].std())
    configs = configurations(model, mode, screens_per_modulation)
    measured = np.concatenate([measure(machine, c, mode, shots) for c in configs])

    number_launch = len(LAUNCH_COLUMNS)
    number_physical = number_launch + 2 * len(inventory.quads)
    prior = np.array([1e-3, 1e-4, 1e-3, 1e-4, 1e-2] + [5e-4] * (2 * len(inventory.quads)))
    nuisance = nuisance_columns(configs, model) if monitor_offsets else np.zeros((measured.size, 0))
    prior = np.concatenate([prior, np.full(nuisance.shape[1], 2e-3)])
    noise = (machine.bpm_noise_m if mode == "bpm" else 2e-6) / np.sqrt(shots) + 1e-6
    # Shot-to-shot launch jitter is amplified downstream to tens of um, far above the BPM
    # noise, and is correlated between the monitors read in the same shots. It enters as a
    # noise covariance (generalised least squares): readings averaged over the same shots
    # (one configuration with BPMs; one screen insertion with screens) share a block
    # J_launch Q J_launch^T, with Q = jitter^2 / shots.
    jitter_covariance = np.diag((np.array([20e-6, 2e-6, 20e-6, 2e-6, 2e-4]) ** 2) / shots)
    blocks = []
    row = 0
    for configuration in configs:
        if mode == "bpm":
            blocks.append(slice(row, row + 2 * len(configuration.monitors)))
        else:
            blocks.extend(slice(row + 2 * i, row + 2 * i + 2) for i in range(len(configuration.monitors)))
        row += 2 * len(configuration.monitors)

    def whitening(jacobian: np.ndarray) -> np.ndarray:
        covariance = np.eye(measured.size) * noise**2
        for block in blocks:
            launch_part = jacobian[block, :number_launch]
            covariance[block, block] += launch_part @ jitter_covariance @ launch_part.T
        return np.linalg.inv(np.linalg.cholesky(covariance))

    def evaluate(parameters: np.ndarray):
        launch = launch_vector(parameters)
        apply_quad_offsets(model, parameters[number_launch:number_physical])
        prediction = np.concatenate([predict(model, c, model_beam, launch) for c in configs])
        jacobian = np.hstack([np.vstack([response_matrix(model, c, sigma_delta, launch) for c in configs]), nuisance])
        residual = measured - prediction - nuisance @ parameters[number_physical:]
        white = whitening(jacobian)
        return residual, jacobian, white @ residual, white @ jacobian

    def cost_of(parameters, white_residual):
        return np.sum(white_residual**2) + np.sum((parameters / prior) ** 2)

    parameters = np.zeros(prior.size)
    residual, jacobian, white_residual, white_jacobian = evaluate(parameters)
    history = [float(residual.std() * 1e6)]
    damping = 1e-3
    for iteration in range(iterations):
        gradient = white_jacobian.T @ white_residual - parameters / prior**2
        information = white_jacobian.T @ white_jacobian + np.diag(1 / prior**2)
        cost = cost_of(parameters, white_residual)
        while True:  # Levenberg-Marquardt: accept only steps that lower the MAP cost
            step = np.linalg.solve(information + damping * np.diag(np.diag(information)), gradient)
            trial = parameters + step
            trial_state = evaluate(trial)
            if cost_of(trial, trial_state[2]) < cost or damping > 1e6:
                break
            damping *= 10
        parameters = trial
        residual, jacobian, white_residual, white_jacobian = trial_state
        damping = max(damping / 10, 1e-6)
        history.append(float(residual.std() * 1e6))
        print(f"iteration {iteration}: residual rms {history[-1]:.1f} um, whitened chi2/dof {np.mean(white_residual**2):.2f}", flush=True)
    apply_quad_offsets(model, np.zeros(number_physical - number_launch))
    covariance = np.linalg.inv(white_jacobian.T @ white_jacobian + np.diag(1 / prior**2))
    posterior_rms = np.sqrt(np.diag(covariance))
    observable = np.abs(jacobian).sum(axis=0) > 0
    singular = np.linalg.svd(jacobian[:, observable] * prior[None, observable], compute_uv=False)

    truth = machine._errors  # read only for scoring
    true_vector = np.concatenate([truth.launch[LAUNCH_COLUMNS]] + [np.asarray(truth.quad_offsets[q]) for q in inventory.quads])
    estimate = parameters[:number_physical]
    error = estimate - true_vector
    offset_error = error[number_launch:].reshape(-1, 2)
    offset_true = true_vector[number_launch:].reshape(-1, 2)
    seen = observable[number_launch:number_physical].reshape(-1, 2).all(axis=1)
    result = {
        "mode": mode,
        "monitor_offsets": monitor_offsets,
        "beam_knowledge": beam_knowledge,
        "beam_scale": beam_scale,
        "screens_per_modulation": screens_per_modulation if mode == "screen" else None,
        "configurations": len(configs),
        "shots": machine.shots,
        "readings": int(measured.size),
        "unknowns": int(prior.size),
        "unobservable": [label for label, flag in zip(LAUNCH_LABELS + [f"{q}_{p}" for q in inventory.quads for p in "xy"], observable) if not flag],
        "condition_number": float(singular[0] / singular[-1]),
        "residual_rms_um_by_iteration": [round(r, 2) for r in history],
        "launch": {
            label: {"true": float(true_vector[i]), "estimate": float(estimate[i]), "posterior_rms": float(posterior_rms[i])}
            for i, label in enumerate(LAUNCH_LABELS)
        },
        "quad_offsets": {
            "true_rms_um": float(offset_true[seen].std() * 1e6),
            "error_rms_um_x": float(offset_error[seen, 0].std() * 1e6),
            "error_rms_um_y": float(offset_error[seen, 1].std() * 1e6),
            "error_median_abs_um": float(np.median(np.abs(offset_error[seen])) * 1e6),
            "per_quad": {
                q: {
                    "true_um": (offset_true[i] * 1e6).round(1).tolist(),
                    "estimate_um": ((offset_true[i] + offset_error[i]) * 1e6).round(1).tolist(),
                    "posterior_rms_um": (posterior_rms[number_launch + 2 * i : number_launch + 2 * i + 2] * 1e6).round(1).tolist(),
                }
                for i, q in enumerate(inventory.quads)
            },
        },
        "seconds": time.time() - started,
    }
    output.write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bpm", "screen"], default="bpm")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--monitor-offsets", action="store_true")
    parser.add_argument("--beam-knowledge", choices=["exact", "nominal"], default="exact")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--beam-scale", type=float, default=1.0)
    parser.add_argument("--screens-per-modulation", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    tag = f"{args.mode}_{args.beam_knowledge}{'_monoff' if args.monitor_offsets else ''}{f'_scale{args.beam_scale:g}' if args.beam_scale != 1 else ''}{f'_spm{args.screens_per_modulation}' if args.mode == 'screen' else ''}_seed{args.seed}"
    output = args.output or Path(__file__).parent / f"result_{tag}.json"
    result = run(args.mode, args.shots, args.seed, output, args.monitor_offsets, args.beam_knowledge, args.iterations, args.beam_scale, args.screens_per_modulation)
    summary = {k: v for k, v in result.items() if k != "quad_offsets"}
    summary["quad_offsets"] = {k: v for k, v in result["quad_offsets"].items() if k != "per_quad"}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
