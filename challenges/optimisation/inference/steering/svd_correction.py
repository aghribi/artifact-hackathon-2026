"""Classic SVD orbit correction on the virtual CLARA: corrector -> BPM response from the
xsuite design model (known quad settings), one BPM shot, pseudo-inverse, apply, repeat.

No error inference: it flattens the measured orbit. Scored on the true beam centroid at
the BPMs and inside the quadrupoles (relative to each quad's hidden offset).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clara_virtual as cv

KICK_PROBE = 1e-5  # rad


def corrector_response(
    model: cv.XsuiteLine, correctors: list[str], bpms: list[str]
) -> np.ndarray:
    """(2 * bpms, 2 * correctors) response of the on-momentum orbit; columns are
    (h, v) kicks per corrector, rows (x, y) per BPM."""
    reference = model.reference_orbit(bpms)
    columns = []
    for corrector in correctors:
        for horizontal in (True, False):
            model.set_corrector(
                corrector,
                KICK_PROBE if horizontal else 0.0,
                0.0 if horizontal else KICK_PROBE,
            )
            orbit = model.reference_orbit(bpms)
            model.set_corrector(corrector, 0.0, 0.0)
            columns.append(
                np.concatenate([(orbit[b] - reference[b])[[0, 2]] for b in bpms])
                / KICK_PROBE
            )
    return np.column_stack(columns)


def svd_pseudo_inverse(
    matrix: np.ndarray, cutoff: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Truncated pseudo-inverse keeping singular values above `cutoff` times the largest
    (the FEH correctors after the last BPM give exact zeros)."""
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    keep = int(np.sum(singular > cutoff * singular[0]))
    inverse = vt[:keep].T @ np.diag(1 / singular[:keep]) @ u[:, :keep].T
    return inverse, singular, keep


def true_centroids(
    machine: cv.VirtualMachine, names: list[str]
) -> dict[str, np.ndarray]:
    """Scoring only: noise-free beam centroid at each element (no jitter)."""
    beam = machine._beam + machine._errors.launch
    recorded = machine._line.track_through(beam, names, allow_losses=True)
    return {n: recorded[n][:, [0, 2]].mean(axis=0) for n in names}


def score(machine: cv.VirtualMachine, bpms: list[str], quads: list[str]) -> dict:
    centroids = true_centroids(machine, bpms + quads)
    bpm_orbit = np.array([centroids[b] for b in bpms])
    in_quads = np.array(
        [centroids[q] - np.asarray(machine._errors.quad_offsets[q]) for q in quads]
    )
    return {
        "bpm_orbit_um": (bpm_orbit * 1e6).round(1).tolist(),
        "quad_orbit_um": (in_quads * 1e6).round(1).tolist(),
        "bpm_rms_um": (bpm_orbit.std(axis=0) * 1e6).round(1).tolist(),
        "bpm_max_um": (np.abs(bpm_orbit).max(axis=0) * 1e6).round(1).tolist(),
        "beam_in_quads_rms_um": (np.sqrt((in_quads**2).mean(axis=0)) * 1e6)
        .round(1)
        .tolist(),
        "screen_at_readout_um": (centroids[bpms[-1]] * 1e6).round(1).tolist(),
    }


def run(seed: int, bpm_offset_rms: float, cutoff: float, iterations: int, shots: int = 1) -> dict:
    machine, model, _ = cv.build_pair(
        beam_particles=5000,
        seed=seed,
        shot_jitter=np.array([20e-6, 2e-6, 20e-6, 2e-6, 0, 2e-4]),
    )
    rng = np.random.default_rng(seed + 7)
    machine._errors.bpm_offsets = {
        b: tuple(rng.normal(0, bpm_offset_rms, 2)) for b in model.inventory.bpms
    }
    bpms, quads = model.inventory.bpms, model.inventory.quads
    correctors = [
        c for c in model.inventory.correctors if c in model.line.element_names
    ]
    response = corrector_response(model, correctors, bpms)
    inverse, singular, keep = svd_pseudo_inverse(response, cutoff)
    kicks = np.zeros(response.shape[1])
    history = [{"iteration": 0, **score(machine, bpms, quads)}]
    for iteration in range(1, iterations + 1):
        readings = [machine.read_bpms(bpms) for _ in range(shots)]
        orbit = np.mean([np.concatenate([r[b] for b in bpms]) for r in readings], axis=0)
        kicks = kicks - inverse @ orbit
        for index, corrector in enumerate(correctors):
            machine.set_corrector(corrector, kicks[2 * index], kicks[2 * index + 1])
        history.append(
            {
                "iteration": iteration,
                "max_kick_urad": float(np.abs(kicks).max() * 1e6),
                **score(machine, bpms, quads),
            }
        )
    return {
        "seed": seed,
        "bpm_offset_rms_um": bpm_offset_rms * 1e6,
        "correctors": len(correctors),
        "cutoff": cutoff,
        "shots_per_iteration": shots,
        "singular_values_kept": keep,
        "singular_values": singular.round(3).tolist(),
        "history": history,
        "kicks_urad": {
            c: (kicks[2 * i : 2 * i + 2] * 1e6).round(1).tolist()
            for i, c in enumerate(correctors)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bpm-offset-rms", type=float, default=0.0)
    parser.add_argument("--cutoff", type=float, default=1e-3)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    result = run(args.seed, args.bpm_offset_rms, args.cutoff, args.iterations, args.shots)
    tag = f"svd_bpmoff{args.bpm_offset_rms * 1e6:g}_cut{args.cutoff:g}_shots{args.shots}_seed{args.seed}"
    (Path(__file__).parent / f"result_{tag}.json").write_text(
        json.dumps(result, indent=1)
    )
    for step in result["history"]:
        print(step)


if __name__ == "__main__":
    main()
