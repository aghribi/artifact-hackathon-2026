"""Monte Carlo error study of one-shot SVD orbit correction on the virtual CLARA.

Reuses `clara_virtual` and `steering/svd_correction.py`. One machine line and one model line
are built per process; errors and correctors are reset for every seed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "steering"))
import svd_correction as svd

import clara_virtual as cv
from clara_virtual import xsuite_model

RANGES_JSON = cv.REPO_CHALLENGE / "data/clara/lattice_data/magnet_ranges.json"
SHOT_JITTER = np.array([20e-6, 2e-6, 20e-6, 2e-6, 0.0, 2e-4])
CUTOFF = 1e-3
UNIT_NAMES = ("x", "y")


@dataclass
class Context:
    machine: cv.VirtualMachine
    model: cv.XsuiteLine
    beam: np.ndarray
    bpms: list[str]
    quads: list[str]
    correctors: list[str]
    optics_stops: list[str]
    readout: str
    response: np.ndarray
    inverses: dict  # relative cutoff -> (pseudo-inverse, modes kept)
    design_twiss: dict
    design_optics: dict
    reference_ip: dict
    corrector_ranges: (
        np.ndarray
    )  # (2 * correctors,) observed max |angle| in rad, h then v
    baseline: dict | None = (
        None  # truth metrics of the error-free machine, no correction
    )


def build_context(
    lattice_json: str | None = None, n_particles: int = 3000, beam_seed: int = 0
) -> Context:
    full_beam, energy_ev, _ = xsuite_model.load_distribution()
    subset = full_beam[
        np.random.default_rng(beam_seed).choice(
            len(full_beam), n_particles, replace=False
        )
    ]
    # match the first moments of the full 200k distribution (removes the sample centroid noise)
    beam = subset - subset.mean(axis=0) + full_beam.mean(axis=0)
    kwargs = {} if lattice_json is None else {"lattice_json": lattice_json}
    machine_line = cv.XsuiteLine(xsuite_model.build_line(energy_ev, **kwargs))
    model = cv.XsuiteLine(
        xsuite_model.build_line(energy_ev, **kwargs), machine_line.inventory
    )
    inventory = machine_line.inventory
    machine = cv.VirtualMachine(
        machine_line, beam, cv.HiddenErrors(), seed=1, shot_jitter=SHOT_JITTER
    )
    correctors = [c for c in inventory.correctors if c in model.line.element_names]
    bpms, quads = inventory.bpms, inventory.quads
    response = svd.corrector_response(model, correctors, bpms)
    readout = xsuite_model.READOUT
    stops = sorted(
        set(quads)
        | set(bpms)
        | {s for s in inventory.screens if s in model.line.element_names}
        | {readout},
        key=model.element_index.__getitem__,
    )
    design_twiss = xsuite_model.twiss_from_coordinates(beam)
    ranges = json.loads(RANGES_JSON.read_text())["elements"]
    corrector_ranges = np.zeros(2 * len(correctors))
    for index, name in enumerate(correctors):
        for plane, key in enumerate(("horizontal_angle", "vertical_angle")):
            corrector_ranges[2 * index + plane] = max(abs(v) for v in ranges[name][key])
    context = Context(
        machine,
        model,
        beam,
        bpms,
        quads,
        correctors,
        stops,
        readout,
        response,
        {},
        design_twiss,
        {},
        {},
        corrector_ranges,
    )
    context.baseline = None
    context.design_optics = optics_about(model, stops, np.zeros(6), design_twiss)
    context.reference_ip = beam_at(model, beam, readout)
    apply_errors(
        context,
        np.zeros(6),
        {q: (0.0, 0.0) for q in context.quads},
        {b: (0.0, 0.0) for b in context.bpms},
        0,
    )
    context.baseline = trajectory_metrics(context)
    centroids = model.track_through(beam, bpms)
    context.design_orbit = np.concatenate(
        [centroids[b][:, [0, 2]].mean(axis=0) for b in bpms]
    )
    return context


def get_inverse(context: Context, cutoff: float, planes: str = "all") -> np.ndarray:
    """Truncated pseudo-inverse; `planes="used"` only drives the corrector planes whose
    observed operating range is at least 100 urad (the FEA and a few other planes stay off)."""
    key = cutoff if planes == "all" else (cutoff, planes)
    if key not in context.inverses:
        if planes == "all":
            inverse, _, kept = svd.svd_pseudo_inverse(context.response, cutoff)
        else:
            mask = context.corrector_ranges >= USED_RANGE_RAD
            sub_inverse, _, kept = svd.svd_pseudo_inverse(
                context.response[:, mask], cutoff
            )
            inverse = np.zeros((context.response.shape[1], context.response.shape[0]))
            inverse[mask] = sub_inverse
        context.inverses[key] = (inverse, kept)
    return context.inverses[key][0]


def apply_errors(
    context: Context,
    launch: np.ndarray,
    quad_offsets: dict[str, tuple[float, float]],
    bpm_offsets: dict[str, tuple[float, float]],
    seed: int,
) -> None:
    """Reset the machine to a fresh draw: errors, zero correctors, new noise stream."""
    machine = context.machine
    errors = machine._errors
    errors.launch, errors.quad_offsets, errors.bpm_offsets = (
        launch,
        quad_offsets,
        bpm_offsets,
    )
    for quad in context.quads:
        machine._line.line[quad].shift_x, machine._line.line[quad].shift_y = (
            quad_offsets[quad]
        )
    for corrector in context.correctors:
        machine.set_corrector(corrector, 0.0, 0.0)
    machine._rng = np.random.default_rng(seed)


def one_shot_correction(
    context: Context,
    cutoff: float = CUTOFF,
    shots: int = 1,
    target: str = "zero",
    planes: str = "all",
) -> np.ndarray:
    """BPM shot(s) averaged, SVD, apply. Returns the kicks (h, v per corrector) in rad."""
    orbit = np.mean(
        [
            np.concatenate([r[b] for b in context.bpms])
            for r in (context.machine.read_bpms(context.bpms) for _ in range(shots))
        ],
        axis=0,
    )
    if (
        target == "design"
    ):  # flatten onto the model's error-free centroid orbit, not onto zero
        orbit = orbit - context.design_orbit
    kicks = -get_inverse(context, cutoff, planes) @ orbit
    for index, corrector in enumerate(context.correctors):
        context.machine.set_corrector(corrector, kicks[2 * index], kicks[2 * index + 1])
    return kicks


def beam_at(line: cv.XsuiteLine, beam: np.ndarray, readout: str) -> dict:
    """Moments of the tracked beam at the readout screen (survivors only)."""
    coords = line.track_through(beam, [readout], allow_losses=True)[readout]
    out = {"transmission": len(coords) / len(beam)}
    if len(coords) < 20:
        return {
            **out,
            **{
                k: np.nan
                for k in ("x", "px", "y", "py", "sigma_x", "sigma_y", "eps_x", "eps_y")
            },
        }
    twiss = xsuite_model.twiss_from_coordinates(coords)
    out.update(
        x=coords[:, 0].mean(),
        px=coords[:, 1].mean(),
        y=coords[:, 2].mean(),
        py=coords[:, 3].mean(),
        sigma_x=coords[:, 0].std(),
        sigma_y=coords[:, 2].std(),
        eps_x=twiss["emittance_x"],
        eps_y=twiss["emittance_y"],
    )
    return out


def optics_about(
    line: cv.XsuiteLine, stops: list[str], reference: np.ndarray, start_twiss: dict
) -> dict:
    """Beta and dispersion at `stops` from the linear map about the trajectory `reference`."""
    try:
        maps = line.linear_map(stops, reference)
    except RuntimeError:
        return {}
    optics = {key: [] for key in ("beta_x", "beta_y", "D_x", "D_y", "Dp_x", "Dp_y")}
    for name in stops:
        transfer = maps[name]
        for plane, offset in (("x", 0), ("y", 2)):
            r11, r12, r21, r22 = (
                transfer[offset, offset],
                transfer[offset, offset + 1],
                transfer[offset + 1, offset],
                transfer[offset + 1, offset + 1],
            )
            beta0, alpha0 = start_twiss[f"beta_{plane}"], start_twiss[f"alpha_{plane}"]
            gamma0 = (1 + alpha0**2) / beta0
            damping = r11 * r22 - r12 * r21
            optics[f"beta_{plane}"].append(
                (beta0 * r11**2 - 2 * alpha0 * r11 * r12 + gamma0 * r12**2) / damping
            )
            optics[f"D_{plane}"].append(transfer[offset, 5])
            optics[f"Dp_{plane}"].append(transfer[offset + 1, 5])
    return {key: np.asarray(values) for key, values in optics.items()}


def trajectory_metrics(context: Context) -> dict:
    """Truth only: residual trajectory rms at BPMs and in quads, beam at the readout, optics."""
    machine = context.machine
    beam = machine._beam + machine._errors.launch
    names = context.bpms + context.quads
    recorded = machine._line.track_through(beam, names, allow_losses=True)
    centroid = {
        n: recorded[n][:, [0, 2]].mean(axis=0)
        if len(recorded[n])
        else np.full(2, np.nan)
        for n in names
    }
    bpm_orbit = np.array([centroid[b] for b in context.bpms])
    in_quads = np.array(
        [
            centroid[q] - np.asarray(machine._errors.quad_offsets[q])
            for q in context.quads
        ]
    )
    rms = lambda a: np.sqrt(np.nanmean(a**2, axis=0))
    out = {
        "bpm_rms_x": rms(bpm_orbit)[0],
        "bpm_rms_y": rms(bpm_orbit)[1],
        "quad_rms_x": rms(in_quads)[0],
        "quad_rms_y": rms(in_quads)[1],
        "bpm_max_x": np.nanmax(np.abs(bpm_orbit[:, 0])),
        "bpm_max_y": np.nanmax(np.abs(bpm_orbit[:, 1])),
    }
    ip = beam_at_machine(context, beam)
    out.update({f"ip_{k}": v for k, v in ip.items()})
    for key in ("sigma_x", "sigma_y", "eps_x", "eps_y"):
        out[f"ip_{key}_rel"] = ip[key] / context.reference_ip[key] - 1
    optics = optics_about(
        machine._line,
        context.optics_stops,
        machine._errors.launch * np.array([1, 1, 1, 1, 0, 0]),
        context.design_twiss,
    )
    design = context.design_optics
    last = len(context.optics_stops) - 1
    if optics:
        for plane in UNIT_NAMES:
            beat = optics[f"beta_{plane}"] / design[f"beta_{plane}"] - 1
            out[f"beatbeat_ip_{plane}"] = beat[last]
            out[f"beatbeat_max_{plane}"] = np.max(np.abs(beat))
            out[f"D_ip_{plane}"] = optics[f"D_{plane}"][last]
            out[f"Dp_ip_{plane}"] = optics[f"Dp_{plane}"][last]
            out[f"dD_max_{plane}"] = np.max(
                np.abs(optics[f"D_{plane}"] - design[f"D_{plane}"])
            )
            out[f"dDp_max_{plane}"] = np.max(
                np.abs(optics[f"Dp_{plane}"] - design[f"Dp_{plane}"])
            )
    else:
        for plane in UNIT_NAMES:
            for key in (
                "beatbeat_ip",
                "beatbeat_max",
                "D_ip",
                "Dp_ip",
                "dD_max",
                "dDp_max",
            ):
                out[f"{key}_{plane}"] = np.nan
    return out


def beam_at_machine(context: Context, beam: np.ndarray) -> dict:
    return beam_at(context.machine._line, beam, context.readout)


USED_RANGE_RAD = 100e-6  # correctors with a larger observed range are in routine use


def kick_metrics(context: Context, kicks: np.ndarray) -> dict:
    """Largest kicks, and how they compare with the observed operating ranges. Planes whose
    observed range is below 100 urad (FEA, VBC, a few S04/S07 planes) count as unused."""
    ranges = np.maximum(context.corrector_ranges, 1e-12)
    used = context.corrector_ranges >= USED_RANGE_RAD
    ratio = np.abs(kicks) / ranges
    active = np.abs(kicks) > 1e-6
    return {
        "max_kick_urad": np.abs(kicks).max() * 1e6,
        "rms_kick_urad": np.sqrt(np.mean(kicks**2)) * 1e6,
        "max_kick_used_urad": np.abs(kicks[used]).max() * 1e6,
        "max_ratio_used": float(ratio[used].max()),
        "n_over_range_used": int(np.sum((ratio > 1) & used)),
        "n_used_planes": int(used.sum()),
        "n_unused_planes_active": int(np.sum(active & ~used)),
        "max_kick_unused_urad": float(np.abs(kicks[~used]).max() * 1e6),
    }


def run_seed(
    context: Context,
    seed: int,
    quad_rms: float,
    bpm_rms_levels: list[float],
    launch_rms: tuple[float, float],
    delta_rms: float = 0.0,
    cutoff: float = CUTOFF,
    shots: int = 1,
    jitter_scale: float = 1.0,
    target: str = "zero",
    planes: str = "all",
) -> list[dict]:
    """Records for one seed: one per BPM offset level (paired draws, same launch and quads)."""
    rng = np.random.default_rng(seed)
    launch_unit = rng.normal(size=6)
    quad_unit = rng.normal(size=(len(context.quads), 2))
    bpm_unit = rng.normal(size=(len(context.bpms), 2))
    launch_scale = np.array(
        [launch_rms[0], launch_rms[1], launch_rms[0], launch_rms[1], 0.0, delta_rms]
    )
    launch = launch_unit * launch_scale
    quad_offsets = {
        q: tuple(quad_unit[i] * quad_rms) for i, q in enumerate(context.quads)
    }
    records = []
    before = None
    context.machine.shot_jitter = SHOT_JITTER * jitter_scale
    for bpm_rms in bpm_rms_levels:
        bpm_offsets = {
            b: tuple(bpm_unit[i] * bpm_rms) for i, b in enumerate(context.bpms)
        }
        apply_errors(context, launch, quad_offsets, bpm_offsets, seed + 1000)
        if before is None:
            before = trajectory_metrics(context)
        kicks = one_shot_correction(context, cutoff, shots, target, planes)
        after = trajectory_metrics(context)
        records.append(
            {
                "seed": seed,
                "quad_rms_um": quad_rms * 1e6,
                "bpm_rms_um": bpm_rms * 1e6,
                "launch_um": launch_rms[0] * 1e6,
                "delta_rms": delta_rms,
                "cutoff": cutoff,
                "shots": shots,
                "jitter_scale": jitter_scale,
                "target": target,
                "planes": planes,
                **{f"before_{k}": v for k, v in before.items()},
                **{f"after_{k}": v for k, v in after.items()},
                **kick_metrics(context, kicks),
            }
        )
    for corrector in context.correctors:
        context.machine.set_corrector(corrector, 0.0, 0.0)
    return records
