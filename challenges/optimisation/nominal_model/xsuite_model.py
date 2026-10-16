"""xsuite model of the CLARA line for the optimisation challenge.

The Cheetah lattice JSON is the source of truth. A segment is cut from it, with whatever
magnet settings it carries, and converted element by element to an xsuite line.

Element mapping (Cheetah -> xsuite):

- Drift -> Drift
- Quadrupole -> thick Quadrupole; misalignment -> shift_x / shift_y, tilt -> rot_s_rad
- Sextupole -> thick Sextupole, with the same misalignment handling
- Dipole -> Bend from length and angle with linear edges; Cheetah's full `gap` is MAD's
  2 * hgap, so hgap = gap / 2 on both faces
- CombinedCorrector -> Drift of the corrector length, then a thin Multipole with
  knl = [-horizontal_angle], ksl = [vertical_angle] (Cheetah kicks at the exit)
- Cavity (standing wave) -> Rosenzweig-Serafini model built from xsuite parts: entrance
  edge lens, sliced xsuite Cavity with ReferenceEnergyIncrease after each piece and
  ponderomotive lenses in between, exit edge lens. Unlike Cheetah's cavity, the
  ponderomotive focusing acts at each particle's own energy.
- TransverseDeflectingCavity with zero voltage -> Drift (the only case in the CLARA file)
- Screen, Marker -> Marker
- Aperture -> LimitRect or LimitEllipse (only present with `apertures=True`)

Distributions are stored in Cheetah's convention (x, px, y, py, tau, delta) with
tau = -zeta; px and py are canonical momenta normalised to the reference momentum.
xsuite's native twiss does not apply to this accelerating line, so `optics_along`
computes Twiss and dispersion from tracked test particles.

Ported from the validation work behind this folder: the conversion reproduces Cheetah's
4x4 map within 1e-4 to CLA-S07-SIM-APER-01 and 5e-4 to the readout, and the reference energy
exactly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import cheetah
import numpy as np
import torch
import xobjects as xo
import xtrack as xt

HERE = Path(__file__).resolve().parent
CHALLENGE_DIR = HERE.parent
# Nominal magnet settings: the strengths stored in the lattice file.
LATTICE_JSON = CHALLENGE_DIR / "data" / "clara" / "lattice_data" / "CLARA_cheetah.json"
# Magnet settings of the 2026-03-23 S07 quadrupole scan (built by make_gpsr_lattice.py).
# Vacuum chambers of the line, from STFC's SimFrame model of CLARA FEBE (see the file).
APERTURES_JSON = CHALLENGE_DIR / "data" / "clara" / "lattice_data" / "apertures.json"
GPSR_LATTICE_JSON = HERE / "CLARA_cheetah_S07_GPSR.json"
INITIAL_DISTRIBUTION = HERE / "initial_distribution.npz"
START = "cla_s02_sim_aper_01"  # the benchmark start, score.LATTICE_START
READOUT = "cla_feh_dia_scr_01"  # the benchmark readout screen, score.READOUT_SCREEN
RECONSTRUCTION_POINT = (
    "cla_s07_sim_aper_01"  # where the GPSR phase space is reconstructed
)

ELECTRON_REST_ENERGY_EV = xt.ELECTRON_MASS_EV
CAVITY_SLICES = 160  # per cavity; 1e-4 agreement with Cheetah to S07
TEST_AMPLITUDE = 1e-6  # m, rad or relative: small enough to stay linear
COORDINATES = ("x", "px", "y", "py", "zeta", "delta")


class AxisymmetricThinLens(xt.BeamElement):
    """Thin RF-focusing kick acting equally in both planes:

        px -= strength * x / (1 + delta)**chromatic_power
        py -= strength * y / (1 + delta)**chromatic_power

    Backtracking applies the opposite kick, so lines containing it run backwards natively.
    """

    _xofields: ClassVar[dict] = {
        "strength": xo.Float64,  # 1/m, positive focuses both planes
        "chromatic_power": xo.Float64,  # 0: independent of energy, 1: scales as 1/(1+delta)
    }

    isthick = False
    has_backtrack = True
    allow_kernel_compilation = True  # custom element: not in xsuite's prebuilt kernels

    _extra_c_sources: ClassVar[list] = [
        r"""
#include "xtrack/headers/track.h"

GPUFUN
void AxisymmetricThinLens_track_local_particle(AxisymmetricThinLensData el,
                                               LocalParticle* part0){
    double strength = AxisymmetricThinLensData_get_strength(el);
    double const chromatic_power = AxisymmetricThinLensData_get_chromatic_power(el);
    if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
        strength = -strength;
    }
    START_PER_PARTICLE_BLOCK(part0, part);
        double const scale = pow(1.0 + LocalParticle_get_delta(part), -chromatic_power);
        LocalParticle_add_to_px(part, -strength * scale * LocalParticle_get_x(part));
        LocalParticle_add_to_py(part, -strength * scale * LocalParticle_get_y(part));
    END_PER_PARTICLE_BLOCK;
}
"""
    ]


@dataclass
class ConvertedLine:
    """An xsuite line built from a Cheetah segment, with its reference energies and, for
    every Cheetah element, the number of xsuite elements up to and including it."""

    line: xt.Line
    initial_energy_ev: float
    final_energy_ev: float
    boundaries: list[tuple[str, int]] = field(default_factory=list)


def momentum_ev(total_energy_ev: float) -> float:
    return math.sqrt(total_energy_ev**2 - ELECTRON_REST_ENERGY_EV**2)


def electron_particles(total_energy_ev: float, **coordinates) -> xt.Particles:
    return xt.Particles(
        mass0=ELECTRON_REST_ENERGY_EV,
        q0=-1,
        p0c=momentum_ev(total_energy_ev),
        **coordinates,
    )


def _alignment(element) -> dict:
    misalignment = getattr(element, "misalignment", None)
    shifts = (
        {}
        if misalignment is None
        else {"shift_x": float(misalignment[0]), "shift_y": float(misalignment[1])}
    )
    tilt = float(getattr(element, "tilt", 0.0))
    return {**shifts, **({"rot_s_rad": tilt} if tilt else {})}


def _standing_wave_cavity(
    name: str, cavity: cheetah.Cavity, incoming_energy_ev: float
) -> tuple[list, float]:
    """Rosenzweig-Serafini standing-wave cavity (eta = 1), as Cheetah's matrix assumes.

    gamma' = energy gain / (m c^2 L). Entrance lens -gamma'/(2 gamma_in) and exit lens
    +gamma'/(2 gamma_out), independent of the particle energy in canonical px.
    Ponderomotive focusing (gamma' / cos phi)^2 / (8 gamma^2) per unit length as thin
    lenses at slice centres, scaling as 1/(1+delta). Acceleration and adiabatic damping
    come from xsuite's Cavity plus ReferenceEnergyIncrease.
    """
    length = float(cavity.length)
    phase = math.radians(float(cavity.phase))
    energy_gain = float(cavity.voltage) * math.cos(
        phase
    )  # electrons: Cheetah's effective voltage is +V
    gradient = energy_gain / ELECTRON_REST_ENERGY_EV / length  # d gamma / ds
    gamma_in = incoming_energy_ev / ELECTRON_REST_ENERGY_EV
    gamma_out = (incoming_energy_ev + energy_gain) / ELECTRON_REST_ENERGY_EV
    ponderomotive_gradient_squared = (gradient / math.cos(phase)) ** 2

    elements = [
        (
            f"{name}..entry_lens",
            AxisymmetricThinLens(
                strength=gradient / (2 * gamma_in), chromatic_power=0.0
            ),
        )
    ]
    energy = incoming_energy_ev
    piece_length, piece_gain = (
        length / (2 * CAVITY_SLICES),
        energy_gain / (2 * CAVITY_SLICES),
    )
    for index in range(CAVITY_SLICES):
        for half in (0, 1):
            elements.append(
                (
                    f"{name}..cav{index}_{half}",
                    xt.Cavity(
                        length=piece_length,
                        voltage=float(cavity.voltage) / (2 * CAVITY_SLICES),
                        frequency=float(cavity.frequency),
                        phase=math.pi / 2 - phase,
                    ),
                )
            )
            new_energy = energy + piece_gain
            elements.append(
                (
                    f"{name}..ref{index}_{half}",
                    xt.ReferenceEnergyIncrease(
                        Delta_p0c=momentum_ev(new_energy) - momentum_ev(energy)
                    ),
                )
            )
            energy = new_energy
            if half == 0:
                gamma_middle = energy / ELECTRON_REST_ENERGY_EV
                strength = (
                    ponderomotive_gradient_squared
                    / (8 * gamma_middle**2)
                    * 2
                    * piece_length
                )
                elements.append(
                    (
                        f"{name}..pond{index}",
                        AxisymmetricThinLens(strength=strength, chromatic_power=1.0),
                    )
                )
    elements.append(
        (
            f"{name}..exit_lens",
            AxisymmetricThinLens(
                strength=-gradient / (2 * gamma_out), chromatic_power=0.0
            ),
        )
    )
    return elements, energy


def _dipole(element: cheetah.Dipole) -> xt.Bend:
    length, angle = float(element.length), float(element.angle)
    edges = {
        "both": ("linear", "linear"),
        "entrance": ("linear", "suppressed"),
        "exit": ("suppressed", "linear"),
        "no_edge": ("suppressed", "suppressed"),
    }
    entry_model, exit_model = edges[element.fringe_at]
    half_gap = float(element.gap) / 2  # Cheetah applies its entrance gap on both faces
    return xt.Bend(
        length=length,
        angle=angle,
        k0=angle / length if length else 0.0,
        k1=float(element.k1),
        edge_entry_angle=float(element.dipole_e1),
        edge_exit_angle=float(element.dipole_e2),
        edge_entry_hgap=half_gap,
        edge_exit_hgap=half_gap,
        edge_entry_fint=float(element.fringe_integral),
        edge_exit_fint=float(element.fringe_integral_exit),
        edge_entry_model=entry_model,
        edge_exit_model=exit_model,
        **_alignment(element),
    )


def _convert_element(element, energy_ev: float) -> tuple[list, float]:
    """xsuite (name, element) pairs for one Cheetah element, and the outgoing energy."""
    name = element.name
    if isinstance(element, cheetah.Cavity):
        if float(element.voltage) == 0.0:
            return [(name, xt.Drift(length=float(element.length)))], energy_ev
        if element.cavity_type != "standing_wave":
            raise NotImplementedError(f"{name}: cavity type {element.cavity_type}")
        return _standing_wave_cavity(name, element, energy_ev)
    if isinstance(element, cheetah.Drift):
        converted = xt.Drift(length=float(element.length))
    elif isinstance(element, cheetah.Quadrupole):
        converted = xt.Quadrupole(
            length=float(element.length), k1=float(element.k1), **_alignment(element)
        )
    elif isinstance(element, cheetah.Sextupole):
        converted = xt.Sextupole(
            length=float(element.length), k2=float(element.k2), **_alignment(element)
        )
    elif isinstance(element, cheetah.Dipole):
        converted = _dipole(element)
    elif isinstance(element, cheetah.CombinedCorrector):
        kick = xt.Multipole(
            knl=[-float(element.horizontal_angle)], ksl=[float(element.vertical_angle)]
        )
        if float(element.length) == 0.0:
            return [(name, kick)], energy_ev
        return [
            (f"{name}..body", xt.Drift(length=float(element.length))),
            (name, kick),
        ], energy_ev
    elif isinstance(element, cheetah.TransverseDeflectingCavity):
        if float(element.voltage) != 0.0:
            raise NotImplementedError(f"{name}: TDC with non-zero voltage")
        converted = xt.Drift(length=float(element.length))
    elif isinstance(element, (cheetah.Screen, cheetah.Marker)):
        converted = xt.Marker()
    elif isinstance(element, cheetah.Aperture):
        converted = _aperture(element)
    else:
        raise NotImplementedError(
            f"{name}: no xsuite mapping for {type(element).__name__}"
        )
    return [(name, converted)], energy_ev


def _aperture(element: cheetah.Aperture):
    """xsuite limit for a Cheetah aperture; an inactive one is a marker."""
    if not element.is_active:
        return xt.Marker()
    x_max, y_max = float(element.x_max), float(element.y_max)
    if element.shape == "elliptical":
        return xt.LimitEllipse(a=x_max, b=y_max)
    return xt.LimitRect(min_x=-x_max, max_x=x_max, min_y=-y_max, max_y=y_max)


def load_apertures(path: str | Path = APERTURES_JSON) -> dict:
    """The aperture table: `pipe` and `collimators` entries keyed by element name."""
    with open(path) as handle:
        table = json.load(handle)
    return {
        kind: {entry["element"]: entry for entry in table[kind]}
        for kind in ("pipe", "collimators")
    }


def _aperture_element(entry: dict, name: str, dtype: torch.dtype) -> cheetah.Aperture:
    return cheetah.Aperture(
        x_max=torch.tensor(entry["x_max"], dtype=dtype),
        y_max=torch.tensor(entry["y_max"], dtype=dtype),
        shape=entry["shape"],
        name=name,
    )


def with_apertures(
    segment: cheetah.Segment, apertures: dict | None = None
) -> cheetah.Segment:
    """The segment with its vacuum chamber added as thin Cheetah apertures.

    A `pipe` entry sets the chamber from its element onwards: the beam is checked
    there and at the exit of every element with a length until the next entry.
    SimFrame applies its aperture markers at the marker only; extending each one
    to the next is our reading of them as the chamber of that section. A `collimator` entry is checked at the entrance and
    exit of its own element. Apertures are named `<element>_pipe`, `<element>_col_in`
    and `<element>_col_out` and stay on the design axis when magnets are misaligned.
    """
    apertures = load_apertures() if apertures is None else apertures
    pipe, collimators = apertures["pipe"], apertures["collimators"]
    dtype = torch.get_default_dtype()
    elements, chamber = [], None
    for element in segment.elements:
        name = element.name
        collimator = collimators.get(name)
        if collimator is not None:
            elements.append(_aperture_element(collimator, f"{name}_col_in", dtype))
        elements.append(element)
        if name in pipe:
            chamber = pipe[name]
            elements.append(_aperture_element(chamber, f"{name}_pipe", dtype))
        elif collimator is not None:
            elements.append(_aperture_element(collimator, f"{name}_col_out", dtype))
        elif chamber is not None and float(getattr(element, "length", 0.0)) > 0.0:
            elements.append(_aperture_element(chamber, f"{name}_pipe", dtype))
    return cheetah.Segment(elements, name=segment.name)


def cheetah_segment(
    lattice_json: str | Path = LATTICE_JSON,
    start: str = START,
    end: str = READOUT,
    apertures: bool = False,
) -> cheetah.Segment:
    """The Cheetah segment after `start` (excluded) up to `end` (included).

    With `apertures`, the vacuum chamber of `APERTURES_JSON` is added (`with_apertures`);
    the chamber at `start` comes along as the first element.
    """
    full_line = cheetah.Segment.from_lattice_json(str(lattice_json))
    if apertures:
        full_line = with_apertures(full_line)
    after_start = full_line.partition_at(start, mode="after")[-1]
    return after_start.partition_at(end, mode="after")[0]


def build_line(
    initial_energy_ev: float,
    lattice_json: str | Path = LATTICE_JSON,
    start: str = START,
    end: str = READOUT,
    apertures: bool = False,
) -> ConvertedLine:
    """xsuite line from `start` to `end` with the magnet settings of `lattice_json`, for a
    reference particle of `initial_energy_ev` total energy at `start`. With `apertures`
    the line carries the vacuum chamber and loses particles on it."""
    names, elements, boundaries = [], [], []
    energy = initial_energy_ev
    for element in cheetah_segment(lattice_json, start, end, apertures).elements:
        pieces, energy = _convert_element(element, energy)
        for piece_name, piece in pieces:
            names.append(piece_name)
            elements.append(piece)
        boundaries.append((element.name, len(elements)))
    line = xt.Line(elements=elements, element_names=names)
    line.particle_ref = electron_particles(initial_energy_ev)
    line.build_tracker()
    return ConvertedLine(
        line=line,
        initial_energy_ev=initial_energy_ev,
        final_energy_ev=energy,
        boundaries=boundaries,
    )


def _particles(coordinates: np.ndarray, energy_ev: float) -> xt.Particles:
    """xsuite particles from Cheetah-convention (x, px, y, py, tau, delta)."""
    x, px, y, py, tau, delta = np.asarray(coordinates, dtype=float).T
    return electron_particles(energy_ev, x=x, px=px, y=y, py=py, zeta=-tau, delta=delta)


def _coordinates(particles: xt.Particles) -> np.ndarray:
    """Cheetah-convention coordinates in the original particle order."""
    if np.any(particles.state < 1):
        raise RuntimeError(
            f"{int(np.sum(particles.state < 1))} particles lost while tracking"
        )
    order = np.argsort(particles.particle_id)
    coordinates = np.column_stack(
        [getattr(particles, key)[order] for key in COORDINATES]
    )
    coordinates[:, 4] *= -1  # zeta -> tau
    return coordinates


def track_forward(converted: ConvertedLine, coordinates: np.ndarray) -> np.ndarray:
    """Track (N, 6) Cheetah-convention coordinates from the start to the end of the line."""
    particles = _particles(coordinates, converted.initial_energy_ev)
    converted.line.track(particles)
    return _coordinates(particles)


def track_backward(converted: ConvertedLine, coordinates: np.ndarray) -> np.ndarray:
    """Track (N, 6) coordinates given at the end of the line back to its start (exact
    reverse of track_forward)."""
    particles = _particles(coordinates, converted.final_energy_ev)
    converted.line.track(particles, backtrack=True)
    return _coordinates(particles)


def twiss_from_coordinates(coordinates: np.ndarray) -> dict:
    """Projected Twiss and geometric emittance per plane from (x, px, y, py, ...)."""
    twiss = {}
    for plane, column in (("x", 0), ("y", 2)):
        covariance = np.cov(coordinates[:, column], coordinates[:, column + 1])
        emittance = float(np.sqrt(np.linalg.det(covariance)))
        twiss.update(
            {
                f"beta_{plane}": float(covariance[0, 0] / emittance),
                f"alpha_{plane}": float(-covariance[0, 1] / emittance),
                f"emittance_{plane}": emittance,
            }
        )
    return twiss


def beam_summary(coordinates: np.ndarray, energy_ev: float) -> dict:
    """Twiss, normalised emittance, rms sizes and centroids, bunch length and energy spread."""
    summary = twiss_from_coordinates(coordinates)
    beta_gamma = math.sqrt((energy_ev / ELECTRON_REST_ENERGY_EV) ** 2 - 1)
    for plane, column in (("x", 0), ("y", 2)):
        summary[f"normalised_emittance_{plane}"] = (
            summary[f"emittance_{plane}"] * beta_gamma
        )
        summary[f"sigma_{plane}"] = float(coordinates[:, column].std())
        summary[f"sigma_{plane}_prime"] = float(coordinates[:, column + 1].std())
        summary[f"mean_{plane}"] = float(coordinates[:, column].mean())
    summary.update(
        sigma_tau=float(coordinates[:, 4].std()),
        sigma_delta=float(coordinates[:, 5].std()),
        correlation_xy=float(np.corrcoef(coordinates[:, 0], coordinates[:, 2])[0, 1]),
        energy_mev=energy_ev / 1e6,
    )
    return summary


def optics_along(converted: ConvertedLine, initial_twiss: dict) -> dict:
    """Beta, alpha and dispersion after every Cheetah element, from tracked test particles,
    starting from `initial_twiss` (keys beta_x, alpha_x, beta_y, alpha_y)."""
    test = np.zeros((8, 6))
    for row in range(6):
        test[row + 1, row] = TEST_AMPLITUDE
    test[7, 5] = -TEST_AMPLITUDE
    particles = _particles(test, converted.initial_energy_ev)
    converted.line.track(particles, turn_by_turn_monitor="ONE_TURN_EBE")
    record = converted.line.record_last_track
    order = np.argsort(record.particle_id[:, 0])
    s_entrances = np.asarray(
        converted.line.get_table().s
    )  # one entry per element plus the end point
    columns: dict = {
        key: []
        for key in (
            "element",
            "s",
            "energy_mev",
            "beta_x",
            "alpha_x",
            "beta_y",
            "alpha_y",
            "dispersion_x",
            "dispersion_y",
        )
    }
    for name, index in converted.boundaries:
        state = np.column_stack(
            [getattr(record, key)[order, index] for key in COORDINATES]
        )
        transfer = ((state[1:5, :4] - state[0, :4]) / TEST_AMPLITUDE).T
        columns["element"].append(name)
        columns["s"].append(float(s_entrances[index]))
        columns["energy_mev"].append(
            math.hypot(float(record.p0c[0, index]), ELECTRON_REST_ENERGY_EV) / 1e6
        )
        for plane, offset in (("x", 0), ("y", 2)):
            r11, r12 = transfer[offset, offset], transfer[offset, offset + 1]
            r21, r22 = transfer[offset + 1, offset], transfer[offset + 1, offset + 1]
            beta0, alpha0 = (
                initial_twiss[f"beta_{plane}"],
                initial_twiss[f"alpha_{plane}"],
            )
            gamma0 = (1 + alpha0**2) / beta0
            damping = r11 * r22 - r12 * r21  # p0 / p for this plane
            columns[f"beta_{plane}"].append(
                (beta0 * r11**2 - 2 * alpha0 * r11 * r12 + gamma0 * r12**2) / damping
            )
            columns[f"alpha_{plane}"].append(
                -(
                    beta0 * r11 * r21
                    - alpha0 * (r11 * r22 + r12 * r21)
                    + gamma0 * r12 * r22
                )
                / damping
            )
            momentum_span = state[6, 5] - state[7, 5]
            columns[f"dispersion_{plane}"].append(
                (state[6, offset] - state[7, offset]) / momentum_span
            )
    return {key: np.asarray(values) for key, values in columns.items()}


def load_distribution(
    path: str | Path = INITIAL_DISTRIBUTION,
) -> tuple[np.ndarray, float, dict]:
    """(particles (N, 6) Cheetah convention, reference total energy in eV, metadata)."""
    with np.load(path, allow_pickle=False) as data:
        metadata = {
            key: data[key].item()
            for key in data.files
            if key not in ("particles", "energy_ev")
        }
        return data["particles"], float(data["energy_ev"]), metadata


def save_distribution(
    path: str | Path, particles: np.ndarray, energy_ev: float, **metadata: str
) -> None:
    np.savez(
        path,
        particles=particles.astype(np.float64),
        energy_ev=energy_ev,
        **{key: np.asarray(str(value)) for key, value in metadata.items()},
    )
