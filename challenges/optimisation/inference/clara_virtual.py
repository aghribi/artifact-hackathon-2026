"""Virtual CLARA for steering and optics inference, built on the nominal xsuite model.

Two copies of the same xsuite line are used everywhere in this folder:

- the *machine* (`VirtualMachine`): the line plus hidden errors (launch offset, mean
  energy error, quadrupole misalignments) and a hidden 6D beam. It only answers with
  what an instrument gives: BPM centroids with noise, or the beam on one inserted YAG
  screen (destructive: the beam stops there).
- the *model* (`DesignModel`): the error-free line the inference uses. Its transfer maps
  come from xsuite test-particle tracking, so every effect in the xsuite line
  (per-particle thick maps, Rosenzweig-Serafini cavities) is in the inference.

Quadrupoles are driven in amperes through the CLARA calibration in
`data/clara/lattice_data/utils.py`, at the model reference momentum of each magnet,
limited to +-100 A (all quadrupoles are bipolar).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_CHALLENGE = Path(__file__).resolve().parents[1]  # challenges/optimisation
sys.path.insert(0, str(REPO_CHALLENGE / "nominal_model"))
sys.path.insert(0, str(REPO_CHALLENGE / "data" / "clara" / "lattice_data"))

import utils as clara_calibration
import xsuite_model

QUAD_CURRENT_LIMIT_A = 100.0
TEST_AMPLITUDE = 1e-6
PHASE_SPACE_LABELS = ("x", "px", "y", "py", "tau", "delta")


@dataclass
class LineInventory:
    """Names, positions and reference momenta of instruments and adjustable magnets."""

    quads: list[str]
    correctors: list[str]
    bpms: list[str]
    screens: list[str]
    momentum_mevc_at: dict[str, float]
    s_at: dict[str, float]

    @classmethod
    def from_converted(cls, converted: xsuite_model.ConvertedLine) -> LineInventory:
        line = converted.line
        table = line.get_table()
        s_at = dict(zip(table.name, table.s))
        particles = xsuite_model._particles(
            np.zeros((1, 6)), converted.initial_energy_ev
        )
        line.track(particles, turn_by_turn_monitor="ONE_TURN_EBE")
        p0c_along = np.asarray(line.record_last_track.p0c[0])
        momentum_mevc_at = {
            name: float(p0c_along[index]) / 1e6 for name, index in converted.boundaries
        }
        # apertures (`<element>_pipe`, `_col_in`, `_col_out`) carry their element's name
        names = [name for name, _ in converted.boundaries
                 if not name.endswith(("_pipe", "_col_in", "_col_out"))]
        return cls(
            quads=[n for n in names if "_mag_quad_" in n],
            correctors=[n for n in names if "_mag_hvcor_" in n or "_mag_vcor_" in n],
            bpms=[n for n in names if "_dia_bpm_" in n],
            screens=[n for n in names if "_dia_scr_" in n or n.endswith("_dcp_scr_01")],
            momentum_mevc_at=momentum_mevc_at,
            s_at={n: float(s_at[n]) for n in names if n in s_at},
        )


class XsuiteLine:
    """Magnet settings and tracking on one converted line."""

    def __init__(
        self,
        converted: xsuite_model.ConvertedLine,
        inventory: LineInventory | None = None,
    ):
        self.converted = converted
        self.line = converted.line
        self.inventory = inventory or LineInventory.from_converted(converted)
        self.nominal_k1 = {q: float(self.line[q].k1) for q in self.inventory.quads}
        self.element_index = {name: i for i, name in enumerate(self.line.element_names)}

    # magnets ---------------------------------------------------------------
    def quad_k1_from_current(self, quad: str, current_a: float) -> float:
        if abs(current_a) > QUAD_CURRENT_LIMIT_A + 1e-9:
            raise ValueError(
                f"{quad}: {current_a:.1f} A is beyond +-{QUAD_CURRENT_LIMIT_A:.0f} A"
            )
        return clara_calibration.quadrupole_k_from_current(
            quad, current_a, self.inventory.momentum_mevc_at[quad]
        )

    def quad_current_from_k1(self, quad: str, k1: float) -> float:
        return clara_calibration.quadrupole_current_from_k(
            quad, k1, self.inventory.momentum_mevc_at[quad]
        )

    def quad_k1_limits(self, quad: str) -> tuple[float, float]:
        return (
            self.quad_k1_from_current(quad, -QUAD_CURRENT_LIMIT_A),
            self.quad_k1_from_current(quad, QUAD_CURRENT_LIMIT_A),
        )

    def set_quad_k1(self, settings: dict[str, float]) -> None:
        for quad, k1 in settings.items():
            self.line[quad].k1 = float(k1)

    def set_quad_currents(self, currents: dict[str, float]) -> None:
        self.set_quad_k1(
            {q: self.quad_k1_from_current(q, i) for q, i in currents.items()}
        )

    def restore_nominal_quads(self) -> None:
        self.set_quad_k1(self.nominal_k1)

    def set_corrector(
        self, name: str, horizontal_rad: float = 0.0, vertical_rad: float = 0.0
    ) -> None:
        self.line[name].knl[0] = -horizontal_rad
        self.line[name].ksl[0] = vertical_rad

    # tracking --------------------------------------------------------------
    def track_through(
        self, coordinates: np.ndarray, stops: list[str], allow_losses: bool = False
    ) -> dict[str, np.ndarray]:
        """Track from the line start, returning (N, 6) Cheetah coordinates at the exit of
        each element in `stops` (any order; tracking ends at the last one). With
        `allow_losses`, lost particles are dropped instead of raising."""
        ordered = sorted(stops, key=self.element_index.__getitem__)
        particles = xsuite_model._particles(
            coordinates, self.converted.initial_energy_ev
        )
        recorded, start = {}, 0
        for name in ordered:
            stop = self.element_index[name] + 1
            if stop > start:
                self.line.track(particles, ele_start=start, ele_stop=stop)
            recorded[name] = _coordinates_keep_order(particles, allow_losses)
            start = stop
        return recorded

    def linear_map(
        self, stops: list[str], reference: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """6x6 linear map (Cheetah coordinates) from the line start to each stop, about a
        reference trajectory (default on axis), by finite differences of tracked particles."""
        reference = np.zeros(6) if reference is None else np.asarray(reference, float)
        test = np.tile(reference, (13, 1))
        for column in range(6):
            test[1 + 2 * column, column] += TEST_AMPLITUDE
            test[2 + 2 * column, column] -= TEST_AMPLITUDE
        recorded = self.track_through(test, stops)
        maps = {}
        for name, state in recorded.items():
            columns = [
                (state[1 + 2 * c] - state[2 + 2 * c]) / (2 * TEST_AMPLITUDE)
                for c in range(6)
            ]
            maps[name] = np.column_stack(columns)
        return maps

    def reference_orbit(
        self, stops: list[str], start_state: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Single-particle trajectory from `start_state` (default on axis)."""
        start_state = (
            np.zeros(6) if start_state is None else np.asarray(start_state, float)
        )
        return {
            k: v[0] for k, v in self.track_through(start_state[None, :], stops).items()
        }


def _coordinates_keep_order(particles, allow_losses: bool = False) -> np.ndarray:
    """Cheetah-convention copy of the particles in particle-id order; lost particles raise
    unless `allow_losses`, in which case only surviving particles are returned."""
    if not allow_losses:
        return xsuite_model._coordinates(particles)
    alive = particles.state > 0
    order = np.argsort(particles.particle_id[alive])
    coordinates = np.column_stack([getattr(particles, key)[alive][order] for key in xsuite_model.COORDINATES])
    coordinates[:, 4] *= -1
    return coordinates


@dataclass
class HiddenErrors:
    """What the machine knows and the inference must not read."""

    launch: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )  # added to every particle at the start
    quad_offsets: dict[str, tuple[float, float]] = field(
        default_factory=dict
    )  # (dx, dy) in m
    bpm_offsets: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def random(
        cls,
        inventory: LineInventory,
        seed: int = 0,
        launch_rms=(3e-4, 3e-5, 3e-4, 3e-5, 0.0, 2e-3),
        quad_offset_rms: float = 1e-4,
        bpm_offset_rms: float = 0.0,
    ) -> HiddenErrors:
        rng = np.random.default_rng(seed)
        return cls(
            launch=rng.normal(0, launch_rms),
            quad_offsets={
                q: tuple(rng.normal(0, quad_offset_rms, 2)) for q in inventory.quads
            },
            bpm_offsets={
                b: tuple(rng.normal(0, bpm_offset_rms, 2)) for b in inventory.bpms
            },
        )


class VirtualMachine:
    """The 'real' CLARA: hidden beam and errors, answers only with instrument readings."""

    def __init__(
        self,
        line: XsuiteLine,
        beam: np.ndarray,
        errors: HiddenErrors,
        seed: int = 1,
        bpm_noise_m: float = 10e-6,
        screen_resolution_m: float = 20e-6,
        shot_jitter: np.ndarray | None = None,
    ):
        self._line = line
        self._beam = beam
        self._errors = errors
        self._rng = np.random.default_rng(seed)
        self.bpm_noise_m = bpm_noise_m
        self.screen_resolution_m = screen_resolution_m
        # shot-to-shot jitter of the launch (rms per coordinate), on top of the fixed offset
        self.shot_jitter = (
            np.zeros(6) if shot_jitter is None else np.asarray(shot_jitter)
        )
        for quad, (dx, dy) in errors.quad_offsets.items():
            line.line[quad].shift_x = dx
            line.line[quad].shift_y = dy
        self.inventory = line.inventory
        self.shots = 0

    def set_quad_currents(self, currents: dict[str, float]) -> None:
        self._line.set_quad_currents(currents)

    def restore_nominal_quads(self) -> None:
        self._line.restore_nominal_quads()

    def set_corrector(
        self, name: str, horizontal_rad: float = 0.0, vertical_rad: float = 0.0
    ) -> None:
        self._line.set_corrector(name, horizontal_rad, vertical_rad)

    def _shot_beam(self) -> np.ndarray:
        self.shots += 1
        return (
            self._beam
            + self._errors.launch
            + self._rng.normal(0, 1, 6) * self.shot_jitter
        )

    def read_bpms(self, bpms: list[str] | None = None) -> dict[str, np.ndarray]:
        """One shot, all screens retracted: (x, y) centroid at each BPM, with noise."""
        bpms = bpms or self.inventory.bpms
        recorded = self._line.track_through(self._shot_beam(), bpms, allow_losses=True)
        readings = {}
        for name, coords in recorded.items():
            offset = np.asarray(self._errors.bpm_offsets.get(name, (0.0, 0.0)))
            centroid = coords[:, [0, 2]].mean(axis=0) - offset
            readings[name] = centroid + self._rng.normal(0, self.bpm_noise_m, 2)
        return readings

    def screen_moments(self, screen: str) -> dict[str, float]:
        """One shot on one inserted screen: centroid and second moments of (x, y), with the
        screen resolution added in quadrature and 1 % relative noise on the sizes."""
        coords = self._line.track_through(self._shot_beam(), [screen], allow_losses=True)[screen]
        x, y = coords[:, 0], coords[:, 2]
        noise = 1 + self._rng.normal(0, 0.01, 3)
        resolution = self.screen_resolution_m**2
        return {
            "mean_x": float(x.mean() + self._rng.normal(0, 2e-6)),
            "mean_y": float(y.mean() + self._rng.normal(0, 2e-6)),
            "xx": float((x.var() + resolution) * noise[0]),
            "yy": float((y.var() + resolution) * noise[1]),
            "xy": float(np.cov(x, y)[0, 1] * noise[2]),
        }

    def screen_image(
        self, screen: str, half_width_m: float = 5e-3, pixels: int = 200
    ) -> tuple[np.ndarray, np.ndarray]:
        """One shot on one inserted screen: 2D histogram (y rows, x columns) and pixel edges."""
        coords = self._line.track_through(self._shot_beam(), [screen], allow_losses=True)[screen]
        blur = self._rng.normal(0, self.screen_resolution_m, (len(coords), 2))
        edges = np.linspace(-half_width_m, half_width_m, pixels + 1)
        image, _, _ = np.histogram2d(
            coords[:, 2] + blur[:, 1], coords[:, 0] + blur[:, 0], bins=[edges, edges]
        )
        return image, edges


def build_pair(
    start: str = xsuite_model.START,
    end: str = xsuite_model.READOUT,
    beam_particles: int = 20000,
    errors: HiddenErrors | None = None,
    seed: int = 0,
    **machine_kwargs,
):
    """(machine, model, true start beam) on the segment start..end."""
    beam, energy_ev, _ = xsuite_model.load_distribution()
    rng = np.random.default_rng(seed)
    beam = beam[rng.choice(len(beam), beam_particles, replace=False)]
    machine_line = XsuiteLine(xsuite_model.build_line(energy_ev, start=start, end=end))
    model_line = XsuiteLine(
        xsuite_model.build_line(energy_ev, start=start, end=end), machine_line.inventory
    )
    errors = errors or HiddenErrors.random(machine_line.inventory, seed=seed)
    machine = VirtualMachine(
        machine_line, beam, errors, seed=seed + 1, **machine_kwargs
    )
    return machine, model_line, beam
