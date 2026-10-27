"""A simulated accelerator, described by a `MachineSpec`.

    env = SimEnv(RLConfig.clara())                       # the benchmark machine
    env = SimEnv(RLConfig(spec="machines/clear.json"))   # or any other

Which accelerator this is comes from the spec and nothing else: the lattice, the
beam, the vacuum chamber, which elements are switched off, which diagnostics are
BPMs and which are screens, which magnets move together and with what signs, and
what every magnet may be set to. `RLConfig.clara()` loads the one `score.py` is
judged on — the measured beam at the start of S02 tracked to the FEBE readout —
and gives exactly what a bare `RLConfig()` used to.

Tracking is Cheetah, the first-order physics the benchmark is scored on. The
thick, chromatic xsuite backend used to live here behind `config.backend` and
does not any more; `nominal_model/xsuite_model.py` is still there for the code
that uses it directly.

An action is a dictionary::

    env.step({"magnets": {"cla_s04_mag_quad_01": 1.2,
                          "cla_s04_mag_hvcor_01_h": 1e-4},
              "screen": "cla_s04_dia_scr_03"})

Only the magnets named are moved; every other magnet stays where it was. Both entries
are optional, and a bare mapping of magnet settings is read as `{"magnets": ...}`.
Settings may also be given in the nested form `score.py` uses,
`{"cla_s04_mag_hvcor_01": {"horizontal_angle": 1e-4}}`. Values are clipped to the
magnet's operating range (`env.magnet_limits`).

`screen` picks the one screen that is in the beam — only one may be, and a screen in the
beam intercepts it, so tracking stops there and every BPM downstream reads nothing.

An observation is a dictionary of the BPM readings and the image on that screen::

    {"bpm_x", "bpm_y":  metres off centre, in `env.bpm_names` order
     "bpm_valid":       1 where enough beam reached that BPM to give a reading
     "screen_image":    the viewed screen, rows are y, columns x, origin at the bottom
     "screen_index":    which of `env.screen_names` the image came from
     "target":          what the beam is asked for on that screen this episode,
                        (mu_x, mu_y, sigma_x, sigma_y) in metres off its centre}

Positions are relative to the diagnostic's own centre, as a camera or a BPM reports
them, and particles that leave the beam pipe stop counting towards any reading
further down the line.

Every `reset()` also draws a different request — the spot the beam is wanted at,
from the same distribution `score.sample_target` uses — and the reward is how far
the reading is from it, in units of the spot size asked for, exactly as
`score.per_case_scores` books it, less what the shot cost. `config.target` fixes
the request instead.

The request belongs to one screen, `config.target_screen`, the FEBE readout the
benchmark scores. An episode starts with that screen in the beam and is judged on
the last reading of it: looking at an upstream screen costs a shot and tells you
something, but brings back no new reading of the request. `info["benchmark_score"]`
is `score.Report.calc_score` for the episode as one case — the mismatch plus
`EVAL_WEIGHT` per doubling of `EVAL_BUDGET` shots — so it can be compared with a
scored tuner directly, while the reward itself stays dense enough to learn from.

Every `reset()` draws a different machine: a beam of its own size, shape and
position, and a survey error on the quadrupoles, sextupoles and diagnostics. How far
each is thrown is `config.beam_variation` and `config.misalignment_variation`, where
0 is the nominal model exactly and 1 is the realistic machine. The magnets start at
the strengths the lattice file asks for; the survey error an agent is stuck with, as
on the real machine. What was drawn is in `env.beam_offset` and `env.misalignments`.

Within an episode the machine then holds still but a shot of it does not. Every step
draws its own beam around the episode's, at `config.beam_jitter` of
`config.beam_variation` and left in `env.shot_offset`, and puts every magnet where
its power supply actually has it rather than where it was set, at
`config.power_supply_ripple` of its setting. Both are undone afterwards, so the
magnets read back what was asked for. Two identical actions do not give identical
readings, which is what reading a screen on a real machine is like.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cheetah
import numpy as np
import torch

from rl_framework.config import RLConfig
from rl_framework.envs.base_env import (
    BaseAcceleratorEnv,
    Diagnostic,
    TrackingTransport,
    lattice_io,
)

# Re-exported, not used here: `Refined/machines.py` takes the benchmark's shot
# pricing from this module by name, and it is the environment's to give even now
# that it is defined one layer down.
from rl_framework.envs.base_env import evaluation_cost  # noqa: F401

DTYPE = torch.float32  # what the Cheetah lattice file is parsed into

# The knob model, the diagnostics, the scoring and `step` are shared with the real
# machine and live in `base_env.py`; what is left here is the simulation. Magnet
# operating ranges, coupled-magnet groups, pole faces that follow a body angle, and
# which elements are switched off all used to be constants here, every one of them
# CLARA. They are properties of a machine, so they live in its `MachineSpec` — see
# `machine_spec/spec.py` and `machines/clara_febe.json`.

# ---- how far one episode's machine is from the next, at a variation of 1 ---- #

# Injection jitter, absolute and roughly what CLARA sees shot to shot, as in
# `score.sample_beam`. Order: x, px, y, py, tau, delta.
CENTROID_JITTER = np.array([50e-6, 50e-6, 50e-6, 50e-6, 100e-6, 1e-3])
# Multiplicative spreads, each landing in [1/s, s] and widening geometrically with
# the variation: a positive quantity cannot be widened by adding to it.
SIGMA_SPREAD = 1.3  # rms size of a plane, at constant shape
BETA_SPREAD = 1.2  # mismatch: size against divergence, at constant emittance
# Survey error, per plane, as in `score.sample_misalignments`.
MISALIGNMENT_RMS = 50e-6  # quadrupoles and sextupoles
DIAGNOSTIC_OFFSET_RMS = 30e-6  # the centring of a camera or a BPM


class CheetahTransport(TrackingTransport):
    """Cheetah: first-order magnets and achromatic cavities, tracked in torch.

    The physics `score.py` is scored on, and the faster of the two to set up — there
    is no tracking kernel to compile.
    """

    name = "cheetah"

    def __init__(self, segment: cheetah.Segment, energy_ev: float, stops: list[str]):
        self._segment = segment
        self._elements = {element.name: element for element in segment.elements}
        self._energy = torch.tensor(float(energy_ev), dtype=DTYPE)
        for element in segment.elements:
            if isinstance(element, cheetah.Screen):
                # The environment does its own imaging, and an active screen costs a
                # full camera frame at every diagnostic in the line.
                element.is_active = False
        # One sub-segment per stop, over the same element objects, so each stretch is
        # tracked as one fused transfer map instead of element by element.
        self._chunks, previous = {}, 0
        for stop in stops:
            index = segment.element_index(stop)
            self._chunks[stop] = cheetah.Segment(segment.elements[previous:index])
            previous = index

    def read(self, element: str, prop: str) -> float:
        return float(getattr(self._elements[element], prop))

    # `phase` needs nothing special here: Cheetah stores it on the cavity, in
    # degrees, and rebuilds the transfer map from it on every track.

    def write(self, element: str, prop: str, value: float,
              follow: Sequence[str] = ()) -> None:
        """Set a property, and anything that has to move with it.

        `follow` is the pole-face angles of a chicane dipole and the like: a
        property that is not independently settable but is not constant either.
        Which faces follow is a fact about the magnet, so it comes from the
        machine spec rather than from a table in here — setting a chicane angle
        without its edges is a different, far more lossy optic, and the error is
        invisible until the knob is first turned.
        """
        tensor = torch.tensor(value, dtype=DTYPE)
        setattr(self._elements[element], prop, tensor)
        for extra in follow:
            setattr(self._elements[element], extra, tensor)

    def read_shift(self, element: str) -> tuple[float, float]:
        misalignment = self._elements[element].misalignment
        return float(misalignment[0]), float(misalignment[1])

    def shift(self, element: str, dx: float, dy: float) -> None:
        self._elements[element].misalignment = torch.tensor([dx, dy], dtype=DTYPE)

    def run(self, coordinates, stops):
        particles = torch.tensor(
            np.column_stack([coordinates, np.ones(len(coordinates))]), dtype=DTYPE
        )
        beam = cheetah.ParticleBeam(particles=particles, energy=self._energy)
        for stop in stops:
            chunk = self._chunks[stop]
            if len(chunk.elements):
                beam = chunk.track(beam)
            yield (
                stop,
                beam.x.detach().numpy(),
                beam.y.detach().numpy(),
                beam.survival_probabilities.detach().numpy() > 0.5,
            )


def _spread(rng, spread: float, variation: float) -> float:
    """A factor log-uniform in [1/spread, spread], widened geometrically by
    `variation` so that a larger variation is more of the same, not something else."""
    reach = np.log(spread) * np.sqrt(variation)
    return float(np.exp(rng.uniform(-reach, reach)))


class SimEnv(BaseAcceleratorEnv):
    """A simulated machine, described by a `MachineSpec`.

    Nothing here knows which accelerator it is running. The lattice, the beam,
    the vacuum chamber, which elements are switched off, which diagnostics are
    BPMs and which are screens, and what every magnet may be set to all come
    from the spec. `RLConfig.clara()` loads the one the benchmark is scored on.

    The knob model, the diagnostics, the scoring and the gym surface are
    `BaseAcceleratorEnv`'s, shared with `RealEnv`. What is added here is the
    simulation: a beam, a tracker, and the ways one episode's machine differs
    from the next.
    """

    def __init__(self, config: RLConfig):
        super().__init__(config)

        # The strengths the lattice file carries, for every magnet of the line and
        # not just the settable ones — what each episode starts near.
        self._nominal_strengths = {
            name: knob.value for name, knob in self._knobs.items()
        }
        # Elements a survey error can move, and where the lattice file already has
        # them. Quadrupoles and sextupoles, as in `score.sample_misalignments`.
        self._misalignable = [
            element.name
            for element in self.segment.elements
            if isinstance(element, (cheetah.Quadrupole, cheetah.Sextupole))
        ]
        self._nominal_shifts = {
            name: self.transport.read_shift(name) for name in self._misalignable
        }
        self.misalignments = dict(self._nominal_shifts)
        self.beam_offset = np.zeros(6)
        self.shot_offset = np.zeros(6)
        # Where each diagnostic's centre is this episode, its own offset included.
        self._offsets = {
            name: np.array(diagnostic.misalignment)
            for name, diagnostic in self._diagnostics.items()
        }
        # Magnets ripple; a cavity phase is not a value a percentage means
        # anything against, so it does not.
        self._rippled = [
            name for name, knob in self._knobs.items() if knob.prop != "phase"
        ]

        self._simulate()
        self._scored = self._readings["stats"]
        self._scored_transmission = self._readings["transmission"]

    # ---------- the model doing the tracking ----------

    def _prepare(self) -> None:
        """Load the beam. Before the transport, which is built around its energy."""
        coordinates, energy_ev, self.beam_metadata = lattice_io.load_beam(
            self.spec, self.config.num_particles
        )
        # The beam as it was measured, and the one this episode was dealt.
        self._reference_coordinates = np.ascontiguousarray(coordinates)
        self._coordinates = self._reference_coordinates
        self.energy_ev = energy_ev

    def _make_transport(self) -> CheetahTransport:
        return CheetahTransport(
            self.segment, self.energy_ev, [d.name for d in self._stops]
        )

    def _extent_of(self, name: str) -> tuple[float, float, float, float]:
        """What the model says the chip spans. A simulated camera is the model."""
        return self._diagnostics[name].extent

    def _acquire(self) -> None:
        self._simulate()
    # ---------- the machine this episode was dealt ----------

    def _perturb_beam(self, coordinates, variation: float):
        """A beam of its own size, shape and position, from the one given.

        Each plane gets an rms size factor and a mismatch factor — size against
        divergence at constant emittance — and the whole beam an offset in all six
        coordinates. Returns the beam and the offset it was given.
        """
        if not variation:
            return coordinates, np.zeros(6)

        centre = coordinates.mean(axis=0)
        drawn = coordinates - centre
        for position, momentum in ((0, 1), (2, 3)):  # x-x', then y-y'
            size = _spread(self._rng, SIGMA_SPREAD, variation)
            mismatch = _spread(self._rng, BETA_SPREAD, variation)
            drawn[:, position] *= size * mismatch
            drawn[:, momentum] *= size / mismatch
        for column in (4, 5):  # bunch length and energy spread
            drawn[:, column] *= _spread(self._rng, SIGMA_SPREAD, variation)
        offset = self._rng.uniform(-CENTROID_JITTER, CENTROID_JITTER) * variation
        return drawn + centre + offset, offset

    @property
    def shot_variation(self) -> float:
        """How far one shot is from the next, against how far one episode's beam is
        from another's."""
        return self.config.beam_variation * self.config.beam_jitter

    def _draw_beam(self) -> np.ndarray:
        """The beam this episode was dealt, from the measured one."""
        coordinates, self.beam_offset = self._perturb_beam(
            self._reference_coordinates, self.config.beam_variation
        )
        return coordinates
    def _restore_nominal(self) -> None:
        """Every magnet and cavity back to what the lattice file asks for, exactly
        and whatever the limits say."""
        for name, knob in self._knobs.items():
            knob.set(self._nominal_strengths[name], clip=False)

    def _apply_ripple(self) -> dict[str, float] | None:
        """Put every magnet where its power supply actually has it for this shot.

        A supply holds its output to a fraction of what it was asked for, and that
        fraction wanders shot to shot, so what the beam sees is never quite the set
        point. The error is proportional, so a magnet parked at zero does not
        fluctuate, and it is drawn fresh for every magnet on every shot. Cavity
        phases are left alone: RF is a different mechanism and a phase is not a
        value a percentage means anything against.

        Returns what the magnets were set to, for `_restore_setpoint` to put back,
        or None when there is no ripple to apply and nothing to undo.
        """
        fraction = self.config.power_supply_ripple
        if not fraction:
            return None
        errors = self._rng.normal(0.0, fraction, len(self._rippled))
        commanded = {}
        for name, error in zip(self._rippled, errors):
            knob = self._knobs[name]
            commanded[name] = knob.value
            knob.set(commanded[name] * (1.0 + error), clip=False)
        return commanded

    def _restore_setpoint(self, commanded: dict[str, float] | None) -> None:
        """Undo the ripple: back to the values that were actually set."""
        if commanded is None:
            return
        for name, value in commanded.items():
            self._knobs[name].set(value, clip=False)

    def _draw_misalignments(self) -> dict[str, tuple[float, float]]:
        """A survey error on the magnets, and a centring error on the diagnostics."""
        variation = self.config.misalignment_variation

        def error(rms):
            return self._rng.normal(0.0, rms * variation, 2) if variation else np.zeros(2)

        self.misalignments = {}
        for name in self._misalignable:
            dx, dy = np.array(self._nominal_shifts[name]) + error(MISALIGNMENT_RMS)
            self.transport.shift(name, float(dx), float(dy))
            self.misalignments[name] = (float(dx), float(dy))
        # Readers get a centring error; an aperture marker is not a reader and
        # has nothing to be off-centre about.
        readers = set(self.bpm_names) | set(self.screen_names)
        for name, diagnostic in self._diagnostics.items():
            offset = np.array(diagnostic.misalignment)
            if name in readers:
                offset = offset + error(DIAGNOSTIC_OFFSET_RMS)
            self._offsets[name] = offset
        return self.misalignments
    # ---------- tracking ----------

    def _simulate(self) -> None:
        """One shot: the episode's beam jittered again, through magnets sitting
        where their power supplies actually have them, and read."""
        commanded = self._apply_ripple()
        try:
            self._track()
        finally:
            self._restore_setpoint(commanded)

    def _track(self) -> None:
        """Track the beam to the screen in the beam, reading every BPM on the way."""
        shot, self.shot_offset = self._perturb_beam(
            self._coordinates, self.shot_variation
        )
        count = len(shot)
        minimum = self.min_screen_fraction * count
        radius = self.beam_pipe_radius

        bpm_x = np.zeros(len(self.bpm_names))
        bpm_y = np.zeros(len(self.bpm_names))
        bpm_valid = np.zeros(len(self.bpm_names), dtype=np.int8)
        bpm_slot = {name: i for i, name in enumerate(self.bpm_names)}
        screen = self._diagnostics[self._target_screen]

        live = np.ones(count, dtype=bool)
        image = np.zeros(self._shape)
        fraction = 0.0
        stats = (0.0, 0.0, 0.0, 0.0, False)

        # Everything past the screen is in its shadow: a screen in the beam stops it.
        stops = [stop.name for stop in self._stops if stop.index <= screen.index]
        for name, x, y, alive in self.transport.run(shot, stops):
            live &= alive
            if radius is not None:
                live &= x**2 + y**2 <= radius**2  # scraped on the beam pipe

            if name in bpm_slot and live.sum() >= minimum:
                slot = bpm_slot[name]
                offset = self._offsets[name]
                bpm_x[slot] = x[live].mean() - offset[0]
                bpm_y[slot] = y[live].mean() - offset[1]
                bpm_valid[slot] = 1
            if name == screen.name:
                image, fraction, stats = self._expose(screen, x, y, live)

        if self.config.bpm_noise:
            noise = self._rng.normal(0.0, self.config.bpm_noise, (2, len(bpm_x)))
            bpm_x += noise[0] * bpm_valid
            bpm_y += noise[1] * bpm_valid

        self._readings = {
            "bpm_x": bpm_x,
            "bpm_y": bpm_y,
            "bpm_valid": bpm_valid,
            "image": image,
            "on_screen_fraction": fraction,
            "stats": stats,
            "transmission": float(live.sum()) / count,
        }

    def _expose(self, screen: Diagnostic, x: np.ndarray, y: np.ndarray, live: np.ndarray):
        """The image on a screen, the fraction of the beam that made it onto the chip,
        and the moments a camera would measure from it."""
        offset = self._offsets[screen.name]
        x = x[live] - offset[0]
        y = y[live] - offset[1]
        on_chip = (np.abs(x) <= screen.extent[1]) & (np.abs(y) <= screen.extent[3])
        x, y = x[on_chip], y[on_chip]

        rows, columns = self._bins(screen.name)
        image, _, _ = np.histogram2d(y, x, bins=(rows, columns))
        if self.config.normalise_image and image.max() > 0:
            image = image / image.max()

        fraction = float(on_chip.sum()) / len(live)
        if fraction < self.min_screen_fraction:
            return image, fraction, (0.0, 0.0, 0.0, 0.0, False)
        stats = (
            float(x.mean()),
            float(y.mean()),
            float(x.std()),
            float(y.std()),
            True,
        )
        return image, fraction, stats

    # ---------- base class hooks ----------

    def _hard_reset(self) -> dict[str, Any]:
        """A new machine: its own beam, set point and survey error, at the configured
        screen. With every variation at 0 this is the nominal model exactly."""
        self.target = self._draw_target()
        self._coordinates = self._draw_beam()
        self._scored, self._scored_transmission = (0.0, 0.0, 0.0, 0.0, False), 0.0
        self._restore_nominal()
        self._draw_misalignments()

        self.target_screen = self.scored_screen
        self._simulate()
        self._scored = self._readings["stats"]
        self._scored_transmission = self._readings["transmission"]
        return self._get_obs()

    def _apply_action(self, native_values: np.ndarray):
        """Set every magnet, in `magnet_names` order and in native units."""
        values = np.asarray(native_values, dtype=np.float64).reshape(-1)
        if len(values) != len(self._controllable):
            raise ValueError(
                f"expected one value per settable magnet "
                f"({len(self._controllable)}), got {len(values)}"
            )
        self.set_magnets(dict(zip(self.magnet_names, values)))
        self._simulate()

    def _get_screen_statistics(self, screen_name: str) -> tuple[float, float, float, float, bool]:
        """(mu_x, mu_y, sigma_x, sigma_y, on_screen) in metres, off the screen centre.

        Only the screen in the beam has a reading: the others are out, and anything
        downstream of this one is in its shadow.
        """
        if screen_name != self._target_screen:
            raise ValueError(
                f"{screen_name} is not in the beam — only one screen may be, and it is "
                f"{self._target_screen}. Set `env.target_screen` or pass a `screen` "
                f"entry in the action to move it."
            )
        return self._readings["stats"]
