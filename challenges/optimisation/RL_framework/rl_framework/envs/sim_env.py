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

import importlib.util
import json
import sys
from abc import ABC, abstractmethod
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import cheetah
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from rl_framework.config import RLConfig
from rl_framework.envs.base_env import BaseAcceleratorEnv

# The shared machine description, which `Refined/` also reads. Added to the path
# rather than packaged, matching how the rest of this challenge is laid out.
_SHARED = Path(__file__).resolve().parents[3]
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))
from machine_spec import lattice_io  # noqa: E402

# .../challenges/optimisation, which holds `machine_spec/`, `machines/` and `data/`.
CHALLENGE_DIR = Path(__file__).resolve().parents[3]


def _is_diagnostic(element) -> bool:
    """Anything that reads the beam or scrapes it, whatever the lattice calls it.

    Cheetah has a `BPM` type and a `Screen` type, and lattices disagree about
    which to use: CLARA models BPMs, screens *and* aperture markers all as
    `Screen`, while CLEAR has real `BPM` elements. Both are stops.
    """
    return isinstance(element, (cheetah.Screen, cheetah.BPM))

DTYPE = torch.float32  # what the Cheetah lattice file is parsed into

# What one knob of each magnet type is called, and the suffix its knob name carries.
PROPERTY_SUFFIX = {"k1": "", "k2": "", "horizontal_angle": "_h", "vertical_angle": "_v",
                   "angle": "", "phase": ""}
# Magnet operating ranges, coupled-magnet groups, pole faces that follow a body
# angle, and which elements are switched off all used to be constants here, every
# one of them CLARA. They are properties of a machine, so they live in its
# `MachineSpec` — see `machine_spec/spec.py` and `machines/clara_febe.json`.

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

# What the beam is asked for, as `score.sample_target` asks for it: a spot size
# log-uniform between these, and a position uniform within this of the screen centre.
REQUESTED_SPOT = (80e-6, 400e-6)  # m
REQUESTED_OFFSET = 2e-3  # m
# What a case books when the beam is lost, and the most any one reading can cost,
# in units of the spot that was asked for. `score.FAILURE_SCORE`.
FAILURE_SCORE = 10.0
# What beam that never arrives costs, as `score.TRANSMISSION_WEIGHT` charges it:
# delivering half the bunch is worth a spot size.
TRANSMISSION_WEIGHT = 2.0
TARGET_LABELS = ("mu_x", "mu_y", "sigma_x", "sigma_y")
# What shots cost, as `score.Report.calc_score` prices them: EVAL_WEIGHT for every
# doubling of EVAL_BUDGET, because beam time costs orders of magnitude rather than
# increments — 40 shots to 80 is priced like 2000 to 4000.
EVAL_WEIGHT = 0.5
EVAL_BUDGET = 50.0
# The worst a reading can be for the purpose of the reward. The benchmark stops
# counting at ten spot sizes, which is the whole reachable range for an untuned
# machine — an 85 um request and a 3 mm beam is thirty-five spot sizes out — so the
# reward needs to see further than that or it is flat where the learning happens.
MAX_MISMATCH = 100.0


def evaluation_cost(shots: int) -> float:
    """What `shots` shots have cost so far, in requested spot sizes."""
    return EVAL_WEIGHT * float(np.log2(1 + shots / EVAL_BUDGET))


class Transport(ABC):
    """One of the two models the environment can track its beam through.

    Both carry the same elements under the same names, so the environment works in
    names and in Cheetah's units and leaves each model to find and write its own
    elements. `config.backend` picks between them.
    """

    name: str

    @abstractmethod
    def read(self, element: str, prop: str) -> float:
        """What a magnet is set to, in the lattice file's units."""

    @abstractmethod
    def write(self, element: str, prop: str, value: float) -> None:
        """Set a magnet, in the lattice file's units."""

    @abstractmethod
    def read_shift(self, element: str) -> tuple[float, float]:
        """Where an element sits against the design axis."""

    @abstractmethod
    def shift(self, element: str, dx: float, dy: float) -> None:
        """Move an element off the design axis."""

    @abstractmethod
    def run(
        self, coordinates: np.ndarray, stops: list[str]
    ) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        """Track from the start of the line, yielding (stop, x, y, alive) at each of
        `stops` in turn. `stops` runs from the start of the line to wherever the beam
        is being read, and x, y and alive are in the order the particles were given."""


class CheetahTransport(Transport):
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


@dataclass(frozen=True)
class Diagnostic:
    """A screen or BPM: where it sits in the line and how big its chip is."""

    name: str
    index: int  # into `line.element_names`
    extent: tuple[float, float, float, float]  # x_min, x_max, y_min, y_max, metres
    misalignment: tuple[float, float]


@dataclass
class Knob:
    """One number an agent can set: a quadrupole k1, a sextupole k2, or one plane of a
    corrector. Always in the lattice file's units, so the numbers match
    `magnet_ranges.json` and `score.py` whichever model is doing the tracking."""

    name: str
    element_name: str
    prop: str
    transport: Transport
    low: float
    high: float
    # Elements driven off the same setting, each with its sign — the four dipoles
    # of a chicane are one knob on the machine, not four.
    coupled: tuple[tuple[str, float], ...] = ()
    # Properties that must be written with the main one, per element: a chicane's
    # pole faces live here. From the spec, because it is a fact about the magnet.
    follow: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.transport.read(self.element_name, self.prop)

    def set(self, value: float, clip: bool = True) -> float:
        """Set it, clipped to the operating range. Returns what the magnet ended up
        at, which is not quite what was asked for — Cheetah's lattice is float32."""
        value = float(value)
        if clip:
            value = float(np.clip(value, self.low, self.high))
        self.transport.write(self.element_name, self.prop, value,
                             self.follow.get(self.element_name, ()))
        for element, sign in self.coupled:
            self.transport.write(element, self.prop, sign * value,
                                 self.follow.get(element, ()))
        return self.value


class SimEnv(BaseAcceleratorEnv):
    """A simulated machine, described by a `MachineSpec`.

    Nothing here knows which accelerator it is running. The lattice, the beam,
    the vacuum chamber, which elements are switched off, which diagnostics are
    BPMs and which are screens, and what every magnet may be set to all come
    from the spec. `RLConfig.clara()` loads the one the benchmark is scored on.
    """

    def __init__(self, config: RLConfig):
        super().__init__(config)
        self.spec = config.machine_spec()
        spec = self.spec

        coordinates, energy_ev, self.beam_metadata = lattice_io.load_beam(
            spec, config.num_particles
        )
        # The beam as it was measured, and the one this episode was dealt.
        self._reference_coordinates = np.ascontiguousarray(coordinates)
        self._coordinates = self._reference_coordinates
        self.energy_ev = energy_ev

        # The Cheetah segment is the lattice: element types, magnet strengths and
        # screen geometry, cut to the modelled stretch, with the chamber inserted
        # and the spec's `disabled` elements replaced by drifts of their length.
        self.segment = lattice_io.build_segment(spec)

        self._diagnostics = {
            element.name: Diagnostic(
                name=element.name,
                index=index,
                extent=tuple(float(v) for v in getattr(element, "extent", (0,) * 4)),
                misalignment=tuple(
                    float(v) for v in getattr(element, "misalignment", (0.0, 0.0))
                ),
            )
            for index, element in enumerate(self.segment.elements)
            if _is_diagnostic(element)
        }
        self.bpm_names, self.screen_names = self._classify(spec)
        if not self.screen_names:
            raise ValueError(
                f"no screens between {spec.start} and {spec.end}. The spec may "
                f"name them in `screens`; otherwise they are found by element "
                f"type and by the tags in `config.diagnostic_tags`."
            )
        # Where tracking pauses to read a diagnostic and to scrape the beam pipe.
        # Apertures are stops too: they read nothing but they can lose beam.
        read = set(self.bpm_names) | set(self.screen_names)
        aperture_tag = config.diagnostic_tags.get("aperture", "")
        self._stops = sorted(
            (
                diagnostic
                for name, diagnostic in self._diagnostics.items()
                if name in read or (aperture_tag and aperture_tag in name)
            ),
            key=lambda diagnostic: diagnostic.index,
        )
        self.transport = CheetahTransport(
            self.segment, energy_ev, [d.name for d in self._stops]
        )

        self._shape = tuple(int(n) for n in config.screen_shape)
        self._bin_edges = {
            name: (
                np.linspace(*diagnostic.extent[2:], self._shape[0] + 1),  # rows: y
                np.linspace(*diagnostic.extent[:2], self._shape[1] + 1),  # columns: x
            )
            for name, diagnostic in self._diagnostics.items()
        }

        self._knobs = self._build_knobs(self.segment)
        self.magnet_names = list(self._controllable)
        self.magnet_limits = {
            name: (self._knobs[name].low, self._knobs[name].high)
            for name in self._controllable
        }
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

        self.action_space = spaces.Dict(
            {
                "magnets": spaces.Dict(
                    {
                        name: spaces.Box(
                            low=self._knobs[name].low,
                            high=self._knobs[name].high,
                            shape=(),
                            dtype=np.float32,
                        )
                        for name in self._controllable
                    }
                ),
                "screen": spaces.Discrete(len(self.screen_names)),
            }
        )
        image_high = 1.0 if config.normalise_image else np.inf
        self.observation_space = spaces.Dict(
            {
                "bpm_x": spaces.Box(-np.inf, np.inf, (len(self.bpm_names),), np.float32),
                "bpm_y": spaces.Box(-np.inf, np.inf, (len(self.bpm_names),), np.float32),
                "bpm_valid": spaces.MultiBinary(len(self.bpm_names)),
                "screen_image": spaces.Box(0.0, image_high, self._shape, np.float32),
                "screen_index": spaces.Discrete(len(self.screen_names)),
                "target": spaces.Box(-np.inf, np.inf, (4,), np.float32),
            }
        )

        # Magnets ripple; a cavity phase is not a value a percentage means
        # anything against, so it does not.
        self._rippled = [
            name for name, knob in self._knobs.items() if knob.prop != "phase"
        ]

        self._rng = np.random.default_rng()
        self.target = self._draw_target()
        # The spec's readout unless the configuration names another.
        # The spec's readout unless the configuration names another. Held on
        # the environment, because the config's may be None.
        # The chamber and the reading threshold describe the machine, so the
        # spec owns them; the config may override, which is how a study varies
        # them. Read from the config alone, a 20 mm CLEAR pipe was scraped at the
        # 50 mm CLARA default while the twin used 20 — the same physical fact
        # with two values, visible only as unexplained transmission error.
        self.beam_pipe_radius = (config.beam_pipe_radius
                                 if config.beam_pipe_radius is not None
                                 else spec.pipe_radius)
        self.min_screen_fraction = (config.min_screen_fraction
                                    if config.min_screen_fraction is not None
                                    else spec.min_fraction)
        self.scored_screen = config.target_screen or spec.readout
        self.target_screen = self.scored_screen
        self._readings: dict[str, Any] = {}
        self._simulate()
        self._scored = self._readings["stats"]
        self._scored_transmission = self._readings["transmission"]

    # ---------- the model doing the tracking ----------

    def _classify(self, spec) -> tuple[list[str], list[str]]:
        """(BPM names, screen names), in lattice order.

        The spec's own lists come first, because only the machine knows: CLARA
        models a BPM, a screen and an aperture marker with the same Cheetah type
        and tells them apart by name, which is a convention and not a fact about
        Cheetah. Where the spec is silent, the element type decides, and where
        that cannot (every diagnostic is a `Screen`), the name tags do.
        """
        found = list(self._diagnostics)
        named_bpms, named_screens = set(spec.bpms), set(spec.screens)
        if named_bpms or named_screens:
            return ([n for n in found if n in named_bpms],
                    [n for n in found if n in named_screens])

        tags = self.config.diagnostic_tags
        elements = {e.name: e for e in self.segment.elements}
        typed = [n for n in found if isinstance(elements[n], cheetah.BPM)]
        if typed:
            return typed, [n for n in found if n not in set(typed)
                           and tags.get("aperture", "\0") not in n]
        return ([n for n in found if tags.get("bpm", "\0") in n],
                [n for n in found if tags.get("screen", "\0") in n])

    def _build_knobs(self, segment) -> dict[str, Knob]:
        """Every knob the spec declares that this stretch of lattice actually has.

        A knob is not an element. It may drive several elements at once with
        different signs — the four dipoles of a chicane are one number on the
        machine — and setting it may require writing more than one property of
        each, because a chicane's pole faces turn with its body angle. All of
        that is in the spec, so there is no special case for any machine here.

        **Signs are normalised so the first element takes the knob value
        unchanged.** A spec records the absolute sense of each magnet, and which
        end of a chicane counts as positive is a convention the two files need
        not share: CLARA's spec says `[-1, 1, 1, -1]` where this environment has
        always driven the knob as `[1, -1, -1, 1]`. Normalising keeps every
        existing set point, response matrix and published result meaning what it
        did, and leaves the two descriptions differing only by a stated
        convention rather than silently.
        """
        present = {element.name for element in segment.elements}
        wanted = self.config.controllable_magnets
        overrides = self.config.magnet_limits or {}

        knobs: dict[str, Knob] = {}
        for declared in self.spec.knobs:
            if not all(name in present for name in declared.elements):
                continue
            low, high = declared.low, declared.high
            if declared.name in overrides:
                low, high = (float(v) for v in overrides[declared.name])
            if low is None or high is None:
                raise ValueError(
                    f"{declared.name} has no operating range. Ranges come from "
                    f"the spec now, not from a calibration table in here; "
                    f"`describe.py` fills them in, and `MachineSpec.validate` "
                    f"lists any it could not."
                )
            lead = declared.signs[0] if declared.signs else 1.0
            sense = 1.0 / lead if lead else 1.0
            # The range turns with the sense. A spec records limits in its own
            # sign convention, so normalising the signs without normalising these
            # silently widens the knob one way and narrows it the other: CLARA's
            # chicane would read [-0.11561, 0.1317] where the machine has always
            # had [-0.1317, 0.11561], and a request for 0.12 rad would be clipped
            # to a different angle and track a different beam.
            low, high = sorted((low * sense, high * sense))
            knobs[declared.name] = Knob(
                name=declared.name,
                element_name=declared.elements[0],
                prop=declared.prop,
                transport=self.transport,
                low=low, high=high,
                coupled=tuple(
                    (name, sign * sense)
                    for name, sign in zip(declared.elements[1:], declared.signs[1:])
                ),
                follow={k: tuple(v) for k, v in declared.follow.items()},
            )

        missing = set(wanted or ()) - set(knobs)
        if missing:
            raise KeyError(
                f"no such magnets between {self.spec.start} and {self.spec.end}: "
                f"{sorted(missing)}"
            )
        self._controllable = list(wanted) if wanted is not None else list(knobs)
        return knobs

    def magnet_settings(self) -> dict[str, float]:
        """What every controllable magnet is set to, in native units."""
        return {name: self._knobs[name].value for name in self._controllable}

    def set_magnets(self, settings: Mapping[str, Any]) -> dict[str, float]:
        """Set some of the magnets, leaving the rest alone. Returns what was applied.

        Keys are knob names (`cla_s04_mag_quad_01`, `cla_s04_mag_hvcor_01_h`) with a
        value each, or element names with a `{property: value}` mapping.
        """
        applied = {}
        for name, value in settings.items():
            if isinstance(value, Mapping):
                for prop, number in value.items():
                    if prop not in PROPERTY_SUFFIX:
                        raise KeyError(f"{name}: not a magnet property: {prop}")
                    knob = self._knob(name + PROPERTY_SUFFIX[prop])
                    applied[knob.name] = knob.set(number)
            else:
                knob = self._knob(name)
                applied[knob.name] = knob.set(value)
        return applied

    def _knob(self, name: str) -> Knob:
        if name in self._knobs and name not in self._controllable:
            raise KeyError(
                f"{name} is a magnet of this line but not one the agent may set. "
                f"`config.controllable_magnets` decides that."
            )
        try:
            return self._knobs[name]
        except KeyError:
            raise KeyError(
                f"{name} is not a controllable magnet. Correctors take a _h or _v "
                f"suffix; `env.magnet_names` lists all {len(self._controllable)}."
            ) from None

    def _denormalise_action(self, action: np.ndarray) -> np.ndarray:
        """Map a normalised [-1, 1] vector to native units, in `magnet_names` order."""
        low = np.array([self._knobs[n].low for n in self._controllable], float)
        high = np.array([self._knobs[n].high for n in self._controllable], float)
        return (low + high) / 2 + np.asarray(action, dtype=np.float64) * (high - low) / 2

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

    def _draw_target(self) -> np.ndarray:
        """What the beam is asked for at the observed screen this episode:
        (mu_x, mu_y, sigma_x, sigma_y) in metres, off the screen centre.

        Drawn as `score.sample_target` draws it — position uniform, size log-uniform,
        so a tight request is no rarer than a loose one — unless `config.target`
        fixes it. Nothing guarantees it is reachable; the task is to get close.
        """
        if self.config.target is not None:
            return np.asarray(self.config.target, dtype=float)
        low, high = self.config.requested_spot
        offset = self.config.requested_offset
        return np.array([
            self._rng.uniform(-offset, offset),
            self._rng.uniform(-offset, offset),
            float(np.exp(self._rng.uniform(np.log(low), np.log(high)))),
            float(np.exp(self._rng.uniform(np.log(low), np.log(high)))),
        ])

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

    # ---------- the observed screen ----------

    @property
    def target_screen(self) -> str:
        """The one screen in the beam. Everything downstream of it is in its shadow."""
        return self._target_screen

    @target_screen.setter
    def target_screen(self, screen: str | int) -> None:
        if isinstance(screen, (int, np.integer)):
            screen = self.screen_names[int(screen)]
        if screen not in self.screen_names:
            raise KeyError(
                f"{screen} is not a screen of this line. `env.screen_names` lists all "
                f"{len(self.screen_names)}."
            )
        self._target_screen = screen

    def screen_extent(self, screen: str | None = None) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) in metres that a screen image spans."""
        return self._diagnostics[screen or self._target_screen].extent

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

        rows, columns = self._bin_edges[screen.name]
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

    # ---------- gym API ----------

    def _get_obs(self) -> dict[str, Any]:
        return {
            "bpm_x": self._readings["bpm_x"].astype(np.float32),
            "bpm_y": self._readings["bpm_y"].astype(np.float32),
            "bpm_valid": self._readings["bpm_valid"],
            "screen_image": self._readings["image"].astype(np.float32),
            "screen_index": self.screen_names.index(self._target_screen),
            "target": self.target.astype(np.float32),
        }

    def _get_info(self) -> dict[str, Any]:
        mu_x, mu_y, sigma_x, sigma_y, on_screen = self._readings["stats"]
        achieved = np.array([mu_x, mu_y, sigma_x, sigma_y])
        return {
            "screen": self._target_screen,
            "screen_extent": self.screen_extent(),
            "target": self.target,
            "error": achieved - self.target if on_screen else None,
            "relative_error": self._relative_error(self._readings["stats"]),
            # The benchmark's view: the last reading of the screen the request is
            # for, however many screens have been looked at since. `benchmark_score`
            # is `score.Report.calc_score` for this episode as one case, so it is
            # comparable to a scored tuner and lower is better.
            "scored_screen": self.scored_screen,
            "evaluations": self._step_count,
            "scored_transmission": self._scored_transmission,
            "scored_mismatch": self._mismatch(self._scored,
                                              self._scored_transmission),
            "scored_relative_error": self._relative_error(self._scored),
            "benchmark_score": self._mismatch(self._scored,
                                              self._scored_transmission)
            + evaluation_cost(self._step_count),
            "mu_x": mu_x,
            "mu_y": mu_y,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "on_screen": on_screen,
            "on_screen_fraction": self._readings["on_screen_fraction"],
            "transmission": self._readings["transmission"],
            "magnets": self.magnet_settings(),
        }

    def step(self, action):
        settings, screen = self._parse_action(action)
        if settings:
            self.set_magnets(settings)
        if screen is not None:
            self.target_screen = screen
        self._simulate()
        if self._target_screen == self.scored_screen:
            # a fresh reading of the request, and of how much of it arrived
            self._scored = self._readings["stats"]
            self._scored_transmission = self._readings["transmission"]

        # Dense: the mismatch every step, and the marginal cost of the shot it took.
        # The costs telescope, so over an episode they come to `evaluation_cost` of
        # the shots taken, exactly what the benchmark charges.
        self._step_count += 1
        reward = self._compute_reward(*self._scored) - (
            evaluation_cost(self._step_count) - evaluation_cost(self._step_count - 1)
        )
        truncated = self._step_count >= self.config.max_episode_steps
        return self._get_obs(), reward, False, truncated, self._get_info()

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)  # the base class reset returns no info
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        return self._hard_reset(), self._get_info()

    def _parse_action(self, action) -> tuple[Mapping[str, Any], str | int | None]:
        """(magnet settings, screen) out of an action.

        A mapping with a `magnets` or `screen` entry is read as those two; any other
        mapping is read as magnet settings alone. A plain sequence is read as a
        normalised [-1, 1] setting for every magnet, in `magnet_names` order.
        """
        if action is None:
            return {}, None
        if isinstance(action, Mapping):
            if "magnets" in action or "screen" in action or "target_screen" in action:
                unknown = set(action) - {"magnets", "screen", "target_screen"}
                if unknown:
                    raise KeyError(
                        f"an action with a `magnets` or `screen` entry takes no other "
                        f"entries: {sorted(unknown)}"
                    )
                screen = action.get("screen", action.get("target_screen"))
                return action.get("magnets") or {}, screen
            return action, None
        values = self._denormalise_action(np.asarray(action, dtype=np.float64).reshape(-1))
        if len(values) != len(self._controllable):
            raise ValueError(
                f"an action vector has one entry per settable magnet "
                f"({len(self._controllable)}), got {len(values)}"
            )
        return dict(zip(self.magnet_names, values)), None

    def _mismatch(self, stats, transmission: float,
                  cap: float = FAILURE_SCORE) -> float:
        """How far a reading is from the request, in requested spot sizes.

        `score.per_case_scores` exactly: each component's error against the spot size
        asked for, position and size weighted equally, plus what the beam that never
        arrived costs, clipped at `cap` — which is also what a lost beam books, so
        losing the beam is never cheaper than delivering a bad one. The benchmark's
        cap is `FAILURE_SCORE`; the reward looks further, out to `MAX_MISMATCH`.
        """
        mu_x, mu_y, sigma_x, sigma_y, on_screen = stats
        if not on_screen:
            return cap
        error = np.abs(
            np.array([mu_x, mu_y, sigma_x, sigma_y]) - self.target
        ) / float(self.target[2:].mean())
        return float(min(
            (error[0] + error[1]) / 2 + (error[2] + error[3]) / 2
            + TRANSMISSION_WEIGHT * (1.0 - transmission),
            cap,
        ))

    def _relative_error(self, stats) -> float:
        """`score.CaseResult.relative_error`: the mean component error in requested
        spot sizes, which is what the benchmark's hit rates are thresholded on."""
        mu_x, mu_y, sigma_x, sigma_y, on_screen = stats
        if not on_screen:
            return FAILURE_SCORE
        achieved = np.array([mu_x, mu_y, sigma_x, sigma_y])
        return float(np.abs(achieved - self.target).mean() / self.target[2:].mean())

    def _compute_reward(self, mu_x, mu_y, sigma_x, sigma_y, on_screen) -> float:
        """How far the beam is from what was asked for, negated.

        `score.per_case_scores` exactly: each component's error in units of the spot
        size that was requested, position and size weighted equally, clipped at
        `FAILURE_SCORE` — which is also what a lost beam books, so losing the beam is
        never cheaper than delivering a bad one. Dimensionless, and bounded, so a
        tight request is no harder to score well on than a loose one.

        `config.reward_shape` decides how the mismatch becomes a reward. "log" is
        -log1p(mismatch), which falls off slowly and so keeps a gradient however far
        out the beam is, while still being -mismatch to first order once the beam is
        close, where the benchmark's own scale is the one that matters. "linear" is
        the benchmark's -min(mismatch, FAILURE_SCORE) itself, flat beyond ten spot
        sizes. Either way a lost beam is the worst a step can do.

        The base class instead rewards the smallest centred spot it can get. Override
        this to ask for something else.
        """
        stats = (mu_x, mu_y, sigma_x, sigma_y, on_screen)
        if self.config.reward_shape == "linear":
            return -self._mismatch(stats, self._scored_transmission)
        if self.config.reward_shape != "log":
            raise ValueError(
                f"{self.config.reward_shape!r} is not a reward shape: "
                f"'log' or 'linear'."
            )
        return -float(np.log1p(self._mismatch(
            stats, self._scored_transmission, cap=MAX_MISMATCH)))

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
