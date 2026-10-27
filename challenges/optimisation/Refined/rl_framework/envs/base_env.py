"""What a machine looks like to an agent, whether or not it is made of iron.

`SimEnv` tracks a beam through Cheetah; `RealEnv` sets power supplies and reads
cameras. Almost everything else about them is the same, and all of that is here:
the knob model, the diagnostics, the observation and `info` dictionaries, the
scoring, and `step`. A subclass supplies four things —

    _make_transport()        how a magnet is read and written
    _extent_of(screen)       how big a camera chip is, in metres
    _acquire()               take one shot, fill `self._readings`
    _hard_reset()            start an episode

— and gets the rest, which matters more than it sounds: it means a reading off
the real machine is scored by the same arithmetic as a simulated one, so the
numbers in `Refined/` are comparable across the two rather than merely similar.

Which accelerator this is comes from a `MachineSpec` and nothing else. See
`sim_env.py` for the action and observation formats, which are documented there
in full.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import cheetah
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_framework.config import RLConfig

# The shared machine description. A sibling package inside `Refined/`, so this
# is an ordinary import — it used to need the path adjusting because the two
# lived in different trees.
from machine_spec import lattice_io  # noqa: F401

# `Refined/`, which holds `machine_spec/`, `machines/`, `artifacts/` and the
# tuning modules. Data the specs point at lives above it and is found by
# `MachineSpec.load`, which walks up from the spec.
REFINED_DIR = Path(__file__).resolve().parents[2]

DTYPE = "float32"  # what the Cheetah lattice file is parsed into; see sim_env

# What one knob of each magnet type is called, and the suffix its knob name carries.
PROPERTY_SUFFIX = {"k1": "", "k2": "", "horizontal_angle": "_h", "vertical_angle": "_v",
                   "angle": "", "phase": ""}

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


def _is_diagnostic(element) -> bool:
    """Anything that reads the beam or scrapes it, whatever the lattice calls it.

    Cheetah has a `BPM` type and a `Screen` type, and lattices disagree about
    which to use: CLARA models BPMs, screens *and* aperture markers all as
    `Screen`, while CLEAR has real `BPM` elements. Both are stops.
    """
    return isinstance(element, (cheetah.Screen, cheetah.BPM))


class Transport(ABC):
    """How a magnet is read and written, whichever machine it belongs to.

    Cheetah elements and CLARA power supplies both answer to this: the
    environment works in element names and in the lattice file's units, and
    leaves each implementation to find its own magnets. A simulation needs more
    than this — it has to track a beam — and that is `TrackingTransport`.
    """

    name: str

    @abstractmethod
    def read(self, element: str, prop: str) -> float:
        """What a magnet is set to, in the lattice file's units."""

    @abstractmethod
    def write(self, element: str, prop: str, value: float,
              follow: Sequence[str] = ()) -> None:
        """Set a magnet, in the lattice file's units.

        `follow` is the pole-face angles of a chicane dipole and the like: a
        property that is not independently settable but is not constant either.
        """


class TrackingTransport(Transport):
    """A `Transport` that is a model, so a beam can be pushed through it."""

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


class BaseAcceleratorEnv(gym.Env, ABC):
    """A machine described by a `MachineSpec`, simulated or real.

    Nothing here knows which accelerator it is running, and nothing here knows
    whether the readings come from a tracker or a camera.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: RLConfig):
        self.config = config
        self.spec = config.machine_spec()
        spec = self.spec
        # Anything a subclass needs before its transport can be built — the
        # simulation loads its beam here, because the energy is a constructor
        # argument of the tracker.
        self._prepare()

        # The Cheetah segment is the lattice: element types, magnet strengths and
        # screen geometry, cut to the modelled stretch, with the chamber inserted
        # and the spec's `disabled` elements replaced by drifts of their length.
        # A real machine builds it too — not to track through, but because it is
        # the statement of which elements exist and in what order, which is what
        # decides the knobs and the diagnostics.
        self.segment = self._build_segment()

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
        self.transport = self._make_transport()

        self._shape = tuple(int(n) for n in config.screen_shape)
        # How many metres of chip each screen image covers, and the grid an
        # image is binned onto. Both are worked out the first time a diagnostic
        # is actually read, not here: a simulation can answer for all of them
        # instantly, but a real camera has to be asked, and asking all
        # twenty-two at construction means talking to twenty-two cameras to use
        # one of them.
        self._extents: dict[str, tuple[float, float, float, float]] = {}
        self._bin_edges: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        self._knobs = self._build_knobs(self.segment)
        self.magnet_names = list(self._controllable)
        self.magnet_limits = {
            name: (self._knobs[name].low, self._knobs[name].high)
            for name in self._controllable
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
        # A machine with no working BPMs is still a machine — CLARA is one at
        # the moment. The readings stay in the observation as empty arrays so
        # every consumer keeps working; only the *space* needs care, because
        # `spaces.MultiBinary(0)` is rejected outright ("n (counts) have to be
        # positive, got 0") and takes the environment down at construction.
        count = len(self.bpm_names)
        self.observation_space = spaces.Dict(
            {
                "bpm_x": spaces.Box(-np.inf, np.inf, (count,), np.float32),
                "bpm_y": spaces.Box(-np.inf, np.inf, (count,), np.float32),
                "bpm_valid": (spaces.MultiBinary(count) if count
                              else spaces.Box(0, 1, (0,), np.int8)),
                "screen_image": spaces.Box(0.0, image_high, self._shape, np.float32),
                "screen_index": spaces.Discrete(len(self.screen_names)),
                "target": spaces.Box(-np.inf, np.inf, (4,), np.float32),
            }
        )

        self._step_count = 0
        self._rng = np.random.default_rng()
        self.target = self._draw_target()
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
        # What the benchmark has seen of the request so far. A subclass finishes
        # its own constructor by taking the first reading and filling these in;
        # until it does, `_get_info` reports a machine that has delivered nothing,
        # which is the truthful answer before a shot has been taken.
        self._readings: dict[str, Any] = {}
        self._scored = (0.0, 0.0, 0.0, 0.0, False)
        self._scored_transmission = 0.0

    # ---------- what a subclass supplies ----------

    def _prepare(self) -> None:
        """Run after the spec is loaded and before anything is built from it."""

    def _build_segment(self):
        """The lattice, as Cheetah elements. The spec says which one."""
        return lattice_io.build_segment(self.spec)

    @abstractmethod
    def _make_transport(self) -> Transport:
        """How this machine's magnets are read and written."""

    @abstractmethod
    def _extent_of(self, name: str) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) in metres that a diagnostic's chip spans."""

    @abstractmethod
    def _acquire(self) -> None:
        """Take one shot and leave it in `self._readings`.

        Must set `bpm_x`, `bpm_y`, `bpm_valid`, `image`, `on_screen_fraction`,
        `stats` and `transmission`, which is what `_get_obs` and `_get_info` read.
        """

    @abstractmethod
    def _hard_reset(self) -> dict[str, Any]:
        """Begin an episode, and return the first observation."""

    # ---------- the machine description ----------

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

        **A knob value is the first element's strength**, so the spec's first
        sign must be positive. The environment used to re-base any spec that
        disagreed, by dividing through by `signs[0]` — which meant the same
        number drove opposite angles here and in a model reading the same file,
        and the only symptom was a tune clipped at the wrong end of its range.
        A disagreement is now an error.
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
            # Lead-positive, so a knob value is the first element's strength.
            # This used to be re-based here by dividing through by `signs[0]`,
            # which made the same number mean opposite angles in the environment
            # and in a model reading the same spec — and the only symptom was a
            # tune clipped at the wrong end of its range. Said once, in the spec,
            # and an error if a spec disagrees rather than a silent correction.
            if declared.signs and declared.signs[0] < 0:
                raise ValueError(
                    f"{declared.name} leads with sign {declared.signs[0]}. A knob "
                    f"value is the first element's strength, so the first sign "
                    f"must be positive; flip the signs and the limits together."
                )
            knobs[declared.name] = Knob(
                name=declared.name,
                element_name=declared.elements[0],
                prop=declared.prop,
                transport=self.transport,
                low=low, high=high,
                coupled=tuple(zip(declared.elements[1:], declared.signs[1:])),
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
        """(x_min, x_max, y_min, y_max) in metres that a screen image spans.

        Asked of the machine once per diagnostic and then remembered — a chip
        does not change size, but on a real camera finding out costs a round
        trip to the control system.
        """
        name = screen or self._target_screen
        if name not in self._extents:
            self._extents[name] = tuple(float(v) for v in self._extent_of(name))
        return self._extents[name]

    def _bins(self, screen: str) -> tuple[np.ndarray, np.ndarray]:
        """(row edges, column edges) an image of this screen is binned onto."""
        if screen not in self._bin_edges:
            extent = self.screen_extent(screen)
            self._bin_edges[screen] = (
                np.linspace(*extent[2:], self._shape[0] + 1),  # rows: y
                np.linspace(*extent[:2], self._shape[1] + 1),  # columns: x
            )
        return self._bin_edges[screen]

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
        self._acquire()
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

    # ---------- scoring ----------

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
