"""What a tuning method is driving.

    machine = SimEnvMachine(spec, seed=1000)
    reading = machine.set({"cla_feh_mag_quad_01": 1.7})
    reading.sizes, reading.bpm, machine.score

One implementation, because `SimEnv` now takes a `MachineSpec` and runs whatever
machine the spec describes. There used to be a second, `LineMachine`, which wrapped
a `Refined.lattice.Line` so the pipeline could be tested without CLARA. It was a
poor test and is gone: it shared its code with the planning twin, so a model-based
method faced no model error at all, and it had no shot noise — identical settings
gave identical readings where `SimEnv`'s differ by about 5 %. Lattice-agnosticism
is now tested the honest way, by pointing `SimEnv` at a different machine.

**The knob mapping is the awkward part and it is done up front.** A `MachineSpec`
knob may be a group — two sextupoles on one supply, four chicane dipoles on one —
and `SimEnv` has its own ideas about which of those exist. The mapping is resolved
in the constructor and anything that cannot be mapped is an error there, not a
`KeyError` eight shots into an episode.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The thresholds come from the spec, not from literals here. What the benchmark
# books for a lost beam, and what a screen move costs, have to be the same number
# in the environment that scores a shot and in the tuner that reads the score —
# and they were two literals apiece until they were not.
from machine import FAILURE_SCORE, SCREEN_SWITCH_SECONDS, MachineSpec  # noqa: E402


@dataclass
class Reading:
    """One shot, as the machine reports it."""

    mu_x: float
    mu_y: float
    sigma_x: float
    sigma_y: float
    on_screen: bool
    transmission: float
    # Centroids at every diagnostic, (n_diagnostics, 2), and which of them read.
    # Rows follow `Machine.diagnostics`.
    bpm: np.ndarray
    bpm_valid: np.ndarray
    settings: dict[str, float] = field(default_factory=dict)
    # Which screen was in the beam. Only one may be, and it intercepts: while it
    # is an upstream screen, `mu_x`, `sigma_x` and the rest describe *that*
    # screen and not the request, and nothing downstream of it reads at all.
    screen: str = ""
    # What the episode books if it stops here: the benchmark's mismatch on this
    # reading plus what the shots have cost. Lower is better.
    score: float = FAILURE_SCORE

    @property
    def spot(self) -> np.ndarray:
        return np.array([self.mu_x, self.mu_y, self.sigma_x, self.sigma_y])


# The benchmark's own arithmetic, imported rather than restated. Every one of
# these was once a local copy: `evaluation_cost` priced a shot five times too
# cheaply (`weight=0.1, budget=10.0` against `score.EVAL_WEIGHT = 0.5` and
# `EVAL_BUDGET = 50.0`), so the BO stopping rule read "worth another shot" long
# after it was not, and `mismatch` was the scoring formula written out a second
# time beside the import that existed to avoid exactly that.
from rl_framework.envs.base_env import (  # noqa: E402,F401
    evaluation_cost,   # re-exported: `bayes.py` imports it from here
    mismatch,
    relative_error,
)


class Machine:
    """The interface a tuning method sees. Both implementations keep the figures
    of merit — shots, screen switches, score — so a method cannot forget to."""

    target: np.ndarray  # (mu_x, mu_y, sigma_x, sigma_y), metres
    diagnostics: list[str]
    knob_names: list[str]
    limits: dict[str, tuple[float, float]]

    def __init__(self) -> None:
        self.shots = 0
        self.screen_switches = 0
        self.history: list[Reading] = []

    def set(self, settings: dict[str, float]) -> Reading:
        raise NotImplementedError

    def look_at(self, screen: str) -> Reading:
        """Move the beam-intercepting screen and read it.

        Costs a shot like any other, and about half a minute of machine time on
        top — only one screen may be in the beam, and swapping which is
        mechanical. That is why `screen_switches` is counted separately from
        `shots`: they are not the same expense, and a method that treats them
        as interchangeable will plan a very slow tune.
        """
        raise NotImplementedError

    def clip(self, name: str, value: float) -> float:
        low, high = self.limits.get(name, (-np.inf, np.inf))
        return float(np.clip(value, low, high))

    @property
    def last(self) -> Reading:
        return self.history[-1]

    @property
    def readout(self) -> str:
        """The screen the request is about, and where the score is read."""
        return self.spec.readout

    @property
    def looking_at_readout(self) -> bool:
        return self.last.screen == self.readout

    @property
    def score(self) -> float:
        """What the episode books: the benchmark on the **last** reading.

        Not the best. `SimEnv` scores the most recent reading of the scored
        screen and adds what the shots cost, so a method that finds a good tune
        and then wanders off books the wandering. Every method here therefore has
        to end on its best setting, which costs a shot and is worth it; compare
        `best` to see what it would have booked if it had stopped at the right
        moment.
        """
        return self.history[-1].score if self.history else FAILURE_SCORE

    @property
    def best(self) -> float:
        """The best reading taken, shot cost included. A diagnostic, not the
        score: the gap between this and `score` is a method failing to go back."""
        return min((r.score for r in self.history), default=FAILURE_SCORE)

    def restore_best(self) -> Reading | None:
        """Re-set whatever scored best. One shot, and usually worth it."""
        if not self.history:
            return None
        target = min(self.history, key=lambda r: r.score)
        if target is self.history[-1]:
            return None
        return self.set(dict(target.settings))

    def mismatch_of(self, reading: Reading) -> float:
        return mismatch(reading.spot, self.target, reading.transmission,
                        reading.on_screen)

    @property
    def relative_error(self) -> float:
        """`score.CaseResult.relative_error`, what hit rates are thresholded on.

        The *best* reading of the episode, not the last — which is why this is
        computed here rather than read from `info["scored_relative_error"]`.
        The environment reports the reading the benchmark books; this reports
        the best the tuner ever saw, and the gap between them is a method
        failing to go back to it.
        """
        best = min(self.history, key=self.mismatch_of, default=None)
        if best is None:
            return FAILURE_SCORE
        return relative_error(best.spot, self.target, best.on_screen)

    @property
    def screen_switch_seconds(self) -> float:
        """What one screen move costs. The machine's own number if it has one."""
        env = getattr(self, "env", None)
        return SCREEN_SWITCH_SECONDS if env is None else env.screen_switch_seconds

    @property
    def machine_seconds(self) -> float:
        """What this episode would have cost on the real machine."""
        return self.shots + self.screen_switches * self.screen_switch_seconds


# ------------------------------------------------- SimEnv and the real thing


def _config_for(spec: MachineSpec, config):
    """A config that describes *this* spec, or an error saying why it cannot.

    Defaulting to CLARA here meant `optimise.py --spec machines/clear.json`
    tuned CLARA and said nothing: the caller had to pass the spec a second
    time, through the config, and the two could disagree without complaint.
    """
    from rl_framework.config import RLConfig

    if config is None:
        return RLConfig(spec=spec)
    if config.spec is None:
        return replace(config, spec=spec)
    if config.spec is not spec and config.machine_spec().name != spec.name:
        raise ValueError(
            f"this machine was given the spec {spec.name!r} but a config for "
            f"{config.machine_spec().name!r}; they must be the same machine"
        )
    return config


class EnvMachine(Machine):
    """A `Machine` driven through the `SimEnv`/`RealEnv` interface.

    All of the driving is here, because the two environments present the same
    actions, observations and `info` — which is the whole point of them sharing
    `BaseAcceleratorEnv`. A subclass only has to build the environment and take
    the first reading.
    """

    def __init__(self, spec: MachineSpec) -> None:
        super().__init__()
        self.spec = spec

    def _adopt(self, env, observation, info) -> None:
        """Take up an environment that has been reset and is ready to read."""
        self.env = env
        self.target = np.asarray(observation["target"], dtype=float)
        self.diagnostics = list(self.env.bpm_names)
        self.knob_names = _knob_names(self.spec, self.env)
        self.limits = {n: tuple(self.env.magnet_limits[n]) for n in self.knob_names}
        self._settings = {
            n: float(self.env._knobs[n].value) for n in self.knob_names
        }
        # `reset` returns a full info dictionary, so the starting reading is a
        # real one and costs no shot.
        self._record(observation, info)

    # ---------- driving ----------

    def set(self, settings: dict[str, float]) -> Reading:
        magnets: dict[str, float] = {}
        for name, value in settings.items():
            if name not in self.limits:
                raise KeyError(
                    f"{name} is not a knob this machine can set; it has "
                    f"{len(self.knob_names)}"
                )
            magnets[name] = self._settings[name] = self.clip(name, float(value))
        observation, _, _, _, info = self.env.step({"magnets": magnets})
        self.shots += 1
        return self._record(observation, info)

    def look_at(self, screen: str) -> Reading:
        """Move the screen into the beam and read it. One shot, one switch."""
        if screen not in self.env.screen_names:
            raise KeyError(
                f"{screen} is not a screen of this machine; it has "
                f"{len(self.env.screen_names)}"
            )
        if screen == self.env.target_screen:
            # Already there, so this is an ordinary shot and costs no switch.
            return self.set({})
        observation, _, _, _, info = self.env.step({"screen": screen})
        self.shots += 1
        self.screen_switches += 1
        return self._record(observation, info)

    def _record(self, observation, info) -> Reading:
        reading = Reading(
            mu_x=float(info.get("mu_x", np.nan)),
            mu_y=float(info.get("mu_y", np.nan)),
            sigma_x=float(info.get("sigma_x", np.nan)),
            sigma_y=float(info.get("sigma_y", np.nan)),
            on_screen=bool(info.get("on_screen", False)),
            transmission=float(info.get("transmission", 0.0)),
            bpm=np.stack([
                np.asarray(observation["bpm_x"], dtype=float),
                np.asarray(observation["bpm_y"], dtype=float),
            ], axis=-1),
            bpm_valid=np.asarray(observation["bpm_valid"], dtype=float),
            settings=dict(self._settings),
            # The environment's own number, not a reimplementation of it: this is
            # what the benchmark books and what every other method in this repo
            # is quoted against. While an upstream screen is in the beam this
            # holds at the last reading of the scored one, which is exactly what
            # looking away costs.
            score=float(info["benchmark_score"]),
            screen=str(info.get("screen", self.env.target_screen)),
        )
        self.history.append(reading)
        return reading


class SimEnvMachine(EnvMachine):
    """CLARA as `rl_framework` deals it, scored by the benchmark."""

    def __init__(self, spec: MachineSpec, seed: int = 0, config=None):
        super().__init__(spec)
        from rl_framework.envs.sim_env import SimEnv

        env = SimEnv(_config_for(spec, config))
        # `reset` returns a full info dictionary, so the starting reading is a
        # real one and costs no shot.
        self._adopt(env, *env.reset(seed=seed))


class RealMachine(EnvMachine):
    """The actual accelerator, through `RealEnv`.

        machine = RealMachine(spec, target=(0.0, 0.0, 266e-6, 150e-6))

    Identical to `SimEnvMachine` from a tuning method's point of view, which is
    the point: `steer` and the rest cannot tell them apart, so what they do on
    the machine is what they were measured doing in simulation.

    There is no seed. `RealEnv.reset()` draws nothing and moves nothing — it
    reads the machine as found — so an episode is not reproducible and should
    not pretend to be. `target` is required, because a real machine is being
    asked for something specific.
    """

    def __init__(self, spec: MachineSpec, target=None, config=None,
                 hardware=None):
        """`hardware` is the control-system module, or its name.

        Defaults to `rl_framework.envs.hardware`, whose placeholders raise —
        so forgetting to name a driver stops the episode on the first shot
        instead of doing something. `"hardware_CLARA"` is the CATAP one.
        """
        super().__init__(spec)
        from rl_framework.envs.real_env import RealEnv

        if isinstance(hardware, str):
            from importlib import import_module

            hardware = import_module(f"rl_framework.envs.{hardware}")
        config = _config_for(spec, config)
        if target is not None:
            config = replace(config, target=tuple(float(v) for v in target))
        if config.target is None:
            raise ValueError(
                "RealMachine needs a target: (mu_x, mu_y, sigma_x, sigma_y) in "
                "metres off the readout's centre. Pass `target=` or set it on "
                "the config."
            )
        env = (RealEnv(config) if hardware is None
               else RealEnv(config, hardware=hardware))
        self._adopt(env, *env.reset())


def _knob_names(spec: MachineSpec, env) -> list[str]:
    """The spec's knobs, checked against the ones the environment built.

    This used to be a reconciliation layer: three cases and an overall sign flip,
    because the spec driving the mapping could describe a different machine from
    the one inside the environment, and the two disagreed about which end of a
    chicane counts as positive. Neither is true any more — the environment builds
    its knobs *from* this spec — so a knob either exists there under its own name
    or does not exist at all. Measured on both shipped machines: 109 of 109 and
    23 of 23 mapped straight through.

    A knob can legitimately be absent: the environment skips any whose magnets
    are not all inside the modelled stretch. Anything else is a bug in the spec
    and is raised here rather than eight shots into an episode.
    """
    available = set(env.magnet_names)
    present = [k.name for k in spec.knobs if k.name in available]
    extra = available - {k.name for k in spec.knobs}
    if extra:
        raise KeyError(
            f"the environment has {len(extra)} knob(s) the spec does not declare: "
            f"{sorted(extra)[:6]}. They cannot have come from this spec."
        )
    return present





