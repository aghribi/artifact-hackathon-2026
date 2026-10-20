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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lattice import Line  # noqa: E402
from machine import MachineSpec  # noqa: E402

# What the benchmark books for a beam that never arrived, `score.FAILURE_SCORE`.
FAILURE_SCORE = 10.0
# Transmission's weight in the mismatch, `sim_env.TRANSMISSION_WEIGHT`.
TRANSMISSION_WEIGHT = 2.0
# What moving the beam-intercepting screen costs on CLARA: only one may be in the
# beam and swapping takes about half a minute of mechanics, far longer than the
# shot itself, so it is counted separately from the shot budget.
SCREEN_SWITCH_SECONDS = 30.0


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
    # What the episode books if it stops here: the benchmark's mismatch on this
    # reading plus what the shots have cost. Lower is better.
    score: float = FAILURE_SCORE

    @property
    def spot(self) -> np.ndarray:
        return np.array([self.mu_x, self.mu_y, self.sigma_x, self.sigma_y])


def evaluation_cost(shots: int, weight: float = 0.1, budget: float = 10.0) -> float:
    """What shots have cost so far, in requested spot sizes.

    `sim_env.evaluation_cost`. It is why a method cannot simply scan forever: the
    twentieth shot is nearly free, the two hundredth is not.
    """
    return weight * float(np.log2(1 + shots / budget))


def mismatch(spot: np.ndarray, target: np.ndarray, transmission: float,
             on_screen: bool, cap: float = FAILURE_SCORE) -> float:
    """`score.per_case_scores`, as `SimEnv._mismatch` computes it.

    Errors in units of the requested spot size, position and size weighted
    equally, plus what the beam that never arrived costs, clipped at `cap` —
    which is also what a lost beam books, so losing the beam is never cheaper
    than delivering a bad one.
    """
    if not on_screen:
        return cap
    error = np.abs(np.asarray(spot) - np.asarray(target)) / float(
        np.mean(target[2:])
    )
    return float(min(
        (error[0] + error[1]) / 2 + (error[2] + error[3]) / 2
        + TRANSMISSION_WEIGHT * (1.0 - transmission),
        cap,
    ))


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

    def clip(self, name: str, value: float) -> float:
        low, high = self.limits.get(name, (-np.inf, np.inf))
        return float(np.clip(value, low, high))

    @property
    def last(self) -> Reading:
        return self.history[-1]

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
        """`score.CaseResult.relative_error`, what hit rates are thresholded on."""
        best = min(self.history, key=self.mismatch_of, default=None)
        if best is None or not best.on_screen:
            return FAILURE_SCORE
        return float(np.abs(best.spot - self.target).mean()
                     / float(np.mean(self.target[2:])))

    @property
    def machine_seconds(self) -> float:
        """What this episode would have cost on the real machine."""
        return self.shots + self.screen_switches * SCREEN_SWITCH_SECONDS


# ---------------------------------------------------------------- SimEnv


class SimEnvMachine(Machine):
    """CLARA as `RL_framework` deals it, scored by the benchmark."""

    def __init__(self, spec: MachineSpec, seed: int = 0, config=None):
        super().__init__()
        here = Path(__file__).resolve().parent
        for extra in (here.parent, here.parent / "RL_framework"):
            if str(extra) not in sys.path:
                sys.path.insert(0, str(extra))
        from rl_framework.config import RLConfig
        from rl_framework.envs.sim_env import SimEnv

        self.spec = spec
        self.env = SimEnv(config or RLConfig.clara())
        observation, info = self.env.reset(seed=seed)
        self.target = np.asarray(observation["target"], dtype=float)
        self.diagnostics = list(self.env.bpm_names)
        self._mapping = _map_knobs(spec, self.env)
        self.knob_names = sorted(self._mapping)
        self.limits = {
            name: _group_limits(self.env, plan) for name, plan in self._mapping.items()
        }
        self._settings = {
            name: _present(self.env, plan) for name, plan in self._mapping.items()
        }
        # `reset` returns a full info dictionary, so the starting reading is a
        # real one and costs no shot.
        self._record(observation, info)

    # ---------- driving ----------

    def set(self, settings: dict[str, float]) -> Reading:
        magnets: dict[str, float] = {}
        for name, value in settings.items():
            if name not in self._mapping:
                raise KeyError(
                    f"{name} is not a knob this machine can set; it has "
                    f"{len(self.knob_names)}"
                )
            value = self.clip(name, float(value))
            self._settings[name] = value
            for target, sign in self._mapping[name]:
                magnets[target] = sign * value
        observation, _, _, _, info = self.env.step({"magnets": magnets})
        self.shots += 1
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
            # is quoted against.
            score=float(info["benchmark_score"]),
        )
        self.history.append(reading)
        return reading


def _map_knobs(spec: MachineSpec, env) -> dict[str, list[tuple[str, float]]]:
    """Each spec knob as a list of `(SimEnv magnet, sign)`.

    Three cases, in order:

    1. **The name is already one of SimEnv's.** Single-element knobs land here:
       `machine.PLANE_SUFFIX` and `sim_env.PROPERTY_SUFFIX` agree, by design.
    2. **SimEnv has one knob driving the same elements.** The VBC is this: four
       dipoles on one supply, which SimEnv exposes as `cla_vbc_chicane` and will
       not let anybody set individually. Matched on the element set, allowing an
       overall sign flip, since which end of a chicane counts as positive is a
       convention and the two files disagree about it.
    3. **Expand the group.** Each element gets its own SimEnv knob and the spec's
       sign, which is how the paired sextupoles are driven.

    Anything left over raises here rather than mid-episode.
    """
    from rl_framework.envs.sim_env import PROPERTY_SUFFIX

    available = set(env.magnet_names)
    coupled = {}
    for name in available:
        knob = env._knobs[name]
        members = ((knob.element_name, 1.0), *knob.coupled)
        coupled[frozenset(e for e, _ in members)] = (name, dict(members))

    mapping: dict[str, list[tuple[str, float]]] = {}
    unmapped: list[str] = []
    for knob in spec.knobs:
        if knob.name in available:
            mapping[knob.name] = [(knob.name, 1.0)]
            continue
        match = coupled.get(frozenset(knob.elements))
        if match is not None:
            target, signs = match
            theirs = np.array([signs[e] for e in knob.elements])
            ours = np.array(knob.signs)
            flip = np.sign(np.dot(theirs, ours)) or 1.0
            if not np.allclose(theirs * flip, ours):
                unmapped.append(
                    f"{knob.name}: SimEnv's {target} drives the same elements with "
                    f"signs {list(theirs)}, the spec says {list(ours)}"
                )
                continue
            mapping[knob.name] = [(target, float(flip))]
            continue
        suffix = PROPERTY_SUFFIX.get(knob.prop)
        members = ([(e + suffix, s) for e, s in zip(knob.elements, knob.signs)]
                   if suffix is not None else [])
        if members and all(e in available for e, _ in members):
            mapping[knob.name] = members
            continue
        unmapped.append(f"{knob.name} ({knob.prop} on {knob.elements})")
    if unmapped:
        raise KeyError(
            f"{len(unmapped)} spec knob(s) have no SimEnv magnet:\n    "
            + "\n    ".join(unmapped[:8])
        )
    return mapping


def _group_limits(env, plan) -> tuple[float, float]:
    """The tightest limits that hold for every magnet a knob drives."""
    lows, highs = [], []
    for target, sign in plan:
        low, high = env.magnet_limits[target]
        lows.append(low / sign if sign > 0 else high / sign)
        highs.append(high / sign if sign > 0 else low / sign)
    return float(max(lows)), float(min(highs))


def _present(env, plan) -> float:
    target, sign = plan[0]
    return float(env._knobs[target].value) / sign


# ---------------------------------------------------------------- Line
