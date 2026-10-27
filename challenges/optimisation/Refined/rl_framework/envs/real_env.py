"""CLARA itself, behind the same interface as the simulation.

    env = RealEnv(RLConfig(spec="machines/clara_febe_no_bpms.json",
                           target=(0.0, 0.0, 266e-6, 150e-6)))

Same actions, same observations, same `info`, same score as `SimEnv` — so the
tuners in `Refined/` drive this without knowing which one they have, and the
numbers they report are comparable with the simulated ones rather than merely
similar. Everything shared lives in `base_env.py`; what is here is the four
things a real machine does differently:

- magnets are set and read through the control system, not a tracker;
- a screen is a mechanism that takes half a minute to move, not an index;
- a camera frame has to be turned into four numbers, as `screen.measure` does;
- there is no new machine to draw at `reset()`, so it does not draw one.

**Every point where a real value is written or read is a placeholder** in
`hardware.py`, raising `NotImplementedError` until populated. Nothing here
imports CATAP or EPICS, so this module is importable and testable with no
control network present — pass `hardware=` a substitute to exercise it.

**`reset()` never moves a magnet.** It takes one reading and leaves the tune
exactly as it found it: the machine you walk up to is the starting point, which
is both the safe behaviour and the honest one, since the survey error and the
beam a simulation would redraw are not ours to choose. The consequence is that
`config.target` must say what the beam is being asked for — there is no
distribution to sample a request from on a real machine.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rl_framework.config import RLConfig
from rl_framework.envs import hardware as _hardware
from rl_framework.envs.base_env import (
    PROPERTY_SUFFIX,
    BaseAcceleratorEnv,
    Transport,
)

# `screen.measure` turns a camera frame into (mu_x, mu_y, sigma_x, sigma_y) in
# metres, using the measure written for real CLARA captures. It sits beside the
# tuning modules in `Refined/`, and is imported where it is used so that a
# machine lacking its dependencies can still import and read `RealEnv`.
_REFINED = Path(__file__).resolve().parents[2]


class HardwareTransport(Transport):
    """Magnets as the control system has them, in native lattice units.

    Readbacks are cached for the duration of a shot. Without that, one `info`
    dictionary costs a control-system read per magnet — 109 of them on CLARA,
    every shot, to answer a question the optimiser asks once.
    """

    name = "hardware"

    def __init__(self, hardware=_hardware):
        self._hw = hardware
        self._cache: dict[str, float] = {}
        self._pending: list[str] = []

    @staticmethod
    def _supply(element: str, prop: str) -> str:
        """The name of the thing that is actually powered.

        A model's `cla_s02_mag_hvcor_01` is one element with a horizontal and a
        vertical angle on it; the machine has two separate magnets, HCOR and
        VCOR, on two separate supplies. So the plane has to travel with the
        name or the hardware layer cannot tell which one is being set — and it
        would quietly drive one plane with the other's demand.

        The suffix convention is the spec's own: this is exactly the knob name
        that `base_env` builds, `cla_s02_mag_hvcor_01_h`.
        """
        if prop not in PROPERTY_SUFFIX:
            raise KeyError(f"{element}: not a settable property: {prop}")
        return element + PROPERTY_SUFFIX[prop]

    def invalidate(self) -> None:
        """Forget the readbacks. Called once per acquisition."""
        self._cache.clear()

    def read(self, element: str, prop: str) -> float:
        supply = self._supply(element, prop)
        if supply not in self._cache:
            self._cache[supply] = float(self._hw.GET_REAL_MAGNET_POWER(supply))
        return self._cache[supply]

    def write(self, element: str, prop: str, value: float,
              follow: Sequence[str] = ()) -> None:
        """Set a magnet in native units.

        `follow` is a pole face that turns with a chicane's body angle. On a
        model those are separate attributes to write; on the machine the iron
        does it, so there is nothing to send — the face is not independently
        powered. Writing it would mean inventing a supply that does not exist.
        """
        supply = self._supply(element, prop)
        self._hw.SET_REAL_MAGNET_POWER(supply, float(value))
        self._cache[supply] = float(value)
        self._pending.append(supply)

    def settle(self) -> None:
        """Block until everything written since the last shot has got there."""
        if self._pending:
            self._hw.WAIT_FOR_MAGNETS_SETTLED(list(dict.fromkeys(self._pending)))
            self._pending.clear()
            # What was asked for is not what the supplies ended up at.
            self._cache.clear()


def _without_rf(config: RLConfig) -> RLConfig:
    """The same configuration, with the RF phases taken out of the agent's hands.

    Nothing in the pipeline tunes RF: the optics family is quadrupoles and
    correctors, and no method ever chooses a phase. The cavities were reachable
    anyway, because `controllable_magnets` defaults to every knob the spec
    declares and CLARA's spec declares three — and `restore_best` re-sets
    *every* knob from a snapshot, so the last shot of every episode wrote three
    RF phases back. Writing back what was read is nominally a no-op, but it is
    still a caput to a linac on the strength of a readback nobody checked, in
    exchange for nothing at all.

    So on a real machine they are not offered. The spec is unchanged, the
    simulation is unchanged, and `hardware.py`'s cavity branches stay as a
    backstop for anything that reaches them another way.
    """
    spec = config.machine_spec()
    phases = {knob.name for knob in spec.knobs if knob.prop == "phase"}
    if not phases:
        return config
    if config.controllable_magnets is None:
        drivable = [knob.name for knob in spec.knobs if knob.name not in phases]
        return replace(config, controllable_magnets=drivable)
    asked = sorted(phases.intersection(config.controllable_magnets))
    if asked:
        raise ValueError(
            f"{asked} are RF phases, and this environment does not set RF. "
            f"Nothing in the pipeline tunes them, so they are excluded by "
            f"default; asking for them explicitly is refused rather than "
            f"quietly dropped. Implement them in the hardware module first."
        )
    return config


class RealEnv(BaseAcceleratorEnv):
    """The machine, scored exactly as the simulation is."""

    def __init__(self, config: RLConfig, hardware=_hardware):
        # Checked before anything else, so a missing request fails on the way in
        # rather than after the construction has read half the control system.
        if config.target is None:
            raise ValueError(
                "RealEnv needs `config.target`: (mu_x, mu_y, sigma_x, sigma_y) in "
                "metres off the readout's centre. A simulation draws a request "
                "every episode from `score.sample_target`; a real machine is being "
                "asked for something specific, and nothing here should guess what."
            )
        config = _without_rf(config)
        self._hw = hardware
        # Which screen is physically in the beam. None until we have put one
        # there: we cannot know what an operator left inserted, so the first
        # acquisition inserts the readout explicitly rather than assuming.
        self._inserted: str | None = None
        super().__init__(config)

        self._acquire()
        self._scored = self._readings["stats"]
        self._scored_transmission = self._readings["transmission"]

    # ---------- what the base class asks for ----------

    def _make_transport(self) -> HardwareTransport:
        return HardwareTransport(self._hw)

    def _extent_of(self, name: str) -> tuple[float, float, float, float]:
        """What the camera actually sees, in metres. Required, not inferred.

        Asked for every diagnostic including the BPMs, which have no chip; they
        answer with the model's zeros and nothing reads it.
        """
        if name not in self.screen_names:
            return self._diagnostics[name].extent
        extent = self._hw.GET_REAL_SCREEN_EXTENT(name)
        if extent is None or len(tuple(extent)) != 4:
            raise ValueError(
                f"GET_REAL_SCREEN_EXTENT({name!r}) must return "
                f"(x_min, x_max, y_min, y_max) in metres, got {extent!r}"
            )
        return tuple(float(v) for v in extent)

    def _hard_reset(self) -> dict[str, Any]:
        """Start an episode without touching the machine.

        No magnet is written and nothing is restored: the tune in force is the
        starting point. The only state reset is the shot counter, which the base
        class has already done, and the request, which comes from the config.
        """
        self.target = self._draw_target()  # `config.target`, required above
        self.target_screen = self.scored_screen
        self._acquire()
        self._scored = self._readings["stats"]
        self._scored_transmission = self._readings["transmission"]
        return self._get_obs()

    def _acquire(self) -> None:
        """One shot: settle the magnets, put the right screen in, and read."""
        self.transport.settle()
        self.transport.invalidate()
        self._select_screen(self._target_screen)

        image = np.asarray(self._hw.GET_REAL_SCREEN_IMAGE(self._target_screen),
                           dtype=float)
        extent = self.screen_extent()
        stats, fraction = self._measure_frame(image, extent)
        transmission = float(np.clip(self._hw.GET_REAL_TRANSMISSION(), 0.0, 1.0))

        count = len(self.bpm_names)
        bpm_x, bpm_y = np.zeros(count), np.zeros(count)
        bpm_valid = np.zeros(count, dtype=np.int8)
        for slot, name in enumerate(self.bpm_names):
            x, y, valid = self._hw.GET_REAL_BPM_READING(name)
            bpm_x[slot], bpm_y[slot], bpm_valid[slot] = float(x), float(y), bool(valid)

        self._readings = {
            "bpm_x": bpm_x,
            "bpm_y": bpm_y,
            "bpm_valid": bpm_valid,
            "image": self._rebin(image),
            "on_screen_fraction": fraction,
            "stats": stats,
            "transmission": transmission,
        }

    # ---------- the parts that are mechanical ----------

    def _select_screen(self, screen: str) -> None:
        """Put `screen` in the beam, taking out whatever was there.

        Only one may be in at a time. Doing nothing when it is already in is the
        whole reason `Refined` can steer on screens affordably — the common case
        is that the beam is already on the readout and no mechanism moves.
        """
        if self._inserted == screen:
            return
        if self._inserted is not None:
            self._hw.RETRACT_REAL_SCREEN(self._inserted)
            self._hw.WAIT_FOR_SCREEN_SETTLED(self._inserted)
        self._hw.INSERT_REAL_SCREEN(screen)
        self._hw.WAIT_FOR_SCREEN_SETTLED(screen)
        self._inserted = screen

    def _measure_frame(self, image: np.ndarray, extent):
        """A camera frame as (mu_x, mu_y, sigma_x, sigma_y, on_screen), in metres.

        `screen.py` already does this, with the measure written for real
        CLARA captures and the corrections it needs — the half-pixel corner bias
        and a background floor, without which the dark half of the chip drags the
        spot wider than it is. Using it here is the point: the agent is driven by
        the same quantity on the machine as in simulation.
        """
        if str(_REFINED) not in sys.path:
            sys.path.insert(0, str(_REFINED))
        from screen import measure  # noqa: PLC0415 — see the module docstring

        reading = measure(image, extent)
        if not reading.ok:
            return (0.0, 0.0, 0.0, 0.0, False), 0.0
        # How much of the frame's light is inside the chip is not a thing a
        # camera can answer — what fell off the edge was never recorded. The
        # charge that failed to arrive is `GET_REAL_TRANSMISSION`'s business, and
        # it is what the score charges for; this stays 1.0 so the two are not
        # counted twice.
        return (*reading.moments, True), 1.0

    def _rebin(self, image: np.ndarray) -> np.ndarray:
        """The camera frame on the observation's fixed grid.

        `config.screen_shape` is part of the observation space, and a real camera
        is whatever resolution it is, so the frame is summed into that grid the
        way `SimEnv` histograms its particles into it. The moments are taken from
        the *full* frame above, not from this, so nothing is lost to the binning.
        """
        rows, columns = self._shape
        if image.shape == (rows, columns):
            binned = image.astype(float)
        else:
            source_rows, source_columns = image.shape
            row_edges = (np.arange(rows + 1) * source_rows) // rows
            column_edges = (np.arange(columns + 1) * source_columns) // columns
            binned = np.add.reduceat(
                np.add.reduceat(image.astype(float), row_edges[:-1], axis=0),
                column_edges[:-1], axis=1,
            )
        if self.config.normalise_image and binned.max() > 0:
            binned = binned / binned.max()
        return binned

    # ---------- what a simulation has and a machine does not ----------

    @property
    def screen_switch_seconds(self) -> float:
        return _hardware.SCREEN_SWITCH_SECONDS
