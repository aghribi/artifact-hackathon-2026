"""The vector of numbers a stage may turn, and the translation to and from SimEnv.

SimEnv names one knob per settable number — `cla_fea_mag_quad_13` for a quadrupole's
k1, `cla_feh_mag_hvcor_01_h` and `_v` for the two planes of a corrector — and holds the
operating range of each in `env.magnet_limits`. A `KnobSet` is a chosen subset of those,
fixed in order, so a stage can work in a plain vector and hand SimEnv back the mapping
it wants.

The ranges are SimEnv's own. They come from what +-100 A reaches through the CLARA
current-to-strength calibration, which is what the hardware can physically do, and
re-deriving them here would only risk disagreeing with the environment about what is
reachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

# What one knob of each element type is called, and the suffix SimEnv gives its name.
# `sim_env.PROPERTY_SUFFIX`, restricted to the properties this agent turns.
PROPERTIES = {
    "Quadrupole": (("k1", ""),),
    "Sextupole": (("k2", ""),),
    "CombinedCorrector": (("horizontal_angle", "_h"), ("vertical_angle", "_v")),
    "Cavity": (("phase", ""),),
    # Knobs that drive several magnets at once, named by the machine spec and
    # behaving like one magnet of the type they are made of: the VBC chicane is
    # four dipoles on one supply, the FEA sextupole pair is two on one.
    "Chicane": (("angle", ""),),
    "SextupolePair": (("k2", ""),),
}
VBC_CHICANE = "cla_vbc_chicane"

# A natural step for each kind of knob, in its own units: roughly the smallest move
# that does something you can see, and small enough that the response to it is still
# linear.
#
# Steps have to be measured against these and not against the operating range. The
# ranges come from what +-100 A reaches, and for a CLARA corrector that is +-50 to
# +-80 mrad — a kick that puts the beam into the wall several times over. One per
# cent of that span, which looks like a cautious probe, is a 1 mrad kick: the
# finite-difference response it measures is not a derivative but the beam being
# thrown away, and steering built on it diverges. `MFopt.optics_env` probes its
# correctors at 2e-5 rad and its quadrupoles at 0.05 m^-2 for the same reason.
SCALE = {
    "horizontal_angle": 1e-4,  # rad
    "vertical_angle": 1e-4,  # rad
    "k1": 0.5,  # m^-2
    "k2": 5.0,  # m^-3
    "phase": 2.0,  # degrees
    "angle": 5e-3,  # rad, the chicane
}


@dataclass(frozen=True)
class Knob:
    """One settable number: what SimEnv calls it, and what it is on the twin."""

    name: str  # SimEnv's knob name, e.g. "cla_feh_mag_hvcor_01_h"
    element: str  # the element it lives on, e.g. "cla_feh_mag_hvcor_01"
    prop: str  # "k1", "k2", "horizontal_angle", "vertical_angle", "phase", "angle"
    low: float
    high: float


class KnobSet:
    """An ordered set of knobs, with their ranges and the conversions a stage needs."""

    def __init__(self, knobs: Sequence[Knob]):
        self.knobs = list(knobs)
        self.names = [knob.name for knob in self.knobs]
        self.elements = [knob.element for knob in self.knobs]
        self.props = [knob.prop for knob in self.knobs]
        self.low = np.array([knob.low for knob in self.knobs], dtype=float)
        self.high = np.array([knob.high for knob in self.knobs], dtype=float)
        self.span = self.high - self.low
        self.middle = (self.high + self.low) / 2
        # What a step of one means for each knob, in its own units. Everything that
        # moves a knob — probing, planning, the trust region — works in these.
        self.scales = np.array([SCALE[knob.prop] for knob in self.knobs], dtype=float)

    def __len__(self) -> int:
        return len(self.knobs)

    @classmethod
    def build(
        cls,
        element_names: Sequence[str],
        element_types: Mapping[str, str],
        limits: Mapping[str, tuple[float, float]],
    ) -> "KnobSet":
        """The knobs of `element_names`, in order, each plane of a corrector its own.

        `element_types` says what kind of element each one is (the twin knows), and
        `limits` is `env.magnet_limits`. An element SimEnv will not let the agent set
        is a configuration error, not something to skip quietly.
        """
        knobs, seen = [], set()
        for element in element_names:
            kind = "Chicane" if element == VBC_CHICANE else element_types.get(element)
            if kind is None:
                raise KeyError(f"{element} is not an element of the modelled stretch")
            if kind not in PROPERTIES:
                raise KeyError(f"{element} is a {kind}, which has no knob to turn")
            for prop, suffix in PROPERTIES[kind]:
                name = element + suffix
                if name in seen:  # a corrector shared between the two stages
                    continue
                if name not in limits:
                    raise KeyError(
                        f"{name} is not settable in this environment. Add it to "
                        f"`RLConfig.controllable_magnets`."
                    )
                low, high = limits[name]
                knobs.append(Knob(name, element, prop, float(low), float(high)))
                seen.add(name)
        return cls(knobs)

    @classmethod
    def from_knob_names(
        cls,
        names: Sequence[str],
        element_types: Mapping[str, str],
        limits: Mapping[str, tuple[float, float]],
    ) -> "KnobSet":
        """The knobs SimEnv calls `names`, in order.

        `build` takes elements and expands each into its planes; this takes the
        settable numbers themselves, which is what an SVD ranking hands back — it
        ranks `cla_s04_mag_hvcor_02_h` and `_v` separately, because one plane of a
        corrector can matter and the other not.
        """
        knobs, seen = [], set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            if name.endswith(("_h", "_v")):
                element = name[:-2]
                prop = "horizontal_angle" if name.endswith("_h") else "vertical_angle"
            else:
                element = name
                kind = "Chicane" if name == VBC_CHICANE else element_types.get(element)
                if kind is None:
                    raise KeyError(f"{name} is not a knob of the modelled stretch")
                prop = PROPERTIES[kind][0][0]
            if name not in limits:
                raise KeyError(
                    f"{name} is not settable in this environment. Add it to "
                    f"`RLConfig.controllable_magnets`."
                )
            low, high = limits[name]
            knobs.append(Knob(name, element, prop, float(low), float(high)))
        return cls(knobs)

    # ---------- talking to SimEnv ----------

    def action(self, values: np.ndarray) -> dict:
        """The SimEnv action that sets these knobs to `values` and nothing else."""
        values = self.clip(values)
        return {"magnets": dict(zip(self.names, (float(v) for v in values)))}

    def read(self, settings: Mapping[str, float]) -> np.ndarray:
        """These knobs' values out of `env.magnet_settings()`, in order."""
        return np.array([float(settings[name]) for name in self.names])

    def clip(self, values: np.ndarray) -> np.ndarray:
        """Into the operating range, and finite. Nothing that is not a number may
        reach a magnet: Cheetah stops tracking entirely rather than passing one on."""
        values = np.asarray(values, dtype=float).reshape(-1)
        values = np.where(np.isfinite(values), values, self.middle)
        return np.clip(values, self.low, self.high)

    # ---------- network and solver units ----------

    def normalise(self, values: np.ndarray) -> np.ndarray:
        """Native units to [-1, 1] over the operating range."""
        return (np.asarray(values, dtype=float) - self.middle) / (self.span / 2)

    def denormalise(self, unit: np.ndarray) -> np.ndarray:
        """[-1, 1] back to native units, clipped to the range."""
        return self.clip(self.middle + np.asarray(unit, dtype=float) * (self.span / 2))

    def rescale(self, sensitivity: np.ndarray, effect: float, limit: float = 1e3) -> None:
        """Redefine a step as whatever moves the reading by `effect`.

        `sensitivity[j]` is how much the thing being optimised changes per unit of
        knob j, in whatever units that thing is measured in; `effect` is how much of
        it one step should be worth. A fixed step in physical units cannot serve
        here, because the same 0.1 mrad is a large kick from a corrector at the top
        of the line, with forty quadrupoles of lever arm behind it, and almost
        nothing from the one a metre before the screen — and it is the last one that
        has to move the spot two millimetres to the position that was requested. In
        units of effect both are a step of one.

        Knobs the reading does not respond to keep a large step, bounded by `limit`;
        they will not be chosen anyway, and dividing by nearly zero is not useful.
        """
        # A probe that lost the beam reports no response, or not a number; either
        # way it is a knob the reading does not respond to. Left as NaN it would
        # poison the scale, then every step taken with it, and eventually arrive at
        # Cheetah as a magnet strength that is not a number.
        sensitivity = np.nan_to_num(
            np.abs(np.asarray(sensitivity, dtype=float)).reshape(-1),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        floor = effect / (limit * np.maximum(self.scales, 1e-30))
        self.scales = effect / np.maximum(sensitivity, floor)

    def trust_region(self, values: np.ndarray, steps: float) -> tuple[np.ndarray, np.ndarray]:
        """How far from `values` one shot may move each knob, clipped to the range.

        `steps` is in units of each knob's natural step (`SCALE`), so one number
        gives a corrector and a quadrupole moves that are comparable in effect
        rather than comparable as fractions of two unrelated operating ranges.
        """
        reach = self.scales * steps
        return (
            np.maximum(values - reach, self.low),
            np.minimum(values + reach, self.high),
        )
