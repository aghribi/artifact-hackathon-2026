"""Which knobs invalidate the matched optics, and therefore have to be scanned over.

    python -I sensitivity.py machines/clara_febe.json
    python -I sensitivity.py machines/clara_febe.json --apply   # write the axes back

The precompute answers one question: *which quadrupole strengths give this spot?*
That has a single answer only if everything else is pinned. This works out what
"everything else" is, so the grid spans it — and, just as usefully, so it does not
span things that do not matter, because every axis multiplies a job measured in
hours.

**The measurement.** Hold the matching quadrupoles where they are, sweep one other
knob across the range it will really be used over, and watch the spot at the
readout — the beam's own rms about its own centroid, not the clipped camera
reading. That distinction is the whole measurement: a corrector that walks the beam
half off the chip changes what a camera reports enormously and changes the optics
not at all, and scoring the camera makes every steerer look like the most important
knob on the machine. If the spot moves, then to get back to the requested size the matched
quadrupoles would have had to be different — which is precisely what it means for
the family to depend on that knob. Reported as |Δ ln σ|, the fractional change in
spot size, because that is the quantity the family would have to undo.

It is deliberately *not* a derivative. A knob with a small derivative and a wide
range invalidates the family just as thoroughly as a sharp one with a narrow range,
and the question is always "across the range this knob will actually be used over".

**What is excluded.** The matching quadrupoles themselves: they are what the family
solves *for*, not a condition it is solved *under*. Changing them is supposed to
change the spot.

**Ranges are narrowed, not discarded.** A knob's limits are what the supply can
deliver, which is not the same as what the machine can be run at. CLARA's
compressor is a clear case: swept to zero the bunch is never compressed, the last
linac puts a large energy spread on a long bunch, and the dogleg throws it into the
wall — so most of its nominal range loses the beam entirely. The knob is not
irrelevant; its usable span is narrower than its limits. The sweep therefore keeps
the run of points around the present setting that holds the beam, judges the knob
over *that*, and proposes it with the narrowed range.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lattice import Line  # noqa: E402
from machine import MachineSpec, ScanAxis  # noqa: E402

# Points sampled across each knob's range, including both ends.
SWEEP_POINTS = 5
# A knob that changes the spot by more than this fraction, anywhere in its range,
# is proposed as an axis. 5 % of a spot size is already a tenth of the tolerance a
# tight request is scored against, so a family that ignores it is wrong by more
# than the tuning is trying to achieve.
AXIS_THRESHOLD = 0.05
# Below this transmission the beam is being lost rather than resized, and the size
# reading means nothing.
HEALTHY = 0.5
# Correctors. Excluded from the optics scan altogether, not swept narrowly.
#
# A corrector adds an orbit. That is a linear, separable response, and the pipeline
# already has a stage that solves it: the SVD of corrector against position, applied
# after the optics is set. It is not part of what a matched family has to span, and
# precomputing over it would be precomputing the thing that is cheapest to solve
# live.
#
# They do perturb the measured spot — walk a beam off axis and dispersion, the
# sextupoles and the chamber all reshape it, which is why an earlier version of this
# scan ranked four correctors above every real optics knob at up to 60 %. That is
# the orbit talking, not the optics, and the orbit has its own stage.
STEERING_PROPS = {"horizontal_angle", "vertical_angle"}


@dataclass
class Effect:
    """What sweeping one knob does to the spot the family is supposed to deliver."""

    knob: str
    prop: str
    nonlinear: bool
    low: float
    high: float
    size_change: float  # max |d ln sigma| over the usable span, worst plane
    change_x: float
    change_y: float
    transmission: float  # the worst seen over the usable span
    usable: bool  # was any span around the present setting able to hold the beam
    narrowed: bool = False  # the limits were wider than the machine can be run at
    note: str = ""

    @property
    def proposes_axis(self) -> bool:
        return self.usable and self.size_change >= AXIS_THRESHOLD


def sweep_range(line: Line, name: str) -> tuple[float, float]:
    """The span a knob may be swept over: its limits, or a window about nominal."""
    knob = line.knobs[name]
    nominal = line.nominal(name)
    if knob.low is None or knob.high is None:
        reach = abs(nominal) * 0.5 or 1.0
        return nominal - reach, nominal + reach
    return knob.low, knob.high


def sweep_values(line: Line, name: str, points: int) -> np.ndarray:
    """Sample points around the present setting, finely near it and coarsely far.

    Knobs differ enormously in how sharp they are, and one uniform grid cannot
    serve them all. A cavity phase is smooth over its whole thirty-degree window;
    CLARA's compressor loses the beam 0.03 rad from where it sits, so a uniform
    sweep of its range samples nothing usable at all and the knob looks irrelevant
    when it is in fact the sharpest on the machine.

    A geometric ladder of offsets either side of the present value resolves a span
    a thousandth of the range wide and still reaches the limits.
    """
    low, high = sweep_range(line, name)
    present = float(np.clip(line.value(name), low, high))
    reach = max(high - present, present - low)
    if reach <= 0:
        return np.array([present])
    rungs = max(1, (points - 1) // 2)
    offsets = np.geomspace(reach * 1e-3, reach, rungs)
    values = np.concatenate([[present], present + offsets, present - offsets])
    return np.unique(np.clip(values, low, high))


def _usable_span(good: np.ndarray, values: np.ndarray, present: float):
    """The run of sweep points around the present setting that holds the beam.

    A knob is used from where it is now, so the span that matters is the one
    containing its present value — not the widest run anywhere in its range.
    """
    if not good.any():
        return None
    start = int(np.argmin(np.abs(values - present)))
    if not good[start]:
        # The present setting itself loses the beam at this sampling; fall back to
        # the longest healthy run, which is the best that can be said.
        best = (None, 1)  # a single point is not a span
        lo = None
        for i, ok in enumerate(list(good) + [False]):
            if ok and lo is None:
                lo = i
            elif not ok and lo is not None:
                if i - lo > best[1]:
                    best = ((lo, i - 1), i - lo)
                lo = None
        return best[0]
    lo = hi = start
    while lo > 0 and good[lo - 1]:
        lo -= 1
    while hi < len(good) - 1 and good[hi + 1]:
        hi += 1
    return (lo, hi) if hi > lo else None


def measure(line: Line, candidates: list[str], points: int = SWEEP_POINTS) -> list[Effect]:
    """Sweep each candidate across its range with everything else held still."""
    base = line.track()
    _, base_size = base.beam_at(line.readout)
    reference = base_size.numpy().astype(float)
    if not np.all(reference > 0):
        raise RuntimeError(
            "the nominal tune does not put a measurable spot on the readout; "
            "fix the starting optics before asking what perturbs it"
        )

    effects: list[Effect] = []
    for name in candidates:
        low, high = sweep_range(line, name)
        values = sweep_values(line, name, points)
        # One batched track for the whole sweep: the knob is a column of values and
        # every element downstream of it carries the batch.
        readings = line.track(
            overrides={name: torch.tensor(values, dtype=torch.float32)},
            batch=len(values),
        )
        _, sizes = readings.beam_at(line.readout)
        sizes = sizes.numpy().astype(float)
        transmission = readings.transmission.numpy().astype(float)

        good = (transmission >= HEALTHY) & np.all(sizes > 0, axis=-1)
        span = _usable_span(good, values, line.value(name))
        knob = line.knobs[name]
        if span is None:
            effects.append(Effect(
                name, knob.prop, knob.nonlinear, low, high,
                float("nan"), float("nan"), float("nan"),
                float(transmission.min()), usable=False,
                note="no span around the present setting holds the beam",
            ))
            continue
        lo, hi = span
        inside = slice(lo, hi + 1)
        shift = np.abs(np.log(sizes[inside] / reference))
        narrowed = not good.all()
        effects.append(Effect(
            knob=name, prop=knob.prop, nonlinear=knob.nonlinear,
            low=float(values[lo]), high=float(values[hi]),
            size_change=float(shift.max()),
            change_x=float(shift[:, 0].max()), change_y=float(shift[:, 1].max()),
            transmission=float(transmission[inside].min()),
            usable=True, narrowed=narrowed,
            note=(f"range narrowed from [{low:g}, {high:g}]: the rest loses the beam"
                  if narrowed else ""),
        ))
    return effects


def candidates_for(spec: MachineSpec) -> tuple[list[str], list[str], list[str]]:
    """Split the knobs into what the scan judges, and what it has no business judging.

    Three groups:

    - **matching**: what the family solves *for*. Changing them is supposed to
      change the spot; that is their job, not a defect in the family.
    - **steering**: correctors. A separable linear orbit, solved live by its own
      SVD stage, never precomputed.
    - **candidates**: everything else — the conditions the family is solved
      *under*, and the only knobs that can invalidate it.
    """
    matching = set(spec.matching_quads) or {
        k.name for k in spec.knobs if k.prop == "k1"
    }
    steering = {k.name for k in spec.knobs if k.prop in STEERING_PROPS}
    candidates = [
        k.name for k in spec.knobs
        if k.name not in matching and k.name not in steering
    ]
    return candidates, sorted(matching), sorted(steering)


def report(effects: list[Effect], spec: MachineSpec) -> list[Effect]:
    """Print the ranking and return what should become axes."""
    ranked = sorted(
        effects, key=lambda e: (-1 if np.isnan(e.size_change) else e.size_change),
        reverse=True,
    )
    print(f"\n  sweeping {len(effects)} candidate knobs, optics held still")
    print(f"  the family is invalidated by anything that moves the spot it promises\n")
    print(f"  {'knob':<30}{'prop':<20}{'|dln sigma|':>12}{'min trans':>11}  note")
    print("  " + "-" * 86)
    for effect in ranked[:20]:
        change = ("   lost" if np.isnan(effect.size_change)
                  else f"{effect.size_change:>11.3f}")
        mark = " <- axis" if effect.proposes_axis else ""
        print(
            f"  {effect.knob:<30}{effect.prop:<20}{change}"
            f"{effect.transmission:>11.2f}  {effect.note}{mark}"
        )
    if len(ranked) > 20:
        print(f"  ... {len(ranked) - 20} more, all below the leaders")

    proposed = [e for e in ranked if e.proposes_axis]
    lost = [e for e in ranked if not e.usable]
    print(f"\n  {len(proposed)} knob(s) move the spot by more than "
          f"{AXIS_THRESHOLD:.0%} and keep the beam:")
    for effect in proposed:
        print(
            f"      {effect.knob}  ({effect.low:g} to {effect.high:g}, "
            f"x {effect.change_x:.1%}, y {effect.change_y:.1%})"
            + ("  [non-linear]" if effect.nonlinear else "")
            + ("  [range narrowed to where the beam survives]"
               if effect.narrowed else "")
        )
    if not proposed:
        print("      none — a two-dimensional family over the requested spot is enough")
    if lost:
        print(f"\n  {len(lost)} knob(s) hold the beam nowhere near their present "
              f"setting: {[e.knob for e in lost][:6]}")
        print("      these are knobs to bound, not knobs to scan")
    return proposed


def to_axes(proposed: list[Effect], points: int) -> list[ScanAxis]:
    return [
        ScanAxis(
            name=e.knob, kind="knob", knob=e.knob, low=e.low, high=e.high,
            points=points, spacing="linear",
            note=f"sweeping it moves the readout spot by {e.size_change:.1%} with "
                 f"the quadrupoles held still, so the matched family depends on it",
        )
        for e in proposed
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--points", type=int, default=SWEEP_POINTS)
    parser.add_argument("--axis-points", type=int, default=5,
                        help="grid points to give each proposed axis")
    parser.add_argument("--apply", action="store_true",
                        help="write the proposed axes into the spec")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    spec = MachineSpec.load(args.spec)
    line = Line(spec, num_particles=args.particles)
    print(f"  {spec.name}: {len(line.segment.elements)} elements, "
          f"{len(spec.knobs)} knobs, readout {spec.readout}")

    candidates, matching, steering = candidates_for(spec)
    print(f"  {len(matching)} matching quadrupole(s) and {len(steering)} corrector(s) "
          f"are not candidates;")
    print(f"  the first are what the family solves for, the second are a separable "
          f"orbit with their own stage.")
    effects = measure(line, candidates, args.points)
    proposed = report(effects, spec)

    if args.apply:
        spec.axes = to_axes(proposed, args.axis_points)
        out = args.out or args.spec
        spec.save(out)
        print(f"\n  wrote {len(spec.axes)} axis/axes to {out}")
        print(f"  grid is now {spec.grid_shape()} = {spec.grid_size():,} matched tunes")
    else:
        print("\n  (nothing written; pass --apply to put these axes in the spec)")


if __name__ == "__main__":
    main()
