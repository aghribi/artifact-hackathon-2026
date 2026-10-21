"""Any Cheetah lattice, driven by a `MachineSpec`, tracked in torch.

The one piece of machinery everything else sits on. `MBRL_opt/twin.py` is the same
code with CLARA written through it — a hard-coded chicane, hard-coded diagnostic
tags, a hard-coded lattice path. Here all of that comes out of the spec, and what is
left is the part that was hard to get right and is worth keeping exactly as it is:

- **Coupled knobs.** One number drives several elements with signs, and may have to
  write more than one property of each — a chicane's pole faces turn with its body
  angle, and setting one without the other is a different magnet.
- **The transfer-map cache.** Cheetah caches an element's map keyed on beam energy,
  not on the strengths it was built from, so reassigning `k1` leaves the old map in
  place. A batched evaluation then leaves a *batched* map behind and the next
  unbatched track silently returns the wrong shape. Every write clears it.
- **Apertures with a usable gradient.** Whether a particle is inside a wall is a
  step function; a sum of ten thousand step functions is a staircase, not a slope.
  Blurring the wall is badly biased, so the value comes from the hard edge and the
  gradient from a sigmoid.
- **Batching.** Element parameters take a leading dimension, so a population of
  candidate settings tracks in little more time than one — but only the elements
  downstream of a batched knob carry it, so readings come back at mixed shapes and
  have to be broadcast before they can be stacked.

    line = Line(spec, num_particles=4000)
    line.set({"cla_feh_mag_quad_01": 1.7, "vbc_chicane": 0.12})
    reading = line.track()
    reading.sizes[line.index("cla_feh_dia_scr_01")]    # sigma_x, sigma_y there
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cheetah
import numpy as np
import torch

from machine import MachineSpec
from machine_spec import lattice_io

DTYPE = torch.float32
def _forget_maps(element) -> None:
    """Drop an element's cached transfer maps after its parameters change."""
    cache = getattr(element, "_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def _aperture(margin: torch.Tensor, soft: float | None) -> torch.Tensor:
    """Whether a particle is inside a wall it is `margin` metres clear of.

    The value is the step function the machine applies. With `soft` set it also
    carries the derivative of a sigmoid of that width, which is what lets anything
    downstream be optimised; the value itself is untouched, so nothing is biased.
    """
    hard = (margin >= 0).to(DTYPE)
    if soft is None:
        return hard
    smooth = torch.sigmoid(margin / soft)
    return hard + (smooth - smooth.detach())


def _masked_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    total = weight.sum(dim=-1).clamp(min=1.0)
    return (values * weight).sum(dim=-1) / total


def _masked_std(values, weight, mean) -> torch.Tensor:
    total = weight.sum(dim=-1).clamp(min=1.0)
    spread = (weight * (values - mean.unsqueeze(-1)) ** 2).sum(dim=-1) / total
    return torch.sqrt(spread.clamp(min=0.0) + 1e-24)


@dataclass
class Readings:
    """What every diagnostic says, with gradient to whatever was being varied.

    Rows follow `Line.diagnostics`; a leading batch dimension is present whenever
    one was asked for. `valid` is 0 where too little beam arrived to read.
    """

    centroids: torch.Tensor  # (..., n_diagnostics, 2)
    sizes: torch.Tensor  # (..., n_diagnostics, 2)
    valid: torch.Tensor  # (..., n_diagnostics)
    transmission: torch.Tensor  # (...,)
    scrape: torch.Tensor  # (...,) mean metres outside the pipe
    # The beam as the optics made it: moments over every surviving particle, about
    # its own centroid, with no clipping to the camera chip. `sizes` is what an
    # instrument reports and is the right thing to score a tune on; these are the
    # right thing to ask physics questions with. A beam steered half off the chip
    # reads as a tiny spot in `sizes` and an unchanged one here, and conflating the
    # two makes every corrector look like it reshapes the beam.
    beam_centroids: torch.Tensor  # (..., n_diagnostics, 2)
    beam_sizes: torch.Tensor  # (..., n_diagnostics, 2)

    def at(self, index: int):
        """(centroid, size) at one diagnostic, as an instrument would report it."""
        return self.centroids[..., index, :], self.sizes[..., index, :]

    def beam_at(self, index: int):
        """(centroid, size) at one diagnostic, as the optics made it."""
        return self.beam_centroids[..., index, :], self.beam_sizes[..., index, :]


@dataclass
class Waypoint:
    """The beam partway down the line, kept so the tail can be retracked cheaply.

    Matching moves the last few quadrupoles, so every evaluation retracks the same
    long upstream stretch to reach the few metres that changed. On CLARA the stop
    before the final focus is 92nd of 102 and costs 75 ms of an 84 ms track: hoist
    it out of the loop and what is left is 9 ms. Over a grid of twelve thousand
    tunes that is the difference between an hour and ten minutes.

    This is the `Line`-native form of `MFopt.batched.Stations`. The state kept is
    everything the loop in `track` carries between stops, so resuming is exactly
    equivalent to never having stopped.
    """

    stop: str
    stop_index: int  # into `line.stops`
    element_index: int  # into `line.segment.elements`
    beam: "cheetah.ParticleBeam"
    alive: torch.Tensor
    margin: torch.Tensor | None
    outside: list[torch.Tensor]
    rows: dict[str, tuple]
    # What every knob was at when this was taken. A waypoint is only valid for
    # settings that agree with it upstream, and `track` checks that.
    setpoints: dict[str, float]


class Line:
    """A lattice, its knobs and its diagnostics, as the spec describes them."""

    def __init__(
        self,
        spec: MachineSpec,
        num_particles: int | None = None,
        coordinates: np.ndarray | None = None,
        energy_ev: float | None = None,
        device: str | None = None,
    ):
        self.spec = spec
        # Per-line rather than a global default, because a `Line` is often not
        # the only thing in the process: `SimEnvMachine` runs a Cheetah model of
        # its own and would be dragged onto the GPU with it. Batched tracking is
        # worth 2-4x on even a modest card once a batch is past about a million
        # particle-rows, and nothing below that.
        self.device = torch.device(device or "cpu")
        self.segment = self._build(spec).to(self.device)
        self._elements = {e.name: e for e in self.segment.elements}
        # Positions, once. `cheetah.Segment.element_index` rebuilds the whole
        # 655-name list and scans it — 18 us a call — and the resume guard used
        # to make one per knob per track, 2.65 ms of pure lookup on every track
        # however short the tail was.
        self._at = {e.name: i for i, e in enumerate(self.segment.elements)}

        # Where the beam is read and where it is scraped, in lattice order. A
        # diagnostic the spec never named is still a stop if it can lose particles.
        named = set(spec.bpms) | set(spec.screens)
        self.stops: list[str] = []
        self.diagnostics: list[str] = []
        for element in self.segment.elements:
            if not isinstance(element, cheetah.Screen):
                continue
            self.stops.append(element.name)
            if element.name in named:
                self.diagnostics.append(element.name)
        if spec.readout and spec.readout not in self.diagnostics:
            raise ValueError(f"the readout {spec.readout} is not a diagnostic of this line")
        self._row = {name: i for i, name in enumerate(self.diagnostics)}
        self._stop_row = {name: i for i, name in enumerate(self.stops)}
        self.readout = self._row.get(spec.readout, -1)

        self._extent = {
            e.name: tuple(float(v) for v in e.extent)
            for e in self.segment.elements if isinstance(e, cheetah.Screen)
        }
        # One sub-segment per stop, so each stretch is one fused transfer map
        # rather than an element-by-element walk.
        self._chunks, previous = {}, 0
        for name in self.stops:
            index = self._at[name]
            self._chunks[name] = cheetah.Segment(self.segment.elements[previous:index])
            previous = index

        self.knobs = {k.name: k for k in spec.knobs}
        self._nominal = {
            name: float(getattr(self._elements[knob.elements[0]], knob.prop))
            * (1.0 / knob.signs[0] if knob.signs[0] else 1.0)
            for name, knob in self.knobs.items()
        }
        self._setpoints: dict[str, float] = {}
        self._written: set[str] = set()
        # Knobs that still carry a batch, and knobs whose set point has moved
        # since they were last written. `_apply` rewrites exactly these.
        self._batched: set[str] = set()
        self._dirty: set[str] = set()

        self._load_beam(spec, num_particles, coordinates, energy_ev)

    # ---------- construction ----------

    @staticmethod
    def _build(spec: MachineSpec) -> cheetah.Segment:
        """The spec's segment, built by the shared loader the environment uses.

        This used to be a second implementation of the same thing. Two readings
        of one aperture file is one too many: the model the optimiser plans
        against and the machine it is tuning must be the same lattice, and the
        surest way to make them the same is to build both with one function.
        """
        full = lattice_io.build_segment(spec)
        for element in full.elements:
            if isinstance(element, cheetah.Screen):
                # Readings are taken from the particles; an active screen would
                # render a camera frame at every diagnostic for nothing.
                element.is_active = False
        return full

    def _load_beam(self, spec, num_particles, coordinates, energy_ev) -> None:
        if coordinates is None:
            # The spec's distribution file, or its Twiss where it has none — the
            # same loader the environment uses, so a machine with no measured
            # beam works here too.
            coordinates, loaded, _ = lattice_io.load_beam(spec, num_particles)
            energy_ev = energy_ev or loaded
        coordinates = np.asarray(coordinates)[:num_particles]
        self.reference = torch.tensor(np.ascontiguousarray(coordinates), dtype=DTYPE,
                                      device=self.device)
        self.centre = self.reference.mean(dim=0)
        self.energy_ev = float(energy_ev)
        self._energy = torch.tensor(self.energy_ev, dtype=DTYPE, device=self.device)
        self.num_particles = len(coordinates)

    # ---------- knobs ----------

    def index(self, diagnostic: str) -> int:
        return self._row[diagnostic]

    def nominal(self, name: str) -> float:
        return self._nominal[name]

    def value(self, name: str) -> float:
        return self._setpoints.get(name, self._nominal[name])

    def set(self, values: Mapping[str, float]) -> None:
        """Mirror a set of knob values; they persist until changed."""
        for name, value in values.items():
            if name not in self.knobs:
                raise KeyError(f"{name} is not a knob of {self.spec.name}")
            self._setpoints[name] = float(value)
            self._dirty.add(name)

    def reset(self) -> None:
        # Everything written goes back to the lattice file, so everything
        # written is stale until it has been.
        self._dirty |= set(self._setpoints) | self._written
        self._setpoints.clear()

    def _write(self, name: str, value) -> None:
        """Set one knob: every element it drives, with its sign and its faces."""
        knob = self.knobs[name]
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(float(value), dtype=DTYPE, device=self.device)
        if not bool(torch.isfinite(value).all()):
            value = torch.full_like(value, float(self.value(name)))
        # Clipped to what the supply can deliver. A model that will happily set a
        # strength the machine cannot reach is a model that matches to tunes
        # nobody can load, and the error surfaces only when someone tries.
        if knob.low is not None and knob.high is not None:
            value = value.clamp(min=float(knob.low), max=float(knob.high))
        self._written.add(name)
        # A batched write leaves a column behind that the next track must not
        # inherit, so it stays on the rewrite list until it is written scalar.
        if value.ndim:
            self._batched.add(name)
        else:
            self._batched.discard(name)
        for element, sign in zip(knob.elements, knob.signs):
            target = self._elements[element]
            setattr(target, knob.prop, sign * value)
            for extra in knob.follow.get(element, ()):
                setattr(target, extra, sign * value)
            _forget_maps(target)

    def _apply(self, overrides: Mapping[str, torch.Tensor] | None) -> None:
        """Every knob that could be stale to its set point, then the overrides.

        The thing that must not happen is inheriting a previous call's batch: an
        override leaves a column of candidate values on an element, and the next
        track would read it as this track's setting. So every knob last written
        with a batch is rewritten, along with any whose set point has moved since
        it was last written.

        Rewriting *everything* ever written also works and is what this did, but
        it costs a tensor, a finiteness reduction, a clamp and a cache clear per
        knob per track — 1.2 ms a track with the 47 knobs a precompute holds, to
        put back values that were already there.
        """
        overrides = overrides or {}
        stale = self._batched | self._dirty | set(overrides)
        for name in stale - set(overrides):
            self._write(name, self.value(name))
        for name, value in overrides.items():
            self._write(name, value)
        self._dirty = set()

    # ---------- waypoints ----------

    def _first_element(self, names: Iterable[str]) -> int:
        """Index of the most upstream element any of these knobs drives."""
        return min(self._at[element]
                   for name in names for element in self.knobs[name].elements)

    def waypoint(self, downstream_of: Sequence[str]) -> Waypoint:
        """The beam at the last stop before anything these knobs can touch.

        Everything upstream of that stop is unaffected by them, so it is tracked
        once here and reused for every setting of them afterwards.
        """
        first = self._first_element(downstream_of)
        before = [n for n in self.stops if self._at[n] < first]
        if not before:
            raise ValueError(
                f"nothing upstream of {list(downstream_of)} to stop at: these knobs "
                f"start before the first diagnostic, so there is no tail to save"
            )
        return self.track(upto=before[-1], capture=before[-1])

    def _check_resumable(self, resume: Waypoint, overrides) -> None:
        """Refuse to resume when the settings disagree upstream of the waypoint.

        A stale waypoint is a silently wrong answer — the right beam size at the
        readout for a machine nobody asked about — so this is an error and not a
        warning.
        """
        def upstream_of_waypoint(name: str) -> bool:
            return self._first_element([name]) <= resume.element_index

        # An override on an upstream knob is always a disagreement: it is a column
        # of values and the waypoint was taken at one.
        upstream = [n for n in (overrides or ()) if upstream_of_waypoint(n)]
        # Only knobs that have actually moved need checking, not all 109.
        upstream += [
            name for name in (self._written | set(self._setpoints))
            if name not in (overrides or ())
            and upstream_of_waypoint(name)
            and self.value(name) != resume.setpoints[name]
        ]
        if upstream:
            raise ValueError(
                f"waypoint at {resume.stop} cannot be resumed: {sorted(upstream)} "
                f"sit upstream of it and have moved since it was taken"
            )

    # ---------- tracking ----------

    def track(
        self,
        overrides: Mapping[str, torch.Tensor] | None = None,
        batch: int | None = None,
        soft: float | None = None,
        upto: str | None = None,
        beam: torch.Tensor | None = None,
        resume: Waypoint | None = None,
        capture: str | None = None,
    ) -> Readings | Waypoint:
        """Track to `upto` (the readout by default), reading every diagnostic.

        `resume` starts from a `Waypoint` instead of from the front of the line,
        and `capture` stops at that diagnostic and returns a `Waypoint` rather
        than readings. Together they let a loop that only moves magnets near the
        end of the line pay for the end of the line.
        """
        self._apply(overrides)
        last = upto or self.spec.readout or self.stops[-1]
        stop_index = self._stop_row[last]

        minimum = self.spec.min_fraction * self.num_particles
        radius = self.spec.pipe_radius

        if resume is None:
            particles = self.reference if beam is None else beam
            travelling = cheetah.ParticleBeam(
                particles=torch.cat(
                    [particles, torch.ones(len(particles), 1, dtype=DTYPE,
                                           device=self.device)], dim=1
                ),
                energy=self._energy,
                # Said explicitly, because the fields Cheetah fills in for itself
                # — `survival_probabilities` above all — are built without one and
                # land on the CPU, and the first aperture then tries to multiply a
                # CPU mask by a GPU beam.
                device=self.device,
            )
            first = 0
            alive = margin = None
            outside: list[torch.Tensor] = []
            rows: dict[str, tuple] = {}
        else:
            if beam is not None:
                raise ValueError("a waypoint already carries its beam")
            self._check_resumable(resume, overrides)
            travelling = resume.beam
            first = resume.stop_index + 1
            alive, margin = resume.alive, resume.margin
            # Copied, not shared: the loop appends to both and a waypoint is meant
            # to be resumed from many times.
            outside, rows = list(resume.outside), dict(resume.rows)

        live = alive if margin is None else alive * _aperture(margin, soft)
        for name in self.stops[first: stop_index + 1]:
            chunk = self._chunks[name]
            if len(chunk.elements):
                travelling = chunk.track(travelling)
            x, y = travelling.x, travelling.y
            survived = (travelling.survival_probabilities > 0.5).to(DTYPE).expand_as(x)
            alive = survived if alive is None else alive * survived
            if radius is not None:
                # How far inside the pipe each particle is at the worst stop it has
                # passed: a running minimum is exactly the product of the per-stop
                # tests, and unlike that product it is one number that can be
                # softened, which is what gives anything downstream a gradient.
                here = radius - torch.sqrt(x**2 + y**2 + 1e-18)
                margin = here if margin is None else torch.minimum(margin, here)
                outside.append((-here).clamp(min=0.0).mean(dim=-1))
            live = alive if margin is None else alive * _aperture(margin, soft)

            if name in self._row:
                extent = self._extent[name]
                on_chip = live * _aperture(
                    torch.minimum(extent[1] - x.abs(), extent[3] - y.abs()), soft
                )
                count = on_chip.sum(dim=-1)
                mu_x, mu_y = _masked_mean(x, on_chip), _masked_mean(y, on_chip)
                bx, by = _masked_mean(x, live), _masked_mean(y, live)
                rows[name] = (
                    mu_x, mu_y,
                    _masked_std(x, on_chip, mu_x), _masked_std(y, on_chip, mu_y),
                    (count >= minimum).to(DTYPE),
                    bx, by,
                    _masked_std(x, live, bx), _masked_std(y, live, by),
                )

            if name == capture:
                return Waypoint(
                    stop=name, stop_index=self.stops.index(name),
                    element_index=self._at[name],
                    beam=travelling, alive=alive, margin=margin,
                    outside=list(outside), rows=dict(rows),
                    setpoints={n: self.value(n) for n in self.knobs},
                )
        if capture is not None:
            raise ValueError(f"{capture} is not a stop at or before {last}")

        transmission = live.sum(dim=-1) / self.num_particles
        scrape = (
            torch.stack(_broadcast(outside), dim=-1).mean(dim=-1)
            if outside else torch.zeros_like(transmission)
        )
        shape = torch.broadcast_shapes(
            transmission.shape, scrape.shape,
            *(v[0].shape for v in rows.values()),
            *(() if batch is None else ((batch,),)),
        )
        blank = torch.zeros(shape, dtype=DTYPE, device=self.device)
        take = lambda name, i: (  # noqa: E731
            rows[name][i].expand(shape) if name in rows else blank
        )
        centroids = torch.stack(
            [torch.stack([take(n, 0), take(n, 1)], -1) for n in self.diagnostics], -2
        )
        sizes = torch.stack(
            [torch.stack([take(n, 2), take(n, 3)], -1) for n in self.diagnostics], -2
        )
        valid = torch.stack([take(n, 4) for n in self.diagnostics], -1)
        beam_centroids = torch.stack(
            [torch.stack([take(n, 5), take(n, 6)], -1) for n in self.diagnostics], -2
        )
        beam_sizes = torch.stack(
            [torch.stack([take(n, 7), take(n, 8)], -1) for n in self.diagnostics], -2
        )
        return Readings(centroids, sizes, valid, transmission.expand(shape),
                        scrape.expand(shape), beam_centroids, beam_sizes)

    # ---------- batching ----------

    def columns(self, names: Sequence[str], population: np.ndarray) -> dict:
        """A population of knob vectors as a batched override mapping."""
        tensor = torch.tensor(np.asarray(population), dtype=DTYPE,
                              device=self.device)
        return {name: tensor[:, i] for i, name in enumerate(names)}

    def probe(self, names: Sequence[str], steps: Sequence[float], base=None) -> dict:
        """Overrides that evaluate the base point and each knob nudged, in one batch.

        Column 0 is the machine as it stands and column i + 1 is knob i stepped, so
        a response matrix over n knobs costs one track rather than n.
        """
        base = {n: self.value(n) for n in names} if base is None else dict(base)
        population = np.repeat(
            np.array([[base[n] for n in names]], dtype=float), len(names) + 1, axis=0
        )
        for i, step in enumerate(steps):
            population[i + 1, i] += step
        return self.columns(names, population)


def _broadcast(tensors: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    tensors = list(tensors)
    shape = torch.broadcast_shapes(*(t.shape for t in tensors))
    return [t.expand(shape) for t in tensors]
