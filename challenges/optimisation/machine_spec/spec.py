"""What a lattice file does not say, and somebody has to.

A Cheetah lattice says there is a `Dipole` called `cla_vbc_mag_dip_01` with an angle
of 0.1317 rad. It does not say that this dipole is one of four on a single power
supply, that they alternate in sign, that moving the angle has to move the pole faces
with it, that the group is a bunch compressor and therefore a knob worth turning, or
that the four dipoles two sections downstream look identical in the file and are
fixed geometry that must never be touched.

All of that is knowledge about the hardware. `MachineSpec` is where it lives — one
file per machine, written once, read by both stages. `describe.py` drafts it by
inference and leaves the rest as stubs.

    spec = MachineSpec.load("machine.yaml")
    spec.validate(lattice)       # names resolve, limits make sense, no overlaps
    for knob in spec.knobs(lattice):
        ...                      # name, elements, property, signs, limits, linearity
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

# Cheetah element types and the properties of each that are, in principle,
# settable. Whether a given element is *actually* a knob on a given machine is the
# spec's business, not this table's: a dipole may be a bunch compressor or it may be
# the geometry of an arc, and only the user knows which.
SETTABLE = {
    "Quadrupole": ("k1",),
    "Sextupole": ("k2",),
    "Octupole": ("k3",),
    "CombinedCorrector": ("horizontal_angle", "vertical_angle"),
    "HorizontalCorrector": ("angle",),
    "VerticalCorrector": ("angle",),
    "Cavity": ("phase",),
    "Dipole": ("angle",),
    "Solenoid": ("k",),
}

# Properties that steer rather than shape. A corrector adds an orbit: a separable,
# linear, per-element effect with its own stage in the pipeline. It is never part
# of an optics family, and it is never on a shared supply with its neighbours —
# independent control is the entire point of a steerer.
STEERING_PROPERTIES = {"horizontal_angle", "vertical_angle"}

# Properties whose effect on the beam is not a matrix. A linear response and an SVD
# of it can describe everything else; these have to be searched rather than solved,
# and the optimiser keeps them in a separate group for that reason.
NONLINEAR_PROPERTIES = {"k2", "k3", "phase"}

# The suffix each plane of a two-plane corrector gets, so one element can carry two
# independently settable numbers under distinct names.
PLANE_SUFFIX = {"horizontal_angle": "_h", "vertical_angle": "_v"}

# ---- thresholds everything downstream has to agree on ----
#
# Each of these was a separate literal in three or four modules, with comments
# saying "kept in step" — by hand. The one that matters most is `HEALTHY`: the
# threshold deciding which tunes go *into* a precomputed family and the one
# deciding which come back *out* of it have to be the same number, or the lookup
# hands back nodes the precompute rejected and says nothing.

# Below this fraction of the beam arriving, a tune is not one anybody would load,
# however well its spot matched.
HEALTHY = 0.5
# What a beam that never arrives costs, against the error in the spot that did.
# `score.TRANSMISSION_WEIGHT`, which the benchmark charges.
TRANSMISSION_WEIGHT = 2.0
# Finite-difference kick for measuring a corrector's response, in rad. Small
# enough to still be a derivative on a 50 mrad corrector.
KICK = 2e-5


@dataclass
class Knob:
    """One number the optimiser may set, and everything needed to set it.

    A knob is not an element. It may drive several elements at once with different
    signs (a chicane), and setting it may require writing more than one property of
    each (an angle and the pole face that follows it).
    """

    name: str
    prop: str
    elements: list[str]
    signs: list[float] = field(default_factory=list)
    # Extra properties that must be written with the main one, per element:
    # {element: [property, ...]}. Pole-face angles on a chicane live here.
    follow: dict[str, list[str]] = field(default_factory=dict)
    low: float | None = None
    high: float | None = None
    # A natural step, in the knob's own units, used before the optimiser has
    # measured the real response. None lets the optimiser pick from the response.
    step: float | None = None
    nonlinear: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if not self.signs:
            self.signs = [1.0] * len(self.elements)
        if len(self.signs) != len(self.elements):
            raise ValueError(
                f"{self.name}: {len(self.elements)} elements but "
                f"{len(self.signs)} signs"
            )

    @property
    def is_group(self) -> bool:
        return len(self.elements) > 1


@dataclass
class ScanAxis:
    """A direction the precomputed optics family has to span.

    Two of these are always present and implicit: the requested horizontal and
    vertical spot size at the readout. The interesting ones are the rest.

    A matched family answers "what quadrupole strengths give this spot?". That
    question only has one answer if everything *else* about the beam is fixed. It
    often is not. CLARA's L4H linearizer is currently run as a drift, but powered it
    sets the chirp, and therefore the energy spread, and therefore what the dogleg
    turns into horizontal size at the readout — so the right strengths for a 100 µm
    spot would depend on its phase, and a family that did not span the phase would
    be wrong everywhere except the one value it was built at. The same goes for a
    bunch compressor's angle, or a working energy.

    Each axis multiplies the size of the precompute, so they are the user's call
    and nobody should add one speculatively. Declaring none is the common case and
    gives exactly the two-dimensional family `optics_matching` builds today.
    """

    name: str
    # What the axis varies: a knob of this machine ("knob"), or a property of the
    # incoming beam that the caller will supply at run time ("beam").
    kind: str = "knob"
    knob: str | None = None  # which knob, when kind == "knob"
    low: float = 0.0
    high: float = 0.0
    points: int = 5
    spacing: str = "linear"  # "linear" or "log"
    note: str = ""

    def values(self):
        """The grid points along this axis."""
        import numpy as np

        if self.spacing == "log":
            if self.low <= 0:
                raise ValueError(f"{self.name}: log spacing needs a positive low")
            return np.geomspace(self.low, self.high, self.points)
        return np.linspace(self.low, self.high, self.points)


@dataclass
class MachineSpec:
    """Everything about a machine that its lattice file does not record."""

    # ---- what to load ----
    lattice: str
    beam: str | None = None  # a distribution .npz; None means the caller supplies one
    # Twiss for machines with no distribution file: beta/alpha/emittance per
    # plane, energy, and optionally sigma_tau, sigma_p, total_charge. A measured
    # distribution is always better — it carries correlations and tails a Twiss
    # ellipse does not — so this is the fallback, and one worth recording as such.
    beam_twiss: dict[str, float] | None = None
    apertures: str | None = None
    # The stretch that is modelled. None takes the whole line.
    start: str | None = None
    end: str | None = None

    # ---- what to read ----
    readout: str = ""  # the screen the request is about
    bpms: list[str] = field(default_factory=list)
    screens: list[str] = field(default_factory=list)
    # A round beam pipe applied at every diagnostic, in metres. None trusts the
    # lattice's own apertures alone.
    pipe_radius: float | None = None
    # Fraction of the launched beam a diagnostic needs before its reading counts.
    min_fraction: float = 0.10

    # ---- what may be turned ----
    knobs: list[Knob] = field(default_factory=list)
    # Elements that look settable and must never be set: arc geometry, the dipoles
    # that define the beamline rather than tune it.
    fixed: list[str] = field(default_factory=list)
    # Elements that are in the file but switched off on the machine. Replaced by a
    # drift of the same length, which is what being off means for the beam.
    disabled: list[str] = field(default_factory=list)

    # ---- what the precomputed family spans ----
    # The requested spot, always the first two dimensions of the grid.
    spot_low: float = 1e-4   # m
    spot_high: float = 2e-2  # m
    spot_points: int = 49
    # The quadrupoles the family varies to hit a requested spot. Empty means all of
    # them, which is what a first pass should do.
    matching_quads: list[str] = field(default_factory=list)
    # Anything else the optics depends on strongly enough to be worth spanning.
    axes: list[ScanAxis] = field(default_factory=list)

    # ---- bookkeeping ----
    name: str = ""
    notes: list[str] = field(default_factory=list)
    # Stubs `describe.py` could not fill in. A spec with these left in is a draft:
    # `validate` refuses it unless `allow_draft`.
    unresolved: list[str] = field(default_factory=list)

    # ---------- persistence ----------

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["knobs"] = [asdict(k) for k in self.knobs]
        out["axes"] = [asdict(a) for a in self.axes]
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MachineSpec":
        raw = dict(raw)
        raw["knobs"] = [Knob(**k) for k in raw.get("knobs", [])]
        raw["axes"] = [ScanAxis(**a) for a in raw.get("axes", [])]
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"not fields of a machine spec: {sorted(unknown)}")
        return cls(**raw)

    def save(self, path: str | Path) -> None:
        """Written as JSON, which every environment can read without a dependency."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "MachineSpec":
        """Read a spec, resolving the files it points at.

        The paths in a spec are whatever the interview was given, which is
        usually relative to the directory it was run from — not to the spec
        itself. That works until something loads the spec from somewhere else,
        at which point the lattice is simply not found; a plotting script two
        directories away is the obvious way to discover this.

        So each path is tried as written, then beside the spec, then at each
        directory above it in turn, and the first that exists is kept as an
        absolute path. A path that resolves nowhere is left exactly as written,
        so the error it eventually raises names what the spec asked for.
        """
        path = Path(path)
        spec = cls.from_dict(json.loads(path.read_text()))
        for field_name in ("lattice", "beam", "apertures"):
            value = getattr(spec, field_name)
            if not value:
                continue
            roots = [Path.cwd(), *path.resolve().parents]
            for candidate in (Path(value), *(root / value for root in roots)):
                if candidate.exists():
                    setattr(spec, field_name, str(candidate.resolve()))
                    break
        return spec

    # ---------- using it ----------

    def knob(self, name: str) -> Knob:
        for k in self.knobs:
            if k.name == name:
                return k
        raise KeyError(f"{name} is not a knob of {self.name or 'this machine'}")

    def linear_knobs(self) -> list[Knob]:
        """The knobs an SVD of a linear response can speak for."""
        return [k for k in self.knobs if not k.nonlinear]

    def nonlinear_knobs(self) -> list[Knob]:
        """The knobs that have to be searched: sextupoles, RF phases, compressors."""
        return [k for k in self.knobs if k.nonlinear]


    def grid_shape(self) -> tuple[int, ...]:
        """How many matched tunes the precompute will have to compute."""
        return (self.spot_points, self.spot_points, *(a.points for a in self.axes))

    def grid_size(self) -> int:
        total = 1
        for n in self.grid_shape():
            total *= n
        return total

    # ---------- checking it ----------

    def validate(self, element_types: dict[str, str], allow_draft: bool = False):
        """Everything the spec claims, checked against the lattice it claims it of.

        Returns the list of problems; empty means it is usable. Checked here rather
        than discovered at run time because a misnamed magnet in a precompute that
        takes hours should fail in the first second.
        """
        problems: list[str] = []
        if self.unresolved and not allow_draft:
            problems.append(
                f"{len(self.unresolved)} unresolved stub(s) left by describe.py: "
                f"{self.unresolved[:3]}{'...' if len(self.unresolved) > 3 else ''}. "
                f"Fill them in, or pass allow_draft=True to proceed anyway."
            )

        for where, names in (
            ("readout", [self.readout] if self.readout else []),
            ("bpms", self.bpms),
            ("screens", self.screens),
            ("fixed", self.fixed),
        ):
            for name in names:
                if name not in element_types:
                    problems.append(f"{where}: {name} is not in the lattice")
        if not self.readout:
            problems.append("no readout screen named; the request is about something")

        seen: dict[str, str] = {}
        for knob in self.knobs:
            for element in knob.elements:
                kind = element_types.get(element)
                if kind is None:
                    problems.append(f"{knob.name}: {element} is not in the lattice")
                    continue
                if knob.prop not in SETTABLE.get(kind, ()):
                    problems.append(
                        f"{knob.name}: a {kind} has no {knob.prop} to set"
                    )
                if element in self.fixed:
                    problems.append(
                        f"{knob.name}: {element} is also listed as fixed geometry"
                    )
                # Two knobs may share an element only if they set different
                # properties — the two planes of a corrector do exactly that.
                key = f"{element}.{knob.prop}"
                if key in seen:
                    problems.append(
                        f"{knob.name} and {seen[key]} both set {key}"
                    )
                seen[key] = knob.name
            if knob.low is not None and knob.high is not None and knob.low >= knob.high:
                problems.append(f"{knob.name}: empty range [{knob.low}, {knob.high}]")
        for axis in self.axes:
            if axis.kind == "knob":
                if axis.knob is None:
                    problems.append(f"axis {axis.name}: no knob named")
                elif not any(k.name == axis.knob for k in self.knobs):
                    problems.append(f"axis {axis.name}: {axis.knob} is not a knob")
            if axis.points < 2:
                problems.append(f"axis {axis.name}: needs at least 2 points")
            if axis.high <= axis.low:
                problems.append(f"axis {axis.name}: empty range")
            # An axis is a knob the family will be *driven* over, so it has to be
            # a knob the machine can actually reach across that range. Scanning a
            # knob with no limits, or past them, bakes unreachable optics into the
            # artifact and the error only shows up at run time.
            driven = next((k for k in self.knobs if k.name == axis.knob), None)
            if driven is not None:
                if driven.low is None or driven.high is None:
                    problems.append(
                        f"axis {axis.name} scans {axis.knob}, which has no limits — "
                        f"set them, or the scan may leave the machine's range"
                    )
                elif axis.low < driven.low or axis.high > driven.high:
                    problems.append(
                        f"axis {axis.name} scans {axis.knob} over "
                        f"[{axis.low:g}, {axis.high:g}], outside its limits "
                        f"[{driven.low:g}, {driven.high:g}]"
                    )
        for knob in self.knobs:
            if knob.nonlinear != (knob.prop in NONLINEAR_PROPERTIES) and not knob.note:
                problems.append(
                    f"{knob.name}: marked nonlinear={knob.nonlinear} but its property "
                    f"{knob.prop} suggests otherwise — say why in `note` if deliberate"
                )
        return problems


def plane_name(element: str, prop: str) -> str:
    """The conventional knob name for one property of one element."""
    return element + PLANE_SUFFIX.get(prop, "")


def default_knobs(
    element_types: dict[str, str], order: Iterable[str], skip: set[str] = frozenset()
) -> list[Knob]:
    """One independent knob per settable property, in lattice order.

    The honest default before anybody has said anything about the machine: every
    magnet is its own knob, nothing is coupled, nothing is fixed. Dipoles are left
    out, because a dipole is far more often the geometry of the beamline than a thing
    to tune, and the cost of wrongly turning one is a different accelerator.
    """
    knobs: list[Knob] = []
    for element in order:
        kind = element_types.get(element)
        if kind is None or element in skip or kind == "Dipole":
            continue
        for prop in SETTABLE.get(kind, ()):
            knobs.append(
                Knob(
                    name=plane_name(element, prop),
                    prop=prop,
                    elements=[element],
                    nonlinear=prop in NONLINEAR_PROPERTIES,
                )
            )
    return knobs
