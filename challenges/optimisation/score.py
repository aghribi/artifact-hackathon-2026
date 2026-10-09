"""Score a full-lattice CLARA tuning function.

The task: the machine is on a **bad set point** — every magnet sits somewhere it has
genuinely sat before, but the combination is not a working optics and usually does not
transport beam to the interaction point at all. You are given a request for the beam at
the IP. Get as close to it as you can.

You write one function::

    def my_tuner(beam, target, lattice) -> cheetah.Segment:
        '''beam: cheetah.ParticleBeam at the lattice entrance.
           target: np.array([mu_x, mu_y, sigma_x, sigma_y]) in metres, requested at
                   the readout screen.
           lattice: cheetah.Segment, the machine starting from the bad set point.
           returns: a configured cheetah.Segment.'''

        # set magnets however you like
        return lattice

and hand it over::

    import score
    report = score.score(my_tuner)
    print(report)

`score()` runs your function against 100 cases, tracks each beam through the lattice
you hand back, measures at `CLA-FEH-DIA-SCR-01` (the FEBE hutch screen, s = 67.93 m,
just ahead of the IP) and reports how close you got.

**You always have something to work from.** A set point is only used if the beam is
still visible on at least the first three diagnostic screens. Past those it is usually
lost, and recovering it is the job. All 28 screens are yours to measure — the score
only looks at the IP, but nothing stops you steering your way down the line.

**Evaluations are counted, not capped.** Every `track()` call on a lattice you got from
`load_lattice()` is counted and reported. Nothing stops you brute-forcing, but a method
that needs 10,000 simulator calls per beam would need 10,000 machine shots, so report
the number honestly — it is the figure that costs beam time.

**Only magnets may change.** The segment you return is checked: same elements in the
same order, and only quadrupole `k1`, corrector angles and sextupole `k2` may differ.
Anything else (drift lengths, cavity voltages, screen positions, misalignments) is a
failed case.

**Baselines to beat.** 
`do_nothing_tuner` hands the bad set point straight back and never delivers beam, which
is the starting position. `reference_tuner` loads the last known-good configuration and
stops — the first thing an operator does, and a decent baseline, though it ignores the
request. `random_search_tuner` loads it and then probes around it.

**Magnet ranges are real.** `magnet_ranges.json` holds what every quadrupole and
corrector actually ran between over six months of CLARA operation (33,159 archiver
readings per PV, 0.5/99.5 percentile after dropping disconnect sentinels, converted
from current with `data/clara/lattice_data/utils.py`). Set points and baseline search
steps are fractions of each magnet's own range. 

**Difficulty.** `severity` scales how far the set point is thrown from the reference,
how much injection jitter the beam carries, and (with `misalign=True`) how large the
survey error is. 
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cheetah
import numpy as np
import torch

HERE = Path(__file__).parent
LATTICE_JSON = HERE / "data" / "clara" / "lattice_data" / "CLARA_cheetah.json"
REFERENCE_TUNE_JSON = HERE / "data" / "clara" / "lattice_data" / "reference_tune.json"
MAGNET_RANGES_JSON = HERE / "data" / "clara" / "lattice_data" / "magnet_ranges.json"

LATTICE_START = "cla_s01_sim_aper_01"  # matches the kickoff notebook

# FEBE hutch screen, s = 67.93 m, just ahead of the IP. 
READOUT_SCREEN = "cla_feh_dia_scr_01"

TUNABLE = {  # what a tuner may change
    cheetah.Quadrupole: ("k1",),
    cheetah.CombinedCorrector: ("horizontal_angle", "vertical_angle"),
    cheetah.Sextupole: ("k2",),
}
TARGET_LABELS = ("mu_x", "mu_y", "sigma_x", "sigma_y")

# A hit is a mean error within this fraction of the target spot size — scale-free, so a
# tight request is not easier than a loose one. Several, because one is 0% or 100%.
HIT_TOLERANCES = (0.50, 0.25, 0.10)
_PENALTY_UM = 1e4  # booked against a case that fails, loses the beam, or errors

MISALIGNMENT_RMS = 50e-6  # m, per plane, quadrupoles and sextupoles, times `severity`
SCREEN_MISALIGNMENT_RMS = 30e-6  # m, the readout screen's own offset

BEAM_PIPE_RADIUS = 50e-3  # m
APERTURE_BEFORE = ("sim_aper", "dia_scr")
MIN_SCREEN_FRACTION = 0.10  # of the launched beam, for a reading to count

REQUESTED_SPOT = (80e-6, 400e-6)  # m, log-uniform — requested sigma_x and sigma_y
# How far off axis a request may sit. Several times the spot size, deliberately: the
# measured beam here is ~300-400 um across, so a smaller request would be inside one
# sigma and steering would not really be part of the task.
REQUESTED_OFFSET = 2e-3  # m
REQUIRED_VISIBLE_SCREENS = 3  # a set point must leave this much signal to work from
RANDOM_SEARCH_REACH = 0.25  # of a magnet's half-span, for `random_search_tuner`

# Plot colours
REQUESTED_COLOUR = "#7c3aed"
MEASURED_COLOUR = "#f97316"
ROI_COLOUR = "#1e3a8a"  # the region-of-interest box on the full-screen view

# Set by `score()` per case so `load_lattice()` hands back the machine being scored,
# however the tuner chooses to get hold of one. Empty means the clean reference machine.
_ACTIVE_MISALIGNMENT: dict = {}
_ACTIVE_SETPOINT: dict = {}
_LAST_SURVIVAL = [1.0]  # on-screen fraction from the most recent `measure`


# --------------------------------------------------------------------------------- #
# lattice
# --------------------------------------------------------------------------------- #


def elements_named(segment, name):
    """All elements with this name. `getattr` gives a list only for repeated names which there should not be."""
    found = getattr(segment, name)
    return found if isinstance(found, list) else [found]


@lru_cache(maxsize=1)
def _lattice_template() -> cheetah.Segment:
    """Parsed once; `load_lattice` clones it. Parsing costs a few seconds."""
    segment = cheetah.Segment.from_lattice_json(str(LATTICE_JSON)).partition_at(
        LATTICE_START, mode="after"
    )[-1]
    apply_tune(segment, reference_tune())
    elements = []
    for element in segment.elements:
        if any(tag in element.name for tag in APERTURE_BEFORE):
            elements.append(
                cheetah.Aperture(
                    x_max=torch.tensor(BEAM_PIPE_RADIUS),
                    y_max=torch.tensor(BEAM_PIPE_RADIUS),
                    shape="elliptical",
                    name=f"{element.name}_pipe",
                    sanitize_name=True,
                )
            )
        elements.append(element)
    return cheetah.Segment(elements)


def load_lattice(count_evaluations: bool = True) -> cheetah.Segment:
    """A fresh lattice. During a case it carries that case's set point and misalignment.

    `track()` calls on it are counted — see `evaluations_of`.
    """
    segment = _lattice_template().clone()
    apply_misalignments(segment, _ACTIVE_MISALIGNMENT)
    apply_tune(segment, _ACTIVE_SETPOINT)
    if count_evaluations:
        _instrument(segment)
    return segment


@lru_cache(maxsize=1)
def reference_tune() -> dict:
    """The last known-good magnet settings, captured from the machine."""
    return json.loads(REFERENCE_TUNE_JSON.read_text())["elements"]


@lru_cache(maxsize=1)
def magnet_ranges() -> dict:
    """{element: {property: [low, high]}} — what each magnet actually runs between."""
    return json.loads(MAGNET_RANGES_JSON.read_text())["elements"]


def _span(name: str, prop: str, fallback: float) -> tuple[float, float, float]:
    """(low, high, half-span) for one knob, from the archiver where we have it."""
    limits = magnet_ranges().get(name, {}).get(prop)
    if limits is None:
        return -fallback, fallback, fallback
    return limits[0], limits[1], (limits[1] - limits[0]) / 2


def apply_tune(segment, tune: dict) -> None:
    """Write a {element: {property: value}} mapping into a segment."""
    for name, properties in tune.items():
        try:
            elements = elements_named(segment, name)
        except AttributeError:
            continue  # downstream of a partition — nothing to set
        for element in elements:
            for prop, value in properties.items():
                setattr(element, prop, torch.tensor(float(value), dtype=torch.float32))


def read_tune(segment) -> dict:
    """Pull the tunable settings back out of a segment."""
    return {
        element.name: {p: float(getattr(element, p)) for p in properties}
        for element in segment.elements
        for kind, properties in TUNABLE.items()
        if isinstance(element, kind)
    }


def apply_misalignments(segment, misalignments: dict) -> None:
    """Write a {element: (dx, dy)} mapping into a segment."""
    for name, (dx, dy) in misalignments.items():
        try:
            elements = elements_named(segment, name)
        except AttributeError:
            continue
        for element in elements:
            element.misalignment = torch.tensor([dx, dy], dtype=torch.float32)


def sample_misalignments(rng, severity: float = 1.0) -> dict:
    """ Generate misalignment offsets for quadrupoles, sextupoles and the readout screen."""
    misalignments = {
        element.name: tuple(rng.normal(0.0, MISALIGNMENT_RMS * severity, 2))
        for element in _lattice_template().elements
        if isinstance(element, (cheetah.Quadrupole, cheetah.Sextupole))
    }
    misalignments[READOUT_SCREEN] = tuple(
        rng.normal(0.0, SCREEN_MISALIGNMENT_RMS * severity, 2)
    )
    return misalignments


def _instrument(segment) -> None:
    """Count `track()` calls on a segment and on anything partitioned out of it."""
    segment._evaluations = [0]

    def wrap(target):
        original = target.track

        def counted(incoming, *args, **kwargs):
            segment._evaluations[0] += 1
            return original(incoming, *args, **kwargs)

        target.track = counted

    wrap(segment)
    original_partition = segment.partition_at

    def partition(*args, **kwargs):
        parts = original_partition(*args, **kwargs)
        for part in parts:
            if isinstance(part, cheetah.Segment):
                part._evaluations = segment._evaluations
                wrap(part)
        return parts

    segment.partition_at = partition


def evaluations_of(segment) -> int:
    """How many times this segment (or a partition of it) has been tracked."""
    return getattr(segment, "_evaluations", [0])[0]


# --------------------------------------------------------------------------------- #
# beams, measurement, cases
# --------------------------------------------------------------------------------- #


def sample_beam(rng, num_particles: int = 10_000, severity: float = 1.0):
    """A starting beam at the lattice entrance, 4.2 MeV out of the gun.

    Randomised around the example beam in section 3 of the kickoff notebook. 

    Particles come from torch's global RNG, so it is seeded from `rng` and restored
    afterwards — without that the benchmark is not reproducible. `severity` widens
    offsets linearly and sizes geometrically (a positive quantity cannot be widened by
    adding), the latter as sqrt(severity) so a large severity stays a harder version of
    this problem rather than a different one.
    """

    def offset(limit):
        return torch.tensor(float(rng.uniform(-limit, limit) * severity))

    def size(low, high):
        middle = np.sqrt(low * high)
        ratio = np.sqrt(high / low) ** np.sqrt(severity)
        return torch.tensor(float(rng.uniform(middle / ratio, middle * ratio)))

    torch_state = torch.get_rng_state()
    torch.manual_seed(int(rng.integers(2**31)))
    try:
        return cheetah.ParticleBeam.from_parameters(
            num_particles=num_particles,
            energy=torch.tensor(float(rng.uniform(4.0e6, 4.4e6))),
            mu_x=offset(5e-5),
            mu_px=offset(1e-5),
            mu_y=offset(5e-5),
            mu_py=offset(1e-5),
            sigma_x=size(0.7e-5, 1.4e-5),
            sigma_px=size(0.7e-6, 1.4e-6),
            sigma_y=size(0.7e-5, 1.4e-5),
            sigma_py=size(0.7e-6, 1.4e-6),
            sigma_tau=torch.tensor(1e-6),
            sigma_p=torch.tensor(1e-4),
            dtype=torch.float32,
        )
    finally:
        torch.set_rng_state(torch_state)


def _imaged(screen, beam):
    """(fraction of the beam on the chip, x, y) relative to the screen centre."""
    x = beam.x.detach() - screen.misalignment[0]
    y = beam.y.detach() - screen.misalignment[1]
    inside = (
        (beam.survival_probabilities.detach() > 0.5)
        & (x.abs() <= float(screen.extent[1]))
        & (y.abs() <= float(screen.extent[3]))
    )
    return float(inside.sum()) / float(beam.num_particles), x[inside], y[inside]


def measure(segment, beam) -> np.ndarray:
    """(mu_x, mu_y, sigma_x, sigma_y) in metres, as the readout camera would see them.

    Moments are over the visible spot only, relative to the screen centre: a clipped
    image gives clipped statistics, and a misaligned screen reports a beam on the design
    axis as off-centre. Below `MIN_SCREEN_FRACTION` there is no reading and the result
    is all-NaN, which `score` books as a lost beam.
    """
    out = segment.partition_at(READOUT_SCREEN, mode="after")[0].track(beam)
    fraction, x, y = _imaged(elements_named(segment, READOUT_SCREEN)[0], out)
    _LAST_SURVIVAL[0] = fraction
    if fraction < MIN_SCREEN_FRACTION:
        return np.full(4, np.nan)
    return np.array([x.mean().item(), y.mean().item(), x.std().item(), y.std().item()])


def plot_target(segment, beam, target, n_sigma=1.0, margin=4.0, pad=0.2,
                min_aspect=0.7):
    """Plot the whole readout camera beside a zoom on the requested spot.

    Left is the full chip with the region of interest outlined; right is that region,
    with the request as a dashed violet ellipse and the measurement as a solid orange
    one, both `n_sigma` and in screen-centre coordinates as in `measure`. The region is
    sized to hold the request at `margin` times its size and the whole visible beam,
    plus `pad` of its own span so nothing touches the frame, and never thinner than
    `min_aspect` of its own length.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Rectangle

    achieved = measure(segment, beam)  # also fills in the screen's reading
    screen = elements_named(segment, READOUT_SCREEN)[0]
    image = screen.reading.detach().numpy()
    chip = [float(v) * 1e3 for v in screen.extent]
    found = bool(np.isfinite(achieved).all())

    xs = np.linspace(chip[0], chip[1], image.shape[1])
    ys = np.linspace(chip[2], chip[3], image.shape[0])

    # A region holding the requested ellipse and the whole visible beam. The beam is
    # bounded from the image rather than from n sigma: these spots have long tails, and
    # a few sigma leaves the streak clipped by the frame.
    low = target[:2] * 1e3 - margin * target[2:] * 1e3
    high = target[:2] * 1e3 + margin * target[2:] * 1e3
    lit = image > 0.02 * image.max() if image.max() > 0 else np.zeros_like(image, bool)
    if lit.any():
        columns, rows = np.where(lit.any(axis=0))[0], np.where(lit.any(axis=1))[0]
        low = np.minimum(low, [xs[columns[0]], ys[rows[0]]])
        high = np.maximum(high, [xs[columns[-1]], ys[rows[-1]]])
    low, high = low - pad * (high - low), high + pad * (high - low)

    # Keep the panel from becoming a sliver: neither side may be shorter than
    # `min_aspect` of the other. Not forced square — when the request and the beam are
    # millimetres apart, squaring that up zooms the spots down to nothing.
    halves = np.maximum((high - low) / 2, min_aspect * (high - low).max() / 2)
    centre = (low + high) / 2
    window = []
    for axis, (lo, hi) in enumerate(((chip[0], chip[1]), (chip[2], chip[3]))):
        half = min(halves[axis], (hi - lo) / 2)
        middle = min(max(centre[axis], lo + half), hi - half)  # slide inside the chip
        window[2 * axis : 2 * axis] = [middle - half, middle + half]
    crop = np.ix_((ys >= window[2]) & (ys <= window[3]),
                  (xs >= window[0]) & (xs <= window[1]))

    figure, (whole, region) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, extent in ((whole, image, chip), (region, image[crop], window)):
        ax.imshow(data, extent=extent, origin="lower", cmap="Blues", aspect="equal")
        ax.set(xlabel="x (mm)", ylabel="y (mm)", xlim=extent[:2], ylim=extent[2:])

    def requested():
        """A fresh patch each time — an artist belongs to one axes."""
        return Ellipse(target[:2] * 1e3, *(2 * n_sigma * target[2:] * 1e3),
                       fill=False, edgecolor=REQUESTED_COLOUR, lw=0.9, ls="--",
                       label=f"requested ({n_sigma:g} sigma)")

    whole.add_patch(Rectangle((window[0], window[2]), window[1] - window[0],
                              window[3] - window[2], fill=False,
                              edgecolor=ROI_COLOUR, lw=0.8, label="ROI"))
    whole.add_patch(requested())
    whole.set_title("Full Screen", fontsize=9)

    region.add_patch(requested())
    if found:
        whole.plot(*(achieved[:2] * 1e3), "+", color=MEASURED_COLOUR, ms=6, mew=0.9)
        region.add_patch(Ellipse(achieved[:2] * 1e3, *(2 * n_sigma * achieved[2:] * 1e3),
                                 fill=False, edgecolor=MEASURED_COLOUR, lw=0.9,
                                 label=f"measured ({n_sigma:g} sigma)"))
        region.plot(*(achieved[:2] * 1e3), "+", color=MEASURED_COLOUR, ms=7, mew=0.9)
        title = (f"sigma ({achieved[2] * 1e6:.0f}, {achieved[3] * 1e6:.0f}) vs requested "
                 f"({target[2] * 1e6:.0f}, {target[3] * 1e6:.0f}) um\n"
                 f"centroid off by "
                 f"{np.linalg.norm(achieved[:2] - target[:2]) * 1e3:.2f} mm")
    else:
        title = "no beam on the screen"
    region.set_title("ROI", fontsize=9)
    region.legend(frameon=False, fontsize=8, loc="best")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    return figure


def visible_screens(segment, beam, limit: int = 8) -> int:
    """How many diagnostic screens in a row, from the start, still show a spot."""
    seen, current = 0, beam
    for element in segment.elements:
        current = element.track(current)
        if not (isinstance(element, cheetah.Screen) and "dia_scr" in element.name):
            continue
        if _imaged(element, current)[0] < MIN_SCREEN_FRACTION:
            break
        seen += 1
        if seen >= limit:
            break
    return seen


def sample_target(rng) -> np.ndarray:
    """What is asked for at the IP: (mu_x, mu_y, sigma_x, sigma_y) in metres."""
    low, high = REQUESTED_SPOT
    return np.array(
        [
            rng.uniform(-REQUESTED_OFFSET, REQUESTED_OFFSET),
            rng.uniform(-REQUESTED_OFFSET, REQUESTED_OFFSET),
            float(np.exp(rng.uniform(np.log(low), np.log(high)))),
            float(np.exp(rng.uniform(np.log(low), np.log(high)))),
        ]
    )


def sample_setpoint(rng, segment, beam, severity: float = 1.0, max_tries: int = 40):
    """A bad set point: every magnet thrown somewhere in its real operating range.

    Not a detuned good solution — a configuration that generally does not transport to
    the end at all, which is what an operator walks in to after a power cycle. The only
    requirement is that the beam still reaches `REQUIRED_VISIBLE_SCREENS` screens, so
    there is something to tune against. Returns None if no draw managed that.
    """
    reference = reference_tune()
    names = sorted(
        {
            e.name
            for e in segment.elements
            if isinstance(e, (cheetah.Quadrupole, cheetah.CombinedCorrector))
        }
    )
    try:
        for _ in range(max_tries):
            setpoint = {}
            for name in names:
                setpoint[name] = {}
                for prop, value in reference.get(name, {}).items():
                    low, high, half = _span(name, prop, 3.0 if prop == "k1" else 5e-4)
                    setpoint[name][prop] = float(
                        np.clip(value + rng.normal(0.0, severity * half), low, high)
                    )
            apply_tune(segment, setpoint)
            if visible_screens(segment, beam) >= REQUIRED_VISIBLE_SCREENS:
                return setpoint
        return None
    finally:
        apply_tune(segment, reference)


@lru_cache(maxsize=4)
def make_cases(
    n_cases: int = 100,
    seed: int = 0,
    num_particles: int = 10_000,
    misalign: bool = False,
    severity: float = 1.0,
):
    """The benchmark: n_cases (beam, target, misalignments, setpoint) tuples.

    Each case draws from its own `default_rng([seed, index, attempt])`, so case `i` does
    not depend on the ones before it. The set is cached under `cases/` and reused —
    generating it is the slow part, and a benchmark wants a fixed test set. Delete that
    directory to rebuild.
    """
    cache = HERE / "cases" / (
        f"n{n_cases}_s{seed}_p{num_particles}"
        f"_{'mis' if misalign else 'aligned'}_sev{severity:g}.pt"
    )
    if cache.exists():
        return tuple(
            (cheetah.ParticleBeam(particles=p, energy=e), np.asarray(t),
             {k: tuple(v) for k, v in m.items()}, s)
            for p, e, t, m, s in torch.load(cache, weights_only=False)
        )

    max_attempts = 12
    cases = []
    for index in range(n_cases):
        for attempt in range(max_attempts):
            rng = np.random.default_rng([seed, index, attempt])
            misalignments = sample_misalignments(rng, severity) if misalign else {}
            scratch = load_lattice(count_evaluations=False)
            apply_misalignments(scratch, misalignments)
            beam = sample_beam(rng, num_particles, severity)
            setpoint = sample_setpoint(rng, scratch, beam, severity)
            if setpoint is not None:
                cases.append((beam, sample_target(rng), misalignments, setpoint))
                break
        else:
            raise RuntimeError(
                f"case {index}: no set point in {max_attempts} attempts left the beam "
                f"visible on {REQUIRED_VISIBLE_SCREENS} screens. Lower `severity`."
            )
    cache.parent.mkdir(exist_ok=True)
    torch.save([(b.particles, b.energy, np.asarray(t), m, s) for b, t, m, s in cases], cache)
    return tuple(cases)


# --------------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------------- #


def _structure(segment) -> list[str]:
    return [f"{type(e).__name__}:{e.name}" for e in segment.elements]


def _fixed_features(segment) -> dict:
    """Everything a tuner is *not* allowed to touch, as rounded floats."""
    fixed = {}
    for index, element in enumerate(segment.elements):
        allowed = next(
            (props for kind, props in TUNABLE.items() if isinstance(element, kind)), ()
        )
        for feature in element.defining_features:
            value = getattr(element, feature, None)
            if feature not in allowed and isinstance(value, torch.Tensor):
                fixed[index, feature] = tuple(
                    np.round(value.detach().flatten().numpy().astype(float), 9)
                )
    return fixed


def check_segment(returned, structure, features) -> str | None:
    """None if the returned segment is a legal retune of the reference, else why not.

    `structure` and `features` describe the machine the tuner was handed, from
    `_structure` and `_fixed_features`.
    """
    if not isinstance(returned, cheetah.Segment):
        return f"expected a cheetah.Segment, got {type(returned).__name__}"
    if _structure(returned) != structure:
        return "lattice structure changed (elements added, removed, renamed or reordered)"
    expected, actual = features, _fixed_features(returned)
    changed = [
        f"element {i} {returned.elements[i].name}.{feature}"
        for (i, feature), value in expected.items()
        if actual.get((i, feature)) != value
    ]
    if not changed:
        return None
    more = f" (+{len(changed) - 3} more)" if len(changed) > 3 else ""
    return f"non-tunable properties changed: {', '.join(changed[:3])}{more}"


# --------------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------------- #


@dataclass
class CaseResult:
    index: int
    target: np.ndarray
    achieved: np.ndarray | None
    error_um: float
    relative_error: float
    evaluations: int
    seconds: float
    survival: float = 1.0
    failure: str | None = None

    def hit(self, tolerance: float) -> bool:
        return self.failure is None and self.relative_error <= tolerance


@dataclass
class Report:
    name: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def errors_um(self) -> np.ndarray:
        return np.array([c.error_um for c in self.cases])

    @property
    def mean_error_um(self) -> float:
        return float(self.errors_um.mean())

    @property
    def median_error_um(self) -> float:
        return float(np.median(self.errors_um))

    @property
    def worst_error_um(self) -> float:
        return float(self.errors_um.max())

    @property
    def failures(self) -> list[CaseResult]:
        return [c for c in self.cases if c.failure is not None]

    @property
    def mean_survival(self) -> float:
        kept = [c.survival for c in self.cases if np.isfinite(c.survival)]
        return float(np.mean(kept)) if kept else float("nan")

    @property
    def evaluations_per_case(self) -> float:
        return float(np.mean([c.evaluations for c in self.cases]))

    @property
    def seconds(self) -> float:
        return sum(c.seconds for c in self.cases)

    def hit_rate(self, tolerance: float) -> float:
        return float(np.mean([c.hit(tolerance) for c in self.cases]))

    def per_component_um(self) -> dict:
        rows = [np.abs(c.achieved - c.target) for c in self.cases if c.achieved is not None]
        if not rows:
            return dict.fromkeys(TARGET_LABELS, float("nan"))
        return dict(zip(TARGET_LABELS, np.array(rows).mean(axis=0) * 1e6))

    def __str__(self) -> str:
        lines = [
            f"{self.name}  —  {len(self.cases)} cases at {READOUT_SCREEN}",
            f"  mean error       {self.mean_error_um:10.2f} um",
            f"  median error     {self.median_error_um:10.2f} um",
            f"  worst error      {self.worst_error_um:10.2f} um",
            "  hit rate (mean error within x% of the requested spot size):",
            *(f"      within {t:.0%} {self.hit_rate(t):10.1%}" for t in HIT_TOLERANCES),
            f"  evaluations      {self.evaluations_per_case:10.1f} per case",
            f"  beam on screen   {self.mean_survival:10.1%} of particles imaged",
            f"  failed cases     {len(self.failures):10d}",
            f"  wall clock       {self.seconds:10.1f} s",
            "  per component (mean |error|, um):",
            *(f"      {k:9s} {v:10.2f}" for k, v in self.per_component_um().items()),
        ]
        if self.failures:
            grouped: dict[str, list[int]] = {}
            for case in self.failures:
                grouped.setdefault(case.failure, []).append(case.index)
            lines.append("  failures:")
            for reason, indices in grouped.items():
                more = f" (+{len(indices) - 5} more)" if len(indices) > 5 else ""
                shown = ", ".join(str(i) for i in indices[:5])
                lines.append(f"      [{shown}{more}] {reason}")
        return "\n".join(lines)


def score(
    tune,
    n_cases: int = 100,
    seed: int = 0,
    num_particles: int = 10_000,
    misalign: bool = False,
    severity: float = 1.0,
    name: str | None = None,
    progress: bool = True,
) -> Report:
    """Score a tuning function against the benchmark.

    `tune` is called once per case as `tune(beam, target)`, or as
    `tune(beam, target, lattice)` if it takes a third argument — in which case it is
    handed a fresh, evaluation-counting lattice rather than having to call
    `load_lattice()` itself. Either way the lattice it gets is the case's machine: the
    bad set point, plus survey error if `misalign` is on. It must return a configured
    `cheetah.Segment`.

    `target` is `np.array([mu_x, mu_y, sigma_x, sigma_y])` in **metres**, requested at
    `READOUT_SCREEN`. Nothing guarantees it is reachable from the set point given — the
    task is to get as close as the machine allows.

    `severity` scales how far the set point is thrown from the known-good tune, how much
    jitter the incoming beam carries, and how large the survey error is. With
    `misalign=True` the quadrupoles and sextupoles are displaced and the readout screen
    picks up an offset of its own, so "centred" means centred on the camera rather than
    on the design axis. A tuner cannot undo misalignment: it is not a tunable property,
    and a segment that changes one fails validation.

    A case scores the error between what was asked for and what the camera sees. If too
    little of the beam lands on the screen there is no reading and the case is booked as
    a lost beam at the penalty value. Returns a `Report`; print it for a summary.
    """
    global _ACTIVE_MISALIGNMENT, _ACTIVE_SETPOINT
    report = Report(name=name or getattr(tune, "__name__", "tuner"))
    wants_lattice = len(inspect.signature(tune).parameters) >= 3

    try:
        for index, (beam, target, misalignments, setpoint) in enumerate(
            make_cases(n_cases, seed, num_particles, misalign, severity)
        ):
            _ACTIVE_MISALIGNMENT, _ACTIVE_SETPOINT = misalignments, setpoint
            # Validation compares against this case's own machine, survey error and
            # all, or everything would fail on "misalignment changed".
            expected = load_lattice(count_evaluations=False)
            structure, features = _structure(expected), _fixed_features(expected)
            lattice = load_lattice()
            started = time.time()
            failure, achieved, survival = None, None, float("nan")
            try:
                returned = (
                    tune(beam, target, lattice) if wants_lattice else tune(beam, target)
                )
            except Exception as exc:  # a crashing tuner fails that case, not the run
                failure, returned = f"{type(exc).__name__}: {exc}", None

            # Count what the tuner spent, before the scorer adds its own measurement.
            evaluations = evaluations_of(lattice) + (
                evaluations_of(returned)
                if isinstance(returned, cheetah.Segment) and returned is not lattice
                else 0
            )
            if failure is None:
                failure = check_segment(returned, structure, features)
            if failure is None:
                achieved = measure(returned, beam)
                survival = _LAST_SURVIVAL[0]
                if not np.isfinite(achieved).all():
                    failure, achieved = (
                        f"beam not on the screen ({survival:.0%} of the beam imaged)",
                        None,
                    )
            if failure is None:
                error_um = float(np.abs(achieved - target).mean() * 1e6)
                relative = error_um / float(target[2:].mean() * 1e6)
            else:
                error_um = relative = _PENALTY_UM

            report.cases.append(
                CaseResult(index, target, achieved, error_um, relative, evaluations,
                           time.time() - started, survival, failure)
            )
            _ACTIVE_MISALIGNMENT, _ACTIVE_SETPOINT = {}, {}
            if progress:
                case = report.cases[-1]
                status = case.failure or f"{case.error_um:8.2f} um ({case.relative_error:5.1%})"
                print(f"  [{index + 1:3d}/{n_cases}] {status}", flush=True)
    finally:
        _ACTIVE_MISALIGNMENT, _ACTIVE_SETPOINT = {}, {}
    return report


# --------------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------------- #


def do_nothing_tuner(beam, target, lattice):
    """Hand the bad set point straight back. The floor, and usually no beam at all."""
    return lattice


def reference_tuner(beam, target, lattice):
    """Load the last known-good configuration. What an operator does first."""
    apply_tune(lattice, reference_tune())
    return lattice


def random_search_tuner(beam, target, lattice, n_calls: int = 40, seed: int = 0):
    """Load the known-good tune, then probe around it at random and keep the best.

    Only the 20-odd magnets in S07 and the FEA arc, and steps are a fraction of each
    magnet's archiver range rather than of its present value — most quadrupoles sit near
    zero in the reference tune, so a relative step would barely move them.
    """
    rng = np.random.default_rng(seed)
    apply_tune(lattice, reference_tune())
    names = sorted(
        {
            e.name
            for e in lattice.partition_at(READOUT_SCREEN, mode="after")[0].elements
            if isinstance(e, (cheetah.Quadrupole, cheetah.CombinedCorrector))
            and e.name.split("_")[1] in ("s07", "fea")
        }
    )
    start = {n: read_tune(lattice)[n] for n in names}
    best_error, best = np.inf, start
    for _ in range(n_calls):
        trial = {}
        for name, properties in start.items():
            trial[name] = {}
            for prop, current in properties.items():
                low, high, half = _span(name, prop, 3.0 if prop == "k1" else 5e-4)
                step = RANDOM_SEARCH_REACH * half
                trial[name][prop] = float(
                    np.clip(rng.uniform(current - step, current + step), low, high)
                )
        apply_tune(lattice, trial)
        moments = measure(lattice, beam)
        error = np.abs(moments - target).mean() if np.isfinite(moments).all() else np.inf
        if error < best_error:
            best_error, best = error, trial
    apply_tune(lattice, best)
    return lattice
