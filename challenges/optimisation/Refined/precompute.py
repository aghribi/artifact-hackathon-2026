"""Stage 1: match an optics family over the grid the spec asks for, once.

    python -I precompute.py machines/clara_febe.json --out artifacts/clara_febe.npz
    python -I precompute.py machines/clara_febe.json --pilot      # a coarse dry run

The expensive half of the pipeline, and the reason the pipeline is split in two. The
family answers "which quadrupole strengths give this spot at the readout?" for every
spot in a grid, and for every value of whatever else the optics turns out to depend
on (`sensitivity.py` decides that; `describe.py` records it). It depends only on the
lattice, so it is computed once and read by every episode afterwards.

**It runs in one process.** Cheetah takes batched element parameters, and the
matching quadrupoles sit at the end of the line, so a thousand candidate tunes cost
the long upstream stretch once and the short tail a thousand times — 0.85 ms per
tune against 93 ms unbatched. See `match.Matcher`. Multiprocessing would buy a
factor of a few on top of a factor of a hundred, at the price of the orphaned
workers this project has twice had to hunt down.

**Coarse to fine.** Each level is matched as one batch and interpolated onto the
next as its starting point, so only the coarsest level pays for finding its basins
and everything after it starts a few iterations from the answer.

**Then a repair pass.** Nodes that stalled are re-seeded from their nearest matched
neighbour and tried again. After the first level most of the grid is solved, so a
stalled node usually has a neighbour sitting in exactly the basin it needed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lattice import Line  # noqa: E402
from machine import KICK, MachineSpec  # noqa: E402
from match import SEED_SAMPLES, MatchResult, Matcher  # noqa: E402

# Spot-grid sizes the solve steps through. Each level is one batched solve warm
# started from the last, so the coarse levels are nearly free and the fine one
# converges in a few iterations.
#
# The first level is the only one that chooses *branches* — `solve_coarse` does
# the continuation, the Sobol rescue and the coherence relaxation there, and
# everything above it inherits those choices through interpolation and only
# refines locally. So it has to be fine enough to find every branch the family
# needs: at 7 x 7 only four nodes land in CLARA's sub-millimetre corner, which
# is not enough to notice that it is reachable at all.
LEVELS = (13, 25)
# Repair rounds: re-seed whatever has not matched from its nearest matched
# neighbour and try again. Two rounds reaches anything a neighbour can reach.
REPAIRS = 2
# Points per plane for the response sub-grid. The responses cost far more than a
# bare match and vary smoothly, so they are sampled coarsely and interpolated, as
# `MFopt.responses` does with its 12 x 12 nodes behind a 49 x 49 family.
RESPONSE_POINTS = 7
# Standing-optics repair, `repair_optics`. CLARA is at 0.995 by the fifth round and
# 0.998 by the twelfth, so there is little past that.
REPAIR_ITERATIONS = 14
REPAIR_POPULATION = 192
# First spread of the search, as a fraction of each quadrupole's own strength.
REPAIR_SPREAD = 0.15
# What drifting from the lattice's tune costs against transmission, so the repair
# stays a repair. At 0.02, a 10 % mean drift is worth a fifth of a percent of beam.
REPAIR_DRIFT = 0.02
# Rounds without improvement before the repair is called done.
REPAIR_PATIENCE = 4
# What counts as improvement rather than the noise of a finite beam.
REPAIR_PROGRESS = 1e-4
# Relaxation passes that pull rescued nodes onto their local branch.
COHERE_PASSES = 6
# Layers the matched region is grown outwards by, from each neighbour.
GROW_ROUNDS = 12


def spot_grid(spec: MachineSpec, points: int | None = None) -> np.ndarray:
    """The requested rms sizes, log-spaced, as `(points, points, 2)`."""
    n = points or spec.spot_points
    axis = np.geomspace(spec.spot_low, spec.spot_high, n)
    return np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)


def _interpolate(coarse: np.ndarray, source: np.ndarray,
                 wanted: np.ndarray) -> np.ndarray:
    """Lift a solved coarse grid onto a finer one, linear in log spot size."""
    from scipy.interpolate import RegularGridInterpolator

    log = np.log(source)
    grid = RegularGridInterpolator(
        (log, log), coarse, bounds_error=False, fill_value=None
    )
    return grid(np.log(wanted).reshape(-1, 2))


def _repair(matcher: Matcher, targets: np.ndarray, result: MatchResult,
            rounds: int = REPAIRS, say=print) -> MatchResult:
    """Re-seed whatever has not matched from its nearest matched neighbour.

    Gauss-Newton is local, so a node fails when its start is in the wrong basin.
    Once most of the grid is solved the right basin is usually sitting next door,
    which a cold Sobol sweep cannot know and a neighbour can.
    """
    log = np.log(targets)
    for round_ in range(1, rounds + 1):
        stuck = np.flatnonzero(~result.converged)
        good = np.flatnonzero(result.converged)
        if not len(stuck) or not len(good):
            return result
        distance = np.abs(log[stuck, None, :] - log[None, good, :]).sum(axis=-1)
        start = result.k1[good[distance.argmin(axis=1)]]
        patched = matcher.match(targets[stuck], start=start)
        better = patched.relative_error < result.relative_error[stuck]
        _assign(result, stuck[better], patched, np.flatnonzero(better))
        say(f"      repair {round_}: {len(stuck)} stalled, "
            f"{int(patched.converged[better].sum())} rescued")
    return result


def _rings(n: int, seed: tuple[int, int]) -> list[np.ndarray]:
    """Grid indices grouped by how far they are from the seed node.

    Ring `r` is everything at Chebyshev distance `r`, so every node in it has a
    neighbour in ring `r - 1` that has already been solved. One ring is one
    batch, which keeps continuation and batching compatible: a 49 x 49 grid is
    25 rings of up to 192 nodes rather than 2401 solves in a row.
    """
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    distance = np.maximum(np.abs(i - seed[0]), np.abs(j - seed[1]))
    return [np.flatnonzero((distance == r).ravel())
            for r in range(distance.max() + 1)]


def _neighbour_start(k1: np.ndarray, solved: np.ndarray, ring: np.ndarray,
                     n: int) -> np.ndarray:
    """Each node in the ring started from the mean of its solved neighbours."""
    start = np.zeros((len(ring), k1.shape[1]))
    for position, flat in enumerate(ring):
        i, j = divmod(int(flat), n)
        picks = [
            k1[(i + di) * n + (j + dj)]
            for di in (-1, 0, 1) for dj in (-1, 0, 1)
            if (di or dj) and 0 <= i + di < n and 0 <= j + dj < n
            and solved[(i + di) * n + (j + dj)]
        ]
        start[position] = np.mean(picks, axis=0) if picks else k1[int(flat)]
    return start


def solve_coarse(matcher: Matcher, spec: MachineSpec, n: int,
                 samples: int = SEED_SAMPLES, say=print) -> MatchResult:
    """The coarsest level, solved outward from one node by continuation.

    This is the part that has to be got right, and the obvious thing is wrong.
    Seeding every node independently — one Sobol sweep, each node taking whichever
    sample came closest — matches more nodes and produces a family that **cannot
    be interpolated**, because neighbouring requests land in unrelated basins.
    Measured on CLARA: adjacent nodes differing by up to 20 m^-2, the entire span
    of the knob, with every node individually correct. Interpolating +9.4 and
    -4.3 gives +2.5, which is not an answer to anything, and the family quietly
    hands out tunes that are nothing like what it promises.

    So one node is seeded and the rest follow it. Each ring starts from the mean
    of its already-solved neighbours, which keeps the whole grid on one branch —
    the branch the machine is already near, because the seed node is the request
    closest to what the present tune delivers.
    """
    targets = spot_grid(spec, n).reshape(-1, 2)
    line = matcher.line
    _, here = line.track().beam_at(line.readout)
    base = np.maximum(here.detach().cpu().numpy().astype(float), 1e-9)
    seed_flat = int(np.abs(np.log(targets / base)).sum(axis=1).argmin())
    seed_ij = divmod(seed_flat, n)

    k1 = np.repeat(matcher.nominal[None], len(targets), axis=0)
    achieved = np.zeros((len(targets), 2))
    transmission = np.zeros(len(targets))
    clipped = np.zeros_like(k1, dtype=bool)
    converged = np.zeros(len(targets), dtype=bool)

    for ring in _rings(n, seed_ij):
        if not len(ring):
            continue
        if not converged.any():
            # The very first node: nothing to continue from, so this is the one
            # place a Sobol sweep is used. It only has to find one basin.
            start = matcher.seed(targets[ring], samples=samples)
        else:
            start = _neighbour_start(k1, converged, ring, n)
        result = matcher.match(targets[ring], start=start)
        k1[ring] = result.k1
        achieved[ring] = result.achieved
        transmission[ring] = result.transmission
        clipped[ring] = result.clipped
        converged[ring] = result.converged

    whole = MatchResult(
        knobs=list(matcher.knobs), targets=targets, k1=k1, achieved=achieved,
        transmission=transmission, clipped=clipped,
        converged=converged,
    )
    say(f"      continuation: {converged.sum()} of {len(targets)} matched")
    whole = _rescue(matcher, targets, whole, samples, say=say)
    whole = _grow(matcher, targets, whole, n, say=say)
    return _cohere(matcher, targets, whole, n, say=say)


def _assign(result: MatchResult, rows, patch: MatchResult, take) -> None:
    """Copy `patch[take]` into `result[rows]`, field by field.

    `rows` and `take` are parallel: node `rows[i]` of the grid takes solve
    `take[i]` of the patch. They are separate because the callers need them to
    be — `_grow` writes one node from whichever of its several attempts won,
    where `_rescue` and `_cohere` write a selection of nodes from the matching
    entries of a same-shaped solve.

    One place that knows which fields a `MatchResult` carries. This was written
    out three times, and adding a field meant finding all three or watching the
    grid quietly fill with a mix of patched and unpatched values.
    """
    for field in ("k1", "achieved", "transmission", "relative_error",
                  "clipped", "converged"):
        getattr(result, field)[rows] = getattr(patch, field)[take]


def _rescue(matcher: Matcher, targets: np.ndarray, result: MatchResult,
            samples: int, say=print) -> MatchResult:
    """Sobol-seed whatever continuation could not reach.

    Continuation keeps the grid on one branch, which is what makes it
    interpolable — and it can only ever extend the branch it started on. On
    CLARA that branch stops at about 0.5 mm vertically, and the whole region
    below it came back unmatched: a third of the grid, and precisely the third
    the machine is actually asked for. A random sweep of the knob box reaches it
    easily (vertical spots down to 0.023 mm), it just has no idea the rest of the
    grid exists.

    So the two are used for what each is good at — continuation for coherence,
    a Sobol sweep for the branches continuation cannot see — and `_cohere`
    afterwards makes the rescued region internally smooth.
    """
    stuck = np.flatnonzero(~result.converged)
    if not len(stuck):
        return result
    patch = matcher.match(targets[stuck], start=matcher.seed(targets[stuck],
                                                             samples=samples))
    _assign(result, stuck[patch.converged], patch,
            np.flatnonzero(patch.converged))
    say(f"      sobol rescue: {int(patch.converged.sum())} of {len(stuck)} reached")
    return result


def _grow(matcher: Matcher, targets: np.ndarray, result: MatchResult, n: int,
          rounds: int = GROW_ROUNDS, say=print) -> MatchResult:
    """Push the matched region outwards one layer at a time, from each neighbour.

    The gap this closes is specific and it is the one that matters. After a Sobol
    rescue the grid is matched in patches, and the boundary between a patch and
    the unmatched region is exactly where a start is available that nothing else
    has tried: a node sitting next to a matched node is one small step from a
    matched tune, even when the branch it needs is nowhere near the one
    continuation began on.

    On CLARA the corner where *both* planes are sub-millimetre is the hard one —
    a random sweep of 20,000 tunes found two that reach it — and it is the corner
    `SimEnv` asks for. Growing into it from the row above, which is matched, gets
    there when neither a cold sweep nor a mean-of-neighbours relaxation does.

    Each neighbour is tried **separately** rather than averaged. Averaging two
    branches gives a start on neither, which is the whole lesson of the
    incoherent first version.
    """
    for round_ in range(1, rounds + 1):
        nodes, starts, owners = [], [], []
        grid = result.k1.reshape(n, n, -1)
        good = result.converged.reshape(n, n)
        for flat in np.flatnonzero(~result.converged):
            i, j = divmod(int(flat), n)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if (di or dj) and 0 <= a < n and 0 <= b < n and good[a, b]:
                        nodes.append(int(flat))
                        starts.append(grid[a, b])
                        owners.append(flat)
        if not nodes:
            break
        patch = matcher.match(targets[np.array(nodes)], start=np.array(starts))
        # Each node keeps the best of the attempts that converged. The attempts
        # are indexed as they are built rather than re-scanned per node, which
        # was an O(nodes x attempts) array rebuild inside the loop.
        attempts: dict[int, list[int]] = {}
        for position, flat in enumerate(owners):
            attempts.setdefault(flat, []).append(position)
        gained = 0
        for flat, tries in attempts.items():
            won = [t for t in tries if patch.converged[t]]
            if not won:
                continue
            pick = min(won, key=lambda t: patch.relative_error[t])
            _assign(result, [flat], patch, [pick])
            gained += 1
        say(f"      grow {round_}: {gained} node(s) reached from a neighbour")
        if not gained:
            break
    return result


def _cohere(matcher: Matcher, targets: np.ndarray, result: MatchResult, n: int,
            passes: int = COHERE_PASSES, say=print) -> MatchResult:
    """Relax the grid towards its own neighbourhood, keeping only improvements.

    After a rescue the grid is right everywhere and smooth only in patches. Each
    pass re-matches every node starting from the mean of its matched neighbours
    and keeps the result when it both converges and sits closer to that mean than
    what was there before. Nodes already agreeing with their neighbours do not
    move; nodes that were a random basin away get pulled onto the local branch.

    It cannot damage the family: a node is only ever replaced by another matched
    solution of the same request.
    """
    for round_ in range(1, passes + 1):
        k1 = result.k1
        neighbourhood = _neighbour_start(
            k1, result.converged, np.arange(len(targets)), n
        )
        before = np.abs(k1 - neighbourhood).max(axis=1)
        patch = matcher.match(targets, start=neighbourhood)
        after = np.abs(patch.k1 - neighbourhood).max(axis=1)
        better = patch.converged & (after < before)
        if not better.any():
            break
        _assign(result, np.flatnonzero(better), patch,
                np.flatnonzero(better))
        say(f"      cohere {round_}: {int(better.sum())} nodes pulled onto "
            f"the local branch")
    return result


def solve_grid(matcher: Matcher, spec: MachineSpec, levels=LEVELS,
               samples: int = SEED_SAMPLES, points: int | None = None,
               say=print) -> MatchResult:
    """Match the whole spot grid at one setting of the scan axes."""
    final = points or spec.spot_points
    ladder = [n for n in levels if n < final] + [final]
    result = None
    previous = None
    for n in ladder:
        axis = np.geomspace(spec.spot_low, spec.spot_high, n)
        flat = spot_grid(spec, n).reshape(-1, 2)
        began = time.perf_counter()
        if previous is None:
            result = solve_coarse(matcher, spec, n, samples, say=say)
        else:
            result = matcher.match(
                flat, start=_interpolate(previous[0], previous[1], flat)
            )
        result = _repair(matcher, flat, result, say=say)
        say(f"    {n:3d} x {n:<3d} = {n * n:5d} nodes   "
            f"{result.converged.sum():5d} matched  "
            f"{result.feasible.sum():5d} feasible  "
            f"{_roughness(result, n):6.2f} rough  "
            f"{time.perf_counter() - began:6.1f} s")
        previous = (result.k1.reshape(n, n, -1), axis)
    return result


def _roughness(result: MatchResult, n: int) -> float:
    """Median jump in k1 between neighbouring nodes, in the knobs' own units.

    The number that says whether the family can be interpolated at all. Printed
    every level because a grid that matches every node and is rough is worse than
    one that misses a few and is smooth: the first lies, the second says so.
    """
    k1 = result.k1.reshape(n, n, -1)
    good = result.converged.reshape(n, n)
    pairs = [
        np.abs(k1[a] - k1[b]).max()
        for a, b in (
            *(((i, j), (i, j + 1)) for i in range(n) for j in range(n - 1)),
            *(((i, j), (i + 1, j)) for i in range(n - 1) for j in range(n)),
        )
        if good[a] and good[b]
    ]
    return float(np.median(pairs)) if pairs else float("nan")


# ---------------------------------------------------------------- standing optics


def repair_optics(line: Line, spec: MachineSpec, iterations: int = REPAIR_ITERATIONS,
                  population: int = REPAIR_POPULATION, start=None, seed: int = 0,
                  say=print) -> dict[str, float]:
    """Fix the standing optics before the family is built on top of it.

    A lattice file records a tune somebody saved, not a tune that works. CLARA's
    ships one that scrapes **26 % of the beam at a single aperture**,
    `cla_s05_sim_aper_01`, before anything is tuned at all — and because the
    benchmark weights transmission at twice the size error, that one aperture is
    worth about 0.5 of score, more than any amount of clever tuning downstream.
    Build a family on top of it and every tune in the family inherits the loss.

    So the quadrupoles the family does *not* match are first set to a tune that
    gets the beam through. `score.TUNABLE` allows every quadrupole, so this is a
    tune anybody could load; it is the optics matching an operator would do once,
    not something the agent spends shots on.

    CEM rather than gradient descent, and that is not a preference. Scraping here
    is Cheetah's own aperture elements, whose `survival_probabilities` is a hard
    step: the straight-through estimator in `lattice._aperture` softens the round
    chamber and the camera chip, not those, so the gradient of transmission with
    respect to an upstream quadrupole is exactly zero. Measured: 120 Adam steps
    moved transmission by 0.005. CEM needs no gradient and the whole population is
    one batched track — 0.741 to 0.998 in about two minutes.
    """
    free = [
        element.name for element in line.segment.elements
        if element.name in line.knobs
        and line.knobs[element.name].prop == "k1"
        and element.name not in spec.matching_quads
    ]
    if not free:
        return {}
    nominal = np.array([line.nominal(n) for n in free])
    low = np.array([line.knobs[n].low if line.knobs[n].low is not None else -np.inf
                    for n in free])
    high = np.array([line.knobs[n].high if line.knobs[n].high is not None else np.inf
                     for n in free])
    rng = np.random.default_rng(seed)
    mean = nominal.copy() if start is None else np.array(
        [start.get(n, line.nominal(n)) for n in free]
    )
    sigma = REPAIR_SPREAD * np.maximum(np.abs(nominal), 0.1)

    best_value, best, stalled = -np.inf, mean.copy(), 0
    for iteration in range(1, iterations + 1):
        candidates = np.clip(rng.normal(mean, sigma, (population, len(free))),
                             low, high)
        candidates[0] = mean  # never lose the incumbent
        readings = line.track(
            overrides=line.columns(free, candidates), batch=population
        )
        transmission = readings.transmission.detach().cpu().numpy().astype(float)
        # Transmission alone would happily walk the whole upstream optics
        # somewhere unrecognisable for a last fraction of a percent. The drift
        # term keeps it a repair of this machine's tune rather than a new one.
        drift = (np.abs(candidates - nominal)
                 / np.maximum(np.abs(nominal), 0.1)).mean(axis=1)
        score = transmission - REPAIR_DRIFT * drift
        order = np.argsort(-score)
        if score[order[0]] > best_value + REPAIR_PROGRESS:
            best_value, best, stalled = score[order[0]], candidates[order[0]], 0
        else:
            stalled += 1
        elite = candidates[order[:max(2, population // 8)]]
        mean, sigma = elite.mean(axis=0), elite.std(axis=0) + 1e-5
        if iteration % 5 == 0 or iteration == iterations:
            say(f"      repair optics {iteration:3d}: "
                f"T {transmission.max():.3f}")
        # Warm-started from the previous axis value this is often done by the
        # fifth round, and every further round is a full population of full-line
        # tracks for nothing.
        if stalled >= REPAIR_PATIENCE:
            say(f"      repair optics {iteration:3d}: "
                f"T {transmission.max():.3f}, settled")
            break
    return dict(zip(free, (float(v) for v in best)))


# ---------------------------------------------------------------- responses


def responses(line: Line, k1: dict[str, float], correctors: list[str]) -> dict:
    """The model's orbit at this optics, and its response to every corrector.

    Both are what steering needs and neither can be had from the family alone.
    The orbit matters because with dispersion the error-free beam is *not* on
    axis, so steering to zero is steering to the wrong place; `MFopt.steering`
    makes the same point and reads the same two things.
    """
    line.set(k1)
    population = np.repeat(
        np.array([[line.value(c) for c in correctors]]), 2 * len(correctors) + 1,
        axis=0,
    )
    for j in range(len(correctors)):
        population[2 * j + 1, j] += KICK
        population[2 * j + 2, j] -= KICK
    batched = line.track(
        overrides=line.columns(correctors, population), batch=len(population)
    )
    # Row 0 of the population is the unkicked point, so the orbit comes off the
    # same batched track rather than a second unbatched one: that duplicate cost
    # a full 127 ms line track at each of the 245 response nodes, 25 s a run.
    centroids = batched.centroids.detach().cpu().numpy().astype(float)  # (B, n_diag, 2)
    plus, minus = centroids[1::2], centroids[2::2]
    # Rows are every diagnostic's x then its y, to match `MFopt.responses`.
    response = ((plus - minus) / (2 * KICK)).transpose(1, 2, 0).reshape(-1, len(correctors))
    return {
        "orbit": centroids[0].ravel(),
        "R": np.nan_to_num(response),
        "valid": batched.valid.detach().cpu().numpy().astype(float)[0],
    }


# ---------------------------------------------------------------- driving


def run(spec: MachineSpec, out: Path, particles: int, samples: int,
        points: int | None, response_points: int, levels=LEVELS,
        repair: bool = True, say=print) -> dict:
    line = Line(spec, num_particles=particles)
    matcher = Matcher(line)
    quads = matcher.knobs
    correctors = [
        k.name for k in spec.knobs
        if k.prop in ("horizontal_angle", "vertical_angle")
    ]
    final = points or spec.spot_points
    axis_values = [a.values() for a in spec.axes]
    axis_names = [a.knob for a in spec.axes]
    combinations = list(itertools.product(*axis_values)) or [()]

    say(f"  {spec.name}: {len(line.segment.elements)} elements, "
        f"{len(spec.knobs)} knobs, readout {spec.readout}")
    say(f"  matching {quads}")
    say(f"  grid {final} x {final}"
        + "".join(f" x {len(v)}" for v in axis_values)
        + f" = {final * final * len(combinations):,} tunes, "
        f"{particles} particles")

    shape = (final, final, *(len(v) for v in axis_values))
    store = {
        "k1": np.zeros((*shape, len(quads))),
        "achieved": np.zeros((*shape, 2)),
        "transmission": np.zeros(shape),
        "relative_error": np.zeros(shape),
        "converged": np.zeros(shape, dtype=bool),
        "feasible": np.zeros(shape, dtype=bool),
        "clipped": np.zeros((*shape, len(quads)), dtype=bool),
    }
    rshape = (response_points, response_points, *(len(v) for v in axis_values))
    orbit = R = None

    began = time.perf_counter()
    standing: dict[str, float] = {}
    standing_grid = None
    for position, values in enumerate(combinations):
        where = np.unravel_index(position, shape[2:]) if axis_names else ()
        line.reset()
        axis_setting = dict(zip(axis_names, values))
        if axis_names:
            say("  " + ", ".join(f"{n} = {v:g}" for n, v in zip(axis_names, values)))
        line.set(axis_setting)
        if repair:
            # Warm-started from the previous axis value: the standing optics moves
            # smoothly along a scan axis, so only the first one pays in full.
            standing = repair_optics(line, spec, start=standing or None, say=say)
            line.set(standing)
        matcher.rebase()
        if standing_grid is None:
            standing_grid = np.zeros((*shape[2:], len(standing))) if standing else None
        if standing_grid is not None:
            standing_grid[where] = [standing[n] for n in sorted(standing)]
        result = solve_grid(matcher, spec, levels, samples, points, say=say)
        index = (slice(None), slice(None), *where)
        for key in ("k1", "achieved", "clipped"):
            store[key][index] = getattr(result, key).reshape(final, final, -1)
        for key in ("transmission", "relative_error", "converged"):
            store[key][index] = getattr(result, key).reshape(final, final)
        store["feasible"][index] = result.feasible.reshape(final, final)

        # Responses on the coarse sub-grid, at the optics this family just matched.
        coarse = spot_grid(spec, response_points).reshape(-1, 2)
        fine = np.geomspace(spec.spot_low, spec.spot_high, final)
        near = _interpolate(result.k1.reshape(final, final, -1), fine, coarse)
        rows = []
        for node in near:
            rows.append(responses(line, dict(zip(quads, node)), correctors))
        if orbit is None:
            orbit = np.zeros((*rshape, rows[0]["orbit"].size))
            R = np.zeros((*rshape, *rows[0]["R"].shape))
        rindex = (slice(None), slice(None), *where)
        orbit[rindex] = np.array([r["orbit"] for r in rows]).reshape(
            response_points, response_points, -1)
        R[rindex] = np.array([r["R"] for r in rows]).reshape(
            response_points, response_points, *rows[0]["R"].shape)
        # `responses` left the matched quadrupoles of the last node written; put
        # the machine back where this axis value says it is before the next one.
        line.reset()
        line.set({**axis_setting, **standing})
        matcher.rebase()
        say(f"      responses at {len(rows)} nodes")

    seconds = time.perf_counter() - began
    provenance = {
        "spec": spec.name,
        "spec_sha1": hashlib.sha1(
            json.dumps(spec.to_dict(), sort_keys=True).encode()
        ).hexdigest(),
        "lattice": spec.lattice,
        "particles": particles,
        "seed_samples": samples,
        "levels": list(levels),
        "repaired_standing_optics": bool(repair),
        "seconds": round(seconds, 1),
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        spot_sizes=np.geomspace(spec.spot_low, spec.spot_high, final),
        response_spot_sizes=np.geomspace(spec.spot_low, spec.spot_high,
                                         response_points),
        axis_names=np.array(axis_names, dtype=object),
        axis_values=np.array(axis_values, dtype=object),
        quadrupoles=np.array(quads),
        standing_quadrupoles=np.array(sorted(standing)),
        standing_k1=(standing_grid if standing_grid is not None
                     else np.zeros((*shape[2:], 0))),
        correctors=np.array(correctors),
        diagnostics=np.array(line.diagnostics),
        readout=np.array(spec.readout),
        orbit=orbit, R=R,
        provenance=np.array(json.dumps(provenance)),
        **store,
    )
    total = store["converged"].size
    say("")
    say(f"  {store['converged'].sum():,} of {total:,} matched, "
        f"{store['feasible'].sum():,} feasible "
        f"({store['feasible'].mean():.0%})")
    say(f"  relative error over feasible: median "
        f"{np.median(store['relative_error'][store['feasible']]):.2e}")
    say(f"  {seconds:.0f} s  ->  {out}")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=4,
                        help="torch threads; 4 is the measured sweet spot")
    parser.add_argument("--device", default="cpu",
                        help="cpu, or cuda if a GPU is worth it. Batched tracking "
                             "is 3-4x faster on even a modest GPU once a batch is "
                             "past about a million particle-rows; below that the "
                             "launch overhead eats the gain.")
    parser.add_argument("--samples", type=int, default=SEED_SAMPLES)
    parser.add_argument("--points", type=int, default=None,
                        help="override the spec's spot_points")
    parser.add_argument("--response-points", type=int, default=RESPONSE_POINTS)
    parser.add_argument("--no-repair", action="store_true",
                        help="build on the lattice's own standing optics, warts "
                             "and all; see repair_optics for what that costs")
    parser.add_argument("--pilot", action="store_true",
                        help="a coarse dry run: 9 x 9 spots, 5 response nodes")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_default_device(args.device)
    spec = MachineSpec.load(args.spec)
    points = 9 if args.pilot else args.points
    response_points = 5 if args.pilot else args.response_points
    out = args.out or Path("artifacts") / (
        args.spec.stem + ("_pilot" if args.pilot else "") + ".npz"
    )
    run(spec, out, args.particles, args.samples, points, response_points,
        repair=not args.no_repair)


if __name__ == "__main__":
    main()
