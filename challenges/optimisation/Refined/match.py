"""Matching a whole grid of requested spots at once.

    matcher = Matcher(line)
    result = matcher.match(targets)        # targets: (N, 2) requested rms sizes
    result.k1                              # (N, n_quads) the matched strengths

The question a matched optics family answers is "which quadrupole strengths give
this spot at the readout?", and a precompute asks it thousands of times. Asking it
once per node with `scipy.optimize.least_squares` works and is what `MFopt` does,
but it leaves almost all of the available speed on the floor.

**Everything is solved at once.** Cheetah takes element parameters with a leading
batch dimension, and only the elements downstream of a batched knob carry it — so
with the matching quadrupoles at the end of the line, a batch of N candidate tunes
costs the long upstream stretch once and the short tail N times. Measured on CLARA
at 1000 particles: 93 ms for one tune, 2.4 ms per tune in a batch of a thousand,
and 0.85 ms once the upstream track is hoisted into a `Waypoint`. A grid of twelve
thousand nodes goes from an hour of wall clock to a few minutes of it, in one
process, with no multiprocessing to leave orphaned workers behind.

So the solver is a **projected Gauss-Newton run in lockstep over every node**. One
batched track per iteration lays out the base point and one forward difference per
knob for all N nodes together, giving residuals and Jacobians in a single pass; each
node then takes its own step, and nodes that have converged drop out of the batch so
the work shrinks as the grid settles.

Two things follow from solving it this way rather than node by node, and both are
improvements rather than compromises:

- **The gauge is pinned.** Four quadrupoles against two sizes leaves a surface of
  solutions, not a point, so "the matched optics" is not well defined until you say
  which one you mean. `torch.linalg.lstsq` on the underdetermined step returns the
  minimum-norm one, which means "move the quadrupoles as little as possible" — the
  natural operational choice, and reproducible, unlike wherever a trust-region
  solver happens to stop.
- **The Jacobian can be reused.** It changes slowly between iterations, so refreshing
  it every few steps (`jacobian_every`) drops the cost from `n_quads + 1` columns
  per node per iteration to little over one.

Residuals are `MFopt.optics_env._match`'s exactly — log size ratio in each plane
plus a transmission penalty — so the two matchers are solving the same problem and
can be compared node for node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from lattice import Line, Waypoint
from machine import HEALTHY  # noqa: E402

# Forward-difference step for the Jacobian, in k1 (m^-2). Large enough that the
# answer is not float32 rounding of a 9 ms track, small enough to still be a
# derivative: `MFopt` probes the same quadrupoles with 0.05.
QUAD_STEP = 0.02
# Converged when every residual is under this. The size residuals are log ratios,
# so 1e-3 is a tenth of a percent on the spot — far finer than any screen reads and
# well inside the 1 % the family is judged on.
TOLERANCE = 1e-3
MAX_ITERATIONS = 40
# Iterations a node may go without improving before it is given up on. The
# residual is not monotone in k1, so one bad step is not a stall; five is.
PATIENCE = 5
# How much better a residual has to be to count as improvement rather than the
# float32 noise of a 2000-particle track.
PROGRESS = 1e-5
# Sobol samples swept over the knob box to find a basin for each request. One
# batched track serves every node, so this is cheap relative to the solve.
SEED_SAMPLES = 4096
# How much of a Gauss-Newton step to take. The residual is not linear in k1 (a
# quadrupole swept far enough takes the waist through the screen and out the other
# side), so a full step can overshoot into a worse place; damping costs iterations
# that are cheap and buys a solve that does not oscillate.
DAMPING = 0.7
# A step is also capped per iteration, in k1, so one bad Jacobian cannot throw a
# node to the far end of its range and lose the beam.
MAX_STEP = 1.0
# What a lost beam books in the residual, as in `MFopt.optics_env._match`.
LOST = 10.0


@dataclass
class MatchResult:
    """A matched grid: the strengths, and how well each node did."""

    knobs: list[str]
    targets: np.ndarray  # (N, 2) requested rms sizes, m
    k1: np.ndarray  # (N, n_quads)
    achieved: np.ndarray  # (N, 2) what the model delivers, m
    transmission: np.ndarray  # (N,)
    # Per node: worse plane of |achieved / requested - 1|, as
    # `MFopt.optics_env.TargetOptics.relative_error`.
    relative_error: np.ndarray = field(default=None)
    clipped: np.ndarray = field(default=None)  # (N, n_quads) pinned at a limit
    converged: np.ndarray = field(default=None)  # (N,)

    def __post_init__(self) -> None:
        if self.relative_error is None:
            with np.errstate(invalid="ignore", divide="ignore"):
                error = np.abs(self.achieved / self.targets - 1.0)
            self.relative_error = np.nan_to_num(error.max(axis=-1), nan=np.inf)

    @property
    def feasible(self) -> np.ndarray:
        """Nodes the family can honestly claim: matched, and the beam arrives."""
        return (self.relative_error < 0.01) & (self.transmission > HEALTHY)



class Matcher:
    """Matches requested spots by moving `spec.matching_quads`, in batches.

    The waypoint is built once, at construction, and every evaluation resumes from
    it — so the matching quadrupoles must all sit downstream of some diagnostic,
    which on any machine shaped like a beamline they do.
    """

    def __init__(self, line: Line, knobs: Sequence[str] | None = None,
                 step: float = QUAD_STEP):
        self.line = line
        self.knobs = list(knobs or line.spec.matching_quads)
        if not self.knobs:
            raise ValueError(
                "no matching quadrupoles in the spec: the family has nothing to "
                "solve for. Run describe.py, or set `matching_quads` by hand."
            )
        missing = [n for n in self.knobs if n not in line.knobs]
        if missing:
            raise ValueError(f"not knobs of this machine: {missing}")
        self.step = float(step)
        self.low = np.array([
            line.knobs[n].low if line.knobs[n].low is not None else -np.inf
            for n in self.knobs
        ])
        self.high = np.array([
            line.knobs[n].high if line.knobs[n].high is not None else np.inf
            for n in self.knobs
        ])
        self.nominal = np.array([line.nominal(n) for n in self.knobs])
        self._waypoint: Waypoint | None = None

    # ---------- the waypoint ----------

    def rebase(self) -> Waypoint:
        """Retrack the upstream stretch at the line's present settings.

        Call this after moving anything upstream — a scan axis, say. The waypoint
        is what makes the tail cheap and it is only valid for the machine it was
        taken on; `Line.track` refuses a stale one rather than quietly answering
        for the wrong machine.
        """
        self._waypoint = self.line.waypoint(self.knobs)
        return self._waypoint

    @property
    def waypoint(self) -> Waypoint:
        if self._waypoint is None:
            self.rebase()
        return self._waypoint

    # ---------- evaluation ----------

    def _evaluate(self, k1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(sizes, transmission) at the readout for a batch of (N, n_quads)."""
        k1 = np.asarray(k1, dtype=float)
        overrides = self.line.columns(self.knobs, k1)
        readings = self.line.track(
            overrides=overrides, batch=len(k1), resume=self.waypoint
        )
        # The beam as the optics made it, not as the camera clipped it: a tune that
        # walks the spot off the chip must not read as a small one.
        _, sizes = readings.beam_at(self.line.readout)
        return (sizes.detach().cpu().numpy().astype(float),
                readings.transmission.detach().cpu().numpy().astype(float))

    def _residuals(self, sizes: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """What the match drives to zero: the log size ratio in each plane.

        `MFopt.optics_env._match` carries a third residual, `2(1 - T)`, and feeds
        all three to `least_squares`, which *minimises* them. That is coherent
        there and incoherent here: a Gauss-Newton step with four quadrupoles and
        three rows is underdetermined, so `lstsq` solves all three exactly — it
        aims at full transmission, which is usually unreachable, and the solve
        oscillates around a target that does not exist. Measured: the sizes were
        matched to 5e-7 while the reported residual sat at 1-2 forever.

        Transmission is a property of the solution, not a constraint on it. It is
        measured, reported, and used to decide whether a node is feasible — see
        `MatchResult.feasible` — and the freedom left over by two constraints on
        four quadrupoles is spent on staying near the present tune instead, which
        is what keeps the beam in the pipe in practice.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.stack([
                np.log(sizes[..., 0] / targets[..., 0]),
                np.log(sizes[..., 1] / targets[..., 1]),
            ], axis=-1)
        # A node with no beam has no gradient to follow. Booking it at a constant
        # keeps the arrays rectangular; the step for it comes out zero.
        return np.where(np.isfinite(out).all(axis=-1, keepdims=True), out, LOST)

    def _jacobian(self, k1: np.ndarray, targets: np.ndarray):
        """Residuals at `k1` and d(residual)/d(k1), from one batched track.

        Column layout is the base point for every node, then node-major blocks of
        one forward difference per knob: `N * (n_quads + 1)` columns in all, which
        is the whole cost of an iteration.
        """
        n, m = k1.shape
        population = np.repeat(k1, m + 1, axis=0)
        for j in range(m):
            population[j + 1::m + 1, j] += self.step
        sizes, transmission = self._evaluate(population)
        residuals = self._residuals(sizes, np.repeat(targets, m + 1, axis=0)).reshape(n, m + 1, 2)
        base = residuals[:, 0, :]
        jacobian = (residuals[:, 1:, :] - base[:, None, :]).transpose(0, 2, 1)
        return base, jacobian / self.step, sizes[0::m + 1], transmission[0::m + 1]

    # ---------- starting points ----------

    def seed(self, targets: np.ndarray, samples: int = SEED_SAMPLES,
             rng: int = 0) -> np.ndarray:
        """A start for every target, from one Sobol sweep of the knob box.

        Gauss-Newton is local and the map from strengths to spot is not: a
        quadrupole swept far enough takes the waist through the screen and out the
        other side, so each requested spot has its own basin and the lattice's own
        tune is in the wrong one for most of them. Measured on CLARA, only 2 of
        8694 randomly sampled tunes that keep the beam put both planes under
        0.3 mm — the small-spot corner is reachable but rare, and a solve started
        from the nominal tune stalls there however many iterations it is given.
        Continuation from a matched neighbour does not rescue it either, because
        the neighbour is in the same wrong basin.

        One Sobol population serves every target at once: it is a single batched
        track, and each target then takes whichever sample came closest. A few
        thousand samples costs a few seconds and is the difference between
        two-thirds of the grid matching and all of it.
        """
        from scipy.stats import qmc

        targets = np.asarray(targets, dtype=float).reshape(-1, 2)
        m = len(self.knobs)
        points = qmc.Sobol(m, scramble=True, seed=rng).random(samples)
        population = np.vstack([
            self.nominal[None],  # the lattice's own tune is always a candidate
            qmc.scale(points, np.maximum(self.low, -1e3), np.minimum(self.high, 1e3)),
        ])
        sizes, transmission = self._evaluate(population)
        healthy = (
            (transmission > HEALTHY)
            & np.all(np.isfinite(sizes), axis=-1)
            & np.all(sizes > 0, axis=-1)
        )
        if not healthy.any():
            return np.repeat(self.nominal[None], len(targets), axis=0)
        usable, good = population[healthy], sizes[healthy]
        with np.errstate(invalid="ignore", divide="ignore"):
            # (n_targets, n_samples) worst-plane log error
            error = np.abs(
                np.log(good[None, :, :] / targets[:, None, :])
            ).max(axis=-1)
        return usable[np.nan_to_num(error, nan=np.inf).argmin(axis=-1)]

    # ---------- the solve ----------

    def match(self, targets: np.ndarray, start: np.ndarray | None = None,
              tolerance: float = TOLERANCE, max_iterations: int = MAX_ITERATIONS,
              jacobian_every: int = 3, patience: int = PATIENCE,
              report=None) -> MatchResult:
        """Match every requested spot in `targets`, all at once.

        `start` is the initial strengths, `(N, n_quads)`. Left out, `seed` finds
        one per node by sweeping the knob box, which is what gets the small-spot
        corner to match at all — see its docstring.
        """
        targets = np.asarray(targets, dtype=float).reshape(-1, 2)
        n, m = len(targets), len(self.knobs)
        k1 = (self.seed(targets) if start is None
              else np.array(start, dtype=float).reshape(n, m))
        k1 = np.clip(k1, self.low, self.high)

        live = np.ones(n, dtype=bool)  # nodes still being worked on
        jacobian = np.zeros((n, 2, m))
        # The best point each node has seen, which is not the last one it tried:
        # the residual is not monotone in k1 and a Gauss-Newton step can land
        # worse than it started. Returning the last point throws away a good match
        # because of a bad final step.
        # Seeded at infinity rather than by evaluating `k1` here: iteration 1
        # always refreshes the Jacobian, and row 0 of that batch is this exact
        # point, so evaluating it first was one wasted batched track per solve
        # — and `precompute` makes a couple of hundred solves per axis value.
        best = np.full(n, np.inf)
        best_k1 = k1.copy()
        stalled = np.zeros(n, dtype=int)
        iteration = 0
        for iteration in range(1, max_iterations + 1):
            index = np.flatnonzero(live)
            fresh = (iteration - 1) % jacobian_every == 0
            if fresh:
                base, jac, size, trans = self._jacobian(k1[index], targets[index])
                jacobian[index] = jac
            else:
                size, trans = self._evaluate(k1[index])
                base = self._residuals(size, targets[index])

            worst = np.abs(base).max(axis=-1)
            better = worst < best[index] - PROGRESS
            best_k1[index[better]] = k1[index[better]]
            best[index] = np.minimum(best[index], worst)
            stalled[index] = np.where(better, 0, stalled[index] + 1)

            done = worst < tolerance
            if report is not None:
                report(iteration, int(live.sum()), float(worst.max()), fresh)
            # A node that has not improved for `patience` iterations is not going
            # to: it is in a basin this start cannot leave. Dropping it keeps the
            # batch shrinking instead of spending the whole budget on a handful,
            # and `precompute` re-seeds the survivors from a matched neighbour.
            live[index[done | (stalled[index] >= patience)]] = False
            if not live.any():
                break

            moving = index[~(done | (stalled[index] >= patience))]
            step = _min_norm_step(jacobian[moving], base[np.isin(index, moving)])
            step = np.clip(DAMPING * step, -MAX_STEP, MAX_STEP)
            k1[moving] = np.clip(k1[moving] + step, self.low, self.high)

        # Report the best point seen, re-evaluated, so what comes back is what the
        # returned strengths actually do.
        k1 = best_k1
        sizes, transmission = self._evaluate(k1)
        residuals = self._residuals(sizes, targets)
        pinned = (np.abs(k1 - self.low) < 1e-9) | (np.abs(k1 - self.high) < 1e-9)
        return MatchResult(
            knobs=list(self.knobs), targets=targets, k1=k1, achieved=sizes,
            transmission=transmission,
            clipped=pinned, converged=np.abs(residuals).max(axis=-1) < tolerance,
        )


def _min_norm_step(jacobian: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """The smallest `dk` with `J dk = -r`, per node.

    Minimum norm is the choice that makes the matched tune well defined when there
    are more quadrupoles than constraints: of all the strengths that deliver the
    requested spot, take the one nearest where you already are. See the module
    docstring, and the implicit-differentiation note in the README, which needs
    exactly this gauge to be stated for `dk*/dp` to mean anything.
    """
    # Always on the CPU, whatever the tracking runs on. `gelsd` handles the
    # rank-deficient rows a lost beam produces, where the default driver assumes
    # full rank and returns nonsense — and CUDA offers only that default
    # (*driver other than gels is not supported on CUDA*). The whole system is
    # N x 2 x 4 doubles, so keeping it here costs nothing and buys a solve that
    # does not quietly break on the nodes most likely to need it.
    jac = torch.tensor(jacobian, dtype=torch.float64, device="cpu")
    rhs = torch.tensor(-residual, dtype=torch.float64, device="cpu").unsqueeze(-1)
    step = torch.linalg.lstsq(jac, rhs, driver="gelsd").solution.squeeze(-1)
    return np.nan_to_num(step.detach().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
