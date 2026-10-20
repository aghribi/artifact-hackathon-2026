"""Reading a precomputed optics family back, and asking it for a tune.

    family = Family.load("artifacts/clara_febe.npz")
    tune = family.at(sigma_x=2e-4, sigma_y=3e-4)
    tune.k1            # {quadrupole: strength} to load on the machine
    tune.orbit         # where the model says the beam will be at each diagnostic
    tune.R             # d(centroid)/d(corrector), for the steering solve

Stage 2 reads this and nothing else of stage 1: the whole point of splitting the
pipeline is that an episode never re-matches anything. A lookup is microseconds
against the minutes a match costs, which is why the agent can afford to load the
optics before it has taken a single shot.

Interpolation is linear in **log** spot size, because that is how the grid is laid
out and how the optics actually varies — halving a spot is the same kind of change
at 100 µm as at 10 mm. Scan axes interpolate linearly in the knob's own units.

Infeasible nodes are not silently averaged in. A node the precompute could not
match, or could not keep the beam through, carries no usable strengths, and
blending it with its neighbours produces a tune that looks reasonable and is not.
`at()` interpolates over the feasible nodes and tells you, in `confidence`, how
much of the surrounding cell it had.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Below this transmission a node's strengths are not a tune anybody would load,
# however close its spot came. `precompute.HEALTHY`, kept in step.
HEALTHY = 0.5


@dataclass
class Tune:
    """What the family says about one request."""

    # Every quadrupole the tune sets: the matched ones and, in front of them, the
    # standing optics the family was built on. They travel together because they
    # only mean anything together — load the matched four onto a machine whose
    # upstream optics is the lattice's and you get a different beam arriving at
    # the final focus than the one the family solved for.
    k1: dict[str, float]
    # Model centroid at every diagnostic, (n_diagnostics, 2). With dispersion the
    # error-free beam is not on axis, so this — not zero — is what steering aims
    # at. `MFopt.steering` makes the same point.
    orbit: np.ndarray
    # d(centroid)/d(corrector): rows are each diagnostic's x then y, columns are
    # `Family.correctors`.
    R: np.ndarray
    achieved: np.ndarray  # (2,) what the model delivers at these strengths
    transmission: float
    relative_error: float
    # Fraction of the interpolation cell that was feasible. 1.0 is a request the
    # family answers outright; 0.0 means it is extrapolating and the caller should
    # treat the tune as a starting point rather than an answer.
    confidence: float
    clamped: bool  # the request was outside the grid and pulled to its edge


class Family:
    """A precomputed optics family: strengths and responses against the request."""

    def __init__(self, data: dict):
        self.spot_sizes = np.asarray(data["spot_sizes"], dtype=float)
        self.response_spot_sizes = np.asarray(
            data["response_spot_sizes"], dtype=float
        )
        self.axis_names = [str(n) for n in np.atleast_1d(data["axis_names"])
                           if str(n) not in ("", "None")]
        raw_axes = np.atleast_1d(data["axis_values"])
        self.axis_values = [np.asarray(v, dtype=float)
                            for v in raw_axes[:len(self.axis_names)]]
        self.quadrupoles = [str(q) for q in data["quadrupoles"]]
        self.standing_quadrupoles = [
            str(q) for q in np.atleast_1d(data.get("standing_quadrupoles", []))
        ]
        self.standing_k1 = np.asarray(
            data.get("standing_k1", np.zeros(0)), dtype=float
        )
        self.correctors = [str(c) for c in data["correctors"]]
        self.diagnostics = [str(d) for d in data["diagnostics"]]
        self.readout = str(data["readout"])
        self.k1 = np.asarray(data["k1"], dtype=float)
        self.achieved = np.asarray(data["achieved"], dtype=float)
        self.transmission = np.asarray(data["transmission"], dtype=float)
        self.relative_error = np.asarray(data["relative_error"], dtype=float)
        self.feasible = np.asarray(data["feasible"], dtype=bool)
        self.orbit = np.asarray(data["orbit"], dtype=float)
        self.R = np.asarray(data["R"], dtype=float)
        self.provenance = json.loads(str(data["provenance"]))
        if not self.feasible.any():
            raise ValueError(
                "this family has no feasible node: every tune in it either failed "
                "to match or lost the beam. Check the scan-axis ranges — a range "
                "wider than the machine can run is the usual cause."
            )

    @classmethod
    def load(cls, path: str | Path) -> "Family":
        with np.load(Path(path), allow_pickle=True) as data:
            return cls({k: data[k] for k in data.files})

    # ---------- geometry ----------

    def _weights(self, axis: np.ndarray, value: float, log: bool):
        """(low index, high index, weight on high), clamped to the grid."""
        grid = np.log(axis) if log else axis
        want = np.log(value) if log else value
        clamped = not (grid[0] <= want <= grid[-1])
        want = float(np.clip(want, grid[0], grid[-1]))
        high = int(np.clip(np.searchsorted(grid, want), 1, len(grid) - 1))
        low = high - 1
        span = grid[high] - grid[low]
        return low, high, (0.0 if span == 0 else (want - grid[low]) / span), clamped

    def _corners(self, sigma_x: float, sigma_y: float, axes: dict):
        """Every corner of the interpolation cell, with its weight."""
        picks = [
            self._weights(self.spot_sizes, sigma_x, log=True),
            self._weights(self.spot_sizes, sigma_y, log=True),
        ]
        for name, values in zip(self.axis_names, self.axis_values):
            if len(values) == 1:
                picks.append((0, 0, 0.0, False))
            else:
                picks.append(self._weights(values, axes.get(name, values[0]),
                                           log=False))
        clamped = any(p[3] for p in picks)
        corners = [((), 1.0)]
        for low, high, weight, _ in picks:
            corners = [
                (index + (which,), share * part)
                for index, share in corners
                for which, part in ((low, 1.0 - weight), (high, weight))
                if part > 0.0
            ]
        return corners, clamped

    # ---------- lookup ----------

    def at(self, sigma_x: float, sigma_y: float, best_effort: bool = True,
           **axes) -> Tune:
        """The tune for a requested rms spot, interpolated over feasible nodes.

        `best_effort` decides what happens when no node in the cell matched:
        take the closest the precompute got *here* (True), or the nearest node
        that did match, elsewhere (False). Neither is always better, so
        `choose` tries both and keeps whichever scores.
        """
        unknown = set(axes) - set(self.axis_names)
        if unknown:
            raise KeyError(
                f"{sorted(unknown)} is not an axis of this family; it spans "
                f"{self.axis_names or 'the requested spot only'}"
            )
        corners, clamped = self._corners(sigma_x, sigma_y, axes)
        usable = [(i, w) for i, w in corners if self.feasible[i]]
        confidence = sum(w for _, w in usable)
        if not usable:
            # Nothing here matched the request — but the precompute still left
            # the closest it could get, because `match` returns the best point it
            # saw rather than only converged ones. That best effort is an answer
            # to *this* request; a converged tune three columns away in sigma_y
            # is an answer to a different one. Measured on CLARA, roughly a third
            # of what the machine is asked for is not reachable at all (the
            # vertical plane will not go below ~270 um at most compressor
            # angles), so this is the common case and not an edge case.
            #
            # The only thing that disqualifies a node is losing the beam: its
            # strengths are not a tune anybody would load.
            arrived = [(i, w) for i, w in corners
                       if self.transmission[i] > HEALTHY] if best_effort else []
            usable = arrived or [(self._nearest(sigma_x, sigma_y, axes), 1.0)]
            confidence = 0.0
        total = sum(w for _, w in usable)
        blend = lambda field: sum(  # noqa: E731
            w / total * np.asarray(field)[i] for i, w in usable
        )
        rcorners, _ = self._response_corners(sigma_x, sigma_y, axes)
        return Tune(
            k1={**self._standing(corners), **dict(zip(self.quadrupoles, blend(self.k1)))},
            orbit=sum(w * self.orbit[i] for i, w in rcorners).reshape(-1, 2),
            R=sum(w * self.R[i] for i, w in rcorners),
            achieved=blend(self.achieved),
            transmission=float(blend(self.transmission)),
            relative_error=float(blend(self.relative_error)),
            confidence=float(confidence),
            clamped=clamped,
        )

    def choose(self, sigma_x: float, sigma_y: float,
               present: dict[str, float] | None = None) -> tuple[Tune, dict]:
        """The best tune for a request, and the scan-axis setting that gives it.

        A scan axis is a knob, so the family is not only a lookup — it is a
        statement about which setting of that knob the request is reachable at,
        and that is often the single most valuable thing in it. On CLARA the
        smallest vertical spot the final focus can deliver runs from 343 um at
        the compressor angle the lattice ships to 70 um at the bottom of its
        usable band: a factor of five, from one knob, decided before any shot is
        taken. Looking the request up at whatever the machine happens to be set
        to throws that away and asks the family a question it has a worse answer
        to.

        Every combination of axis values is tried — there are a handful, and each
        is an interpolation rather than a track — and the one whose tune is both
        feasible and closest to the request wins. Ties go to the setting nearest
        where the machine already is, since moving a compressor is a real change
        to the machine and not worth making for nothing.
        """
        import itertools

        if not self.axis_names:
            return self.at(sigma_x, sigma_y), {}
        present = present or {}
        best = None
        wanted = np.array([sigma_x, sigma_y])
        for values, effort in itertools.product(
            itertools.product(*self.axis_values), (True, False)
        ):
            axes = dict(zip(self.axis_names, (float(v) for v in values)))
            tune = self.at(sigma_x, sigma_y, best_effort=effort, **axes)
            # Feasibility first, then how close the family thinks it can get,
            # then how far the knob has to move.
            moved = sum(
                abs(v - present.get(n, v)) / (max(a) - min(a) or 1.0)
                for (n, v), a in zip(axes.items(), self.axis_values)
            )
            # Ranked on the benchmark's own measure of size error — mean
            # absolute, in units of the mean requested spot — and not on
            # `relative_error`, which says how well a node matched its *own*
            # target (5e-4 everywhere, and silent about a fallback), nor on the
            # worst-plane log ratio the matcher minimises. Those three disagree:
            # for one 300 x 100 um request the best-effort tune won on log ratio
            # and lost on the benchmark. Only one of them is the score.
            miss = float(np.abs(tune.achieved - wanted).mean() / wanted.mean())
            key = (-tune.confidence, miss, moved)
            if best is None or key < best[0]:
                best = (key, tune, axes)
        return best[1], best[2]

    def _standing(self, corners) -> dict[str, float]:
        """The standing optics at this point of the scan axes.

        It does not depend on the requested spot — it is the machine the family
        was built on, not part of what the family solves — so only the axis part
        of the cell is interpolated over.
        """
        if not self.standing_quadrupoles:
            return {}
        total = blend = 0.0
        for index, weight in corners:
            total += weight
            blend = blend + weight * self.standing_k1[index[2:]]
        return {n: float(v) for n, v in
                zip(self.standing_quadrupoles, np.asarray(blend) / total)}

    def _response_corners(self, sigma_x: float, sigma_y: float, axes: dict):
        """The same cell on the coarser grid the responses were sampled on."""
        saved, self.spot_sizes = self.spot_sizes, self.response_spot_sizes
        try:
            return self._corners(sigma_x, sigma_y, axes)
        finally:
            self.spot_sizes = saved

    def _nearest(self, sigma_x: float, sigma_y: float, axes: dict):
        """Index of the feasible node closest to the request, in log spot size."""
        shape = self.feasible.shape
        grids = np.meshgrid(
            np.log(self.spot_sizes), np.log(self.spot_sizes),
            *[v for v in self.axis_values], indexing="ij",
        )
        want = [np.log(sigma_x), np.log(sigma_y)] + [
            axes.get(n, v[0]) for n, v in zip(self.axis_names, self.axis_values)
        ]
        # Spot sizes are already logarithmic; the axes are scaled by their own
        # span so no one of them dominates the distance by its units.
        spans = [1.0, 1.0] + [
            float(v.max() - v.min()) or 1.0 for v in self.axis_values
        ]
        distance = sum(
            ((g - w) / s) ** 2 for g, w, s in zip(grids, want, spans)
        )
        distance[~self.feasible] = np.inf
        return np.unravel_index(int(distance.argmin()), shape)

    # ---------- reporting ----------

    def summary(self) -> str:
        shape = " x ".join(str(n) for n in self.feasible.shape)
        axes = ", ".join(
            f"{n} in [{v.min():g}, {v.max():g}] ({len(v)})"
            for n, v in zip(self.axis_names, self.axis_values)
        )
        return (
            f"{shape} = {self.feasible.size:,} tunes, "
            f"{self.feasible.sum():,} feasible ({self.feasible.mean():.0%})\n"
            f"  spot {self.spot_sizes[0] * 1e3:g} to "
            f"{self.spot_sizes[-1] * 1e3:g} mm"
            + (f", {axes}" if axes else "")
            + f"\n  {len(self.quadrupoles)} quadrupoles, "
            f"{len(self.correctors)} corrector planes, "
            f"{len(self.diagnostics)} diagnostics"
            f"\n  built {self.provenance.get('built')} in "
            f"{self.provenance.get('seconds')} s from "
            f"{self.provenance.get('particles')} particles"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a precomputed family.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--sigma-x", type=float, default=None)
    parser.add_argument("--sigma-y", type=float, default=None)
    args = parser.parse_args()

    family = Family.load(args.artifact)
    print(f"  {args.artifact}")
    print("  " + family.summary())
    if args.sigma_x and args.sigma_y:
        tune = family.at(args.sigma_x, args.sigma_y)
        print(f"\n  request {args.sigma_x * 1e3:g} x {args.sigma_y * 1e3:g} mm")
        print(f"    model delivers {tune.achieved[0] * 1e3:.4f} x "
              f"{tune.achieved[1] * 1e3:.4f} mm at T = {tune.transmission:.2f}")
        print(f"    confidence {tune.confidence:.2f}"
              + ("  (clamped to the grid edge)" if tune.clamped else ""))
        for name, value in tune.k1.items():
            print(f"      {name:<26} {value:+.4f}")


if __name__ == "__main__":
    main()
