"""Working out which machine the agent is on, from the shots it has taken.

The twin carries the right lattice but the wrong machine: every `reset()` deals a beam
of its own size, shape and launch angle, and a survey error the agent is stuck with.
Those are `LatentState`, and this module fits them by gradient descent through Cheetah —
the same differentiability that lets the planner ask "which way should this quadrupole
go" lets the estimator ask "which way should this misalignment go".

What makes it affordable is that the whole history fits in one tracking call. Every
shot sat at different magnet settings, and a batched element parameter is exactly a
column of per-shot values, so eight shots at eight different tunes are eight rows of a
single batched track rather than eight tracks. A fit is then a hundred-odd tracks, not
a thousand.

Two things are deliberately not fitted. Shot-to-shot jitter is noise, not state — the
beam is redrawn around the episode's every step — so it sets the residual the fit should
*not* chase, and appears here as the weights in `Residuals`. And the magnet
misalignments are off unless asked for: with only 31 BPMs there is far more survey error
than there is data to pin it down, and a launch offset absorbs most of its effect on the
orbit anyway. `MBRLConfig.fit_misalignments` turns them on as an ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from twin import DTYPE, CheetahTwin, LatentState

# What a reading is worth trusting to, which is the shot-to-shot jitter it carries
# rather than the precision of the number. `beam_jitter` is 0.1 of `beam_variation`,
# which puts the centroid wander at a few microns and the size wander at some 8 %.
BPM_NOISE = 2e-5  # m, how much a BPM reading moves between identical shots
POSITION_NOISE = 1e-5  # m, the same for a centroid on the readout screen
RELATIVE_SIZE_NOISE = 0.09  # of the spot size, the jitter in an rms measurement
SIZE_FLOOR = 2e-5  # m, so a tiny predicted spot cannot make its own residual huge

# What a sigma of unlikely machine costs against a sigma of disagreement with the
# data. Below 1 the fit is free to invent a strange machine to explain shot noise.
PRIOR_WEIGHT = 1.0
# Largest step the optimiser may take, in sigmas of the prior. The beam leaves the
# pipe a few sigmas off axis, and past that every reading is zero and the fit is lost
# in a flat region it cannot climb out of — so it is worth never going there.
MAX_GRAD_NORM = 1.0
# How far the pipe wall and the chip edge are blurred while fitting, in metres.
# Only the gradient sees it; the residuals reported below are measured hard.
SOFT_APERTURE = 1e-3


@dataclass
class Shot:
    """One reading of the machine, and what the magnets were at when it was taken."""

    knobs: dict[tuple[str, str], float]  # (element, property) -> native value
    bpm_x: np.ndarray
    bpm_y: np.ndarray
    bpm_valid: np.ndarray
    moments: np.ndarray | None  # (mu_x, mu_y, sigma_x, sigma_y) in m, or None
    on_screen: bool = False
    # Fraction of the launched bunch that reached the readout. Part of the score
    # in its own right, at `TRANSMISSION_WEIGHT` per unit lost.
    transmission: float = 1.0
    # What the beam was asked for on this shot, and the environment's own account of
    # it. Carried along so the agent can read the request and book the score without
    # keeping a second history in step with this one.
    target: np.ndarray | None = None
    info: dict | None = None


@dataclass
class FitReport:
    """How the fit went, for logging and for deciding whether to trust the twin."""

    loss: float
    orbit_rms: float  # m, residual between predicted and measured orbit
    moment_error: float  # m, mean residual on the screen moments
    shots: int
    iterations: int
    history: list[float] = field(default_factory=list)


class SystemID:
    """Fits a `LatentState` to an episode's shots, warm-started from the last fit."""

    def __init__(self, twin: CheetahTwin, fit_misalignments: bool = False):
        self.twin = twin
        self.fit_misalignments = fit_misalignments
        self.state = self._fresh()

    def _fresh(self) -> LatentState:
        """The machine as the lattice file describes it, before any evidence."""
        return LatentState.nominal(
            n_magnets=len(self.twin.misalignable) if self.fit_misalignments else 0,
            n_diagnostics=(
                len(self.twin.diagnostic_names) if self.fit_misalignments else 0
            ),
        )

    def reset(self) -> None:
        """Forget the last episode: a new `reset()` is a different machine."""
        self.state = self._fresh()

    # ---------- the fit ----------

    def fit(
        self,
        history: list[Shot],
        iterations: int = 80,
        learning_rate: float = 0.02,
    ) -> FitReport:
        """Fit the latent state to `history`, in place, and report the residual."""
        if not history:
            return FitReport(0.0, 0.0, 0.0, 0, 0)

        batched = self._batch_knobs(history)
        measured = self._batch_measurements(history)
        state = self.state.detached().requires_grad_(True)
        optimiser = torch.optim.Adam(state.parameters(), lr=learning_rate)

        trace: list[float] = []
        for _ in range(iterations):
            optimiser.zero_grad(set_to_none=True)
            readings = self.twin.readings(
                state=state, overrides=batched, batch=len(history), soft=SOFT_APERTURE
            )
            loss = self._loss(readings, measured) + self._prior(state)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(state.parameters(), MAX_GRAD_NORM)
            optimiser.step()
            trace.append(float(loss.detach()))

        self.state = state.detached()
        with torch.no_grad():
            readings = self.twin.readings(
                state=self.state, overrides=batched, batch=len(history)
            )
            orbit, moment = self._residuals(readings, measured)
        return FitReport(
            loss=trace[-1] if trace else 0.0,
            orbit_rms=orbit,
            moment_error=moment,
            shots=len(history),
            iterations=iterations,
            history=trace,
        )

    # ---------- turning a history into one batched track ----------

    def _batch_knobs(self, history: list[Shot]) -> dict[tuple[str, str], torch.Tensor]:
        """Every knob any shot moved, as a column of its value in each shot.

        A knob a given shot did not name was wherever it already was, which is what
        `twin.setting` reports, so the column is always full.
        """
        keys = sorted({key for shot in history for key in shot.knobs})
        return {
            key: torch.tensor(
                [shot.knobs.get(key, self.twin.setting(*key)) for shot in history],
                dtype=DTYPE,
            )
            for key in keys
        }

    def _batch_measurements(self, history: list[Shot]) -> dict[str, torch.Tensor]:
        """The readings to fit against, stacked over shots."""
        tensor = lambda rows: torch.tensor(np.array(rows), dtype=DTYPE)  # noqa: E731
        blank = np.zeros(4)
        return {
            "bpm_x": tensor([shot.bpm_x for shot in history]),
            "bpm_y": tensor([shot.bpm_y for shot in history]),
            "bpm_valid": tensor([shot.bpm_valid for shot in history]),
            "moments": tensor(
                [shot.moments if shot.moments is not None else blank for shot in history]
            ),
            "has_moments": tensor(
                [1.0 if (shot.on_screen and shot.moments is not None) else 0.0
                 for shot in history]
            ),
        }

    # ---------- what the fit is minimising ----------

    def _loss(self, readings, measured) -> torch.Tensor:
        """Mismatch between what the twin predicts and what the machine said.

        Each residual is divided by how much that reading moves between identical
        shots, so the fit weighs a BPM against a spot size on the only common scale
        there is — how surprising the disagreement would be. A BPM the machine could
        not read, and a screen with too little beam on it, say nothing and are masked
        out rather than counted as zero.
        """
        valid = measured["bpm_valid"] * readings.bpm_valid
        orbit = (
            ((readings.bpm_x - measured["bpm_x"]) / BPM_NOISE) ** 2
            + ((readings.bpm_y - measured["bpm_y"]) / BPM_NOISE) ** 2
        ) * valid
        loss = orbit.sum() / valid.sum().clamp(min=1.0)

        # Positions are trusted absolutely, sizes in proportion: an rms measurement
        # jitters by a fraction of itself, not by a fixed number of microns.
        gate = measured["has_moments"] * readings.on_screen
        predicted, target = readings.planning_moments(), measured["moments"]
        scale = torch.stack(
            [
                torch.full_like(target[:, 0], POSITION_NOISE),
                torch.full_like(target[:, 1], POSITION_NOISE),
                target[:, 2].abs().clamp(min=SIZE_FLOOR) * RELATIVE_SIZE_NOISE,
                target[:, 3].abs().clamp(min=SIZE_FLOOR) * RELATIVE_SIZE_NOISE,
            ],
            dim=-1,
        )
        moments = (((predicted - target) / scale) ** 2).sum(dim=-1) * gate
        return loss + moments.sum() / gate.sum().clamp(min=1.0)

    def _prior(self, state: LatentState) -> torch.Tensor:
        """How surprising this machine would be, given how SimEnv deals them.

        Without this the fit is wildly underdetermined — 31 BPMs against a launch
        offset, four size factors and, with misalignments on, a hundred-odd survey
        errors, most of which the orbit cannot separate from each other. `LatentState`
        already holds every parameter in units of the spread `sim_env` draws it from,
        so this is just how many sigmas from nominal the fit has wandered, and
        `PRIOR_WEIGHT` says how much a sigma of strangeness is worth against a sigma
        of disagreement with the data.
        """
        return PRIOR_WEIGHT * sum(
            (parameter**2).sum() for parameter in state.parameters()
        )

    def _residuals(self, readings, measured) -> tuple[float, float]:
        """Orbit and moment residuals in metres, for reporting rather than fitting."""
        valid = measured["bpm_valid"] * readings.bpm_valid
        count = valid.sum().clamp(min=1.0)
        orbit = (
            ((readings.bpm_x - measured["bpm_x"]) ** 2
             + (readings.bpm_y - measured["bpm_y"]) ** 2) * valid
        ).sum() / (2 * count)
        gate = measured["has_moments"] * readings.on_screen
        moment = (
            (readings.moments - measured["moments"]).abs().mean(dim=-1) * gate
        ).sum() / gate.sum().clamp(min=1.0)
        return float(torch.sqrt(orbit)), float(moment)
