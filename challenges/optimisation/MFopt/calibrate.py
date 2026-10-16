"""Calibrating the day's beam in the twin, and steering the spot size with it.

Most of what separates the machine's spot from the twin's is the beam the day was
dealt: `SimEnv.reset` scales each plane's rms size (up to x1.3) and mismatches it
(size against divergence, up to x1.2) and scales the energy spread, and with the
twin given that beam its spot matches the machine's within the shot jitter. So
instead of learning a discrepancy over the knobs, this fits those five numbers,

    theta = ln(size_x, mismatch_x, size_y, mismatch_y, energy_spread),

to every reading of the spot, MAP with SimEnv's own spreads as the prior (a
Kennedy-O'Hagan calibration parameter), and re-matches the final focus for the
fitted beam. A reading at any optics informs theta, including the one `reset` gives
for free, so the size is right from the first shot on the target optics and
improves with every shot after it.

The policy (`calibrated`) keeps the approved scheme: the model decides the
quadrupoles, the FEH correctors put the spot where it was asked for linearly.

Only readings and the twin are used: theta is fitted, never read from the episode.
"""

from __future__ import annotations

import numpy as np

from .batched import probe_beams
from .harness import Session
from .optics_env import FINAL_FOCUS, OpticsEnv
from .steering import (
    FINAL_CORRECTORS,
    _clip,
    _corrector_zero,
    steer_orbit,
    table_responses,
    thread,
)

PARAMETERS = ("size_x", "mismatch_x", "size_y", "mismatch_y", "energy_spread",
              "extra_x", "extra_y")
BEAM = 5  # the first five act on the beam; the last two are added after tracking
# SimEnv draws each factor log-uniform in [1/s, s] at variation 1: sd = ln(s)/sqrt(3).
PRIOR_SD = np.append(np.log([1.3, 1.2, 1.3, 1.2, 1.3]) / np.sqrt(3), [0.5, 0.5])
# extra_x, extra_y: what the beam picks up on the machine that the twin's clean
# orbit does not (spurious dispersion from the survey error, mostly), added in
# quadrature: sigma^2 = sigma_twin^2 + extra * sigma_requested^2. Negative values
# down to EXTRA_FLOOR are allowed, for a machine spot smaller than the twin's.
EXTRA_FLOOR = -0.6
# Shot-to-shot jitter of ln(sigma) at the readout (measured in the twin with SimEnv's
# own per-shot spreads): 0.11 to 0.16 in x, 0.07 to 0.12 in y.
NOISE = np.array([0.14, 0.10])
# Readings taken before the orbit is steered see a different path through the
# chromatic optics: they count, with more noise.
UNSTEERED_NOISE = np.array([0.30, 0.30])
STEP = 0.03  # finite-difference step in theta
TRANSMISSION_WEIGHT = 2.0


def beam(nominal: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """The nominal beam with the five factors applied about its centre, as
    `SimEnv._perturb_beam` applies its draws."""
    size_x, mismatch_x, size_y, mismatch_y, spread = np.exp(theta[:BEAM])
    centre = nominal.mean(axis=0)
    drawn = nominal - centre
    drawn[:, 0] *= size_x * mismatch_x
    drawn[:, 1] *= size_x / mismatch_x
    drawn[:, 2] *= size_y * mismatch_y
    drawn[:, 3] *= size_y / mismatch_y
    drawn[:, 5] *= spread
    return drawn + centre


class BeamCalibration:
    """MAP estimate of theta from the spot readings, by Gauss-Newton."""

    def __init__(self, env: OpticsEnv, target: np.ndarray):
        self.env = env
        self.requested = np.asarray(target[2:4], dtype=float)
        self.nominal = env.twin._coordinates.copy()
        self.theta = np.zeros(len(PARAMETERS))
        self.cov = np.diag(PRIOR_SD**2)
        self.readings: list[tuple[dict, np.ndarray, np.ndarray]] = []

    def add(
        self,
        settings: dict[str, float],
        sigma: tuple[float, float],
        noise: np.ndarray = NOISE,
    ) -> None:
        """A reading of (sigma_x, sigma_y) at the readout with the twin's magnets at
        `settings` (on top of the lattice file)."""
        if not np.all(np.isfinite(sigma)) or min(sigma) <= 0:
            return
        self.readings.append((dict(settings), np.log(sigma), np.asarray(noise)))

    def predict(self, settings: dict[str, float], thetas: np.ndarray) -> np.ndarray:
        """ln(sigma_x, sigma_y) and transmission at the readout for each theta row."""
        thetas = np.atleast_2d(thetas)
        beams, index = np.unique(thetas[:, :BEAM], axis=0, return_inverse=True)
        coords = np.stack([beam(self.nominal, t) for t in beams])
        stats, transmission, names = probe_beams(self.env, coords, settings)
        row = stats[:, names.index(self.env.config.target_screen)][index.ravel()]
        variance = row[:, 2:4] ** 2 + thetas[:, BEAM:] * self.requested**2
        sigma = np.sqrt(np.maximum(variance, (0.2 * self.requested) ** 2))
        return np.column_stack([np.log(sigma), transmission[index.ravel()]])

    def twin_request(self) -> tuple[float, float]:
        """The spot to ask the twin for so that the machine gives the request."""
        variance = self.requested**2 * (1 - self.theta[BEAM:])
        sigma = np.sqrt(np.maximum(variance, (0.3 * self.requested) ** 2))
        return float(sigma[0]), float(sigma[1])

    def fit(self, iterations: int = 3) -> np.ndarray:
        if not self.readings:
            return self.theta
        p = len(PARAMETERS)
        prior_inv = np.diag(1 / PRIOR_SD**2)
        theta = self.theta.copy()
        for _ in range(iterations):
            rows, residuals, weights = [], [], []
            for settings, measured, noise in self.readings:
                thetas = np.vstack(
                    [theta, theta + STEP * np.eye(p), theta - STEP * np.eye(p)]
                )
                out = self.predict(settings, thetas)[:, :2]
                if not np.all(np.isfinite(out)):
                    continue
                jac = (out[1 : p + 1] - out[p + 1 :]) / (2 * STEP)  # (p, 2)
                rows.append(jac.T)
                residuals.append(measured - out[0])
                weights.append(1 / noise**2)
            if not rows:
                break
            J = np.vstack(rows)
            r = np.concatenate(residuals)
            W = np.concatenate(weights)
            H = J.T @ (W[:, None] * J) + prior_inv
            g = J.T @ (W * r) - prior_inv @ theta
            delta = np.linalg.solve(H, g)
            theta = np.clip(theta + delta, -3 * PRIOR_SD, 3 * PRIOR_SD)
            theta[BEAM:] = np.maximum(theta[BEAM:], EXTRA_FLOOR)
            self.cov = np.linalg.inv(H)
            if np.abs(delta).max() < 0.005:
                break
        self.theta = theta
        return theta

    def coordinates(self) -> np.ndarray:
        return beam(self.nominal, self.theta)



def aim_kicks(env: OpticsEnv, model: dict, target: np.ndarray) -> dict[str, float]:
    """The FEH corrector settings that put the twin's spot on the requested centroid
    (linear, from the table's response at this optics). They come after the last
    BPM, so they change nothing that steering sees."""
    zero = _corrector_zero(env)
    kicks = np.linalg.pinv(model["J_spot"]) @ (np.asarray(target[:2]) - model["spot"])
    return {name: _clip(env, name, zero[name] + dk)
            for name, dk in zip(FINAL_CORRECTORS, kicks)}


def _model_settings(env: OpticsEnv, magnets: dict[str, float],
                    aim: dict[str, float]) -> dict[str, float]:
    """The machine's focusing magnets on the twin's clean orbit, aimed at the
    request: steering puts the machine's beam on the twin's orbit and the FEH
    correctors put it on the requested centroid, so the twin's correctors are its
    own clean ones plus `aim`."""
    focusing = {n: v for n, v in magnets.items()
                if env._knobs[n].prop in ("k1", "k2")}
    return {**_corrector_zero(env), **aim, **focusing}


def rematch(
    env: OpticsEnv, cal: BeamCalibration, target: np.ndarray, start: dict[str, float],
    aim: dict[str, float],
) -> tuple[dict[str, float], np.ndarray]:
    """The final focus re-matched to the request for the calibrated beam, and the
    calibrated prediction (ln sigma_x, ln sigma_y, T) at it."""
    twin = env.twin
    saved = twin._coordinates
    twin._coordinates = cal.coordinates()
    try:
        k, _ = env._match(cal.twin_request(), {**_corrector_zero(env), **aim, **start})
    finally:
        twin._coordinates = saved
    quads = {n: v for n, v in k.items() if env._knobs[n].prop == "k1"}
    return quads, cal.predict(_model_settings(env, quads, aim), cal.theta)[0]


def _size_mismatch(prediction: np.ndarray, target: np.ndarray) -> float:
    """The size and transmission terms of the benchmark mismatch for a prediction."""
    sigma = np.exp(prediction[:2])
    return float(
        np.abs(sigma - target[2:]).sum() / 2 / target[2:].mean()
        + TRANSMISSION_WEIGHT * (1 - prediction[2])
    )


def calibrated(session: Session, orbit_shots: int = 2, max_shots: int = 6,
               margin: float = 0.01, rematch_above: float = 0.05) -> BeamCalibration:
    """Optics for the calibrated beam with the FEH correctors already aimed at the
    request, SVD orbit steering, then shots that each re-fit the beam, re-match the
    final focus when the predicted size error is worth it, and re-aim the FEH
    correctors on the measured centroid, while the predicted gain is worth a step."""
    from .mfbo import step_cost

    env = session.env
    target = session.target
    model = table_responses(env, target[2], target[3])
    aim = aim_kicks(env, model, target)
    cal = BeamCalibration(env, target)
    first = session.last.info
    if first["on_screen"]:
        # the lattice file's own optics and correctors: aimed nowhere
        cal.add(_model_settings(env, first["magnets"], {}),
                (first["sigma_x"], first["sigma_y"]), UNSTEERED_NOISE)
    cal.fit()
    nodes = env.optics_for_target(target[2], target[3], modes=False)
    quads, _ = rematch(env, cal, target, dict(nodes.k1), aim)
    session.step({"magnets": {**quads, **aim}})
    thread(session, model)
    for _ in range(orbit_shots):
        steer_orbit(session, model)
        info = session.last.info
        if info["on_screen"]:
            cal.add(_model_settings(env, info["magnets"], aim),
                    (info["sigma_x"], info["sigma_y"]))
    for _ in range(max_shots):
        if session.steps_left <= 1:
            break
        info = session.last.info
        if not info["on_screen"]:
            break
        cal.fit()
        now = session.settings()
        current = {n: now[n] for n in quads}
        predicted_now = cal.predict(_model_settings(env, now, aim), cal.theta)[0]
        size_now = _size_mismatch(predicted_now, target)
        new_quads, size_gain = current, 0.0
        if size_now > rematch_above:
            new_quads, predicted_new = rematch(env, cal, target, current, aim)
            size_gain = size_now - _size_mismatch(predicted_new, target)
        position_error = np.array([info["mu_x"], info["mu_y"]]) - target[:2]
        position_gain = float(np.abs(position_error).sum() / 2 / target[2:].mean())
        # A position error is fixed by the FEH correctors to within the jitter.
        gain = size_gain + max(position_gain - 0.03, 0.0)
        if gain < step_cost(session.steps) + margin:
            break
        kicks = -np.linalg.pinv(model["J_spot"]) @ position_error
        action = dict(new_quads) if size_gain > margin else {}
        for name, dk in zip(FINAL_CORRECTORS, kicks):
            action[name] = _clip(env, name, now[name] + dk)
        reading = session.step({"magnets": action})
        if reading.info["on_screen"]:
            cal.add(_model_settings(env, reading.info["magnets"], aim),
                    (reading.info["sigma_x"], reading.info["sigma_y"]))
    return cal


def calibrated_mfbo(session: Session, max_steps: int | None = None):
    """Calibration first (optics for the day's beam in a few steps), then MF-BO
    with the calibrated twin as its low fidelity: it spends steps only where the
    expected improvement is worth them, which is mostly the requests the final
    focus cannot reach within its limits."""
    from .mfbo import SpotBO

    env = session.env
    target = session.target
    cal = calibrated(session)
    model = table_responses(env, target[2], target[3])
    aim = aim_kicks(env, model, target)
    now = session.settings()
    k1 = {n: now[n] for n in env.nodes.quadrupoles}
    saved = env.twin._coordinates
    env.twin._coordinates = cal.coordinates()
    try:
        bo = SpotBO(session, k1, model, aim=aim)
    finally:
        env.twin._coordinates = saved
    bo.run(max_steps)
    return cal, bo


__all__ = ["FINAL_FOCUS", "BeamCalibration", "beam", "calibrated", "calibrated_mfbo",
           "rematch"]
