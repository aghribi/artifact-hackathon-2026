"""Multi-fidelity Bayesian optimisation of the readout, after the orbit is set.

Imported only by `methods.bo`, and only when it is asked for: botorch is not a
dependency of anything else here and is not installed by default.

Two fidelities, following `MFopt/mfbo.py`:

- **Low: the twin.** Free, so it is sampled densely over the knob box and a GP
  per readout is fitted to it — `f_L(x)`.
- **High: the machine.** Modelled as `f_H(x) = f_L(x) + delta(x)`, with an
  independent GP on the discrepancy (Kennedy-O'Hagan with rho = 1). After one
  shot `delta` is already a constant offset, so the first proposal is the twin's
  own answer shifted by what today's beam does differently — which is the same
  anchoring `methods.mpc` does by subtraction, learnt instead of assumed, and it
  keeps improving as shots accumulate where a constant cannot.

The objective is the benchmark mismatch computed from posterior samples of the
readouts and the transmission — a composite objective, so the GPs model physical
quantities that vary smoothly rather than a score with a clamp in it. The
acquisition is expected improvement, maximised over a dense candidate set, and
the loop stops when the expected improvement is worth less than what one more
shot costs in the score.
"""

from __future__ import annotations

import numpy as np

from machine import TRANSMISSION_WEIGHT  # noqa: E402
import torch

# The GPs run in double precision, set per tensor. The tracking in the same
# process is float32 and setting torch's global default to float64 breaks it —
# which is exactly how `MFopt/mfbo.py` failed the first time it was run.
DTYPE = torch.float64
# Twin samples the low-fidelity GP is fitted to.
MODEL_SAMPLES = 256
# Candidates the acquisition is maximised over.
CANDIDATES = 4096
# Posterior samples the composite objective is averaged over.
POSTERIOR_SAMPLES = 128
# How far around the present setting the search box reaches, as a fraction of
# each knob's usable span.
REACH = 0.3


def _fit(x: torch.Tensor, y: torch.Tensor):
    """One GP per output column, standardised."""
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    model = SingleTaskGP(x, y, outcome_transform=Standardize(m=y.shape[-1]))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def _mismatch(samples: torch.Tensor, target: np.ndarray) -> torch.Tensor:
    """The benchmark's mismatch from posterior samples of (sigma_x, sigma_y, T)."""
    scale = float(np.mean(target[2:]))
    wanted = torch.tensor(target[2:], dtype=DTYPE)
    size_error = (samples[..., :2] - wanted).abs() / scale
    missing = TRANSMISSION_WEIGHT * (1.0 - samples[..., 2].clamp(max=1.0))
    return size_error.mean(dim=-1) + missing


def multi_fidelity(session, knobs: list[str], shots: int) -> None:
    """Propose, measure, refit, until one more shot is not worth its cost."""
    from scipy.stats import qmc

    if not knobs:
        session.say("bo: no knob to search")
        return
    session.stage = "bayes"
    machine = session.machine
    target = machine.target
    twin = session.model()

    present = np.array([
        machine.last.settings.get(n, twin.value(n)) for n in knobs
    ])
    low = np.array([machine.limits[n][0] for n in knobs])
    high = np.array([machine.limits[n][1] for n in knobs])
    reach = REACH * (high - low)
    box_low = np.maximum(low, present - reach)
    box_high = np.minimum(high, present + reach)
    span = np.where(box_high > box_low, box_high - box_low, 1.0)

    to_unit = lambda v: (v - box_low) / span  # noqa: E731
    to_native = lambda u: box_low + u * span  # noqa: E731

    # ---- low fidelity: the twin, sampled densely and fitted ----
    points = qmc.Sobol(len(knobs), scramble=True, seed=0).random(MODEL_SAMPLES)
    population = to_native(points)
    readings = twin.track(
        overrides=twin.columns(knobs, population), batch=len(population)
    )
    _, size = readings.beam_at(twin.readout)
    model_y = torch.tensor(
        np.column_stack([
            size.detach().cpu().numpy().astype(float),
            readings.transmission.detach().cpu().numpy().astype(float),
        ]), dtype=DTYPE,
    )
    model_x = torch.tensor(points, dtype=DTYPE)
    low_fidelity = _fit(model_x, model_y)

    def predict(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return low_fidelity.posterior(x).mean

    # ---- high fidelity: the machine, as the twin plus a discrepancy ----
    seen_x = [to_unit(present)]
    seen_y = [np.array([machine.last.sigma_x, machine.last.sigma_y,
                        machine.last.transmission])]
    best = machine.last.score

    for _ in range(min(shots, max(0, session.left - 1))):
        x = torch.tensor(np.array(seen_x), dtype=DTYPE)
        delta = torch.tensor(np.array(seen_y), dtype=DTYPE) - predict(x)
        discrepancy = (_fit(x, delta) if len(seen_x) > 2 else None)

        candidates = torch.tensor(
            qmc.Sobol(len(knobs), scramble=True,
                      seed=len(seen_x)).random(CANDIDATES),
            dtype=DTYPE,
        )
        with torch.no_grad():
            posterior = low_fidelity.posterior(candidates)
            samples = posterior.rsample(torch.Size([POSTERIOR_SAMPLES]))
            if discrepancy is not None:
                offset = discrepancy.posterior(candidates)
                samples = samples + offset.rsample(
                    torch.Size([POSTERIOR_SAMPLES])
                )
            else:
                samples = samples + delta.mean(dim=0)
            cost = _mismatch(samples, target)
        # Expected improvement on the composite objective.
        improvement = (best - cost).clamp(min=0.0).mean(dim=0)
        if float(improvement.max()) < _shot_cost(machine.shots):
            session.say("bo: no proposal is worth another shot")
            break
        choice = candidates[int(improvement.argmax())].detach().cpu().numpy()
        reading = session.set(dict(zip(knobs, to_native(choice))))
        seen_x.append(choice)
        seen_y.append(np.array([reading.sigma_x, reading.sigma_y,
                                reading.transmission]))
        best = min(best, reading.score)


def _shot_cost(shots: int) -> float:
    """What one more shot adds to the score, from `machines.evaluation_cost`."""
    from machines import evaluation_cost

    return evaluation_cost(shots + 1) - evaluation_cost(shots)
