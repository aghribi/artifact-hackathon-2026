# Grading & Scoring

How the three challenges are judged. Each challenge has two competing teams working the
same tasks, so scoring needs to be comparable within a challenge — it does **not** need to
be comparable *across* challenges (an anomaly-detection score and a surrogate-models score
are different units and are never compared to each other).

## Common structure

Every challenge's `objectives.md` follows the same shape, set by the surrogate-models
(PALLAS) challenge, which has the most mature version of this pattern:

- **Three tiers — Easy, Medium, Hard** (plus optional stretch objectives outside the
  sequence). Everyone goes Easy → Medium → Hard in the same order, so teams are compared on
  the same tasks, not whichever one they picked.
- **Each objective states a signature** (what goes in, what comes out), **a score** (one
  metric, computed by a script in the challenge folder — `pallas_score.py`,
  `score.py`, ...), and **a bar** (a real reference number to beat, not an abstract target).
- A bar is either a reference baseline the organisers ran themselves, or — where a case
  holder hasn't provided one yet — explicitly marked `TBD`.

What the metric *is** differs by challenge, because the challenges differ:

| Challenge | Metric | Why |
|---|---|---|
| Surrogate models | R² per target (zone-aware), Wasserstein distance for particle clouds | Simulation campaigns with known ground truth — see `pallas_score.py` |
| Anomaly detection | Reconstruction MSE on a fixed holdout split | No confirmed fault labels exist (see the challenge's `kickoff_notebook.ipynb`), so the score is model quality on unseen data, not a label match |
| Optimisation | Best objective value (mm, real CLARA spot size) found, *and* evaluations needed to reach it | Real machine data; cost-per-evaluation matters on a real machine, so efficiency is reported alongside the best value, not just the value alone — see `kickoff_notebook.ipynb` |

## Logging to Weights & Biases

Each team logs their score to a shared W&B project so progress is visible without manually
collecting files.

- **Project:** `aissai-hackathon-2026` (one shared project, all challenges and teams)
- **Each run's config** should set:
  - `challenge`: `"anomaly-detection"` | `"optimisation"` | `"surrogate-models"`
  - `team`: your team name
  - `tier`: `"easy"` | `"medium"` | `"hard"` | `"optional"`
- **Each run logs its score** under a challenge-specific key, read straight from that
  challenge's scoring script:
  - Anomaly detection: `anomaly/holdout_mse` (lower is better)
  - Surrogate models: `pallas/mean_r2` for R²-based tasks, or the task-specific key
    (`pallas/median_ratio` for particle clouds, `pallas/median_rel_l2_*` for fields) —
    see `pallas_score.py`'s own output
  - Optimisation: `optim/best_mm` (lower is better) and `optim/n_evals`

Logging is one or two lines around your existing scoring call:

```python
import wandb
wandb.init(project="aissai-hackathon-2026", config={"challenge": "anomaly-detection", "team": "<your-team>", "tier": "easy"})
wandb.log({"anomaly/holdout_mse": mean_mse})
```

## Ranking

Within a challenge, a team's standing is: **highest tier with a bar beaten, tied-break by
that tier's metric value.** Reaching Medium with a worse Medium score still outranks staying
at Easy. Optional/stretch objectives don't move the ranking — they're for teams who finish
early or want to explore further, and are judged qualitatively (does it hold up, is the
report honest about its limits), same as the write-up every team gives Friday morning.
