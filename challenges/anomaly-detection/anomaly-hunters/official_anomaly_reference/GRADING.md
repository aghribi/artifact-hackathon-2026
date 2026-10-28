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
| Optimisation | `report.calc_score()` — mean per-case error in units of the requested spot size, plus a penalty for evaluation count (lower is better) | Real machine data; cost-per-evaluation matters on a real machine, so the score prices evaluation budget alongside tuning accuracy rather than reporting accuracy alone — see `score.py` |

## Logging to Weights & Biases

Each team logs to the shared W&B team (entity) **`artifact-hackathon-2026`**, one project per
challenge, grouped by team so Agora-Conf's leaderboard can pick up your runs automatically.
**The exact snippet for your team — entity, project and `group` already filled in — is on
your team's page in Agora-Conf** (Hackathon guide → *Training runs and the leaderboard*);
copy it from there rather than retyping it, since a wrong `group` silently drops your runs
from the leaderboard (they still log to W&B, Agora just won't match them to your team).

- **Entity:** `artifact-hackathon-2026`
- **Project:** one per challenge — `anomaly-detection`, `optimisation`, `surrogate-models`
- **`group`:** your team's slug, exactly as shown on your team's page in Agora (e.g.
  `anomaly-hunters`, `the-outliers`, `fast-approximate`, `overfit-proud`, `laser-focused`,
  `plasma-pirates`)
- **Auth:** each team has its own W&B API key (a service account, shared by the whole team —
  no personal W&B account needed). It's on `cca.in2p3.fr`, readable by your own CC-IN2P3
  account (see the [access guide](access-guide.md) for getting one), at:
  `/sps/m4cast/artifact_hackathon_2026/<challenge>/teams/<your-team-slug>/wandb_api_key.txt`
  (`<challenge>` is `anomaly-detection`, `optimisation` or `surrogate-models`; `<your-team-slug>`
  is the same slug as your `group` above). **Never commit it** — put it in a local `.env` /
  shell export, never in a notebook cell you push:
  ```
  export WANDB_API_KEY=$(cat /sps/m4cast/artifact_hackathon_2026/<challenge>/teams/<your-team-slug>/wandb_api_key.txt)
  ```
  (or `wandb.login(key=...)` typed interactively, not hardcoded). That file is only
  `chmod 640` on a shared NFS mount all participants can log into — it isn't private from
  other teams the way a real secrets manager would be, so treat it as "don't hand it to
  someone outside the hackathon," not as a strongly-guarded credential.
- **Each run's config** should set:
  - `tier`: `"easy"` | `"medium"` | `"hard"` | `"optional"`
- **Each run logs its score** under a challenge-specific key, read straight from that
  challenge's scoring script:
  - Anomaly detection: `anomaly/holdout_mse` (lower is better)
  - Surrogate models: `pallas/mean_r2` for R²-based tasks, or the task-specific key
    (`pallas/median_ratio` for particle clouds, `pallas/median_rel_l2_*` for fields) —
    see `pallas_score.py`'s own output. The leaderboard ranks on `pallas/mean_r2`; log the
    other keys too (for your own tracking), they just don't drive the ranking.
  - Optimisation: `optim/score` — `report.calc_score()` from `score.py` (lower is better,
    drives the ranking); log `optim/evals_per_case` alongside for context, not ranked on

Logging is one or two lines around your existing scoring call — replace `<your-team-slug>`
with the slug from your team's page in Agora (the `project` below is for anomaly-detection;
use `optimisation` or `surrogate-models` in the other challenges):

```python
import os, wandb
wandb.login(key=os.environ["WANDB_API_KEY"])  # set once in your shell, never hardcoded
wandb.init(entity="artifact-hackathon-2026", project="anomaly-detection",
           group="<your-team-slug>", config={"tier": "easy"})
wandb.log({"anomaly/holdout_mse": mean_mse})
```

## Ranking

Within a challenge, a team's standing is: **highest tier with a bar beaten, tied-break by
that tier's metric value.** Reaching Medium with a worse Medium score still outranks staying
at Easy. Optional/stretch objectives don't move the ranking — they're for teams who finish
early or want to explore further, and are judged qualitatively (does it hold up, is the
report honest about its limits), same as the write-up every team gives Friday morning.
