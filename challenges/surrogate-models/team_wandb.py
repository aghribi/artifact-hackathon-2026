"""Log plasma-pirates experiments to the team's group in the shared W&B project.

Usage:
    from team_wandb import start_run, log_scores
    run = start_run(task="trajectory", tier="medium", model="mlp", lr=1e-3)
    ...                                   # train, wandb.log({"train/loss": ...}) as you go
    log_scores(run, score_submission(sub, key))   # or log_scores(run, 0.87)
    run.finish()

The official score is not ours to write: the organisers score the CSV uploaded to DALIA and
publish `pallas/mean_r2` themselves. Our own holdout numbers go under `holdout/`, which the
leaderboard does not read. Run details are visible to our team, mentors and organisers.

The key is read from WANDB_API_KEY, or else from the first `.env` found in this folder or
above it (the team keeps it in `AI Hackathon/.env`, outside the repo). Never commit it.
Entity, project and group follow GRADING.md; a wrong `group` drops runs from the leaderboard,
so it is fixed here. Use `tags` or config fields to tell your own experiments apart.
"""

import os
from pathlib import Path

ENTITY = "artifact-hackathon-2026"
PROJECT = "surrogate-models"
GROUP = "plasma-pirates"
TIERS = ("easy", "medium", "hard", "optional")


def load_env():
    """Set the variables of the nearest `.env` that are not already in the environment."""
    for folder in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        path = folder / ".env"
        if path.is_file():
            for line in path.read_text().splitlines():
                name, sep, value = line.strip().partition("=")
                if sep and name and not name.startswith("#"):
                    os.environ.setdefault(name.strip(), value.strip().strip("'\""))
            return path
    return None


def start_run(task, tier, name=None, tags=None, **config):
    """A run in the team's group, with `task` and `tier` in its config as GRADING.md asks."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    load_env()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is not set and no .env with it was found")
    import wandb                       # here, so the constants import without wandb installed
    wandb.login(key=os.environ["WANDB_API_KEY"])
    return wandb.init(entity=ENTITY, project=PROJECT, group=GROUP, name=name,
                      tags=tags, config={"task": task, "tier": tier, **config})


def log_scores(run, scores):
    """Log a `pallas_score.score_submission` result on our own holdout, or a bare mean R2."""
    if not isinstance(scores, dict):
        scores = {"mean": scores}
    logged = {"holdout/mean_r2": float(scores["mean"])}
    for key, value in scores.items():
        if key == "mean":
            continue
        logged[f"holdout/{key}" if key.startswith("n_") else f"holdout/r2/{key}"] = value
    run.log(logged)
    run.summary.update(logged)
    return logged
