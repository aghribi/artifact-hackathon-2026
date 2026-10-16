"""Per-episode statistics of an MFopt run: scores, steps, wall time, screen moves.

Reads the JSON that `MFopt.run_baselines` writes for one policy, stores every
episode's numbers in one JSON, and draws a histogram of each metric. Run from
challenges/optimisation:

    python talk_figures/scripts/mfopt_episode_stats.py MFopt/results/mfbo_100.json \
        talk_figures/mfopt --label "stage A + MF-BO" --workers 16

A screen move is a step whose reading comes from a different screen than the one
before it (the episode starts on the readout screen, FEH-SCR-01). Wall time is the
time `run_episode` took for the episode, model work included, in a worker process
that shared the machine with `--workers - 1` others.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Shared script: the figure style is set here, not taken from a local rc file.
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "figure.figsize": (4.0, 2.5),
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300,
    }
)

METRICS = (
    ("score", "Benchmark score per episode", None),
    ("steps", "Steps (environment calls) per episode", 1),
    ("wall_time_s", "Wall time per episode (s)", None),
    ("screen_moves", "Screen moves per episode", 1),
)


def screen_moves(trace: list[dict]) -> int:
    screens = [t["screen"] for t in trace]
    return sum(a != b for a, b in pairwise(screens))


def episodes(path: Path) -> list[dict]:
    rows = []
    for e in json.loads(path.read_text())["episodes"]:
        rows.append(
            {
                "seed": e["seed"],
                "target_m": e["target"],
                "score": e["score"],
                "steps": e["steps"],
                "wall_time_s": e["seconds"],
                "screen_moves": screen_moves(e["trace"]),
                "mismatch": e["mismatch"],
                "relative_error": e["relative_error"],
                "transmission": e["transmission"],
                "achieved_m": e["achieved"],
                "trace": e["trace"],
            }
        )
    return rows


def summary(rows: list[dict]) -> dict:
    out = {}
    for key, _, _ in METRICS:
        v = np.array([r[key] for r in rows], dtype=float)
        out[key] = {
            "mean": v.mean(),
            "median": float(np.median(v)),
            "std": v.std(ddof=1),
            "min": v.min(),
            "max": v.max(),
            "p10": float(np.quantile(v, 0.1)),
            "p90": float(np.quantile(v, 0.9)),
        }
    err = np.array([r["relative_error"] for r in rows])
    out["hit_10"] = float((err <= 0.10).mean())
    out["hit_25"] = float((err <= 0.25).mean())
    out["hit_50"] = float((err <= 0.50).mean())
    return {
        k: ({kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
        for k, v in out.items()
    }


def histogram(ax, values, title, integer_bins, label):
    values = np.asarray(values, dtype=float)
    if integer_bins:
        bins = np.arange(values.min() - 0.5, values.max() + 1.5, 1)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlim(values.min() - 1.5, values.max() + 1.5)
    else:
        bins = 20
    ax.hist(values, bins=bins, label=label)
    ax.axvline(values.mean(), color="C1", lw=1, label=f"mean {values.mean():.2f}")
    ax.axvline(
        np.median(values),
        color="C2",
        lw=1,
        ls="--",
        label=f"median {np.median(values):.2f}",
    )
    ax.set_xlabel(title)
    ax.set_ylabel("Episodes")
    ax.legend(frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--label", default="stage A + MF-BO")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--stem", default="mfopt_100_episodes")
    args = parser.parse_args()
    rows = episodes(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    data = {
        "policy": args.label,
        "source": str(args.results),
        "episodes_count": len(rows),
        "parallel_workers": args.workers,
        "summary": summary(rows),
        "episodes": rows,
    }
    (args.out / f"{args.stem}.json").write_text(json.dumps(data, indent=1))
    for key, title, integer in METRICS:
        fig, ax = plt.subplots()
        histogram(
            ax,
            [r[key] for r in rows],
            title,
            integer,
            f"{args.label}, {len(rows)} days",
        )
        for ext in ("png", "pdf"):
            fig.savefig(args.out / f"{args.stem}_{key}.{ext}")
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.1))
    for ax, (key, title, integer) in zip(axes.flat, METRICS):
        histogram(ax, [r[key] for r in rows], title, integer, None)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"{args.stem}_all.{ext}")
    print(json.dumps(data["summary"], indent=1))


if __name__ == "__main__":
    main()
