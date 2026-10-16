"""Figures for the map-based reconstruction and the image-count study (fast_ment.py outputs)."""

from __future__ import annotations

import sys
from pathlib import Path

import common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = common.HERE / "figures"
RUNS = [("fast", 68), ("lean1", 55), ("lean2", 45), ("lean3", 39), ("lean4", 24), ("lean5", 21), ("lean6", 16)]
PLANES = [(0, 1, "x (mm)", "px (mrad)"), (2, 3, "y (mm)", "py (mrad)"), (0, 2, "x (mm)", "y (mm)"),
          (4, 5, "tau (mm)", "delta (%)")]
UNIT = np.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e2])


def load(tag):
    return common.read_json(common.HERE / f"ment_results_{tag}.json"), np.load(common.HERE / f"ment_full_{tag}.npz")


def quantities_vs_images(out):
    groups = [
        ("emittances", [("core_eps_x", "eps x 95%"), ("core_eps_y", "eps y 95%"), ("eps_x", "eps x rms"),
                        ("eps_y", "eps y rms"), ("eps_4d_ratio", "eps4d/(ex ey)")]),
        ("longitudinal", [("sigma_z", "sigma z"), ("sigma_delta", "sigma delta"), ("chirp", "chirp"),
                          ("core_eps_z", "eps z 95%")]),
    ]
    results = [load(tag)[0] for tag, _ in RUNS]
    n = [r["n_images"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, (title, keys) in zip(axes, groups):
        for key, label in keys:
            ax.plot(n, [r["reco"][key] / r["truth"][key] for r in results], "o-", ms=3, label=label)
        ax.axhline(1, color="k", lw=0.6)
        ax.axhspan(0.9, 1.1, color="0.9", zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(n, [str(v) for v in n], fontsize=7)
        ax.minorticks_off()
        ax.set_yscale("log")
        ax.set_yticks([0.5, 0.7, 1, 1.5, 2, 3, 4], ["0.5", "0.7", "1", "1.5", "2", "3", "4"])
        ax.set_xlabel("number of images", fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("reconstructed / true", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def projections_vs_images(beam, out, tags=("fast", "lean3", "lean4", "lean6"), bins=60, extent_sd=3.5):
    sd, mean = beam.std(0), beam.mean(0)
    fig, axes = plt.subplots(len(tags), len(PLANES), figsize=(8, 2 * len(tags)))
    for row, tag in enumerate(tags):
        res, full = load(tag)
        for col, (i, j, li, lj) in enumerate(PLANES):
            ax = axes[row, col]
            rng = [(mean[k] - extent_sd * sd[k], mean[k] + extent_sd * sd[k]) for k in (i, j)]
            h, xe, ye = np.histogram2d(full["sample"][:, i], full["sample"][:, j], bins=bins, range=rng,
                                       weights=full["weights"])
            t, _, _ = np.histogram2d(beam[:, i], beam[:, j], bins=bins, range=rng)
            ax.imshow(h.T, origin="lower", aspect="auto", cmap="inferno",
                      extent=[xe[0] * UNIT[i], xe[-1] * UNIT[i], ye[0] * UNIT[j], ye[-1] * UNIT[j]])
            xc, yc = 0.5 * (xe[1:] + xe[:-1]) * UNIT[i], 0.5 * (ye[1:] + ye[:-1]) * UNIT[j]
            ax.contour(xc, yc, t.T / t.max(), levels=[0.1, 0.5], colors="white", linewidths=0.6)
            ax.tick_params(labelsize=6)
            if row == len(tags) - 1:
                ax.set_xlabel(li, fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{res['n_images']} images\n{lj}", fontsize=7)
            else:
                ax.set_ylabel(lj, fontsize=7)
    fig.suptitle("Reconstruction at S02 (colour) against the true beam (white, 10 and 50 % of peak)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def profiles_vs_images(beam, out, tags=("fast", "lean3", "lean4", "lean6"), bins=80):
    fig, axes = plt.subplots(1, 6, figsize=(8, 2))
    for k, ax in enumerate(axes):
        lo, hi = np.percentile(beam[:, k], [0.5, 99.5])
        edges = np.linspace(lo, hi, bins + 1)
        t, _ = np.histogram(beam[:, k], edges, density=True)
        c = 0.5 * (edges[1:] + edges[:-1]) * UNIT[k]
        ax.plot(c, t / t.max(), "k", lw=1.2, label="true")
        for tag in tags:
            res, full = load(tag)
            h, _ = np.histogram(full["sample"][:, k], edges, weights=full["weights"], density=True)
            ax.plot(c, h / t.max(), lw=0.8, label=f"{res['n_images']}")
        ax.set_xlabel(["x (mm)", "px (mrad)", "y (mm)", "py (mrad)", "tau (mm)", "delta (%)"][k], fontsize=7)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
    axes[0].legend(fontsize=5, title="images", title_fontsize=5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def timing(out):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    labels = ["xsuite tracking\n(14 Oct)", "fitted maps\n68 images", "fitted maps\n39 images"]
    seconds = [9000, load("fast")[0]["seconds"], load("lean3")[0]["seconds"]]
    ax.bar(labels, np.array(seconds) / 60, color=["C1", "C0", "C0"])
    for k, s in enumerate(seconds):
        ax.text(k, s / 60, f"{s / 60:.0f} min" if s > 600 else f"{s / 60:.1f} min", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("reconstruction time (min, 4 cores)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def map_accuracy(out):
    rows = common.read_json(common.HERE / "surrogate_validation.json")
    err = np.array([r["rms_err_um"] for r in rows])
    inside = np.array([r["inside"] for r in rows])
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.scatter(inside * 100, np.maximum(err, 1e-2), s=8)
    ax.axhline(100, color="k", lw=0.6, ls="--")
    ax.text(2, 110, "pixel (100 um)", fontsize=6)
    ax.axhline(20, color="C1", lw=0.6, ls="--")
    ax.text(2, 22, "screen resolution (20 um)", fontsize=6, color="C1")
    ax.set_yscale("log")
    ax.set_xlabel("beam inside the screen window (%)", fontsize=8)
    ax.set_ylabel("map error, rms (um)", fontsize=8)
    ax.set_title("68 fitted maps against xsuite tracking of the true beam", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    beam = np.load(common.HERE / "images.npz")["beam"]
    quantities_vs_images(FIG / "fast_quantities_vs_images.png")
    projections_vs_images(beam, FIG / "fast_projections_vs_images.png")
    profiles_vs_images(beam, FIG / "fast_profiles_vs_images.png")
    timing(FIG / "fast_timing.png")
    map_accuracy(FIG / "fast_map_accuracy.png")
    if len(sys.argv) > 1:
        dest = Path(sys.argv[1])
        dest.mkdir(parents=True, exist_ok=True)
        for f in FIG.glob("fast_*.png"):
            (dest / f.name).write_bytes(f.read_bytes())


if __name__ == "__main__":
    main()
