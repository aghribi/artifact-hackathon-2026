"""Figures for the image-based 6D reconstruction at the S02 start (run_ment.py outputs)."""

from __future__ import annotations

import sys
from pathlib import Path

import common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = common.HERE / "figures"
UNITS = {"x": (1e3, "mm"), "px": (1e3, "mrad"), "y": (1e3, "mm"), "py": (1e3, "mrad"),
         "tau": (1e3, "mm"), "delta": (1e2, "%")}
QUANTITIES = [
    ("eps_x", "eps x rms"), ("core_eps_x", "eps x 95%"), ("eps_y", "eps y rms"),
    ("core_eps_y", "eps y 95%"), ("eps_4d_ratio", "eps4d/(ex ey)"), ("sigma_x", "sigma x"),
    ("sigma_y", "sigma y"), ("sigma_z", "sigma z"), ("sigma_delta", "sigma delta"),
    ("core_eps_z", "eps z 95%"), ("slice_sigma_delta", "slice sigma delta"),
]
CORRELATIONS = [("corr_tau_delta", "tau,delta"), ("corr_x_y", "x,y"), ("corr_x_py", "x,py"),
                ("corr_px_y", "px,y"), ("corr_x_delta", "x,delta"), ("corr_y_tau", "y,tau")]


def projections(sample, weights, beam, out, extent_sd=3.5, bins=60):
    labels = common.LABELS
    sd, mean = beam.std(0), beam.mean(0)
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    for row in range(5):
        for col in range(5):
            ax = axes[row, col]
            i, j = col, row + 1
            if col > row:
                ax.axis("off")
                continue
            ri = (mean[i] - extent_sd * sd[i], mean[i] + extent_sd * sd[i])
            rj = (mean[j] - extent_sd * sd[j], mean[j] + extent_sd * sd[j])
            h, xe, ye = np.histogram2d(sample[:, i], sample[:, j], bins=bins, range=[ri, rj], weights=weights)
            t, _, _ = np.histogram2d(beam[:, i], beam[:, j], bins=bins, range=[ri, rj])
            si, sj = UNITS[labels[i]][0], UNITS[labels[j]][0]
            ax.imshow(h.T, origin="lower", aspect="auto", cmap="inferno",
                      extent=[xe[0] * si, xe[-1] * si, ye[0] * sj, ye[-1] * sj])
            xc, yc = 0.5 * (xe[1:] + xe[:-1]) * si, 0.5 * (ye[1:] + ye[:-1]) * sj
            ax.contour(xc, yc, t.T / t.max(), levels=[0.1, 0.5], colors="white", linewidths=0.6)
            if row == 4:
                ax.set_xlabel(f"{labels[i]} ({UNITS[labels[i]][1]})", fontsize=7)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(f"{labels[j]} ({UNITS[labels[j]][1]})", fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6)
    fig.suptitle("Reconstruction at S02 (colour) and true beam (white contours, 10 and 50 % of peak)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def quantities(results, out):
    truth, reco = results["truth"], results["reco"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={"width_ratios": [1.8, 1]})
    ratios = [reco[k] / truth[k] for k, _ in QUANTITIES]
    ax1.bar(range(len(QUANTITIES)), ratios, color="C0")
    ax1.axhline(1, color="k", lw=0.6)
    ax1.set_yscale("log")
    ax1.set_xticks(range(len(QUANTITIES)), [lab for _, lab in QUANTITIES], rotation=60, ha="right", fontsize=7)
    ax1.set_ylabel("reconstructed / true", fontsize=8)
    x = np.arange(len(CORRELATIONS))
    ax2.bar(x - 0.2, [truth[k] for k, _ in CORRELATIONS], 0.4, label="true", color="C1")
    ax2.bar(x + 0.2, [reco[k] for k, _ in CORRELATIONS], 0.4, label="reconstructed", color="C0")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xticks(x, [lab for _, lab in CORRELATIONS], rotation=60, ha="right", fontsize=7)
    ax2.set_ylabel("correlation", fontsize=8)
    ax2.legend(fontsize=6)
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chirp_scan(scan, out):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    tr = scan["true_transverse"]
    ax.plot([p["chirp"] for p in tr], [p["chi2_s1.3e-3"] for p in tr], "o-", label="true transverse beam")
    rc = scan["reconstructed"]
    ax.plot([p["chirp"] for p in rc], [p["chi2_s4e-3"] for p in rc], "s-", label="reconstructed transverse")
    ax.axvline(13.46, color="k", lw=0.6, ls="--")
    ax.set_xlabel("S02 chirp <tau delta>/<tau tau> (1/m)", fontsize=8)
    ax.set_ylabel("image chi2 per pixel", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    results = common.read_json(common.HERE / "ment_results.json")
    full = np.load(common.HERE / "ment_full.npz")
    beam = np.load(common.HERE / "images.npz")["beam"]
    projections(full["sample"], full["weights"], beam, FIG / "start_projections.png")
    quantities(results, FIG / "start_quantities.png")
    chirp_scan(common.read_json(common.HERE / "chirp_scan.json"), FIG / "start_chirp_scan.png")
    if len(sys.argv) > 1:
        dest = Path(sys.argv[1])
        dest.mkdir(parents=True, exist_ok=True)
        for f in FIG.glob("start_*.png"):
            (dest / f.name).write_bytes(f.read_bytes())


if __name__ == "__main__":
    main()
