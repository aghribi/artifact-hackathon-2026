"""Figures for README.md from the results_*.json files."""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
mpl.rcParams.update(
    {
        "font.size": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "figure.figsize": (4.0, 2.5),
        "figure.constrained_layout.use": True,
        "savefig.dpi": 200,
    }
)
s02 = json.loads((HERE / "results_s02.json").read_text())
s07 = json.loads((HERE / "results_s07.json").read_text())

# 1. beam sizes on the screens with nominal optics
fig, ax = plt.subplots()
positions = [v["s"] for v in s02["screens"].values()]
sx = [v["sigma_x_mm"] for v in s02["screens"].values()]
sy = [v["sigma_y_mm"] for v in s02["screens"].values()]
ax.semilogy(positions, sx, "o", ms=3, label="rms x")
ax.semilogy(positions, sy, "s", ms=3, label="rms y")
ax.axhline(4.0, color="k", ls="--", lw=0.8, label="imageable limit 4 mm")
ax.set_xlabel("s from cla_s02_sim_aper_01 (m)")
ax.set_ylabel("rms beam size (mm)")
ax.legend(loc="upper left")
fig.savefig(HERE / "fig_screen_sizes.png")

# 2. forward-model closure per screen
fig, ax = plt.subplots()
index = np.arange(len(positions))
for label, marker in (("chromatic", "o"), ("chromatic+orbit", "s")):
    values = np.array(s02["closure_true_sigma_xx_yy"][label])
    ax.plot(
        index,
        np.abs(values[:, 0]) * 100,
        marker + "-",
        ms=3,
        lw=0.7,
        label=f"{label}, x",
    )
ax.plot(
    index,
    np.abs(np.array(s02["closure_true_sigma_xx_yy"]["chromatic+orbit"])[:, 1]) * 100,
    "^-",
    ms=3,
    lw=0.7,
    label="chromatic+orbit, y",
)
ax.set_yscale("log")
ax.set_xlabel("screen index along the line")
ax.set_ylabel("|model - tracked| of variance (%)")
ax.legend()
fig.savefig(HERE / "fig_closure.png")

# 3. conditioning and predicted precision for cumulative screen groups
fig, ax = plt.subplots()
names, kappas, ex, ey = [], [], [], []
for key, row in s02["conditioning_cumulative"].items():
    names.append(row.get("group", key))
    kappas.append(row["kappa"] if np.isfinite(row["kappa"]) else np.nan)
    ex.append(row.get("d_eps_n_x_um", np.nan))
    ey.append(row.get("d_eps_n_y_um", np.nan))
x = np.arange(len(names))
ax.semilogy(x, ex, "o-", ms=3, label="rel. error eps_x")
ax.semilogy(x, ey, "s-", ms=3, label="rel. error eps_y")
ax.semilogy(x, kappas, "^--", ms=3, lw=0.7, label="condition number")
ax.set_xticks(x, names, rotation=45)
ax.set_xlabel("screens up to and including")
ax.set_ylabel("2 % data error propagated")
ax.legend()
fig.savefig(HERE / "fig_conditioning.png")

# 4. energy-spread identifiability
fig, ax = plt.subplots()
for results, subset, label in (
    (s02, "all 23", "S02 start, all 23"),
    (s02, "FEA+FEH (dispersive)", "S02 start, FEA+FEH only"),
    (s02, "S02-S05", "S02 start, S02-S05"),
    (s07, "all 10", "S07, all 10"),
):
    scan = results["energy_spread"]["scan"]
    scales = np.array(sorted(float(k) for k in scan))
    chi2 = np.array([scan[f"{k:g}"][subset]["chi2_total"] for k in scales])
    ax.semilogy(scales, chi2 / np.nanmin(chi2), "o-", ms=3, lw=0.7, label=label)
ax.set_xlabel("assumed / true energy spread")
ax.set_ylabel("chi2 / minimum")
ax.set_ylim(0.5, 1e4)
ax.legend()
fig.savefig(HERE / "fig_energy_spread.png")
