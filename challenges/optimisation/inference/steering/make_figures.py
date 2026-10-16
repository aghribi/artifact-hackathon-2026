"""Figures for README.md from the result_*.json files."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RUNS = {
    "BPMs, nominal beam": "result_bpm_nominal_seed0.json",
    "screens (4/quad), nominal beam": "result_screen_nominal_spm4_seed0.json",
}


def load(name):
    return json.loads((HERE / name).read_text())


def short(quad):
    section, number = quad.split("_")[1], quad.split("_")[-1]
    return f"{section.upper()}-Q{int(number)}"


def offsets_figure():
    bpm, screen = load(RUNS["BPMs, nominal beam"]), load(RUNS["screens (4/quad), nominal beam"])
    quads = list(bpm["quad_offsets"]["per_quad"])
    index = np.arange(len(quads))
    figure, axes = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True, layout="constrained")
    for plane, axis in enumerate(axes):
        true = [bpm["quad_offsets"]["per_quad"][q]["true_um"][plane] for q in quads]
        axis.plot(index, true, "k_", markersize=9, mew=2, label="true")
        for shift, (label, run) in zip((-0.15, 0.15), (("BPMs", bpm), ("screens only", screen))):
            estimate = [run["quad_offsets"]["per_quad"][q]["estimate_um"][plane] for q in quads]
            error = [run["quad_offsets"]["per_quad"][q]["posterior_rms_um"][plane] for q in quads]
            axis.errorbar(index + shift, estimate, yerr=error, fmt="o", markersize=3, elinewidth=0.8, label=label)
        axis.set_ylabel(f"{'xy'[plane]} offset (um)")
        axis.set_ylim(-450, 450)
        axis.axhline(0, color="0.8", lw=0.5, zorder=0)
    axes[0].legend(ncol=3, loc="upper left")
    axes[1].set_xticks(index[::2], [short(q) for q in quads][::2], rotation=90)
    axes[1].set_xlabel("quadrupole (beam direction)")
    figure.savefig(HERE / "fig_quad_offsets.png", dpi=150)


def convergence_figure():
    """Whitened chi2/dof per iteration, parsed from the run logs (jitter-aware, so 1 means
    the fit explains the data down to the noise)."""
    logs = {
        "BPMs, exact beam": "log_bpm_exact.txt",
        "BPMs, nominal beam": "log_bpm_nominal.txt",
        "screens 4/quad, nominal beam": "log_screen_nominal_spm4.txt",
        "screens 2/quad, nominal beam": "log_screen_nominal_spm2.txt",
    }
    figure, axis = plt.subplots(figsize=(4, 2.5), layout="constrained")
    for label, name in logs.items():
        values = [float(line.split("chi2/dof")[1]) for line in (HERE / name).read_text().splitlines() if "chi2/dof" in line]
        axis.semilogy(range(1, len(values) + 1), values, "o-", markersize=3, label=label)
    axis.axhline(1, color="0.5", lw=0.8, ls="--")
    axis.set_xlabel("Levenberg-Marquardt iteration")
    axis.set_ylabel("whitened chi2 / dof")
    axis.legend(fontsize="small")
    figure.savefig(HERE / "fig_convergence.png", dpi=150)


if __name__ == "__main__":
    offsets_figure()
    convergence_figure()
