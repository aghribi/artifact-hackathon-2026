"""Figure: true beam centroid at the BPMs before and after single-shot SVD correction."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
result = json.loads((HERE / "result_svd_bpmoff0_cut0.001_shots1_seed0.json").read_text())
history = result["history"]
figure, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True, layout="constrained")
for plane, axis in enumerate(axes):
    for step, label in ((0, "before"), (1, "after 1 shot"), (3, "after 3 shots")):
        orbit = np.asarray(history[step]["bpm_orbit_um"])[:, plane]
        axis.plot(np.arange(len(orbit)), orbit / 1e3, "o-", markersize=3, label=label)
    axis.set_ylabel(f"{'xy'[plane]} at BPM (mm)")
    axis.axhline(0, color="0.8", lw=0.5, zorder=0)
axes[0].legend(ncol=3)
axes[1].set_xlabel("BPM index (31 BPMs, S02 to FEH)")
figure.savefig(HERE / "fig_svd_orbit.png", dpi=150)
