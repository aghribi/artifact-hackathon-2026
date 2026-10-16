"""k1 of the 30 rematched quadrupoles (after the VBC) against target spot size.
Each panel: sigma_x = sigma_y diagonal, a sigma_x scan at sigma_y = 1.1 mm and a sigma_y scan at
sigma_x = 1.1 mm (rematch, solid), Eiad's diagonal (dashed grey), and the +-100 A range (shaded).

Usage (from challenges/optimisation): python optics_matching_vbc_on/plot_quads.py <results_dir> optics_matching/results/validated_grid.npz [figure_dir]"""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
d = Path(sys.argv[1]); g = np.load(sys.argv[2]); fig_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else d
# +-100 A ranges, as SimEnv derives them from the magnet calibration
sys.path[:0] = [str(Path.cwd() / "RL_framework"), str(Path.cwd())]
from rl_framework.config import RLConfig
from rl_framework.envs.sim_env import SimEnv
limits = SimEnv(RLConfig(backend="cheetah", num_particles=1000, beam_variation=0.0,
                         misalignment_variation=0.0, beam_jitter=0.0)).magnet_limits
z = np.load(d / "rematched_grid_vbc_on.npz")
Q = z["quadrupoles"].tolist(); s = z["spot_sizes_m"] * 1e3; K, KE = z["k1"], g["k1"]
varied = [q for q in Q if np.ptp(K[..., Q.index(q)]) > 1e-9]
m = 5  # index of 1.1 mm
fig, axes = plt.subplots(6, 5, figsize=(20, 21), sharex=True)
for ax, q in zip(axes.flat, varied):
    j = Q.index(q)
    lo, hi = limits[q]
    ax.axhspan(lo, hi, color="0.92", zorder=0)
    ax.plot(s, KE.diagonal(axis1=0, axis2=1)[j], "--", color="0.5", lw=1.3, label="Eiad, σx=σy")
    ax.plot(s, K.diagonal(axis1=0, axis2=1)[j], "-o", color="#2a6fdb", ms=3, lw=1.8, label="rematch, σx=σy")
    ax.plot(s, K[:, m, j], "-", color="#e8590c", lw=1.3, label="rematch, σx scan (σy=1.1 mm)")
    ax.plot(s, K[m, :, j], "-", color="#2b8a3e", lw=1.3, label="rematch, σy scan (σx=1.1 mm)")
    vals = np.concatenate([K[..., j].ravel(), KE[..., j].ravel()])
    pad = max(0.3, 0.1 * np.ptp(vals))
    ax.set_ylim(vals.min() - pad, vals.max() + pad)   # zoom on the data; band only shows if near the limit
    ax.set_xscale("log"); ax.set_title(q.replace("cla_", "").replace("_mag_quad_", " Q"), fontsize=11)
    ax.grid(alpha=0.3)
for ax in axes[-1]: ax.set_xlabel("target size [mm]")
for ax in axes[:, 0]: ax.set_ylabel("k1 [m$^{-2}$]")
for ax in axes.flat[len(varied):]: ax.axis("off")
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h + [plt.Rectangle((0, 0), 1, 1, color="0.92")], l + ["±100 A range"], loc="upper center", ncol=5, fontsize=12, bbox_to_anchor=(0.5, 0.965))
fig.suptitle("Quadrupoles varied by the VBC-on rematch (VBC 131.7 mrad); Eiad's 42-quad prefix is fixed", y=0.995, fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.945)); fig.savefig(fig_dir / "quads_vs_target.png", dpi=100)
dk = np.abs(K - KE)[..., [Q.index(q) for q in varied]]
print("max |k1 change| per quad:", {q.split('_mag_')[0][4:] + q[-2:]: round(float(dk[..., i].max()), 2) for i, q in enumerate(varied)})
