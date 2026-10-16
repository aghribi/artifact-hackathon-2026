"""Matrix plots for the VBC-on rematch: transmission and spot-size error, Eiad vs rematched.
Also (re)writes rematched_grid_vbc_on.npz from rematch_results.json.

Usage: python plot_rematch.py <results_dir> optics_matching/results/validated_grid.npz [vbc_angle_rad] [figure_dir]"""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
d = Path(sys.argv[1]); grid_path = Path(sys.argv[2])  # .../optics_matching/results/validated_grid.npz
res = json.loads((d / "rematch_results.json").read_text())
g = np.load(grid_path); s = g["spot_sizes_m"]; n = len(s)
k1 = np.array(g["k1"], float); T = np.full((n, n, 2), np.nan); S = np.full((n, n, 2, 2), np.nan)
for r in res:
    i, j = r["i"], r["j"]; k1[i, j] = r["k1"]
    T[i, j] = r["before"][0], r["after"][0]; S[i, j, 0], S[i, j, 1] = r["before"][1:], r["after"][1:]
np.savez(d / "rematched_grid_vbc_on.npz", spot_sizes_m=s, k1=k1, quadrupoles=g["quadrupoles"],
         transmission_eiad=T[..., 0], transmission=T[..., 1], actual_sizes_eiad_m=S[:, :, 0],
         actual_sizes_m=S[:, :, 1], vbc_angle_rad=float(sys.argv[3]) if len(sys.argv) > 3 else 0.13169935)
tgt = np.stack(np.meshgrid(s, s, indexing="ij"), -1)
err = np.abs(S / tgt[:, :, None] - 1).max(-1) * 100   # worst plane, % (n, n, 2)
lab = [f"{v*1e3:.2g}" for v in s]
fig, axes = plt.subplots(2, 2, figsize=(17, 14))
for col, name in enumerate(["Eiad's nodes", "Rematched (30 quads after VBC)"]):
    for row, (Z, cmap, vmax, unit, fmt) in enumerate([(T[..., col] * 100, "viridis", 100, "transmission [%]", "{:.0f}"),
                                                      (err[..., col], "magma_r", 100, "worst-plane size error [%]", "{:.0f}")]):
        ax = axes[row, col]
        im = ax.imshow(Z.T, origin="lower", cmap=cmap, vmin=0, vmax=vmax)
        for i in range(n):
            for j in range(n):
                v = Z[i, j]
                light = (v > 55) if row == 0 else (v < 45)
                ax.text(i, j, fmt.format(v) if np.isfinite(v) else "", ha="center", va="center",
                        fontsize=7.5, color="black" if light else "white")
        ax.set_xticks(range(n), lab, rotation=45); ax.set_yticks(range(n), lab)
        ax.set_xlabel(r"target $\sigma_x$ [mm]"); ax.set_ylabel(r"target $\sigma_y$ [mm]")
        ax.set_title(f"{name}: {unit}", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(f"VBC on ({(float(sys.argv[3]) if len(sys.argv) > 3 else 0.13169935)*1e3:.1f} mrad, R56 = -57.6 mm), master lattice, Cheetah 2nd order, apertures on, 50k particles", fontsize=13)
fig.tight_layout(); fig.savefig((Path(sys.argv[4]) if len(sys.argv) > 4 else d) / "rematch_vbc_on.png", dpi=110)
print("median transmission eiad/rematch", np.nanmedian(T[..., 0]), np.nanmedian(T[..., 1]),
      "min", np.nanmin(T[..., 0]), np.nanmin(T[..., 1]))
print("size err median/max eiad", np.nanmedian(err[..., 0]), np.nanmax(err[..., 0]),
      "rematch", np.nanmedian(err[..., 1]), np.nanmax(err[..., 1]))
print("nodes rematch T>=0.95 & err<=5%", int(((T[..., 1] >= .95) & (err[..., 1] <= 5)).sum()))
