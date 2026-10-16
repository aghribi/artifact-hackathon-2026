"""Matrix plots of node transmission: sigma_x target on x, sigma_y target on y.

Usage: python plot_matrix.py <results_dir> [figure_dir]"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
d = Path(sys.argv[1]); fig_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else d
TITLES = {"vbc_on": "Master lattice (second order), VBC on (118.5 mrad), L4H on",
          "vbc_off": "Master lattice (second order), VBC off, L4H on",
          "design": "Eiad's design lattice (VBC off, L4H off)"}
for case, title in TITLES.items():
    f = d / f"transmission_{case}.npz"
    if not f.exists(): continue
    z = np.load(f); s = z["sizes_m"] * 1e3; T = z["transmission"] * 100   # T[ix, iy]
    n = len(s)
    fig, ax = plt.subplots(figsize=(9, 7.6))
    im = ax.imshow(T.T, origin="lower", cmap="viridis", vmin=0, vmax=100)
    for i in range(n):
        for j in range(n):
            v = T[i, j]
            ax.text(i, j, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="black" if v > 55 else "white")
    lab = [f"{v:.2g}" for v in s]
    ax.set_xticks(range(n), lab, rotation=45); ax.set_yticks(range(n), lab)
    ax.set_xlabel(r"target $\sigma_x$ at FEH screen [mm]"); ax.set_ylabel(r"target $\sigma_y$ at FEH screen [mm]")
    base = float(z["baseline"][0]) * 100
    ax.set_title(f"{title}\nTransmission [%] of Eiad's 144 optics nodes (Cheetah 2nd order, apertures on, {int(z['num_particles'])//1000}k particles); nominal optics: {base:.0f}%", fontsize=10)
    fig.colorbar(im, ax=ax, label="transmission to cla_feh_dia_scr_01 [%]")
    fig.tight_layout(); fig.savefig(fig_dir / f"transmission_matrix_{case}.png", dpi=130); plt.close(fig)
    print(case, "saved")
