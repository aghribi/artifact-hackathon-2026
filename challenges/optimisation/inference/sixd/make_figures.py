"""Figures for the README (small, default colour cycle)."""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import common
import matplotlib.pyplot as plt
import numpy as np
import observability as obs
import reconstruct as rec

FIG = common.HERE / "figures"
FIG.mkdir(exist_ok=True)


def system_for(model, sigma, settings, voltages=(-5e6, 0.0, 5e6)):
    screens = model.inventory.screens
    after = [
        s
        for s in screens
        if model.element_index[s] > model.element_index["cla_s07_dia_tdc_01"]
    ]
    maps = []
    for k1 in settings:
        model.set_quad_k1(k1)
        for v in voltages:
            m = obs.maps_for(model, screens, v)
            maps += [m[s] for s in screens if v == 0.0 or s in after]
    model.restore_nominal_quads()
    return obs.scaled_system(maps, sigma)


def fig_observability() -> None:
    _, model, beam = common.make_pair(particles=20000)
    sigma = common.covariance(beam)
    base = system_for(model, sigma, rec.quad_settings(model, "baseline"))
    scan = system_for(model, sigma, rec.quad_settings(model, "scan"))
    off_only = system_for(
        model, sigma, rec.quad_settings(model, "baseline"), voltages=(0.0,)
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), layout="constrained")
    for label, system in (
        ("screens, TDC off", off_only),
        ("screens, TDC off/+/-", base),
        ("+ 2 optics settings", scan),
    ):
        sv = np.linalg.svd(system, compute_uv=False)
        axes[0].semilogy(np.arange(1, 22), sv / sv[0] + 1e-18, "o-", ms=3, label=label)
    axes[0].set_xlabel("index")
    axes[0].set_ylabel("sing. value / largest")
    axes[0].set_ylim(1e-8, 2)
    axes[0].legend(fontsize=5)
    positions = np.arange(21)
    width = 0.4
    unc_base = obs.summarise(base)["uncertainty"]
    unc_scan = obs.summarise(scan)["uncertainty"]
    axes[1].bar(positions - width / 2, unc_base, width, label="screens only")
    axes[1].bar(positions + width / 2, unc_scan, width, label="+ 2 optics settings")
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(
        [obs.name_of(c).strip("<>").replace(" ", ",") for c in range(21)],
        rotation=90,
        fontsize=5,
    )
    axes[1].set_ylabel("1-sigma, scaled")
    axes[1].legend(fontsize=5, loc="upper right")
    fig.savefig(FIG / "observability.png", dpi=200)


def fig_scan() -> None:
    history = common.read_json(common.HERE / "optics_scan.json")["history"]
    fig, ax = plt.subplots(figsize=(4.0, 2.5), layout="constrained")
    for key in ("eps_x", "eps_y", "eps_z", "sigma_tau", "sigma_delta", "chirp"):
        ax.semilogy(
            range(1, len(history) + 1), [h[key] for h in history], "o-", ms=3, label=key
        )
    ax.set_xlabel("extra optics settings (S07 quads)")
    ax.set_ylabel("relative 1-sigma")
    ax.legend(fontsize=5, ncol=2)
    fig.savefig(FIG / "optics_scan.png", dpi=200)


def fig_demo() -> None:
    labels = ["baseline", "q01q02", "scan"]
    fig, ax = plt.subplots(figsize=(4.0, 2.5), layout="constrained")
    keys = [
        "eps_x (nm, geometric)",
        "eps_y (nm, geometric)",
        "sigma_z (mm)",
        "sigma_delta (%)",
        "chirp h (1/m) <tau delta>/<tau tau>",
    ]
    short = ["eps_x", "eps_y", "sigma_z", "sigma_delta", "chirp"]
    width = 0.2
    truth = None
    for n, label in enumerate(labels):
        try:
            data = common.read_pickle(common.HERE / f"demo_results_{label}.pkl")
        except FileNotFoundError:
            continue
        sigma_true = data["true"]
        truth = rec.summary(sigma_true)
        reconstructed = rec.summary(data[label]["nonlinear"])
        ratio = [reconstructed[k] / truth[k] for k in keys]
        ax.bar(np.arange(len(keys)) + n * width, ratio, width, label=label)
    ax.axhline(1.0, color="k", lw=0.5)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(keys)) + width)
    ax.set_xticklabels(short, rotation=30)
    ax.set_ylabel("reconstructed / true")
    ax.legend(fontsize=5)
    fig.savefig(FIG / "demo_ratio.png", dpi=200)


if __name__ == "__main__":
    which = sys.argv[1:] or ["obs", "scan", "demo"]
    if "obs" in which:
        fig_observability()
    if "scan" in which:
        fig_scan()
    if "demo" in which:
        fig_demo()
