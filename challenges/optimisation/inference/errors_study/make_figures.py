"""Figures and summary tables for the error study (reads results/*.jsonl)."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import analysis_utils as au
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = au.HERE
BPM_LEVELS = [0, 50, 100, 200]
QUAD_LEVELS = [50, 100, 200, 300]


def p90(values) -> float:
    return float(np.nanpercentile(values, 90))


def ensure_reftune_reference() -> None:
    path = au.RESULTS / "meta_reftune.json"
    if path.exists():
        return
    import error_study as es
    import run_scan

    context = es.build_context(run_scan.LATTICES["reftune"])
    path.write_text(
        json.dumps(
            {
                "ip": {k: float(v) for k, v in context.reference_ip.items()},
                "optics": {k: float(v[-1]) for k, v in context.design_optics.items()},
            },
            indent=1,
        )
    )


def line_with_band(
    ax, frame, x_column, group_column, value_column, labels, color_offset=0
):
    """Median (solid) and 90th percentile (dashed) of `value_column` against `x_column`."""
    for index, level in enumerate(labels):
        subset = frame[frame[group_column] == level]
        stats = subset.groupby(x_column)[value_column].agg(["median", p90])
        color = f"C{index + color_offset}"
        ax.plot(
            stats.index, stats["median"], "o-", color=color, ms=3, label=f"{level:g}"
        )
        ax.plot(stats.index, stats["p90"], "--", color=color)


def save(fig, name: str) -> None:
    fig.savefig(HERE / name, dpi=200)
    plt.close(fig)
    print("wrote", name)


def fig_hist(main: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8, 4.5), layout="constrained", sharex="col")
    bins_bpm = np.linspace(0, 1500, 40)
    bins_quad = np.linspace(0, 1500, 40)
    for column, (bins, key) in enumerate(
        ((bins_bpm, "after_bpm_rms"), (bins_quad, "after_quad_rms"))
    ):
        for level in BPM_LEVELS:
            subset = main[(main.quad_rms_um == 100) & (main.bpm_rms_um == level)]
            axes[0, column].hist(
                subset[key], bins=bins, histtype="step", label=f"BPM {level} um"
            )
        for level in QUAD_LEVELS:
            subset = main[(main.quad_rms_um == level) & (main.bpm_rms_um == 100)]
            axes[1, column].hist(
                subset[key], bins=bins, histtype="step", label=f"quad {level} um"
            )
    axes[0, 0].set_title("BPM trajectory rms, after (quad 100 um)")
    axes[0, 1].set_title("beam rms in quads, after (quad 100 um)")
    axes[1, 0].set_title("BPM trajectory rms, after (BPM 100 um)")
    axes[1, 1].set_title("beam rms in quads, after (BPM 100 um)")
    for ax in axes[1]:
        ax.set_xlabel("rms of true x,y centroid (um, combined)")
    for ax in axes[:, 0]:
        ax.set_ylabel("seeds")
    for ax in axes.flat:
        ax.legend()
    save(fig, "fig_hist_residual.png")


def fig_residual(main: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.8), layout="constrained", sharey="col")
    columns = (
        ("after_bpm_rms", "BPM rms (um)"),
        ("after_quad_rms", "in-quad rms (um)"),
        ("after_dpos", "IP centroid shift (um)"),
    )
    for column, (key, label) in enumerate(columns):
        line_with_band(
            axes[0, column], main, "quad_rms_um", "bpm_rms_um", key, BPM_LEVELS
        )
        line_with_band(
            axes[1, column],
            main[main.quad_rms_um > 0],
            "bpm_rms_um",
            "quad_rms_um",
            key,
            QUAD_LEVELS,
        )
        axes[0, column].set_xlabel("quad offset rms (um)")
        axes[1, column].set_xlabel("BPM offset rms (um)")
        axes[0, column].set_ylabel(label)
        axes[1, column].set_ylabel(label)
    axes[0, 0].legend(title="BPM rms (um)", fontsize=5.5, title_fontsize=6)
    axes[1, 0].legend(title="quad rms (um)", fontsize=5.5, title_fontsize=6)
    save(fig, "fig_residual_vs_errors.png")


def fig_optics(main: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8, 4.5), layout="constrained")
    panels = (
        ("after_beatbeat_ip_pct", "beta-beat at IP (%)"),
        ("after_beatbeat_max_pct", "max beta-beat (%)"),
        ("after_dD_abs", "|dD| at IP (mm)"),
        ("after_dD_max", "max |dD| (mm)"),
    )
    main = main.copy()
    main["after_dD_abs"] = (
        np.maximum(main.after_dD_x.abs(), main.after_dD_y.abs()) * 1e3
    )
    main["after_dD_max"] = np.maximum(main.after_dD_max_x, main.after_dD_max_y) * 1e3
    for ax, (key, label) in zip(axes.flat, panels):
        line_with_band(ax, main, "quad_rms_um", "bpm_rms_um", key, BPM_LEVELS)
        ax.set_xlabel("quad offset rms (um)")
        ax.set_ylabel(label)
        ax.set_yscale("log")
    axes[0, 0].legend(title="BPM rms (um)", fontsize=5.5, title_fontsize=6)
    save(fig, "fig_optics_change.png")


def fig_size(main: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.8), layout="constrained")
    main = main.copy()
    for key in ("sigma_x", "sigma_y", "eps_y"):
        main[f"abs_{key}"] = main[f"after_{key}_pct"].abs()
    for column, (key, label) in enumerate(
        (
            ("abs_sigma_x", "|sigma_x change| (%)"),
            ("abs_sigma_y", "|sigma_y change| (%)"),
            ("abs_eps_y", "|eps_y change| (%)"),
        )
    ):
        line_with_band(
            axes[0, column], main, "quad_rms_um", "bpm_rms_um", key, BPM_LEVELS
        )
        line_with_band(
            axes[1, column],
            main[main.quad_rms_um > 0],
            "bpm_rms_um",
            "quad_rms_um",
            key,
            QUAD_LEVELS,
        )
        for row, xlabel in ((0, "quad offset rms (um)"), (1, "BPM offset rms (um)")):
            axes[row, column].set_xlabel(xlabel)
            axes[row, column].set_ylabel(label)
            axes[row, column].set_yscale("log")
    axes[0, 0].legend(title="BPM rms (um)", fontsize=5.5, title_fontsize=6)
    axes[1, 0].legend(title="quad rms (um)", fontsize=5.5, title_fontsize=6)
    save(fig, "fig_ip_size_change.png")


def fig_kicks(main: pd.DataFrame, meta: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.6), layout="constrained")
    line_with_band(
        axes[0], main, "quad_rms_um", "bpm_rms_um", "max_kick_used_urad", BPM_LEVELS
    )
    axes[0].set_xlabel("quad offset rms (um)")
    axes[0].set_ylabel("max |kick| (urad)")
    axes[0].legend(title="BPM rms (um)", fontsize=5.5, title_fontsize=6)
    ranges = np.array(
        [v for values in meta["corrector_ranges_urad"].values() for v in values]
    )
    axes[1].hist(np.sort(ranges[ranges >= 100]), bins=np.logspace(2, 4, 15))
    axes[1].set_xscale("log")
    axes[1].set_xlabel("observed corrector range, planes in use (urad)")
    axes[1].set_ylabel("planes")
    save(fig, "fig_kicks.png")


def fig_side(frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.2), layout="constrained", sharey=True)
    order = [
        ("main", "baseline"),
        ("no_launch", "no launch error"),
        ("big_launch", "launch 1 mm, 100 urad"),
        ("no_jitter", "no shot jitter"),
        ("shots10", "10 shots averaged"),
        ("cutoff_1e-2", "SVD cutoff 1e-2 (6 modes)"),
        ("cutoff_5e-5", "SVD cutoff 5e-5 (51 modes)"),
        ("delta_0p2pct", "launch dp/p 0.2 %"),
        ("target_design", "target = model centroid orbit"),
        ("used_correctors", "no FEA/VBC correctors"),
        ("reftune", "reference tune"),
        ("reftune_target_design", "ref. tune, model orbit"),
    ]
    panels = (
        ("after_quad_rms", "in-quad rms (um)"),
        ("after_dpos", "IP centroid shift (um)"),
        ("after_size_growth_pct", "IP size change (%)"),
    )
    positions = np.arange(len(order))
    for ax, (key, label) in zip(axes, panels):
        for offset, bpm_level in enumerate((0, 100)):
            medians, tops = [], []
            for config, _ in order:
                frame = frames[config]
                subset = frame[
                    (frame.quad_rms_um == 100) & (frame.bpm_rms_um == bpm_level)
                ]
                medians.append(subset[key].median())
                tops.append(p90(subset[key]))
            medians, tops = np.array(medians), np.array(tops)
            ax.barh(
                positions + (offset - 0.5) * 0.38,
                medians,
                0.38,
                color=f"C{offset}",
                label=f"BPM {bpm_level} um",
            )
            ax.plot(tops, positions + (offset - 0.5) * 0.38, "k|", ms=4)
        ax.set_xlabel(label)
        if key == "after_size_growth_pct":
            ax.set_xscale("log")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([label for _, label in order])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    save(fig, "fig_side_scans.png")


def summary_tables(main: pd.DataFrame) -> str:
    rows = []
    keys = {
        "bpm_rms": "after_bpm_rms",
        "quad_rms": "after_quad_rms",
        "dpos": "after_dpos",
        "dang": "after_dang",
        "sigma_pct": "after_size_growth_pct",
        "beat_ip": "after_beatbeat_ip_pct",
        "beat_max": "after_beatbeat_max_pct",
        "kick": "max_kick_used_urad",
    }
    for (quad, bpm), group in main.groupby(["quad_rms_um", "bpm_rms_um"]):
        row = {"quad_um": quad, "bpm_um": bpm}
        for name, column in keys.items():
            row[f"{name}_med"] = group[column].median()
            row[f"{name}_p90"] = p90(group[column])
        row["lost_frac"] = float((group.after_ip_transmission < 0.999).mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "summary_main.csv", index=False, float_format="%.4g")
    return au.to_markdown(table, index=False)


def main() -> None:
    ensure_reftune_reference()
    reference = au.reference_tables()
    frames = {}
    for config in (
        "main",
        "big_launch",
        "no_launch",
        "no_jitter",
        "shots10",
        "cutoff_5e-5",
        "cutoff_1e-2",
        "delta_0p2pct",
        "reftune",
        "target_design",
        "reftune_target_design",
        "used_correctors",
    ):
        frames[config] = au.add_derived(
            au.load(config),
            reference["reftune" if config.startswith("reftune") else "design"],
        )
    meta = json.loads((au.RESULTS / "meta.json").read_text())
    main_frame = frames["main"]
    fig_hist(main_frame)
    fig_residual(main_frame)
    fig_optics(main_frame)
    fig_size(main_frame)
    fig_kicks(main_frame, meta)
    fig_side(frames)
    (HERE / "summary_main.md").write_text(summary_tables(main_frame))
    for name, frame in frames.items():
        if name == "main":
            continue
        group = frame.groupby(["quad_rms_um", "bpm_rms_um"])[
            [
                "after_bpm_rms",
                "after_quad_rms",
                "after_dpos",
                "after_dang",
                "after_size_growth_pct",
                "after_beatbeat_ip_pct",
                "max_kick_used_urad",
            ]
        ].agg(["median", p90])
        (HERE / f"summary_{name}.md").write_text(au.to_markdown(group))


if __name__ == "__main__":
    main()
