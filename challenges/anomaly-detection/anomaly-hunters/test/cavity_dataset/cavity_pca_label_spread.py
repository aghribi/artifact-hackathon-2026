"""Describe Normal/Trip spread inside the existing PCA clusters.

Use the plotted coordinates and assignments verbatim. Variances describe pulse
positions, with a separate centroid for each cluster/label; no model is refitted.
"""
from pathlib import Path

import numpy as np
import pandas as pd


VIEWS = ("Whole waveform", "Plateau")
COLORS = {0: "#3574ae", 1: "#e56b32"}


def make_result(points):
    required = {"representation", "cluster", "label", "PC1", "PC2", "source", "event_id", "dataset_index"}
    if not required.issubset(points.columns):
        raise ValueError(f"Missing membership columns: {sorted(required-set(points.columns))}")
    points = points.loc[points.representation.isin(VIEWS)].copy()
    if set(points.representation) != set(VIEWS):
        raise ValueError("Both existing Whole waveform and Plateau memberships are required")
    if points.duplicated(["representation", "dataset_index"]).any():
        raise ValueError("Repeated pulse in the same representation")
    if not np.isfinite(points[["PC1", "PC2"]].to_numpy()).all() or not points.label.isin([0, 1]).all():
        raise ValueError("Expected finite PCA scores and binary labels")
    rows, comparisons = [], []
    for name in VIEWS:
        view = points.loc[points.representation.eq(name)]
        for cluster, members in view.groupby("cluster", sort=True):
            groups = {}
            for label in (0, 1):
                subset = members.loc[members.label.eq(label)]
                xy = subset[["PC1", "PC2"]].to_numpy()
                n = len(xy)
                mean = xy.mean(axis=0) if n else np.full(2, np.nan)
                cov = np.cov(xy, rowvar=False, ddof=1) if n >= 2 else np.full((2, 2), np.nan)
                captures = len(subset[["source", "event_id"]].drop_duplicates())
                row = {
                    "representation": name, "cluster": int(cluster), "label": label,
                    "label_name": "Normal" if label == 0 else "Trip", "n_waveforms": n,
                    "n_captures": captures, "mean_PC1": mean[0], "mean_PC2": mean[1],
                    "var_PC1": cov[0, 0], "var_PC2": cov[1, 1], "cov_PC1_PC2": cov[0, 1],
                    "std_PC1": np.sqrt(cov[0, 0]), "std_PC2": np.sqrt(cov[1, 1]),
                    "variance_2D": np.trace(cov),
                    "status": "no_waveforms" if n == 0 else "one_waveform" if n == 1 else
                              "single_capture" if captures == 1 else "descriptive",
                }
                rows.append(row)
                groups[label] = row
            normal, trip = groups[0], groups[1]
            variance_ratio = (trip["variance_2D"] / normal["variance_2D"]
                              if normal["variance_2D"] > 0 and trip["n_waveforms"] >= 2 else np.nan)
            comparison = {"representation": name, "cluster": int(cluster)}
            for label_name, row in (("Normal", normal), ("Trip", trip)):
                for column in ("n_waveforms", "n_captures", "var_PC1", "var_PC2", "variance_2D", "status"):
                    comparison[f"{label_name}_{column}"] = row[column]
            comparison["Trip_over_Normal_variance_2D"] = variance_ratio
            comparison["centroid_distance_2D"] = float(np.hypot(
                trip["mean_PC1"]-normal["mean_PC1"], trip["mean_PC2"]-normal["mean_PC2"]))
            comparisons.append(comparison)
    return {"group_spread": pd.DataFrame(rows), "cluster_comparison": pd.DataFrame(comparisons), "points": points}


def save_results(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for table in ("group_spread", "cluster_comparison"):
        result[table].to_csv(output / f"{table}.csv", index=False)


def plot_cluster_details(result, representation):
    """Per-cluster zooms, retaining all points and the original PC directions."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib.lines import Line2D

    points = result["points"].loc[result["points"].representation.eq(representation)]
    stats = result["group_spread"].loc[result["group_spread"].representation.eq(representation)]
    clusters = sorted(points.cluster.unique())
    ncols = 4 if len(clusters) > 6 else 3
    nrows = int(np.ceil(len(clusters)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1*ncols, 4.0*nrows), squeeze=False)
    for ax, cluster in zip(axes.flat, clusters):
        members = points.loc[points.cluster.eq(cluster)]
        origin = members[["PC1", "PC2"]].mean().to_numpy()
        summaries = []
        for label in (0, 1):
            subset = members.loc[members.label.eq(label)]
            row = stats.loc[stats.cluster.eq(cluster) & stats.label.eq(label)].iloc[0]
            name = "Normal" if label == 0 else "Trip"
            summaries.append(f"{name}: n={len(subset)}, captures={int(row.n_captures)}")
            if subset.empty:
                continue
            xy = subset[["PC1", "PC2"]].to_numpy() - origin
            ax.scatter(xy[:, 0], xy[:, 1], s=12 if label == 0 else 26,
                       marker="o" if label == 0 else "x", color=COLORS[label],
                       alpha=.45 if label == 0 else .8, linewidths=.8)
            mean = xy.mean(axis=0)
            ax.scatter(*mean, marker="D", s=34, facecolors="white", edgecolors=COLORS[label], zorder=5)
            if len(xy) >= 2:
                covariance = np.array([[row.var_PC1, row.cov_PC1_PC2], [row.cov_PC1_PC2, row.var_PC2]])
                values, vectors = np.linalg.eigh(covariance)
                values = np.maximum(values, 0.)
                angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
                ellipse = Ellipse(mean, width=2*np.sqrt(values[1]), height=2*np.sqrt(values[0]),
                                  angle=angle, fill=False, edgecolor=COLORS[label], linewidth=1.6, zorder=4)
                ax.add_patch(ellipse)
        ax.set(title=f"C{cluster}\n" + "\n".join(summaries),
               xlabel="PC1 - pooled cluster mean", ylabel="PC2 - pooled cluster mean")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 3), useOffset=False)
        ax.grid(alpha=.2)
        ax.margins(.15)
    for ax in axes.flat[len(clusters):]:
        ax.set_visible(False)
    handles = [Line2D([], [], marker="o" if label == 0 else "x", color=COLORS[label], linestyle="none",
                      label="Normal" if label == 0 else "Trip") for label in (0, 1)]
    handles += [Line2D([], [], marker="D", markerfacecolor="white", color="gray", linestyle="none", label="Label centroid"),
                Line2D([], [], color="gray", label="1-SD covariance ellipse (spread, not a CI)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(f"{representation}: Normal / Trip within each existing cluster\n"
                 "Every point retained; each panel zooms independently; centering does not change variance", fontsize=12)
    fig.tight_layout(rect=(0, .045, 1, .925))
    return fig


def plot_variance_comparison(result):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, name in zip(axes, VIEWS):
        table = result["cluster_comparison"].loc[result["cluster_comparison"].representation.eq(name)]
        x = np.arange(len(table))
        for label, offset in ((0, -.18), (1, .18)):
            label_name = "Normal" if label == 0 else "Trip"
            values = table[f"{label_name}_variance_2D"].to_numpy()
            valid = np.isfinite(values) & (values > 0)
            ax.bar(x[valid]+offset, values[valid], width=.34, color=COLORS[label], alpha=.85)
        ax.set(yscale="log", xticks=x, xticklabels=[f"C{c}" for c in table.cluster],
               title=name, ylabel="Var(PC1) + Var(PC2) [PCA score squared]")
        ax.grid(axis="y", alpha=.2)
        if table.Trip_n_waveforms.eq(0).any():
            ax.text(.02, .98, "C6: no Trip waveforms; variance is NA", transform=ax.transAxes,
                    va="top", fontsize=9)
    fig.legend(handles=[Patch(color=COLORS[label], label="Normal" if label == 0 else "Trip") for label in (0, 1)],
               loc="lower center", ncol=2)
    fig.suptitle("Within-label sample variances (ddof=1); scales differ between representations")
    fig.tight_layout(rect=(0, .06, 1, .94))
    return fig
