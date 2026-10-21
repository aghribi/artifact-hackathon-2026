"""Within-cluster variance of the PCA coordinates already displayed in the notebook.

Use existing coordinates and assignments. No PCA, clustering, or classifier is fitted.
"""
from pathlib import Path

import numpy as np
import pandas as pd


VIEWS = ("Whole waveform", "Plateau", "Centered plateau")


def points_from_views(views, annotated_points=None):
    """Read the exact live scatter coordinates; attach capture IDs when available."""
    parts = []
    for name in VIEWS:
        view = views[name]
        xy = np.asarray(view["xy"], dtype=float)
        part = view["meta"][["dataset_index", "label"]].copy().reset_index(drop=True)
        if len(part) != len(xy) or len(part) != len(view["clusters"]):
            raise ValueError(f"PCA points and memberships disagree: {name}")
        part["representation"] = name
        part["cluster"] = np.asarray(view["clusters"], dtype=int)
        part[["PC1", "PC2"]] = xy
        parts.append(part)
    points = pd.concat(parts, ignore_index=True)
    if annotated_points is not None:
        keys = ["representation", "dataset_index"]
        fields = [c for c in ("source", "event_id", "cavity", "beam_mode", "event_year")
                  if c in annotated_points]
        annotations = annotated_points[keys + fields]
        points = points.merge(annotations, on=keys, how="left", validate="one_to_one")
        if "event_id" in points and points["event_id"].isna().any():
            raise ValueError("Some plotted points have no matching capture metadata")
    return points


def _statistics(group):
    xy = group[["PC1", "PC2"]].to_numpy(dtype=float)
    n = len(xy)
    mean = xy.mean(axis=0) if n else np.full(2, np.nan)
    covariance = np.cov(xy, rowvar=False, ddof=1) if n >= 2 else np.full((2, 2), np.nan)
    if n >= 2:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0)
        direction = eigenvectors[:, -1]
        angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    else:
        eigenvalues, angle = np.full(2, np.nan), np.nan
    captures = (len(group[["source", "event_id"]].drop_duplicates())
                if {"source", "event_id"}.issubset(group) else np.nan)
    return {
        "n_waveforms": n, "n_captures": captures,
        "mean_PC1": mean[0], "mean_PC2": mean[1],
        "var_PC1": covariance[0, 0], "var_PC2": covariance[1, 1],
        "cov_PC1_PC2": covariance[0, 1], "variance_2D": np.trace(covariance),
        "std_PC1": np.sqrt(covariance[0, 0]), "std_PC2": np.sqrt(covariance[1, 1]),
        "principal_variance_major": eigenvalues[-1], "principal_variance_minor": eigenvalues[0],
        "major_axis_angle_degrees": angle,
        "status": ("no_waveforms" if n == 0 else "one_waveform_variance_undefined"
                   if n == 1 else "descriptive_sample_variance"),
    }


def make_result(points):
    required = {"representation", "cluster", "dataset_index", "label", "PC1", "PC2"}
    if not required.issubset(points):
        raise ValueError(f"Missing point columns: {sorted(required - set(points))}")
    points = points.loc[points["representation"].isin(VIEWS)].copy()
    if points.empty or points.duplicated(["representation", "dataset_index"]).any():
        raise ValueError("Expected unique existing pulses within each representation")
    if not np.isfinite(points[["PC1", "PC2"]].to_numpy()).all():
        raise ValueError("PCA coordinates must be finite")
    if not points["label"].isin([0, 1]).all():
        raise ValueError("Expected normal/trip labels")
    rows, label_rows, comparisons = [], [], []
    for name in VIEWS:
        selected = points.loc[points["representation"] == name]
        for cluster, group in selected.groupby("cluster", sort=True):
            rows.append({"representation": name, "cluster": int(cluster),
                         "normal_waveforms": int(group["label"].eq(0).sum()),
                         "trip_waveforms": int(group["label"].eq(1).sum()),
                         **_statistics(group)})
            comparison = {"representation": name, "cluster": int(cluster)}
            statistics = {}
            for label in (0, 1):
                subset = group.loc[group["label"] == label]
                stats = _statistics(subset)
                statistics[label] = stats
                label_name = "Normal" if label == 0 else "Trip"
                label_rows.append({"representation": name, "cluster": int(cluster),
                                   "label": label, "label_name": label_name,
                                   "circle_radius_1sigma": np.sqrt(stats["variance_2D"]), **stats})
                for column in ("n_waveforms", "n_captures", "var_PC1", "var_PC2", "variance_2D", "status"):
                    comparison[f"{label_name}_{column}"] = stats[column]
            normal, trip = statistics[0], statistics[1]
            comparison["Trip_over_Normal_variance_2D"] = (
                trip["variance_2D"] / normal["variance_2D"]
                if normal["n_waveforms"] >= 2 and trip["n_waveforms"] >= 2
                and normal["variance_2D"] > 0 else np.nan
            )
            comparison["centroid_distance_2D"] = (
                float(np.hypot(trip["mean_PC1"] - normal["mean_PC1"],
                               trip["mean_PC2"] - normal["mean_PC2"]))
                if normal["n_waveforms"] and trip["n_waveforms"] else np.nan
            )
            comparisons.append(comparison)
    return {"cluster_variance": pd.DataFrame(rows), "label_variance": pd.DataFrame(label_rows),
            "label_comparison": pd.DataFrame(comparisons), "points": points}


def save_results(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("cluster_variance", "label_variance", "label_comparison"):
        result[name].to_csv(output / f"{name}.csv", index=False)


def plot_cluster_zooms(result, representation):
    """Draw separate Normal/Trip 1-sigma circles in each existing cluster.

    The explicitly defined radial scale is sqrt(Var(PC1) + Var(PC2)), using
    sample variances about each label's own center. This circular summary does
    not imply Gaussian coverage or a confidence region.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    points = result["points"].loc[result["points"]["representation"] == representation]
    table = result["label_variance"].loc[result["label_variance"]["representation"] == representation]
    clusters = sorted(points["cluster"].unique())
    if not clusters:
        raise ValueError(f"No existing clusters for {representation}")
    ncols = min(4 if len(clusters) > 6 else 3, len(clusters))
    nrows = int(np.ceil(len(clusters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.1 * nrows), squeeze=False)
    for ax, cluster in zip(axes.flat, clusters):
        group = points.loc[points["cluster"] == cluster]
        center = group[["PC1", "PC2"]].mean().to_numpy(dtype=float)
        details = []
        for label, color, marker in ((0, "#3574ae", "o"), (1, "#e56b32", "x")):
            stats = table.loc[(table["cluster"] == cluster) & (table["label"] == label)].iloc[0]
            label_name = "Normal" if label == 0 else "Trip"
            variance_text = (f"V1={stats['var_PC1']:.2e}, V2={stats['var_PC2']:.2e}"
                             if stats["n_waveforms"] >= 2 else "variance NA")
            details.append(f"{label_name} n={int(stats['n_waveforms'])}: {variance_text}")
            xy = group.loc[group["label"] == label, ["PC1", "PC2"]].to_numpy() - center
            ax.scatter(xy[:, 0], xy[:, 1], color=color, marker=marker,
                       s=11 if label == 0 else 27, alpha=0.5 if label == 0 else 0.85,
                       linewidths=0.8)
            if stats["n_waveforms"]:
                label_center = stats[["mean_PC1", "mean_PC2"]].to_numpy(dtype=float) - center
                ax.scatter(*label_center, marker="D", s=31, facecolors="white", edgecolors=color, zorder=5)
            if stats["n_waveforms"] >= 2:
                radius = float(stats["circle_radius_1sigma"])
                ax.add_patch(Circle(label_center, radius, facecolor=color, alpha=0.07,
                                    edgecolor="none"))
                ax.add_patch(Circle(label_center, radius, fill=False, edgecolor=color,
                                    linewidth=1.25, linestyle="-" if label == 0 else "--"))
        ax.axhline(0, color="gray", alpha=0.2, linewidth=0.6)
        ax.axvline(0, color="gray", alpha=0.2, linewidth=0.6)
        ax.set_title(f"C{cluster}\n" + "\n".join(details), fontsize=8.5)
        ax.set(xlabel="PC1 - cluster mean", ylabel="PC2 - cluster mean")
        ax.set_aspect("equal", adjustable="datalim")
        ax.ticklabel_format(style="sci", axis="both", scilimits=(-2, 2), useOffset=False)
        ax.grid(alpha=0.15)
    for ax in list(axes.flat)[len(clusters):]:
        ax.set_visible(False)
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color="#3574ae", label="Normal"),
        Line2D([], [], marker="x", linestyle="", color="#e56b32", label="Trip"),
        Line2D([], [], color="#3574ae", label="Normal 1-sigma circle"),
        Line2D([], [], color="#e56b32", linestyle="--", label="Trip 1-sigma circle"),
        Line2D([], [], marker="D", linestyle="", markerfacecolor="white",
               markeredgecolor="#3574ae", label="Normal mean (centroid)"),
        Line2D([], [], marker="D", linestyle="", markerfacecolor="white",
               markeredgecolor="#e56b32", label="Trip mean (centroid)"),
    ], loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"{representation}: separate Normal / Trip variances and centers\n"
                 "Circle radius = sqrt[Var(PC1) + Var(PC2)]; equal axis units; panel zooms differ", fontsize=11)
    fig.tight_layout(rect=(0, 0.13 if nrows == 1 else 0.085, 1, 0.93))
    return fig


def plot_variance_bars(result):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(VIEWS), figsize=(16, 4.8))
    for ax, name in zip(axes, VIEWS):
        table = result["label_comparison"].loc[result["label_comparison"]["representation"] == name]
        x = np.arange(len(table))
        for label, offset, color in (("Normal", -0.18, "#3574ae"), ("Trip", 0.18, "#e56b32")):
            values = table[f"{label}_variance_2D"].to_numpy()
            valid = np.isfinite(values) & (values > 0)
            ax.bar(x[valid] + offset, values[valid], width=0.34, color=color, label=label)
        ax.set(yscale="log", xticks=x, xticklabels=[f"C{int(c)}" for c in table["cluster"]],
               title=name, ylabel="Var(PC1) + Var(PC2) [PCA score squared]")
        unavailable = table.loc[table["Trip_n_waveforms"] < 2, "cluster"].tolist()
        if unavailable:
            ax.text(0.02, 0.98, "Trip variance NA: " + ", ".join(f"C{int(c)}" for c in unavailable),
                    transform=ax.transAxes, va="top", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Normal and Trip measured around their own mean in each existing cluster | ddof=1\n"
                 "Each representation has its own PCA scale; no clusters or PCA coordinates were refitted")
    fig.tight_layout()
    return fig
