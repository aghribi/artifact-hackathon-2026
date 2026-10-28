"""Statistics for the notebook's existing four-cavity PCA/KMeans plots.

Retain the existing coordinates, cluster IDs and k choices. Compute separation
in the full standardized feature space, and only then inspect labels/metadata.
Capture bootstrap intervals keep fitted clusters fixed; they are not a test of
physical causality or a measurement of refitted-cluster uncertainty.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from cavity_signal_comparison import SEED
from cavity_trip_cause_pca import load_annotated_data, rebuild_cluster_views, annotate_cluster_views


TABLES = ("view_metrics", "cluster_metrics", "point_statistics", "initialization_stability",
          "conditional_association", "year_label_counts", "retention")


def _capture_bootstrap(meta, clusters, n_bootstrap=2000):
    """Resample entire captures within each label; retain pulse multiplicities."""
    captures = meta["source"].astype(str) + ":" + meta["event_id"].astype(str)
    table = pd.DataFrame({"capture": captures, "label": meta.label.to_numpy(), "cluster": clusters})
    if table.groupby("capture").label.nunique().gt(1).any():
        raise ValueError("A capture has mixed weak labels")
    ids = sorted(np.unique(clusters))
    rng = np.random.default_rng(SEED)
    boot = {}
    weighted = {}
    for label in (0, 1):
        part = table.loc[table.label.eq(label)]
        counts = pd.crosstab(part.capture, part.cluster).reindex(columns=ids, fill_value=0).to_numpy(float)
        weights = rng.multinomial(len(counts), np.full(len(counts), 1/len(counts)), size=n_bootstrap)
        boot[label] = weights @ counts
        weighted[label] = (counts / counts.sum(axis=1, keepdims=True)).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = boot[1] / (boot[0] + boot[1])
        balanced = weighted[1] / (weighted[0] + weighted[1])
    return ids, shares, balanced


def make_result(root, existing_views=None, n_bootstrap=2000):
    from sklearn.cluster import KMeans
    from sklearn.metrics import (silhouette_samples, davies_bouldin_score,
                                 adjusted_rand_score, adjusted_mutual_info_score, mutual_info_score)
    from threadpoolctl import threadpool_limits

    metadata, waves, *_ = load_annotated_data(root)
    views = existing_views if existing_views is not None else rebuild_cluster_views(metadata, waves["field_amp"])
    annotated, _ = annotate_cluster_views(views, metadata)
    view_rows, cluster_rows, stability_rows, conditional_rows, retention_rows, point_parts = [], [], [], [], [], []
    base_counts = metadata.label.value_counts()
    with threadpool_limits(limits=2):
        for name, view in views.items():
            group = annotated.loc[annotated.representation.eq(name)].reset_index(drop=True).copy()
            labels = np.asarray(view["clusters"])
            x = view["scaler"].transform(view["x"])
            if len(group) != len(x) or not np.array_equal(labels, group.cluster):
                raise ValueError(f"View and annotated memberships disagree: {name}")
            silhouettes = silhouette_samples(x, labels)
            group["silhouette"] = silhouettes
            point_parts.append(group)
            k = int(view["best_k"])
            # Repeat the chosen k only, without any label-based model selection.
            for seed in range(10):
                repeated = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(x)
                stability_rows.append({"representation": name, "seed": seed, "k_fixed": k,
                                       "ARI_vs_existing_clusters": adjusted_rand_score(labels, repeated)})
            for control_columns, control_name in ((["event_year"], "year"),
                                                  (["event_year", "beam_mode", "beam_present"], "year + beam mode + beam presence")):
                conditional_mi = 0.
                for _, stratum in group.groupby(control_columns, sort=False):
                    conditional_mi += len(stratum)/len(group) * mutual_info_score(stratum.label, stratum.cluster) / np.log(2)
                conditional_rows.append({"representation": name, "conditioned_on": control_name,
                                         "label_cluster_MI_bits": mutual_info_score(group.label, labels)/np.log(2),
                                         "conditional_label_cluster_MI_bits": conditional_mi})
            stability = [row["ARI_vs_existing_clusters"] for row in stability_rows if row["representation"] == name]
            view_rows.append({"representation": name, "waveforms": len(group), "k": k,
                              "normal_waveforms": group.label.eq(0).sum(), "trip_waveforms": group.label.eq(1).sum(),
                              "normal_captures": group.loc[group.label.eq(0), "event_id"].nunique(),
                              "trip_captures": group.loc[group.label.eq(1), "event_id"].nunique(),
                              "silhouette_for_k_selection": view["scores"][k],
                              "silhouette_all_points": silhouettes.mean(),
                              "negative_silhouette_fraction": np.mean(silhouettes < 0),
                              "Davies_Bouldin": davies_bouldin_score(x, labels),
                              "PCA_2D_variance": view["pca"].explained_variance_ratio_.sum(),
                              "AMI_label": adjusted_mutual_info_score(group.label, labels),
                              "AMI_cavity": adjusted_mutual_info_score(group.cavity, labels),
                              "AMI_year": adjusted_mutual_info_score(group.event_year, labels),
                              "ARI_year": adjusted_rand_score(group.event_year, labels),
                              "initialization_ARI_mean": np.mean(stability), "initialization_ARI_min": min(stability)})
            ids, bootstrap, balanced = _capture_bootstrap(group, labels, n_bootstrap)
            for position, cluster in enumerate(ids):
                points = group.loc[group.cluster.eq(cluster)]
                normal, trip = points.loc[points.label.eq(0)], points.loc[points.label.eq(1)]
                normal_captures, trip_captures = normal.event_id.nunique(), trip.event_id.nunique()
                supported = normal_captures >= 2 and trip_captures >= 2
                valid = np.isfinite(bootstrap[:, position])
                lo, hi = np.percentile(bootstrap[valid, position], [2.5, 97.5]) if supported else (np.nan, np.nan)
                cluster_rows.append({"representation": name, "cluster": cluster, "waveforms": len(points),
                                     "normal_waveforms": len(normal), "trip_waveforms": len(trip),
                                     "normal_captures": normal_captures, "trip_captures": trip_captures,
                                     "trip_fraction": len(trip)/len(points),
                                     "trip_enrichment_vs_representation": (len(trip)/len(points))/group.label.mean(),
                                     "capture_balanced_trip_fraction": balanced[position],
                                     "bootstrap_CI_low": lo, "bootstrap_CI_high": hi,
                                     "bootstrap_status": "reported" if supported else "fewer_than_two_captures_in_a_label",
                                     "bootstrap_replicates": n_bootstrap, "valid_bootstrap_fraction": valid.mean(),
                                     "silhouette_mean": points.silhouette.mean(),
                                     "silhouette_p10": points.silhouette.quantile(.1),
                                     "silhouette_median": points.silhouette.median(),
                                     "negative_silhouette_fraction": points.silhouette.lt(0).mean()})
            for label in (0, 1):
                kept = group.label.eq(label).sum()
                retention_rows.append({"representation": name, "label": label, "original_waveforms": base_counts[label],
                                       "retained_waveforms": kept, "excluded_waveforms": base_counts[label]-kept,
                                       "retained_fraction": kept/base_counts[label]})
            print(f"Measured {name}: n={len(group)}, exact silhouette={silhouettes.mean():.6f}, seed stability min={min(stability):.4f}", flush=True)
    points = pd.concat(point_parts, ignore_index=True)
    year_counts = points.groupby(["representation", "cluster", "event_year", "label"], sort=False).agg(
        waveforms=("dataset_index", "size"), captures=("event_id", "nunique")
    ).reset_index()
    return {"view_metrics": pd.DataFrame(view_rows), "cluster_metrics": pd.DataFrame(cluster_rows),
            "point_statistics": points, "initialization_stability": pd.DataFrame(stability_rows),
            "conditional_association": pd.DataFrame(conditional_rows), "year_label_counts": year_counts,
            "retention": pd.DataFrame(retention_rows)}


def save_results(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        result[name].to_csv(output_dir / f"{name}.csv", index=False)


def load_results(output_dir):
    return {name: pd.read_csv(Path(output_dir) / f"{name}.csv", float_precision="round_trip") for name in TABLES}


def cache_fresh(root, output_dir):
    output_dir, root = Path(output_dir), Path(root)
    paths = [output_dir / f"{name}.csv" for name in TABLES]
    from cavity_all_cavity_trip_clusters import input_paths
    inputs = input_paths(root) + [Path(__file__), root / "cavity_trip_cause_pca.py", root / "signal_comparison" / "waveforms.npz"]
    return all(p.is_file() for p in paths) and min(p.stat().st_mtime for p in paths) >= max(p.stat().st_mtime for p in inputs)


def plot_silhouettes(result):
    import matplotlib.pyplot as plt
    names = result["view_metrics"].representation.tolist()
    fig, axes = plt.subplots(1, len(names), figsize=(16, 6), squeeze=False)
    for ax, name in zip(axes[0], names):
        points = result["point_statistics"].query("representation == @name")
        bottom, ticks, names_ = 0, [], []
        for cluster, group in points.groupby("cluster"):
            values = np.sort(group.silhouette.to_numpy())
            y = np.arange(bottom, bottom+len(values))
            ax.fill_betweenx(y, 0, values, color=plt.get_cmap("tab10")(int(cluster)-1), alpha=.8)
            ticks.append(bottom+len(values)/2)
            names_.append(f"C{cluster} (n={len(values)})")
            bottom += len(values) + 15
        mean = points.silhouette.mean()
        ax.axvline(mean, color="black", linestyle="--", label=f"Mean={mean:.4f}")
        ax.axvline(0, color="gray", linewidth=.5)
        ax.set(yticks=ticks, yticklabels=names_, xlabel="Silhouette in full standardized feature space",
               title=name, xlim=(min(-.02, points.silhouette.min()-.02), 1.02))
        ax.legend(fontsize=8)
    fig.suptitle("Every pulse evaluated; cluster assignments and PCA coordinates unchanged")
    fig.tight_layout()
    return fig


def plot_associations(result):
    import matplotlib.pyplot as plt
    metrics = result["view_metrics"].set_index("representation")
    columns = ["AMI_label", "AMI_cavity", "AMI_year"]
    values = metrics[columns].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), ["Normal/Trip label", "Cavity", "Acquisition year"])
    ax.set_yticks(range(len(metrics)), metrics.index)
    for i in range(len(metrics)):
        for j in range(3):
            ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", color="white" if values[i,j]>.55 else "black")
    ax.set_title("What do the clusters correspond to? Adjusted mutual information\n1 = identical partition; around 0 = chance-level agreement; negative values are possible")
    fig.tight_layout()
    return fig


def plot_trip_fractions(result):
    import matplotlib.pyplot as plt
    names = result["view_metrics"].representation.tolist()
    fig, axes = plt.subplots(1, len(names), figsize=(16, 5), squeeze=False)
    for ax, name in zip(axes[0], names):
        table = result["cluster_metrics"].query("representation == @name").sort_values("cluster")
        x = np.arange(len(table))
        supported = table.bootstrap_status.eq("reported").to_numpy()
        ax.scatter(x[supported], table.trip_fraction.to_numpy()[supported], color="#d56b32", s=50,
                    label="Pulse Trip fraction; capture-bootstrap 95% CI")
        if (~supported).any():
            ax.scatter(x[~supported], table.trip_fraction.to_numpy()[~supported], facecolors="none", edgecolors="#d56b32", s=65,
                        label="CI not reported: <2 captures in a label")
        for pos, row in enumerate(table.itertuples()):
            if row.bootstrap_status == "reported":
                ax.vlines(pos, row.bootstrap_CI_low, row.bootstrap_CI_high, color="#d56b32")
                ax.hlines([row.bootstrap_CI_low, row.bootstrap_CI_high], pos-.08, pos+.08, color="#d56b32")
        info = result["view_metrics"].query("representation == @name").iloc[0]
        baseline = info.trip_waveforms/info.waveforms
        ax.axhline(baseline, color="gray", linestyle="--", label=f"View baseline={baseline:.1%}")
        ax.set(xticks=x, xticklabels=[f"C{k}" for k in table.cluster], ylim=(-.03, 1.06), title=name, ylabel="Trip waveform fraction")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Resample complete captures within each label; fitted clusters are held fixed")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parent
    output = root / "signal_comparison" / "pca_cluster_statistics"
    result = make_result(root)
    save_results(result, output)
    for name, plotter in (("silhouettes.png", plot_silhouettes), ("label_cavity_year_AMI.png", plot_associations),
                          ("trip_fraction_capture_bootstrap.png", plot_trip_fractions)):
        figure = plotter(result)
        figure.savefig(output/name, dpi=140, bbox_inches="tight")
        plt.close(figure)
    print(result["view_metrics"].round(6).to_string(index=False))
    print(result["conditional_association"].round(8).to_string(index=False))
