"""Annotate the notebook's unsupervised PCA/K-means exploration with trip systems.

Trip annotations join on the exact UTC event_ts, with event_id checked afterwards.
Recorded system combinations are preserved; absent annotations stay unlisted.
Neither trip labels nor recorded systems enter PCA, scaling, k selection or K-means.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from cavity_signal_comparison import CAVITIES, SEED
from cavity_signal_beam_mode import load_beam_datasets


TABLES = ("metadata", "coverage", "unlisted_trip_events", "unused_cause_events",
          "memberships", "view_summary", "cluster_summary", "system_counts", "combination_counts")


def load_annotated_data(notebook_dir, cause_path=None):
    """Use original Parquet event_ts, never nearest-time or year-based matching."""
    notebook_dir = Path(notebook_dir)
    metadata, waves = load_beam_datasets(notebook_dir / "signal_comparison")
    timestamp_parts = []
    for source in ("DD", "PM"):
        mask = metadata["source"].eq(source)
        table = pd.read_parquet(
            notebook_dir.parent / "notebooks" / f"ess_{source.lower()}" / "cavity_pulses.parquet",
            columns=["event_id", "cycle_id", "cavity", "event_ts"],
        ).iloc[metadata.loc[mask, "source_row"].to_numpy()].copy()
        for key in ("event_id", "cycle_id", "cavity"):
            if not np.array_equal(table[key].to_numpy(), metadata.loc[mask, key].to_numpy()):
                raise ValueError(f"Original event_ts alignment failed: {source}, {key}")
        table["metadata_row"] = metadata.index[mask].to_numpy()
        timestamp_parts.append(table[["metadata_row", "event_ts"]])
    timestamps = pd.concat(timestamp_parts).set_index("metadata_row").sort_index()
    if not timestamps.index.equals(metadata.index):
        raise ValueError("Incomplete original event_ts mapping")
    metadata["event_ts"] = pd.to_datetime(timestamps["event_ts"], utc=True)
    metadata["event_year"] = metadata["event_ts"].dt.year

    if cause_path is None:
        candidates = [
            Path("/workspace/lustre/work/pdl18409/shared/Anomaly_Hunter/trip_label/trip_events.csv"),
            Path("/lustre/work/pdl18409/shared/Anomaly_Hunter/trip_label/trip_events.csv"),
        ]
        cause_path = next((p for p in candidates if p.is_file()), None)
        if cause_path is None:
            raise FileNotFoundError("trip_events.csv was not found at either mounted path")
    causes = pd.read_csv(cause_path)
    causes["event_ts"] = pd.to_datetime(causes["event_ts"], utc=True)
    if causes["event_ts"].isna().any() or causes["event_ts"].duplicated().any():
        raise ValueError("Each recorded trip annotation must have one unique, valid event_ts")
    causes = causes.rename(columns={"event_id": "cause_event_id", "systems": "trip_systems",
                                    "n_labels": "cause_n_labels"})
    if causes["trip_systems"].isna().any() or causes["trip_systems"].str.strip().eq("").any():
        raise ValueError("Empty systems value in trip_events.csv")
    causes["trip_systems"] = causes["trip_systems"].str.strip()
    before_indices = metadata["dataset_index"].to_numpy().copy()
    metadata = metadata.merge(causes, on="event_ts", how="left", sort=False, validate="many_to_one")
    if not np.array_equal(before_indices, metadata["dataset_index"].to_numpy()):
        raise AssertionError("Annotation join changed waveform order")
    matched = metadata["cause_event_id"].notna()
    if not metadata.loc[matched, "event_id"].eq(metadata.loc[matched, "cause_event_id"]).all():
        raise ValueError("Exact timestamp match disagrees with original event_id")
    metadata["trip_cause_status"] = np.where(
        metadata["label"].eq(0), "normal", np.where(matched, "matched", "unlisted"),
    )
    metadata["trip_cause"] = np.where(
        metadata["label"].eq(0), "Normal (DD)",
        metadata["trip_cause_status"].eq("matched").map({True: "", False: "Unlisted trip"}),
    )
    annotated_trip = metadata["label"].eq(1) & matched
    metadata.loc[annotated_trip, "trip_cause"] = metadata.loc[annotated_trip, "trip_systems"]
    metadata["cause_n_labels"] = metadata["cause_n_labels"].astype("Int64")
    metadata["cause_event_id"] = metadata["cause_event_id"].astype("Int64")
    metadata["cavity_short"] = metadata["cavity"].str[-3:]
    metadata["label_name"] = metadata["label"].map({0: "Normal (DD)", 1: "Trip (PM)"})
    coverage_rows = []
    for status, group in metadata.groupby("trip_cause_status", sort=False):
        coverage_rows.append({"status": status, "waveforms": len(group),
                              "captures": len(group[["source", "event_id"]].drop_duplicates())})
    unlisted = metadata.loc[metadata["trip_cause_status"].eq("unlisted"), [
        "source", "event_id", "event_ts", "event_year", "beam_mode", "beam_present", "trig_delay",
    ]].drop_duplicates()
    used_times = metadata.loc[annotated_trip, "event_ts"]
    unused = causes.loc[~causes["event_ts"].isin(used_times)].copy()
    return metadata, waves, pd.DataFrame(coverage_rows), unlisted, unused


def _extract_plateau(wave):
    # Exact rule used in the notebook's existing cluster_extract_plateau().
    flat_mask = np.zeros(len(wave), dtype=bool)
    flat_mask[1:] = np.abs(np.diff(wave)) <= 0.1
    flat_mask &= (wave >= 18) & (wave <= 24)
    edges = np.diff(np.r_[False, flat_mask, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        return None
    best = np.argmax(ends - starts)
    flat = wave[starts[best]:ends[best]]
    if len(flat) < 6:
        return None
    return np.interp(np.linspace(0, 1, 12), np.linspace(0, 1, len(flat)), flat)


def rebuild_cluster_views(metadata, field_waveforms):
    """Reproduce existing notebook clustering when its live variables are absent."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from threadpoolctl import threadpool_limits

    full = np.asarray(field_waveforms, dtype=float)
    if full.shape != (len(metadata), 64) or not np.isfinite(full).all():
        raise ValueError("Expected complete, aligned 64-bin field-amplitude waveforms")
    plateaux = [_extract_plateau(w) for w in full]
    keep = np.array([w is not None for w in plateaux])
    plateau = np.vstack([w for w in plateaux if w is not None])
    plateau_meta = metadata.loc[keep].reset_index(drop=True)
    views = {
        "Whole waveform": {"x": full, "meta": metadata},
        "Plateau": {"x": plateau, "meta": plateau_meta},
        "Centered plateau": {"x": plateau - plateau.mean(axis=1, keepdims=True), "meta": plateau_meta},
    }
    with threadpool_limits(limits=2):
        for name, view in views.items():
            scaler = StandardScaler()
            scaled = scaler.fit_transform(view["x"])
            pca = PCA(n_components=2, svd_solver="full")
            xy = pca.fit_transform(scaled)
            scores, models = {}, {}
            for k in range(2, 9):
                model = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(scaled)
                if len(np.unique(model.labels_)) < 2:
                    continue
                scores[k] = silhouette_score(scaled, model.labels_, sample_size=min(1200, len(scaled)),
                                              random_state=SEED)
                models[k] = model
            best_k = max(scores, key=scores.get)
            view.update(xy=xy, pca=pca, scaler=scaler, model=models[best_k],
                        clusters=models[best_k].labels_ + 1, scores=scores, best_k=best_k)
            print(f"Reproduced {name}: n={len(scaled)}, k={best_k}", flush=True)
    return views


def annotate_cluster_views(views, metadata):
    """Retain existing coordinates and cluster IDs; only append event annotations."""
    indexed = metadata.set_index("dataset_index", drop=False)
    memberships, summaries = [], []
    for name, view in views.items():
        existing = view["meta"]
        annotated = indexed.loc[existing["dataset_index"].to_numpy()].reset_index(drop=True).copy()
        for key in ("cavity", "label"):
            if not np.array_equal(existing[key].to_numpy(), annotated[key].to_numpy()):
                raise ValueError(f"Existing cluster metadata disagrees with waveform keys: {name}, {key}")
        xy, labels = np.asarray(view["xy"]), np.asarray(view["clusters"])
        if xy.shape != (len(annotated), 2) or len(labels) != len(annotated):
            raise ValueError("Existing PCA/cluster arrays are not aligned")
        annotated["representation"], annotated["cluster"] = name, labels
        annotated[["PC1", "PC2"]] = xy
        memberships.append(annotated)
        variance = view["pca"].explained_variance_ratio_
        summaries.append({"representation": name, "n_waveforms": len(annotated),
                          "n_features": view["x"].shape[1], "selected_k": view["best_k"],
                          "silhouette": view["scores"][view["best_k"]],
                          "PC1_variance": variance[0], "PC2_variance": variance[1]})
    return pd.concat(memberships, ignore_index=True), pd.DataFrame(summaries)


def summarize_memberships(memberships):
    summary_rows = []
    for (representation, cluster), group in memberships.groupby(["representation", "cluster"], sort=False):
        trip = group.loc[group["label"].eq(1)]
        normal = group.loc[group["label"].eq(0)]
        summary_rows.append({"representation": representation, "cluster": cluster,
                             "normal_waveforms": len(normal), "trip_waveforms": len(trip),
                             "normal_captures": normal["event_id"].nunique(),
                             "trip_captures": trip["event_id"].nunique(),
                             "matched_trip_waveforms": trip["trip_cause_status"].eq("matched").sum(),
                             "unlisted_trip_waveforms": trip["trip_cause_status"].eq("unlisted").sum()})
    trip = memberships.loc[memberships["label"].eq(1)].copy()
    combinations = trip.groupby(["representation", "cluster", "trip_cause"], sort=False).agg(
        waveforms=("label", "size"), captures=("event_id", "nunique"),
    ).reset_index()
    trip["system"] = trip["trip_cause"].str.split(",")
    exploded = trip.explode("system")
    exploded["system"] = exploded["system"].str.strip()
    count_keys = ["representation", "cluster", "system"]
    waveform_counts = exploded.groupby(count_keys, sort=False).size().rename("waveforms")
    event_counts = exploded.drop_duplicates(count_keys + ["source", "event_id"]).groupby(
        count_keys, sort=False,
    ).size().rename("captures")
    system_counts = pd.concat([waveform_counts, event_counts], axis=1).reset_index()
    return pd.DataFrame(summary_rows), system_counts, combinations


def make_result(notebook_dir, cause_path=None, existing_views=None):
    metadata, waves, coverage, unlisted, unused = load_annotated_data(notebook_dir, cause_path)
    views = existing_views if existing_views is not None else rebuild_cluster_views(metadata, waves["field_amp"])
    memberships, summary = annotate_cluster_views(views, metadata)
    cluster_summary, systems, combinations = summarize_memberships(memberships)
    return {"metadata": metadata, "coverage": coverage, "unlisted_trip_events": unlisted,
            "unused_cause_events": unused, "memberships": memberships, "view_summary": summary,
            "cluster_summary": cluster_summary, "system_counts": systems,
            "combination_counts": combinations}


def save_results(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        result[name].to_csv(output_dir / f"{name}.csv", index=False)


def load_results(output_dir):
    result = {name: pd.read_csv(Path(output_dir) / f"{name}.csv", float_precision="round_trip")
              for name in TABLES}
    for name in ("metadata", "memberships", "unlisted_trip_events", "unused_cause_events"):
        result[name]["event_ts"] = pd.to_datetime(result[name]["event_ts"], utc=True)
    return result


def plot_cause_scatter(result):
    """Same coordinates colored by cluster, recorded trip systems, and actual year."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    memberships, summary = result["memberships"], result["view_summary"]
    known = sorted(result["metadata"].loc[result["metadata"]["trip_cause_status"].eq("matched"), "trip_cause"].unique())
    palette = {cause: plt.get_cmap("tab20")(i % 20) for i, cause in enumerate(known)}
    for _, info in summary.iterrows():
        name = info["representation"]
        group = memberships.loc[memberships["representation"].eq(name)]
        fig, axes = plt.subplots(1, 3, figsize=(19, 8.5))
        for cluster, points in group.groupby("cluster", sort=True):
            axes[0].scatter(points["PC1"], points["PC2"], s=11, alpha=0.5,
                            color=plt.get_cmap("tab10")(int(cluster) - 1), label=f"C{cluster}: {len(points)}")
        normal = group.loc[group["label"].eq(0)]
        axes[1].scatter(normal["PC1"], normal["PC2"], s=9, color="#bdbdbd", alpha=0.28)
        handles = [Line2D([0], [0], color="#bdbdbd", marker="o", linestyle="", label=f"Normal (DD): {len(normal)} waveforms")]
        for cause in known:
            points = group.loc[group["trip_cause"].eq(cause)]
            if points.empty:
                continue
            axes[1].scatter(points["PC1"], points["PC2"], s=32, marker="x", linewidths=1.1,
                            color=palette[cause], alpha=0.85)
            handles.append(Line2D([0], [0], color=palette[cause], marker="x", linestyle="",
                                  label=f"{cause}: {points['event_id'].nunique()} events"))
        unlisted = group.loc[group["trip_cause_status"].eq("unlisted")]
        axes[1].scatter(unlisted["PC1"], unlisted["PC2"], s=32, marker="+", linewidths=1,
                        color="#222222", alpha=0.8)
        handles.append(Line2D([0], [0], color="#222222", marker="+", linestyle="",
                              label=f"Unlisted trip: {unlisted['event_id'].nunique()} events"))
        for cluster, points in group.groupby("cluster", sort=True):
            axes[1].annotate(f"C{cluster}", points[["PC1", "PC2"]].mean().to_numpy(),
                             fontsize=8, bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})
        for year_index, year in enumerate(sorted(group["event_year"].unique())):
            for label, marker in ((0, "o"), (1, "x")):
                points = group.loc[group["event_year"].eq(year) & group["label"].eq(label)]
                axes[2].scatter(points["PC1"], points["PC2"], s=11 if label == 0 else 26,
                                marker=marker, color=plt.get_cmap("tab10")(year_index),
                                alpha=0.35 if label == 0 else 0.85,
                                label=f"{year}, label={label}: {len(points)}")
        for ax, title in zip(axes, ("K-means cluster", "Recorded trip systems (event_ts match)", "Acquisition year / label")):
            ax.set(title=title, xlabel=f"PC1 ({info.PC1_variance:.1%})", ylabel=f"PC2 ({info.PC2_variance:.1%})")
            ax.grid(alpha=0.15)
        axes[0].legend(fontsize=8)
        axes[2].legend(fontsize=8)
        fig.suptitle(f"{name} | n={len(group):,} | k={int(info.selected_k)} | existing PCA coordinates", fontsize=14)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                   ncol=2, fontsize=8, title="Recorded system combinations; multi-system labels retained")
        fig.tight_layout(rect=(0, 0.29, 1, 0.94))
        yield name, fig


def plot_system_heatmaps(result):
    """Count unique trip captures per cluster and recorded system, without cavity duplication."""
    import matplotlib.pyplot as plt

    names = result["view_summary"]["representation"].tolist()
    fig, axes = plt.subplots(1, len(names), figsize=(19, 6), squeeze=False)
    systems = sorted(result["system_counts"]["system"].unique())
    if "Unlisted trip" in systems:
        systems.remove("Unlisted trip")
        systems.append("Unlisted trip")
    maximum = int(result["system_counts"]["captures"].max())
    for ax, name in zip(axes.flat, names):
        selected = result["system_counts"].loc[result["system_counts"]["representation"].eq(name)]
        clusters = sorted(result["cluster_summary"].loc[result["cluster_summary"]["representation"].eq(name), "cluster"])
        table = selected.pivot(index="cluster", columns="system", values="captures").reindex(
            index=clusters, columns=systems,
        ).fillna(0)
        shown = ax.imshow(table.to_numpy(), vmin=0, vmax=maximum, cmap="YlGnBu", aspect="auto")
        for row in range(len(table)):
            for col in range(len(table.columns)):
                value = int(table.iloc[row, col])
                if value:
                    ax.text(col, row, str(value), ha="center", va="center", fontsize=8,
                            color="white" if value > maximum * 0.55 else "black")
        ax.set(title=name, xticks=range(len(systems)), xticklabels=systems,
               yticks=range(len(clusters)), yticklabels=[f"C{c}" for c in clusters])
        ax.tick_params(axis="x", rotation=65, labelsize=8)
    colorbar_ax = fig.add_axes([0.95, 0.35, 0.012, 0.5])
    fig.colorbar(shown, cax=colorbar_ax, label="Unique trip captures per cluster/system")
    fig.suptitle("Trip systems in each cluster | multi-system events contribute to every listed system", fontsize=13)
    fig.subplots_adjust(left=0.04, right=0.92, bottom=0.38, top=0.88, wspace=0.25)
    return fig


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = make_result(root)
    output = root / "signal_comparison" / "trip_cause_pca"
    save_results(result, output)
    print(result["coverage"].to_string(index=False))
    print(result["view_summary"].to_string(index=False))
    print(f"Saved annotated clustering results to {output}")
