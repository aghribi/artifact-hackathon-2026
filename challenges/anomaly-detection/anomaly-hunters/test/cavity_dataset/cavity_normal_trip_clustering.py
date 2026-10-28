"""Unsupervised clustering of every labeled cavity-pulse, including Normal.

One point is one cavity on one pulse: 35,414 DD + 4,809 final PM rows.
Labels, source, timestamps, year, cavity ID and causes never enter the fit.
They are appended afterwards for interpretation. KMeans fits every row;
only the silhouette calculation uses a fixed random subset for speed.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from cavity_kickoff_models import load_data, pulse_inputs


SEED = 42
TABLES = ("support", "feature_missingness", "model_selection", "view_summary", "memberships",
          "cluster_composition", "cluster_context", "cause_composition", "cluster_feature_profiles")
VIEWS = ("Waveforms", "Waveforms 80% + parameters 20%")


def make_result(root, output_dir):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
    from threadpoolctl import threadpool_limits

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, waves, prior = load_data(root)
    frame, waveform_columns = pulse_inputs(metadata, waves)
    metadata = metadata.reset_index(drop=True)
    if len(metadata) != 40223 or metadata.dataset_index.nunique() != 40223:
        raise ValueError("Expected all 40,223 labeled cavity-pulses, without aggregation")
    metadata["label_name"] = metadata.label.map({0: "Normal (DD)", 1: "Trip (PM last)"})
    causes = prior["event_metadata"].set_index("event_id").trip_cause
    metadata["trip_cause"] = metadata.event_id.map(causes).fillna("Unlisted trip")
    metadata.loc[metadata.label.eq(0), "trip_cause"] = "Normal (DD)"
    missingness = frame[waveform_columns].isna().sum().rename("missing_rows").rename_axis("feature").reset_index()
    support = metadata.groupby(["label", "label_name"], sort=True).agg(
        points=("dataset_index", "size"), captures=("capture", "nunique"), cavities=("cavity", "nunique")
    ).reset_index()
    selection_rows, summary_rows, membership_parts, composition_rows, profile_rows = [], [], [], [], []
    bundles = {}
    with threadpool_limits(limits=2):
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        imputed = imputer.fit_transform(frame[waveform_columns])
        scaler = StandardScaler()
        standardized = scaler.fit_transform(imputed)
        # No label-dependent normalization, no plateau rejection, no class resampling.
        wave_pca = PCA(n_components=.95, svd_solver="full")
        waveform_space = wave_pca.fit_transform(standardized)
        wave_rms = np.sqrt(np.mean(np.sum(waveform_space ** 2, axis=1)))
        waveform_space /= max(wave_rms, 1e-12)

        parameter_columns = ["beam_present", "beam_len", "trig_delay"]
        parameter_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        parameter_scaler = StandardScaler()
        parameters = parameter_scaler.fit_transform(parameter_imputer.fit_transform(frame[parameter_columns]))
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        categories = encoder.fit_transform(frame[["beam_mode"]])
        context = np.c_[parameters, categories]
        context_center = context.mean(axis=0)
        context -= context_center
        context_rms = np.sqrt(np.mean(np.sum(context ** 2, axis=1)))
        context /= max(context_rms, 1e-12)
        spaces = {VIEWS[0]: waveform_space,
                  VIEWS[1]: np.c_[np.sqrt(.8) * waveform_space, np.sqrt(.2) * context]}
        for view_name, space in spaces.items():
            best_score, best_model = -np.inf, None
            for k in range(2, 9):
                model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(space)
                counts = np.bincount(model.labels_, minlength=k)
                # Same random sample for every candidate. Fit/assignment still use all rows.
                try:
                    score = silhouette_score(space, model.labels_, sample_size=1500, random_state=SEED)
                except ValueError:
                    score = np.nan  # e.g. a tiny group absent from the diagnostic subset
                selection_rows.append({"representation": view_name, "k": k, "silhouette": score,
                                       "fit_points": len(space), "silhouette_sample_points": min(1500, len(space)),
                                       "min_cluster_points": counts.min()})
                if np.isfinite(score) and score > best_score:
                    best_score, best_model = score, model
                print(f"{view_name}: k={k}, all {len(space):,} points fitted, silhouette={score:.4f}", flush=True)
            if best_model is None:
                raise ValueError("Silhouette subset did not support any candidate; inspect model_selection")
            projection = PCA(n_components=2, svd_solver="full")
            xy = projection.fit_transform(space)
            labels = best_model.labels_ + 1
            selected = metadata[["dataset_index", "capture", "source", "event_id", "cycle_id", "cavity", "section",
                                 "label", "label_name", "event_ts", "event_year", "beam_mode", "beam_present",
                                 "beam_len", "trig_delay", "trip_cause"]].copy()
            selected["representation"], selected["cluster"] = view_name, labels
            selected[["PC1", "PC2"]] = xy
            membership_parts.append(selected)
            summary_rows.append({"representation": view_name, "points": len(space),
                                 "normal_points": metadata.label.eq(0).sum(), "trip_points": metadata.label.eq(1).sum(),
                                 "captures": metadata.capture.nunique(), "cavities": metadata.cavity.nunique(),
                                 "waveform_features": len(waveform_columns), "waveform_PCs": waveform_space.shape[1],
                                 "waveform_variance_retained": wave_pca.explained_variance_ratio_.sum(),
                                 "cluster_space_dimensions": space.shape[1], "selected_k": best_model.n_clusters,
                                 "silhouette": best_score, "PC1_variance": projection.explained_variance_ratio_[0],
                                 "PC2_variance": projection.explained_variance_ratio_[1],
                                 "label_ARI": adjusted_rand_score(metadata.label, labels),
                                 "year_AMI": adjusted_mutual_info_score(metadata.event_year, labels),
                                 "beam_mode_AMI": adjusted_mutual_info_score(metadata.beam_mode, labels),
                                 "cavity_AMI": adjusted_mutual_info_score(metadata.cavity, labels)})
            for cluster, group in selected.groupby("cluster", sort=True):
                normal, trip = group.loc[group.label.eq(0)], group.loc[group.label.eq(1)]
                composition_rows.append({"representation": view_name, "cluster": cluster, "points": len(group),
                                         "normal_points": len(normal), "trip_points": len(trip),
                                         "trip_fraction": len(trip) / len(group),
                                         "normal_captures": normal.capture.nunique(), "trip_captures": trip.capture.nunique(),
                                         "cavities": group.cavity.nunique()})
                indices = group.index.to_numpy()
                mean = standardized[indices].mean(axis=0)
                for rank, j in enumerate(np.argsort(-abs(mean))[:8], 1):
                    profile_rows.append({"representation": view_name, "cluster": cluster, "rank": rank,
                                         "feature": waveform_columns[j], "standardized_cluster_mean": mean[j],
                                         "cluster_mean": imputed[indices, j].mean(), "overall_mean": scaler.mean_[j]})
            bundles[view_name] = {"model": best_model, "projection": projection}
    memberships = pd.concat(membership_parts, ignore_index=True)
    context_counts = memberships.groupby(["representation", "cluster", "event_year", "beam_mode", "beam_present", "section"],
                                          sort=False).size().rename("points").reset_index()
    cause_counts = memberships.groupby(["representation", "cluster", "trip_cause"], sort=False).agg(
        points=("dataset_index", "size"), captures=("capture", "nunique")
    ).reset_index()
    selection = pd.DataFrame(selection_rows)
    summary = pd.DataFrame(summary_rows)
    chosen = summary.set_index("representation").selected_k
    selection["selected"] = selection.k.eq(selection.representation.map(chosen))
    result = {"support": support, "feature_missingness": missingness, "model_selection": selection,
              "view_summary": summary, "memberships": memberships, "cluster_composition": pd.DataFrame(composition_rows),
              "cluster_context": context_counts, "cause_composition": cause_counts,
              "cluster_feature_profiles": pd.DataFrame(profile_rows)}
    for name in TABLES:
        result[name].to_csv(output_dir / f"{name}.csv", index=False)
    joblib.dump({"views": bundles, "waveform_columns": waveform_columns, "imputer": imputer, "scaler": scaler,
                 "wave_pca": wave_pca, "wave_rms": wave_rms, "parameter_columns": parameter_columns,
                 "parameter_imputer": parameter_imputer, "parameter_scaler": parameter_scaler, "encoder": encoder,
                 "context_center": context_center, "context_rms": context_rms}, output_dir / "clustering_models.joblib", compress=3)
    manifest = {"implementation": Path(__file__).stat().st_mtime_ns,
                "feature_implementation": (Path(root) / "cavity_kickoff_models.py").stat().st_mtime_ns,
                "inputs": {str(p): p.stat().st_mtime_ns for p in [Path(root).parent / "notebooks" / f"ess_{s}" / "cavity_pulses.parquet" for s in ("dd", "pm")]},
                "cause_metadata": (Path(root) / "signal_comparison" / "all_cavity_trip_clusters" / "event_metadata.csv").stat().st_mtime_ns,
                "points_per_representation": len(metadata), "seed": SEED}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return result


def load_results(output_dir):
    return {name: pd.read_csv(Path(output_dir) / f"{name}.csv", float_precision="round_trip") for name in TABLES}


def cache_fresh(root, output_dir):
    output_dir, root = Path(output_dir), Path(root)
    if not all((output_dir / f"{name}.csv").is_file() for name in TABLES) or not (output_dir / "manifest.json").is_file():
        return False
    manifest = json.loads((output_dir / "manifest.json").read_text())
    inputs = {str(p): p.stat().st_mtime_ns for p in [root.parent / "notebooks" / f"ess_{s}" / "cavity_pulses.parquet" for s in ("dd", "pm")]}
    return (manifest["implementation"] == Path(__file__).stat().st_mtime_ns and manifest["inputs"] == inputs and
            manifest["feature_implementation"] == (root / "cavity_kickoff_models.py").stat().st_mtime_ns and
            manifest["cause_metadata"] == (root / "signal_comparison" / "all_cavity_trip_clusters" / "event_metadata.csv").stat().st_mtime_ns)


def plot_all_points(result, view_name):
    import matplotlib.pyplot as plt
    selected = result["memberships"].query("representation == @view_name")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    settings = [("cluster", "KMeans clusters", "tab10"), ("label_name", "Normal and Trip", None),
                ("beam_mode", "Beam mode", "tab10"), ("event_year", "Acquisition year", "tab10"),
                ("section", "Cavity section", "tab10")]
    for ax, (column, title, cmap) in zip(axes.flat, settings):
        for i, (category, group) in enumerate(selected.groupby(column, sort=True)):
            if column == "label_name":
                color = "#3678ad" if group.label.iloc[0] == 0 else "#db6b29"
                size, alpha = (2, .13) if group.label.iloc[0] == 0 else (5, .5)
            else:
                color, size, alpha = plt.get_cmap(cmap)(i % 10), 3, .25
            ax.scatter(group.PC1, group.PC2, color=color, s=size, alpha=alpha,
                       linewidths=0, label=f"{category} (n={len(group):,})", rasterized=True)
        ax.set(title=title, xlabel="PC1", ylabel="PC2")
        legend = ax.legend(fontsize=7, loc="best", markerscale=3)
        for handle in getattr(legend, "legend_handles", getattr(legend, "legendHandles", [])):
            handle.set_alpha(1.)
        ax.grid(alpha=.15)
    ax = axes[1, 2]
    image = ax.hexbin(selected.PC1, selected.PC2, gridsize=70, mincnt=1, bins="log", cmap="viridis")
    if int(image.get_array().sum()) != len(selected):
        raise AssertionError("Density plot failed to count every point")
    ax.set(title=f"Density: all {len(selected):,} points counted", xlabel="PC1", ylabel="PC2")
    fig.colorbar(image, ax=ax, label="Points per hexagon (log color scale)")
    fig.suptitle(f"{view_name}\n1 point = 1 cavity-pulse; Normal=35,414, Trip=4,809; no plotting subsample", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93))
    return fig


def plot_composition(result):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for i, view_name in enumerate(VIEWS):
        table = result["cluster_composition"].query("representation == @view_name")
        x = np.arange(len(table))
        axes[i, 0].bar(x, table.normal_points, color="#3678ad", label="Normal")
        axes[i, 0].bar(x, table.trip_points, bottom=table.normal_points, color="#db6b29", label="Trip")
        axes[i, 1].bar(x, table.trip_fraction, color="#db6b29")
        axes[i, 1].axhline(4809/40223, color="gray", linestyle="--", label="Overall Trip fraction")
        for ax, ylabel in zip(axes[i], ("Cavity-pulse count", "Trip fraction")):
            ax.set(title=view_name, ylabel=ylabel)
            ax.set_xticks(x, [f"C{k}" for k in table.cluster])
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=.15)
        axes[i, 1].set_ylim(0, 1)
    fig.suptitle("Normal/Trip labels are used only after unsupervised cluster fitting")
    fig.tight_layout(rect=(0, 0, 1, .95))
    return fig


def plot_causes(result):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    causes = sorted(set(result["memberships"].trip_cause) - {"Normal (DD)"})
    palette = {cause: plt.get_cmap("tab20")(i % 20) for i, cause in enumerate(causes)}
    palette["Unlisted trip"] = "black"
    for ax, view_name in zip(axes, VIEWS):
        table = result["memberships"].query("representation == @view_name")
        normal = table.loc[table.label.eq(0)]
        ax.scatter(normal.PC1, normal.PC2, s=2, alpha=.12, color="gray", label="Normal (DD)", linewidths=0)
        for cause, group in table.loc[table.label.eq(1)].groupby("trip_cause", sort=True):
            ax.scatter(group.PC1, group.PC2, color=palette[cause], s=6, alpha=.6,
                       linewidths=.7 if cause == "Unlisted trip" else 0,
                       marker="x" if cause == "Unlisted trip" else "o", label=cause)
        ax.set(title=view_name, xlabel="PC1", ylabel="PC2")
        ax.grid(alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, markerscale=2)
    fig.suptitle("Every Normal waveform remains in the background; all 4,809 Trip waveforms are annotated by recorded systems")
    fig.tight_layout(rect=(0, .16, 1, .93))
    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parent
    output = root / "signal_comparison" / "normal_trip_clustering"
    results = make_result(root, output)
    for i, name in enumerate(VIEWS):
        fig = plot_all_points(results, name)
        fig.savefig(output / f"all_points_{i+1}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    for filename, plotter in (("cluster_normal_trip_counts.png", plot_composition), ("trip_causes_with_normal.png", plot_causes)):
        fig = plotter(results)
        fig.savefig(output / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(results["view_summary"].to_string(index=False))
    print(results["cluster_composition"].to_string(index=False))
