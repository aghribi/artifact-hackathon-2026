"""All-cavity, event-level exploration of interpretable trip signatures.

Use the measured change from the earlier half of each PM capture to its final
pulse in each cavity. Never use cause annotations or time-to-trip as features.
All 82 observed cavities are retained. Missing measurements are not zero-filled
for clustering: distances use common finite features, normalized for overlap.
"""
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
BIN_SIGNALS = ("field_amp", "field_phase", "fwd_amp", "refl_amp")
GROUP = ["source", "event_id", "cavity"]
SCALARS = ["field_level", "flat_len", "flat_std_rel", "flat_tilt_rel", "flat_step_max_rel",
           "phase_mean_deg", "phase_std_deg", "phase_tilt_deg", "fwd_flat", "fwd_flat_std_rel",
           "refl_ratio", "field_end_after_rf_off"]
FEATURES = ["field_level_change_rel", "flat_length_change_fraction", "flat_noise_change_rel",
            "flat_tilt_change_rel", "flat_step_change_rel", "field_phase_change_sin",
            "field_phase_change_cos_minus_one", "phase_jitter_change_deg", "phase_tilt_change_deg",
            "forward_level_change_rel", "forward_noise_change_rel", "reflection_ratio_change",
            "field_end_vs_rf_off_change_fraction", "field_shape_change_rel", "missing_pulse"]
FLOORS = dict(zip(FEATURES, [.005, .005, .0001, .0002, .0002, .005, .005, .02, .05,
                            .005, .001, .005, .005, .005, .1]))
TABLES = ("coverage", "cavity_support", "event_metadata", "pulse_metadata", "trip_features",
          "memberships", "selection", "metrics", "cluster_causes", "cluster_drivers", "distance_overlap")
INPUT_COLUMNS = ["event_id", "cycle_id", "event_ts", "cycle_ts", "cavity", "section", "array_size",
                 "cycle_rank_from_last", "cycles_in_event", "beam_mode", "beam_present", "beam_len",
                 "trig_delay", "has_pulse", *SCALARS, *[s + "_bins" for s in BIN_SIGNALS]]


def _cause_path():
    candidates = [Path("/workspace/lustre/work/pdl18409/shared/Anomaly_Hunter/trip_label/trip_events.csv"),
                  Path("/lustre/work/pdl18409/shared/Anomaly_Hunter/trip_label/trip_events.csv")]
    return next(p for p in candidates if p.is_file())


def input_paths(root):
    root = Path(root)
    return [root.parent / "notebooks" / f"ess_{s}" / "cavity_pulses.parquet" for s in ("dd", "pm")] + [_cause_path()]


def load_source_tables(root):
    parts = []
    for source, path in zip(("DD", "PM"), input_paths(root)[:2]):
        frame = pd.read_parquet(path, columns=INPUT_COLUMNS)
        frame["source_row"] = np.arange(len(frame))
        frame["source"] = source
        frame["event_ts"] = pd.to_datetime(frame.event_ts, utc=True)
        frame["cycle_ts"] = pd.to_datetime(frame.cycle_ts, utc=True)
        parts.append(frame)
    full = pd.concat(parts, ignore_index=True)
    if full.duplicated(["source", "event_id", "cycle_id", "cavity"]).any():
        raise ValueError("Duplicate original pulse identity")
    eligible = full.source.eq("DD") | full.cycle_rank_from_last.eq(0)
    labeled = full.loc[eligible].copy()
    labeled["label"] = labeled.source.eq("PM").astype(int)
    labeled["dataset_index"] = np.arange(len(labeled))
    return full, labeled


def pulse_feature_changes(full):
    """Final PM minus an earlier-half reference, with circular phase means.

    Reference cycles satisfy rank_from_last*2 >= cycles_in_event, the existing
    feature extractor's rule. Singleton captures have no reference, stay missing.
    """
    early = full.cycle_rank_from_last * 2 >= full.cycles_in_event
    reference = full.loc[early].groupby(GROUP, sort=False)[SCALARS].median()
    # Replace the scalar phase median by a circular reference.
    phase = np.deg2rad(full.loc[early, "phase_mean_deg"])
    phase_components = full.loc[early, GROUP].copy()
    phase_components["s"], phase_components["c"] = np.sin(phase), np.cos(phase)
    circular = phase_components.groupby(GROUP)[["s", "c"]].mean()
    reference["phase_mean_deg"] = np.rad2deg(np.arctan2(circular.s, circular.c))
    requested = full.loc[full.source.eq("PM") & full.cycle_rank_from_last.eq(0)].copy()
    idx = pd.MultiIndex.from_frame(requested[GROUP])
    ref = reference.reindex(idx).reset_index(drop=True)
    requested = requested.reset_index(drop=True)
    delta = requested[SCALARS] - ref
    out = pd.DataFrame(index=requested.index)
    for target, original in (("field_level_change_rel", "field_level"),
                             ("forward_level_change_rel", "fwd_flat")):
        denominator = ref[original].where(ref[original].abs() > 1e-6)
        out[target] = delta[original] / denominator.abs()
    out["flat_length_change_fraction"] = delta.flat_len / requested.array_size
    for name, original in (("flat_noise_change_rel", "flat_std_rel"), ("flat_tilt_change_rel", "flat_tilt_rel"),
                           ("flat_step_change_rel", "flat_step_max_rel"), ("phase_jitter_change_deg", "phase_std_deg"),
                           ("phase_tilt_change_deg", "phase_tilt_deg"), ("forward_noise_change_rel", "fwd_flat_std_rel"),
                           ("reflection_ratio_change", "refl_ratio")):
        out[name] = delta[original]
    phase_delta = np.deg2rad(delta.phase_mean_deg)
    out["field_phase_change_sin"] = np.sin(phase_delta)
    out["field_phase_change_cos_minus_one"] = np.cos(phase_delta) - 1
    out["field_end_vs_rf_off_change_fraction"] = delta.field_end_after_rf_off / requested.array_size
    # Waveform changes retain missing/no-pulse records, without requiring a flat top.
    binned = np.stack(full.field_amp_bins).astype(float)
    codes, uniques = pd.factorize(pd.MultiIndex.from_frame(full[GROUP]))
    total, count = np.zeros((len(uniques), 64)), np.zeros(len(uniques))
    np.add.at(total, codes[early], binned[early])
    np.add.at(count, codes[early], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        earlier_waves = total / count[:, None]
    selected_codes = pd.Series(np.arange(len(uniques)), index=uniques).reindex(idx).to_numpy()
    current = np.stack(requested.field_amp_bins).astype(float)
    shape = np.sqrt(np.mean((current - earlier_waves[selected_codes]) ** 2, axis=1))
    out["field_shape_change_rel"] = shape / ref.field_level.abs().where(ref.field_level.abs() > 1e-6)
    out["missing_pulse"] = (~requested.has_pulse.astype(bool)).astype(float)
    out = out[FEATURES].replace([np.inf, -np.inf], np.nan)
    if out.notna().sum(axis=1).lt(2).any():
        # Keep the row and its missing reference in the audit; do not fabricate a change.
        print("PM rows with fewer than two usable change features:", int(out.notna().sum(axis=1).lt(2).sum()), flush=True)
    keys = ["source", "event_id", "event_ts", "cycle_id", "cycle_ts", "cavity", "section",
            "source_row", "beam_mode", "beam_present", "beam_len", "trig_delay"]
    return pd.concat([requested[keys], out], axis=1)


def annotate_events(trip):
    causes = pd.read_csv(_cause_path())
    causes["event_ts"] = pd.to_datetime(causes.event_ts, utc=True)
    if causes.event_ts.duplicated().any() or causes.event_ts.isna().any():
        raise ValueError("Trip annotations require unique valid event_ts values")
    rows = []
    for (source, event_id), group in trip.groupby(["source", "event_id"], sort=False):
        if group.event_ts.nunique() != 1:
            raise ValueError("event_id maps to multiple event_ts values")
        # Metadata states can differ for a few cavities whose last available cycle is older.
        # Record all of this; use the most frequent state solely for conditional diagnostics.
        state = group.groupby(["beam_mode", "beam_present", "beam_len"], dropna=False).size().sort_values(ascending=False)
        mode, present, length = state.index[0]
        rows.append({"source": source, "event_id": event_id, "event_ts": group.event_ts.iloc[0],
                     "event_date": str(group.event_ts.iloc[0].date()),
                     "event_year": group.event_ts.dt.year.iloc[0], "beam_mode": mode,
                     "beam_present": present, "beam_len": length, "n_metadata_states": len(state),
                     "n_cycle_ids": group.cycle_id.nunique(), "n_cavities": group.cavity.nunique(),
                     "max_last_cycle_age_s": (group.cycle_ts.max() - group.cycle_ts.min()).total_seconds()})
    events = pd.DataFrame(rows).merge(causes.rename(columns={"event_id": "cause_event_id", "systems": "trip_cause"}),
                                      on="event_ts", how="left", validate="one_to_one", sort=False)
    matched = events.cause_event_id.notna()
    if not events.loc[matched, "event_id"].eq(events.loc[matched, "cause_event_id"]).all():
        raise ValueError("Exact event_ts join disagrees with event_id")
    events["cause_status"] = np.where(matched, "matched", "unlisted")
    events["trip_cause"] = events.trip_cause.fillna("Unlisted trip")
    events["cause_event_id"] = events.cause_event_id.astype("Int64")
    events["n_labels"] = events.n_labels.astype("Int64")
    return events


def cluster_trip_events(trip, events, all_cavities):
    from sklearn.cluster import AgglomerativeClustering, HDBSCAN
    from sklearn.manifold import MDS
    from sklearn.metrics import adjusted_mutual_info_score, nan_euclidean_distances, silhouette_score
    from threadpoolctl import threadpool_limits

    index = pd.MultiIndex.from_product([all_cavities, FEATURES], names=["cavity", "feature"])
    wide = trip.set_index(["event_id", "cavity"])[FEATURES].unstack("cavity")
    wide = wide.swaplevel(0, 1, axis=1).reindex(columns=index).reindex(events.event_id)
    x = wide.to_numpy(float)
    finite_count = np.isfinite(x).sum(axis=1)
    coverage = finite_count / x.shape[1]
    # Require broad machine coverage for a machine-wide distance; retain partial events separately.
    usable = (events.n_cavities.to_numpy() >= .75 * len(all_cavities)) & (coverage >= .60)
    selected = events.loc[usable].copy().reset_index(drop=True)
    raw = x[usable]
    floors = np.array([FLOORS[f] for _, f in index])
    with np.errstate(invalid="ignore"):
        center = np.nanmedian(raw, axis=0)
        scale = np.maximum(np.nanstd(raw, axis=0), floors)
    center = np.nan_to_num(center)
    scale = np.where(np.isfinite(scale), scale, floors)
    z = np.clip((raw - center) / scale, -8, 8)
    distances = nan_euclidean_distances(z)
    np.fill_diagonal(distances, 0)
    if not np.isfinite(distances).all():
        raise ValueError("Two selected captures have no common finite features")
    available = np.isfinite(raw).astype(float)
    overlap = available @ available.T / x.shape[1]
    overlap_rows = []
    for i in range(len(selected)):
        for j in range(i):
            overlap_rows.append({"event_id_a": selected.event_id.iloc[i], "event_id_b": selected.event_id.iloc[j],
                                 "common_feature_fraction": overlap[i, j], "distance": distances[i, j]})
    candidates = []
    with threadpool_limits(limits=2):
        for k in range(2, min(6, len(selected) // 3) + 1):
            labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(distances)
            _, counts = np.unique(labels, return_counts=True)
            candidates.append({"k": k, "silhouette": silhouette_score(distances, labels, metric="precomputed"),
                               "min_cluster_events": counts.min(), "eligible": counts.min() >= 3, "labels": labels + 1})
        eligible = [c for c in candidates if c["eligible"]]
        # Do not force tiny outlier groups into a cause partition.
        chosen = max(eligible, key=lambda c: c["silhouette"]) if eligible else None
        hierarchical = chosen["labels"] if chosen else np.zeros(len(selected), dtype=int)
        density = HDBSCAN(min_cluster_size=4, min_samples=3, metric="precomputed").fit_predict(distances)
        density = np.where(density < 0, -1, density + 1)
        xy = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED,
                 n_init=4, normalized_stress="auto").fit_transform(distances)
    memberships, metrics, compositions, drivers = [], [], [], []
    strata = selected.groupby(["event_date", "beam_mode", "beam_present"], sort=False).indices
    annotated = selected.cause_status.eq("matched").to_numpy()
    causes = selected.trip_cause.to_numpy()
    for method, labels in (("Hierarchical", hierarchical), ("HDBSCAN", density)):
        membership = selected.copy()
        membership["method"], membership["cluster"] = method, labels
        membership[["MDS1", "MDS2"]] = xy
        membership["feature_coverage"] = coverage[usable]
        membership["status"] = "clustered"
        memberships.append(membership)
        observed_ami = adjusted_mutual_info_score(causes[annotated], labels[annotated])
        rng = np.random.default_rng(SEED)
        permuted_ami = []
        for _ in range(999):
            permuted = causes.copy()
            # Unlisted causes never enter AMI or the permutation pool.
            for indices in strata.values():
                known = indices[annotated[indices]]
                permuted[known] = rng.permutation(permuted[known])
            permuted_ami.append(adjusted_mutual_info_score(permuted[annotated], labels[annotated]))
        pvalue = (1 + np.sum(np.asarray(permuted_ami) >= observed_ami - 1e-12)) / 1000
        metrics.append({"method": method, "events": len(selected), "annotated_events": int(annotated.sum()),
                        "n_clusters": len(set(labels) - {-1, 0}), "noise_events": int((labels == -1).sum()),
                        "cause_AMI": observed_ami, "conditional_permutation_p": pvalue,
                        "permutation_controls": "event_date,beam_mode,beam_present",
                        "year_AMI": adjusted_mutual_info_score(selected.event_year, labels),
                        "beam_mode_AMI": adjusted_mutual_info_score(selected.beam_mode, labels)})
        for cluster in np.unique(labels):
            mask = labels == cluster
            part = selected.loc[mask]
            for cause, number in part.trip_cause.value_counts().items():
                compositions.append({"method": method, "cluster": cluster, "trip_cause": cause, "events": number})
            if cluster <= 0 or not (~mask).any():
                continue
            with np.errstate(invalid="ignore"):
                median = pd.DataFrame(raw[mask]).median().to_numpy()
                other = pd.DataFrame(raw[~mask]).median().to_numpy()
            effect = (median - other) / scale
            # Interpret only features observed in at least half of both comparison groups.
            sufficient = (np.isfinite(raw[mask]).mean(axis=0) >= .5) & (np.isfinite(raw[~mask]).mean(axis=0) >= .5)
            effect[~sufficient] = np.nan
            top = np.argsort(-np.nan_to_num(abs(effect), nan=-1), kind="stable")[:10]
            for rank, j in enumerate(top, 1):
                if not np.isfinite(effect[j]):
                    continue
                cavity, feature = index[j]
                drivers.append({"method": method, "cluster": cluster, "rank": rank, "cavity": cavity,
                                "feature": feature, "cluster_median_change": median[j], "other_trip_median_change": other[j],
                                "scaled_difference": effect[j], "events_with_feature": int(np.isfinite(raw[mask, j]).sum())})
    partial = events.loc[~usable].copy()
    partial["method"], partial["cluster"], partial["status"] = "Not clustered", -2, "partial_capture"
    partial["feature_coverage"] = coverage[~usable]
    partial[["MDS1", "MDS2"]] = np.nan
    memberships.append(partial)
    metrics = pd.DataFrame(metrics)
    # Holm correction for the two methods in this fixed comparison.
    order = np.argsort(metrics.conditional_permutation_p.to_numpy())
    corrected = np.maximum.accumulate(metrics.conditional_permutation_p.to_numpy()[order] * np.arange(len(order), 0, -1))
    metrics.loc[metrics.index[order], "conditional_p_holm"] = np.minimum(corrected, 1)
    selection = pd.DataFrame([{k: v for k, v in c.items() if k != "labels"} for c in candidates])
    selection["selected"] = selection.k.eq(chosen["k"]) if chosen else False
    return {"memberships": pd.concat(memberships, ignore_index=True), "selection": selection, "metrics": metrics,
            "cluster_causes": pd.DataFrame(compositions), "cluster_drivers": pd.DataFrame(drivers),
            "distance_overlap": pd.DataFrame(overlap_rows)}


def make_result(root):
    full, labeled = load_source_tables(root)
    trip = pulse_feature_changes(full)
    events = annotate_events(trip)
    cavities = sorted(labeled.cavity.unique())
    result = cluster_trip_events(trip, events, cavities)
    support_rows = []
    for cavity, part in labeled.groupby("cavity", sort=True):
        row = {"cavity": cavity, "section": part.section.iloc[0]}
        for label, name in ((0, "normal"), (1, "trip")):
            group = part.loc[part.label.eq(label)]
            row[name + "_waveform_sets"] = len(group)
            row[name + "_captures"] = group.event_id.nunique()
        changes = trip.loc[trip.cavity.eq(cavity)]
        row["trip_with_shape_reference"] = changes.field_shape_change_rel.notna().sum()
        analyzed = result["memberships"].query("method == 'HDBSCAN'").event_id
        row["trip_in_main_analysis"] = changes.event_id.isin(analyzed).sum()
        support_rows.append(row)
    coverage_rows = []
    for label, name in ((0, "Normal (DD all pulses)"), (1, "Trip (PM last per cavity)")):
        group = labeled.loc[labeled.label.eq(label)]
        row = {"label": label, "population": name, "waveform_sets": len(group), "captures": group.event_id.nunique(),
               "cavities": group.cavity.nunique()}
        for signal in BIN_SIGNALS:
            row[signal + "_valid"] = group[signal + "_bins"].map(
                lambda a: isinstance(a, (list, np.ndarray)) and len(a) == 64 and np.isfinite(a).all()).sum()
        coverage_rows.append(row)
    result.update(coverage=pd.DataFrame(coverage_rows), cavity_support=pd.DataFrame(support_rows),
                  event_metadata=events, trip_features=trip,
                  pulse_metadata=labeled.drop(columns=[s + "_bins" for s in BIN_SIGNALS]))
    return result


def save_results(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        result[name].to_csv(output_dir / f"{name}.csv", index=False)


def load_results(output_dir):
    return {name: pd.read_csv(Path(output_dir) / f"{name}.csv", float_precision="round_trip") for name in TABLES}


def plot_scatter(result, method="HDBSCAN"):
    import matplotlib.pyplot as plt
    membership = result["memberships"].query("method == @method")
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.3))
    for cluster, part in membership.groupby("cluster"):
        name = f"C{cluster}" if cluster > 0 else ("Noise" if cluster == -1 else "Unpartitioned")
        axes[0].scatter(part.MDS1, part.MDS2, label=f"{name} (n={len(part)})", s=50)
    palette = plt.get_cmap("tab20")
    for i, (cause, part) in enumerate(membership.groupby("trip_cause")):
        axes[1].scatter(part.MDS1, part.MDS2, label=f"{cause} (n={len(part)})", s=50,
                        marker="x" if cause == "Unlisted trip" else "o",
                        color="black" if cause == "Unlisted trip" else palette(i))
    for year, part in membership.groupby("event_year"):
        axes[2].scatter(part.MDS1, part.MDS2, label=f"{year} (n={len(part)})", s=50)
    for ax, title in zip(axes, (f"{method} clusters", "Recorded systems (annotation only)", "Acquisition year")):
        ax.set(title=title, xlabel="MDS1", ylabel="MDS2")
        if ax is not axes[1]:
            ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=.2)
    fig.suptitle("All-cavity within-capture changes; one point = one trip capture\n"
                 "MDS is a display of common-feature distances; clustering uses the full distance matrix")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(.5, .01))
    fig.tight_layout(rect=(0, .18, 1, .90))
    return fig


def plot_cause_counts(result, method="HDBSCAN"):
    import matplotlib.pyplot as plt
    table = result["cluster_causes"].query("method == @method").pivot(
        index="cluster", columns="trip_cause", values="events").fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(13, max(4, 1.2 * len(table))))
    ax.imshow(table, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=55, ha="right", fontsize=9)
    ax.set_yticks(range(len(table)), [f"C{k}" if k > 0 else ("Noise" if k == -1 else "Unpartitioned") for k in table.index])
    for i in range(len(table)):
        for j in range(len(table.columns)):
            count = table.iloc[i, j]
            ax.text(j, i, str(count), ha="center", va="center", color="white" if count > table.to_numpy().max() / 2 else "black")
    ax.set_title(f"{method}: recorded trip systems per cluster (capture counts)")
    fig.tight_layout()
    return fig


def plot_cavity_changes(result, feature="field_level_change_rel", method="HDBSCAN"):
    import matplotlib.pyplot as plt
    membership = result["memberships"].query("method == @method").sort_values(["cluster", "event_id"])
    cavities = result["cavity_support"].cavity.tolist()
    matrix = result["trip_features"].pivot(index="event_id", columns="cavity", values=feature).reindex(
        index=membership.event_id, columns=cavities)
    values = matrix.to_numpy()
    limit = max(np.nanpercentile(abs(values), 95), FLOORS[feature])
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#4b4b4b")
    fig, ax = plt.subplots(figsize=(22, max(8, len(matrix) * .23)))
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(cavities)), [c.replace("RFC:RFS-DIG-", "/") for c in cavities], rotation=90, fontsize=7)
    ax.set_yticks(range(len(membership)), [f"{'C' + str(r.cluster) if r.cluster > 0 else 'Noise'} | {r.event_id} | {r.trip_cause}" for r in membership.itertuples()], fontsize=7)
    ax.set_title(f"{feature}: final PM pulse minus earlier-half reference\nDark gray = unavailable; near white = small change; colors clipped at 95th percentile for display only")
    fig.colorbar(image, ax=ax, shrink=.5, label="relative change" if feature.endswith("rel") else "change (feature units)")
    fig.tight_layout()
    return fig


def plot_top_features(result, method="HDBSCAN"):
    import matplotlib.pyplot as plt
    drivers = result["cluster_drivers"].query("method == @method and rank <= 2")
    if drivers.empty:
        return None
    choices = drivers[["cavity", "feature"]].drop_duplicates().reset_index(drop=True)
    membership = result["memberships"].query("method == @method")
    fig, axes = plt.subplots(len(choices), 1, figsize=(13, max(4, len(choices) * 2.4)), squeeze=False)
    palette = {c: plt.get_cmap("tab20")(i % 20) for i, c in enumerate(sorted(membership.trip_cause.unique()))}
    for ax, row in zip(axes[:, 0], choices.itertuples()):
        part = membership.merge(result["trip_features"].query("cavity == @row.cavity")[["event_id", row.feature]], on="event_id")
        for i, (cluster, group) in enumerate(part.groupby("cluster")):
            # Deterministic horizontal offsets prevent hiding event counts.
            offsets = np.linspace(-.15, .15, len(group))
            ax.scatter(i + offsets, group[row.feature], c=[palette[c] for c in group.trip_cause], s=35, alpha=.8)
        clusters = sorted(part.cluster.unique())
        ax.set_xticks(range(len(clusters)), [f"C{k}" if k > 0 else "Noise" for k in clusters])
        ax.axhline(0, color="gray", linewidth=.7)
        ax.set_title(f"{row.cavity} | {row.feature}", fontsize=10)
        ax.set_ylabel("change vs early PM")
        ax.grid(axis="y", alpha=.2)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color, label=c) for c, color in palette.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, .94))
    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parent
    output = root / "signal_comparison" / "all_cavity_trip_clusters"
    results = make_result(root)
    save_results(results, output)
    for method in ("Hierarchical", "HDBSCAN"):
        for suffix, plotter in (("scatter", plot_scatter), ("cause_counts", plot_cause_counts)):
            figure = plotter(results, method)
            figure.savefig(output / f"{method.lower()}_{suffix}.png", dpi=140, bbox_inches="tight")
            plt.close(figure)
    for feature in ("field_level_change_rel", "phase_jitter_change_deg", "flat_step_change_rel"):
        figure = plot_cavity_changes(results, feature)
        figure.savefig(output / f"{feature}.png", dpi=140, bbox_inches="tight")
        plt.close(figure)
    figure = plot_top_features(results)
    if figure is not None:
        figure.savefig(output / "cluster_feature_profiles.png", dpi=140, bbox_inches="tight")
        plt.close(figure)
    print(results["coverage"].to_string(index=False))
    print(results["metrics"].to_string(index=False))
    print(results["selection"].to_string(index=False))
