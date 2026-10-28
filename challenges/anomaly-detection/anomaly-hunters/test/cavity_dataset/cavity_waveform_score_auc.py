"""ROC-AUC of fixed waveform deviation scores, without fitting a classifier.

Reference envelopes, phase centers, medians and RF windows use reference normal
captures only. All cavities of a capture share the same held-out fold. Matching
uses beam_mode, beam_len, trig_delay and beam_present, with separate cavity
references. No trip-driven score, bin, direction or hyperparameter selection.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from cavity_signal_comparison import SEED, SIGNALS, SIGNAL_NAMES, CAVITIES
from cavity_signal_change import CONDITION_COLUMNS, _envelope, _normal_coordinates, _rf_window
from cavity_signal_beam_mode import load_beam_datasets


TABLES = ("support", "reference_audit", "scores", "fold_metrics", "condition_summary",
          "beam_summary", "bin_fold_metrics", "bin_summary", "roc_curves")


def _capture_weights(frame):
    counts = frame.groupby(["source", "event_id"])["label"].transform("size")
    return 1.0 / counts.to_numpy()


def evaluate_score_auc(metadata, signals, max_splits=5):
    """Fixed score: fraction of bins outside a reference normal 95% envelope.

    A separate per-bin score is absolute distance from the normal median, using
    circular angular distance for phase. High scores always mean more deviation;
    AUCs below 0.5 are retained. Primary waveform scores use the full RF-on window
    determined only from reference normal amplitudes, not the explored bins 7-14.
    ROC metrics give each test capture equal total weight within a cavity.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedGroupKFold

    common = np.logical_and.reduce([np.isfinite(signals[s]).all(axis=1) for s in SIGNALS])
    meta = metadata.loc[common].reset_index(drop=True).copy()
    waves = {s: np.asarray(signals[s][common], dtype=float) for s in SIGNALS}
    if meta[CONDITION_COLUMNS].isna().any().any():
        raise ValueError("Missing beam matching conditions")
    if not meta["dataset_index"].is_unique:
        raise ValueError("Duplicate pulse indices")
    groups = meta["source"].astype(str) + ":" + meta["event_id"].astype(str)
    support_rows, audit_rows, score_rows, metric_rows, bin_rows, curve_rows = [], [], [], [], [], []
    fpr_grid = np.linspace(0, 1, 201)

    for condition_id, (conditions, condition) in enumerate(
        meta.groupby(CONDITION_COLUMNS, sort=True, dropna=False), 1,
    ):
        context = dict(zip(CONDITION_COLUMNS, conditions))
        context["condition_id"] = condition_id
        indices = condition.index.to_numpy()
        local_y = condition["label"].to_numpy()
        local_groups = groups.iloc[indices]
        support = {**context, "n_splits": 0, "status": "excluded", "reason": ""}
        counts = []
        for cavity in CAVITIES:
            cavity_frame = condition.loc[condition["cavity"].eq(cavity)]
            counts.extend(cavity_frame.loc[cavity_frame["label"].eq(label), "event_id"].nunique()
                          for label in (0, 1))
        support.update(normal_pulses=int((local_y == 0).sum()), trip_pulses=int((local_y == 1).sum()),
                       normal_events=condition.loc[condition["label"].eq(0), "event_id"].nunique(),
                       trip_events=condition.loc[condition["label"].eq(1), "event_id"].nunique(),
                       minimum_normal_events_per_cavity=min(counts[::2]),
                       minimum_trip_events_per_cavity=min(counts[1::2]))
        folds = None
        for n_splits in range(min(max_splits, *counts), 1, -1):
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            candidate = list(splitter.split(np.zeros((len(indices), 1)), local_y, local_groups))
            valid = True
            for tr, te in candidate:
                for cavity in CAVITIES:
                    train = condition.iloc[tr].query("cavity == @cavity")
                    test = condition.iloc[te].query("cavity == @cavity")
                    if not train["label"].eq(0).any() or test["label"].nunique() != 2:
                        valid = False
            if valid:
                folds = candidate
                support.update(n_splits=n_splits, status="evaluated")
                break
        if folds is None:
            support["reason"] = ("Need at least two independent captures per label in every cavity; "
                                 "every test fold must include both labels")
            support_rows.append(support)
            continue
        support_rows.append(support)

        for fold, (local_train, local_test) in enumerate(folds, 1):
            test_ids = indices[local_test]
            reference_ids = indices[local_train][local_y[local_train] == 0]
            if set(groups.iloc[test_ids]) & set(groups.iloc[reference_ids]):
                raise AssertionError("A test capture occurs in the normal reference")
            for cavity in CAVITIES:
                reference = reference_ids[meta.iloc[reference_ids]["cavity"].eq(cavity).to_numpy()]
                test = test_ids[meta.iloc[test_ids]["cavity"].eq(cavity).to_numpy()]
                test_frame = meta.iloc[test]
                y = test_frame["label"].to_numpy()
                weights = _capture_weights(test_frame)
                combined = np.concatenate([reference, test])
                reference_mask = np.arange(len(combined)) < len(reference)
                rf_window = _rf_window(waves, combined, reference_mask)
                if not rf_window.any():
                    raise ValueError(f"Empty normal RF-on window: {context}, {cavity}, fold {fold}")
                common_context = {**context, "cavity": cavity, "fold": fold}
                reference_events = groups.iloc[reference].nunique()
                for role, rows in (("reference", reference), ("test", test)):
                    audit = meta.iloc[rows][["dataset_index", "source", "event_id", "label", "cavity"]].copy()
                    for key, value in context.items():
                        audit[key] = value
                    audit["fold"], audit["role"] = fold, role
                    audit_rows.append(audit)
                for signal in SIGNALS:
                    phase = signal.endswith("_phase")
                    coordinates, _ = _normal_coordinates(waves[signal][combined], reference_mask, phase)
                    normal = coordinates[reference_mask]
                    test_values = coordinates[~reference_mask]
                    low, high = _envelope(normal)
                    outside = (test_values < low) | (test_values > high)
                    median = np.median(normal, axis=0)
                    distance = test_values - median
                    if phase:
                        distance = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(distance))))
                    distance = np.abs(distance)
                    for window_name, window in (
                        ("rf_on_reference", rf_window), ("whole_record", np.ones(len(rf_window), bool)),
                    ):
                        score = outside[:, window].mean(axis=1)
                        fpr, tpr, _ = roc_curve(y, score, sample_weight=weights)
                        metrics = {
                            **common_context, "signal": signal, "window": window_name,
                            "score_definition": "outside_normal_95pct_fraction",
                            "n_reference_normal_pulses": len(reference),
                            "n_reference_normal_events": reference_events,
                            "n_test_normal_events": test_frame.loc[test_frame["label"].eq(0), "event_id"].nunique(),
                            "n_test_trip_events": test_frame.loc[test_frame["label"].eq(1), "event_id"].nunique(),
                            "n_test": len(test), "n_bins": int(window.sum()),
                            "roc_auc": roc_auc_score(y, score, sample_weight=weights),
                            "roc_auc_pulse_weighted": roc_auc_score(y, score),
                            "average_precision": average_precision_score(y, score, sample_weight=weights),
                            "recall_at_5pct_fpr": float(tpr[fpr <= 0.05].max()),
                        }
                        metric_rows.append(metrics)
                        scored = test_frame[["dataset_index", "source", "event_id", "label"]].copy()
                        for key, value in common_context.items():
                            scored[key] = value
                        scored["signal"], scored["window"] = signal, window_name
                        scored["score"], scored["capture_weight"] = score, weights
                        score_rows.append(scored)
                        grid_tpr = np.interp(fpr_grid, fpr, tpr)
                        grid_tpr[0], grid_tpr[-1] = tpr[fpr == 0].max(), 1.0
                        curve_rows.append(pd.DataFrame({
                            **common_context, "signal": signal, "window": window_name,
                            "fpr": fpr_grid, "tpr": grid_tpr,
                        }))
                    for bin_index in range(distance.shape[1]):
                        bin_rows.append({
                            **common_context, "signal": signal, "bin": bin_index + 1,
                            "rf_on_reference": bool(rf_window[bin_index]),
                            "roc_auc": roc_auc_score(y, distance[:, bin_index], sample_weight=weights),
                            "score_definition": "absolute_distance_from_normal_median",
                        })
        print(f"Evaluated fixed waveform scores: {context}, {len(folds)} event-grouped folds", flush=True)

    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise ValueError("No matched acquisition condition supports held-out evaluation")
    condition_keys = [*CONDITION_COLUMNS, "condition_id", "cavity", "signal", "window"]
    condition_summary = metrics.groupby(condition_keys, sort=False).agg(
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        recall_at_5pct_fpr_mean=("recall_at_5pct_fpr", "mean"),
        recall_at_5pct_fpr_std=("recall_at_5pct_fpr", "std"),
        n_splits=("fold", "nunique"),
        minimum_reference_normal_events=("n_reference_normal_events", "min"),
    ).reset_index()
    beam_summary = condition_summary.groupby(["beam_mode", "beam_present", "signal", "window"], sort=False).agg(
        roc_auc_macro=("roc_auc_mean", "mean"),
        cavity_auc_min=("roc_auc_mean", "min"), cavity_auc_max=("roc_auc_mean", "max"),
        recall_at_5pct_fpr_macro=("recall_at_5pct_fpr_mean", "mean"),
        matched_condition_cavity_cells=("condition_id", "size"),
        minimum_reference_normal_events=("minimum_reference_normal_events", "min"),
    ).reset_index()
    bin_metrics = pd.DataFrame(bin_rows)
    bin_summary = bin_metrics.groupby(
        [*CONDITION_COLUMNS, "condition_id", "cavity", "signal", "bin"], sort=False,
    ).agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
          rf_on_fold_fraction=("rf_on_reference", "mean")).reset_index()
    scores = pd.concat(score_rows, ignore_index=True)
    if scores.duplicated(["dataset_index", "signal", "window"]).any():
        raise AssertionError("Repeated held-out pulse scores")
    return {"support": pd.DataFrame(support_rows), "reference_audit": pd.concat(audit_rows, ignore_index=True),
            "scores": scores, "fold_metrics": metrics, "condition_summary": condition_summary,
            "beam_summary": beam_summary, "bin_fold_metrics": bin_metrics,
            "bin_summary": bin_summary, "roc_curves": pd.concat(curve_rows, ignore_index=True)}


def save_results(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        result[name].to_csv(output_dir / f"{name}.csv", index=False)


def load_results(output_dir):
    return {name: pd.read_csv(Path(output_dir) / f"{name}.csv", float_precision="round_trip")
            for name in TABLES}


def cavity_sensitivity_report(result, metadata, window="rf_on_reference"):
    """All condition/cavity/signal cells, retaining unavailable cells as NaN."""
    conditions = result["support"]
    selected = result["condition_summary"].loc[
        result["condition_summary"]["window"].eq(window)
    ]
    rows = []
    for _, condition in conditions.iterrows():
        matched = np.logical_and.reduce([
            metadata[column].eq(condition[column]).to_numpy()
            for column in CONDITION_COLUMNS
        ])
        for cavity in CAVITIES:
            pulses = metadata.loc[matched & metadata["cavity"].eq(cavity).to_numpy()]
            counts = {}
            for label, name in ((0, "normal"), (1, "trip")):
                part = pulses.loc[pulses["label"].eq(label)]
                counts[f"{name}_pulses"] = len(part)
                counts[f"{name}_events"] = len(part[["source", "event_id"]].drop_duplicates())
            for signal in SIGNALS:
                measured = selected.loc[
                    selected["condition_id"].eq(condition["condition_id"])
                    & selected["cavity"].eq(cavity) & selected["signal"].eq(signal)
                ]
                if len(measured) > 1:
                    raise ValueError("Duplicate condition/cavity/signal AUC results")
                row = {
                    **{column: condition[column] for column in CONDITION_COLUMNS},
                    "condition_id": condition["condition_id"], "cavity": cavity,
                    "signal": signal, "signal_name": SIGNAL_NAMES[signal], "window": window,
                    **counts, "n_splits": int(condition["n_splits"]),
                    "roc_auc_mean": np.nan, "roc_auc_std": np.nan,
                    "recall_at_5pct_fpr_mean": np.nan, "recall_at_5pct_fpr_std": np.nan,
                    "minimum_reference_normal_events": np.nan,
                    "status": condition["status"], "reason": "",
                }
                if not measured.empty:
                    value = measured.iloc[0]
                    for metric in ("roc_auc_mean", "roc_auc_std", "recall_at_5pct_fpr_mean",
                                   "recall_at_5pct_fpr_std", "minimum_reference_normal_events"):
                        row[metric] = value[metric]
                elif counts["normal_events"] == 0:
                    row["reason"] = "No normal capture in the matched acquisition condition"
                elif counts["trip_events"] == 0:
                    row["reason"] = "No trip capture in the matched acquisition condition"
                elif counts["normal_events"] < 2:
                    row["reason"] = "Only one normal capture; separate reference/test captures unavailable"
                elif counts["trip_events"] < 2:
                    row["reason"] = "Fewer than two trip captures for grouped cross-validation"
                else:
                    row["reason"] = str(condition["reason"])
                rows.append(row)
    report = pd.DataFrame(rows)
    if report.duplicated(["condition_id", "cavity", "signal", "window"]).any():
        raise AssertionError("Duplicate sensitivity report cells")
    evaluated = report["status"].eq("evaluated")
    if report.loc[evaluated, "roc_auc_mean"].isna().any():
        raise AssertionError("Missing evaluated AUC in the sensitivity report")
    return report


def plot_cavity_sensitivity(report):
    """Yield paired AUC and low-FPR recall heatmaps for each evaluated condition."""
    import matplotlib.pyplot as plt

    for condition_id, group in report.loc[report["status"].eq("evaluated")].groupby("condition_id"):
        context = group.iloc[0]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax, metric, multiplier, title, color_map, maximum in (
            (axes[0], "roc_auc_mean", 1, "ROC-AUC (fold mean)", "coolwarm", 1),
            (axes[1], "recall_at_5pct_fpr_mean", 100, "Recall at FPR <= 5% (fold mean, %)", "viridis", 100),
        ):
            table = group.pivot(index="signal", columns="cavity", values=metric).reindex(
                index=SIGNALS, columns=CAVITIES,
            ) * multiplier
            values = table.to_numpy()
            shown = ax.imshow(values, vmin=0, vmax=maximum, cmap=color_map, aspect="auto")
            ax.set(xticks=range(len(CAVITIES)), xticklabels=[c[-3:] for c in CAVITIES],
                   yticks=range(len(SIGNALS)), yticklabels=[SIGNAL_NAMES[s] for s in SIGNALS],
                   xlabel="Cavity", title=title)
            for row in range(len(SIGNALS)):
                for col in range(len(CAVITIES)):
                    value = values[row, col]
                    label = f"{value:.3f}" if multiplier == 1 else f"{value:.1f}"
                    color = ("white" if value < 0.23 or value > 0.85 else "black") if multiplier == 1 else (
                        "white" if value < 50 else "black"
                    )
                    ax.text(col, row, label, ha="center", va="center", color=color, fontsize=11)
            fig.colorbar(shown, ax=ax, shrink=0.8)
        minimum_reference = int(group["minimum_reference_normal_events"].min())
        fig.suptitle(
            f"{context.beam_mode} | beam_present={bool(context.beam_present)} | "
            f"{int(context.n_splits)} folds\n"
            f"beam_len={context.beam_len:g}, trig_delay={context.trig_delay:.6f} | "
            f"minimum normal reference captures per cavity/fold: {minimum_reference}", fontsize=12,
        )
        fig.text(0.5, 0.025,
                 "Fixed normal-envelope fraction score; held-out captures. "
                 "Fold SDs are in tables. Recall uses each test ROC, not a deployment threshold.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 0.89))
        yield condition_id, fig


def plot_roc(result):
    """Average ROC curves over folds and cavities within each matched condition."""
    import matplotlib.pyplot as plt

    conditions = result["support"].query("status == 'evaluated'")
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5), squeeze=False)
    for ax, (_, condition) in zip(axes.flat, conditions.iterrows()):
        curves = result["roc_curves"].query("condition_id == @condition.condition_id and window == 'rf_on_reference'")
        metrics = result["condition_summary"].query("condition_id == @condition.condition_id and window == 'rf_on_reference'")
        for signal in SIGNALS:
            curve = curves.loc[curves["signal"].eq(signal)].groupby("fpr")["tpr"].mean()
            auc_mean = metrics.loc[metrics["signal"].eq(signal), "roc_auc_mean"].mean()
            ax.plot(np.r_[0, curve.index.to_numpy()], np.r_[0, curve.to_numpy()],
                    label=f"{SIGNAL_NAMES[signal]} ({auc_mean:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="Trip recall",
               title=f"{condition.beam_mode} | beam={bool(condition.beam_present)}\n"
                     f"{int(condition.n_splits)} folds | normal envelope fraction score")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.15)
    fig.suptitle("ROC without a classifier | held-out captures, matched conditions", fontsize=14)
    fig.tight_layout()
    return fig


def plot_bin_auc(result, beam_mode="RfTest", beam_present=True):
    import matplotlib.pyplot as plt

    selected = result["bin_summary"].query("beam_mode == @beam_mode and beam_present == @beam_present")
    if selected["condition_id"].nunique() != 1:
        raise ValueError("Choose a mode/beam combination with one matched acquisition condition")
    fig, axes = plt.subplots(1, len(CAVITIES), figsize=(18, 4.5), squeeze=False)
    for ax, cavity in zip(axes.flat, CAVITIES):
        data = selected.loc[selected["cavity"].eq(cavity)]
        matrix = data.pivot(index="signal", columns="bin", values="roc_auc_mean").reindex(SIGNALS)
        rf = data.pivot(index="signal", columns="bin", values="rf_on_fold_fraction").reindex(SIGNALS)
        values = np.ma.masked_where(rf.to_numpy() < 1.0, matrix.to_numpy())
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("#eeeeee")
        shown = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto", extent=(0.5, 64.5, 5.5, -0.5))
        ax.set(yticks=range(6), yticklabels=[SIGNAL_NAMES[s] for s in SIGNALS],
               xticks=[1, 8, 16, 24, 32, 48, 64], xlabel="Waveform bin (whole record)", title=cavity[-3:])
    colorbar_ax = fig.add_axes([0.93, 0.18, 0.012, 0.64])
    fig.colorbar(shown, cax=colorbar_ax, label="Mean held-out AUC of absolute median deviation")
    fig.suptitle(f"{beam_mode} | beam={beam_present} | gray: outside reference RF-on in any fold", fontsize=13)
    fig.subplots_adjust(left=0.1, right=0.89, bottom=0.15, top=0.82, wspace=0.65)
    return fig


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    metadata, waves = load_beam_datasets(root / "signal_comparison")
    results = evaluate_score_auc(metadata, waves)
    output = root / "signal_comparison" / "waveform_score_auc"
    save_results(results, output)
    print(results["support"].to_string(index=False))
    print(results["beam_summary"].query("window == 'rf_on_reference'").to_string(index=False))
    print(f"Saved classifier-free ROC-AUC results to {output}")
