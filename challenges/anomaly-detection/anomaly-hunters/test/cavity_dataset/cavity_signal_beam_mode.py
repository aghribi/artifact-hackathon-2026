"""Compare pooled and beam-mode-specific training on identical held-out captures.

Each beam mode has its own event-grouped cross-validation partition. For each
test partition, the pooled model can train on all other modes but must exclude
every test capture globally, even if a capture spans more than one beam mode.
The specialized model trains only on that mode's remaining captures.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from cavity_signal_comparison import (
    SEED, SIGNALS, SIGNAL_NAMES, baseline_models, load_datasets, signal_features,
)


def load_beam_datasets(output_dir):
    """Attach cached beam metadata by checked original pulse identities."""
    output_dir = Path(output_dir)
    metadata, signals = load_datasets(output_dir)
    cached_path = output_dir / "direct_change_pulse_metadata.csv"
    if cached_path.is_file():
        cached = pd.read_csv(cached_path)
        if not cached["dataset_index"].is_unique:
            raise ValueError("Duplicate pulse indices in the beam metadata cache")
        if set(cached["dataset_index"]) != set(metadata["dataset_index"]):
            raise ValueError("Beam metadata cache and waveform bundle contain different pulses")
        cached = cached.set_index("dataset_index").loc[metadata["dataset_index"]].reset_index()
        for key in metadata.columns:
            if not np.array_equal(cached[key].to_numpy(), metadata[key].to_numpy()):
                raise ValueError(f"Beam metadata is not aligned: {key}")
        metadata = cached
    else:
        from cavity_signal_change import attach_beam_metadata
        metadata = attach_beam_metadata(metadata, output_dir.parent)
    if metadata["beam_mode"].isna().any():
        raise ValueError("Missing beam_mode")
    return metadata, signals


def evaluate_by_beam_mode(metadata, signals, max_splits=5):
    """Keep original class proportions and compare both training scopes fairly.

    The number of folds is capped by the rarer label's independent captures.
    StratifiedGroupKFold must also produce both labels in every train/test set;
    if it cannot, reduce the fold count. Modes with fewer than two captures per
    label cannot support this evaluation and are reported without a score.
    """
    from sklearn.base import clone
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedGroupKFold
    from threadpoolctl import threadpool_limits

    common = np.logical_and.reduce([np.isfinite(signals[s]).all(axis=1) for s in SIGNALS])
    meta = metadata.loc[common].reset_index(drop=True).copy()
    groups = meta["source"].astype(str) + ":" + meta["event_id"].astype(str)
    y = meta["label"].to_numpy()
    features = {s: signal_features(signals[s][common], s.endswith("_phase")) for s in SIGNALS}
    models = baseline_models()
    support_rows, fold_rows, prediction_rows, partition_rows = [], [], [], []

    with threadpool_limits(limits=2):
        for mode in sorted(meta["beam_mode"].unique()):
            indices = np.flatnonzero(meta["beam_mode"].eq(mode).to_numpy())
            mode_y, mode_groups = y[indices], groups.iloc[indices]
            event_counts = [mode_groups.iloc[np.flatnonzero(mode_y == label)].nunique()
                            for label in (0, 1)]
            support = {
                "beam_mode": mode, "normal_pulses": int((mode_y == 0).sum()),
                "trip_pulses": int((mode_y == 1).sum()),
                "normal_events": event_counts[0], "trip_events": event_counts[1],
                "trip_prevalence": float(mode_y.mean()),
                "n_splits": 0, "status": "insufficient_events",
                "reason": "Need at least two independent captures per label",
            }
            folds = None
            for n_splits in range(min(max_splits, *event_counts), 1, -1):
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
                candidate = list(splitter.split(np.zeros((len(indices), 1)), mode_y, mode_groups))
                if all(len(np.unique(mode_y[tr])) == len(np.unique(mode_y[te])) == 2
                       for tr, te in candidate):
                    folds = candidate
                    support.update(n_splits=n_splits, status="evaluated", reason="")
                    break
            support_rows.append(support)
            if folds is None:
                print(f"Skipped {mode}: {support['reason']}", flush=True)
                continue

            partitions = []
            for fold, (local_train, local_test) in enumerate(folds, 1):
                test = indices[local_test]
                mode_train = indices[local_train]
                test_groups = set(groups.iloc[test])
                pooled_train = np.flatnonzero(~groups.isin(test_groups).to_numpy())
                if test_groups & set(groups.iloc[pooled_train]):
                    raise AssertionError("Capture leakage in pooled training")
                if test_groups & set(groups.iloc[mode_train]):
                    raise AssertionError("Capture leakage in mode-specific training")
                if not set(mode_train).issubset(set(pooled_train)):
                    raise AssertionError("Training cohorts are inconsistent")
                partitions.append((fold, test, pooled_train, mode_train))
                for scope, train in (("pooled", pooled_train), ("per_mode", mode_train)):
                    partition_rows.append({
                        "beam_mode": mode, "fold": fold, "training_scope": scope,
                        "n_train": len(train), "n_train_events": groups.iloc[train].nunique(),
                        "n_train_normal": int((y[train] == 0).sum()),
                        "n_train_trip": int((y[train] == 1).sum()),
                        "n_test": len(test), "n_test_events": groups.iloc[test].nunique(),
                        "n_test_normal_events": groups.iloc[test[y[test] == 0]].nunique(),
                        "n_test_trip_events": groups.iloc[test[y[test] == 1]].nunique(),
                    })

            for signal in SIGNALS:
                x = features[signal]
                for model_name, template in models.items():
                    for fold, test, pooled_train, mode_train in partitions:
                        for scope, train in (("pooled", pooled_train), ("per_mode", mode_train)):
                            model = clone(template).fit(x[train], y[train])
                            probability = model.predict_proba(x[test])[:, 1]
                            fpr, tpr, _ = roc_curve(y[test], probability)
                            fold_rows.append({
                                "beam_mode": mode, "signal": signal, "model": model_name,
                                "training_scope": scope, "fold": fold,
                                "n_train": len(train), "n_test": len(test),
                                "trip_prevalence": float(y[test].mean()),
                                "roc_auc": roc_auc_score(y[test], probability),
                                "average_precision": average_precision_score(y[test], probability),
                                "recall_at_5pct_fpr": float(tpr[fpr <= 0.05].max()),
                            })
                            prediction = meta.iloc[test][[
                                "dataset_index", "source", "event_id", "cycle_id", "cavity",
                                "label", "beam_mode",
                            ]].copy()
                            prediction["signal"], prediction["model"] = signal, model_name
                            prediction["training_scope"], prediction["fold"] = scope, fold
                            prediction["prob_trip"] = probability
                            prediction_rows.append(prediction)
                    print(f"Evaluated {mode}: {signal}, {model_name} (pooled and per_mode)", flush=True)

    support = pd.DataFrame(support_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    if fold_metrics.empty:
        raise ValueError("No beam mode supports event-grouped classification")
    summary = fold_metrics.groupby(
        ["beam_mode", "model", "signal", "training_scope"], sort=False,
    ).agg(
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        average_precision_mean=("average_precision", "mean"),
        average_precision_std=("average_precision", "std"),
        recall_at_5pct_fpr_mean=("recall_at_5pct_fpr", "mean"),
        recall_at_5pct_fpr_std=("recall_at_5pct_fpr", "std"),
    ).reset_index().merge(support, on="beam_mode", validate="many_to_one")
    keys = ["beam_mode", "model", "signal"]
    comparison = summary.query("training_scope == 'pooled'").merge(
        summary.query("training_scope == 'per_mode'"), on=keys,
        suffixes=("_pooled", "_per_mode"), validate="one_to_one",
    )
    for metric in ("roc_auc", "average_precision", "recall_at_5pct_fpr"):
        comparison[f"{metric}_delta"] = (comparison[f"{metric}_mean_per_mode"]
                                          - comparison[f"{metric}_mean_pooled"])
    predictions = pd.concat(prediction_rows, ignore_index=True)
    if predictions.duplicated(["dataset_index", "signal", "model", "training_scope"]).any():
        raise AssertionError("Duplicate out-of-fold predictions")
    expected = int(support.query("status == 'evaluated'")[["normal_pulses", "trip_pulses"]].to_numpy().sum())
    coverage = predictions.groupby(["signal", "model", "training_scope"]).size()
    if not (coverage == expected).all() or not np.isfinite(predictions["prob_trip"]).all():
        raise AssertionError("Incomplete out-of-fold predictions")
    return {"support": support, "summary": summary, "comparison": comparison,
            "fold_metrics": fold_metrics, "predictions": predictions,
            "partitions": pd.DataFrame(partition_rows), "metadata": meta}


def save_results(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("support", "summary", "comparison", "fold_metrics", "predictions", "partitions"):
        result[name].to_csv(output_dir / f"{name}.csv", index=False)


def load_results(output_dir):
    output_dir = Path(output_dir)
    # Preserve nearly tied tree scores exactly when reconstructing ROC/PR metrics.
    return {name: pd.read_csv(output_dir / f"{name}.csv", float_precision="round_trip")
            for name in ("support", "summary", "comparison", "fold_metrics", "predictions", "partitions")}


def plot_comparison(result):
    """Show paired training scopes; compare AP only within the same test cohort."""
    import matplotlib.pyplot as plt

    modes = result["support"].query("status == 'evaluated'")["beam_mode"].tolist()
    fig, axes = plt.subplots(2, len(modes), figsize=(6 * len(modes), 8), squeeze=False)
    summary = result["summary"]
    for row, model in enumerate(("ExtraTrees", "LogisticRegression")):
        for col, mode in enumerate(modes):
            ax = axes[row, col]
            selected = summary.query("model == @model and beam_mode == @mode")
            positions = np.arange(len(SIGNALS))
            for offset, scope, color, label in (
                (-0.18, "pooled", "#3574ae", "Train on all modes"),
                (0.18, "per_mode", "#e56b32", "Train on this mode"),
            ):
                table = selected.query("training_scope == @scope").set_index("signal").loc[list(SIGNALS)]
                ax.barh(positions + offset, table["average_precision_mean"], height=0.34,
                        xerr=table["average_precision_std"], color=color, alpha=0.8,
                        capsize=2, label=label)
            support = result["support"].set_index("beam_mode").loc[mode]
            ax.axvline(support["trip_prevalence"], color="gray", linestyle=":",
                       label="Trip fraction in mode")
            ax.set(yticks=positions, yticklabels=[SIGNAL_NAMES[s] for s in SIGNALS],
                   xlim=(0, 1.05), xlabel="Average precision (fold mean +/- SD)",
                   title=f"{mode} | {model} | {support['n_splits']} folds")
            ax.grid(axis="x", alpha=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3)
    fig.suptitle("Identical held-out captures for pooled and mode-specific training", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    metadata, signals = load_beam_datasets(root / "signal_comparison")
    result = evaluate_by_beam_mode(metadata, signals)
    destination = root / "signal_comparison" / "beam_mode_classifier"
    save_results(result, destination)
    print(result["support"].to_string(index=False))
    print(result["summary"].query("training_scope == 'per_mode'").to_string(index=False))
    print(f"Saved results to {destination}")
