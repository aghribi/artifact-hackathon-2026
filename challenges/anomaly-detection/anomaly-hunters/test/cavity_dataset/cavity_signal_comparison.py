"""Build aligned datasets for six cavity signals and compare trip sensitivity.

The normal label is every DD pulse; the trip label is the last PM pulse.
All numerical analysis uses the same pulses and event-grouped folds for every signal.
Parquet inputs require pandas and pyarrow; analysis of saved bundles additionally
uses scikit-learn and matplotlib. No autoencoder training is required.
"""
from pathlib import Path
import tarfile
import re

import numpy as np
import pandas as pd


CAVITIES = [f"HBL-010RFC:RFS-DIG-{n}01" for n in (1, 2, 3, 4)]
SIGNALS = ("field_amp", "field_phase", "fwd_amp", "fwd_phase", "refl_amp", "refl_phase")
SIGNAL_NAMES = {
    "field_amp": "Field amplitude", "field_phase": "Field phase",
    "fwd_amp": "Forward amplitude", "fwd_phase": "Forward phase",
    "refl_amp": "Reflected amplitude", "refl_phase": "Reflected phase",
}
N_BINS = 64
SEED = 42


def bin_waveform(wave, phase=False):
    """Match the original extractor, including circular means for phase."""
    wave = np.asarray(wave, dtype=float)
    if len(wave) < N_BINS:
        return np.full(N_BINS, np.nan, dtype=np.float32)
    edges = np.linspace(0, len(wave), N_BINS + 1).astype(int)
    if phase:
        wave = np.exp(1j * np.deg2rad(wave))
    result = np.add.reduceat(wave, edges[:-1]) / np.diff(edges)
    if phase:
        result = np.rad2deg(np.angle(result))
    return result.astype(np.float32)


def save_datasets(output_dir, metadata, signals):
    """Save one original-schema CSV per signal plus a compact aligned bundle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_dir / "pulse_metadata.csv", index=False)
    np.savez_compressed(output_dir / "waveforms.npz", **signals)
    availability = []
    for signal in SIGNALS:
        matrix = signals[signal]
        if matrix.shape != (len(metadata), N_BINS):
            raise ValueError(f"Unexpected shape for {signal}: {matrix.shape}")
        frame = pd.DataFrame(matrix, columns=[f"value_{i}" for i in range(1, N_BINS + 1)])
        frame.insert(0, "cavity", metadata["cavity"].to_numpy())
        frame["label"] = metadata["label"].to_numpy()
        frame.to_csv(output_dir / f"cavity_pulses_{signal}_labeled.csv", index=False)
        valid = np.isfinite(matrix).all(axis=1)
        availability.append({
            "signal": signal, "usable": int(valid.sum()), "missing": int((~valid).sum()),
            "normal": int((valid & (metadata["label"] == 0)).sum()),
            "trip": int((valid & (metadata["label"] == 1)).sum()),
        })
    pd.DataFrame(availability).to_csv(output_dir / "signal_availability.csv", index=False)


def build_signal_datasets(notebook_dir, raw_dir):
    """Rebuild the inputs from the feature Parquets and original SDS archives.

    Forward/reflected phase bins are absent from the existing feature tables.
    Read those two channels from matching event/cycle/cavity rows in HBL.parquet.
    Arrow batches bound memory when scanning the raw waveform column.
    """
    import pyarrow.parquet as pq

    notebook_dir, raw_dir = Path(notebook_dir), Path(raw_dir)
    existing = ("field_amp", "field_phase", "fwd_amp", "refl_amp")
    parts, matrices = [], {signal: [] for signal in existing}
    for kind, label in (("dd", 0), ("pm", 1)):
        columns = ["event_id", "cycle_id", "cavity", "cycle_rank_from_last", "array_size"]
        columns += [f"{signal}_bins" for signal in existing]
        frame = pd.read_parquet(notebook_dir.parent / f"notebooks/ess_{kind}/cavity_pulses.parquet", columns=columns)
        frame["source_row"] = np.arange(len(frame))
        if label == 0:
            frame["dataset_index"] = np.arange(len(frame))
            dd_count = len(frame)
        else:
            frame = frame.loc[frame["cycle_rank_from_last"] < 1].copy()
            frame["dataset_index"] = dd_count + np.arange(len(frame))
        frame = frame.loc[frame["cavity"].isin(CAVITIES)].copy()
        frame["source"], frame["label"] = kind.upper(), label
        parts.append(frame.drop(columns=[f"{signal}_bins" for signal in existing]))
        for signal in existing:
            vectors = []
            for wave in frame[f"{signal}_bins"]:
                if isinstance(wave, (list, np.ndarray)) and len(wave) == N_BINS:
                    vectors.append(np.asarray(wave, dtype=np.float32))
                else:
                    vectors.append(np.full(N_BINS, np.nan, dtype=np.float32))
            matrices[signal].append(np.stack(vectors))
    metadata = pd.concat(parts, ignore_index=True)
    signals = {signal: np.vstack(parts) for signal, parts in matrices.items()}
    for signal in ("fwd_phase", "refl_phase"):
        signals[signal] = np.full((len(metadata), N_BINS), np.nan, dtype=np.float32)

    for event_id, requested in metadata.groupby("event_id", sort=True):
        archives = list(raw_dir.glob(f"*_{event_id}_*.tar"))
        if len(archives) != 1:
            raise ValueError(f"Expected one raw archive for event {event_id}, found {archives}")
        keys = {(int(row.cycle_id), row.cavity): index for index, row in requested.iterrows()}
        with tarfile.open(archives[0], "r:") as archive:
            member = next(m for m in archive if m.isfile() and m.name.endswith("/HBL.parquet"))
            with archive.extractfile(member) as raw:
                parquet = pq.ParquetFile(raw)
                names = parquet.read(columns=["pv", "cycle_id", "array_size"]).to_pandas()
                wanted = {}
                for row_index, row in names.iterrows():
                    match = re.fullmatch(r"(.+):Dwn([56])-Cmp1SDS-\w+", row["pv"])
                    if match:
                        cavity, dwn = match.groups()
                        key = (int(row["cycle_id"]), cavity)
                        if key in keys and row["array_size"] >= N_BINS:
                            wanted[row_index] = (keys[key], "fwd_phase" if dwn == "5" else "refl_phase")
                start = 0
                for batch in parquet.iter_batches(batch_size=16, columns=["data_array"]):
                    stop = start + batch.num_rows
                    chosen = [i for i in wanted if start <= i < stop]
                    if chosen:
                        arrays = batch.column(0)
                        offsets = arrays.offsets.to_numpy(zero_copy_only=False)
                        values = arrays.values.to_numpy(zero_copy_only=False)
                        for raw_index in chosen:
                            index, signal = wanted[raw_index]
                            local = raw_index - start
                            wave = values[offsets[local]:offsets[local + 1]]
                            # Last duplicate wins, matching ess_dd_features.py.
                            signals[signal][index] = bin_waveform(wave, phase=True)
                    start = stop
        print(f"Loaded phase channels for event {event_id}")
    save_datasets(notebook_dir / "signal_comparison", metadata, signals)
    return metadata, signals


def load_datasets(output_dir):
    output_dir = Path(output_dir)
    metadata = pd.read_csv(output_dir / "pulse_metadata.csv")
    with np.load(output_dir / "waveforms.npz", allow_pickle=False) as bundle:
        signals = {signal: bundle[signal] for signal in SIGNALS}
    if metadata.duplicated(["source", "event_id", "cycle_id", "cavity"]).any():
        raise ValueError("Duplicate pulse keys in the aligned dataset")
    if not metadata["dataset_index"].is_unique:
        raise ValueError("Duplicate original dataset indices")
    for signal, matrix in signals.items():
        if matrix.shape != (len(metadata), N_BINS):
            raise ValueError(f"Incorrect dimensions for {signal}")
    return metadata, signals


def signal_features(matrix, phase=False):
    """Encode phase on the unit circle to respect the -180/180-degree boundary."""
    if phase:
        radians = np.deg2rad(matrix.astype(float))
        return np.hstack([np.sin(radians), np.cos(radians)])
    return matrix.astype(float)


def baseline_models():
    """Return fresh copies of the fixed models used in every signal comparison."""
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2,
                                            class_weight="balanced", random_state=SEED, n_jobs=2),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED)),
    }


def evaluate_signals(metadata, signals, n_splits=5):
    """Fixed baselines with common, event-grouped folds; no hyperparameter search."""
    from sklearn.base import clone
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedGroupKFold
    from threadpoolctl import threadpool_limits

    common = np.logical_and.reduce([np.isfinite(signals[s]).all(axis=1) for s in SIGNALS])
    meta = metadata.loc[common].reset_index(drop=True)
    y = meta["label"].to_numpy()
    # A capture is kept entirely in one fold, including all its cavities.
    groups = meta["source"].astype(str) + ":" + meta["event_id"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros((len(meta), 1)), y, groups))
    for train, test in folds:
        if set(groups.iloc[train]) & set(groups.iloc[test]):
            raise AssertionError("Event leakage between train and test")
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError("Each fold must contain both labels")
    models = baseline_models()
    fold_rows, prediction_rows, cavity_rows = [], [], []
    with threadpool_limits(limits=2):
        for signal in SIGNALS:
            x = signal_features(signals[signal][common], phase=signal.endswith("_phase"))
            for model_name, template in models.items():
                probabilities = np.full(len(meta), np.nan)
                fold_ids = np.full(len(meta), -1, dtype=int)
                for fold, (train, test) in enumerate(folds, 1):
                    model = clone(template).fit(x[train], y[train])
                    probability = model.predict_proba(x[test])[:, 1]
                    probabilities[test], fold_ids[test] = probability, fold
                    fpr, tpr, _ = roc_curve(y[test], probability)
                    fold_rows.append({
                        "signal": signal, "model": model_name, "fold": fold,
                        "n_test": len(test), "trip_prevalence": float(y[test].mean()),
                        "roc_auc": roc_auc_score(y[test], probability),
                        "average_precision": average_precision_score(y[test], probability),
                        "recall_at_5pct_fpr": float(tpr[fpr <= 0.05].max()),
                    })
                if not np.isfinite(probabilities).all() or (fold_ids < 1).any():
                    raise AssertionError("Incomplete out-of-fold predictions")
                predicted = meta[["dataset_index", "source", "event_id", "cycle_id", "cavity", "label"]].copy()
                predicted["signal"], predicted["model"] = signal, model_name
                predicted["fold"], predicted["prob_trip"] = fold_ids, probabilities
                prediction_rows.append(predicted)
                for cavity in sorted(meta["cavity"].unique()):
                    selected = meta["cavity"].to_numpy() == cavity
                    cavity_rows.append({
                        "signal": signal, "model": model_name, "cavity": cavity,
                        "n_pulses": int(selected.sum()),
                        "roc_auc": roc_auc_score(y[selected], probabilities[selected]),
                        "average_precision": average_precision_score(y[selected], probabilities[selected]),
                    })
                print(f"Evaluated {signal}: {model_name}", flush=True)
    fold_metrics = pd.DataFrame(fold_rows)
    summary = fold_metrics.groupby(["model", "signal"], sort=False).agg(
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        average_precision_mean=("average_precision", "mean"),
        average_precision_std=("average_precision", "std"),
        recall_at_5pct_fpr_mean=("recall_at_5pct_fpr", "mean"),
        recall_at_5pct_fpr_std=("recall_at_5pct_fpr", "std"),
    ).reset_index().sort_values(["model", "average_precision_mean"], ascending=[True, False])
    return {"summary": summary, "fold_metrics": fold_metrics,
            "predictions": pd.concat(prediction_rows, ignore_index=True),
            "cavity_metrics": pd.DataFrame(cavity_rows), "common_mask": common,
            "metadata": meta, "folds": folds}


def plot_waveforms(metadata, signals):
    """Plot matching random pulse examples for every signal and cavity."""
    import matplotlib.pyplot as plt
    figures = []
    for phase in (False, True):
        selected_signals = [s for s in SIGNALS if s.endswith("_phase") == phase]
        fig, axes = plt.subplots(3, 4, figsize=(17, 9), squeeze=False)
        for row, signal in enumerate(selected_signals):
            for col, cavity in enumerate(CAVITIES):
                ax = axes[row, col]
                for label, color in ((0, "#3574ae"), (1, "#e56b32")):
                    indices = np.flatnonzero((metadata["cavity"].to_numpy() == cavity)
                                             & (metadata["label"].to_numpy() == label)
                                             & np.isfinite(signals[signal]).all(axis=1))
                    # Same selected pulse indices for each channel where available.
                    rng = np.random.RandomState(SEED)
                    chosen = rng.choice(indices, size=min(10, len(indices)), replace=False)
                    for number, index in enumerate(chosen):
                        ax.plot(np.arange(1, N_BINS + 1), signals[signal][index],
                                color=color, alpha=0.4, linewidth=0.8,
                                label=("Normal (DD)" if label == 0 else "Trip (PM)") if number == 0 else None)
                ax.set_title(f"{SIGNAL_NAMES[signal]} | {cavity[-3:]}", fontsize=10)
                if col == 0:
                    ax.set_ylabel("Phase (degrees)" if phase else "Amplitude (source units)")
                if row == 2:
                    ax.set_xlabel("Waveform bin")
                ax.grid(alpha=0.15)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Phase waveforms (wrapped degrees)" if phase else "Amplitude waveforms", fontsize=14)
        fig.tight_layout()
        figures.append(fig)
    return figures


def plot_pca(metadata, signals, common):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    y = metadata.loc[common, "label"].to_numpy()
    with threadpool_limits(limits=2):
        for signal, ax in zip(SIGNALS, axes.flat):
            features = signal_features(signals[signal][common], signal.endswith("_phase"))
            pca = PCA(n_components=2, svd_solver="full")
            xy = pca.fit_transform(StandardScaler().fit_transform(features))
            for label, color, marker in ((0, "#3574ae", "o"), (1, "#e56b32", "x")):
                select = y == label
                ax.scatter(xy[select, 0], xy[select, 1], color=color, marker=marker,
                           s=10 if label == 0 else 22, alpha=0.35 if label == 0 else 0.7,
                           label="Normal (DD)" if label == 0 else "Trip (PM)")
            ax.set(title=SIGNAL_NAMES[signal], xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                   ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            ax.grid(alpha=0.15)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("PCA of aligned whole waveforms | phase encoded as sine/cosine")
    fig.tight_layout()
    return fig


def plot_performance(result):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    summary = result["summary"]
    selected = summary[summary["model"] == "ExtraTrees"].sort_values("average_precision_mean")
    names = [SIGNAL_NAMES[s] for s in selected["signal"]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for ax, metric, title in zip(axes,
            ("roc_auc", "average_precision", "recall_at_5pct_fpr"),
            ("ROC AUC", "Average precision", "Recall at FPR <= 5%")):
        ax.barh(names, selected[metric + "_mean"], xerr=selected[metric + "_std"],
                color="#3574ae", alpha=0.85, capsize=3)
        ax.set(xlim=(0, 1.05), title=title, xlabel="Mean +/- fold standard deviation")
        ax.grid(axis="x", alpha=0.2)
    axes[0].axvline(0.5, linestyle="--", color="gray", linewidth=1)
    prevalence = result["metadata"]["label"].mean()
    axes[1].axvline(prevalence, linestyle="--", color="gray", linewidth=1)
    fig.suptitle("ExtraTrees | identical 5-fold event-grouped evaluation")
    fig.tight_layout()
    curves, ax = plt.subplots(figsize=(7, 5))
    predictions = result["predictions"]
    for signal in SIGNALS:
        rows = predictions[(predictions["signal"] == signal) & (predictions["model"] == "ExtraTrees")]
        fpr, tpr, _ = roc_curve(rows["label"], rows["prob_trip"])
        ax.plot(fpr, tpr, label=SIGNAL_NAMES[signal])
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.axvline(0.05, linestyle=":", color="gray", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="Trip recall", xlim=(0, 1), ylim=(0, 1.02),
           title="ROC curves from out-of-fold predictions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    curves.tight_layout()
    return fig, curves
