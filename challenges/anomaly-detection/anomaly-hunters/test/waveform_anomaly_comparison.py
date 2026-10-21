"""Compare Isolation Forest with full-waveform PCA + kNN, including beam_mode.

Each normalized PV has its own models. PV suffixes
-DODData and -PMData are ignored for model selection; source names are retained.
Isolation Forest uses statistics
of the existing 256-bin min/max envelopes. PCA uses every ORIGINAL sample,
without resampling or per-waveform normalization. Short tails are NaN-padded;
PCA uses sample-position mean imputation, and kNN also receives an explicit
padding mask and one-hot beam mode.
Both methods use the same training
captures and the same exploratory median + MAD threshold rule, not confirmed fault labels.

Usage:
    python waveform_anomaly_comparison.py
    python waveform_anomaly_comparison.py --pv MBL-090RFC:RFS-DIG-101:Dwn0-Cmp0SDS-PMData
    python waveform_anomaly_comparison.py --resume --raw-cache anomaly-pca-reuse-20261015
    python waveform_anomaly_comparison.py --rethreshold --threshold-method median_mad

Per-PV PCA checkpoints are saved after every PV. Run global_waveform_pca_knn.py
after this script to add the single shared model to the comparison.

Dependencies: numpy, scipy, scikit-learn, matplotlib, Pillow.
An intermediate full-waveform cache is removed after use. Original files are
read-only. Results include all input captures, including unscored captures.
"""
import argparse
import collections
import csv
import html
import hashlib
import json
import mmap
import os
import shutil
from pathlib import Path
import sys
import tarfile
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/anomalyhunter-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "waveform_plots"))
from beam_modes import load_beam_modes
from plot_all_cycles import BINS, waveforms as decode_waveforms


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def restore_classification(records, path):
    """Restore numeric types and verify row identity before reusing a checkpoint."""
    with Path(path).open() as stream:
        reader = csv.DictReader(stream)
        for record in records:
            saved = next(reader, None)
            if saved is None or record["original_row"] != int(saved["original_row"]):
                raise ValueError("Checkpoint original rows do not match current inputs")
            for field in ["pv", "source_pv", "cycle_id", "array_size", "beam_mode"]:
                if str(record[field]) != saved[field]:
                    raise ValueError(f"Checkpoint {field} does not match current inputs")
            for key, value in saved.items():
                template = record.get(key)
                if value == "":
                    record[key] = None
                elif value in ("True", "False"):
                    record[key] = value == "True"
                elif isinstance(template, str):
                    record[key] = value
                elif isinstance(template, int) or key.endswith("_label") or key in ["duplicate_of", "global_padded_size", "global_missing_samples"]:
                    record[key] = int(value)
                else:
                    record[key] = float(value)
        if next(reader, None) is not None:
            raise ValueError("Checkpoint contains additional original rows")


def restore_forests(records, groups, args):
    apply_score_thresholds(records, groups, args)
    summaries, modeled = [], {}
    for pv, indices in sorted(groups.items()):
        rows = [records[i] for i in indices]
        training = [i for i in indices if records[i]["duplicate_of"] is None]
        unique_cycles = len({r["cycle_id"] for r in rows})
        active = unique_cycles >= args.minimum
        if active and any(r["iforest_score"] is None or r["status"] != "modeled" for r in rows):
            raise ValueError(f"Incomplete Isolation Forest checkpoint: {pv}")
        summary = dict(pv=pv, sample_lengths=", ".join(map(str, sorted({r["array_size"] for r in rows}))),
                       padded_size=max(r["array_size"] for r in rows), rows=len(rows),
                       training_captures=len(training), unique_cycles=unique_cycles,
                       status="modeled" if active else "insufficient_samples",
                       beam_modes=", ".join(sorted({r["beam_mode"] for r in rows})),
                       iforest_candidates=sum(r["iforest_label"] == -1 for r in rows), pca_knn_candidates=0,
                       iforest_wave_only_candidates=sum(r["iforest_wave_only_label"] == -1 for r in rows),
                       iforest_mode_changes=sum(r["iforest_mode_changed"] for r in rows),
                       pca_wave_only_candidates=0, pca_mode_changes=0, both_candidates=0,
                       candidate_jaccard=None, pca_components=None, pca_retained_variance=None,
                       knn_neighbors=None, pca_knn_threshold=None,
                       iforest_threshold=score_threshold([records[i]["iforest_score"] for i in training], args.threshold_method, args.fraction, mad_multiplier_for("iforest", args))["threshold"] if active else None)
        summaries.append(summary)
        if active:
            modeled[pv] = (indices, training, summary)
    print(f"Restored completed Isolation Forest results: {len(modeled)} PVs", flush=True)
    return summaries, modeled


def normalized_pv(pv):
    for suffix in ("-doddata", "-pmdata"):
        if pv.lower().endswith(suffix):
            return pv[:-len(suffix)]
    return pv


def envelope_features(envelopes, blocks=16):
    """Amplitude extrema are exact; centers/widths are envelope statistics."""
    a = np.asarray(envelopes, dtype=np.float64).reshape(-1, BINS, 2)
    low, high = a[:, :, 0], a[:, :, 1]
    center, width = (low + high) / 2, high - low
    minimum, maximum = low.min(axis=1), high.max(axis=1)
    span = maximum - minimum
    delta = np.abs(np.diff(center, axis=1))
    globals_ = [minimum, maximum, span, center.mean(axis=1), center.std(axis=1),
                *np.percentile(center, [10, 50, 90], axis=1), width.mean(axis=1),
                width.max(axis=1), delta.mean(axis=1), delta.max(axis=1),
                center[:, -1] - center[:, 0], center.argmax(axis=1) / (BINS - 1)]
    names = ["waveform_min", "waveform_max", "waveform_range", "envelope_center_mean",
             "envelope_center_std", "envelope_center_q10", "envelope_center_q50",
             "envelope_center_q90", "envelope_width_mean", "envelope_width_max",
             "envelope_delta_mean", "envelope_delta_max", "envelope_end_minus_start",
             "envelope_peak_position"]
    divisor = np.maximum(span, 1e-12)[:, None]
    normalized = (center - minimum[:, None]) / divisor
    parts = [normalized.reshape(-1, blocks, BINS // blocks).mean(axis=2),
             ((low - minimum[:, None]) / divisor).reshape(-1, blocks, BINS // blocks).min(axis=2),
             ((high - minimum[:, None]) / divisor).reshape(-1, blocks, BINS // blocks).max(axis=2),
             (width / divisor).reshape(-1, blocks, BINS // blocks).mean(axis=2)]
    for stat in ["center", "minimum", "maximum", "width"]:
        names.extend(f"shape_{stat}_{i:02d}" for i in range(blocks))
    return np.column_stack([*globals_, *parts]).astype(np.float32), names


def raw_pca(samples, retained=0.95):
    """Exact dual PCA: center full samples, diagonalize X @ X.T, no downsampling.

    A small capture-by-capture Gram matrix avoids forming a 35,221-square
    covariance matrix. The resulting scores and eigenvalues equal ordinary PCA.
    """
    x = np.array(samples, dtype=np.float64, copy=True)
    if np.isinf(x).any():
        raise ValueError("Infinite original samples are not valid PCA input")
    means = np.nanmean(x, axis=0)
    if np.isnan(means).any():
        raise ValueError("PCA has a sample position with no observed training data")
    x -= means
    # This replaces NaN with its sample-position mean BEFORE centering; missing
    # positions become zero only in centered coordinates. The mask is retained.
    np.nan_to_num(x, copy=False, nan=0.)
    gram = x @ x.T
    eigenvalues, vectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues[::-1], 0)
    vectors = vectors[:, ::-1]
    total = float(eigenvalues.sum())
    if total <= 1e-24:
        return np.zeros((len(x), 1)), 0, 1.0
    count = int(np.searchsorted(np.cumsum(eigenvalues) / total, retained) + 1)
    count = min(count, len(x) - 1, x.shape[1])
    scores = vectors[:, :count] * np.sqrt(eigenvalues[:count])
    # One common scale preserves the relative PC variances and waveform amplitude
    # differences, while giving the waveform block total variance of one.
    scale = np.sqrt(eigenvalues[:count].sum() / max(len(x) - 1, 1))
    return scores / max(scale, 1e-12), count, float(eigenvalues[:count].sum() / total)


def neighbor_scores(points, neighbors=10, workers=2):
    """Mean distance to k OTHER captures; exclude the training point itself."""
    k = min(neighbors, len(points) - 1)
    search = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=workers).fit(points)
    distances = search.kneighbors(points, return_distance=True)[0]
    return distances[:, 1:].mean(axis=1), k


def mad_multiplier_for(detector, args):
    family = "iforest" if detector.startswith("iforest") else "global_pca_knn" if detector.startswith("global") else "pca_knn"
    specific = getattr(args, family + "_mad_multiplier", None)
    common = getattr(args, "mad_multiplier", None)
    return float(specific if specific is not None else common if common is not None else 1.)


def score_threshold(scores, method="median_mad", fraction=.05, mad_multiplier=1.):
    """Use raw (unscaled) MAD. Larger scores are more anomalous; ties are nominal."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Threshold scores must be a nonempty finite vector")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if not np.isfinite(mad_multiplier) or mad_multiplier <= 0:
        raise ValueError("MAD multiplier must be finite and positive")
    if method == "median_mad":
        threshold = median + mad_multiplier * mad
    elif method == "quantile":
        threshold = float(np.quantile(values, 1 - fraction))
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    return dict(threshold_method=method, training_count=len(values), median=median, mad=mad,
                mad_multiplier=float(mad_multiplier), mad_consistency_scale=1.,
                threshold=threshold, threshold_percentile=float(np.mean(values <= threshold) * 100))


def distance_labels(scores, fraction=.05, method="median_mad", mad_multiplier=1.):
    threshold = score_threshold(scores, method, fraction, mad_multiplier)["threshold"]
    return np.where(np.asarray(scores) > threshold, -1, 1), threshold


def apply_score_thresholds(records, groups, args):
    """Relabel existing scores; exactly duplicated captures never affect a cutoff."""
    method = getattr(args, "threshold_method", "median_mad")
    primary = ["iforest", "pca_knn"]
    has_global = any("global_pca_knn_label" in row for row in records)
    if has_global:
        primary.append("global_pca_knn")
    for row in records:
        for detector in primary:
            row[detector + "_threshold"] = None
            row[detector + "_threshold_percentile"] = None
    details = []
    def apply(indices, detector, pv, scope):
        scored = [i for i in indices if records[i].get(detector + "_score") is not None]
        training = [i for i in scored if records[i]["duplicate_of"] is None]
        if not training:
            return
        stats = score_threshold([records[i][detector + "_score"] for i in training], method, args.fraction,
                                mad_multiplier_for(detector, args))
        details.append(dict(scope=scope, pv=pv, detector=detector, **stats))
        for i in scored:
            row = records[i]
            row[detector + "_label"] = -1 if row[detector + "_score"] > stats["threshold"] else 1
            if detector in primary:
                row[detector + "_threshold"] = stats["threshold"]
                row[detector + "_threshold_percentile"] = stats["threshold_percentile"]
    for pv, indices in sorted(groups.items()):
        for detector in ["iforest", "pca_knn", "iforest_wave_only", "pca_wave_only"]:
            apply(indices, detector, pv, "normalized_pv")
    if has_global:
        apply(range(len(records)), "global_pca_knn", "", "all_pvs")
    for row in records:
        row["agreement"] = bool(row["iforest_label"] == row["pca_knn_label"]) if row["iforest_label"] is not None and row["pca_knn_label"] is not None else None
        row["iforest_mode_changed"] = bool(row["iforest_label"] != row["iforest_wave_only_label"]) if row["iforest_label"] is not None else False
        row["pca_mode_changed"] = bool(row["pca_knn_label"] != row["pca_wave_only_label"]) if row["pca_knn_label"] is not None else False
    return details


def refresh_group_summaries(groups, records, thresholds):
    by_pv = collections.defaultdict(list)
    for row in records:
        if row["status"] == "modeled":
            by_pv[row["pv"]].append(row)
    details = {(r["pv"], r["detector"]): r for r in thresholds if r["scope"] == "normalized_pv"}
    for group in groups:
        rows = by_pv[group["pv"]]
        stats = metrics(rows)
        for key in ["iforest_candidates", "pca_knn_candidates", "both_candidates", "candidate_jaccard", "iforest_mode_changes", "pca_mode_changes"]:
            group[key] = stats[key]
        for detector in ["iforest", "pca_knn", "iforest_wave_only", "pca_wave_only"]:
            detail = details.get((group["pv"], detector))
            if detail:
                for key in ["threshold", "median", "mad", "threshold_percentile", "mad_multiplier"]:
                    group[detector + "_" + key] = detail[key]
        group["iforest_wave_only_candidates"] = sum(r["iforest_wave_only_label"] == -1 for r in rows)
        group["pca_wave_only_candidates"] = sum(r["pca_wave_only_label"] == -1 for r in rows)


def attach_known_anomalies(output, records):
    path = output / "known_anomalies.csv"
    if not path.exists():
        return None
    with path.open() as stream:
        known = list(csv.DictReader(stream))
    identities = {(normalized_pv(r["pv"]), int(r["cycle_id"]), int(r["original_row"])) for r in known}
    matched = []
    for row in records:
        row["confirmed_label"] = -1 if (row["pv"], row["cycle_id"], row["original_row"]) in identities else None
        if row["confirmed_label"] == -1:
            matched.append(row)
    return dict(confirmed_anomalies=len(matched), supplied_labels=len(known),
                detected={d: sum(r.get(d + "_label") == -1 for r in matched) for d in ["iforest", "pca_knn", "global_pca_knn"]},
                interpretation="User-confirmed positive examples used for provisional threshold selection; not held-out validation. Other waveforms remain unlabelled.")


def padding_mask_features(lengths, padded_size):
    """Exact compressed tail mask; distances equal full masks / sqrt(width).

    A mask is constant between observed waveform lengths, so weighting those
    intervals preserves the complete mask information without 35,221 columns.
    """
    lengths = np.asarray(lengths)
    edges = np.unique(np.r_[0, lengths, padded_size])
    return (lengths[:, None] <= edges[:-1][None, :]) * np.sqrt(np.diff(edges) / padded_size)


def load_inputs(root, pv=None):
    waveform_rows = json.loads((root / "waveforms.json").read_text())
    modes = load_beam_modes(root, waveform_rows)
    inventory = json.loads((root / "inventory.json").read_text())
    with (HERE / "dd_by_cycle/cycle_summary.csv").open() as stream:
        cycles = {r["cycle_id"]: r for r in csv.DictReader(stream)}
    sources = collections.defaultdict(list)
    archive_ids = {path: i for i, path in enumerate(dict.fromkeys(s["archive"] for s in inventory["files"]), 1)}
    for source in inventory["files"]:
        for source_row, (cycle, name) in enumerate(zip(source["cycle_ids"], source["pvs"])):
            sources[str(cycle)].append((name, source["archive"], source["member"], source_row))
    records, features, envelopes = [], [], []
    source_lookup, groups = {}, collections.defaultdict(list)
    original_row = 0
    feature_names = None
    for cycle in sorted(waveform_rows, key=int):
        rows = waveform_rows[cycle]
        assert len(rows) == int(cycles[cycle]["rows"]) == len(sources[cycle])
        cached = np.memmap(root / "cache" / f"{cycle}.f32", dtype="<f4", mode="r", shape=(len(rows), BINS * 2))
        wanted = [i for i, r in enumerate(rows) if pv is None or normalized_pv(r["pv"]) == normalized_pv(pv)]
        good = [i for i in wanted if rows[i]["samples"] > 0 and rows[i]["min"] is not None and np.isfinite(cached[i]).all()]
        computed = {}
        if good:
            matrix, feature_names = envelope_features(cached[good])
            computed = dict(zip(good, matrix))
        for i, row in enumerate(rows):
            name, archive, member, source_row = sources[cycle][i]
            assert (row["pv"], row["archive"], row["member"]) == (name, archive_ids[archive], member)
            if i in wanted:
                index = len(records)
                record = dict(original_row=original_row, pv=normalized_pv(name), source_pv=name, cycle_id=int(cycle),
                              cycle_ts=cycles[cycle]["cycle_ts"], beam_mode=modes[cycle][i] or "Unknown",
                              array_size=row["samples"], source_archive=archive, source_parquet=member,
                              source_row=source_row, waveform_min=row["min"], waveform_max=row["max"],
                              status="pending" if i in computed else "invalid_waveform",
                              duplicate_of=None,
                              mode_count=0, rare_mode=False,
                              padded_size=0, missing_samples=0, missing_fraction=0.,
                              iforest_score=None, iforest_label=None, iforest_percentile=None,
                              iforest_wave_only_score=None, iforest_wave_only_label=None, iforest_mode_changed=False,
                              pca_knn_score=None, pca_knn_label=None, pca_knn_percentile=None,
                              pca_wave_only_score=None, pca_wave_only_label=None, pca_mode_changed=False,
                              PC1=None, PC2=None, agreement=None)
                records.append(record)
                features.append(computed.get(i))
                envelopes.append(np.asarray(cached[i]).copy())
                source_lookup[(archive, member, source_row)] = index
                if i in computed:
                    groups[normalized_pv(name)].append(index)
            original_row += 1
        del cached
    manifest = json.loads((HERE / "dd_by_cycle/manifest.json").read_text())
    assert original_row == manifest["rows"] == sum(s["rows"] for s in inventory["files"])
    if not records:
        raise ValueError(f"PV not found: {pv}")
    dimension = len(feature_names)
    features = np.asarray([f if f is not None else np.full(dimension, np.nan) for f in features], dtype=np.float32)
    maximum_lengths = {name: max(records[i]["array_size"] for i in indices) for name, indices in groups.items()}
    for row in records:
        maximum = maximum_lengths.get(row["pv"], row["array_size"])
        row.update(padded_size=maximum, missing_samples=maximum - row["array_size"],
                   missing_fraction=(maximum - row["array_size"]) / maximum if maximum else 0.)
    features = np.column_stack([features, np.asarray([r["missing_fraction"] for r in records], dtype=np.float32)])
    feature_names.append("padding_missing_fraction")
    envelopes = np.asarray(envelopes, dtype=np.float32)
    print(f"Inputs verified: {len(records):,} captures / {len({r['pv'] for r in records})} PVs", flush=True)
    return records, features, envelopes, groups, source_lookup, inventory, feature_names


def fit_forests(records, features, groups, args):
    summaries, modeled = [], {}
    started = time.monotonic()
    for number, (pv, indices) in enumerate(sorted(groups.items()), 1):
        rows = [records[i] for i in indices]
        training = [i for i in indices if records[i]["duplicate_of"] is None]
        unique_cycles = len({r["cycle_id"] for r in rows})
        summary = dict(pv=pv, sample_lengths=", ".join(map(str, sorted({r["array_size"] for r in rows}))),
                       padded_size=max(r["array_size"] for r in rows), rows=len(indices),
                       training_captures=len(training), unique_cycles=unique_cycles,
                       status="modeled" if unique_cycles >= args.minimum else "insufficient_samples",
                       beam_modes=", ".join(sorted({r["beam_mode"] for r in rows})),
                       iforest_candidates=0, pca_knn_candidates=0,
                       iforest_wave_only_candidates=0, iforest_mode_changes=0,
                       pca_wave_only_candidates=0, pca_mode_changes=0,
                       both_candidates=0, candidate_jaccard=None, pca_components=None,
                       pca_retained_variance=None, knn_neighbors=None,
                       iforest_threshold=None, pca_knn_threshold=None)
        summaries.append(summary)
        if unique_cycles < args.minimum:
            for row in rows:
                row["status"] = "insufficient_samples"
                row["mode_count"] = len(training)
                row["rare_mode"] = True
            continue
        numeric = features[training]
        varying = np.ptp(numeric, axis=0) > 0
        base = numeric[:, varying] if varying.any() else np.zeros((len(training), 1))
        queries = features[indices][:, varying] if varying.any() else np.zeros((len(indices), 1))
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        mode_train = encoder.fit_transform([[records[i]["beam_mode"]] for i in training])
        mode_all = encoder.transform([[r["beam_mode"]] for r in rows])
        active = np.ptp(mode_train, axis=0) > 0
        augmented = np.column_stack([base, mode_train[:, active]])
        augmented_queries = np.column_stack([queries, mode_all[:, active]])
        parameters = dict(n_estimators=args.trees, contamination="auto",
                          max_samples=min(256, len(training)), random_state=args.seed, n_jobs=args.workers)
        baseline = IsolationForest(**parameters).fit(base)
        model = IsolationForest(**parameters).fit(augmented)
        scores = -model.score_samples(augmented_queries)
        base_scores = -baseline.score_samples(queries)
        train_position = {records[index]["original_row"]: position for position, index in enumerate(indices)}
        training_scores = np.asarray([scores[train_position[records[index]["original_row"]]] for index in training])
        base_training_scores = np.asarray([base_scores[train_position[records[index]["original_row"]]] for index in training])
        cutoff = score_threshold(training_scores, args.threshold_method, args.fraction, mad_multiplier_for("iforest", args))["threshold"]
        base_cutoff = score_threshold(base_training_scores, args.threshold_method, args.fraction, mad_multiplier_for("iforest", args))["threshold"]
        labels = np.where(scores > cutoff, -1, 1)
        base_labels = np.where(base_scores > base_cutoff, -1, 1)
        training_percentiles = rankdata(training_scores, method="average") / len(training) * 100
        percentile_by_original = {records[index]["original_row"]: float(value) for index, value in zip(training, training_percentiles)}
        mode_counts = collections.Counter(records[i]["beam_mode"] for i in training)
        for position, row in enumerate(rows):
            representative = row["duplicate_of"] if row["duplicate_of"] is not None else row["original_row"]
            row.update(status="modeled", mode_count=mode_counts[row["beam_mode"]],
                       rare_mode=mode_counts[row["beam_mode"]] < args.minimum,
                       iforest_score=float(scores[position]), iforest_label=int(labels[position]),
                       iforest_percentile=percentile_by_original[representative],
                       iforest_wave_only_score=float(base_scores[position]),
                       iforest_wave_only_label=int(base_labels[position]),
                       iforest_mode_changed=bool(labels[position] != base_labels[position]))
        summary.update(iforest_candidates=int((labels == -1).sum()),
                       iforest_wave_only_candidates=int((base_labels == -1).sum()),
                       iforest_mode_changes=int((labels != base_labels).sum()),
                       iforest_threshold=cutoff)
        modeled[pv] = (indices, training, summary)
        if number % 40 == 0 or number == len(groups):
            print(f"Isolation Forest: {number}/{len(groups)} groups ({time.monotonic() - started:.0f}s)", flush=True)
    return summaries, modeled


def materialize_raw(directory, modeled, records, lookup, inventory):
    """Decode original Parquet arrays once and write capture-ordered memmaps."""
    arrays, positions = {}, {}
    for number, (key, (indices, _, _)) in enumerate(sorted(modeled.items())):
        path = directory / f"raw_{number:04d}.npy"
        arrays[key] = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(len(indices), key[1]))
        for local, global_ in enumerate(indices):
            positions[global_] = (key, local)
    remaining = set(positions)
    archives = collections.defaultdict(list)
    for source in inventory["files"]:
        if any((source["archive"], source["member"], j) in lookup and lookup[(source["archive"], source["member"], j)] in positions for j in range(source["rows"])):
            archives[source["archive"]].append(source)
    started = time.monotonic()
    for number, (path, sources) in enumerate(archives.items(), 1):
        touched = set()
        with tarfile.open(path, "r:*") as archive:
            for source in sources:
                position = 0
                with archive.extractfile(source["member"]) as stream:
                    for group in source["metadata"]["4"]:
                        metadata = next(c["3"] for c in group["1"] if c["3"]["3"][0] == "data_array")
                        count = 0
                        for values in decode_waveforms(stream, metadata):
                            index = lookup.get((path, source["member"], position))
                            if index in positions:
                                key, local = positions[index]
                                row = records[index]
                                if len(values) != row["array_size"] or not np.isfinite(values).all():
                                    raise ValueError(f"Original waveform no longer matches: {row['original_row']}")
                                if float(values.min()) != row["waveform_min"] or float(values.max()) != row["waveform_max"]:
                                    raise ValueError(f"Original amplitude no longer matches: {row['original_row']}")
                                arrays[key][local] = values
                                touched.add(key)
                                remaining.remove(index)
                            position += 1
                            count += 1
                        assert count == group["3"]
                assert position == source["rows"]
        for key in touched:
            arrays[key].flush()
            if hasattr(arrays[key]._mmap, "madvise"):
                arrays[key]._mmap.madvise(mmap.MADV_DONTNEED)
        print(f"Original samples: {number}/{len(archives)} archives; {len(remaining):,} captures remaining ({time.monotonic() - started:.0f}s)", flush=True)
    if remaining:
        raise ValueError(f"Missing original waveforms: {len(remaining)}")
    return arrays


class RawCache:
    """Read complete sample rows in bulk from reusable, capture-indexed arrays."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.lookup = {}
        for group in self.manifest["groups"]:
            path = self.directory / group["file"]
            array = np.load(path, mmap_mode="r")
            if array.shape != (len(group["original_rows"]), group["array_size"]) or array.dtype != np.float32:
                raise ValueError(f"Raw cache shape/dtype mismatch: {path}")
            offset = array.offset
            array._mmap.close()
            for position, original in enumerate(group["original_rows"]):
                self.lookup[original] = (path, position, group["array_size"], offset)

    def read(self, originals, pad_to=None):
        length = pad_to if pad_to is not None else self.lookup[originals[0]][2]
        result = np.full((len(originals), length), np.nan, dtype=np.float32)
        by_file = collections.defaultdict(list)
        for destination, original in enumerate(originals):
            path, position, size, offset = self.lookup[original]
            if size > length or (pad_to is None and size != length):
                raise ValueError("Mixed waveform lengths requested from raw cache")
            by_file[(path, offset, size)].append((position, destination))
        for (path, offset, size), requests in by_file.items():
            requests.sort()
            runs = []
            for position, destination in requests:
                if not runs or position != runs[-1][-1][0] + 1:
                    runs.append([])
                runs[-1].append((position, destination))
            with path.open("rb") as stream:
                for run in runs:
                    stream.seek(offset + run[0][0] * size * 4)
                    values = np.frombuffer(stream.read(len(run) * size * 4), dtype=np.float32)
                    if values.size != len(run) * size:
                        raise ValueError(f"Incomplete raw cache: {path}")
                    result[[destination for _, destination in run], :size] = values.reshape(len(run), size)
        return result


def prepare_raw_cache(directory, records, lookup, inventory):
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    fingerprint = json.loads((HERE / "waveform_plots/waveform_beam_modes.json").read_text())["inventory_sha256"]
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"inventory_sha256": fingerprint, "groups": []}
    if manifest["inventory_sha256"] != fingerprint:
        raise ValueError("Raw cache belongs to a different source inventory")
    present = {original for group in manifest["groups"] for original in group["original_rows"]}
    missing = collections.defaultdict(list)
    for index, row in enumerate(records):
        if row["status"] != "invalid_waveform" and row["original_row"] not in present:
            missing[(row["source_pv"], row["array_size"])].append(index)
    if missing:
        part = directory / f"part_{len(manifest['groups']):04d}"
        part.mkdir()
        plan = {key: (indices, None, None) for key, indices in missing.items()}
        arrays = materialize_raw(part, plan, records, lookup, inventory)
        for number, (key, indices) in enumerate(sorted(missing.items())):
            arrays[key]._mmap.close()
            manifest["groups"].append(dict(pv=key[0], array_size=key[1],
                original_rows=[records[i]["original_row"] for i in indices], file=f"{part.name}/raw_{number:04d}.npy"))
        manifest_path.write_text(json.dumps(manifest))
    cache = RawCache(directory)
    assert all(r["status"] == "invalid_waveform" or r["original_row"] in cache.lookup for r in records)
    return cache


def mark_exact_duplicates(records, groups, cache):
    duplicates = 0
    for indices in groups.values():
        by_cycle = collections.defaultdict(list)
        for index in indices:
            row = records[index]
            matched = None
            identity = (row["cycle_id"], row["beam_mode"], row["array_size"])
            for previous in by_cycle[identity]:
                values = cache.read([records[previous]["original_row"], row["original_row"]])
                if np.array_equal(values[0], values[1]):
                    matched = records[previous]["original_row"]
                    break
            if matched is None:
                by_cycle[identity].append(index)
            else:
                row["duplicate_of"] = matched
                duplicates += 1
    print(f"Exact same-cycle duplicate captures excluded from training: {duplicates}", flush=True)


def fit_pca_knn(records, modeled, cache, args, sketch=None):
    started = time.monotonic()
    checkpoints = Path(args.output) / "pca_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    with threadpool_limits(limits=args.workers):
        for number, (key, (indices, training, summary)) in enumerate(sorted(modeled.items()), 1):
            lengths = [records[i]["array_size"] for i in training]
            originals = np.asarray([records[i]["original_row"] for i in training], dtype=np.int64)
            checkpoint = checkpoints / (hashlib.sha256(key.encode()).hexdigest()[:20] + ".npz")
            signature = json.dumps(dict(version=1, retained=args.retained, seed=args.seed,
                                        global_components=args.global_components,
                                        global_training=len(sketch.training) if sketch else 0,
                                        padded_size=summary["padded_size"]))
            reused = checkpoint.exists()
            if reused:
                with np.load(checkpoint, allow_pickle=False) as saved:
                    if str(saved["signature"]) != signature or not np.array_equal(saved["originals"], originals):
                        raise ValueError(f"PCA checkpoint differs from current inputs/settings: {key}")
                    points, count, retained = saved["points"], int(saved["count"]), float(saved["retained"])
                    if sketch:
                        sketch.restore(training, {k: saved[k] for k in saved.files if k.startswith("global_")})
            else:
                raw = cache.read(originals, pad_to=summary["padded_size"])
                assert all(np.isnan(raw[j, size:]).all() and np.isfinite(raw[j, :size]).all() for j, size in enumerate(lengths))
                part = sketch.update(training, raw) if sketch else {}
                points, count, retained = raw_pca(raw, args.retained)
                del raw
                temporary = checkpoint.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, signature=signature, originals=originals,
                                    points=points, count=count, retained=retained, **part)
                temporary.replace(checkpoint)
            mask = padding_mask_features(lengths, summary["padded_size"]) * args.missing_weight
            waveform_and_mask = np.column_stack([points, mask])
            modes = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform([[records[i]["beam_mode"]] for i in training])
            augmented = np.column_stack([waveform_and_mask, modes * args.mode_weight / np.sqrt(2)])
            baseline, k = neighbor_scores(waveform_and_mask, args.neighbors, args.workers)
            scores, _ = neighbor_scores(augmented, args.neighbors, args.workers)
            base_labels, _ = distance_labels(baseline, args.fraction, args.threshold_method, mad_multiplier_for("pca_knn", args))
            labels, threshold = distance_labels(scores, args.fraction, args.threshold_method, mad_multiplier_for("pca_knn", args))
            percentiles = rankdata(scores, method="average") / len(scores) * 100
            position_by_original = {records[index]["original_row"]: position for position, index in enumerate(training)}
            for index in indices:
                row = records[index]
                representative = row["duplicate_of"] if row["duplicate_of"] is not None else row["original_row"]
                position = position_by_original[representative]
                row.update(pca_knn_score=float(scores[position]), pca_knn_label=int(labels[position]),
                           pca_knn_percentile=float(percentiles[position]),
                           pca_wave_only_score=float(baseline[position]),
                           pca_wave_only_label=int(base_labels[position]),
                           pca_mode_changed=bool(labels[position] != base_labels[position]),
                           PC1=float(points[position, 0]), PC2=float(points[position, 1]) if points.shape[1] > 1 else 0.0,
                           agreement=bool(row["iforest_label"] == int(labels[position])))
            both = int(sum(records[i]["iforest_label"] == -1 and records[i]["pca_knn_label"] == -1 for i in indices))
            candidates = sum(records[i]["pca_knn_label"] == -1 for i in indices)
            union = summary["iforest_candidates"] + candidates - both
            summary.update(pca_knn_candidates=candidates, both_candidates=both,
                           pca_wave_only_candidates=sum(records[i]["pca_wave_only_label"] == -1 for i in indices),
                           pca_mode_changes=sum(records[i]["pca_mode_changed"] for i in indices),
                           candidate_jaccard=both / union if union else 1.0,
                           pca_components=count, pca_retained_variance=retained,
                           knn_neighbors=k, pca_knn_threshold=threshold)
            print(f"Full waveform PCA + kNN: {number}/{len(modeled)} PVs; {count} PCs; "
                  f"{'restored' if reused else 'checkpoint saved'} ({time.monotonic() - started:.0f}s)", flush=True)


def metrics(rows):
    paired = [r for r in rows if r["agreement"] is not None]
    if_count = sum(r["iforest_label"] == -1 for r in paired)
    pca_count = sum(r["pca_knn_label"] == -1 for r in paired)
    both = sum(r["iforest_label"] == -1 and r["pca_knn_label"] == -1 for r in paired)
    union = if_count + pca_count - both
    result = dict(rows=len(rows), modeled=len(paired), unscored=len(rows) - len(paired),
                iforest_candidates=if_count, pca_knn_candidates=pca_count, both_candidates=both,
                iforest_only=if_count - both, pca_knn_only=pca_count - both,
                candidate_jaccard=both / union if union else None,
                label_agreement=sum(r["agreement"] for r in paired) / len(paired) if paired else None,
                training_duplicates=sum(r["duplicate_of"] is not None for r in rows),
                unique_modeled=sum(r["duplicate_of"] is None for r in paired),
                iforest_mode_changes=sum(r["iforest_mode_changed"] for r in paired),
                pca_mode_changes=sum(r["pca_mode_changed"] for r in paired))
    if any("global_pca_knn_label" in r for r in rows):
        triple = [r for r in paired if r.get("global_pca_knn_label") is not None]
        global_count = sum(r["global_pca_knn_label"] == -1 for r in triple)
        ig = sum(r["iforest_label"] == -1 and r["global_pca_knn_label"] == -1 for r in triple)
        pg = sum(r["pca_knn_label"] == -1 and r["global_pca_knn_label"] == -1 for r in triple)
        all_three = sum(r["iforest_label"] == -1 and r["pca_knn_label"] == -1 and r["global_pca_knn_label"] == -1 for r in triple)
        result.update(global_pca_knn_candidates=global_count,
                      iforest_global_both=ig, local_global_both=pg, all_three_candidates=all_three,
                      iforest_global_jaccard=ig / (if_count + global_count - ig) if if_count + global_count - ig else None,
                      local_global_jaccard=pg / (pca_count + global_count - pg) if pca_count + global_count - pg else None,
                      global_modeled=len(triple),
                      union_candidates=if_count + pca_count + global_count - both - ig - pg + all_three)
    return result


def write_viewer(output, records, envelopes, pv_rows, args):
    has_global = any("global_pca_knn_label" in r for r in records)
    global_option = '<option value="global_pca_knn">全PV共通 PCA＋kNN（生波形＋PV＋モード＋欠損）</option>' if has_global else ""
    global_headers = "<th>共通PCA順位%</th>" if has_global else ""
    global_label_header = "<th>共通PCA判定</th>" if has_global else ""
    pages = output / "pv"
    pages.mkdir(exist_ok=True)
    assets = HERE / "waveform_comparison_viewer"
    for filename in ["viewer.js", "viewer.css"]:
        (output / filename).write_text((assets / filename).read_text())
    pv_summaries = []
    started = time.monotonic()
    for number, (pv, indices) in enumerate(sorted(pv_rows.items()), 1):
        rows = [records[i] for i in indices]
        filename = f"pv_{number:03d}.html"
        summary = dict(pv=pv, source_pvs=", ".join(sorted({r["source_pv"] for r in rows})),
                       beam_modes=", ".join(sorted({r["beam_mode"] for r in rows})),
                       **metrics(rows), file=f"pv/{filename}")
        pv_summaries.append(summary)
        data = []
        references = {}
        for i in indices:
            r = records[i]
            if r["status"] != "invalid_waveform":
                a = envelopes[i].reshape(64, 4, 2)
                lo, hi = a[:, :, 0].min(axis=1), a[:, :, 1].max(axis=1)
                preview = [[float(f"{v:.6g}") for v in lo], [float(f"{v:.6g}") for v in hi]]
            else:
                preview = [[], []]
            data.append({**r, "preview": preview})
        for length, mode in sorted({(r["array_size"], r["beam_mode"]) for r in rows}):
            options = [j for j, r in enumerate(rows) if r["array_size"] == length and r["beam_mode"] == mode and r["iforest_label"] == 1 and r["pca_knn_label"] == 1 and r.get("global_pca_knn_label", 1) == 1]
            if not options:
                options = [j for j, r in enumerate(rows) if r["array_size"] == length and r["beam_mode"] == mode and r["status"] != "invalid_waveform"]
            if options:
                references[f"{length}|{mode}"] = min(options, key=lambda j: abs((rows[j]["iforest_percentile"] or 50) - 50))
        thresholds = {}
        for detector in ["iforest", "pca_knn", "global_pca_knn"]:
            source = next((r for r in rows if r.get(detector + "_threshold") is not None), None)
            if source:
                thresholds[detector] = dict(value=source[detector + "_threshold"], percentile=source[detector + "_threshold_percentile"])
        names = dict(iforest="IF", pca_knn="PV別PCA", global_pca_knn="共通PCA")
        threshold_text = " ／ ".join(f"{names[k]}: {v['value']:.6g}" for k, v in thresholds.items())
        payload = json.dumps(dict(pv=pv, rows=data, references=references, summary=summary, thresholds=thresholds), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
        document = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(pv)}</title><link rel="stylesheet" href="../viewer.css"><body><main><a href="../index.html">全PVの比較へ</a><h1 id="title"></h1><p id="summary"></p><p>スコア閾値：{html.escape(threshold_text)}</p><p class="note">赤い輪は選択した手法の異常候補、金色の輪はユーザー確認済み異常です。破線は実際の閾値に対応する順位です。点を選ぶと波形を確認できます。判定は正常・故障の確定ラベルではありません。</p><div class="controls"><label>判定手法 <select id="method"><option value="iforest">Isolation Forest</option><option value="pca_knn">PV別 PCA＋kNN（生波形＋モード＋欠損マスク）</option>{global_option}</select></label><label>モード <select id="mode"><option value="">すべて</option></select></label><label>表示 <select id="filter"><option value="all">すべて</option><option value="candidate">異常候補</option><option value="different">IFと選択PCAの判定が異なる</option><option value="both">IFと選択PCAの共通候補</option></select></label><label>散布図 <select id="axes"><option value="rank">異常スコアの順位を比較</option><option value="pca">生波形のPC1・PC2</option></select></label></div><div id="legend"></div><svg id="scatter" viewBox="0 0 1000 440" role="img" aria-label="異常判定の比較"></svg><h2 id="detailTitle">波形を選択してください</h2><p id="detail"></p><svg id="waveform" viewBox="0 0 1000 300" role="img" aria-label="選択波形と同条件の代表例"></svg><p class="note">波形表示は64区間の最小値・最大値です。PCAの学習には要約前の全サンプルを使っています。灰色の区間はNaNで表したデータのない部分です。破線は同じモード・波形長の実際の代表例です。</p><h2>サイクル一覧</h2><div class="tablebox"><table><thead><tr><th>cycle_id</th><th>beam_mode</th><th>波形長</th><th>IF順位%</th><th>PV別PCA順位%</th>{global_headers}<th>IF判定</th><th>PV別PCA判定</th>{global_label_header}</tr></thead><tbody id="cycles"></tbody></table></div></main><script>window.DATA={payload};</script><script src="../viewer.js"></script></body></html>'''
        (pages / filename).write_text(document)
        if number % 100 == 0 or number == len(pv_rows):
            print(f"Comparison viewers: {number}/{len(pv_rows)} PVs ({time.monotonic() - started:.0f}s)", flush=True)
    return pv_summaries


def publish(output, records, envelopes, groups, feature_names, features, args):
    pv_rows, mode_rows = collections.defaultdict(list), collections.defaultdict(list)
    for i, row in enumerate(records):
        pv_rows[row["pv"]].append(i)
        mode_rows[row["beam_mode"]].append(row)
    method = getattr(args, "threshold_method", "median_mad")
    thresholds = apply_score_thresholds(records, pv_rows, args)
    known_evaluation = attach_known_anomalies(output, records)
    refresh_group_summaries(groups, records, thresholds)
    write_csv(output / "thresholds.csv", thresholds)
    has_global = any("global_pca_knn_label" in r for r in records)
    global_details = json.loads((output / "global_model_summary.json").read_text()) if has_global else None
    if has_global:
        global_threshold = next(t for t in thresholds if t["detector"] == "global_pca_knn")
        global_details.update({k: global_threshold[k] for k in ["threshold_method", "median", "mad", "mad_multiplier", "threshold", "threshold_percentile"]})
        global_details["candidate_fraction"] = args.fraction if method == "quantile" else None
        global_details["candidate_count_unique"] = sum(r.get("global_pca_knn_label") == -1 and r["duplicate_of"] is None for r in records)
        (output / "global_model_summary.json").write_text(json.dumps(global_details, indent=2) + "\n")
    pv_summaries = write_viewer(output, records, envelopes, pv_rows, args)
    mode_summaries = [dict(beam_mode=mode, **metrics(rows)) for mode, rows in sorted(mode_rows.items())]
    totals = metrics(records)
    summary = dict(**totals, pvs=len(pv_rows), modeled_pvs=sum(r["modeled"] > 0 for r in pv_summaries),
                   statuses=dict(collections.Counter(r["status"] for r in records)),
                   candidate_fraction=args.fraction if method == "quantile" else None,
                   threshold_method=method, mad_multiplier=mad_multiplier_for("iforest", args) if len({mad_multiplier_for(d, args) for d in ["iforest", "pca_knn", "global_pca_knn"]}) == 1 else None, mad_consistency_scale=1.,
                   mad_multipliers={d: mad_multiplier_for(d, args) for d in ["iforest", "pca_knn", "global_pca_knn"]},
                   known_anomaly_check=known_evaluation,
                   minimum_unique_cycles=args.minimum,
                   group_by=["normalized_pv"],
                   pv_normalization="Ignore -DODData/-PMData suffix; preserve source_pv for provenance",
                   source_pvs=len({r["source_pv"] for r in records}),
                   iforest_input="79 waveform/envelope/missing-fraction statistics plus one-hot beam_mode; one model per normalized PV",
                   pca_input="Every original float32 sample; NaN tail padding; sample-position mean imputation and centering; no resampling or per-waveform normalization",
                   pca_retained_variance=args.retained,
                   knn_input="Retained full-waveform PCs, exact tail-missingness mask and one-hot beam_mode; one model per normalized PV",
                   knn_mode_weight=args.mode_weight, missing_mask_weight=args.missing_weight,
                   knn_neighbors=args.neighbors,
                   global_model=global_details,
                   global_candidate_distribution=dict(
                       pvs_with_candidates=sum(r.get("global_pca_knn_candidates", 0) > 0 for r in pv_summaries),
                       pvs_with_all_waveforms_candidates=sum(r["modeled"] > 0 and r.get("global_pca_knn_candidates", 0) == r["modeled"] for r in pv_summaries)) if has_global else None,
                   global_knn_input="Shared raw-waveform PCs plus one-hot normalized PV, one-hot beam_mode and exact missingness mask" if has_global else None,
                   deduplication="Exactly equal full waveforms from the same normalized PV, mode, cycle and length are fitted once; scores are copied to their original rows",
                   threshold="score > median(score) + k * median(abs(score - median(score))); raw unscaled MAD, k recorded per detector in mad_multipliers; per normalized PV for local methods and one pooled threshold for the global model; exact duplicate training captures excluded" if method == "median_mad" else "Per-PV unique-training-score quantile for local methods; one pooled unique-training-score quantile for the global model",
                   interpretation="Exploratory in-sample rankings. User-confirmed positives were used for provisional threshold calibration; other captures remain unlabelled. No held-out accuracy evaluation. Agreement is not accuracy." if known_evaluation else "Exploratory in-sample rankings. No confirmed nominal/fault labels, no held-out accuracy evaluation. Agreement is not accuracy.",
                   versions=dict(numpy=np.__version__, sklearn=sklearn.__version__),
                   checkpoint_provenance=json.loads((output / "checkpoint_provenance.json").read_text()) if (output / "checkpoint_provenance.json").exists() else None,
                   parameters=dict(trees=args.trees, seed=args.seed, workers=args.workers),
                   feature_names=feature_names, mode_summary=mode_summaries)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    methods = ["iforest", "pca_knn"] + (["global_pca_knn"] if has_global else [])
    write_csv(output / "classification.csv", records)
    write_csv(output / "anomaly_candidates.csv", [r for r in records if any(r.get(m + "_label") == -1 for m in methods)])
    if has_global:
        write_csv(output / "global_pca_knn_candidates.csv", sorted([r for r in records if r.get("global_pca_knn_label") == -1], key=lambda r: r["global_pca_knn_score"], reverse=True))
    write_csv(output / "pv_summary.csv", pv_summaries)
    write_csv(output / "mode_summary.csv", mode_summaries)
    write_csv(output / "group_summary.csv", groups)
    np.savez_compressed(output / "iforest_features.npz", features=features, names=np.asarray(feature_names), original_row=np.asarray([r["original_row"] for r in records]))
    pairwise = []
    for first, second, field in [("Isolation Forest", "PV-specific PCA+kNN", "candidate_jaccard"),
                                 ("Isolation Forest", "Global PCA+kNN", "iforest_global_jaccard"),
                                 ("PV-specific PCA+kNN", "Global PCA+kNN", "local_global_jaccard")]:
        if field in totals:
            pairwise.append(dict(first=first, second=second, candidate_jaccard=totals[field]))
    write_csv(output / "pairwise_comparison.csv", pairwise)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained")
    x = np.arange(len(mode_summaries))
    bars = [("iforest_candidates", "#2563eb", "PV-specific IF"),
            ("pca_knn_candidates", "#dc2626", "PV-specific PCA+kNN")]
    if has_global:
        bars.append(("global_pca_knn_candidates", "#16965c", "Global PCA+kNN"))
    width = .8 / len(bars)
    for j, (field, color, label) in enumerate(bars):
        axes[0].bar(x + (j - (len(bars) - 1) / 2) * width,
                    [r[field] / max(r["modeled"], 1) * 100 for r in mode_summaries], width, color=color, label=label)
    axes[0].set_xticks(x, [r["beam_mode"] for r in mode_summaries])
    axes[0].set_ylabel("Candidate rate (%)")
    axes[0].set_title("Candidates by beam mode")
    axes[0].legend(fontsize=8)
    axes[1].bar([r["first"].replace("Isolation Forest", "IF").replace("PV-specific PCA+kNN", "PV PCA").replace("Global PCA+kNN", "Global PCA") + "\nvs " + r["second"].replace("PV-specific PCA+kNN", "PV PCA").replace("Global PCA+kNN", "Global PCA") for r in pairwise],
                [100 * (r["candidate_jaccard"] or 0) for r in pairwise], color=["#7c3aed", "#2563eb", "#16965c"][:len(pairwise)])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Candidate Jaccard (%)")
    axes[1].set_title("Candidate overlap (not accuracy)")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    def rate(value):
        return f"{value:.1%}" if value is not None else "—"
    table = []
    for r in pv_summaries:
        global_cell = f"<td>{r['global_pca_knn_candidates']}</td>" if has_global else ""
        table.append(f'<tr><td><a href="{r["file"]}">{html.escape(r["pv"])}</a></td><td>{html.escape(r["beam_modes"])}</td><td>{r["modeled"]:,}/{r["rows"]:,}</td><td>{r["iforest_candidates"]}</td><td>{r["pca_knn_candidates"]}</td>{global_cell}<td>{r["both_candidates"]}</td><td>{rate(r["candidate_jaccard"])}</td></tr>')
    global_header = "<th>共通PCA候補</th>" if has_global else ""
    coefficient_text = " ／ ".join(f"{name}={mad_multiplier_for(d, args):g}" for d, name in [("iforest", "IF"), ("pca_knn", "PV別PCA"), ("global_pca_knn", "共通PCA")] if has_global or d != "global_pca_knn")
    threshold_text = f"中央値＋k×MAD（k：{coefficient_text}、正規分布補正なし）" if method == "median_mad" else f"スコア上位{args.fraction:.0%}"
    zero_mad_note = "MADが0の場合は中央値を閾値にし、" if method == "median_mad" else ""
    label_note = "確認済み異常は上記の実例のみで、それ以外は未確認です。" if known_evaluation else "正常・異常の確定ラベルはありません。"
    global_text = '<p><a href="all_waveforms_threshold_map.html">全波形の集合と判定閾値のマップ（MAD・中央値＋nσを切替）</a> · <a href="all_waveforms_5sigma_map.png">中央値＋5σの全体図 PNG</a> · <a href="sigma5_anomalous_cycles.csv">5σ候補サイクルCSV</a> · <a href="sigma5_candidates.csv">5σ候補波形CSV</a> · <a href="sigma5_summary.json">5σの集計・条件</a></p>' if has_global and method == "median_mad" else ""
    if known_evaluation:
        found = known_evaluation["detected"]
        global_text += f'<p class="note">ユーザー確認済み異常{known_evaluation["confirmed_anomalies"]}波形の検出数：IF {found["iforest"]}、PV別PCA {found["pca_knn"]}、共通PCA {found["global_pca_knn"]}。この実例を使った暫定調整であり、独立した精度評価ではありません。他の波形の正常・異常は未確認です。</p><p><a href="known_anomalies.csv">確認済み異常</a> · <a href="threshold_diagnostics/index.html">MAD倍率と波形の診断</a></p>'
    if has_global:
        global_text += f"""<p><b>全PV共通モデル: PCA＋kNN候補 {totals['global_pca_knn_candidates']:,}件 ／ 3手法すべての共通候補 {totals['all_three_candidates']:,}件</b><br>候補集合のJaccard: IFと共通PCA {rate(totals['iforest_global_jaccard'])} ／ PV別PCAと共通PCA {rate(totals['local_global_jaccard'])}。</p><p>共通モデルは、全{len(pv_rows)} PVの元波形を1つのPCAで学習し、正規化PV名・beam_mode・欠損マスクを特徴量に加えています。最大{global_details['padded_size']:,}サンプル、主成分{global_details['components']}個、実際の保持分散{global_details['retained_variance']:.2%}。PVごとの振幅の標準化は行っていません。全体のスコアから{threshold_text}で1つの閾値を求めるため、PVごとの候補率は一定ではありません。</p><p><a href="global_pca_knn_candidates.csv">共通PCA候補CSV</a> · <a href="global_model_summary.json">共通モデルの条件</a> · <a href="global_pca_knn_model.npz">共通PCA＋kNNモデル</a> · <a href="pairwise_comparison.csv">手法間比較CSV</a></p>"""
    if has_global:
        distribution = summary["global_candidate_distribution"]
        global_text += f'<p class="note">共通モデルでは{distribution["pvs_with_candidates"]} PVに候補があり、{distribution["pvs_with_all_waveforms_candidates"]} PVでは全波形が候補です。全PVで同じスケールとしきい値を使うため、PVごとの候補率は異なります。各PVの詳細で実波形を確認してください。</p>'
    page = f"""<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>波形異常候補モデル比較</title><link rel="stylesheet" href="viewer.css"><main><h1>波形異常候補モデル比較</h1><p>{len(pv_rows)} PV・{totals['rows']:,}波形、判定対象 {totals['modeled']:,}波形。PV名末尾のDODData／PMDataを統合しています。</p><p>PV別Isolation ForestとPV別PCA＋kNNは、それぞれ各PVにつき1モデルです。運転モード・波形長でモデルを分割していません。beam_modeはカテゴリ特徴量です。IFは256区間の最小値・最大値から作った特徴量、PCAは元波形の全サンプルを使用しています。</p><p>短い波形の末尾をNaNとし、PCA計算時にサンプル位置ごとの平均で補完しています。PV別PCAでは分散の{args.retained:.0%}を残し、beam_modeと欠損マスクを加えて近傍距離を計算しています。波形・モード・欠損の重みは実行条件に記録しています。</p><p><b>PV別IF候補 {totals['iforest_candidates']:,}件 ／ PV別PCA候補 {totals['pca_knn_candidates']:,}件 ／ 共通 {totals['both_candidates']:,}件</b><br>候補集合のJaccard: {rate(totals['candidate_jaccard'])}。</p>{global_text}<p>閾値は{threshold_text}で決定し、スコアが閾値を超える波形を候補にしています。PV別モデルはPVごと、共通モデルは全体から閾値を計算します。{zero_mad_note}閾値と同点の波形は候補に含めません。同一サイクルの完全一致波形は学習で重複排除し、結果には元の全行を残しています。無効波形は保留しています。</p><p class="note">{label_note}一致率は精度を表しません。各PVのリンクから候補と代表波形を確認できます。</p><p><a href="classification.csv">全判定CSV</a> · <a href="anomaly_candidates.csv">異常候補CSV</a> · <a href="pv_summary.csv">PV別集計</a> · <a href="mode_summary.csv">モード別集計</a> · <a href="thresholds.csv">中央値・MAD・閾値一覧</a> · <a href="summary.json">実行条件</a></p><img class="overview" src="comparison.png" alt="モード別候補率と候補集合の一致率"><p><input id="search" placeholder="PV名・beam_modeで検索"></p><div class="tablebox"><table><thead><tr><th>PV（詳細へ）</th><th>beam_mode</th><th>判定対象/全波形</th><th>IF候補</th><th>PV別PCA候補</th>{global_header}<th>IF・PV別PCA共通</th><th>IF・PV別PCA Jaccard</th></tr></thead><tbody>{''.join(table)}</tbody></table></div></main><script>document.getElementById('search').oninput=function(){{const q=this.value.toLowerCase();for(const row of document.querySelectorAll('tbody tr'))row.hidden=!row.textContent.toLowerCase().includes(q);}};</script></html>"""
    (output / "index.html").write_text(page)
    if has_global and method == "median_mad":
        from build_all_waveform_threshold_map import build
        build(output)
    print("FINAL", json.dumps({k: summary[k] for k in ["pvs", "modeled_pvs", "modeled", "unscored", "iforest_candidates", "pca_knn_candidates", "both_candidates", "candidate_jaccard", "label_agreement", "statuses"]}), flush=True)


def sync_iforest_checkpoint(output, records):
    """Keep the standalone IF result current while retaining its IF-only schema."""
    path = output / "iforest_classification.csv"
    if not path.exists():
        return
    by_original = {r["original_row"]: r for r in records}
    fields = ["iforest_label", "iforest_wave_only_label", "iforest_mode_changed",
              "iforest_threshold", "iforest_threshold_percentile"]
    temporary = path.with_suffix(".tmp.csv")
    with path.open() as source, temporary.open("w", newline="", encoding="utf-8") as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=list(dict.fromkeys([*reader.fieldnames, *fields])))
        writer.writeheader()
        count = 0
        for row in reader:
            current = by_original[int(row["original_row"])]
            if row["iforest_score"]:
                assert float(row["iforest_score"]) == current["iforest_score"]
            for field in fields:
                row[field] = current.get(field)
            writer.writerow(row)
            count += 1
        assert count == len(records)
    temporary.replace(path)


def rethreshold(args):
    """Change decision rules from saved scores without recomputing any model."""
    output = Path(args.output).resolve()
    previous = json.loads((output / "summary.json").read_text())
    previous_method = previous.get("threshold_method", "quantile")
    previous_multipliers = previous.get("mad_multipliers", {d: previous.get("mad_multiplier", 1.) or 1. for d in ["iforest", "pca_knn", "global_pca_knn"]})
    tag = f"quantile_{previous['candidate_fraction']:g}" if previous_method == "quantile" else "median_mad_" + "_".join(f"{d}-{k:g}" for d, k in previous_multipliers.items())
    history = output / "threshold_history" / tag
    history.mkdir(parents=True, exist_ok=True)
    for filename in ["classification.csv", "iforest_classification.csv", "anomaly_candidates.csv", "pv_summary.csv", "mode_summary.csv",
                     "group_summary.csv", "pairwise_comparison.csv", "summary.json", "global_model_summary.json",
                     "comparison.png", "thresholds.csv", "validation.json"]:
        source, destination = output / filename, history / filename
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
    args.minimum = previous["minimum_unique_cycles"]
    args.retained = previous["pca_retained_variance"]
    args.neighbors = previous["knn_neighbors"]
    args.mode_weight = previous["knn_mode_weight"]
    args.missing_weight = previous["missing_mask_weight"]
    args.trees, args.seed = previous["parameters"]["trees"], previous["parameters"]["seed"]
    records, features, envelopes, groups, _, _, names = load_inputs(Path(args.plot_root).resolve(), args.pv)
    restore_classification(records, output / "classification.csv")
    methods = ["iforest", "pca_knn"] + (["global_pca_knn"] if previous.get("global_model") else [])
    old_candidates = {method: {r["original_row"] for r in records if r.get(method + "_label") == -1} for method in methods}
    thresholds = apply_score_thresholds(records, groups, args)
    if "global_pca_knn" in methods:
        global_stats = next(t for t in thresholds if t["detector"] == "global_pca_knn")
        model_path = output / "global_pca_knn_model.npz"
        with np.load(model_path, allow_pickle=False) as saved:
            model = {key: saved[key] for key in saved.files}
        model.update(threshold=np.asarray(global_stats["threshold"]),
                     threshold_method=np.asarray(args.threshold_method),
                     threshold_median=np.asarray(global_stats["median"]),
                     threshold_mad=np.asarray(global_stats["mad"]),
                     threshold_mad_multiplier=np.asarray(global_stats["mad_multiplier"]),
                     threshold_percentile=np.asarray(global_stats["threshold_percentile"]))
        temporary = model_path.with_suffix(".tmp.npz")
        np.savez(temporary, **model)
        temporary.replace(model_path)
    with (output / "group_summary.csv").open() as stream:
        summaries = list(csv.DictReader(stream))
    publish(output, records, envelopes, summaries, names, features, args)
    sync_iforest_checkpoint(output, records)
    changes = {}
    for method in methods:
        before = old_candidates[method]
        after = {r["original_row"] for r in records if r.get(method + "_label") == -1}
        changes[method] = dict(before=len(before), after=len(after), added=len(after - before),
                               removed=len(before - after), common=len(before & after))
    result = dict(previous_method=previous_method, threshold_method=args.threshold_method,
                  mad_multipliers={d: mad_multiplier_for(d, args) for d in methods},
                  rule="score > median(score) + k * median(abs(score - median(score)))" if args.threshold_method == "median_mad" else "score > training-score quantile",
                  scores_reused=True, history=str(history.relative_to(output)), changes=changes)
    (output / "threshold_change_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RETHRESHOLDED", json.dumps(result), flush=True)


def run(args):
    if args.rethreshold:
        return rethreshold(args)
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True, exist_ok=args.resume)
    root = Path(args.plot_root).resolve()
    if args.resume and (output / "interruption_status.json").exists():
        saved_configuration = json.loads((output / "interruption_status.json").read_text())["configuration"]
        for name, saved_name in [("minimum", "minimum_unique_cycles"),
                                 ("trees", "trees"), ("seed", "seed")]:
            if getattr(args, name) != saved_configuration[saved_name]:
                raise ValueError(f"Cannot change {name} while reusing the completed Isolation Forest checkpoint")
    records, features, envelopes, groups, lookup, inventory, names = load_inputs(root, args.pv)
    def compute(directory):
        cache = prepare_raw_cache(directory, records, lookup, inventory)
        if args.resume:
            restore_classification(records, output / "iforest_classification.csv")
            summaries, modeled = restore_forests(records, groups, args)
        else:
            mark_exact_duplicates(records, groups, cache)
            summaries, modeled = fit_forests(records, features, groups, args)
            write_csv(output / "iforest_classification.csv", records)
        print("PV-specific Isolation Forest complete; starting NaN-padded full waveform PCA + kNN", flush=True)
        from global_waveform_pca_knn import GlobalSketch
        training = [i for i, r in enumerate(records) if r["status"] == "modeled" and r["duplicate_of"] is None]
        sketch = GlobalSketch(records, training, args.global_components, args.seed)
        fit_pca_knn(records, modeled, cache, args, sketch)
        sketch.save(output / "global_pca_sketch.npz")
        return summaries
    if args.raw_cache:
        summaries = compute(Path(args.raw_cache).resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="anomaly-pca-raw-", dir=HERE) as temporary:
            summaries = compute(Path(temporary))
    publish(output, records, envelopes, summaries, names, features, args)
    print(f"Complete: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-root", default=str(HERE / "waveform_plots"))
    parser.add_argument("--output", default=str(HERE / "waveform_anomaly_comparison"))
    parser.add_argument("--pv", help="Optional exact PV; by default process all PVs")
    parser.add_argument("--fraction", type=float, default=.05)
    parser.add_argument("--threshold-method", choices=["median_mad", "quantile"], default="median_mad")
    parser.add_argument("--mad-multiplier", type=float, default=1.)
    for detector in ["iforest", "pca_knn", "global_pca_knn"]:
        parser.add_argument("--" + detector.replace("_", "-") + "-mad-multiplier", type=float)
    parser.add_argument("--rethreshold", action="store_true", help="Update all labels and reports from saved scores without retraining")
    parser.add_argument("--retained", type=float, default=.95)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--raw-cache", help="Optional reusable original-sample cache")
    parser.add_argument("--resume", action="store_true", help="Reuse saved Isolation Forest and per-PV PCA checkpoints")
    parser.add_argument("--mode-weight", type=float, default=1.)
    parser.add_argument("--missing-weight", type=float, default=1.)
    parser.add_argument("--global-components", type=int, default=96)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--minimum", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 0 < args.fraction <= .5 or not 0 < args.retained <= 1 or args.neighbors < 1 or args.minimum < 2 or args.trees < 1 or args.workers < 1 or args.mode_weight < 0 or args.missing_weight < 0:
        parser.error("Invalid candidate fraction, PCA variance, neighbors, sample count, tree count or workers")
    run(args)
