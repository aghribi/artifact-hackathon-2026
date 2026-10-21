"""Streaming shared PCA + kNN using raw samples, PV, mode and missingness.

Run after waveform_anomaly_comparison.py has completed its per-PV models:
    python global_waveform_pca_knn.py --resume --raw-cache anomaly-pca-reuse-20261015

Reusable inference:
    model = SharedPCAModel("waveform_anomaly_comparison/global_pca_knn_model.npz")
    result = model.predict([original_waveform], [source_pv], [beam_mode])

Labels -1 and +1 mean exploratory anomaly and nominal candidates. Training
scores exclude the query itself; inference queries use the saved training set.
"""
import json
import argparse
import csv
import time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import pairwise_distances_chunked
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits


class GlobalSketch:
    """Accumulate a randomized PCA range while local models read all samples.

    No sample positions or waveforms are subsampled. Missing tails are excluded
    from means and retained separately through observed lengths.
    """
    def __init__(self, records, training, components=96, seed=42):
        self.training = list(training)
        self.position = {index: j for j, index in enumerate(training)}
        self.originals = np.asarray([records[i]["original_row"] for i in training], dtype=np.int64)
        self.lengths = np.asarray([records[i]["array_size"] for i in training], dtype=np.int32)
        self.width = int(self.lengths.max())
        self.components = components
        self.rank = min(components + 16, len(training) - 1, self.width)
        self.omega = np.random.default_rng(seed).standard_normal((self.width, self.rank))
        self.projected = np.zeros((len(training), self.rank), dtype=np.float64)
        self.sums = np.zeros(self.width, dtype=np.float64)
        self.squares = np.zeros(self.width, dtype=np.float64)
        self.counts = np.zeros(self.width, dtype=np.int64)
        self.seen = np.zeros(len(training), dtype=bool)
        self.pv_categories = sorted({records[i]["pv"] for i in training})
        self.mode_categories = sorted({records[i]["beam_mode"] for i in training})
        pv_lookup = {value: j for j, value in enumerate(self.pv_categories)}
        mode_lookup = {value: j for j, value in enumerate(self.mode_categories)}
        self.pv_codes = np.asarray([pv_lookup[records[i]["pv"]] for i in training], dtype=np.int32)
        self.mode_codes = np.asarray([mode_lookup[records[i]["beam_mode"]] for i in training], dtype=np.int32)

    def update(self, indices, raw):
        destinations = np.asarray([self.position[index] for index in indices])
        if self.seen[destinations].any():
            raise ValueError("A waveform was included twice in shared PCA")
        clean = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0.)
        width = clean.shape[1]
        part = dict(global_projected=clean @ self.omega[:width],
                    global_sums=clean.sum(axis=0),
                    global_squares=np.einsum("ij,ij->j", clean, clean),
                    global_counts=np.zeros(width, dtype=np.int64))
        lengths, counts = np.unique(self.lengths[destinations], return_counts=True)
        for length, count in zip(lengths, counts):
            part["global_counts"][:length] += count
        self.restore(indices, part)
        return part

    def restore(self, indices, part):
        destinations = np.asarray([self.position[index] for index in indices])
        if self.seen[destinations].any():
            raise ValueError("A waveform was included twice in shared PCA")
        width = len(part["global_sums"])
        self.projected[destinations] = part["global_projected"]
        self.sums[:width] += part["global_sums"]
        self.squares[:width] += part["global_squares"]
        self.counts[:width] += part["global_counts"]
        self.seen[destinations] = True

    def save(self, path):
        if not self.seen.all():
            raise ValueError(f"Shared PCA is missing {int((~self.seen).sum())} training captures")
        np.savez(path, originals=self.originals, lengths=self.lengths, omega=self.omega,
                 projected=self.projected, sums=self.sums, squares=self.squares,
                 counts=self.counts, pv_codes=self.pv_codes, mode_codes=self.mode_codes,
                 pv_categories=np.asarray(self.pv_categories), mode_categories=np.asarray(self.mode_categories),
                 max_components=np.asarray(self.components))
        print(f"Shared PCA sketch saved: {len(self.training):,} complete original waveforms", flush=True)


def mask_features(lengths, edges, weight=1.):
    """Preserve query-to-training distances even for unseen intermediate lengths.

    All training masks are constant inside each interval. An extra orthogonal
    coordinate retains the residual when a query ends inside an interval.
    """
    widths = np.diff(edges)
    fractions = np.clip((edges[1:][None, :] - np.asarray(lengths)[:, None]) / widths, 0, 1)
    scale = widths / edges[-1]
    return np.column_stack([fractions * np.sqrt(scale),
                            np.sqrt((fractions * (1 - fractions) * scale).sum(axis=1))]) * weight


def categorical_neighbor_scores(points, pv_codes, neighbors=10, pv_weight=1., workers=2,
                                queries=None, query_pvs=None, exclude_self=True):
    """Exact kNN for [points, one_hot(PV) * weight/sqrt(2)].

    Different PVs have a known distance lower bound of pv_weight. Searching
    within a category is only an optimization; queries that can have closer
    cross-category neighbors are checked against EVERY training waveform.
    There is one global embedding, distance metric, and decision threshold.
    """
    training_query = queries is None
    queries = points if training_query else np.asarray(queries)
    query_pvs = pv_codes if query_pvs is None else np.asarray(query_pvs)
    k = min(neighbors, len(points) - int(training_query and exclude_self))
    if k < 1:
        raise ValueError("At least two training waveforms are required")
    scores = np.empty(len(queries), dtype=np.float64)
    unresolved = []
    for code in np.unique(query_pvs):
        source = np.flatnonzero(pv_codes == code) if code >= 0 else np.array([], dtype=int)
        destination = np.flatnonzero(query_pvs == code)
        count = k + int(training_query and exclude_self)
        if len(source) < count or pv_weight <= 0:
            unresolved.extend(destination)
            continue
        search = NearestNeighbors(n_neighbors=count, algorithm="brute", n_jobs=workers).fit(points[source])
        distances = search.kneighbors(queries[destination])[0]
        if training_query and exclude_self:
            distances = distances[:, 1:]
        certified = distances[:, -1] < pv_weight
        scores[destination[certified]] = distances[certified].mean(axis=1)
        unresolved.extend(destination[~certified])
    unresolved = np.asarray(sorted(unresolved), dtype=int)
    print(f"Global kNN: {len(queries) - len(unresolved):,} queries certified by categorical distance; "
          f"{len(unresolved):,} queries require all-PV search", flush=True)
    completed = 0
    # Bounded batches keep the pairwise-distance matrix below 64 MiB.
    batch = max(1, int(64 * 1024 ** 2 / (8 * len(points))))
    for start in range(0, len(unresolved), batch):
        indices = unresolved[start:start + batch]
        chunks = pairwise_distances_chunked(queries[indices], points, metric="euclidean",
                                            n_jobs=workers, working_memory=64)
        offset = 0
        for distances in chunks:
            current = indices[offset:offset + len(distances)]
            distances **= 2
            distances += (query_pvs[current, None] != pv_codes[None, :]) * pv_weight ** 2
            if training_query and exclude_self:
                distances[np.arange(len(current)), current] = np.inf
            smallest = np.partition(distances, k - 1, axis=1)[:, :k]
            scores[current] = np.sqrt(np.maximum(smallest, 0)).mean(axis=1)
            offset += len(distances)
        completed += len(indices)
        if completed % (batch * 100) == 0 or completed == len(unresolved):
            print(f"Global kNN exhaustive queries: {completed:,}/{len(unresolved):,}", flush=True)
    return scores, k, len(unresolved)


def centered_batches(cache, originals, lengths, pv_codes, means):
    """Read every original sample, grouping only for efficient file access."""
    for code in np.unique(pv_codes):
        positions = np.flatnonzero(pv_codes == code)
        width = int(lengths[positions].max())
        raw = cache.read(originals[positions], pad_to=width).astype(np.float64)
        raw -= means[:width]
        np.nan_to_num(raw, copy=False, nan=0.)
        yield positions, raw


def fit_shared_pca(sketch, cache, retained=.95, workers=2):
    """Randomized PCA on complete raw samples, with exact variance accounting."""
    originals, lengths, codes = (sketch[key] for key in ["originals", "lengths", "pv_codes"])
    counts, sums = sketch["counts"], sketch["sums"]
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    total_energy = float(np.maximum(sketch["squares"] - sums * means, 0).sum())
    projected, omega = sketch["projected"].copy(), sketch["omega"]
    for length in np.unique(lengths):
        projected[lengths == length] -= means[:length] @ omega[:length]
    print(f"Global PCA range: {len(originals):,} x {len(means):,}, rank {omega.shape[1]}", flush=True)
    started = time.monotonic()
    with threadpool_limits(limits=workers):
        q = np.linalg.qr(projected, mode="reduced")[0]
        del projected
        b = np.zeros((q.shape[1], len(means)), dtype=np.float64)
        for number, (positions, raw) in enumerate(centered_batches(cache, originals, lengths, codes, means), 1):
            b[:, :raw.shape[1]] += q[positions].T @ raw
            if number % 100 == 0:
                print(f"Global PCA covariance pass: {number} PV batches ({time.monotonic() - started:.0f}s)", flush=True)
        del q
        eigenvalues, u = np.linalg.eigh(b @ b.T)
        eigenvalues, u = np.maximum(eigenvalues[::-1], 0), u[:, ::-1]
        positive = eigenvalues > max(eigenvalues[0] * 1e-14, 1e-24)
        available = int(positive.sum())
        if available == 0:
            raise ValueError("Global waveforms have no usable variance")
        cumulative = np.cumsum(eigenvalues) / total_energy
        desired = int(np.searchsorted(cumulative, retained) + 1)
        count = min(desired, available, int(sketch["max_components"]))
        if cumulative[count - 1] < retained and desired <= available:
            count = desired  # Allow the oversampling columns to meet the target.
        components = (u[:, :count].T @ b) / np.sqrt(eigenvalues[:count, None])
        del b
        scores = np.empty((len(originals), count), dtype=np.float64)
        for number, (positions, raw) in enumerate(centered_batches(cache, originals, lengths, codes, means), 1):
            scores[positions] = raw @ components[:, :raw.shape[1]].T
            if number % 100 == 0:
                print(f"Global PCA transform pass: {number} PV batches ({time.monotonic() - started:.0f}s)", flush=True)
    captured = float(np.einsum("ij,ij->", scores, scores))
    scale = np.sqrt(captured / max(len(originals) - 1, 1))
    details = dict(algorithm="Streaming randomized PCA; all raw samples and all unique training waveforms",
                   training_captures=len(originals), padded_size=len(means), components=count,
                   sketch_rank=omega.shape[1], target_retained_variance=retained,
                   retained_variance=captured / total_energy, total_centered_energy=total_energy,
                   waveform_scale=scale, elapsed_seconds=time.monotonic() - started)
    print("Global PCA fitted: " + json.dumps(details), flush=True)
    return means, components, scores / max(scale, 1e-12), scale, details


class SharedPCAModel:
    """Reusable global model. predict accepts raw arrays, source PVs and modes."""
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as saved:
            self.values = {key: saved[key] for key in saved.files}

    def transform(self, waveforms, pvs, modes):
        from waveform_anomaly_comparison import normalized_pv
        v = self.values
        if not (len(waveforms) == len(pvs) == len(modes)):
            raise ValueError("Waveform, PV and mode counts must match")
        lengths = np.asarray([len(w) for w in waveforms], dtype=int)
        if (lengths < 1).any() or (lengths > len(v["means"])).any():
            raise ValueError(f"Waveform lengths must be 1..{len(v['means'])}; retrain for longer signals")
        raw = np.full((len(waveforms), int(lengths.max())), np.nan, dtype=np.float64)
        for j, waveform in enumerate(waveforms):
            if not np.isfinite(waveform).all():
                raise ValueError("Observed waveform samples must be finite")
            raw[j, :len(waveform)] = waveform
        raw -= v["means"][:raw.shape[1]]
        np.nan_to_num(raw, copy=False, nan=0.)
        pcs = raw @ v["components"][:, :raw.shape[1]].T / float(v["waveform_scale"])
        pv_lookup = {str(name): j for j, name in enumerate(v["pv_categories"])}
        mode_lookup = {str(name): j for j, name in enumerate(v["mode_categories"])}
        pv_codes = np.asarray([pv_lookup.get(normalized_pv(name), -1) for name in pvs])
        mode_codes = np.asarray([mode_lookup.get(name, len(mode_lookup)) for name in modes])
        mode_features = np.eye(len(mode_lookup) + 1)[mode_codes] * float(v["mode_weight"]) / np.sqrt(2)
        mask = mask_features(lengths, v["mask_edges"], float(v["missing_weight"]))
        return np.column_stack([pcs, mask, mode_features]), pv_codes

    def predict(self, waveforms, pvs, modes, workers=2):
        points, pv_codes = self.transform(waveforms, pvs, modes)
        v = self.values
        with threadpool_limits(limits=workers):
            scores, _, _ = categorical_neighbor_scores(v["training_embedding"], v["pv_codes"],
                int(v["neighbors"]), float(v["pv_weight"]), workers,
                queries=points, query_pvs=pv_codes, exclude_self=False)
        return dict(score=scores, label=np.where(scores > float(v["threshold"]), -1, 1))


def run(args):
    import waveform_anomaly_comparison as local
    output = Path(args.output).resolve()
    previous = json.loads((output / "summary.json").read_text())
    if args.mad_multiplier is None:
        inherited = previous.get("mad_multipliers", {})
        for detector in ["iforest", "pca_knn", "global_pca_knn"]:
            if getattr(args, detector + "_mad_multiplier") is None:
                setattr(args, detector + "_mad_multiplier", inherited.get(detector, previous.get("mad_multiplier") or 1.))
    with np.load(output / "global_pca_sketch.npz", allow_pickle=False) as saved:
        sketch = {key: saved[key] for key in saved.files}
    cache = local.RawCache(args.raw_cache)
    fit_path = output / "global_pca_fit.npz"
    if args.resume and fit_path.exists():
        with np.load(fit_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["originals"], sketch["originals"]) or float(saved["target"]) != args.retained:
                raise ValueError("Global PCA checkpoint differs from current inputs/settings")
            means, components, pcs, scale = saved["means"], saved["components"], saved["pcs"], float(saved["scale"])
            details = json.loads(str(saved["details"]))
        print("Restored fitted global PCA checkpoint", flush=True)
    else:
        means, components, pcs, scale, details = fit_shared_pca(sketch, cache, args.retained, args.workers)
        temporary = fit_path.with_suffix(".tmp.npz")
        np.savez(temporary, means=means, components=components, pcs=pcs, scale=scale,
                 originals=sketch["originals"], target=args.retained, details=json.dumps(details))
        temporary.replace(fit_path)
    details["retained_variance_target_met"] = details["retained_variance"] >= args.retained
    details["component_limit"] = int(sketch["max_components"])
    edges = np.unique(np.r_[0, sketch["lengths"], len(means)])
    mask = mask_features(sketch["lengths"], edges, args.missing_weight)
    modes = np.eye(len(sketch["mode_categories"]) + 1)[sketch["mode_codes"]] * args.mode_weight / np.sqrt(2)
    embedding = np.column_stack([pcs, mask, modes])
    with threadpool_limits(limits=args.workers):
        scores, k, exhaustive = categorical_neighbor_scores(embedding, sketch["pv_codes"],
                                                             args.neighbors, args.pv_weight, args.workers)
    multiplier = local.mad_multiplier_for("global_pca_knn", args)
    labels, threshold = local.distance_labels(scores, args.fraction, args.threshold_method, multiplier)
    threshold_stats = local.score_threshold(scores, args.threshold_method, args.fraction, multiplier)
    percentiles = rankdata(scores, method="average") / len(scores) * 100
    details.update(neighbors=k, **threshold_stats, candidate_fraction=args.fraction if args.threshold_method == "quantile" else None,
                   pv_weight=args.pv_weight, mode_weight=args.mode_weight, missing_weight=args.missing_weight,
                   categorical_distance="Different PV or beam_mode adds weight squared to squared Euclidean distance",
                   threshold_scope="One global threshold on unique training waveforms", exhaustive_knn_queries=exhaustive,
                   candidate_count_unique=int((labels == -1).sum()))
    np.savez(output / "global_pca_knn_model.npz", means=means, components=components,
             waveform_scale=scale, training_embedding=embedding, originals=sketch["originals"],
             pv_codes=sketch["pv_codes"], pv_categories=sketch["pv_categories"],
             mode_categories=sketch["mode_categories"], mask_edges=edges, neighbors=k,
             pv_weight=args.pv_weight, mode_weight=args.mode_weight, missing_weight=args.missing_weight,
             threshold=threshold, threshold_method=np.asarray(args.threshold_method),
             threshold_median=threshold_stats["median"], threshold_mad=threshold_stats["mad"],
             threshold_mad_multiplier=multiplier,
             threshold_percentile=threshold_stats["threshold_percentile"],
             training_scores=scores, retained_variance=details["retained_variance"])
    (output / "global_model_summary.json").write_text(json.dumps(details, indent=2) + "\n")
    records, features, envelopes, _, _, _, names = local.load_inputs(Path(args.plot_root).resolve())
    # Reload the completed per-PV comparison without fitting it a second time.
    local.restore_classification(records, output / "classification.csv")
    positions = {int(original): j for j, original in enumerate(sketch["originals"])}
    for record in records:
        original = record["duplicate_of"] if record["duplicate_of"] is not None else record["original_row"]
        j = positions.get(original)
        record.update(global_pca_knn_score=float(scores[j]) if j is not None else None,
                      global_pca_knn_label=int(labels[j]) if j is not None else None,
                      global_pca_knn_percentile=float(percentiles[j]) if j is not None else None,
                      global_PC1=float(pcs[j, 0]) if j is not None else None,
                      global_PC2=float(pcs[j, 1]) if j is not None and pcs.shape[1] > 1 else 0. if j is not None else None,
                      global_padded_size=len(means),
                      global_missing_samples=len(means) - record["array_size"])
    with (output / "group_summary.csv").open() as stream:
        groups = list(csv.DictReader(stream))
    previous = json.loads((output / "summary.json").read_text())
    args.minimum = previous["minimum_unique_cycles"]
    args.trees = previous["parameters"]["trees"]
    args.seed = previous["parameters"]["seed"]
    local.publish(output, records, envelopes, groups, names, features, args)
    print(f"All three comparisons complete: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="waveform_anomaly_comparison")
    parser.add_argument("--raw-cache", default="anomaly-pca-reuse-20261015")
    parser.add_argument("--plot-root", default="waveform_plots")
    parser.add_argument("--retained", type=float, default=.95)
    parser.add_argument("--fraction", type=float, default=.05)
    parser.add_argument("--threshold-method", choices=["median_mad", "quantile"], default="median_mad")
    parser.add_argument("--mad-multiplier", type=float)
    for detector in ["iforest", "pca_knn", "global_pca_knn"]:
        parser.add_argument("--" + detector.replace("_", "-") + "-mad-multiplier", type=float)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--pv-weight", type=float, default=1.)
    parser.add_argument("--mode-weight", type=float, default=1.)
    parser.add_argument("--missing-weight", type=float, default=1.)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Reuse the fitted global PCA checkpoint")
    run(parser.parse_args())
