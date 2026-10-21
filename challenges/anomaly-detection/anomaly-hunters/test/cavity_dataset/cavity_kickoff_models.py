"""Kickoff sections 7/8 applied to the complete, current ESS cavity dataset.

Weak detection labels are DD=0 and final PM pulse per cavity=1, not confirmed
cavity faults. Cause targets are recorded systems, with unlisted events excluded.
All validation splits hold out whole event IDs. Explanations describe the fitted
prediction, not physical causality. No SHAP/LIME approximation is claimed.
"""
from pathlib import Path
import json
import hashlib
import inspect

import joblib
import numpy as np
import pandas as pd

import cavity_all_cavity_trip_clusters as all_cavity


SEED = 42
SCOPES = ("waveform", "parameters", "combined")
MODEL_NAMES = ("Forest", "Small tree")
TABLES = ("data_support", "weak_fold_metrics", "weak_summary", "weak_predictions", "weak_event_predictions",
          "year_metrics", "split_audit", "weak_global_explanations", "weak_local_cases", "weak_local_contributions",
          "cluster_memberships", "cluster_summary", "cluster_profiles", "cause_support", "cause_metrics",
          "cause_predictions", "cause_global_explanations", "cause_local_contributions", "cause_permutation",
          "conditional_cause_metrics", "unknown_cause_scores")


def stack_bins(series):
    return np.stack([np.asarray(a, float) if isinstance(a, (list, np.ndarray)) and len(a) == 64
                     else np.full(64, np.nan) for a in series])


def load_data(root):
    full, labeled = all_cavity.load_source_tables(root)
    labeled = labeled.reset_index(drop=True)
    waves = {s: stack_bins(labeled[s + "_bins"]) for s in all_cavity.BIN_SIGNALS}
    metadata = labeled.drop(columns=[s + "_bins" for s in all_cavity.BIN_SIGNALS]).copy()
    metadata["event_year"] = metadata.event_ts.dt.year
    metadata["capture"] = metadata.source + ":" + metadata.event_id.astype(str)
    trip_path = Path(root) / "signal_comparison" / "all_cavity_trip_clusters"
    # Refresh the source analysis if source tables or implementation changed.
    paths = [trip_path / f"{name}.csv" for name in all_cavity.TABLES]
    inputs = all_cavity.input_paths(root) + [Path(all_cavity.__file__)]
    fresh = all(p.is_file() for p in paths) and min(p.stat().st_mtime for p in paths) >= max(p.stat().st_mtime for p in inputs)
    if fresh:
        prior = all_cavity.load_results(trip_path)
    else:
        prior = all_cavity.make_result(root)
        all_cavity.save_results(prior, trip_path)
    del full
    return metadata, waves, prior


def pulse_inputs(metadata, waves):
    """Instantaneous features only; never source, event ID, year, rank or cause.

    All 64 bins contribute to 16 contiguous four-bin blocks per waveform.
    Amplitudes use their own robust range; phase sin/cos is weighted by field
    amplitude, suppressing phase outside the RF pulse. Scalar levels retain
    amplitude information removed by waveform normalization.
    """
    frame = pd.DataFrame(index=metadata.index)
    for signal in ("field_amp", "fwd_amp", "refl_amp"):
        x = waves[signal]
        valid = np.isfinite(x).all(axis=1)
        low, high = np.full(len(x), np.nan), np.full(len(x), np.nan)
        low[valid], high[valid] = np.percentile(x[valid], [5, 95], axis=1)
        normalized = (x - low[:, None]) / np.maximum((high - low)[:, None], 1e-8)
        frame[f"wave.{signal}.level"] = high - low
        for block, column in enumerate(normalized.reshape(len(x), 16, 4).mean(axis=2).T, 1):
            frame[f"wave.{signal}.block{block:02}"] = column
        if signal == "field_amp":
            rf_weight = np.clip(normalized, 0, 1)
    phase = np.deg2rad(waves["field_phase"])
    for name, matrix in (("sin", np.sin(phase)), ("cos", np.cos(phase))):
        for block, column in enumerate((matrix * rf_weight).reshape(len(matrix), 16, 4).mean(axis=2).T, 1):
            frame[f"wave.field_phase_{name}.block{block:02}"] = column
    for name in all_cavity.SCALARS:
        if name == "phase_mean_deg":
            frame["shape.phase_mean_sin"] = np.sin(np.deg2rad(metadata[name]))
            frame["shape.phase_mean_cos"] = np.cos(np.deg2rad(metadata[name]))
        elif name in ("flat_len", "field_end_after_rf_off"):
            frame["shape." + name + "_fraction"] = metadata[name] / metadata.array_size
        else:
            frame["shape." + name] = metadata[name]
    frame["shape.has_pulse"] = metadata.has_pulse.astype(float)
    waveform_columns = frame.columns.tolist()
    frame = frame.copy()
    frame["beam_mode"], frame["cavity"] = metadata.beam_mode, metadata.cavity
    for name in ("beam_present", "beam_len", "trig_delay"):
        frame[name] = metadata[name].astype(float)
    return frame.replace([np.inf, -np.inf], np.nan), waveform_columns


def capture_weights(meta):
    """Each capture totals one, with equal weight per cavity inside a capture."""
    count = meta.groupby(["capture", "cavity"]).capture.transform("size")
    cavities = meta.groupby("capture").cavity.transform("nunique")
    return (1 / (count * cavities)).to_numpy(float)


class ExplainedForest:
    """Training-only imputation/encoding plus a forest or a small rule tree."""
    def __init__(self, name="Forest"):
        self.name = name

    def fit(self, frame, y, sample_weight=None):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        self.columns = frame.columns.tolist()
        categorical = frame.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        numeric = [c for c in self.columns if c not in categorical]
        transformers = []
        if numeric:
            transformers.append(("numeric", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), numeric))
        if categorical:
            transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
        self.preprocessor = ColumnTransformer(transformers, verbose_feature_names_out=False)
        x = self.preprocessor.fit_transform(frame)
        self.feature_names = self.preprocessor.get_feature_names_out()
        if self.name == "Forest":
            self.model = RandomForestClassifier(n_estimators=80, max_depth=5, min_samples_leaf=2,
                                                min_weight_fraction_leaf=.025, max_features="sqrt",
                                                bootstrap=False, random_state=SEED, n_jobs=2)
        else:
            self.model = DecisionTreeClassifier(max_depth=3, min_weight_fraction_leaf=.05,
                                                min_samples_leaf=2, random_state=SEED)
        self.model.fit(x, y, sample_weight=sample_weight)
        return self

    def predict(self, frame):
        return self.model.predict_proba(self.preprocessor.transform(frame[self.columns]))[:, 1]

    def explain(self, frame):
        """Exact additive path contributions in probability units (not SHAP).

        A feature receives child-node probability minus parent-node probability
        along the actual decision path, averaged over trees. Interactions and
        correlated features make the attribution depend on the tree paths.
        """
        # sklearn traverses float32 inputs using double thresholds. Compare as
        # double AFTER rounding inputs to float32, matching its C tree traversal.
        x = self.preprocessor.transform(frame[self.columns]).astype(np.float32).astype(float)
        contributions = np.zeros_like(x, dtype=float)
        base = 0.
        estimators = self.model.estimators_ if hasattr(self.model, "estimators_") else [self.model]
        for estimator in estimators:
            tree = estimator.tree_
            counts = tree.value[:, 0, :]
            probabilities = counts[:, 1] / counts.sum(axis=1)
            base += probabilities[0] / len(estimators)
            pending = [(0, np.arange(len(x)))]
            while pending:
                node, indices = pending.pop()
                feature = tree.feature[node]
                if feature < 0 or not len(indices):
                    continue
                left = x[indices, feature] <= tree.threshold[node]
                for child, selected in ((tree.children_left[node], indices[left]),
                                        (tree.children_right[node], indices[~left])):
                    if len(selected):
                        contributions[selected, feature] += (probabilities[child] - probabilities[node]) / len(estimators)
                        pending.append((child, selected))
        scores = self.predict(frame)
        if not np.allclose(base + contributions.sum(axis=1), scores, atol=1e-10):
            raise AssertionError(f"Path contributions do not reproduce predictions: max error {np.max(abs(base + contributions.sum(axis=1) - scores))}")
        return base, contributions


def scope_columns(frame, waveform_columns, scope, cavity=True):
    context = [c for c in frame.columns if c not in waveform_columns and (c != "cavity" or not cavity)]
    # For pulse models, cavity is present in all three comparisons.
    if cavity:
        context = [c for c in context if c != "cavity"]
    columns = waveform_columns.copy() if scope in ("waveform", "combined") else []
    if scope in ("parameters", "combined"):
        columns += context
    if cavity:
        columns += ["cavity"]
    return columns


def metrics(y, score, weights=None):
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(np.unique(y)) < 2:
        return {"roc_auc": np.nan, "average_precision": np.nan}
    return {"roc_auc": roc_auc_score(y, score, sample_weight=weights),
            "average_precision": average_precision_score(y, score, sample_weight=weights)}


def event_scores(meta, probability):
    table = meta[["capture", "source", "event_id", "cavity", "label", "event_year"]].copy()
    table["score"] = probability
    cavity = table.groupby(["capture", "source", "event_id", "cavity", "label", "event_year"], sort=False).score.mean()
    return cavity.groupby(level=["capture", "source", "event_id", "label", "event_year"], sort=False).mean().reset_index()


def weak_validation(metadata, inputs, waveform_columns):
    from sklearn.model_selection import StratifiedGroupKFold
    capture = metadata.drop_duplicates("capture")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(splitter.split(capture, capture.label, groups=capture.event_id))
    y = metadata.label.to_numpy()
    rows, audits, case_rows, local_rows = [], [], [], []
    predictions = metadata[["dataset_index", "capture", "source", "event_id", "cycle_id", "cavity", "label", "event_year"]].copy()
    global_parts = []
    for fold, (train_cap, test_cap) in enumerate(folds, 1):
        train_captures, test_captures = set(capture.iloc[train_cap].capture), set(capture.iloc[test_cap].capture)
        train = np.flatnonzero(metadata.capture.isin(train_captures))
        test = np.flatnonzero(metadata.capture.isin(test_captures))
        if set(metadata.iloc[train].event_id) & set(metadata.iloc[test].event_id):
            raise AssertionError("Capture leakage")
        weight = capture_weights(metadata.iloc[train])
        for position, indices in (("train", train), ("test", test)):
            for cap, group in metadata.iloc[indices].groupby("capture", sort=False):
                audits.append({"task": "weak DD/PM", "target": "DD/PM", "fold": fold, "partition": position,
                               "capture": cap, "event_id": group.event_id.iloc[0], "label": group.label.iloc[0]})
        for scope in SCOPES:
            columns = scope_columns(inputs, waveform_columns, scope)
            for model_name in MODEL_NAMES:
                model = ExplainedForest(model_name).fit(inputs.iloc[train][columns], y[train], sample_weight=weight)
                score = model.predict(inputs.iloc[test][columns])
                column = f"{model_name}__{scope}"
                predictions.loc[test, column] = score
                event = event_scores(metadata.iloc[test], score)
                rows.append({"model": model_name, "scope": scope, "fold": fold,
                             "train_captures": len(train_captures), "test_captures": len(event),
                             "normal_test_captures": event.label.eq(0).sum(), "trip_test_captures": event.label.eq(1).sum(),
                             **metrics(event.label, event.score)})
                if scope == "combined" and model_name == "Forest":
                    base, contribution = model.explain(inputs.iloc[test][columns])
                    weights = capture_weights(metadata.iloc[test])
                    global_parts.append(pd.DataFrame({"feature": model.feature_names,
                        "weighted_abs_sum": (abs(contribution) * weights[:, None]).sum(axis=0),
                        "weight_sum": weights.sum(), "fold": fold}))
                    # Show one measured waveform per held-out capture: its highest PM score.
                    selection = metadata.iloc[test][["capture"]].copy()
                    selection["local_row"], selection["score"] = np.arange(len(test)), score
                    chosen = selection.sort_values("score", ascending=False).drop_duplicates("capture")
                    for _, example in chosen.iterrows():
                        local = int(example.local_row)
                        original = int(test[local])
                        info = metadata.iloc[original]
                        case_rows.append({"dataset_index": info.dataset_index, "capture": info.capture,
                                          "event_id": info.event_id, "cavity": info.cavity, "label": info.label,
                                          "fold": fold, "score": score[local], "base_score": base})
                        top = np.argsort(-abs(contribution[local]))[:12]
                        for j in top:
                            local_rows.append({"dataset_index": info.dataset_index, "capture": info.capture,
                                               "fold": fold, "feature": model.feature_names[j],
                                               "contribution": contribution[local, j], "score": score[local], "base_score": base})
                        local_rows.append({"dataset_index": info.dataset_index, "capture": info.capture, "fold": fold,
                                           "feature": "Other features (sum)", "score": score[local], "base_score": base,
                                           "contribution": contribution[local].sum() - contribution[local, top].sum()})
            print(f"  fold {fold}: {scope} models complete", flush=True)
        print(f"Weak-label grouped validation: fold {fold}/5 complete", flush=True)
    fold_metrics = pd.DataFrame(rows)
    summary = fold_metrics.groupby(["model", "scope"], sort=False).agg(
        mean_event_auc=("roc_auc", "mean"), std_event_auc=("roc_auc", "std"),
        mean_event_AP=("average_precision", "mean"), std_event_AP=("average_precision", "std")
    ).reset_index()
    event_predictions = []
    for model_name in MODEL_NAMES:
        for scope in SCOPES:
            column = f"{model_name}__{scope}"
            if predictions[column].isna().any():
                raise AssertionError("Missing held-out predictions")
            event = event_scores(metadata, predictions[column].to_numpy())
            event["model"], event["scope"] = model_name, scope
            event_predictions.append(event)
    global_frame = pd.concat(global_parts, ignore_index=True)
    global_frame["importance"] = global_frame.weighted_abs_sum / global_frame.weight_sum
    global_summary = global_frame.groupby("feature").agg(
        mean_abs_probability_contribution=("importance", "mean"), fold_std=("importance", "std")
    ).reset_index().sort_values("mean_abs_probability_contribution", ascending=False)
    return {"weak_fold_metrics": fold_metrics, "weak_summary": summary, "weak_predictions": predictions,
            "weak_event_predictions": pd.concat(event_predictions, ignore_index=True), "split_audit": pd.DataFrame(audits),
            "weak_global_explanations": global_summary, "weak_local_cases": pd.DataFrame(case_rows),
            "weak_local_contributions": pd.DataFrame(local_rows)}


def year_validation(metadata, inputs, waveform_columns):
    rows = []
    for year in sorted(metadata.event_year.unique()):
        train = np.flatnonzero(metadata.event_year.ne(year))
        test = np.flatnonzero(metadata.event_year.eq(year))
        train_events = metadata.iloc[train].drop_duplicates("capture")
        test_events = metadata.iloc[test].drop_duplicates("capture")
        for scope in SCOPES:
            columns = scope_columns(inputs, waveform_columns, scope)
            model = ExplainedForest().fit(inputs.iloc[train][columns], metadata.iloc[train].label,
                                            sample_weight=capture_weights(metadata.iloc[train]))
            score = model.predict(inputs.iloc[test][columns])
            event = event_scores(metadata.iloc[test], score)
            rows.append({"test_year": year, "scope": scope, "normal_train_captures": train_events.label.eq(0).sum(),
                         "trip_train_captures": train_events.label.eq(1).sum(), "normal_test_captures": test_events.label.eq(0).sum(),
                         "trip_test_captures": test_events.label.eq(1).sum(), **metrics(event.label, event.score)})
        print(f"Leave-year-out diagnostic: {year} complete", flush=True)
    return pd.DataFrame(rows)


def event_inputs(prior):
    trip = prior["trip_features"].copy()
    events = prior["event_metadata"].copy()
    for column in ("event_ts",):
        events[column] = pd.to_datetime(events[column], utc=True)
    cavities = prior["cavity_support"].cavity.tolist()
    columns = pd.MultiIndex.from_product([cavities, all_cavity.FEATURES])
    wide = trip.set_index(["event_id", "cavity"])[all_cavity.FEATURES].unstack("cavity").swaplevel(0, 1, axis=1)
    wide = wide.reindex(index=events.event_id, columns=columns)
    # Fixed physical unit scaling also avoids tiny differences being lost in tree float32 comparisons.
    for cavity, feature in columns:
        wide[(cavity, feature)] /= all_cavity.FLOORS[feature]
    wide.columns = [c + " | " + f for c, f in columns]
    waveform_columns = wide.columns.tolist()
    inputs = wide.reset_index(drop=True)
    for name in ("beam_mode", "beam_present", "beam_len"):
        inputs[name] = events[name].to_numpy()
    inputs["beam_present"] = inputs.beam_present.astype(float)
    for section in sorted(trip.section.unique()):
        delay = trip.loc[trip.section.eq(section)].groupby("event_id").trig_delay.median()
        inputs["trig_delay_" + section] = events.event_id.map(delay).to_numpy()
    return events, inputs, waveform_columns


def mixed_clustering(prior, events, inputs, waveform_columns):
    from sklearn.cluster import HDBSCAN
    from sklearn.manifold import MDS
    from sklearn.metrics import adjusted_mutual_info_score, pairwise_distances
    from sklearn.preprocessing import OneHotEncoder
    from threadpoolctl import threadpool_limits
    used = prior["memberships"].query("method == 'HDBSCAN'").event_id.tolist()
    positions = events.set_index("event_id").index.get_indexer(used)
    meta = events.iloc[positions].reset_index(drop=True)
    context = inputs.iloc[positions][scope_columns(inputs, waveform_columns, "parameters", cavity=False)].copy()
    categories = OneHotEncoder(sparse_output=False).fit_transform(context[["beam_mode"]])
    numbers = context.drop(columns="beam_mode").to_numpy(float)
    medians = np.nanmedian(numbers, axis=0)
    numbers = np.where(np.isfinite(numbers), numbers, medians)
    # A scale floor of one recorded unit avoids amplifying tiny delay differences.
    scale = np.maximum(numbers.std(axis=0), 1.)
    numbers = (numbers - medians) / scale
    d_context = pairwise_distances(np.c_[categories, numbers])
    lookup = {event: i for i, event in enumerate(used)}
    d_wave = np.zeros((len(used), len(used)))
    for row in prior["distance_overlap"].itertuples():
        i, j = lookup[row.event_id_a], lookup[row.event_id_b]
        d_wave[i, j] = d_wave[j, i] = row.distance
    off_diagonal = np.triu_indices(len(used), 1)
    d_wave /= np.median(d_wave[off_diagonal])
    context_scale = np.median(d_context[off_diagonal])
    if context_scale > 0:
        d_context /= context_scale
    memberships, summaries, profiles = [], [], []
    for name, distance in (("Waveform changes", d_wave),
                           ("Waveform 80% + parameters 20%", np.sqrt(.8 * d_wave ** 2 + .2 * d_context ** 2))):
        with threadpool_limits(limits=2):
            labels = HDBSCAN(min_cluster_size=4, min_samples=3, metric="precomputed").fit_predict(distance)
            labels = np.where(labels < 0, -1, labels + 1)
            xy = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED, n_init=4,
                     normalized_stress="auto").fit_transform(distance)
        membership = meta.copy()
        membership["representation"], membership["cluster"] = name, labels
        membership[["MDS1", "MDS2"]] = xy
        memberships.append(membership)
        known = meta.cause_status.eq("matched")
        summaries.append({"representation": name, "events": len(meta), "clusters": len(set(labels) - {-1}),
                          "noise_events": (labels == -1).sum(),
                          "cause_AMI": adjusted_mutual_info_score(meta.loc[known, "trip_cause"], labels[known]),
                          "year_AMI": adjusted_mutual_info_score(meta.event_year, labels),
                          "beam_mode_AMI": adjusted_mutual_info_score(meta.beam_mode, labels)})
        profiles.append(membership.groupby(["representation", "cluster", "beam_mode", "beam_present", "event_year", "trip_cause"],
                                           sort=False).size().rename("events").reset_index())
    return {"cluster_memberships": pd.concat(memberships, ignore_index=True), "cluster_summary": pd.DataFrame(summaries),
            "cluster_profiles": pd.concat(profiles, ignore_index=True)}


def cause_validation(prior, events, inputs, waveform_columns):
    from sklearn.model_selection import StratifiedKFold
    used = set(prior["memberships"].query("method == 'HDBSCAN'").event_id)
    eligible = events.event_id.isin(used)
    known = eligible & events.cause_status.eq("matched")
    indices = np.flatnonzero(known)
    cause_sets = events.trip_cause.str.split(",").map(lambda a: {s.strip() for s in a})
    systems = sorted(set().union(*cause_sets[known]))
    support, rows, predictions, globals_, locals_, permutations, conditional, unknown_scores, audits = [], [], [], [], [], [], [], [], []
    fitted = {}
    for system in systems:
        y = cause_sets.iloc[indices].map(lambda a: int(system in a)).to_numpy()
        positives, negatives = int(y.sum()), int((1-y).sum())
        support.append({"system": system, "positive_captures": positives, "other_annotated_captures": negatives,
                        "status": "evaluated" if min(positives, negatives) >= 4 else "too_few_labeled_captures"})
        if min(positives, negatives) < 4:
            continue
        folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(indices, y))
        for scope in SCOPES:
            columns = scope_columns(inputs, waveform_columns, scope, cavity=False)
            out = np.full(len(indices), np.nan)
            fold_scores = []
            for fold, (train, test) in enumerate(folds, 1):
                model = ExplainedForest().fit(inputs.iloc[indices[train]][columns], y[train])
                score = model.predict(inputs.iloc[indices[test]][columns])
                out[test] = score
                fold_scores.append(metrics(y[test], score))
                if scope == "combined":
                    for position, part in (("train", train), ("test", test)):
                        for j in part:
                            audits.append({"task": "recorded cause", "target": system, "fold": fold,
                                           "partition": position, "capture": "PM:" + str(events.event_id.iloc[indices[j]]),
                                           "event_id": events.event_id.iloc[indices[j]], "label": y[j]})
                    base, contribution = model.explain(inputs.iloc[indices[test]][columns])
                    globals_.append(pd.DataFrame({"system": system, "fold": fold, "feature": model.feature_names,
                                                   "mean_abs_contribution": abs(contribution).mean(axis=0)}))
                    for local, j in enumerate(test):
                        top = np.argsort(-abs(contribution[local]))[:8]
                        for feature in top:
                            locals_.append({"system": system, "event_id": events.event_id.iloc[indices[j]], "fold": fold,
                                            "label": y[j], "score": score[local], "base_score": base,
                                            "feature": model.feature_names[feature], "contribution": contribution[local, feature]})
                        locals_.append({"system": system, "event_id": events.event_id.iloc[indices[j]], "fold": fold,
                                        "label": y[j], "score": score[local], "base_score": base,
                                        "feature": "Other features (sum)",
                                        "contribution": contribution[local].sum() - contribution[local, top].sum()})
                    # Event-level, held-out block permutations preserve each waveform block internally.
                    blocks = {"All waveform changes": waveform_columns}
                    for section in ("HBL", "MBL", "Spk"):
                        blocks[section + " cavity changes"] = [c for c in waveform_columns if c.startswith(section)]
                    for name in columns:
                        if name not in waveform_columns:
                            blocks[name] = [name]
                    baseline = metrics(y[test], score)["roc_auc"]
                    rng = np.random.default_rng(SEED + fold)
                    test_input = inputs.iloc[indices[test]][columns].copy()
                    for block_name, selected in blocks.items():
                        changes = []
                        for _ in range(3):
                            altered = test_input.copy()
                            altered[selected] = test_input.iloc[rng.permutation(len(test))][selected].to_numpy()
                            changes.append(baseline - metrics(y[test], model.predict(altered))["roc_auc"])
                        permutations.append({"system": system, "fold": fold, "block": block_name,
                                             "mean_auc_drop": np.mean(changes), "repeat_std": np.std(changes)})
            rows.append({"system": system, "scope": scope, "positive_captures": positives, "negative_captures": negatives,
                         "folds": 3, "mean_fold_auc": np.mean([m["roc_auc"] for m in fold_scores]),
                         "mean_fold_AP": np.mean([m["average_precision"] for m in fold_scores]), **metrics(y, out)})
            for j, original in enumerate(indices):
                predictions.append({"system": system, "scope": scope, "event_id": events.event_id.iloc[original],
                                    "label": y[j], "score": out[j]})
            if scope == "combined":
                fitted[system] = ExplainedForest().fit(inputs.iloc[indices][columns], y)
                unknown = np.flatnonzero(eligible & events.cause_status.eq("unlisted"))
                for j, score in zip(unknown, fitted[system].predict(inputs.iloc[unknown][columns])):
                    unknown_scores.append({"event_id": events.event_id.iloc[j], "candidate_system": system,
                                           "model_score": score, "status": "unvalidated candidate, not a confirmed cause"})
        # MEBT has enough positives and negatives inside one acquisition context.
        if system == "MEBT buncher":
            match = events.iloc[indices].event_year.eq(2025).to_numpy() & events.iloc[indices].beam_mode.eq("Probe").to_numpy() & events.iloc[indices].beam_present.to_numpy()
            sub_indices, sub_y = indices[match], y[match]
            if min(sub_y.sum(), (1-sub_y).sum()) >= 3:
                for scope in SCOPES:
                    columns = scope_columns(inputs, waveform_columns, scope, cavity=False)
                    score = np.full(len(sub_indices), np.nan)
                    fold_scores = []
                    for train, test in StratifiedKFold(3, shuffle=True, random_state=SEED).split(sub_indices, sub_y):
                        model = ExplainedForest().fit(inputs.iloc[sub_indices[train]][columns], sub_y[train])
                        score[test] = model.predict(inputs.iloc[sub_indices[test]][columns])
                        fold_scores.append(metrics(sub_y[test], score[test]))
                    conditional.append({"system": system, "condition": "2025 / Probe / beam=True", "scope": scope,
                                        "positive_captures": int(sub_y.sum()), "negative_captures": int((1-sub_y).sum()),
                                        "mean_fold_auc": np.mean([m["roc_auc"] for m in fold_scores]),
                                        "mean_fold_AP": np.mean([m["average_precision"] for m in fold_scores]),
                                        **metrics(sub_y, score)})
        print(f"Recorded-cause held-out validation: {system} ({positives} positives) complete", flush=True)
    global_frame = pd.concat(globals_, ignore_index=True).groupby(["system", "feature"]).agg(
        mean_abs_probability_contribution=("mean_abs_contribution", "mean")).reset_index()
    return {"cause_support": pd.DataFrame(support), "cause_metrics": pd.DataFrame(rows), "cause_predictions": pd.DataFrame(predictions),
            "cause_global_explanations": global_frame, "cause_local_contributions": pd.DataFrame(locals_),
            "cause_permutation": pd.DataFrame(permutations), "conditional_cause_metrics": pd.DataFrame(conditional),
            "unknown_cause_scores": pd.DataFrame(unknown_scores)}, fitted, pd.DataFrame(audits)


def make_result(root, output_dir):
    from threadpoolctl import threadpool_limits
    from sklearn.tree import export_text
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    implementation_stamp = Path(__file__).stat().st_mtime_ns
    metadata, waves, prior = load_data(root)
    inputs, waveform_columns = pulse_inputs(metadata, waves)
    weak_tables = ["weak_fold_metrics", "weak_summary", "weak_predictions", "weak_event_predictions", "split_audit",
                   "weak_global_explanations", "weak_local_cases", "weak_local_contributions", "year_metrics"]
    weak_code = "".join(inspect.getsource(f) for f in [load_data, pulse_inputs, scope_columns, capture_weights,
                       metrics, event_scores, ExplainedForest, weak_validation, year_validation])
    weak_fingerprint = {"code": hashlib.sha256(weak_code.encode()).hexdigest(),
                        "inputs": {str(p): p.stat().st_mtime_ns for p in all_cavity.input_paths(root)}}
    stage_path = output_dir / "weak_stage.json"
    staged = stage_path.is_file() and json.loads(stage_path.read_text()) == weak_fingerprint and all(
        (output_dir / f"{name}.csv").is_file() for name in weak_tables)
    with threadpool_limits(limits=2):
        if staged:
            result = {name: pd.read_csv(output_dir / f"{name}.csv", float_precision="round_trip") for name in weak_tables}
            result["split_audit"] = result["split_audit"].query("task == 'weak DD/PM'").copy()
            print("Reused unchanged completed weak-label validation.", flush=True)
        else:
            result = weak_validation(metadata, inputs, waveform_columns)
            result["year_metrics"] = year_validation(metadata, inputs, waveform_columns)
            for name in weak_tables:
                result[name].to_csv(output_dir / f"{name}.csv", index=False)
            stage_path.write_text(json.dumps(weak_fingerprint, indent=2))
        events, event_frame, event_waveform_columns = event_inputs(prior)
        result.update(mixed_clustering(prior, events, event_frame, event_waveform_columns))
        cause_result, cause_models, cause_audit = cause_validation(prior, events, event_frame, event_waveform_columns)
        result.update(cause_result)
        result["split_audit"] = pd.concat([result["split_audit"], cause_audit], ignore_index=True)
        columns = scope_columns(inputs, waveform_columns, "combined")
        full_models = {name: ExplainedForest(name).fit(inputs[columns], metadata.label, capture_weights(metadata)) for name in MODEL_NAMES}
    result["data_support"] = metadata.groupby(["source", "label", "event_year", "beam_mode", "beam_present"], sort=True).agg(
        waveform_sets=("cavity", "size"), captures=("capture", "nunique"), cavities=("cavity", "nunique")
    ).reset_index()
    rules = export_text(full_models["Small tree"].model, feature_names=list(full_models["Small tree"].feature_names), decimals=5)
    (output_dir / "small_tree_rules.txt").write_text(rules)
    joblib.dump({"weak_models": full_models, "cause_models": cause_models,
                 "weak_input_columns": inputs.columns.tolist(), "event_input_columns": event_frame.columns.tolist(),
                 "description": "Weak DD/PM capture label and recorded-system association models; not validated physical fault diagnosis"},
                output_dir / "models.joblib", compress=3)
    for name in TABLES:
        result[name].to_csv(output_dir / f"{name}.csv", index=False)
    manifest = {"inputs": {str(p): p.stat().st_mtime_ns for p in all_cavity.input_paths(root)},
                "implementation": implementation_stamp,
                "all_cavity_implementation": Path(all_cavity.__file__).stat().st_mtime_ns,
                "waveform_sets": len(metadata), "cavities": metadata.cavity.nunique(),
                "seed": SEED, "weak_folds": 5, "cause_folds": 3}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return result


def load_results(output_dir):
    return {name: pd.read_csv(Path(output_dir) / f"{name}.csv", float_precision="round_trip") for name in TABLES}


def cache_fresh(root, output_dir):
    output_dir = Path(output_dir)
    if not all((output_dir / f"{name}.csv").is_file() for name in TABLES) or not (output_dir / "manifest.json").is_file():
        return False
    manifest = json.loads((output_dir / "manifest.json").read_text())
    return (manifest["implementation"] == Path(__file__).stat().st_mtime_ns and
            manifest["all_cavity_implementation"] == Path(all_cavity.__file__).stat().st_mtime_ns and
            manifest["inputs"] == {str(p): p.stat().st_mtime_ns for p in all_cavity.input_paths(root)})


if __name__ == "__main__":
    import sys
    sys.modules["cavity_kickoff_models"] = sys.modules[__name__]
    ExplainedForest.__module__ = "cavity_kickoff_models"
    root = Path(__file__).resolve().parent
    result = make_result(root, root / "signal_comparison" / "kickoff_models")
    print(result["weak_summary"].to_string(index=False))
    print(result["year_metrics"].to_string(index=False))
    print(result["cause_metrics"].to_string(index=False))
    print(result["conditional_cause_metrics"].to_string(index=False))
