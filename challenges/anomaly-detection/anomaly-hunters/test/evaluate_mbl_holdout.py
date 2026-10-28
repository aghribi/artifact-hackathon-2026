"""Evaluate saved MBL models on held-out raw captures and compare official MSE bars.

Native-input models receive full raw waveforms, so their raw-input evaluation is
reported separately from the unchanged official 256-point-input benchmark.
The kickoff model is not retrained: only its published 0.0194 bar is used.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from predict_mbl_autoencoder import MBLDetector, OfficialWaveformAdapter
from waveform_anomaly_comparison import RawCache


def evaluate(directory):
    root = Path(directory).resolve()
    workspace = Path(__file__).resolve().parent
    if json.loads((root/"progress.json").read_text())["phase"] != "complete":
        raise RuntimeError("Finish model training before the final evaluation")
    config = json.loads((root/"configuration.json").read_text())
    sample_root = workspace/"official_anomaly_reference/data/sample"
    sample = pd.read_parquet(sample_root/"ess_llrf_sample.parquet")
    split = json.loads((sample_root/"holdout_ids.json").read_text())
    held = sample.iloc[split["holdout_idx"]].copy().reset_index(drop=True)
    held.insert(0, "sample_row", split["holdout_idx"])
    held["pv"] = held.pv_base.str.replace("CmpSDS", "Cmp1SDS", regex=False).str.replace(r"-(?:PMData|DODData)$", "", case=False, regex=True)
    inventory = pd.read_csv(workspace/"waveform_autoencoder_comparison/sample_splits.csv",
                            usecols=["original_row", "cycle_id", "pv", "source_pv", "length"])
    # Unrelated PVs can have multiple PM/DoD captures; validate only held-out pairs.
    keys = pd.MultiIndex.from_frame(held[["cycle_id", "pv"]])
    inventory = inventory.loc[pd.MultiIndex.from_frame(inventory[["cycle_id", "pv"]]).isin(keys)]
    held = held.merge(inventory, on=["cycle_id", "pv"], validate="one_to_one").sort_values("sample_row").reset_index(drop=True)
    assert len(held) == 42
    prepared_splits = pd.read_csv(root/"sample_splits.csv")
    assert not prepared_splits.cycle_id.isin(held.cycle_id).any()
    cache = RawCache(workspace/"anomaly-pca-reuse-20261015")
    waveforms = [cache.read([int(row.original_row)])[0] for row in held.itertuples()]
    np.savez(root/"official_holdout_native_waveforms.npz", **{f"waveform_{i}": value for i, value in enumerate(waveforms)})
    held.drop(columns=["I", "Q"]).to_csv(root/"official_holdout_native_metadata.csv", index=False)
    methods = pd.read_csv(root/"comparison.csv").method.tolist()
    rows, details = [], []
    for method in methods:
        detector = MBLDetector(root, method)
        scores = np.zeros(len(held))
        for length in sorted(held.length.unique()):
            idx = np.flatnonzero(held.length.eq(length))
            scores[idx], _ = detector.predict([waveforms[i] for i in idx],
                held.iloc[idx].source_pv.tolist(), held.iloc[idx].beam_mode.tolist())
        candidate = scores > detector.threshold
        compatible = method.startswith("conv256") or method.startswith("pca256_")
        rows.append(dict(method=method, events=len(held), input_grid="256" if compatible else "native (5032/35221)",
            mean_native_normalized_mse=float(scores.mean()), candidate_waveforms=int(candidate.sum()),
            threshold=detector.threshold, official_256_input_compatible=compatible))
        detail = held[["sample_row", "original_row", "cycle_id", "pv", "beam_mode", "length"]].copy()
        detail.insert(0, "method", method)
        detail["native_normalized_mse"] = scores
        detail["threshold"] = detector.threshold
        detail["anomaly_candidate"] = candidate
        details.append(detail)
        print(f"Held-out native evaluation {method}: MSE={scores.mean():.6g}; candidates={candidate.sum()}/42", flush=True)
        del detector
    pd.DataFrame(rows).to_csv(root/"heldout_native_comparison.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(root/"heldout_native_waveform_scores.csv", index=False)

    official = pd.read_csv(root/"official_holdout_comparison.csv")
    official["kickoff_published_mse"] = .0194
    official["mse_reduction_percent"] = (1-official.official_holdout_mse/.0194)*100
    for name in ["MBL-010RFC", "MBL-090RFC"]:
        official[name+"_mse"] = [
            pd.read_csv(root/f"{method}_official_holdout_scores.csv").loc[
                lambda frame: frame.pv_base.str.startswith(name), "mse"].mean()
            for method in official.method
        ]
    official.to_csv(root/"kickoff_score_comparison.csv", index=False)

    # Cross-check the waveform-only model using the actual, unchanged score.py.
    spec = importlib.util.spec_from_file_location("official_reference_score", workspace/"official_anomaly_reference/score.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    q = np.stack(sample.Q).astype(np.float32)
    normalized = (q-config["official_mean"])/config["official_std"]
    value, errors = reference.score_holdout(OfficialWaveformAdapter(root), normalized, np.array(split["holdout_idx"]))
    expected = float(official.loc[official.method.eq("conv256"), "official_holdout_mse"].iloc[0])
    assert abs(value-expected) < 1e-6
    validation = dict(official_score_py_mean_mse=value, recorded_mean_mse=expected,
        absolute_difference=abs(value-expected), events=len(errors), waveform_only_adapter=True,
        metadata_passed_to_adapter=False, kickoff_model_retrained=False, training_executed_by_evaluation=False,
        native_holdout_models_evaluated=len(rows), evaluation_cycle_overlap=0,
        note="Native-input results use additional raw samples and are not the official 256-input MSE benchmark")
    (root/"official_score_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print("Verified official score.py compatibility and saved the published-kickoff comparison", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="mbl_autoencoder_comparison")
    evaluate(parser.parse_args().directory)
