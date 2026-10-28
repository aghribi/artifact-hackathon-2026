"""Check saved splits, calibration thresholds, inference replay and curve coverage."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from predict_mbl_autoencoder import MBLDetector


def validate(directory):
    root = Path(directory).resolve()
    config = json.loads((root/"configuration.json").read_text())
    progress = json.loads((root/"progress.json").read_text())
    assert progress["phase"] == "complete"
    meta = pd.read_csv(root/"sample_splits.csv")
    assert len(meta) == sum(config["counts"].values()) == 13530
    assert meta.pv.nunique() == 36
    assert not meta.cycle_id.isin(config["official_holdout_cycle_ids"]).any()
    assert meta.groupby("cycle_id").split.nunique().max() == 1
    assert meta.groupby("source_archive").split.nunique().max() == 1
    frame = pd.read_pickle(config["input"])
    actual = frame.loc[meta.original_row]
    assert np.array_equal(actual.cycle_id, meta.cycle_id)
    names = actual.pv.str.replace(r"-(?:PMData|DODData)$", "", case=False, regex=True)
    assert np.array_equal(names, meta.pv)
    methods = pd.read_csv(root/"comparison.csv")
    assert len(methods) == progress["completed_methods"] == 18
    test_positions = []
    for length in [5032, 35221]:
        test_positions.extend(np.flatnonzero(meta.split.eq("test") & meta.length.eq(length))[:3].tolist())
    replay = []
    histories = []
    calibration = np.flatnonzero(meta.split.eq("calibration"))
    for result in methods.itertuples():
        scores = np.load(root/f"{result.method}_scores.npy")
        assert len(scores) == len(meta) and np.isfinite(scores).all()
        expected_cutoff = np.median(scores[calibration])+5*np.std(scores[calibration], ddof=0)
        assert abs(expected_cutoff-result.threshold) < 1e-10
        detector = MBLDetector(root, result.method)
        maximum_error = 0.
        for length in [5032, 35221]:
            idx = [i for i in test_positions if meta.length.iloc[i] == length]
            selected = actual.iloc[idx]
            inferred, reconstructed = detector.predict(selected.data_array.tolist(), selected.pv.tolist(), selected.beam_mode.tolist())
            assert reconstructed.shape == (len(idx), length)
            assert np.isfinite(reconstructed).all()
            error = float(np.max(np.abs(inferred-scores[idx])))
            maximum_error = max(maximum_error, error)
            np.testing.assert_allclose(inferred, scores[idx], rtol=4e-3, atol=2e-5)
            if result.method == "conv256_context":
                aliases = selected.pv.str.replace(r"-(?:PMData|DODData)$", "-DoDData", case=False, regex=True)
                alias_scores, _ = detector.predict(selected.data_array.tolist(), aliases.tolist(), selected.beam_mode.tolist())
                np.testing.assert_array_equal(alias_scores, inferred)
        replay.append(dict(method=result.method, compared_waveforms=len(test_positions), max_score_difference=maximum_error))
        history_path = root/f"{result.method}_history.csv"
        if result.epochs:
            history = pd.read_csv(history_path)
            assert len(history) == result.epochs
            np.testing.assert_array_equal(history.epoch, np.arange(1, len(history)+1))
            assert np.isfinite(history[["train_mse", "validation_mse", "validation_waveform_mse"]]).all().all()
            best = int(history.loc[history.validation_mse.idxmin(), "epoch"])
            assert best == result.best_epoch
            assert (root/"learning_curves"/f"{result.method}.png").stat().st_size > 1000
            assert (root/f"{result.method}.pt").is_file()
            histories.append(dict(method=result.method, epochs=len(history), best_epoch=best,
                                  stopped_at_budget=bool(result.epoch_limit_reached)))
        assert (root/f"{result.method}_reconstructions.png").stat().st_size > 1000
        print(f"Verified {result.method}: native inference difference <= {maximum_error:.3g}", flush=True)
        del detector
    native = pd.read_csv(root/"heldout_native_comparison.csv")
    assert len(native) == 18 and native.events.eq(42).all()
    official = pd.read_csv(root/"kickoff_score_comparison.csv")
    assert len(official) == 8 and official.holdout_events.eq(42).all()
    assert official.kickoff_published_mse.eq(.0194).all()
    verified_score = json.loads((root/"official_score_validation.json").read_text())
    assert verified_score["absolute_difference"] < 1e-6
    assert verified_score["kickoff_model_retrained"] is False
    for file in ["learning_curves/all_models.png", "learning_curves/all_models.pdf", "learning_curves/learning_curves.pdf",
                 "report.html", "mse_comparison.png", "heldout_native_comparison.png"]:
        assert (root/file).stat().st_size > 1000
    report = dict(complete=True, waveforms=len(meta), pvs=36, official_holdout_cycle_overlap=0,
        cycle_and_archive_split_overlap=0, comparison_methods=len(methods), trained_ae_models=len(histories),
        native_evaluation_events=42, official_compatible_methods=8, official_score_adapter_absolute_difference=verified_score["absolute_difference"],
        inference_replay=replay, histories=histories, pmdata_doddata_alias_invariance=True,
        calibration_thresholds_verified=True, learning_curves_and_reports_present=True,
        kickoff_model_retrained=False, wandb_upload_performed=False,
        limits=["Normal candidates were selected upstream by full-data PCA+kNN", "No confirmed abnormal MBL labels were supplied",
                "Epoch-budget limits do not prove convergence", "Metadata-conditioned benchmark variants extend the official model inputs"])
    (root/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print("All saved model, split, threshold, curve and evaluation checks passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="mbl_autoencoder_comparison")
    validate(parser.parse_args().directory)
