"""Select MBL Dwn0-Cmp1 normal candidates, excluding every official holdout cycle.

This prepares data only: no model, PCA, normalization or threshold is fitted.
The original DataFrame columns, dtypes, indices and waveform samples are retained.
"""
import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from filter_dd_cycles import ProgressReader, fingerprint, hash_waveforms


HERE = Path(__file__).resolve().parent
PV_PATTERN = r"^MBL.*:Dwn0-Cmp1SDS$"


def prepare(input_path, output, official_sample, official_split):
    input_path, output, official_sample, official_split = (
        Path(p).resolve() for p in (input_path, output, official_sample, official_split)
    )
    partial = output.with_name(output.name + ".partial")
    summary_path = output.with_suffix(".summary.json")
    cycles_path = output.with_suffix(".excluded_cycles.csv")
    pvs_path = output.with_suffix(".pv_counts.csv")
    if input_path == output:
        raise ValueError("Output must differ from the source")
    for path in (output, partial, summary_path, cycles_path, pvs_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    sample = pd.read_parquet(official_sample, columns=["cycle_id", "pv_base"])
    split = json.loads(official_split.read_text())
    holdout = sample.iloc[split["holdout_idx"]].copy()
    holdout.insert(0, "sample_row", split["holdout_idx"])
    if holdout.empty or holdout.cycle_id.isna().any() or holdout.cycle_id.duplicated().any():
        raise ValueError("Official holdout must contain unique, non-null cycle IDs")
    evaluation_cycles = set(map(int, holdout.cycle_id))

    started = time.monotonic()
    source_fingerprint = fingerprint(input_path)
    print(f"Reading normal candidates: {input_path}", flush=True)
    with ProgressReader(input_path, "Normal candidates") as stream:
        frame = pd.read_pickle(stream)
    if not isinstance(frame, pd.DataFrame) or not {"cycle_id", "pv", "data_array"}.issubset(frame):
        raise ValueError("Expected a waveform DataFrame")
    if frame.cycle_id.isna().any() or not frame.index.is_unique:
        raise ValueError("Source must have non-null cycle IDs and unique original indices")
    normalized_pvs = frame.pv.str.replace(r"-(?:PMData|DODData)$", "", case=False, regex=True)
    selected = normalized_pvs.str.fullmatch(PV_PATTERN, na=False)
    target = frame.loc[selected].copy()
    target_pvs = normalized_pvs.loc[selected]
    del normalized_pvs
    lengths = target.data_array.map(lambda value: 0 if value is None else len(value))
    official = target.cycle_id.isin(evaluation_cycles)
    keep = ~official & lengths.gt(0)
    result = target.loc[keep].copy()
    if result.empty:
        raise ValueError("No trainable waveforms remain")
    if result.cycle_id.isin(evaluation_cycles).any():
        raise AssertionError("Evaluation cycles remain in the prepared dataset")
    for index, waveform in result.data_array.items():
        if not np.isfinite(waveform).all():
            raise ValueError(f"Nonfinite waveform at original index {index}")

    pv_counts = pd.DataFrame({
        "pv": target_pvs,
        "source_rows": 1,
        "evaluation_rows_removed": official.astype(int),
        "empty_rows_removed": (~official & lengths.eq(0)).astype(int),
        "retained_waveforms": keep.astype(int),
    }).groupby("pv", sort=True).sum().reset_index()
    removed_by_cycle = target.loc[official].groupby("cycle_id").size()
    holdout["removed_mbl_waveforms"] = holdout.cycle_id.map(removed_by_cycle).fillna(0).astype(int)
    source_columns = list(frame.columns)
    source_dtypes = {column: str(dtype) for column, dtype in frame.dtypes.items()}
    pd.testing.assert_series_equal(result.dtypes, frame.dtypes)
    counts = dict(
        source_rows=len(frame), selected_rows_before_exclusion=len(target),
        selected_valid_waveforms_before_exclusion=int(lengths.gt(0).sum()),
        official_evaluation_cycles=len(evaluation_cycles),
        evaluation_cycles_present_in_selected_data=int(target.loc[official, "cycle_id"].nunique()),
        evaluation_waveforms_removed=int(official.sum()),
        empty_waveforms_removed=int((~official & lengths.eq(0)).sum()),
        retained_waveforms=len(result), retained_cycles=int(result.cycle_id.nunique()),
        retained_pvs=int(pv_counts.retained_waveforms.gt(0).sum()),
        retained_lengths={str(k): int(v) for k, v in lengths.loc[keep].value_counts().sort_index().items()},
        retained_beam_modes=result.beam_mode.value_counts().to_dict(),
    )
    assert counts["selected_rows_before_exclusion"] == (
        counts["evaluation_waveforms_removed"] + counts["empty_waveforms_removed"] + counts["retained_waveforms"]
    )
    expected_metadata = result.drop(columns="data_array")
    expected_attrs = dict(result.attrs)
    del frame, target, target_pvs, lengths, official, keep, selected
    gc.collect()
    print("SELECTED " + json.dumps(counts), flush=True)
    expected_hash = hash_waveforms(result.data_array, "Hashing selected waveforms")
    try:
        with partial.open("xb") as stream:
            result.to_pickle(stream, protocol=5)
            stream.flush()
            os.fsync(stream.fileno())
        del result
        gc.collect()
        print("Reading back the prepared dataset", flush=True)
        with ProgressReader(partial, "Prepared dataset") as stream:
            restored = pd.read_pickle(stream)
        pd.testing.assert_frame_equal(restored.drop(columns="data_array"), expected_metadata, check_exact=True)
        assert list(restored.columns) == source_columns
        assert {column: str(dtype) for column, dtype in restored.dtypes.items()} == source_dtypes
        assert restored.attrs == expected_attrs
        assert not restored.cycle_id.isin(evaluation_cycles).any()
        assert len(restored) == counts["retained_waveforms"]
        actual_hash = hash_waveforms(restored.data_array, "Verifying all saved waveform bytes")
        if actual_hash != expected_hash:
            raise ValueError("Prepared waveform data changed during serialization")
        if fingerprint(input_path) != source_fingerprint:
            raise RuntimeError("Source changed during dataset preparation")
        report = dict(
            created_at_utc=datetime.now(timezone.utc).isoformat(), input=str(input_path), output=str(output),
            official_sample=str(official_sample), official_split=str(official_split),
            pv_pattern_after_terminal_pmdata_doddata_removal=PV_PATTERN,
            exclusion_scope="Every selected PV waveform with any official holdout cycle_id",
            excluded_cycle_ids=sorted(evaluation_cycles), **counts,
            source_fingerprint=source_fingerprint, source_file_unchanged=True,
            columns=source_columns, dtypes=source_dtypes, waveform_sha256=actual_hash,
            metadata_exact_readback_verified=True, waveform_bytes_exact_readback_verified=True,
            evaluation_cycle_overlap=0, index="Original source index and order preserved",
            normal_label_status="Normal candidates selected upstream by global PCA+kNN, not confirmed normals",
            upstream_selection_uses_full_original_data=True,
            upstream_selection_note="Removing holdout cycles here does not undo their influence on upstream PCA+kNN selection",
            training_executed=False, pca_fitted=False, normalization_fitted=False,
            output_bytes=partial.stat().st_size, elapsed_seconds=time.monotonic()-started,
        )
        os.link(partial, output)
        with cycles_path.open("x") as stream:
            holdout.to_csv(stream, index=False)
        with pvs_path.open("x") as stream:
            pv_counts.to_csv(stream, index=False)
        with summary_path.open("x") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    finally:
        if partial.exists():
            partial.unlink()
    print("SAVED " + json.dumps(dict(output=str(output), summary=str(summary_path), **counts)), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", default=str(HERE/"dd_all_cycles_without_global_pca_knn_5sigma_cycles.pkl"))
    parser.add_argument("--output", default=str(HERE/"mbl_dwn0_cmp1_normal_without_official_holdout.pkl"))
    parser.add_argument("--official-sample", default=str(HERE/"official_anomaly_reference/data/sample/ess_llrf_sample.parquet"))
    parser.add_argument("--official-split", default=str(HERE/"official_anomaly_reference/data/sample/holdout_ids.json"))
    prepare(**vars(parser.parse_args()))
