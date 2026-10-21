"""Exclude complete candidate cycles from a waveform DataFrame pickle.

Keep all original columns, dtypes, index values, row order and waveform samples.
Write to a new file and read it back to verify metadata and every waveform byte.
"""
import argparse
import gc
import hashlib
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from merge_dd_cycles import update_waveform_hash


class ProgressReader(io.BufferedReader):
    """Report bytes read while retaining the standard binary file interface."""
    def __init__(self, path, label):
        super().__init__(Path(path).open("rb", buffering=0))
        self.total = Path(path).stat().st_size
        self.label = label
        self.last_report = time.monotonic()

    def report(self):
        if time.monotonic() - self.last_report >= 10:
            print(f"{self.label}: {self.tell() / self.total:.1%} read", flush=True)
            self.last_report = time.monotonic()

    def read(self, size=-1):
        result = super().read(size)
        self.report()
        return result

    def readinto(self, target):
        count = super().readinto(target)
        self.report()
        return count

    def readline(self, size=-1):
        result = super().readline(size)
        self.report()
        return result


def fingerprint(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, size_bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def hash_waveforms(series, label):
    digest = hashlib.sha256()
    chunk = 20000
    for start in range(0, len(series), chunk):
        update_waveform_hash(digest, series.iloc[start:start + chunk])
        print(f"{label}: {min(start + chunk, len(series)):,}/{len(series):,} waveforms", flush=True)
    return digest.hexdigest()


def filter_cycles(input_path, cycle_csv, output, classification=None):
    input_path, cycle_csv, output = (Path(p).resolve() for p in (input_path, cycle_csv, output))
    partial = output.with_name(output.name + ".partial")
    report_path = output.with_suffix(".summary.json")
    removed_path = output.with_suffix(".excluded_cycles.csv")
    if input_path == output:
        raise ValueError("Output must differ from the original dataset")
    for p in (output, partial, report_path, removed_path):
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite {p}")
    excluded = pd.read_csv(cycle_csv, dtype={"cycle_id": "int64"})
    if excluded.empty or excluded.cycle_id.duplicated().any():
        raise ValueError("Excluded cycle list must be nonempty and unique")
    if "global_pca_knn_candidates" in excluded and not excluded.global_pca_knn_candidates.gt(0).all():
        raise ValueError("Cycle CSV contains cycles without global PCA+kNN candidates")
    ids = set(map(int, excluded.cycle_id))
    predictions = None
    if classification:
        classification = Path(classification).resolve()
        predictions = pd.read_csv(classification, usecols=["original_row", "pv", "cycle_id", "global_pca_knn_label"])
        detected = set(map(int, predictions.loc[predictions.global_pca_knn_label.eq(-1), "cycle_id"]))
        if detected != ids:
            raise ValueError("Excluded cycles do not match the global classifier's candidate cycles")
    source_before = fingerprint(input_path)
    started = time.monotonic()
    print(f"Reading source: {input_path} ({source_before['size_bytes']:,} bytes)", flush=True)
    with ProgressReader(input_path, "Source pickle") as stream:
        frame = pd.read_pickle(stream)
    if not isinstance(frame, pd.DataFrame) or not {"cycle_id", "pv", "data_array"}.issubset(frame.columns):
        raise ValueError("Expected a DataFrame with cycle_id, pv and data_array")
    if frame.cycle_id.isna().any():
        raise ValueError("Null source cycle_id")
    normalized = frame.pv.str.replace(r"-(?:PMData|DODData)$", "", regex=True, flags=re.IGNORECASE)
    if predictions is not None:
        if len(predictions) != len(frame):
            raise ValueError("Classifier and source row counts differ")
        if not np.array_equal(predictions.original_row, np.arange(len(frame))):
            raise ValueError("Classifier row order does not match source positions")
        if not np.array_equal(predictions.cycle_id, frame.cycle_id):
            raise ValueError("Classifier cycle IDs do not match the source")
        if not np.array_equal(normalized, predictions.pv):
            raise ValueError("Classifier PVs do not match the source")
    missing = ids - set(map(int, frame.cycle_id.unique()))
    if missing:
        raise ValueError(f"Requested cycles absent from source: {sorted(missing)}")
    remove = frame.cycle_id.isin(ids)
    cycle_rows = frame.loc[remove].groupby("cycle_id", sort=True).agg(
        removed_rows=("cycle_id", "size"), removed_source_pvs=("pv", "nunique"))
    removed = excluded.merge(cycle_rows, left_on="cycle_id", right_index=True, validate="one_to_one").sort_values("cycle_id")
    result = frame.loc[~remove].copy()
    pd.testing.assert_series_equal(result.dtypes, frame.dtypes)
    assert result.index.equals(frame.index[~remove])
    assert not result.cycle_id.isin(ids).any()
    counts = dict(source_rows=len(frame), source_cycles=int(frame.cycle_id.nunique()),
                  source_pvs=int(frame.pv.nunique()), removed_cycles=len(ids), removed_rows=int(remove.sum()),
                  retained_rows=len(result), retained_cycles=int(result.cycle_id.nunique()),
                  retained_pvs=int(result.pv.nunique()),
                  source_normalized_pvs=int(normalized.nunique()),
                  retained_normalized_pvs=int(normalized.loc[~remove].nunique()), columns=len(frame.columns))
    assert counts["source_rows"] == counts["removed_rows"] + counts["retained_rows"]
    assert counts["source_cycles"] == counts["removed_cycles"] + counts["retained_cycles"]
    print("FILTERED " + json.dumps(counts), flush=True)
    expected_metadata = result.drop(columns="data_array")
    expected_attrs = dict(result.attrs)
    source_columns = list(frame.columns)
    source_dtypes = {col: str(dtype) for col, dtype in frame.dtypes.items()}
    del frame, remove, predictions
    gc.collect()
    expected_digest = hash_waveforms(result.data_array, "Hashing retained source data")
    try:
        with partial.open("xb") as stream:
            print(f"Writing filtered pickle: {partial}", flush=True)
            result.to_pickle(stream, protocol=5)
            stream.flush()
            os.fsync(stream.fileno())
        del result
        gc.collect()
        print("Reading back the output for full verification", flush=True)
        with ProgressReader(partial, "Output pickle") as stream:
            restored = pd.read_pickle(stream)
        pd.testing.assert_frame_equal(restored.drop(columns="data_array"), expected_metadata, check_exact=True)
        assert list(restored.columns) == source_columns
        assert {col: str(dtype) for col, dtype in restored.dtypes.items()} == source_dtypes
        assert restored.attrs == expected_attrs
        assert not restored.cycle_id.isin(ids).any()
        assert len(restored) == counts["retained_rows"]
        actual_digest = hash_waveforms(restored.data_array, "Verifying saved waveform bytes")
        if expected_digest != actual_digest:
            raise ValueError("Saved waveforms differ from the retained source waveforms")
        if fingerprint(input_path) != source_before:
            raise RuntimeError("Source file changed during filtering")
        report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), input=str(input_path), output=str(output),
            excluded_cycle_source=str(cycle_csv), classification=str(classification) if classification else None,
            detector="global_pca_knn", threshold="median + 5 * std(unique training scores, ddof=0)",
            removal_scope="Every PV capture belonging to an excluded cycle_id",
            **counts, excluded_cycle_ids=sorted(ids),
            original_file_unchanged=True, original_file_fingerprint=source_before,
            index="Original source DataFrame index values and row order preserved; index not reset",
            source_columns=source_columns, dtypes=source_dtypes,
            waveform_sha256=actual_digest, metadata_exact_readback_verified=True,
            waveform_dtypes_shapes_and_all_bytes_readback_verified=True,
            excluded_cycles_absent_verified=True, output_bytes=partial.stat().st_size,
            elapsed_seconds=time.monotonic() - started)
        # Exclusive hard link atomically publishes the verified file without overwriting.
        os.link(partial, output)
        with removed_path.open("x") as stream:
            removed.to_csv(stream, index=False)
        with report_path.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    finally:
        if partial.exists():
            partial.unlink()
    print("SAVED " + json.dumps(dict(output=str(output), summary=str(report_path),
          excluded_cycles=str(removed_path), **counts)), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dd_all_cycles.pkl", dest="input_path")
    parser.add_argument("--cycles", default="waveform_anomaly_comparison/global_pca_knn_sigma5_cycles.csv", dest="cycle_csv")
    parser.add_argument("--output", default="dd_all_cycles_without_global_pca_knn_5sigma_cycles.pkl")
    parser.add_argument("--classification", default=None)
    filter_cycles(**vars(parser.parse_args()))
