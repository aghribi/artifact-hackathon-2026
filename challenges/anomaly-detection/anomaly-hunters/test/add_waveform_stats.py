"""Add per-waveform population standard deviation, maximum and mean.

Usage:
    python add_waveform_stats.py
    python add_waveform_stats.py --input dd_all_cycles.pkl --output with_stats.pkl

By default, update dd_all_cycles.pkl after verifying a temporary output.
Empty or null waveforms receive NaN; NaNs within a waveform propagate.
Mean and standard deviation use float64 accumulation, with ddof=0.
"""

import argparse
import gc
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from merge_dd_cycles import update_waveform_hash


STAT_COLUMNS = ["data_array_std", "data_array_max", "data_array_mean"]


def waveform_statistics(array):
    """Return (population standard deviation, maximum, mean) for one capture."""
    if array is None:
        return np.nan, np.nan, np.nan
    values = np.asarray(array)
    if values.size == 0:
        return np.nan, np.nan, np.nan
    return (np.std(values, dtype=np.float64, ddof=0),
            np.max(values), np.mean(values, dtype=np.float64))


def add_waveform_statistics(input_path="dd_all_cycles.pkl", output_path=None):
    """Preserve all rows and original data, then atomically publish the result."""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve() if output_path else input_path
    partial = output_path.with_name(output_path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Temporary output already exists: {partial}")
    if output_path != input_path and output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    source_stat = input_path.stat()

    print(f"Reading {input_path}", flush=True)
    frame = pd.read_pickle(input_path)
    if not isinstance(frame, pd.DataFrame) or "data_array" not in frame.columns:
        raise ValueError("Input must be a DataFrame with a data_array column")

    results = np.empty((len(frame), len(STAT_COLUMNS)), dtype=np.float64)
    source_digest = hashlib.sha256()
    empty_waveforms = 0
    print(f"Computing waveform statistics for {len(frame):,} rows (ddof=0)", flush=True)
    for index, array in enumerate(frame["data_array"], 1):
        results[index - 1] = waveform_statistics(array)
        update_waveform_hash(source_digest, (array,))
        empty_waveforms += array is None or array.size == 0
        if index % 10000 == 0 or index == len(frame):
            print(f"Computed {index:,}/{len(frame):,} rows", flush=True)
    for position, column in enumerate(STAT_COLUMNS):
        frame[column] = results[:, position]
    del results
    expected_metadata = frame.drop(columns="data_array")
    expected_digest = source_digest.digest()

    with partial.open("xb") as stream:
        try:
            print(f"Writing {len(frame):,} rows / {len(frame.columns)} columns", flush=True)
            frame.to_pickle(stream, protocol=5)
            stream.flush()
            os.fsync(stream.fileno())
            del frame
            gc.collect()

            print("Reading back and checking all metadata, statistics and waveform bytes", flush=True)
            restored = pd.read_pickle(partial)
            pd.testing.assert_frame_equal(
                restored.drop(columns="data_array"), expected_metadata, check_exact=True,
            )
            restored_digest = hashlib.sha256()
            update_waveform_hash(restored_digest, restored["data_array"])
            if restored_digest.digest() != expected_digest:
                raise ValueError("Saved waveform data does not match input data")
            current_stat = input_path.stat()
            if (current_stat.st_ino, current_stat.st_size, current_stat.st_mtime_ns) != (
                    source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns):
                raise RuntimeError("Input file changed during processing")
            if output_path == input_path:
                os.replace(partial, output_path)
            else:
                os.link(partial, output_path)
        finally:
            partial.unlink(missing_ok=True)

    print(f"Saved and verified: {output_path}\n"
          f"{len(restored):,} rows / {len(restored.columns)} columns / "
          f"{output_path.stat().st_size:,} bytes\n"
          f"Empty or null waveforms: {empty_waveforms}\n"
          f"Waveform SHA-256: {restored_digest.hexdigest()}", flush=True)
    print(restored[["cycle_id", "pv", *STAT_COLUMNS]].head().to_string(index=False), flush=True)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dd_all_cycles.pkl", dest="input_path")
    parser.add_argument("--output", dest="output_path")
    args = parser.parse_args()
    add_waveform_statistics(args.input_path, args.output_path)
