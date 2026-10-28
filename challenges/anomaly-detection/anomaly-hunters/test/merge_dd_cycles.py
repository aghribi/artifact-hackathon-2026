"""Combine cycle pickles into one pandas DataFrame, preserving all captures.

Usage:
    python merge_dd_cycles.py
    df = pd.read_pickle("dd_all_cycles.pkl")

The full waveform data must fit in RAM. Existing output files are never
overwritten, and the output is published only after a read-back check.
"""

import argparse
import gc
import hashlib
import os
from pathlib import Path

import pandas as pd

from dd_cycles import CycleDataFrames


def update_waveform_hash(digest, arrays):
    """Hash array boundaries, dtypes, shapes and every original data byte."""
    for array in arrays:
        if array is None:
            digest.update(b"None\n")
        else:
            digest.update(f"{array.dtype.str}:{array.shape}\n".encode())
            digest.update(memoryview(array).cast("B") if array.flags.c_contiguous
                          else array.tobytes(order="C"))


def merge_cycle_pickles(input_dir="dd_by_cycle", output="dd_all_cycles.pkl"):
    """Concatenate cycles by ascending cycle_id and verify the saved result."""
    output = Path(output).resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Output or temporary file already exists: {output}")

    dataset = CycleDataFrames(input_dir)
    expected_files = set(dataset.summary["file"])
    actual_files = {p.name for p in dataset.directory.glob("*.pkl")}
    if actual_files != expected_files:
        raise ValueError("Cycle pickle files do not match cycle_summary.csv")
    if len(dataset) != dataset.manifest["cycles"] or not len(dataset):
        raise ValueError("Cycle count does not match manifest.json")
    expected_rows = dataset.summary.set_index("cycle_id")["rows"].to_dict()
    frames, digest = [], hashlib.sha256()
    print(f"Reading {len(dataset):,} cycle files", flush=True)
    for index, cycle_id in enumerate(sorted(dataset), 1):
        frame = dataset[cycle_id]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"cycle_id={cycle_id}: expected a DataFrame")
        if list(frame.columns) != dataset.manifest["columns"]:
            raise ValueError(f"cycle_id={cycle_id}: column mismatch")
        if len(frame) != expected_rows[cycle_id] or not frame.cycle_id.eq(cycle_id).all():
            raise ValueError(f"cycle_id={cycle_id}: row count or cycle_id mismatch")
        if frames and not frame.dtypes.equals(frames[0].dtypes):
            raise ValueError(f"cycle_id={cycle_id}: dtype mismatch")
        update_waveform_hash(digest, frame["data_array"])
        frames.append(frame)
        if index % 50 == 0 or index == len(dataset):
            print(f"Read {index}/{len(dataset)} cycles", flush=True)

    merged = pd.concat(frames, ignore_index=True)
    if len(merged) != dataset.manifest["rows"]:
        raise ValueError("Total row count does not match manifest.json")
    pd.testing.assert_series_equal(merged.dtypes, frames[0].dtypes)
    expected_metadata = merged.drop(columns="data_array")
    expected_digest = digest.digest()
    del frames, frame
    gc.collect()

    # Exclusive creation avoids overwriting another run or existing data.
    with partial.open("xb") as stream:
        try:
            print(f"Writing {len(merged):,} rows / {len(merged.columns)} columns", flush=True)
            merged.to_pickle(stream, protocol=5)
            stream.flush()
            os.fsync(stream.fileno())
            del merged
            gc.collect()

            print("Reading back and checking metadata and all waveform bytes", flush=True)
            restored = pd.read_pickle(partial)
            pd.testing.assert_frame_equal(
                restored.drop(columns="data_array"), expected_metadata,
                check_exact=True,
            )
            restored_digest = hashlib.sha256()
            update_waveform_hash(restored_digest, restored["data_array"])
            if restored_digest.digest() != expected_digest:
                raise ValueError("Saved waveform data does not match input data")
            os.link(partial, output)
        finally:
            partial.unlink()

    print(f"Saved and verified: {output}\n"
          f"{len(restored):,} rows / {len(dataset):,} cycles / "
          f"{len(restored.columns)} columns / {output.stat().st_size:,} bytes\n"
          f"Waveform SHA-256: {restored_digest.hexdigest()}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dd_by_cycle", dest="input_dir")
    parser.add_argument("--output", default="dd_all_cycles.pkl")
    args = parser.parse_args()
    merge_cycle_pickles(args.input_dir, args.output)
