"""Add sample-window statistics for one PV or a PV prefix to the merged pickle.

Usage:
    python add_waveform_window_stats.py
    python add_waveform_window_stats.py --pv-prefix 'HBL-010RFC:RFS-DIG-101:Dwn0-Cmp0SDS'
    python add_waveform_window_stats.py --pv 'HBL-010RFC:RFS-DIG-101:Dwn0-Cmp0SDS-PMData'

The default window is data_array[5000:15000] (10,000 samples, zero-based).
By default, include every PV starting with HBL-010RFC:RFS-DIG-101:Dwn0-Cmp0SDS.
Standard deviation uses ddof=0; mean and standard deviation accumulate in
float64. Range is max minus min. New columns are NaN for other PVs, and
existing values for other PVs are preserved on subsequent updates.
The original file is replaced only after metadata and waveform verification.
"""

import argparse
import gc
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from merge_dd_cycles import update_waveform_hash


DEFAULT_PV = "HBL-010RFC:RFS-DIG-101:Dwn0-Cmp0SDS-PMData"
DEFAULT_PV_PREFIX = "HBL-010RFC:RFS-DIG-101:Dwn0-Cmp0SDS"


def window_statistics(array, start=5000, stop=15000):
    """Return (population standard deviation, mean, range) for a full window."""
    if start < 0 or stop <= start:
        raise ValueError("Require 0 <= start < stop")
    if array is None:
        return np.nan, np.nan, np.nan
    values = np.asarray(array)
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional waveform")
    if values.size < stop:
        return np.nan, np.nan, np.nan
    window = values[start:stop]
    return (float(np.std(window, dtype=np.float64, ddof=0)),
            float(np.mean(window, dtype=np.float64)),
            float(np.max(window)) - float(np.min(window)))


def add_window_statistics(input_path="dd_all_cycles.pkl", output_path=None,
                          pv=None, start=5000, stop=15000, pv_prefix=None):
    """Update matching rows; default to the whole DEFAULT_PV_PREFIX family."""
    if start < 0 or stop <= start:
        raise ValueError("Require 0 <= start < stop")
    if pv is not None and pv_prefix is not None:
        raise ValueError("Specify either pv or pv_prefix, not both")
    if pv is None and pv_prefix is None:
        pv_prefix = DEFAULT_PV_PREFIX
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve() if output_path else input_path
    partial = output_path.with_name(output_path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Temporary output already exists: {partial}")
    if output_path != input_path and output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    source_stat = input_path.stat()
    columns = [f"data_array_{start}_{stop}_{stat}" for stat in ("std", "mean", "ptp")]

    print(f"Reading {input_path}", flush=True)
    frame = pd.read_pickle(input_path)
    if not isinstance(frame, pd.DataFrame) or not {"pv", "data_array"}.issubset(frame.columns):
        raise ValueError("Input must be a DataFrame with pv and data_array columns")
    mask = frame.pv.str.startswith(pv_prefix, na=False) if pv_prefix is not None else frame.pv.eq(pv)
    if not mask.any():
        raise ValueError(f"No PV matches: {pv_prefix if pv_prefix is not None else pv}")
    original_dtypes = frame.dtypes.copy()
    results = np.asarray([window_statistics(a, start, stop) for a in frame.loc[mask, "data_array"]])
    for position, column in enumerate(columns):
        if column not in frame:
            frame[column] = np.nan
        frame.loc[mask, column] = results[:, position]
    pd.testing.assert_series_equal(frame[original_dtypes.index].dtypes, original_dtypes)
    matching_pvs = frame.loc[mask, "pv"].value_counts()
    print(f"Computed {int(mask.sum()):,} captures across {len(matching_pvs)} PVs\n"
          f"{matching_pvs.to_string()}\n"
          f"Window: [{start}, {stop}) / {stop-start:,} samples / ddof=0\n"
          f"Columns: {', '.join(columns)}", flush=True)
    preview = frame.loc[mask, ["cycle_id", "pv", *columns]].groupby("pv", sort=False).head(3).copy()
    expected_metadata = frame.drop(columns="data_array")
    original_digest = hashlib.sha256()
    print("Hashing all original waveform bytes", flush=True)
    update_waveform_hash(original_digest, frame.data_array)

    with partial.open("xb") as stream:
        try:
            print(f"Writing {len(frame):,} rows / {len(frame.columns)} columns", flush=True)
            frame.to_pickle(stream, protocol=5)
            stream.flush()
            os.fsync(stream.fileno())
            del frame
            gc.collect()

            print("Reading back and verifying all columns and waveform bytes", flush=True)
            restored = pd.read_pickle(partial)
            pd.testing.assert_frame_equal(restored.drop(columns="data_array"),
                                          expected_metadata, check_exact=True)
            restored_digest = hashlib.sha256()
            update_waveform_hash(restored_digest, restored.data_array)
            if restored_digest.digest() != original_digest.digest():
                raise ValueError("Saved waveform data does not match original data")
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
          f"Waveform SHA-256: {restored_digest.hexdigest()}", flush=True)
    print(preview.to_string(index=False), flush=True)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dd_all_cycles.pkl", dest="input_path")
    parser.add_argument("--output", dest="output_path")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--pv", help="Match one exact PV name")
    target.add_argument("--pv-prefix", help="Match all PV names starting with this prefix")
    parser.add_argument("--start", type=int, default=5000)
    parser.add_argument("--stop", type=int, default=15000)
    add_window_statistics(**vars(parser.parse_args()))
