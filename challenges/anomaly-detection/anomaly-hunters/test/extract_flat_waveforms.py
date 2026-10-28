"""Extract flat waveform intervals from one PV before an exclusive sample limit.

Usage:
    python extract_flat_waveforms.py
    segments = pd.read_pickle("flat_waveform_segments/flat_segments.pkl")

Flat means the whole interval's maximum minus minimum is at most 0.02.
Each interval has at least 101 samples. Sample indices are zero-based; the
default search covers [0, 23480). Intervals are disjoint and scanned left to
right: extend a valid interval until the next sample exceeds the tolerance,
then emit it if sufficiently long and start a new interval. Short intervals
are shifted forward until their range fits. Nonfinite samples break intervals.
No waveform resampling or smoothing is applied during extraction.
"""

import argparse
from collections import deque
import gc
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "anomalyhunter-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PV = "HBL-010RFC:RFS-DIG-101:Dwn0-Cmp0SDS-PMData"


def find_flat_intervals(array, stop_sample=23480, minimum_samples=101, range_tolerance=0.02):
    """Return nonoverlapping (start, exclusive end) intervals in original indices."""
    if stop_sample < 1 or minimum_samples < 1 or range_tolerance < 0:
        raise ValueError("Sample limits must be positive and tolerance nonnegative")
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional waveform")
    values = values[:stop_sample]
    minima, maxima, intervals = deque(), deque(), []
    left = 0
    for right, value in enumerate(values):
        if not np.isfinite(value):
            if right - left >= minimum_samples:
                intervals.append((left, right))
            left = right + 1
            minima.clear()
            maxima.clear()
            continue
        while minima and values[minima[-1]] >= value:
            minima.pop()
        while maxima and values[maxima[-1]] <= value:
            maxima.pop()
        minima.append(right)
        maxima.append(right)
        if values[maxima[0]] - values[minima[0]] > range_tolerance:
            if right - left >= minimum_samples:
                intervals.append((left, right))
                left = right
                minima, maxima = deque([right]), deque([right])
            else:
                while values[maxima[0]] - values[minima[0]] > range_tolerance:
                    left += 1
                    while minima[0] < left:
                        minima.popleft()
                    while maxima[0] < left:
                        maxima.popleft()
    if len(values) - left >= minimum_samples:
        intervals.append((left, len(values)))
    return intervals


def extract_flat_waveforms(input_path="dd_all_cycles.pkl", pv=DEFAULT_PV,
                           output_dir="flat_waveform_segments", stop_sample=23480,
                           minimum_samples=101, range_tolerance=0.02):
    input_path, output_dir = Path(input_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    print(f"Reading {input_path}", flush=True)
    all_data = pd.read_pickle(input_path)
    selected = all_data.loc[all_data.pv.eq(pv)].copy()
    if selected.empty:
        raise ValueError(f"PV not found: {pv}")
    if "original_row" not in selected:
        selected["original_row"] = selected.index
    del all_data
    gc.collect()
    selected = selected.sort_values(["cycle_id", "data_ts", "original_row"])
    records, summaries, intervals_by_row = [], [], {}
    provenance = ["cycle_id", "cycle_ts", "pv", "data_ts", "original_row", "beam_mode",
                  "beam_len", "source_archive", "source_parquet", "source_row"]
    for processed, (_, row) in enumerate(selected.iterrows(), 1):
        array = row.data_array
        intervals = find_flat_intervals(array, stop_sample, minimum_samples, range_tolerance)
        intervals_by_row[int(row.original_row)] = intervals
        metadata = {column: row[column] for column in provenance if column in selected}
        for start, end in intervals:
            data = array[start:end].copy()
            minimum, maximum = float(data.min()), float(data.max())
            records.append(dict(
                **metadata, flat_start=start, flat_end=end - 1, flat_end_exclusive=end,
                flat_n_samples=end - start, flat_min=minimum, flat_max=maximum,
                flat_ptp=maximum - minimum, flat_mean=float(data.mean(dtype=np.float64)),
                flat_std=float(data.std(dtype=np.float64)),
                flat_range_tolerance=range_tolerance, flat_data_array=data,
            ))
        longest = max(intervals, key=lambda segment: segment[1] - segment[0], default=None)
        summaries.append(dict(
            **metadata, flat_intervals=len(intervals),
            flat_total_samples=sum(end - start for start, end in intervals),
            longest_flat_start=longest[0] if longest else None,
            longest_flat_end=longest[1] - 1 if longest else None,
            longest_flat_samples=longest[1] - longest[0] if longest else 0,
        ))
        if processed % 50 == 0 or processed == len(selected):
            print(f"Analyzed {processed}/{len(selected)} waveforms", flush=True)
    segment_columns = [*provenance, "flat_start", "flat_end", "flat_end_exclusive",
                       "flat_n_samples", "flat_min", "flat_max", "flat_ptp", "flat_mean",
                       "flat_std", "flat_range_tolerance", "flat_data_array"]
    segments = pd.DataFrame(records, columns=segment_columns)
    summary = pd.DataFrame(summaries)

    # Check every interval and its exact original data before saving.
    originals = selected.set_index("original_row")
    for row in segments.itertuples(index=False):
        original = originals.loc[row.original_row].data_array
        assert 0 <= row.flat_start < row.flat_end_exclusive <= min(stop_sample, len(original))
        assert row.flat_n_samples >= minimum_samples
        assert row.flat_ptp <= range_tolerance
        np.testing.assert_array_equal(row.flat_data_array, original[row.flat_start:row.flat_end_exclusive])
    for intervals in intervals_by_row.values():
        assert all(a[1] <= b[0] for a, b in zip(intervals[:-1], intervals[1:]))

    output_dir.mkdir(parents=True)
    segments.to_pickle(output_dir / "flat_segments.pkl", protocol=5)
    segments.drop(columns="flat_data_array").to_csv(output_dir / "flat_segments.csv", index=False)
    summary.to_csv(output_dir / "cycle_summary.csv", index=False)
    restored = pd.read_pickle(output_dir / "flat_segments.pkl")
    pd.testing.assert_frame_equal(restored, segments, check_exact=True)
    parameters = dict(
        input_path=str(input_path.resolve()), pv=pv, stop_sample_exclusive=stop_sample,
        indexing="zero-based; flat_end inclusive; flat_end_exclusive for slicing",
        minimum_samples=minimum_samples, range_tolerance=range_tolerance,
        definition="max(flat_data_array) - min(flat_data_array) <= range_tolerance",
        interval_policy="left-to-right greedy disjoint intervals; no smoothing or resampling",
        waveforms=len(selected), matched_waveforms=int((summary.flat_intervals > 0).sum()),
        intervals=len(segments), flat_samples=int(summary.flat_total_samples.sum()),
    )
    (output_dir / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")

    # Include the unusual waveform with the lowest mean and representative captures.
    means = np.asarray([a[:stop_sample].mean(dtype=np.float64) for a in selected.data_array])
    indices = list(dict.fromkeys([0, int(np.argmin(means)), len(selected) // 2, len(selected) - 1]))
    fig, axes = plt.subplots(len(indices), 1, figsize=(13, 3.3 * len(indices)), layout="constrained")
    for ax, index in zip(np.atleast_1d(axes), indices):
        row = selected.iloc[index]
        array = row.data_array
        ax.plot(np.arange(array.size), array, color="#6b7f99", lw=0.75, label="Original waveform")
        intervals = intervals_by_row[int(row.original_row)]
        for j, (start, end) in enumerate(intervals):
            ax.plot(np.arange(start, end), array[start:end], color="#16985e", lw=1.8,
                    label="Extracted flat intervals" if j == 0 else None)
        if intervals:
            start, end = max(intervals, key=lambda interval: interval[1] - interval[0])
            mean = np.mean(array[start:end])
            near_top = mean > array.min() + 0.7 * np.ptp(array)
            ax.annotate(f"{start:,} - {end-1:,} ({end-start:,} samples)",
                        ((start + end) / 2, mean),
                        xytext=(0, -18 if near_top else 12), textcoords="offset points",
                        ha="center", va="top" if near_top else "bottom", fontsize=9)
        ax.axvline(stop_sample, color="#df574f", ls="--", lw=1.2, label=f"Search limit: {stop_sample:,}")
        ax.axvspan(stop_sample, array.size, color="#e6eaf0", alpha=0.55)
        ax.set_xlim(0, array.size)
        ax.set_title(f"cycle_id={row.cycle_id} | {row.beam_mode} | beam_len={row.beam_len} | {len(intervals)} intervals", fontsize=10)
        ax.set_xlabel("Sample index (zero-based)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(pv + f"\nFlat: interval range <= {range_tolerance}; length >= {minimum_samples} samples", fontsize=12)
    fig.savefig(output_dir / "flat_examples.png", dpi=160)
    plt.close(fig)
    print(json.dumps(parameters, indent=2), flush=True)
    if not segments.empty:
        print(segments[["cycle_id", "flat_start", "flat_end", "flat_n_samples", "flat_ptp", "flat_mean"]].head(10).to_string(index=False), flush=True)
    print(f"Saved and verified: {output_dir.resolve()}", flush=True)
    return segments, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dd_all_cycles.pkl", dest="input_path")
    parser.add_argument("--pv", default=DEFAULT_PV)
    parser.add_argument("--output", default="flat_waveform_segments", dest="output_dir")
    parser.add_argument("--stop-sample", type=int, default=23480)
    parser.add_argument("--minimum-samples", type=int, default=101)
    parser.add_argument("--range-tolerance", type=float, default=0.02)
    extract_flat_waveforms(**vars(parser.parse_args()))
