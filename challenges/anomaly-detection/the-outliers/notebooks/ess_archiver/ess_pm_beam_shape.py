"""Beam current waveforms of the captures: does the shape of the beam pulse change in the pulses before a trip?

Input is the raw directory holding SDS_Data_*_PM_*.tar (taken when the beam trips) and SDS_Data_*_DD_*.tar (taken
on demand while the beam holds). Each holds BCM.parquet: one waveform of 80,000 samples per beam current monitor
and cycle, one cycle per second. The beam pulse covers 25 to 270 samples of it.

The samples of the beam pulse are those where the waveform, taken as its median over the first 5 cycles of the
capture, is above half of its peak. Each monitor and cycle becomes one row:

    level            mean of the waveform over the beam pulse, above its baseline
    shape            the beam pulse divided by its level, averaged in at most 32 blocks
    shape_distance   root mean square distance of the shape to the median shape of the earlier pulses
    shape_ratio      shape_distance over its median for the earlier pulses, each compared with the others

A monitor reads beam in a cycle when its level is above half of its level in the first 5 cycles. The beam stops
at the first cycle where fewer than half of the monitors read beam. Pulses are counted back from there: 1 is the
last pulse with beam. A PM capture counts when the beam stops inside it. A DD capture counts when it does not,
and its pulses are counted back from its end. The earlier pulses are those 5 or more before the stop.

    <out>/beam_shape_pulses.parquet   one row per capture, cycle and monitor
    <out>/beam_shape_before.parquet   one row per pulses before the stop: PM captures against DD captures

    python ess_pm_beam_shape.py <raw> <out>
"""

import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FIRST = 5            # cycles at the start of a capture that place the beam pulse in the waveform
EARLIER = 5          # pulses this many or more before the stop are the reference
MIN_EARLIER = 6      # reference pulses a capture needs
BLOCKS = 32
SHOWN = [4, 3, 2, 1]
NORMAL_SHARE = 0.99


def capture_rows(path: Path) -> list[dict]:
    """One row per cycle and monitor of the BCM file of a capture, an empty list when it has none."""
    with tarfile.open(path) as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith("/BCM.parquet")), None)
        if member is None:
            return []
        d = pq.ParquetFile(tar.extractfile(member)).read(
            columns=["pv", "cycle_id", "cycle_ts", "event_id", "beam_current", "data_array"]).to_pandas()
    capture_type = member.name.split("/")[2]
    d["monitor"] = d.pv.str.rsplit(":", n=1).str[0]
    rows = []
    # The same waveform can sit twice in a file.
    for monitor, g in d.drop_duplicates(["pv", "cycle_id"]).groupby("monitor"):
        g = g[g.data_array.str.len() == g.data_array.str.len().max()].sort_values("cycle_ts")
        waves = np.stack(g.data_array.to_numpy()).astype(float)
        if len(waves) < FIRST + MIN_EARLIER:
            continue
        waves -= np.median(waves, axis=1, keepdims=True)
        typical = np.median(waves[:FIRST], axis=0)
        # Some monitors read the beam as a negative pulse.
        if -typical.min() > typical.max():
            waves, typical = -waves, -typical
        # No beam at the start of the capture: the peak stays within the noise, taken as 1.4826 times the median
        # absolute deviation.
        if typical.max() < 8 * 1.4826 * np.median(np.abs(typical)):
            continue
        pulse = np.flatnonzero(typical > 0.5 * typical.max())
        if len(pulse) < 8:
            continue
        inside = waves[:, pulse[0]:pulse[-1] + 1]
        level = inside.mean(axis=1)
        blocks = np.array_split(np.arange(inside.shape[1]), min(BLOCKS, inside.shape[1]))
        shape = np.stack([inside[:, b].mean(axis=1) for b in blocks], axis=1) / np.where(level > 0, level, np.nan)[:, None]
        for i, r in enumerate(g.itertuples()):
            rows.append({"capture_type": capture_type, "event_id": int(r.event_id), "cycle_id": int(r.cycle_id),
                         "cycle_ts": r.cycle_ts, "monitor": monitor, "beam_current": int(r.beam_current),
                         "pulse_samples": inside.shape[1], "level": level[i],
                         "has_beam": bool(level[i] > 0.5 * np.median(level[:FIRST])), "shape": shape[i]})
    return rows


def count_back(pulses: pd.DataFrame) -> pd.DataFrame:
    """Add pulses_before_stop and beam_stops, true when the beam stops inside the capture. Captures with too few
    pulses before the stop are dropped."""
    pulses = pulses.sort_values(["event_id", "monitor", "cycle_ts"], ignore_index=True)
    cycle = pulses.groupby(["event_id", "cycle_ts"]).has_beam.mean().ge(0.5).reset_index(name="beam")
    cycle["order"] = cycle.groupby("event_id").cycle_ts.rank(method="dense").astype(int)
    without = cycle[~cycle.beam].groupby("event_id").order.min()
    stop = cycle.groupby("event_id").order.max().add(1).to_frame("stop").join(without.rename("without"))
    stop = stop.without.fillna(stop.stop)
    cycle["pulses_before_stop"] = (cycle.event_id.map(stop) - cycle.order).astype(int)
    cycle["beam_stops"] = cycle.event_id.isin(without.index)
    pulses = pulses.merge(cycle[["event_id", "cycle_ts", "pulses_before_stop", "beam_stops"]], on=["event_id", "cycle_ts"])
    return pulses[pulses.event_id.map(stop) > 1 + EARLIER + MIN_EARLIER - 1].reset_index(drop=True)


def shape_distances(pulses: pd.DataFrame) -> pd.DataFrame:
    """shape_distance and shape_ratio of each pulse with beam. An earlier pulse is compared with the other earlier
    pulses, so that it is treated like a later one."""
    distance = np.full(len(pulses), np.nan)
    ratio = np.full(len(pulses), np.nan)
    for _, rows in pulses.groupby(["event_id", "monitor"]).indices.items():
        g = pulses.iloc[rows]
        shapes = np.stack(g["shape"].to_numpy())
        usable = (g.has_beam & (g.pulses_before_stop >= 1)).to_numpy()
        earlier = usable & (g.pulses_before_stop >= EARLIER).to_numpy()
        if earlier.sum() < MIN_EARLIER:
            continue
        for i in np.flatnonzero(usable):
            reference = np.median(shapes[earlier & (np.arange(len(g)) != i)], axis=0)
            distance[rows[i]] = np.sqrt(np.mean((shapes[i] - reference) ** 2))
        normal = np.median(distance[rows[earlier]])
        if normal > 0:
            ratio[rows] = distance[rows] / normal
    return pulses.assign(shape_distance=distance, shape_ratio=ratio)


def above(a: np.ndarray, b: np.ndarray) -> float:
    """Chance that a value of a is above a value of b: the ROC-AUC of telling a from b."""
    return float(((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


def main(raw: Path, out: Path, workers: int = 32) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tars = sorted(p for p in raw.glob("SDS_Data_*.tar") if "_PM_" in p.name or "_DD_" in p.name)
    with Pool(workers) as pool:
        rows = [r for part in pool.imap_unordered(capture_rows, tars) for r in part]
    pulses = shape_distances(count_back(pd.DataFrame(rows)))
    pulses.to_parquet(out / "beam_shape_pulses.parquet", index=False)
    # A PM capture counts when the beam is seen stopping in it: some are taken with no beam in the machine.
    pulses = pulses[pulses.beam_stops == (pulses.capture_type == "PM")].dropna(subset="shape_ratio")
    print(f"{len(pulses):,} pulses with beam, captures:", pulses.groupby("capture_type").event_id.nunique().to_dict(), flush=True)

    # One value per capture and pulse: the median over its monitors.
    capture = pulses.groupby(["capture_type", "event_id", "pulses_before_stop"]).agg(
        shape_ratio=("shape_ratio", "median"), monitors=("monitor", "size"), beam_current=("beam_current", "first")).reset_index()
    pm, dd = capture[capture.capture_type == "PM"], capture[capture.capture_type == "DD"]
    limit = pm.shape_ratio[pm.pulses_before_stop >= EARLIER].quantile(NORMAL_SHARE)
    table = []
    for back in SHOWN:
        a, b = pm.shape_ratio[pm.pulses_before_stop == back].to_numpy(), dd.shape_ratio[dd.pulses_before_stop == back].to_numpy()
        table.append({"pulses_before_stop": back, "pm_captures": len(a), "dd_captures": len(b), "pm_median_ratio": np.median(a),
                      "dd_median_ratio": np.median(b), "pm_over_limit": int((a > limit).sum()),
                      "dd_over_limit": int((b > limit).sum()), "limit": limit, "roc_auc": above(a, b)})
    before = pd.DataFrame(table)
    before.to_parquet(out / "beam_shape_before.parquet", index=False)

    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 50)
    print("\nPM against DD captures, by pulses before the stop")
    print(before.round(2).to_string(index=False))
    print("\nshape_ratio per PM capture")
    print(pm[pm.pulses_before_stop.isin(SHOWN)].pivot(index="event_id", columns="pulses_before_stop",
                                                     values="shape_ratio")[SHOWN].round(2).to_string())
    section = pulses[(pulses.capture_type == "PM") & pulses.pulses_before_stop.isin(SHOWN)]
    print("\nmedian shape_ratio of the PM captures by monitor")
    print(section.groupby(["monitor", "pulses_before_stop"]).shape_ratio.median().unstack()[SHOWN].round(2).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
