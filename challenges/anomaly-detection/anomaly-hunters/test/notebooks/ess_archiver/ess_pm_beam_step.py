"""The step the beam leaves in the cavity field: does it change in the pulses before a trip?

The beam passing a cavity shifts its field amplitude and phase by a small step, set by the beam charge and by the
phase at which the beam arrives. The front end of the linac has no waveform in the captures, so this step is the
nearest reading of what it delivers.

Input is the raw directory holding SDS_Data_*_PM_*.tar and SDS_Data_*_DD_*.tar, and beam_shape_pulses.parquet
written by ess_pm_beam_shape.py, which gives the pulses before the beam stops. The beam pulse is placed in time with
the first beam current monitor of the capture, as in ess_pm_beam_shape.py. Each cavity and cycle becomes one row:

    amp_step        field amplitude after the beam pulse minus before it, over the amplitude before
    phase_step      field phase after the beam pulse minus before it, degrees
    *_control       the same two, measured 180 microseconds later in the pulse, where no beam passes

Before and after are 34 microseconds each, 6 microseconds away from the beam pulse. Only records of 35,221 samples
are used: the sampling step of the shorter ones is not documented.

Each step of each cavity is compared with its own pulses 5 or more before the beam stops, in units of their
spread. These earlier pulses, each compared with the others, say what is normal: a step is "out" beyond the
distance 99 % of them stay within. A capture counts as in ess_pm_beam_shape.py.

    <out>/beam_step_pulses.parquet   one row per capture, cycle and cavity
    <out>/beam_step_before.parquet   one row per pair of steps and pulses before the stop: PM against DD captures

    python ess_pm_beam_step.py <raw> <beam_shape_pulses.parquet> <out>
"""

import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ess_dd_features import BCM_DT_US, CAVITY_PV, RF_DT_US, RF_SAMPLES
from ess_pm_beam_shape import EARLIER, FIRST, MIN_EARLIER, NORMAL_SHARE, SHOWN, above

GUARD = 50           # cavity samples left out on each side of the beam pulse
SIDE = 280           # cavity samples averaged before and after
CONTROL = 1500       # cavity samples between the beam pulse and the control measurement
PAIRS = {"beam": ["amp_step", "phase_step"], "control": ["amp_control", "phase_control"]}
COLUMNS = ["pv", "cycle_id", "cycle_ts", "data_ts", "event_id", "data_array"]


def steps(amp: np.ndarray, phase: np.ndarray, first: int, last: int) -> tuple[float, float]:
    """Relative amplitude step and phase step in degrees across the samples first..last."""
    before, after = slice(first - GUARD - SIDE, first - GUARD), slice(last + GUARD, last + GUARD + SIDE)
    turn = np.exp(1j * np.deg2rad(phase))
    return (amp[after].mean() / amp[before].mean() - 1,
            float(np.rad2deg(np.angle(turn[after].mean() / turn[before].mean()))))


def capture_rows(path: Path) -> list[dict]:
    """One row per cycle and cavity of a capture, an empty list when it has no beam current or cavity file."""
    rows = []
    with tarfile.open(path) as tar:
        members = {m.name.split("/")[-1]: m for m in tar.getmembers() if m.name.endswith(".parquet")}
        if "BCM.parquet" not in members:
            return rows
        capture_type = members["BCM.parquet"].name.split("/")[2]
        bcm = pq.ParquetFile(tar.extractfile(members["BCM.parquet"])).read(columns=COLUMNS).to_pandas()
        # The beam pulse in time after the start of the cycle, from the first monitor that reads it.
        window = None
        for _, g in bcm.drop_duplicates(["pv", "cycle_id"]).groupby("pv"):
            g = g[g.data_array.str.len() == g.data_array.str.len().max()].sort_values("cycle_ts")
            if len(g) < FIRST:
                continue
            typical = np.median(np.stack(g.data_array.to_numpy()[:FIRST]).astype(float), axis=0)
            typical -= np.median(typical)
            if -typical.min() > typical.max():
                typical = -typical
            if typical.max() < 8 * 1.4826 * np.median(np.abs(typical)):
                continue
            inside = np.flatnonzero(typical > 0.5 * typical.max())
            start = (g.data_ts.iloc[0] - g.cycle_ts.iloc[0]).total_seconds() * 1e6
            window = (start + inside[0] * BCM_DT_US, start + inside[-1] * BCM_DT_US)
            break
        if window is None:
            return rows
        for family in ("SPOKES.parquet", "MBL.parquet", "HBL.parquet"):
            if family not in members:
                continue
            d = pq.ParquetFile(tar.extractfile(members[family])).read(columns=COLUMNS).to_pandas()
            d = d.join(d.pv.str.extract(CAVITY_PV)).dropna(subset="cavity")
            d = d[(d.dwn == "0") & (d.data_array.str.len() == RF_SAMPLES)].drop_duplicates(["pv", "cycle_id"])
            for (cavity, cycle), g in d.groupby(["cavity", "cycle_id"]):
                if set(g.cmp) != {"0", "1"}:
                    continue
                amp, phase = (np.asarray(g.data_array[g.cmp == c].iloc[0], dtype=float) for c in ("0", "1"))
                start = (g.data_ts.iloc[0] - g.cycle_ts.iloc[0]).total_seconds() * 1e6
                first, last = (int(round((t - start) / RF_DT_US)) for t in window)
                if first - GUARD - SIDE < 0 or last + CONTROL + GUARD + SIDE > RF_SAMPLES:
                    continue
                amp_step, phase_step = steps(amp, phase, first, last)
                amp_control, phase_control = steps(amp, phase, first + CONTROL, last + CONTROL)
                rows.append({"capture_type": capture_type, "event_id": int(g.event_id.iloc[0]), "cycle_id": int(cycle),
                             "cavity": cavity, "section": g.section.iloc[0], "amp_step": amp_step, "phase_step": phase_step,
                             "amp_control": amp_control, "phase_control": phase_control})
    return rows


def distances(pulses: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Distance of each step to the earlier pulses of the same cavity and capture, in units of their spread.

    An earlier pulse is compared with the other earlier pulses, so that it is treated like a later one.
    """
    out = np.full((len(pulses), len(features)), np.nan)
    for _, rows in pulses.groupby(["event_id", "cavity"]).indices.items():
        g = pulses.iloc[rows]
        earlier = (g.pulses_before_stop >= EARLIER).to_numpy()
        if earlier.sum() < MIN_EARLIER:
            continue
        for f, feature in enumerate(features):
            v = g[feature].to_numpy()
            for i in range(len(v)):
                ref = v[earlier & (np.arange(len(v)) != i)]
                centre = np.median(ref)
                spread = 1.4826 * np.median(np.abs(ref - centre))
                out[rows[i], f] = abs(v[i] - centre) / spread if spread > 0 else np.nan
    return pd.DataFrame(out, columns=features, index=pulses.index)


def main(raw: Path, shape_path: Path, out: Path, workers: int = 32) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tars = sorted(p for p in raw.glob("SDS_Data_*.tar") if "_PM_" in p.name or "_DD_" in p.name)
    with Pool(workers) as pool:
        rows = [r for part in pool.imap_unordered(capture_rows, tars) for r in part]
    cycles = pd.read_parquet(shape_path, columns=["event_id", "cycle_id", "pulses_before_stop", "beam_stops", "beam_current"])
    pulses = pd.DataFrame(rows).merge(cycles.drop_duplicates(["event_id", "cycle_id"]), on=["event_id", "cycle_id"])
    pulses = pulses[(pulses.beam_stops == (pulses.capture_type == "PM")) & (pulses.pulses_before_stop >= 1)].reset_index(drop=True)
    features = [f for pair in PAIRS.values() for f in pair]
    z = distances(pulses, features)
    pulses.join(z.add_suffix("_distance")).to_parquet(out / "beam_step_pulses.parquet", index=False)
    print(f"{len(pulses):,} rows, {pulses.cavity.nunique()} cavities, captures:",
          pulses.groupby("capture_type").event_id.nunique().to_dict(), flush=True)
    print("median step:", pulses[features].abs().median().to_dict(), flush=True)

    normal = (pulses.pulses_before_stop >= EARLIER).to_numpy()
    is_out = (z > z[normal].quantile(NORMAL_SHARE)).where(z.notna())
    table, shares = [], {}
    for pair, columns in PAIRS.items():
        frame = pulses[["capture_type", "event_id", "pulses_before_stop"]].assign(
            out=is_out[columns].sum(axis=1), n=is_out[columns].notna().sum(axis=1), furthest=z[columns].max(axis=1))
        share = frame.groupby(["capture_type", "event_id", "pulses_before_stop"]).agg(
            out=("out", "sum"), n=("n", "sum"), furthest=("furthest", "max")).reset_index()
        share = share[share.n > 0].assign(share_out=lambda d: d.out / d.n)
        shares[pair] = share
        pm, dd = share[share.capture_type == "PM"], share[share.capture_type == "DD"]
        limit = pm.share_out[pm.pulses_before_stop >= EARLIER].quantile(NORMAL_SHARE)
        far = pm.furthest[pm.pulses_before_stop >= EARLIER].quantile(NORMAL_SHARE)
        for back in SHOWN:
            a, b = pm[pm.pulses_before_stop == back], dd[dd.pulses_before_stop == back]
            table.append({"steps": pair, "pulses_before_stop": back, "pm_captures": len(a), "dd_captures": len(b),
                          "pm_share_out": a.share_out.median(), "dd_share_out": b.share_out.median(),
                          "pm_over_limit": int((a.share_out > limit).sum()), "dd_over_limit": int((b.share_out > limit).sum()),
                          "roc_auc": above(a.share_out.to_numpy(), b.share_out.to_numpy()),
                          "pm_furthest_over_limit": int((a.furthest > far).sum()),
                          "roc_auc_furthest": above(a.furthest.to_numpy(), b.furthest.to_numpy())})
    before = pd.DataFrame(table)
    before.to_parquet(out / "beam_step_before.parquet", index=False)

    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 50)
    print("\nPM against DD captures, by pulses before the stop")
    print(before.round(3).to_string(index=False))
    share = shares["beam"]
    print("\nshare of beam steps out per PM capture, in %")
    print((100 * share[(share.capture_type == "PM") & (share.pulses_before_stop <= 8)].pivot(
        index="event_id", columns="pulses_before_stop", values="share_out")).round(0).iloc[:, ::-1].to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
