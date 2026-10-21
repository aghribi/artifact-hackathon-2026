"""Beam position waveforms of the trip captures: does the beam move in the pulses before it is lost?

Input is the raw directory holding SDS_Data_*_PM_*.tar and mps_transitions.parquet written by ess_dd_features.py.
31 PM captures, all from 2025, hold BPM.parquet: 81 beam position monitors, one beam pulse per second, four
waveforms per monitor and pulse (XP and YP position, SA phase, SM signal strength; the last two are named from
their ranges). The beam pulse covers a few tens of samples of each waveform.

Each monitor and pulse becomes one row. The samples of the beam pulse are those where the signal strength, taken as
its median over the first 5 pulses of the capture, is above half of its peak. Over these samples:

    strength, x, y, phase                               mean of each waveform
    strength_spread, x_spread, y_spread, phase_spread   its spread within the pulse (relative for the strength)

Pulses are counted back from the last one that still has beam at the first MEBT monitor: 0 is that pulse, 1 the
one a second earlier. A monitor has beam when its strength is above half of its level in the first 5 pulses.

Each feature of each monitor is compared with its own pulses 5 or more seconds before the last one, in units of
their spread. These earlier pulses, each compared with the others, say what is normal: a feature is "out" beyond
the distance 99 % of them stay within, and a pulse is flagged when more of its features are out than in 99 % of
them.

    <out>/bpm_pulses.parquet        one row per capture, pulse and monitor: the features
    <out>/bpm_before_trip.parquet   one row per capture, pulse and group of features: share of features out, flagged

    python ess_pm_bpm_pretrip.py <raw> <mps_transitions.parquet> <out>
"""

import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ess_archiver_pretrip import PERMIT, trip_kinds

WAVEFORMS = {"SM": "strength", "XP": "x", "YP": "y", "SA": "phase"}
GROUPS = {"position": ["x", "y"], "phase": ["phase"], "signal strength": ["strength"],
          "spread within the pulse": ["strength_spread", "x_spread", "y_spread", "phase_spread"]}
FEATURES = [f for features in GROUPS.values() for f in features]
FIRST = 5            # pulses at the start of a capture that place the beam pulse in the waveform
EARLIER = 5          # pulses this many seconds or more before the last one are the reference
MIN_EARLIER = 6      # reference pulses a capture needs
SHOWN = [4, 3, 2, 1, 0, -1]
NORMAL_SHARE = 0.99


def wrapped(degrees):
    """A phase difference brought back to -180..180."""
    return (degrees + 180) % 360 - 180


def capture_rows(path: Path) -> list[dict]:
    """One row per pulse and monitor of the BPM file of a capture, an empty list when it has none."""
    with tarfile.open(path) as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith("/BPM.parquet")), None)
        if member is None:
            return []
        d = pq.ParquetFile(tar.extractfile(member)).read(
            columns=["pv", "cycle_id", "cycle_ts", "event_id", "event_ts", "beam_mode", "beam_current", "data_array"]).to_pandas()
    d["monitor"] = d.pv.str.rsplit(":", n=1).str[0]
    d["waveform"] = d.pv.str.split(":").str[-1].str[:2].map(WAVEFORMS)
    rows = []
    for monitor, g in d.dropna(subset="waveform").groupby("monitor"):
        cycles = np.sort(g.cycle_id.unique())
        wave = {name: {r.cycle_id: np.asarray(r.data_array, dtype=float) for r in w.itertuples()} for name, w in g.groupby("waveform")}
        if set(wave) != set(WAVEFORMS.values()) or any(set(w) != set(cycles) for w in wave.values()):
            continue
        typical = np.median([wave["strength"][c] for c in cycles[:FIRST]], axis=0)
        pulse = typical > 0.5 * typical.max()
        meta = g.drop_duplicates("cycle_id").set_index("cycle_id")
        for cycle in cycles:
            strength = wave["strength"][cycle][pulse]
            phase = np.deg2rad(wave["phase"][cycle][pulse])
            centre = np.angle(np.exp(1j * phase).mean())
            m = meta.loc[cycle]
            rows.append({
                "event_id": int(m.event_id), "cycle_id": int(cycle), "monitor": monitor, "section": monitor.split("-")[0].upper(),
                "cycle_ts": m.cycle_ts, "event_ts": m.event_ts, "beam_mode": m.beam_mode, "beam_current": int(m.beam_current),
                "pulse_samples": int(pulse.sum()),
                "strength": strength.mean(), "strength_spread": strength.std() / max(strength.mean(), 1e-12),
                "x": wave["x"][cycle][pulse].mean(), "x_spread": wave["x"][cycle][pulse].std(),
                "y": wave["y"][cycle][pulse].mean(), "y_spread": wave["y"][cycle][pulse].std(),
                "phase": np.rad2deg(centre), "phase_spread": np.rad2deg(np.abs(np.angle(np.exp(1j * (phase - centre))))).mean(),
            })
    return rows


def count_back(pulses: pd.DataFrame) -> pd.DataFrame:
    """Add has_beam per monitor and pulses_before_last, counted from the last pulse with beam at the first MEBT monitor."""
    pulses = pulses.sort_values(["event_id", "monitor", "cycle_id"], ignore_index=True)
    group = pulses.groupby(["event_id", "monitor"])
    level = group.strength.transform(lambda s: s.iloc[:FIRST].median())
    pulses["has_beam"] = pulses.strength > 0.5 * level
    first_mebt = sorted(m for m in pulses.monitor.unique() if m.startswith("MEBT"))[0]
    order = pulses.groupby("event_id").cycle_id.rank(method="dense").astype(int)
    at_source = pulses[(pulses.monitor == first_mebt) & pulses.has_beam]
    last = order[at_source.index].groupby(at_source.event_id).max()
    pulses["pulses_before_last"] = pulses.event_id.map(last) - order
    return pulses.dropna(subset="pulses_before_last").astype({"pulses_before_last": int})


def distances(pulses: pd.DataFrame) -> pd.DataFrame:
    """Distance of each feature to the earlier pulses of the same monitor and capture, in units of their spread.

    An earlier pulse is compared with the other earlier pulses, so that it is treated like a later one.
    """
    out = np.full((len(pulses), len(FEATURES)), np.nan)
    for _, rows in pulses.groupby(["event_id", "monitor"]).indices.items():
        g = pulses.iloc[rows]
        earlier = (g.pulses_before_last >= EARLIER).to_numpy()
        if earlier.sum() < MIN_EARLIER:
            continue
        for f, feature in enumerate(FEATURES):
            v = g[feature].to_numpy()
            for i in range(len(v)):
                ref = v[earlier & (np.arange(len(v)) != i)]
                diff = wrapped(v[i] - ref) if feature == "phase" else v[i] - ref
                centre = np.median(diff)
                spread = 1.4826 * np.median(np.abs(diff - centre))
                out[rows[i], f] = abs(centre) / spread if spread > 0 else np.nan
    return pd.DataFrame(out, columns=FEATURES, index=pulses.index)


def main(raw: Path, mps_path: Path, out: Path, workers: int = 32) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with Pool(workers) as pool:
        rows = [r for part in pool.imap_unordered(capture_rows, sorted(raw.glob("SDS_Data_*_PM_*.tar"))) for r in part]
    pulses = count_back(pd.DataFrame(rows))
    print(f"{pulses.event_id.nunique()} captures with beam position waveforms, {pulses.monitor.nunique()} monitors, "
          f"{len(pulses):,} rows", flush=True)

    # The trip of each capture: the logged trip nearest to the capture, and the interlocks that drop with it.
    mps = pd.read_parquet(mps_path)
    logged = mps[(mps.pv == PERMIT) & (mps.value == 0)].sort_values("data_ts").data_ts.reset_index(drop=True)
    events = pulses.drop_duplicates("event_id")[["event_id", "event_ts"]].sort_values("event_ts")
    events = pd.merge_asof(events, pd.DataFrame({"trip_ts": logged, "kind": trip_kinds(mps, logged)}), left_on="event_ts",
                           right_on="trip_ts", direction="nearest", tolerance=pd.Timedelta("5s"))
    pulses["kind"] = pulses.event_id.map(events.set_index("event_id").kind).fillna("no trip logged")

    z = distances(pulses)
    normal = (pulses.pulses_before_last >= EARLIER).to_numpy()
    limits = z[normal].quantile(NORMAL_SHARE)
    is_out = (z > limits).where(z.notna())
    print("distance 99 % of the earlier pulses stay within:", limits.round(1).to_dict(), flush=True)
    pulses.join(z.add_suffix("_distance")).to_parquet(out / "bpm_pulses.parquet", index=False)

    # Share of features out per capture and pulse, for all features and for each group, and the pulses flagged.
    tables = []
    for group, features in {"all": FEATURES, **GROUPS}.items():
        frame = pulses[["event_id", "kind", "pulses_before_last"]].assign(out=is_out[features].sum(axis=1), n=is_out[features].notna().sum(axis=1))
        share = frame.groupby(["event_id", "kind", "pulses_before_last"])[["out", "n"]].sum().reset_index()
        share = share[share.n > 0].assign(group=group, share_out=lambda d: d.out / d.n)
        share["flagged"] = share.share_out > share.share_out[share.pulses_before_last >= EARLIER].quantile(NORMAL_SHARE)
        tables.append(share)
    before = pd.concat(tables, ignore_index=True)
    before.to_parquet(out / "bpm_before_trip.parquet", index=False)

    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 50)
    late = before[before.pulses_before_last.isin(SHOWN)]
    cell = late.groupby(["group", "kind", "pulses_before_last"]).flagged.agg(lambda f: f"{f.sum()}/{len(f)}").unstack()[SHOWN]
    print("\ncaptures flagged / captures, by pulses before the last one with beam (-1 is the pulse after it)")
    print(cell.to_string())
    earlier = before[before.pulses_before_last >= EARLIER].groupby("group").flagged.agg(lambda f: f"{f.sum()}/{len(f)}")
    print("\nearlier pulses flagged:", earlier.to_dict())
    print("\nshare of features out, all features, per capture")
    print(late[late.group == "all"].pivot(index=["kind", "event_id"], columns="pulses_before_last", values="share_out")[SHOWN].round(2).to_string())
    by_section = pulses[["event_id", "section", "pulses_before_last"]].assign(out=is_out.sum(axis=1), n=is_out.notna().sum(axis=1))
    by_section = by_section[by_section.pulses_before_last.isin(SHOWN)].groupby(["section", "pulses_before_last"])[["out", "n"]].sum()
    print("\nshare of features out by section, all captures")
    print((by_section.out / by_section.n).unstack()[SHOWN].round(2).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
