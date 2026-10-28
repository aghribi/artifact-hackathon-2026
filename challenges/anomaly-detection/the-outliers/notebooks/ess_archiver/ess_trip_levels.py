"""Level of the Archiver signals just before a beam trip: does the beam trip when some signal sits unusually high or low?

ess_archiver_pretrip.py compares a signal with its own level shortly before, so it cannot see a signal that builds
up over minutes. Here the level itself is compared, between stretches of beam. A stretch runs from the beam coming
back to the beam stopping, as read by the first MEBT beam current monitor.

The level of a signal at a moment is its median 10 to 3 s before. Each trip is compared with moments of other
stretches that held: same beam current within 10 %, close in time, the same time since the beam came back within a
factor 1.5, and at least 30 s before the end of their stretch. The place of a trip is the share of those moments
with a lower level: 0.5 when the trip looks like them, near 1 or 0 when the signal is higher or lower than in the
stretches that held. A signal is scored over the trips by the mean of its places, as a z value (0 is no difference).

What chance gives comes from mock trips: each trip replaced by one of the moments it is compared with, scored the
same way against the others, 200 times. `chance` is the share of these draws in which some signal reaches the z of
this one, so a small value means no mock draw produced a signal as far off.

The beam current regimes are taken apart: a reading under 10, compared within 6 hours, and over 40, compared within
90 minutes.

    <out>/trip_levels.parquet     one row per regime and signal: trips, mean place, z, chance
    <out>/trip_levels_trips.parquet   one row per regime: trips, trips compared, signals beyond z 3 and the furthest z,
                                      for the trips and for the mock trips

    python ess_trip_levels.py <raw> <mps_transitions.parquet> <out>
"""

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ess_archiver_pretrip import BEAM_GROUP, NS, PERMIT, RUN_GAP, add_transmission, load_series

LEVEL_SPAN = np.arange(-10, -2.9, 0.5)    # seconds before a moment whose median is the level of a signal
QUIET_EVERY = 5                           # seconds between the moments trips are compared with
BEFORE_END = 30                           # seconds a moment must be from the end of its stretch
BEAM_RATIO = 1.1
CLOCK_RATIO = 1.5
REGIMES = {"reading under 10": (0, 10, 360), "reading over 40": (40, np.inf, 90)}    # low, high, minutes apart
MIN_MATCHED = 20
MATCH_CAP = 200
DRAWS = 200

SERIES: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
TIMES = np.empty(0, dtype="int64")


def levels(pv: str) -> np.ndarray:
    """Level of one signal at each moment: median of the value it holds over LEVEL_SPAN."""
    warnings.simplefilter("ignore", RuntimeWarning)
    _, ts, value = SERIES[pv]
    idx = np.searchsorted(ts, TIMES[:, None] + np.round(LEVEL_SPAN * NS).astype("int64")[None, :], side="right") - 1
    return np.nanmedian(np.where(idx >= 0, value[np.clip(idx, 0, None)], np.nan), axis=1).astype("float32")


def place(level: np.ndarray, own: np.ndarray) -> np.ndarray:
    """Share of the compared moments with a lower level than `own`, per signal. Equal levels count for half."""
    with np.errstate(invalid="ignore"):
        lower = (level < own[:, None]).sum(axis=1) + 0.5 * (level == own[:, None]).sum(axis=1)
        return np.where(np.isnan(own), np.nan, lower / (~np.isnan(level)).sum(axis=1))


def z_of(places: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trips, mean place and z per signal from the places of the trips (signal by trip)."""
    warnings.simplefilter("ignore", RuntimeWarning)
    n = (~np.isnan(places)).sum(axis=1)
    mean = np.nanmean(places, axis=1)
    return n, mean, np.nan_to_num((mean - 0.5) * np.sqrt(12 * n))


def main(raw: Path, mps_path: Path, out: Path, workers: int = 32) -> None:
    global SERIES, TIMES
    out.mkdir(parents=True, exist_ok=True)
    SERIES = load_series(raw, workers)
    add_transmission(SERIES)
    pvs = sorted(pv for pv, (group, _, _) in SERIES.items() if group != BEAM_GROUP)
    groups = np.array([SERIES[pv][0] for pv in pvs])

    # The beam second by second: on or off, its stretch, the minutes since it came back and the seconds to its stop.
    _, ts, value = SERIES[sorted(pv for pv, (group, _, _) in SERIES.items() if group == BEAM_GROUP and pv.startswith("MEBT"))[0]]
    run = np.concatenate([[0], np.cumsum(np.diff(ts) > RUN_GAP * NS)])
    top = pd.Series(value).groupby(run).transform(lambda v: v.quantile(0.95)).to_numpy()
    on = value > 0.2 * top
    beam = pd.DataFrame({"ts": ts, "value": value, "run": run, "on": on})
    beam["stretch"] = (beam.on & ~beam.on.shift(fill_value=False) | (beam.run != beam.run.shift())).cumsum()
    beam["clock"] = (beam.ts - beam.ts.where(~beam.on).groupby(beam.run).ffill()) / NS / 60
    beam["to_end"] = (beam.groupby("stretch").ts.transform("max") - beam.ts) / NS

    mps = pd.read_parquet(mps_path)
    permit = mps[mps.pv == PERMIT].sort_values("data_ts")
    permit = permit[permit.value != permit.value.shift()]
    logged = permit.data_ts[permit.value == 0].dt.tz_localize(None).astype("datetime64[ns]").astype("int64").to_numpy()
    last = []
    for t in logged:
        lo, hi = np.searchsorted(ts, [t - 15 * NS, t + 3 * NS])
        off = np.flatnonzero(~on[lo:hi])
        if len(off) and off[0] > 0:
            last.append(lo + off[0] - 1)
    span = -LEVEL_SPAN[0] / 60
    trips = beam.iloc[last][lambda d: d.clock >= span].assign(is_trip=True)
    quiet = beam[beam.on & (beam.clock >= span) & (beam.to_end >= BEFORE_END)].iloc[::QUIET_EVERY].assign(is_trip=False)
    events = pd.concat([trips, quiet], ignore_index=True)
    print(f"{len(logged)} trips logged, {len(last)} with the beam seen stopping, {len(trips)} with a known time since "
          f"the beam came back, {len(quiet):,} moments to compare with", flush=True)

    TIMES = events.ts.to_numpy()
    with Pool(workers) as pool:
        level = np.stack(pool.map(levels, pvs, chunksize=16))            # signal, event

    rng = np.random.default_rng(0)
    e = {c: events[c].to_numpy() for c in ["ts", "value", "stretch", "clock"]}
    is_quiet = ~events.is_trip.to_numpy()
    tables, counts = [], []
    for regime, (low, high, apart) in REGIMES.items():
        places, mock = [], [[] for _ in range(DRAWS)]
        in_regime = np.flatnonzero(events.is_trip & (events.value >= low) & (events.value < high))
        for i in in_regime:
            matched = np.flatnonzero(
                is_quiet & (e["stretch"] != e["stretch"][i]) & (np.abs(e["ts"] - e["ts"][i]) <= apart * 60 * NS)
                & (e["value"] > e["value"][i] / BEAM_RATIO) & (e["value"] < e["value"][i] * BEAM_RATIO)
                & (e["clock"] > e["clock"][i] / CLOCK_RATIO) & (e["clock"] < e["clock"][i] * CLOCK_RATIO))
            if len(matched) < MIN_MATCHED or len(np.unique(e["stretch"][matched])) < 2:
                continue
            matched = rng.choice(matched, min(len(matched), MATCH_CAP), replace=False)
            places.append(place(level[:, matched], level[:, i]))
            # A mock trip is one of the compared moments, placed among those of the other stretches.
            for d in range(DRAWS):
                j = rng.choice(matched)
                others = matched[e["stretch"][matched] != e["stretch"][j]]
                mock[d].append(place(level[:, others], level[:, j]))
        if not places:
            counts.append({"regime": regime, "trips": len(in_regime), "trips_compared": 0})
            continue
        n, mean, z = z_of(np.stack(places, axis=1))
        mock_z = np.stack([z_of(np.stack(m, axis=1))[2] for m in mock])                  # draw, signal
        furthest = np.abs(mock_z).max(axis=1)
        tables.append(pd.DataFrame({"regime": regime, "pv": pvs, "group": groups, "trips": n, "mean_place": mean, "z": z,
                                    "chance": [(furthest >= abs(v)).mean() for v in z]}))
        counts.append({"regime": regime, "trips": len(in_regime), "trips_compared": len(places),
                       "signals_beyond_z3": int((np.abs(z) > 3).sum()),
                       "mock_signals_beyond_z3": float((np.abs(mock_z) > 3).sum(axis=1).mean()),
                       "furthest_z": float(np.abs(z).max()), "mock_furthest_z_median": float(np.median(furthest)),
                       "mock_furthest_z_95": float(np.quantile(furthest, 0.95))})
        print(f"\n{regime}: {len(places)} of {len(in_regime)} trips compared. Signals beyond z 3: {(np.abs(z) > 3).sum()}, "
              f"mock trips give {(np.abs(mock_z) > 3).sum(axis=1).mean():.1f}. Furthest z {np.abs(z).max():.1f}, "
              f"mock trips reach {np.median(furthest):.1f} (median) and {np.quantile(furthest, 0.95):.1f} (95 %)", flush=True)
    table = pd.concat(tables, ignore_index=True)
    table.to_parquet(out / "trip_levels.parquet", index=False)
    pd.DataFrame(counts).to_parquet(out / "trip_levels_trips.parquet", index=False)

    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 70)
    for regime, g in table.groupby("regime", sort=False):
        print(f"\n{regime}: the 20 signals furthest from 0.5")
        print(g.reindex(g.z.abs().sort_values(ascending=False).index).head(20).drop(columns="regime").round(3).to_string(index=False))
        by_group = g.assign(far=g.z.abs() > 3).groupby("group").agg(signals=("pv", "size"), beyond_z3=("far", "sum"), mean_z=("z", "mean"))
        print(by_group.round(2).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
