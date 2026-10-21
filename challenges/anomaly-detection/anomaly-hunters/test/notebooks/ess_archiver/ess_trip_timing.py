"""Beam trips against the time since the beam came back and against the beam current: when is a trip more likely?

Input is the raw directory holding Archiver_Data_*.tar and mps_transitions.parquet written by ess_dd_features.py.
The first MEBT beam current monitor gives one sample per second. The beam is on when it reads more than 20 % of the
usual top current of its run. Every second with beam gets a clock: the time since the beam was last off. A trip is
FBIS::beam_permit_ok going to 0, placed on the last sample with beam before it, and kept when the beam is seen
stopping. The trip rate of a span of the clock is the trips that fall in it over the hours of beam spent in it.
The same rate is taken per level of the beam current, as read by that monitor at each second (the file gives no unit).

    <out>/trip_timing.parquet       one row per span of the clock and kind of trip: hours of beam, trips, trips per hour
    <out>/trip_current.parquet      one row per level of the beam current: hours of beam, trips, trips per hour
    <out>/trip_clock.parquet        one row per trip: time, kind, run, minutes since the beam came back, beam current
    <out>/trip_timing_runs.parquet  one row per run: hours of beam and trips in the first 15 minutes and after

A run with a bad day has more restarts and more trips, which alone would put more trips close to a restart. The
check within runs keeps the number of trips of each run and asks how many would fall in the first 15 minutes if
they came at random moments of that run's beam time (10,000 draws).

    python ess_trip_timing.py <raw> <mps_transitions.parquet> <out>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ess_archiver_pretrip import NS, PERMIT, RUN_GAP, load_series, trip_kinds

SPANS = {"under 1 min": (0, 1), "1 to 2 min": (1, 2), "2 to 5 min": (2, 5), "5 to 15 min": (5, 15),
         "15 to 60 min": (15, 60), "over 60 min": (60, np.inf)}
LEVELS = {"under 10": (0, 10), "10 to 40": (10, 40), "over 40": (40, np.inf)}    # reading of the beam current monitor
EARLY = 15       # minutes after the beam came back that count as early
DRAWS = 10_000


def main(raw: Path, mps_path: Path, out: Path, workers: int = 32) -> None:
    out.mkdir(parents=True, exist_ok=True)
    series = load_series(raw, workers, ["SCBCMS"])
    _, ts, value = series[sorted(pv for pv in series if pv.startswith("MEBT"))[0]]

    # Beam on or off at each sample, and the minutes since it was last off. The clock is unknown from the start of
    # a run to its first sample without beam.
    run = np.concatenate([[0], np.cumsum(np.diff(ts) > RUN_GAP * NS)])
    top = pd.Series(value).groupby(run).transform(lambda v: v.quantile(0.95)).to_numpy()
    on = value > 0.2 * top
    last_off = pd.Series(np.where(~on, ts, np.nan)).groupby(run).ffill().to_numpy()
    clock = (ts - last_off) / NS / 60
    step = np.minimum(np.diff(ts, append=ts[-1]) / NS, 2.0)    # seconds of beam a sample stands for

    mps = pd.read_parquet(mps_path)
    permit = mps[mps.pv == PERMIT].sort_values("data_ts")
    permit = permit[permit.value != permit.value.shift()]
    logged = permit.data_ts[permit.value == 0].reset_index(drop=True)
    kinds = trip_kinds(mps, logged)
    rows = []
    for t, kind in zip(logged.dt.tz_localize(None).astype("datetime64[ns]").astype("int64"), kinds):
        lo, hi = np.searchsorted(ts, [t - 15 * NS, t + 3 * NS])
        off = np.flatnonzero(~on[lo:hi])
        if len(off) and off[0] > 0:
            i = lo + off[0] - 1
            rows.append({"time": pd.Timestamp(ts[i], tz="UTC"), "kind": kind, "run": int(run[i]),
                         "minutes_since_beam_back": clock[i], "beam_current": value[i]})
    trips = pd.DataFrame(rows)
    trips.to_parquet(out / "trip_clock.parquet", index=False)
    print(f"{len(logged)} trips logged, {len(trips)} with the beam seen stopping, "
          f"{trips.minutes_since_beam_back.isna().sum()} of them before the first beam stop of their run", flush=True)

    table = []
    for span, (lo, hi) in SPANS.items():
        hours = step[on & (clock >= lo) & (clock < hi)].sum() / 3600
        inside = trips[(trips.minutes_since_beam_back >= lo) & (trips.minutes_since_beam_back < hi)]
        for kind, n in [("all", len(inside)), *inside.kind.value_counts().items()]:
            table.append({"span": span, "kind": kind, "hours_of_beam": hours, "trips": n, "trips_per_hour": n / hours})
    table = pd.DataFrame(table)
    table.to_parquet(out / "trip_timing.parquet", index=False)
    pd.set_option("display.width", 200)
    print(table[table.kind == "all"].drop(columns="kind").round(2).to_string(index=False))
    print(table.pivot(index="span", columns="kind", values="trips").loc[list(SPANS)].fillna(0).astype(int).to_string())
    print("minutes since the beam came back, per trip:", trips.minutes_since_beam_back.describe().round(1).to_dict())
    print("hours of beam with the clock unknown:", round(step[on & np.isnan(clock)].sum() / 3600, 1))

    current = pd.DataFrame([{"beam_current": level, "hours_of_beam": step[on & (value >= lo) & (value < hi)].sum() / 3600,
                             "trips": int(trips.beam_current.between(lo, hi, inclusive="left").sum())}
                            for level, (lo, hi) in LEVELS.items()])
    current["trips_per_hour"] = current.trips / current.hours_of_beam
    current.to_parquet(out / "trip_current.parquet", index=False)
    print(current.round(2).to_string(index=False), flush=True)

    # Within runs: trips of each run drawn at random moments of its beam time with a known clock.
    known = on & ~np.isnan(clock)
    timed = trips.dropna(subset="minutes_since_beam_back")
    runs = pd.DataFrame({"run": run[known], "hours": step[known] / 3600, "early": clock[known] < EARLY}).groupby("run").apply(
        lambda g: pd.Series({"hours_early": g.hours[g.early].sum(), "hours_late": g.hours[~g.early].sum()}), include_groups=False)
    runs["trips_early"] = timed[timed.minutes_since_beam_back < EARLY].groupby("run").size().reindex(runs.index, fill_value=0)
    runs["trips_late"] = timed[timed.minutes_since_beam_back >= EARLY].groupby("run").size().reindex(runs.index, fill_value=0)
    runs.reset_index().to_parquet(out / "trip_timing_runs.parquet", index=False)
    n = (runs.trips_early + runs.trips_late).to_numpy()
    share = (runs.hours_early / (runs.hours_early + runs.hours_late)).to_numpy()
    drawn = np.random.default_rng(0).binomial(n, share, size=(DRAWS, len(runs))).sum(axis=1)
    seen = int(runs.trips_early.sum())
    print(runs.round(2).to_string())
    print(f"trips in the first {EARLY} min: {seen} seen, {drawn.mean():.1f} expected at random within runs, "
          f"{(drawn >= seen).mean():.4f} of the draws reach {seen}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
