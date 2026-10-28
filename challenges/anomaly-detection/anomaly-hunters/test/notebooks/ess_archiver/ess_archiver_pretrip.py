"""Signals of the Archiver around the beam trips: does a modulator, a klystron, the front end, the beam position, a
magnet, the vacuum or the cryogenics move first?

Input is the raw directory holding Archiver_Data_*.tar and mps_transitions.parquet written by ess_dd_features.py.
The Archiver stores one row per pv each time its value changes, so a value holds until the next row. The signals
read here:

    modulator voltage         PulseVoltage-RB of the 17 modulators (RFQ, DTL, MBL, HBL)
    klystron current          PulseCurrent-K<n>-RB of their klystrons
    modulator oil             OilTankTemp-RB and OilTankHum-RB
    front end RF level        AI<n>-SMonAvg-Mag of the RFQ, DTL and MEBT buncher digitisers (powers and field)
    front end cavity tuning   DetuneDecayQL and DetuneDecayMovAvg of the same cavities
    front end temperature     Temperature of the DTL tanks and the cooling water
    beam position             XP-AvgValue and YP-AvgValue of the beam position monitors, one value per beam pulse
    beam phase                SA-AvgValue of the same monitors (named from its range of -180 to 180)
    beam signal strength      SM-AvgValue of the same monitors (named from its range of 0 to 1)
    beam current              FlatTopCurrentR of the beam current monitors, kept as the reference
    beam transmission         the reading of each beam current monitor over that of the first MEBT one
    magnet power supply       Vol-R, DCLnkVol-R, Cur-R and HeatTmp-R of the magnet power supplies
    vacuum pressure           PrsR of the vacuum gauges
    cryomodule                MeasValue and ValvePosition of the cryomodules of the superconducting linac
    cryo plant                the values of the cryogenic plant and its distribution
    superconducting RF temperature   MeasValue of the RF temperature sensors of the Spokes, MBL and HBL

A trip is FBIS::beam_permit_ok going to 0. The logged time runs late, so each trip is placed on the last sample of
the first MEBT beam current monitor that still reads beam: everything up to that sample happened with the beam on.
A trip is kept when the beam was steady over the 2 minutes before.

Each signal is compared with its own level 120 to 20 s earlier, in units of its spread there, over four spans:
20 to 5 s before, the last 5 s, the second of the trip, and 1 to 3 s after. Two numbers per span: how far the median
moved (a shift) and how far the furthest sample went (a spike).

What is normal comes from quiet moments: times with the beam permit held and a steady beam, taken every 10 s. A
trip is compared with the quiet moments of the same run and the same beam current (bins of 20 %), so the beam
setting is held fixed. A signal is "out" when at most 1 % of those quiet moments moved as far. A trip, or a quiet
moment, then has a count of signals out, and a trip counts as flagged when it has more than 99 % of the quiet
moments.

    <out>/pretrip_events.parquet    one row per trip and quiet moment: time, kind of trip, beam current, signals
                                    out per span and group
    <out>/pretrip_signals.parquet   one row per trip, signal and span: shift, spike and their share of quiet moments
    <out>/pretrip_summary.parquet   one row per kind of trip, span and group: trips, trips flagged
    <out>/pretrip_model.parquet     one row per span and set of signals: how well a model tells trips from quiet moments

The model is an isolation forest: it learns what the quiet moments look like on all signals at once, and gives every
moment a score of how unusual it is. It is fitted on the quiet moments of some runs and scores the quiet moments and
the trips of the other runs. Its input is, per signal, how rare the shift and the spike are among the quiet moments
of the same run and beam current. ROC-AUC 0.5 means trips look like quiet moments, 1 means every trip is more
unusual than every quiet moment.

With `short` as a fourth argument the comparison is made for the hours of high beam current, where the beam trips
every few minutes: the earlier level is 40 to 10 s before, the spans are 10 to 3 s before and the last 3 s, quiet
moments are taken every 2 s among the moments with a beam current reading above 10, and the runs of an afternoon
count as one.

    python ess_archiver_pretrip.py <raw> <mps_transitions.parquet> <out> [short]
"""

import io
import re
import sys
import tarfile
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold

PERMIT = "FBIS::beam_permit_ok"
# Archiver families read: whether only their front end pvs are kept (the RF levels of the superconducting cavities are
# covered by the waveform features), and the group of a pv by a part of its signal name ("" fits any signal).
FAMILIES = {
    "MODULATORS": (False, {"PulseVoltage": "modulator voltage", "OilTank": "modulator oil"}),
    "KLYSTRONS": (False, {"PulseCurrent": "klystron current"}),
    "LLRF": (True, {"SMonAvg": "front end RF level"}),
    "RF_PWR_FLD": (True, {"SMonAvg": "front end RF level"}),
    "RF_DET_Q_W": (True, {"DetuneDecay": "front end cavity tuning"}),
    "RF_TEMP": (False, {"Temperature": "front end temperature", "MeasValue": "superconducting RF temperature"}),
    "SCBPMS": (False, {"XP-AvgValue": "beam position", "YP-AvgValue": "beam position", "SA-AvgValue": "beam phase",
                       "SM-AvgValue": "beam signal strength"}),
    "SCBCMS": (False, {"FlatTopCurrent": "beam current"}),
    "MAGNETS": (False, {"": "magnet power supply"}),
    "VACCUM": (False, {"PrsR": "vacuum pressure"}),
    "SCL": (False, {"": "cryomodule"}),
    "CRYO": (False, {"": "cryo plant"}),
}
FRONT_END = re.compile(r"^(RFQ|DTL|MEBT|CWM)-")
BEAM_GROUP = "beam current"
# Groups counted together, by the name of their sum.
TOGETHER = {
    "all RF signals": {"modulator voltage", "klystron current", "modulator oil", "front end RF level",
                       "front end cavity tuning", "front end temperature"},
    "all beam position monitor signals": {"beam position", "beam phase", "beam signal strength"},
    "all magnet, vacuum and cryo signals": {"magnet power supply", "vacuum pressure", "cryomodule", "cryo plant",
                                            "superconducting RF temperature"},
}
STEP = 0.1               # seconds between the points a signal is read at
BEAM_BIN = 1.2           # quiet moments within this ratio of beam current are one reference
OUT_SHARE = 0.01
FOLDS = 5
MIN_READING = 0.5        # beam current reading under which a transmission is not computed: there is no beam
NS = 1_000_000_000

# Windows of the comparison, in seconds around the last beam pulse. "long" is the default. "short" is for the hours
# of high beam current, where the beam trips every few minutes: a shorter earlier level, quiet moments closer to
# each other and to the trips, the runs of an afternoon joined, and only the moments with a beam current reading
# above 10.
#   EARLIER         span that gives a signal its own level
#   QUIET_EVERY     seconds between quiet moments
#   QUIET_CLEAR     seconds the beam permit must hold after a quiet moment
#   RUN_GAP         seconds without a beam current sample that end a run
#   MIN_REFERENCE   quiet moments a reference needs
WINDOWS = {
    "long": dict(EARLIER=(-120, -20), SPANS={"20 to 5 s before": (-20, -5), "last 5 s": (-5, 0),
                                              "second of the trip": (0, 1), "1 to 3 s after": (1, 3)},
                 QUIET_EVERY=10, QUIET_CLEAR=60, QUIET_PER_RUN=1500, RUN_GAP=600, MIN_REFERENCE=100, MIN_BEAM=0),
    "short": dict(EARLIER=(-40, -10), SPANS={"10 to 3 s before": (-10, -3), "last 3 s": (-3, 0),
                                              "second of the trip": (0, 1), "1 to 3 s after": (1, 3)},
                  QUIET_EVERY=2, QUIET_CLEAR=10, QUIET_PER_RUN=5000, RUN_GAP=3600, MIN_REFERENCE=50, MIN_BEAM=10),
}


def use_windows(name: str) -> None:
    """Set the windows and the grid of offsets they give, for this process and the workers it forks."""
    g = globals()
    g.update(WINDOWS[name])
    g["OFFSETS"] = np.arange(EARLIER[0], max(hi for _, hi in SPANS.values()) + STEP / 2, STEP)
    g["EARLIER_MASK"] = (OFFSETS >= EARLIER[0]) & (OFFSETS < EARLIER[1])
    g["SPAN_MASKS"] = [(OFFSETS > lo + 1e-9) & (OFFSETS <= hi + 1e-9) for lo, hi in SPANS.values()]
    g["OFFSETS_NS"] = np.round(OFFSETS * NS).astype("int64")


use_windows("long")

# Filled in the parent before the workers fork: pv -> (group, times in ns, values), and the times of all events.
SERIES: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
TIMES = np.empty(0, dtype="int64")


def read_member(job: tuple[str, int, int, str]) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
    """One Archiver parquet as pv -> (group, times in ns, values). The file is read from its offset in the tar."""
    tar_path, offset, size, family = job
    front_end_only, groups = FAMILIES[family]
    with open(tar_path, "rb") as f:
        f.seek(offset)
        pf = pq.ParquetFile(io.BytesIO(f.read(size)), read_dictionary=["pv"])
    # The value sits in data_Int64 or data_Float64 depending on the pv, so the first filled one is taken.
    values = [c for c in pf.schema_arrow.names if c.startswith("data_") and c != "data_ts"]
    d = pf.read(columns=["pv", "data_ts"] + values).to_pandas()
    d["value"] = d[values].bfill(axis=1).iloc[:, 0].astype(float)
    d["ts"] = d.data_ts.dt.tz_localize(None).astype("datetime64[ns]").astype("int64")
    out = {}
    for pv, g in d.dropna(subset="value").groupby("pv", observed=True):
        # A pv without a group is a setting, not a signal: pulse width, frequency, set point, valve state.
        group = next((group for key, group in groups.items() if key in pv.split(":")[-1]), None)
        if group and (not front_end_only or FRONT_END.match(pv)):
            out[pv] = (group, g.ts.to_numpy(), g.value.to_numpy())
    return out


def load_series(raw: Path, workers: int, families=tuple(FAMILIES)) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
    """Every signal of the given families as one sorted series over all the Archiver windows."""
    jobs = []
    for path in sorted(raw.glob("Archiver_Data_*.tar")):
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                family = re.sub(r"(_\d+)+$", "", Path(member.name).stem)
                if member.name.endswith(".parquet") and family in families:
                    jobs.append((str(path), member.offset_data, member.size, family))
    parts: dict[str, list] = {}
    with Pool(workers) as pool:
        for part in pool.imap_unordered(read_member, jobs, chunksize=4):
            for pv, arrays in part.items():
                parts.setdefault(pv, []).append(arrays)
    series = {}
    for pv, arrays in parts.items():
        ts = np.concatenate([a[1] for a in arrays])
        value = np.concatenate([a[2] for a in arrays])
        # The same sample sits in every window folder that covers its time.
        ts, first = np.unique(ts, return_index=True)
        series[pv] = (arrays[0][0], ts, value[first])
    return series


def on_grid(ts: np.ndarray, value: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Value held by a signal at each offset around each time: rows are times, columns are OFFSETS."""
    idx = np.searchsorted(ts, times[:, None] + OFFSETS_NS[None, :], side="right") - 1
    return np.where(idx >= 0, value[np.clip(idx, 0, None)], np.nan)


def signal_moves(pv: str) -> np.ndarray:
    """Shift and spike of one signal in each span of each event, in units of its own earlier spread."""
    warnings.simplefilter("ignore", RuntimeWarning)
    _, ts, value = SERIES[pv]
    x = on_grid(ts, value, TIMES)
    level = np.nanmedian(x[:, EARLIER_MASK], axis=1)
    spread = 1.4826 * np.nanmedian(np.abs(x[:, EARLIER_MASK] - level[:, None]), axis=1)
    # A signal that did not change in the earlier span has a spread of 0: its usual step between two samples stands in.
    steps = np.abs(np.diff(value))
    steps = steps[steps > 0]
    scale = np.maximum(spread, np.median(steps) if len(steps) else 1.0)
    moves = np.full((len(TIMES), len(SPANS), 2), np.nan)
    for s, mask in enumerate(SPAN_MASKS):
        moves[:, s, 0] = np.abs(np.nanmedian(x[:, mask], axis=1) - level) / scale
        moves[:, s, 1] = np.nanmax(np.abs(x[:, mask] - level[:, None]), axis=1) / scale
    return moves


def trip_kinds(mps: pd.DataFrame, trips: pd.Series) -> pd.Series:
    """Which interlocks drop within a few seconds of the beam permit."""
    drops = mps[(mps.pv != PERMIT) & (mps.value == 0)]

    def kind(t):
        near = drops[(drops.data_ts >= t - pd.Timedelta("5s")) & (drops.data_ts <= t + pd.Timedelta("3s"))]
        systems = {pv.split("::")[-1].replace("-FBIS_ok", "") for pv in near.pv}
        if not systems:
            return "nothing else logged"
        if systems == {"RFLPS"}:
            return "RF protection"
        return "RF protection and beam" if "RFLPS" in systems else "beam instruments"

    return trips.map(kind)


def quiet_share(quiet: np.ndarray, trips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Share of the quiet moments that moved at least as far, for each quiet moment (itself left out) and each trip."""
    ref = np.sort(quiet[~np.isnan(quiet)])
    n = len(ref)
    if n < MIN_REFERENCE:
        return np.full(len(quiet), np.nan), np.full(len(trips), np.nan)
    share_quiet = (n - np.searchsorted(ref, quiet, side="left") - 1) / (n - 1)
    share_trips = (n - np.searchsorted(ref, trips, side="left")) / n
    return np.where(np.isnan(quiet), np.nan, share_quiet), np.where(np.isnan(trips), np.nan, share_trips)


def model_scores(share: np.ndarray, events: pd.DataFrame, members: np.ndarray, span: int, workers: int) -> dict:
    """Isolation forest fitted on quiet moments, scored on the quiet moments and trips of runs it has not seen."""
    # Rarity of each signal among the quiet moments of its run and beam current: 0 is usual, 2 is 1 in 100.
    x = -np.log10(np.clip(share[members][:, :, span, :], 1e-3, 1))
    x = np.nan_to_num(x.transpose(1, 0, 2).reshape(len(events), -1))
    is_trip = events.is_trip.to_numpy()
    place = np.full(len(events), np.nan)    # share of the held-out quiet moments a trip is more unusual than
    for train, test in GroupKFold(FOLDS).split(x, groups=events.run):
        forest = IsolationForest(n_estimators=300, random_state=0, n_jobs=workers).fit(x[train][~is_trip[train]])
        score = -forest.score_samples(x[test])
        quiet = np.sort(score[~is_trip[test]])
        place[test] = np.searchsorted(quiet, score, side="left") / len(quiet)
    return {"signals": int(members.sum()), "trips": int(is_trip.sum()), "auc": float(place[is_trip].mean()),
            "trips_above_99": int((place[is_trip] > 0.99).sum())}


def add_transmission(series: dict) -> None:
    """Add, for each beam current monitor after the first MEBT one, its reading over that of the first."""
    monitors = sorted(pv for pv, (group, _, _) in series.items() if group == BEAM_GROUP)
    first = next(pv for pv in monitors if pv.startswith("MEBT"))
    _, first_ts, first_value = series[first]
    for pv in monitors:
        if pv == first:
            continue
        _, ts, value = series[pv]
        at = first_value[np.clip(np.searchsorted(first_ts, ts, side="right") - 1, 0, None)]
        series[pv.rsplit(":", 1)[0] + ":Transmission"] = ("beam transmission", ts, np.where(at > MIN_READING, value / at, np.nan))


def main(raw: Path, mps_path: Path, out: Path, windows: str = "long", workers: int = 32) -> None:
    global SERIES, TIMES
    use_windows(windows)
    out.mkdir(parents=True, exist_ok=True)
    SERIES = load_series(raw, workers)
    add_transmission(SERIES)
    pvs = sorted(SERIES)
    groups = np.array([SERIES[pv][0] for pv in pvs])
    print(f"{len(pvs)} signals, {sum(len(ts) for _, ts, _ in SERIES.values()):,} samples", flush=True)
    print(pd.Series(groups).value_counts().to_string(), flush=True)

    # 1. The beam: the first MEBT beam current monitor, its runs, and where it reads a steady beam.
    beam_pv = next(pv for pv in pvs if groups[pvs.index(pv)] == BEAM_GROUP and pv.startswith("MEBT"))
    _, beam_ts, beam_value = SERIES[beam_pv]
    run_of_sample = np.concatenate([[0], np.cumsum(np.diff(beam_ts) > RUN_GAP * NS)])
    runs = pd.DataFrame({"start": beam_ts, "end": beam_ts, "top": beam_value}).groupby(run_of_sample).agg(
        start=("start", "min"), end=("end", "max"), top=("top", lambda v: v.quantile(0.95)))
    print(f"beam from {beam_pv}, a sample every {np.median(np.diff(beam_ts)) / NS:.2f} s, {len(runs)} runs", flush=True)

    def steady_beam(times: np.ndarray, until: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run, beam level over the earlier span, and whether the beam held that level up to `until` seconds."""
        run = np.searchsorted(runs.start.to_numpy(), times, side="right") - 1
        x = on_grid(beam_ts, beam_value, times)
        level = np.nanmedian(x[:, EARLIER_MASK], axis=1)
        held = np.nanmin(x[:, OFFSETS <= until], axis=1) >= 0.5 * level
        inside = (times + OFFSETS_NS[0] >= runs.start.to_numpy()[run]) & (times + OFFSETS_NS[-1] <= runs.end.to_numpy()[run])
        return run, level, held & inside & (level > 0.2 * runs.top.to_numpy()[run]) & (level > MIN_BEAM)

    # 2. The trips, each placed on the last beam current sample that still reads beam.
    mps = pd.read_parquet(mps_path)
    permit = mps[mps.pv == PERMIT].sort_values("data_ts")
    permit = permit[permit.value != permit.value.shift()]
    logged = permit.data_ts[permit.value == 0].reset_index(drop=True)
    kinds = trip_kinds(mps, logged)
    logged_ns = logged.dt.tz_localize(None).astype("datetime64[ns]").astype("int64").to_numpy()
    level_at_log = np.nanmedian(on_grid(beam_ts, beam_value, logged_ns)[:, EARLIER_MASK], axis=1)
    last_pulse = np.full(len(logged_ns), -1, dtype="int64")
    for i, (t, level) in enumerate(zip(logged_ns, level_at_log)):
        lo, hi = np.searchsorted(beam_ts, [t - 15 * NS, t + 3 * NS])
        low = np.flatnonzero(beam_value[lo:hi] < 0.5 * level)
        if len(low) and low[0] > 0:
            last_pulse[i] = beam_ts[lo + low[0] - 1]
    found = last_pulse >= 0
    trip_run, trip_beam, trip_ok = steady_beam(last_pulse[found], 0)
    trips = pd.DataFrame({"is_trip": True, "kind": kinds[found].to_numpy(), "time": last_pulse[found],
                          "logged": logged_ns[found], "run": trip_run, "beam": trip_beam})[trip_ok]
    print(f"{len(logged)} trips logged, {found.sum()} with the beam seen stopping, {len(trips)} after a steady beam", flush=True)
    print(f"logged time minus last beam pulse, s: {((trips.logged - trips.time) / NS).describe().round(2).to_dict()}", flush=True)

    # 3. The quiet moments: beam permit held from 30 s before the earlier span to QUIET_CLEAR after, steady beam up to
    # the end of the last span.
    rng = np.random.default_rng(0)
    permit_ts = permit.data_ts.dt.tz_localize(None).astype("datetime64[ns]").astype("int64").to_numpy()
    permit_value = permit.value.to_numpy()
    quiet = []
    for run, r in runs.iterrows():
        t = np.arange(r.start - OFFSETS_NS[0] + 30 * NS, r.end - QUIET_CLEAR * NS, QUIET_EVERY * NS, dtype="int64")
        before = np.searchsorted(permit_ts, t + OFFSETS_NS[0] - 30 * NS, side="right") - 1
        after = np.searchsorted(permit_ts, t + QUIET_CLEAR * NS, side="right") - 1
        t = t[(before >= 0) & (before == after) & (permit_value[np.clip(before, 0, None)] == 1)]
        _, level, ok = steady_beam(t, OFFSETS[-1])
        keep = np.flatnonzero(ok)
        keep = np.sort(rng.choice(keep, min(len(keep), QUIET_PER_RUN), replace=False))
        quiet.append(pd.DataFrame({"is_trip": False, "kind": "quiet moment", "time": t[keep], "logged": t[keep],
                                   "run": run, "beam": level[keep]}))
    events = pd.concat([trips] + quiet, ignore_index=True)
    events["beam_bin"] = np.round(np.log(events.beam) / np.log(BEAM_BIN)).astype(int)
    print(events.groupby(["run", "is_trip"]).size().unstack(fill_value=0).to_string(), flush=True)

    # 4. Shift and spike of every signal around every event, then the share of quiet moments that moved as far.
    TIMES = events.time.to_numpy()
    with Pool(workers) as pool:
        moves = np.stack(pool.map(signal_moves, pvs))            # signal, event, span, shift or spike
    share = np.full(moves.shape, np.nan)
    is_trip = events.is_trip.to_numpy()
    for _, rows in events.groupby(["run", "beam_bin"]).indices.items():
        q, t = rows[~is_trip[rows]], rows[is_trip[rows]]
        for index in np.ndindex(len(pvs), len(SPANS), 2):
            p, s, k = index
            share[p, q, s, k], share[p, t, s, k] = quiet_share(moves[p, q, s, k], moves[p, t, s, k])
    referenced = ~np.isnan(share).all(axis=(0, 2, 3))
    out_flag = np.nanmin(share, axis=3) <= OUT_SHARE             # signal, event, span
    print(f"{(referenced & is_trip).sum()} trips and {(referenced & ~is_trip).sum()} quiet moments have a reference "
          f"of {MIN_REFERENCE} quiet moments at their beam current", flush=True)

    # 5. Signals out per event, and the trips with more of them than 99 % of the quiet moments.
    together = {name: np.isin(groups, list(members)) for name, members in TOGETHER.items()}
    group_names = list(together) + sorted(set(groups))
    summary = []
    for s, span in enumerate(SPANS):
        for group in group_names:
            members = together.get(group, groups == group)
            count = out_flag[members][:, :, s].sum(axis=0)
            events[f"{span} | {group}"] = count
            limit = np.quantile(count[referenced & ~is_trip], 0.99)
            for kind, rows in events[referenced].groupby("kind").indices.items():
                rows = np.flatnonzero(referenced)[rows]
                summary.append({"kind": kind, "span": span, "group": group, "signals": int(members.sum()),
                                "limit": limit, "events": len(rows), "flagged": int((count[rows] > limit).sum())})
    summary = pd.DataFrame(summary)
    events["referenced"] = referenced
    for column in ("time", "logged"):
        events[column] = pd.to_datetime(events[column], utc=True)
    events.to_parquet(out / "pretrip_events.parquet", index=False)
    summary.to_parquet(out / "pretrip_summary.parquet", index=False)

    trip_rows = np.flatnonzero(is_trip & referenced)
    p, e, s = np.meshgrid(np.arange(len(pvs)), trip_rows, np.arange(len(SPANS)), indexing="ij")
    signals = pd.DataFrame({
        "time": events.time.to_numpy()[e.ravel()], "kind": events.kind.to_numpy()[e.ravel()],
        "pv": np.array(pvs)[p.ravel()], "group": groups[p.ravel()], "span": np.array(list(SPANS))[s.ravel()],
        "shift": moves[p, e, s, 0].ravel(), "spike": moves[p, e, s, 1].ravel(),
        "shift_quiet_share": share[p, e, s, 0].ravel(), "spike_quiet_share": share[p, e, s, 1].ravel(),
        "out": out_flag[p, e, s].ravel()})
    signals.to_parquet(out / "pretrip_signals.parquet", index=False)

    # 6. All signals at once: a model of the quiet moments, scored on runs it has not seen.
    kept = events[referenced].reset_index(drop=True)
    sets = {"all signals but the beam monitors": ~np.isin(groups, [BEAM_GROUP, "beam transmission", *TOGETHER["all beam position monitor signals"]]),
            **{name: together[name] for name in TOGETHER}}
    # The model is scored on runs it has not seen, so it needs runs to leave out.
    model = pd.DataFrame([{"signal_set": name, "span": span, **model_scores(share[:, referenced], kept, members, s, workers)}
                          for name, members in sets.items() for s, span in enumerate(SPANS)] if kept.run.nunique() >= FOLDS else
                         [], columns=["signal_set", "span", "signals", "trips", "auc", "trips_above_99"])
    model.to_parquet(out / "pretrip_model.parquet", index=False)

    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 50)
    if len(model):
        print("\nisolation forest, ROC-AUC of trips against quiet moments")
        print(model.pivot(index="signal_set", columns="span", values="auc")[list(SPANS)].round(2).to_string())
        print(model.pivot(index="signal_set", columns="span", values="trips_above_99")[list(SPANS)].to_string())
    flagged = summary.assign(cell=summary.flagged.astype(str) + "/" + summary.events.astype(str))
    for group in group_names:
        print(f"\n{group}: events flagged / events")
        print(flagged[flagged.group == group].pivot(index="kind", columns="span", values="cell")[list(SPANS)].to_string())
    print("\nsignals out most often, trips only, share of trips")
    often = signals.groupby(["span", "kind", "pv"]).out.mean().rename("share").reset_index()
    for span in SPANS:
        print(often[(often.span == span) & (often.share > 0)].sort_values("share", ascending=False).groupby("kind").head(6)
              .sort_values(["kind", "share"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), *sys.argv[4:5])
