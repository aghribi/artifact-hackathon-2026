"""Clusters of machine states in the Archiver: is there a state of the machine in which trips come more often than
its beam current says?

One machine state is the machine every 10 s of beam: the value each Archiver signal holds at that moment, and how
much each group of signals moved over the minute before. Input is the raw directory holding Archiver_Data_*.tar and
mps_transitions.parquet written by ess_dd_features.py. The signals are those of ess_archiver_pretrip.py. The beam is
the first MEBT beam current monitor; a moment has beam when it reads more than a fifth of the top of its run, as in
ess_trip_timing.py.

Each signal is scaled by its median and spread over all moments with beam; signals that never move are dropped.
The levels are reduced to 10 principal components and joined with the movement of each group. The states are
clustered with k-means and a Gaussian mixture, k picked by silhouette and BIC, and with HDBSCAN.

A trip is placed on the last beam current sample that still reads beam, as in ess_trip_timing.py, and takes the
cluster of the state LEAD seconds before it, for each lead. A cluster is riskier than its beam current when it holds
more trips than its hours of beam at each level of the beam current give, with the trip rate of that level over all
the data. Chance comes from drawing each trip again at a random moment of the same run, the same beam current level
and the same time since the beam came back (under 5, 5 to 15, over 15 minutes, unknown).

    <out>/archiver_states.parquet          one row per moment: time, run, beam current, minutes since the beam came
                                           back, cluster of each method
    <out>/archiver_cluster_summary.parquet one row per method, lead and cluster: hours, trips, expected, chance
    <out>/archiver_cluster_trips.parquet   one row per trip: time, kind, beam current, cluster of each method and lead

    python ess_archiver_clusters.py <raw> <mps_transitions.parquet> <out>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from ess_archiver_pretrip import BEAM_GROUP, NS, PERMIT, RUN_GAP, load_series, trip_kinds
from ess_trip_timing import LEVELS

EVERY = 10           # seconds between two machine states
MOVE = 60            # seconds over which the movement of a signal is taken
LEADS = [10, 60, 300]
SINCE = [0, 5, 15, np.inf]   # minutes since the beam came back, in bins: the first minutes after a restart differ
CLIP = 10            # scaled values are clipped here so that one stuck signal does not own a component
COMPONENTS = 10
K_RANGE = range(2, 11)
DRAWS = 2000


def main(raw: Path, mps_path: Path, out: Path, workers: int = 32) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    series = load_series(raw, workers)
    pvs = sorted(series)
    beam_pv = next(pv for pv in pvs if series[pv][0] == BEAM_GROUP and pv.startswith("MEBT"))
    _, beam_ts, beam_value = series[beam_pv]

    # Moments every EVERY seconds inside the runs, kept where the beam is on.
    run_of_sample = np.concatenate([[0], np.cumsum(np.diff(beam_ts) > RUN_GAP * NS)])
    runs = pd.DataFrame({"ts": beam_ts, "v": beam_value}).groupby(run_of_sample).agg(
        start=("ts", "min"), end=("ts", "max"), top=("v", lambda v: v.quantile(0.95)))
    times = np.concatenate([np.arange(r.start + MOVE * NS, r.end, EVERY * NS) for r in runs.itertuples()])
    run = np.searchsorted(runs.start.to_numpy(), times, side="right") - 1
    held = lambda ts, v, t: np.where((i := np.searchsorted(ts, t, side="right") - 1) >= 0, v[np.clip(i, 0, None)], np.nan)
    beam = held(beam_ts, beam_value, times)
    on = beam > 0.2 * runs.top.to_numpy()[run]
    # Minutes since the beam was last off in the same run; unknown before the first stop of a run.
    off = ~(beam_value > 0.2 * runs.top.to_numpy()[run_of_sample])
    last_off = pd.Series(np.where(off, beam_ts, np.nan)).groupby(run_of_sample).ffill().to_numpy()
    i = np.searchsorted(beam_ts, times, side="right") - 1
    since = np.where(run_of_sample[i] == run, (times - last_off[i]) / NS / 60, np.nan)
    times, run, beam, since = times[on], run[on], beam[on], since[on]
    since_bin = np.where(np.isnan(since), -1, np.searchsorted(SINCE, since, side="right") - 1)
    level = np.select([(beam >= lo) & (beam < hi) for lo, hi in LEVELS.values()], list(LEVELS), "none")
    print(f"{len(times):,} moments with beam, {len(times) * EVERY / 3600:.1f} hours, {len(runs)} runs", flush=True)

    # Levels and movements of every signal at every moment.
    levels, moves, groups = [], {}, {}
    for pv in pvs:
        group, ts, v = series[pv]
        now, before = held(ts, v, times), held(ts, v, times - MOVE * NS)
        centre = np.nanmedian(now)
        spread = 1.4826 * np.nanmedian(np.abs(now - centre))
        spread = spread if spread > 0 else np.nanstd(now)
        if not spread > 0 or np.isnan(now).mean() > 0.5:
            continue
        levels.append(np.clip((now - centre) / spread, -CLIP, CLIP))
        moves.setdefault(group, []).append(np.clip(np.abs(now - before) / spread, 0, CLIP))
        groups[pv] = group
    x = np.nan_to_num(np.stack(levels, axis=1))
    x = PCA(COMPONENTS, random_state=0).fit_transform((x - x.mean(0)) / np.maximum(x.std(0), 1e-12))
    movement = np.stack([np.nanmedian(np.stack(m, axis=1), axis=1) for m in moves.values()], axis=1)
    x = np.hstack([x / x[:, 0].std(), np.nan_to_num(movement / np.maximum(np.nanstd(movement, axis=0), 1e-12))])
    print(f"{len(groups)} signals that move, groups: {pd.Series(groups).value_counts().to_dict()}", flush=True)

    sample = rng.choice(len(x), size=min(len(x), 5000), replace=False)
    fits = {k: KMeans(k, n_init=10, random_state=0).fit(x) for k in K_RANGE}
    k = max(K_RANGE, key=lambda k: silhouette_score(x[sample], fits[k].labels_[sample]))
    mixtures = {n: GaussianMixture(n, random_state=0).fit(x) for n in K_RANGE}
    n = min(K_RANGE, key=lambda n: mixtures[n].bic(x))
    labels = {f"k-means, k {k}": fits[k].labels_, f"Gaussian mixture, {n}": mixtures[n].predict(x),
              "HDBSCAN": HDBSCAN(min_cluster_size=len(x) // 100).fit_predict(x)}
    states = pd.DataFrame({"time": pd.to_datetime(times, utc=True), "run": run, "beam_current": beam, "level": level,
                           "minutes_since_beam_back": since, **labels})
    states.to_parquet(out / "archiver_states.parquet", index=False)

    # The trips, on the last beam current sample that still reads beam, as in ess_trip_timing.py.
    mps = pd.read_parquet(mps_path)
    permit = mps[mps.pv == PERMIT].sort_values("data_ts")
    permit = permit[permit.value != permit.value.shift()]
    logged = permit.data_ts[permit.value == 0].reset_index(drop=True)
    kinds = trip_kinds(mps, logged)
    beam_on = beam_value > 0.2 * runs.top.to_numpy()[run_of_sample]
    rows = []
    for t, kind in zip(logged.dt.tz_localize(None).astype("datetime64[ns]").astype("int64"), kinds):
        lo, hi = np.searchsorted(beam_ts, [t - 15 * NS, t + 3 * NS])
        off = np.flatnonzero(~beam_on[lo:hi])
        if len(off) and off[0] > 0:
            rows.append({"time": beam_ts[lo + off[0] - 1], "kind": kind})
    trips = pd.DataFrame(rows)
    trips["moment"] = np.searchsorted(times, trips.time.to_numpy(), side="right") - 1
    # A trip counts when the moment before it is a moment with beam no more than one step away.
    trips = trips[(trips.moment >= 0) & (trips.time - times[trips.moment.clip(lower=0)] <= EVERY * NS)].reset_index(drop=True)
    trips["run"], trips["level"], trips["since"] = run[trips.moment], level[trips.moment], since_bin[trips.moment]
    rate = {lv: (trips.level == lv).sum() / ((level == lv).sum() * EVERY / 3600) for lv in LEVELS}
    print(f"{len(trips)} trips with beam up to them, per hour by level: { {k: round(v, 2) for k, v in rate.items()} }", flush=True)

    # For each lead, the moment that far before each trip; random draws keep its run, its beam current level and
    # its time since the beam came back.
    strata = ["run", "level", "since"]
    pools = {(r, lv, sb): np.flatnonzero((run == r) & (level == lv) & (since_bin == sb)) for (r, lv, sb), _ in trips.groupby(strata)}
    table, at_trip = [], trips.copy()
    for lead in LEADS:
        moment = np.searchsorted(times, trips.time.to_numpy() - lead * NS, side="right") - 1
        valid = (moment >= 0) & (run[moment.clip(min=0)] == trips.run.to_numpy())
        draws = np.stack([np.concatenate([rng.choice(pools[key], size=len(g)) for key, g in trips[valid].groupby(strata)])
                          for _ in range(DRAWS)])
        for method, lab in labels.items():
            at_trip[f"{method}, {lead} s before"] = np.where(valid, lab[moment.clip(min=0)], -99)
            clusters = np.unique(lab)
            hours = pd.crosstab(lab, level).reindex(index=clusters, columns=list(LEVELS), fill_value=0) * EVERY / 3600
            expected = hours.mul(pd.Series(rate)).sum(axis=1).to_numpy() * valid.sum() / len(trips)
            seen = np.array([(lab[moment[valid]] == c).sum() for c in clusters])
            score = lambda counts: ((counts - expected) / np.sqrt(np.maximum(expected, 1))).max()
            drawn = np.array([score(np.array([(lab[d] == c).sum() for c in clusters])) for d in draws])
            chance = float((drawn >= score(seen)).mean())
            for i, c in enumerate(clusters):
                table.append({"method": method, "lead_s": lead, "cluster": int(c), "hours": float(hours.iloc[i].sum()),
                              "median_beam_current": float(np.median(beam[lab == c])), "trips": int(seen[i]),
                              "expected": float(expected[i]), "chance": chance})
    summary = pd.DataFrame(table)
    summary.to_parquet(out / "archiver_cluster_summary.parquet", index=False)
    at_trip.to_parquet(out / "archiver_cluster_trips.parquet", index=False)

    pd.set_option("display.width", 250, "display.max_rows", 500)
    for (method, lead), g in summary.groupby(["method", "lead_s"], sort=False):
        print(f"\n{method}, state {lead} s before the trip: chance of the largest excess {g.chance.iloc[0]:.3f}")
        print(g[["cluster", "hours", "median_beam_current", "trips", "expected"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:4]))
