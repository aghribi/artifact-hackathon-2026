"""Clusters of machine states in the PM and DD captures: is there a group of pulses that marks a coming trip?

One machine state is one cycle of one capture: what the 82 cavities, the 11 beam current monitors and the beam step
look like in that pulse. Input is the output of ess_dd_features.py (cavity_pulses.parquet of PM and DD),
ess_pm_beam_shape.py and ess_pm_beam_step.py. Captures count as in ess_pm_beam_shape.py: PM where the beam is seen
stopping, DD where it holds. Pulses are counted back from the stop: 1 is the last pulse with beam, 0 and below
come after it.

Each cavity feature becomes a distance, in units of spread, to a reference, seen two ways:

    nominal    against all DD pulses of the same cavity: how far the machine is from its usual state
    own        against the pulses of the same cavity and capture 5 or more before the stop: what changed

Per section (Spokes, MBL, HBL) and cycle, the state holds the median distance of each feature over the cavities,
the share of cavities more than 5 spreads off on any feature and the share without an RF pulse. The beam adds its
level and, in the own view, the distance of the beam step. The states are scaled, reduced to 10 principal
components and clustered with k-means, a Gaussian mixture, Ward linkage and HDBSCAN.

Each cluster is then read against the pulses: 5 or more before the stop, 2 to 4 before, the last one, after; for
PM and for DD, where the end of the capture stands for the stop. The gain of a cluster is its share of the pulses 2
to 4 before the stop minus its share of the earlier pulses, averaged over the captures of one type. Any capture
drifts over its 16 seconds, so later pulses sit further from the earlier ones in DD too. Chance of the largest PM
gain comes from swapping PM and DD at random between whole captures.

The trips are clustered too: one row per PM capture, its own-view state at 2, 1 and 0 pulses from the stop, set
against the kind of trip in the interlock log and the year.

    <out>/cluster_states.parquet    one row per capture and cycle: group, state, cluster of each method and view
    <out>/cluster_summary.parquet   one row per view, method and cluster: pulses of each group, gain, chance
    <out>/cluster_trips.parquet     one row per PM capture: kind, year, cluster of each method

    python ess_pm_clusters.py <out_pm> <out_dd> <out_pm_beam> <mps_transitions.parquet> <out>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ess_archiver_pretrip import PERMIT, trip_kinds
from ess_pm_beam_shape import EARLIER

FEATURES = ["field_level", "flat_len", "flat_std_rel", "flat_tilt_rel", "phase_mean_deg", "phase_std_deg", "fwd_flat",
            "fwd_flat_std_rel", "refl_ratio", "decay_tau"]
KEY = ["event_id", "cycle_id"]
OFF = 5              # spreads beyond which a cavity counts as off
CLIP = 20            # distances are clipped here so that a cut pulse does not own a component alone
COMPONENTS = 10
K_RANGE = range(2, 9)
TRIP_K = range(2, 7)
PERMUTATIONS = 1000
GROUPS = {"DD earlier": ("DD", EARLIER, 99), "DD 2 to 4 before": ("DD", 2, 4), "DD last pulse": ("DD", 1, 1),
          "PM earlier": ("PM", EARLIER, 99), "PM 2 to 4 before": ("PM", 2, 4), "PM last pulse": ("PM", 1, 1),
          "PM after": ("PM", -99, 0)}


def distance(x: np.ndarray, is_ref: np.ndarray, phase: bool) -> np.ndarray:
    """Signed distance of each value to the median of the reference values in units of their spread; phases are
    wrapped. A reference value is compared with the other reference values, so that it is treated like the others."""
    diff = lambda v, c: (v - c + 180) % 360 - 180 if phase else v - c
    out = np.full(len(x), np.nan)
    ref_rows = np.flatnonzero(is_ref & ~np.isnan(x))
    for i in range(len(x)):
        ref = x[ref_rows[ref_rows != i]]
        if len(ref) < 5 or np.isnan(x[i]):
            continue
        centre = np.median(ref)
        spread = 1.4826 * np.median(np.abs(diff(ref, centre)))
        out[i] = diff(x[i], centre) / spread if spread > 0 else np.nan
    return out


def cavity_states(cav: pd.DataFrame, reference: pd.Series, by: list[str]) -> pd.DataFrame:
    """Per capture, cycle and section: median distance of each feature, share of cavities off and without pulse."""
    z = pd.DataFrame(index=cav.index, columns=FEATURES, dtype=float)
    for _, rows in cav.groupby(by).indices.items():
        g = cav.iloc[rows]
        is_ref = reference.iloc[rows].to_numpy() & g.has_pulse.to_numpy()
        if is_ref.sum() < 6:
            continue
        for f in FEATURES:
            x = g[f].where(g.has_pulse).to_numpy(dtype=float)
            z.iloc[rows, z.columns.get_loc(f)] = distance(x, is_ref, f.startswith("phase_mean"))
    frame = pd.concat([cav[KEY + ["section", "has_pulse"]], z], axis=1)
    frame["off"] = (z.abs() > OFF).any(axis=1).where(cav.has_pulse)
    state = frame.groupby(KEY + ["section"]).agg(**{f: (f, "median") for f in FEATURES}, off=("off", "mean"),
                                                 no_pulse=("has_pulse", lambda h: 1 - h.mean()))
    state = state.unstack("section")
    state.columns = [f"{section} {f}" for f, section in state.columns]
    return state


def gain(labels: np.ndarray, group: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    """Per capture type and cluster: its share of the pulses 2 to 4 before the stop minus its share of the earlier
    pulses, averaged over the captures. Each capture counts once, whatever its number of pulses."""
    frame = pd.DataFrame({"label": labels, "group": group, "event": events})
    frame = frame[frame.group.str.endswith(("earlier", "2 to 4 before"))]
    frame["type"], frame["late"] = frame.group.str[:2], frame.group.str.endswith("2 to 4 before")
    share = pd.crosstab([frame.type, frame.event, frame.late], frame.label, normalize="index").reindex(
        columns=np.unique(labels), fill_value=0)
    return (share.xs(True, level="late") - share.xs(False, level="late")).dropna().groupby(level="type").mean()


def chance(labels: np.ndarray, group: np.ndarray, events: np.ndarray, rng) -> float:
    """Share of draws, with PM and DD swapped at random between whole captures, whose largest PM gain reaches the
    one seen. Any capture drifts over its 16 seconds; the draws hold DD captures, which show what that alone gives."""
    seen = gain(labels, group, events).loc["PM"].max()
    captures = np.unique(events)
    is_pm = np.isin(captures, np.unique(events[np.char.startswith(group.astype(str), "PM")]))
    plain = np.char.replace(np.char.replace(group.astype(str), "PM ", ""), "DD ", "")
    best = []
    for _ in range(PERMUTATIONS):
        pm = captures[rng.permutation(is_pm)]
        prefix = np.where(np.isin(events, pm), "PM ", "DD ")
        best.append(gain(labels, np.char.add(prefix, plain), events).loc["PM"].max())
    return float((np.array(best) >= seen).mean())


def cluster(x: np.ndarray, seed: int, k_range=K_RANGE) -> dict[str, np.ndarray]:
    """Labels of each method; k and the number of mixture components are picked by silhouette and BIC."""
    sample = np.random.default_rng(seed).choice(len(x), size=min(len(x), 5000), replace=False)
    fits = {k: KMeans(k, n_init=10, random_state=seed).fit(x) for k in k_range}
    k = max(k_range, key=lambda k: silhouette_score(x[sample], fits[k].labels_[sample]))
    mixtures = {n: GaussianMixture(n, covariance_type="full", random_state=seed).fit(x) for n in k_range}
    n = min(k_range, key=lambda n: mixtures[n].bic(x))
    out = {f"k-means, k {k}": fits[k].labels_, f"Gaussian mixture, {n}": mixtures[n].predict(x),
           f"Ward, k {k}": AgglomerativeClustering(k).fit_predict(x)}
    if len(x) >= 200:
        out["HDBSCAN"] = HDBSCAN(min_cluster_size=max(20, len(x) // 200)).fit_predict(x)
    return out


def main(out_pm: Path, out_dd: Path, out_beam: Path, mps_path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    columns = KEY + ["event_id", "cycle_ts", "cavity", "section", "has_pulse"] + FEATURES
    cav = pd.concat([pd.read_parquet(p / "cavity_pulses.parquet", columns=list(dict.fromkeys(columns))) for p in (out_pm, out_dd)])
    cav = cav.drop_duplicates(KEY + ["cavity"])
    beam = pd.read_parquet(out_beam / "beam_shape_pulses.parquet",
                           columns=KEY + ["capture_type", "cycle_ts", "level", "pulses_before_stop", "beam_stops"])
    beam = beam[beam.beam_stops == (beam.capture_type == "PM")]
    cycles = beam.groupby(KEY).agg(capture_type=("capture_type", "first"), cycle_ts=("cycle_ts", "first"),
                                   back=("pulses_before_stop", "first"), beam_level=("level", "median")).reset_index()
    cycles = cycles[cycles.back >= -2]
    for name, (capture_type, low, high) in GROUPS.items():
        cycles.loc[(cycles.capture_type == capture_type) & cycles.back.between(low, high), "group"] = name
    cav = cav.merge(cycles[KEY + ["capture_type", "back"]], on=KEY).reset_index(drop=True)

    step = pd.read_parquet(out_beam / "beam_step_pulses.parquet", columns=KEY + ["section", "amp_step_distance", "phase_step_distance"])
    step = step.groupby(KEY + ["section"])[["amp_step_distance", "phase_step_distance"]].median().unstack("section")
    step.columns = [f"{section} beam {f.replace('_distance', '')}" for f, section in step.columns]

    states = {
        "nominal": cavity_states(cav, cav.capture_type.eq("DD"), ["cavity"]),
        "own": cavity_states(cav, cav.back.ge(EARLIER), ["event_id", "cavity"]).join(step),
    }
    cycles = cycles.set_index(KEY)
    own_beam = cycles.groupby("event_id", group_keys=False).apply(lambda c: pd.Series(
        distance(c.beam_level.to_numpy(dtype=float), (c.back >= EARLIER).to_numpy(), False), index=c.index))
    states["nominal"]["beam level"] = np.log(cycles.beam_level.clip(lower=0.01)).reindex(states["nominal"].index)
    states["own"]["beam level"] = own_beam.reindex(states["own"].index)

    table, saved = [], cycles[["capture_type", "cycle_ts", "back", "group", "beam_level"]].copy()
    for view, state in states.items():
        state = state.reindex(cycles.index)
        x = state.clip(-CLIP, CLIP).fillna(0).to_numpy()
        x = PCA(COMPONENTS, random_state=0).fit_transform(StandardScaler().fit_transform(x))
        saved = saved.join(state.add_prefix(f"{view}: "))
        for method, labels in cluster(x, 0).items():
            saved[f"{view}: {method}"] = labels
            group, events = cycles.group.to_numpy(), cycles.index.get_level_values("event_id").to_numpy()
            counts = pd.crosstab(labels, group).reindex(columns=list(GROUPS), fill_value=0)
            gains = gain(labels, group, events)
            counts["PM gain"], counts["DD gain"] = gains.loc["PM"].to_numpy(), gains.loc["DD"].to_numpy()
            counts["chance"] = chance(labels, group, events, rng)
            year = pd.Series(cycles.cycle_ts.dt.year.to_numpy()).groupby(labels).agg(lambda y: (y == 2026).mean())
            counts["share 2026"] = year
            counts["median beam level"] = pd.Series(cycles.beam_level.to_numpy()).groupby(labels).median()
            table.append(counts.reset_index(names="cluster").assign(view=view, method=method))
    summary = pd.concat(table, ignore_index=True)
    summary.to_parquet(out / "cluster_summary.parquet", index=False)
    saved.reset_index().to_parquet(out / "cluster_states.parquet", index=False)

    pd.set_option("display.width", 300, "display.max_rows", 500, "display.max_columns", 50)
    for (view, method), g in summary.groupby(["view", "method"], sort=False):
        print(f"\n{view} view, {method}: chance of the largest PM gain {g.chance.iloc[0]:.3f}")
        print(g[["cluster", *GROUPS, "PM gain", "DD gain", "share 2026", "median beam level"]].round(2).to_string(index=False))

    # The trips: own-view state at 2, 1 and 0 pulses from the stop, one row per PM capture.
    own = states["own"].reindex(cycles.index).clip(-CLIP, CLIP).fillna(0)
    pm = cycles[(cycles.capture_type == "PM") & cycles.back.isin([2, 1, 0])]
    wide = own.loc[pm.index].assign(back=pm.back).reset_index().drop(columns="cycle_id").pivot(index="event_id", columns="back").dropna()
    mps = pd.read_parquet(mps_path)
    permit = mps[mps.pv == PERMIT].sort_values("data_ts")
    logged = permit.data_ts[(permit.value == 0) & (permit.value != permit.value.shift())].reset_index(drop=True)
    stop = cycles[(cycles.capture_type == "PM") & (cycles.back == 0)].reset_index().set_index("event_id").cycle_ts
    trips = pd.merge_asof(stop.loc[wide.index].sort_values().rename("stop").reset_index(),
                          pd.DataFrame({"trip_ts": logged, "kind": trip_kinds(mps, logged)}), left_on="stop",
                          right_on="trip_ts", direction="nearest", tolerance=pd.Timedelta("10s")).set_index("event_id").loc[wide.index]
    trips["kind"] = trips.kind.fillna("no trip logged")
    trips["year"] = trips.stop.dt.year
    trips["beam_level"] = cycles[(cycles.capture_type == "PM") & (cycles.back == 1)].reset_index().set_index("event_id").beam_level
    x = PCA(min(COMPONENTS, len(wide) - 1), random_state=0).fit_transform(StandardScaler().fit_transform(wide.to_numpy()))
    for method, labels in cluster(x, 0, TRIP_K).items():
        trips[method] = labels
        print(f"\ntrips, {method}")
        print(pd.crosstab(labels, [trips.year, trips.kind]).to_string())
        print("median beam level per cluster:", trips.groupby(labels).beam_level.median().round(1).to_dict())
    trips.reset_index().to_parquet(out / "cluster_trips.parquet", index=False)


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:6]))
