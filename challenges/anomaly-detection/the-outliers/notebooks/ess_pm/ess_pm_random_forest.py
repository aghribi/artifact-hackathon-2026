"""Random forest on the cavity pulse features: can a model tell a pulse near a trip from a nominal pulse?

Input is cavity_pulses.parquet of the DD captures (nominal) and of the PM captures (taken at each trip), both written
by ess_dd_features.py. A capture holds one pulse per second. A feature goes in as the size of its distance to the
level the same cavity had 6 to 12 s before the end of the same capture, in units of its spread there. That needs no
matching of beam settings, and a nominal capture is treated exactly like a trip capture. Only the days that hold
both kinds of capture are used, so that the two differ by the trip and not by the date.

Each question is a forest scored on captures it has not seen (ROC-AUC, 0.5 = no better than chance):

    trip pulse                       last pulse of a PM capture against the last pulse of a nominal capture
    before trip                      pulses 3 to 5 s before the end of a PM capture against the same pulses of a
                                     nominal capture
    before trip, same beam current   the same, kept to the beam currents that both kinds of capture ran at
    control                          last pulse of nominal captures split in two arbitrary halves: what a forest
                                     finds when there is nothing to find

Trips happen at beam currents the nominal captures never ran at, so a forest can tell the two apart from the beam
setting alone. "beam current alone" is the AUC of the beam current used as the only predictor of "before trip".

    <out.parquet>   one row per question and feature: auc of the question, importance of the feature (drop of the
                    AUC when the feature is shuffled in the held-out captures)

    python ess_pm_random_forest.py <dd cavity_pulses.parquet> <pm cavity_pulses.parquet> <out.parquet>
"""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

KEY = ["event_id", "cavity"]
FEATURES = ["field_level", "pulse_len", "flat_len", "rise_len", "fall_len", "decay_tau", "flat_std_rel", "flat_p2p_rel",
            "flat_tilt_rel", "flat_max_dev_rel", "flat_step_max_rel", "phase_mean_deg", "phase_std_deg", "phase_tilt_deg",
            "fwd_peak", "fwd_flat", "fwd_flat_std_rel", "refl_flat", "refl_fill_peak", "refl_end_peak", "refl_ratio",
            "field_end_after_rf_off"]
EARLIER = (6, 12)    # seconds before the end of a capture whose pulses are its reference; the same span in every capture
MIN_REFERENCE = 5    # reference pulses a cavity needs in a capture
BEFORE = (3, 5)      # seconds before the end that count as before the trip: the beam is still there
FOLDS = 5
MIN_CAPTURES = 5     # captures of each kind a day needs to be used
CLIP = 50            # distances are capped: one extreme pulse must not stretch the scale


def own_distances(pulses: pd.DataFrame) -> pd.DataFrame:
    """Size of the distance of each feature to the earlier level of the same cavity in the same capture."""
    earlier = pulses[pulses.cycle_rank_from_last.between(*EARLIER)]
    group = earlier.groupby(KEY)[FEATURES]
    median, count = group.median(), group.size()
    spread = 1.4826 * (earlier[FEATURES] - group.transform("median")).abs().groupby([earlier[c] for c in KEY]).median()
    # A feature that takes few distinct values has a median spread of 0: the standard deviation, then 1, stands in.
    spread = spread.where(spread > 0, group.std()).fillna(0)
    spread = spread.where(spread > 0, 1.0)
    rows = pd.MultiIndex.from_frame(pulses[KEY])
    z = np.abs(pulses[FEATURES].to_numpy() - median.reindex(rows).to_numpy()) / spread.reindex(rows).to_numpy()
    out = pd.DataFrame(z, columns=FEATURES, index=pulses.index).clip(upper=CLIP).fillna(0)
    return out[count.reindex(rows).fillna(0).to_numpy() >= MIN_REFERENCE]


def score(x: pd.DataFrame, y: np.ndarray, captures: np.ndarray, name: str) -> pd.DataFrame:
    """Out-of-capture AUC of a forest and the importance of each feature on the held-out captures."""
    predicted, importance = np.zeros(len(x)), np.zeros(len(FEATURES))
    for train, test in GroupKFold(FOLDS).split(x, y, captures):
        forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, class_weight="balanced", n_jobs=-1, random_state=0)
        forest.fit(x.iloc[train], y[train])
        predicted[test] = forest.predict_proba(x.iloc[test])[:, 1]
        if len(np.unique(y[test])) == 2:
            importance += permutation_importance(forest, x.iloc[test], y[test], scoring="roc_auc", n_repeats=3, random_state=0, n_jobs=-1).importances_mean
    auc = roc_auc_score(y, predicted)
    print(f"{name}: {int(y.sum())} against {int((1 - y).sum())} pulses, {len(np.unique(captures[y == 1]))} against "
          f"{len(np.unique(captures[y == 0]))} captures, AUC {auc:.3f}", flush=True)
    return pd.DataFrame({"question": name, "auc": auc, "feature": FEATURES, "importance": importance / FOLDS,
                         "positive_captures": len(np.unique(captures[y == 1])), "negative_captures": len(np.unique(captures[y == 0]))})


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    columns = KEY + ["capture_window", "has_pulse", "cycle_rank_from_last", "seconds_to_next_trip", "beam_current"] + FEATURES
    dd, pm = (pd.read_parquet(path, columns=columns) for path in sys.argv[1:3])
    # A day counts when it holds at least MIN_CAPTURES captures of each kind: 2025 has trips and one nominal capture.
    enough = lambda t: set(t.groupby("capture_window").event_id.nunique().loc[lambda n: n >= MIN_CAPTURES].index)
    days = enough(dd) & enough(pm)
    dd, pm = dd[dd.capture_window.isin(days)], pm[pm.capture_window.isin(days)]
    near_trip = lambda t: t.groupby("event_id").seconds_to_next_trip.transform("min") < 1
    # Nominal: DD captures that do not end at a trip. Trip: PM captures that end within a second of one.
    nominal, trip = dd[dd.has_pulse & ~near_trip(dd)], pm[pm.has_pulse & near_trip(pm)]
    zn, zt = own_distances(nominal), own_distances(trip)

    def versus(ranks: tuple[int, int], name: str, same_current: bool = False) -> pd.DataFrame:
        pick = lambda z, t: z[t.cycle_rank_from_last.reindex(z.index).between(*ranks)]
        positive, negative = pick(zt, trip), pick(zn, nominal)
        if same_current:
            # Beam currents that at least FOLDS captures of each kind ran at.
            per = lambda z, t: t.loc[z.index].groupby("beam_current").event_id.nunique().loc[lambda n: n >= FOLDS].index
            shared = set(per(positive, trip)) & set(per(negative, nominal))
            positive = positive[trip.beam_current.reindex(positive.index).isin(shared).to_numpy()]
            negative = negative[nominal.beam_current.reindex(negative.index).isin(shared).to_numpy()]
        y = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
        captures = np.r_[trip.event_id.reindex(positive.index).to_numpy(), nominal.event_id.reindex(negative.index).to_numpy()]
        return score(pd.concat([positive, negative]), y, captures, name)

    results = [versus((0, 0), "trip pulse"), versus(BEFORE, "before trip"), versus(BEFORE, "before trip, same beam current", same_current=True)]
    before = lambda t: t[t.cycle_rank_from_last.between(*BEFORE)].beam_current
    alone = roc_auc_score(np.r_[np.ones(len(before(trip))), np.zeros(len(before(nominal)))], np.r_[before(trip), before(nominal)])
    print(f"beam current alone: AUC {alone:.3f}", flush=True)
    results.append(pd.DataFrame({"question": ["beam current alone"], "auc": [alone]}))
    # Control: every other nominal capture is called positive.
    last = zn[nominal.cycle_rank_from_last.reindex(zn.index) == 0]
    captures = nominal.event_id.reindex(last.index).to_numpy()
    results.append(score(last, np.isin(captures, np.sort(np.unique(captures))[::2]).astype(float), captures, "control"))
    pd.concat(results, ignore_index=True).to_parquet(sys.argv[3], index=False)


if __name__ == "__main__":
    main()
