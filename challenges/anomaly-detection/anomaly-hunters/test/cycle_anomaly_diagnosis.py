"""Summarize waveform reconstruction-error decisions by cycle, using logical OR.

The AE's score and decision threshold are unchanged. A cycle is an anomaly
candidate if any of its PV waveforms exceeds that threshold. PVs are returned
as suspects, never labelled faulty merely because their cycle is labelled faulty.
No autoencoder is trained or resumed by this module.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


class CycleDiagnoser:
    def __init__(self, threshold):
        self.threshold = float(threshold)
        if not np.isfinite(self.threshold):
            raise ValueError("Waveform threshold must be finite")

    @classmethod
    def from_ae_metrics(cls, path):
        """Reuse the exact error threshold selected for an existing AE."""
        return cls(json.loads(Path(path).read_text())["threshold"])

    def predict(self, scores, score_column="score"):
        required = {"cycle_id", "pv", score_column}
        if not required.issubset(scores):
            raise ValueError(f"Missing columns: {sorted(required-set(scores))}")
        waves = scores.copy()
        if waves.cycle_id.isna().any() or waves.pv.isna().any():
            raise ValueError("cycle_id and pv must be present")
        waves["diagnosis_pv"] = waves.pv.astype(str).str.replace(r"-(?:PMData|DODData)$", "", case=False, regex=True)
        waves["waveform_score"] = pd.to_numeric(waves[score_column], errors="raise")
        if np.isinf(waves.waveform_score).any():
            raise ValueError("Scores must be finite or NaN (unscored)")
        waves["threshold"] = self.threshold
        waves["pv_anomaly_candidate"] = waves.waveform_score > self.threshold
        rows = []
        for cycle_id, group in waves.groupby("cycle_id", sort=True):
            scored = group.loc[group.waveform_score.notna()]
            ranked = scored.groupby("diagnosis_pv").waveform_score.max().sort_values(ascending=False)
            suspects = ranked.loc[ranked > self.threshold]
            candidate = bool(group.pv_anomaly_candidate.any())
            score = float(ranked.iloc[0]) if len(ranked) else np.nan
            assert candidate == bool(len(ranked) and score > self.threshold)
            row = dict(cycle_id=int(cycle_id), cycle_score=score, threshold=self.threshold,
                anomaly_candidate=candidate if len(ranked) else None,
                status="anomaly_candidate" if candidate else "normal_candidate" if len(ranked) else "unscored",
                observed_pvs=int(group.diagnosis_pv.nunique()), scored_pvs=len(ranked),
                unscored_waveforms=int(group.waveform_score.isna().sum()),
                candidate_pv_count=len(suspects), candidate_pvs="|".join(suspects.index),
                highest_score_pv=ranked.index[0] if len(ranked) else "",
                top5_pvs="|".join(ranked.index[:5]))
            for column in ["cycle_ts", "beam_mode", "split"]:
                if column in group:
                    values = group[column].dropna().astype(str).unique()
                    if len(values)>1:
                        raise ValueError(f"cycle_id={cycle_id}: inconsistent {column}")
                    row[column] = values[0] if len(values) else None
            rows.append(row)
        return waves, pd.DataFrame(rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores",required=True,help="CSV with cycle_id, pv and waveform error")
    parser.add_argument("--metrics",required=True,help="Saved AE metrics JSON containing its threshold")
    parser.add_argument("--score-column",default="score")
    parser.add_argument("--output",required=True,help="Cycle predictions CSV")
    parser.add_argument("--candidate-pvs",required=True,help="Suspect PV captures CSV")
    args=parser.parse_args()
    for path in [args.output,args.candidate_pvs]:
        if Path(path).exists():raise FileExistsError(path)
    model=CycleDiagnoser.from_ae_metrics(args.metrics)
    waves,cycles=model.predict(pd.read_csv(args.scores),args.score_column)
    cycles.to_csv(args.output,index=False)
    waves.loc[waves.pv_anomaly_candidate].to_csv(args.candidate_pvs,index=False)
    print(f"{len(cycles):,} cycles, {cycles.anomaly_candidate.eq(True).sum():,} anomaly candidates")
