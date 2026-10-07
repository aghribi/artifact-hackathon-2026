# Challenge Objectives — Anomaly Detection

Objectives only — see [`scientific_case.md`](scientific_case.md) for background (still TBD
from the case holder) and [`GRADING.md`](../../GRADING.md) for how scoring and W&B logging
work across all three challenges.

**No confirmed fault labels exist for this dataset** (see `kickoff_notebook.ipynb`, section
2–3): `trig_code` looks like a plausible weak label but is unconfirmed, and `Cmp0`/`Cmp1`
don't behave like a standard I/Q pair. Every objective below is built around
**reconstruction-error model quality**, which needs no labels, rather than a
precision/recall score against a fault list.

**Score for every tier:** mean reconstruction MSE (autoencoder output vs. input, normalised
scale) on the held-out split in `data/sample/holdout_ids.json` — lower is better. Log it to
the shared W&B project (see `GRADING.md`) as `anomaly/holdout_mse` each run.

## 🟢 Easy

- Run `kickoff_notebook.ipynb` to the end on the provided 210-event sample.
- **Bar:** reference autoencoder, held-out mean MSE **0.0194** (42 held-out events,
  stratified 80/20 split, seed 42 — reproduce with `score.py`).

## 🟡 Medium

- Beat the Easy bar using the full raw archive instead of the 210-event sample (CC-IN2P3
  account required — see the [access guide](../../access-guide.md); raw data at
  `/sps/m4cast/artifact_hackathon_2026/anomaly-detection/raw/`). More data, better features,
  a deeper model, or a better architecture are all fair game.
- The notebook's pipeline (sections 4–7) is built to have its pieces swapped and tuned:
  feature set, autoencoder architecture/latent dimension, Isolation Forest/LOF
  hyperparameters, number of clusters. Change one piece, report the reconstruction-MSE
  effect *and* whether it changes the cluster structure (section 7) or the
  method-agreement numbers (section 6).
- Pick one concrete thing to improve and report *why* it helped.

## 🔴 Hard / stretch goal

- Bring in `Archiver_Data` (slow EPICS trends) to look for **precursors** — signal in the
  slow channels *before* a high-scoring `SDS_Data` event — echoing the SPIRAL2 pipeline's
  Precursor Root-Cause Atlas. Report only; no fixed bar, since nobody has done this on ESS
  data yet.
- Confirm (or rule out) `trig_code` as a real fault label by cross-referencing with the case
  holder or ESS's own operations log, and re-score as a proper supervised task if it holds up.
