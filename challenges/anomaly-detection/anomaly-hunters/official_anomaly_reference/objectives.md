# Challenge Objectives — Anomaly Detection

Objectives only — see [`scientific_case.md`](scientific_case.md) for background and
[`GRADING.md`](../../GRADING.md) for how scoring and W&B logging work across all three
challenges. This structure follows the case holder's own tiers: binary normal-vs-anomalous,
then fault characterisation/grouping, then precursor prediction.

**No confirmed fault labels exist yet for this specific sample** (`trig_code` is a plausible
but unconfirmed weak label — see `kickoff_notebook.ipynb`, sections 2–3). The case holder's
own labelling effort is ongoing and **labels for certain fault classes will be provided**
during the event for the Medium tier — until then, the scoring below is built around
reconstruction-error model quality, which needs no labels.

**Score for every tier:** mean reconstruction MSE (autoencoder output vs. input, normalised
scale) on the held-out split in `data/sample/holdout_ids.json` — lower is better. Log it to
the shared W&B project (see `GRADING.md`) as `anomaly/holdout_mse` each run.

## 🟢 Easy — binary: normal vs. anomalous

Learn the nominal envelope of the machine and detect datasets that fall outside it. Both
supervised and unsupervised approaches are valid — `kickoff_notebook.ipynb` builds an
unsupervised one (autoencoder + reconstruction error, cross-checked against Isolation
Forest/LOF), since no labels exist yet for this sample.

- Run `kickoff_notebook.ipynb` to the end on the provided 210-event sample.
- **Bar:** reference autoencoder, held-out mean MSE **0.0194** (42 held-out events,
  stratified 80/20 split, seed 42 — reproduce with `score.py`).

## 🟡 Medium — characterise and classify the faults

Identify different types of faults and group together those exhibiting similar behaviour.
If the case holder's labels for certain fault classes land in time, use them directly and
compare against your own unsupervised grouping; if not, this stays an unsupervised
clustering task.

- Beat the Easy bar using the full raw archive instead of the 210-event sample (CC-IN2P3
  account required — see the [access guide](../../access-guide.md); raw data at
  `/sps/m4cast/artifact_hackathon_2026/anomaly-detection/raw/`). More data, better features,
  a deeper model, or a better architecture are all fair game.
- Extend the notebook's K-Means clustering (section 7) into real fault characterisation:
  more clusters, better features, or — once labels arrive — check how well your clusters
  line up with the provided fault classes.
- The notebook's pipeline (sections 4–7) is built to have its pieces swapped and tuned:
  feature set, autoencoder architecture/latent dimension, Isolation Forest/LOF
  hyperparameters, number of clusters. Change one piece, report the reconstruction-MSE
  effect *and* whether it changes the cluster structure or the method-agreement numbers.

## 🔴 Hard — precursors and early prediction

Identify precursors to the failures to provide early prediction of an anomaly developing
towards a fault.

- Bring in `Archiver_Data` (slow EPICS trends) to look for precursors — signal in the slow
  channels *before* a high-scoring `SDS_Data` event — echoing the SPIRAL2 pipeline's
  Precursor Root-Cause Atlas. No fixed bar: nobody has done this on ESS data yet, so a
  well-reasoned negative result (no detectable precursor signal) is a legitimate finding.
- Confirm (or rule out) `trig_code` as a real fault label by cross-referencing with the case
  holder or ESS's own operations log, and re-score as a proper supervised task if it holds up.
