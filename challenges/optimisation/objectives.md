# Challenge Objectives — Optimisation (CLEAR / CLARA)

**⚠️ DRAFT — not written or reviewed by the case owners (Amelia Pollard, Antonio Gilardi).**
This is a first pass from what the organisers could find in the data itself, so there's
something concrete to react to instead of a blank page. Expect it to change.

Objectives only — see [`scientific_case.md`](scientific_case.md) (still TBD) and
[`GRADING.md`](../../GRADING.md) for how scoring and W&B logging work across all three
challenges.

## What we found in the data (CLARA side only — CLEAR is still unexplored)

- A live, already-working REST API over CLARA's archiver + diagnostic-camera database:
  `http://apml1.dl.ac.uk:9876` (search/filter by PV, time window, camera; see
  `data/clara/machine_data/image_db_examples.ipynb` for a full walkthrough — connecting,
  listing PVs/cameras, filtered search, time series).
- Several camera stations across CLARA's beamline (`CLA-S02-DIA-CAM-03`,
  `CLA-S04-DIA-CAM-05`, `CLA-S07-DIA-CAM-04`, `CLA-LAS-DIA-CAM-04`, ...), each an HDF5 image
  capture, timestamped, alongside magnet/RF archiver PVs (`MAG:...:SETI`, `...:getPhase`,
  `...:getPower`) queryable at the same time window — i.e. (machine settings) ↔ (beam image)
  pairs are there, just not pre-packaged into a training table the way PALLAS's data is.
  A couple of months of this archive also sits raw at
  `/sps/m4cast/artifact_hackathon_2026/optimisation/raw/` on CC-IN2P3 (~0.9TB,
  reassemble with `cat archive.tar.gz.part_* | tar -xzf -`) if the API doesn't cover
  something you need.
- `data/clara/lattice_data/CLARA_cheetah.json` — **explicitly marked
  `"info": "This is a placeholder lattice description"` inside the file itself.** Don't build
  on it as if it were the real lattice yet.
- Monday's Lecture 02 explicitly points here: its second notebook is "Beam tuning with
  Bayesian optimisation and Cheetah" — this challenge is presumably that idea, for real.

## Proposed shape (confirm with case owners)

Everyone goes Easy → Medium → Hard, so teams are comparable, mirroring the surrogate-models
challenge's structure.

## 🟢 Easy

- Connect to the CLARA API, pull a dataset of (one or two magnet/RF settings) ↔ (a beam-size
  or centroid statistic from one camera) for a time window with real variation, and fit a
  baseline regression settings → beam property.
  **Score:** R² on a held-out time split. **Bar:** _TBD — needs a reference run from the case
  owners._

## 🟡 Medium

- Multi-camera / multi-section version of the Easy task, or the inverse direction (target
  beam property → required settings).
- A real lattice (once provided) opens up a simulation-informed baseline: compare a
  data-driven model against a Cheetah prediction from the real CLARA lattice.

## 🔴 Hard / stretch goal

- A genuine optimisation loop: use a trained surrogate (or Cheetah once the real lattice is
  available) inside Bayesian optimisation to propose settings that minimise a beam-quality
  objective, continuing where Lecture 02's second notebook leaves off.
- Bring in CLEAR once its data/lattice situation is understood — currently unexplored by the
  organisers.

## Open questions for Amelia / Antonio

- Is CLEAR data structured the same way (API + HDF5 images), or different?
- What's the real CLARA lattice, to replace the placeholder Cheetah JSON?
- Is there a specific beam-quality objective you want "optimisation" to mean here (spot
  size? emittance? something else)?
- Any existing reference numbers/baselines to set the bars?
