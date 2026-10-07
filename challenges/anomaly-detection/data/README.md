# Data — Anomaly Detection

## Description

_TBD — full dataset description to be added by the case holder (Ishkhan Gorgisyan)._

Raw data (ESS `SDS_Data` diagnostic/post-mortem captures + `Archiver_Data` slow EPICS
trends, ~71GB) lives on CC-IN2P3 storage at
`/sps/m4cast/artifact_hackathon_2026/anomaly-detection/raw/` — see the
[access guide](../../../access-guide.md) for CC-IN2P3 account setup.

## Sample

`sample/ess_llrf_sample.parquet` — 210 real events (not synthetic), built from the raw
`SDS_Data` DD captures: one RF chain (`DIG-101:Dwn0`), decimated to 256 points per
component, plus the `trig_code`/`beam_mode`/`beam_state` metadata columns. Used by
`kickoff_notebook.ipynb` so it runs straight from a clone, no CC-IN2P3 access required.

**Open questions for the case holder:** whether `trig_code` is a confirmed fault/trip
code, and what `Cmp0`/`Cmp1` physically represent (`Cmp0` is near-flat in this sample
despite the I/Q-sounding name — see the notebook's section 3).

## Ontologies / schema

See `ontologies/` (placeholder — to be added).
