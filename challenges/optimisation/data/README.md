# Data — Optimisation (double case)

This challenge covers two facilities. Each has its own subfolder with machine data and lattice data:

- [`clear/`](clear/) — CLEAR (CERN)
- [`clara/`](clara/) — CLARA (Daresbury Laboratory)

## Sample

`sample/clara_s07_sample.parquet` — 1,370 real captures from camera `CLA-S07-DIA-CAM-04`
(20 dates spanning March–April 2026): a background-subtracted centroid + RMS spot size
computed from each real beam image, paired with the real magnet `SETI` settings read from
that same capture's HDF5 attributes. Used by `kickoff_notebook.ipynb` so it runs straight
from a clone, no CC-IN2P3 access required. Built from the full raw archive at
`/sps/m4cast/artifact_hackathon_2026/optimisation/` on CC-IN2P3 (`raw/` = the original
split `.tar.gz`, `extracted/` = the real file tree, 2.4TB, 17,885 HDF5 files across ~20
camera stations) — see the [access guide](../../../access-guide.md).

Each HDF5 file holds `Capture000001`...`Capture0000NN` datasets: a 2560×2160 `uint16`
greyscale image, with ~150 real magnet PVs (`SETI`/`READI`/`GETSETI`/`RPOWER`) as HDF5
attributes on each dataset — image and settings are already paired, no separate query
needed. Some magnet families carry `RPOWER=0` (section powered off for that capture) —
real operational detail, not a data error.

There's also a live archiver+camera search API (documented in
`clara/machine_data/image_db_examples.ipynb`) for finding captures by PV/time/camera
without opening every HDF5 file — useful at scale, not required to get started.

## Ontologies / schema

CLARA PV naming follows `<AREA>-<SYSTEM>-<DEVICE>-<NUMBER>:<FIELD>`, e.g.
`CLA-S02-MAG-QUAD-03:SETI` = section S02, magnet system, quadrupole 03, commanded current.
No formal ontology file beyond this convention yet. CLEAR: _TBD — to be added._
