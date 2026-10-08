# Scientific Case — Anomaly Detection (ESS Time-series)

## Background

European Spallation Source (ESS), currently under commissioning, is a multidisciplinary
neutron research facility based in Lund, Sweden [1]. It is driven by a 5 MW proton linac
that consists of both normal and superconducting parts to accelerate the proton beam up to
2 GeV design energy. The initial user operation is planned with a reduced 2 MW power at beam
energy of 800 MeV.

Being a user facility, the main aim of ESS is to provide uninterrupted neutron beam for the
users. From the ESS linac point of view, this translates to a demanding availability
requirement of 95%. Furthermore, as a multi-megawatt power machine, ESS linac has strict
limits on beam loss and activation of components to allow hands-on maintenance on the
accelerator.

## Motivation

ESS linac is a complex system with millions of process variables (PVs) generating a large
amount of data during operation that can have a strategic importance. In particular, data
from various accelerator systems (e.g. LLRF, instrumentation, machine protection etc.) can
be leveraged for anomaly detection in the accelerator.

Early detection of anomalous behavior and identifying precursors to different system faults
can help preventing trips and beam loss events that may result in activation or damage of
machine components. Furthermore, accurate and efficient identification and classification of
faults helps with troubleshooting and reduces the valuable recovery time, hence, increasing
the availability of the machine [2].

## Current state / prior work

Currently, there are ongoing efforts to establish a standardized procedure for operational
data retrieval and inventory, automated fault tracking and labelling and general MLOps
development. Deployment of the Synchronous Data Service (SDS) [3] and Post Mortem tools have
been finalized — this is exactly the `SDS_Data`/`Archiver_Data` raw archive this challenge's
data comes from (see `data/README.md`). **No prior work was done on model development for
this purpose** — there is no existing pipeline to adapt for ESS specifically (unlike
SPIRAL2 — see `kickoff_notebook.ipynb`'s pointers to the SPIRAL2 LLRF pipeline, a
methodological reference only, on a different machine).

## References

[1] Garoby R. et al (2018) The European spallation source design. Phys. Scr. 93(1):014001.
https://doi.org/10.1088/1402-4896/aa9bff

[2] Gorgisyan I. et al (2026). Strategy of AI Application and Current Activities at European
Spallation Source Proton Linac. EPJ Res. Infrastruct. 10(1):28.
https://doi.org/10.1007/s41781-026-00182-7

[3] Martins J. et al (2025) The ESS synchronous data service (SDS) development and first
results. In: ICALEPCS25, Chicago, IL, United States, Paper TUMG018.
https://doi.org/10.18429/JACoW-ICALEPCS2025-TUMG018
