# Challenge: Surrogate Models (PALLAS)

**Case holder:** Mykyta Lenivenko (IJCLab)
**Team captains:** _TBD_
**Challenge coordinator (organizing team):** Hayg Guler (backup: Damien)

## Overview

PALLAS is a laser-plasma accelerator. A high-power laser pulse is focused into a
few-millimetre gas cell holding a helium/nitrogen mixture; the pulse drives a plasma wave in
its wake, and electrons stripped from the nitrogen inner shell are injected into that wave and
accelerated to roughly 40–500 MeV in a few millimetres. A conventional accelerator needs metres
to do the same thing.

The operator sets a handful of knobs — gas pressure, nitrogen concentration, where the laser is
focused, laser intensity, the shape of the gas density profile — and one electron bunch comes
out. Predicting that bunch takes a particle-in-cell simulation, which costs hours on a cluster.
**The challenge is to replace that simulation with a model that answers in milliseconds, and to
find out honestly what such a model gives up.**

Everything in this challenge is simulation, run with Smilei. That is deliberate: it gives every
sample an exact ground-truth label, so a disagreement between a model and the data is the
model's, never the diagnostic's.

### Two teams, three questions

Team A works physics-informed or hybrid; Team B works purely data-driven. The three questions
both teams are asked to answer are the organisers' own:

1. **What accuracy is lost by removing the physics?**
2. **How much data does each approach need?** — measured as a curve, not a single score.
3. **How does each behave out of distribution?** — at plasma densities and at positions along
   the accelerator that neither team trained on.

Beating a score in-distribution is the warm-up. The three questions are the challenge, and the
dataset ships the splits that let you answer each of them in a way the other team's answer can
be compared against.

## What you get

A public data pack of about 200 MB: three simulation campaigns of the PALLAS accelerator, the
beam's trajectory along it, the bunch's full phase-space moments at twenty planes, five held-out
test sets with their targets removed, and the scorer the organisers will run. See
[`data/README.md`](data/README.md) for the files, the units and the splits.

**The data pack is not in this repository** — it is too large for git. `data/README.md` says
where to download it and where to unpack it.

## Structure

- `scientific_case.md` — background and motivation
- `objectives.md` — easy / medium / hard objectives (bullet points only)
- `requirements.txt` — Python dependencies
- `kickoff_notebook.ipynb` — starter notebook: from the raw pack to a scored submission
- `pallas_score.py` — the scorer, imported by the kickoff notebook
- `data/` — dataset description and schema

The kickoff notebook runs on a laptop CPU in a few minutes and ends with a valid submission
file. Do that first; it also walks through the pack's id conventions and its single injection
flag, which are what a join or a quoted rate gets wrong.

## Getting help

Reach out to the case holder or challenge coordinator listed above, or ask during the Tuesday
hands-on session and the Wednesday mid-challenge check-in.
