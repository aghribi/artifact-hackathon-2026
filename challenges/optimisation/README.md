# Challenge: Optimisation (CLEAR / CLARA — double case)

**Co-case owners:** Amelia Pollard, Antonio Gilardi
**Team captains:** Amelia Pollard, Antonio Gilardi
**Challenge coordinator (organizing team):** Barbara Dalena (backup: Adnan Ghribi)

## Overview

_Organiser draft — not yet reviewed by the case owners._ This is a **double case**, covering
two accelerator facilities:
- **CLEAR** (CERN Linear Electron Accelerator for Research) — data situation unexplored so far
- **CLARA** (Compact Linear Accelerator for Research and Applications, Daresbury Laboratory)
  — real beam-image + magnet-settings captures available (see `data/README.md`)

The task: tune real machine settings to hit a beam-quality target (spot size), using a
surrogate trained on real CLARA operational data standing in for a physics simulation (the
real CLARA lattice isn't ready yet — see `data/clara/lattice_data/`). The actual subject is
the **optimisation methods themselves** — classical (grid search, Nelder-Mead, Powell,
gradient-based) through advanced (Bayesian Optimisation, reinforcement learning) — compared
on the same real objective, continuing where Lecture 02's Bayesian-optimisation-with-Cheetah
notebook leaves off. See `kickoff_notebook.ipynb` and `objectives.md`.

## Structure

- `scientific_case.md` — background and motivation
- `objectives.md` — easy / medium / hard objectives (bullet points only)
- `requirements.txt` — Python dependencies
- `kickoff_notebook.ipynb` — starter notebook
- `data/` — dataset description & ontologies, split into `clear/` and `clara/` (placeholders)

## Getting help

Reach out to the case owners or challenge coordinator listed above, or ask during the Tuesday hands-on session and the Wednesday mid-challenge check-in.
