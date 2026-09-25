# Data — Surrogate Models (PALLAS)

## Description

Particle-in-cell simulations of the PALLAS laser-plasma accelerator, run with Smilei. Every row
is one simulation: the machine settings that were used, and the electron bunch that came out.
**Simulation only — there is no measured data here, and therefore no diagnostic uncertainty and
no shot-to-shot jitter.**

## Getting the pack

The pack is about 200 MB zipped and is **not stored in this repository**.

> **Download:** _TBD — link to be added by the organisers._

**If `pallas_hackathon_data/` is already sitting in this `data/` directory, it is in place and
there is nothing for you to download** — the line above is for the downloadable copy. A
`.gitignore` beside this file keeps those ~200 MB out of any repository this folder is copied
into.

Unpack it so that the folder `pallas_hackathon_data/` sits inside this `data/` directory:

```
challenges/surrogate-models/
├── kickoff_notebook.ipynb
├── pallas_score.py
└── data/
    └── pallas_hackathon_data/     <- unpack here
        ├── campaign_A.parquet
        ├── ...
        └── test_trajectory_ood.parquet
```

The kickoff notebook finds it there by itself. If you keep the pack somewhere else, set
`PALLAS_PACK` to its path before starting Jupyter.

## Files

| file | rows | what it is |
|---|---|---|
| `campaign_A.parquet` | 13,604 | Campaign A — four scalar knobs, both scans in one table; `scan` is `random` (9,816 simulations) or `uniform` (3,788 on a grid) |
| `campaign_B.parquet` | 4,886 | Campaign B — laser fixed, the gas density profile varied and shipped as a 2000-point curve |
| `campaign_B_trajectories.parquet` | 277,415 | the bunch along the accelerator: 2,455 configurations × 113 positions |
| `campaign_B_moments.parquet` | 195,360 | the bunch's full 6×6 covariance at 20 planes, twice: `group` = `all` (every electron; the answer key) and `cohort` (a sub-population) |
| `common_energy_grid_MeV.npy` | 200 bins | the shared spectrum axis, 25.8–510.3 MeV |
| `test_direct.parquet` | 714 | held-out inputs: settings → bunch |
| `test_inverse.parquet` | 1,915 | held-out inputs: bunch → settings |
| `test_trajectory.parquet` | 80,682 | held-out inputs: settings + position → bunch |
| `test_trajectory_ood.parquet` | 60,568 | the same, at unseen plasma densities (536 configurations) |
| `test_moments_planes.parquet` | 6,426 | held-out inputs: settings + plane → the bunch's moments |
| `pack_reference.json` | — | **one file for everything you need to be comparable**: five id lists in its `sections`, and every reference number in its `scores` |
| `DATA_CARD.md` | — | the pack's own detailed card |

`pack_reference.json` replaces the six split files and the separate score file earlier drafts
shipped. Read it once and index it by name:

```python
ref = json.load(open(f"{PACK}/pack_reference.json"))
SPLITS, SCORES = ref["sections"], ref["scores"]
SPLITS["participant_split"]["holdout"]      # the 603 configs you score yourself on
SCORES["direct_published_emulator"]         # what our internal forward model reaches
```

Its `sections` are `reference_split`, `ood_split`, `participant_split`,
`participant_ood_split` and `plane_holdout`; each carries its own `what` line saying what it is
for. Every id list is a list of `config` values.

The two campaigns **do not share a knob set**, which is why they are two files. Campaign A
varies four scalars (`p_1`, `a_0`, `c_N2`, `x_of`) and sampled them twice — once at random and
once on a grid — which is what the `scan` column distinguishes; Campaign B fixes the laser and
varies the shape of the gas density profile (`P_max`, `cN2_max`, `L_inj`, `dip_frac`, `x_of`).
The hidden inverse test set is drawn from the random scan only.

**No target column appears in any shipped test file.** This is checked on every build of the
pack. The answer keys are held by the organisers.

## Particle clouds (a separate download)

Not in the pack itself, because of its size: Campaign B's bunch as individual electrons, for the
Hard objective "Predict the bunch as particles". Up to 2,000 tracked electrons per simulation (a
few configurations with a small bunch have fewer), the same electrons followed along the accelerator, for the 4,882 Campaign B configurations that are in no
test file. One `Config_<id>.npz` per configuration, the same `config` as everywhere else.

| part | what | size |
|---|---|---|
| `planes20/` | the twenty planes of `campaign_B_moments.parquet`; start here | 4.9 GB |
| full set | every snapshot from the bunch's first (1.5–2.1 mm) to 7.04 mm, every 22.5 µm — 222 to 247 per configuration; same keys | 50.0 GB |

The keys that matter:

- `state` (snapshots × electrons × 6), float32, in the order `zeta_um, y_um, z_um, ux, uy, uz`:
  `zeta_um` is the position along the beam (µm), measured from a point moving at the speed of
  light, NOT from the laser's front as in the moment table: the two differ by a few tens of µm
  (about 35 µm at 2 mm, growing slowly along the accelerator), the same orientation; positions
  within the bunch are unaffected. To compare with the moment table, compare widths and shapes, or
  shift each plane by the difference of the two `zeta` means. `y_um` and
  `z_um` are transverse (µm — as in the moment table, `z` here is a TRANSVERSE coordinate), and
  `ux, uy, uz` are momenta in units of m_e c, `ux` along the beam.
- `present` (snapshots × electrons), bool: the electron exists at that snapshot. Electrons are lost
  along the way, and a late-injecting bunch has none at its first planes. Where `present` is
  false, `state` is NaN.
- `weight` (electrons): each electron's share of the bunch's charge. **The clouds are
  charge-weighted, not equal-weight** — use `weight` in every mean, width and loss, or you model
  the simulation's sampling rather than the beam.
- `plane_mm` (`planes20/` only) and `z_win_mm`: the position along the accelerator of each
  snapshot, in mm. The snapshot taken for a plane sits within 10 µm of it.
- `theta` with `theta_keys`: the settings, as a check against `campaign_B.parquet`.

The moments computed from these clouds are close to, but not identical to,
`campaign_B_moments.parquet`: the moment table was computed from every simulated electron, and
the clouds are a sample of up to 2,000 of them.

## Units

Units are the thing people get wrong here, so each column carries one.

- **Energies are MeV everywhere.** Campaign B was stored natively in Lorentz gamma and is
  converted in the pack; the native columns are not shipped, so there is nothing to pick up by
  mistake and no unit flag to check.
- `scan` is `random` or `uniform` in `campaign_A.parquet`, and is absent from `campaign_B.parquet`.
- `dE_mad` and `dE_std` are **relative** spreads, dimensionless.
- `q_end` is in **coulombs** — a 3 pC cut is `3e-12`. `q_pC` beside it is in picocoulombs.
- `p_1` is in mbar; `P_max` is in **pascals**; `x_of` in µm; `L_inj` and `plane_mm` and `z_mm` in mm.
- In `campaign_B_moments.parquet` only, **`z` is a transverse coordinate**; the position along the
  accelerator is `plane_mm`.

## What was done to the raw output

1. **Energy units harmonised** — MeV everywhere, native columns dropped.
2. **Spectra resampled** onto one shared 200-bin grid. Each row's native axis is kept as
   `ener_axis_MeV`, and `spec` on it is charge per MeV (it integrates to `q_pC`). **In this pack
   version `spec_common` is not charge per bin:** each row is off by the width of its own native
   bin, so multiply it by `np.diff(ener_axis_MeV).mean()` of the same row to get pC per shared
   bin. The shape is unaffected. The next pack version ships it corrected.
3. **One injection flag** — `injected` is `q_pC >= 3.0`, computed identically for all three
   campaigns. The source files' own labels are not shipped; they were produced under different
   rules and were not comparable.
4. **273 unphysical runs deleted**, not flagged — runs whose laser envelope oscillates (seven or
   more turning points at 5 % depth or more). They are simply absent; there is no column to filter.

## Two id conventions

- **`config` is one simulation, everywhere.** It is unique in both Campaign A scans and in
  Campaign B, it is the column every test file joins on, and it is what every submission is keyed
  by. Campaign A's random scan stacks five sub-scans that each numbered their draws from zero, so
  its id folds the two together as `sub_scan * 2401 + draw + 1`, running 1 to 12005; the uniform
  scan's start at 100000; Campaign B's source index was already one per simulation and is
  unchanged. Because one row is one simulation, an ordinary row-wise split is already a split on
  simulations — there is nothing to group.
  Two columns beside it are **not** keys: `draw` is the index the source scan gave a run and
  repeats across sub-scans, and `row_id` is a readable provenance label, `scan:draw:sub_scan`.
  Five rows sharing a `draw` are five different settings, not one setting five times — between
  sub-scans 0 and 1 only `a_0` is repeated, while `p_1` differs by up to 10 mbar, `c_N2` across
  its whole range and `x_of` by 300 µm.
- **The moments table spells the id differently.** It names the simulation in `sim_id` as the
  string `"Config_1001"`; everywhere else in the pack, `test_moments_planes.parquet` included, the
  same id is the integer `1001`. Strip the prefix before joining:
  `mom["config"] = mom.sim_id.str.removeprefix("Config_").astype(int)`.
- **The moments table spells three centroids differently too.** It stores them as `mean_zeta_um`,
  `mean_y_um` and `mean_z_um`; the scorer and the answer key call them `mean_zeta`, `mean_y` and
  `mean_z`, in the same µm. Rename before you submit, or the scorer stops on a missing column.

## Splits, and the three axes beyond accuracy

In-distribution R² with all the data is one axis, and on Campaign A's settings → endpoint task it
is saturated. These are the axes the challenge's three questions actually live on. Each ships as
an **id list** inside `pack_reference.json`, not as a pre-cut training file: the whole corpus is
here, and the section says which configurations a comparable model may train on. A team that
trains on everything simply stops being comparable.

- **Your own fold** (`participant_split`) — 603 configurations held out of the training corpus,
  leaving a pool of 1,852. Score yourself here.
- **Data budget** (`participant_split`'s `pool`) — train at 100, 300 and 1,000 configurations and on
  the whole pool, all scored on the same fold. **No subsets ship: you choose which configurations
  make up each size**, and choosing them well is part of the task. Our reference curve is the mean
  and spread over ten random draws of each size, so compare your choice with random draws too.
- **Out of distribution, density** (`participant_ood_split`) — train on 1,714 configurations below
  6,528.6 Pa of `P_max`, then score **one** model on two held-out sets: 563 configurations inside
  that range and 178 above it. Neither is trained on, so the gap between the two numbers is
  extrapolation and nothing else.
- **Out of distribution, position** (`plane_holdout`) — train on 11 planes and predict the bunch at
  9 it has never seen (2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 5.0 mm). **Read 3.5 and 5.0 apart
  from the rest:** the other seven are interpolated across a 0.2 mm gap, those two across 0.6 mm
  and 2.0 mm.

The `reference_split` and `ood_split` sections are the folds the internal reference numbers were scored on.
They are listed for reference: their test configurations are not in the public tables, so **the
internal reference numbers are a target to aim at, not a number you can reproduce.**

## Scoring

`pallas_score.py`, in the challenge folder above this one, is the scorer the organisers run.

**Submit physical values.** The scale each target is scored on is applied for you — energy
linearly, charge as log₁₀ of pC floored at 0.1, energy spread as log₁₀ of percent.

**The position-dependent tasks are scored against the per-position mean.** R² divides by the
spread across configurations *at each position*, summed over positions — not by the spread over
every cell at once. Most of the variance along a trajectory is the z-trend itself, and a global
denominator would reward reproducing the average curve shape, which is trivial. This baseline
asks whether configurations can be told apart at each position, and it is the baseline every
reference number here uses.

**Every trajectory score is reported twice**, inside the plasma and in the drift after it, split
at each configuration's own plasma end, `L_inj + 3.2` mm — the shipped density profile's own
support, not a flat plane. Because that boundary moves with `L_inj`, a position is not wholly one
zone or the other: 44 positions are scored in-plasma and 77 in the drift, and they overlap. Most
of the stored range is drift, where the beam is coasting, so **a single pooled number is mostly a
score on the easy part.** A position carrying fewer than 30 configurations of a zone is dropped
from that zone's sum, and the scorer reports how many it scored and dropped.

Two of the six quantities, `emit_um` and `div_mrad`, are scored **in-plasma only**. Past the
plasma end `div_mrad` is a waist proxy that falls as β grows, and `emit_um` falls as halo
particles leave the diagnostic; both measure a definition there rather than the beam.

## Ontologies / schema

`ontologies/pack_schema.json` lists every file in the pack, every column, its dtype, its unit and
one line on what it means — generated from the pack itself, not written by hand.
