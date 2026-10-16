# CLARA starting beam: working log

Team Overfit & Proud, ARTIFACT hackathon optimisation challenge. Written 14 October 2026; covers the work of 13 October.

The question: which beam should the benchmark start from, just after CLA-S02-DIA-SCR-01 at 35.8 MeV, and how does the beam the benchmark already uses compare with the one reconstructed from machine data?

## Names

### Beams already in the repository

Amelia's fork, `amylizzle/artifact-hackathon-2026`, up to commit 74eee85 (RL framework, fetched but not yet pulled into our clone).

| Name | Defined in | Location, energy | What it is | Key numbers |
|---|---|---|---|---|
| `bench-nominal` | `score.sample_beam(severity=0)` | just after S02-SCR-01, 35.8 MeV | the benchmark's starting beam, full 6D correlated covariance, no jitter | β 0.85 / 32.8 m, α −7.75 / 26.0, εn 2.33 / 2.31 µm, σδ 0.93 %, σz 0.69 mm |
| `bench-case` | `score.make_cases` → `sample_beam(severity=1)` | same | `bench-nominal` plus per-case jitter: centroid ±50 µm and ±50 µrad, energy ±0.1 %, each rms ×/÷1.3, correlations fixed | used for the 100 scored cases; notebook cell 21 uses case seed 1 |
| `nb-gun-example` | kickoff notebook, cell 7 | after S01-SIM-APER-01 (gun), 4.2 MeV | toy beam for the notebook's `plot_overview` | σ 10 µm, σ′ 1 µrad, β 10 m, α 0, εn 0.08 nm |
| `nb-s04-easy` | kickoff notebook, cell 10 | S04 entrance, 193.2 MeV | starting beams of the Easy-tier BO example | εn 1 µm, β 8 to 16 m, α ±1, offsets ±0.3 mm and ±0.1 mrad, σδ 0.1 % |
| `rl-config` | `RL_framework/rl_framework/config.py` | start of `FEBE_full.json` (the gun), but 250 MeV | Twiss beam of Amelia's RL environment (target S04-SCR-03) | β 14.2 / 925 m, α −0.95 / 72.0, εn about 1.6 / 2.1 µm, σδ 1 % |
| `s07-gpsr` | `data/clara/lattice_data/CLA-S07-SIM-APER-01.hdf` | S07-SIM-APER-01, 247.6 MeV in the model | 4D GPSR reconstruction from the 23 March 2026 scan of S07-QUAD-05 on S07-CAM-04; transverse only, 10⁶ particles | β 79.1 / 47.0 m, α −3.69 / −3.83, εn 7.92 / 2.89 µm, x–y correlation −0.48 |

`s07-gpsr` is the only beam here that comes from measurement.

### Lattice settings

| Name | File | What it is |
|---|---|---|
| `file` | `CLARA_cheetah.json` | the strengths stored in the lattice file |
| `ref` | `reference_tune.json` | archiver capture of 17 April 2026, correctors at zero (Amelia set the sextupoles to zero on 13 October) |
| `gpsr` | `CLARA_cheetah_S07_GPSR.json`, `s07_gpsr_tune.json` | the magnet currents of the 23 March scan, built by us (see below) |
| `gpsr35.8` | `gpsr` with the S02 quads × 32.8 / 35.8 | sensitivity check of the S02 conversion energy |

### Beams we derived

`back[settings, order, longitudinal]` is `s07-gpsr` tracked back to the benchmark start, so every one of them compares directly with `bench-nominal`.

- `order`: `lin` is first order everywhere, as the lattice file sets it; `2nd` puts drifts, quadrupoles and dipoles at second order. Cavities are always Cheetah's first-order map.
- `longitudinal`: `δ0` keeps the file's τ = δ = 0; `δmodel` attaches the (τ, δ) of `bench-nominal` tracked forward to S07 (σδ 2.1 %, σz 0.68 mm, chirp 31 m⁻¹), paired, with no correlation to the transverse coordinates.

`back_xsuite[settings, exact, longitudinal]` is the same construction done in xsuite by the parallel xsuite session (`work/xsuite/`), described below.

## Method

Reading `s07-gpsr`: its longitudinal momentum is a single 0.86 MeV/c, which cannot be the momentum at S07. We take x′ = px/pz with the file's own pz and the energy from the model (247.6 MeV). This assumes the reconstruction got positions and angles right and only the pz value is a placeholder.

Linear backtracking inverts the 4×4 map from the benchmark start to S07-SIM-APER-01, measured with test particles. Tracking the result forward with Cheetah reproduces the S07 covariance to 5 × 10⁻⁷.

Second-order backtracking cannot invert a matrix. Each particle starts from the linear inverse and is corrected with the linear inverse of its forward-tracking residual, in float64, until every particle is below 10⁻⁶ of the rms (up to 100 passes for the energy tail). Cheetah 0.8.4 accepts only `linear`, `second_order` and `drift_kick_drift` as tracking methods; it silently keeps the old method for any other name, so the script checks the switch.

The `δmodel` longitudinal phase space is built in two steps. First, `bench-nominal` (200,000 particles) is tracked from the S02 start to S07-SIM-APER-01 through the same settings, at the same order, and only its (τ, δ) columns are kept. Second, each of the 10⁶ reconstructed particles draws one (τ, δ) pair at random, with replacement (seed 1). The pairs stay intact, so the chirp survives; the pairing is random, so the longitudinal coordinates carry no information about x, x′, y or y′. The S02 to S07 line has no dispersion, so the magnet settings barely matter here: σδ at S07 is 2.108 % for `gpsr`, `ref` and `file` alike. The RF does matter, and it is the lattice file's RF, not the scan's.

xsuite backtrack (`work/xsuite/backtrack_s07_xsuite.py`, run by the xsuite session on the validated port, which reproduces the Cheetah 4×4 map to S07 within 5 × 10⁻⁵ and the energy exactly). The construction is the same, with three differences: `bench-nominal` is tracked forward in xsuite, whose cavities apply the Rosenzweig–Serafini matrix at each particle's own energy (σδ 2.106 % at S07, ζ = −τ); the backtracking uses xsuite's native exact reverse tracking (`line.track(..., backtrack=True)`) instead of an iterative inversion; and it uses a 200,000-particle random subsample of the reconstruction (seed 0), paired one to one with 200,000 freshly sampled nominal particles. Without energy spread, xsuite and Cheetah agree particle by particle within 0.8 % of the rms.

`gpsr` lattice (`make_gpsr_lattice.py`): the scan has 645 captures; only S07-QUAD-05 moves (−100 to +100 A). Each magnet takes its median READI current, converted with `utils.update_cheetah_element`. As in `reference_tune.json`, correctors and sextupoles are set to zero. The RF readbacks are not applied: they are machine phases (L02 −86°, L03 −107°, L04 +125°), not Cheetah's off-crest angles. During the scan only the S02 triplet was on between the start and S07 (k1 +4.09, −1.03, −4.80 m⁻² for QUAD-03, 04, 05); the S03 to S07 quads sat at about −0.02 m⁻², and the FEA dipoles were off.

## Results at the benchmark start

| Beam | βx / βy (m) | αx / αy | εn x / y (µm) | B_mag vs `bench-nominal`, x / y |
|---|---|---|---|---|
| `bench-nominal` | 0.85 / 32.8 | −7.75 / 26.0 | 2.33 / 2.31 | 1 / 1 |
| `back[file, lin, δ0]` | 63.2 / 25.6 | 163.5 / 26.4 | 7.92 / 2.89 | 3723 / 25 |
| `back[file, 2nd, δmodel]` | 12.9 / 22.0 | 33.3 / 22.6 | 39.2 / 3.43 | 759 / 21 |
| `back[ref, lin, δ0]` | 9.91 / 53.7 | −0.49 / −26.8 | 7.92 / 2.89 | 353 / 1472 |
| `back[ref, 2nd, δmodel]` | 9.83 / 52.3 | −0.49 / −26.1 | 7.97 / 2.97 | 350 / 1435 |
| `back[gpsr, lin, δ0]` | 46.2 / 10.9 | −7.86 / 14.1 | 7.92 / 2.89 | 1603 / 47 |
| `back[gpsr, 2nd, δmodel]` | 45.9 / 9.70 | −7.83 / 12.5 | 7.98 / 3.29 | 1593 / 41 |
| `back[gpsr35.8, lin, δ0]` | 44.9 / 6.10 | −6.56 / 9.54 | 7.92 / 2.89 | 1566 / 62 |
| `back_xsuite[gpsr, exact, δ0]` | 46.2 / 10.9 | −7.87 / 14.1 | 7.92 / 2.89 | 1605 / 47 |
| `back_xsuite[gpsr, exact, δmodel]` | 45.3 / 8.96 | −7.75 / 11.5 | 8.09 / 3.59 | 1574 / 38 |

`[·, 2nd, δ0]` matches `[·, lin, δ0]` within 0.2 % in every case, because every chromatic term needs δ.

The two codes tell the same story for `gpsr`. With `δ0` they coincide. With `δmodel`, xsuite finds a larger chromatic effect in y than Cheetah (εn,y 3.59 against 3.29 µm, βy 8.96 against 9.70 m), which is the share Cheetah's achromatic cavities miss (finding 3 below). The x plane agrees within 1.3 % in β and 1.4 % in εn.

| `gpsr`, `δmodel`, at the start | σx (mm) | σx′ (mrad) | σy (mm) | σy′ (mrad) | x–y corr. | σδ |
|---|---|---|---|---|---|---|
| Cheetah, `back[gpsr, 2nd, δmodel]` | 2.29 | 0.393 | 0.674 | 0.874 | +0.30 | 0.93 % |
| xsuite, `back_xsuite[gpsr, exact, δmodel]` | 2.29 | 0.394 | 0.678 | 0.875 | +0.30 | 0.94 % |

The distributions, at the start and on common axes. Rows: `bench-nominal`, then the Cheetah and xsuite backtracks; the last column is the longitudinal phase space, which returns to `bench-nominal`'s by construction (σz 0.688 mm in all three, σδ 0.931 to 0.936 %).

![Phase space at the start: bench-nominal, Cheetah backtrack, xsuite backtrack](s07_backtrack/figures/compare_codes_phase_space.png)

![Projections at the start, log density](s07_backtrack/figures/compare_codes_projections.png)

In the projections the two backtracks lie on top of each other down to 10⁻⁴ of the peak density, tails included: the x tails out to ±9 mm and the shoulder near −3 mrad in x′, which is the halo already present in the S07 reconstruction.

Against `bench-nominal`, `back[gpsr, 2nd, δmodel]` is 14 times wider in x (σx 2.29 against 0.17 mm) and 4 times narrower in angle (σx′ 0.39 against 1.55 mrad). Its emittances are 3.4 times larger in x and 1.4 times in y, and it carries an x–y correlation of +0.30. The y plane is the closer one: σy′ 0.87 against 0.83 mrad, σy 0.67 against 1.04 mm. Figures: `work/s07_backtrack/figures/compare_start_phase_space.png` and `compare_start_projections.png`.

## Consistency check: forward to S07 in xsuite (14 October)

Decision of 14 October: from now on we track with xsuite only, and the nominal starting beam is `back_xsuite[gpsr, exact, δmodel]` (`work/xsuite/backtrack/xsuite_gpsr_delta_model.npy`).

`work/s07_backtrack/forward_consistency_xsuite.py` tracks that 6D beam forward in xsuite, through the `gpsr` lattice, to S07-SIM-APER-01 and compares it with the 200,000 reconstructed particles it came from. xsuite's backtracking is the exact reverse of its forward tracking, so the 6D beam as it is must return the reconstruction exactly; that is the closure test. To see what the longitudinal plane does, two altered versions are tracked as well:

- no longitudinal plane: the same starting transverse coordinates with τ = δ = 0;
- longitudinal attached at S02: the transverse-only backtrack `back_xsuite[gpsr, exact, δ0]` with `bench-nominal`'s own (τ, δ) paired at random at the start, which is what one would build without pairing at S07.

| At S07-SIM-APER-01 | βx / βy (m) | αx / αy | εn x / y (µm) | B_mag vs `s07-gpsr`, x / y | 4D emittance / `s07-gpsr` | rms displacement x, x′, y, y′ (in σ) |
|---|---|---|---|---|---|---|
| `s07-gpsr` | 79.1 / 47.0 | −3.70 / −3.83 | 7.92 / 2.90 | 1 / 1 | 1 | |
| closure | 79.1 / 47.0 | −3.70 / −3.83 | 7.92 / 2.90 | 1 / 1 | 1 | 0, largest 5 × 10⁻¹³ |
| no longitudinal plane | 76.4 / 37.4 | −3.56 / −3.01 | 8.10 / 3.59 | 1.001 / 1.027 | 1.31 | 0.12, 0.06, 0.18, 0.006 |
| longitudinal attached at S02 | 79.6 / 40.3 | −3.69 / −3.13 | 8.11 / 3.59 | 1.000 / 1.026 | 1.31 | 0.12, 0.06, 0.18, 0.007 |

![Forward to S07: reconstruction, closure and the two altered versions](s07_backtrack/figures/forward_consistency_phase_space.png)

![Projections at S07](s07_backtrack/figures/forward_consistency_projections.png)

The two-way view of the closure alone, the measured 4D against the 4D part of the nominal 6D start tracked to S07 (`plot_s07_data_vs_6d.py`):

![Measured 4D at S07 against the nominal 6D start tracked to S07](s07_backtrack/figures/s07_data_vs_6d_start.png)

The construction is self-consistent: the closure is exact.

The longitudinal plane matters at the 20 to 25 % level in y and very little in x. With the model's energy spread (0.93 % at S02, 2.1 % at S07), the stretch from S02 to S07 adds 24 % to the vertical emittance (2.90 → 3.59 µm) and 2 % to the horizontal one, and moves βy from 47 to 37 to 40 m. The Twiss mismatch stays small (B_mag y 1.03); the growth is in the emittance, 31 % in 4D. Particles move by 0.18 σ rms in y. The y′ distribution barely changes; the reconstruction's sharp edges in x (±3.5 mm) and y (±1.5 mm) get smeared.

All the forward tracking is 6D, and both versions that have a longitudinal plane start at S02 with the nominal one. In the 6D start, σz is 0.688 mm (`bench-nominal`: 0.688), σδ 0.936 % (0.931) and the τ–δ correlation 0.988 (0.992). Row by row, its (τ, δ) also stay close to the `bench-nominal` particles they were paired with: rms differences of 2 % of σ in τ and 8 % in δ. The residual comes from path-length coupling, since each longitudinal pair travels back with a different transverse partner than it travelled forward with.

What differs between the closure and "longitudinal attached at S02" is how the transverse distribution depends on δ. Cut the 6D start into ten energy slices and every slice has εn,y between 2.87 and 3.02 µm and εn,x between 7.6 and 8.4 µm, the measured values at S07, while the projected values are 3.59 and 8.09 µm. Each energy slice carries its own orientation in phase space, which the chromatic transport to S07 brings back into line. When the longitudinal plane is attached at S02 the slices all share one transverse distribution, and the same transport spreads them apart instead.

`back_xsuite[gpsr, exact, δmodel]` carries that growth in reverse. At S02 it has εn,y = 3.59 µm with δ-dependent correlations that the chromatic transport to S07 undoes, so the measured 2.90 µm comes back. A beam whose energy deviations are uncorrelated with its transverse coordinates at S02, as real injector beams usually are, cannot do this: started at 2.9 µm with 0.93 % uncorrelated spread, it reaches S07 at 3.6 µm, 24 % above the measurement. So either the beam at S02 carried those correlations, or the energy spread during the scan was well below the model's. The real energy spread at S07 is the number that decides between them, which makes question 4 for Amelia the most important one.

## What we learned about the model

1. The lattice file tracks every drift, quadrupole and dipole at first order, so most chromatic effects are missing. The two exceptions are the switched-off TDC (Cheetah only offers `drift_kick_drift` for it) and the sextupoles (`second_order`). With the whole line at second order, `bench-nominal` reaches S07 with εn 12.7 / 15.0 µm through `ref` and 95.6 / 3.3 µm through `file`, against 2.33 / 2.31 µm at first order.
2. Correction to something said on 13 October: the tenfold jump of the vertical emittance at `cla_s07_dia_tdc_01` is not a Cheetah bug. An exact drift moves each particle by −L·py·δ/(1+δ) more than the linear map, and the beam arrives there with αy around 3600, so that term multiplies its emittance by ten. A drift tracked with `drift_kick_drift` does the same. The earlier "bug" verdict came from a control test that asked for the non-existent method `bmadx` and silently stayed linear.
3. Cheetah's `Cavity` applies one transfer matrix, computed at the reference energy, to every particle, so it has no chromatic focusing: L02 gives R11 0.3298 and R12 2.0581 at δ of −1, 0 and +1 %. xsuite evaluates the Rosenzweig–Serafini matrix at each particle's energy and finds about 45 % more chromatic growth to S07 (18.5 / 23.2 µm against Cheetah's 12.6 / 15.0 through `ref`). With the cavities replaced by drifts the two codes agree within 2 % (`work/xsuite/chromatic_comparison.py`). Our Cheetah backtrack therefore understates chromatic effects by roughly 15 %.
4. Upstream of S07 the reconstructed bunch is far less sensitive to energy spread than `bench-nominal`: through `ref` its start Twiss changes by about 3 % at second order with `δmodel`, against a fivefold emittance growth for `bench-nominal`. Through `gpsr` the change is 14 % in εn,y.
5. `utils.beam_energy` converts the S02 currents at 32.8 MeV, while the model has 35.8 MeV there. Rescaling moves βy from 10.9 to 6.1 m and B_mag y from 47 to 62; x hardly moves.
6. During the scan the 4HC linearizer was off (power about 3 × 10⁻⁵), and the RF powers convert to 69, 69 and 49 MV for L02, L03 and L04, against 87.5, 78.8 and 93.4 MV in the lattice file. `δmodel` (2.1 % at S07) comes from the lattice file's RF, so it probably does not describe the scan.
7. With `bench-nominal` and `ref` as of 13 October (sextupoles still at ±226 in the organisers' clone), only 5.7 % of the beam lands on the FEH-SCR-01 chip, below the scorer's 10 % threshold. Not re-checked since Amelia zeroed the sextupoles.

## Best guess for the starting beam

The benchmark has to start at S02, after S02-SCR-01. Our best guess there is `back_xsuite[gpsr, exact, δmodel]`, with `back[gpsr, 2nd, δmodel]` from Cheetah as the cross-check, used as a particle distribution rather than reduced to Twiss parameters, because its tails and x–y coupling matter. It is the only candidate built from a measurement, and `gpsr` is the magnet state that was actually on the machine when that measurement was taken; the `file` and `ref` settings belong to other days. The xsuite version is the central value because its cavities are chromatic, as real cavities are; Cheetah's are not.

| Plane | Central value (xsuite) | Range across our variants | Confidence |
|---|---|---|---|
| x | β 45.3 m, α −7.75, εn 8.1 µm | β 44.9 to 46.2 m, α −6.6 to −7.9, εn 7.9 to 8.1 µm | good: insensitive to energy spread, code and S02 calibration |
| y | β 8.96 m, α 11.5, εn 3.6 µm | β 6.1 to 10.9 m, α 9.5 to 14.1, εn 2.9 to 3.6 µm | moderate: set by the S02 conversion energy and the energy spread |
| x–y | correlation +0.30 | +0.29 to +0.30 | as good as the reconstruction |
| longitudinal | `bench-nominal`'s: σδ 0.93 %, σz 0.69 mm | unknown | weak: model value, and the scan ran with different RF |

The ranges come from single-factor changes, not a joint scan, so they are indicative.

For anything downstream of S07, start from `s07-gpsr` itself at S07-SIM-APER-01 and skip the backtracking. That removes the S02 calibration, the cavity model and the inversion from the chain, and leaves only the missing longitudinal phase space.

`bench-nominal` and `bench-case` stay what the leaderboard scores, but they do not describe the machine of 23 March. `nb-gun-example`, `nb-s04-easy` and `rl-config` are synthetic and should not be used as machine beams.

## Open questions for Amelia

1. pz in the HDF file is a single 0.86 MeV/c. Are positions and angles correct, and which energy and lattice did the reconstruction assume?
2. Which conversion energy is right for the S02 quads, 32.8 MeV (`utils.beam_energy`) or the 35.8 MeV of `sample_beam`?
3. Can we have the RF crest phases, to turn the getPhase readbacks into off-crest angles, and the beam energy at S07 during the scan?
4. Is there a longitudinal measurement for the scan: bunch length from the S07 TDC, energy spread from a dispersive screen?
5. Is the x–y correlation of the reconstruction (−0.48 at S07) real?
6. Where do the `BASE_*` numbers of `sample_beam` come from? Their σx is 14 times smaller than what the reconstruction implies.
7. The scorer tracks magnets at first order and Cheetah's cavities are achromatic. Should the benchmark move to second order?
8. `rl-config` is a 250 MeV beam tracked from the gun of `FEBE_full.json`. Intended?

## Files

| Path | Content |
|---|---|
| `work/s07_backtrack/make_gpsr_lattice.py` | builds `CLARA_cheetah_S07_GPSR.json` and `s07_gpsr_tune.json` in the fork's `lattice_data` (not committed) |
| `work/s07_backtrack/backtrack_s07_bunch.py` | all `back[…]` variants; `figures/backtracked_second_order.png`, `figures/summary_second_order.json` |
| `work/s07_backtrack/compare_with_benchmark_beam.py` | `bench-nominal` against `back[gpsr, 2nd, δmodel]`; cached particles in `figures/backtracked_gpsr_second_order.npy` |
| `work/s07_backtrack/forward_consistency_xsuite.py` | xsuite forward tracking of the starting beam to S07 (closure and the two altered versions): `figures/forward_consistency_phase_space.png`, `figures/forward_consistency_projections.png`, `figures/forward_consistency_xsuite.json` |
| `work/s07_backtrack/compare_codes_at_start.py` | `bench-nominal`, Cheetah and xsuite backtracks together: `figures/compare_codes_phase_space.png`, `figures/compare_codes_projections.png` |
| `work/xsuite/backtrack/xsuite_gpsr_delta_model.npy` | `back_xsuite[gpsr, exact, δmodel]`, 200,000 particles, Cheetah convention (x, px, y, py, τ, δ) at the start; the same run wrote `xsuite_reference_delta_model.npy` and `xsuite_file_delta_model.npy` |
| `work/optics/compute_optics.py` | shared helpers (also imported by the xsuite session) |
| `work/xsuite/` | the xsuite port (`cheetah_to_xsuite.py`, `validate_port.py`, `rf_focusing.py`), its backtracks (`backtrack_s07_xsuite.py`, `backtrack/summary.json`) and the code comparison (`chromatic_comparison.py`) |
| `environments/optimisation` | Cheetah 0.8.4, the challenge's `requirements.txt`, h5py |

Optics page from 13 October: https://claude.ai/artifact/PwfevaSpBUTvAMKcuKNCpp. It does not include the S07 work yet.

Pull requests to Amelia's fork are on hold until the longitudinal phase space is sorted out.
