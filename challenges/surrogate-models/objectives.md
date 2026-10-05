# Challenge Objectives — Surrogate Models (PALLAS)

Short version. The physics, the traps and the full reference numbers are in
[`objectives_detailed.md`](objectives_detailed.md); the files and scoring rules are in
[`data/README.md`](data/README.md), the background in [`scientific_case.md`](scientific_case.md).

Each objective gives a **signature** (the literal pack columns in → out), a **score**, and a
**bar** to beat. Score yourself with `pallas_score.py`; a submission is keyed by `config`, plus
`z_mm` or `plane_mm` where the task has a position.

Three rules for every trajectory score:
- **Only the in-plasma score is ranked**, split at each configuration's plasma end
  `L_inj + 3.2` mm. After that the beam drifts through vacuum: free-space transport, not the
  plasma physics the simulation is run for. The scorer reports the drift too, for information
  only, and never pools the two.
- **R² is taken per position, then summed**, so reproducing the average curve earns nothing.
- **`emit_um` and `div_mrad` are scored inside the plasma only**: past the plasma end the
  diagnostic stops describing the beam.

## Hand-in and scoring

_TBD — announced by the organisers._

## How the week runs

Everyone goes Easy → Medium → Hard, so teams are compared on the same tasks.

| Day | Objectives |
|---|---|
| Monday, after the lectures | the first Easy objective (the kickoff notebook) |
| Tuesday | the other three Easy objectives and the first Medium one |
| Wednesday morning | the second Medium objective |
| Thursday | the third Medium objective, then start on the Hard ones |
| Friday | the two Hard objectives and the final ranking |

The **optional objectives** sit outside the sequence: take one once you are done, or when stuck.

## 🟢 Easy

- **Inverse A — read the machine backwards.** Run `kickoff_notebook.ipynb` to the end and write
  `submission_inverse.csv`. Injected Campaign A rows (≥ 3 pC).
  **Signature:** `E_med_MeV, dE_mad, q_end, i_peak, n_emit_x, sigma_z, sigma_y, div_rms` + `x_of` →
  `p_1, a_0, c_N2`
  **Score:** R² per target and their mean. **Bar:** our reference ensemble, mean R² 0.9100
  (in review; measured with measurement noise, so your noise-free test is the easier one).
- **Direct B — read it forwards.** Write `submission_direct.csv` from the same notebook.
  **Signature:** `P_max, cN2_max, L_inj, dip_frac, x_of` → `E_med_MeV, dE_mad, q_end`
  **Score:** R² per quantity. **Bar:** the notebook's linear baseline, about 0.94 / 0.26 / 0.81;
  the room is in the energy spread. A stretch target, not the bar: our internal forward model
  reaches 0.974 / 0.804 / 0.942 (`direct_reference_emulator`), trained on our own split.
- **Warm-up, not scored — Campaign A forwards.** The one published reference in the pack
  (Kane et al., MLST 7, 030502 (2026)): R² 0.99 / 0.96 / 0.99 / 0.95 on energy, spread, charge,
  vertical emittance with a neural network. Kane's spread is in percent, `dE_mad` is a fraction.
  **Signature:** `p_1, a_0, c_N2, x_of` → `E_med_MeV, dE_mad, q_end`
- **Trajectory — the bunch along the accelerator.** Beat a linear model you fit yourself on
  your own held-out split; the organisers then score you on `test_trajectory.parquet`.
  **Signature:** `P_max, cN2_max, L_inj, dip_frac, x_of` + `z_mm` →
  `E_MeV, q_pC, dE_pct, emit_um, div_mrad, sigz_um`
  **Score:** R² per quantity, inside the plasma (the drift is reported, not ranked).
  The hidden test leans to high density: 136 of its 357 configurations (38 %) lie above the
  out-of-distribution cut, against 20 % of the public ones, so a random holdout of your own
  reads optimistic.

## 🟡 Medium

- **Beat the internal reference on the trajectory** (same signature and test as above).
  **Bar (in-plasma, hidden test, 357 configurations):** energy 0.9633, charge 0.9331, energy
  spread 0.8092, emittance 0.8610, divergence 0.9469, bunch length 0.8623, mean 0.8960
  (`emulator_inplasma` in `pack_reference.json`). You can't reproduce it yourself; your own
  validation split is only a guide. Don't compare with its 0.9069 on the 714-configuration fold.
- **Inverse B — Campaign B backwards.** Train on injected `campaign_B.parquet` rows, write
  `submission_inverse_B.csv` for `test_inverse_B.parquet` (357 configurations).
  **Signature:** `E_med_MeV, dE_mad, q_end, i_peak, n_emit_x, sigma_z, sigma_y, div_rms` + `x_of` →
  `P_max, cN2_max, L_inj, dip_frac`
  **Score:** R² per target and mean. **Bar:** a linear model (StandardScaler + LinearRegression,
  `q_end`, `i_peak`, `n_emit_x`, `sigma_y` logged), mean R² 0.6480 — `P_max` 0.8665, `cN2_max`
  0.8400, `L_inj` 0.7987, `dip_frac` 0.0869 (`inverse_B_linear_baseline` in `pack_reference.json`).
  Most of the room is in `dip_frac`.
- **Moments at the planes.** Train on `campaign_B_moments.parquet`, `group == "all"` rows only,
  every plane; score on `test_moments_planes.parquet`. Each width is √ of its covariance
  diagonal (`sigma_y` = √`cov_11`); off-diagonals are optional, not scored.
  **Signature:** `P_max, cN2_max, L_inj, dip_frac, x_of` + `plane_mm` →
  `mean_zeta, mean_y, mean_z, mean_ux, mean_uy, mean_uz`,
  `sigma_zeta, sigma_y, sigma_z, sigma_ux, sigma_uy, sigma_uz`, `q_tot_pC`
  **Score:** R² per column, over every test plane, 5.0 mm included. The hidden answers have a
  bunch at every test plane; drop the empty rows of the training table (`data/README.md`).
  Name the centroids `mean_zeta, mean_y, mean_z` — the table's own names end in `_um`.

## 🔴 Hard

- **Out of distribution in density.** Train only on the 2,277 configurations in `ood_split.train`
  of `pack_reference.json` (`P_max` < 6,528.6 Pa) — you must drop the 578 public ones above the
  cut yourself. One model predicts `test_trajectory.parquet`; the organisers score it on the 221
  hidden configurations inside the range and the 136 above it. `test_trajectory_ood.parquet` is
  not submitted on its own: it holds the rows of `test_trajectory` for those 136, so you can see
  which ones lie above the cut.
  **Signature:** unchanged from the trajectory task.
  **Deliverable: the gap between the two scores, honestly measured.** This is extrapolation:
  every out-of-range configuration lies above the highest trained pressure (6,532.5–6,999.1 Pa
  vs 4,001.5–6,528.4 Pa). A model tuned until the gap looks small has usually learned the cut.
- **Particles, not moments.** Generate a cloud of electrons whose full 6D shape (tails, skew,
  correlations) matches the simulation's. Training: the particle download, 5,282 configurations,
  up to 2,000 charge-weighted electrons each — `planes20/` (5.3 GB) first, the full path (54.2 GB)
  only once a model works.
  For scale: our own conditional-flow generator (width 512, depth 4) trains to its final state
  in about 100 minutes on one RTX A6000 (500 epochs), so a first model fits in an afternoon.
  Submit one `Config_<id>.npz` per configuration of
  `test_moments_planes.parquet`, holding `plane_mm` and `state` (planes × particles × 6).
  **Signature:** `P_max, cN2_max, L_inj, dip_frac, x_of` + `plane_mm` →
  (`zeta_um, y_um, z_um, ux, uy, uz`), one row per electron
  **Score:** `score_particles` — sliced Wasserstein distance in units of the cloud's own sampling
  noise; perfect ≈ 1, lower is better, median over rows. Planes with < 50 true electrons are
  skipped. **Bars** (100 test configurations): a Gaussian with the *true* mean and covariance 2.13;
  copying the nearest training simulation 4.26. The cost is reading data, not training.

## ⚪ Optional — outside the sequence

- **Out of distribution in position.** Train on the eleven planes of `plane_holdout` in
  `pack_reference.json`, predict the other nine; scored on `test_moments_planes.parquet`. This is
  interpolation (all nine lie inside 2.0–7.0 mm); 3.5 mm and 5.0 mm sit in the widest gaps.
  **Signature:** as the Medium moments objective, at planes never seen in training.
- **Data-budget curve.** Train the same model on 100, 300, 1,000 and all 2,855 configurations,
  submit all four predictions of `test_trajectory.parquet`, report where mean R² first reaches
  0.90. Choosing *which* configurations is part of the task: say how, compare with random draws.
- **Inverse as a distribution.** A 90 % interval for each of `p_1, a_0, c_N2`, scored by coverage
  and mean width. Degeneracy caps even a perfect point predictor at R² 0.9909.
  **Signature:** the Easy inverse inputs → a lower and upper bound per target.
- **Show the degeneracy.** Find Campaign A settings pairs giving indistinguishable bunches, state
  your tolerance, name the knob that can't be recovered. Report only.
- **The whole energy spectrum.** Campaign B injected rows, settings → charge per bin on the
  200-bin grid (`common_energy_grid_MeV.npy`). Your own fold: earth-mover distance
  (`scipy.stats.wasserstein_distance(grid, grid, true, yours)`, in MeV) and total charge vs `q_pC`.
  `spec_common` is already pC per bin and sums to `q_pC`.
- **The density profile as a curve.** Same outputs from the four settings and from the curve
  `n_e_p` (+ `x_of` in both), on your own fold. The curve is built from the four numbers, so the
  question is whether reading the curve loses nothing.
- **Find the physics that helps.** Raise the in-plasma energy R² with a physics ingredient.
  Integrating a fixed fraction of the wave-breaking field failed: R² 0.26 alone, −0.006 to +0.001
  as an extra input, because it ignores beam loading (residual vs charge −0.74).
  **Bar:** 0.820 ± 0.016 at 100 and 0.908 ± 0.008 at 300 configurations (gradient boosting, ten
  random subsets, hidden test). Say *why* it works; a term that needs the charge must predict it.
  **Signature:** `P_max, cN2_max, L_inj, dip_frac, x_of` + `z_mm` (+ what you derive) → `E_MeV`
- **Fast and honest.** Inference time per sample with its hardware, and a refusal region written
  down before scoring. Report only.
- **The wake itself.** Predict three on-axis field maps per configuration — `ex` (the wake's
  electric field), `rho_e` (plasma electron density), `laser` (envelope) — each 157 snapshots
  along the accelerator × 320 points across the moving window. Training: `level6/`, 5,286
  configurations. Submit `field_axes.npz` (`config`) + `field_ex.npy`, `field_rho_e.npy`,
  `field_laser.npy` for the 357 configurations of `test_direct.parquet`.
  **Signature:** `P_max, cN2_max, L_inj, dip_frac, x_of` → the maps `ex`, `rho_e`, `laser`
  **Score:** `score_field` — relative L2 per map after 1 µm averaging, median per field, lower
  is better. **Bar:** 20-component PCA with a linear map from the settings: `ex` 0.533,
  `rho_e` 0.839, `laser` 0.092 (`field_reference` in `pack_reference.json`).
  The error lives in the sharp features (the density spike, the injection jump); choose on
  purpose which axis your model treats as time.
