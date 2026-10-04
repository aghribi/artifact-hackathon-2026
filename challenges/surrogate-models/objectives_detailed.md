# Challenge Objectives — Surrogate Models (PALLAS) — detailed version

The long form of [`objectives.md`](objectives.md): the same objectives, with the physics, the traps and every reference number spelled out.

Objectives only — see [`scientific_case.md`](scientific_case.md) for background and motivation,
and [`data/README.md`](data/README.md) for the files, splits and scoring rules named below.

**Every objective that asks for a prediction carries a signature line**: the columns that go in,
the position column if the task has one, then an arrow, then the columns that come out. The names
are the literal column names in the pack, so a signature is enough to start writing code —
`data/README.md` gives the unit of each one.

Score yourself at any point with `pallas_score.py`. A submission is keyed by `config`, plus
`z_mm` or `plane_mm` where the task has a position.

Two rules run through every scored objective here, because a surrogate can look good by
accident under either one.

- **Only the in-plasma score is ranked, and it is never pooled with the drift.** Inside the
  plasma the bunch is being accelerated — the physics the simulation is run for. After the plasma
  ends, at each configuration's own `L_inj + 3.2` mm, the beam drifts through vacuum: free-space
  transport, which the scorer reports for information but which does not count. Every ranked
  trajectory number in these objectives — the bar, the out-of-distribution gap, the data-budget
  curve — is the in-plasma one.
- **A position-dependent score is measured against the per-position mean.** R² is computed at
  each position along the accelerator and then summed over positions, not over every cell at
  once. Most of the variation along a trajectory is the trend with distance, so a single global
  denominator would reward reproducing the average curve and nothing else.

One warning about the data rather than a rule of the score: **emittance (`emit_um`) and
divergence (`div_mrad`) are scored inside the plasma only.** Past the plasma end the simulation's
diagnostic stops describing the beam — the numbers keep changing, but not because the bunch does.
A surrogate trained on those values past the plasma end learns the diagnostic, not the beam.

## How the week runs

Every team works through the objectives in the same order — Easy, then Medium, then Hard — so
the two teams of a group are compared on the same tasks. The suggested pace:

| Day | Objectives |
|---|---|
| Monday, after the lectures | the first Easy objective (the kickoff notebook) |
| Tuesday | the other three Easy objectives and the first Medium one |
| Wednesday morning | the second Medium objective |
| Thursday | the third Medium objective, then start on the Hard ones |
| Friday | the two Hard objectives and the final ranking |

The **optional objectives** at the end sit outside that sequence: pick any of them once the
sequence is done, or instead of a step you are stuck on.

## 🟢 Easy

- Run `kickoff_notebook.ipynb` from start to finish and write a valid `submission_inverse.csv`:
  in go the eight beam columns of an injected Campaign A row (a bunch of 3 pC or more) and the
  focal position `x_of`, which the operator sets and so knows; out come the three settings
  `p_1`, `a_0` and `c_N2`, scored by R² on each and their mean. These are the reference
  ensemble's own inputs and targets (ours, in review, mean R² 0.9100), so the comparison is
  direct — except that the reference was measured with realistic measurement noise on the beam
  columns and the test file is noise-free, which makes your task the easier of the two.
  The reference number was trained on the uniform scan of Campaign A and tested on the random
  scan of Campaign A as it stood in that study, 9,846 simulations that produced a bunch. The
  pack's copy of that scan is smaller after the cleaning in `DATA_CARD.md`; the hidden test set
  is 1,915 of its simulations.
  **Signature:** `E_med_MeV, dE_mad, q_end, i_peak, n_emit_x, sigma_z, sigma_y, div_rms` + `x_of` →
  `p_1, a_0, c_N2`
- Write a valid `submission_direct.csv` from the same notebook, the machine read forwards: in go
  Campaign B's five density-profile settings, out comes the bunch at the end of the accelerator
  (`E_med_MeV`, `dE_mad`, `q_end`), scored by R² per quantity. The notebook's linear baseline
  reaches about 0.94, 0.26 and 0.81 — that is the reference to beat, and the energy spread is
  where the room is. Our internal forward model's 0.974 / 0.804 / 0.942
  (`direct_reference_emulator`) is a stretch target, not the bar: it was trained on our own split.
  **Signature:** `P_max, cN2_max, L_inj, dip_frac, x_of` → `E_med_MeV, dE_mad, q_end`
- Know the warm-up that is *not* scored: the same forward question on Campaign A's four settings.
  This is the one published reference in the pack — Kane et al., MLST 7, 030502 (2026), trained
  on Campaign A's uniform scan and tested on the 9,846 simulations of its random scan that produced
  a bunch in that study (the pack's copy is smaller after cleaning), report
  R² on median energy, energy spread, charge and vertical emittance of 0.99 / 0.96 / 0.99 / 0.95
  with a neural network, 0.99 / 0.95 / 0.99 / 0.90 with a Gaussian process and 0.97 / 0.88 /
  0.96 / 0.78 with gradient boosting. Kane quotes the energy spread in percent; the pack's
  `dE_mad` is a fraction. Campaign B is a different set of simulations with different settings,
  so these numbers are not a bar for the scored forward task above.
  **Signature (not scored):** `p_1, a_0, c_N2, x_of` → `E_med_MeV, dE_mad, q_end`
- Fit the scored task and beat a linear baseline inside the plasma: in go the settings plus a
  position `z_mm` along the accelerator, out come the six trajectory quantities (energy, charge,
  energy spread, emittance, divergence, bunch length), scored by R² in each zone. The pack
  quotes no linear number for this task, so fit the linear model yourself: hold part of the
  public data out as your own validation set, fit both models on the rest, score both the same
  way, and beat it. Your submission is then scored by the organisers on the hidden test
  (`test_trajectory.parquet`).
  **Signature:** `P_max, cN2_max, L_inj, dip_frac, x_of` + `z_mm` →
  `E_MeV, q_pC, dE_pct, emit_um, div_mrad, sigz_um`

## 🟡 Medium

- Beat the internal reference model inside the plasma on the same six trajectory quantities. It
  reaches energy 0.9633, charge 0.9331, energy spread 0.8092, emittance 0.8610, divergence
  0.9469, bunch length 0.8623, mean 0.8960 (`emulator_inplasma` in the `scores` section of
  `pack_reference.json`). These were scored on the hidden test — the same 357 configurations your
  `test_trajectory.parquet` submission is scored on — so once the organisers score you, the two
  numbers are directly comparable. You cannot reproduce them yourself, because the answers are
  not in the pack; your own validation set gives you a rough guide only. (The same model scores
  0.9069 on the whole 714-configuration reference test fold, which contains fewer
  out-of-range configurations; do not compare against that number.)
  **Signature:** `P_max, cN2_max, L_inj, dip_frac, x_of` + `z_mm` →
  `E_MeV, q_pC, dE_pct, emit_um, div_mrad, sigz_um`
- Read Campaign B's machine backwards and write a valid `submission_inverse_B.csv`: in go the
  same eight beam columns as the Easy inverse task and the focal position `x_of`, which the
  operator knows; out come the four density-profile settings `P_max`, `cN2_max`, `L_inj` and
  `dip_frac`, scored by R² on each and their mean on the 357 configurations of
  `test_inverse_B.parquet`. Train on the injected rows of `campaign_B.parquet`. The bar is a
  linear model on the same inputs (StandardScaler + LinearRegression, `q_end`, `i_peak`,
  `n_emit_x` and `sigma_y` logged), scored by the organisers on the same 357: mean R² 0.6480 — `P_max` 0.8665, `cN2_max` 0.8400,
  `L_inj` 0.7987, `dip_frac` 0.0869 (`inverse_B_linear_baseline` in the `scores` section of
  `pack_reference.json`).
  Most of the room is in `dip_frac`, which the linear model barely sees.
  **Signature:** `E_med_MeV, dE_mad, q_end, i_peak, n_emit_x, sigma_z, sigma_y, div_rms` + `x_of` →
  `P_max, cN2_max, L_inj, dip_frac`
- Predict the bunch's centroid and size at the stored planes instead of the six summary
  quantities: in go the settings plus the plane, out come the six centroids, the six RMS widths
  and the charge, scored by R² per column. The training table `campaign_B_moments.parquet`
  ships the full 6×6 covariance as `cov_00 … cov_55`; each width is the square root of its
  diagonal entry (`sigma_y` = √`cov_11`). The table holds every plane twice, in the `group`
  column: `all` (every electron) and `cohort` (a selected sub-population); the hidden truth is
  `all`, so train on `all` only. Predicting the off-diagonal entries as well is
  optional and not scored. Train on every plane the table holds and score on
  `test_moments_planes.parquet`: the task here is to build an architecture that learns these
  quantities at all. The optional position objective reuses the same test file with a stricter
  training rule.
  **Signature:** `P_max, cN2_max, L_inj, dip_frac, x_of` + `plane_mm` →
  `mean_zeta, mean_y, mean_z, mean_ux, mean_uy, mean_uz`,
  `sigma_zeta, sigma_y, sigma_z, sigma_ux, sigma_uy, sigma_uz`, `q_tot_pC`

## 🔴 Hard

- Go out of distribution in density: train only on the 2,277 configurations listed in `train`
  of the `ood_split` section of `pack_reference.json` — every public configuration below the
  cut on `P_max`, the peak pressure of the gas density profile, at 6,528.6 Pa. The 578 public
  configurations above the cut are rows of your own training table
  `campaign_B_trajectories.parquet`, so remove them before you train — nothing in the pack does
  it for you. Then predict `test_trajectory.parquet` with that one model and submit it; the
  organisers score it separately on the 221 hidden configurations inside the trained range
  (`test_indist`) and the 136 above it (`test_ood`, also shipped on their own as
  `test_trajectory_ood.parquet`), and the gap between the two numbers is your result.
  `test_trajectory_ood.parquet` is not submitted on its own: its rows are a subset of
  `test_trajectory.parquet`, shipped so you can see which hidden configurations lie above the cut.
  **Signature:** unchanged from the scored task; what changes is which configurations you may
  train on, and that one model is scored on two held-out sets rather than one.

  > **⚠ Read this before you start: this is an extrapolation question, not a harder
  > interpolation one.** The out-of-distribution set does not overlap the training range at
  > all. Training runs from `P_max` 4001.5 to 6528.4 Pa; the out-of-distribution set runs from
  > 6532.5 to 6999.1 Pa, so **every one of its 136 configurations sits above the highest
  > pressure the model ever saw**, reaching 1.07× the trained maximum. That is the same shape
  > of question as forecasting a time series past the end of its record: nothing in the
  > training data constrains the answer, and a model that is excellent inside the range can be
  > badly wrong outside it while its own diagnostics still look healthy.
  >
  > So read the deliverable carefully. **What is asked for is the measured gap between the two
  > held-out sets — not a good absolute score on the out-of-distribution one.** A report with a large, honestly measured gap answers the
  > question. A model tuned until the gap looks small on this particular cut has usually
  > learned the cut.
- Predict the bunch as particles, not as moments: for a configuration and a plane, generate a
  cloud of electrons whose full six-dimensional distribution matches the simulation's — not only
  its means and widths, but its shape: tails, skew, correlations beyond the covariance. Training
  data is the separate particle download (see `data/README.md`, "Level 5 — particle clouds"): up to 2,000
  tracked, charge-weighted electrons per simulation for 5,282 configurations of Campaign B — none
  of them a test configuration — as `planes20/`, the twenty planes of the moment table (5.3 GB),
  and, optionally, the full path, a snapshot every 22.5 µm (54.2 GB). It is scored on the
  same rows as `test_moments_planes.parquet`: submit one `Config_<id>.npz` per configuration of
  that file, holding `plane_mm` (its nine planes) and `state` (planes × particles × 6, in the
  coordinate order below). The number of particles is yours to choose; 2,000 is a good default.
  The scorer first redraws your cloud (by your weights) to the true cloud's count, so sending
  more particles does not change the score. A plane with fewer than 50 true electrons is not
  scored and is counted in `n_rows_sparse`.
  **Signature:** `P_max, cN2_max, L_inj, dip_frac, x_of` + `plane_mm` → a cloud of
  (`zeta_um, y_um, z_um, ux, uy, uz`), one row per electron.
  **Score:** `score_particles` in `pallas_score.py` — a sliced Wasserstein distance to the true
  cloud, measured in units of the true cloud's own spread and divided by the true cloud's own
  sampling noise, so a perfect prediction scores about 1 and lower is better; the median over
  rows is the headline. Two references, measured on 100 of the 357 test configurations: a
  Gaussian drawn from the **true** mean and covariance scores 2.13 — a model that gets every
  moment exactly right but ignores the shape still leaves a factor of two on the table — and
  copying the nearest training simulation, with no model at all, scores 4.26.

  > **⚠ Start with caution: this objective is heavy on data, not on training.** Training a
  > particle model takes about as long as training a moments model (1.03× to 1.25× on one
  > A6000 at matched steps and batch, measured); our own conditional-flow generator (width 512,
  > depth 4) reaches its final state in about 100 minutes there (500 epochs). What costs is
  > reading the data: the full set is 54.2 GB and takes minutes to load, against a fraction of a
  > second for the moment table.
  > Start from `planes20/`; reach for the full set only once a model works on the thinned one.
  > Some simulations inject their bunch late: at their earliest planes no electron is present
  > yet, and those planes carry `present` all false. A plane with no true bunch is not scored.

## ⚪ Optional — outside the sequence

- Go out of distribution in position: train on the eleven planes of the moment tables named in
  the `plane_holdout` section of `pack_reference.json`, and predict the bunch at the nine the
  model has never seen. The moment table ships all twenty planes, so drop the nine held-out
  ones from your training rows yourself. This is scored on the same `test_moments_planes.parquet`
  as the Medium moments objective; the difference is the training rule. There the architecture
  only had to learn the moments; here it has to predict them at planes it never saw, which is
  the real test of it.
  Read 3.5 mm and 5.0 mm apart from the other seven: those two sit in gaps of 0.6 mm and 2.0 mm
  rather than 0.2 mm, so they ask a harder question.
  **Signature:** `P_max, cN2_max, L_inj, dip_frac, x_of` + `plane_mm` → the same thirteen
  columns as the Medium moments objective, with `plane_mm` taking nine values the model never saw in training.

  > **Note the contrast with the density objective above: this one is interpolation.** All nine
  > held-out planes lie inside the trained span of 2.0 to 7.0 mm; none is beyond it. The model
  > fills gaps rather than predicting past the end, which is why 3.5 mm and 5.0 mm are called
  > out separately — they sit in the widest gaps, not outside the range. The two objectives are
  > labelled "out of distribution" but they ask different questions, and a method that handles
  > one need not handle the other.
- Build the data-budget curve: train the same model four times, on 100, 300 and 1,000
  configurations and on all 2,855 public configurations that have trajectories, predict
  `test_trajectory.parquet` with each of the four, and submit all four prediction files; the
  organisers score them on the hidden test, and report the number of configurations at which your model first
  reaches a mean R² of 0.90 over the six quantities. **Which configurations go into the 100,
  300 and 1,000 is your choice** — the pack ships no fixed lists. Picking them well (spread over
  the settings, dense where the physics changes fast) is part of the task: say how you chose,
  and compare against random draws of the same size, which is how our reference curve is made.
  **Signature:** unchanged from the scored task above; the only thing that changes across the
  four points is how many configurations you are allowed to train on, and which.
- Solve the inverse task as a distribution rather than a point: in go the beam columns, out come
  `p_1`, `a_0` and `c_N2` each with a 90 % interval, scored by coverage — does the true value
  fall inside the interval 90 % of the time — and by the mean width of those intervals. The bar
  here is the calibration, not a point R²: different settings can produce nearly the same bunch,
  which is why even a perfect point predictor stops at R² 0.9909 on this task (the mean of our
  internal ceilings 0.9940 for `p_1`, 0.9858 for `a_0` and 0.9928 for `c_N2`).
  **Signature:** `E_med_MeV, dE_mad, q_end, i_peak, n_emit_x, sigma_z, sigma_y, div_rms` + `x_of` →
  a 90 % interval (a lower and an upper bound) for each of `p_1, a_0, c_N2`
- Show that degeneracy directly: in Campaign A, find pairs of settings (`p_1`, `a_0`, `c_N2`)
  that produce indistinguishable bunches in the eight beam columns of the inverse task above, and
  say which knob is the one that cannot be recovered. What counts as indistinguishable is yours
  to choose — state the tolerance you used. Not scored; answered in your report.
- Predict the whole energy spectrum on the shared 200-bin grid instead of its summary statistics:
  in Campaign B, on the rows where `injected` is true (the others carry no spectrum: their
  `spec_common` is all NaN), in go the settings, out comes the charge in each energy bin — the
  spectrum itself, not normalised. It is scored by two numbers, both on your own held-out fold,
  because `pallas_score.py` does not score this one and the organisers hold no hidden answers
  for it:
  1. **its shape**, by the earth-mover distance between your spectrum and the true one —
     `scipy.stats.wasserstein_distance(grid, grid, true_spectrum, your_spectrum)`, where `grid`
     is `common_energy_grid_MeV.npy`; in one dimension that is exactly the earth-mover distance,
     in MeV. It normalises both spectra first, so it sees the shape only;
  2. **its total charge**, the sum of your spectrum against the true total, which is `q_pC`.

  **Signature:** `P_max, cN2_max, L_inj, dip_frac, x_of` → `spec_common`, the
  charge in pC in each of the 200 bins of `common_energy_grid_MeV.npy` (25.8 to 510.3 MeV)
- Treat the gas density profile as a curve rather than as four numbers, in Campaign B: train one
  model given the settings and another given the 2,000-point curve `n_e_p` the pack ships in
  place of the four density settings, predict the same outputs with both and score them the same
  way, on your own held-out fold — the test files carry the settings but not the curve. The curve is the helium density the simulation was started from, and it is built from
  `P_max`, `cN2_max`, `L_inj` and `dip_frac` alone, so it carries no information the four numbers
  do not. The question is therefore whether a model that reads the curve loses nothing against
  one that reads the numbers — which has to hold before a measured profile could ever replace
  them. `x_of` is where the laser is focused, not part of the profile, so both models get it.
  **Signature, twice over:** `P_max, cN2_max, L_inj, dip_frac, x_of` → the same outputs, against
  `n_e_p` (sampled at `x_p`) + `x_of` → those same outputs.
- Find the physics that helps: a physics-derived input, constraint or model structure that
  measurably raises the in-plasma energy score. **We tried the obvious one and it failed**, so
  this has an honest reference to beat. Our test (internal) integrated a fixed fraction η of the
  wave-breaking field, E0 = 96 √(n_e[cm⁻³]) V/m, along each configuration's own density profile:
  - alone, with one η fitted on the training configurations and the *true* energy at injection
    handed over, it reaches only R² 0.26 inside the plasma; the η that each configuration would
    need spans 0.20 to 0.49 (10th to 90th percentile);
  - given to a gradient-boosted model as a sixth input beside the settings and `z_mm`, it moves
    the in-plasma energy R² by −0.006 to +0.001 across the four sizes (100, 300, 1,000 and all
    2,855; ten random subsets per size) — never a gain as large as the spread between subsets;
  - the reason is visible in the residual: it correlates −0.74 with the bunch's charge. A
    heavy bunch weakens the field it rides on (beam loading), and the law has no charge in it.
  Beat our gradient-boosting reference, which uses no physics input, at 100 and 300 configurations
  (0.820 ± 0.016 and 0.908 ± 0.008, mean ± standard deviation over ten random subsets, pointwise
  gradient boosting, scored on the hidden test) with a physics ingredient,
  and say *why* it works. Beam loading and dephasing are the two places to start
  (`scientific_case.md`). A physics term that needs the charge as an input has to predict the
  charge too. Scored with `pallas_score.py` on the energy column, in-plasma zone.
  **Signature:** `P_max, cN2_max, L_inj, dip_frac, x_of` + `z_mm` (+ anything you derive from
  them or from `n_e_p`) → `E_MeV`
- Report a surrogate that is both fast and honest: quote inference time per sample alongside
  accuracy, and state the region of settings where the model should refuse to answer. Give the
  time per sample together with the hardware it was measured on, and write the refusal rule down
  before you score the model on any held-out set. Not scored; answered in your report.
- Predict the wake itself, not the bunch it makes: for a configuration, predict three maps of
  the fields on the laser axis — the longitudinal electric field `ex`, the plasma electron
  density `rho_e` and the laser envelope `laser` — each over the 157 snapshots along the
  accelerator (one every 45 µm, from 0 to 7.02 mm) and the 320 points across the 63.9 µm window
  that moves with the laser. Training data is level 6, the separate field download (`level6/`,
  see `data/README.md`): the three maps for the 5,286 configurations of
  `campaign_B.parquet`. It is scored on the 357 configurations of `test_direct.parquet`:
  submit a folder laid out like the download — `field_axes.npz` holding `config`, and
  `field_ex.npy`, `field_rho_e.npy`, `field_laser.npy`, each configurations × 157 × 320, rows in
  the order of `config`.
  **Signature:** `P_max, cN2_max, L_inj, dip_frac, x_of` → the maps `ex`, `rho_e`, `laser`
  (snapshot × position in the window)
  **Score:** `score_field` in `pallas_score.py` — for each configuration and each field, the
  relative L2 error of the whole map, ‖prediction − truth‖ / ‖truth‖, after both maps are
  averaged over 1 µm along the window; the median over configurations is the headline, one
  number per field, and lower is better. The averaging means that a sharp feature placed a
  cell or two off counts as nearly right. The reference is a principal-component fit (20
  components per field) with a linear map from the five settings to the components:
  `ex` 0.533, `rho_e` 0.839, `laser` 0.092 (`field_reference` in the `scores` section of
  `pack_reference.json`). A map of all zeros scores 1; the average training map scores
  0.689 / 0.882 / 0.184, so the density map is where a fit of this kind barely helps.

  > **⚠ This objective is data-driven, and the method is open.** Most of each map is smooth and
  > easy; the error lives in the sharp features — the density spike at the back of the bubble,
  > the jump where electrons are injected — which a smooth fit blurs. Two axes are not the same
  > thing: across a map's 320 points you read one moment of the wake (laser in front, wake
  > behind); down its 157 snapshots you follow that wake as the laser travels through the
  > plasma. Which axis your model treats as "time" is a design choice worth making on purpose.
