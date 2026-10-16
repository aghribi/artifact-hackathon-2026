# CLARA steering: working log

Team Overfit & Proud, ARTIFACT hackathon optimisation challenge. Written 14 October 2026; covers the steering work of that day. The code is in `challenges/optimisation/inference/`, run with `python`.

The question: with the xsuite nominal model (PR #1 on `amylizzle/artifact-hackathon-2026`, merged 14 October), how well can we infer and correct the trajectory from start (`cla_s02_sim_aper_01`) to the IP (`cla_feh_dia_scr_01`), first with BPMs and then with screens only, and is BPM-only correction good enough to use online?

## Setup common to everything

- `inference/clara_virtual.py` holds two copies of the same xsuite line, built from the lattice-file strengths.
  - The machine (`VirtualMachine`) holds a hidden beam and hidden errors: a launch offset, quad offsets, BPM offsets and shot jitter. It answers only with BPM centroids, with 10 um noise, or with one inserted screen per shot. Screens are destructive.
  - The model (`XsuiteLine`) is error free. Its maps come from finite differences of xsuite tracking, so cavities and chromatic effects are included.
- Quads are set in amperes through the CLARA calibration (`utils.quadrupole_k_from_current`), limited to +-100 A and bipolar. At the model momentum this gives about +-160 m^-2 in S02, +-27 in S07 and +-22 in FEA.
- Instruments on the line: 46 quads, 31 BPMs, 23 screens and 29 combined correctors. The two FEH correctors and the two FEH quads come after the last BPM.
- Default errors: launch rms 0.3 mm / 30 urad (plus 0.2 % in delta), quad offsets 100 um rms, shot jitter 20 um / 2 urad / 20 um / 2 urad / 0 / 2e-4.

## 1. Inferring the errors (`inference/steering/steering_inference.py`)

This fits the launch (x, px, y, py, delta) and all 92 quad offsets. Each quad is stepped by 5 A towards zero, one at a time (beam-based alignment by modulation). That is 45 settings with BPMs; with screens only, the 2 or 4 screens after each stepped quad are read.

### What it took to make it work

1. A linear response about an on-axis particle fails, with a 155 um residual. With the lattice-file optics the beam is millimetres to centimetres wide and carries 0.9 to 2.1 % energy spread, so its centroid follows the energy-averaged map. The fix was a Jacobian averaged over 7 Gauss-Hermite energy slices.
2. Even for a beam shrunk by 100, a one-step linear fit stalls at a 70 um residual, because the orbit reaches 15 mm at FEA. The fix was Levenberg-Marquardt, re-linearised every iteration, with predictions from tracking the model beam through the model carrying the current offset estimate.
3. 20 um of launch jitter turns into hundreds of microns downstream and is correlated across all BPMs in a shot. The fix was to treat it as a block noise covariance (generalised least squares): per configuration with BPMs, per insertion with screens.
4. 30 A quad steps lose beam with these optics, so the step is 5 A.
5. A free offset per monitor, meant to absorb the beam-shape centroid shift, is nearly degenerate with the launch and made the quad errors grow to 170 to 260 um. It is off by default.

### Results (one error seed, 99 um rms true quad offsets)

| Measurements | Beam the fit uses | Shots | Quad offset error x / y (rms, um) | Median abs (um) | Fitted delta (true 0.072 %) |
|---|---|---|---|---|---|
| 31 BPMs | exact (the machine's own particles) | 225 | 6.6 / 5.1 | 2.6 | 0.045 % |
| 31 BPMs | an independent sample of the nominal beam | 225 | 6.1 / 14.9 | 4.6 | 0.48 % |
| screens, 2 per quad | exact | 555 | 78 / 102 | 25 | not measurable |
| screens, 4 per quad | exact | 915 | 38 / 97 | 15 | not measurable |
| screens, 4 per quad | nominal sample | 915 | 36 / 92 | 17 | not measurable |

- With the exact beam the whitened chi2/dof is 1.03. With the nominal sample it stalls at 8.2: two samples of the same distribution differ by up to 1.5 mm in centroid at the FEA BPMs. The fit absorbs that into delta, while the quads stay good.
- With screens only, everything up to S06 is good to 10 to 40 um. S07 Q3/Q4, most of FEA and the FEH quads stay at 90 to 350 um in y, because few screens see them and the beam is large there.
- Figures: `inference/steering/fig_quad_offsets.png` (true vs fitted offsets with 1 sigma bars) and `fig_convergence.png` (chi2/dof per iteration).

## 2. One-shot SVD correction (`inference/steering/svd_correction.py`)

This is Francesco's suggestion. The corrector to BPM response matrix comes from the xsuite model at the known quad settings. One BPM shot is taken and the correction is dtheta = -R^+ orbit, with a truncated SVD that keeps singular values above 1e-3 of the largest (27 of 58 modes; FEH correctors have zero response).

- On seed 0, one shot takes the orbit from 2.2 / 4.2 mm rms to 0.13 / 0.17 mm, with kicks under 0.43 mrad. Three shots give 0.08 / 0.11 mm. A second seed gives 0.10 / 0.19 mm after one shot.
- The beam stays about 0.2 mm off-centre in the quads, because BPMs cannot see quad centres.
- Figure: `inference/steering/fig_svd_orbit.png` (true centroid at each BPM, before and after).

## 3. Error study of BPM-only one-shot correction (`inference/errors_study/`)

The grid: 200 seeds per point, quad offsets {0, 50, 100, 200, 300} um, BPM offsets {0, 50, 100, 200} um, launch 0.3 mm / 30 urad, plus side scans. Correction is one shot, cutoff 1e-3, and the BPM-error case is re-corrected from scratch. Values are medians with p90 in brackets, x and y combined in quadrature, BPM offsets 0 to 200 um pooled.

| Quad rms (um) | BPM rms after (um) | In-quad rms after (um) | IP centroid shift (um) | IP size change (%) |
|---|---|---|---|---|
| 0 | 253 (520) | 260 (478) | 363 (633) | 0.20 (0.42) |
| 50 | 263 (528) | 304 (497) | 465 (912) | 0.22 (0.47) |
| 100 | 297 (572) | 399 (580) | 668 (1403) | 0.22 (0.58) |
| 200 | 420 (687) | 633 (856) | 1120 (2419) | 0.24 (1.02) |
| 300 | 565 (867) | 883 (1169) | 1568 (3283) | 0.44 (1.81) |

- Quad offsets dominate. BPM offsets up to 100 um change any metric by less than 5 %, and 200 um by 5 to 25 %. A 1 mm launch raises the IP shift at 100 um quads only from 0.67 to 0.85 mm.
- The IP centroid stays 2 to 10 times larger than the 80 to 400 um requested spots; at 100 um quads only 27 % of seeds land within 0.4 mm. The 0.36 mm floor with perfect quads comes from single-shot jitter and the 27 kept modes. Averaging 10 shots takes 0.67 to 0.56 mm.
- Optics at 100 um quads: beta-beat at the IP 0.4 % (p90 2 %), maximum along the line 10 % (p90 26 %).
- Spurious dispersion at the IP is about 14 mm, and already 13 mm with perfect quads. With 2.1 % energy spread that adds about 0.3 mm of beam size. It is negligible against today's 3 x 25 mm beam, but as large as the target spots.
- Correctors: the SVD drives the FEA and VBC correctors to 0.3 to 1 mrad. The archiver shows they have never moved (about 0.1 urad). Restricting the SVD to used planes doubles the IP error.
- Reference tune: the same conclusion holds. Flattening to zero is wrong there, because the error-free beam sits 1.7 mm off axis from dispersion. Targeting the model's error-free orbit gives 0.23 to 0.39 mm.
- Figures (in `inference/errors_study/`):
  - `fig_hist_residual.png`: residual histograms vs BPM and quad errors
  - `fig_residual_vs_errors.png`: residuals and IP shift vs error level
  - `fig_optics_change.png`: beta-beat and dispersion
  - `fig_ip_size_change.png`: IP size change
  - `fig_kicks.png`, `fig_side_scans.png`

**Verdict:** BPM-only one-shot correction is a good first stage that gets the beam through. It is not enough on its own for spots of 80 to 400 um. Online, use averaged BPM correction against the model's target orbit first, then screen-based fine steering of the last correctors, then a dispersion correction.

## Caveats

- Only quad offsets were randomised: no rolls, gradient errors, BPM gain errors or corrector calibration errors, and no corrector limits enforced.
- The xsuite line has no apertures, so "no losses" says nothing about margins.
- The lattice-file optics give a 3 x 25 mm beam at the IP with beta up to 1e5 m. Absolute numbers will change on matched optics.
- The fits used one or two error seeds; the error study used 200 per point.
