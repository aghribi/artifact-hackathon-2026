# CLARA 6D phase-space measurement at S02, no TDC: draft procedure

15 Oct 2026, L4H off, VBC off. This procedure measures the 6D beam distribution at the S02 start (`cla_s02_sim_aper_01`) from 17 YAG images, without the TDC. Beam time is about 16 min. Analysis takes about 13 min.

The plan comes from `simenv_acquire.no_tdc_plan()` with `fea_match=FEA_L4H_MATCH`. The k1 values come from `full6d.make_plan()`. Currents were computed with `XsuiteLine.quad_current_from_k1` on the line from `full6d.build_line()`, which has VBC off and L4H off.

## 1. Machine state before starting

1. **VBC off.** All four VBC dipoles are at angle 0 and the beam goes straight through.
2. **L4H off.** `cla_l4h_lin_cav_01` is unpowered. Its 4.12 mm iris stays in the beam path.
3. **TDC not used.** It is broken. Leave it off for the whole measurement.
4. **Repetition rate 10 Hz.**
5. **Nominal optics.** All magnets are at their saved nominal currents. The L02 phase (`cla_l02_lin_cav_01`) is at nominal, which is 23 deg in the lattice file. Save this state as the restore file.
6. **Orbit.** Run one one-shot SVD orbit correction on the nominal optics, with all screens out. Use 3 iterations with the design-model response (`simenv_acquire.svd_steer`). Do not run beam-based alignment. Record the corrector settings, because the analysis needs the applied kicks.
7. Do not retune anything upstream of S02 during the measurement. The beam at S02 is the beam being measured.

## 2. Acquisition rules

- Only one screen is in at a time.
- Take 10 shots per image at 10 Hz (1 s per image). Store every shot as raw counts on the full chip. Do not crop or centre the image.
- With every image, record the readings of all BPMs upstream of the inserted screen, for every shot. The BPMs for each screen are listed below.
- Some images clip at the screen edge. Take them as they are and do not re-steer to centre them. The offset fit leaves out the centroids of clipped images automatically.
- After every magnet or phase change, wait 30 s before taking shots.

BPMs to record for each screen:

| screen | BPMs upstream (record all) |
|---|---|
| cla_s02_dia_scr_03 | cla_s02_dia_bpm_01, cla_s02_dia_bpm_02 (2 BPMs) |
| cla_s03_dia_scr_01 | the 2 above, plus cla_s03_dia_bpm_01 (3 BPMs) |
| cla_s04_dia_scr_03 | the 3 above, plus cla_s04_dia_bpm_01, cla_s04_dia_bpm_02 (5 BPMs) |
| cla_fea_dia_scr_01 | cla_s02_dia_bpm_01 to _02, cla_s03_dia_bpm_01, cla_s04_dia_bpm_01 to _06, cla_s05_dia_bpm_01 to _02, cla_vbc_dia_bpm_01, cla_s06_dia_bpm_01 to _02, cla_s07_dia_bpm_01 to _07, cla_fea_dia_bpm_01 (23 BPMs) |

## 3. Measurement order

Each screen goes in once. The S02 quad settings are cycled while the screen is in. The settings run forwards on the first screen, backwards on the second and forwards again on the third, so each new screen starts at the setting the previous one ended on. This needs 3 screen insertions and 9 magnet changes for the 12 transverse images. Taking one setting at a time over all three screens would instead need 12 insertions.

The FEA-01 block comes last. It changes 23 quads and the L02 phase, so putting it first or in the middle would add an extra full change and restore. The phase scan runs from -30 deg to 0 deg, so it ends at the nominal L02 phase and the phase needs no separate restore.

### 3.1 S02 quad settings (blocks A to C)

All other magnets stay at nominal. The momentum at the S02 quads is 35.8 MeV/c.

| setting | cla_s02_mag_quad_01 k1 (1/m^2) | current (A) | cla_s02_mag_quad_02 k1 (1/m^2) | current (A) |
|---|---|---|---|---|
| nominal | +13.911 | +8.38 | -12.938 | -7.79 |
| s02a | -18.751 | -11.30 | +43.510 | +26.30 |
| s02b | -13.939 | -8.40 | +29.859 | +18.00 |
| s02c | -15.266 | -9.20 | +26.552 | +16.00 |

### 3.2 Step table

L02 offset is the change from the nominal L02 phase, in degrees. Each image is 10 shots.

| step | action | screen in | S02 setting | other magnets | L02 offset (deg) | shots | record |
|---|---|---|---|---|---|---|---|
| A0 | insert screen | cla_s02_dia_scr_03 | nominal | nominal | 0 | | |
| A1 | image | cla_s02_dia_scr_03 | nominal | nominal | 0 | 10 | image, 2 BPMs |
| A2 | set s02a, wait 30 s, image | cla_s02_dia_scr_03 | s02a | nominal | 0 | 10 | image, 2 BPMs |
| A3 | set s02b, wait 30 s, image | cla_s02_dia_scr_03 | s02b | nominal | 0 | 10 | image, 2 BPMs |
| A4 | set s02c, wait 30 s, image | cla_s02_dia_scr_03 | s02c | nominal | 0 | 10 | image, 2 BPMs |
| A5 | retract screen | | s02c | nominal | 0 | | |
| B0 | insert screen | cla_s03_dia_scr_01 | s02c | nominal | 0 | | |
| B1 | image (no change) | cla_s03_dia_scr_01 | s02c | nominal | 0 | 10 | image, 3 BPMs |
| B2 | set s02b, wait 30 s, image | cla_s03_dia_scr_01 | s02b | nominal | 0 | 10 | image, 3 BPMs |
| B3 | set s02a, wait 30 s, image | cla_s03_dia_scr_01 | s02a | nominal | 0 | 10 | image, 3 BPMs |
| B4 | set nominal, wait 30 s, image | cla_s03_dia_scr_01 | nominal | nominal | 0 | 10 | image, 3 BPMs |
| B5 | retract screen | | nominal | nominal | 0 | | |
| C0 | insert screen | cla_s04_dia_scr_03 | nominal | nominal | 0 | | |
| C1 | image (no change) | cla_s04_dia_scr_03 | nominal | nominal | 0 | 10 | image, 5 BPMs |
| C2 | set s02a, wait 30 s, image | cla_s04_dia_scr_03 | s02a | nominal | 0 | 10 | image, 5 BPMs |
| C3 | set s02b, wait 30 s, image | cla_s04_dia_scr_03 | s02b | nominal | 0 | 10 | image, 5 BPMs |
| C4 | set s02c, wait 30 s, image | cla_s04_dia_scr_03 | s02c | nominal | 0 | 10 | image, 5 BPMs |
| C5 | retract screen | | s02c | nominal | 0 | | |
| D0 | S02 quads to nominal, FEA quad set at the -30 deg currents (table 3.3), L02 to -30 deg, scale the other downstream magnets (table 3.4), wait 30 s | | nominal | FEA set | -30 | | |
| D1 | insert screen | cla_fea_dia_scr_01 | nominal | FEA set | -30 | | |
| D2 | image | cla_fea_dia_scr_01 | nominal | FEA set | -30 | 10 | image, 23 BPMs |
| D3 | L02 to -23 deg, magnets to the -23 deg column, wait 30 s, image | cla_fea_dia_scr_01 | nominal | FEA set | -23 | 10 | image, 23 BPMs |
| D4 | L02 to -15 deg, magnets to the -15 deg column, wait 30 s, image | cla_fea_dia_scr_01 | nominal | FEA set | -15 | 10 | image, 23 BPMs |
| D5 | L02 to -8 deg, magnets to the -8 deg column, wait 30 s, image | cla_fea_dia_scr_01 | nominal | FEA set | -8 | 10 | image, 23 BPMs |
| D6 | L02 to 0 deg, magnets to the 0 deg column, wait 30 s, image | cla_fea_dia_scr_01 | nominal | FEA set | 0 | 10 | image, 23 BPMs |
| D7 | retract screen | | nominal | FEA set | 0 | | |
| E1 | restore the nominal file, wait 30 s | | nominal | nominal | 0 | | |

Image keys for the analysis file use the form "config|screen", for example `s02b|cla_s03_dia_scr_01` or `fea_01_ph-15|cla_fea_dia_scr_01`. The config names are `nominal_off`, `s02a`, `s02b`, `s02c`, `fea_01_ph-30`, `fea_01_ph-23`, `fea_01_ph-15`, `fea_01_ph-8` and `fea_01_ph+0`.

### 3.3 FEA-01 quad set (block D)

> **Final.** These are the values to use. FEA-01 has only about 2 cm of dispersion in this setting. A re-match does not give more: the FEA dogleg is achromatic (four 14 deg bends), with at most 3.5 cm of dispersion at FEA-01 and FEA-03 and none at FEA-02 and FEA-04. Optimising FEA quads 01 to 06 does not get FEA-02 past about 4 cm, and no screen from S02 to FEH has more. The phase scan therefore measures the projected energy spread and the chirp, and the slice energy spread is only an upper bound (see section 6).

The S06, S07 and FEA quads are the `fea_01_off` design from `optics_design.json`. The S03 quad_01 and S04 quads are `FEA_L4H_MATCH`, which takes the beam through the 4.12 mm L4H iris. In simulation, 98.6 % of the beam then reaches FEA-01 at every phase, against 70 % with the nominal S03 and S04 setting.

The k1 values stay fixed. The currents follow the beam energy, which changes with the L02 phase. The 0 deg column is at nominal momentum: 116.4 MeV/c at S03, 194.4 MeV/c at S04 to S06 and 260.4 MeV/c at S07 and FEA. The other columns add the L02 energy change, computed from the lattice L02 settings (87.54 MV, 23 deg nominal phase). An offset of -23 deg puts L02 on crest.

| quad | k1 (1/m^2) | I at -30 deg (A) | I at -23 deg (A) | I at -15 deg (A) | I at -8 deg (A) | I at 0 deg (A) | nominal I (A) |
|---|---|---|---|---|---|---|---|
| cla_s03_mag_quad_01 | -1.4224 | -3.14 | -3.15 | -3.13 | -3.08 | -2.97 | -2.03 |
| cla_s04_mag_quad_01 | +2.6981 | +9.70 | +9.73 | +9.69 | +9.59 | +9.40 | +2.17 |
| cla_s04_mag_quad_02 | -0.4489 | -1.62 | -1.63 | -1.62 | -1.60 | -1.57 | +0.58 |
| cla_s04_mag_quad_03 | -0.7350 | -2.65 | -2.66 | -2.65 | -2.62 | -2.57 | -1.36 |
| cla_s04_mag_quad_04 | -1.6278 | -5.86 | -5.88 | -5.85 | -5.79 | -5.67 | -0.73 |
| cla_s04_mag_quad_05 | -0.1423 | -0.51 | -0.52 | -0.51 | -0.51 | -0.50 | +0.89 |
| cla_s04_mag_quad_06 | -0.2843 | -1.03 | -1.03 | -1.03 | -1.02 | -1.00 | +1.52 |
| cla_s04_mag_quad_07 | +0.8083 | +2.91 | +2.92 | +2.91 | +2.87 | +2.82 | +1.82 |
| cla_s04_mag_quad_08 | -0.1372 | -0.49 | -0.50 | -0.49 | -0.49 | -0.48 | -0.94 |
| cla_s06_mag_quad_01 | -1.2409 | -3.76 | -3.78 | -3.76 | -3.72 | -3.65 | +3.09 |
| cla_s06_mag_quad_02 | +0.0831 | +0.25 | +0.25 | +0.25 | +0.25 | +0.24 | -1.21 |
| cla_s07_mag_quad_01 | +3.5264 | +14.22 | +14.25 | +14.21 | +14.09 | +13.88 | +9.52 |
| cla_s07_mag_quad_02 | -3.6790 | -14.83 | -14.87 | -14.82 | -14.70 | -14.48 | -3.04 |
| cla_s07_mag_quad_03 | +6.4738 | +26.10 | +26.16 | +26.08 | +25.87 | +25.48 | -5.00 |
| cla_s07_mag_quad_04 | -10.0620 | -40.57 | -40.67 | -40.54 | -40.21 | -39.61 | -4.45 |
| cla_s07_mag_quad_05 | +13.2045 | +53.24 | +53.37 | +53.20 | +52.77 | +51.98 | +4.05 |
| cla_s07_mag_quad_06 | -16.9650 | -68.40 | -68.56 | -68.35 | -67.80 | -66.78 | +0.52 |
| cla_s07_mag_quad_07 | +6.1388 | +24.75 | +24.81 | +24.73 | +24.53 | +24.16 | -5.91 |
| cla_s07_mag_quad_08 | -11.7476 | -56.93 | -57.07 | -56.88 | -56.42 | -55.55 | +24.83 |
| cla_s07_mag_quad_09 | +16.2046 | +78.90 | +79.10 | +78.84 | +78.19 | +76.96 | +12.21 |
| cla_s07_mag_quad_10 | -16.2833 | -79.68 | -79.88 | -79.62 | -78.96 | -77.72 | -35.47 |
| cla_fea_mag_quad_01 | +14.9245 | +72.20 | +72.38 | +72.14 | +71.56 | +70.46 | +49.19 |
| cla_fea_mag_quad_02 | -15.1142 | -73.17 | -73.36 | -73.12 | -72.52 | -71.42 | -35.03 |

The FEA currents here are higher than the ones in `optics_design.json` (+66.95 A and -67.86 A). The design file assumed L4H on. With L4H off, the momentum at S07 is higher.

### 3.4 Other magnets downstream of L02 (block D)

All other magnets downstream of L02 follow the energy at each phase. This includes the quads not listed in 3.3, the FEA dogleg dipoles and the correctors. Multiply their field (B L) by these factors. The quads in table 3.3 are already scaled.

| L02 offset (deg) | energy change (MeV) | factor at S03 | factor at S04 to S06 | factor at S07 and FEA |
|---|---|---|---|---|
| -30 | +6.31 | 1.0542 | 1.0325 | 1.0242 |
| -23 | +6.96 | 1.0598 | 1.0358 | 1.0267 |
| -15 | +6.11 | 1.0525 | 1.0314 | 1.0235 |
| -8 | +3.98 | 1.0342 | 1.0205 | 1.0153 |
| 0 | 0 | 1 | 1 | 1 |

## 4. Time budget

| item | count | unit time | time |
|---|---|---|---|
| screen moves (4 insertions, 4 retractions) | 8 | 1 min | 8 min 0 s |
| magnet changes (9 S02 changes, 1 FEA set plus L02, 4 phase steps, 1 restore) | 15 | 30 s | 7 min 30 s |
| images (10 shots at 10 Hz) | 17 | 1 s | 17 s |
| **total** | | | **15 min 47 s, about 16 min** |

You can save about 1 min by overlapping two magnet changes with screen moves. Set the FEA quad set (D0) while the S04 screen retracts, and restore the nominal file (E1) while the FEA screen retracts. The orbit correction before the start is not included.

## 5. Analysis

Run in `challenges/optimisation/inference/sixd/` with `XO_OPT=-O1 XSUITE_ALLOW_KERNEL_COMPILATION=1`. Building the model line (`full6d.build_line()`) takes about 4 min to 7 min and is done once.

1. **Data file.** Write the 17 images and their BPM readings in the `images.npz` layout that `simenv_acquire.acquire` produces. It holds `keys`, the images summed over 10 shots, the mean BPM readings per image with their valid flags, the BPM names and the screen geometry. The normalisation is the bunch charge. A loader for measured data still has to be written.
2. **Quad offset fit, about 3.7 min.** `offset_fit.fit(line, configs, data, beam, quads, kicks=...)`. It fits the quad offsets, the launch and each diagnostic's own offset from the plan's own BPM and screen centroids. `configs` comes from `no_tdc_plan(line, fea_match=FEA_L4H_MATCH)` and `kicks` are the SVD corrector kicks from step 1.6. Readings taken after any beam scraping are left out.
3. **Aperture-aware maps, about 6.5 min.** `surrogate.fit_with_offsets(line, configs, quad_offsets, tag, aperture_line=full6d.build_line(apertures=True), kicks=...)`. This fits the polynomial maps from the S02 start to each of the 17 images, with the fitted offsets in. The maps also say which particles the apertures lose.
4. **Reconstruction, about 2.8 min.** `fast_ment.run(line, images_path=<file>, set_name="full", maps=surrogate.load_combined(tag))`. The output is `ment_full_fast.npz` (weighted 6D particles at S02) and `ment_results_fast.json` (emittances, sizes, correlations).
5. **Check.** The final image chi2 per pixel in the log should be about 1. If it is 2 or more, check the offset fit chi2 first, then whether the FEA magnets followed the energy at each phase.

| analysis step | time |
|---|---|
| offset fit | about 3.7 min |
| aperture-aware maps | about 6.5 min |
| reconstruction | about 2.8 min |
| **total** | **about 13 min** |

The times above are for one analysis using all 4 cores.

## 6. What it recovers (SimEnv check)

Three SimEnv episodes were run end to end on the plan lattice, with VBC off and L4H off: steering, acquisition, offset fit, maps and reconstruction. They used 50 um quad misalignments, 30 um diagnostic offsets, beam variation and jitter, and apertures on. The table gives reconstructed over true at the S02 start.

| quantity | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| core eps_x | 1.21 | 0.88 | 1.01 |
| core eps_y | 1.13 | 1.24 | 0.86 |
| rms eps_x | 0.74 | 0.48 | 0.48 |
| rms eps_y | 1.20 | 1.23 | 0.84 |
| sigma_z | 1.12 | 1.16 | 1.19 |
| sigma_delta | 0.97 | 1.19 | 0.85 |
| chirp | 0.42 | 0.94 | 0.41 |
| slice sigma_delta | 6.2 | 3.8 | 5.2 |

- **Reliable to about 25 %:** the core emittances, sigma_z and sigma_delta.
- **Under-recovered:** rms eps_x. The 1 % halo falls outside the screens.
- **Under by up to 2.4x:** the chirp. The limiting factor is the small dispersion at FEA-01.
- **Not resolved:** the slice energy spread. Treat the result as an upper bound.
- **Offset fit:** the residual quad offset error is 42 to 48 um rms, against 45 to 50 um true. It fixes the launch and the diagnostic offsets, but the quad offsets themselves are barely constrained.
- **Seed 1 transmission:** only 73 % of the beam reached FEA-01. The rest was lost on the L4H iris, and the aperture-aware maps account for this.
