# CLARA 6D beam measurement at S02: 21-image procedure

Reconstructs the full 6D distribution (x, px, y, py, tau, delta) at the S02 start (`cla_s02_sim_aper_01`) from 21 YAG images. Beam time is about 20 minutes and the reconstruction takes about 3 minutes on 4 cores.

## Simulated accuracy

The tests included hidden quad offsets (100 um rms), launch errors and launch jitter up to 3x the day-to-day level. Values are reconstructed over true:

| core eps x | core eps y | eps4d ratio | sigma_z | sigma_delta | chirp |
|---|---|---|---|---|---|
| 0.93 | 0.94 | 0.98 | 1.01 | 0.97 | 0.75 |

Two quantities are not measurable with these screens:
- **rms eps_x:** a 1 % halo falls outside the +-5 mm windows.
- **Slice energy spread and eps_z:** at the spectrometer, x is dominated by the linac chirp and the betatron size.

## 0. Before the shift (once, offline)

1. **Quad alignment.** If there is no recent result, run the BPM beam-based alignment scan:
   - Screens out, beam transmitted to the end.
   - Each of the 46 quads is stepped by 5 A towards zero, one at a time, with 5 shots each. That is about 45 magnet changes, so about 23 min.
   - Fit with `python steering/steering_inference.py --mode bpm --beam-knowledge nominal`. This gives the quad offsets to about 10 um.
   - Quad offsets drift slowly, so a recent result can be reused.
2. **Fit the transfer maps** of the model with those quad offsets (about 5 min, needs no beam):
   ```python
   import full6d, surrogate
   line = full6d.build_line()
   maps = surrogate.fit_with_offsets(line, full6d.make_plan(line), surrogate.steering_estimate(<bpm result json>), "_steer")
   ```

## 1. Machine state for the whole measurement

- VBC off: dipoles at angle 0 and the straight-through line.
- Linac and gun at the nominal settings of the reference run. The starting beam is the one being measured, so do not retune upstream of S02 during the measurement.
- All quads at their saved nominal currents, except where a step below says otherwise. Save the nominal file first; every step ends by returning to it.
- TDC (CLA-TDC1-RF-CTRL-01) off at the start.
- Bunch charge recorded with every shot.

## 2. Image acquisition rules

- 10 shots per image at 10 Hz (1 s). Shot-to-shot jitter does not matter at this level; 10 shots only beat down the counting noise.
- Store every shot as raw counts:
  - 200 x 200 pixels of 50 um over +-5 mm, centred on the screen axis;
  - rows are y, columns are x;
  - the background is a shot with the beam blocked (laser shutter or gun RF off), taken once per screen insertion;
  - store the bunch charge too.
- Do not crop or centre the images. The reconstruction needs the absolute position and the fraction outside the window.
- Only one screen is in at a time. Each insertion or retraction takes about 1 min.

## 3. Transverse block (TDC off): 12 images, about 11 min

S02 quad settings (S02 Q01 / Q02, A):

| name | CLA-S02-MAG-QUAD-01 | CLA-S02-MAG-QUAD-02 |
|---|---|---|
| nominal | +8.4 | -7.8 |
| s02a | -11.3 | +26.3 |
| s02b | -8.4 | +18.0 |
| s02c | -9.2 | +16.0 |

All other quads stay nominal. Wait 30 s after every current change.

Expected in-window rms sizes, for sanity checks:

| step | screen | setting | sx / sy (mm) | in window |
|---|---|---|---|---|
| T1 | CLA-S02-DIA-SCR-03 | nominal | 0.59 / 0.77 | 99 % |
| T2 | CLA-S02-DIA-SCR-03 | s02a | 0.63 / 2.73 | 64 % |
| T3 | CLA-S02-DIA-SCR-03 | s02b | 1.43 / 0.88 | 98 % |
| T4 | CLA-S02-DIA-SCR-03 | s02c | 1.72 / 1.95 | 97 % |
| T5 | CLA-S03-DIA-SCR-01 | s02c | 0.40 / 1.44 | 100 % |
| T6 | CLA-S03-DIA-SCR-01 | s02b | 1.91 / 0.15 | 95 % |
| T7 | CLA-S03-DIA-SCR-01 | s02a | 2.78 / 2.34 | 39 % |
| T8 | CLA-S03-DIA-SCR-01 | nominal | 0.64 / 1.58 | 99 % |
| T9 | CLA-S04-DIA-SCR-03 | nominal | 0.90 / 2.00 | 98 % |
| T10 | CLA-S04-DIA-SCR-03 | s02a | 2.86 / 0.87 | 22 % |
| T11 | CLA-S04-DIA-SCR-03 | s02b | 2.55 / 0.57 | 70 % |
| T12 | CLA-S04-DIA-SCR-03 | s02c | 1.02 / 0.15 | 99 % |

Sequence:
1. S02 at nominal. Insert CLA-S02-DIA-SCR-03 and take the background. Then take T1, set s02a and take T2, set s02b and take T3, set s02c and take T4. Retract.
2. Keep s02c. Insert CLA-S03-DIA-SCR-01 and take the background. Take T5, set s02b and take T6, set s02a and take T7, set nominal and take T8. Retract. Running the settings backwards saves a magnet change.
3. Keep nominal. Insert CLA-S04-DIA-SCR-03 and take the background. Take T9, set s02a and take T10, set s02b and take T11, set s02c and take T12. Retract.
4. Return S02 Q01/Q02 to nominal. The reconstruction expects the clipped images in T2, T7, T10 and T11 as they are; do not re-steer to centre them.

Time: 6 screen moves (6 min), 9 magnet changes (4.5 min) and 12 s of shots.

## 4. Streak and spectrometer block: 9 images, about 9 min

There is one quad setting per screen. Each screen is imaged with the TDC off, then +V, then -V.
- **+V:** 7.0 MW at the zero-crossing phase, 78 deg in the 23 March scan.
- **-V:** the same power at the opposite zero crossing (78 + 180 deg).

The model assumes V = 5.9 MV, which follows from an assumed 5 MOhm shunt impedance (see step 4.4).

**S1, CLA-S07-DIA-SCR-02** (all other quads nominal):

| CLA-S06-MAG-QUAD-01 | S06-Q02 | S07-Q01 | S07-Q02 | S07-Q03 | S07-Q04 |
|---|---|---|---|---|---|
| +18.70 | -11.93 | -14.29 | +18.66 | +9.58 | -56.80 |

Expected: off 0.29 / 0.29 mm; on 0.29 / 0.78 mm; 100 % in window.

**S2, CLA-S07-DCP-SCR-01** (all other quads nominal):

| S06-Q01 | S06-Q02 | S07-Q01 | S07-Q02 | S07-Q03 | S07-Q04 | S07-Q05 | S07-Q06 |
|---|---|---|---|---|---|---|---|
| -2.58 | -2.34 | +17.50 | -18.58 | +10.96 | -36.44 | +52.46 | -57.83 |

Expected: off 0.81 / 0.25 mm; on 0.81 / 1.01 mm; 99 % in window.

**S3, CLA-FEA-DIA-SCR-01, the spectrometer** (FEA dogleg on as in nominal operation; all other quads nominal):

| S06-Q01 | S06-Q02 | S07-Q01 | S07-Q02 | S07-Q03 | S07-Q04 | S07-Q05 | S07-Q06 | S07-Q07 | S07-Q08 | S07-Q09 | S07-Q10 | FEA-Q01 | FEA-Q02 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -3.41 | +0.23 | +13.20 | -13.77 | +24.23 | -37.66 | +49.42 | -63.50 | +22.98 | -52.75 | +73.03 | -73.75 | +66.95 | -67.86 |

Expected: off 2.4 / 0.12 mm; on 2.4 / 0.7 mm; 84 % in window (x clips by design).

Sequence, for each of S1, S2 and S3 in turn:
1. Set the currents in the table and wait 30 s. This can overlap with the previous retraction.
2. Insert the screen and take the background.
3. Take the TDC off image. Set +V and take the image. Set -V and take the image. Switch the TDC off.
4. **Optional TDC calibration** (recommended once, at S2): take two extra images at +V with the phase shifted by +-2 deg. The centroid shift against phase gives the true streak voltage, which replaces the assumed shunt impedance. Costs about 30 s.
5. Retract.

Finish by restoring the nominal quad file.

Time: 6 screen moves (6 min), 3 settings plus the restore (2 min), and the TDC changes.

## 5. Reconstruction (about 3 min)

1. Convert the data into the `images.npz` layout:
   - `keys`, named "config|screen", e.g. `s02a|cla_s03_dia_scr_01` or `s07_dcp_01_plus|cla_s07_dcp_scr_01`, as in `full6d.make_plan`;
   - `images`: 200 x 200, background-subtracted, summed over the 10 shots;
   - the normalisation is the total charge.

   The loader for measured data is still to be written: `ment6d.load_images` currently takes the particle count from the simulated beam. Charge normalisation replaces it.
2. Run:
   ```python
   import fast_ment, surrogate
   fast_ment.run(line, images_path=<file>, set_name="three screens four settings", maps=surrogate.load_combined("_steer"))
   ```
3. Check the result:
   - The final image chi2 per pixel in the log should be about 1.
   - If it is 2 or more, check the TDC voltage calibration and the quad-offset file first.

   Outputs are `ment_full_fast.npz` (weighted 6D particles at S02) and `ment_results_fast.json` (emittances, sizes, correlations).

## Total time

| block | time |
|---|---|
| transverse (12 images) | about 11 min |
| streak and spectrometer (9 images) | about 9 min |
| reconstruction | about 3 min |
| BPM alignment scan, when needed | about 23 min |
| map fit, offline | about 5 min |
