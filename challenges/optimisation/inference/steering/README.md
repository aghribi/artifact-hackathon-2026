# Steering inference

Fits the beam launch at `cla_s02_sim_aper_01` (x, px, y, py, delta) and the (dx, dy) offset of all 46 quadrupoles from centroid measurements on the virtual CLARA in `../clara_virtual.py`. The forward model is the xsuite nominal line: predictions are centroids of a tracked model beam, and the Jacobian is the first-order response averaged over 7 Gauss-Hermite energy slices about the current launch estimate. The fit is a Levenberg-Marquardt MAP fit with priors of 1 mm / 0.1 mrad / 1 % on the launch and 0.5 mm on each quad offset.

Quad offsets only show up when quads change, so each quad is stepped by 5 A towards zero current, one at a time (beam-based alignment by modulation). Steps of 30 A lose beam with the lattice-file optics.

## Test case

The truth has launch errors of (38 um, -4 urad, 192 um, 3 urad, 0.072 %) and 99 um rms quad offsets. Shot-to-shot launch jitter is (20 um, 2 urad, 20 um, 2 urad, 0.02 %) rms, BPM noise is 10 um, and every reading is averaged over 5 shots. The machine beam is 5000 particles of the nominal starting distribution.

## Results

| Measurements | Beam known to the fit | Shots | Quad offset error x / y (rms, um) | Median abs (um) | Launch delta (true 0.072 %) |
|---|---|---|---|---|---|
| 31 BPMs, beam transmitted | exact | 225 | 6.6 / 5.1 | 2.6 | 0.045 +- 0.004 % |
| 31 BPMs, beam transmitted | nominal sample | 225 | 6.1 / 14.9 | 4.6 | 0.48 +- 0.004 % |
| screens only, 2 per modulated quad | exact | 555 | 78 / 102 | 25 | not measured (+-0.3 %) |
| screens only, 4 per modulated quad | exact | 915 | 38 / 97 | 15 | not measured (+-0.2 %) |
| screens only, 4 per modulated quad | nominal sample | 915 | 36 / 92 | 17 | not measured (+-0.2 %) |

The two FEH quads sit after the last BPM, so the BPM rows leave them out. The launch x and y come out within 10 to 40 um in every BPM case.

"Exact" means the fit tracks the machine's own particles: it tests the method. "Nominal sample" is an independent sample of the same starting distribution, which is closer to reality, where the fit only has a model of the beam.

## What the runs taught

**The beam centroid is not the single-particle orbit.** With the lattice-file optics the beam is millimetres to centimetres wide downstream and carries 0.9 to 2.1 % energy spread. Its centroid follows the energy-averaged map. A plain linear response about an on-axis particle leaves a 155 um residual, and the fit returns nonsense.

**The fit has to be re-linearised.** The orbit reaches 15 mm at the FEA BPMs, so even for a beam shrunk by 100 the linear fit stalls at a 70 um residual. Gauss-Newton with xsuite-tracked predictions converges in 3 to 4 iterations.

**Jitter dominates the noise.** 20 um of launch jitter turns into hundreds of microns downstream, correlated across all BPMs in a shot. Treating it as a block noise covariance (generalised least squares) brings chi2/dof to 1.03 when the beam is known exactly.

**An imperfect beam model biases the energy, not the quads.** With the nominal sample the quad offsets stay at 6 / 15 um, but the launch energy offset absorbs the beam-shape centroid shift and lands at 0.48 % instead of 0.07 % (chi2/dof 8). Two samples of the same distribution already differ by up to 1.5 mm in centroid at the FEA BPMs. Adding a free offset per monitor to absorb this made things worse: it is nearly degenerate with the launch, and the quad errors grew to 170 to 260 um.

**Screens only are much weaker.** Each insertion stops the beam and gives one location, so a modulated quad is seen on 2 to 4 downstream screens instead of every BPM. Quads with few screens behind them, and with a small beam offset inside them, stay poorly determined: S07 Q03/Q04, most of FEA and the two FEH quads have y posteriors of 90 to 350 um. Everything upstream is good to 10 to 40 um. The screens cannot measure the energy offset.

## Running

```bash
python steering_inference.py --mode bpm --beam-knowledge nominal
python steering_inference.py --mode screen --screens-per-modulation 4
```

Each run takes 3 to 4 minutes and writes `result_<mode>_<beam>_..._seed0.json` with per-quad true, estimated and posterior offsets.

## Next steps

- Feed the reconstructed beam from the tomography into the fit as its beam model. That should remove the energy bias in the nominal-sample row.
- Use a less aggressive optics for alignment. The lattice-file optics makes the beam so large that centroids depend on its shape at the millimetre level.
- Choose the screen set per modulated quad by expected information rather than "the next N screens".
