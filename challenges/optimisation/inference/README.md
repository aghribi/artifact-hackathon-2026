# Steering inference and correction on the xsuite model

Team Overfit & Proud. This is the steering work done on the nominal xsuite model (`../nominal_model/`): inferring the trajectory errors, one-shot SVD orbit correction, and a Monte Carlo study of whether BPM-only correction is good enough to use online. The full story, with every number, is in [`logs/steering-log.md`](logs/steering-log.md).

```bash
pip install -r ../nominal_model/requirements.txt scipy
```

| Path | What it is |
|---|---|
| `clara_virtual.py` | Virtual CLARA. `VirtualMachine` holds a hidden beam and hidden errors (launch, quad and BPM offsets, shot jitter) and answers only with BPM centroids or one destructive screen per shot. `XsuiteLine` is the error-free model: maps by finite differences of xsuite tracking, quads set in A (+-100 A, bipolar) through `data/clara/lattice_data/utils.py`. |
| `steering/steering_inference.py` | Fits the launch and all 46 quad offsets by stepping each quad 5 A, with BPMs or screens only. Levenberg-Marquardt with tracked predictions, an energy-sliced Jacobian, and jitter as a correlated noise covariance. |
| `steering/svd_correction.py` | One-shot SVD orbit correction from the model's corrector to BPM response. |
| `steering/README.md` | Results, `fig_quad_offsets.png`, `fig_convergence.png`, `fig_svd_orbit.png`. |
| `errors_study/` | 200 seeds per point: quad offsets 0 to 300 um, BPM offsets 0 to 200 um. Covers residual orbit, IP centroid, beta-beat, dispersion, IP size and corrector kicks, plus side scans (jitter, shots, SVD cutoff, launch, reference tune). Raw `results/*.jsonl` are not committed; `run_scan.py` regenerates them. |

## Main results

- With all 31 BPMs, quad offsets of 99 um rms are recovered to 6 / 15 um rms (x / y). This holds when the fit only knows the nominal beam. The beam-shape mismatch biases the fitted energy offset instead (0.48 % vs 0.07 %).
- With screens only, offsets are recovered to a median of 15 to 25 um. They are poorly determined from S07 Q3 onwards, where few screens see each quad.
- One BPM shot of SVD correction takes the orbit from 2 to 4 mm rms to about 0.15 mm.
- BPM-only one-shot correction is a good first stage, but not enough for 80 to 400 um spots.
  - The IP centroid stays at 0.36 / 0.67 / 1.1 / 1.6 mm (median) for 0 / 100 / 200 / 300 um quad offsets.
  - Quad offsets dominate; BPM offsets up to 200 um barely matter.
  - Spurious dispersion of about 14 mm at the IP is the main optics effect.
  - The FEA and VBC correctors would need kicks they have never had in the archiver.
