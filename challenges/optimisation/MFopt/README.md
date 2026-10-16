# MFopt

Getting through a `SimEnv` episode (a new day on CLARA) to the requested spot at
`cla_feh_dia_scr_01` in at most 50 steps: steering first, then multi-fidelity
Bayesian optimisation between the model and the machine. The plan is in the
project notes (`mfopt/plan.md`).

## How an episode is solved

The policy to use is `calibrate.calibrated_mfbo` (`--policies calibrated_mfbo`),
drawn in
[`talk_figures/mfopt/mfopt_workflow_calibrated.svg`](../talk_figures/mfopt/mfopt_workflow_calibrated.svg).
The earlier workflow, stage A then MF-BO (`--policies mfbo`), is in
[`talk_figures/mfopt/mfopt_workflow.svg`](../talk_figures/mfopt/mfopt_workflow.svg). Model work in the
error-free twin costs no step; every shot on the machine is one step.

1. **Model, free.** The request (sigma_x, sigma_y) picks the target optics from the
   VBC-on node table and the orbit responses at that optics from the precomputed
   table (`responses.py`). The FEH correctors' setting that puts the twin's spot on
   the requested centroid (`aim_kicks`) is computed from the same table.
2. **Calibrate the day's beam, free** (`calibrate.py`). The reading `reset` returns
   (the lattice file's optics, a free reading) is the first data point of a MAP fit
   of seven numbers: the day's rms size and mismatch in each plane and its energy
   spread (the factors `SimEnv.reset` draws, with SimEnv's spreads as the prior), and
   an extra size in quadrature per plane for what the twin's clean orbit misses. The
   final focus (FEA Q15/Q16, FEH Q1/Q2) is re-matched for the calibrated beam.
3. **Load optics, aimed** (1 step): the re-matched quadrupoles with the FEH
   correctors already aimed at the request. They come after the last BPM, so aiming
   them changes nothing steering sees.
4. **Linear steering** (2 steps, more only if the beam is lost part way): thread the
   beam when BPMs are missing (R cropped to the BPMs that read and the correctors
   upstream of them), then two SVD shots of the orbit onto the twin's orbit. Each
   reading also refits the beam.
5. **Calibrated shots**: refit, re-match the final focus if the predicted size error
   is worth it, and re-aim the FEH correctors on the measured centroid, while the
   predicted gain is worth more than the score cost of one more step.
6. **MF-BO of the spot size, linear steering of its position** (`mfbo.py`), with the
   calibrated twin as low fidelity. The BO searches only the knobs that shape the
   spot: the five quadrupoles closest to the IP (FEA Q14, Q15, Q16, FEH Q1, Q2) and
   the two FEA sextupoles, about their current values. Position is left out of the
   search because it is linear in the correctors: on every BO shot the FEH
   correctors are set, from their known linear response, to cancel the centroid
   error the GP predicts, and that known kick is taken back out of the measured
   centroid before it reaches the GP.
   - Low fidelity: 256 Sobol points of the calibrated twin over the 7 knobs, in
     one batched Cheetah track, with a GP per readout (mu_x, mu_y, sigma_x, sigma_y, T).
   - High fidelity: the machine, modelled as the low-fidelity GP plus a discrepancy
     (a constant offset after the first reading, a GP from the third).
   - Each shot maximises the expected improvement of the benchmark mismatch. The
     loop stops when that improvement is worth less than one more step, and the
     episode ends by returning to the best reading. On days the calibration already
     solved, it stops at once.
7. **Score.** The benchmark score of the last reading of FEH-SCR-01: the mismatch in
   requested spot sizes plus 2 (1 - T), capped at 10, plus 0.5 log2(1 + steps / 50).

## Model settings

The environment and the twin are `SimEnv` defaults on master: Cheetah, every
drift, quadrupole and dipole at second order with second-order dipole edges (the
lattice file), VBC on at 131.7 mrad with edges, apertures on, L4H off (tracked as a
drift of its own length, `RLConfig.l4h_off`, as in `orbit_matching/`), and every
magnet limited to what +-100 A reaches.

## Target optics: `optics_env.py`

```python
from MFopt.optics_env import OpticsEnv

env = OpticsEnv()                        # a SimEnv; episodes are unchanged
optics = env.optics_for_target(150e-6, 250e-6)
k1, quad_sv, corrector_sv = optics.vectors

optics.quadrupoles.ranked(4)             # quadrupoles by their weight in the modes
optics.correctors.ranked(6)              # corrector planes, `_h` and `_v`
env.step(env.optics_action(optics))      # load the optics (one step)
```

- **Optics.** Starts from the 144 nodes rematched with the VBC on at 131.7 mrad
  (`optics_matching_vbc_on/`), interpolated in log size. Those were fitted with L4H
  powered and no magnet limits, so the four final-focus quadrupoles (FEA Q15/Q16,
  FEH Q1/Q2) are re-matched to the request in an error-free copy of the environment
  (see Model settings), within the magnet limits. Below about 0.13 mm in x the
  nodes need FEH Q2 above its limit (12.5 m^-2 at 100 A; the nodes go to 17.6), and
  no re-match within limits gets lower, even with all 12 downstream quadrupoles or
  the sextupoles free.
  `optics.achieved`, `optics.transmission` and `optics.clipped` say how well that went.
- **Quadrupole modes.** SVD of d ln(rms size) / d k1 at every screen and the readout,
  for all 46 quadrupoles. Rows and weights follow `SizeModes` (the readout carries
  half of the squared weight), but it is the beam-size response in this environment's
  model, so dispersion and the VBC are included.
- **Corrector modes.** The same for the centroid against all 58 corrector planes.
- **Scores.** An SVD has one singular value per mode, so each magnet also gets
  `scores[name] = sqrt(sum_i s_i^2 V_ij^2)` over the modes holding 99 % of the squared
  response. `ranked()` sorts by it.

Everything runs on the error-free model with the nominal beam: no environment steps.
The re-match tracks the beam once to just before FEA Q15 and then only the last few
dozen elements per evaluation, with the finite-difference Jacobian as one batched
track; the 208 finite-difference tracks of the modes start from the beam cached at the
diagnostic before each knob, batched per stretch. On one thread a target takes 2 to 3 s
without modes (was 20 s to 3.5 min) and about 30 s with them (was 2 to 5 min), and
results are cached per target.

## Steering responses and MF-BO knobs per spot: `responses.py`

```python
from MFopt.responses import DOF, ResponseTable

table = ResponseTable()                  # MFopt/data/responses_vbc_on.npz
at = table.at(150e-6, 250e-6)            # bilinear in log size between nodes
at["R"]      # centroid response: BPM x, BPM y, screen x, y  x  all 58 corrector planes
at["orbit"]  # the model's centroids at the same rows: what steering aims for
at["J"]      # readout (mu_x, mu_y, sigma_x, sigma_y, T)  x  DOF
```

- **MF-BO knobs (`DOF`).** The five quadrupoles closest to the IP (FEA Q14, Q15,
  Q16, FEH Q1, Q2), the five correctors per plane closest to it (FEA HVCOR 04, 05,
  06, FEH HVCOR 01, 02) and the two FEA sextupoles.
- **Table.** Built at every node of the VBC-on grid after the final-focus re-match,
  in the error-free twin: `k1`, `readout`, `clipped`, `feasible`
  (T >= 95 % and both sizes within 5 %), `relative_error`.
- **Twin.** The error-free twin tracks 5,000 particles (`OpticsEnv(twin_particles=...)`),
  half of the environment's 10,000, which keeps the rms sizes to about 1 %.
- **Rebuild.** `python -m MFopt.responses --workers 16` from `challenges/optimisation`,
  about 4 minutes on 16 cores. 105 of the 144 nodes are feasible (T >= 95 %, both
  sizes within 5 %). Inside the benchmark's 80 to 400 um range the misses are the
  nodes with sigma_x <= 0.162 mm, which need FEH Q2 beyond its limit (see above).
- **Check against `orbit_matching/`.** At the 105 feasible nodes our orbit response
  R agrees with Eiad's Cheetah responses (same BPMs and correctors, L4H a drift) to
  3.6 % median and 11 % at the 90th percentile (relative Frobenius norm). The rest
  is our re-matched final focus.

Stage A steering (`steering.py`) reads the orbit and spot responses from this table
instead of tracking them per episode.

## Running episodes: `run_baselines.py`

```bash
python -m MFopt.run_baselines --seeds 30 --workers 16
python -m MFopt.run_baselines --misalignment 5 --seeds 12 --tag lost_beam --policies optics_and_steering mfbo
```

One process per episode. Policies: `do_nothing`, `optics_only`, `optics_and_steering`
(stage A), `mfbo` (stage A, then MF-BO in `mfbo.py`), `calibrated` (beam
calibration alone) and `calibrated_mfbo` (calibration, then MF-BO: the one to use). `--misalignment 5` gives
days on which the beam is lost part way along the line; stage A then threads it
(`steering.thread`). Results go to `MFopt/results/<policy>[_<tag>].json`. MF-BO needs
`botorch`.

## Results (15 Oct evening): calibration + MF-BO, 100 fresh days

`calibrated_mfbo` on seeds 2000-2099 (default `RLConfig`: second order, L4H off, VBC
131.7 mrad, apertures, 100 A limits). Histograms of score, steps, wall time and screen
moves, and every episode's numbers, are in `talk_figures/mfopt/mfopt_100_episodes*`
(`talk_figures/scripts/mfopt_episode_stats.py` makes them from the run's JSON).

| | Mean | Median | 10th pct | 90th pct |
|---|---|---|---|---|
| Score | 0.40 | 0.29 | 0.16 | 0.72 |
| Steps | 8.7 | 7 | 4 | 14 |
| Wall time per episode (16 episodes in parallel) | 258 s | 239 s | 139 s | 409 s |
| Screen moves | 0 | 0 | 0 | 0 |

62 % of days end within 10 % of the request, 87 % within 25 %, 96 % within 50 %; no
day fails, the longest takes 42 steps, transmission is at least 91 %. The 29 days
asking for sigma_x below 0.135 mm average 0.51 (FEH Q2 limit); the other 71 average
0.35, median 0.25.

On the 30 test days (seeds 1000-1029), paired by seed:

| Policy | Mean | Median | Within 10 % | Steps |
|---|---|---|---|---|
| stage A | 0.59 | 0.47 | 17 % | 4.7 |
| stage A + MF-BO | 0.41 | 0.36 | 53 % | 7.9 |
| calibration | 0.41 | 0.34 | 47 % | 5.3 |
| calibration + MF-BO | 0.39 | 0.35 | 53 % | 9.6 |

Calibration reaches MF-BO's score in two to three fewer steps, and is better on the
days the final focus can reach (0.35 against 0.41); MF-BO finds a better compromise
on the small-x days (0.42 against 0.50). The combination takes the best of both.
Also tried, none better than MF-BO by more than its error on these 30 days: a Newton
correction of the size on the quadrupoles before BO (0.39 to 0.41), a jitter-robust
BO objective (0.40 to 0.46), a trust-region BO (0.61), a linear-mean discrepancy
(0.48), a GP on ln(sigma) (0.43), and choosing the final setting by posterior mean
(0.45).

What keeps the score above 0.2 on this environment: the shot-to-shot jitter of the
spot size (about 0.1 of score, see below), the steps (about 0.08 at 8 steps), and the
requests below 0.13 mm in x, about a third of all requests, which FEH Q2 cannot reach
within 100 A.

## Results (15 Oct, second order, L4H off, VBC 131.7 mrad)

| Days | Policy | Mean score | Median | Within 10 % | Within 25 % | Steps |
|---|---|---|---|---|---|---|
| 30 default (seeds 1000-1029) | do nothing | 9.30 | 10.00 | 0 % | 0 % | 0 |
| | optics only | 6.99 | 7.05 | 0 % | 0 % | 1 |
| | stage A | 0.59 | 0.47 | 17 % | 67 % | 4.7 |
| | stage A + MF-BO | 0.41 | 0.36 | 53 % | 87 % | 7.9 |
| 12 lost-beam (misalignment x5, seeds 1000-1011) | stage A | 0.62 | 0.65 | 0 % | 42 % | 6.2 |
| | stage A + MF-BO | 0.49 | 0.49 | 33 % | 83 % | 13.2 |

Every episode ends within 50 steps (longest 20) with the beam on the screen
(transmission at least 85 %). MF-BO improves on stage A by 0.18 +- 0.04 per day
(paired over the 30 days).

Two things limit it on this model, both worse than on the linear lattice it was
first tested on (mean 0.32, 77 % within 10 %):

- **Shot-to-shot jitter.** At fixed settings the readout sigma_x moves by about
  0.15 in ln(sigma) rms (175 to 270 um on a 200 um spot, seed 1000). It is SimEnv's
  per-shot redraw of the beam (`beam_jitter` 0.1: each plane's size within +-9 % and
  its mismatch within +-6 % at the start), which the optics amplifies to 10 to 14 %
  at the IP; the centroid and energy offsets contribute under 1 % (measured in the
  twin with each part of the redraw alone). The score is a single reading, so this
  costs about 0.1 whatever the tuner does. Treating
  the sizes as ln(sigma) in the GPs and ranking settings by posterior mean instead of
  by the best reading made no significant difference (+0.02 +- 0.02 and +0.04 +-
  0.03), so `mfbo.py` keeps the simpler scheme.
- **Small horizontal spots.** Requests below about 0.13 mm in x need FEH Q2 beyond
  its limit, so those days end with sigma_x too large.

## Log: energy discrepancy at the VBC

The lattice file's cavities bring the reference beam to 181.6 MeV at the VBC, while
`data/clara/lattice_data/utils.py` converts S05, VBC and S06 magnet currents at
`beam_energy` 196.6 MeV, about 8 % higher (see `nominal_model/README.md`). It does not
matter for MFopt: every knob here is set in model units (k1, k2, kick angle) and the
responses come from the same model, so no current is converted anywhere. It would
matter only when the settings are turned into currents for the real machine, where
the S05 to S06 magnets would come out about 8 % strong. Logged, not corrected.

## Batched Cheetah tracks: `batched.py`

`probe_batch(env, {knob: values}, base)` tracks N settings of the error-free twin in
one vectorised Cheetah pass and returns the same readings as N calls to `_probe`
(they agree to about 1e-7 relative). It runs on a private deep copy of the twin's
segment, so the twin that `_probe` and the re-match use is never left batched. The
MF-BO low fidelity (256 points) and every finite difference in `responses.py` go
through it: on the linear lattice a table node took 3 s instead of 15 s, the low-fidelity sample 1.5 s
instead of 10 s.
