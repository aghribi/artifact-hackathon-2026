# Inference work: handoff (14 October 2026)

Team Overfit & Proud. This folder moves the steering, tomography and 6D work from Francesco's laptop (`work/inference/`) into the repo, so it can continue in a cloud session. It sits on branch `overfit-proud/inference-wip`, cut from `origin/master` at 5fb0a7f.

## Environment

```bash
pip install -r ../nominal_model/requirements.txt scipy
```

Pinned versions: cheetah 0.8.4, xtrack 0.115.5, xpart 0.23.22, xobjects 0.7.1, xdeps 0.10.21. The first tracking call compiles a custom kernel, which takes a few seconds. A full start-to-readout line builds in about 10 s, or a few minutes with a cold kernel cache.

## What is here

| Folder | Status | Read first |
|---|---|---|
| `clara_virtual.py` | done | Virtual CLARA. `VirtualMachine` holds a hidden beam and errors and answers only with BPM centroids or one destructive screen per shot. `XsuiteLine` is the error-free model. Quads are set in A (+-100 A) through `data/clara/lattice_data/utils.py`. |
| `steering/` | done | `README.md`, `logs/steering-log.md`. Error fit by quad modulation, and one-shot SVD correction (`svd_correction.py`). |
| `errors_study/` | done | `README.md`. 200-seed Monte Carlo of BPM-only one-shot correction. Raw `results/*.jsonl` are not committed; rerun `run_scan.py`. |
| `tomography/` | done | `README.md`. 4D tomography from all screens on nominal optics, plus the extra S02 Q01/Q02 settings for eps_y. |
| `sixd/` | in progress | `README.md` (first part done). The full 6D reconstruction at the start was stopped mid-run for the move; see below. |
| `logs/` | | `steering-log.md` (14 Oct) and `beam-initial-conditions-log.md` (13 Oct). |

Generated `.npz`, `.pkl` and `.jsonl` files are not committed. Regenerate them:
- `sixd/make_s07_beam.py` and `tomography/make_s07_beam.py` make `s07_beam.npz`.
- `sixd/run_acquire.py` makes `images.npz`.

## Decisions already taken (Francesco)

- Track with xsuite. Cheetah is only a cross-check.
- The nominal starting beam is `nominal_model/initial_distribution.npz`.
- YAG screens are destructive. BPM studies assume the beam is transmitted.
- Tomography uses all screens on nominal optics, not quad scans. Quad changes are added only where something is unobservable.
- The VBC chicane is off: the 23 March scan has `CLA-VBC-MAG-DIP-01` SETI 0. The repo has no operating compressor angle.
- No AI attribution in commits or PRs.

## 6D reconstruction: where it stands

Goal: the full 6D beam at the benchmark start `cla_s02_sim_aper_01`, emittances included, chicane off.

Done before the move:
- `sixd/tdc.py`: a thin TDC kick with Panofsky-Wenzel. The voltage is assumed to be 5 MV, S-band 2998.5 MHz, vertical. The scan file has `CLA-TDC1-RF-CTRL-01` at 7.0 MW and phase 78 deg, but there is no power-to-voltage calibration yet.
- First 6D study at S07, moments only (`sixd/README.md`):
  - 21 moments need the TDC plus the FEA dispersion.
  - S07 Q01/Q02 x0.7 conditions the set.
  - sigma_z, sigma_delta, chirp and correlations come back; emittances do not, because the beam at S07 is nearly singular (corr 0.99994).
- `sixd/full6d.py`: a global forward model through the whole line, with configurations:
  - nominal, TDC off/+/-
  - three S02 Q01/Q02 settings (-11.3/26.3, -8.4/18.0, -9.2/16.0 A)
  - S07 Q01/Q02 x0.7, TDC off/+/-
- `run_acquire.py` acquired the virtual images; this is cached locally only and not committed.
- `observe_start.py` (observability of the 21 moments at the start with an exact tracked Jacobian) had started and has no result yet.

Next steps:
1. Finish `observe_start.py`: which of the 21 moments are observable at the start, and with which configurations.
2. Density reconstruction from images. The plan is sample reweighting (MENT-like): draw a broad 6D prior sample at the start, track it once per configuration with xsuite, then fit per-particle weights so the weighted, resolution-blurred projections match every image. Only S02 to S05 screens are imageable (under 4 mm rms) on nominal optics; consider an S07 optics change to make the S07/FEA screens imageable.
3. Validate against the true beam: 6D covariance, projected and slice emittances, sigma_z, sigma_delta, chirp, x-y coupling, and 2D projections. Report the number of images needed.
4. Add a section to `sixd/README.md` with the recipe (knob values and currents) and figures.

Plotting conventions: matplotlib default colour cycle, inferno for colormaps, small figures (about 4 x 2.5 in, up to 8 in wide when rectangular). No em dashes in any text.

## Update from the cloud session (14 October, evening)

Branch `claude/project-thread-vjmyln` (cut from `overfit-proud/inference-wip`). The 6D reconstruction at S02 now runs end to end; results and recipe are in `sixd/README.md`, section "6D reconstruction at the S02 start from images".

- TDC at 5.92 MV (7.0 MW, assumed 5 MOhm). Designed S07/FEA quad settings in `sixd/optics_design.json`.
- Recovered: sigma_x, sigma_y (2 %), sigma_z (+8 %), sigma_delta (-7 %), core emittances (within 20 %), x-y coupling.
- Not recovered: chirp sign (two minima), slice energy spread and eps_z. rms eps_x is set by a 1 % halo outside the screens.
- Environment: set `XSUITE_ALLOW_KERNEL_COMPILATION=1` without the `xsuite` package. `common.py` shares compiled kernels between lines in a process (the first build costs 3 to 5 min).
- Fixed: `analyse_start.py` (undefined `config_of` after the lint change), `common.py` import path for `xsuite_model`.
