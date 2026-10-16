# Error study: one-shot SVD orbit correction with BPMs only

Question: is a single BPM shot plus a truncated SVD good enough to steer CLARA online, given misaligned quadrupoles, BPM offsets and a launch error? Everything runs on the virtual CLARA in `../clara_virtual.py` (xsuite line `cla_s02_sim_aper_01` to `cla_feh_dia_scr_01`, lattice-file magnet strengths), reusing `../steering/svd_correction.py` (`corrector_response`, `svd_pseudo_inverse`).

## Method

- Per seed: launch (x, x', y, y') Gaussian (default 0.3 mm / 30 urad rms, one scan at 1 mm / 100 urad; no energy or time error unless stated), quad offsets (dx, dy) Gaussian per quad, BPM offsets Gaussian per BPM and plane. The same unit draws are scaled to each level, so the BPM-offset levels of one seed share launch and quad errors (paired comparison). BPM offset 0 is the perfect-BPM case, each other level is a fresh one-shot correction from zero correctors.
- Correction: ONE BPM shot (31 BPMs), kicks = -pinv(R) * orbit with the response R of the error-free model, relative cutoff 1e-3 on the singular values. That keeps **27 of the 58 modes** (the brief quoted 52 of 58, which is an absolute 1e-3; the code's relative cutoff gives 27). 29 correctors (2 FEH ones after the last BPM stay at zero).
- Noise: shot jitter (20 um, 2 urad, 20 um, 2 urad, 0, 2e-4) as in `svd_correction.py` (the `clara_virtual` default is zero jitter) and 10 um BPM noise.
- Beam: 3000 particles of the nominal start distribution, shifted so the 6D first moments equal the full 200k sample (otherwise the sample centroid alone is 1 mm off axis at the IP). The same sample is the machine beam, the model beam and the error-free reference.
- Truth metrics (scoring only, no jitter or noise): rms of the beam centroid at the BPMs and relative to the quad centres (x and y combined as sqrt(rms_x^2 + rms_y^2)), centroid position and angle at the IP screen relative to the error-free beam, sigma_x, sigma_y and projected emittance of the tracked beam at the IP relative to the error-free reference, beta-beat and dispersion from the linear map about the actual launch trajectory (`XsuiteLine.linear_map`, finite differences, start Twiss from `twiss_from_coordinates` of the start beam) at the 100 quads, BPMs, screens and IP, max kick and its ratio to `magnet_ranges.json`, transmission. Optics is at delta = 0, not energy-sliced; the energy spread enters only through the tracked sigma and emittance.
- Code: `error_study.py` (one line pair per process, errors reset per seed), `run_scan.py` (15 processes, `results/*.jsonl`), `analysis_utils.py`, `make_figures.py`, `make_reference_tune_lattice.py`. Ruff clean.

## Scan grid

| set | quad rms (um) | BPM rms (um) | launch | seeds |
|---|---|---|---|---|
| main | 0, 50, 100, 200, 300 | 0, 50, 100, 200 | 0.3 mm / 30 urad | 200 (4000 corrections) |
| big_launch | 50, 100, 200, 300 | 0, 50, 100, 200 | 1 mm / 100 urad | 100 |
| side scans, quad 100 (and 300 for some) | | 0, 50, 100, 200 | default | 100 each |

Side scans: no launch error, no jitter, 10 shots averaged, cutoff 1e-2 (6 modes) and 5e-5 (51 modes), launch dp/p 0.2 % rms, target = model centroid orbit (flatten onto the error-free predicted centroids instead of zero), correctors restricted to planes with observed range of at least 100 urad, and the `reference_tune.json` lattice (with and without the model-orbit target). Transmission is 1.0 in every run, also uncorrected: the xsuite line has no aperture, so no losses can occur.

## Results (main grid, median and (90th percentile), BPM-offset 0 / 100 / 200 um)

| quad rms (um) | BPM rms (um) | BPM rms before -> after (um) | in-quad rms after (um) | IP centroid shift (um) | IP angle shift (urad) | IP size change (%) | beta-beat IP (%) | max kick (urad) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 3031 -> 253 (520) | 260 (478) | 363 (633) | 335 (579) | 0.20 (0.42) | 0.37 (1.96) | 112 (276) |
| 0 | 100 | 3031 -> 267 (539) | 279 (497) | 364 (762) | 317 (645) | 0.21 (0.44) | 0.40 (2.03) | 124 (277) |
| 0 | 200 | 3031 -> 305 (566) | 323 (541) | 451 (892) | 378 (699) | 0.23 (0.53) | 0.42 (1.97) | 176 (316) |
| 50 | 0 | 3189 -> 263 (528) | 304 (497) | 465 (912) | 435 (817) | 0.22 (0.47) | 0.37 (1.97) | 151 (305) |
| 50 | 100 | 3189 -> 278 (546) | 318 (521) | 451 (894) | 450 (848) | 0.22 (0.51) | 0.39 (2.06) | 157 (312) |
| 50 | 200 | 3189 -> 316 (578) | 355 (554) | 525 (1048) | 480 (911) | 0.21 (0.57) | 0.43 (2.20) | 195 (333) |
| 100 | 0 | 5175 -> 297 (572) | 399 (580) | 668 (1403) | 661 (1284) | 0.22 (0.58) | 0.41 (1.97) | 222 (393) |
| 100 | 100 | 5175 -> 318 (578) | 404 (593) | 680 (1427) | 663 (1294) | 0.24 (0.65) | 0.43 (2.02) | 230 (405) |
| 100 | 200 | 5175 -> 362 (602) | 443 (623) | 706 (1408) | 711 (1312) | 0.25 (0.65) | 0.47 (2.28) | 248 (427) |
| 200 | 0 | 9446 -> 420 (687) | 633 (856) | 1120 (2419) | 1187 (2230) | 0.24 (1.02) | 0.67 (2.25) | 405 (690) |
| 200 | 100 | 9446 -> 433 (688) | 648 (869) | 1108 (2379) | 1185 (2280) | 0.24 (1.09) | 0.64 (2.47) | 399 (680) |
| 200 | 200 | 9446 -> 463 (721) | 660 (881) | 1150 (2396) | 1201 (2286) | 0.26 (1.16) | 0.69 (2.45) | 401 (673) |
| 300 | 0 | 14010 -> 565 (867) | 883 (1169) | 1568 (3283) | 1711 (3111) | 0.44 (1.81) | 1.07 (2.91) | 593 (976) |
| 300 | 100 | 14010 -> 572 (857) | 890 (1183) | 1565 (3236) | 1718 (3125) | 0.43 (1.88) | 1.02 (3.11) | 588 (972) |
| 300 | 200 | 14010 -> 584 (860) | 905 (1198) | 1570 (3232) | 1741 (3172) | 0.40 (2.20) | 1.04 (3.05) | 592 (988) |

Uncorrected, the BPM orbit is 3.0 mm (no quad error, launch only) to 14 mm rms (300 um quads), and the IP size grows by 1.6 to 80 %.

Optics after correction (quads 100 um, BPM 0 to 200 um): beta-beat 0.4 % at the IP (p90 2 %), max along the line 10 % (p90 26 %); spurious dispersion change at the IP about 14 mm in Dx or Dy (p90 25 mm) and D' about 13 mrad, almost independent of the error level (13 mm already with no quad errors, 20 mm at 300 um) because the correctors and the orbit through the quads themselves generate it. The beam has 2.1 % energy spread at the IP, so 14 mm of dispersion is about 0.3 mm of beam size: invisible against 3 x 25 mm, but comparable to or larger than the requested 80 to 400 um spots.

Correctors: median max kick 110 to 590 urad (p90 280 to 980 urad) for quads 0 to 300 um. Against `magnet_ranges.json` the planes in routine use (39 planes, range of 140 urad to 5 mrad) stay below their observed range in 99 to 100 % of seeds (largest kick is 0.14 to 0.43 of the range, median, and 0.33 to 0.77, p90). The other 19 planes (all 12 FEA, VBC, parts of S04/S07) have observed ranges below 100 urad (FEA about 0.1 urad) and the correction still drives all of them, with 0.3 to 1.0 mrad in the FEA correctors. Restricting the SVD to the planes in use doubles the IP centroid error (1.2 mm median at quad 100, 3.5 mm at 300) and puts 3 % (quad 100) to 52 % (quad 300) of seeds over a historical range.

## Figures

- `fig_hist_residual.png`: histograms over seeds of the post-correction BPM rms and in-quad rms. Top row: quad 100 um, BPM offset 0 to 200 um (barely shifts the distribution). Bottom row: BPM 100 um, quad offset 50 to 300 um (shifts and broadens it strongly).
- `fig_residual_vs_errors.png`: median (solid) and p90 (dashed) of BPM rms, in-quad rms and IP centroid shift against quad offset (top, one curve per BPM offset) and against BPM offset (bottom, one curve per quad offset). Quad errors dominate, the BPM curves are nearly flat.
- `fig_optics_change.png`: beta-beat at the IP and its maximum along the line, and the change of dispersion at the IP and its maximum along the line, against quad offset for each BPM offset (log y).
- `fig_ip_size_change.png`: absolute change of sigma_x, sigma_y and eps_y at the IP relative to the error-free beam, against quad offset (top) and BPM offset (bottom).
- `fig_kicks.png`: largest kick among the planes in use against quad offset, and the distribution of the observed operating ranges of those planes.
- `fig_side_scans.png`: in-quad rms, IP centroid shift and IP size change for the side scans at quad 100 um, BPM 0 and 100 um (bar = median, tick = p90).

Tables for every set: `summary_main.md/.csv`, `summary_<set>.md`.

## Verdict

BPM-only one-shot correction is a good coarse first stage but not sufficient on its own to centre the beam on the IP within the 80 to 400 um requested spot sizes.

| criterion | holds for | does not hold for |
|---|---|---|
| Orbit reduced 12 to 25 times, size growth below 1 % (p90 below 1.2 %), no extreme kicks (median max kick up to 0.4 mrad, below 45 % of the observed ranges) | quad offsets up to 200 um rms, BPM offsets up to 200 um rms | 300 um quads: p90 size growth 1.8 to 2.2 %, 27 % of seeds above 1 % |
| IP centroid within 0.7 mm (median), size change below 0.25 % | quad rms up to 100 um, any BPM offset up to 200 um | 200 um quads (1.1 mm), 300 um (1.6 mm) |
| IP centroid within 0.4 mm (about the largest requested spot, median) | quad rms below about 20 um only (median 0.36 mm already with perfect quads) | essentially every realistic case: at quad 100 um only 27 % of seeds are below 0.4 mm and 7 % below 0.2 mm |
| Beta-beat at IP below 1 % (p90 below 2.5 %) | quads up to 200 um | max beta-beat along the line reaches 10 % (p90 26 %) at 100 um |

What dominates: the quadrupole offsets. The IP centroid shift is about 0.36 mm at zero quad error (a floor set by shot jitter, the 27-mode truncation and the unmonitored last section), 0.47, 0.67, 1.1, 1.6 mm at 50, 100, 200, 300 um quad rms. A BPM offset of 100 um changes any metric by less than 5 % and 200 um by 5 to 25 % (in-quad rms 399 to 443 um at quad 100); BPM offsets are not the problem up to 200 um. The launch error is removed by the correction (0.3 mm to 1 mm launch raises the IP shift at quad 100 um from 0.67 to 0.85 mm, a dp/p launch of 0.2 % rms raises the median size change from 0.2 to 1.4 % because the orbit correction cannot remove energy errors). The reason is structural: flattening the BPM orbit leaves the beam off the quad centres: the in-quad rms is 400 um at quad 100 um (141 um combined x,y quad rms) and 1.2 to 1.6 times the BPM residual, which is what drives the IP error and the spurious dispersion; BPMs cannot see quad centres.

Improvements that were tested: averaging 10 shots recovers the no-jitter value (IP shift 0.67 to 0.56 mm, in-quad 0.40 to 0.33 mm), so online use should average a few shots. Cutoff 1e-3 is a good compromise: 1e-2 doubles the residual (IP 1.5 mm), 5e-5 needs 2.4 mrad kicks and gives 1.6 % size growth. Targeting the model-predicted centroid orbit instead of zero changes nothing for the lattice-file optics but matters for `reference_tune.json` (below).

Reference tune (`reference_tune.json`, FEA/FEH quads about zero, IP beam 83 x 7 mm, Dx = -0.45 m, quads 100 um): the uncorrected BPM orbit is 1.6 mm rms and the correction brings it to 120 to 240 um (BPM offset 0 to 200 um), with IP size change 0.03 to 0.1 % (negligible) and beta-beat at the IP 7 to 9 %. Flattening onto zero leaves a 1.65 mm IP shift (the error-free centroid is 1.7 mm off axis because of dispersion with the nonzero mean energy, so zero is the wrong target); targeting the model orbit gives 0.23 to 0.39 mm and 20 to 85 urad angle shift. Conclusion changes in magnitude, not in kind: size effects are negligible, centroid is limited by the quad offsets, and the target orbit has to come from the model.

Recommended use online: apply the one-shot correction (averaged over a few shots, correctors restricted by the hardware limits once known) as the first step, then follow with screen-based fine tuning of the last correctors for the position, and treat the spurious dispersion (about 14 mm) as part of the optics tuning for small spots.

## Caveats

- Quad offsets are Gaussian per plane; no roll, tilt, gradient errors, BPM gain or corrector calibration errors; correctors are exact and unlimited. A single iteration only.
- The model knows the exact start beam (same sample), so the model-orbit target and the response are optimistic; in reality the nominal distribution is only an estimate and the centroid at dispersive BPMs depends on the unknown mean energy.
- The lattice-file optics give a 3 x 25 mm IP beam with huge beta (up to 1e5 m): absolute centroid shifts are small against the beam size (2 to 7 % of sigma_y) and relative size changes of 0.2 to 2 % correspond to about 10 to 500 um. The requested 80 to 400 um spots need other optics, so these numbers do not carry over directly; dispersion is the quantity most likely to matter there.
- The xsuite line has no apertures, so zero losses is not evidence about aperture margins. The 2.1 % energy spread makes the finite-difference optics at delta = 0 incomplete; size and emittance come from tracking.
- The FEA correctors have about zero observed range and the real limit is unknown; the used-planes-only scenario above is the pessimistic case.
- 200 seeds (main) and 100 seeds (scans) give medians good to about 5 to 10 % and p90 to 15 %.
