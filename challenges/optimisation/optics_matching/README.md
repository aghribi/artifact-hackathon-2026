# CLARA spot-size optics

This directory packages the smooth quadrupole-strength family for desired RMS
beam sizes at **`cla_feh_dia_scr_01`**, the IP proxy. Both sizes range from
**0.1 to 20 mm**, independently. All Python API sizes are **metres**; quadrupole
`k1` values are **m⁻²**, rather than currents.

The saved configuration uses the nominal 200,000-particle bunch, **zero VBC
excitation, no wakefields, and L4H off**. Apply all **46** quadrupole settings:
42 share a retuned constant prefix; FEA Q15/Q16 and FEH Q1/Q2 vary with the
requested sizes. Using only the four variable settings on the nominal lattice
would give a different beam.

The default lattice file and the SimEnv now run the VBC at 131.7 mrad (R56 =
−57.6 mm) with L4H on, so this family does not reproduce its target sizes in the
default SimEnv; `MFopt/` re-matches the final focus for that case.

## Quick start

Run from the repository root using the existing environment:

```bash
nominal_model/.venv/bin/python -m optics_matching \
  --sigma-x-mm 1 --sigma-y-mm 3 --output /tmp/clara_1x3mm.json
```

This exports a complete **Cheetah-format lattice JSON**, with the saved geometry
and RF settings. It can be loaded by Cheetah or passed as `lattice_json` to
`nominal_model.xsuite_model.build_line(..., wakefields=False)`.

```python
from optics_matching import MatchedOptics

model = MatchedOptics()
tune = model.evaluate(1e-3, 3e-3, derivatives=True)
settings = tune['settings']                  # name -> k1, all 46 magnets
k1 = tune['k1']                              # order: model.quadrupoles
Jx = tune['d_k1_d_log_sigma_x']              # d k1 / d ln(sigma_x)
Jy = tune['d_k1_d_log_sigma_y']
# d_k1_d_sigma_x and d_k1_d_sigma_y are also returned, in m^-3.
```

### Xsuite

```python
from optics_matching import build_xsuite, apply_to_xsuite

converted = build_xsuite(1e-3, 3e-3, model=model)
line = converted.line
# Reference energy defaults to the saved nominal bunch energy.
# Or update an existing compatible no-wake, zero-VBC lattice:
apply_to_xsuite(line, 2e-3, 4e-3, model=model)
```

`build_xsuite` uses the repository's nominal converter, disables wakes, and
returns its `ConvertedLine` object. It requires the repository and the nominal
bunch archive for the default reference energy. Pass `initial_energy_ev` to
avoid reading that archive; the match is validated only at the nominal energy.
The factory does not build the tracker or create the particle bunch.

### Cheetah

```python
from optics_matching import build_cheetah, apply_to_cheetah

segment = build_cheetah(1e-3, 3e-3, model=model)  # float64 by default
apply_to_cheetah(segment, 2e-3, 4e-3, model=model)
```

The factory returns the same benchmark slice: after `cla_s02_sim_aper_01`,
through the IP proxy. It replaces disabled L4H by a drift. Optional `device`
and `dtype` arguments are forwarded to the segment. The adapter preserves each
strength tensor's dtype/device and supports nested segments.

**Backend scope:** both adapters apply identical magnetic strengths. The
full-bunch matching errors below were validated with Xsuite. Native Cheetah
transport uses different approximations, particularly for chromatic transport
and RF; these settings do not establish the same beam-size accuracy in native
Cheetah tracking. Use Xsuite to reproduce the quoted physical validation.

Both `apply_to_*` functions change quadrupole strengths only. Existing RF,
reference energy, VBC and wake settings must already match the saved family.
Missing magnets are rejected before changes are made. The lookup returns NumPy
values; it is not an end-to-end PyTorch graph, even when used with Cheetah.
Use the explicit derivatives above for sensitivity to requested sizes.

## Interpolation and accuracy

`data/strength_splines.npz` contains precompiled tensor-product cubic B-spline
knots and coefficients for **all 46 strengths**, in `ln(sigma_x), ln(sigma_y)`,
from the 23×23 auxiliary grid. The first 42 are held exactly constant.

A fixed six-step smooth, regularized Gauss–Newton correction adjusts the last
four strengths using the saved momentum-resolved bunch-moment model. This is
part of the delivered **C² parameterization**: it has no adaptive iteration
count or branch selection. It requires no lattice tracking at lookup time.
**Keep refinement enabled.** `evaluate(..., refine=False)` exposes the raw
splines for diagnostics; these alone can produce large off-grid spot errors
near a tight focus. No extrapolation outside the size domain is allowed.

The inherited validation covers:

| Check | Result |
|---|---|
| 144 requested nodes, full 200,000-particle Xsuite tracking | maximum relative size error 0.01639% |
| 23 off-grid full-bunch Xsuite checks | maximum relative size error 0.01345% |
| 2,025 corrected-surface model checks | maximum relative size error 1.01×10⁻⁹ |
| Dense quadrupole-body aperture check | maximum radius usage 0.960235 |

These are nominal-bunch results, not a machine-error tolerance study or a
proof of feasibility at every point in the continuous domain. Sizes are about
the centroid; no waist/alpha or IP-centroid target is imposed.

The circular aperture criterion is
`norm(mean_xy) + 3*sqrt(lambda_max(cov_xy)) <= radius`, with x–y correlations
included. Radii are **68 mm** for FEH Q1/Q2 and **20 mm** for every preceding
quadrupole. Body sampling is included in validation. These are envelope
constraints; the adapters do not insert particle-clipping apertures. Apertures
of other devices and screen field of view were not supplied.

**Quadrupole currents were not constrained to ±100 A.** The calibrated family
requires up to **148.667 A**; 77 of the 144 grid settings exceed 100 A. The earlier
±100 A limit applies to the chicane scan, not to this aperture-only matching
family. No clipping of interpolated strengths to current limits is performed.

## Optics response and saved compression effects

```python
from optics_matching import CompressionEffects

response = model.optics_response(1e-3, 3e-3)
A = response['weighted_response']  # 46 screen/plane rows × 46 k1 columns
compression = CompressionEffects().evaluate(0.8)
```

The response is `d ln(beta) / d k1`, calculated separately from the bunch RMS
matching objective. The IP carries half the total squared-error weight: each
IP-plane row has weight 1/4; each of the 44 upstream screen-plane rows has
weight 1/88. The two IP rows vary with the matched optics.

Compression is tabulated at 101 VBC strengths for this family's common prefix.
Its parameter is `u = theta/theta_max` in [-1,1], with coordinated bend signs
`(+,-,-,+)`; `theta_max = 0.116736694 rad`. Interpolation uses shape-preserving
PCHIP. Outputs include calibrated currents, bunch length, compression factor,
charge widths, peak current and skewness; see `data/compression_metadata.json`.
**This does not extend the spot-size match to nonzero VBC excitation.**

## Files and dependencies

- `matched_optics.py`, `runtime_model.py`, `response_surface.py`: portable lookup,
  smooth correction, and optics response. Require NumPy, SciPy and PyTorch.
- `lattices.py`: lazy-imported backend adapters and lattice export.
- `data/`: compiled splines, source interpolation grid, small moment/response
  bundles, anchor lattice, aperture and compression metadata. Keep together.
- `results/quadrupole_settings.csv`: all 144 original settings and tracked sizes.
- `results/validated_grid.npz`, `matching_grid.npz`: tracked/model grid arrays.
- `results/summary.json`, `off_grid_validation.*`, `*_validation.json`: numerical
  checks, including package parity and backend strength assignment checks.
- `results/matching_details.md`: detailed original matching report; its file
  references refer to the original archive in
  `inference/optics/matching/smooth_final_focus/`.
- `results/provenance.json`: original input hashes and matching provenance.

The lookup and Cheetah adapter can be copied with this directory; the Xsuite
factory additionally depends on `nominal_model/`. Reuse one `MatchedOptics`
instance for repeated evaluations to cache the numerical bundles.

## Symmetric target scan

Run `python -m optics_matching.plot_symmetric_scan` to plot all 46 strengths
against `sigma_x = sigma_y` from 0.1 to 20 mm (401 logarithmic samples).
The PNG, PDF and numerical CSV are saved as
`results/symmetric_target_strength_scan.*`; the lower panel shows the four
varying final-focus magnets separately.

## SVD modes versus both target sizes

`from optics_matching import SizeModes` loads the 36 resolved quadrupole modes
of the weighted fractional-beta response, parametrized by both target sizes.
The 49×49 grid, compiled splines, dominant-variable examples and validations
are described in [the 2D SVD report](results/size_svd/README.md).

Use `SizeModes().evaluate(sigma_x_m, sigma_y_m)` for sorted values and
orthonormal vectors. Keep its default analytic response refinement enabled;
raw response splines are inaccurate near sharp focuses. Local sign transport
and grouped projectors are supplied because verified degeneracies prevent a
globally smooth individually sorted vector field across the whole domain.
