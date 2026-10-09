# Physics formula card — "Find the physics that helps"

A starting kit for a team, not a derivation task. Compute these quantities from the settings,
add them as inputs or use them to scale the target, and measure whether the score at 100 and
300 simulations moves. Everything here is implemented in `physics_formulas.py` (numpy only;
`python physics_formulas.py` runs its self-checks against the pack).

## Knobs → density profile

The public tables ship the density profile per configuration (`x_p` in µm, `n_e_p` in m⁻³);
**the test files ship only the knobs**, so rebuild the profile from them. A profile feature
that cannot be rebuilt from the knobs is useless on the hidden test.

```python
import numpy as np
from physics_formulas import (species_A, species_B, electrons_background,
                              electrons_injected, m3_to_cm3, LASER,
                              focus_position_um, a0_along_z)

# Campaign A: six fixed nodes, straight lines in between; knobs p_1 (mbar), c_N2
x = np.linspace(0, 5265, 500)                      # um, the shipped nodes span 65..5265
n_He, n_N = species_A(x, p_1, c_N2)                # atoms per m^3

# Campaign B: the organisers' generator (Fermi edges); knobs P_max (Pa), cN2_max, L_inj (mm), dip_frac
x = np.linspace(0, (L_inj + 3.2) * 1e3, 2000)      # um, the shipped x_p axis
n_He, n_N = species_B(x, P_max, cN2_max, L_inj, dip_frac)

n_e_background = electrons_background(n_He, n_N)   # 2*n_He + 5*n_N: He fully, N to 5+ -- the wake's plasma
n_e_injected   = electrons_injected(n_N)           # 2*n_N: the two inner N electrons, freed at the laser peak
n_e_cm3        = m3_to_cm3(n_e_background)         # the formulas below take cm^-3

# Laser along the same axis (Campaign B shown; for A use LASER["A"] and the row's a_0)
x_focus = focus_position_um("B", x_of)             # focal position on the x_p axis, um
a_z     = a0_along_z(x, LASER["B"]["a0"], LASER["B"]["w0_um"], x_focus)
```

**What the shipped curve is.** On the public rows `n_e_p` equals 2 × n_He only (helium, with
the local nitrogen fraction already taken out); add the nitrogen parts yourself as above.
In Campaign A the plateau sits at 1.006 × 2 p₁/k_BT at 300 K (`species_A` reproduces every
node on every public row to 2 × 10⁻⁷ with that factor). In Campaign B the fixed geometry is hard-coded as `GEO_B`; `species_B` reproduces `n_e_p` on all public rows (relative residual 4 × 10⁻⁷). The acceleration plateau sits at (1 − 0.3 cN2_max) × 2 P_max/k_BT.
Alternative without any formula: interpolate `n_e_p` over the knobs from the public rows.

## Laser parameters

| campaign | a₀ | w₀ | t_L (FWHM in intensity) | FGB order | focus origin on `x_p` |
|---|---|---|---|---|---|
| A | per row (knob `a_0`) | 19 µm | 35 fs | 5 | 1865 µm + `x_of` |
| B | 1.25 | 19 µm | 40 fs | 5 | 2900 µm + `x_of` |

Read from the Smilei namelists of the two campaigns. w₀ is the waist parameter of a
flattened-Gaussian beam of order 5, not the 1/e² radius of a plain Gaussian, so formula 3 with
w₀ = 19 µm is the Gaussian approximation of that beam. `focus_position_um(campaign, x_of)`
applies the origin; `LASER` and `FOCUS_ORIGIN_UM` hold the numbers.

## Symbols and constants

λ₀ = 0.8 µm (laser wavelength); n_c = 1.1 × 10²¹ / λ₀[µm]² ≈ 1.7 × 10²¹ cm⁻³ (critical
density); m_e c² = 0.511 MeV.
n_e is the electron density along the accelerator, from `n_e_p` or from `species_A` /
`species_B` (background + injected as the formula needs). **Note the unit:** m⁻³ in the pack,
cm⁻³ in the formulas below, a factor 10⁻⁶ (`m3_to_cm3`).
a₀ is the normalised laser strength (a₀ = 1 means the electrons' quiver motion is
relativistic), w₀ the focal spot radius, τ_L the pulse duration, x_of the focal position
along z.

## The formulas

| | Quantity | Expression | What it is for | Function |
|---|---|---|---|---|
| 1 | Plasma wavelength | λ_p[µm] ≈ 3.34 × 10¹⁰ / √(n_e[cm⁻³]), k_p = 2π/λ_p | the length scale of the wave; everything below is in units of it | `plasma_wavelength_um`, `k_p_per_um` |
| 2 | Wave-breaking field | E₀[V/m] ≈ 96 √(n_e[cm⁻³]) | the strongest field the wave can carry; the accelerating field is a fraction η of it | `wave_breaking_field_V_per_m` |
| 3 | Laser strength along z (vacuum focusing) | a(z) = a₀ / √(1 + ((z − x_of)/z_R)²), z_R = π w₀²/λ₀ | turns the focus knob into a field at each position; guiding in the plasma makes the real decay slower | `a0_along_z` |
| 4 | Matched spot and bubble radius | k_p w₀ ≈ 2√a₀, R_b ≈ 2√a₀ / k_p | whether the laser is matched to the plasma; R_b enters the loading limit (7) | `matched_spot_um`, `bubble_radius_um` |
| 5 | Energy gain, simple form | ΔE(z) = e η ∫₀^z E₀(n_e(z′)) dz′, η a fraction to fit | the textbook estimate along the actual density profile; a cheap first test as an input or as a target scale | `energy_gain_MeV` |
| 6 | Pump depletion length | L_pd ≈ (n_c / n_e) c τ_L | where the laser is spent; whether it is reached in these simulations is not established — compare it with the profile length first | `pump_depletion_length_um` |
| 7 | Beam loading | E_z,eff = E_z (1 − Q/Q_s), Q_s ∝ (k_p R_b)⁴ / √n_e (Tzoufras 2008) | a heavy bunch flattens the field it rides on; the first candidate for what formula 5 leaves out | `beam_loading_factor` |

## Three ways to use them, cheapest first

**(a) Extra inputs.** Add columns λ_p, E₀, a(z), L_pd and ΔE(z) from (5) beside the settings
and z. Train the same model with and without them at 100 and 300 simulations (ten random
subsets each). The result is the difference and its spread, not one number.

**(b) Scaled target.** Predict E / ΔE(z) instead of E. The formula carries the trend and
extrapolates above the density cut; the network learns the deviation. This is the version to
try on the out-of-distribution objective.

**(c) Beam loading.** Fit Q_s (or a per-configuration η) as a function of the settings, then
put (7) into (5). Caveat from the pack: a term that needs the charge must predict the charge
too, since the hidden test gives only the settings.

## Pitfalls

- A profile feature that cannot be rebuilt from the knobs is useless on the hidden test.
- a₀ in Campaign B is fixed (1.25); the focus x_of is the knob that moves the field. x_of is
  measured from a campaign-specific origin on the `x_p` axis (1865 µm in A, 2900 µm in B):
  use `focus_position_um`, do not add x_of to zero.
- The density profile has a dip (`dip_frac`) and an injection length (`L_inj`): integrate (5)
  along the actual profile, not with a plateau value.
- Units: the pack is in m⁻³ and µm; the formulas take cm⁻³. Keep the conversion at one place.
- Dephasing is not on this card: in these simulations the laser is tied to the moving window
  and the bunch does not outrun the wave over the simulated length.
- Depletion (6) is not established in this data: compute L_pd and compare it with the profile
  length before building on it.

## References

- E. Esarey, C. B. Schroeder, W. P. Leemans, *Rev. Mod. Phys.* **81**, 1229 (2009) — formulas 1–3, 5.
- W. Lu *et al.*, *Phys. Rev. ST Accel. Beams* **10**, 061301 (2007) — 4, 6.
- M. Tzoufras *et al.*, *Phys. Rev. Lett.* **101**, 145002 (2008) — 7.
