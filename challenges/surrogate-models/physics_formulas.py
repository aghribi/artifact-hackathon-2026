"""
physics_formulas.py -- the PALLAS surrogate challenge's physics starting kit.

Two things live here:

  1. Knobs -> density profile.  The public tables ship the gas-density curve
     (`x_p` in um, `n_e_p` in m^-3) per configuration; the hidden test files
     ship only the knobs.  `species_A` and `species_B` rebuild the curve from
     the knobs, and `electrons_background` / `electrons_injected` turn the two
     gas species into electron densities.

  2. Seven textbook formulas of laser-wakefield acceleration (plasma
     wavelength, wave-breaking field, laser strength along z, matched spot and
     bubble radius, energy gain, pump-depletion length, beam loading), each a
     small numpy function with the unit in its name or its docstring.

numpy only.  Nothing is loaded at import.  `python physics_formulas.py` runs the
self-checks (they look for the data pack, see `_find_pack`) and prints one
worked example.  The companion note PHYSICS.md says what the formulas are for.

Units: the pack gives positions in um and densities in m^-3.  Inside the
profile functions positions are in mm (the generator's unit); the conversion
happens at the function boundary.  The wakefield formulas take n_e in cm^-3,
as in the textbooks: use `m3_to_cm3` to get there.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_B = 1.380649e-23          # Boltzmann constant, J/K
T_GAS = 300.0               # gas temperature assumed by the profile generator, K
LAMBDA0_UM = 0.8            # laser wavelength, um
N_CRIT_CM3 = 1.1e21 / LAMBDA0_UM**2   # critical density for lambda0, cm^-3 (about 1.7e21)
M_E_C2_MEV = 0.511          # electron rest energy, MeV
C_UM_PER_FS = 0.299792458   # speed of light, um per fs

_KT = K_B * T_GAS           # J; P / kT is the molecule density in m^-3

# Laser parameters of the two campaigns, read from their Smilei namelists.
# t_L is the pulse duration, FWHM in INTENSITY, fs.  w0 is the waist parameter
# at focus of a flattened-Gaussian beam of order N = 5 (Smilei
# LaserEnvelopeGaussianAM / FBPIC FlattenedGaussianLaser), NOT the 1/e^2 radius
# of a plain Gaussian, so `a0_along_z` with w0 = 19 um is the Gaussian
# approximation of that beam.
LASER = {
    # RandomScan 2022 deck namelist_env_ii_4p7-rps.py; FBPIC port
    # phase40_fbpic_one_run_sim.py:61-63 and kit corpus table phase40_kit_corpus.py:33
    "A": dict(t_L_fs=35.0, w0_um=19.0, fgb_order=5, a0="per row (knob a_0)"),
    # namelist_analyticalpressureprofile.py:330-332, 361 (t_L = 40e-15 s, N = 5, w_0 = 19e-6 m)
    "B": dict(t_L_fs=40.0, w0_um=19.0, fgb_order=5, a0=1.25),
}

# Where the focus knob x_of is measured from, on the shipped x_p axis (um).
# Focus position on x_p = FOCUS_ORIGIN_UM + x_of.
FOCUS_ORIGIN_UM = {"A": 1865.0, "B": 2900.0}


def focus_position_um(campaign, x_of_um):
    """Focal position on the shipped x_p axis, um: FOCUS_ORIGIN_UM[campaign] + x_of.

    campaign is "A" or "B"; x_of_um is the focus knob `x_of` of the row, um.
    Use the result as `x_focus_um` in `a0_along_z`, so that x_of is never added
    to the wrong origin.
    """
    return FOCUS_ORIGIN_UM[campaign] + np.asarray(x_of_um, float)


def m3_to_cm3(n_m3):
    """Convert a density from m^-3 (the pack's unit) to cm^-3 (the formulas' unit)."""
    return np.asarray(n_m3, float) * 1e-6


# ---------------------------------------------------------------------------
# Elementary shape: the Fermi (logistic) edge used by the profile generator
# ---------------------------------------------------------------------------
def fermi_edge(x, x_edge, width, rising=True):
    """Smooth step centred at `x_edge` with transition width `width`.

    rising=True : 0 below the edge -> 1 above it.
    rising=False: 1 below the edge -> 0 above it.
    `x`, `x_edge`, `width` share one unit (mm in the profile functions).
    """
    width = max(float(width), 1e-9)
    z = np.clip((np.asarray(x, float) - x_edge) / width, -80.0, 80.0)
    s = 1.0 / (1.0 + np.exp(-z))
    return s if rising else 1.0 - s


# ---------------------------------------------------------------------------
# Campaign A: six fixed nodes, linear in between
# ---------------------------------------------------------------------------
X_NODES_A_UM = np.array([65.0, 1065.0, 1565.0, 1865.0, 2765.0, 5265.0])  # the same for every row
GAS_SHAPE_A = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0])      # gas density shape on the nodes
N2_SHAPE_A = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])       # where the nitrogen is (injection zone)

# Read off the public pack: median over all Campaign A rows of
# max(n_e_p) / (2 * p_1*100 / kT) = 1.006079 (identical on every row).
PLATEAU_FACTOR_A = 1.006079


def species_A(x_um, p_1_mbar, c_N2):
    """Campaign A gas profile from its two gas knobs.

    Parameters
    ----------
    x_um     : positions along the accelerator, um (array or scalar).
    p_1_mbar : plateau pressure knob `p_1`, mbar.
    c_N2     : nitrogen knob `c_N2`, the N2 molecular fraction in the injection zone.

    Returns
    -------
    (n_He, n_N) : helium atom density and nitrogen ATOM density, both in m^-3
                  (one N2 molecule gives two N atoms).

    The shape is six fixed nodes with straight lines in between: gas on
    [0, 1, 1, 1, 1, 0], nitrogen fraction c_N2 on [1, 1, 1, 0, 0, 0].  The
    public `n_e_p` equals 2 * n_He (helium only) to seven digits with
    PLATEAU_FACTOR_A; the nitrogen parts are added by `electrons_*`.
    """
    x = np.asarray(x_um, float)
    n = p_1_mbar * 100.0 / _KT * PLATEAU_FACTOR_A          # mbar -> Pa -> molecules per m^3
    gas = np.interp(x, X_NODES_A_UM, GAS_SHAPE_A)
    f_N2 = np.interp(x, X_NODES_A_UM, N2_SHAPE_A) * c_N2  # N2 molecular fraction
    n_He = (1.0 - f_N2) * gas * n
    n_N = 2.0 * f_N2 * gas * n
    return n_He, n_N


# ---------------------------------------------------------------------------
# Campaign B: the organisers' region-based generator with Fermi edges
# ---------------------------------------------------------------------------
# Fitted on the public rows, 2026-10-09: least squares of 2*n_He(x_p) against
# n_e_p over 200 rows spread across the knob ranges (scipy least_squares, all
# values free, including P_min, an overall factor and the generator's
# he_smooth blend).  Every value converged to a round number; with the values
# below, max_x |2*n_He - n_e_p| / max(n_e_p) over ALL 5286 public rows is
# median 3.9e-7, p90 4.1e-7, max 4.3e-7, so the geometry IS the generator's,
# not an approximation.
#
#   * Positions are in mm on the shipped x axis (x_p = 0 is 0.5 mm before the
#     target starts); the x grid ends at L_inj + 3.2 mm, i.e. the target
#     (0.4 + L_inj + 0.3 + 1.2 + 0.3 = L_inj + 2.2 mm) sits between two 0.5 mm
#     pads.
#   * Only L_acc + L_outramp/2 = 1.35 mm is determined by the curves (the
#     out-edge is centred in the out-ramp).  The split 1.2 / 0.3 is the one
#     for which the out-ramp has the same sharpness as the in-ramp
#     (width = length/4 on both sides) and the two pads are equal.
#   * The factor is 1.0: no plateau correction is needed once the he_smooth
#     term is in.  Without it the residual is 1.5e-2 rms and 4e-2 max, so the
#     term is kept.  Its effect: on the acceleration plateau, where the
#     nitrogen has gone, the pressure is lowered to (1 - 0.3*cN2_max) * P_max.
GEO_B = dict(
    x0=0.5,          # target start on the shipped axis, mm
    L_inramp=0.4,    # in-ramp length, mm
    L_outlet=0.3,    # outlet (dip) length, mm
    L_acc=1.2,       # acceleration plateau length, mm (see note on the split)
    L_outramp=0.3,   # out-ramp length, mm
    w_in=0.1,        # Fermi width of the in-edge, mm   (= L_inramp / 4)
    w_out=0.075,     # Fermi width of the out-edge, mm  (= L_outramp / 4)
    w_dip=0.1,       # Fermi width of the dip corners, mm
    w_cN2=0.018,     # Fermi width of the nitrogen roll-off, mm
    k_off=3.0,       # the roll-off centre sits k_off * w_cN2 after the end of injection
    P_min=100.0,     # pressure floor at the ramp feet, Pa
    factor=1.0,      # overall plateau factor (none needed)
    he_smooth=0.3,   # generator's helium-step smoothing blend, 0 = off, 1 = full
)


def species_B(x_um, P_max_Pa, cN2_max, L_inj_mm, dip_frac, geo=GEO_B):
    """Campaign B gas profile from its four knobs.

    Parameters
    ----------
    x_um      : positions on the shipped axis, um (the pack's `x_p` runs from 0
                to L_inj + 3200 um; any array is accepted).
    P_max_Pa  : plateau pressure knob `P_max`, Pa.
    cN2_max   : nitrogen knob `cN2_max`, the N2 molecular fraction in the injection zone.
    L_inj_mm  : injection-plateau length knob `L_inj`, mm.
    dip_frac  : outlet dip knob `dip_frac`: the dip goes down to (1 - dip_frac) * P_max.
    geo       : the fixed geometry (GEO_B); pass a modified copy to experiment.

    Returns
    -------
    (n_He, n_N) : helium atom density and nitrogen ATOM density, both in m^-3.

    Construction (the organisers' generator, regions head to tail):
        in-ramp | injection (L_inj) | outlet dip | acceleration | out-ramp
    A Fermi in-edge lifts a P_min floor to the plateau and a Fermi out-edge
    drops it back; the dip is a window of depth dip_frac subtracted between the
    end of injection and the start of acceleration.  The nitrogen fraction is
    flat at cN2_max up to the end of injection and then rolls off with one
    Fermi edge.  The generator's `he_smooth` term lowers the pressure where
    the nitrogen has gone, so that n_He has no step at the roll-off.
    The public `n_e_p` equals 2 * n_He (helium only); see `electrons_*`.
    """
    g = geo
    x = np.asarray(x_um, float) * 1e-3                       # um -> mm, the generator's unit
    x1 = g["x0"] + g["L_inramp"]                              # start of injection
    x2 = x1 + L_inj_mm                                        # end of injection
    x3 = x2 + g["L_outlet"]                                   # end of the dip
    x4 = x3 + g["L_acc"]                                      # end of acceleration
    x5 = x4 + g["L_outramp"]                                  # target end

    plateau = fermi_edge(x, 0.5 * (g["x0"] + x1), g["w_in"], rising=True) * \
              fermi_edge(x, 0.5 * (x4 + x5), g["w_out"], rising=False)
    notch = fermi_edge(x, x2, g["w_dip"], rising=True) * \
            fermi_edge(x, x3, g["w_dip"], rising=False)       # 1 inside the outlet, 0 outside
    shape = np.clip(plateau - dip_frac * notch, 0.0, 1.0)
    P = g["P_min"] + (P_max_Pa - g["P_min"]) * shape          # Pa

    c_N2 = cN2_max * fermi_edge(x, x2 + g["k_off"] * g["w_cN2"], g["w_cN2"], rising=False)

    hs = g["he_smooth"]
    if hs > 0.0:
        safe = np.clip(1.0 - c_N2, 1e-6, 1.0)
        P = P * (1.0 - hs + hs * (1.0 - cN2_max) / safe)

    n = P / _KT * g["factor"]                                 # molecules per m^3
    n_He = (1.0 - c_N2) * n
    n_N = 2.0 * c_N2 * n
    return n_He, n_N


# ---------------------------------------------------------------------------
# From gas atoms to electrons
# ---------------------------------------------------------------------------
def electrons_background(n_He, n_N):
    """Electron density of the wake's plasma, m^-3: 2*n_He + 5*n_N.

    Before the laser peak arrives the leading edge of the pulse has already
    stripped helium fully (2 electrons) and nitrogen to N5+ (5 electrons).
    This is the density that sets the plasma wavelength and the fields.
    """
    return 2.0 * np.asarray(n_He, float) + 5.0 * np.asarray(n_N, float)


def electrons_injected(n_N):
    """Electron density available for injection, m^-3: 2*n_N.

    The two inner-shell electrons of nitrogen (N5+ -> N7+) are freed only near
    the laser peak, inside the wake, and form the injected bunch.
    """
    return 2.0 * np.asarray(n_N, float)


# ---------------------------------------------------------------------------
# The seven formulas (n_e in cm^-3, lengths in um, fields in V/m)
# ---------------------------------------------------------------------------
def plasma_wavelength_um(n_e_cm3):
    """1. Plasma wavelength, um: lambda_p = 3.34e10 / sqrt(n_e[cm^-3]).

    The length scale of the wave; the other quantities are in units of it.
    Zero density gives inf.
    """
    n = np.asarray(n_e_cm3, float)
    with np.errstate(divide="ignore"):
        return 3.34e10 / np.sqrt(n)


def k_p_per_um(n_e_cm3):
    """1. Plasma wavenumber, 1/um: k_p = 2*pi / lambda_p."""
    return 2.0 * np.pi / plasma_wavelength_um(n_e_cm3)


def wave_breaking_field_V_per_m(n_e_cm3):
    """2. Cold wave-breaking field, V/m: E_0 = 96 * sqrt(n_e[cm^-3]).

    The strongest field the wave can carry; the accelerating field is a
    fraction eta of it.
    """
    return 96.0 * np.sqrt(np.asarray(n_e_cm3, float))


def a0_along_z(z_um, a0, w0_um, x_focus_um):
    """3. Laser strength along z for vacuum focusing (dimensionless).

    a(z) = a0 / sqrt(1 + ((z - x_focus) / z_R)^2),  z_R = pi * w0^2 / lambda0.
    z_um in um on the shipped x_p axis; x_focus_um the focal position on that
    same axis, from `focus_position_um(campaign, x_of)`; w0_um the focal spot
    radius in um.  With the campaigns' w0 = 19 um (LASER) this is the Gaussian
    approximation of the flattened-Gaussian beam of order 5 the decks use.
    Turns the focus knob into a field at each position; guiding in the plasma
    makes the real decay slower than this.
    """
    z_R = np.pi * w0_um**2 / LAMBDA0_UM                       # Rayleigh length, um
    return a0 / np.sqrt(1.0 + ((np.asarray(z_um, float) - x_focus_um) / z_R)**2)


def matched_spot_um(a0, n_e_cm3):
    """4. Matched spot radius, um: k_p * w0 = 2 * sqrt(a0)  ->  w0 = 2*sqrt(a0)/k_p.

    Whether the laser spot is matched to the plasma.
    """
    return 2.0 * np.sqrt(a0) / k_p_per_um(n_e_cm3)


def bubble_radius_um(a0, n_e_cm3):
    """4. Bubble (blow-out cavity) radius, um: R_b = 2 * sqrt(a0) / k_p.

    Enters the beam-loading limit (formula 7).
    """
    return 2.0 * np.sqrt(a0) / k_p_per_um(n_e_cm3)


def energy_gain_MeV(z_um, n_e_cm3_along_z, eta):
    """5. Energy gain along z, MeV: dE(z) = e * eta * integral_0^z E_0(n_e(z')) dz'.

    z_um : positions, um, increasing.  n_e_cm3_along_z : the electron density
    at those positions, cm^-3.  eta : the fraction of the wave-breaking field
    that accelerates (a number to fit).  Returns the cumulative trapezoid, one
    value per z, starting at 0.  Integrate along the actual profile, not with
    a plateau value.
    """
    z_m = np.asarray(z_um, float) * 1e-6                      # um -> m
    E = eta * wave_breaking_field_V_per_m(n_e_cm3_along_z)    # V/m
    steps = 0.5 * (E[1:] + E[:-1]) * np.diff(z_m)             # eV gained per step (e * E * dz)
    return np.concatenate([[0.0], np.cumsum(steps)]) * 1e-6   # eV -> MeV


def pump_depletion_length_um(n_e_cm3, tau_fs):
    """6. Pump-depletion length, um: L_pd = (n_c / n_e) * c * tau_L.

    tau_fs : laser pulse duration, fs.  Where the laser is spent.  Whether it
    is reached in these simulations is not established; compare it with the
    profile length first.
    """
    n = np.asarray(n_e_cm3, float)
    with np.errstate(divide="ignore"):
        return N_CRIT_CM3 / n * C_UM_PER_FS * tau_fs


def beam_loading_factor(Q, Q_s):
    """7. Beam-loading factor (dimensionless): E_z,eff = E_z * (1 - Q / Q_s).

    Q : bunch charge, Q_s : the charge that flattens the field completely,
    Q_s proportional to (k_p R_b)^4 / sqrt(n_e)  (Tzoufras et al., PRL 101,
    145002, 2008).  Any consistent charge unit.  A heavy bunch flattens the
    field it rides on; the first candidate for what formula 5 leaves out.
    """
    return 1.0 - np.asarray(Q, float) / Q_s


# ---------------------------------------------------------------------------
# Self-checks (run only as a script; the data pack is needed for the first two)
# ---------------------------------------------------------------------------
def _find_pack():
    """The pack directory: $PALLAS_PACK, else data/pallas_hackathon_data, else data/sample."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.environ.get("PALLAS_PACK"),
                 os.path.join(here, "data", "pallas_hackathon_data"),
                 os.path.join(here, "data", "sample")):
        if cand and os.path.isfile(os.path.join(cand, "campaign_B.parquet")):
            return cand
    return None


def _check_A(pack):
    import pandas as pd
    df = pd.read_parquet(f"{pack}/campaign_A.parquet", columns=["config", "p_1", "c_N2", "x_p", "n_e_p"])
    factors, errs = [], []
    for r in df.itertuples():
        x, ne = np.asarray(r.x_p), np.asarray(r.n_e_p)
        factors.append(ne.max() / (2.0 * r.p_1 * 100.0 / _KT))
        n_He, _ = species_A(x, r.p_1, r.c_N2)
        nz = ne > 0
        errs.append(max(np.max(np.abs(2 * n_He[nz] - ne[nz]) / ne[nz]), np.max(np.abs(n_He[~nz]))))
    print(f"[A] {len(df)} rows: plateau factor median {np.median(factors):.6f} "
          f"(coded {PLATEAU_FACTOR_A}); max relative error of 2*n_He at the nodes {max(errs):.2e}")


def _check_B(pack):
    import pandas as pd
    df = pd.read_parquet(f"{pack}/campaign_B.parquet",
                         columns=["config", "P_max", "cN2_max", "L_inj", "dip_frac", "x_p", "n_e_p"])
    res = []
    for r in df.itertuples():
        x, ne = np.asarray(r.x_p), np.asarray(r.n_e_p)
        n_He, _ = species_B(x, r.P_max, r.cN2_max, r.L_inj, r.dip_frac)
        res.append(np.max(np.abs(2 * n_He - ne)) / ne.max())
    res = np.array(res)
    print(f"[B] {len(df)} rows: max_x |2*n_He - n_e_p| / max(n_e_p): median {np.median(res):.2e}, "
          f"p90 {np.percentile(res, 90):.2e}, max {res.max():.2e}")
    if res.max() > 0.03:
        worst = np.argsort(res)[::-1][:5]
        print("[B] residual above 3 % -- worst five configs:")
        for i in worst:
            print(f"     config {df.config.iloc[i]}  residual {res[i]:.3f}")


def _worked_example():
    # One public Campaign B row (config 1065): its knobs, as the test files would give them.
    P_max, cN2_max, L_inj, dip_frac, a0, x_of = 4029.9268, 0.071580, 0.400605, 0.118237, 1.25, 1492.414
    z = np.linspace(0.0, (L_inj + 3.2) * 1e3, 2000)                    # the shipped x axis, um
    n_He, n_N = species_B(z, P_max, cN2_max, L_inj, dip_frac)
    n_bg = m3_to_cm3(electrons_background(n_He, n_N))
    n_inj = m3_to_cm3(electrons_injected(n_N))
    i = np.argmax(n_bg)                                                 # the plateau
    n_pl = n_bg[i]
    print(f"[example] config 1065: P_max {P_max:.0f} Pa, cN2_max {cN2_max:.3f}, L_inj {L_inj:.3f} mm, "
          f"dip_frac {dip_frac:.3f}, a0 {a0}, x_of {x_of:.0f} um")
    print(f"   background electrons at the plateau  {n_pl:.3e} cm^-3  (z = {z[i]:.0f} um); "
          f"injectable electrons there {n_inj[i]:.3e} cm^-3")
    print(f"   1. lambda_p = {plasma_wavelength_um(n_pl):.2f} um,  k_p = {k_p_per_um(n_pl):.4f} 1/um")
    print(f"   2. E_0 = {wave_breaking_field_V_per_m(n_pl):.3e} V/m")
    w0, tau = LASER["B"]["w0_um"], LASER["B"]["t_L_fs"]
    x_focus = focus_position_um("B", x_of)
    print(f"   3. a(z) with w0 = {w0} um, focus at x_p = {x_focus:.0f} um: at focus "
          f"{a0_along_z(x_focus, a0, w0, x_focus):.3f}, 1 mm before focus "
          f"{a0_along_z(x_focus - 1000, a0, w0, x_focus):.3f}, at the plateau (z = {z[i]:.0f} um) "
          f"{a0_along_z(z[i], a0, w0, x_focus):.3f}")
    print(f"   4. matched spot {matched_spot_um(a0, n_pl):.2f} um,  bubble radius {bubble_radius_um(a0, n_pl):.2f} um")
    dE = energy_gain_MeV(z, n_bg, eta=1.0)
    print(f"   5. energy gain at eta = 1 over the whole profile: {dE[-1]:.1f} MeV (scale by your fitted eta)")
    print(f"   6. pump-depletion length with t_L = {tau} fs: {pump_depletion_length_um(n_pl, tau) * 1e-3:.2f} mm")
    print(f"   7. beam-loading factor at Q = Q_s/4: {beam_loading_factor(1.0, 4.0):.2f}")


if __name__ == "__main__":
    pack = _find_pack()
    if pack is None:
        print("[A]/[B] skipped: no data pack found (set PALLAS_PACK or unpack it under data/).")
    else:
        print(f"pack: {pack}")
        _check_A(pack)
        _check_B(pack)
    _worked_example()
