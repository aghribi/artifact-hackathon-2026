"""Global forward model and virtual acquisition for the 6D reconstruction at the S02 start.

One xsuite line from `cla_s02_sim_aper_01` to the FEH screen (RF cavities, TDC kick, FEA
dogleg, VBC dipoles at angle 0), a measurement configuration = quadrupole settings plus TDC
voltage, and a screen-image acquisition on the virtual machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import common
import numpy as np
import tdc

cv = common.cv
xsuite_model = common.xsuite_model

LINE_END = "cla_feh_dia_scr_01"
TDC_ELEMENT = tdc.TDC_NAME
FIELD_OF_VIEW_M = 5e-3  # half width of the YAG window
PIXELS = 200  # at 50 um, as VirtualMachine.screen_image
SCREEN_RESOLUTION_M = 20e-6
# TDC voltage from the 23 March scan (CLA-TDC1-RF-CTRL-01 at 7.0 MW, phase 78 deg). No
# calibration is in the repo, so V = sqrt(P R) with an assumed effective shunt impedance of
# 5 MOhm (5 MV at 5 MW, the VELA/CLARA design figure). The phase is taken as the zero
# crossing (pure streak); the absolute LLRF phase offset is a calibration constant.
TDC_POWER_W = 7.0e6
TDC_SHUNT_IMPEDANCE_OHM = 5e6  # assumed
TDC_VOLTAGE_V = (TDC_POWER_W * TDC_SHUNT_IMPEDANCE_OHM) ** 0.5
S02_Q_CURRENTS_A = {  # tomography/README.md section 6, S02 quad_01 / quad_02
    "s02a": (-11.3, 26.3),
    "s02b": (-8.4, 18.0),
    "s02c": (-9.2, 16.0),
}


VBC_DIPOLES = ("cla_vbc_mag_dip_01", "cla_vbc_mag_dip_02", "cla_vbc_mag_dip_03", "cla_vbc_mag_dip_04")


L4H_CAVITY = "cla_l4h_lin_cav_01"
PLAN_LATTICE = common.HERE / "CLARA_cheetah_plan.json"


def plan_lattice(vbc_off: bool = True, l4h_off: bool = True, path=PLAN_LATTICE):
    """The lattice file the measurement runs on, written next to this module: the VBC
    chicane at angle 0 (faces square) and the L4H X-band cavity unpowered (it converts to
    a drift; its iris aperture stays). Francesco, 15 Oct: the plan runs with the VBC off,
    and L4H is off on the machine."""
    import json

    lattice = json.loads(xsuite_model.LATTICE_JSON.read_text())
    elements = lattice["elements"]
    if vbc_off:
        for name in VBC_DIPOLES:
            for key in ("angle", "dipole_e1", "dipole_e2"):
                elements[name][1][key] = 0.0
    if l4h_off:
        elements[L4H_CAVITY][1]["voltage"] = 0.0
    path.write_text(json.dumps(lattice))
    return path


def build_line(end: str = LINE_END, apertures: bool = False, vbc_off: bool = True,
               l4h_off: bool = True) -> cv.XsuiteLine:
    """The model line, by default on the plan's lattice (VBC off, L4H off); `apertures`
    adds the vacuum chamber, which loses particles."""
    _, energy_ev, _ = xsuite_model.load_distribution()
    lattice = plan_lattice(vbc_off, l4h_off) if (vbc_off or l4h_off) else xsuite_model.LATTICE_JSON
    converted = xsuite_model.build_line(energy_ev, lattice, end=end, apertures=apertures)
    tdc.install_tdc(converted, voltage_v=0.0)
    line = cv.XsuiteLine(converted)
    return line


def tdc_momentum_ev(line: cv.XsuiteLine) -> float:
    return line.inventory.momentum_mevc_at[TDC_ELEMENT] * 1e6


def set_tdc_voltage(line: cv.XsuiteLine, voltage_v: float, phase_rad: float = 0.0) -> None:
    tdc.set_tdc(line.line, voltage_v, phase_rad, p0c_ev=tdc_momentum_ev(line))


@dataclass
class Config:
    """Quadrupole k1 overrides (absolute) and a TDC voltage; images taken at `screens`."""

    name: str
    k1: dict[str, float] = field(default_factory=dict)
    tdc_voltage_v: float = 0.0
    screens: list[str] = field(default_factory=list)
    rf_phase_deg: dict[str, float] = field(default_factory=dict)  # cavity -> phase change
    rf_rescale: bool = True  # downstream magnets follow the new energy


def apply_config(line: cv.XsuiteLine, config: Config) -> None:
    line.restore_nominal_quads()
    line.set_quad_k1(config.k1)
    set_tdc_voltage(line, config.tdc_voltage_v)
    set_rf_phases(line.line, config.rf_phase_deg, config.rf_rescale)


_RF_NOMINAL: dict[int, dict] = {}


def chip_geometry(pixel_m=50e-6, rebin=2):
    """Each screen's real chip, from the Cheetah lattice (as SimEnv images them):
    screen -> (xmin, xmax, ymin, ymax, nx, ny) at about `pixel_m`, with nx, ny multiples
    of `rebin`."""
    import cheetah

    out = {}
    for el in common.xsuite_model.cheetah_segment().elements:
        if isinstance(el, cheetah.Screen) and "dia_scr" in el.name:
            x0, x1, y0, y1 = (float(v) for v in el.extent)
            nx = max(rebin, int(round((x1 - x0) / pixel_m / rebin)) * rebin)
            ny = max(rebin, int(round((y1 - y0) / pixel_m / rebin)) * rebin)
            out[el.name] = (x0, x1, y0, y1, nx, ny)
    return out


def set_rf_phases(xline, changes: dict[str, float], rescale: bool = True) -> None:
    """Shift the phase of whole linac cavities (cheetah convention, degrees) and scale the
    reference energy gain of each cavity with cos(phi), i.e. every downstream magnet is
    rescaled to the new beam energy, as operators do for a phase scan. The edge and
    ponderomotive lenses are left at their nominal strength (small). Cavities not named
    go back to nominal. rescale=False keeps the magnets at their currents: the beam
    then comes out off momentum, as when only the phase is changed. Works on any xsuite line built by xsuite_model (SimEnv's too)."""
    key = id(xline)
    if key not in _RF_NOMINAL:
        nominal = {}
        for name in xline.element_names:
            if "..cav" in name or "..ref" in name:
                cavity = name.split("..")[0]
                el = xline.element_dict[name]
                entry = nominal.setdefault(cavity, {"cav": {}, "ref": {}})
                if "..cav" in name:
                    entry["cav"][name] = float(el.phase)
                else:
                    entry["ref"][name] = float(el.Delta_p0c)
        _RF_NOMINAL[key] = nominal
    for cavity, entry in _RF_NOMINAL[key].items():
        dphi = np.radians(changes.get(cavity, 0.0))
        any_phase = next(iter(entry["cav"].values()))
        phi0 = np.pi / 2 - any_phase  # cheetah phase
        ratio = np.cos(phi0 + dphi) / np.cos(phi0) if rescale else 1.0
        for name, phase in entry["cav"].items():
            xline.element_dict[name].phase = phase - dphi
        for name, dp in entry["ref"].items():
            xline.element_dict[name].Delta_p0c = dp * ratio


def screens_after(line: cv.XsuiteLine, element: str) -> list[str]:
    order = line.element_index
    return [s for s in line.inventory.screens if order[s] > order[element]]


def make_configs(line: cv.XsuiteLine, imageable: list[str], s07_factors=(0.7,)) -> list[Config]:
    """Measurement plan: nominal optics (all screens TDC off; screens after the TDC with
    +V and -V), three S02 quad_01/quad_02 settings (imageable screens, TDC off), and the
    S07 quad_01/quad_02 scaled settings (all screens after them, TDC off/+/-)."""
    screens = line.inventory.screens
    after_tdc = screens_after(line, TDC_ELEMENT)
    configs = [Config("nominal_off", {}, 0.0, screens)]
    for sign, label in ((+1, "plus"), (-1, "minus")):
        configs.append(Config(f"nominal_{label}", {}, sign * TDC_VOLTAGE_V, after_tdc))
    for name, (q1, q2) in S02_Q_CURRENTS_A.items():
        k1 = {
            "cla_s02_mag_quad_01": line.quad_k1_from_current("cla_s02_mag_quad_01", q1),
            "cla_s02_mag_quad_02": line.quad_k1_from_current("cla_s02_mag_quad_02", q2),
        }
        configs.append(Config(name, k1, 0.0, imageable))
    after_q07 = screens_after(line, "cla_s07_mag_quad_02")
    for factor in s07_factors:
        k1 = {
            q: line.nominal_k1[q] * factor
            for q in ("cla_s07_mag_quad_01", "cla_s07_mag_quad_02")
        }
        configs.append(Config(f"s07x{factor}_off", k1, 0.0, after_q07))
        for sign, label in ((+1, "plus"), (-1, "minus")):
            configs.append(Config(f"s07x{factor}_{label}", k1, sign * TDC_VOLTAGE_V, after_q07))
    return configs


def blur_and_histogram(coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Screen image as VirtualMachine.screen_image: 20 um Gaussian blur per particle, 200 x
    200 histogram over +-5 mm, rows = y, columns = x. Counts (not normalised)."""
    blur = rng.normal(0, SCREEN_RESOLUTION_M, (len(coords), 2))
    edges = np.linspace(-FIELD_OF_VIEW_M, FIELD_OF_VIEW_M, PIXELS + 1)
    image, _, _ = np.histogram2d(
        coords[:, 2] + blur[:, 1], coords[:, 0] + blur[:, 0], bins=[edges, edges]
    )
    return image


def acquire_images(machine: cv.VirtualMachine, configs: list[Config], seed: int = 3):
    """One machine shot per (config), recorded at every screen of the config. In reality
    each screen is a separate shot; here they share the shot particles (zero jitter), which
    only correlates the counting noise between screens. Returns {(config, screen): image}."""
    rng = np.random.default_rng(seed)
    images = {}
    for config in configs:
        apply_config(machine._line, config)
        tracked = machine._line.track_through(machine._shot_beam(), config.screens)
        for screen in config.screens:
            images[(config.name, screen)] = blur_and_histogram(tracked[screen], rng)
        machine.shots += len(config.screens) - 1
    machine._line.restore_nominal_quads()
    set_tdc_voltage(machine._line, 0.0)
    return images


def window_fraction(image: np.ndarray, particles: int) -> float:
    return float(image.sum() / particles)


def load_design() -> dict:
    """Designed per-screen settings (design_optics.py)."""
    return common.read_json(common.HERE / "optics_design.json")


def make_plan(line: cv.XsuiteLine, design: dict | None = None) -> list[Config]:
    """Measurement plan of the image-based reconstruction:

    - nominal optics, TDC off, every screen (screens from S06 on clip at the window edge;
      the reconstruction uses the in-window part and the clipped fraction);
    - three S02 quad_01/quad_02 settings, S02 to S05 screens, TDC off;
    - for each designed screen after the TDC (design_optics.py), its own quadrupole
      setting with the TDC off, +V and -V."""
    design = load_design() if design is None else design
    imageable = [s for s in line.inventory.screens if s.split("_")[1] in ("s02", "s03", "s04", "s05")]
    configs = [Config("nominal_off", {}, 0.0, list(line.inventory.screens))]
    for name, (q1, q2) in S02_Q_CURRENTS_A.items():
        k1 = {
            "cla_s02_mag_quad_01": line.quad_k1_from_current("cla_s02_mag_quad_01", q1),
            "cla_s02_mag_quad_02": line.quad_k1_from_current("cla_s02_mag_quad_02", q2),
        }
        configs.append(Config(name, k1, 0.0, imageable))
    for screen, entry in design.items():
        short = screen.replace("cla_", "").replace("_dia", "").replace("_scr", "")
        for sign, label in ((0, "off"), (+1, "plus"), (-1, "minus")):
            configs.append(Config(f"{short}_{label}", entry["k1"], sign * TDC_VOLTAGE_V, [screen]))
    return configs
