"""Check the TDC model: kick, Panofsky-Wenzel energy kick, streak vs voltage, calibration."""

from __future__ import annotations

import json

import common
import numpy as np
import tdc
import xsuite_model as xm

SCREEN = "cla_s07_dia_scr_02"


def map_between(model, first_element: str, last_element: str) -> np.ndarray:
    """6x6 map from the entrance of `first_element` to the exit of `last_element`."""
    names = list(model.line.element_names)
    start, stop = names.index(first_element), names.index(last_element) + 1
    test = np.zeros((12, 6))
    for c in range(6):
        test[2 * c, c], test[2 * c + 1, c] = 1e-6, -1e-6
    particles = xm._particles(test, model.converted.initial_energy_ev)
    model.line.track(particles, ele_start=start, ele_stop=stop)
    out = xm._coordinates(particles)
    return np.column_stack([(out[2 * c] - out[2 * c + 1]) / 2e-6 for c in range(6)])


def main() -> None:
    _, model, beam = common.make_pair(particles=20000)
    line = model.line
    energy = model.converted.initial_energy_ev
    p0c = xm.momentum_ev(energy)
    k = 2 * np.pi * tdc.TDC_FREQUENCY_HZ / tdc.SPEED_OF_LIGHT
    report = {"k_1_per_m": k, "p0c_MeV": p0c / 1e6}

    # 1. thin kick alone (tau = +1 mm is a late particle, zeta = -1 mm)
    tdc.set_tdc(line, 5e6, 0.0)
    gain = 5e6 / p0c
    probe = np.zeros((2, 6))
    probe[:, 4] = 1e-3
    probe[1, 2] = 1e-3
    first = list(line.element_names).index(tdc.TDC_NAME + "..kick")
    particles = xm._particles(probe, energy)
    line.track(particles, ele_start=first, ele_stop=first + 1)
    kicked = xm._coordinates(particles)
    report["py_kick_vs_expected"] = [
        float(kicked[0, 3]),
        float(-gain * np.sin(k * 1e-3)),
    ]
    report["delta_kick_y1mm_vs_expected"] = [
        float(kicked[1, 5]),
        float(gain * k * 1e-3 * np.cos(k * 1e-3)),
    ]

    # 2. streak vs voltage at the screen against the linear map prediction
    rows = []
    for voltage in (0.0, 1e6, 2e6, 3e6, 5e6):
        tdc.set_tdc(line, voltage, 0.0)
        sigma_y_tracked = model.track_through(beam, [SCREEN])[SCREEN][:, 2].std()
        transfer = model.linear_map([SCREEN])[SCREEN]
        sigma_linear = np.sqrt(transfer[2] @ np.cov(beam.T) @ transfer[2])
        rows.append((voltage / 1e6, sigma_y_tracked * 1e3, sigma_linear * 1e3))
    report["sigma_y_mm: (MV, tracked, linear)"] = rows

    # 3. time calibration: dy/dtau at the screen from the TDC exit map
    tdc.set_tdc(line, 1e6, 0.0)
    r34 = map_between(model, tdc.TDC_NAME, SCREEN)[2, 3]
    slope_expected = -r34 * (1e6 / p0c) * k  # per MV, dy/dtau (tau = -zeta)
    off = model.linear_map([SCREEN])[SCREEN][2, 4]
    tdc.set_tdc(line, 0.0, 0.0)
    off0 = model.linear_map([SCREEN])[SCREEN][2, 4]
    report["dy_dtau_per_MV_measured_vs_expected"] = [
        float(off - off0),
        float(slope_expected),
    ]
    report["R34_TDC_exit_to_screen_m"] = float(r34)
    report["fs_per_um_at_1MV"] = float(1e-6 / abs(slope_expected) / 3e8 * 1e15)
    print(json.dumps(report, indent=1))
    common.write_json(common.HERE / "verify_tdc.json", report, indent=1)


if __name__ == "__main__":
    main()
