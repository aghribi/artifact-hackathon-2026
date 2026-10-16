"""6D second-moment reconstruction at cla_s07_sim_aper_01 from screen moments.

Step 1: weighted linear least squares with the 6x6 maps of the error-free model.
Step 2: nonlinear refinement. The covariance is parametrised by its Cholesky factor, a fixed
whitened Gaussian sample is tracked through the xsuite model (exact maps, chromatic
effects included) and the 3 moments on every screen are matched.
The machine (hidden errors, true beam) is only used through screen_moments.
"""

from __future__ import annotations

import common
import numpy as np
import observability as obs
from scipy.optimize import least_squares

SHOT_NOISE = 0.01  # relative noise on each moment, as in VirtualMachine
MODEL_ERROR = 0.03  # extra relative weight floor for non-Gaussianity / hidden errors
PRIOR_SCALE = np.array([1e-3, 3e-4, 1e-3, 3e-4, 1e-3, 1e-2])  # generic, not the truth
TDC_VOLTAGE = 5e6
EXTRA_SETTING = {"cla_s07_mag_quad_01": 0.7, "cla_s07_mag_quad_02": 0.7}


def quad_settings(model, variant: str) -> list[dict[str, float]]:
    """k1 settings (all quads) for a measurement variant:
    'baseline'  nominal optics only;
    'q01q02'    plus one setting with S07 quad_01 and quad_02 scaled by 0.7;
    'scan'      plus the two greedy-optimised S07 settings of optics_scan.json."""
    settings = [{**model.nominal_k1}]
    if variant == "q01q02":
        alt = dict(model.nominal_k1)
        for quad, factor in EXTRA_SETTING.items():
            alt[quad] = model.nominal_k1[quad] * factor
        settings.append(alt)
    elif variant == "scan":
        picked = common.read_json(common.HERE / "optics_scan.json")["picked"][:2]
        for changes in picked:
            alt = dict(model.nominal_k1)
            alt.update(changes)
            settings.append(alt)
    return settings


def apply_k1(line_holder, k1: dict[str, float]) -> None:
    line_holder.set_quad_k1(k1)


def acquire(machine, model, settings, screens):
    """Measure moments: list of dict(setting index, voltage, screen, xx, yy, xy)."""
    order = model.element_index
    shots = []
    for index, k1 in enumerate(settings):
        machine._line.set_quad_k1(k1)
        for voltage in (0.0, TDC_VOLTAGE, -TDC_VOLTAGE):
            common.set_machine_tdc(machine, voltage)
            for screen in screens:
                if voltage != 0.0 and order[screen] < order["cla_s07_dia_tdc_01"]:
                    continue
                reading = machine.screen_moments(screen)
                shots.append(
                    {
                        "setting": index,
                        "voltage": voltage,
                        "screen": screen,
                        "xx": reading["xx"],
                        "yy": reading["yy"],
                        "xy": reading["xy"],
                    }
                )
    machine.restore_nominal_quads()
    common.set_machine_tdc(machine, 0.0)
    return shots


def shot_vector(shots):
    return np.array([[s["xx"], s["yy"], s["xy"]] for s in shots])


def noise_of(moments: np.ndarray) -> np.ndarray:
    return np.hypot(SHOT_NOISE, MODEL_ERROR) * np.column_stack(
        [moments[:, 0], moments[:, 1], np.sqrt(np.abs(moments[:, 0] * moments[:, 1]))]
    )


def linear_reconstruction(model, shots, settings, ridge: float = 1e-6):
    rows = []
    cache = {}
    for shot in shots:
        key = (shot["setting"], shot["voltage"])
        if key not in cache:
            model.set_quad_k1(settings[shot["setting"]])
            cache[key] = obs.maps_for(model, model.inventory.screens, shot["voltage"])
        rows.append(obs.moment_rows(cache[key][shot["screen"]]))
    model.restore_nominal_quads()
    design = np.vstack(rows)
    measured = shot_vector(shots)
    noise = noise_of(measured).reshape(-1)
    scales = np.array([PRIOR_SCALE[i] * PRIOR_SCALE[j] for i, j in common.PAIR_INDEX])
    lhs = np.vstack([design * scales / noise[:, None], ridge * np.eye(21)])
    rhs = np.concatenate([measured.reshape(-1) / noise, np.zeros(21)])
    theta = np.linalg.lstsq(lhs, rhs, rcond=None)[0] * scales
    return common.unique_to_matrix(theta)


def project_psd(sigma: np.ndarray, floor_fraction: float = 1e-4) -> np.ndarray:
    """Nearest PSD matrix in the correlation-scaled space, eigenvalues floored at a small
    fraction of the mean eigenvalue (the beam planes are strongly correlated)."""
    sizes = np.sqrt(np.abs(np.diag(sigma)))
    scaled = sigma / np.outer(sizes, sizes)
    values, vectors = np.linalg.eigh(scaled)
    values = np.maximum(values, floor_fraction * values.mean())
    scaled = (vectors * values) @ vectors.T
    d = np.sqrt(np.diag(scaled))
    scaled = scaled / np.outer(d, d)
    return scaled * np.outer(sizes, sizes)


class NonlinearFit:
    """sigma = D L L^T D with D fixed (sizes from the starting estimate) and L lower
    triangular (21 parameters). Near-singular covariances (strongly correlated planes) are
    then described by small L diagonal entries, which is well conditioned."""

    def __init__(self, model, shots, settings, sigma_start, particles=4000, seed=5):
        self.model, self.shots, self.settings = model, shots, settings
        rng = np.random.default_rng(seed)
        z = rng.normal(size=(particles, 6))
        z -= z.mean(0)
        z = z @ np.linalg.inv(np.linalg.cholesky(np.cov(z.T))).T  # exactly white
        self.white = z
        self.measured = shot_vector(shots)
        self.noise = noise_of(self.measured)
        self.sizes = np.sqrt(np.diag(sigma_start))
        self.groups = sorted({(s["setting"], s["voltage"]) for s in shots})

    def factor(self, theta: np.ndarray) -> np.ndarray:
        lower = np.zeros((6, 6))
        lower[np.tril_indices(6)] = theta
        return self.sizes[:, None] * lower

    def theta_from_covariance(self, sigma: np.ndarray) -> np.ndarray:
        scaled = sigma / np.outer(self.sizes, self.sizes)
        return np.linalg.cholesky(scaled)[np.tril_indices(6)]

    def predict(self, theta: np.ndarray) -> np.ndarray:
        sample = self.white @ self.factor(theta).T
        resolution = 20e-6**2
        tracked = {}
        for setting, voltage in self.groups:
            self.model.set_quad_k1(self.settings[setting])
            common.tdc.set_tdc(self.model.line, voltage)
            wanted = [
                s["screen"]
                for s in self.shots
                if s["setting"] == setting and s["voltage"] == voltage
            ]
            tracked[(setting, voltage)] = self.model.track_through(sample, wanted)
        self.model.restore_nominal_quads()
        common.tdc.set_tdc(self.model.line, 0.0)
        out = []
        for s in self.shots:
            c = tracked[(s["setting"], s["voltage"])][s["screen"]]
            x, y = c[:, 0], c[:, 2]
            out.append(
                [
                    x.var() + resolution,
                    y.var() + resolution,
                    np.mean((x - x.mean()) * (y - y.mean())),
                ]
            )
        return np.array(out)

    def residual(self, theta: np.ndarray) -> np.ndarray:
        try:
            return ((self.predict(theta) - self.measured) / self.noise).reshape(-1)
        except RuntimeError:  # particles lost: penalise
            self.model.restore_nominal_quads()
            return np.full(self.measured.size, 1e3)

    def solve(self, sigma_start: np.ndarray, max_nfev: int = 40):
        theta0 = self.theta_from_covariance(sigma_start)
        lower = np.full(21, -2.0)
        diag_positions = [
            i for i, (r, c) in enumerate(zip(*np.tril_indices(6))) if r == c
        ]
        lower[diag_positions] = 1e-7
        result = least_squares(
            self.residual,
            theta0,
            bounds=(lower, np.full(21, 2.0)),
            x_scale="jac",
            diff_step=1e-6,
            max_nfev=max_nfev,
            loss="soft_l1",
            f_scale=3.0,
        )
        factor = self.factor(result.x)
        return factor @ factor.T, result


def summary(sigma: np.ndarray) -> dict:
    """Key beam quantities from a 6x6 covariance (Cheetah coordinates)."""
    out = {}
    for plane, i in (("x", 0), ("y", 2)):
        out[f"eps_{plane} (nm, geometric)"] = 1e9 * np.sqrt(
            np.linalg.det(sigma[i : i + 2, i : i + 2])
        )
    out["sigma_z (mm)"] = 1e3 * np.sqrt(sigma[4, 4])
    out["sigma_delta (%)"] = 1e2 * np.sqrt(sigma[5, 5])
    out["chirp corr(tau,delta)"] = sigma[4, 5] / np.sqrt(sigma[4, 4] * sigma[5, 5])
    out["chirp h (1/m) <tau delta>/<tau tau>"] = sigma[4, 5] / sigma[4, 4]
    for name, (i, j) in {
        "x,tau": (0, 4),
        "y,tau": (2, 4),
        "x,delta": (0, 5),
        "y,delta": (2, 5),
        "x,y": (0, 2),
        "px,delta": (1, 5),
    }.items():
        out[f"corr({name})"] = sigma[i, j] / np.sqrt(sigma[i, i] * sigma[j, j])
    return out


def correlation_matrix(sigma: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(sigma))
    return sigma / np.outer(d, d)
