"""Shared helpers for the 6D analysis: virtual S07 segment with a working TDC."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import xtrack as xt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nominal_model"))
import tdc
import xsuite_model

import clara_virtual as cv

HERE = Path(__file__).resolve().parent

# Share compiled tracking kernels between lines with the same element classes, so that a
# process compiles each kernel once instead of once per line (each compile takes minutes).
_SHARED_KERNELS: dict[tuple, dict] = {}
_build_tracker = xt.Line.build_tracker


def _build_tracker_shared(self, *args, **kwargs):
    key = tuple(sorted({type(e).__name__ for e in self.element_dict.values()}))
    if "track_kernel" not in kwargs and _SHARED_KERNELS.get(key):
        kwargs["track_kernel"] = _SHARED_KERNELS[key]
    out = _build_tracker(self, *args, **kwargs)
    _SHARED_KERNELS[key] = self.tracker.track_kernel
    return out


xt.Line.build_tracker = _build_tracker_shared
RECONSTRUCTION_POINT = xsuite_model.RECONSTRUCTION_POINT
LAST_ELEMENT = "cla_fea_dia_scr_04"
LABELS = ("x", "px", "y", "py", "tau", "delta")
PAIR_INDEX = [(i, j) for i in range(6) for j in range(i, 6)]  # the 21 unknowns


def load_s07_beam() -> tuple[np.ndarray, float]:
    with np.load(HERE / "s07_beam.npz") as data:
        return data["particles"], float(data["energy_ev"])


def build_segment(energy_ev: float, end: str = LAST_ELEMENT) -> cv.XsuiteLine:
    converted = xsuite_model.build_line(energy_ev, start=RECONSTRUCTION_POINT, end=end)
    tdc.install_tdc(converted, voltage_v=0.0)
    return cv.XsuiteLine(converted)


def make_pair(
    end: str = LAST_ELEMENT,
    particles: int = 20000,
    seed: int = 0,
    hidden_errors: bool = True,
):
    """(machine, model, true beam at the reconstruction point)."""
    beam, energy_ev = load_s07_beam()
    rng = np.random.default_rng(seed)
    beam = beam[rng.choice(len(beam), particles, replace=False)]
    machine_line = build_segment(energy_ev, end)
    model_line = cv.XsuiteLine(_rebuilt(energy_ev, end), machine_line.inventory)
    errors = (
        cv.HiddenErrors.random(machine_line.inventory, seed=seed)
        if hidden_errors
        else cv.HiddenErrors()
    )
    machine = cv.VirtualMachine(machine_line, beam, errors, seed=seed + 1)
    return machine, model_line, beam


def _rebuilt(energy_ev: float, end: str):
    converted = xsuite_model.build_line(energy_ev, start=RECONSTRUCTION_POINT, end=end)
    tdc.install_tdc(converted, voltage_v=0.0)
    return converted


def set_machine_tdc(
    machine: cv.VirtualMachine, voltage_v: float, phase_rad: float = 0.0
):
    tdc.set_tdc(machine._line.line, voltage_v, phase_rad)


def covariance(particles: np.ndarray) -> np.ndarray:
    return np.cov(particles.T)


def unique_to_matrix(values: np.ndarray) -> np.ndarray:
    sigma = np.zeros((6, 6))
    for value, (i, j) in zip(values, PAIR_INDEX):
        sigma[i, j] = sigma[j, i] = value
    return sigma


def matrix_to_unique(sigma: np.ndarray) -> np.ndarray:
    return np.array([sigma[i, j] for i, j in PAIR_INDEX])


def read_json(path: Path):
    with open(path) as handle:
        return json.load(handle)


def write_json(path: Path, payload, **kwargs) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, **kwargs)


def read_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def write_pickle(path: Path, payload) -> None:
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
