"""Build the nominal starting beam at the benchmark start, cla_s02_sim_aper_01.

1. Transverse plane: the GPSR reconstruction at CLA-S07-SIM-APER-01
   (data/clara/lattice_data/CLA-S07-SIM-APER-01.hdf), read as x, x' = px/pz, y, y'. The
   file's pz is a single placeholder value (0.86 MeV/c), so it only sets the angles; the
   energy comes from the model. A random subsample (seed 0) keeps the file small.
2. Longitudinal plane: a source beam is injected at the start and tracked forward in
   xsuite through the magnet settings of the scan (CLARA_cheetah_S07_GPSR.json) to the
   reconstruction point. Its (zeta, delta) there are paired row by row with the
   reconstructed particles, so the chirp survives but carries no transverse correlation.
   Sources: "bench-nominal", score.sample_beam at severity 0 (default), or "astra", the
   longitudinal shape of S02_input_beam.astra, centred on its own reference particle and
   applied around bench-nominal's reference energy.
3. The 6D beam is tracked back to the start in xsuite through the same settings and saved.
4. Closure: tracking the result forward again must return the reconstruction.
5. The beam is centred at the start in x, px, y and py, so errors can be added on top of
   an on-axis beam; the removed offset is stored in the metadata.

Run: python make_initial_distribution.py [--longitudinal astra] [--output FILE]
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

import cheetah
import h5py
import numpy as np

import xsuite_model as model

sys.path.insert(0, str(model.CHALLENGE_DIR))
import score

RECONSTRUCTION_FILE = (
    model.CHALLENGE_DIR / "data" / "clara" / "lattice_data" / "CLA-S07-SIM-APER-01.hdf"
)
ASTRA_FILE = (
    model.CHALLENGE_DIR / "data" / "clara" / "lattice_data" / "S02_input_beam.astra"
)
PARTICLES = 200_000
SUBSAMPLE_SEED = 0
ASTRA_SEED = 1


def load_reconstruction(path=RECONSTRUCTION_FILE) -> np.ndarray:
    """(N, 4) x (m), x' (rad), y (m), y' (rad) of the reconstruction, angles as px / pz."""
    with h5py.File(path, "r") as handle:
        electrons = handle["particles/electron"]
        longitudinal_momentum = float(electrons["momentum/z"].attrs["value"])
        return np.column_stack(
            [
                electrons["position/x"][:].astype(float),
                electrons["momentum/x"][:].astype(float) / longitudinal_momentum,
                electrons["position/y"][:].astype(float),
                electrons["momentum/y"][:].astype(float) / longitudinal_momentum,
            ]
        )


def bench_nominal(count: int) -> tuple[np.ndarray, float]:
    """score.sample_beam at severity 0: Cheetah-convention coordinates and total energy (eV)."""
    beam = score.sample_beam(np.random.default_rng(0), count, severity=0.0)
    return beam.particles.detach().double().numpy()[:, :6], float(beam.energy)


def astra_longitudinal(count: int) -> np.ndarray:
    """(count, 2) (tau, delta) of the ASTRA beam relative to its own reference particle."""
    beam = cheetah.ParticleBeam.from_astra(str(ASTRA_FILE))
    longitudinal = beam.particles.detach().double().numpy()[:, 4:6]
    picks = np.random.default_rng(ASTRA_SEED).choice(
        len(longitudinal), count, replace=len(longitudinal) < count
    )
    return longitudinal[picks] - longitudinal.mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--longitudinal", choices=("bench-nominal", "astra"), default="bench-nominal"
    )
    parser.add_argument("--particles", type=int, default=PARTICLES)
    parser.add_argument("--output", default=str(model.INITIAL_DISTRIBUTION))
    arguments = parser.parse_args()

    reconstruction = load_reconstruction()
    picks = np.random.default_rng(SUBSAMPLE_SEED).choice(
        len(reconstruction), arguments.particles, replace=False
    )
    reconstruction = reconstruction[picks]

    source, energy_ev = bench_nominal(arguments.particles)
    if arguments.longitudinal == "astra":
        source[:, 4:6] = astra_longitudinal(arguments.particles)
    scan_line = model.build_line(
        energy_ev, lattice_json=model.GPSR_LATTICE_JSON, end=model.RECONSTRUCTION_POINT
    )
    longitudinal_at_reconstruction = model.track_forward(scan_line, source)[:, 4:6]

    at_reconstruction = np.column_stack(
        [reconstruction, longitudinal_at_reconstruction]
    )
    start = model.track_backward(scan_line, at_reconstruction)
    closure = model.track_forward(scan_line, start)
    worst = float(
        np.max(np.abs(closure[:, :4] - reconstruction) / reconstruction.std(axis=0))
    )
    print(
        f"closure: largest particle difference at {model.RECONSTRUCTION_POINT} is {worst:.1e} rms"
    )
    removed_offset = start[:, :4].mean(axis=0)
    start[:, :4] -= removed_offset

    model.save_distribution(
        arguments.output,
        start,
        energy_ev,
        location=model.START,
        convention="Cheetah (x [m], px, y [m], py, tau [m], delta), px and py canonical / p0, tau = -zeta",
        transverse_source=f"{RECONSTRUCTION_FILE.name}, {arguments.particles} particles (seed {SUBSAMPLE_SEED})",
        longitudinal_source=arguments.longitudinal,
        magnet_settings_for_backtracking=model.GPSR_LATTICE_JSON.name,
        removed_offset_x_px_y_py=" ".join(f"{value:.6e}" for value in removed_offset),
        built=datetime.now(UTC).date().isoformat(),
    )
    summary = model.beam_summary(start, energy_ev)
    print(
        f"wrote {arguments.output}: {arguments.particles} particles at {model.START}, {energy_ev / 1e6:.4f} MeV"
    )
    for plane in ("x", "y"):
        print(
            f"  {plane}: beta {summary[f'beta_{plane}']:.3f} m, alpha {summary[f'alpha_{plane}']:.3f}, "
            f"eps_n {summary[f'normalised_emittance_{plane}'] * 1e6:.3f} um, sigma {summary[f'sigma_{plane}'] * 1e3:.3f} mm"
        )
    print(
        f"  sigma_z {summary['sigma_tau'] * 1e3:.3f} mm, sigma_delta {summary['sigma_delta'] * 100:.3f} %, "
        f"x-y correlation {summary['correlation_xy']:.3f}"
    )


if __name__ == "__main__":
    main()
