"""Track the nominal beam from S02 to the reconstruction point cla_s07_sim_aper_01 and cache it.

The result is the true 6D beam used as the hidden truth in the demo (scoring only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xsuite_model

import clara_virtual  # noqa: F401  (sets sys.path for xsuite_model)

OUT = Path(__file__).with_name("s07_beam.npz")
RECONSTRUCTION_POINT = xsuite_model.RECONSTRUCTION_POINT


def main() -> None:
    beam, energy_ev, _ = xsuite_model.load_distribution()
    converted = xsuite_model.build_line(energy_ev, end=RECONSTRUCTION_POINT)
    tracked = xsuite_model.track_forward(converted, beam)
    np.savez(OUT, particles=tracked, energy_ev=converted.final_energy_ev)
    print(
        "energy MeV",
        converted.final_energy_ev / 1e6,
        "mean",
        tracked.mean(0),
        "std",
        tracked.std(0),
    )


if __name__ == "__main__":
    main()
