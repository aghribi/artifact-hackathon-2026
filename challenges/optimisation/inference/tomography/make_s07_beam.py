"""Track the nominal S02 beam to cla_s07_sim_aper_01 once and cache it (the S07 benchmark beam)."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xsuite_model as xm

t = time.time()
beam, energy_ev, _ = xm.load_distribution()
conv = xm.build_line(energy_ev, end=xm.RECONSTRUCTION_POINT)
print("built", time.time() - t, conv.final_energy_ev / 1e6, flush=True)
out = xm.track_forward(conv, beam)
np.savez(
    Path(__file__).parent / "s07_beam.npz",
    particles=out,
    energy_ev=conv.final_energy_ev,
)
print("done", time.time() - t, out.mean(0), out.std(0), flush=True)
