"""Transmission and beam size along s for every matched-optics node (Eiad's 12x12 grid).

Cheetah SimEnv from upstream master (apertures on, VBC 118.5 mrad, nominal 200k-particle
beam, no errors/jitter/ripple). Every element that supports it is tracked at second
order. The beam is tracked element by element; after each element we record the
fraction of the launched bunch still alive and the rms size of the survivors.

Run from challenges/optimisation:
    python track_along_s.py <out.npz> <case> [node_start node_stop] [--linear]
case: vbc_on | vbc_off ; nodes are flat indices i*12+j into the grid (i: sigma_x, j: sigma_y).
"""
import sys, time
from pathlib import Path
import numpy as np, torch, cheetah

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "RL_framework"), str(ROOT)]
from rl_framework.config import RLConfig
from machine_spec.sources.clara import VBC_CHICANE
from rl_framework.envs.sim_env import SimEnv, DTYPE

args = [a for a in sys.argv[1:] if not a.startswith("--")]
linear = "--linear" in sys.argv
out, case = Path(args[0]), args[1]
grid = np.load(ROOT / "optics_matching/results/validated_grid.npz")
K, quads = grid["k1"], grid["quadrupoles"].tolist()
n = K.shape[0]
nodes = range(int(args[2]), int(args[3])) if len(args) > 3 else range(n * n)

cfg = RLConfig.clara(backend="cheetah", num_particles=None, apertures=True,
               beam_variation=0.0, misalignment_variation=0.0, beam_jitter=0.0,
               power_supply_ripple=0.0, target=(0.0, 0.0, 1e-3, 1e-3))
env = SimEnv(cfg); env.reset(seed=0)
if case == "vbc_off":
    env._knobs[VBC_CHICANE].set(0.0, clip=False)
seg = env.segment
if not linear:
    for e in seg.elements:
        if "second_order" in getattr(e, "supported_tracking_methods", ()):
            e.tracking_method = "second_order"
els = seg.elements
s_end = np.cumsum([float(e.length) for e in els])
coords = env._coordinates
energy = torch.tensor(float(env.energy_ev), dtype=DTYPE)

def profile():
    p = torch.tensor(np.column_stack([coords, np.ones(len(coords))]), dtype=DTYPE)
    beam = cheetah.ParticleBeam(particles=p, energy=energy)
    T, SX, SY = (np.empty(len(els)) for _ in range(3))
    with torch.no_grad():
        for k, e in enumerate(els):
            beam = e.track(beam)
            alive = beam.survival_probabilities > 0.5
            T[k] = float(alive.float().mean())
            x, y = beam.x[alive], beam.y[alive]
            SX[k] = float(x.std()) if len(x) > 1 else np.nan
            SY[k] = float(y.std()) if len(y) > 1 else np.nan
    return T, SX, SY

res = {"nominal": profile()}
for idx in nodes:
    i, j = divmod(idx, n)
    for q, k in zip(quads, K[i, j]):
        env.transport.write(q, "k1", float(k))
    t0 = time.time()
    res[idx] = profile()
    print(case, idx, f"T_end={res[idx][0][-1]:.3f}", f"{time.time()-t0:.1f}s", flush=True)
idxs = [k for k in res if k != "nominal"]
np.savez(out, s=s_end, names=np.array([e.name for e in els]),
         nodes=np.array(idxs, dtype=int),
         T=np.array([res[k][0] for k in idxs]).reshape(len(idxs), -1),
         SX=np.array([res[k][1] for k in idxs]).reshape(len(idxs), -1),
         SY=np.array([res[k][2] for k in idxs]).reshape(len(idxs), -1),
         nominal=np.array(res["nominal"]), vbc=env._knobs[VBC_CHICANE].value,
         order="linear" if linear else "second_order")
