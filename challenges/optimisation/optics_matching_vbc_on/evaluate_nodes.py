"""Transmission of Eiad Hamwi's matched-optics nodes (optics_matching, 12x12 validated grid).

Each node's 46 quadrupole k1 values are written straight into the Cheetah SimEnv
(apertures on, nominal beam, no errors, no jitter, no clipping to the +-100 A ranges),
with every element tracked at second order. Transmission is the fraction of the
launched bunch alive at cla_feh_dia_scr_01, as SimEnv reports it. Rows of the grid run
in parallel worker processes, one torch thread each.

Run from challenges/optimisation:
    python optics_matching_vbc_on/evaluate_nodes.py optics_matching_vbc_on/results 50000 4
"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
NUM = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 4

CASES = {
    # master lattice converted to second-order tracking (VBC 118.5 mrad, L4H on)
    "vbc_on":  dict(lattice=str(HERE / "CLARA_cheetah_second_order.json"), vbc=None),
    # same, VBC chicane knob set to 0
    "vbc_off": dict(lattice=str(HERE / "CLARA_cheetah_second_order.json"), vbc=0.0),
    # Eiad's anchor lattice: already second order, VBC off, L4H off
    "design":  dict(lattice=str(ROOT / "optics_matching/data/anchor_lattice.json"), vbc=None),
}
_envs = {}

def _env(case):
    if case not in _envs:
        import torch; torch.set_num_threads(1)
        sys.path[:0] = [str(ROOT / "RL_framework"), str(ROOT)]
        from rl_framework.config import RLConfig
        from machine_spec.sources.clara import VBC_CHICANE
        from rl_framework.envs.sim_env import SimEnv
        spec = CASES[case]
        env = SimEnv(RLConfig.clara(backend="cheetah", num_particles=NUM, apertures=True,
                              beam_variation=0.0, misalignment_variation=0.0, beam_jitter=0.0,
                              power_supply_ripple=0.0, nominal_lattice_json=spec["lattice"],
                              target=(0.0, 0.0, 1e-3, 1e-3)))
        env.reset(seed=0)
        if spec["vbc"] is not None:
            env._knobs[VBC_CHICANE].set(spec["vbc"], clip=False)
        env.vbc_angle = env._knobs[VBC_CHICANE].value
        _envs[case] = env
    return _envs[case]

def _shoot(env):
    env._simulate(); r = env._readings
    return r["transmission"], r["stats"][2], r["stats"][3]

def task(args):
    case, i = args
    env = _env(case)
    grid = np.load(ROOT / "optics_matching/results/validated_grid.npz")
    K, quads = grid["k1"], grid["quadrupoles"].tolist()
    row = []
    if i < 0:   # the lattice's own optics, for reference
        return case, i, [_shoot(env)], env.vbc_angle
    for j in range(K.shape[1]):     # grid layout [sigma_x index, sigma_y index]
        for q, k in zip(quads, K[i, j]):
            env.transport.write(q, "k1", float(k))
        row.append(_shoot(env))
    return case, i, row, env.vbc_angle

if __name__ == "__main__":
    t0 = time.time()
    sizes = np.load(ROOT / "optics_matching/results/validated_grid.npz")["spot_sizes_m"]
    n = len(sizes)
    tasks = [(c, -1) for c in CASES] + [(c, i) for c in CASES for i in range(n)]
    res = {c: dict(T=np.full((n, n), np.nan), SX=np.full((n, n), np.nan),
                   SY=np.full((n, n), np.nan)) for c in CASES}
    with Pool(WORKERS) as pool:
        for case, i, row, vbc in pool.imap_unordered(task, tasks):
            res[case]["vbc"] = vbc
            if i < 0:
                res[case]["base"] = row[0]
            else:
                res[case]["T"][i], res[case]["SX"][i], res[case]["SY"][i] = np.array(row).T
            print(f"{case} row {i} done {time.time()-t0:.0f}s", flush=True)
    summary = {}
    for case, r in res.items():
        np.savez(OUT / f"transmission_{case}.npz", sizes_m=sizes, transmission=r["T"],
                 sigma_x_m=r["SX"], sigma_y_m=r["SY"], vbc_angle_rad=r["vbc"],
                 baseline=np.array(r["base"]), num_particles=NUM)
        summary[case] = dict(vbc_angle_rad=r["vbc"], baseline_transmission=r["base"][0],
                             min=float(np.nanmin(r["T"])), median=float(np.nanmedian(r["T"])),
                             max=float(np.nanmax(r["T"])))
    summary["num_particles"] = NUM; summary["seconds"] = round(time.time() - t0)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
