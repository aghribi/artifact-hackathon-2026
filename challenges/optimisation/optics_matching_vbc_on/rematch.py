"""Rematch Eiad's optics nodes with the VBC on (master lattice, second-order Cheetah, apertures).

For each node the 30 quadrupoles after the VBC (S06, S07, FEA, FEH) start from Eiad's k1
and are tuned by L-BFGS through differentiable Cheetah tracking; Eiad's fixed 42-magnet prefix
is kept. Loss = ln(sigma_x/target)^2 + ln(sigma_y/target)^2 + W_LOSS * (1 - T_soft): sizes are
the rms about the centroid at cla_feh_dia_scr_01 of the particles that survive the hard
apertures, and T_soft replaces each aperture by a sigmoid so the fit sees beam about to scrape.
The beam up to the VBC exit is tracked once and reused. Every result is then checked with
SimEnv itself (hard apertures, more particles).

Run from challenges/optimisation:
    python optics_matching_vbc_on/rematch.py optics_matching_vbc_on/results 4 3000
"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
LATTICE = str(HERE / "CLARA_cheetah_second_order.json")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
FIT_N = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
CHECK_N = 50_000
SPLIT = "cla_vbc_mag_dip_04"
SCREEN = "cla_feh_dia_scr_01"
import os
W_LOSS = float(os.environ.get("W_LOSS", 5)); EPS = float(os.environ.get("EPS", 0.03))
STEPS = int(os.environ.get("STEPS", 60)); LR = float(os.environ.get("LR", 0.03)); K_MAX = 30.0
OPT = os.environ.get("OPT", "lbfgs")
# VBC bend per dipole [rad]. 0.13170 rad gives R56 = -57.6 mm, the measured maximum
# compression into FEBE (Jones et al., IPAC'26 THP2100); the master lattice has 0.1185.
VBC_ANGLE = float(os.environ.get("VBC_ANGLE", 0.13169935))
_state = {}


def _setup():
    if _state:
        return _state
    import torch, cheetah
    torch.set_num_threads(1)
    sys.path[:0] = [str(ROOT / "RL_framework"), str(ROOT)]
    from rl_framework.config import RLConfig
    from rl_framework.envs.sim_env import SimEnv
    cfg = dict(backend="cheetah", apertures=True, beam_variation=0.0, misalignment_variation=0.0,
               beam_jitter=0.0, nominal_lattice_json=LATTICE, target=(0.0, 0.0, 1e-3, 1e-3))
    check = SimEnv(RLConfig.clara(num_particles=CHECK_N, **cfg)); check.reset(seed=0)
    fit = SimEnv(RLConfig.clara(num_particles=FIT_N, **cfg)); fit.reset(seed=0)
    from machine_spec.sources.clara import VBC_CHICANE
    for env in (check, fit):   # knob sign: dipole 01 bends negative in the lattice file
        env._knobs[VBC_CHICANE].set(-VBC_ANGLE, clip=False)
    seg = fit.segment.to(dtype=torch.float64)
    names = [e.name for e in seg.elements]
    split, end = names.index(SPLIT) + 1, names.index(SCREEN) + 1
    up, down = seg.elements[:split], seg.elements[split:end]
    # Eiad's 42-magnet prefix is the same at every node: set it before the cached upstream track
    grid = np.load(ROOT / "optics_matching/results/validated_grid.npz")
    k0 = dict(zip(grid["quadrupoles"].tolist(), grid["k1"][0, 0]))
    for e in seg.elements:
        if isinstance(e, cheetah.Quadrupole) and e.name in k0:
            e.k1 = torch.tensor(float(k0[e.name]), dtype=torch.float64)
    coords = torch.tensor(fit._coordinates, dtype=torch.float64)
    beam = cheetah.ParticleBeam(particles=torch.column_stack([coords, torch.ones(len(coords), dtype=torch.float64)]),
                                energy=torch.tensor(float(fit.energy_ev), dtype=torch.float64))
    alive = torch.ones(len(coords), dtype=torch.float64)
    with torch.no_grad():
        for e in up:
            if isinstance(e, cheetah.Aperture):
                alive = alive * _hard(e, beam)
            else:
                beam = e.track(beam)
    quads = [e for e in down if isinstance(e, cheetah.Quadrupole)]
    _state.update(torch=torch, cheetah=cheetah, check=check, beam=beam, alive=alive, down=down,
                  quads=quads, grid=grid, Q=grid["quadrupoles"].tolist())
    return _state


def _rho(e, x, y):
    if e.shape == "elliptical":
        return (x.square() / e.x_max**2 + y.square() / e.y_max**2).sqrt()
    return (x.abs() / e.x_max).maximum(y.abs() / e.y_max)


def _hard(e, beam):
    return (_rho(e, beam.x, beam.y) <= 1).double() if e.is_active else 1.0


def _forward(s, k):
    torch, cheetah = s["torch"], s["cheetah"]
    for q, v in zip(s["quads"], k):
        q.k1 = v
    beam, w, soft = s["beam"], s["alive"], s["alive"]
    for e in s["down"]:
        if isinstance(e, cheetah.Aperture):
            if e.is_active:
                rho = _rho(e, beam.x, beam.y)
                soft = soft * torch.sigmoid((1 - rho) / EPS)
                w = w * (rho.detach() <= 1).double()
        elif not isinstance(e, cheetah.Screen):
            beam = e.track(beam)
    # sizes over the particles that really survive; the soft weight only drives transmission
    W = w.sum()
    mx, my = (w * beam.x).sum() / W, (w * beam.y).sum() / W
    sx = ((w * (beam.x - mx).square()).sum() / W).sqrt()
    sy = ((w * (beam.y - my).square()).sum() / W).sqrt()
    return sx, sy, soft.sum() / len(w)


def _check(s, settings):
    env = s["check"]
    for q, v in settings.items():
        env.transport.write(q, "k1", float(v))
    env._simulate()
    T = env._readings["transmission"]
    # rms of surviving particles at the screen, no chip cut
    stops = [d.name for d in env._stops if d.index <= env._diagnostics[SCREEN].index]
    live = np.ones(CHECK_N, bool)
    for name, x, y, a in env.transport.run(env._coordinates, stops):
        live &= a
    return T, float(x[live].std()), float(y[live].std())


def task(ij):
    s = _setup(); torch = s["torch"]
    i, j = ij
    tx, ty = float(s["grid"]["spot_sizes_m"][i]), float(s["grid"]["spot_sizes_m"][j])
    eiad = dict(zip(s["Q"], s["grid"]["k1"][i, j]))
    t0 = time.time()
    before = _check(s, eiad)
    k = torch.tensor([eiad[q.name] for q in s["quads"]], dtype=torch.float64, requires_grad=True)
    best = [np.inf, None]

    def closure():
        opt.zero_grad()
        sx, sy, T = _forward(s, k)
        loss = torch.log(sx / tx) ** 2 + torch.log(sy / ty) ** 2 + W_LOSS * (1 - T)
        if float(loss.detach()) < best[0] and bool((k.detach().abs() <= K_MAX).all()):
            best[:] = [float(loss.detach()), k.detach().clone()]
        loss.backward()
        return loss

    if OPT == "lbfgs":
        opt = torch.optim.LBFGS([k], lr=1, max_iter=STEPS, history_size=20, line_search_fn="strong_wolfe")
        opt.step(closure)
    else:
        opt = torch.optim.Adam([k], lr=LR)
        for step in range(STEPS):
            opt.step(closure)
            with torch.no_grad():
                k.clamp_(-K_MAX, K_MAX)
    with torch.no_grad():
        fx, fy, fT = _forward(s, best[1])
    new = dict(eiad); new.update({q.name: float(v) for q, v in zip(s["quads"], best[1])})
    after = _check(s, new)
    return dict(i=i, j=j, target=(tx, ty), before=before, after=after, fit_loss=best[0], fit=(float(fT), float(fx), float(fy)),
                k1=[new[q] for q in s["Q"]], seconds=time.time() - t0)


if __name__ == "__main__":
    nodes = json.loads(sys.argv[4]) if len(sys.argv) > 4 else [(i, j) for i in range(12) for j in range(12)]
    t0 = time.time(); results = []
    with Pool(WORKERS) as pool:
        for r in pool.imap_unordered(task, [tuple(n) for n in nodes]):
            results.append(r)
            b, a = r["before"], r["after"]; tx, ty = r["target"]
            print(f"({r['i']},{r['j']}) target {tx*1e3:.2f}x{ty*1e3:.2f} mm | Eiad T={b[0]:.3f} "
                  f"{b[1]*1e3:.3f}x{b[2]*1e3:.3f} | rematch T={a[0]:.3f} {a[1]*1e3:.3f}x{a[2]*1e3:.3f} "
                  f"| fit T={r['fit'][0]:.3f} {r['fit'][1]*1e3:.3f}x{r['fit'][2]*1e3:.3f} | {r['seconds']:.0f}s, total {time.time()-t0:.0f}s", flush=True)
    (OUT / "rematch_results.json").write_text(json.dumps(results))
    # machine-readable node table, same layout as optics_matching/results/validated_grid.npz
    grid = np.load(ROOT / "optics_matching/results/validated_grid.npz")
    n = len(grid["spot_sizes_m"])
    k1 = np.array(grid["k1"], dtype=float)
    T = np.full((n, n, 2), np.nan); sizes = np.full((n, n, 2, 2), np.nan)
    for r in results:
        i, j = r["i"], r["j"]
        k1[i, j] = r["k1"]
        T[i, j] = r["before"][0], r["after"][0]
        sizes[i, j, 0], sizes[i, j, 1] = r["before"][1:], r["after"][1:]
    np.savez(OUT / "rematched_grid_vbc_on.npz", spot_sizes_m=grid["spot_sizes_m"], k1=k1,
             quadrupoles=grid["quadrupoles"], transmission_eiad=T[..., 0], transmission=T[..., 1],
             actual_sizes_eiad_m=sizes[:, :, 0], actual_sizes_m=sizes[:, :, 1],
             vbc_angle_rad=VBC_ANGLE, lattice=LATTICE)
