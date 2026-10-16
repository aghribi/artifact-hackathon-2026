"""Run a measurement plan on Amy's SimEnv (RL_framework, xsuite backend) and store the
images in the images.npz layout, with each screen's own chip geometry.

The episode's machine is SimEnv's: its own beam (size, shape, offset), magnet set-point
errors, 50 um quad/sextupole misalignments, 30 um diagnostic offsets and shot jitter.
Only `env.step` is used to set magnets and cavity phases, pick the screen and read BPMs
and images. The one exception is the true beam, read for scoring only.
"""

from __future__ import annotations

import sys

import common
import full6d
import numpy as np

sys.path.insert(0, str(common.HERE.parents[1] / "RL_framework"))
sys.path.insert(0, str(common.HERE.parent / "steering"))

SHAPE = (1024, 1024)  # camera grid; the reconstruction rebins it by REBIN
REBIN = 4
L02 = "cla_l02_lin_cav_01"
# SimEnv phase knobs are Cheetah phases in degrees; a Config's rf_phase_deg offset p
# (full6d.set_rf_phases, xsuite phase - p) is Cheetah phase + PHASE_SIGN * p
PHASE_SIGN = 1.0


def make_env(num_particles=20_000, seed=0, lattice=None, **kwargs):
    """SimEnv (xsuite backend) on the plan's lattice (full6d.plan_lattice: VBC off, L4H
    off) unless `lattice` names another file."""
    from rl_framework.config import RLConfig
    from rl_framework.envs.sim_env import SimEnv

    lattice = str(lattice or full6d.plan_lattice())
    env = SimEnv(RLConfig(backend="xsuite", num_particles=num_particles, normalise_image=False,
                          nominal_lattice_json=lattice,
                          screen_shape=SHAPE, max_episode_steps=10**9, **kwargs))
    env.reset(seed=seed)
    return env


def nominal_settings(env):
    """Every magnet at the strength the lattice file carries. In SimEnv a commanded
    setting is exact, so this removes the episode's set-point errors (survey errors and
    the beam stay)."""
    return dict(env._nominal_strengths)


def shot(env, magnets, screen):
    obs, *_ = env.step({"magnets": magnets, "screen": screen})
    return obs


def svd_steer(env, model, base, iterations=3, cutoff=1e-3, screen="cla_feh_dia_scr_01"):
    """One-shot SVD orbit correction (steering/svd_correction.py): corrector response
    from the design model, one BPM shot per iteration. Returns the corrector settings
    {knob: angle} on top of `base` and the kicks {corrector: (h, v)} for the model."""
    import svd_correction

    bpms = [b for b in env.bpm_names if b in model.line.element_names]
    correctors = [c for c in model.inventory.correctors
                  if c in model.line.element_names and f"{c}_h" in env.magnet_names]
    response = svd_correction.corrector_response(model, correctors, bpms)
    inverse, _, _ = svd_correction.svd_pseudo_inverse(response, cutoff)
    kicks = np.zeros(response.shape[1])
    settings = dict(base)
    index = [env.bpm_names.index(b) for b in bpms]
    for _ in range(iterations):
        obs = shot(env, settings, screen)
        valid = obs["bpm_valid"][index].astype(bool)
        orbit = np.ravel(np.column_stack([obs["bpm_x"][index], obs["bpm_y"][index]]) * valid[:, None])
        kicks = kicks - inverse @ orbit
        for i, c in enumerate(correctors):
            settings[f"{c}_h"] = base[f"{c}_h"] + kicks[2 * i]
            settings[f"{c}_v"] = base[f"{c}_v"] + kicks[2 * i + 1]
    applied = {c: (settings[f"{c}_h"], settings[f"{c}_v"]) for c in correctors}
    return settings, applied


def config_magnets(base, config):
    """Magnet and phase settings for a Config. A SimEnv phase change moves the reference
    energy downstream and its magnet strengths are normalised to it, so the magnets
    follow the energy: that is Config.rf_rescale = True (checked on FEA-01, 15 Oct)."""
    out = dict(base)
    out.update(config.k1)
    for cavity, p in config.rf_phase_deg.items():
        out[cavity] = base[cavity] + PHASE_SIGN * p
    return out


def acquire(env, configs, keys, base, shots=10, out=common.HERE / "images_simenv.npz"):
    """Images for (config, screen) keys: `shots` steps each, summed. Also the mean BPM
    readings of every shot (valid ones), for orbit work."""
    by_name = {c.name: c for c in configs}
    images, bpm, valid = [], [], []
    for cname, screen in keys:
        config = by_name[cname]
        magnets = config_magnets(base, config)
        img, xs, ys, vs = 0, [], [], []
        for _ in range(shots):
            obs = shot(env, magnets, screen)
            img = img + obs["screen_image"].astype(float)
            xs.append(obs["bpm_x"]); ys.append(obs["bpm_y"]); vs.append(obs["bpm_valid"])
        images.append(img)
        bpm.append(np.stack([np.mean(xs, 0), np.mean(ys, 0)], 1))
        valid.append(np.min(vs, 0))
    screens = sorted({s for _, s in keys})
    geometry = [(*env.screen_extent(s), SHAPE[1] // REBIN, SHAPE[0] // REBIN, 0.0) for s in screens]
    beam = env._coordinates  # the episode's true beam: scoring only
    np.savez_compressed(out, keys=np.array([f"{c}|{s}" for c, s in keys]), images=np.array(images),
                        beam=beam, n_particles=len(beam) * shots, geometry_screens=np.array(screens),
                        geometry=np.array(geometry), bpm=np.array(bpm), bpm_valid=np.array(valid),
                        bpm_names=np.array(env.bpm_names))
    return out


# S03/S04 quadrupoles for the FEA-01 images that take the beam through the 4.12 mm L4H
# iris with the apertures in (VBC off): 98.6 % reaches FEA-01 at every phase, against
# 70 % with the nominal S03/S04 setting. Nelder-Mead on SimEnv transmission, 15 Oct.
FEA_L4H_MATCH = {
    "cla_s03_mag_quad_01": -1.4224,
    "cla_s04_mag_quad_01": 2.6981,
    "cla_s04_mag_quad_02": -0.4489,
    "cla_s04_mag_quad_03": -0.7350,
    "cla_s04_mag_quad_04": -1.6278,
    "cla_s04_mag_quad_05": -0.1423,
    "cla_s04_mag_quad_06": -0.2843,
    "cla_s04_mag_quad_07": 0.8083,
    "cla_s04_mag_quad_08": -0.1372,
}


def no_tdc_plan(line, phases=(-30.0, -23.0, -15.0, -8.0, 0.0), rescale=True, fea_match=None,
                fea_screen="cla_fea_dia_scr_01"):
    """Transverse block of the 21-image plan (S02-03, S03-01, S04-03 under nominal, s02a,
    s02b, s02c) and the FEA-01 spectrometer setting at several L02 phases, TDC off.
    `fea_match` adds quadrupole settings to the FEA images (FEA_L4H_MATCH, spectrometer
    optics); `fea_screen` is the screen they are taken on."""
    plan = {c.name: c for c in full6d.make_plan(line)}
    screens = ["cla_s02_dia_scr_03", "cla_s03_dia_scr_01", "cla_s04_dia_scr_03"]
    configs = [full6d.Config(n, plan[n].k1, 0.0, screens) for n in ("nominal_off", "s02a", "s02b", "s02c")]
    fea_k1 = {**plan["fea_01_off"].k1, **(fea_match or {})}
    for p in phases:
        configs.append(full6d.Config(f"fea_{fea_screen[-2:]}_ph{p:+.0f}", fea_k1, 0.0, [fea_screen], {L02: p}, rescale))
    keys = [(c.name, s) for c in configs for s in c.screens]
    return configs, keys
