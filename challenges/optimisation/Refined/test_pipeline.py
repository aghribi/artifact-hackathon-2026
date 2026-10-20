"""What has to keep being true. Run it after touching anything here.

    python -I test_pipeline.py            # the quick set, about a minute
    python -I test_pipeline.py --full     # and the slow end-to-end ones

Not a unit test suite. These are the handful of facts that were each learned by
getting them wrong, and each of which was invisible until something far away
produced a plausible wrong number:

- CLARA is unchanged. The environment was generalised to take a machine
  description; every published result depends on it still dealing the same
  machine, to the last digit.
- The chicane drives four dipoles *and their pole faces*. Setting an angle
  without the edges is a different, far lossier optic, and nothing shows it
  until the knob is first turned.
- The spec and the environment agree on which way is positive. They disagreed
  once, and the only symptom was a tune clipped at the wrong angle.
- The family is smooth enough to interpolate, and delivers what it promises.
- The pipeline runs on a machine nobody wrote it for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT, ROOT / "RL_framework"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

PASSED, FAILED = [], []


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASSED if ok else FAILED).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    return ok


# ---------------------------------------------------------------- the machine


def test_clara_unchanged() -> None:
    """The environment deals the machine it has always dealt.

    The reference is checked in beside this file rather than recomputed, because
    the whole point is to catch a change in what the code produces.
    """
    print("\n1. CLARA is the same machine it was")
    from rl_framework.config import RLConfig
    from rl_framework.envs.sim_env import SimEnv

    reference = json.loads((HERE / "reference" / "clara_baseline.json").read_text())
    env = SimEnv(RLConfig.clara())

    check("element count", len(env.segment.elements) == reference["n_elements"],
          f"{len(env.segment.elements)}")
    check("BPMs, in order", env.bpm_names == reference["bpm_names"],
          f"{len(env.bpm_names)}")
    check("screens, in order", env.screen_names == reference["screen_names"],
          f"{len(env.screen_names)}")

    shared = [k for k in env.magnet_names if k in reference["magnet_limits"]]
    off = [k for k in shared
           if tuple(env.magnet_limits[k]) != tuple(reference["magnet_limits"][k])]
    check("magnet limits exact", not off, f"{len(shared)} knobs, {len(off)} off")

    same = True
    for seed, expected in reference["episodes"].items():
        env.reset(seed=int(seed))
        _, _, _, _, info = env.step({"magnets": expected["action"]})
        got = [float(info[k]) for k in
               ("mu_x", "mu_y", "sigma_x", "sigma_y", "transmission",
                "benchmark_score")]
        same &= got == expected["after"]
    check("seeded episodes bit-identical", same,
          f"{len(reference['episodes'])} seeds")


def test_coupled_knobs() -> None:
    """A knob that drives several magnets drives all of them, faces included."""
    print("\n2. coupled knobs move everything they should")
    from rl_framework.config import RLConfig
    from rl_framework.envs.sim_env import SimEnv

    env = SimEnv(RLConfig.clara())
    env.reset(seed=1000)
    elements = {e.name: e for e in env.segment.elements}
    dipoles = [f"cla_vbc_mag_dip_0{i}" for i in (1, 2, 3, 4)]

    env.set_magnets({"cla_vbc_chicane": 0.10})
    angles = [float(elements[n].angle) for n in dipoles]
    check("four dipoles, alternating signs",
          np.allclose(angles, [0.10, -0.10, -0.10, 0.10]),
          f"{[round(a, 4) for a in angles]}")
    # The chicane's outer faces stay square to the beamline and the inner ones
    # follow the bend, so it alternates exit, entry, exit, entry along the string.
    inner = [float(elements[dipoles[0]].dipole_e2),
             float(elements[dipoles[1]].dipole_e1),
             float(elements[dipoles[2]].dipole_e2),
             float(elements[dipoles[3]].dipole_e1)]
    check("inner pole faces follow the body angle",
          np.allclose(inner, angles))
    check("outer pole faces stay square",
          float(elements[dipoles[0]].dipole_e1) == 0.0
          and float(elements[dipoles[3]].dipole_e2) == 0.0)


def test_sign_agreement() -> None:
    """The spec and the environment mean the same thing by a knob value.

    They disagreed once. The spec drove the chicane lead-negative and the
    environment lead-positive, so the same number was two different angles and
    the only symptom was a tune clipped at the wrong end of its range.
    """
    print("\n3. the spec and the environment agree on sign")
    from machine_spec import MachineSpec
    from rl_framework.config import RLConfig
    from rl_framework.envs.sim_env import SimEnv

    from lattice import Line

    spec = MachineSpec.load(ROOT / "machines" / "clara_febe.json")
    env = SimEnv(RLConfig.clara())
    line = Line(spec, num_particles=200)

    lead_positive = [k.name for k in spec.knobs if k.signs and k.signs[0] < 0]
    check("every knob is lead-positive in the spec", not lead_positive,
          f"{lead_positive}" if lead_positive else "")

    off = []
    for knob in spec.knobs:
        if knob.name not in env.magnet_limits:
            continue
        if not np.allclose(env.magnet_limits[knob.name], (knob.low, knob.high)):
            off.append(knob.name)
    check("limits identical in spec and environment", not off, f"{off[:3]}")

    # Same knob, same value, same magnet: the model and the machine must agree.
    name = "cla_vbc_chicane"
    env.set_magnets({name: 0.05})
    line.set({name: 0.05})
    line.track()
    first = spec.knob(name).elements[0]
    theirs = float({e.name: e for e in env.segment.elements}[first].angle)
    ours = float({e.name: e for e in line.segment.elements}[first].angle)
    check("model and machine write the same angle", np.isclose(theirs, ours),
          f"{theirs:+.4f} vs {ours:+.4f}")


# ---------------------------------------------------------------- the family


def test_family(artifact: str = "artifacts/clara_febe.npz") -> None:
    """The precomputed optics is smooth, and delivers what it promises."""
    print("\n4. the optics family is usable")
    from machine_spec import MachineSpec

    from artifact import Family
    from lattice import Line

    path = HERE / artifact
    if not path.exists():
        print(f"      skipped: no {artifact}")
        return
    family = Family.load(path)
    spec = MachineSpec.load(ROOT / "machines" / "clara_febe.json")

    check("some of it is feasible", family.feasible.any(),
          f"{family.feasible.mean():.0%} of {family.feasible.size:,}")
    check("scan axes ascend",
          all(np.all(np.diff(v) > 0) for v in family.axis_values),
          "searchsorted silently misreads a descending axis")

    # A tune the family calls feasible must deliver, on the model it was built
    # for. This is what catches an axis or a grid that has been shuffled.
    line = Line(spec, num_particles=2000)
    rng = np.random.default_rng(0)
    errors = []
    for _ in range(8):
        sx, sy = np.exp(rng.uniform(np.log(family.spot_sizes[2]),
                                    np.log(family.spot_sizes[-3]), 2))
        tune, axes = family.choose(sx, sy)
        if tune.confidence < 0.99:
            continue
        line.reset()
        line.set({**axes, **tune.k1})
        _, got = line.track().beam_at(line.readout)
        errors.append(float(np.abs(got.detach().cpu().numpy()
                                   / np.array([sx, sy]) - 1).max()))
    check("feasible tunes deliver the requested spot",
          bool(errors) and np.median(errors) < 0.05,
          f"median {np.median(errors):.2%} over {len(errors)} requests"
          if errors else "no feasible request found")


# ---------------------------------------------------------------- end to end


def test_tuning(full: bool) -> None:
    """The methods still score what they scored."""
    print("\n5. the pipeline tunes CLARA")
    from machine_spec import MachineSpec

    from artifact import Family
    from machines import SimEnvMachine
    from methods import METHODS, Session

    path = HERE / "artifacts" / "clara_febe.npz"
    if not path.exists():
        print("      skipped: no artifacts/clara_febe.npz")
        return
    spec = MachineSpec.load(ROOT / "machines" / "clara_febe.json")
    family = Family.load(path)

    # Known-good scores. They are not a target to optimise, they are a tripwire:
    # a change that moves them has changed the tuning, and should say so.
    for seed, expected in ((1000, 0.522), (1002, 0.273), (1003, 0.477)):
        machine = SimEnvMachine(spec, seed=seed)
        METHODS["steer"](Session(machine, family, spec, budget=25))
        check(f"steer, seed {seed}", abs(machine.score - expected) < 6e-4,
              f"{machine.score:.3f} against {expected:.3f}")

    if not full:
        return
    # A tripwire on one seed, not a comparison between methods: which of `steer`
    # and `mpc` wins a *single* episode is noise — over eight seeds `mpc` means
    # 0.55 against `steer`'s 2.08, and still loses on some of them individually.
    machine = SimEnvMachine(spec, seed=1002)
    METHODS["mpc"](Session(machine, family, spec, budget=25))
    check("mpc, seed 1002", abs(machine.score - 0.623) < 0.01,
          f"{machine.score:.3f} against 0.623, in {machine.shots} shots")


def test_other_machine(full: bool) -> None:
    """The point of all of it: a machine nobody wrote this for."""
    print("\n6. the pipeline runs on a lattice it was not written for")
    spec_path = ROOT / "machines" / "clear.json"
    artifact = HERE / "artifacts" / "clear.npz"
    if not spec_path.exists():
        print("      skipped: no machines/clear.json")
        return
    from machine_spec import MachineSpec
    from rl_framework.config import RLConfig
    from rl_framework.envs.sim_env import SimEnv

    spec = MachineSpec.load(spec_path)
    config = RLConfig(spec=spec, num_particles=1000,
                      requested_spot=(1.2e-4, 8e-4), requested_offset=5e-4)
    env = SimEnv(config)
    check("the environment builds it", len(env.segment.elements) > 0,
          f"{len(env.segment.elements)} elements, {len(env.bpm_names)} BPMs, "
          f"{len(env.screen_names)} screens")
    _, info = env.reset(seed=1)
    check("it reads a beam", np.isfinite(info["sigma_x"]) and info["transmission"] > 0,
          f"T {info['transmission']:.2f}, "
          f"spot {info['sigma_x'] * 1e6:.0f} x {info['sigma_y'] * 1e6:.0f} um")

    if not full or not artifact.exists():
        return
    from artifact import Family
    from machines import SimEnvMachine
    from methods import METHODS, Session

    family = Family.load(artifact)
    scores = {}
    for method in ("none", "steer"):
        runs = []
        for seed in (1, 2, 3, 4):
            machine = SimEnvMachine(spec, seed=seed, config=RLConfig(
                spec=spec, num_particles=2000,
                requested_spot=(1.2e-4, 8e-4), requested_offset=5e-4))
            METHODS[method](Session(machine, family, spec, budget=25))
            runs.append(machine.score)
        scores[method] = float(np.mean(runs))
    check("steering beats doing nothing", scores["steer"] < scores["none"],
          f"{scores['steer']:.3f} against {scores['none']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="also the slow end-to-end checks")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    test_clara_unchanged()
    test_coupled_knobs()
    test_sign_agreement()
    test_family()
    test_tuning(args.full)
    test_other_machine(args.full)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"   failed: {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
