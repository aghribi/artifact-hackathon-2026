"""Animate one `Refined` episode: the four stages, shot by shot.

    python3 -I make_refined_episode_animation.py --seed 1000
    python3 -I make_refined_episode_animation.py --method steer --seed 1003
    python3 -I make_refined_episode_animation.py --scan 1000 1007

One frame is one shot of the machine, in the order the agent took them — nothing
interpolated, nothing re-ordered. The drawing is `make_mbrl_episode_animation`'s:
the readout camera on the left, zoomed to a window fixed on the *requested*
position so the spot flies into a stationary target, and the score so far on the
right. Only the recording differs, because the pipeline does.

What there is to see is the shape of the pipeline:

1. **optics** — one shot. The family is looked up and its strengths loaded, with
   the compressor set to whatever value makes the request reachable. This is
   where the spot gets roughly the right size and is usually still in the wrong
   place.
2. **steer** — the orbit flattened against the model's own orbit, then the spot
   placed by a linear solve on the two correctors the BPMs cannot see. This is
   the biggest single jump in the score, because the benchmark divides every
   error by the requested spot and the request is a fifth of a millimetre.
3. **focus** — CEM against a twin anchored on the last reading, moving the
   matched quadrupoles, the sextupoles and the chicane. Size only.
4. **steer again** — the focus moved the quadrupoles, so the spot moved with
   them; placed once more against a response re-measured at the new optics.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHALLENGE = HERE.parents[1]
for path in (CHALLENGE, CHALLENGE / "RL_framework", CHALLENGE / "Refined"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _drawing():
    """The sibling script's rendering, loaded by path rather than by name.

    `talk_figures/scripts` is not a package and the file name is not an
    identifier anybody would import, so this is the honest way to share it.
    """
    spec = importlib.util.spec_from_file_location(
        "_mbrl_animation", HERE / "make_mbrl_episode_animation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Which stage each shot belongs to. The methods announce it on the `Session`, so
# this is read rather than inferred. An earlier version guessed from which knobs
# the action moved and mislabelled every restore shot as an optics load, because
# restoring re-writes all 109 magnets and looks identical from outside.
STAGE_TEXT = {
    "start": "the machine as the day dealt it",
    "optics": "1 · family optics loaded, compressor chosen for the request",
    "steer": "2 · steering the orbit on the BPMs",
    "place": "2 · placing the spot, one linear solve",
    "screens": "2 · reading the orbit off an upstream screen",
    "hunt": "2 · the beam is lost; bisecting the screens to find it",
    "return": "2 · the readout back in the beam, 30 s a switch",
    "focus": "3 · matching the spot size against the twin (CEM)",
    "bayes": "3 · matching the spot size against the twin (multi-fidelity BO)",
    "final": "4 · placing the spot again, at the new optics",
    "restore": "back to the best shot taken",
}


def record(seed: int, method: str, budget: int, particles: int,
           spec_path: Path, artifact: Path):
    """Run one episode, keeping the camera frame and the numbers of every shot."""
    from artifact import Family
    from machine import MachineSpec
    from machines import SimEnvMachine
    from methods import METHODS, Session

    spec = MachineSpec.load(spec_path)
    family = Family.load(artifact)
    machine = SimEnvMachine(spec, seed=seed)
    env = machine.env
    extent = env.screen_extent()

    frames: list[dict] = []
    session = Session(machine, family, spec, budget=budget,
                      twin_particles=particles)

    def note(observation, info, stage):
        on_screen = bool(info["on_screen"])
        # Which camera this is. The orbit stage puts an upstream screen in the
        # beam, and that picture is not the readout's — it must not be drawn
        # against the request, so the frame carries the screen and its extent.
        screen = info["screen"]
        elsewhere = None if screen == env.scored_screen else screen
        frames.append({
            "elsewhere": elsewhere,
            "extent": env.screen_extent(screen),
            "image": np.asarray(observation["screen_image"], dtype=float),
            "target": np.asarray(observation["target"], dtype=float),
            "moments": (np.array([info["mu_x"], info["mu_y"],
                                  info["sigma_x"], info["sigma_y"]])
                        if on_screen else None),
            "on_screen": on_screen,
            "transmission": float(info["transmission"]),
            "shot": int(info["evaluations"]),
            "score": float(info["benchmark_score"]),
            "mismatch": float(info["scored_mismatch"]),
            "stage": stage,
        })

    original_step = env.step

    def step(action):
        observation, reward, terminated, truncated, info = original_step(action)
        note(observation, info, session.stage)
        return observation, reward, terminated, truncated, info

    note(env._get_obs(), env._get_info(), "start")
    env.step = step
    METHODS[method](session)
    return frames, machine, session, extent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--spec", type=Path,
                        default=CHALLENGE / "Refined/machines/clara_febe.json")
    parser.add_argument("--artifact", type=Path,
                        default=CHALLENGE / "Refined/artifacts/clara_febe.npz")
    parser.add_argument("--method", default="mpc")
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--particles", type=int, default=1000,
                        help="particles the planning twin tracks; the "
                             "machine itself is SimEnv's 10,000")
    parser.add_argument(
        "--scan", type=int, nargs=2, metavar=("FIRST", "LAST"),
        help="report shots and score for a range of seeds, and render nothing",
    )
    parser.add_argument(
        "--out", type=Path, default=HERE.parent / "refined_episode",
    )
    args = parser.parse_args()

    import torch

    torch.set_num_threads(4)
    drawing = _drawing()
    drawing.STAGE_TEXT.update(STAGE_TEXT)

    if args.scan:
        for seed in range(args.scan[0], args.scan[1] + 1):
            frames, machine, _, _ = record(
                seed, args.method, args.budget, args.particles,
                args.spec, args.artifact
            )
            print(f"  seed {seed}  shots {machine.shots:>2}  "
                  f"score {machine.score:6.3f}  "
                  f"stages {[f['stage'] for f in frames]}", flush=True)
        return

    print(f"running {args.method}, seed {args.seed}")
    frames, machine, session, extent = record(
        args.seed, args.method, args.budget, args.particles,
        args.spec, args.artifact
    )
    target = machine.target
    print(f"  {machine.shots} shots, score {machine.score:.3f}, "
          f"requested {target[2] * 1e6:.0f} x {target[3] * 1e6:.0f} um")
    for line in session.log:
        print(f"    {line}")

    # The machine names itself, and says whether it has BPMs to steer on — both
    # read off the spec, so any lattice titles its own animation.
    spec = session.spec
    title = (
        f"{spec.name} · one episode of the Refined pipeline"
        f" ({args.method}{'' if spec.bpms else ', no BPMs'})"
        f"        final score {machine.score:.2f}"
        f" in {machine.shots} shots, {machine.screen_switches} screen changes"
    )
    # One directory per run: the frames are numbered, so two methods rendered
    # into the same place would overwrite each other's. The spec names the
    # machine, so two specs do not collide either.
    tag = f"{args.spec.stem}_{args.method}_{args.seed}"
    drawing.render(frames, extent, args.out / tag, machine, title=title,
                   name=f"refined_{tag}")


if __name__ == "__main__":
    main()
