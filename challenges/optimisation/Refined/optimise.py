"""Stage 2: tune one episode from a precomputed family, by a method of your choice.

    python -I optimise.py machines/clara_febe.json artifacts/clara_febe.npz \
        --method steer --seed 1000

    python -I optimise.py machines/clara_febe.json artifacts/clara_febe.npz \
        --method mpc --seed 1002

The cheap half. Everything expensive happened in `precompute.py`; this reads the
artifact, looks the request up, and spends shots. A lookup is microseconds against
the minutes a match costs, which is the whole reason the agent can afford to load
the optics before it has taken a single shot.

Methods are in `methods.py` and stack: `optics` loads the family's answer,
`steer` adds the orbit and the placement, `mpc` adds a planned tweak against a
twin, `bo` adds Bayesian optimisation. `none` is the do-nothing baseline, and
every number is only meaningful beside it.

The figures of merit are the ones that cost real time on a real machine: the
benchmark score, the number of shots, the number of times the beam-intercepting
screen had to be moved (about half a minute each), and the wall clock.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from artifact import Family  # noqa: E402
from machine import MachineSpec  # noqa: E402
from machines import SimEnvMachine  # noqa: E402
from methods import METHODS, OutOfShots, Session  # noqa: E402


def build_machine(spec: MachineSpec, seed: int):
    return SimEnvMachine(spec, seed=seed)


def run(spec: MachineSpec, family: Family, method: str, seed: int,
        budget: int, particles: int, say=print) -> dict:
    began = time.perf_counter()
    machine = build_machine(spec, seed)
    session = Session(machine, family, spec, budget=budget)
    start = machine.last

    try:
        METHODS[method](session)
    except OutOfShots as stop:
        session.say(str(stop))

    seconds = time.perf_counter() - began
    final = machine.last
    result = {
        "method": method, "seed": seed,
        "score": machine.score,
        "best_seen": machine.best,
        "score_before": start.score,
        "relative_error": machine.relative_error,
        "shots": machine.shots,
        "screen_switches": machine.screen_switches,
        "seconds": round(seconds, 1),
        "machine_seconds": machine.machine_seconds,
        "target": machine.target.tolist(),
        "achieved": final.spot.tolist(),
        "transmission": final.transmission,
        "on_screen": final.on_screen,
        "log": session.log,
    }

    say("")
    say(f"  {method}, seed {seed}")
    for line in session.log:
        say(f"      {line}")
    say("")
    t, a = machine.target, final.spot
    say(f"  requested  mu {t[0] * 1e3:+8.4f} {t[1] * 1e3:+8.4f} mm   "
        f"sigma {t[2] * 1e3:8.4f} {t[3] * 1e3:8.4f} mm")
    say(f"  achieved   mu {a[0] * 1e3:+8.4f} {a[1] * 1e3:+8.4f} mm   "
        f"sigma {a[2] * 1e3:8.4f} {a[3] * 1e3:8.4f} mm"
        + ("" if final.on_screen else "   [off screen]"))
    say(f"  transmission {final.transmission:.3f}")
    say("")
    say(f"  score            {machine.score:8.3f}   "
        f"(do nothing: {start.score:.3f}, best seen: {machine.best:.3f})")
    say(f"  relative error   {machine.relative_error:8.3f}")
    say(f"  shots            {machine.shots:8d}")
    say(f"  screen switches  {machine.screen_switches:8d}")
    say(f"  wall clock       {seconds:8.1f} s")
    say(f"  machine time     {machine.machine_seconds:8.1f} s")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--method", default="steer", choices=sorted(METHODS))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=25,
                        help="shots the method may take")
    parser.add_argument("--particles", type=int, default=2000,
                        help="particles, for the `line` machine and the twin")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    spec = MachineSpec.load(args.spec)
    family = Family.load(args.artifact)
    print(f"  {args.artifact}")
    print("  " + family.summary())
    if family.provenance.get("spec") != spec.name:
        print(f"  warning: this family was built for "
              f"{family.provenance.get('spec')!r}, not {spec.name!r}")

    result = run(spec, family, args.method, args.seed,
                 args.budget, args.particles)
    if args.json:
        import json

        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1, default=float))
        print(f"\n  -> {args.json}")


if __name__ == "__main__":
    main()
