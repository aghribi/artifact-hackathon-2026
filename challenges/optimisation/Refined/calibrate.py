"""How many screens is it worth moving? Measure it, once, per machine.

    python -I calibrate.py machines/clara_febe_no_bpms.json artifacts/clara_febe.npz
    python -I calibrate.py machines/clara_febe_no_bpms.json artifacts/clara_febe.npz --apply

A machine with no working BPMs has to read its orbit off a screen, and a screen
is expensive: only one may be in the beam, it intercepts so nothing downstream
reads, and moving it costs about half a minute against roughly a second for a
shot. How many are worth moving is not a thing to guess — it depends on the
lattice, on where the screens are, and on how far the orbit actually wanders.

So this runs the tuner at each budget over a set of seeds and reports what each
one buys. The answer for CLARA is one: measured over 12 seeds, 2.028 with no
screen, 1.671 with one, 1.793 with two, 1.645 with three, against 1.680 on the
same machine with its BPMs working. One screen recovers the BPMs; the rest is
flat and costs 30 seconds apiece.

**Why it lives here and not in the interview.** The obvious homes are
`describe.py`, which asks the user everything else about a machine, and
`sensitivity.py`, which measures the things the user should not have to guess.
Neither can: the screen stage needs the family's response matrix and model
orbit, and both of those scripts run before `precompute.py` has built one. It
is not a question of time — a sweep is about two minutes — but of what exists
when. So the pipeline is:

    describe.py -> sensitivity.py -> precompute.py -> calibrate.py -> optimise.py

and this is the only step in it that needs both a machine and a family.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Everything this needs is in `Refined/`: the tuning modules, `machine_spec`
# and `rl_framework` are all siblings here now.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from artifact import Family  # noqa: E402
from machine import MachineSpec  # noqa: E402
from machines import SimEnvMachine  # noqa: E402
from methods import Session, steer  # noqa: E402

# How much of the available improvement a budget has to capture before the
# cheaper one is preferred. Switches cost machine time the benchmark does not
# charge for, so a budget that buys the last few percent is not worth it.
ENOUGH = 0.8


def sweep(spec: MachineSpec, family: Family, budgets, seeds,
          say=print) -> list[dict]:
    """Run the tuner at each budget over the same seeds."""
    rows = []
    for budget in budgets:
        began = time.perf_counter()
        episodes = []
        for seed in seeds:
            machine = SimEnvMachine(spec, seed=seed)
            steer(Session(machine, family, spec, budget=25),
                  screen_budget=budget)
            episodes.append((machine.score, machine.shots,
                             machine.screen_switches, machine.machine_seconds,
                             machine.last.sigma_y))
        episodes = np.array(episodes, dtype=float)
        rows.append({
            "budget": budget,
            "score": float(episodes[:, 0].mean()),
            "median": float(np.median(episodes[:, 0])),
            "shots": float(episodes[:, 1].mean()),
            "switches": float(episodes[:, 2].mean()),
            "machine_seconds": float(episodes[:, 3].mean()),
            "sigma_y": float(episodes[:, 4].mean()),
            "seconds": round(time.perf_counter() - began, 1),
        })
        say("  %6d %8.3f %8.3f %7.1f %9.1f %8.0f %8.0f" % (
            budget, rows[-1]["score"], rows[-1]["median"], rows[-1]["shots"],
            rows[-1]["switches"], rows[-1]["machine_seconds"],
            rows[-1]["sigma_y"] * 1e6))
    return rows


def choose(rows: list[dict]) -> int:
    """The cheapest budget that captures most of what is on offer.

    Lower score is better, and a switch costs half a minute the benchmark never
    charges for — so the right budget is not the best-scoring one but the
    smallest that gets most of the way there. With the improvements within
    noise of each other this picks zero, which is the honest answer when
    looking at a screen buys nothing.
    """
    base = rows[0]["score"]
    best = min(row["score"] for row in rows)
    if base - best <= 0:
        return rows[0]["budget"]
    for row in rows:
        if (base - row["score"]) >= ENOUGH * (base - best):
            return row["budget"]
    return rows[-1]["budget"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs=2, default=[1000, 1012],
                        metavar=("FIRST", "LAST"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--apply", action="store_true",
                        help="write the chosen budget into the spec")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    spec = MachineSpec.load(args.spec)
    family = Family.load(args.artifact)
    seeds = range(args.seeds[0], args.seeds[1])
    print(f"  {spec.name}: {len(spec.bpms)} BPM(s), {len(spec.screens)} screens, "
          f"readout {spec.readout}")
    if spec.bpms:
        print("  this machine can read its orbit every shot, so the screen stage "
              "never engages; expect a flat sweep.")
    print(f"  {len(list(seeds))} seeds per budget\n")
    print("  budget    score   median   shots  switches  mach.s  sigma_y")
    rows = sweep(spec, family, args.budgets, seeds)

    pick = choose(rows)
    base, chosen = rows[0], next(r for r in rows if r["budget"] == pick)
    print(f"\n  chosen budget: {pick}")
    if pick == 0:
        print("  looking at a screen does not pay on this machine.")
    else:
        print(f"  it takes the score from {base['score']:.3f} to "
              f"{chosen['score']:.3f} for {chosen['switches']:.0f} switch(es), "
              f"{chosen['machine_seconds'] - base['machine_seconds']:.0f} s more "
              f"machine time per episode.")

    if args.apply:
        raw = json.loads(args.spec.read_text())
        raw["screen_budget"] = int(pick)
        raw.setdefault("notes", []).append(
            f"screen_budget {pick} measured by calibrate.py over "
            f"{len(list(seeds))} seeds: "
            + ", ".join(f"{r['budget']}->{r['score']:.3f}" for r in rows)
        )
        args.spec.write_text(json.dumps(raw, indent=1))
        print(f"  -> written to {args.spec}")
    else:
        print("  (nothing written; pass --apply to put it in the spec)")


if __name__ == "__main__":
    main()
