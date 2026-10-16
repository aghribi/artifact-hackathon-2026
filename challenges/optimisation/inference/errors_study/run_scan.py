"""Run the Monte Carlo scan in parallel; each worker builds its lines once per lattice.

Usage: run_scan.py [--workers 14] [--quick]
Writes results/<config>.jsonl (one record per seed and BPM offset level) and results/meta.json.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
from pathlib import Path

import error_study as es
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LATTICES = {"design": None, "reftune": str(HERE / "CLARA_cheetah_reference_tune.json")}
UM = 1e-6
QUADS_UM = [0, 50, 100, 200, 300]
BPMS_UM = [0, 50, 100, 200]
DEFAULT_LAUNCH = (300 * UM, 30 * UM)
BIG_LAUNCH = (1000 * UM, 100 * UM)

# name: lattice, quad levels (um), BPM levels (um), launch (m, rad), delta rms, cutoff, shots, jitter, seeds
CONFIGS = {
    "main": {
        "lattice": "design",
        "quads": QUADS_UM,
        "bpms": BPMS_UM,
        "launch": DEFAULT_LAUNCH,
        "seeds": 200,
    },
    "big_launch": {
        "lattice": "design",
        "quads": QUADS_UM[1:],
        "bpms": BPMS_UM,
        "launch": BIG_LAUNCH,
        "seeds": 100,
    },
    "no_launch": {
        "lattice": "design",
        "quads": [100],
        "bpms": BPMS_UM,
        "launch": (0.0, 0.0),
        "seeds": 100,
    },
    "no_jitter": {
        "lattice": "design",
        "quads": [100, 300],
        "bpms": BPMS_UM,
        "launch": DEFAULT_LAUNCH,
        "jitter_scale": 0.0,
        "seeds": 100,
    },
    "shots10": {
        "lattice": "design",
        "quads": [100, 300],
        "bpms": BPMS_UM,
        "launch": DEFAULT_LAUNCH,
        "shots": 10,
        "seeds": 100,
    },
    "cutoff_5e-5": {
        "lattice": "design",
        "quads": [100],
        "bpms": BPMS_UM,
        "launch": DEFAULT_LAUNCH,
        "cutoff": 5e-5,
        "seeds": 100,
    },
    "cutoff_1e-2": {
        "lattice": "design",
        "quads": [100],
        "bpms": BPMS_UM,
        "launch": DEFAULT_LAUNCH,
        "cutoff": 1e-2,
        "seeds": 100,
    },
    "delta_0p2pct": {
        "lattice": "design",
        "quads": [100],
        "bpms": BPMS_UM,
        "launch": DEFAULT_LAUNCH,
        "delta": 2e-3,
        "seeds": 100,
    },
    "target_design": {
        "lattice": "design",
        "quads": [100, 300],
        "bpms": BPMS_UM,
        "launch": DEFAULT_LAUNCH,
        "target": "design",
        "seeds": 100,
    },
    "reftune_target_design": {
        "lattice": "reftune",
        "quads": [100],
        "bpms": BPMS_UM,
        "launch": DEFAULT_LAUNCH,
        "target": "design",
        "seeds": 100,
    },
    "used_correctors": {
        "lattice": "design",
        "quads": [100, 300],
        "bpms": BPMS_UM,
        "launch": DEFAULT_LAUNCH,
        "planes": "used",
        "seeds": 100,
    },
    "reftune": {
        "lattice": "reftune",
        "quads": [100],
        "bpms": BPMS_UM,
        "launch": DEFAULT_LAUNCH,
        "seeds": 100,
    },
}

_contexts: dict[str, es.Context] = {}


def _context(lattice: str) -> es.Context:
    if lattice not in _contexts:
        _contexts[lattice] = es.build_context(LATTICES[lattice])
    return _contexts[lattice]


def _task(args: tuple[str, int, float]) -> tuple[str, list[dict]]:
    name, seed, quad_um = args
    config = CONFIGS[name]
    context = _context(config["lattice"])
    records = es.run_seed(
        context,
        seed,
        quad_um * UM,
        [b * UM for b in config["bpms"]],
        config["launch"],
        delta_rms=config.get("delta", 0.0),
        cutoff=config.get("cutoff", es.CUTOFF),
        shots=config.get("shots", 1),
        jitter_scale=config.get("jitter_scale", 1.0),
        target=config.get("target", "zero"),
        planes=config.get("planes", "all"),
    )
    for record in records:
        record["config"] = name
    return name, records


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    tasks = []
    for name, config in CONFIGS.items():
        if args.only and name not in args.only:
            continue
        seeds = 4 if args.quick else config["seeds"]
        tasks += [(name, seed, q) for seed in range(seeds) for q in config["quads"]]
    # interleave so slow lattices do not end up on one worker
    with (
        contextlib.ExitStack() as stack,
        mp.get_context("spawn").Pool(args.workers) as pool,
    ):
        handles = {
            name: stack.enter_context(open(RESULTS / f"{name}.jsonl", "w"))
            for name in {t[0] for t in tasks}
        }
        for done, (name, records) in enumerate(pool.imap_unordered(_task, tasks), 1):
            for record in records:
                handles[name].write(
                    json.dumps({k: _jsonable(v) for k, v in record.items()}) + "\n"
                )
            handles[name].flush()
            if done % 50 == 0:
                print(f"{done}/{len(tasks)}", flush=True)
    context = _context("design")
    for cutoff in (1e-2, es.CUTOFF, 5e-5):
        es.get_inverse(context, cutoff)
    meta = {
        "configs": CONFIGS,
        "n_particles": len(context.beam),
        "modes_kept": {str(k): v[1] for k, v in context.inverses.items()},
        "n_correctors": len(context.correctors),
        "baseline_design": {k: float(v) for k, v in context.baseline.items()},
        "reference_ip_design": {k: float(v) for k, v in context.reference_ip.items()},
        "design_optics_ip": {k: float(v[-1]) for k, v in context.design_optics.items()},
        "corrector_ranges_urad": {
            c: (context.corrector_ranges[2 * i : 2 * i + 2] * 1e6).tolist()
            for i, c in enumerate(context.correctors)
        },
    }
    (RESULTS / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    main()
