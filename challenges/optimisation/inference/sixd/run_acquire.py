"""Acquire the full image set from the virtual machine (true beam hidden) and cache it."""

from __future__ import annotations

import sys
import time

import common
import full6d
import numpy as np

cv = common.cv
xsuite_model = common.xsuite_model

OUT = common.HERE / "images.npz"
PARTICLES = 100000


def main() -> None:
    hidden = "--errors" in sys.argv
    t0 = time.time()
    machine_line = full6d.build_line()
    line = full6d.build_line()
    beam, _, _ = xsuite_model.load_distribution()
    rng = np.random.default_rng(0)
    beam = beam[rng.choice(len(beam), PARTICLES, replace=False)]
    errors = (
        cv.HiddenErrors.random(machine_line.inventory, seed=0) if hidden else cv.HiddenErrors()
    )
    machine = cv.VirtualMachine(machine_line, beam, errors, seed=1)
    for q, (dx, dy) in errors.quad_offsets.items():
        machine_line.line[q].shift_x, machine_line.line[q].shift_y = dx, dy
    configs = full6d.make_plan(line)
    print("built", time.time() - t0, "screens", len(line.inventory.screens), flush=True)
    for c in configs:
        print(c.name, c.tdc_voltage_v, len(c.screens), flush=True)
    images = full6d.acquire_images(machine, configs)
    keys = sorted(images)
    np.savez_compressed(
        OUT if not hidden else common.HERE / "images_errors.npz",
        keys=np.array([f"{c}|{s}" for c, s in keys]),
        images=np.array([images[k] for k in keys]),
        beam=beam,
        shots=machine.shots,
    )
    print("done", time.time() - t0, "images", len(keys), "shots", machine.shots)


if __name__ == "__main__":
    main()
