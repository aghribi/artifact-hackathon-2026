"""Retake the images of some configs (after an optics redesign) and replace them in the
cached image set; the previous file is kept as <name>_prev.npz."""

from __future__ import annotations

import shutil
import sys

import common
import full6d
import numpy as np

cv = common.cv


def reacquire(prefix: str, path=common.HERE / "images.npz", line=None, machine_line=None, hidden=False):
    """`hidden`: the machine has the run_acquire.py --errors hidden errors (seed 0)."""
    old = np.load(path)
    keys, images = list(old["keys"]), list(old["images"])
    beam = old["beam"]
    machine_line = machine_line or full6d.build_line()
    errors = cv.HiddenErrors.random(machine_line.inventory, seed=0) if hidden else cv.HiddenErrors()
    machine = cv.VirtualMachine(machine_line, beam, errors, seed=1)
    for q, (dx, dy) in errors.quad_offsets.items():
        machine_line.line[q].shift_x, machine_line.line[q].shift_y = dx, dy
    configs = [c for c in full6d.make_plan(line or machine_line) if c.name.startswith(prefix)]
    new = full6d.acquire_images(machine, configs, seed=4)
    for (cname, screen), img in new.items():
        key = f"{cname}|{screen}"
        if key in keys:
            images[keys.index(key)] = img
        else:
            keys.append(key)
            images.append(img)
    shutil.copy(path, path.with_name(path.stem + "_prev.npz"))
    order = np.argsort(keys)
    np.savez_compressed(path, keys=np.array(keys)[order], images=np.array(images)[order], beam=beam,
                        shots=old["shots"] + machine.shots)
    return sorted(new)


if __name__ == "__main__":
    print(reacquire(sys.argv[1] if len(sys.argv) > 1 else "fea"))
