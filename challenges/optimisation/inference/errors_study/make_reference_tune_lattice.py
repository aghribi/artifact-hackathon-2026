"""Write CLARA_cheetah_reference_tune.json: the lattice file with reference_tune.json applied."""

from __future__ import annotations

import sys
from pathlib import Path

import cheetah
import torch

CHALLENGE = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "CLARA_cheetah_reference_tune.json"


def main() -> None:
    sys.path.insert(0, str(CHALLENGE))
    import json

    lattice_dir = CHALLENGE / "data" / "clara" / "lattice_data"
    lattice = cheetah.Segment.from_lattice_json(str(lattice_dir / "CLARA_cheetah.json"))
    tune = json.loads((lattice_dir / "reference_tune.json").read_text())["elements"]
    by_name = {element.name: element for element in lattice.elements}
    for name, properties in tune.items():
        if name in by_name:
            for prop, value in properties.items():
                setattr(by_name[name], prop, torch.tensor(float(value)))
    lattice.to_lattice_json(
        str(OUTPUT), title="CLARA_FEBE", info="reference_tune applied"
    )
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
