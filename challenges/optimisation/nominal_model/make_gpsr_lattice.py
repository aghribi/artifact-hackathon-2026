"""Build a Cheetah lattice with the magnet settings of the S07 GPSR quad scan.

The settings are `S07-GPSR-Settings.csv` from Amelia's fork: 645 captures on
CLA-S07-DIA-CAM-04, 2026-03-23 14:22-15:21 UTC, used for the 4D phase-space
reconstruction at CLA-S07-SIM-APER-01. Only CLA-S07-MAG-QUAD-05 is scanned; every other
magnet current is constant, so each magnet takes the median ':READI' over the scan.

Currents are converted with `data/clara/lattice_data/utils.update_cheetah_element`, the
routine that built `reference_tune.json`, and its conventions are kept:
- corrector angles are set to zero (the ideal lattice has no misalignments to correct),
- sextupoles are set to zero (utils has no sextupole calibration and would write amps
  into k2),
- RF readbacks are not applied: they are machine phases, not Cheetah's off-crest angles,
  so the cavities keep the lattice file's voltages and phases.
The scanned quadrupole sits at its median current and is listed in the provenance; it is
downstream of the reconstruction point.

Writes into this folder:
- CLARA_cheetah_S07_GPSR.json  the full lattice with these settings
- s07_gpsr_tune.json           the tunable values in reference_tune.json's format
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import cheetah
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
LATTICE_DIR = HERE.parent / "data" / "clara" / "lattice_data"
SETTINGS_CSV = LATTICE_DIR / "S07-GPSR-Settings.csv"
SOURCE_LATTICE = LATTICE_DIR / "CLARA_cheetah.json"
OUTPUT_LATTICE = HERE / "CLARA_cheetah_S07_GPSR.json"
OUTPUT_TUNE = HERE / "s07_gpsr_tune.json"
SCAN_THRESHOLD_A = (
    0.05  # a current spanning more than this (and 1% of itself) is scanned
)
CORRECTOR_TYPES = (
    cheetah.CombinedCorrector,
    cheetah.HorizontalCorrector,
    cheetah.VerticalCorrector,
)
RECORDED = {  # element type: properties written to the tune file
    cheetah.Quadrupole: ("k1",),
    cheetah.CombinedCorrector: ("horizontal_angle", "vertical_angle"),
    cheetah.Sextupole: ("k2",),
    cheetah.Dipole: ("angle",),
}

sys.path.insert(0, str(LATTICE_DIR))
import utils


def scan_medians(settings: pd.DataFrame) -> tuple[dict, dict]:
    """Median ':READI' current per magnet PV, and the PVs that are scanned (min, max)."""
    medians, scanned = {}, {}
    for column in (name for name in settings.columns if name.endswith(":READI")):
        current = settings[column].astype(float)
        medians[column] = float(current.median())
        if current.max() - current.min() > max(
            SCAN_THRESHOLD_A, 0.01 * abs(medians[column])
        ):
            scanned[column] = (float(current.min()), float(current.max()))
    return medians, scanned


def apply_currents(lattice: cheetah.Segment, medians: dict) -> list[str]:
    """Write every current into the lattice through utils; return what utils rejected."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        for pv, current in medians.items():
            utils.update_cheetah_element(lattice, pv, current)
    return [line for line in printed.getvalue().splitlines() if line.strip()]


def zero_correctors_and_sextupoles(lattice: cheetah.Segment) -> None:
    for element in lattice.elements:
        if isinstance(element, cheetah.CombinedCorrector):
            element.horizontal_angle = torch.tensor(0.0)
            element.vertical_angle = torch.tensor(0.0)
        elif isinstance(
            element, (cheetah.HorizontalCorrector, cheetah.VerticalCorrector)
        ):
            element.angle = torch.tensor(0.0)
        elif isinstance(element, cheetah.Sextupole):
            element.k2 = torch.tensor(0.0)


def tune_of(lattice: cheetah.Segment) -> dict:
    return {
        element.name: {prop: float(getattr(element, prop)) for prop in props}
        for element in lattice.elements
        for kind, props in RECORDED.items()
        if isinstance(element, kind)
    }


def main() -> None:
    settings = pd.read_csv(SETTINGS_CSV)
    medians, scanned = scan_medians(settings)
    original = cheetah.Segment.from_lattice_json(str(SOURCE_LATTICE))
    lattice = cheetah.Segment.from_lattice_json(str(SOURCE_LATTICE))
    rejected = apply_currents(lattice, medians)
    zero_correctors_and_sextupoles(lattice)

    provenance = {
        "source": f"{SETTINGS_CSV.name}: {len(settings)} captures on {settings.camera.iloc[0]}, "
        f"{settings.timestamp.min()} to {settings.timestamp.max()}",
        "built_by": "median ':READI' per magnet over the scan, converted with "
        "data/clara/lattice_data/utils.update_cheetah_element (nominal_model/make_gpsr_lattice.py)",
        "scanned": {
            pv: {"min_A": low, "max_A": high, "median_A": medians[pv]}
            for pv, (low, high) in scanned.items()
        },
        "conventions": "Corrector angles and sextupole k2 set to zero, as in reference_tune.json. RF "
        "readbacks not applied (machine phases, not off-crest angles); cavities keep "
        "the lattice file's voltages and phases. Conversion energies are utils.beam_energy.",
        "conversion_energies_mev": utils.beam_energy,
        "pvs_not_in_lattice": rejected,
    }
    OUTPUT_TUNE.write_text(
        json.dumps({"_provenance": provenance, "elements": tune_of(lattice)}, indent=1)
    )
    lattice.to_lattice_json(
        str(OUTPUT_LATTICE), title="CLARA_FEBE", info=json.dumps(provenance)
    )

    reloaded = cheetah.Segment.from_lattice_json(str(OUTPUT_LATTICE))
    assert [e.name for e in reloaded.elements] == [e.name for e in lattice.elements], (
        "element sequence changed"
    )
    assert tune_of(reloaded) == tune_of(lattice) or all(
        abs(a - b) < 1e-6
        for name in tune_of(lattice)
        for a, b in zip(
            tune_of(lattice)[name].values(), tune_of(reloaded)[name].values()
        )
    ), "values changed on reload"

    original_tune = tune_of(original)
    print(f"{len(medians)} READI PVs, scanned: {scanned}")
    print(f"{len(rejected)} PVs not mapped by utils, e.g. {rejected[:4]}")
    print(f"{'element':26s} {'property':16s} {'lattice file':>14s} {'GPSR scan':>12s}")
    for name, props in tune_of(lattice).items():
        for prop, value in props.items():
            before = original_tune[name][prop]
            if abs(value - before) > 1e-9:
                print(f"{name:26s} {prop:16s} {before:14.5g} {value:12.5g}")
    print("wrote", OUTPUT_LATTICE.name, "and", OUTPUT_TUNE.name)


if __name__ == "__main__":
    main()
