"""CLARA's own names, for the CLARA-specific code that still wants them.

`SimEnv` used to carry these as module constants, which is what made it a CLARA
environment rather than an environment. They are facts about one machine, so
they live beside that machine's spec — and they are *derived* from it, not
restated, so there is still one place to change them.

    from machine_spec.sources.clara import VBC_CHICANE, VBC_EDGE, BPM_TAG, SCREEN_TAG

New code should read the spec instead: `MachineSpec.knob("cla_vbc_chicane")`
carries the elements, the signs and the pole faces, and `spec.bpms` /
`spec.screens` name the diagnostics outright, so nothing has to match on a name.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from machine_spec import MachineSpec  # noqa: E402

SPEC_PATH = _ROOT / "machines" / "clara_febe.json"
SPEC = MachineSpec.load(SPEC_PATH)

# How CLARA's lattice names its diagnostics. It models BPMs, screens and aperture
# markers all as `cheetah.Screen`, so the name is the only thing that tells them
# apart — a convention of this lattice, not a fact about Cheetah.
BPM_TAG = "dia_bpm"
SCREEN_TAG = "dia_scr"
APERTURE_TAG = "sim_aper"

# The variable bunch compressor: four dipoles on one supply.
VBC_CHICANE = "cla_vbc_chicane"
_CHICANE = SPEC.knob(VBC_CHICANE)
VBC_DIPOLES = tuple(_CHICANE.elements)
VBC_SIGNS = tuple(_CHICANE.signs)
# The furthest the *supply* reaches, 131.7 mrad — not the 135 mrad design figure
# quoted for the compressor (IPAC THP2100), which the magnets cannot be driven to.
VBC_SUPPLY_LIMIT = float(max(abs(_CHICANE.low), abs(_CHICANE.high)))

# Which face of each dipole turns with its angle. The chicane's outer faces stay
# square to the beamline and the inner ones follow the bend, so it alternates
# exit, entry, exit, entry along the string. Given here in the `"exit"` /
# `"entry"` form the older code expects; the spec holds the same thing as the
# Cheetah attribute names `dipole_e2` / `dipole_e1`.
VBC_EDGE = {
    element: ("exit" if faces[0] == "dipole_e2" else "entry")
    for element, faces in _CHICANE.follow.items()
}
