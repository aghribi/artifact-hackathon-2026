"""Re-export of the shared machine description, which now lives in `machine_spec/`.

It moved because `RL_framework` needs it too and may not import `Refined`. This
shim keeps `from machine import MachineSpec` working for everything in this
directory, and for the spec files that name it.
"""

import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parents[1] / "machine_spec"
if str(_SHARED.parent) not in sys.path:
    sys.path.insert(0, str(_SHARED.parent))

from machine_spec.spec import *  # noqa: F401,F403
