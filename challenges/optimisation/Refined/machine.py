"""The shared machine description, which lives in `machine_spec/`.

It moved there because `RL_framework` needs it too and may not import
`Refined`. This shim keeps `from machine import MachineSpec` working for the
modules in this directory; new code outside it should import `machine_spec`.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from machine_spec import *  # noqa: F401,F403,E402
