"""The shared machine description, which lives in `machine_spec/`.

It sits in its own package because `rl_framework` needs it too and must not
import the tuning modules. Both are now siblings inside `Refined/`, so this is
a plain re-export: it keeps `from machine import MachineSpec` working for the
modules in this directory, and new code should import `machine_spec` directly.
"""

from machine_spec import *  # noqa: F401,F403
