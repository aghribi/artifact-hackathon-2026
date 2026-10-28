"""What a lattice file does not say, and somebody has to.

A machine description, shared by everything that drives a machine: the
environment that simulates one (`rl_framework`) and the pipeline that tunes one
(`Refined`). It lives here rather than in either of them because both need it and
neither may depend on the other.

    from machine_spec import MachineSpec, Knob
    spec = MachineSpec.load("machines/clara_febe.json")

`spec.py` is the description itself. `lattice_io.py` builds a Cheetah segment
from one. `sources/` holds the machine-specific readers that *author* a spec —
CLARA's magnet calibration, say — which belong with the authoring and not with
the running.
"""

from .spec import (  # noqa: F401
    FAILURE_SCORE,
    HEALTHY,
    KICK,
    NONLINEAR_PROPERTIES,
    PLANE_SUFFIX,
    SCREEN_SWITCH_SECONDS,
    SETTABLE,
    STEERING_PROPERTIES,
    TRANSMISSION_WEIGHT,
    Knob,
    MachineSpec,
    ScanAxis,
    default_knobs,
    plane_name,
)
