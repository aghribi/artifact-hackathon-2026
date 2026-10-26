"""Every point where `RealEnv` touches CLARA. Populate these and it runs.

Nothing here is implemented. Each function raises `NotImplementedError` naming
what it has to return and in what units, and `RealEnv` is built so that these
are the *only* places a real value is written or read — so this file is the
whole of the control-system surface, and the rest of the environment can be
read, tested and trusted without a control network anywhere near it.

**Units.** Everything is in the units the rest of the challenge uses, which are
the lattice file's: magnet strengths are native (quadrupole `k1` in 1/m²,
corrector `angle` in rad, dipole `angle` in rad), and positions and chip extents
are **metres**. Converting native strength to a power-supply current is the job
of `SET_REAL_MAGNET_POWER`, inside this file, because the calibration is a fact
about the magnet rather than about the environment. CLARA has a service for it:
`POST http://dsastvx10.dl.ac.uk:8001/magnet/convert` with
`{"current": ..., "magnet_name": ..., "momentum": ...}`, which an earlier
version of `real_env.py` called.

**Names** are the lattice's, lower case with underscores —
`cla_s04_mag_quad_01`, `cla_feh_dia_scr_01`. The CATAP name is the same thing
upper case with hyphens, and correctors split: `cla_s04_mag_hvcor_01_h` is
CATAP's `CLA-S04-MAG-HCOR-01`. That mapping belongs in here too.

**Safety.** No function in this module is called at import time or at
construction. `RealEnv.reset()` is read-only and never writes a magnet, so the
first write happens only when an optimiser asks for one.

To wire it up, replace the bodies. To test without hardware, hand `RealEnv` a
substitute module via `RealEnv(config, hardware=my_fake)`.
"""

from __future__ import annotations

import numpy as np

# How long a screen takes to move, in seconds. Only one screen may be in the
# beam, and swapping is mechanical: this is what makes a switch expensive and
# why `Refined` counts switches separately from shots.
SCREEN_SWITCH_SECONDS = 30.0


def _todo(name: str, returns: str) -> NotImplementedError:
    return NotImplementedError(
        f"{name} is a placeholder and has not been populated yet. It must "
        f"{returns}. See RL_framework/rl_framework/envs/hardware.py."
    )


# ---------------------------------------------------------------- magnets


def SET_REAL_MAGNET_POWER(magnet_name: str, value: float) -> None:
    """Set one magnet to `value` in **native lattice units**.

    `value` is a quadrupole's `k1` in 1/m², a corrector's `angle` in rad, or a
    dipole's `angle` in rad — whatever property the spec declares for that knob.
    Converting to a power-supply current happens in here.

    Called once per element, so a coupled group (the four VBC chicane dipoles on
    one supply) arrives as four calls with the signs already applied. Nothing
    above this knows about currents, interlocks or ramp rates.
    """
    raise _todo("SET_REAL_MAGNET_POWER", "set the magnet and return None")


def GET_REAL_MAGNET_POWER(magnet_name: str) -> float:
    """What a magnet is actually at, in **native lattice units**.

    A readback, not a set point, if the two can differ. `RealEnv` caches this
    for the duration of a shot, so it is called at most once per magnet per
    acquisition even though the optimisers ask for all of them every shot.
    """
    raise _todo("GET_REAL_MAGNET_POWER", "return the strength as a float")


def WAIT_FOR_MAGNETS_SETTLED(magnet_names: list[str]) -> None:
    """Block until these magnets have reached their new set points.

    Returning early is the expensive failure mode here: the shot that follows
    reads a machine mid-ramp and the optimiser books it as the response to a
    setting that was never applied.
    """
    raise _todo("WAIT_FOR_MAGNETS_SETTLED", "block until the magnets are settled")


# ---------------------------------------------------------------- screens


def INSERT_REAL_SCREEN(screen_name: str) -> None:
    """Put this screen into the beam."""
    raise _todo("INSERT_REAL_SCREEN", "insert the screen and return None")


def RETRACT_REAL_SCREEN(screen_name: str) -> None:
    """Take this screen out of the beam.

    `RealEnv` retracts the screen it was looking at before inserting another,
    because only one may be in at a time and an intercepting screen blinds
    everything downstream of it.
    """
    raise _todo("RETRACT_REAL_SCREEN", "retract the screen and return None")


def WAIT_FOR_SCREEN_SETTLED(screen_name: str) -> None:
    """Block until the screen has finished moving."""
    raise _todo("WAIT_FOR_SCREEN_SETTLED", "block until the screen has stopped")


def GET_REAL_SCREEN_IMAGE(screen_name: str) -> np.ndarray:
    """One camera frame, as a 2-D array of intensities.

    **Orientation matters.** Rows are y increasing *upward* and columns are x
    increasing right, with the origin at the bottom left — which is how `SimEnv`
    lays its images out and what the moment measure assumes. A frame handed over
    flipped or transposed gives a correct-looking spot size at a mirrored
    position, which reads as a steering error that no corrector can fix.

    Background subtraction is welcome but not required: the measure applies its
    own floor at 5 % of the brightest pixel.
    """
    raise _todo("GET_REAL_SCREEN_IMAGE", "return a 2-D intensity array")


def GET_REAL_SCREEN_EXTENT(screen_name: str) -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) in **metres** that this camera's frame covers.

    The calibration, in other words: pixels times metres-per-pixel, about the
    chip centre. This is asked for rather than taken from the lattice model on
    purpose — a model's idea of a field of view is not a measurement, and every
    spot size the optimiser is scored on is this number times a pixel count.

    Most of this is already measured and in the repo:
    `image_processing/image_processing.py` has a `camera_conversions` table of
    metres per pixel for the CLARA screens, keyed by exactly these names. With
    the frame's shape that gives the extent directly::

        m_per_px_x, m_per_px_y = camera_conversions[screen_name]
        rows, columns = GET_REAL_SCREEN_IMAGE(screen_name).shape
        half_x, half_y = columns * m_per_px_x / 2, rows * m_per_px_y / 2
        return (-half_x, half_x, -half_y, half_y)

    That assumes the chip centre is the optical axis. Where a camera is known to
    be off-centre, that offset belongs here too — it is indistinguishable from a
    steering error to everything downstream.
    """
    raise _todo(
        "GET_REAL_SCREEN_EXTENT",
        "return (x_min, x_max, y_min, y_max) in metres about the chip centre",
    )


# ---------------------------------------------------------------- BPMs


def GET_REAL_BPM_READING(bpm_name: str) -> tuple[float, float, bool]:
    """(x, y, valid) for one BPM: metres off its electrical centre, and whether
    there was enough charge for the reading to mean anything.

    CLARA's BPMs are not working at the time of writing, which is what
    `machines/clara_febe_no_bpms.json` is for: with `bpms: []` in the spec this
    is never called and the optimiser steers on screens instead. Populate it when
    they come back and the BPM path lights up again with no other change.
    """
    raise _todo("GET_REAL_BPM_READING", "return (x, y, valid)")


# ---------------------------------------------------------------- charge


def GET_REAL_TRANSMISSION() -> float:
    """The fraction of the bunch that reaches the readout, in [0, 1].

    The benchmark charges two spot sizes for a bunch half lost, so this is not a
    decoration: a tune that delivers a beautiful spot with a third of the charge
    scores worse than a mediocre spot with all of it. However you measure it —
    wall-current monitors, BPM sums, a charge monitor either end — it has to be
    the ratio of what arrives to what was launched, not an absolute charge.
    """
    raise _todo("GET_REAL_TRANSMISSION", "return a fraction in [0, 1]")
