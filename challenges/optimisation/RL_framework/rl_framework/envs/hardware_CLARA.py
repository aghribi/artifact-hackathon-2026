"""`hardware.py` for CLARA, through CATAP. The real implementation.

    from rl_framework.envs import hardware_CLARA
    env = RealEnv(config, hardware=hardware_CLARA)

`USE_VIRTUAL` picks the virtual accelerator over the real one, and is the only
switch that matters here: with it `True` nothing in this file can move a magnet
on CLARA. Leave it `True` until a run is intended.

**Units in, units out.** Everything crossing this boundary is in the units the
lattice file uses — quadrupole `k1` in 1/m², corrector and dipole `angle` in
rad, sextupole `k2` in 1/m³, positions and extents in metres. CATAP works in
amps and millimetres, and the conversions live here so that nothing above this
line knows a power supply exists. They are the same calibrations the Cheetah
model is built from (`data/clara/lattice_data/utils.py`), which matters more
than it sounds: a tuner plans against the model and applies through here, so a
disagreement between the two shows up as model error that no amount of tuning
removes.

**Names.** The environment works in lattice names, lower case with underscores,
and *with the plane on the end for a corrector* — `cla_s02_mag_hvcor_01_h`,
because the model's one element is two magnets on the machine. `_catap_name`
turns that into CATAP's `CLA-S02-MAG-HCOR-01`.

**pyCATAP is incomplete and buggy.** So: every wait has a timeout rather than a
bare `while`, every readback that can come back empty is checked before it is
believed, and the frame shape is taken from the camera rather than assumed. The
places where that caution is load-bearing are commented individually.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# Use the virtual accelerator. **This is the safety switch.** It is read when a
# factory is first built, not at import, so flipping it in a session before
# touching anything still takes effect.
USE_VIRTUAL = True

if USE_VIRTUAL:
    os.environ["EPICS_CA_ADDR_LIST"] = "172.18.0.6"
    os.environ["EPICS_CA_SERVER_PORT"] = "6090"
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    os.environ["EPICS_CA_MAX_ARRAY_BYTES"] = "1000000"

# The lattice's own magnet calibrations, so the machine and the model agree.
_LATTICE_DATA = Path(__file__).resolve().parents[3] / "data" / "clara" / "lattice_data"
if str(_LATTICE_DATA) not in sys.path:
    sys.path.insert(0, str(_LATTICE_DATA))
from utils import (  # noqa: E402
    corrector_angle_from_current,
    corrector_current_from_angle,
    dipole_angle_from_current,
    dipole_current_from_angle,
    quadrupole_current_from_k,
    quadrupole_k_from_current,
    sextupole_current_from_k2,
    sextupole_k2_from_current,
)

# How long a screen takes to move, in seconds. Only one screen may be in the
# beam, and swapping is mechanical: this is what makes a switch expensive and
# why `Refined` counts switches separately from shots.
SCREEN_SWITCH_SECONDS = 30.0

# Beam momentum, MeV/c. Every current conversion needs it, and getting it wrong
# scales every magnet by the same factor — which looks like a perfectly healthy
# machine that is simply mistuned, so it is worth being sure of.
#
# TODO: confirm this and, better, read it. 250 MeV/c is CLARA FEBE's design
# value and what `RLConfig.beam["energy"]` carries, but the machine is not
# always at its design energy and nothing here would notice.
BEAM_MOMENTUM_MEVC = 250.0

# How long to wait for mechanisms before giving up, in seconds. A bare
# `while thing.moving:` is an infinite loop the first time a status PV sticks,
# and these PVs do stick.
MAGNET_SETTLE_TIMEOUT = 30.0
SCREEN_SETTLE_TIMEOUT = 60.0
POLL_SECONDS = 0.5
# How close a readback has to be to its demand to count as arrived, in amps.
CURRENT_TOLERANCE_A = 0.05


# ---------------------------------------------------------------- plumbing

_factories: dict[str, object] = {}


def _factory(kind: str):
    """A CATAP factory, built on first use.

    Lazily, because building one connects to the control system, and importing
    a module should not do that — `rl_framework` imports `real_env`, so an
    eager factory here would mean that merely importing the package talks to an
    accelerator. It also means `USE_VIRTUAL` can be set by a caller before the
    first call and still be honoured.
    """
    if kind not in _factories:
        if kind == "magnet":
            from CATAP.magnet import MagnetFactory

            _factories[kind] = MagnetFactory(is_virtual=USE_VIRTUAL)
        elif kind == "screen":
            from CATAP.diagnostics.screen import ScreenFactory

            _factories[kind] = ScreenFactory(is_virtual=USE_VIRTUAL)
        elif kind == "camera":
            from CATAP.diagnostics.camera import CameraFactory

            _factories[kind] = CameraFactory(is_virtual=USE_VIRTUAL)
        else:
            raise KeyError(kind)
    return _factories[kind]


def _catap_name(lattice_name: str) -> str:
    """`cla_s02_mag_hvcor_01_h` -> `CLA-S02-MAG-HCOR-01`.

    Two things happen. The style changes, lower-underscore to upper-hyphen. And
    a corrector's plane, which arrives as a suffix because the model has one
    element where the machine has two magnets, is folded into the name: the
    combined `HVCOR` becomes `HCOR` or `VCOR` and the suffix comes off. An
    element that is already single-plane (`cla_vbc_mag_vcor_01_v`) keeps the
    name it has, rather than becoming `VVCOR`.
    """
    name, plane = lattice_name, ""
    if name.endswith(("_h", "_v")):
        name, plane = name[:-2], name[-1]
    if plane and "hvcor" in name:
        name = name.replace("hvcor", "hcor" if plane == "h" else "vcor")
    return name.upper().replace("_", "-")


def _kind(lattice_name: str) -> str:
    """Which calibration a magnet takes, from the lattice name.

    Read off the name we were given rather than CATAP's `type`/`subtype`,
    because this has to agree with `data/clara/lattice_data/utils.py`, whose
    tables are keyed by name. Asking CATAP what kind of magnet it is and then
    looking the answer up in a table keyed by something else is how the two
    drift apart.
    """
    for tag, kind in (("cor", "corrector"), ("quad", "quadrupole"),
                      ("sext", "sextupole"), ("dip", "dipole"),
                      ("cav", "cavity")):
        if tag in lattice_name:
            return kind
    raise ValueError(
        f"cannot tell what kind of magnet {lattice_name!r} is. The name is "
        f"expected to contain one of quad, cor, dip, sext or cav."
    )


def _magnet(lattice_name: str):
    """The CATAP magnet for a lattice name, or a useful error."""
    catap = _catap_name(lattice_name)
    magnet = _factory("magnet").get_magnet(catap)
    if magnet is None:
        # `get_magnet` returns None for a name it does not know rather than
        # raising, so an unmapped name would otherwise surface as an
        # AttributeError on None several lines later.
        raise KeyError(
            f"CATAP has no magnet {catap!r} (from {lattice_name!r}). Either the "
            f"name mapping in `_catap_name` is wrong for this one, or it is not "
            f"on this machine."
        )
    return magnet


def _wait_until(predicate, timeout: float, what: str) -> None:
    """Poll until `predicate()` or the timeout, then say which it was."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except Exception as error:  # a status PV that is not connected yet
            print(f"  waiting for {what}: {type(error).__name__}: {error}")
        time.sleep(POLL_SECONDS)
    # Not an exception: a timeout here means a mechanism is slow or a status PV
    # is stuck, and the shot that follows is still worth taking — but it has to
    # be said out loud, because the reading may be of a machine mid-move.
    print(f"  WARNING: gave up waiting for {what} after {timeout:.0f} s; "
          f"the next reading may not be of a settled machine.")


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
    kind = _kind(magnet_name)
    if kind == "cavity":
        # The tune carries three cavity phases and they are not magnets. Loudly
        # unimplemented rather than quietly skipped: a tuner that thinks it has
        # set a phase and has not is planning against a machine it is not on.
        raise NotImplementedError(
            f"{magnet_name} is a cavity phase, not a magnet, and setting RF is "
            f"not wired up. Either implement it here or keep cavities out of "
            f"`config.controllable_magnets`."
        )

    magnet = _magnet(magnet_name)
    convert = {
        "dipole": dipole_current_from_angle,
        "quadrupole": quadrupole_current_from_k,
        "corrector": corrector_current_from_angle,
        "sextupole": sextupole_current_from_k2,
    }[kind]
    # The calibration tables are keyed by CATAP's name, and normalise case and
    # separators themselves — so the plane-resolved name is what they want.
    current = convert(_catap_name(magnet_name), float(value), BEAM_MOMENTUM_MEVC)
    if not np.isfinite(current):
        raise ValueError(
            f"{magnet_name}: {value} gives a current of {current}. The "
            f"conversion has gone out of range rather than the magnet."
        )
    magnet.seti = float(current)


def GET_REAL_MAGNET_POWER(magnet_name: str) -> float:
    """What a magnet is actually at, in **native lattice units**.

    A readback, not a set point, if the two can differ. `RealEnv` caches this
    for the duration of a shot, so it is called at most once per magnet per
    acquisition even though the optimisers ask for all of them every shot.
    """
    kind = _kind(magnet_name)
    if kind == "cavity":
        raise NotImplementedError(
            f"{magnet_name} is a cavity phase, not a magnet; reading RF is not "
            f"wired up. See SET_REAL_MAGNET_POWER."
        )

    magnet = _magnet(magnet_name)
    # The virtual accelerator does not drive the readback PV, so `readi` sits at
    # zero there however the magnet is set — which would read as an untuned
    # machine on every shot.
    current = magnet.seti if USE_VIRTUAL else magnet.readi
    if current is None or not np.isfinite(float(current)):
        raise ValueError(
            f"{magnet_name}: CATAP gave a current of {current!r}. Not treated "
            f"as zero, because zero is a plausible-looking setting and a tuner "
            f"would build a response matrix out of it."
        )

    convert = {
        "dipole": dipole_angle_from_current,
        "quadrupole": quadrupole_k_from_current,
        "corrector": corrector_angle_from_current,
        "sextupole": sextupole_k2_from_current,
    }[kind]
    return float(convert(_catap_name(magnet_name), float(current),
                         BEAM_MOMENTUM_MEVC))


def WAIT_FOR_MAGNETS_SETTLED(magnet_names: list[str]) -> None:
    """Block until these magnets have reached their new set points.

    Returning early is the expensive failure mode here: the shot that follows
    reads a machine mid-ramp and the optimiser books it as the response to a
    setting that was never applied.
    """
    if USE_VIRTUAL:
        return  # the virtual accelerator applies a demand immediately

    magnets = {name: _magnet(name) for name in magnet_names
               if _kind(name) != "cavity"}

    def arrived() -> bool:
        for magnet in magnets.values():
            # `is_current_steady` is the direct answer, when the PV exists.
            # Where it does not, fall back to comparing the readback with the
            # demand, which is the thing it is supposed to be telling us.
            steady = getattr(magnet, "is_current_steady", None)
            if steady is not None:
                if not steady:
                    return False
                continue
            readback, demand = magnet.readi, magnet.seti
            if readback is None or demand is None:
                return False
            if abs(float(readback) - float(demand)) > CURRENT_TOLERANCE_A:
                return False
        return True

    _wait_until(arrived, MAGNET_SETTLE_TIMEOUT,
                f"{len(magnets)} magnet(s) to settle")


# ---------------------------------------------------------------- screens


def _screen_and_camera(screen_name: str):
    """The CATAP screen and its camera, for a lattice screen name."""
    catap = _catap_name(screen_name)
    screen = _factory("screen").get_screen(catap)
    if screen is None:
        raise KeyError(f"CATAP has no screen {catap!r} (from {screen_name!r})")
    # Ask the screen which camera is on it rather than rewriting the name.
    # `SCR`->`CAM` is right for most of CLARA and wrong for enough of it to be
    # worth not guessing.
    camera_name = getattr(screen, "camera_name", None) or catap.replace("SCR", "CAM")
    camera = _factory("camera").get_camera(camera_name)
    if camera is None:
        raise KeyError(
            f"CATAP has no camera {camera_name!r} for screen {catap!r}"
        )
    return screen, camera


def INSERT_REAL_SCREEN(screen_name: str) -> None:
    """Put this screen into the beam."""
    screen, camera = _screen_and_camera(screen_name)
    screen.insert_yag()
    if not camera.is_acquiring:
        camera.start_acquiring()


def RETRACT_REAL_SCREEN(screen_name: str) -> None:
    """Take this screen out of the beam.

    `RealEnv` retracts the screen it was looking at before inserting another,
    because only one may be in at a time and an intercepting screen blinds
    everything downstream of it.
    """
    screen, camera = _screen_and_camera(screen_name)
    if camera.is_acquiring:
        camera.stop_acquiring()
    screen.remove_yag()


def WAIT_FOR_SCREEN_SETTLED(screen_name: str) -> None:
    """Block until the screen has finished moving."""
    screen, _ = _screen_and_camera(screen_name)
    # A moment first: `moving` reads False for a beat after the command, before
    # the mechanism starts, and polling it immediately returns "settled" on a
    # screen that has not begun to move.
    time.sleep(1.0)
    _wait_until(lambda: not screen.moving, SCREEN_SETTLE_TIMEOUT,
                f"screen {screen_name} to stop moving")


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
    _, camera = _screen_and_camera(screen_name)
    if not camera.is_acquiring:
        camera.start_acquiring()
        time.sleep(1.0)  # let a frame arrive before asking for one

    frame = np.asarray(camera.get_array_data(), dtype=float)
    if frame.size == 0:
        raise ValueError(
            f"{screen_name}: the camera returned an empty frame. Not passed on "
            f"as a blank image, because that measures as 'no beam' and a tuner "
            f"will spend its budget steering away from a camera fault."
        )

    if frame.ndim == 1:
        # `get_array_data` hands back the raw waveform; the shape is the
        # camera's business and it knows it.
        height, width = int(camera.image_height), int(camera.image_width)
        if height * width != frame.size:
            raise ValueError(
                f"{screen_name}: {frame.size} pixels does not fit the camera's "
                f"{width} x {height}. Reshaping on a guess would transpose the "
                f"beam rather than fail."
            )
        frame = frame.reshape(height, width)
    elif frame.ndim != 2:
        raise ValueError(f"{screen_name}: expected a 2-D frame, got {frame.shape}")

    # Cameras index from the top-left, the convention here is bottom-left. Only
    # the row order differs; flipping it is the difference between a beam
    # measured above the axis and below it.
    return np.flipud(frame)


def GET_REAL_SCREEN_EXTENT(screen_name: str) -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) in **metres** that this camera's frame covers.

    The calibration: pixel count times metres per pixel, about the chip centre.
    Asked for rather than taken from the lattice model on purpose — a model's
    idea of a field of view is not a measurement, and every spot size the
    optimiser is scored on is this number times a pixel count.

    Taken from the camera's own `pixel_to_mm` where it has one, and from the
    measured table in `image_processing.camera_conversions` where it does not.
    The two are cross-checked when both exist: they are the same physical
    quantity from different sources, so a disagreement means one of them is
    wrong and it matters which.

    Note this asks the camera for its dimensions rather than grabbing a frame —
    `RealEnv` calls this for every screen of the line when it starts, and
    grabbing frames would mean starting acquisition on all twenty-two.
    """
    _, camera = _screen_and_camera(screen_name)
    height, width = int(camera.image_height), int(camera.image_width)
    if not height or not width:
        raise ValueError(
            f"{screen_name}: the camera reports a {width} x {height} frame."
        )

    measured = _conversion_table().get(screen_name)
    from_camera = getattr(camera, "pixel_to_mm", None)
    if from_camera:
        metres_per_pixel = float(from_camera) / 1000.0
        if measured and abs(metres_per_pixel - measured[0]) > 0.1 * measured[0]:
            print(f"  WARNING: {screen_name} pixel size is "
                  f"{metres_per_pixel * 1e6:.2f} um from the camera and "
                  f"{measured[0] * 1e6:.2f} um from the measured table — "
                  f"a 10 % disagreement scales every spot size on this screen.")
        x_per_pixel = y_per_pixel = metres_per_pixel
    elif measured:
        x_per_pixel, y_per_pixel = measured
    else:
        raise KeyError(
            f"no pixel calibration for {screen_name}: the camera has no "
            f"`pixel_to_mm` and it is not in `camera_conversions`. Add it "
            f"rather than falling back on the model, which would make every "
            f"spot size on this screen a guess."
        )

    half_x, half_y = width * x_per_pixel / 2, height * y_per_pixel / 2
    return (-half_x, half_x, -half_y, half_y)


def _conversion_table() -> dict[str, list[float]]:
    """The measured metres-per-pixel table, keyed by lattice screen name."""
    root = Path(__file__).resolve().parents[3] / "image_processing"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from image_processing import camera_conversions

    return camera_conversions


# ---------------------------------------------------------------- BPMs


def GET_REAL_BPM_READING(bpm_name: str) -> tuple[float, float, bool]:
    """(x, y, valid) for one BPM: metres off its electrical centre, and whether
    there was enough charge for the reading to mean anything.

    CLARA's BPMs are not working at the time of writing, which is what
    `machines/clara_febe_no_bpms.json` is for: with `bpms: []` in the spec this
    is never called and the optimiser steers on screens instead. Populate it when
    they come back and the BPM path lights up again with no other change.

    Until then this reports *invalid* rather than zero. A BPM claiming a
    confident 0.0 is a BPM saying the beam is perfectly on axis, and `steer`
    would correct a real orbit towards that fiction; `valid=False` makes it skip
    the BPM entirely, which is the truth.
    """
    return 0.0, 0.0, False


# ---------------------------------------------------------------- charge


def GET_REAL_TRANSMISSION() -> float:
    """The fraction of the bunch that reaches the readout, in [0, 1].

    The benchmark charges two spot sizes for a bunch half lost, so this is not a
    decoration: a tune that delivers a beautiful spot with a third of the charge
    scores worse than a mediocre spot with all of it.

    TODO: the intended measure is the ratio of `CLA-ICT-FEH-01:Q` at the end to
    `CLA-SP1-DIA-FCUP-01:Q` at the start. It is not wired up, so this returns 1.0
    and the score loses its transmission term — which means a tune that throws
    charge away is scored as though it did not. Everything else works without
    it; this is the one number still missing.
    """
    return 1.0
