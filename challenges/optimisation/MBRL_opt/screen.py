"""Reading a spot off a camera frame, the way the real screens are read.

`SimEnv` hands back both a picture and, in `info`, the moments of the particles that
made it. The moments are the easy path and they are not what a machine gives you: on
CLARA there is a camera, and somebody has to turn its frame into four numbers. This
module does that with `image_processing.image_center_and_deviation` — the measure
written for the real CLARA screen captures — so the agent is driven by the same
quantity in simulation as it would be on the machine.

Two corrections sit on top of it. The pixel grid is indexed from its corner, so the
coordinates it builds are half a pixel low in each axis; on a 128-wide grid across a
22 mm chip that is 87 µm of bias, which is as large as the smallest spot the benchmark
ever asks for. And an intensity-weighted rms over a whole frame counts every pixel of
background as beam, so a faint, wide pedestal reads as a wide spot. `BACKGROUND_FLOOR`
drops the pixels that are plausibly background before the moments are taken.

    stats = measure(observation["screen_image"], env.screen_extent())
    stats.ok           # enough signal to believe it
    stats.moments      # (mu_x, mu_y, sigma_x, sigma_y) in metres off the chip centre
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

CHALLENGE_DIR = Path(__file__).resolve().parents[1]
if str(CHALLENGE_DIR / "image_processing") not in sys.path:
    sys.path.insert(0, str(CHALLENGE_DIR / "image_processing"))

from image_processing import image_center_and_deviation  # noqa: E402

# Pixels dimmer than this fraction of the brightest are treated as background. The
# rms of an image is dominated by its tails, so a floor matters more than it looks:
# without one, the dark half of the chip drags the measured spot wider than it is.
BACKGROUND_FLOOR = 0.05
# Least of the frame's total intensity that has to survive the floor for the reading
# to mean anything. Below it there is no spot, only noise.
MIN_SIGNAL = 1e-9


@dataclass(frozen=True)
class ScreenReading:
    """What the camera says the beam is doing, in metres off the chip centre."""

    mu_x: float
    mu_y: float
    sigma_x: float
    sigma_y: float
    ok: bool

    @property
    def moments(self) -> np.ndarray:
        return np.array([self.mu_x, self.mu_y, self.sigma_x, self.sigma_y])

    @classmethod
    def blank(cls) -> "ScreenReading":
        return cls(0.0, 0.0, 0.0, 0.0, False)


def measure(
    image: np.ndarray,
    extent: tuple[float, float, float, float],
    floor: float = BACKGROUND_FLOOR,
) -> ScreenReading:
    """The spot on a camera frame: centroid and rms, in metres off the chip centre.

    `image` is rows by columns with y increasing up the rows and x along the columns,
    which is how `SimEnv` lays its screen images out, and `extent` is the
    (x_min, x_max, y_min, y_max) of the chip it covers.
    """
    frame = np.asarray(image, dtype=np.float64)
    if frame.ndim != 2 or not frame.size:
        return ScreenReading.blank()

    peak = frame.max()
    if not np.isfinite(peak) or peak <= 0:
        return ScreenReading.blank()
    frame = np.where(frame >= floor * peak, frame, 0.0)
    if frame.sum() <= MIN_SIGNAL:
        return ScreenReading.blank()

    x_min, x_max, y_min, y_max = (float(v) for v in extent)
    width, height = x_max - x_min, y_max - y_min
    (cx, cy), (sx, sy) = image_center_and_deviation(
        torch.from_numpy(frame), width, height
    )
    if not all(np.isfinite(v) for v in (cx, cy, sx, sy)):
        return ScreenReading.blank()

    # The measure lays its coordinates on pixel corners, so both centroids come back
    # half a pixel low. The rms is unaffected: shifting every coordinate by the same
    # amount does not change the spread.
    rows, columns = frame.shape
    cx += 0.5 * width / columns
    cy += 0.5 * height / rows
    return ScreenReading(float(cx), float(cy), float(sx), float(sy), True)


def pixel_size(
    shape: tuple[int, int], extent: tuple[float, float, float, float]
) -> tuple[float, float]:
    """(x, y) metres per pixel, which is the floor on what a frame can resolve."""
    x_min, x_max, y_min, y_max = (float(v) for v in extent)
    return (x_max - x_min) / shape[1], (y_max - y_min) / shape[0]
