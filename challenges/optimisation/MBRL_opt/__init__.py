"""The model-based stage of the CLARA tuning pipeline, against `RL_framework`'s SimEnv.

Reference optics and SVD control rankings come from `MFopt`/`optics_matching`, the
screen measure from `image_processing`; what is here is a differentiable Cheetah twin
of the machine, a fit of the episode's beam onto it, and the planner that tweaks the
reference tune until the camera agrees with the request. See `README.md`.

    from MBRL_opt.agent import MBRLAgent
    from MBRL_opt.config import MBRLConfig
"""

from agent import EpisodeResult, MBRLAgent
from config import MBRLConfig
from knobs import KnobSet
from screen import ScreenReading, measure
from twin import CheetahTwin, LatentState, Readings, tune_threads

__all__ = [
    "CheetahTwin",
    "EpisodeResult",
    "KnobSet",
    "LatentState",
    "MBRLAgent",
    "MBRLConfig",
    "Readings",
    "ScreenReading",
    "measure",
    "tune_threads",
]
