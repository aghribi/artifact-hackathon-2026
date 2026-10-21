"""What `SimEnv` looks like to a stable-baselines3 policy.

`SimEnv` takes a dictionary action — some magnets and a screen to look at — because
that is what an operator does to the machine. No stable-baselines3 algorithm accepts a
`Dict` action space (PPO takes Box, Discrete, MultiDiscrete or MultiBinary), so a
policy needs a fixed vector of magnets instead, and `SB3Wrapper` is that view of it:

    env = SB3Wrapper(SimEnv(config))
    PPO("MultiInputPolicy", env).learn(10_000)

One entry per magnet in `env.magnets`, each in [-1, 1] over that magnet's operating
range, and the screen stays wherever `SimEnv` has it — the policy steers towards one
screen rather than choosing between them.

The observation stays a dictionary, which `MultiInputPolicy` handles, with the pieces
put in the units a network wants: BPM readings as a fraction of the beam pipe radius
rather than metres, the screen image as an 8-bit single-channel picture, which is what
makes stable-baselines3 put a CNN behind it instead of flattening 16,384 floats into
the MLP, and the request as a fraction of the screen's half-width, so it is in the
same units as the image's own axes. `screen_index` is dropped, being constant here.

The requested spot is a small part of that range — the chip is tens of millimetres
and the request hundreds of microns — so if a policy struggles to resolve it, scaling
`target` by something tighter than the chip is the line to change.

Because of those units, `gymnasium.wrappers.NormalizeObservation` is neither needed nor
usable on top of this (it expects a single Box, and a running mean would undo the 8-bit
image). `NormalizeReward` is fine.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

DEFAULT_PIPE_RADIUS = 50e-3  # m, to scale BPM readings by when there is no pipe


class SB3Wrapper(gym.Wrapper):
    """A fixed vector of magnets and a network-shaped observation, around `SimEnv`."""

    def __init__(self, env, magnets: list[str] | None = None, image_as_uint8: bool = True):
        super().__init__(env)
        sim = env.unwrapped
        self.magnets = list(magnets if magnets is not None else sim.magnet_names)
        unknown = [name for name in self.magnets if name not in sim.magnet_limits]
        if unknown:
            raise KeyError(f"not magnets of this environment: {unknown}")

        limits = np.array([sim.magnet_limits[name] for name in self.magnets], float)
        self._middle = limits.mean(axis=1)
        self._half = (limits[:, 1] - limits[:, 0]) / 2
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.magnets),), dtype=np.float32
        )

        self._scale = sim.beam_pipe_radius or DEFAULT_PIPE_RADIUS
        self._image_as_uint8 = image_as_uint8
        # Position and size of the request, each against the half-width of the chip
        # it is asked for on.
        half = np.abs(sim.screen_extent())[[1, 3, 1, 3]]
        self._target_scale = half
        bpms = len(sim.bpm_names)
        image = sim.observation_space["screen_image"].shape
        self.observation_space = spaces.Dict(
            {
                "bpm_x": spaces.Box(-1.0, 1.0, (bpms,), np.float32),
                "bpm_y": spaces.Box(-1.0, 1.0, (bpms,), np.float32),
                "bpm_valid": spaces.MultiBinary(bpms),
                "target": spaces.Box(-1.0, 1.0, (4,), np.float32),
                "screen_image": (
                    spaces.Box(0, 255, (1, *image), np.uint8)
                    if image_as_uint8
                    else spaces.Box(0.0, 1.0, (1, *image), np.float32)
                ),
            }
        )

    def native_settings(self, action) -> dict[str, float]:
        """The magnet settings a normalised action asks for, in native units."""
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        return dict(zip(self.magnets, self._middle + action * self._half))

    def observation(self, observation: dict) -> dict:
        image = np.asarray(observation["screen_image"], dtype=np.float64)
        peak = image.max()
        if peak > 0:
            image = image / peak
        return {
            "bpm_x": (observation["bpm_x"] / self._scale).astype(np.float32),
            "bpm_y": (observation["bpm_y"] / self._scale).astype(np.float32),
            "bpm_valid": observation["bpm_valid"],
            "target": (observation["target"] / self._target_scale).astype(np.float32),
            "screen_image": (
                (image * 255).astype(np.uint8)
                if self._image_as_uint8
                else image.astype(np.float32)
            )[np.newaxis],
        }

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            {"magnets": self.native_settings(action)}
        )
        return self.observation(observation), reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self.observation(observation), info
