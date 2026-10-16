from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import skimage

from rl_framework.config import RLConfig


class BaseAcceleratorEnv(gym.Env, ABC):
    """
    Abstract base environment for accelerator RL.
    Subclasses implement _apply_action() and _get_screen_statistics().
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: RLConfig):
        self.config = config

        # Observation space: 4 stats per observation screen
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Per-magnet denormalisation buffers
        self._magnet_names = [m[0] for m in config.magnet_config]
        self._magnet_low = np.array([m[1] for m in config.magnet_config], dtype=np.float32)
        self._magnet_high = np.array([m[2] for m in config.magnet_config], dtype=np.float32)

        self.action_space = spaces.Box(
            low=self._magnet_low , high=self._magnet_high, dtype=np.float32
        )
        # Internal state
        self._step_count = 0

    def _denormalise_action(self, action: np.ndarray) -> np.ndarray:
        """Map normalised [-1, 1] actions to native [low, high]."""
        mid = (self._magnet_low + self._magnet_high) / 2.0
        half = (self._magnet_high - self._magnet_low) / 2.0
        return mid + action * half

    def _compute_reward(self, mu_x: float, mu_y: float, sigma_x: float, sigma_y: float, on_screen: bool) -> float:
        """Negative sum of beam sizes at the target screen."""
        if not on_screen:
            return float(-1000)  # Penalise if beam is not on screen
        return float(-(sigma_x + sigma_y) + 0.1 * (abs(mu_x) + abs(mu_y)))  # Also penalise large centroids to encourage beam centering

    def _get_obs(self) -> np.ndarray:
        """Collect statistics from the target screen."""
        mu_x, mu_y, sigma_x, sigma_y, on_screen = self._get_screen_statistics(self.config.target_screen)
        return np.array([mu_x, mu_y, sigma_x, sigma_y, on_screen], dtype=np.float32)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        native = self._denormalise_action(action)
        self._apply_action(native)

        obs = self._get_obs()
        # Compute reward at target screen
        mu_x, mu_y, sigma_x, sigma_y, on_screen = self._get_screen_statistics(self.config.target_screen)
        reward = self._compute_reward(mu_x, mu_y, sigma_x, sigma_y, on_screen)

        self._step_count += 1
        truncated = self._step_count >= self.config.max_episode_steps
        terminated = False  # no early termination (can be added later)

        return obs, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return self._hard_reset(), {}

    # ---------- Subclass API ----------

    @abstractmethod
    def _hard_reset(self) -> np.ndarray:
        """Full reset: return initial observation."""

    @abstractmethod
    def _apply_action(self, native_values: np.ndarray):
        """Set magnets to given native-unit values."""

    @abstractmethod
    def _get_screen_statistics(self, screen_name: str) -> tuple[float, float, float, float, bool]:
        """Return (mu_x, mu_y, sigma_x, sigma_y, on_screen) for a screen."""


    def _read_screen_stats(self, image: np.ndarray) -> tuple[float, float, float, float, bool]:
        """
        Extract centroid and sigma from a screen image
        Returns (mu_x, mu_y, sigma_x, sigma_y, on_screen).
        """
        threshold = skimage.filters.threshold_otsu(image)
        # set values below the threshold intensity to zero
        image[image < threshold] = 0.0
        # calculate second-order image moments
        moments = skimage.measure.moments(image, order=2)
        if((moments == 0).all()):
            return 0.0, 0.0, 0.0, 0.0, False
        # calculate first-order image moments
        centroid_row = moments[1, 0] / moments[0, 0]
        centroid_col = moments[0, 1] / moments[0, 0]
        
        # calculate the second-order image moments
        row_variance = (moments[2, 0] / moments[0, 0]) - centroid_row**2
        covariance = (moments[1, 1] / moments[0, 0]) - (centroid_row * centroid_col)
        col_variance = (moments[0, 2] / moments[0, 0]) - centroid_col**2
        
        mu_x, mu_y = centroid_row, centroid_col    
        sigma_x, sigma_y = np.sqrt(row_variance), np.sqrt(col_variance)

        return mu_x, mu_y, sigma_x, sigma_y, True
