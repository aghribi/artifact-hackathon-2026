import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from rl_framework.config import RLConfig
from rl_framework.envs.sim_env import SimEnv
from rl_framework.wrappers import SB3Wrapper


class ProgressCallback(BaseCallback):
    """Print rolling reward stats every `report_every` steps."""

    def __init__(self, report_every: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.report_every = report_every
        self._last_report = 0
        self._start_time = time.monotonic()
        self._rewards = []

    def _on_step(self) -> bool:
        steps = self.num_timesteps
        self._rewards.append(self.locals.get("rewards", [float("nan")])[0])
        if steps - self._last_report >= self.report_every:
            self._last_report = steps
            elapsed = time.monotonic() - self._start_time
            recent = self._rewards[-100:]
            if recent:
                print(
                    f"  [{steps:>6d} steps | {elapsed:.1f}s] "
                    f"rew100_mean={np.mean(recent):+.3f}  "
                    f"rew100_std={np.std(recent):.3f}"
                )
            else:
                print(f"  [{steps:>6d} steps | {elapsed:.1f}s]")
        return True


def train(config: RLConfig, total_timesteps: int = 100, model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)

    # Single environment. SB3Wrapper turns SimEnv's dictionary action into the vector
    # of magnets PPO needs and puts the observation in network units, which is also
    # why NormalizeObservation is gone: it takes a single Box, not a dictionary.
    env = SimEnv(config)
    env = SB3Wrapper(env)
    env = gym.wrappers.NormalizeReward(env)
    env = Monitor(env, filename=os.path.join(model_dir, "monitor.csv"))

    # Checkpoint every 5000 steps
    checkpoint_cb = CheckpointCallback(
        save_freq=10,
        save_path=model_dir,
        name_prefix="ppo_accelerator",
    )

    # Custom progress callback
    progress_cb = ProgressCallback(report_every=1)

    model = PPO(
        "MultiInputPolicy",  # the observation is a dictionary: BPMs and a camera
        env,
        verbose=1,
        n_steps=10
    )
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, progress_cb])
    model.save(os.path.join(model_dir, "ppo_accelerator_final"))
    print(f"Training complete. Model saved to {model_dir}/")
    return model


if __name__ == "__main__":
    cfg = RLConfig.clara()
    # Train on the magnets RealEnv drives, over the ranges it denormalises with, so an
    # action means the same thing here and in `deploy.py`. Left alone, SimEnv offers
    # every magnet of the line over its archiver range instead.
    cfg.controllable_magnets = [magnet[0] for magnet in cfg.magnet_config]
    cfg.magnet_limits = {magnet[0]: magnet[1:] for magnet in cfg.magnet_config}
    train(cfg)
