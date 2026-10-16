import numpy as np
from stable_baselines3 import PPO

from rl_framework.config import RLConfig
from rl_framework.envs.real_env import RealEnv


def deploy(
    config: RLConfig,
    model_path: str,
    episodes: int = 5,
    virtual: bool = False,
):
    env = RealEnv(config, virtual=virtual)
    model = PPO.load(model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        step = 0
        ep_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1
            print(f"  Step {step}: reward={reward:.4f}, action={np.round(action, 4)}")

        print(f"Episode {ep + 1}: total_reward={ep_reward:.4f}, steps={step}")

    env.close()


if __name__ == "__main__":
    cfg = RLConfig()
    deploy(cfg, model_path="models/ppo_accelerator_final.zip", virtual=True)
