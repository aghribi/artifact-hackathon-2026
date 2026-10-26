import numpy as np
from stable_baselines3 import PPO

from rl_framework.config import RLConfig
from rl_framework.envs.real_env import RealEnv
from rl_framework.wrappers import SB3Wrapper


def deploy(
    config: RLConfig,
    model_path: str,
    episodes: int = 5,
):
    # The same wrapper `train.py` trains through. Without it the policy is handed
    # a dictionary observation it was never trained on — `RealEnv` and `SimEnv`
    # present the same interface now, and a policy trained on one has to meet the
    # other through the same view of it.
    env = SB3Wrapper(RealEnv(config))
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
    # `RealEnv` needs to be told what the beam is being asked for: it does not
    # draw a request, because a real machine is not being asked for a random one.
    cfg = RLConfig.clara(target=(0.0, 0.0, 266e-6, 150e-6))
    deploy(cfg, model_path="models/ppo_accelerator_final.zip")
