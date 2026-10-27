from rl_framework.config import RLConfig
from rl_framework.envs.real_env import RealEnv
from rl_framework.envs.sim_env import SimEnv
from rl_framework.wrappers import SB3Wrapper

# `RealEnv` used to be imported lazily because it pulled in CATAP, which only
# exists on the control network. It no longer imports anything of the sort: every
# point where it touches the machine is a placeholder in `envs/hardware.py`, so
# it is an ordinary import that works anywhere. Constructing one is still
# harmless — the placeholders raise until they are populated.
__all__ = ["RLConfig", "SimEnv", "RealEnv", "SB3Wrapper"]
