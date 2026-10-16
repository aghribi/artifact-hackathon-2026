from rl_framework.config import RLConfig
from rl_framework.envs.sim_env import SimEnv
from rl_framework.wrappers import SB3Wrapper

__all__ = ["RLConfig", "SimEnv", "SB3Wrapper", "RealEnv"]


def __getattr__(name):
    """`RealEnv` on demand: it needs CATAP, which only exists on the control network,
    and the simulation has to import without it."""
    if name == "RealEnv":
        from rl_framework.envs.real_env import RealEnv

        return RealEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
