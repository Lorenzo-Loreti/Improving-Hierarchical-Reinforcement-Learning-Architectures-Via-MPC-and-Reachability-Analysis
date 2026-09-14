import gymnasium.error
from gymnasium.envs.registration import register

from .config import SlalomEnvConfig, make_env
from .slalom_env import SlalomEnv
from .width_profile import WidthSegment, WidthProfile, constant_profile, slalom_profile

try:
    register(id="SlalomEnv-v0", entry_point="envs.slalom_env:SlalomEnv")
except gymnasium.error.Error:
    pass

__all__ = [
    "SlalomEnv",
    "SlalomEnvConfig",
    "make_env",
    "WidthSegment",
    "WidthProfile",
    "constant_profile",
    "slalom_profile",
]
