import gymnasium.error
from gymnasium.envs.registration import register

from .config import SlalomEnvConfig
from .slalom_env import SlalomEnv
from .width_profile import WidthSegment, WidthProfile, constant_profile, slalom_profile

try:
    register(id="SlalomEnv-v0", entry_point="envs.slalom_env:SlalomEnv")
except gymnasium.error.Error:
    pass

__all__ = [
    "SlalomEnv",
    "SlalomEnvConfig",
    "WidthSegment",
    "WidthProfile",
    "constant_profile",
    "slalom_profile",
]
