import gymnasium.error
from gymnasium.envs.registration import register

from .config import TunnelEnvConfig
from .tunnel_env import TunnelEnv
from .width_profile import WidthSegment, WidthProfile, constant_profile

try:
    register(id="TunnelEnv-v0", entry_point="envs.tunnel_env:TunnelEnv")
except gymnasium.error.Error:
    pass

__all__ = [
    "TunnelEnv",
    "TunnelEnvConfig",
    "WidthSegment",
    "WidthProfile",
    "constant_profile",
]
