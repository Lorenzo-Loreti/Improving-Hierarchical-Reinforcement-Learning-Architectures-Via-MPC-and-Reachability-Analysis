"""Gymnasium adapter of :class:`hrlmpc.env.NavigationEnv` for a single agent.

The training loops use :class:`~hrlmpc.env.NavigationEnv` directly, which steps
many agents at once. This adapter exposes one agent through the Gymnasium API
(D5), for Gymnasium's checks and tools; importing it requires Gymnasium.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from hrlmpc.config import EnvConfig
from hrlmpc.env import NavigationEnv

FloatArray = NDArray[np.float64]


class NavigationGymEnv(gym.Env[FloatArray, FloatArray]):
    """One agent of the navigation MDP behind the Gymnasium API.

    The action is ``x`` in ``[-1, 1]^2`` (D15), the observation ``(p, v)``
    mapped to ``[-1, 1]`` (D13). A reset with a seed restarts the random
    streams; ``options`` may set the initial ``"position"`` and ``"velocity"``.
    The ``info`` of a step holds the quantities of
    :class:`~hrlmpc.env.StepInfo` and, when the episode ends, its
    :class:`~hrlmpc.env.EpisodeStats` under ``"episode_stats"`` (the key
    ``"episode"`` belongs to Gymnasium's ``RecordEpisodeStatistics``). After
    the end of an episode, :meth:`reset` must be called before the next step.

    Args:
        config: Physics, geometry, task and reward of the layout.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(NavigationEnv.observation_size,), dtype=np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(NavigationEnv.action_size,), dtype=np.float64)
        self._env: NavigationEnv | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[FloatArray, dict[str, Any]]:
        super().reset(seed=seed)
        options = dict(options or {})
        unknown = set(options) - {"position", "velocity"}
        if unknown:
            raise ValueError(f"unknown reset options {sorted(unknown)}; known: 'position', 'velocity'")
        if seed is not None or self._env is None:
            inner_seed = seed if seed is not None else int(self.np_random.integers(0, 2**63 - 1))
            self._env = NavigationEnv(self.config, num_envs=1, seed=inner_seed, autoreset=False)
        position, velocity = options.get("position"), options.get("velocity")
        obs = self._env.reset(
            positions=None if position is None else [position],
            velocities=None if velocity is None else [velocity],
        )
        return obs[0], {}

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        if self._env is None:
            raise ValueError("call reset() before step()")
        x = np.asarray(action, dtype=np.float64)
        if x.shape != self.action_space.shape:
            raise ValueError(f"the action must have shape {self.action_space.shape}, got {x.shape}")
        out = self._env.step(x[None, :])
        info: dict[str, Any] = {}
        for field in dataclasses.fields(out.info):
            value = getattr(out.info, field.name)[0]
            info[field.name] = value.item() if np.ndim(value) == 0 else value.copy()
        if out.episodes:
            info["episode_stats"] = dataclasses.asdict(out.episodes[0])
        return out.obs[0], float(out.reward[0]), bool(out.terminated[0]), bool(out.truncated[0]), info
