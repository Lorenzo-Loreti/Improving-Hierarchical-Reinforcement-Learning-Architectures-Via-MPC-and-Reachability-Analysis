"""Tests of the Gymnasium adapter; they run where Gymnasium is installed (the project's virtual environment)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402
from gymnasium.wrappers import RecordEpisodeStatistics  # noqa: E402

from hrlmpc.config import EnvConfig, env_config_from_dict  # noqa: E402
from hrlmpc.gym_env import NavigationGymEnv  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "env" / "slalom.yaml"


def _config(**sections: dict[str, Any]) -> EnvConfig:
    data = copy.deepcopy(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    for section, values in sections.items():
        data[section].update(values)
    return env_config_from_dict(data)


def test_the_adapter_passes_the_gymnasium_checks() -> None:
    check_env(NavigationGymEnv(_config(physics={"d_bar": 1.0})), skip_render_check=True)


def test_a_seeded_episode_is_reproducible() -> None:
    """With disturbances: the same seed and actions give the same trajectory."""
    env = NavigationGymEnv(_config(physics={"d_bar": 1.0}, task={"horizon": 50}))
    runs = []
    for _ in range(2):
        obs, _ = env.reset(seed=7)
        trajectory, rng = [obs], np.random.default_rng(0)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(rng.uniform(-1.0, 1.0, 2))
            trajectory.append(np.append(obs, reward))
        runs.append(np.concatenate(trajectory))
    np.testing.assert_array_equal(runs[0], runs[1])


def test_an_episode_ends_cleanly() -> None:
    env = NavigationGymEnv(_config())
    env.reset(seed=7)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))
        steps += 1
    assert truncated and not terminated and steps == 200  # at rest, with no disturbance, until the horizon
    assert isinstance(reward, float) and info["episode_stats"]["length"] == 200
    assert env.observation_space.contains(obs)
    with pytest.raises(ValueError, match="reset"):
        env.step(np.zeros(2))


def test_reset_options_set_the_initial_state() -> None:
    env = NavigationGymEnv(_config())
    env.reset(seed=0, options={"position": [9.95, 0.0], "velocity": [1.0, 0.0]})
    _, reward, terminated, _, info = env.step(np.zeros(2))
    assert terminated and info["episode_stats"]["success"]
    assert reward == pytest.approx(999.5)
    with pytest.raises(ValueError, match="unknown reset options"):
        env.reset(options={"pos": [1.0, 0.0]})


def test_actions_must_have_the_shape_of_the_action_space() -> None:
    env = NavigationGymEnv(_config())
    env.reset(seed=0)
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros((2, 1)))


def test_gymnasiums_episode_statistics_wrapper_works() -> None:
    env = RecordEpisodeStatistics(NavigationGymEnv(_config(task={"horizon": 3})))
    env.reset(seed=0)
    for _ in range(3):
        *_, info = env.step(np.zeros(2))
    assert info["episode"]["l"] == 3 and info["episode_stats"]["length"] == 3
