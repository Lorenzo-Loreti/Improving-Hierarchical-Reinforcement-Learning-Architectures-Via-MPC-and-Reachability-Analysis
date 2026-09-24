"""Tests for ppo_train.py's evaluation controller and checkpoint loader, the
two pieces the study scripts (algorithms/study.py) replay trained policies
with."""

import numpy as np
import torch
from gymnasium import spaces

from ppo import PPOAgent
from ppo_train import load_agent, make_policy_fn

OBS_LOW = np.array([-1.0, -2.0, -1.2, -1.2], dtype=np.float32)
OBS_HIGH = np.array([11.0, 2.0, 1.2, 1.2], dtype=np.float32)


class _Env:
    """Just the two spaces load_agent reads."""
    observation_space = spaces.Box(OBS_LOW, OBS_HIGH, dtype=np.float32)
    action_space = spaces.Box(-2.5, 2.5, shape=(2,), dtype=np.float32)


def test_a_reloaded_agent_drives_the_controller_identically(tmp_path):
    torch.manual_seed(0)
    agent = PPOAgent(obs_dim=4, act_dim=2, act_limit_low=-2.5, act_limit_high=2.5,
                     obs_low=OBS_LOW, obs_high=OBS_HIGH)
    path = str(tmp_path / "agent.pt")
    agent.save(path)
    reloaded = load_agent(path, _Env())
    for obs in (np.array([0.5, -0.5, 0.0, 0.0]), np.array([7.3, 0.4, 1.1, -0.2])):
        np.testing.assert_array_equal(make_policy_fn(agent)(obs), make_policy_fn(reloaded)(obs))


def test_the_controller_is_deterministic_and_within_the_action_limits():
    torch.manual_seed(1)
    agent = PPOAgent(obs_dim=4, act_dim=2, act_limit_low=-2.5, act_limit_high=2.5,
                     obs_low=OBS_LOW, obs_high=OBS_HIGH)
    policy_fn = make_policy_fn(agent)
    obs = np.array([2.0, 1.0, 0.3, 0.0])
    action = policy_fn(obs)
    np.testing.assert_array_equal(action, policy_fn(obs))
    assert action.shape == (2,) and np.all(np.abs(action) <= 2.5)
