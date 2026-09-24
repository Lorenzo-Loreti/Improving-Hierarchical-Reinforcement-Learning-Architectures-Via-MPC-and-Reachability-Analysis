"""Tests for hppo_train.py's evaluation controller and checkpoint loader, the
two pieces the study scripts (algorithms/study.py) replay trained hierarchies
with."""

import numpy as np
import torch
from gymnasium import spaces

from hppo import HPPOAgent
from hppo_train import load_agent, make_policy_fn

OBS_LOW = np.array([-1.0, -2.0, -1.2, -1.2], dtype=np.float32)
OBS_HIGH = np.array([11.0, 2.0, 1.2, 1.2], dtype=np.float32)


class _Env:
    """Just the two spaces load_agent reads."""
    observation_space = spaces.Box(OBS_LOW, OBS_HIGH, dtype=np.float32)
    action_space = spaces.Box(-2.5, 2.5, shape=(2,), dtype=np.float32)


def _agent(manager_freq=3, seed=0):
    torch.manual_seed(seed)
    return HPPOAgent(obs_dim=4, goal_dim=2, act_dim=2, worker_act_limit_low=-2.5, worker_act_limit_high=2.5,
                     obs_low=OBS_LOW, obs_high=OBS_HIGH, manager_freq=manager_freq)


def _observations(n):
    """An agent moving right and slightly up."""
    return [np.array([1.0 + 0.1 * k, 0.02 * k, 1.0, 0.2], dtype=np.float32) for k in range(n)]


def test_the_manager_replans_every_manager_freq_steps_and_the_goal_decays_between():
    agent = _agent(manager_freq=3)
    trace = []
    policy_fn = make_policy_fn(agent, trace)
    observations = _observations(7)
    for obs in observations:
        policy_fn(obs)
    assert [r for r, _ in trace] == [True, False, False, True, False, False, True]
    # Between re-plans the goal is the previous one minus the displacement.
    for k in (1, 2, 4, 5):
        displacement = observations[k][:2] - observations[k - 1][:2]
        np.testing.assert_allclose(trace[k][1], trace[k - 1][1] - displacement, atol=1e-6)


def test_tracing_the_goals_changes_no_action():
    agent = _agent()
    traced, plain = make_policy_fn(agent, []), make_policy_fn(agent)
    for obs in _observations(8):
        np.testing.assert_array_equal(traced(obs), plain(obs))


def test_a_reloaded_hierarchy_drives_the_controller_identically(tmp_path):
    agent = _agent(manager_freq=4, seed=2)
    path = str(tmp_path / "agent.pt")
    agent.save(path)
    reloaded = load_agent(path, _Env())
    assert reloaded.manager_freq == 4
    original, again = make_policy_fn(agent), make_policy_fn(reloaded)
    for obs in _observations(9):
        np.testing.assert_array_equal(original(obs), again(obs))
