"""Tests for solved_check.py.

Uses TunnelEnv (see test_optimal_solver.py's docstring for why: it and
SlalomEnv are identical apart from class name, so exercising this through
either is equivalent, and avoids the two scenarios' same-named `envs`
packages colliding on sys.path within one test session).
"""

import numpy as np
import pytest

from optimal_solver import spawn_grid, precompute_optimal_grid, MinTimeSolver
from solved_check import check_solved
from envs.tunnel_env import TunnelEnv
from envs.config import TunnelEnvConfig
from envs.width_profile import WidthSegment, WidthProfile


def _make_env(width_profile=None, seed=0):
    config = TunnelEnvConfig(width_profile=width_profile) if width_profile is not None else TunnelEnvConfig()
    env = TunnelEnv(config=config)
    env.reset(seed=seed)
    return env


def _gate_profile():
    return WidthProfile([
        WidthSegment(-np.inf, 4.0, 2.0, 0.0),
        WidthSegment(4.0, 5.0, 0.75, 1.0),
        WidthSegment(5.0, np.inf, 2.0, 0.0),
    ])


def _replay_policy(actions):
    """A stateful policy_fn that plays back a stored action sequence,
    one action per call -- holds the last action if called more times than
    the sequence is long (shouldn't happen if it's truly optimal, but keeps
    this from raising instead of just under/over-performing in that case)."""
    state = {"i": 0}

    def policy_fn(obs):
        i = min(state["i"], len(actions) - 1)
        state["i"] += 1
        return actions[i]

    return policy_fn


def test_optimal_policy_is_judged_solved_at_every_point():
    env = _make_env(width_profile=_gate_profile(), seed=0)
    solver = MinTimeSolver()
    grid = spawn_grid(env, n_x=3, n_y=3)
    optimal_grid = precompute_optimal_grid(env, grid, solver=solver)

    for init_state, optimal_result in optimal_grid:
        make_policy_fn = lambda actions=optimal_result.actions: _replay_policy(actions)
        result = check_solved(make_policy_fn, env, [(init_state, optimal_result)], tolerance=1e-2)
        assert result.solved, f"gap={result.worst_gap} at {init_state}"
        assert result.worst_gap < 1e-2


def test_zero_action_policy_is_judged_not_solved():
    env = _make_env(seed=0)
    solver = MinTimeSolver()
    grid = spawn_grid(env, n_x=3, n_y=3)
    optimal_grid = precompute_optimal_grid(env, grid, solver=solver)

    result = check_solved(lambda: (lambda obs: np.zeros(2, dtype=np.float32)), env, optimal_grid, tolerance=5.0)

    assert not result.solved
    # Never reaches the goal (drifts/idles until truncation), so every gap
    # should be large -- not just barely over tolerance.
    assert np.all(result.gaps > 100.0)
    assert result.worst_point.shape == (4,)


def test_make_policy_fn_is_called_fresh_once_per_grid_point():
    """Regression guard for the reason check_solved takes a *factory*: a
    stateful policy (hPPO/PPO+MPC's manager-replan cadence) must not
    carry state left over from a previous grid point's episode."""
    env = _make_env(seed=0)
    solver = MinTimeSolver()
    grid = spawn_grid(env, n_x=2, n_y=2)
    optimal_grid = precompute_optimal_grid(env, grid, solver=solver)

    call_count = {"n": 0}

    def make_policy_fn():
        call_count["n"] += 1
        return lambda obs: np.zeros(2, dtype=np.float32)

    check_solved(make_policy_fn, env, optimal_grid, tolerance=5.0)
    assert call_count["n"] == len(optimal_grid)
