"""Tests for the minimum-time/maximum-return oracle in optimal_solver.py.

Uses TunnelEnv (whichever scenario's `envs` conftest.py puts on sys.path --
see that module's docstring) with hand-built WidthProfiles even for the
"offset gate" cases: TunnelEnv and SlalomEnv are identical apart from class
name (both are a plain LTI double integrator plus a WidthProfile lookup), so
exercising gate behaviour through TunnelEnv with a synthetic profile is
equivalent to doing it through SlalomEnv, and avoids the two scenarios'
same-named `envs` packages colliding on sys.path within one test session.
"""

import numpy as np
import pytest

from optimal_solver import solve_min_time, spawn_grid, precompute_optimal_grid, MinTimeSolver
from envs.tunnel_env import TunnelEnv
from envs.config import TunnelEnvConfig
from envs.width_profile import WidthSegment, WidthProfile


def _make_env(width_profile=None, seed=0, **overrides):
    config = TunnelEnvConfig(width_profile=width_profile, **overrides)
    env = TunnelEnv(config=config)
    env.reset(seed=seed)
    return env


def _gate_profile():
    """One narrow, laterally-offset gate -- same shape as slalom_profile's
    gate 1 (half_width=0.75, offset=1.0 against tunnel_width=4)."""
    return WidthProfile([
        WidthSegment(-np.inf, 4.0, 2.0, 0.0),
        WidthSegment(4.0, 5.0, 0.75, 1.0),
        WidthSegment(5.0, np.inf, 2.0, 0.0),
    ])


def test_constant_profile_never_needs_lateral_effort():
    env = _make_env(seed=1)
    result = solve_min_time(env)
    assert result.success
    assert result.collision_count == 0
    assert np.allclose(result.actions[:, 1], 0.0, atol=1e-3)


def test_constant_profile_matches_closed_form_step_count():
    env = _make_env(seed=2)
    p_x0 = float(env.state[0])
    result = solve_min_time(env)

    d = env.L - p_x0
    t_to_vmax = env.v_max / env.u_max
    d_to_vmax = 0.5 * env.u_max * t_to_vmax ** 2
    if d <= d_to_vmax:
        t_star = np.sqrt(2 * d / env.u_max)
    else:
        t_star = t_to_vmax + (d - d_to_vmax) / env.v_max
    expected_n = int(np.ceil(t_star / env.dt - 1e-9))

    assert abs(result.length - expected_n) <= 1


def test_offset_gate_is_threaded_without_collision():
    env = _make_env(width_profile=_gate_profile(), seed=3)
    result = solve_min_time(env)
    assert result.success
    assert result.collision_count == 0

    gate_mask = (result.states[:, 0] > 4.0) & (result.states[:, 0] <= 5.0)
    assert gate_mask.any()
    assert np.all(result.states[gate_mask, 1] <= 1.75 + 1e-6)
    assert np.all(result.states[gate_mask, 1] >= 0.25 - 1e-6)


def test_return_matches_closed_form_reward_formula():
    """goal_reward - (N - 1) + progress_reward_coef * (L - p_x0) plus the
    effort term: the potential is cut at L (since 2026-10-04), so the
    overshoot the solver adds to survive float32 replay noise
    (`_TERMINAL_MARGIN`) earns nothing, and the effort is the replayed
    actions' own. The model's prediction agrees with the replay."""
    for profile, seed in ((None, 4), (_gate_profile(), 3)):
        env = _make_env(width_profile=profile, seed=seed)
        p_x0 = float(env.state[0])
        result = solve_min_time(env)
        assert result.collision_count == 0
        assert result.states[-1, 0] > env.L

        effort = env.config.effort_penalty / env.u_max ** 2 * np.sum(result.actions ** 2)
        expected = (
            env.config.goal_reward
            - (result.length - 1)
            + env.config.progress_reward_coef * (env.L - p_x0)
            + effort
        )
        assert effort < 0
        assert result.total_return == pytest.approx(expected, abs=1e-3)
        assert result.predicted_return == pytest.approx(result.total_return, abs=1e-3)


def test_plan_respects_the_disk_limits():
    """The oracle's model is the environment's: ||u|| <= u_max and
    ||v|| <= v_max (envs/actuation.py), not the per-axis box it used until
    2026-10-04. Through an offset gate, where the per-axis limits let the
    old oracle steer at no cost in time, the disks make it pay: it arrives
    later than the x-only lower bound."""
    from optimal_solver import _x_only_min_time_steps
    env = _make_env(width_profile=_gate_profile(), seed=0)
    env.reset(options={"init_state": np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32)})
    result = solve_min_time(env)
    assert np.all(np.linalg.norm(result.actions, axis=1) <= env.u_max + 1e-4)
    assert np.all(np.linalg.norm(result.states[:, 2:], axis=1) <= env.v_max + 1e-4)
    assert result.length > _x_only_min_time_steps(0.0, env.L, env.v_max, env.u_max, env.dt)


def test_a_heavy_effort_penalty_trades_time_for_effort():
    """The search over N maximises the return, not just the arrival time:
    with an effort weight large enough to outweigh a few steps, the oracle
    arrives later than the minimum time and earns more than the minimum-time
    plan would in the same environment."""
    start = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    cheap = _make_env(effort_penalty=0.0)
    cheap.reset(options={"init_state": start})
    fastest = solve_min_time(cheap)

    costly = _make_env(effort_penalty=-20.0)
    costly.reset(options={"init_state": start})
    best = solve_min_time(costly)
    costly.reset(options={"init_state": start})
    fastest_there = sum(costly.step(a)[1] for a in fastest.actions.astype(np.float32))

    assert best.length > fastest.length
    assert best.total_return > fastest_there + 1.0
    assert best.predicted_return == pytest.approx(best.total_return, abs=1e-3)


def test_spawn_grid_shape_bounds_and_zero_velocity():
    env = _make_env(seed=0)
    grid = spawn_grid(env, n_x=3, n_y=4)
    assert grid.shape == (12, 4)
    assert np.all(grid[:, 0] >= 0.0) and np.all(grid[:, 0] <= 2.0)
    assert np.all(grid[:, 1] >= -env.W / 4.0) and np.all(grid[:, 1] <= env.W / 4.0)
    np.testing.assert_array_equal(grid[:, 2:], 0.0)
    # Endpoints included: the corners of the spawn box are actually on the grid.
    assert np.any(np.isclose(grid[:, 0], 0.0)) and np.any(np.isclose(grid[:, 0], 2.0))
    assert np.any(np.isclose(grid[:, 1], -env.W / 4.0)) and np.any(np.isclose(grid[:, 1], env.W / 4.0))


def test_precompute_optimal_grid_solves_every_point_feasibly():
    env = _make_env(width_profile=_gate_profile(), seed=0)
    grid = spawn_grid(env, n_x=3, n_y=3)
    solver = MinTimeSolver()
    results = precompute_optimal_grid(env, grid, solver=solver)

    assert len(results) == len(grid)
    for point, result in results:
        assert result.success
        assert result.collision_count == 0
    # Same QP cache reused across all 9 points (see MinTimeSolver's class
    # docstring) rather than one solver built per point.
    assert len(solver._qp_cache) == 1


def test_infeasible_profile_raises_instead_of_lying():
    # A segment covering the known spawn p_x0, whose bound excludes the
    # known spawn p_y0 by a wide margin -- infeasible from the very first
    # (fixed) state, for every candidate horizon. Guards against SCP
    # quietly returning a trajectory that isn't actually valid.
    probe = _make_env(seed=5)
    p_x0, p_y0 = float(probe.state[0]), float(probe.state[1])

    excluding_profile = WidthProfile([
        WidthSegment(-np.inf, p_x0 + 0.5, 0.05, p_y0 + 3.0),
        WidthSegment(p_x0 + 0.5, np.inf, 2.0, 0.0),
    ])
    env = _make_env(width_profile=excluding_profile, seed=5)
    with pytest.raises(RuntimeError):
        solve_min_time(env)


@pytest.mark.parametrize("start", [(0.0, -1.0), (1.0, 0.0), (2.0, 1.0)])
def test_full_thrust_does_not_beat_the_oracle(start):
    """Regression for the speed-limit defect fixed on 2026-09-24 (see
    envs/actuation.py): full thrust down the open corridor used to arrive 7
    steps before the oracle, which was then no upper bound on the env. Now
    the oracle's ||v|| <= v_max model is the env's, and it arrives no later
    -- nor earns more."""
    env = _make_env()
    init = np.array([start[0], start[1], 0.0, 0.0], dtype=np.float32)
    env.reset(options={"init_state": init})
    steps, total, done = 0, 0.0, False
    while not done:
        _, reward, terminated, truncated, _ = env.step(np.array([env.u_max, 0.0], dtype=np.float32))
        steps += 1
        total += reward
        done = terminated or truncated
    env.reset(options={"init_state": init})
    oracle = MinTimeSolver().solve(env)
    assert steps >= oracle.length
    assert total <= oracle.total_return + 1e-4
