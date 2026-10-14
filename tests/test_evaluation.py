"""Tests of the evaluation protocol and of the minimum-time bound (decision log D6, D20, F7, D43, D50)."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from hrlmpc.action_map import square_to_disk
from hrlmpc.config import Box, EnvConfig, env_config_from_dict
from hrlmpc.env import EpisodeStats, NavigationEnv
from hrlmpc.evaluation import (
    EVAL_ACTIONS,
    Evaluation,
    append_details,
    arrival_lower_bound,
    console_line,
    evaluate,
    evaluation_name,
    min_steps_lower_bound,
    progress_bound,
    spawn_grid,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "env"


def _config(layout: str = "tunnel", **sections: dict[str, Any]) -> EnvConfig:
    data = copy.deepcopy(yaml.safe_load((CONFIGS / f"{layout}.yaml").read_text(encoding="utf-8")))
    for section, values in sections.items():
        data[section].update(values)
    return env_config_from_dict(data)


def _full_thrust(obs: np.ndarray) -> np.ndarray:
    return np.tile([1.0, 0.0], (len(obs), 1))


# --------------------------------------------------------------------------
# Starts
# --------------------------------------------------------------------------


def test_the_grid_spans_the_spawn_box_row_by_row() -> None:
    grid = spawn_grid(Box(x=(0.0, 2.0), y=(-1.0, 1.0)), (5, 5))
    assert grid.shape == (25, 2)
    np.testing.assert_array_equal(grid[:5], [[0.0, -1.0], [0.0, -0.5], [0.0, 0.0], [0.0, 0.5], [0.0, 1.0]])
    np.testing.assert_array_equal(grid[-1], [2.0, 1.0])
    assert sorted(set(grid[:, 0])) == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_a_single_point_grid_is_the_centre_of_the_box() -> None:
    np.testing.assert_array_equal(spawn_grid(Box(x=(0.0, 2.0), y=(-1.0, 1.0)), (1, 1)), [[1.0, 0.0]])
    np.testing.assert_array_equal(spawn_grid(Box(x=(0.0, 2.0), y=(-1.0, 1.0)), (2, 1))[:, 1], [0.0, 0.0])


@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (-1, 2)])
def test_the_grid_needs_at_least_one_point_per_axis(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="at least one"):
        spawn_grid(Box(x=(0.0, 2.0), y=(-1.0, 1.0)), shape)


# --------------------------------------------------------------------------
# Minimum-time bound (F7)
# --------------------------------------------------------------------------


def test_full_thrust_in_the_tunnel_attains_the_bound_exactly() -> None:
    """F7: without obstacles or disturbance, full thrust along +x is time-optimal and follows the bound bit for bit."""
    config = _config()
    starts = spawn_grid(config.task.spawn, (5, 5))
    bound = progress_bound(starts[:, 0], 120, config.physics)
    steps = min_steps_lower_bound(starts[:, 0], config.geometry.goal_x, config.physics)
    env = NavigationEnv(config, num_envs=25, seed=0, autoreset=False)
    env.reset(positions=starts)
    lengths = np.zeros(25, dtype=np.int64)
    for k in range(1, 121):
        out = env.step(_full_thrust(starts))
        running = lengths == 0
        np.testing.assert_array_equal(env.positions[running, 0], bound[running, k])
        lengths[running & out.terminated] = k
        if (lengths > 0).all():
            break
        if out.terminated.any():
            env.reset(positions=starts[out.terminated], mask=out.terminated)
    np.testing.assert_array_equal(lengths, steps)
    assert steps.min() == 69 and steps.max() == 86  # from p_x0 = 2 and from p_x0 = 0


@pytest.mark.parametrize("layout", ["tunnel", "slalom"])
@pytest.mark.parametrize("d_bar", [0.0, 1.0])
def test_no_policy_progresses_faster_than_the_bound(layout: str, d_bar: float) -> None:
    """F7 holds on every layout, with disturbances, initial speeds and wall contacts."""
    config = _config(layout, physics={"d_bar": d_bar}, task={"horizon": 400})
    rng = np.random.default_rng(5)
    candidates = rng.uniform([-1.0, -2.0], [10.0, 2.0], (4000, 2))
    env = NavigationEnv(config, num_envs=1, seed=0)
    free = candidates[env.layout.in_free_space(candidates) & ~env.layout.in_goal(candidates)][:60]
    speeds = rng.uniform(0.0, config.physics.v_max, len(free))
    angles = rng.uniform(0.0, 2.0 * np.pi, len(free))
    velocities = speeds[:, None] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    accel = config.physics.a_max + d_bar
    bound = progress_bound(free[:, 0], 400, config.physics, speed0=speeds, accel=accel)
    least = min_steps_lower_bound(free[:, 0], config.geometry.goal_x, config.physics, speed0=speeds, accel=accel)
    least_free = arrival_lower_bound(env.layout, free, config.physics, speed0=speeds, accel=accel)
    assert np.all(least_free >= least)
    env = NavigationEnv(config, num_envs=len(free), seed=1)
    env.reset(positions=free, velocities=velocities)
    first_done = np.zeros(len(free), dtype=bool)
    contacts = 0
    for k in range(1, 401):
        lateral = rng.uniform(-1.0, 1.0, len(free))
        forward = np.where(np.arange(len(free)) % 2 == 0, 1.0, rng.uniform(-1.0, 1.0, len(free)))
        out = env.step(np.stack([forward, lateral], axis=1))
        contacts += int(out.info.contact[~first_done].sum())
        running = ~first_done & ~(out.terminated | out.truncated)
        assert np.all(env.positions[running, 0] <= bound[running, k] + 1e-12)
        succeeded = ~first_done & out.terminated
        assert np.all(k >= least_free[succeeded])
        first_done |= out.terminated | out.truncated
        if first_done.all():
            break
    assert contacts > 0  # the walls were exercised


def test_the_bound_counts_zero_steps_from_the_goal_region_and_rejects_bad_inputs() -> None:
    params = _config().physics
    np.testing.assert_array_equal(min_steps_lower_bound([10.0, 10.5], 10.0, params), [0, 0])
    with pytest.raises(ValueError, match="speed0"):
        progress_bound([0.0], 3, params, speed0=[2.0])
    with pytest.raises(ValueError, match="accel"):
        progress_bound([0.0], 3, params, accel=0.0)
    with pytest.raises(ValueError, match="steps"):
        progress_bound([0.0], -1, params)


def test_the_bound_gives_up_after_max_steps() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        min_steps_lower_bound([0.0], 10.0, _config().physics, max_steps=10)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def test_the_evaluation_runs_one_episode_from_each_start() -> None:
    config = _config()
    starts = spawn_grid(config.task.spawn, (5, 5))
    result = evaluate(_full_thrust, config, starts, seed=4)
    assert [ep.env for ep in result.episodes] == list(range(25))
    np.testing.assert_array_equal(result.starts, starts)
    np.testing.assert_array_equal([ep.length for ep in result.episodes], result.lower_bounds)
    metrics = result.metrics()
    assert metrics["eval/success_rate"] == 1.0
    assert metrics["eval/arrival_gap_mean"] == 0.0 and metrics["eval/arrival_gap_max"] == 0.0
    assert metrics["eval/contact_fraction"] == 0.0 and metrics["eval/impulse"] == 0.0
    expected = np.mean([1000.0 - n - 0.01 * n for n in result.lower_bounds])  # bonus, time and full effort
    assert metrics["eval/unshaped_return"] == pytest.approx(expected, abs=1e-9)


def test_the_evaluation_is_repeatable_and_uses_its_own_environment() -> None:
    config = _config("slalom", physics={"d_bar": 1.0})
    starts = spawn_grid(config.task.spawn, (3, 3))

    def policy(obs: np.ndarray) -> np.ndarray:
        return np.clip(np.stack([np.ones(len(obs)), -2.0 * obs[:, 1]], axis=1), -1.0, 1.0)

    first = evaluate(policy, config, starts, seed=4)
    again = evaluate(policy, config, starts, seed=4)
    other = evaluate(policy, config, starts, seed=5)
    assert first.episodes == again.episodes
    assert first.episodes != other.episodes  # the disturbances differ


def test_the_evaluation_rejects_bad_starts() -> None:
    config = _config("slalom")
    with pytest.raises(ValueError, match="free space"):
        evaluate(_full_thrust, config, [[4.5, -1.0]], seed=0)
    with pytest.raises(ValueError, match="shape"):
        evaluate(_full_thrust, config, [1.0, 0.0], seed=0)


def test_commanded_accelerations_can_be_evaluated() -> None:
    """With ``inputs=True`` the commands are accelerations, the MPC Worker's interface (D29, step 9)."""
    config = _config("slalom", physics={"d_bar": 0.5})
    starts = spawn_grid(config.task.spawn, (2, 3))
    u = square_to_disk(np.array([[1.0, 0.25]]), config.physics.a_max)
    by_actions = evaluate(lambda obs: np.tile([1.0, 0.25], (len(obs), 1)), config, starts, seed=4)
    by_inputs = evaluate(lambda obs: np.tile(u[0], (len(obs), 1)), config, starts, seed=4, inputs=True)
    assert by_actions.details() == by_inputs.details()
    too_strong = evaluate(lambda obs: np.tile([5.0, 0.0], (len(obs), 1)), config, starts, seed=4, inputs=True)
    assert all(ep.saturation_steps > 0 for ep in too_strong.episodes)  # the commands were not mapped by D15


def test_the_evaluation_stops_at_the_horizon() -> None:
    config = _config(task={"horizon": 7})
    result = evaluate(lambda obs: np.zeros((len(obs), 2)), config, [[0.0, 0.0], [1.0, 0.5]], seed=0)
    assert [ep.length for ep in result.episodes] == [7, 7]
    metrics = result.metrics()
    assert metrics["eval/success_rate"] == 0.0
    assert math.isnan(metrics["eval/arrival_gap_mean"]) and math.isnan(metrics["eval/arrival_gap_max"])


def _stats(env: int, length: int, success: bool, contact_steps: int = 0, impulse: float = 0.0) -> EpisodeStats:
    return EpisodeStats(
        env=env, length=length, success=success, episode_return=float(length), unshaped_return=-float(length),
        contact_steps=contact_steps, contact_events=min(contact_steps, 1), impulse=impulse, wall_steps=contact_steps,
        limiter_steps=length // 2, limiter_total=0.5 * length, saturation_steps=0, saturation_total=0.0,
    )


def test_the_metrics_summarise_the_episodes() -> None:
    result = Evaluation(
        starts=np.zeros((4, 2)),
        episodes=(_stats(0, 105, True), _stats(1, 110, True, 3, 0.6), _stats(2, 200, False), _stats(3, 100, True)),
        lower_bounds=np.array([100, 100, 100, 100]),
        path_lengths=np.full(4, 11.0),
    )
    m = result.metrics()
    assert m["eval/success_rate"] == 0.75
    assert m["eval/arrival_gap_mean"] == pytest.approx((0.05 + 0.10 + 0.0) / 3)
    assert m["eval/arrival_gap_max"] == pytest.approx(0.10)
    assert m["eval/contact_fraction"] == 0.25 and m["eval/contact_steps"] == 0.75
    assert m["eval/impulse"] == pytest.approx(0.15) and m["eval/length"] == pytest.approx(128.75)
    assert m["eval/limiter_total"] == pytest.approx(0.5 * 128.75)
    assert set(result.metrics(prefix="x/")) == {k.replace("eval/", "x/") for k in m}


def test_the_oracle_gap_is_measured_against_the_oracle_steps() -> None:
    """D44: with the oracle's minimum times, the gap of a successful episode is its length over them, minus one."""
    episodes = (_stats(0, 105, True), _stats(1, 110, True), _stats(2, 200, False), _stats(3, 100, True))
    bounds = np.array([100, 100, 100, 100])
    without = Evaluation(np.zeros((4, 2)), episodes, bounds, np.full(4, 11.0))
    assert math.isnan(without.metrics()["eval/oracle_gap_mean"])
    assert math.isnan(without.metrics()["eval/oracle_gap_max"])
    assert without.details()["oracle_steps"] is None
    with_oracle = Evaluation(
        np.zeros((4, 2)), episodes, bounds, np.full(4, 11.0), oracle_steps=np.array([105, 100, 150, 100])
    )
    m = with_oracle.metrics()
    assert m["eval/oracle_gap_mean"] == pytest.approx((0.0 + 0.10 + 0.0) / 3)
    assert m["eval/oracle_gap_max"] == pytest.approx(0.10)
    assert m["eval/arrival_gap_mean"] == pytest.approx((0.05 + 0.10 + 0.0) / 3)  # the bound of F7 stays
    assert with_oracle.details()["oracle_steps"] == [105, 100, 150, 100]
    none_succeed = Evaluation(np.zeros((1, 2)), (_stats(0, 200, False),), np.array([100]), np.full(1, 11.0),
                              oracle_steps=np.array([100]))
    assert math.isnan(none_succeed.metrics()["eval/oracle_gap_mean"])


def test_the_evaluation_takes_the_oracle_table_of_its_starts() -> None:
    """D44: ``evaluate`` reads the minimum times from a table of the same starts, and refuses another's."""
    from hrlmpc.oracle import OracleEntry, OracleSettings, OracleTable, layout_signature

    config = _config()
    starts = spawn_grid(config.task.spawn, (2, 2))
    physics, geometry = layout_signature(config)

    def table(points: np.ndarray, steps: list[int]) -> OracleTable:
        entries = tuple(
            OracleEntry((float(p[0]), float(p[1])), int(n), int(n), 0.0, (), int(n), True, 0, 0, 0.0)
            for p, n in zip(points, steps, strict=True)
        )
        return OracleTable(physics, geometry, {"rho": OracleSettings.rho}, entries)

    result = evaluate(_full_thrust, config, starts, seed=0, oracle=table(starts, [86, 86, 69, 69]))
    np.testing.assert_array_equal(result.oracle_steps, [86, 86, 69, 69])
    np.testing.assert_array_equal(result.lower_bounds, result.oracle_steps)
    assert result.metrics()["eval/oracle_gap_max"] == 0.0
    details = result.details()
    assert json.loads(json.dumps(details))["oracle_steps"] == [86, 86, 69, 69]
    with pytest.raises(ValueError, match="starts"):
        evaluate(_full_thrust, config, starts, seed=0, oracle=table(starts[:3], [86, 86, 69]))


def test_the_details_are_plain_json() -> None:
    result = evaluate(_full_thrust, _config(), [[0.0, 0.0], [2.0, 1.0]], seed=0)
    details = result.details()
    assert json.loads(json.dumps(details)) == details
    assert details["starts"] == [[0.0, 0.0], [2.0, 1.0]]
    assert details["path_length"] == [10.0, 8.0]
    assert details["length"] == details["lower_bound"] == [86, 69]
    assert details["success"] == [True, True]


def test_the_default_acceleration_bound_includes_the_disturbance() -> None:
    disturbed, nominal = _config(physics={"d_bar": 1.0}).physics, _config().physics
    np.testing.assert_array_equal(progress_bound([0.0], 50, disturbed), progress_bound([0.0], 50, nominal, accel=3.5))
    assert min_steps_lower_bound([0.0], 10.0, disturbed)[0] < min_steps_lower_bound([0.0], 10.0, nominal)[0]


def test_reaching_the_goal_line_exactly_counts() -> None:
    """As in NavigationEnv, p_x = goal_x is in the goal region."""
    params = _config().physics
    goal = float(progress_bound([0.0], 60, params)[0, 40])
    assert min_steps_lower_bound([0.0], goal, params)[0] == 40
    np.testing.assert_array_equal(min_steps_lower_bound([0.0, 0.0], [goal, 10.0], params), [40, 86])


def test_obstacles_lengthen_the_bound_only_where_they_block_the_straight_run() -> None:
    """F7 with the shortest free path: equal to the straight bound in the tunnel, larger behind the slalom's gates."""
    for layout_name in ("tunnel", "slalom"):
        config = _config(layout_name)
        env = NavigationEnv(config, num_envs=1, seed=0)
        starts = spawn_grid(config.task.spawn, (5, 5))
        free = arrival_lower_bound(env.layout, starts, config.physics)
        straight = min_steps_lower_bound(starts[:, 0], config.geometry.goal_x, config.physics)
        if layout_name == "tunnel":
            np.testing.assert_array_equal(free, straight)
        else:
            assert np.all(free >= straight) and (free > straight).sum() == 19  # short detours cost no step
    assert free[spawn_grid(config.task.spawn, (5, 5)).tolist().index([2.0, -1.0])] == 73


def test_the_evaluation_keeps_the_first_episode_of_each_start() -> None:
    def policy(obs: np.ndarray) -> np.ndarray:
        actions = np.zeros((len(obs), 2))
        actions[0, 0] = 1.0  # agent 0 drives, finishes and goes on in new episodes; agent 1 waits for the horizon
        return actions

    result = evaluate(policy, _config(), [[2.0, 0.0], [0.0, 0.0]], seed=0)
    assert [(ep.length, ep.success) for ep in result.episodes] == [(69, True), (200, False)]


def test_the_contact_fraction_counts_contact_steps() -> None:
    """Lying on a face without a wall reaction is contact too (D14)."""
    episodes = (_stats(0, 10, True, contact_steps=2), _stats(1, 10, True))
    episodes = (EpisodeStats(**{**episodes[0].__dict__, "wall_steps": 0}), episodes[1])
    result = Evaluation(np.zeros((2, 2)), episodes, np.array([10, 10]), np.full(2, 1.0))
    assert result.metrics()["eval/contact_fraction"] == 0.5


def test_an_evaluation_touches_neither_the_training_environment_nor_global_randomness() -> None:
    """D6: training goes on exactly as without the evaluations in between."""
    config = _config("slalom", physics={"d_bar": 1.0})
    plain, interleaved = NavigationEnv(config, num_envs=4, seed=7), NavigationEnv(config, num_envs=4, seed=7)
    plain.reset()
    interleaved.reset()
    actions = np.tile([1.0, 0.3], (4, 1))
    state = np.random.get_state()[1].copy()
    for k in range(30):
        if k % 10 == 0:
            evaluate(_full_thrust, config, [[0.0, 0.0], [1.0, 0.5]], seed=12345)
        a, b = plain.step(actions), interleaved.step(actions)
        np.testing.assert_array_equal(a.obs, b.obs)
        np.testing.assert_array_equal(a.info.disturbance, b.info.disturbance)
    assert plain.counter.env_steps == interleaved.counter.env_steps == 120
    np.testing.assert_array_equal(np.random.get_state()[1], state)


def test_the_evaluation_measures_against_the_free_path_bound() -> None:
    """In the slalom the reference of the arrival gap is the bound with the shortest free path, not the straight one."""
    result = evaluate(lambda obs: np.zeros((len(obs), 2)), _config("slalom"), [[2.0, -1.0], [1.0, 0.0]], seed=0)
    np.testing.assert_array_equal(result.lower_bounds, [73, 78])
    detour = math.hypot(2.0, 1.25) + 1.0 + math.hypot(2.0, 0.5) + 1.0 + 2.0
    assert result.path_lengths[0] == pytest.approx(detour, rel=1e-12)


# --------------------------------------------------------------------------
# Both actions and the compute (D43, D50)
# --------------------------------------------------------------------------


def test_the_evaluation_times_the_controller_per_step_and_agent() -> None:
    """D50: the controller's wall-clock time per call, divided by the agents it steers, at every step; its mean and
    largest under ``time/``, with the evaluation's duration, apart from the metrics of the outcome."""

    def slow(obs: np.ndarray) -> np.ndarray:
        until = time.perf_counter() + 0.004  # busy: a sleep may end early on Windows
        while time.perf_counter() < until:
            pass
        return np.zeros((len(obs), 2))

    result = evaluate(slow, _config(task={"horizon": 6}), [[0.0, 0.0], [1.0, 0.5]], seed=0)
    assert result.step_ms.shape == (6,) and np.all(result.step_ms >= 2.0)  # 4 ms per call for two agents
    assert result.duration_s >= 6 * 0.004
    timing = result.timing("eval_mode")
    assert set(timing) == {"time/eval_mode_s", "time/eval_mode_step_ms_mean", "time/eval_mode_step_ms_max"}
    assert timing["time/eval_mode_step_ms_mean"] == pytest.approx(float(result.step_ms.mean()), rel=1e-12)
    assert timing["time/eval_mode_step_ms_max"] == float(result.step_ms.max())
    assert timing["time/eval_mode_s"] == result.duration_s
    assert not any(key.startswith("time/") for key in result.metrics())  # the outcome stays comparable bit for bit


def test_an_evaluation_built_without_its_times_has_none() -> None:
    result = Evaluation(np.zeros((1, 2)), (_stats(0, 10, True),), np.array([10]), np.full(1, 1.0))
    assert result.step_ms.shape == (0,) and math.isnan(result.duration_s)
    assert all(math.isnan(value) for value in result.timing("eval").values())


def test_the_evaluations_are_named_by_their_action() -> None:
    """D50: the evaluation with the mean keeps the names of steps 7-10; the one with the mode adds ``_mode``."""
    assert EVAL_ACTIONS == ("mean", "mode")
    assert [evaluation_name(action) for action in EVAL_ACTIONS] == ["eval", "eval_mode"]
    assert [evaluation_name(action, "mpc") for action in EVAL_ACTIONS] == ["eval_mpc", "eval_mode_mpc"]
    assert evaluation_name("mode", "hierarchy") == "eval_mode_hierarchy"
    assert evaluation_name("mean", "reach") == "eval_reach"
    with pytest.raises(ValueError, match="action"):
        evaluation_name("median")


def test_the_details_of_each_action_are_appended_as_one_line(tmp_path: Path) -> None:
    """D50: ``evaluations.jsonl`` holds one line per evaluation and action, marked with its action."""
    result = evaluate(_full_thrust, _config(), [[0.0, 0.0], [2.0, 1.0]], seed=0)
    path = tmp_path / "evaluations.jsonl"
    append_details(path, result, env_steps=0, action="mean")
    append_details(path, result, env_steps=16, action="mode")
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [(line["env_steps"], line["action"]) for line in lines] == [(0, "mean"), (16, "mode")]
    for line in lines:
        assert {k: v for k, v in line.items() if k not in ("env_steps", "action")} == result.details()
    with pytest.raises(ValueError, match="action"):
        append_details(path, result, env_steps=32, action="median")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_the_console_line_shows_both_actions() -> None:
    metrics = {
        "eval/success_rate": 1.0, "eval_mode/success_rate": 0.75,
        "eval/unshaped_return": 950.3, "eval_mode/unshaped_return": -12.0,
        "eval/arrival_gap_mean": 0.05, "eval_mode/arrival_gap_mean": float("nan"),
        "eval/contact_fraction": 0.0, "eval_mode/contact_fraction": 0.25,
    }
    assert console_line(metrics) == (
        "mean/mode: success 1.00/0.75 | unshaped return 950.3/-12.0 | gap 0.050/  -   | contact 0.00/0.25"
    )

