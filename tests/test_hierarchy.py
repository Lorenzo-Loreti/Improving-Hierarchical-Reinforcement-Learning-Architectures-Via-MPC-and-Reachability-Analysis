"""Tests of the common Manager/Worker loop (decision log D6, D24-D27, D29; rows AR2-AR5, R2, X1)."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from hrlmpc.action_map import square_to_disk
from hrlmpc.config import EnvConfig, env_config_from_dict
from hrlmpc.env import NavigationEnv, ObservationMap
from hrlmpc.evaluation import evaluate
from hrlmpc.hierarchy import (
    CommandWorker,
    DiskTargets,
    HierarchicalController,
    HierarchicalRollout,
    Hierarchy,
    HierarchyState,
    LearnedWorker,
    collect_hierarchical_rollout,
    disk_hierarchy,
    segment_gae,
    segment_reach,
    start_segments,
    worker_inputs,
    worker_reward,
)
from hrlmpc.rollout import compute_gae

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "env"
GAMMA = 0.99
H = 10
M_WEIGHTS = np.array([1.0, 2.0, 3.0, 4.0])
W_WEIGHTS = np.arange(1.0, 7.0)


def _config(layout: str = "tunnel", **sections: dict[str, Any]) -> EnvConfig:
    data = copy.deepcopy(yaml.safe_load((CONFIGS / f"{layout}.yaml").read_text(encoding="utf-8")))
    for section, values in sections.items():
        data[section].update(values)
    return env_config_from_dict(data)


def _hierarchy(config: EnvConfig, segment_steps: int = H) -> Hierarchy:
    return disk_hierarchy(config, segment_steps=segment_steps, reach_factor=1.5, discount=GAMMA)


# Deterministic stubs: the Manager aims forward and toward y = 0, the Worker toward its target.


def _manager_policy(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.clip(np.stack([np.full(len(obs), 0.9), -0.8 * obs[:, 1]], axis=1), -1.0, 1.0)
    return x, -np.sum(x**2, axis=1)


def _manager_value(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs) @ M_WEIGHTS


def _worker_policy(inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.clip(2.0 * inputs[:, 4:6] - 0.3 * inputs[:, 2:4], -1.0, 1.0)
    return x, -np.sum(x**2, axis=1)


def _worker_value(inputs: np.ndarray) -> np.ndarray:
    return np.asarray(inputs) @ W_WEIGHTS


def _random_policy(seed: int) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)

    def policy(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return rng.uniform(-1.0, 1.0, (len(obs), 2)), rng.normal(0.0, 1.0, len(obs))

    return policy


LEARNED = LearnedWorker(policy=_worker_policy, value=_worker_value)


def _run(
    config: EnvConfig,
    *,
    batch: int = 3,
    steps: int = 35,
    seed: int = 0,
    worker: LearnedWorker | CommandWorker = LEARNED,
    manager: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] = _manager_policy,
    positions: Any = None,
    segment_steps: int = H,
) -> tuple[NavigationEnv, Hierarchy, HierarchyState, HierarchicalRollout, HierarchyState]:
    env = NavigationEnv(config, batch, seed=seed)
    hierarchy = _hierarchy(config, segment_steps)
    state = start_segments(env, hierarchy, manager, _manager_value, positions=positions)
    rollout, after = collect_hierarchical_rollout(env, state, steps, hierarchy, manager, _manager_value, worker)
    return env, hierarchy, state, rollout, after


def _reference_segments(
    rewards: np.ndarray, done: np.ndarray, first_steps: np.ndarray, segment_steps: int, gamma: float
) -> list[list[tuple[int, int, float]]]:
    """Per agent, the closed segments as (closing step, length, discounted reward), from the step-level data."""
    steps, batch = rewards.shape
    result: list[list[tuple[int, int, float]]] = []
    for b in range(batch):
        closed, length, total = [], int(first_steps[b]), 0.0
        if length:  # the segment in flight at the start began before this rollout
            total = math.nan
        for t in range(steps):
            total += gamma**length * rewards[t, b]
            length += 1
            if length == segment_steps or done[t, b]:
                closed.append((t, length, total))
                length, total = 0, 0.0
        result.append(closed)
    return result


# --------------------------------------------------------------------------
# Observation map, targets, Worker inputs
# --------------------------------------------------------------------------


def test_observations_invert_to_the_states() -> None:
    config = _config("slalom")
    env = NavigationEnv(config, 1)
    obs_map = ObservationMap.from_config(config)
    rng = np.random.default_rng(0)
    p = rng.uniform([-1.0, -2.0], [11.0, 2.0], (200, 2))
    angle, speed = rng.uniform(0.0, 2.0 * np.pi, 200), rng.uniform(0.0, 1.2, 200)
    v = speed[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    obs = obs_map.observe(p, v)
    np.testing.assert_array_equal(obs, env.observe(p, v))
    np.testing.assert_allclose(obs_map.positions(obs), p, atol=1e-12)
    np.testing.assert_allclose(obs_map.velocities(obs), v, atol=1e-12)
    batched = obs.reshape(20, 10, 4)  # any leading dimensions
    np.testing.assert_array_equal(obs_map.positions(batched), obs_map.positions(obs).reshape(20, 10, 2))


@pytest.mark.parametrize("layout", ["slalom", "tunnel"])
def test_the_target_radius_is_one_and_a_half_segment_reaches(layout: str) -> None:
    """D24: a segment covers at most v_max H T_s = 1.2 m; the targets reach R = 1.8 m."""
    config = _config(layout)
    assert segment_reach(config.physics, 10) == pytest.approx(1.2)
    hierarchy = _hierarchy(config)
    assert isinstance(hierarchy.targets, DiskTargets)
    assert hierarchy.targets.radius == pytest.approx(1.8)
    assert (hierarchy.segment_steps, hierarchy.discount) == (10, GAMMA)


def test_disk_targets_map_the_square_onto_the_disk_around_the_position() -> None:
    targets = DiskTargets(radius=1.8)
    rng = np.random.default_rng(1)
    x = rng.uniform(-1.0, 1.0, (500, 2))
    p = rng.uniform(-1.0, 1.0, (500, 2))
    v = rng.uniform(-1.0, 1.0, (500, 2))
    g = targets(x, p, v)
    np.testing.assert_allclose(g, p + square_to_disk(x, 1.8), atol=1e-15)
    np.testing.assert_allclose(targets(x, p, np.zeros_like(v)), g)  # the disk ignores the velocity
    offset = np.hypot(*(g - p).T)
    assert np.all(offset <= 1.8 + 1e-12)
    np.testing.assert_allclose(offset, 1.8 * np.max(np.abs(x), axis=1), rtol=1e-12)
    cross = np.cross(np.c_[x, np.zeros(500)], np.c_[g - p, np.zeros(500)])[:, 2]
    np.testing.assert_allclose(cross, 0.0, atol=1e-12)  # same direction as the action
    assert np.all(np.sum(x * (g - p), axis=1) >= 0.0)
    edge = np.array([[1.0, 0.3], [-0.2, -1.0], [1.0, 1.0]])
    np.testing.assert_allclose(np.hypot(*(targets(edge, np.zeros((3, 2)), np.zeros((3, 2)))).T), 1.8)
    np.testing.assert_array_equal(targets(np.zeros((1, 2)), [[4.0, -1.0]], [[0.0, 0.0]]), [[4.0, -1.0]])


def test_disk_targets_reject_bad_input() -> None:
    targets = DiskTargets(radius=1.8)
    with pytest.raises(ValueError, match="square"):
        targets([[1.01, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]])
    with pytest.raises(ValueError, match="shape"):
        targets([[0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        targets([[math.nan, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]])
    for radius in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError, match="radius"):
            DiskTargets(radius=radius)


def test_worker_inputs_hold_the_observation_and_the_scaled_offset() -> None:
    obs = np.array([[0.1, -0.2, 0.3, -0.4], [0.0, 0.5, -1.0, 1.0]])
    p, g = np.array([[2.0, 0.0], [5.0, 1.0]]), np.array([[3.8, 0.0], [4.1, 1.9]])
    inputs = worker_inputs(obs, p, g, 1.8)
    np.testing.assert_array_equal(inputs[:, :4], obs)
    np.testing.assert_allclose(inputs[:, 4:], [[1.0, 0.0], [-0.5, 0.5]])
    with pytest.raises(ValueError, match="shape"):
        worker_inputs(obs, p[:1], g, 1.8)


# --------------------------------------------------------------------------
# Worker reward (D25)
# --------------------------------------------------------------------------


def _one_step(
    config: EnvConfig, p: list[float], v: list[float], action: list[float]
) -> tuple[np.ndarray, np.ndarray, Any]:
    env = NavigationEnv(config, 1)
    env.reset(positions=[p], velocities=[v])
    out = env.step([action])
    return np.array([p]), env.obs_map.positions(out.final_obs), out


def test_worker_reward_of_free_motion_toward_the_target() -> None:
    config = _config()
    p, p_next, out = _one_step(config, [2.0, 0.0], [0.0, 0.0], [1.0, 0.0])
    np.testing.assert_allclose(p_next, [[2.025, 0.0]], atol=1e-12)
    reward = worker_reward(config, p, p_next, np.array([[3.0, 0.0]]), out.info)
    np.testing.assert_allclose(reward.progress, [10.0 * (1.0 - 0.975)], atol=1e-10)
    np.testing.assert_allclose(reward.contact, [0.0])
    np.testing.assert_allclose(reward.effort, [-0.01], atol=1e-12)
    np.testing.assert_allclose(reward.total, [0.25 - 0.01], atol=1e-10)


def test_worker_reward_of_a_head_on_impact() -> None:
    """D25 with D14: the Worker pays the environment's contact penalty, at the same price."""
    config = _config()
    p, p_next, out = _one_step(config, [2.0, -1.95], [0.0, -1.2], [0.0, 0.0])
    np.testing.assert_allclose(out.info.impulse, [0.64], atol=1e-12)
    np.testing.assert_allclose(p_next, [[2.0, -2.0]], atol=1e-12)
    reward = worker_reward(config, p, p_next, np.array([[2.0, -1.0]]), out.info)
    np.testing.assert_allclose(reward.progress, [10.0 * (0.95 - 1.0)], atol=1e-10)
    np.testing.assert_allclose(reward.contact, [-50.0 * 0.64 - 1.0], atol=1e-10)
    np.testing.assert_allclose(reward.total, [-0.5 - 33.0], atol=1e-10)


@pytest.mark.parametrize("layout", ["slalom", "tunnel"])
def test_the_worker_reward_has_no_term_of_the_task(layout: str) -> None:
    """D25: apart from the progress to its target, the Worker's reward is the environment's contact and effort terms."""
    config = _config(layout, task={"horizon": 120}, physics={"d_bar": 1.0})
    env = NavigationEnv(config, 16, seed=3)
    rng = np.random.default_rng(3)
    obs = env.reset()
    seen_contact = seen_success = False
    r = config.reward
    for _ in range(300):
        p = env.obs_map.positions(obs)
        g = p + rng.uniform(-1.8, 1.8, p.shape)
        action = np.clip(rng.normal([0.9, 0.0], 0.6, (16, 2)), -1.0, 1.0)
        out = env.step(action)
        p_next = env.obs_map.positions(out.final_obs)
        reward = worker_reward(config, p, p_next, g, out.info)
        task = -r.time_penalty + r.success_bonus * out.terminated
        np.testing.assert_allclose(reward.contact + reward.effort, out.info.reward_unshaped - task, atol=1e-9)
        np.testing.assert_allclose(
            reward.progress, r.progress_coef * (np.hypot(*(p - g).T) - np.hypot(*(p_next - g).T)), atol=1e-12
        )
        assert np.all(np.abs(reward.progress) <= r.progress_coef * config.physics.v_max * config.physics.dt + 1e-9)
        seen_contact |= bool(out.info.contact.any())
        seen_success |= bool(out.terminated.any())
        obs = out.obs
    assert seen_contact if layout == "slalom" else seen_success


@pytest.mark.parametrize("layout", ["slalom", "tunnel"])
def test_shipped_configs_meet_the_contact_criterion_for_the_worker(layout: str) -> None:
    """D25: one saved step is worth at most c_p v_max T_s + c_e to the Worker, so c_n must exceed that / (a_max T_s)."""
    config = _config(layout)
    r, p = config.reward, config.physics
    threshold = (r.progress_coef * p.v_max * p.dt + r.effort_coef) / (p.a_max * p.dt)
    assert threshold == pytest.approx(4.84)
    assert r.contact_impulse_coef >= threshold


# --------------------------------------------------------------------------
# The loop: segments, Manager transitions, Worker transitions
# --------------------------------------------------------------------------


def test_the_manager_decides_at_the_start_and_after_every_segment() -> None:
    env, hierarchy, state, rollout, after = _run(_config(), batch=3, steps=35)
    assert rollout.num_samples == 105 and env.counter.env_steps == 105
    assert len(rollout.segments) == 9 and env.counter.manager_decisions == 12  # 3 at the start, then 3 per agent
    np.testing.assert_array_equal(np.sort(rollout.segments.steps), np.full(9, 10))
    np.testing.assert_array_equal(after.segment_steps, [5, 5, 5])
    np.testing.assert_array_equal(state.segment_steps, [0, 0, 0])
    assert not rollout.done.any()


def test_segment_rewards_discounts_and_bootstraps() -> None:
    """D27: reward sum_k gamma^k r_k over the segment, discount gamma^tau, next value of the next segment's start."""
    env, hierarchy, state, rollout, after = _run(_config(task={"horizon": 23}), batch=2, steps=50)
    seg = rollout.segments
    expected = _reference_segments(rollout.rewards, rollout.done, state.segment_steps, H, GAMMA)
    for b in range(2):
        mine = np.flatnonzero(seg.env == b)
        closed = expected[b]
        assert len(mine) == len(closed)
        np.testing.assert_array_equal(seg.steps[mine], [length for _, length, _ in closed])
        np.testing.assert_allclose(seg.rewards[mine], [total for _, _, total in closed], rtol=1e-12)
        np.testing.assert_allclose(seg.discounts[mine], GAMMA ** seg.steps[mine], rtol=1e-12)
        for k, (t, _, _) in enumerate(closed):
            i = mine[k]
            assert seg.truncated[i] == rollout.truncated[t, b] and seg.terminated[i] == rollout.terminated[t, b]
            if seg.truncated[i]:
                assert seg.next_values[i] == pytest.approx(_manager_value(rollout.final_obs[t, b][None])[0])
            elif k + 1 < len(mine):
                assert seg.next_values[i] == seg.values[mine[k + 1]]  # the next segment starts where this one ended
            else:
                assert seg.next_values[i] == after.segment_values[b]  # the segment in flight
        np.testing.assert_allclose(seg.values[mine], _manager_value(seg.obs[mine]))
    assert seg.truncated.any() and not seg.terminated.any()
    assert sorted(seg.steps[seg.truncated]) == [3, 3, 3, 3]  # episodes of 23 steps: segments of 10, 10, 3


def test_a_segment_that_reaches_the_goal_is_terminal_for_the_manager_only() -> None:
    config = _config()
    forward = LearnedWorker(policy=lambda w: (np.tile([1.0, 0.0], (len(w), 1)), np.zeros(len(w))), value=_worker_value)
    env, hierarchy, state, rollout, after = _run(config, batch=1, steps=3, worker=forward, positions=[[9.95, 0.0]])
    assert rollout.terminated[:, 0].tolist() == [False, True, False]
    seg = rollout.segments
    assert len(seg) == 1 and seg.terminated[0] and not seg.truncated[0] and seg.next_values[0] == 0.0
    assert seg.steps[0] == 2 and seg.discounts[0] == pytest.approx(GAMMA**2)
    worker = rollout.worker
    assert worker is not None and not worker.terminated.any()  # D25: no terminal state for the Worker
    assert worker.truncated[:, 0].tolist() == [False, True, False]
    start_obs = seg.obs[:1]  # the target set at the start: p + R m(x)
    target = env.obs_map.positions(start_obs) + square_to_disk(seg.actions[:1], 1.8)
    np.testing.assert_allclose(rollout.targets[:2, 0], np.vstack([target, target]), atol=1e-15)
    final = rollout.final_obs[1]
    expected = _worker_value(worker_inputs(final, env.obs_map.positions(final), target, 1.8))
    assert worker.next_values[1, 0] == pytest.approx(expected[0])


def test_segments_in_flight_continue_in_the_next_rollout() -> None:
    """D27: a segment in flight keeps the decision's log-probability and value, though the policy changed since."""
    config = _config()
    env = NavigationEnv(config, 1, seed=2)
    hierarchy = _hierarchy(config)
    manager = _random_policy(5)
    offset = [0.0]

    def value(obs: np.ndarray) -> np.ndarray:
        return _manager_value(obs) + offset[0]

    state = start_segments(env, hierarchy, manager, value)
    first, middle = collect_hierarchical_rollout(env, state, 15, hierarchy, manager, value, LEARNED)
    offset[0] = 1000.0  # an update between the rollouts
    second, _ = collect_hierarchical_rollout(env, middle, 15, hierarchy, manager, value, LEARNED)
    assert first.segments.steps.tolist() == [10] and middle.segment_steps.tolist() == [5]
    assert abs(second.segments.values[1] - _manager_value(second.segments.obs[1:2])[0] - 1000.0) < 1e-9
    straddling = 0
    assert second.segments.steps[straddling] == 10
    rewards = np.concatenate([first.rewards[10:, 0], second.rewards[:5, 0]])
    assert second.segments.rewards[straddling] == pytest.approx(np.sum(GAMMA ** np.arange(10) * rewards), rel=1e-12)
    np.testing.assert_array_equal(second.segments.obs[straddling], middle.segment_obs[0])
    np.testing.assert_array_equal(second.segments.actions[straddling], middle.segment_actions[0])
    assert second.segments.log_probs[straddling] == middle.segment_log_probs[0]
    assert second.segments.values[straddling] == middle.segment_values[0]
    assert abs(middle.segment_values[0] - _manager_value(middle.segment_obs)[0]) < 1e-9  # the old value function
    assert first.segments.next_values[0] == middle.segment_values[0]


def test_a_rollout_leaves_its_input_state_unchanged() -> None:
    config = _config(task={"horizon": 13}, physics={"d_bar": 1.0})
    env = NavigationEnv(config, 3, seed=4)
    hierarchy = _hierarchy(config)
    manager = _random_policy(8)
    state = start_segments(env, hierarchy, manager, _manager_value)
    _, middle = collect_hierarchical_rollout(env, state, 14, hierarchy, manager, _manager_value, LEARNED)
    snapshot = {name: np.copy(getattr(middle, name)) for name in HierarchyState.__dataclass_fields__}
    collect_hierarchical_rollout(env, middle, 23, hierarchy, manager, _manager_value, LEARNED)
    for name, before in snapshot.items():
        np.testing.assert_array_equal(getattr(middle, name), before, err_msg=name)


def test_the_manager_does_not_replan_on_contact() -> None:
    """D27: a wall contact in the middle of a segment ends neither the segment nor its target."""
    config = _config()
    down = lambda obs: (np.tile([0.0, -1.0], (len(obs), 1)), np.zeros(len(obs)))  # noqa: E731 - a target in the floor
    env, hierarchy, state, rollout, after = _run(config, batch=1, steps=30, manager=down, positions=[[1.0, -1.5]])
    contact = rollout.worker_reward.contact[:, 0] < 0.0
    assert contact[3:9].any() and contact[13:19].any()
    assert rollout.segments.steps.tolist() == [10, 10, 10]
    for begin in (0, 10, 20):
        block = rollout.targets[begin : begin + 10, 0]
        np.testing.assert_array_equal(block, np.broadcast_to(block[0], block.shape))


def test_targets_stay_fixed_in_the_plane_during_a_segment() -> None:
    """D24: g = p + R m(x) with p the position at the segment's start, held until the segment ends."""
    env, hierarchy, state, rollout, after = _run(
        _config(task={"horizon": 27}, physics={"d_bar": 1.0}), batch=4, steps=60, manager=_random_policy(1)
    )
    worker = rollout.worker
    assert worker is not None
    obs_map, radius = env.obs_map, hierarchy.targets.radius
    g = obs_map.positions(worker.obs[..., :4]) + radius * worker.obs[..., 4:]
    np.testing.assert_allclose(g, rollout.targets, atol=1e-12)
    seg = rollout.segments
    starts = obs_map.positions(seg.obs) + square_to_disk(seg.actions, radius)
    expected = _reference_segments(rollout.rewards, rollout.done, state.segment_steps, H, GAMMA)
    for b in range(4):
        mine = np.flatnonzero(seg.env == b)
        begin = 0
        for k, (t, _, _) in enumerate(expected[b]):
            block = rollout.targets[begin : t + 1, b]
            np.testing.assert_array_equal(block, np.broadcast_to(block[0], block.shape))
            np.testing.assert_array_equal(block[0], starts[mine[k]])
            begin = t + 1
    end = np.hypot(*(obs_map.positions(rollout.final_obs) - g).transpose(2, 0, 1))
    closing = [(t, b) for b in range(4) for t, _, _ in expected[b]]
    np.testing.assert_allclose(sorted(seg.end_distances), sorted(end[t, b] for t, b in closing), atol=1e-12)


def test_the_worker_bootstraps_at_every_episode_end() -> None:
    """D25: the Worker's transitions are never terminal; at an episode's end it bootstraps under its old target."""
    config = _config(task={"horizon": 17}, physics={"d_bar": 0.5})
    env, hierarchy, state, rollout, after = _run(config, batch=3, steps=40, manager=_random_policy(2))
    worker = rollout.worker
    assert worker is not None
    assert not worker.terminated.any()
    np.testing.assert_array_equal(worker.truncated, rollout.done)
    np.testing.assert_allclose(worker.values, _worker_value(worker.obs.reshape(-1, 6)).reshape(40, 3))
    obs_map = env.obs_map
    g = obs_map.positions(worker.obs[..., :4]) + 1.8 * worker.obs[..., 4:]
    for t in range(40):
        for b in range(3):
            if rollout.done[t, b]:
                final = rollout.final_obs[t, b][None]
                bootstrap = _worker_value(worker_inputs(final, obs_map.positions(final), g[t, b][None], 1.8))
                assert worker.next_values[t, b] == pytest.approx(bootstrap[0])
            elif t + 1 < 40:
                assert worker.next_values[t, b] == worker.values[t + 1, b]
    last = _worker_value(worker_inputs(after.obs, obs_map.positions(after.obs), after.targets, 1.8))
    live = ~rollout.done[-1]
    np.testing.assert_allclose(worker.next_values[-1, live], last[live])
    assert rollout.done.any()


def test_the_worker_rewards_of_the_rollout() -> None:
    config = _config("slalom", task={"horizon": 30})
    env, hierarchy, state, rollout, after = _run(config, batch=4, steps=60, manager=_random_policy(4))
    worker = rollout.worker
    assert worker is not None
    np.testing.assert_allclose(worker.rewards, rollout.worker_reward.total, rtol=1e-12)
    obs_map = env.obs_map
    g = obs_map.positions(worker.obs[..., :4]) + 1.8 * worker.obs[..., 4:]
    p, p_next = obs_map.positions(rollout.obs), obs_map.positions(rollout.final_obs)
    progress = 10.0 * (np.linalg.norm(p - g, axis=-1) - np.linalg.norm(p_next - g, axis=-1))
    np.testing.assert_allclose(rollout.worker_reward.progress, progress, atol=1e-9)
    assert np.all(rollout.worker_reward.contact <= 0.0) and np.all(rollout.worker_reward.effort <= 0.0)


def test_rollouts_are_reproducible() -> None:
    results = []
    for _ in range(2):
        env, hierarchy, state, rollout, after = _run(
            _config(physics={"d_bar": 1.0}, task={"horizon": 25}), batch=2, steps=40, seed=7, manager=_random_policy(3)
        )
        results.append((rollout, after))
    (a, after_a), (b, after_b) = results
    for name in ("obs", "actions", "log_probs", "values", "rewards", "env", "steps", "next_values"):
        np.testing.assert_array_equal(getattr(a.segments, name), getattr(b.segments, name))
    assert a.worker is not None and b.worker is not None
    np.testing.assert_array_equal(a.worker.obs, b.worker.obs)
    np.testing.assert_array_equal(a.worker.rewards, b.worker.rewards)
    np.testing.assert_array_equal(after_a.targets, after_b.targets)


def test_segment_gae_runs_agent_by_agent() -> None:
    env, hierarchy, state, rollout, after = _run(
        _config(task={"horizon": 19}, physics={"d_bar": 1.0}), batch=3, steps=70, manager=_random_policy(6)
    )
    seg = rollout.segments
    advantages, returns = segment_gae(seg, gae_lambda=0.95)
    assert advantages.shape == returns.shape == (len(seg),)
    for b in range(3):
        mine = np.flatnonzero(seg.env == b)
        column = [a[mine][:, None] for a in (seg.rewards, seg.values, seg.next_values)]
        flags = [seg.terminated[mine][:, None], seg.done[mine][:, None]]
        adv, ret = compute_gae(*column, *flags, discount=seg.discounts[mine][:, None], gae_lambda=0.95)
        np.testing.assert_allclose(advantages[mine], adv[:, 0], rtol=1e-12)
        np.testing.assert_allclose(returns[mine], ret[:, 0], rtol=1e-12)
    with pytest.raises(ValueError, match="gae_lambda"):
        segment_gae(seg, gae_lambda=1.5)


def test_invalid_arguments_are_rejected() -> None:
    config = _config()
    hierarchy = _hierarchy(config)
    env = NavigationEnv(config, 2)
    state = start_segments(env, hierarchy, _manager_policy, _manager_value)
    with pytest.raises(ValueError, match="num_steps"):
        collect_hierarchical_rollout(env, state, 0, hierarchy, _manager_policy, _manager_value, LEARNED)
    with pytest.raises(ValueError, match="autoreset"):
        collect_hierarchical_rollout(
            NavigationEnv(config, 2, autoreset=False), state, 5, hierarchy, _manager_policy, _manager_value, LEARNED
        )
    bad_worker = LearnedWorker(policy=lambda w: (np.zeros((len(w), 3)), np.zeros(len(w))), value=_worker_value)
    with pytest.raises(ValueError, match="actions"):
        collect_hierarchical_rollout(env, state, 5, hierarchy, _manager_policy, _manager_value, bad_worker)
    with pytest.raises(ValueError, match="values"):  # the Manager's values are first needed when a segment ends
        collect_hierarchical_rollout(env, state, 12, hierarchy, _manager_policy, lambda o: np.zeros(1), LEARNED)
    with pytest.raises(ValueError, match="agents"):
        other = NavigationEnv(config, 3)
        collect_hierarchical_rollout(other, state, 5, hierarchy, _manager_policy, _manager_value, LEARNED)
    for steps, discount in ((0, 0.99), (10, 0.0), (10, 1.5)):
        with pytest.raises(ValueError):
            Hierarchy(DiskTargets(1.8), steps, discount, env.obs_map)


# --------------------------------------------------------------------------
# Interchangeable Worker: commanded accelerations
# --------------------------------------------------------------------------


def _tracking_command(obs_map: ObservationMap) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    def command(obs: np.ndarray, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        u = 1.2 * (g - p) - 0.8 * obs_map.velocities(obs)
        norm = np.maximum(np.hypot(*u.T), 1e-12)
        return u * np.minimum(1.0, 2.4 / norm)[:, None]  # inside U, so the saturation never acts

    return command


def test_a_command_worker_drives_the_inputs() -> None:
    config = _config(task={"horizon": 21})
    obs_map = ObservationMap.from_config(config)
    resets: list[np.ndarray] = []
    worker = CommandWorker(command=_tracking_command(obs_map), reset=lambda mask: resets.append(mask.copy()))
    starts = [[9.0, 0.0], [0.0, 0.0]]  # the first agent reaches the goal, the second runs into the horizon
    env, hierarchy, state, rollout, after = _run(config, batch=2, steps=45, worker=worker, positions=starts)
    assert rollout.worker is None
    expected = _reference_segments(rollout.rewards, rollout.done, state.segment_steps, H, GAMMA)
    assert len(rollout.segments) == sum(len(e) for e in expected)
    done_steps = np.flatnonzero(rollout.done.any(axis=1))
    assert len(resets) == len(done_steps)
    for mask, t in zip(resets, done_steps, strict=True):
        np.testing.assert_array_equal(mask, rollout.done[t])
    assert any(mask.sum() == 1 for mask in resets) and rollout.terminated[:, 0].any()
    # The commands reached the environment as accelerations: the effort term is that of the commands.
    p = obs_map.positions(rollout.obs)
    u = np.stack([_tracking_command(obs_map)(rollout.obs[t], p[t], rollout.targets[t]) for t in range(45)])
    effort = -0.01 * np.sum(u**2, axis=-1) / 2.5**2
    np.testing.assert_allclose(rollout.worker_reward.effort, effort, atol=1e-12)


def test_the_worker_pays_effort_on_the_saturated_input() -> None:
    """D25: the effort term is that of the input after the saturation onto U, at most c_e per step."""
    config = _config()
    strong = CommandWorker(command=lambda obs, p, g: np.tile([5.0, 0.0], (len(obs), 1)))
    env, hierarchy, state, rollout, after = _run(config, batch=2, steps=12, worker=strong)
    np.testing.assert_allclose(rollout.worker_reward.effort, -0.01, rtol=1e-12)


# --------------------------------------------------------------------------
# The deterministic controller of the evaluation (D29)
# --------------------------------------------------------------------------


STARTS = np.array([[0.0, -1.0], [1.0, 0.5], [2.0, 1.0]])


def test_the_controller_replans_as_in_training() -> None:
    """Without disturbance the evaluation's controller reproduces the episodes of the training loop."""
    config = _config()
    env, hierarchy, state, rollout, after = _run(config, batch=3, steps=130, positions=STARTS)
    first = {}
    for episode in rollout.episodes:
        first.setdefault(episode.env, episode)
    assert sorted(first) == [0, 1, 2] and all(ep.success for ep in first.values())
    controller = HierarchicalController(hierarchy, lambda o: _manager_policy(o)[0], lambda w: _worker_policy(w)[0])
    assert not controller.inputs
    result = evaluate(controller, config, STARTS, seed=99)
    for b in range(3):
        mine, theirs = result.episodes[b], first[b]
        assert (mine.length, mine.success) == (theirs.length, theirs.success)
        assert mine.unshaped_return == pytest.approx(theirs.unshaped_return, abs=1e-9)


def test_the_controller_starts_afresh_for_every_evaluation() -> None:
    config = _config(physics={"d_bar": 1.0})
    env, hierarchy, state, rollout, after = _run(config, batch=1, steps=1)
    controller = HierarchicalController(hierarchy, lambda o: _random_policy(0)(o)[0], lambda w: _worker_policy(w)[0])
    first = evaluate(controller, config, STARTS, seed=5)
    controller(np.zeros((3, 4)))  # a stray call in between
    second = evaluate(controller, config, STARTS, seed=5)
    fresh = HierarchicalController(hierarchy, lambda o: _random_policy(0)(o)[0], lambda w: _worker_policy(w)[0])
    assert first.details() == second.details() == evaluate(fresh, config, STARTS, seed=5).details()


def test_the_controller_resets_a_command_worker_and_declares_its_inputs() -> None:
    config = _config()
    obs_map = ObservationMap.from_config(config)
    resets: list[np.ndarray] = []
    worker = CommandWorker(command=_tracking_command(obs_map), reset=lambda mask: resets.append(mask.copy()))
    env, hierarchy, state, rollout, after = _run(config, batch=1, steps=1)
    controller = HierarchicalController(hierarchy, lambda o: _manager_policy(o)[0], worker)
    by_default = evaluate(controller, config, STARTS, seed=1)  # inputs taken from the controller
    assert [mask.tolist() for mask in resets] == [[True, True, True]]
    assert by_default.details() == evaluate(controller, config, STARTS, seed=1, inputs=True).details()
    with pytest.raises(ValueError, match="contradicts"):
        evaluate(controller, config, STARTS, seed=1, inputs=False)


def test_the_controller_with_a_command_worker() -> None:
    config = _config()
    obs_map = ObservationMap.from_config(config)
    worker = CommandWorker(command=_tracking_command(obs_map))
    env, hierarchy, state, rollout, after = _run(config, batch=3, steps=200, positions=STARTS, worker=worker)
    first = {}
    for episode in rollout.episodes:
        first.setdefault(episode.env, episode)
    assert sorted(first) == [0, 1, 2]
    controller = HierarchicalController(hierarchy, lambda o: _manager_policy(o)[0], worker)
    assert controller.inputs
    result = evaluate(controller, config, STARTS, seed=99, inputs=True)
    for b in range(3):
        mine, theirs = result.episodes[b], first[b]
        assert (mine.length, mine.success, mine.saturation_steps) == (theirs.length, theirs.success, 0)
        assert mine.unshaped_return == pytest.approx(theirs.unshaped_return, abs=1e-9)
