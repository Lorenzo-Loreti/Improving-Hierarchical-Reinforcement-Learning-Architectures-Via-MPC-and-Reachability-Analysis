"""Tests of the common Manager/Worker loop (decision log D6, D24-D27, D29, D37; rows AR2-AR5, R2, X1)."""

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
from hrlmpc.geometry import Layout, ray_distances
from hrlmpc.hierarchy import (
    CommandWorker,
    DiskTargets,
    HierarchicalController,
    HierarchicalRollout,
    Hierarchy,
    HierarchyState,
    LearnedWorker,
    MapFeatures,
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


C_P = 10.0
"""The Worker's progress coefficient of the shipped configurations (D25, D66): the shaped reward's (D13)."""


def _hierarchy(config: EnvConfig, segment_steps: int = H, progress_coef: float = C_P) -> Hierarchy:
    return disk_hierarchy(
        config, segment_steps=segment_steps, reach_factor=1.5, discount=GAMMA, progress_coef=progress_coef
    )


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
    reward = worker_reward(config, p, p_next, np.array([[3.0, 0.0]]), out.info, progress_coef=C_P)
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
    reward = worker_reward(config, p, p_next, np.array([[2.0, -1.0]]), out.info, progress_coef=C_P)
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
        reward = worker_reward(config, p, p_next, g, out.info, progress_coef=C_P)
        task = -r.time_penalty + r.success_bonus * out.terminated
        np.testing.assert_allclose(reward.contact + reward.effort, out.info.reward_unshaped - task, atol=1e-9)
        np.testing.assert_allclose(
            reward.progress, C_P * (np.hypot(*(p - g).T) - np.hypot(*(p_next - g).T)), atol=1e-12
        )
        assert np.all(np.abs(reward.progress) <= C_P * config.physics.v_max * config.physics.dt + 1e-9)
        seen_contact |= bool(out.info.contact.any())
        seen_success |= bool(out.terminated.any())
        obs = out.obs
    assert seen_contact if layout == "slalom" else seen_success


@pytest.mark.parametrize("layout", sorted(path.stem for path in CONFIGS.glob("*.yaml")))
def test_shipped_configs_meet_the_contact_criterion_for_the_worker(layout: str) -> None:
    """D25: one saved step is worth at most c_p v_max T_s + c_e to the Worker, so c_n must exceed that / (a_max T_s);
    with c_p the hierarchy's price (D66), in the sparse files too, whose own ``progress_coef`` is 0."""
    config = _config(layout)
    r, p = config.reward, config.physics
    threshold = (C_P * p.v_max * p.dt + r.effort_coef) / (p.a_max * p.dt)
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


def test_the_workers_progress_is_priced_by_the_hierarchy_not_by_the_environment() -> None:
    """D66 (the realization of D51): under the sparse reward the environment's ``progress_coef`` is 0, and the
    Worker keeps its dense reward of D25 at the hierarchy's price. Before the fix the Worker of every sparse run
    was paid no progress at all."""
    config = _config("slalom", task={"horizon": 30}, reward={"progress_coef": 0.0})
    assert config.reward.progress_coef == 0.0
    env, hierarchy, state, rollout, after = _run(config, batch=4, steps=60, manager=_random_policy(4))
    assert hierarchy.progress_coef == C_P
    obs_map = env.obs_map
    g = rollout.targets
    p, p_next = obs_map.positions(rollout.obs), obs_map.positions(rollout.final_obs)
    progress = C_P * (np.linalg.norm(p - g, axis=-1) - np.linalg.norm(p_next - g, axis=-1))
    np.testing.assert_allclose(rollout.worker_reward.progress, progress, atol=1e-9)
    assert np.abs(progress).max() > 0.1  # the Worker is paid for its progress
    assert rollout.worker is not None
    np.testing.assert_allclose(rollout.worker.rewards, rollout.worker_reward.total, rtol=1e-12)
    p0, p1, out = _one_step(config, [2.0, 0.0], [0.0, 0.0], [1.0, 0.0])
    reward = worker_reward(config, p0, p1, np.array([[3.0, 0.0]]), out.info, progress_coef=3.0)
    np.testing.assert_allclose(reward.progress, [3.0 * (1.0 - 0.975)], atol=1e-10)  # the given price, not 0
    with pytest.raises(ValueError, match="progress_coef"):
        _hierarchy(config, progress_coef=0.0)
    with pytest.raises(ValueError, match="progress_coef"):
        worker_reward(config, p0, p1, np.array([[3.0, 0.0]]), out.info, progress_coef=-1.0)


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
    for steps, discount, price in ((0, 0.99, C_P), (10, 0.0, C_P), (10, 1.5, C_P), (10, 0.99, 0.0)):
        with pytest.raises(ValueError):
            Hierarchy(DiskTargets(1.8), steps, discount, env.obs_map, price)


# --------------------------------------------------------------------------
# Interchangeable Worker: commanded accelerations
# --------------------------------------------------------------------------


def _tracking_command(
    obs_map: ObservationMap,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    def command(obs: np.ndarray, p: np.ndarray, g: np.ndarray, steps_left: np.ndarray) -> np.ndarray:
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
    u = np.stack([_tracking_command(obs_map)(rollout.obs[t], p[t], rollout.targets[t], None) for t in range(45)])
    effort = -0.01 * np.sum(u**2, axis=-1) / 2.5**2
    np.testing.assert_allclose(rollout.worker_reward.effort, effort, atol=1e-12)


def test_the_worker_pays_effort_on_the_saturated_input() -> None:
    """D25: the effort term is that of the input after the saturation onto U, at most c_e per step."""
    config = _config()
    strong = CommandWorker(command=lambda obs, p, g, left: np.tile([5.0, 0.0], (len(obs), 1)))
    env, hierarchy, state, rollout, after = _run(config, batch=2, steps=12, worker=strong)
    np.testing.assert_allclose(rollout.worker_reward.effort, -0.01, rtol=1e-12)


class _CountingWorker:
    """A commanding Worker that records the steps left it is given, and tracks its targets."""

    def __init__(self, obs_map: ObservationMap) -> None:
        self.track = _tracking_command(obs_map)
        self.calls: list[np.ndarray] = []

    def command(self, obs: np.ndarray, p: np.ndarray, g: np.ndarray, steps_left: np.ndarray) -> np.ndarray:
        assert steps_left.dtype == np.int64 and steps_left.shape == (len(obs),)
        self.calls.append(steps_left.copy())
        return self.track(obs, p, g, steps_left)


def _expected_steps_left(done: np.ndarray, first: np.ndarray, segment_steps: int) -> np.ndarray:
    """``H - j`` at every step, ``j`` the steps since the agent's last decision (after ``H`` steps or at an episode's
    end)."""
    expected = np.zeros(done.shape, dtype=np.int64)
    j = first.copy()
    for t in range(done.shape[0]):
        expected[t] = segment_steps - j
        j += 1
        j[(j == segment_steps) | done[t]] = 0
    return expected


def test_a_command_worker_gets_the_steps_left_in_each_segment() -> None:
    """D37: the MPC Worker plans to pass the target when the segment ends, so it gets H - j at step j, also
    across rollouts and after an episode that ends in the middle of a segment."""
    config = _config(task={"horizon": 15})  # episodes of 15 steps: segments of 10 and 5 steps
    obs_map = ObservationMap.from_config(config)
    recorder = _CountingWorker(obs_map)
    worker = CommandWorker(command=recorder.command)
    env = NavigationEnv(config, 2, seed=0)
    hierarchy = _hierarchy(config)
    state = start_segments(env, hierarchy, _manager_policy, _manager_value, positions=[[0.0, 0.0], [9.5, 0.0]])
    rollouts = []
    for steps in (7, 26):  # the first rollout stops in the middle of a segment
        rollout, after = collect_hierarchical_rollout(env, state, steps, hierarchy, _manager_policy, _manager_value,
                                                      worker)
        rollouts.append((state, rollout))
        state = after
    calls = np.array(recorder.calls)
    expected = np.concatenate(
        [_expected_steps_left(r.done, s.segment_steps, H) for s, r in rollouts]
    )
    np.testing.assert_array_equal(calls, expected)
    assert set(np.unique(calls)) == set(range(1, H + 1))
    assert rollouts[1][1].terminated[:, 1].any() and rollouts[1][1].truncated[:, 0].any()


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


def test_the_controller_records_how_far_each_segment_ends_from_its_target() -> None:
    """D40: at every decision after the first, the distances of the agents from the targets of the segments that
    ended, with the call at which they ended; a reset forgets them."""
    config = _config()
    obs_map = ObservationMap.from_config(config)
    hierarchy = _hierarchy(config, segment_steps=3)
    controller = HierarchicalController(hierarchy, lambda o: _manager_policy(o)[0],
                                        CommandWorker(command=_tracking_command(obs_map)))
    rng = np.random.default_rng(7)
    positions = rng.uniform([0.0, -1.0], [8.0, 1.0], size=(8, 2, 2))
    targets = []
    for t, p in enumerate(positions):
        obs = obs_map.observe(p, np.zeros((2, 2)))
        if t % 3 == 0:
            targets.append(hierarchy.targets(_manager_policy(obs)[0], p, np.zeros((2, 2))))
        controller(obs)
    assert [t for t, _ in controller.segment_ends] == [3, 6]
    for (t, distances), target in zip(controller.segment_ends, targets[:-1], strict=True):
        np.testing.assert_allclose(distances, np.linalg.norm(positions[t] - target, axis=1))
    controller.reset(2)
    assert controller.segment_ends == []


def test_the_controller_exposes_the_targets_of_the_current_segments() -> None:
    """The targets in force, ``(B, 2)``, for the replays of :mod:`hrlmpc.trajectories`: empty before the first call
    and after a reset, those of the last decision otherwise (a copy)."""
    config = _config()
    obs_map = ObservationMap.from_config(config)
    hierarchy = _hierarchy(config, segment_steps=3)
    controller = HierarchicalController(hierarchy, lambda o: _manager_policy(o)[0], lambda w: _worker_policy(w)[0])
    assert controller.targets.shape == (0, 2)
    positions = np.array([[1.0, 0.0], [2.0, 0.5]])
    obs = obs_map.observe(positions, np.zeros((2, 2)))
    controller(obs)
    expected = hierarchy.targets(_manager_policy(obs)[0], positions, np.zeros((2, 2)))
    np.testing.assert_allclose(controller.targets, expected)
    seen = controller.targets
    seen[:] = 0.0  # a copy: the controller's targets are untouched
    controller(obs_map.observe(positions + 0.1, np.zeros((2, 2))))  # the second call of the segment keeps them
    np.testing.assert_allclose(controller.targets, expected)
    controller.reset(2)
    assert controller.targets.shape == (0, 2)


def test_the_controller_with_a_command_worker() -> None:
    config = _config()
    obs_map = ObservationMap.from_config(config)
    recorder = _CountingWorker(obs_map)
    worker = CommandWorker(command=recorder.command)
    env, hierarchy, state, rollout, after = _run(config, batch=3, steps=200, positions=STARTS, worker=worker)
    first = {}
    for episode in rollout.episodes:
        first.setdefault(episode.env, episode)
    assert sorted(first) == [0, 1, 2]
    training = np.array(recorder.calls)
    recorder.calls.clear()
    controller = HierarchicalController(hierarchy, lambda o: _manager_policy(o)[0], worker)
    assert controller.inputs
    result = evaluate(controller, config, STARTS, seed=99, inputs=True)
    for b in range(3):
        mine, theirs = result.episodes[b], first[b]
        assert (mine.length, mine.success, mine.saturation_steps) == (theirs.length, theirs.success, 0)
        assert mine.unshaped_return == pytest.approx(theirs.unshaped_return, abs=1e-9)
    evaluation = np.array(recorder.calls)  # every agent is stepped until the last episode ends
    np.testing.assert_array_equal(evaluation[:, 0], H - np.arange(len(evaluation)) % H)
    for b in range(3):  # within the first episode the steps left are those of training
        np.testing.assert_array_equal(evaluation[: first[b].length, b], training[: first[b].length, b])


# --------------------------------------------------------------------------
# The Worker's local map features (D75)
# --------------------------------------------------------------------------


def _map_hierarchy(config: EnvConfig, rays: int = 4, max_range: float = 2.0) -> Hierarchy:
    layout = Layout.from_config(config.geometry)
    return disk_hierarchy(config, segment_steps=H, reach_factor=1.5, discount=GAMMA, progress_coef=C_P,
                          map_features=MapFeatures.rays(layout, rays, max_range))


def test_map_features_are_the_scaled_ray_distances() -> None:
    """D75: ``K`` rays at equal angles from ``+x``; each feature is the distance to the first wall along its ray,
    clipped at the range and divided by it: 1 where nothing is within range, 0 against a wall."""
    config = _config("slalom")  # arena [-1, 11] x [-2, 2]; gate 1 blocks [4, 5] x [-2, 0.25] and [4, 5] x [1.75, 2]
    layout = Layout.from_config(config.geometry)
    features = MapFeatures.rays(layout, 4, 2.0)
    assert features.size == 4 and features.max_range == 2.0
    np.testing.assert_allclose(features.directions, [[1, 0], [0, 1], [-1, 0], [0, -1]], atol=1e-15)
    values = features([[2.0, 0.0], [4.5, 1.0], [-1.0, 0.0]])
    assert values.shape == (3, 4)
    np.testing.assert_allclose(values[0], [1.0, 1.0, 1.0, 1.0])  # the block 2 m east is at the range, walls 2 m away
    np.testing.assert_allclose(values[1], [1.0, 0.375, 1.0, 0.375])  # inside the gate: the blocks 0.75 m up and down
    np.testing.assert_allclose(values[2], [1.0, 1.0, 0.0, 1.0])  # against the west wall
    np.testing.assert_array_equal(ray_distances(layout, [[4.5, 1.0]], features.directions, 2.0), 2.0 * values[1:2])
    with pytest.raises(ValueError, match="rays"):
        MapFeatures.rays(layout, 0, 2.0)
    with pytest.raises(ValueError, match="range"):
        MapFeatures.rays(layout, 4, 0.0)


def test_the_hierarchy_appends_the_map_features_to_the_workers_input() -> None:
    """Without features the hierarchy's inputs are those of :func:`worker_inputs`, width 6; with ``K`` features
    the width is ``6 + K`` and the last ``K`` entries are the features at the positions."""
    config = _config("slalom")
    plain, mapped = _hierarchy(config), _map_hierarchy(config)
    assert plain.map_features is None and plain.worker_input_size == 6 and mapped.worker_input_size == 10
    obs = np.array([[0.1, -0.2, 0.3, -0.4], [0.0, 0.5, -1.0, 1.0]])
    p, g = np.array([[2.0, 0.0], [4.5, 1.0]]), np.array([[3.8, 0.0], [4.1, 1.9]])
    np.testing.assert_array_equal(plain.worker_inputs(obs, p, g), worker_inputs(obs, p, g, plain.targets.radius))
    inputs = mapped.worker_inputs(obs, p, g)
    assert inputs.shape == (2, 10)
    np.testing.assert_array_equal(inputs[:, :6], worker_inputs(obs, p, g, mapped.targets.radius))
    assert mapped.map_features is not None
    np.testing.assert_array_equal(inputs[:, 6:], mapped.map_features(p))


def test_the_rollout_and_the_controller_feed_the_map_features_to_the_worker() -> None:
    """The learned Worker's transitions and the evaluation's controller carry the ``6 + K`` inputs, with the
    features of the positions the Worker acted at; the Manager's inputs stay the observations."""
    config = _config("slalom")
    hierarchy = _map_hierarchy(config)
    seen: list[np.ndarray] = []

    def policy(inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seen.append(np.asarray(inputs).copy())
        assert inputs.shape[1] == 10
        x = np.clip(2.0 * inputs[:, 4:6] - 0.3 * inputs[:, 2:4], -1.0, 1.0)
        return x, -np.sum(x**2, axis=1)

    def value(inputs: np.ndarray) -> np.ndarray:
        return np.asarray(inputs)[:, :6] @ W_WEIGHTS + np.asarray(inputs)[:, 6:].sum(axis=1)

    worker = LearnedWorker(policy=policy, value=value)
    env = NavigationEnv(config, 3, seed=0)
    state = start_segments(env, hierarchy, _manager_policy, _manager_value, positions=STARTS)
    rollout, after = collect_hierarchical_rollout(env, state, 25, hierarchy, _manager_policy, _manager_value, worker)
    assert rollout.worker is not None and rollout.worker.obs.shape == (25, 3, 10)
    assert rollout.segments.obs.shape[1] == 4 and state.obs.shape == (3, 4)
    assert hierarchy.map_features is not None
    for t in range(25):
        positions = hierarchy.obs_map.positions(rollout.obs[t])
        np.testing.assert_allclose(rollout.worker.obs[t, :, 6:], hierarchy.map_features(positions))
        np.testing.assert_allclose(rollout.worker.obs[t], seen[t])
    assert np.any(rollout.worker.obs[:, :, 6:] < 1.0)  # some wall came within range
    seen.clear()
    controller = HierarchicalController(hierarchy, lambda o: _manager_policy(o)[0], lambda w: policy(w)[0])
    result = evaluate(controller, config, STARTS, seed=99)
    assert len(result.episodes) == 3 and seen and all(inputs.shape == (3, 10) for inputs in seen)
