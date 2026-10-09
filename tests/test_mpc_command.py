"""Tests of the MPC Worker's adapter to the common loop and of its statistics, without solvers (decision log D34)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from hrlmpc.config import load_env_config
from hrlmpc.env import ObservationMap
from hrlmpc.hierarchy import CommandWorker
from hrlmpc.mpc_command import (
    CANDIDATE,
    EMERGENCY,
    KEPT,
    SOLVER,
    STAT_KEYS,
    MPCCommand,
    WorkerStep,
    episode_mask,
    mpc_metrics,
)


class Recorder:
    """A Worker that records its calls: ``u = x_v + target``, outcome and tube ratio by slot."""

    def __init__(self, num_slots: int) -> None:
        self.num_slots = num_slots
        self.calls: list[tuple[np.ndarray, np.ndarray, int]] = []
        self.resets: list[int | None] = []

    def act(self, x: Any, target: Any, slot: int = 0) -> tuple[np.ndarray, WorkerStep]:
        x, target = np.asarray(x, dtype=np.float64), np.asarray(target, dtype=np.float64)
        self.calls.append((x, target, slot))
        info = WorkerStep(outcome=(SOLVER, CANDIDATE, KEPT, EMERGENCY)[slot % 4], solve_ms=1.0 + slot,
                          free_binaries=2 * slot, tube_ratio=float("nan") if slot == 0 else 0.1 * slot,
                          cost=1.0, cut_rounds=slot, plan=None, invalid_candidate=int(slot == 1),
                          solver_errors=int(slot == 3))
        return x[2:] + target, info

    def reset(self, slot: int | None = None) -> None:
        self.resets.append(slot)


def _obs_map() -> ObservationMap:
    return ObservationMap.from_config(load_env_config("configs/env/slalom.yaml"))


def test_the_command_steers_every_agent_with_its_own_slot() -> None:
    obs_map, worker = _obs_map(), Recorder(4)
    command = MPCCommand(worker, obs_map)
    p = np.array([[1.0, 0.5], [2.0, -0.5], [3.0, 1.5]])
    v = np.array([[0.1, 0.2], [-0.3, 0.4], [0.5, -0.6]])
    targets = p + 1.0
    u = command.command(obs_map.observe(p, v), p, targets)
    np.testing.assert_allclose(u, v + targets)
    assert [slot for _, _, slot in worker.calls] == [0, 1, 2]
    for (x, target, _), p_i, v_i, g_i in zip(worker.calls, p, v, targets, strict=True):
        np.testing.assert_allclose(x, np.concatenate([p_i, v_i]))  # the velocity read back from the observation
        np.testing.assert_array_equal(target, g_i)


def test_the_statistics_are_kept_per_step_and_agent_until_popped() -> None:
    obs_map, worker = _obs_map(), Recorder(4)
    command = MPCCommand(worker, obs_map)
    p, v = np.zeros((4, 2)), np.zeros((4, 2))
    for _ in range(3):
        command.command(obs_map.observe(p, v), p, p)
    stats = command.pop_stats()
    assert set(stats) == set(STAT_KEYS)
    assert all(values.shape == (3, 4) for values in stats.values())
    np.testing.assert_array_equal(stats["outcome"][0], [SOLVER, CANDIDATE, KEPT, EMERGENCY])
    np.testing.assert_array_equal(stats["free_binaries"][2], [0, 2, 4, 6])
    np.testing.assert_array_equal(stats["invalid_candidate"][1], [0, 1, 0, 0])
    np.testing.assert_array_equal(stats["solver_errors"][1], [0, 0, 0, 1])
    assert np.isnan(stats["tube_ratio"][:, 0]).all()
    assert all(values.size == 0 for values in command.pop_stats().values())


def test_too_many_agents_are_refused() -> None:
    obs_map = _obs_map()
    command = MPCCommand(Recorder(2), obs_map)
    p = np.zeros((3, 2))
    with pytest.raises(ValueError, match="slots"):
        command.command(obs_map.observe(p, p), p, p)


def test_a_reset_reaches_the_slots_of_the_finished_agents_only() -> None:
    worker = Recorder(4)
    adapter = MPCCommand(worker, _obs_map()).as_worker()
    assert isinstance(adapter, CommandWorker) and adapter.reset is not None
    adapter.reset(np.array([True, False, True, False]))
    adapter.reset(np.zeros(4, dtype=bool))
    assert worker.resets == [0, 2]


def _stats(outcome: list[list[int]], ratio: list[list[float]]) -> dict[str, np.ndarray]:
    out = np.array(outcome, dtype=np.float64)
    return {
        "outcome": out,
        "solve_ms": np.arange(out.size, dtype=np.float64).reshape(out.shape),
        "free_binaries": np.full(out.shape, 3.0),
        "tube_ratio": np.array(ratio, dtype=np.float64),
        "cut_rounds": np.ones(out.shape),
        "invalid_candidate": (out == CANDIDATE).astype(np.float64),
        "solver_errors": (out == EMERGENCY).astype(np.float64),
    }


def test_the_metrics_summarise_the_counted_steps() -> None:
    nan = float("nan")
    stats = _stats([[SOLVER, CANDIDATE], [KEPT, EMERGENCY], [SOLVER, EMERGENCY]], [[nan, 0.4], [0.2, 0.9], [nan, 0.7]])
    metrics = mpc_metrics(stats, "mpc")
    assert metrics == pytest.approx({
        "mpc/solve_ms_mean": 2.5, "mpc/solve_ms_max": 5.0, "mpc/candidate_rate": 1 / 6, "mpc/kept_rate": 1 / 6,
        "mpc/emergency_count": 2.0, "mpc/invalid_candidate_count": 1.0, "mpc/solver_error_count": 2.0,
        "mpc/free_binaries_mean": 3.0, "mpc/cut_rounds_mean": 1.0, "mpc/steps": 6.0, "mpc/tube_ratio_max": 0.9,
    })
    # only the first two steps of agent 0 and the first step of agent 1
    mask = np.array([[True, True], [True, False], [False, False]])
    masked = mpc_metrics(stats, "eval_mpc", mask)
    assert masked["eval_mpc/steps"] == 3.0 and masked["eval_mpc/emergency_count"] == 0.0
    assert masked["eval_mpc/candidate_rate"] == pytest.approx(1 / 3)
    assert masked["eval_mpc/kept_rate"] == pytest.approx(1 / 3)
    assert masked["eval_mpc/tube_ratio_max"] == pytest.approx(0.4) and masked["eval_mpc/solve_ms_max"] == 2.0


def test_the_tube_ratio_is_logged_only_when_some_step_has_one() -> None:
    nan = float("nan")
    stats = _stats([[SOLVER, SOLVER]], [[nan, nan]])
    metrics = mpc_metrics(stats, "mpc")
    assert "mpc/tube_ratio_max" not in metrics and metrics["mpc/steps"] == 2.0
    stats = _stats([[SOLVER, SOLVER], [SOLVER, SOLVER]], [[nan, 0.3], [nan, nan]])
    assert "eval_mpc/tube_ratio_max" not in mpc_metrics(stats, "eval_mpc", np.array([[True, False], [True, True]]))


def test_no_counted_step_gives_no_metric() -> None:
    assert mpc_metrics({key: np.zeros((0, 2)) for key in STAT_KEYS}, "mpc") == {}
    stats = _stats([[SOLVER, KEPT]], [[0.1, 0.2]])
    assert mpc_metrics(stats, "mpc", np.zeros((1, 2), dtype=bool)) == {}


def test_the_episode_mask_keeps_the_steps_of_each_first_episode() -> None:
    mask = episode_mask([3, 1, 4], 4)
    np.testing.assert_array_equal(mask, [[1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1]])
    assert mask.dtype == np.bool_
