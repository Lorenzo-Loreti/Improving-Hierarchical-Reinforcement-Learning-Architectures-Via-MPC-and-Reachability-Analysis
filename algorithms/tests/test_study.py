"""Tests for the parts of algorithms/study.py that are not plotting: seed
parsing, the progression's checkpoint choice, and the rollout recorder the
oracle and every agent figure go through."""

import numpy as np
import pytest

from envs.config import TunnelEnvConfig
from envs.tunnel_env import TunnelEnv
from optimal_solver import MinTimeSolver
from study import _choose_progression, _parse_seeds, oracle_rollout, rollout


def test_seed_lists_parse_ranges_and_lists():
    assert _parse_seeds("1-13") == list(range(1, 14))
    assert _parse_seeds("3,1,2") == [1, 2, 3]
    assert _parse_seeds("1-3, 9") == [1, 2, 3, 9]


def test_progression_picks_distinct_evaluations_spanning_the_run():
    evals = [10240 * k for k in range(1, 17)]
    chosen = _choose_progression(evals, first_solve=51200)
    assert chosen == sorted(set(chosen))
    assert chosen[0] == evals[0] and chosen[-1] == evals[-1]
    assert 51200 in chosen
    assert len(chosen) <= 6


def test_progression_without_a_solve_still_spans_the_run():
    evals = [10240 * k for k in range(1, 5)]
    chosen = _choose_progression(evals, first_solve=None)
    assert chosen[0] == evals[0] and chosen[-1] == evals[-1]


@pytest.fixture(scope="module")
def env():
    return TunnelEnv(config=TunnelEnvConfig())


def test_oracle_rollout_reproduces_the_solvers_own_replay(env):
    """`oracle_rollout` records the oracle through the same recorder as an
    agent; it must land on exactly what the solver's own replay measured."""
    start = np.array([1.0, 0.5, 0.0, 0.0], dtype=np.float32)
    env.reset(options={"init_state": start})
    solved = MinTimeSolver().solve(env)
    ro = oracle_rollout(env, MinTimeSolver(), start)
    assert ro["success"] and ro["length"] == solved.length
    assert ro["return"] == pytest.approx(solved.total_return)
    np.testing.assert_allclose(ro["states"], solved.states, atol=1e-6)
    assert not ro["contacts"].any()
    assert ro["goals"] is None and ro["replanned"] is None


def test_rollout_records_one_entry_per_step_and_the_goal_trace(env):
    """A controller that fills the goal trace (as hPPO's does) gets it back
    per step; states carry the start plus one entry per step."""
    def make_policy_fn(goal_trace):
        def policy_fn(obs):
            goal_trace.append((len(goal_trace) % 10 == 0, np.array([1.0, 0.0])))
            return np.array([env.u_max, 0.0])
        return policy_fn

    ro = rollout(env, make_policy_fn, [1.0, 0.0, 0.0, 0.0])
    T = ro["length"]
    assert ro["states"].shape == (T + 1, 4) and ro["actions"].shape == (T, 2)
    assert ro["goals"].shape == (T, 2) and ro["replanned"].shape == (T,)
    assert ro["replanned"][0] and ro["replanned"].sum() == int(np.ceil(T / 10))
    assert ro["success"]
