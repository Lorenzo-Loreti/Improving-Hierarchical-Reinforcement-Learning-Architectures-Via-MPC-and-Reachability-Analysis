"""Regression tests for PPO_MPC's vectorized manager rollout buffer.

The single-environment `ManagerRolloutBuffer` is the reference implementation
throughout: every property checked here is stated as "the vectorized buffer
agrees with running the serial one on each environment's own column". That is
the whole correctness claim of the vectorization, and it is the claim that
would break silently -- a GAE recursion that chains one environment's last
segment onto the next environment's first still produces finite advantages of
the right shape, and training would just be quietly worse.

Mirrors HRL/hPPO/tests/test_vec_rollout.py; PPO_MPC has only the manager
buffer, since its worker is the MPC and is not learned.
"""

import numpy as np
import pytest
import torch

from ppo_mpc import (
    ManagerRolloutBuffer,
    ManagerVecRolloutBuffer,
    PPOMPCAgent,
)

OBS_DIM, GOAL_DIM = 4, 4
GAMMA, LAM = 0.99 ** 10, 0.95  # the manager's segment-level discount
OBS_LOW = [-1.0, -2.0, -2.0, -2.0]
OBS_HIGH = [11.0, 2.0, 2.0, 2.0]
GOAL_SCALE = [10.0, 10.0, 2.0, 2.0]


def _agent():
    return PPOMPCAgent(OBS_DIM, GOAL_DIM, device="cpu", obs_low=OBS_LOW,
                       obs_high=OBS_HIGH, goal_scale=GOAL_SCALE)


def _serial_reference(states, actions, logprobs, rewards, values, dones,
                      next_value, next_done):
    """Run the single-environment buffer over one column, as the reference."""
    n = rewards.shape[0]
    buf = ManagerRolloutBuffer(max(n, 1), OBS_DIM, GOAL_DIM, "cpu")
    buf.states[:n] = states
    buf.actions[:n] = actions
    buf.logprobs[:n] = logprobs
    buf.rewards[:n] = rewards
    buf.values[:n] = values
    buf.dones[:n] = dones
    buf.step = n
    buf.compute_returns_and_advantage(next_value, next_done, GAMMA, LAM)
    return buf.advantages[:n].clone(), buf.returns[:n].clone()


def _fill_ragged(buf, lengths, seed=0):
    """Give environment i exactly lengths[i] transitions."""
    rng = np.random.default_rng(seed)
    N = buf.num_envs
    for t in range(int(max(lengths))):
        mask = np.array([t < lengths[i] for i in range(N)])
        buf.add(mask,
                rng.standard_normal((N, buf.obs_dim)).astype(np.float32),
                rng.uniform(-1, 1, (N, buf.goal_dim)).astype(np.float32),
                rng.standard_normal(N).astype(np.float32),
                (rng.standard_normal(N) * 50).astype(np.float32),
                (rng.standard_normal(N) * 50).astype(np.float32),
                (rng.random(N) < 0.3).astype(np.float32))
    return buf


# --------------------------------------------------------------------------
# Ragged GAE
# --------------------------------------------------------------------------

def test_ragged_gae_matches_the_serial_buffer_column_by_column():
    """The property the whole design rests on: each environment's segments
    form their own temporal chain, bootstrapped off their own in-flight value."""
    N = 5
    lengths = np.array([9, 4, 12, 1, 7])
    buf = _fill_ragged(ManagerVecRolloutBuffer(12, N, OBS_DIM, GOAL_DIM, "cpu"),
                       lengths, seed=3)
    next_value = torch.randn(N) * 50
    next_done = (torch.rand(N) < 0.5).float()

    buf.compute_returns_and_advantage(next_value, next_done, GAMMA, LAM)

    for i, n in enumerate(lengths):
        adv, ret = _serial_reference(
            buf.states[:n, i], buf.actions[:n, i], buf.logprobs[:n, i],
            buf.rewards[:n, i], buf.values[:n, i], buf.dones[:n, i],
            next_value[i], next_done[i])
        assert torch.allclose(buf.advantages[:n, i], adv, atol=1e-4), f"env {i}"
        assert torch.allclose(buf.returns[:n, i], ret, atol=1e-4), f"env {i}"


def test_ragged_gae_does_not_chain_one_environment_onto_another():
    N = 3
    buf = _fill_ragged(ManagerVecRolloutBuffer(8, N, OBS_DIM, GOAL_DIM, "cpu"),
                       np.array([5, 0, 5]), seed=4)
    buf.compute_returns_and_advantage(torch.randn(N) * 50, torch.zeros(N), GAMMA, LAM)
    assert buf.steps[1] == 0
    assert torch.equal(buf.advantages[:, 1], torch.zeros(8))


def test_ragged_gae_bootstraps_each_column_off_its_own_in_flight_value():
    N = 2
    buf = ManagerVecRolloutBuffer(4, N, OBS_DIM, GOAL_DIM, "cpu")
    for _ in range(3):
        buf.add(np.array([True, True]), np.zeros((N, OBS_DIM)), np.zeros((N, GOAL_DIM)),
                np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N))
    buf.compute_returns_and_advantage(torch.tensor([10.0, 20.0]), torch.zeros(N), GAMMA, LAM)
    # With zero rewards and zero values the advantage is a pure function of the
    # bootstrap, so the two columns must scale exactly 2:1.
    assert torch.allclose(buf.advantages[:3, 1], 2 * buf.advantages[:3, 0], atol=1e-5)


def test_ragged_gae_stops_bootstrapping_at_a_done():
    N = 1
    buf = ManagerVecRolloutBuffer(2, N, OBS_DIM, GOAL_DIM, "cpu")
    one = np.ones(1, dtype=bool)
    # Transition 0 ends the episode, so it must not see transition 1's value.
    buf.add(one, np.zeros((N, OBS_DIM)), np.zeros((N, GOAL_DIM)),
            np.zeros(N), np.ones(N), np.zeros(N), np.ones(N))
    buf.add(one, np.zeros((N, OBS_DIM)), np.zeros((N, GOAL_DIM)),
            np.zeros(N), np.zeros(N), np.full(N, 100.0), np.zeros(N))
    buf.compute_returns_and_advantage(torch.zeros(N), torch.zeros(N), GAMMA, LAM)
    # reward 1, no bootstrap, value 0 -> advantage exactly 1.
    assert buf.advantages[0, 0].item() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# Storage and flattening
# --------------------------------------------------------------------------

def test_ragged_flattening_keeps_values_aligned_with_the_batch():
    N = 4
    lengths = np.array([3, 1, 5, 2])
    buf = ManagerVecRolloutBuffer(5, N, OBS_DIM, GOAL_DIM, "cpu")
    for t in range(int(max(lengths))):
        mask = np.array([t < lengths[i] for i in range(N)])
        # Encode the (t, env) address in both the state and the value, so a
        # misaligned flattening shows up as a mismatch between them.
        addr = np.arange(N, dtype=np.float32) + 100 * t
        buf.add(mask, np.tile(addr[:, None], (1, OBS_DIM)), np.zeros((N, GOAL_DIM)),
                np.zeros(N), np.zeros(N), addr, np.zeros(N))

    states, _, _, _, _ = buf.get()
    values = buf.get_values()
    assert values.shape[0] == buf.total_steps == int(lengths.sum())
    assert torch.equal(states[:, 0], values)
    expected = torch.tensor([float(i + 100 * t) for i in range(N) for t in range(lengths[i])])
    assert torch.equal(values, expected)


def test_ragged_add_writes_only_the_masked_environments():
    buf = ManagerVecRolloutBuffer(3, 3, OBS_DIM, GOAL_DIM, "cpu")
    buf.add(np.array([True, False, True]),
            np.array([[1.0] * OBS_DIM, [2.0] * OBS_DIM, [3.0] * OBS_DIM]),
            np.zeros((3, GOAL_DIM)), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    assert list(buf.steps) == [1, 0, 1]
    assert buf.states[0, 0, 0] == 1.0
    assert buf.states[0, 2, 0] == 3.0
    assert buf.states[0, 1, 0] == 0.0  # untouched


def test_ragged_add_ignores_an_empty_mask():
    buf = ManagerVecRolloutBuffer(4, 3, OBS_DIM, GOAL_DIM, "cpu")
    buf.add(np.zeros(3, dtype=bool), np.zeros((3, OBS_DIM)), np.zeros((3, GOAL_DIM)),
            np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    assert buf.total_steps == 0


def test_ragged_overflow_is_loud():
    buf = ManagerVecRolloutBuffer(2, 2, OBS_DIM, GOAL_DIM, "cpu")
    full = np.ones(2, dtype=bool)
    args = (np.zeros((2, OBS_DIM)), np.zeros((2, GOAL_DIM)),
            np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
    buf.add(full, *args)
    buf.add(full, *args)
    with pytest.raises(IndexError, match="overflow"):
        buf.add(full, *args)


def test_total_steps_and_reset():
    buf = _fill_ragged(ManagerVecRolloutBuffer(6, 3, OBS_DIM, GOAL_DIM, "cpu"),
                       np.array([6, 2, 4]), seed=5)
    assert buf.total_steps == 12
    buf.reset()
    assert buf.total_steps == 0
    assert buf.get_values().numel() == 0


# --------------------------------------------------------------------------
# The agent consumes it
# --------------------------------------------------------------------------

def test_update_manager_consumes_a_vectorized_buffer():
    agent = _agent()
    buf = _fill_ragged(ManagerVecRolloutBuffer(8, 4, OBS_DIM, GOAL_DIM, "cpu"),
                       np.array([8, 3, 6, 5]), seed=6)
    buf.compute_returns_and_advantage(torch.zeros(4), torch.zeros(4), GAMMA, LAM)
    metrics = agent.update_manager(buf, minibatch_size=8, update_epochs=2)
    assert np.isfinite(metrics["manager/loss_policy"])
    assert np.isfinite(metrics["manager/explained_variance"])


def test_explained_variance_uses_the_vectorized_pre_update_values():
    """The metric must pair each return with the value that produced it; a
    misaligned `get_values()` would show up here as a wrong figure."""
    agent = _agent()
    buf = _fill_ragged(ManagerVecRolloutBuffer(8, 4, OBS_DIM, GOAL_DIM, "cpu"),
                       np.array([8, 3, 6, 5]), seed=7)
    buf.compute_returns_and_advantage(torch.zeros(4), torch.zeros(4), GAMMA, LAM)
    returns = buf.get()[3].numpy()
    values = buf.get_values().numpy()
    expected = 1 - np.var(returns - values) / np.var(returns)

    metrics = agent.update_manager(buf, minibatch_size=8, update_epochs=1)

    assert metrics["manager/explained_variance"] == pytest.approx(expected, abs=1e-5)
    assert metrics["manager/value_bias"] == pytest.approx(float(np.mean(values - returns)), abs=1e-4)
