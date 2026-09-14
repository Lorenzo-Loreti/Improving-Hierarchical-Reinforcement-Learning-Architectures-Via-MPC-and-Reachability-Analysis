"""Regression tests for hPPO's vectorized rollout buffers.

The single-environment `RolloutBuffer` is the reference implementation
throughout: every property checked here is stated as "the vectorized buffer
agrees with running the serial one on each environment's own column". That is
the whole correctness claim of the vectorization, and it is the claim that
would break silently -- a GAE recursion that chains one environment's last
transition onto the next environment's first still produces finite advantages
of the right shape, and training would just be quietly worse.

The manager's buffer is the one with teeth. Its columns are *ragged*: an
environment stores a transition every `manager_freq` steps or whenever its
episode ends, and no two environments agree on when that is. See section 8 of
hppo_explanation.md.
"""

import numpy as np
import pytest
import torch

from hppo import (
    HPPOAgent,
    ManagerVecRolloutBuffer,
    RolloutBuffer,
    VecRolloutBuffer,
)

OBS_DIM, GOAL_DIM, ACT_DIM = 4, 2, 2
GAMMA, LAM = 0.99, 0.95


def _serial_reference(states, actions, logprobs, rewards, values, dones,
                      next_value, next_done, gamma=GAMMA, gae_lambda=LAM):
    """Run the single-environment buffer over one column, as the reference."""
    n = rewards.shape[0]
    buf = RolloutBuffer(max(n, 1), states.shape[-1], actions.shape[-1], "cpu")
    buf.states[:n] = states
    buf.actions[:n] = actions
    buf.logprobs[:n] = logprobs
    buf.rewards[:n] = rewards
    buf.values[:n] = values
    buf.dones[:n] = dones
    buf.step = n
    buf.compute_returns_and_advantage(next_value, next_done, gamma, gae_lambda)
    return buf.advantages[:n].clone(), buf.returns[:n].clone()


# --------------------------------------------------------------------------
# Worker buffer: dense (T, N)
# --------------------------------------------------------------------------

def test_dense_gae_matches_the_serial_buffer_column_by_column():
    torch.manual_seed(0)
    T, N = 12, 5
    buf = VecRolloutBuffer(T, N, OBS_DIM, ACT_DIM, "cpu")
    for _ in range(T):
        buf.add(torch.randn(N, OBS_DIM), torch.rand(N, ACT_DIM) * 2 - 1,
                torch.randn(N), torch.randn(N) * 10, torch.randn(N) * 10,
                (torch.rand(N) < 0.25).float())
    next_value, next_done = torch.randn(N) * 10, (torch.rand(N) < 0.5).float()

    buf.compute_returns_and_advantage(next_value, next_done, GAMMA, LAM)

    for i in range(N):
        adv, ret = _serial_reference(
            buf.states[:, i], buf.actions[:, i], buf.logprobs[:, i],
            buf.rewards[:, i], buf.values[:, i], buf.dones[:, i],
            next_value[i], next_done[i])
        assert torch.allclose(buf.advantages[:, i], adv, atol=1e-5)
        assert torch.allclose(buf.returns[:, i], ret, atol=1e-5)


def test_dense_flattening_keeps_values_aligned_with_the_batch():
    """`get_values()` must index the same transitions, in the same order, as
    `get()` -- explained_variance and value_bias pair them elementwise."""
    torch.manual_seed(1)
    T, N = 6, 4
    buf = VecRolloutBuffer(T, N, OBS_DIM, ACT_DIM, "cpu")
    for t in range(T):
        # A value that encodes its own (t, env) address, so a transposed or
        # rolled flattening is detectable rather than merely wrong-looking.
        buf.add(torch.zeros(N, OBS_DIM), torch.zeros(N, ACT_DIM), torch.zeros(N),
                torch.zeros(N), torch.arange(N, dtype=torch.float32) + 100 * t,
                torch.zeros(N))
    _, _, _, _, _ = buf.get()
    expected = torch.stack([torch.arange(N, dtype=torch.float32) + 100 * t
                            for t in range(T)]).reshape(-1)
    assert torch.equal(buf.get_values(), expected)


def test_dense_get_returns_the_full_batch():
    buf = VecRolloutBuffer(7, 3, OBS_DIM, ACT_DIM, "cpu")
    for _ in range(7):
        buf.add(torch.zeros(3, OBS_DIM), torch.zeros(3, ACT_DIM), torch.zeros(3),
                torch.zeros(3), torch.zeros(3), torch.zeros(3))
    states, actions, logprobs, returns, advantages = buf.get()
    assert states.shape == (21, OBS_DIM)
    assert actions.shape == (21, ACT_DIM)
    for x in (logprobs, returns, advantages):
        assert x.shape == (21,)


# --------------------------------------------------------------------------
# Manager buffer: ragged per-environment columns
# --------------------------------------------------------------------------

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
    """A column of length 0 must stay at zero advantage, and must not pick up
    its neighbours' recursion."""
    N = 3
    lengths = np.array([5, 0, 5])
    buf = _fill_ragged(ManagerVecRolloutBuffer(8, N, OBS_DIM, GOAL_DIM, "cpu"),
                       lengths, seed=4)
    buf.compute_returns_and_advantage(torch.randn(N) * 50, torch.zeros(N), GAMMA, LAM)
    assert buf.steps[1] == 0
    assert torch.equal(buf.advantages[:, 1], torch.zeros(8))


def test_ragged_gae_bootstraps_each_column_off_its_own_in_flight_value():
    """Two columns identical except for their bootstrap value must differ, and
    differ only through that value -- i.e. next_value is read per column."""
    N = 2
    buf = ManagerVecRolloutBuffer(4, N, OBS_DIM, GOAL_DIM, "cpu")
    for _ in range(3):
        buf.add(np.array([True, True]), np.zeros((N, OBS_DIM)), np.zeros((N, GOAL_DIM)),
                np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N))
    buf.compute_returns_and_advantage(torch.tensor([10.0, 20.0]), torch.zeros(N), GAMMA, LAM)
    # Undiscounted-to-the-end with zero rewards and zero values, the advantage
    # is a pure function of the bootstrap, so the columns scale exactly 2:1.
    assert torch.allclose(buf.advantages[:3, 1], 2 * buf.advantages[:3, 0], atol=1e-5)


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
    # Environment-major order: env 0's transitions first.
    expected_first_col = torch.tensor(
        [float(i + 100 * t) for i in range(N) for t in range(lengths[i])])
    assert torch.equal(values, expected_first_col)


def test_ragged_total_steps_and_reset():
    N = 3
    buf = _fill_ragged(ManagerVecRolloutBuffer(6, N, OBS_DIM, GOAL_DIM, "cpu"),
                       np.array([6, 2, 4]), seed=5)
    assert buf.total_steps == 12
    buf.reset()
    assert buf.total_steps == 0
    assert buf.get_values().numel() == 0


def test_ragged_add_ignores_an_empty_mask():
    buf = ManagerVecRolloutBuffer(4, 3, OBS_DIM, GOAL_DIM, "cpu")
    buf.add(np.zeros(3, dtype=bool), np.zeros((3, OBS_DIM)), np.zeros((3, GOAL_DIM)),
            np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    assert buf.total_steps == 0


def test_ragged_overflow_is_loud():
    """Sized at the worker's per-environment step count, overflow means the
    cadence bookkeeping is wrong -- it must not wrap around silently."""
    buf = ManagerVecRolloutBuffer(2, 2, OBS_DIM, GOAL_DIM, "cpu")
    full = np.ones(2, dtype=bool)
    args = (np.zeros((2, OBS_DIM)), np.zeros((2, GOAL_DIM)),
            np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
    buf.add(full, *args)
    buf.add(full, *args)
    with pytest.raises(IndexError, match="overflow"):
        buf.add(full, *args)


def test_ragged_add_writes_only_the_masked_environments():
    buf = ManagerVecRolloutBuffer(3, 3, OBS_DIM, GOAL_DIM, "cpu")
    buf.add(np.array([True, False, True]),
            np.array([[1.0] * OBS_DIM, [2.0] * OBS_DIM, [3.0] * OBS_DIM]),
            np.zeros((3, GOAL_DIM)), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    assert list(buf.steps) == [1, 0, 1]
    assert buf.states[0, 0, 0] == 1.0
    assert buf.states[0, 2, 0] == 3.0
    # The unmasked environment's slot is untouched.
    assert buf.states[0, 1, 0] == 0.0


# --------------------------------------------------------------------------
# The agent consumes both layouts
# --------------------------------------------------------------------------

@pytest.mark.parametrize("head", ["manager", "worker"])
def test_update_consumes_a_vectorized_buffer(head):
    agent = HPPOAgent(OBS_DIM, GOAL_DIM, ACT_DIM, device="cpu",
                      obs_low=[-1.0, -2.0, -2.0, -2.0], obs_high=[11.0, 2.0, 2.0, 2.0])
    if head == "manager":
        buf = _fill_ragged(ManagerVecRolloutBuffer(8, 4, OBS_DIM, GOAL_DIM, "cpu"),
                           np.array([8, 3, 6, 5]), seed=6)
        buf.compute_returns_and_advantage(torch.zeros(4), torch.zeros(4), GAMMA, LAM)
        metrics = agent.update_manager(buf, minibatch_size=8, update_epochs=2)
    else:
        buf = VecRolloutBuffer(8, 4, OBS_DIM + GOAL_DIM, ACT_DIM, "cpu")
        for _ in range(8):
            buf.add(torch.randn(4, OBS_DIM + GOAL_DIM), torch.rand(4, ACT_DIM) * 2 - 1,
                    torch.randn(4), torch.randn(4), torch.randn(4), torch.zeros(4))
        buf.compute_returns_and_advantage(torch.zeros(4), torch.zeros(4), GAMMA, LAM)
        metrics = agent.update_worker(buf, minibatch_size=8, update_epochs=2)
    assert metrics[f"{head}/batch_size"] == buf.total_steps if head == "manager" else True
    assert np.isfinite(metrics[f"{head}/loss_policy"])
    assert np.isfinite(metrics[f"{head}/explained_variance"])


def test_explained_variance_uses_the_vectorized_pre_update_values():
    """The metric must pair each return with the value that produced it; a
    misaligned `get_values()` would show up here as a wrong figure."""
    agent = HPPOAgent(OBS_DIM, GOAL_DIM, ACT_DIM, device="cpu",
                      obs_low=[-1.0, -2.0, -2.0, -2.0], obs_high=[11.0, 2.0, 2.0, 2.0])
    buf = _fill_ragged(ManagerVecRolloutBuffer(8, 4, OBS_DIM, GOAL_DIM, "cpu"),
                       np.array([8, 3, 6, 5]), seed=7)
    buf.compute_returns_and_advantage(torch.zeros(4), torch.zeros(4), GAMMA, LAM)
    returns = buf.get()[3].numpy()
    values = buf.get_values().numpy()
    expected = 1 - np.var(returns - values) / np.var(returns)

    metrics = agent.update_manager(buf, minibatch_size=8, update_epochs=1)

    assert metrics["manager/explained_variance"] == pytest.approx(expected, abs=1e-5)
    assert metrics["manager/value_bias"] == pytest.approx(float(np.mean(values - returns)), abs=1e-4)
