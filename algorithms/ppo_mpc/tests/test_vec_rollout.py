"""The manager's ragged rollout buffer, held to hPPO's: PPO+MPC keeps its own
copy (see ManagerVecRolloutBuffer in ppo_mpc.py), and it must compute the
same GAE and hand the update the same batch. hPPO's own suite
(algorithms/hppo/tests/test_vec_rollout.py) checks that buffer against a
serial reference column by column."""

import numpy as np
import pytest
import torch

import hppo
from ppo_mpc import ManagerVecRolloutBuffer

OBS_DIM, GOAL_DIM = 4, 2


def fill_ragged(buf, lengths, seed=0):
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


@pytest.mark.parametrize("lengths", [[5, 3, 7, 1], [4, 4, 4, 4], [0, 6, 2, 0]])
def test_the_buffer_is_hppos_buffer(lengths):
    ours = fill_ragged(ManagerVecRolloutBuffer(8, 4, OBS_DIM, GOAL_DIM, "cpu"), lengths)
    theirs = fill_ragged(hppo.ManagerVecRolloutBuffer(8, 4, OBS_DIM, GOAL_DIM, "cpu"), lengths)
    next_value = torch.tensor([3.0, -2.0, 10.0, 0.5])
    ours.compute_returns_and_advantage(next_value, gamma=0.99 ** 10, gae_lambda=0.95)
    theirs.compute_returns_and_advantage(next_value, gamma=0.99 ** 10, gae_lambda=0.95)
    assert ours.total_steps == theirs.total_steps == sum(lengths)
    for a, b in zip(ours.get(), theirs.get()):
        assert torch.equal(a, b)


def test_gae_does_not_chain_one_environment_onto_another():
    """Each column bootstraps off its own in-flight value: changing one
    environment's next_value moves only that column's advantages."""
    buf = fill_ragged(ManagerVecRolloutBuffer(8, 3, OBS_DIM, GOAL_DIM, "cpu"), [4, 2, 5])
    buf.dones.zero_()
    buf.compute_returns_and_advantage(torch.zeros(3), gamma=0.9, gae_lambda=0.95)
    before = buf.advantages.clone()
    buf.compute_returns_and_advantage(torch.tensor([0.0, 100.0, 0.0]), gamma=0.9, gae_lambda=0.95)
    changed = (buf.advantages != before).any(dim=0)
    assert changed.tolist() == [False, True, False]


def test_overflow_is_loud():
    buf = ManagerVecRolloutBuffer(2, 1, OBS_DIM, GOAL_DIM, "cpu")
    fill_ragged(buf, [2])
    with pytest.raises(IndexError, match="overflow"):
        fill_ragged(buf, [1])
