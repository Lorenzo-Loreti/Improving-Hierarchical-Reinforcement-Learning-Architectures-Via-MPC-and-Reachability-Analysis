"""The manager's ragged rollout buffer (see ManagerVecRolloutBuffer in ppo_mpc.py).

Its cross-check against hPPO's buffer went with the pre-alignment hPPO in
step 8 of the code alignment (decision log D29); PPO+MPC itself is replaced in
step 9."""

import numpy as np
import pytest
import torch

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
