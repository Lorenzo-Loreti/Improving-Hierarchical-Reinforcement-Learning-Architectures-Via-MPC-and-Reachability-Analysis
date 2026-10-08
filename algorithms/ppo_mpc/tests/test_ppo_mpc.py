"""Regression tests for PPO+MPC's agent (algorithms/ppo_mpc/ppo_mpc.py).

The agent is hPPO's manager. The test that held it there, update for update,
went with the pre-alignment hPPO in step 8 of the code alignment (decision
log D29); PPO+MPC itself is replaced in step 9. The tests left pin what the
agent adds or keeps on its own: the goal scale, the derived discount, the
self-describing checkpoint (now including the worker's settings), and the
defaults the training script relies on.
"""

import sys

import numpy as np
import pytest
import torch

from ppo_mpc import PPOMPCAgent, ManagerVecRolloutBuffer
from test_vec_rollout import fill_ragged

OBS_DIM, GOAL_DIM = 4, 2
OBS_LOW = [-1.0, -2.0, -1.2, -1.2]
OBS_HIGH = [11.0, 2.0, 1.2, 1.2]


def _agent(**kwargs):
    kwargs.setdefault("obs_low", OBS_LOW)
    kwargs.setdefault("obs_high", OBS_HIGH)
    return PPOMPCAgent(OBS_DIM, GOAL_DIM, device="cpu", **kwargs)


def test_an_empty_batch_skips_the_update_with_the_same_metric_keys():
    agent = _agent()
    before = [p.clone() for p in agent.manager_actor.parameters()]
    empty = agent.update_manager(ManagerVecRolloutBuffer(4, 2, OBS_DIM, GOAL_DIM, "cpu"), 4, 2)
    assert all(torch.equal(a, b) for a, b in zip(before, agent.manager_actor.parameters()))
    full_buf = fill_ragged(ManagerVecRolloutBuffer(8, 2, OBS_DIM, GOAL_DIM, "cpu"), [3, 4])
    agent.compute_manager_returns_and_advantage(full_buf, torch.zeros(2))
    assert set(empty) == set(agent.update_manager(full_buf, 4, 2))


def test_scale_goal_maps_the_action_box_to_the_goal_box():
    agent = _agent(max_goal_bound=1.8)
    np.testing.assert_allclose(agent.scale_goal(np.array([1.0, -0.5])), [1.8, -0.9])


def test_the_manager_discount_is_derived_from_the_cadence():
    agent = _agent(gamma=0.99, manager_freq=10)
    assert agent.gamma_manager == pytest.approx(0.99 ** 10)


def test_agent_without_bounds_refuses_to_normalize():
    with pytest.raises(ValueError, match="observation bounds"):
        PPOMPCAgent(OBS_DIM, GOAL_DIM).normalize_obs(np.zeros(4))


def test_the_critic_has_its_own_learning_rate():
    agent = _agent(lr_manager=1e-4, critic_lr_mult=3.0)
    assert [g["lr"] for g in agent.manager_optimizer.param_groups] == pytest.approx([1e-4, 3e-4])


def test_the_checkpoint_carries_the_whole_controller(tmp_path):
    """Manager, observation map, goal box, cadence -- and the worker's
    settings, since the MPC is half of the trained controller."""
    settings = dict(horizon=12, q_pos=5.0, q_vel=1.0, r=0.2, noise_bound_p=0.005, noise_bound_v=0.05,
                    rho=1e-3, rpi_eps=1e-2, mip_gap=1e-4, work_limit=None)
    torch.manual_seed(3)
    agent = _agent(max_goal_bound=2.5, manager_freq=7, mpc_settings=settings)
    path = str(tmp_path / "a.pt")
    agent.save(path)
    reloaded = PPOMPCAgent(OBS_DIM, GOAL_DIM)
    reloaded.load(path)
    assert reloaded.max_goal_bound == 2.5 and reloaded.manager_freq == 7
    assert reloaded.gamma_manager == pytest.approx(0.99 ** 7)
    assert reloaded.mpc_settings == settings
    np.testing.assert_array_equal(reloaded.obs_low, agent.obs_low)
    x = torch.randn(5, OBS_DIM)
    assert torch.equal(reloaded.manager_act(x), agent.manager_act(x))


def test_agent_defaults_match_the_training_script(monkeypatch):
    """PPOMPCAgent's defaults are ppo_mpc_train.py's flag defaults, and its
    goal box is the one the script derives for the canonical plant."""
    import ppo_mpc_train
    from envs.config import TunnelEnvConfig
    from tube_mpc import TubeMPCWorker
    scenario = ppo_mpc_train.Scenario(name="tunnel", env_cls=None, vec_env_cls=None,
                                      make_env_config=lambda **kw: TunnelEnvConfig(**kw),
                                      total_timesteps=1, wandb_project="x", env_u_max_help="", script_dir=".")
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = ppo_mpc_train.parse_args(scenario)
    agent = PPOMPCAgent(OBS_DIM)
    assert agent.gamma == args.gamma and agent.manager_freq == args.manager_freq
    assert agent.gae_lambda == args.gae_lambda and agent.clip_coef == args.clip_coef
    assert agent.ent_coef_manager == args.ent_coef_manager
    assert agent.max_grad_norm == args.max_grad_norm
    lrs = [g["lr"] for g in agent.manager_optimizer.param_groups]
    assert lrs == pytest.approx([args.learning_rate_manager, args.learning_rate_manager * args.critic_lr_mult])
    config = TunnelEnvConfig()
    assert agent.max_goal_bound == pytest.approx(
        args.goal_bound_slack * config.v_max * args.manager_freq * config.dt)
    # And the worker's settings name TubeMPCWorker's constructor arguments.
    assert set(ppo_mpc_train.mpc_settings_from_args(args)) == set(TubeMPCWorker.SETTINGS)
