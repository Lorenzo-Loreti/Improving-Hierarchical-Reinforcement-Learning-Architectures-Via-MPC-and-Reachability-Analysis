"""Tests for ppo_mpc_train.py's evaluation controller, its checkpoint loader
and, end to end, a few updates of the training loop itself -- on the tunnel,
whose obstacle-free MIQP fits the size-limited Gurobi licence even with the
disturbance's tube on."""

import os
import sys

import numpy as np
import pytest
import torch

import ppo_mpc_train
from envs.config import TunnelEnvConfig
from envs.tunnel_env import TunnelEnv
from envs.vec_tunnel_env import TunnelVecEnv
from ppo_mpc import PPOMPCAgent
from ppo_mpc_train import load_agent, make_policy_fn
from tube_mpc import TubeMPCWorker

NOISE = (0.005, 0.05)


def _agent(manager_freq=3, seed=0, **kwargs):
    env = TunnelEnv()
    torch.manual_seed(seed)
    return PPOMPCAgent(4, 2, obs_low=env.observation_space.low, obs_high=env.observation_space.high,
                       manager_freq=manager_freq, **kwargs)


def _observations(n):
    """An agent moving right and slightly up."""
    return [np.array([1.0 + 0.1 * k, 0.02 * k, 1.0, 0.2], dtype=np.float32) for k in range(n)]


def test_the_manager_replans_every_manager_freq_steps_and_the_goal_decays_between():
    agent = _agent(manager_freq=3)
    worker = TubeMPCWorker.from_env(TunnelEnv())
    trace = []
    policy_fn = make_policy_fn(agent, worker, goal_trace=trace)
    observations = _observations(7)
    for obs in observations:
        policy_fn(obs)
    assert [r for r, _ in trace] == [True, False, False, True, False, False, True]
    for k in (1, 2, 4, 5):
        displacement = observations[k][:2] - observations[k - 1][:2]
        np.testing.assert_allclose(trace[k][1], trace[k - 1][1] - displacement, atol=1e-6)


def test_a_new_episodes_controller_forgets_the_last_plan():
    """Building a policy_fn resets its worker slot: the first step of an
    episode must not be handed a candidate from the previous one."""
    agent = _agent()
    worker = TubeMPCWorker.from_env(TunnelEnv())
    policy_fn = make_policy_fn(agent, worker)
    policy_fn(_observations(1)[0])
    assert worker._prev[0] is not None
    make_policy_fn(agent, worker)
    assert worker._prev[0] is None


def test_a_reloaded_controller_acts_identically_with_its_trained_tube(tmp_path):
    """load_agent rebuilds the worker from the checkpoint's settings -- the
    disturbance model included, even for an undisturbed evaluation env."""
    env = TunnelEnv(config=TunnelEnvConfig(noise_bound_p=NOISE[0], noise_bound_v=NOISE[1]))
    settings = dict(TubeMPCWorker.from_env(env).settings)
    agent = _agent(manager_freq=4, seed=2, mpc_settings=settings)
    path = str(tmp_path / "agent.pt")
    agent.save(path)
    reloaded, worker = load_agent(path, TunnelEnv())
    assert worker.settings == settings and worker.Z.s > 0
    original = make_policy_fn(agent, TubeMPCWorker.from_env(env))
    again = make_policy_fn(reloaded, worker)
    for obs in _observations(9):
        np.testing.assert_array_equal(original(obs), again(obs))


@pytest.mark.parametrize("noise", [(0.0, 0.0), NOISE])
def test_a_short_training_run_end_to_end(tmp_path, monkeypatch, noise):
    """Two updates, an evaluation and a solved-check each, on the tunnel,
    with and without the disturbance: the loop runs, writes its checkpoints,
    and the saved controller reloads with the tube it was trained with."""
    scenario = ppo_mpc_train.Scenario(
        name="tunnel", env_cls=TunnelEnv, vec_env_cls=TunnelVecEnv,
        make_env_config=lambda **kw: TunnelEnvConfig(**kw), total_timesteps=256,
        wandb_project="x", env_u_max_help="", script_dir=str(tmp_path))
    monkeypatch.setattr(sys, "argv", [
        "prog", "--num-envs", "2", "--num-steps-worker", "128", "--eval-freq", "1",
        "--eval-episodes", "1", "--solved-grid-nx", "1", "--solved-grid-ny", "1",
        "--noise-bound-p", str(noise[0]), "--noise-bound-v", str(noise[1]),
        "--checkpoint-dir", "ckpt"])
    ppo_mpc_train.main(scenario)
    (run_dir,) = os.listdir(tmp_path / "ckpt")
    files = set(os.listdir(tmp_path / "ckpt" / run_dir))
    assert {"final.pt", "best.pt", "metrics.jsonl", "config.json"} <= files
    _, worker = load_agent(str(tmp_path / "ckpt" / run_dir / "final.pt"), TunnelEnv())
    assert worker.settings["noise_bound_v"] == noise[1]
