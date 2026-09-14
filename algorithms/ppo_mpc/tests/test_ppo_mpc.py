"""Regression tests for the PPO+MPC manager implementation.

Each test pins down a property that is either non-obvious from the source or
easy to break silently. The set mirrors RL/PPO/tests/test_ppo.py and
HRL/hPPO/tests/test_hppo.py for everything this module shares with them --
the GAE indexing convention, the value standardisation round trip, the
self-describing checkpoint, the derived manager discount, the metric
definitions -- since ppo_mpc.py's PPOMPCAgent used to lack all of it: no
observation normalization, no value-target normalization, a single combined
optimiser group, a hardcoded ent_coef_manager default that disagreed with the
training script, and an explained_variance computed from post-update critic
outputs instead of the pre-update values that actually produced the
advantages.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from ppo_mpc import (
    ManagerActor,
    ManagerRolloutBuffer,
    PPOMPCAgent,
    RunningMeanStd,
    ScaledBeta,
    normalize_obs,
)

OBS_DIM, GOAL_DIM = 4, 4
OBS_LOW = [-1.0, -2.0, -2.0, -2.0]
OBS_HIGH = [11.0, 2.0, 2.0, 2.0]
GOAL_SCALE = [10.0, 10.0, 2.0, 2.0]


def _agent(**kwargs):
    kwargs.setdefault("obs_low", OBS_LOW)
    kwargs.setdefault("obs_high", OBS_HIGH)
    kwargs.setdefault("goal_scale", GOAL_SCALE)
    return PPOMPCAgent(OBS_DIM, GOAL_DIM, device="cpu", **kwargs)


def _filled_buffer(num_steps=16, obs_dim=OBS_DIM, goal_dim=GOAL_DIM, seed=0, fill=None):
    """A buffer with plausible, reproducible contents, filled to `fill` steps.

    Partial filling is the norm here, not an edge case: the manager writes one
    transition per c worker steps into a buffer sized for the worker's step
    budget.
    """
    torch.manual_seed(seed)
    buf = ManagerRolloutBuffer(num_steps, obs_dim, goal_dim, "cpu")
    buf.states = torch.randn(num_steps, obs_dim)
    buf.actions = torch.rand(num_steps, goal_dim) * 2 - 1
    buf.logprobs = torch.randn(num_steps)
    buf.rewards = torch.randn(num_steps) * 100
    buf.values = torch.randn(num_steps) * 100
    buf.dones = (torch.rand(num_steps) < 0.3).float()
    buf.step = num_steps if fill is None else fill
    return buf


# --------------------------------------------------------------------------
# Observation map
# --------------------------------------------------------------------------

def test_normalize_obs_maps_bounds_to_plus_minus_one():
    low = np.array(OBS_LOW, dtype=np.float32)
    high = np.array(OBS_HIGH, dtype=np.float32)
    assert normalize_obs(low, low, high) == pytest.approx([-1.0] * 4)
    assert normalize_obs(high, low, high) == pytest.approx([1.0] * 4)
    mid = (low + high) / 2
    assert normalize_obs(mid, low, high) == pytest.approx([0.0] * 4)


def test_agent_without_bounds_refuses_to_normalize():
    """Silently feeding raw physical units (p_x in [-1, 11]) to a policy
    trained on [-1, 1] is the failure this guard exists to make loud."""
    agent = PPOMPCAgent(OBS_DIM, GOAL_DIM, device="cpu")
    with pytest.raises(ValueError, match="observation bounds"):
        agent.normalize_obs(np.zeros(OBS_DIM, dtype=np.float32))


def test_agent_without_goal_scale_refuses_to_scale():
    agent = PPOMPCAgent(OBS_DIM, GOAL_DIM, device="cpu")
    with pytest.raises(ValueError, match="goal_scale"):
        agent.scale_goal(np.zeros(GOAL_DIM, dtype=np.float32))


def test_scale_goal_maps_the_manager_action_box_to_the_physical_goal():
    agent = _agent()
    normalized = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    assert agent.scale_goal(normalized) == pytest.approx([10.0, -10.0, 1.0, -1.0])


# --------------------------------------------------------------------------
# The manager's derived segment-level discount
# --------------------------------------------------------------------------

def test_gamma_manager_is_derived_from_manager_freq():
    """One manager step spans `manager_freq` env steps (SMDP/options
    treatment). A previous version of this class had no manager_freq at all,
    so the script had to independently compute gamma**c and pass it into
    compute_returns_and_advantage by hand -- exactly the kind of duplicated
    discount that drifted apart in HRL/hPPO's history."""
    agent = _agent(gamma=0.99, manager_freq=10)
    assert agent.gamma_manager == pytest.approx(0.99 ** 10)


def test_loading_a_different_manager_freq_updates_the_derived_discount():
    agent = _agent(gamma=0.99, manager_freq=10)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        agent.save(path)

        reloaded = _agent(gamma=0.99, manager_freq=999)  # deliberately wrong
        reloaded.load(path)

    assert reloaded.manager_freq == 10
    assert reloaded.gamma_manager == pytest.approx(0.99 ** 10)


# --------------------------------------------------------------------------
# GAE
# --------------------------------------------------------------------------

def test_gae_matches_transition_indexed_reference():
    """dones[t] means 'transition t ended an episode', so the mask is
    1 - dones[t]. CleanRL stores the other convention and masks with
    dones[t+1]; this test pins ours down so the difference is never 'fixed'
    by mistake."""
    buf = _filled_buffer()
    next_value, next_done = torch.randn(1).item(), 1.0
    gamma, lam = 0.99 ** 10, 0.95
    buf.compute_returns_and_advantage(next_value, next_done, gamma, lam)

    expected = torch.zeros(buf.step)
    last = 0.0
    for t in reversed(range(buf.step)):
        if t == buf.step - 1:
            nonterminal, next_v = 1.0 - next_done, next_value
        else:
            nonterminal, next_v = 1.0 - buf.dones[t].item(), buf.values[t + 1].item()
        delta = buf.rewards[t].item() + gamma * next_v * nonterminal - buf.values[t].item()
        last = delta + gamma * lam * nonterminal * last
        expected[t] = last
    assert torch.allclose(buf.advantages[:buf.step], expected, atol=1e-4)


def test_gae_returns_identity_holds():
    buf = _filled_buffer()
    buf.compute_returns_and_advantage(0.0, 0.0, 0.99 ** 10, 0.95)
    assert torch.allclose(buf.returns[:buf.step], buf.advantages[:buf.step] + buf.values[:buf.step], atol=1e-4)


def test_gae_stops_bootstrapping_at_a_done():
    buf = ManagerRolloutBuffer(2, OBS_DIM, GOAL_DIM, "cpu")
    buf.add(np.zeros(OBS_DIM), np.zeros(GOAL_DIM), 0.0, 5.0, 0.0, 1.0)  # terminal segment
    buf.add(np.zeros(OBS_DIM), np.zeros(GOAL_DIM), 0.0, 7.0, 0.0, 0.0)
    buf.compute_returns_and_advantage(100.0, 0.0, 0.99 ** 10, 0.95)
    assert buf.advantages[0].item() == pytest.approx(5.0, abs=1e-4)


def test_gae_parameters_are_required():
    """They used to default to 0.99/0.95 while the manager is discounted at
    gamma**manager_freq -- a caller who omitted them got silently different
    discounting from the trained configuration."""
    buf = _filled_buffer()
    with pytest.raises(TypeError):
        buf.compute_returns_and_advantage(0.0, 0.0)


def test_gae_handles_a_partially_filled_buffer():
    """The manager writes far fewer transitions than the buffer's step budget
    (one per c worker steps); GAE and get() must only touch what was filled."""
    buf = _filled_buffer(num_steps=64, fill=5)
    buf.compute_returns_and_advantage(0.0, 0.0, 0.99 ** 10, 0.95)
    states, actions, logprobs, returns, advantages = buf.get()
    assert states.shape[0] == 5
    assert returns.shape[0] == 5
    assert advantages.shape[0] == 5


def test_agent_is_the_single_source_of_the_discount():
    agent = _agent(gamma=0.5, manager_freq=2, gae_lambda=0.5)  # gamma_manager = 0.25
    buf_a, buf_b = _filled_buffer(), _filled_buffer()
    agent.compute_manager_returns_and_advantage(buf_a, 0.0, 0.0)
    buf_b.compute_returns_and_advantage(0.0, 0.0, 0.25, 0.5)
    assert torch.allclose(buf_a.advantages, buf_b.advantages, atol=1e-6)

    agent.gamma_manager = 0.999
    agent.compute_manager_returns_and_advantage(buf_a, 0.0, 0.0)
    assert not torch.allclose(buf_a.advantages, buf_b.advantages, atol=1e-6)


# --------------------------------------------------------------------------
# Value-target standardisation
# --------------------------------------------------------------------------

def test_get_manager_value_inverts_the_standardisation():
    agent = _agent()
    agent.manager_ret_rms.mean, agent.manager_ret_rms.var = 350.0, 53570.0
    states = torch.randn(8, OBS_DIM)
    raw = agent.manager_critic(states).squeeze(-1)
    assert torch.allclose(
        agent.get_manager_value(states),
        raw * agent.manager_ret_rms.std + agent.manager_ret_rms.mean,
        atol=1e-4,
    )


def test_running_mean_std_tracks_batch_statistics():
    rms = RunningMeanStd()
    x = torch.randn(4096) * 30 + 400
    rms.update(x)
    assert rms.mean == pytest.approx(float(x.mean()), rel=1e-3)
    assert rms.std == pytest.approx(float(x.std(unbiased=False)), rel=1e-2)


# --------------------------------------------------------------------------
# Checkpoint round trip
# --------------------------------------------------------------------------

def test_checkpoint_is_self_describing():
    """ret_rms, the observation bounds, the goal scale AND manager_freq must
    all survive a save/load round trip, or a reloaded manager's inputs,
    outputs, and discount are all wrong."""
    agent = _agent(gamma=0.99, manager_freq=10)
    agent.manager_ret_rms.mean, agent.manager_ret_rms.var = 350.0, 53570.0
    states = torch.randn(8, OBS_DIM)
    before = agent.get_manager_value(states)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        agent.save(path)
        reloaded = PPOMPCAgent(OBS_DIM, GOAL_DIM, gamma=0.99, manager_freq=10, device="cpu")
        reloaded.load(path)  # must not need weights_only=False

    assert reloaded.obs_low == pytest.approx(OBS_LOW)
    assert reloaded.obs_high == pytest.approx(OBS_HIGH)
    assert reloaded.goal_scale == pytest.approx(GOAL_SCALE)
    assert reloaded.manager_freq == 10
    assert torch.allclose(reloaded.get_manager_value(states), before, atol=1e-5)
    assert reloaded.normalize_obs(np.array(OBS_HIGH, dtype=np.float32)) == pytest.approx([1.0] * 4)
    assert reloaded.scale_goal(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)) == pytest.approx(GOAL_SCALE)


def test_load_tolerates_a_single_group_optimiser_state(capsys):
    """Checkpoints written before actor/critic had separate parameter groups
    hold a one-group optimiser state. Adam rejects it; `load` must warn and
    keep the weights rather than raise."""
    import torch.optim as optim

    old = _agent()
    old.manager_optimizer = optim.Adam(
        list(old.manager_actor.parameters()) + list(old.manager_critic.parameters()),
        lr=3e-4, eps=1e-5,
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.pt")
        old.save(path)

        new = _agent(critic_lr_mult=3.0)
        new.load(path)  # must not raise

    assert "predates the two-group optimiser" in capsys.readouterr().out
    for a, b in zip(old.manager_actor.parameters(), new.manager_actor.parameters()):
        assert torch.allclose(a, b)
    for a, b in zip(old.manager_critic.parameters(), new.manager_critic.parameters()):
        assert torch.allclose(a, b)


# --------------------------------------------------------------------------
# The two-group optimiser
# --------------------------------------------------------------------------

def test_optimiser_splits_actor_and_critic_into_separate_groups():
    agent = _agent(lr_manager=3e-4, critic_lr_mult=3.0)
    groups = agent.manager_optimizer.param_groups
    assert len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(3e-4)
    assert groups[1]["lr"] == pytest.approx(9e-4)

    actor_ids = {id(p) for p in agent.manager_actor.parameters()}
    critic_ids = {id(p) for p in agent.manager_critic.parameters()}
    assert {id(p) for p in groups[0]["params"]} == actor_ids
    assert {id(p) for p in groups[1]["params"]} == critic_ids


def test_critic_lr_mult_of_one_restores_a_single_rate():
    agent = _agent(lr_manager=3e-4, critic_lr_mult=1.0)
    assert [g["lr"] for g in agent.manager_optimizer.param_groups] == pytest.approx([3e-4, 3e-4])


def test_annealing_every_group_keeps_the_critic_ratio():
    agent = _agent(lr_manager=3e-4, critic_lr_mult=3.0)
    base_lrs = [g["lr"] for g in agent.manager_optimizer.param_groups]
    for group, base_lr in zip(agent.manager_optimizer.param_groups, base_lrs):
        group["lr"] = 0.5 * base_lr
    lrs = [g["lr"] for g in agent.manager_optimizer.param_groups]
    assert lrs == pytest.approx([1.5e-4, 4.5e-4])
    assert lrs[1] / lrs[0] == pytest.approx(3.0)


# --------------------------------------------------------------------------
# update_manager()
# --------------------------------------------------------------------------

def _prepared_agent_and_buffer():
    agent = _agent()
    buf = _filled_buffer(num_steps=8)
    with torch.no_grad():
        _, logprobs, _ = agent.manager_policy_forward(buf.states, buf.actions)
    buf.logprobs = logprobs
    agent.compute_manager_returns_and_advantage(buf, 0.0, 0.0)
    return agent, buf


def test_explained_variance_uses_pre_update_values():
    """Scoring the critic after its own update epochs on the batch measures
    training-set fit, not prediction -- the metric must use the values that
    actually produced the advantages. A previous version recomputed the
    critic's output on `states` *after* the update loop, which is optimistically
    biased and inconsistent with RL/PPO's and HRL/hPPO's definition."""
    agent, buf = _prepared_agent_and_buffer()
    returns = buf.returns[:buf.step].numpy()
    values = buf.values[:buf.step].numpy()
    expected = 1 - np.var(returns - values) / np.var(returns)

    metrics = agent.update_manager(buf, minibatch_size=8, update_epochs=10)
    assert metrics["manager/explained_variance"] == pytest.approx(expected, abs=1e-4)
    assert metrics["manager/value_bias"] == pytest.approx(float(np.mean(values - returns)), abs=1e-2)


def test_update_manager_reports_the_expected_metrics():
    agent, buf = _prepared_agent_and_buffer()
    metrics = agent.update_manager(buf, minibatch_size=8, update_epochs=2)
    assert set(metrics) == {
        "manager/loss_policy", "manager/loss_value", "manager/entropy",
        "manager/approx_kl", "manager/clipfrac", "manager/value_bias",
        "manager/value_target_mean", "manager/value_target_std",
        "manager/explained_variance",
    }
    assert all(isinstance(v, float) for v in metrics.values())


def test_update_manager_changes_both_actor_and_critic():
    agent, buf = _prepared_agent_and_buffer()
    actor_before = [p.clone() for p in agent.manager_actor.parameters()]
    critic_before = [p.clone() for p in agent.manager_critic.parameters()]
    agent.update_manager(buf, minibatch_size=8, update_epochs=2)
    assert any(not torch.allclose(a, b) for a, b in zip(actor_before, agent.manager_actor.parameters()))
    assert any(not torch.allclose(a, b) for a, b in zip(critic_before, agent.manager_critic.parameters()))


def test_update_manager_handles_a_partially_filled_buffer():
    """The common case: the manager's buffer ends an update ~c times emptier
    than the worker's step budget."""
    agent = _agent()
    buf = _filled_buffer(num_steps=64, fill=5)
    with torch.no_grad():
        _, logprobs, _ = agent.manager_policy_forward(buf.states[:5], buf.actions[:5])
    buf.logprobs[:5] = logprobs
    agent.compute_manager_returns_and_advantage(buf, 0.0, 0.0)
    metrics = agent.update_manager(buf, minibatch_size=8, update_epochs=2)
    assert np.isfinite(metrics["manager/loss_value"])


# --------------------------------------------------------------------------
# Defaults match the training script (each scenario's script_ppo_mpc.py)
# --------------------------------------------------------------------------

def test_agent_defaults_match_the_training_script():
    """ent_coef_manager used to default to 0.0 while the script always passed
    0.01 -- an agent built directly (e.g. in a notebook) silently behaved
    differently from every trained checkpoint."""
    agent = PPOMPCAgent(OBS_DIM, GOAL_DIM, device="cpu")
    assert agent.gamma == 0.99
    assert agent.manager_freq == 10
    assert agent.ent_coef_manager == pytest.approx(0.01)


# --------------------------------------------------------------------------
# ScaledBeta sanity (shared with RL/PPO and HRL/hPPO; a quick smoke check)
# --------------------------------------------------------------------------

def test_scaled_beta_samples_are_inside_the_action_box():
    dist = ScaledBeta(torch.full((2000,), 1.7), torch.full((2000,), 3.1), low=torch.tensor(-1.0), high=torch.tensor(1.0))
    samples = dist.sample()
    assert samples.min() >= -1.0 and samples.max() <= 1.0


def test_manager_actor_output_shapes():
    actor = ManagerActor(OBS_DIM, GOAL_DIM)
    alpha, beta = actor(torch.randn(5, OBS_DIM))
    assert alpha.shape == (5, GOAL_DIM)
    assert beta.shape == (5, GOAL_DIM)
    assert (alpha >= 1.0).all() and (beta >= 1.0).all()
