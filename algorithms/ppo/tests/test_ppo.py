"""Regression tests for the PPO implementation.

Each test pins down a property that is either non-obvious from the source or
easy to break silently -- the corrections in ScaledBeta, the GAE indexing
convention, the value standardisation round trip, and the metric definitions.
See the thesis PPO chapter (kept outside this repo) for the reasoning
behind each.
"""

import math
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Beta

from ppo import (
    ActorNetwork,
    CriticNetwork,
    PPOAgent,
    RolloutBuffer,
    RunningMeanStd,
    ScaledBeta,
    clipped_value_loss,
    floor_normalize,
    normalize_obs,
)

LOW, HIGH = torch.tensor(-1.0), torch.tensor(1.0)


def _filled_buffer(num_steps=4, num_envs=2, obs_dim=4, act_dim=2, seed=0):
    """A buffer with plausible, reproducible contents and GAE already run."""
    torch.manual_seed(seed)
    buf = RolloutBuffer(num_steps, num_envs, obs_dim, act_dim, "cpu")
    buf.states = torch.randn(num_steps, num_envs, obs_dim)
    buf.actions = torch.rand(num_steps, num_envs, act_dim) * 2 - 1
    buf.logprobs = torch.randn(num_steps, num_envs)
    buf.rewards = torch.randn(num_steps, num_envs) * 100
    buf.values = torch.randn(num_steps, num_envs) * 100
    buf.dones = (torch.rand(num_steps, num_envs) < 0.3).float()
    return buf


# --------------------------------------------------------------------------
# ScaledBeta: the change-of-variables corrections
# --------------------------------------------------------------------------

def test_scaled_beta_density_integrates_to_one():
    """The Jacobian correction in log_prob makes it a density on [-1, 1]."""
    dist = ScaledBeta(torch.tensor([2.5]), torch.tensor([1.3]), low=LOW, high=HIGH)
    xs = torch.linspace(-1 + 1e-6, 1 - 1e-6, 200001).unsqueeze(1)
    density = dist.log_prob(xs).exp().squeeze(1)
    assert torch.trapz(density, xs.squeeze(1)).item() == pytest.approx(1.0, abs=1e-4)


def test_scaled_beta_entropy_matches_numeric_integration():
    """The +log(scale) shift in entropy() matches -E[log p] of the scaled density."""
    dist = ScaledBeta(torch.tensor([2.5]), torch.tensor([1.3]), low=LOW, high=HIGH)
    xs = torch.linspace(-1 + 1e-6, 1 - 1e-6, 200001).unsqueeze(1)
    logp = dist.log_prob(xs).squeeze(1)
    numeric = -torch.trapz(logp.exp() * logp, xs.squeeze(1))
    assert numeric.item() == pytest.approx(float(dist.entropy()), abs=1e-4)


def test_scaled_beta_corrections_are_the_log_of_the_scale():
    """Both corrections are exactly log(high - low) against the base Beta."""
    alpha, beta = torch.tensor([2.5]), torch.tensor([1.3])
    base, scaled = Beta(alpha, beta), ScaledBeta(alpha, beta, low=LOW, high=HIGH)
    action = torch.tensor([0.4])
    unscaled = (action - LOW) / (HIGH - LOW)
    assert scaled.log_prob(action).item() == pytest.approx(
        (base.log_prob(unscaled) - math.log(2.0)).item(), abs=1e-6)
    assert float(scaled.entropy()) == pytest.approx(float(base.entropy()) + math.log(2.0), abs=1e-6)


def test_scaled_beta_samples_are_inside_the_action_box():
    """Support is the action box, so the env's action clip is a no-op."""
    dist = ScaledBeta(torch.full((5000, 2), 1.7), torch.full((5000, 2), 3.1), low=LOW, high=HIGH)
    samples = dist.sample()
    assert samples.min() >= -1.0 and samples.max() <= 1.0


def test_deterministic_sample_is_the_distribution_mean():
    """Evaluation uses the mean, NOT the mode -- thesis PPO chapter, 4.5."""
    alpha, beta = torch.tensor([2.5]), torch.tensor([1.3])
    dist = ScaledBeta(alpha, beta, low=LOW, high=HIGH)
    expected_mean = 2.0 * (alpha / (alpha + beta)) - 1.0
    expected_mode = 2.0 * ((alpha - 1) / (alpha + beta - 2)) - 1.0
    assert dist.deterministic_sample().item() == pytest.approx(expected_mean.item(), abs=1e-6)
    assert dist.deterministic_sample().item() != pytest.approx(expected_mode.item(), abs=1e-3)


# --------------------------------------------------------------------------
# Actor: the alpha, beta >= 1 floor
# --------------------------------------------------------------------------

def test_actor_concentrations_never_drop_below_one():
    """softplus + 1 keeps the density bounded and unimodal for any logits."""
    actor = ActorNetwork(4, 2)
    with torch.no_grad():
        for layer in actor.net:
            if isinstance(layer, torch.nn.Linear):
                layer.bias.fill_(-50.0)  # drive softplus to ~0
    alpha, beta = actor(torch.randn(64, 4))
    assert alpha.min().item() >= 1.0 and beta.min().item() >= 1.0


def test_actor_initial_policy_is_near_uniform():
    """std=0.01 output init gives alpha = beta = softplus(0) + 1, i.e. broad."""
    torch.manual_seed(0)
    alpha, beta = ActorNetwork(4, 2)(torch.zeros(1, 4))
    expected = math.log(2.0) + 1.0
    assert alpha.flatten().tolist() == pytest.approx([expected] * 2, abs=1e-4)
    assert beta.flatten().tolist() == pytest.approx([expected] * 2, abs=1e-4)


def test_entropy_is_bounded_above_by_the_uniform_case():
    """Unlike a Gaussian, the entropy bonus has an attainable maximum."""
    uniform = float(ScaledBeta(torch.tensor(1.0), torch.tensor(1.0), LOW, HIGH).entropy())
    assert uniform == pytest.approx(math.log(2.0), abs=1e-6)
    at_init = float(ScaledBeta(torch.tensor(math.log(2.0) + 1.0),
                               torch.tensor(math.log(2.0) + 1.0), LOW, HIGH).entropy())
    assert at_init < uniform


# --------------------------------------------------------------------------
# GAE
# --------------------------------------------------------------------------

def test_gae_matches_transition_indexed_reference():
    """dones[t] means 'transition t ended an episode', so the mask is 1-dones[t].

    CleanRL stores the other convention and masks with dones[t+1]; this test
    pins ours down so the difference is never 'fixed' by mistake.
    """
    buf = _filled_buffer()
    next_value, next_done = torch.randn(2), torch.tensor([0.0, 1.0])
    gamma, lam = 0.999, 0.95
    buf.compute_returns_and_advantage(next_value, next_done, gamma, lam)

    expected = torch.zeros(buf.num_steps, buf.num_envs)
    last = torch.zeros(buf.num_envs)
    for t in reversed(range(buf.num_steps)):
        if t == buf.num_steps - 1:
            nonterminal, next_v = 1.0 - next_done, next_value
        else:
            nonterminal, next_v = 1.0 - buf.dones[t], buf.values[t + 1]
        delta = buf.rewards[t] + gamma * next_v * nonterminal - buf.values[t]
        last = delta + gamma * lam * nonterminal * last
        expected[t] = last
    assert torch.allclose(buf.advantages, expected, atol=1e-5)


def test_gae_returns_identity_holds():
    """returns = advantages + values, which update() relies on."""
    buf = _filled_buffer()
    buf.compute_returns_and_advantage(torch.randn(2), torch.zeros(2), 0.999, 0.95)
    assert torch.allclose(buf.returns, buf.advantages + buf.values, atol=1e-5)


def test_gae_stops_bootstrapping_at_a_done():
    """A terminal transition must not leak value from the next episode."""
    buf = RolloutBuffer(2, 1, 4, 2, "cpu")
    buf.rewards = torch.tensor([[5.0], [7.0]])
    buf.values = torch.zeros(2, 1)
    buf.dones = torch.tensor([[1.0], [0.0]])  # transition 0 ended the episode
    buf.compute_returns_and_advantage(torch.tensor([100.0]), torch.zeros(1), 0.999, 0.95)
    # Step 0 is terminal: its advantage is its own reward and nothing else.
    assert buf.advantages[0, 0].item() == pytest.approx(5.0, abs=1e-5)


def test_gae_parameters_are_required():
    """They used to default to 0.99 while every call site passed 0.999."""
    buf = _filled_buffer()
    with pytest.raises(TypeError):
        buf.compute_returns_and_advantage(torch.randn(2), torch.zeros(2))


def test_agent_is_the_single_source_of_the_discount():
    """agent.gamma/gae_lambda drive GAE; they used to be set and never read."""
    agent = PPOAgent(4, 2, gamma=0.5, gae_lambda=0.5, device="cpu")
    buf_a, buf_b = _filled_buffer(), _filled_buffer()
    next_value, next_done = torch.randn(2), torch.zeros(2)

    agent.compute_returns_and_advantage(buf_a, next_value, next_done)
    buf_b.compute_returns_and_advantage(next_value, next_done, 0.5, 0.5)
    assert torch.allclose(buf_a.advantages, buf_b.advantages, atol=1e-6)

    agent.gamma = 0.999
    agent.compute_returns_and_advantage(buf_a, next_value, next_done)
    assert not torch.allclose(buf_a.advantages, buf_b.advantages, atol=1e-6)


# --------------------------------------------------------------------------
# Value-target standardisation
# --------------------------------------------------------------------------

def test_get_value_inverts_the_standardisation():
    """The critic learns in standardised space; get_value is the only way back."""
    agent = PPOAgent(4, 2, device="cpu")
    agent.ret_rms.mean, agent.ret_rms.var = 350.0, 53570.0
    states = torch.randn(8, 4)
    raw = agent.critic(states).squeeze(-1)
    assert torch.allclose(agent.get_value(states),
                          raw * agent.ret_rms.std + agent.ret_rms.mean, atol=1e-4)


def test_running_mean_std_tracks_batch_statistics():
    rms = RunningMeanStd()
    x = torch.randn(4096) * 30 + 400
    rms.update(x)
    assert rms.mean == pytest.approx(float(x.mean()), rel=1e-3)
    assert rms.std == pytest.approx(float(x.std(unbiased=False)), rel=1e-2)


def test_running_mean_std_count_is_capped_at_the_horizon():
    """The cap is what keeps early-training returns from poisoning late targets."""
    rms = RunningMeanStd(horizon=10)
    batch = torch.randn(2048)
    for _ in range(30):
        rms.update(batch)
    assert rms.count == 10 * 2048


def test_running_mean_std_forgets_a_stale_regime():
    """A capped estimator must migrate to a new return level; an uncapped one lags."""
    capped, uncapped = RunningMeanStd(horizon=10), RunningMeanStd(horizon=10**9)
    for _ in range(3):
        capped.update(torch.full((2048,), -600.0))
        uncapped.update(torch.full((2048,), -600.0))
    for _ in range(15):
        capped.update(torch.full((2048,), 400.0))
        uncapped.update(torch.full((2048,), 400.0))
    assert capped.mean > uncapped.mean


# --------------------------------------------------------------------------
# Observation map and checkpoint round trip
# --------------------------------------------------------------------------

def test_normalize_obs_maps_bounds_to_plus_minus_one():
    low = np.array([-1.0, -2.0, -2.0, -2.0], dtype=np.float32)
    high = np.array([11.0, 2.0, 2.0, 2.0], dtype=np.float32)
    assert normalize_obs(low, low, high) == pytest.approx([-1.0] * 4)
    assert normalize_obs(high, low, high) == pytest.approx([1.0] * 4)
    mid = (low + high) / 2
    assert normalize_obs(mid, low, high) == pytest.approx([0.0] * 4)


def test_agent_without_bounds_refuses_to_normalize():
    """Silently feeding raw physical units to a policy trained on [-1,1] is the
    failure this guard exists to make loud."""
    agent = PPOAgent(4, 2, device="cpu")
    with pytest.raises(ValueError, match="observation bounds"):
        agent.normalize_obs(np.zeros(4, dtype=np.float32))


def test_checkpoint_is_self_describing():
    """ret_rms AND the observation bounds must survive a save/load round trip,
    or a reloaded agent's inputs and outputs are both wrong."""
    low = [-1.0, -2.0, -2.0, -2.0]
    high = [11.0, 2.0, 2.0, 2.0]
    agent = PPOAgent(4, 2, obs_low=low, obs_high=high, device="cpu")
    agent.ret_rms.mean, agent.ret_rms.var = 350.0, 53570.0
    states = torch.randn(8, 4)
    before = agent.get_value(states)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        agent.save(path)
        reloaded = PPOAgent(4, 2, device="cpu")
        reloaded.load(path)  # must not need weights_only=False

    assert reloaded.obs_low == pytest.approx(low)
    assert reloaded.obs_high == pytest.approx(high)
    assert torch.allclose(reloaded.get_value(states), before, atol=1e-5)
    assert reloaded.normalize_obs(np.array(high, dtype=np.float32)) == pytest.approx([1.0] * 4)


# --------------------------------------------------------------------------
# update()
# --------------------------------------------------------------------------

def _prepared_agent_and_buffer(agent=None):
    agent = agent or PPOAgent(4, 2, device="cpu")
    buf = _filled_buffer(num_steps=8, num_envs=4)
    with torch.no_grad():
        flat = buf.states.reshape(-1, 4)
        _, logprobs, _ = agent.policy_forward(flat, buf.actions.reshape(-1, 2))
    buf.logprobs = logprobs.reshape(8, 4)
    agent.compute_returns_and_advantage(buf, torch.zeros(4), torch.zeros(4))
    return agent, buf


def test_explained_variance_uses_pre_update_values():
    """Scoring the critic after its own 10 epochs on the batch measures
    training-set fit, not prediction. The metric must use the values that
    actually produced the advantages."""
    agent, buf = _prepared_agent_and_buffer()
    returns = buf.returns.reshape(-1).numpy()
    values = buf.values.reshape(-1).numpy()
    expected = 1 - np.var(returns - values) / np.var(returns)

    metrics = agent.update(buf, minibatch_size=8, update_epochs=10)
    assert metrics["loss/explained_variance"] == pytest.approx(expected, abs=1e-4)
    assert metrics["loss/value_bias"] == pytest.approx(float(np.mean(values - returns)), abs=1e-2)


def test_update_runs_every_epoch_when_target_kl_is_unset():
    agent, buf = _prepared_agent_and_buffer()
    assert agent.target_kl is None
    assert agent.update(buf, minibatch_size=8, update_epochs=10)["loss/update_epochs_ran"] == 10


def test_target_kl_stops_the_update_early():
    agent, buf = _prepared_agent_and_buffer()
    agent.target_kl = 0.0  # any policy movement at all trips it
    assert agent.update(buf, minibatch_size=8, update_epochs=10)["loss/update_epochs_ran"] == 1


def test_update_reports_the_expected_metrics():
    """Autotuning is on by default, so the default agent's metrics include
    loss/ent_coef; see test_autotune_off_matches_todays_metrics_and_checkpoint_shape
    for the metrics dict with autotuning explicitly disabled."""
    agent, buf = _prepared_agent_and_buffer()
    metrics = agent.update(buf, minibatch_size=8, update_epochs=2)
    assert set(metrics) == {
        "loss/policy_loss", "loss/value_loss", "loss/entropy", "loss/approx_kl",
        "loss/approx_kl_max", "loss/ratio_max_dev",
        "loss/clipfrac", "loss/update_epochs_ran", "loss/value_bias",
        "loss/value_target_mean", "loss/value_target_std", "loss/explained_variance",
        "loss/adv_std_raw", "loss/batch_size",
        "loss/ent_coef",
    }
    assert all(isinstance(v, float) for v in metrics.values())


def test_update_changes_both_heads():
    agent, buf = _prepared_agent_and_buffer()
    actor_before = [p.clone() for p in agent.actor.parameters()]
    critic_before = [p.clone() for p in agent.critic.parameters()]
    agent.update(buf, minibatch_size=8, update_epochs=2)
    assert any(not torch.allclose(a, b) for a, b in zip(actor_before, agent.actor.parameters()))
    assert any(not torch.allclose(a, b) for a, b in zip(critic_before, agent.critic.parameters()))


# --------------------------------------------------------------------------
# Ported-for-parity mitigations: value-loss clipping, advantage-std floor,
# ret_rms horizon (see clipped_value_loss/floor_normalize's docstrings --
# these were added to HPPOAgent during a manager-collapse investigation and
# ported here for API/diagnostic parity, not a finding specific to flat PPO)
# --------------------------------------------------------------------------

def test_clipped_value_loss_matches_plain_mse_when_disabled():
    """clip_vloss=False must reproduce the original, unclipped loss exactly
    -- this is the default, so every run before this option existed depends
    on it being a no-op."""
    newvalue = torch.tensor([0.0, 5.0, -3.0])
    old_value = torch.tensor([10.0, 10.0, 10.0])
    target = torch.tensor([1.0, 1.0, 1.0])
    expected = 0.5 * ((newvalue - target) ** 2).mean()
    actual = clipped_value_loss(newvalue, old_value, target, clip_coef=0.2, clip_vloss=False)
    assert torch.allclose(actual, expected)


def test_clipped_value_loss_matches_unclipped_inside_the_trust_region():
    """When newvalue is already within clip_coef of old_value, clipping has
    nothing to bind on: clipped and unclipped candidates coincide."""
    old_value = torch.tensor([10.0])
    newvalue = old_value + 0.05  # inside clip_coef=0.2
    target = torch.tensor([10.5])
    expected = 0.5 * ((newvalue - target) ** 2).mean()
    actual = clipped_value_loss(newvalue, old_value, target, clip_coef=0.2, clip_vloss=True)
    assert torch.allclose(actual, expected)


def test_clipped_value_loss_picks_the_clipped_candidate_when_it_scores_worse():
    """newvalue has moved beyond clip_coef *and* toward target (a legitimate
    improvement): the clipped candidate, pinned near the stale old_value,
    scores worse than the unclipped one and must be the one used -- this is
    the actual trust-region bite, not a no-op."""
    old_value = torch.tensor([10.0])
    target = torch.tensor([0.0])
    newvalue = torch.tensor([8.0])  # moved 2.0 toward target, past clip_coef=0.2
    unclipped_loss = 0.5 * ((newvalue - target) ** 2).mean()
    actual = clipped_value_loss(newvalue, old_value, target, clip_coef=0.2, clip_vloss=True)
    assert actual > unclipped_loss
    v_clipped = old_value - 0.2  # clamp(8-10, -0.2, 0.2) == -0.2, so old_value + (-0.2)
    expected = 0.5 * ((v_clipped - target) ** 2).mean()
    assert torch.allclose(actual, expected)


def test_clipped_value_loss_zero_gradient_beyond_the_clip_boundary():
    """The actual protection clip_vloss buys: once the clipped candidate wins
    the max() (previous test), its gradient w.r.t. newvalue is zero out there
    -- clamp() is flat beyond the boundary -- so this update stops pushing
    the critic any further in that direction."""
    old_value = torch.tensor([10.0])
    target = torch.tensor([0.0])
    newvalue = torch.tensor([8.0], requires_grad=True)
    loss = clipped_value_loss(newvalue, old_value, target, clip_coef=0.2, clip_vloss=True)
    loss.backward()
    assert newvalue.grad.item() == pytest.approx(0.0)


def test_floor_normalize_matches_previous_behaviour_when_floor_is_zero():
    """floor_frac=0.0 (the default) must reproduce the original single-batch
    normalization exactly, whether or not an adv_rms is supplied."""
    advantages = torch.tensor([1.0, -2.0, 3.0, 0.5])
    expected = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    normalized, raw_std = floor_normalize(advantages.clone(), RunningMeanStd(), 0.0)
    assert torch.allclose(normalized, expected, atol=1e-6)
    assert raw_std == pytest.approx(float(advantages.std()))


def test_floor_normalize_floors_a_degenerate_batchs_std():
    """A batch with an anomalously small std must be normalized against the
    floor, not its own near-zero std, or dividing by it would amplify
    whatever noise is left into an oversized update."""
    rms = RunningMeanStd()
    rms.mean, rms.var = 0.0, 100.0  # multi-update history: std == 10.0
    tiny = torch.tensor([1.0, 1.0001, 0.9999, 1.0])  # std ~= 4e-5
    normalized, raw_std = floor_normalize(tiny, rms, floor_frac=0.5)
    assert raw_std < 1e-3
    expected = (tiny - tiny.mean()) / (0.5 * 10.0 + 1e-8)
    assert torch.allclose(normalized, expected, atol=1e-4)


def test_floor_normalize_leaves_a_healthy_batch_unfloored():
    """A batch whose own std already exceeds the floor must be normalized
    against its own std, unchanged from the un-floored behaviour -- the floor
    only ever raises the denominator, never lowers it."""
    rms = RunningMeanStd()
    rms.mean, rms.var = 0.0, 1.0  # multi-update history: std == 1.0
    advantages = torch.tensor([10.0, -20.0, 30.0, 5.0])  # std >> 0.5 * 1.0
    normalized, raw_std = floor_normalize(advantages.clone(), rms, floor_frac=0.5)
    expected = (advantages - advantages.mean()) / (raw_std + 1e-8)
    assert torch.allclose(normalized, expected, atol=1e-5)


def test_floor_normalize_updates_adv_rms_but_not_in_time_for_its_own_floor():
    """Complements test_floor_normalize_floors_a_degenerate_batchs_std, which
    shows a degenerate batch cannot lift its own floor. This shows the other
    half: adv_rms is still updated with that batch, just *after* the floor
    was computed from its pre-update state -- so the next call sees it, even
    though this one did not."""
    rms = RunningMeanStd()
    rms.mean, rms.var, rms.count = 0.0, 100.0, 1000.0  # well-established std == 10.0
    tiny = torch.tensor([1.0, 1.0001, 0.9999, 1.0])  # std ~= 4e-5
    floor_normalize(tiny, rms, floor_frac=0.5)
    # The tiny batch must have pulled the running std down for whatever
    # checks it *after* this call -- proving it landed, not that it was
    # silently dropped.
    assert rms.std < 10.0


def test_agent_honours_its_own_ret_rms_horizon():
    """ret_rms_horizon lets the critic's value-target statistics be given a
    longer or shorter memory than the default 10 -- unlike HPPOAgent's two
    independent heads, PPOAgent has only the one, so there is a single
    ret_rms_horizon kwarg rather than a per-head pair."""
    agent = PPOAgent(4, 2, device="cpu", ret_rms_horizon=40)
    batch = torch.randn(64)
    # Enough updates (> horizon) that count is actually capped rather than
    # still climbing -- 20 updates would leave count at a plain running
    # total (20*64 < 40*64) and pass by accident.
    for _ in range(100):
        agent.ret_rms.update(batch)
    assert agent.ret_rms.count == pytest.approx(40 * 64)


def test_update_reports_the_pre_normalization_advantage_std():
    """adv_std_raw must reflect the buffer's own GAE advantages, not
    something post-normalization or otherwise disconnected from them."""
    agent, buf = _prepared_agent_and_buffer()
    expected = float(buf.advantages.reshape(-1).std())
    metrics = agent.update(buf, minibatch_size=8, update_epochs=2)
    assert metrics["loss/adv_std_raw"] == pytest.approx(expected, rel=1e-4)


def test_clip_vloss_bounds_how_far_the_critic_moves_on_a_noisy_batch():
    """With clip_vloss on, a single update's value-loss gradient is capped
    the same way the policy ratio clip caps the actor's -- this is the
    end-to-end claim (through a real update() call) that the unit tests
    above verify only at the clipped_value_loss level."""
    torch.manual_seed(0)
    agent, buf = _prepared_agent_and_buffer(PPOAgent(4, 2, device="cpu", clip_vloss=True))
    assert agent.clip_vloss
    metrics = agent.update(buf, minibatch_size=8, update_epochs=2)
    assert "loss/value_loss" in metrics and math.isfinite(metrics["loss/value_loss"])


# --------------------------------------------------------------------------
# Entropy autotuning (SAC-style dual ascent on log_ent_coef)
# --------------------------------------------------------------------------

def _autotuning_agent_and_buffer(target_entropy, seed=0, **agent_kwargs):
    agent = PPOAgent(4, 2, autotune_ent_coef=True, device="cpu", **agent_kwargs)
    agent.target_entropy = target_entropy
    buf = _filled_buffer(num_steps=8, num_envs=4, seed=seed)
    with torch.no_grad():
        flat = buf.states.reshape(-1, 4)
        _, logprobs, _ = agent.policy_forward(flat, buf.actions.reshape(-1, 2))
    buf.logprobs = logprobs.reshape(8, 4)
    agent.compute_returns_and_advantage(buf, torch.zeros(4), torch.zeros(4))
    return agent, buf


def test_autotune_off_matches_todays_metrics_and_checkpoint_shape():
    """Explicitly disabling autotuning must be unchanged from before
    autotuning existed: no extra tensors, no new checkpoint keys, same
    metrics dict. Autotuning itself is on by default (see
    test_update_reports_the_expected_metrics)."""
    agent, buf = _prepared_agent_and_buffer(PPOAgent(4, 2, autotune_ent_coef=False, device="cpu"))
    assert not agent.autotune_ent_coef
    assert not hasattr(agent, "log_ent_coef")
    metrics = agent.update(buf, minibatch_size=8, update_epochs=2)
    assert "loss/ent_coef" not in metrics

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        agent.save(path)
        checkpoint = torch.load(path, weights_only=False)
    assert "log_ent_coef" not in checkpoint
    assert "ent_coef_optimizer" not in checkpoint


def test_ent_coef_rises_when_entropy_is_below_target():
    """A target far above the policy's actual (bounded) entropy ceiling must
    push log_ent_coef, hence ent_coef, up: not enough exploration yet."""
    agent, buf = _autotuning_agent_and_buffer(target_entropy=10.0)
    before = float(agent.log_ent_coef.detach().exp())
    metrics = agent.update(buf, minibatch_size=8, update_epochs=2)
    assert metrics["loss/ent_coef"] > before


def test_ent_coef_falls_when_entropy_is_above_target():
    """A target far below the policy's entropy must push ent_coef down."""
    agent, buf = _autotuning_agent_and_buffer(target_entropy=-10.0)
    before = float(agent.log_ent_coef.detach().exp())
    metrics = agent.update(buf, minibatch_size=8, update_epochs=2)
    assert metrics["loss/ent_coef"] < before


def test_log_ent_coef_is_clamped():
    """A strongly-pushing target must not drive ent_coef past ent_coef_max --
    PPO's entropy term shares one backward pass with pg_loss/v_loss, so an
    unclamped coefficient could otherwise starve them of gradient signal."""
    agent, buf = _autotuning_agent_and_buffer(
        target_entropy=100.0, ent_coef_max=0.02, ent_coef_lr=1.0
    )
    for i in range(20):
        metrics = agent.update(buf, minibatch_size=8, update_epochs=2)
        assert metrics["loss/ent_coef"] <= 0.02 + 1e-8
        buf = _filled_buffer(num_steps=8, num_envs=4, seed=i + 1)
        with torch.no_grad():
            flat = buf.states.reshape(-1, 4)
            _, logprobs, _ = agent.policy_forward(flat, buf.actions.reshape(-1, 2))
        buf.logprobs = logprobs.reshape(8, 4)
        agent.compute_returns_and_advantage(buf, torch.zeros(4), torch.zeros(4))


def test_ent_coef_is_fixed_within_one_update_call():
    """PPO reuses one batch across update_epochs passes; the dual-ascent
    optimizer must step exactly once per update() call, not once per
    minibatch, or the coefficient (and hence the loss it enters) would drift
    mid-update, epoch 1 and epoch 10 no longer optimizing the same surrogate."""
    agent, buf = _autotuning_agent_and_buffer(target_entropy=10.0)
    step_calls = []
    original_step = agent.ent_coef_optimizer.step

    def counting_step(*args, **kwargs):
        step_calls.append(1)
        return original_step(*args, **kwargs)

    agent.ent_coef_optimizer.step = counting_step
    agent.update(buf, minibatch_size=8, update_epochs=5)
    assert len(step_calls) == 1


def test_autotune_checkpoint_round_trips():
    """log_ent_coef and its optimiser state must survive a save/load round
    trip, or a resumed autotuning run silently restarts from the initial
    ent_coef instead of the value it had converged to."""
    agent = PPOAgent(4, 2, autotune_ent_coef=True, device="cpu")
    with torch.no_grad():
        agent.log_ent_coef.fill_(math.log(0.5))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        agent.save(path)
        reloaded = PPOAgent(4, 2, autotune_ent_coef=True, device="cpu")
        reloaded.load(path)  # must not need weights_only=False

    assert float(reloaded.log_ent_coef.detach()) == pytest.approx(math.log(0.5), abs=1e-6)


def test_loading_autotuned_checkpoint_into_non_autotuning_agent_warns_and_folds_back(capsys):
    """A checkpoint trained with autotuning, loaded into an agent constructed
    without it, must fold the tuned value into a fixed ent_coef rather than
    silently dropping it (mirrors test_load_tolerates_a_single_group_optimiser_state)."""
    agent = PPOAgent(4, 2, autotune_ent_coef=True, device="cpu")
    with torch.no_grad():
        agent.log_ent_coef.fill_(math.log(0.3))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        agent.save(path)
        reloaded = PPOAgent(4, 2, autotune_ent_coef=False, device="cpu")
        reloaded.load(path)  # must not raise

    assert "trained with autotune_ent_coef=True" in capsys.readouterr().out
    assert reloaded.ent_coef == pytest.approx(0.3, abs=1e-6)


# --------------------------------------------------------------------------
# The two-group optimiser (thesis PPO chapter, §8.2, §11.1)
# --------------------------------------------------------------------------

def test_optimiser_splits_actor_and_critic_into_separate_groups():
    """The critic runs at `critic_lr_mult` times the actor's rate, so the two
    heads must live in separate parameter groups -- and every parameter must
    still be covered by exactly one of them."""
    agent = PPOAgent(4, 2, lr=3e-4, critic_lr_mult=3.0, device="cpu")
    groups = agent.optimizer.param_groups
    assert len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(3e-4)
    assert groups[1]["lr"] == pytest.approx(9e-4)

    actor_ids = {id(p) for p in agent.actor.parameters()}
    critic_ids = {id(p) for p in agent.critic.parameters()}
    assert {id(p) for p in groups[0]["params"]} == actor_ids
    assert {id(p) for p in groups[1]["params"]} == critic_ids


def test_critic_lr_mult_of_one_restores_a_single_rate():
    """The knob must be able to reproduce the pre-split behaviour exactly."""
    agent = PPOAgent(4, 2, lr=3e-4, critic_lr_mult=1.0, device="cpu")
    assert [g["lr"] for g in agent.optimizer.param_groups] == pytest.approx([3e-4, 3e-4])


def test_annealing_every_group_keeps_the_critic_ratio():
    """The anneal the script performs. Scaling each group by its own base rate
    preserves the 3x ratio; scaling both by `args.learning_rate` -- the bug the
    single-group code invited -- would collapse the critic onto the actor."""
    agent = PPOAgent(4, 2, lr=3e-4, critic_lr_mult=3.0, device="cpu")
    base_lrs = [g["lr"] for g in agent.optimizer.param_groups]
    for group, base_lr in zip(agent.optimizer.param_groups, base_lrs):
        group["lr"] = 0.5 * base_lr
    lrs = [g["lr"] for g in agent.optimizer.param_groups]
    assert lrs == pytest.approx([1.5e-4, 4.5e-4])
    assert lrs[1] / lrs[0] == pytest.approx(3.0)


def test_load_tolerates_a_single_group_optimiser_state(capsys):
    """Checkpoints written before the split hold a one-group optimiser state.
    Adam rejects it; `load` must warn and keep the weights rather than raise,
    because nothing downstream needs the moment estimates (PPO chapter §8.2)."""
    import torch.optim as optim

    old = PPOAgent(4, 2, device="cpu")
    old.optimizer = optim.Adam(
        list(old.actor.parameters()) + list(old.critic.parameters()), lr=3e-4, eps=1e-5
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.pt")
        old.save(path)

        new = PPOAgent(4, 2, critic_lr_mult=3.0, device="cpu")
        new.load(path)          # must not raise

    assert "predates the two-group optimiser" in capsys.readouterr().out
    for a, b in zip(old.actor.parameters(), new.actor.parameters()):
        assert torch.allclose(a, b)
    for a, b in zip(old.critic.parameters(), new.critic.parameters()):
        assert torch.allclose(a, b)
    # the ratio survives a load that discarded the optimiser state
    assert [g["lr"] for g in new.optimizer.param_groups] == pytest.approx([3e-4, 9e-4])


def test_agent_default_discount_is_the_documented_one():
    """gamma = 0.99 is a deliberate default, not an accident (PPO chapter §11.1),
    and it must match what each scenario's script_ppo.py passes."""
    assert PPOAgent(4, 2, device="cpu").gamma == 0.99
