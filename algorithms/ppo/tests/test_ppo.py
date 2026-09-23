"""Regression tests for the PPO implementation.

Each test pins down a property that is either non-obvious from the source or
easy to break silently -- the GAE indexing convention, the value
standardisation round trip, and the metric definitions. The building blocks
PPO shares with the other algorithms (ScaledBeta's corrections,
RunningMeanStd, normalize_obs) are tested once, in algorithms/tests/test_common.py.
See the thesis PPO chapter (kept outside this repo) for the reasoning
behind each.
"""

import math
import os
import tempfile

import numpy as np
import pytest
import torch

from ppo import (
    ActorNetwork,
    PPOAgent,
    RolloutBuffer,
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


def test_act_is_the_deterministic_action_without_a_critic_pass():
    """Evaluation and the solved-check only need the action; they used to go
    through get_action_and_value and throw its value away."""
    agent = PPOAgent(4, 2, device="cpu")
    state = torch.randn(3, 4)
    expected = agent.policy_forward(state, deterministic=True)[0]

    def no_critic(*_):
        raise AssertionError("act() must not evaluate the critic")

    agent.critic.forward = no_critic
    assert torch.equal(agent.act(state), expected)


# --------------------------------------------------------------------------
# GAE
# --------------------------------------------------------------------------

def test_gae_matches_transition_indexed_reference():
    """dones[t] means 'transition t ended an episode', so the mask is 1-dones[t].

    CleanRL stores the other convention and masks with dones[t+1]; this test
    pins ours down so the difference is never 'fixed' by mistake.
    """
    buf = _filled_buffer()
    next_value = torch.randn(2)
    gamma, lam = 0.999, 0.95
    buf.compute_returns_and_advantage(next_value, gamma, lam)

    expected = torch.zeros(buf.num_steps, buf.num_envs)
    last = torch.zeros(buf.num_envs)
    for t in reversed(range(buf.num_steps)):
        nonterminal = 1.0 - buf.dones[t]
        next_v = next_value if t == buf.num_steps - 1 else buf.values[t + 1]
        delta = buf.rewards[t] + gamma * next_v * nonterminal - buf.values[t]
        last = delta + gamma * lam * nonterminal * last
        expected[t] = last
    assert torch.allclose(buf.advantages, expected, atol=1e-5)


def test_gae_returns_identity_holds():
    """returns = advantages + values, which update() relies on."""
    buf = _filled_buffer()
    buf.compute_returns_and_advantage(torch.randn(2), 0.999, 0.95)
    assert torch.allclose(buf.returns, buf.advantages + buf.values, atol=1e-5)


def test_gae_stops_bootstrapping_at_a_done():
    """A terminal transition must not leak value from the next episode."""
    buf = RolloutBuffer(2, 1, 4, 2, "cpu")
    buf.rewards = torch.tensor([[5.0], [7.0]])
    buf.values = torch.zeros(2, 1)
    buf.dones = torch.tensor([[1.0], [0.0]])  # transition 0 ended the episode
    buf.compute_returns_and_advantage(torch.tensor([100.0]), 0.999, 0.95)
    # Step 0 is terminal: its advantage is its own reward and nothing else.
    assert buf.advantages[0, 0].item() == pytest.approx(5.0, abs=1e-5)


def test_gae_does_not_bootstrap_past_a_done_on_the_last_transition():
    """The case a separate `next_done` argument used to cover: when the last
    stored transition ended its episode, the observation after the rollout
    belongs to the next episode, and its value must not leak in. dones[-1]
    alone has to say so."""
    buf = RolloutBuffer(2, 1, 4, 2, "cpu")
    buf.rewards = torch.tensor([[5.0], [7.0]])
    buf.values = torch.zeros(2, 1)
    buf.dones = torch.tensor([[0.0], [1.0]])  # the last transition ended the episode
    buf.compute_returns_and_advantage(torch.tensor([100.0]), 0.999, 0.95)
    assert buf.advantages[1, 0].item() == pytest.approx(7.0, abs=1e-5)


def test_gae_parameters_are_required():
    """They used to default to 0.99 while every call site passed 0.999."""
    buf = _filled_buffer()
    with pytest.raises(TypeError):
        buf.compute_returns_and_advantage(torch.randn(2))


def test_agent_is_the_single_source_of_the_discount():
    """agent.gamma/gae_lambda drive GAE; they used to be set and never read."""
    agent = PPOAgent(4, 2, gamma=0.5, gae_lambda=0.5, device="cpu")
    buf_a, buf_b = _filled_buffer(), _filled_buffer()
    next_value = torch.randn(2)

    agent.compute_returns_and_advantage(buf_a, next_value)
    buf_b.compute_returns_and_advantage(next_value, 0.5, 0.5)
    assert torch.allclose(buf_a.advantages, buf_b.advantages, atol=1e-6)

    agent.gamma = 0.999
    agent.compute_returns_and_advantage(buf_a, next_value)
    assert not torch.allclose(buf_a.advantages, buf_b.advantages, atol=1e-6)


def test_agent_default_discount_is_the_documented_one():
    """gamma = 0.99 is a deliberate default, not an accident (PPO chapter §11.1),
    and it must match ppo_train.py's --gamma default."""
    assert PPOAgent(4, 2, device="cpu").gamma == 0.99


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


# --------------------------------------------------------------------------
# Observation map and checkpoint round trip
# --------------------------------------------------------------------------

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


def test_checkpoint_from_the_autotuning_era_still_loads():
    """Every checkpoint trained before the entropy autotuner was removed
    carries log_ent_coef and its optimiser state. Loading one must restore
    the weights and leave the fixed ent_coef alone, not raise."""
    agent = PPOAgent(4, 2, device="cpu")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.pt")
        agent.save(path)
        checkpoint = torch.load(path, weights_only=False)
        log_ent_coef = torch.tensor(math.log(0.3), requires_grad=True)
        checkpoint["log_ent_coef"] = math.log(0.3)
        checkpoint["ent_coef_optimizer"] = torch.optim.Adam([log_ent_coef], lr=3e-4).state_dict()
        torch.save(checkpoint, path)

        reloaded = PPOAgent(4, 2, device="cpu")
        reloaded.load(path)  # must not raise, and must not need weights_only=False

    assert reloaded.ent_coef == 0.01
    for a, b in zip(agent.actor.parameters(), reloaded.actor.parameters()):
        assert torch.equal(a, b)


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
    agent.compute_returns_and_advantage(buf, torch.zeros(4))
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


def test_update_reports_the_expected_metrics():
    agent, buf = _prepared_agent_and_buffer()
    metrics = agent.update(buf, minibatch_size=8, update_epochs=2)
    assert set(metrics) == {
        "loss/policy_loss", "loss/value_loss", "loss/entropy", "loss/approx_kl",
        "loss/approx_kl_max", "loss/ratio_max_dev",
        "loss/clipfrac", "loss/value_bias",
        "loss/value_target_mean", "loss/value_target_std", "loss/explained_variance",
        "loss/adv_std_raw",
    }
    assert all(isinstance(v, float) for v in metrics.values())


def test_actor_and_critic_share_no_parameters():
    """The premise behind PPOAgent having no vf_coef: with disjoint
    parameters the critic's gradient comes from the value loss alone, so a
    constant weight on that loss is undone by Adam instead of trading the
    two heads off against each other."""
    agent = PPOAgent(4, 2, device="cpu")
    actor_ids = {id(p) for p in agent.actor.parameters()}
    critic_ids = {id(p) for p in agent.critic.parameters()}
    assert actor_ids.isdisjoint(critic_ids)


def test_update_changes_both_heads():
    agent, buf = _prepared_agent_and_buffer()
    actor_before = [p.clone() for p in agent.actor.parameters()]
    critic_before = [p.clone() for p in agent.critic.parameters()]
    agent.update(buf, minibatch_size=8, update_epochs=2)
    assert any(not torch.allclose(a, b) for a, b in zip(actor_before, agent.actor.parameters()))
    assert any(not torch.allclose(a, b) for a, b in zip(critic_before, agent.critic.parameters()))


def test_update_reports_the_pre_normalization_advantage_std():
    """adv_std_raw must reflect the buffer's own GAE advantages, not
    something post-normalization or otherwise disconnected from them."""
    agent, buf = _prepared_agent_and_buffer()
    expected = float(buf.advantages.reshape(-1).std())
    metrics = agent.update(buf, minibatch_size=8, update_epochs=2)
    assert metrics["loss/adv_std_raw"] == pytest.approx(expected, rel=1e-4)


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
