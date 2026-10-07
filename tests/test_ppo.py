"""Tests of the in-house PPO (decision log D5, D15, D19); they need PyTorch.

The strongest check is the comparison with the pre-alignment implementation
(``algorithms/ppo/ppo.py``, kept until step 8 for this purpose, D21): given the
same weights, batch and seed, the update must end on bit-identical weights.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
stats = pytest.importorskip("scipy.stats")

from hrlmpc.config import env_config_from_dict  # noqa: E402
from hrlmpc.env import NavigationEnv  # noqa: E402
from hrlmpc.ppo import LOG_PROB_EPS, Batch, PPOAgent, RunningMeanStd, make_batch  # noqa: E402
from hrlmpc.ppo_config import NetworkConfig, UpdateConfig  # noqa: E402
from hrlmpc.rollout import collect_rollout, compute_gae  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NETWORK = NetworkConfig(hidden_sizes=(64, 64))
_UPDATE = UpdateConfig(
    discount=0.99,
    epochs=10,
    minibatches=4,
    learning_rate=3e-4,
    anneal_learning_rate=True,
    critic_lr_mult=3.0,
    adam_eps=1e-5,
    gae_lambda=0.95,
    clip_coef=0.2,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    value_norm_horizon=10,
)


def _update(**changes: Any) -> UpdateConfig:
    return dataclasses.replace(_UPDATE, **changes)


def _agent(seed: int = 0, **changes: Any) -> PPOAgent:
    torch.manual_seed(seed)
    return PPOAgent(4, 2, NETWORK, _update(**changes))


def _obs(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1.0, 1.0, (n, 4))


def _legacy() -> ModuleType:
    """The pre-alignment flat PPO, loaded from its file with ``algorithms/`` on the path for its ``common`` module."""
    algorithms = REPO / "algorithms"
    source = algorithms / "ppo" / "ppo.py"
    if not source.exists():
        pytest.skip("the pre-alignment PPO is no longer in the repository")
    sys.path.insert(0, str(algorithms))
    try:
        spec = importlib.util.spec_from_file_location("pre_alignment_ppo", source)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(algorithms))
    return module


def _random_batch(agent: PPOAgent, n: int, seed: int) -> Batch:
    """Observations, actions sampled from the agent's policy, and arbitrary value targets."""
    gen = torch.Generator().manual_seed(seed)
    obs = torch.rand(n, 4, generator=gen) * 2 - 1
    actions, log_probs = agent.act(obs.numpy())
    values = torch.randn(n, generator=gen) * 100
    returns = values + torch.randn(n, generator=gen) * 30
    advantages = returns - values + torch.randn(n, generator=gen)
    return Batch(
        obs=obs,
        actions=torch.as_tensor(actions, dtype=torch.float32),
        log_probs=torch.as_tensor(log_probs, dtype=torch.float32),
        values=values,
        returns=returns,
        advantages=advantages,
    )


# --------------------------------------------------------------------------
# Policy: a Beta per axis on the square (D15)
# --------------------------------------------------------------------------


def test_the_concentrations_are_at_least_one() -> None:
    """softplus + 1 >= 1; very negative logits give exactly 1 in float32, the uniform density."""
    agent = _agent()
    with torch.no_grad():
        last = agent.actor.net[-1]
        last.weight.zero_()
        last.bias.fill_(-50.0)
    alpha, beta = agent.concentrations(_obs(100))
    assert alpha.shape == beta.shape == (100, 2)
    assert np.all(alpha == 1.0) and np.all(beta == 1.0)
    actions = np.random.default_rng(0).uniform(-0.9, 0.9, (100, 2))
    np.testing.assert_allclose(agent.log_prob(_obs(100), actions), -2.0 * math.log(2.0), rtol=1e-6)
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.normal_(0.0, 3.0)
    alpha, beta = agent.concentrations(_obs(500))
    assert np.all(alpha >= 1.0) and np.all(beta >= 1.0)


def test_the_initial_policy_is_nearly_uniform() -> None:
    """The output layer starts small, so alpha and beta start near softplus(0) + 1 on every axis."""
    alpha, beta = _agent().concentrations(_obs(200))
    expected = math.log(2.0) + 1.0
    assert np.allclose(alpha, expected, atol=0.05) and np.allclose(beta, expected, atol=0.05)


def test_log_probabilities_are_those_of_the_beta_on_the_square() -> None:
    """The density of x = 2 y - 1 with y ~ Beta(alpha, beta), per axis: p(y) / 2."""
    agent = _agent(3)
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.mul_(5.0)
    obs = _obs(300, seed=1)
    actions = np.random.default_rng(2).uniform(-0.999, 0.999, (300, 2))
    alpha, beta = agent.concentrations(obs)
    expected = (stats.beta.logpdf((actions + 1.0) / 2.0, alpha, beta) - math.log(2.0)).sum(axis=1)
    np.testing.assert_allclose(agent.log_prob(obs, actions), expected, rtol=1e-4, atol=1e-4)


def test_sampled_actions_stay_inside_the_square_with_finite_log_probabilities() -> None:
    """Even a nearly deterministic policy samples admissible actions with finite log-probabilities."""
    agent = _agent()
    with torch.no_grad():
        last = agent.actor.net[-1]
        last.weight.zero_()
        last.bias.copy_(torch.tensor([1000.0, -1000.0, -1000.0, 1000.0]))  # alpha = (1001, 1), beta = (1, 1001)
    actions, log_probs = agent.act(_obs(2000))
    bound = 1.0 - 2.0 * LOG_PROB_EPS
    assert np.all(np.abs(actions) <= bound + 1e-7) and np.all(np.isfinite(log_probs))
    assert np.any(actions[:, 0] >= bound - 1e-6) and np.any(actions[:, 1] <= -bound + 1e-6)  # the clamp acted
    assert actions[:, 0].mean() > 0.9 and actions[:, 1].mean() < -0.9
    np.testing.assert_allclose(agent.log_prob(_obs(2000), actions), log_probs, rtol=1e-6, atol=1e-5)
    env = NavigationEnv(_env_config(), num_envs=8, seed=0)
    env.reset()
    env.step(actions[:8])  # admissible for the environment


def test_the_deterministic_action_is_the_mean_mapped_to_the_square() -> None:
    agent = _agent(5)
    obs = _obs(50, seed=3)
    alpha, beta = agent.concentrations(obs)
    expected = 2.0 * alpha / (alpha + beta) - 1.0
    np.testing.assert_allclose(agent.deterministic_action(obs), expected, rtol=1e-6, atol=1e-6)


def test_acting_draws_from_the_torch_generator() -> None:
    agent = _agent()
    obs = _obs(10)
    torch.manual_seed(11)
    first = agent.act(obs)
    torch.manual_seed(11)
    again = agent.act(obs)
    np.testing.assert_array_equal(first[0], again[0])
    np.testing.assert_array_equal(first[1], again[1])
    assert not np.array_equal(agent.act(obs)[0], agent.act(obs)[0])  # the generator moves on


# --------------------------------------------------------------------------
# Critic and its target normalisation
# --------------------------------------------------------------------------


def test_values_undo_the_normalisation_of_the_targets() -> None:
    agent = _agent()
    agent.ret_rms.mean, agent.ret_rms.var = 700.0, 170.0**2
    obs = _obs(20)
    with torch.no_grad():
        raw = agent.critic(torch.as_tensor(obs, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(agent.value(obs), raw * (170.0 + 1e-8) + 700.0, rtol=1e-6)


def test_the_running_statistics_match_the_pre_alignment_ones() -> None:
    legacy = _legacy()
    old, new = legacy.RunningMeanStd(), RunningMeanStd(horizon=10)
    gen = torch.Generator().manual_seed(0)
    for i in range(30):
        x = torch.randn(1024, generator=gen) * (i + 1) + 10.0 * i
        old.update(x)
        new.update(x)
    assert new.state_dict() == old.state_dict() and new.std == old.std


# --------------------------------------------------------------------------
# Update
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("steps", "envs", "max_grad_norm"), [(16, 4, 0.5), (14, 5, 0.05)])
def test_the_update_reproduces_the_pre_alignment_update_exactly(steps: int, envs: int, max_grad_norm: float) -> None:
    """Same weights, batches and seed: bit-identical weights, statistics and metrics (D19).

    The second case splits 70 samples unevenly and clips the gradients of both
    networks; two consecutive updates carry Adam's moments and the target
    statistics over.
    """
    legacy = _legacy()
    torch.manual_seed(0)
    old = legacy.PPOAgent(
        4, 2, act_limit_low=-1.0, act_limit_high=1.0, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
        ent_coef=0.01, max_grad_norm=max_grad_norm, critic_lr_mult=3.0, device="cpu",
    )
    new = _agent(epochs=3, max_grad_norm=max_grad_norm)
    new.actor.load_state_dict(old.actor.state_dict())
    new.critic.load_state_dict(old.critic.state_dict())
    initial = [p.clone() for p in new.actor.parameters()]
    gen = torch.Generator().manual_seed(1)
    n = steps * envs
    for round_ in range(2):
        obs = torch.rand(steps, envs, 4, generator=gen) * 2 - 1
        with torch.no_grad():
            actions, log_probs, _ = old.policy_forward(obs.reshape(-1, 4))
        values = torch.randn(n, generator=gen) * 100
        returns = values + torch.randn(n, generator=gen) * 30
        advantages = returns - values + torch.randn(n, generator=gen)
        buffer = legacy.RolloutBuffer(steps, envs, 4, 2, "cpu")
        buffer.states = obs.clone()
        buffer.actions = actions.reshape(steps, envs, 2).clone()
        buffer.logprobs = log_probs.reshape(steps, envs).clone()
        buffer.values = values.reshape(steps, envs).clone()
        buffer.returns = returns.reshape(steps, envs).clone()
        buffer.advantages = advantages.reshape(steps, envs).clone()
        batch = Batch(
            obs=obs.reshape(-1, 4).clone(), actions=actions.clone(), log_probs=log_probs.clone(),
            values=values.clone(), returns=returns.clone(), advantages=advantages.clone(),
        )
        torch.manual_seed(7 + round_)
        old_metrics = old.update(buffer, 4, 3)
        torch.manual_seed(7 + round_)
        new_metrics = new.update(batch)
        for module_old, module_new in ((old.actor, new.actor), (old.critic, new.critic)):
            pairs = zip(module_old.state_dict().items(), module_new.state_dict().items(), strict=True)
            for (name, a), (name_new, b) in pairs:
                assert name == name_new and torch.equal(a, b), (round_, name)
        assert new.ret_rms.state_dict() == old.ret_rms.state_dict()
        for key, value in old_metrics.items():
            mine = new_metrics["train/" + key.split("/", 1)[1]]
            assert mine == value or (math.isnan(mine) and math.isnan(value)), (round_, key)
    changed = [not torch.equal(a, b) for a, b in zip(new.actor.parameters(), initial, strict=True)]
    assert any(changed)  # the updates did something


def test_the_networks_start_like_the_pre_alignment_ones() -> None:
    legacy = _legacy()
    torch.manual_seed(3)
    old_actor, old_critic = legacy.ActorNetwork(4, 2), legacy.CriticNetwork(4)
    new = _agent(3)
    for old, mine in ((old_actor, new.actor), (old_critic, new.critic)):
        for (name, a), (name_new, b) in zip(old.state_dict().items(), mine.state_dict().items(), strict=True):
            assert name == name_new and torch.equal(a, b), name


def test_gae_agrees_with_the_pre_alignment_gae() -> None:
    """Explicit bootstrap values and the old convention (truncations bootstrapped through the reward) agree."""
    legacy = _legacy()
    steps, envs, gamma = 12, 3, 0.99
    rng = np.random.default_rng(0)
    rewards = rng.normal(0.0, 10.0, (steps, envs))
    values = rng.normal(0.0, 50.0, (steps, envs))
    terminated = rng.random((steps, envs)) < 0.1
    truncated = (rng.random((steps, envs)) < 0.1) & ~terminated
    done = terminated | truncated
    reached = rng.normal(0.0, 50.0, (steps, envs))  # value of the state a truncated transition reached
    last = rng.normal(0.0, 50.0, envs)  # value of the observation after the rollout
    following = np.vstack([values[1:], last[None]])
    next_values = np.where(done, np.where(truncated, reached, 0.0), following)
    advantages, returns = compute_gae(rewards, values, next_values, terminated, done, discount=gamma, gae_lambda=0.95)
    buffer = legacy.RolloutBuffer(steps, envs, 4, 2, "cpu")
    buffer.rewards = torch.tensor(rewards + np.where(truncated, gamma * reached, 0.0), dtype=torch.float64)
    buffer.values = torch.tensor(values, dtype=torch.float64)
    buffer.dones = torch.tensor(done, dtype=torch.float64)
    buffer.advantages = torch.zeros(steps, envs, dtype=torch.float64)
    buffer.compute_returns_and_advantage(torch.tensor(last, dtype=torch.float64), gamma=gamma, gae_lambda=0.95)
    np.testing.assert_allclose(advantages, buffer.advantages.numpy(), rtol=1e-9, atol=1e-7)
    np.testing.assert_allclose(returns, buffer.returns.numpy(), rtol=1e-9, atol=1e-7)


def test_the_update_moves_the_policy_towards_positive_advantages() -> None:
    """A sign check independent of the comparison above."""
    agent = _agent(learning_rate=1e-2, epochs=4, minibatches=2)
    obs = _obs(512, seed=4)
    torch.manual_seed(1)
    actions, log_probs = agent.act(obs)
    advantages = torch.as_tensor(np.where(actions[:, 0] > 0.0, 1.0, -1.0), dtype=torch.float32)
    batch = Batch(
        obs=torch.as_tensor(obs, dtype=torch.float32),
        actions=torch.as_tensor(actions, dtype=torch.float32),
        log_probs=torch.as_tensor(log_probs, dtype=torch.float32),
        values=torch.zeros(512),
        returns=advantages.clone(),
        advantages=advantages,
    )
    before = agent.deterministic_action(obs)[:, 0].mean()
    agent.update(batch)
    assert agent.deterministic_action(obs)[:, 0].mean() > before + 0.01


def test_the_update_refuses_batches_too_small_to_split() -> None:
    """Empty minibatches would turn the losses, and then the weights, into NaN."""
    agent = _agent()
    with pytest.raises(ValueError, match="minibatches"):
        agent.update(_random_batch(agent, 3, seed=0))  # 3 samples, 4 minibatches
    single = _agent(minibatches=1)
    with pytest.raises(ValueError, match="minibatches"):
        single.update(_random_batch(single, 1, seed=0))


def test_the_update_is_deterministic_given_the_seed() -> None:
    a, b = _agent(3, epochs=2), _agent(3, epochs=2)
    batch = _random_batch(a, 256, seed=9)
    torch.manual_seed(5)
    metrics_a = a.update(batch)
    torch.manual_seed(5)
    metrics_b = b.update(batch)
    assert metrics_a == metrics_b
    for p, q in zip(a.actor.parameters(), b.actor.parameters(), strict=True):
        assert torch.equal(p, q)


def test_the_update_reports_its_metrics_and_changes_both_networks() -> None:
    agent = _agent(epochs=2)
    actor_before = [p.clone() for p in agent.actor.parameters()]
    critic_before = [p.clone() for p in agent.critic.parameters()]
    metrics = agent.update(_random_batch(agent, 256, seed=2))
    expected = {
        "policy_loss", "value_loss", "entropy", "approx_kl", "approx_kl_max", "ratio_max_dev", "clipfrac",
        "value_bias", "value_target_mean", "value_target_std", "explained_variance", "adv_std_raw",
    }
    assert set(metrics) == {f"train/{k}" for k in expected}
    assert all(isinstance(v, float) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters(), strict=True))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, agent.critic.parameters(), strict=True))
    assert not set(map(id, agent.actor.parameters())) & set(map(id, agent.critic.parameters()))


def test_annealing_scales_every_learning_rate() -> None:
    agent = _agent()
    np.testing.assert_allclose(agent.learning_rates, [3e-4, 9e-4], rtol=1e-12)
    agent.set_learning_rate_fraction(0.25)
    np.testing.assert_allclose(agent.learning_rates, [0.75e-4, 2.25e-4], rtol=1e-12)
    with pytest.raises(ValueError, match="fraction"):
        agent.set_learning_rate_fraction(1.5)


def test_a_checkpoint_restores_the_agent(tmp_path: Path) -> None:
    agent = _agent(1, epochs=2)
    agent.update(_random_batch(agent, 128, seed=3))
    agent.set_learning_rate_fraction(0.5)
    path = tmp_path / "agent.pt"
    agent.save(path)
    restored = _agent(2, epochs=2)
    restored.load(path)
    for mine, theirs in ((agent.actor, restored.actor), (agent.critic, restored.critic)):
        for a, b in zip(mine.state_dict().values(), theirs.state_dict().values(), strict=True):
            assert torch.equal(a, b)
    assert restored.ret_rms.state_dict() == agent.ret_rms.state_dict()
    assert restored.learning_rates == agent.learning_rates
    state_a, state_b = agent.optimizer.state_dict()["state"], restored.optimizer.state_dict()["state"]
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        for name in state_a[key]:
            assert torch.equal(torch.as_tensor(state_a[key][name]), torch.as_tensor(state_b[key][name]))
    obs = _obs(30)
    np.testing.assert_array_equal(agent.deterministic_action(obs), restored.deterministic_action(obs))
    np.testing.assert_array_equal(agent.value(obs), restored.value(obs))


# --------------------------------------------------------------------------
# From a rollout to a batch
# --------------------------------------------------------------------------


def _env_config(**sections: dict[str, Any]) -> Any:
    data = copy.deepcopy(yaml.safe_load((REPO / "configs" / "env" / "tunnel.yaml").read_text(encoding="utf-8")))
    for section, values in sections.items():
        data[section].update(values)
    return env_config_from_dict(data)


def test_a_batch_is_the_rollout_flattened_time_major() -> None:
    agent = _agent()
    env = NavigationEnv(_env_config(task={"horizon": 5}), num_envs=3, seed=0)
    rollout, _ = collect_rollout(env, env.reset(), 7, agent.act, agent.value)
    advantages, returns = compute_gae(
        rollout.rewards, rollout.values, rollout.next_values, rollout.terminated, rollout.done,
        discount=0.99, gae_lambda=0.95,
    )
    batch = make_batch(rollout, advantages, returns)
    assert batch.obs.shape == (21, 4) and batch.actions.shape == (21, 2) and batch.obs.dtype == torch.float32
    for t, b in ((0, 0), (2, 1), (6, 2)):
        i = 3 * t + b
        np.testing.assert_allclose(batch.obs[i].numpy(), rollout.obs[t, b], rtol=1e-6)
        np.testing.assert_allclose(batch.actions[i].numpy(), rollout.actions[t, b], rtol=0, atol=0)
        assert batch.log_probs[i].item() == pytest.approx(rollout.log_probs[t, b], rel=1e-6)
        assert batch.advantages[i].item() == pytest.approx(advantages[t, b], rel=1e-6, abs=1e-4)
        assert batch.returns[i].item() == pytest.approx(returns[t, b], rel=1e-6, abs=1e-4)
    agent.update(batch)  # the batch is what the update consumes
