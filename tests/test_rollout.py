"""Tests of the rollout collection and of generalized advantage estimation (decision log D6, D13, D19)."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from hrlmpc.config import EnvConfig, env_config_from_dict
from hrlmpc.env import NavigationEnv
from hrlmpc.rollout import collect_rollout, compute_gae

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "env"
WEIGHTS = np.array([1.0, 2.0, 3.0, 4.0])


def _config(layout: str = "tunnel", **sections: dict[str, Any]) -> EnvConfig:
    data = copy.deepcopy(yaml.safe_load((CONFIGS / f"{layout}.yaml").read_text(encoding="utf-8")))
    for section, values in sections.items():
        data[section].update(values)
    return env_config_from_dict(data)


def _reference_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    done: np.ndarray,
    discount: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE from its definition: a sum of discounted TD errors up to the end of the episode or rollout."""
    steps, batch = rewards.shape
    discount = np.broadcast_to(discount, (steps, batch))
    advantages = np.zeros((steps, batch))
    for b in range(batch):
        for t in range(steps):
            total, weight = 0.0, 1.0
            for k in range(t, steps):
                delta = rewards[k, b] + discount[k, b] * (1.0 - terminated[k, b]) * next_values[k, b] - values[k, b]
                total += weight * delta
                if done[k, b]:
                    break
                weight *= discount[k, b] * lam
            advantages[t, b] = total
    return advantages, advantages + values


def _random_rollout(seed: int, steps: int = 9, batch: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    terminated = rng.random((steps, batch)) < 0.15
    truncated = (rng.random((steps, batch)) < 0.15) & ~terminated
    return {
        "rewards": rng.normal(0.0, 10.0, (steps, batch)),
        "values": rng.normal(0.0, 50.0, (steps, batch)),
        "next_values": rng.normal(0.0, 50.0, (steps, batch)),
        "terminated": terminated,
        "done": terminated | truncated,
    }


# --------------------------------------------------------------------------
# Generalized advantage estimation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("lam", [0.0, 0.95, 1.0])
def test_gae_matches_its_definition(seed: int, lam: float) -> None:
    r = _random_rollout(seed)
    advantages, returns = compute_gae(**r, discount=0.99, gae_lambda=lam)
    expected_adv, expected_ret = _reference_gae(**r, discount=np.array(0.99), lam=lam)
    np.testing.assert_allclose(advantages, expected_adv, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(returns, expected_ret, rtol=1e-12, atol=1e-9)


def test_gae_accepts_one_discount_per_transition() -> None:
    """A Manager's transition spans several steps and is discounted by gamma to their number (step 8)."""
    r = _random_rollout(7)
    discount = 0.99 ** np.random.default_rng(8).integers(1, 11, r["rewards"].shape)
    advantages, returns = compute_gae(**r, discount=discount, gae_lambda=0.95)
    expected_adv, expected_ret = _reference_gae(**r, discount=discount, lam=0.95)
    np.testing.assert_allclose(advantages, expected_adv, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(returns, expected_ret, rtol=1e-12, atol=1e-9)


def test_with_lambda_one_the_returns_are_the_bootstrapped_discounted_returns() -> None:
    rewards = np.array([[1.0], [2.0], [3.0]])
    values = np.array([[5.0], [-4.0], [7.0]])
    next_values = np.array([[-4.0], [7.0], [10.0]])
    flags = np.zeros((3, 1), dtype=bool)
    _, returns = compute_gae(rewards, values, next_values, flags, flags, discount=0.9, gae_lambda=1.0)
    expected = [1.0 + 0.9 * 2.0 + 0.81 * 3.0 + 0.729 * 10.0, 2.0 + 0.9 * 3.0 + 0.81 * 10.0, 3.0 + 0.9 * 10.0]
    np.testing.assert_allclose(returns[:, 0], expected, rtol=1e-12)


def test_truncated_transitions_bootstrap_and_terminated_ones_do_not() -> None:
    """Partial-episode bootstrapping: a time limit is not a terminal state (D13 has no time in the observation)."""
    rewards, values, next_values = np.ones((1, 2)), np.zeros((1, 2)), np.full((1, 2), 10.0)
    terminated = np.array([[True, False]])
    done = np.array([[True, True]])  # agent 0 terminated, agent 1 truncated
    advantages, _ = compute_gae(rewards, values, next_values, terminated, done, discount=0.99, gae_lambda=0.95)
    np.testing.assert_allclose(advantages[0], [1.0, 1.0 + 0.99 * 10.0], rtol=1e-12)


def test_an_episode_end_stops_the_accumulation() -> None:
    rewards = np.array([[0.0], [100.0]])
    values = np.zeros((2, 1))
    next_values = np.zeros((2, 1))
    done = np.array([[True], [False]])  # a truncation at t = 0, bootstrapped from a value of 0
    advantages, _ = compute_gae(rewards, values, next_values, np.zeros_like(done), done, discount=1.0, gae_lambda=1.0)
    assert advantages[0, 0] == 0.0 and advantages[1, 0] == 100.0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"rewards": np.zeros((3, 4))}, "shape"),
        ({"terminated": np.ones((9, 4), dtype=bool), "done": np.zeros((9, 4), dtype=bool)}, "terminated"),
        ({"done": np.zeros((9, 4))}, "boolean"),
    ],
)
def test_gae_rejects_inconsistent_inputs(change: dict[str, np.ndarray], message: str) -> None:
    r = _random_rollout(0)
    r.update(change)
    with pytest.raises(ValueError, match=message):
        compute_gae(**r, discount=0.99, gae_lambda=0.95)


@pytest.mark.parametrize(("discount", "lam"), [(1.1, 0.95), (-0.1, 0.95), (math.nan, 0.95), (0.99, 1.5), (0.99, -0.5)])
def test_gae_rejects_parameters_outside_the_unit_interval(discount: float, lam: float) -> None:
    with pytest.raises(ValueError, match="discount|gae_lambda"):
        compute_gae(**_random_rollout(0), discount=discount, gae_lambda=lam)


def test_gae_rejects_a_discount_per_time_step_only() -> None:
    """A (T,) discount would broadcast over the agents when T == B; it must be (T, B)."""
    r = _random_rollout(0, steps=4, batch=4)
    with pytest.raises(ValueError, match="shape"):
        compute_gae(**r, discount=np.full(4, 0.99), gae_lambda=0.95)


def test_gae_rejects_integer_flags() -> None:
    r = _random_rollout(0)
    r["terminated"] = r["terminated"].astype(np.int64)
    with pytest.raises(ValueError, match="boolean"):
        compute_gae(**r, discount=0.99, gae_lambda=0.95)


# --------------------------------------------------------------------------
# Rollout collection
# --------------------------------------------------------------------------


def _value(obs: np.ndarray) -> np.ndarray:
    """A value function that tells every observation apart."""
    result: np.ndarray = np.asarray(obs) @ WEIGHTS + 0.5
    return result


def _full_thrust(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full thrust along +x, with a log-probability that records the observation."""
    return np.tile([1.0, 0.0], (len(obs), 1)), np.asarray(obs)[:, 0].copy()


def _mixed_env(horizon: int = 4, seed: int = 3) -> tuple[NavigationEnv, np.ndarray]:
    """Three agents: one terminates at the first step, one truncates, one drifts in between."""
    env = NavigationEnv(_config(task={"horizon": horizon}), num_envs=3, seed=seed)
    obs = env.reset(positions=[[9.95, 0.0], [0.5, 0.0], [5.0, 1.0]], velocities=[[1.0, 0.0], [0.0, 0.0], [0.0, 0.5]])
    return env, obs


def test_the_rollout_records_what_the_agents_saw_and_did() -> None:
    env, obs = _mixed_env()
    rollout, next_obs = collect_rollout(env, obs, 6, _full_thrust, _value)
    assert rollout.obs.shape == (6, 3, 4) and rollout.actions.shape == (6, 3, 2)
    np.testing.assert_array_equal(rollout.obs[0], obs)
    np.testing.assert_array_equal(rollout.actions, np.tile([1.0, 0.0], (6, 3, 1)))
    np.testing.assert_array_equal(rollout.log_probs, rollout.obs[..., 0])
    np.testing.assert_array_equal(rollout.values, _value(rollout.obs))
    assert rollout.num_samples == 18
    assert env.counter.env_steps == 18  # D6: every step of every agent
    twin, twin_obs = _mixed_env()
    np.testing.assert_array_equal(twin_obs, obs)
    for t in range(6):
        out = twin.step(np.tile([1.0, 0.0], (3, 1)))
        np.testing.assert_array_equal(rollout.rewards[t], out.reward)
        np.testing.assert_array_equal(rollout.terminated[t], out.terminated)
        np.testing.assert_array_equal(rollout.truncated[t], out.truncated)
        expected_next = np.where(out.terminated | out.truncated, 0.0, _value(out.obs))
        expected_next[out.truncated] = _value(out.final_obs[out.truncated])
        np.testing.assert_array_equal(rollout.next_values[t], expected_next)
        if t < 5:
            np.testing.assert_array_equal(rollout.obs[t + 1], out.obs)
    np.testing.assert_array_equal(next_obs, out.obs)


def test_the_rollout_bootstraps_truncations_from_the_state_reached() -> None:
    env, obs = _mixed_env(horizon=4)
    rollout, next_obs = collect_rollout(env, obs, 4, _full_thrust, _value)
    assert rollout.terminated[0, 0] and rollout.next_values[0, 0] == 0.0
    assert rollout.truncated[3, 1] and rollout.truncated[3, 2]
    # The next observation already belongs to a new episode, so it must not be the bootstrap.
    assert rollout.next_values[3, 1] != _value(next_obs[1:2])[0]
    assert rollout.next_values[3, 1] > rollout.values[3, 1]  # the agent moved on along +x


def test_the_rollout_returns_the_finished_episodes() -> None:
    env, obs = _mixed_env(horizon=4)
    rollout, _ = collect_rollout(env, obs, 9, _full_thrust, _value)
    assert len(rollout.episodes) == int(rollout.done.sum())
    finished = sorted((ep.env, ep.length, ep.success) for ep in rollout.episodes)
    assert finished[0] == (0, 1, True)
    assert all(success or length == 4 for _, length, success in finished)
    counts = np.bincount([ep.env for ep in rollout.episodes], minlength=3)
    np.testing.assert_array_equal(counts, rollout.done.sum(axis=0))


def test_consecutive_rollouts_continue_where_the_previous_one_stopped() -> None:
    env_a, obs_a = _mixed_env(horizon=3, seed=11)
    whole, _ = collect_rollout(env_a, obs_a, 8, _full_thrust, _value)
    env_b, obs_b = _mixed_env(horizon=3, seed=11)
    first, obs_mid = collect_rollout(env_b, obs_b, 5, _full_thrust, _value)
    second, _ = collect_rollout(env_b, obs_mid, 3, _full_thrust, _value)
    for name in ("obs", "actions", "log_probs", "values", "rewards", "terminated", "truncated", "next_values"):
        joined = np.concatenate([getattr(first, name), getattr(second, name)])
        np.testing.assert_array_equal(joined, getattr(whole, name), err_msg=name)


def test_gae_over_a_rollout_credits_the_success_bonus() -> None:
    env, obs = _mixed_env()
    rollout, _ = collect_rollout(env, obs, 3, _full_thrust, lambda o: np.zeros(len(o)))
    advantages, returns = compute_gae(
        rollout.rewards, rollout.values, rollout.next_values, rollout.terminated, rollout.done,
        discount=0.99, gae_lambda=0.95,
    )
    assert returns[0, 0] == pytest.approx(rollout.rewards[0, 0])  # terminated at once: no bootstrap
    assert rollout.rewards[0, 0] > 900.0 and advantages.shape == (3, 3)


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (lambda o: (np.zeros((len(o), 3)), np.zeros(len(o))), "actions"),
        (lambda o: (np.zeros((len(o), 2)), np.zeros((len(o), 1))), "log-probabilities"),
    ],
)
def test_the_rollout_rejects_malformed_policy_outputs(policy: Any, message: str) -> None:
    env, obs = _mixed_env()
    with pytest.raises(ValueError, match=message):
        collect_rollout(env, obs, 2, policy, _value)


def test_a_one_step_rollout_is_allowed() -> None:
    env, obs = _mixed_env()
    rollout, _ = collect_rollout(env, obs, 1, _full_thrust, _value)
    assert rollout.rewards.shape == (1, 3) and rollout.num_samples == 3


def test_the_rollout_rejects_malformed_values_and_settings() -> None:
    env, obs = _mixed_env()
    with pytest.raises(ValueError, match="values"):
        collect_rollout(env, obs, 2, _full_thrust, lambda o: np.zeros((len(o), 1)))
    with pytest.raises(ValueError, match="num_steps"):
        collect_rollout(env, obs, 0, _full_thrust, _value)
    with pytest.raises(ValueError, match="observations"):
        collect_rollout(env, obs[:2], 2, _full_thrust, _value)
    manual = NavigationEnv(_config(), num_envs=3, seed=0, autoreset=False)
    with pytest.raises(ValueError, match="autoreset"):
        collect_rollout(manual, manual.reset(), 2, _full_thrust, _value)
