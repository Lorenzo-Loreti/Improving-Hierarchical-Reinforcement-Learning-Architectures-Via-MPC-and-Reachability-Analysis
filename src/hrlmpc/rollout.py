"""Rollout collection and generalized advantage estimation (decision log D6, D13, D19).

The training loops collect fixed-length rollouts from the batched
:class:`~hrlmpc.env.NavigationEnv` and estimate advantages with GAE
(Schulman et al., "High-Dimensional Continuous Control Using Generalized
Advantage Estimation", ICLR 2016). Everything here is plain NumPy, so it is
tested without PyTorch; the policy and the value function enter as callables.

Time limits. The observation carries no time (D13), so the truncation at the
horizon is not a terminal state of the task: a truncated transition is
bootstrapped with the value of the state it reached ("partial-episode
bootstrapping", Pardo et al., "Time Limits in Reinforcement Learning", ICML
2018). Only reaching the goal region terminates.

Discounts. :func:`compute_gae` accepts one discount per transition, so a
Manager's transition spanning ``tau`` steps can be discounted by
``gamma ** tau`` (D13, step 8).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrlmpc.env import EpisodeStats, NavigationEnv

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

PolicyFn = Callable[[FloatArray], tuple[ArrayLike, ArrayLike]]
"""Maps ``(B, n)`` observations to ``(B, m)`` actions and their ``(B,)`` log-probabilities."""

ValueFn = Callable[[FloatArray], ArrayLike]
"""Maps ``(k, n)`` observations to ``(k,)`` value estimates."""


def compute_gae(
    rewards: ArrayLike,
    values: ArrayLike,
    next_values: ArrayLike,
    terminated: ArrayLike,
    done: ArrayLike,
    *,
    discount: float | ArrayLike,
    gae_lambda: float,
) -> tuple[FloatArray, FloatArray]:
    """Advantages and returns of a ``(T, B)`` rollout by GAE.

    With ``g_t`` the discount of transition ``t``,

        delta_t = r_t + g_t (1 - terminated_t) next_value_t - value_t,
        A_t     = delta_t + g_t lambda (1 - done_t) A_(t+1),     A_T = 0,

    and the returns are ``A_t + value_t``, the regression targets of the critic.

    Args:
        rewards: ``(T, B)`` rewards.
        values: ``(T, B)`` value estimates of the states acted in.
        next_values: ``(T, B)`` value estimates of the states the transitions
            reached; ignored where ``terminated``.
        terminated: ``(T, B)`` booleans, transitions that ended in a terminal state.
        done: ``(T, B)`` booleans, transitions that ended their episode
            (terminated or truncated); the accumulation stops there.
        discount: Discount in ``[0, 1]``, a scalar or one per transition (``(T, B)``).
        gae_lambda: GAE parameter in ``[0, 1]``.

    Returns:
        ``(advantages, returns)``, both ``(T, B)``.

    Raises:
        ValueError: If the shapes differ, the flags are not boolean, a
            transition is terminated but not done, or a parameter lies outside
            ``[0, 1]``.
    """
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    nv = np.asarray(next_values, dtype=np.float64)
    term, ends = np.asarray(terminated), np.asarray(done)
    if r.ndim != 2 or any(a.shape != r.shape for a in (v, nv, term, ends)):
        raise ValueError("rewards, values, next_values, terminated and done must share one (T, B) shape")
    if term.dtype != np.bool_ or ends.dtype != np.bool_:
        raise ValueError("terminated and done must be boolean arrays")
    if np.any(term & ~ends):
        raise ValueError("every terminated transition must also be done")
    g = np.asarray(discount, dtype=np.float64)
    if g.ndim != 0 and g.shape != r.shape:
        raise ValueError(f"the discount must be a scalar or have the rollout's shape {r.shape}, got {g.shape}")
    if not np.all((g >= 0.0) & (g <= 1.0)):  # also rejects NaN
        raise ValueError("every discount must lie in [0, 1]")
    g = np.broadcast_to(g, r.shape)
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must lie in [0, 1], got {gae_lambda}")
    deltas = r + g * np.where(term, 0.0, nv) - v
    carry = g * gae_lambda * ~ends
    advantages = np.zeros_like(r)
    running = np.zeros(r.shape[1])
    for t in range(r.shape[0] - 1, -1, -1):
        running = deltas[t] + carry[t] * running
        advantages[t] = running
    return advantages, advantages + v


@dataclass(frozen=True, eq=False)
class Rollout:
    """A fixed-length rollout of every agent of a :class:`~hrlmpc.env.NavigationEnv`.

    Attributes:
        obs: ``(T, B, n)`` observations acted on.
        actions: ``(T, B, m)`` actions taken.
        log_probs: ``(T, B)`` log-probabilities of the actions under the
            policy that collected them.
        values: ``(T, B)`` value estimates of ``obs``.
        rewards: ``(T, B)`` rewards.
        terminated: ``(T, B)`` transitions that reached the goal region.
        truncated: ``(T, B)`` transitions that hit the horizon.
        next_values: ``(T, B)`` value estimates of the states reached: of the
            next observation if the episode goes on, of the final observation
            if it was truncated, zero if it terminated.
        episodes: Statistics of the episodes that ended in the rollout.
    """

    obs: FloatArray
    actions: FloatArray
    log_probs: FloatArray
    values: FloatArray
    rewards: FloatArray
    terminated: BoolArray
    truncated: BoolArray
    next_values: FloatArray
    episodes: tuple[EpisodeStats, ...]

    @property
    def done(self) -> BoolArray:
        """``(T, B)`` transitions that ended their episode."""
        result: BoolArray = self.terminated | self.truncated
        return result

    @property
    def num_samples(self) -> int:
        """Number of physical steps in the rollout (decision log D6)."""
        return int(self.rewards.size)


def _checked(values: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"the {name} must have shape {shape}, got {array.shape}")
    return array


def collect_rollout(
    env: NavigationEnv, obs: ArrayLike, num_steps: int, policy: PolicyFn, value: ValueFn
) -> tuple[Rollout, FloatArray]:
    """Step every agent of ``env`` ``num_steps`` times with ``policy``.

    Episodes that end restart inside the environment (automatic resets), so
    the rollout runs across episode boundaries. Every step adds the batch size
    to ``env.counter`` (decision log D6).

    Args:
        env: Environment with automatic resets.
        obs: ``(B, n)`` observations to act on first, as returned by the
            environment's last reset or by the previous rollout.
        num_steps: Steps per agent.
        policy: Actions and log-probabilities for a batch of observations.
        value: Value estimates for a batch of observations.

    Returns:
        The rollout and the ``(B, n)`` observations to act on next.

    Raises:
        ValueError: If ``env`` does not reset automatically, ``num_steps`` is
            not positive, or an input or a callable's output has the wrong shape.
    """
    if not env.autoreset:
        raise ValueError("collect_rollout needs an environment with autoreset=True")
    if num_steps < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    batch, obs_size, action_size = env.num_envs, env.observation_size, env.action_size
    current = _checked(obs, (batch, obs_size), "observations")
    buffers = {
        "obs": np.zeros((num_steps, batch, obs_size)),
        "actions": np.zeros((num_steps, batch, action_size)),
        "log_probs": np.zeros((num_steps, batch)),
        "values": np.zeros((num_steps, batch)),
        "rewards": np.zeros((num_steps, batch)),
        "terminated": np.zeros((num_steps, batch), dtype=bool),
        "truncated": np.zeros((num_steps, batch), dtype=bool),
        "next_values": np.zeros((num_steps, batch)),
    }
    episodes: list[EpisodeStats] = []
    current_values = _checked(value(current), (batch,), "values")
    for t in range(num_steps):
        actions_raw, log_probs_raw = policy(current)
        actions = _checked(actions_raw, (batch, action_size), "actions")
        log_probs = _checked(log_probs_raw, (batch,), "log-probabilities")
        out = env.step(actions)
        buffers["obs"][t] = current
        buffers["actions"][t] = actions
        buffers["log_probs"][t] = log_probs
        buffers["values"][t] = current_values
        buffers["rewards"][t] = out.reward
        buffers["terminated"][t] = out.terminated
        buffers["truncated"][t] = out.truncated
        episodes.extend(out.episodes)
        upcoming = _checked(value(out.obs), (batch,), "values")
        reached = np.where(out.terminated | out.truncated, 0.0, upcoming)
        if out.truncated.any():
            count = int(out.truncated.sum())
            reached[out.truncated] = _checked(value(out.final_obs[out.truncated]), (count,), "values")
        buffers["next_values"][t] = reached
        current, current_values = out.obs, upcoming
    rollout = Rollout(episodes=tuple(episodes), **buffers)
    return rollout, current
