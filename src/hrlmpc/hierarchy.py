"""The common Manager/Worker loop of the hierarchical architectures (decision log D6, D24-D27, D29; rows AR2-AR5).

Segments (D27). The Manager decides at the start of every episode and after
every segment of ``H`` steps; a segment also ends with its episode. Its
decision ``x`` in ``[-1, 1]^2`` sets a target that stays fixed in the plane for
the whole segment. Its reward is the environment's reward discounted within the
segment, ``sum_(k < tau) gamma^k r_k``, and its discount to the next decision
is ``gamma^tau``, with ``tau <= H`` the segment's length: the semi-MDP of
options (Sutton, Precup and Singh, Artificial Intelligence 112, 1999), whose
return is the discounted return of a flat learner. A segment truncated at the
horizon is bootstrapped with the value of the state it reached.

Targets (D24). :class:`DiskTargets` maps ``x`` onto the disk of radius ``R``
around the position at the segment's start, with the bijection of D15. The
reachable sets of architecture 4 will enter through the same interface,
:class:`TargetMap` (step 10).

Workers. The Worker is interchangeable. A :class:`LearnedWorker`, the hPPO
Worker, sees the observation and the target offset ``(g - p) / R`` (D26) and
returns actions in the square, mapped onto ``U`` by the environment (D15). A
:class:`CommandWorker`, the MPC Worker of step 9, returns commanded
accelerations. Every Worker is scored with the reward of D25, for learning
(hPPO) or as a diagnostic. A learned Worker bootstraps its value at the end of
every episode: its stream of targets has no terminal state (D25).

Rollouts straddle updates. A segment still in flight when a rollout ends goes
on in the next one (:class:`HierarchyState`), so the Manager never re-plans
because a rollout ended and training follows the same segments as the
evaluation (D29). Such a segment keeps the log-probability and the value of
the policy that decided it and is stored when it ends; its value then acts
only as its own baseline, since GAE's return target does not depend on it.

Positions are read from the observations (:class:`~hrlmpc.env.ObservationMap`),
so the training loop and the evaluation's controller compute the same targets
and Worker inputs from the same observations. Everything here is NumPy; the
policies enter as callables.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrlmpc.action_map import square_to_disk
from hrlmpc.config import EnvConfig
from hrlmpc.env import ACTION_TOL, EpisodeStats, NavigationEnv, ObservationMap, StepInfo
from hrlmpc.model import PhysicsParams
from hrlmpc.rollout import PolicyFn, Rollout, ValueFn, compute_gae

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def segment_reach(params: PhysicsParams, segment_steps: int) -> float:
    """Longest displacement over ``segment_steps`` steps, ``v_max H T_s``: every speed is at most ``v_max``.

    Raises:
        ValueError: If ``segment_steps`` is not positive.
    """
    if segment_steps < 1:
        raise ValueError(f"segment_steps must be positive, got {segment_steps}")
    return params.v_max * segment_steps * params.dt


def _pairs(values: ArrayLike, name: str) -> FloatArray:
    array = np.array(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"the {name} must have shape (B, 2), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"the {name} must be finite")
    return array


def _checked(values: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.array(values, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"the {name} must have shape {shape}, got {array.shape}")
    return array


class TargetMap(Protocol):
    """Maps the Manager's actions to targets in the plane (D24)."""

    @property
    def radius(self) -> float:
        """Length scale [m] of the target offsets; the Worker's input divides them by it (D26)."""
        ...

    def __call__(self, actions: ArrayLike, positions: ArrayLike, velocities: ArrayLike) -> FloatArray:
        """``(B, 2)`` targets for ``(B, 2)`` actions in the square, at ``(B, 2)`` positions and velocities."""
        ...


@dataclass(frozen=True)
class DiskTargets:
    """Targets ``g = p + R m(x)`` on the disk of radius ``R`` around the position (D24).

    ``m`` is the bijection of D15 from the square onto the unit disk: it keeps
    the direction of ``x`` and maps ``||x||_inf`` to ``||m(x)||_2``. The
    velocity plays no part.

    Raises:
        ValueError: If the radius is not positive and finite.
    """

    radius: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.radius) and self.radius > 0.0):
            raise ValueError(f"the radius must be positive and finite, got {self.radius}")

    def __call__(self, actions: ArrayLike, positions: ArrayLike, velocities: ArrayLike) -> FloatArray:
        """Targets of actions in the square at the given positions.

        Raises:
            ValueError: If the shapes differ or are not ``(B, 2)``, a value is
                not finite, or an action lies outside the square.
        """
        x, p, v = _pairs(actions, "actions"), _pairs(positions, "positions"), _pairs(velocities, "velocities")
        if x.shape != p.shape or v.shape != p.shape:
            raise ValueError(
                f"actions, positions and velocities must share one shape, got {x.shape}, {p.shape}, {v.shape}"
            )
        if np.any(np.abs(x) > 1.0 + ACTION_TOL):
            raise ValueError("the actions must lie within the square [-1, 1]^2")
        result: FloatArray = p + square_to_disk(np.clip(x, -1.0, 1.0), self.radius)
        return result


def worker_inputs(obs: ArrayLike, positions: ArrayLike, targets: ArrayLike, radius: float) -> FloatArray:
    """The learned Worker's input (D26): the observation and the target offset ``(g - p) / R``, ``(..., 6)``.

    Raises:
        ValueError: If the shapes do not match ``(..., 4)``, ``(..., 2)``, ``(..., 2)``.
    """
    o = np.asarray(obs, dtype=np.float64)
    p, g = np.asarray(positions, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if o.ndim == 0 or o.shape[-1] != 4 or p.shape != (*o.shape[:-1], 2) or g.shape != p.shape:
        raise ValueError(f"observations, positions and targets have the wrong shapes: {o.shape}, {p.shape}, {g.shape}")
    result: FloatArray = np.concatenate([o, (g - p) / radius], axis=-1)
    return result


@dataclass(frozen=True, eq=False)
class WorkerReward:
    """The Worker's reward of D25, by term.

    Attributes:
        progress: ``c_p (||p_k - g|| - ||p_(k+1) - g||)``.
        contact: ``-c_n ||dv_wall|| - c_s 1[contact]``, the contact penalty of D14.
        effort: ``-c_e ||u||^2 / a_max^2``.
    """

    progress: FloatArray
    contact: FloatArray
    effort: FloatArray

    @property
    def total(self) -> FloatArray:
        """The reward: the sum of the three terms."""
        result: FloatArray = self.progress + self.contact + self.effort
        return result


def worker_reward(
    config: EnvConfig, positions: ArrayLike, next_positions: ArrayLike, targets: ArrayLike, info: StepInfo
) -> WorkerReward:
    """The Worker's reward of one step (D25), at the environment's prices.

    Args:
        config: The environment's configuration, for ``c_p``, ``c_n``, ``c_s``, ``c_e`` and ``a_max``.
        positions: ``(B, 2)`` positions before the step.
        next_positions: ``(B, 2)`` positions the step reached, before any reset.
        targets: ``(B, 2)`` targets of the step.
        info: The step's :class:`~hrlmpc.env.StepInfo`, for the contact and the input.
    """
    r, params = config.reward, config.physics
    p, q = np.asarray(positions, dtype=np.float64), np.asarray(next_positions, dtype=np.float64)
    g = np.asarray(targets, dtype=np.float64)
    before, after = p - g, q - g
    progress = r.progress_coef * (np.hypot(before[..., 0], before[..., 1]) - np.hypot(after[..., 0], after[..., 1]))
    contact = -(r.contact_impulse_coef * info.impulse + r.contact_step_coef * info.contact)
    effort = -r.effort_coef * np.sum(info.u**2, axis=-1) / params.a_max**2
    return WorkerReward(progress=progress, contact=contact, effort=effort)


@dataclass(frozen=True)
class LearnedWorker:
    """A learned Worker (hPPO): a policy and a value function over the inputs of :func:`worker_inputs`.

    Attributes:
        policy: ``(B, 6)`` inputs to ``(B, 2)`` actions in the square and their log-probabilities.
        value: ``(k, 6)`` inputs to ``(k,)`` value estimates.
    """

    policy: PolicyFn
    value: ValueFn


@dataclass(frozen=True)
class CommandWorker:
    """A Worker that commands accelerations, like the MPC Worker (step 9); it does not learn.

    Attributes:
        command: ``(obs, positions, targets)``, each with one row per agent, to
            ``(B, 2)`` commanded accelerations.
        reset: Called with the ``(B,)`` mask of the agents whose episode just
            ended, before they act in their new episode; ``None`` if the
            Worker keeps no state.
    """

    command: Callable[[FloatArray, FloatArray, FloatArray], ArrayLike]
    reset: Callable[[BoolArray], None] | None = None


Worker = LearnedWorker | CommandWorker


@dataclass(frozen=True, eq=False)
class Hierarchy:
    """The fixed parts of the loop: targets, segment length, the Manager's discount, observation map.

    Attributes:
        targets: Map from the Manager's actions to targets (D24).
        segment_steps: ``H``, the steps of a full segment (D27).
        discount: The Manager's per-step discount ``gamma_RL``: its reward is
            discounted by it within a segment, and by ``gamma_RL^tau`` between decisions.
        obs_map: Positions and velocities of the observations.

    Raises:
        ValueError: If ``segment_steps`` is not positive or ``discount`` lies outside ``(0, 1]``.
    """

    targets: TargetMap
    segment_steps: int
    discount: float
    obs_map: ObservationMap

    def __post_init__(self) -> None:
        if self.segment_steps < 1:
            raise ValueError(f"segment_steps must be positive, got {self.segment_steps}")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError(f"the discount must lie in (0, 1], got {self.discount}")


def disk_hierarchy(config: EnvConfig, *, segment_steps: int, reach_factor: float, discount: float) -> Hierarchy:
    """The hierarchy of D24 and D27: targets on the disk of radius ``reach_factor`` times the segment's reach."""
    radius = reach_factor * segment_reach(config.physics, segment_steps)
    return Hierarchy(DiskTargets(radius), segment_steps, discount, ObservationMap.from_config(config))


@dataclass(frozen=True, eq=False)
class HierarchyState:
    """What one rollout hands to the next: the observations, and the segments in flight.

    Attributes:
        obs: ``(B, 4)`` observations to act on next.
        targets: ``(B, 2)`` targets of the segments in flight.
        segment_obs: ``(B, 4)`` the Manager's observations at their starts.
        segment_actions: ``(B, 2)`` the Manager's actions that set them.
        segment_log_probs: ``(B,)`` the actions' log-probabilities.
        segment_values: ``(B,)`` the Manager's values at their starts.
        segment_steps: ``(B,)`` steps taken in them so far.
        segment_rewards: ``(B,)`` their discounted rewards so far.
    """

    obs: FloatArray
    targets: FloatArray
    segment_obs: FloatArray
    segment_actions: FloatArray
    segment_log_probs: FloatArray
    segment_values: FloatArray
    segment_steps: IntArray
    segment_rewards: FloatArray


@dataclass(frozen=True, eq=False)
class Segments:
    """The Manager's transitions: the segments that ended in a rollout, in the order they ended.

    Attributes:
        env: ``(N,)`` agent of each segment.
        obs: ``(N, 4)`` the Manager's observation at the start.
        actions: ``(N, 2)`` its action.
        log_probs: ``(N,)`` the action's log-probability under the policy that decided it.
        values: ``(N,)`` the Manager's value at the start.
        rewards: ``(N,)`` ``sum_(k < tau) gamma^k r_k``.
        steps: ``(N,)`` length ``tau``.
        discounts: ``(N,)`` ``gamma^tau``.
        terminated: ``(N,)`` segments that reached the goal region.
        truncated: ``(N,)`` segments cut by the horizon.
        next_values: ``(N,)`` value of the state reached: of the next
            segment's start if the episode goes on, of the final state if it
            was truncated, zero if it terminated.
        end_distances: ``(N,)`` distance [m] from the target at the end.
    """

    env: IntArray
    obs: FloatArray
    actions: FloatArray
    log_probs: FloatArray
    values: FloatArray
    rewards: FloatArray
    steps: IntArray
    discounts: FloatArray
    terminated: BoolArray
    truncated: BoolArray
    next_values: FloatArray
    end_distances: FloatArray

    @property
    def done(self) -> BoolArray:
        """``(N,)`` segments that ended their episode."""
        result: BoolArray = self.terminated | self.truncated
        return result

    def __len__(self) -> int:
        return int(self.env.shape[0])


def segment_gae(segments: Segments, gae_lambda: float) -> tuple[FloatArray, FloatArray]:
    """Advantages and returns of the Manager's transitions by GAE, agent by agent, at the discounts ``gamma^tau``.

    Raises:
        ValueError: If ``gae_lambda`` lies outside ``[0, 1]``.
    """
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must lie in [0, 1], got {gae_lambda}")
    advantages, returns = np.zeros(len(segments)), np.zeros(len(segments))
    for b in np.unique(segments.env):
        rows = np.flatnonzero(segments.env == b)
        adv, ret = compute_gae(
            segments.rewards[rows][:, None],
            segments.values[rows][:, None],
            segments.next_values[rows][:, None],
            segments.terminated[rows][:, None],
            segments.done[rows][:, None],
            discount=segments.discounts[rows][:, None],
            gae_lambda=gae_lambda,
        )
        advantages[rows], returns[rows] = adv[:, 0], ret[:, 0]
    return advantages, returns


@dataclass(frozen=True, eq=False)
class HierarchicalRollout:
    """A fixed-length rollout of the hierarchy, ``T`` steps of ``B`` agents.

    Attributes:
        segments: The Manager's transitions. A decision follows every segment
            that ends, so their number is also that of the rollout's decisions.
        worker: The learned Worker's transitions, ``None`` for a
            :class:`CommandWorker`. Its observations are the inputs of
            :func:`worker_inputs` and its rewards those of D25; no transition
            is terminal, and ``truncated`` marks every end of an episode,
            where the Worker bootstraps (D25).
        worker_reward: ``(T, B)`` terms of the Worker's reward, also for a :class:`CommandWorker`.
        obs: ``(T, B, 4)`` observations acted on.
        targets: ``(T, B, 2)`` targets in force.
        rewards: ``(T, B)`` the environment's rewards.
        terminated: ``(T, B)`` steps that reached the goal region.
        truncated: ``(T, B)`` steps that hit the horizon.
        final_obs: ``(T, B, 4)`` observations of the states reached, before any reset.
        episodes: Statistics of the episodes that ended in the rollout.
    """

    segments: Segments
    worker: Rollout | None
    worker_reward: WorkerReward
    obs: FloatArray
    targets: FloatArray
    rewards: FloatArray
    terminated: BoolArray
    truncated: BoolArray
    final_obs: FloatArray
    episodes: tuple[EpisodeStats, ...]

    @property
    def done(self) -> BoolArray:
        """``(T, B)`` steps that ended their episode."""
        result: BoolArray = self.terminated | self.truncated
        return result

    @property
    def num_samples(self) -> int:
        """Physical steps in the rollout (D6)."""
        return int(self.rewards.size)


def _decide(
    hierarchy: Hierarchy, policy: PolicyFn, value: ValueFn, obs: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """The Manager's actions, log-probabilities, values and targets at ``(k, 4)`` observations."""
    k = len(obs)
    actions_raw, log_probs_raw = policy(obs)
    actions = _checked(actions_raw, (k, 2), "Manager's actions")
    log_probs = _checked(log_probs_raw, (k,), "Manager's log-probabilities")
    values = _checked(value(obs), (k,), "Manager's values")
    obs_map = hierarchy.obs_map
    targets = hierarchy.targets(actions, obs_map.positions(obs), obs_map.velocities(obs))
    return actions, log_probs, values, targets


def start_segments(
    env: NavigationEnv,
    hierarchy: Hierarchy,
    manager_policy: PolicyFn,
    manager_value: ValueFn,
    *,
    positions: ArrayLike | None = None,
) -> HierarchyState:
    """Reset every agent of ``env`` and let the Manager decide the first segments.

    Args:
        env: Environment with automatic resets.
        hierarchy: Targets, segments and the Manager's discount.
        manager_policy: The Manager's actions and log-probabilities.
        manager_value: The Manager's values.
        positions: Initial positions, one per agent; by default drawn in the spawn box.

    Returns:
        The state to start the first rollout from; ``B`` decisions are added to ``env.counter``.

    Raises:
        ValueError: If ``env`` does not reset automatically, or as :meth:`~hrlmpc.env.NavigationEnv.reset`.
    """
    if not env.autoreset:
        raise ValueError("the hierarchical loop needs an environment with autoreset=True")
    obs = env.reset(positions=positions)
    actions, log_probs, values, targets = _decide(hierarchy, manager_policy, manager_value, obs)
    env.counter.add_manager_decisions(env.num_envs)
    return HierarchyState(
        obs=obs,
        targets=targets,
        segment_obs=obs.copy(),
        segment_actions=actions,
        segment_log_probs=log_probs,
        segment_values=values,
        segment_steps=np.zeros(env.num_envs, dtype=np.int64),
        segment_rewards=np.zeros(env.num_envs),
    )


def collect_hierarchical_rollout(
    env: NavigationEnv,
    state: HierarchyState,
    num_steps: int,
    hierarchy: Hierarchy,
    manager_policy: PolicyFn,
    manager_value: ValueFn,
    worker: Worker,
) -> tuple[HierarchicalRollout, HierarchyState]:
    """Step every agent ``num_steps`` times under the Manager and the Worker.

    Episodes that end restart inside the environment, and a new segment starts
    with them. Every step adds the batch size to ``env.counter.env_steps``,
    every decision one to ``env.counter.manager_decisions`` (D6).

    Args:
        env: Environment with automatic resets.
        state: Where the previous rollout (or :func:`start_segments`) stopped.
        num_steps: Steps per agent.
        hierarchy: Targets, segments and the Manager's discount.
        manager_policy: The Manager's actions and log-probabilities, at ``(k, 4)`` observations.
        manager_value: The Manager's values.
        worker: A :class:`LearnedWorker` or a :class:`CommandWorker`.

    Returns:
        The rollout and the state to continue from.

    Raises:
        ValueError: If ``env`` does not reset automatically, ``num_steps`` is
            not positive, the state holds another number of agents, or a
            callable returns the wrong shape.
    """
    if not env.autoreset:
        raise ValueError("the hierarchical loop needs an environment with autoreset=True")
    if num_steps < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    batch = env.num_envs
    if state.obs.shape != (batch, 4):
        raise ValueError(f"the state holds {len(state.obs)} agents, the environment {batch}")
    obs_map, horizon, gamma = hierarchy.obs_map, hierarchy.segment_steps, hierarchy.discount
    radius = hierarchy.targets.radius
    learned = isinstance(worker, LearnedWorker)

    obs, targets = state.obs.copy(), state.targets.copy()
    seg_obs, seg_actions = state.segment_obs.copy(), state.segment_actions.copy()
    seg_log_probs, seg_values = state.segment_log_probs.copy(), state.segment_values.copy()
    seg_steps, seg_rewards = state.segment_steps.copy(), state.segment_rewards.copy()

    shape = (num_steps, batch)
    steps = {
        "obs": np.zeros((*shape, 4)),
        "targets": np.zeros((*shape, 2)),
        "rewards": np.zeros(shape),
        "terminated": np.zeros(shape, dtype=bool),
        "truncated": np.zeros(shape, dtype=bool),
        "final_obs": np.zeros((*shape, 4)),
    }
    terms = {name: np.zeros(shape) for name in ("progress", "contact", "effort")}
    w = {
        "obs": np.zeros((*shape, 6)),
        "actions": np.zeros((*shape, 2)),
        "log_probs": np.zeros(shape),
        "values": np.zeros(shape),
        "next_values": np.zeros(shape),
    }
    closed: dict[str, list[ArrayLike]] = {name: [] for name in Segments.__dataclass_fields__}
    episodes: list[EpisodeStats] = []

    if isinstance(worker, LearnedWorker):
        start = worker_inputs(obs, obs_map.positions(obs), targets, radius)
        current_values = _checked(worker.value(start), (batch,), "Worker's values")
    for t in range(num_steps):
        p = obs_map.positions(obs)
        if isinstance(worker, LearnedWorker):
            inputs = worker_inputs(obs, p, targets, radius)
            actions_raw, log_probs_raw = worker.policy(inputs)
            actions = _checked(actions_raw, (batch, 2), "Worker's actions")
            log_probs = _checked(log_probs_raw, (batch,), "Worker's log-probabilities")
            out = env.step(actions)
        else:
            commands = _checked(worker.command(obs, p, targets), (batch, 2), "Worker's accelerations")
            out = env.step_input(commands)
        done = out.terminated | out.truncated
        reached = obs_map.positions(out.final_obs)
        reward = worker_reward(env.config, p, reached, targets, out.info)
        steps["obs"][t], steps["targets"][t], steps["rewards"][t] = obs, targets, out.reward
        steps["terminated"][t], steps["truncated"][t] = out.terminated, out.truncated
        steps["final_obs"][t] = out.final_obs
        terms["progress"][t], terms["contact"][t], terms["effort"][t] = reward.progress, reward.contact, reward.effort
        episodes.extend(out.episodes)

        seg_rewards += gamma**seg_steps * out.reward
        seg_steps += 1
        new_targets = targets.copy()
        ending = np.flatnonzero((seg_steps >= horizon) | done)
        if len(ending):
            decided = _decide(hierarchy, manager_policy, manager_value, out.obs[ending])
            actions_m, log_probs_m, values_m, targets_m = decided
            next_values = np.where(done[ending], 0.0, values_m)
            cut = out.truncated[ending]
            if cut.any():
                final = out.final_obs[ending[cut]]
                next_values[cut] = _checked(manager_value(final), (len(final),), "Manager's values")
            gap = reached[ending] - targets[ending]
            closed["env"].append(ending)
            closed["obs"].append(seg_obs[ending])
            closed["actions"].append(seg_actions[ending])
            closed["log_probs"].append(seg_log_probs[ending])
            closed["values"].append(seg_values[ending])
            closed["rewards"].append(seg_rewards[ending])
            closed["steps"].append(seg_steps[ending])
            closed["discounts"].append(gamma ** seg_steps[ending].astype(np.float64))
            closed["terminated"].append(out.terminated[ending])
            closed["truncated"].append(out.truncated[ending])
            closed["next_values"].append(next_values)
            closed["end_distances"].append(np.hypot(gap[:, 0], gap[:, 1]))
            seg_obs[ending], seg_actions[ending] = out.obs[ending], actions_m
            seg_log_probs[ending], seg_values[ending] = log_probs_m, values_m
            seg_steps[ending], seg_rewards[ending] = 0, 0.0
            new_targets[ending] = targets_m
            env.counter.add_manager_decisions(len(ending))

        if isinstance(worker, LearnedWorker):
            upcoming_inputs = worker_inputs(out.obs, obs_map.positions(out.obs), new_targets, radius)
            upcoming = _checked(worker.value(upcoming_inputs), (batch,), "Worker's values")
            next_w = upcoming.copy()
            if done.any():  # D25: the Worker bootstraps under the target it had
                final_inputs = worker_inputs(out.final_obs[done], reached[done], targets[done], radius)
                next_w[done] = _checked(worker.value(final_inputs), (int(done.sum()),), "Worker's values")
            w["obs"][t], w["actions"][t], w["log_probs"][t] = inputs, actions, log_probs
            w["values"][t], w["next_values"][t] = current_values, next_w
            current_values = upcoming
        elif done.any() and worker.reset is not None:
            worker.reset(done.copy())
        obs, targets = out.obs, new_targets

    def joined(name: str, width: int = 0) -> NDArray[Any]:
        """The records of one field, in the order the segments ended."""
        if closed[name]:
            return np.concatenate([np.asarray(part) for part in closed[name]])
        return np.zeros((0, width) if width else (0,))

    def floats(name: str, width: int = 0) -> FloatArray:
        return np.asarray(joined(name, width), dtype=np.float64)

    def flags(name: str) -> BoolArray:
        return np.asarray(joined(name), dtype=np.bool_)

    segments = Segments(
        env=np.asarray(joined("env"), dtype=np.int64),
        obs=floats("obs", 4),
        actions=floats("actions", 2),
        log_probs=floats("log_probs"),
        values=floats("values"),
        rewards=floats("rewards"),
        steps=np.asarray(joined("steps"), dtype=np.int64),
        discounts=floats("discounts"),
        terminated=flags("terminated"),
        truncated=flags("truncated"),
        next_values=floats("next_values"),
        end_distances=floats("end_distances"),
    )
    worker_reward_terms = WorkerReward(**terms)
    worker_rollout = None
    if learned:
        worker_rollout = Rollout(
            obs=w["obs"],
            actions=w["actions"],
            log_probs=w["log_probs"],
            values=w["values"],
            rewards=worker_reward_terms.total,
            terminated=np.zeros(shape, dtype=bool),
            truncated=steps["terminated"] | steps["truncated"],
            next_values=w["next_values"],
            episodes=tuple(episodes),
        )
    rollout = HierarchicalRollout(
        segments=segments,
        worker=worker_rollout,
        worker_reward=worker_reward_terms,
        episodes=tuple(episodes),
        **steps,
    )
    after = HierarchyState(
        obs=obs.copy(),
        targets=targets.copy(),
        segment_obs=seg_obs,
        segment_actions=seg_actions,
        segment_log_probs=seg_log_probs,
        segment_values=seg_values,
        segment_steps=seg_steps,
        segment_rewards=seg_rewards,
    )
    return rollout, after


class HierarchicalController:
    """The deterministic hierarchy as one controller for :func:`hrlmpc.evaluation.evaluate` (D29).

    The Manager decides at the first call after :meth:`reset` and after every
    ``H`` calls, as in training; the Worker acts at every call. The agents start
    together; :func:`~hrlmpc.evaluation.evaluate` calls :meth:`reset` before
    the first step, and a call with another number of agents resets too.

    Args:
        hierarchy: Targets and segments, as in training.
        manager: ``(B, 4)`` observations to ``(B, 2)`` actions in the square.
        worker: For a learned Worker, ``(B, 6)`` inputs of :func:`worker_inputs`
            to ``(B, 2)`` actions in the square; or a :class:`CommandWorker`,
            whose accelerations call for ``evaluate(..., inputs=True)``.
    """

    def __init__(
        self,
        hierarchy: Hierarchy,
        manager: Callable[[FloatArray], ArrayLike],
        worker: Callable[[FloatArray], ArrayLike] | CommandWorker,
    ) -> None:
        self.hierarchy = hierarchy
        self.manager = manager
        self.worker = worker
        self.inputs = isinstance(worker, CommandWorker)
        """Whether the commands are accelerations (:func:`hrlmpc.evaluation.evaluate`'s ``inputs``)."""
        self._agents = 0
        self._calls = 0
        self._targets = np.zeros((0, 2))

    def reset(self, num_agents: int) -> None:
        """Start new episodes for ``num_agents`` agents: the next call re-plans, and a :class:`CommandWorker` resets."""
        self._agents = num_agents
        self._calls = 0
        self._targets = np.zeros((0, 2))
        if isinstance(self.worker, CommandWorker) and self.worker.reset is not None:
            self.worker.reset(np.ones(num_agents, dtype=bool))

    def __call__(self, obs: ArrayLike) -> FloatArray:
        """The commands for ``(B, 4)`` observations."""
        o = np.asarray(obs, dtype=np.float64)
        h = self.hierarchy
        p = h.obs_map.positions(o)
        if len(o) != self._agents:
            self.reset(len(o))
        if self._calls % h.segment_steps == 0:
            actions = _checked(self.manager(o), (len(o), 2), "Manager's actions")
            self._targets = h.targets(actions, p, h.obs_map.velocities(o))
        self._calls += 1
        if isinstance(self.worker, CommandWorker):
            return _checked(self.worker.command(o, p, self._targets), (len(o), 2), "Worker's accelerations")
        inputs = worker_inputs(o, p, self._targets, h.targets.radius)
        return _checked(self.worker(inputs), (len(o), 2), "Worker's actions")
