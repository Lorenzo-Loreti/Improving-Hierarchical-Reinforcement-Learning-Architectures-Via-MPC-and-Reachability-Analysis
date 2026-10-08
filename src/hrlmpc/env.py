"""The navigation MDP on top of the physical step (decision log D6, D10, D13-D15, D18; rows R1-R5, AM1).

:class:`NavigationEnv` runs ``num_envs`` independent episodes in lockstep. At
each step, for every agent:

1. the action ``x`` in ``[-1, 1]^2`` is mapped onto ``U`` (D15); an MPC Worker
   gives the commanded acceleration directly (:meth:`NavigationEnv.step_input`);
2. a disturbance is drawn uniformly in area on ``D`` (D10);
3. :func:`hrlmpc.physics.physics_step` advances the state;
4. the reward of D13 is computed, with the contact penalty of D14:

       r_k = -c_t - c_n ||dv_wall_k|| - c_s 1[contact at k] + B 1[p_(k+1) in G]
             + s Phi(p_(k+1)) - Phi(p_k) - c_e ||u_k||^2 / a_max^2,

   ``Phi(p) = -c_p max(goal_x - p_x, 0)``, ``s`` the shaping discount (1 by
   D22), ``u_k`` the input after the saturation onto ``U``; the unshaped
   reward omits the shaping term;
5. an episode terminates when ``p_(k+1)`` lies in the goal region and is
   truncated at the horizon. A finished episode restarts at rest, uniformly in
   the spawn box, unless automatic resets are disabled.

Contact is never terminal (D1). Observations are ``(p, v)`` mapped to
``[-1, 1]`` with the arena's bounding box and ``v_max`` (D13); the map is
fixed and invertible (:class:`ObservationMap`).

Random numbers. Each agent owns two generators spawned from the seed: one for
its initial positions and one for its disturbances. An agent's trajectory
therefore depends only on the seed, its index and its own actions, not on the
batch size or on the other agents. Every level of ``d_bar`` draws the same unit
disturbances, scaled by ``d_bar``, which pairs the comparisons across levels.

Samples (D6): every physical step adds ``num_envs`` to ``counter.env_steps``;
resets add nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrlmpc.action_map import square_to_disk
from hrlmpc.config import EnvConfig
from hrlmpc.geometry import TOL, Layout
from hrlmpc.physics import physics_step
from hrlmpc.utils.runlog import SampleCounter

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

CORRECTION_TOL = 1e-6
"""Corrections smaller than this (in m/s or m/s^2) count as inactive in the episode statistics.

It sits far below any physically meaningful correction and far above rounding
and solver noise, e.g. an MPC input that overshoots ``a_max`` by the solver's
tolerance.
"""

ACTION_TOL = 1e-12
"""Slack accepted beyond the square ``[-1, 1]^2`` for an action; such actions are clipped onto it."""


@dataclass(frozen=True, eq=False)
class ObservationMap:
    """The fixed map between states and observations (D13), and its inverse.

    Positions are mapped to ``[-1, 1]`` with the bounding box of the arena,
    velocities by ``v_max``. Every state of the environment lies inside these
    bounds, so the map is invertible there; observations are clipped to
    ``[-1, 1]`` only against rounding.

    Attributes:
        low: ``(2,)`` lower corner of the arena's bounding box.
        span: ``(2,)`` side lengths of the bounding box.
        v_max: Speed limit.
    """

    low: FloatArray
    span: FloatArray
    v_max: float

    @classmethod
    def from_layout(cls, layout: Layout, v_max: float) -> ObservationMap:
        """The map of a layout and a speed limit."""
        lo, hi = layout.arena.vertices.min(axis=0), layout.arena.vertices.max(axis=0)
        return cls(low=lo.astype(np.float64), span=(hi - lo).astype(np.float64), v_max=float(v_max))

    @classmethod
    def from_config(cls, config: EnvConfig) -> ObservationMap:
        """The map of an environment configuration."""
        return cls.from_layout(Layout.from_config(config.geometry), config.physics.v_max)

    def observe(self, p: ArrayLike, v: ArrayLike) -> FloatArray:
        """Observations ``(..., 4)`` of positions and velocities ``(..., 2)``."""
        p_arr, v_arr = np.asarray(p, dtype=np.float64), np.asarray(v, dtype=np.float64)
        obs = np.concatenate([2.0 * (p_arr - self.low) / self.span - 1.0, v_arr / self.v_max], axis=-1)
        result: FloatArray = np.clip(obs, -1.0, 1.0)
        return result

    def positions(self, obs: ArrayLike) -> FloatArray:
        """Positions ``(..., 2)`` of observations ``(..., 4)``."""
        result: FloatArray = self.low + 0.5 * (np.asarray(obs, dtype=np.float64)[..., :2] + 1.0) * self.span
        return result

    def velocities(self, obs: ArrayLike) -> FloatArray:
        """Velocities ``(..., 2)`` of observations ``(..., 4)``."""
        result: FloatArray = self.v_max * np.asarray(obs, dtype=np.float64)[..., 2:4]
        return result


@dataclass(frozen=True, eq=False)
class StepInfo:
    """Per-agent quantities of one step, for evaluation and for the metrics of rows R3-R4.

    Attributes:
        u_cmd: ``(B, 2)`` commanded accelerations.
        u: ``(B, 2)`` inputs after the saturation onto ``U``.
        disturbance: ``(B, 2)`` disturbances ``d_k``.
        reward_unshaped: ``(B,)`` rewards without the shaping term (D13).
        contact: ``(B,)`` contact in the sense of D14.
        impulse: ``(B,)`` ``||dv_wall||``, the normal velocity cancelled by the walls [m/s].
        limiter: ``(B,)`` correction of the speed limiter, ``||v^ - v~|| / dt`` [m/s^2].
        saturation: ``(B,)`` correction of the saturation onto ``U``, ``||u - u_cmd||`` [m/s^2].
        wall_reaction: ``(B,)`` whether the wall reaction acted.
        friction: ``(B,)`` whether the wall friction acted.
    """

    u_cmd: FloatArray
    u: FloatArray
    disturbance: FloatArray
    reward_unshaped: FloatArray
    contact: BoolArray
    impulse: FloatArray
    limiter: FloatArray
    saturation: FloatArray
    wall_reaction: BoolArray
    friction: BoolArray


@dataclass(frozen=True)
class EpisodeStats:
    """Summary of one finished episode.

    Attributes:
        env: Index of the agent in the batch.
        length: Number of steps.
        success: Whether the episode reached the goal region.
        episode_return: Undiscounted sum of the rewards.
        unshaped_return: Undiscounted sum of the unshaped rewards (evaluation, D13).
        contact_steps: Steps with contact (R3, duration).
        contact_events: Maximal runs of contact steps (R3, number).
        impulse: Sum of ``||dv_wall||``, the cancelled normal velocity [m/s] (R3).
        wall_steps: Steps with a wall reaction (R4).
        limiter_steps: Steps in which the speed limiter acted (R4).
        limiter_total: Sum of the limiter corrections [m/s^2] (R4).
        saturation_steps: Steps in which the saturation onto ``U`` acted (R4).
        saturation_total: Sum of the saturation corrections [m/s^2] (R4).
    """

    env: int
    length: int
    success: bool
    episode_return: float
    unshaped_return: float
    contact_steps: int
    contact_events: int
    impulse: float
    wall_steps: int
    limiter_steps: int
    limiter_total: float
    saturation_steps: int
    saturation_total: float


@dataclass(frozen=True, eq=False)
class Transition:
    """Outcome of one step of :class:`NavigationEnv`.

    Attributes:
        obs: ``(B, 4)`` observations to act on next; after an automatic reset,
            the first observation of the new episode.
        reward: ``(B,)`` rewards.
        terminated: ``(B,)`` whether the step reached the goal region.
        truncated: ``(B,)`` whether the episode hit the horizon without success.
        final_obs: ``(B, 4)`` observations of the states the step reached,
            before any reset (for bootstrapping truncated episodes).
        info: Per-agent quantities of the step.
        episodes: Statistics of the episodes that ended at this step.
    """

    obs: FloatArray
    reward: FloatArray
    terminated: BoolArray
    truncated: BoolArray
    final_obs: FloatArray
    info: StepInfo
    episodes: tuple[EpisodeStats, ...]


class NavigationEnv:
    """``num_envs`` agents in one layout, stepped together (see the module docstring).

    Args:
        config: Physics, geometry, task and reward of the layout.
        num_envs: Number of agents.
        seed: Non-negative seed of all the agents' random streams.
        autoreset: Whether finished episodes restart within :meth:`step`.

    Raises:
        ValueError: If ``num_envs`` is not positive or ``seed`` is negative.
    """

    observation_size = 4
    action_size = 2

    def __init__(self, config: EnvConfig, num_envs: int = 1, *, seed: int = 0, autoreset: bool = True) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        self.config = config
        self.params = config.physics
        self.layout = Layout.from_config(config.geometry)
        self.num_envs = num_envs
        self.autoreset = autoreset
        self.counter = SampleCounter()
        streams = [child.spawn(2) for child in np.random.SeedSequence(seed).spawn(num_envs)]
        self._spawn_rngs = [np.random.default_rng(spawn) for spawn, _ in streams]
        self._disturbance_rngs = [np.random.default_rng(disturbance) for _, disturbance in streams]
        self.obs_map = ObservationMap.from_layout(self.layout, self.params.v_max)
        spawn = config.task.spawn
        self._spawn_lo, self._spawn_hi = np.array([spawn.x[0], spawn.y[0]]), np.array([spawn.x[1], spawn.y[1]])
        self._p = np.zeros((num_envs, 2))
        self._v = np.zeros((num_envs, 2))
        self._steps = np.zeros(num_envs, dtype=np.int64)
        self._needs_reset = np.ones(num_envs, dtype=bool)
        self._stats = _EpisodeAccumulator(num_envs)

    @property
    def positions(self) -> FloatArray:
        """``(B, 2)`` current positions (a copy)."""
        return self._p.copy()

    @property
    def velocities(self) -> FloatArray:
        """``(B, 2)`` current velocities (a copy)."""
        return self._v.copy()

    @property
    def episode_steps(self) -> NDArray[np.int64]:
        """``(B,)`` steps taken in the current episodes (a copy)."""
        return self._steps.copy()

    def observe(self, p: ArrayLike, v: ArrayLike) -> FloatArray:
        """Observations of states ``(p, v)``: both mapped to ``[-1, 1]`` (D13, :class:`ObservationMap`)."""
        return self.obs_map.observe(p, v)

    def reset(
        self, positions: ArrayLike | None = None, velocities: ArrayLike | None = None, mask: ArrayLike | None = None
    ) -> FloatArray:
        """Start new episodes for the agents selected by ``mask`` (all by default).

        Args:
            positions: One position per selected agent, free and outside the
                goal region; by default drawn uniformly in the spawn box.
            velocities: One velocity per selected agent; zero by default.
            mask: ``(B,)`` booleans selecting the agents.

        Returns:
            ``(B, 4)`` observations of all the agents.

        Raises:
            ValueError: If the mask is not boolean, a shape is wrong, a given
                position is not free or lies in the goal region, or a speed
                exceeds ``v_max``. Nothing changes then, and no random number is drawn.
        """
        selected = np.ones(self.num_envs, dtype=bool) if mask is None else np.asarray(mask)
        if selected.dtype != np.bool_ or selected.shape != (self.num_envs,):
            raise ValueError(f"mask must be a boolean array of shape ({self.num_envs},)")
        index = np.flatnonzero(selected)
        v = np.zeros((len(index), 2)) if velocities is None else _rows(velocities, len(index), "velocities")
        if np.any(np.hypot(v[:, 0], v[:, 1]) > self.params.v_max + TOL):
            raise ValueError("every initial speed must be at most v_max")
        if positions is None:
            p = np.array([self._spawn_rngs[i].uniform(self._spawn_lo, self._spawn_hi) for i in index]).reshape(-1, 2)
        else:
            p = _rows(positions, len(index), "positions")
            if not self.layout.in_free_space(p).all():
                raise ValueError("every initial position must lie in the free space")
            if self.layout.in_goal(p).any():
                raise ValueError("no initial position may lie in the goal region")
        self._p[index], self._v[index] = p, v
        self._steps[index] = 0
        self._needs_reset[index] = False
        self._stats.clear(index)
        return self.observe(self._p, self._v)

    def step(self, action: ArrayLike) -> Transition:
        """Advance every agent with actions ``x`` in ``[-1, 1]^2``, mapped onto ``U`` (D15).

        Raises:
            ValueError: If the actions are not ``(B, 2)``, not finite or outside
                the square, or an agent must be reset first.
        """
        x = _rows(action, self.num_envs, "actions")
        if np.any(np.abs(x) > 1.0 + ACTION_TOL):
            raise ValueError("actions must lie within [-1, 1]^2")
        return self._advance(square_to_disk(np.clip(x, -1.0, 1.0), self.params.a_max))

    def step_input(self, u_cmd: ArrayLike) -> Transition:
        """Advance every agent with commanded accelerations (the MPC Worker's interface).

        Raises:
            ValueError: If the commands are not ``(B, 2)`` or not finite, or an agent must be reset first.
        """
        return self._advance(_rows(u_cmd, self.num_envs, "commanded accelerations"))

    def _advance(self, u_cmd: FloatArray) -> Transition:
        if self._needs_reset.any():
            raise ValueError(f"agents {np.flatnonzero(self._needs_reset).tolist()} need a reset() before the next step")
        p, dt = self._p, self.params.dt
        d = self._disturbances()
        out = physics_step(p, self._v, u_cmd, d, self.params, self.layout)
        self.counter.add_env_steps(self.num_envs)

        r = self.config.reward
        success = self.layout.in_goal(out.p)
        impulse = np.hypot(out.dv_wall[:, 0], out.dv_wall[:, 1])
        effort = r.effort_coef * np.sum(out.u**2, axis=1) / self.params.a_max**2
        unshaped = (
            -r.time_penalty
            - r.contact_impulse_coef * impulse
            - r.contact_step_coef * out.contact
            + r.success_bonus * success
            - effort
        )
        reward = unshaped + r.shaping_discount * self._potential(out.p) - self._potential(p)
        limiter = np.hypot(*(out.v_limited - out.v_candidate).T) / dt
        info = StepInfo(
            u_cmd=u_cmd,
            u=out.u,
            disturbance=d,
            reward_unshaped=unshaped,
            contact=out.contact,
            impulse=impulse,
            limiter=limiter,
            saturation=np.hypot(*(out.u - u_cmd).T),
            wall_reaction=out.wall_reaction,
            friction=out.friction,
        )

        self._p, self._v = out.p.copy(), out.v.copy()
        self._steps += 1
        terminated = success
        truncated = (self._steps >= self.config.task.horizon) & ~terminated
        done = terminated | truncated
        episodes = self._stats.add(reward, info, done, success)
        final_obs = self.observe(self._p, self._v)
        if done.any():
            if self.autoreset:
                self.reset(mask=done)
            else:
                self._needs_reset |= done
        obs = self.observe(self._p, self._v)
        return Transition(obs, reward, terminated, truncated, final_obs, info, episodes)

    def _disturbances(self) -> FloatArray:
        """One disturbance per agent, uniform in area on ``D`` (D10)."""
        unit = np.array([rng.random(2) for rng in self._disturbance_rngs])
        angle, radius = 2.0 * np.pi * unit[:, 0], np.sqrt(unit[:, 1])
        result: FloatArray = self.params.d_bar * (radius[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1))
        return result

    def _potential(self, p: FloatArray) -> FloatArray:
        """Shaping potential ``Phi(p) = -c_p max(goal_x - p_x, 0)`` (D13)."""
        result: FloatArray = -self.config.reward.progress_coef * np.maximum(self.layout.goal_x - p[:, 0], 0.0)
        return result


class _EpisodeAccumulator:
    """Running sums of the episode statistics, one entry per agent."""

    _FIELDS = (
        "length",
        "episode_return",
        "unshaped_return",
        "contact_steps",
        "contact_events",
        "impulse",
        "wall_steps",
        "limiter_steps",
        "limiter_total",
        "saturation_steps",
        "saturation_total",
    )

    def __init__(self, num_envs: int) -> None:
        self._sums = {name: np.zeros(num_envs) for name in self._FIELDS}
        self._in_contact = np.zeros(num_envs, dtype=bool)

    def clear(self, index: NDArray[np.int64]) -> None:
        for values in self._sums.values():
            values[index] = 0.0
        self._in_contact[index] = False

    def add(
        self, reward: FloatArray, info: StepInfo, done: BoolArray, success: BoolArray
    ) -> tuple[EpisodeStats, ...]:
        """Add one step; return the statistics of the episodes that ``done`` ends."""
        limiter = info.limiter > CORRECTION_TOL
        saturation = info.saturation > CORRECTION_TOL
        s = self._sums
        s["length"] += 1.0
        s["episode_return"] += reward
        s["unshaped_return"] += info.reward_unshaped
        s["contact_steps"] += info.contact
        s["contact_events"] += info.contact & ~self._in_contact
        s["impulse"] += info.impulse
        s["wall_steps"] += info.wall_reaction
        s["limiter_steps"] += limiter
        s["limiter_total"] += np.where(limiter, info.limiter, 0.0)
        s["saturation_steps"] += saturation
        s["saturation_total"] += np.where(saturation, info.saturation, 0.0)
        self._in_contact = info.contact.copy()
        finished = []
        for i in np.flatnonzero(done):
            counts = {name: int(s[name][i]) for name in ("length", "contact_steps", "contact_events", "wall_steps")}
            counts.update({name: int(s[name][i]) for name in ("limiter_steps", "saturation_steps")})
            totals = {name: float(s[name][i]) for name in ("episode_return", "unshaped_return", "impulse")}
            totals.update({name: float(s[name][i]) for name in ("limiter_total", "saturation_total")})
            finished.append(EpisodeStats(env=int(i), success=bool(success[i]), **counts, **totals))
        return tuple(finished)


def _rows(values: ArrayLike, rows: int, name: str) -> FloatArray:
    """``values`` as a finite ``(rows, 2)`` float array."""
    out = np.array(values, dtype=np.float64)
    if out.shape != (rows, 2):
        raise ValueError(f"{name} must have shape ({rows}, 2), got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    return out
