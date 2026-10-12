"""Evaluation of a deterministic controller from fixed starts, and a minimum-time bound (decision log D6, D20, F7,
D43, D50).

Protocol (D20). A controller is evaluated from the points of a grid on the
spawn box, at rest, one episode per start, in an environment of its own:
evaluation steps are not samples (D6) and never influence training. The
evaluation environment is seeded by the configuration, the same seed for every
run, so that every evaluation of every run meets the same disturbances.

Actions (D43, D50). At every evaluation point the training loops evaluate the
policy twice, with the mean of each Beta and then with its mode
(:data:`EVAL_ACTIONS`). The evaluation with the mean keeps the names of steps
7-10 (``eval/``, ``eval_mpc/``, ...); the one with the mode adds ``_mode``
(``eval_mode/``, ``eval_mode_mpc/``, ...; :func:`evaluation_name`). The
run's ``evaluations.jsonl`` holds one line per evaluation and action
(:func:`append_details`).

Compute (D50). :func:`evaluate` times every call of the controller, which
acts for all the agents at once, and divides it by their number: the
controller's time per agent and step, under ``time/`` with the evaluation's
duration (:meth:`Evaluation.timing`), apart from the metrics of the outcome,
which stay reproducible. For the MPC Workers, which solve agent by agent, it
is the mean time of an agent's decision; for the networks, which act on the
whole batch at once, it is less than the latency of a single agent's decision.
The largest is that of the slowest step, a mean over the agents; the slowest
single decision of an MPC Worker is its own ``eval_mpc/solve_ms_max``.

Minimum-time bound (F7). Let ``A = a_max + d_bar`` and

    s_0 = ||v_0||,    s_(k+1) = min((1 - gamma T_s) s_k + T_s A, v_max).

Every stage of the environment step after the input keeps the speed below
this recursion: ``||v~|| <= (1 - gamma T_s) ||v|| + T_s A``, the speed limiter
and the wall friction only shrink the velocity, and the wall reaction is a
projection onto a convex set that contains the origin. The recursion is
monotone, so ``||v_k|| <= s_k`` for every policy, disturbance and layout. The
positions ``p_0, ..., p_k`` are joined by straight segments in ``P_free``, of
total length at most ``T_s (s_1 + ... + s_k)``. An episode from ``p_0``
therefore cannot reach the goal region before the first ``k`` with

    T_s (s_1 + ... + s_k) >= d_free(p_0),

the length of the shortest path in ``P_free`` from ``p_0`` to the goal region
(:mod:`hrlmpc.paths`). Without obstacles ``d_free(p_0) = goal_x - p_x,0``, and
then also ``p_x,k <= p_x,0 + T_s (s_1 + ... + s_k)``. With ``d_bar = 0``, from
rest and away from the walls, full thrust along ``+x`` attains this, so in the
tunnel the bound is the minimum time. The proof is to be written in the
thesis; the tests check both statements numerically.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrlmpc.action_map import EVAL_ACTIONS as EVAL_ACTIONS
from hrlmpc.config import Box, EnvConfig
from hrlmpc.env import EpisodeStats, NavigationEnv
from hrlmpc.geometry import TOL, Layout
from hrlmpc.model import PhysicsParams
from hrlmpc.paths import free_path_distance

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

Controller = Callable[[FloatArray], ArrayLike]
"""Maps ``(B, 4)`` observations to ``(B, 2)`` commands: actions in ``[-1, 1]^2``, or accelerations.

It is called once per step for all the agents together, from their first
step on. A controller with memory (:class:`hrlmpc.hierarchy.HierarchicalController`)
has a ``reset(num_agents)`` method, which :func:`evaluate` calls before the
first step, and may declare with an ``inputs`` attribute that its commands are
accelerations.
"""

PATH_MARGIN = 1e-6
"""Length [m] taken off a bent shortest path, so that rounding and the tolerance ``TOL`` cannot invalidate the bound."""

_MEAN_FIELDS = (
    "contact_steps",
    "contact_events",
    "impulse",
    "wall_steps",
    "limiter_steps",
    "limiter_total",
    "saturation_steps",
    "saturation_total",
)


def spawn_grid(spawn: Box, shape: tuple[int, int]) -> FloatArray:
    """Starts on a ``(nx, ny)`` grid over the spawn box, ordered by ``x``, then by ``y``.

    The grid includes the corners of the box; an axis with a single point
    takes the middle of the interval.

    Raises:
        ValueError: If an axis has fewer than one point.
    """
    nx, ny = shape
    if nx < 1 or ny < 1:
        raise ValueError(f"the grid needs at least one point per axis, got {shape}")

    def axis(interval: tuple[float, float], n: int) -> FloatArray:
        lo, hi = interval
        result: FloatArray = np.array([0.5 * (lo + hi)]) if n == 1 else np.linspace(lo, hi, n)
        return result

    xs, ys = axis(spawn.x, nx), axis(spawn.y, ny)
    grid: FloatArray = np.array([[x, y] for x in xs for y in ys])
    return grid


def progress_bound(
    p_x0: ArrayLike,
    steps: int,
    params: PhysicsParams,
    *,
    speed0: ArrayLike = 0.0,
    accel: float | None = None,
) -> FloatArray:
    """``p_x0`` plus the longest distance coverable in ``k = 0, ..., steps`` steps (F7).

    Since ``p_x`` moves at most by the distance covered, this bounds ``p_x``
    after ``k`` steps on every layout.

    Args:
        p_x0: ``(N,)`` initial ``p_x``.
        steps: Number of steps.
        params: Physical parameters.
        speed0: Initial speeds ``||v_0||``, scalar or ``(N,)``, at most ``v_max``.
        accel: Bound ``A`` on ``||u + d||``; ``a_max + d_bar`` by default.

    Returns:
        ``(N, steps + 1)`` bounds; column ``k`` bounds ``p_x`` after ``k`` steps.

    Raises:
        ValueError: If ``steps`` is negative, a speed exceeds ``v_max`` or
            ``accel`` is not positive.
    """
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    a = params.a_max + params.d_bar if accel is None else accel
    if not a > 0.0:
        raise ValueError(f"accel must be positive, got {a}")
    p = np.atleast_1d(np.asarray(p_x0, dtype=np.float64)).copy()
    s = np.broadcast_to(np.asarray(speed0, dtype=np.float64), p.shape).copy()
    if np.any(s < 0.0) or np.any(s > params.v_max + TOL):
        raise ValueError("every speed0 must lie in [0, v_max]")
    dt = params.dt
    bound = np.empty((p.size, steps + 1))
    bound[:, 0] = p
    for k in range(1, steps + 1):
        # Same operations as stages 2, 3 and 6 of the environment step along +x.
        s = np.minimum(params.velocity_factor * s + dt * a, params.v_max)
        p = p + dt * s
        bound[:, k] = p
    return bound


def min_steps_lower_bound(
    p_x0: ArrayLike,
    goal_x: float | ArrayLike,
    params: PhysicsParams,
    *,
    speed0: ArrayLike = 0.0,
    accel: float | None = None,
    max_steps: int = 100_000,
) -> IntArray:
    """Fewest steps after which :func:`progress_bound` reaches ``goal_x``.

    Without obstacles this bounds the length of every successful episode from
    below (F7); with ``d_bar = 0``, from rest and away from the walls, it is
    the minimum. :func:`arrival_lower_bound` covers layouts with obstacles.

    Args:
        p_x0: ``(N,)`` initial ``p_x``; zero steps from ``p_x0 >= goal_x``.
        goal_x: Target ``p_x``, scalar or one per start.
        params: Physical parameters.
        speed0: Initial speeds, as in :func:`progress_bound`.
        accel: As in :func:`progress_bound`.
        max_steps: Steps after which the search gives up.

    Returns:
        ``(N,)`` numbers of steps.

    Raises:
        ValueError: If some start needs more than ``max_steps`` steps, or as
            :func:`progress_bound`.
    """
    p0 = np.atleast_1d(np.asarray(p_x0, dtype=np.float64))
    goal = np.broadcast_to(np.asarray(goal_x, dtype=np.float64), p0.shape)
    steps = np.where(p0 >= goal, 0, -1).astype(np.int64)
    horizon, start = 64, 0
    while np.any(steps < 0):
        if start >= max_steps:
            raise ValueError(f"some starts need more than max_steps={max_steps} steps")
        n = min(horizon, max_steps)
        bound = progress_bound(p0, n, params, speed0=speed0, accel=accel)
        reached = bound >= goal[:, None]
        first = np.where(reached.any(axis=1), reached.argmax(axis=1), -1)
        pending = steps < 0
        steps[pending] = first[pending]
        horizon, start = 2 * horizon, n
    return steps


def arrival_lower_bound(
    layout: Layout,
    starts: ArrayLike,
    params: PhysicsParams,
    *,
    speed0: ArrayLike = 0.0,
    accel: float | None = None,
    distance: ArrayLike | None = None,
) -> IntArray:
    """Fewest steps in which any episode from each start can reach the goal region (F7).

    The steps needed to cover ``d_free``, the shortest free path to the goal
    region. Where that path is the straight segment along ``+x`` (always
    without obstacles), the bound is :func:`min_steps_lower_bound` with
    ``goal_x`` itself, exact in floating point; otherwise ``PATH_MARGIN`` is
    taken off the path.

    Args:
        layout: The layout.
        starts: ``(N, 2)`` free positions.
        params: Physical parameters.
        speed0: Initial speeds, as in :func:`progress_bound`.
        accel: As in :func:`progress_bound`.
        distance: ``d_free`` of each start, if already computed by
            :func:`hrlmpc.paths.free_path_distance`.

    Returns:
        ``(N,)`` numbers of steps.
    """
    points = np.atleast_2d(np.asarray(starts, dtype=np.float64))
    d = free_path_distance(layout, points) if distance is None else np.asarray(distance, dtype=np.float64)
    straight = d <= layout.goal_x - points[:, 0]
    targets = np.where(straight, layout.goal_x, points[:, 0] + d - PATH_MARGIN)
    return min_steps_lower_bound(points[:, 0], targets, params, speed0=speed0, accel=accel)


@dataclass(frozen=True, eq=False)
class Evaluation:
    """Outcome of an evaluation: one episode from each start.

    Attributes:
        starts: ``(N, 2)`` initial positions, at rest.
        episodes: Statistics of the episode from each start, in the order of ``starts``.
        lower_bounds: ``(N,)`` bounds of F7 on the length of a successful
            episode from each start (:func:`arrival_lower_bound`); the
            minimum time in the tunnel without disturbance.
        path_lengths: ``(N,)`` lengths of the shortest free paths to the goal region.
        step_ms: The controller's wall-clock time [ms] at every step, per agent: each call, for all the
            agents, divided by their number (D50); empty if not measured.
        duration_s: Wall-clock time [s] of the whole evaluation; NaN if not measured.
    """

    starts: FloatArray
    episodes: tuple[EpisodeStats, ...]
    lower_bounds: IntArray
    path_lengths: FloatArray
    step_ms: FloatArray = field(default_factory=lambda: np.zeros(0))
    duration_s: float = math.nan

    def metrics(self, prefix: str = "eval/") -> dict[str, float]:
        """Means over the episodes, the success rate and the arrival-time gap.

        The arrival gap of a successful episode is its length relative to the
        bound, minus one; its mean and maximum are over the successful
        episodes only (NaN if there are none). ``contact_fraction`` is the
        fraction of episodes with at least one contact step.
        """
        eps = self.episodes
        success = np.array([ep.success for ep in eps])
        lengths = np.array([ep.length for ep in eps], dtype=np.float64)
        gaps = lengths[success] / self.lower_bounds[success] - 1.0
        result = {
            "success_rate": float(success.mean()),
            "return": float(np.mean([ep.episode_return for ep in eps])),
            "unshaped_return": float(np.mean([ep.unshaped_return for ep in eps])),
            "length": float(lengths.mean()),
            "arrival_gap_mean": float(gaps.mean()) if gaps.size else float("nan"),
            "arrival_gap_max": float(gaps.max()) if gaps.size else float("nan"),
            "contact_fraction": float(np.mean([ep.contact_steps > 0 for ep in eps])),
        }
        for name in _MEAN_FIELDS:
            result[name] = float(np.mean([getattr(ep, name) for ep in eps]))
        return {prefix + key: value for key, value in result.items()}

    def timing(self, name: str) -> dict[str, float]:
        """The compute of the evaluation ``name`` (e.g. ``"eval_mode"``) under ``time/`` (D50): the controller's
        mean and largest time per agent and step [ms], and the evaluation's duration [s]; NaN where not measured.
        """
        measured = self.step_ms.size > 0
        return {
            f"time/{name}_s": self.duration_s,
            f"time/{name}_step_ms_mean": float(self.step_ms.mean()) if measured else math.nan,
            f"time/{name}_step_ms_max": float(self.step_ms.max()) if measured else math.nan,
        }

    def details(self) -> dict[str, Any]:
        """Per-start results as plain JSON values, for the run's ``evaluations.jsonl``."""
        result: dict[str, Any] = {
            "starts": self.starts.tolist(),
            "path_length": self.path_lengths.tolist(),
            "lower_bound": [int(n) for n in self.lower_bounds],
        }
        for entry in fields(EpisodeStats):
            if entry.name != "env":
                result[entry.name] = [getattr(ep, entry.name) for ep in self.episodes]
        return result


def evaluate(
    controller: Controller, config: EnvConfig, starts: ArrayLike, *, seed: int, inputs: bool | None = None
) -> Evaluation:
    """Run one episode of ``controller`` from each start, at rest.

    The episodes run in a fresh :class:`~hrlmpc.env.NavigationEnv` with one
    agent per start, so they do not count as training samples (D6). The
    disturbances of an agent depend only on ``seed`` and its start index.

    All the agents start together, and agents that have finished keep being
    stepped (in new episodes, ignored) until the last one finishes. A
    controller with memory is reset first, so it sees the episodes it is
    evaluated on from their first step, in step with each other (D29). Every
    call of the controller is timed (D50).

    Args:
        controller: Deterministic map from observations to commands.
        config: Layout to evaluate on.
        starts: ``(N, 2)`` initial positions in the free space, outside the goal region.
        seed: Seed of the evaluation environment.
        inputs: Whether the commands are accelerations, as from an MPC Worker
            (:meth:`~hrlmpc.env.NavigationEnv.step_input`), instead of actions
            in the square mapped by D15; by default the controller's ``inputs``
            attribute, if it has one, and ``False`` otherwise.

    Returns:
        The episodes, the bounds of F7 for each start and the compute.

    Raises:
        ValueError: If ``starts`` is not ``(N, 2)`` or a start is not
            admissible, or ``inputs`` contradicts the controller's ``inputs`` attribute.
    """
    started = time.perf_counter()
    points = np.asarray(starts, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError(f"starts must have shape (N, 2) with N >= 1, got {points.shape}")
    declared = getattr(controller, "inputs", None)
    if inputs is None:
        inputs = bool(declared)
    elif declared is not None and bool(declared) != inputs:
        raise ValueError(f"inputs={inputs} contradicts the controller's inputs={declared}")
    env = NavigationEnv(config, num_envs=len(points), seed=seed, autoreset=True)
    obs = env.reset(positions=points)
    reset = getattr(controller, "reset", None)
    if callable(reset):
        reset(len(points))
    first: dict[int, EpisodeStats] = {}
    step_ms: list[float] = []
    while len(first) < len(points):
        tick = time.perf_counter()
        commands = controller(obs)
        step_ms.append(1e3 * (time.perf_counter() - tick) / len(points))
        out = env.step_input(commands) if inputs else env.step(commands)
        for episode in out.episodes:
            first.setdefault(episode.env, episode)
        obs = out.obs
    lengths = free_path_distance(env.layout, points)
    bounds = arrival_lower_bound(env.layout, points, config.physics, distance=lengths)
    episodes = tuple(first[i] for i in range(len(points)))
    duration = time.perf_counter() - started
    return Evaluation(points.copy(), episodes, bounds, lengths, np.array(step_ms), duration)


def evaluation_name(action: str, group: str = "") -> str:
    """The name under which an evaluation with ``action`` logs the metrics of ``group`` (D50): ``eval`` and
    ``eval_<group>`` for the mean, as in steps 7-10; ``eval_mode`` and ``eval_mode_<group>`` for the mode.

    Raises:
        ValueError: If ``action`` is not one of :data:`EVAL_ACTIONS`.
    """
    if action not in EVAL_ACTIONS:
        raise ValueError(f"an evaluation's action is one of {EVAL_ACTIONS}, got {action!r}")
    name = "eval" if action == "mean" else f"eval_{action}"
    return f"{name}_{group}" if group else name


def append_details(path: str | Path, evaluation: Evaluation, *, env_steps: int, action: str) -> None:
    """Append the per-start results of ``evaluation`` with ``action`` to a run's ``evaluations.jsonl`` as one line,
    with the sample counter and the action (D50).

    Raises:
        ValueError: If ``action`` is not one of :data:`EVAL_ACTIONS`; nothing is written then.
    """
    evaluation_name(action)  # checks the action
    line = json.dumps({"env_steps": env_steps, "action": action, **evaluation.details()})
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _gap(value: float) -> str:
    return "  -  " if math.isnan(value) else f"{value:5.3f}"


def console_line(metrics: Mapping[str, Any]) -> str:
    """The outcome of the evaluations of both actions in one line, ``mean/mode`` (D50): success rate, unshaped
    return, mean arrival gap and contact fraction."""
    mean, mode = (f"{evaluation_name(action)}/" for action in EVAL_ACTIONS)

    def pair(key: str, form: Callable[[float], str]) -> str:
        return f"{form(float(metrics[mean + key]))}/{form(float(metrics[mode + key]))}"

    return (
        f"mean/mode: success {pair('success_rate', lambda v: f'{v:4.2f}')}"
        f" | unshaped return {pair('unshaped_return', lambda v: f'{v:.1f}')}"
        f" | gap {pair('arrival_gap_mean', _gap)}"
        f" | contact {pair('contact_fraction', lambda v: f'{v:4.2f}')}"
    )
