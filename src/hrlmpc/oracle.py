"""Minimum-time oracle (decision log D44): the reference of the arrival times on a layout.

For each start of the evaluation grid (D20) the oracle is the open-loop plan,
without disturbance, that reaches the goal region in the fewest steps and,
among those, with the least effort ``sum_k ||u_k||^2``, within the class of
plans defined below. The minimum time replaces the lower bound of F7
(:func:`hrlmpc.evaluation.arrival_lower_bound`) as the reference of the
arrival times where obstacles make the bound loose; without obstacles the two
agree (F7 is attained by thrusting along ``+x``), a check of the oracle.

Problem for a horizon ``N``, a mixed-integer program with convex quadratic
constraints (exact disks, no polygonal relaxation), solved by Gurobi to its
MIP gap, or by Clarabel as a second-order cone program when the reach boxes
fix every binary (every horizon without obstacles):

    min  sum_{k<N} ||u_k||^2
    s.t. x_0 = (p_0, 0),  x_{k+1} = A x_k + B u_k,                k = 0, ..., N-1
         ||u_k||^2 <= (a_max - eps)^2,                            k = 0, ..., N-1
         ||v_k||^2 <= (v_max - eps)^2,                            k = 1, ..., N
         m_i' p_k <= b_i - rho                   (every face i of the arena),
         n_j' p_k >= c_j + rho - M_j (1 - delta_{k,j}),
         n_j' p_{k+1} >= c_j + rho - M_j (1 - delta_{k,j}),
         sum_j delta_{k,j} >= 1                  (every obstacle, k = 0, ..., N-1; D32),
         p_{x,N} >= goal_x + goal_margin.

The first feasible ``N``, from the bound of F7 upwards, is the minimum time
and its plan the least-effort one. The obstacle constraints are the Worker's
(:mod:`hrlmpc.mpc_problem`): both ends of each segment lie beyond one face of
the obstacle enlarged by ``rho``, so the whole segment does, which is what the
environment's continuous collision detection requires. They are sufficient,
and slightly conservative where a segment rounds a vertex. The oracle is
therefore exact within its class: the contact-free plans that clear every
obstacle by one face per step and keep the margins. The margins are small
(defaults: ``rho`` 1 mm, the Worker's; ``eps`` 1e-6; ``goal_margin`` 1e-5 m)
and exist for the solver's tolerances: ``rho`` covers what a mixed-integer
plan may violate a face by, ``eps`` what the quadratic constraints may be
violated by, so that the replay's saturation and speed limiter stay inactive,
and ``goal_margin`` the tolerance at the goal line. A start whose true
optimum arrives within these margins of a constraint may be reported one step
late. Binaries of faces that no plan can reach within the horizon are fixed
beforehand from the bound of F7, without losing any feasible plan.

Oracle gap. The evaluation reports the arrival time of a policy minus the
oracle's (:mod:`hrlmpc.evaluation`). It is non-negative for every policy
within the oracle's class; a policy may beat the oracle, by a step or so,
where it rounds a vertex more tightly than one face per step allows, or by
braking on a wall (contact, which the oracle excludes and the reward
penalizes), so a small negative gap is possible and not an error.

Return. For a contact-free arrival in ``N`` steps the unshaped return of D13
(and the sparse reward of D51) is ``success_bonus - time_penalty * N -
effort_coef * sum_k ||u_k||^2 / a_max^2``, where the effort term lies in
``[0, effort_coef * N]``. When ``effort_coef * N < time_penalty`` no later
arrival can beat an earlier one, so the oracle's plan also maximizes the
unshaped return within its class, and its replayed return is the reference
return of the start (:func:`entry_for` refuses a plan that fails this
condition). Outside the class, the calibration of the contact penalty (D18,
an engineering estimate) is meant to keep braking on a wall from paying. The
shipped rewards (``effort_coef`` 0.01, ``time_penalty`` 1) satisfy the
condition for ``N < 100``.

Replay. Every plan is replayed through the environment's physical step
(:meth:`hrlmpc.env.NavigationEnv.step_input`) without disturbance, and the
table records what the environment measured: length, success, contacts,
physical corrections and unshaped return. A plan whose replay does not succeed
in exactly ``N`` steps, without contact and without any correction of the
input (saturation, speed limiter, wall reaction), is an error, not an entry:
the oracle's plan is coherent, the environment's trajectory is the model's,
and the table's numbers are facts about the environment whatever the solver's
tolerances were.

Tables. A table (:class:`OracleTable`) holds the physics (without ``d_bar``),
the geometry, the settings, the reward the returns were measured with and one
entry per start, as JSON under ``configs/oracle/`` (:data:`ORACLE_DIR`). The
evaluation finds the table of a layout by content (:func:`find_oracle`), not
by name, so one table serves every reward regime and disturbance level of a
layout; the disturbed runs are thus measured against the undisturbed minimum
time. Computing a table needs the solvers of the extra ``mpc`` (:class:`OracleSolver`:
Clarabel for the convex horizons, Gurobi for the mixed-integer ones); using one does not.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrlmpc.config import EnvConfig, env_config_to_dict, load_env_config
from hrlmpc.env import EpisodeStats, NavigationEnv
from hrlmpc.evaluation import arrival_lower_bound, progress_bound, spawn_grid
from hrlmpc.geometry import Layout
from hrlmpc.model import PhysicsParams, system_matrices
from hrlmpc.mpc_problem import FEAS_TOL, Bounds, Obstacle, face_slack, layout_obstacles, tightened_arena

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ORACLE_DIR = Path(__file__).resolve().parents[2] / "configs" / "oracle"
"""Where the tables of the layouts live in the repository."""

Solve = Callable[[FloatArray, int], FloatArray | None]
"""``solve(start, N)``: the inputs ``(N, 2)`` of a least-effort plan with horizon ``N``, or ``None`` if none exists."""


@dataclass(frozen=True)
class OracleSettings:
    """Settings of the oracle's problem.

    Attributes:
        rho: Margin [m] the planned positions keep from every wall and obstacle face; it must exceed what a
            mixed-integer plan may violate a face by (:func:`hrlmpc.mpc_problem.face_slack`).
        goal_margin: How far beyond the goal line [m] the last position must lie, against the solver's tolerance.
        disk_margin: ``eps``, by how much the radii of the input and speed disks are reduced [m/s^2, m/s]: the
            solver may violate a quadratic constraint by its feasibility tolerance (:data:`hrlmpc.mpc_problem.FEAS_TOL`
            on the squared norm), and the margin must exceed that over twice the smaller radius, so that the plan's
            inputs and speeds stay strictly within the environment's disks.
        mip_gap: Gurobi's relative MIP gap on the effort.
        max_extra_steps: How many horizons beyond the bound of F7 the search tries before giving up.
    """

    rho: float = 1e-3
    goal_margin: float = 1e-5
    disk_margin: float = 1e-6
    mip_gap: float = 1e-4
    max_extra_steps: int = 60

    def __post_init__(self) -> None:
        if not self.rho > 0.0 or not self.goal_margin > 0.0 or not self.disk_margin > 0.0:
            raise ValueError("rho, goal_margin and disk_margin must be positive")
        if not 0.0 <= self.mip_gap < 1.0:
            raise ValueError(f"mip_gap must lie in [0, 1), got {self.mip_gap}")
        if self.max_extra_steps < 0:
            raise ValueError("max_extra_steps must be non-negative")


@dataclass(frozen=True)
class OracleEntry:
    """The oracle of one start.

    Attributes:
        start: The initial position ``(p_x, p_y)``, at rest.
        lower_bound: The bound of F7 on the arrival time, in steps.
        steps: The minimum time ``N``, in steps.
        effort: ``sum_k ||u_k||^2`` of the plan [m^2/s^4].
        inputs: The ``N`` inputs of the plan, ``((u_x, u_y), ...)``.
        length: Length of the replayed episode (``N`` for a valid entry).
        success: Whether the replay reached the goal region (true for a valid entry).
        contact_steps: Contact steps of the replay (0 for a valid entry).
        corrections: Steps of the replay in which the saturation onto ``U``, the speed limiter or a wall reaction
            corrected the plan (0 for a valid entry: the plan is coherent).
        unshaped_return: Unshaped return of the replay (D13): the reference return of the start (see the module).
    """

    start: tuple[float, float]
    lower_bound: int
    steps: int
    effort: float
    inputs: tuple[tuple[float, float], ...]
    length: int
    success: bool
    contact_steps: int
    corrections: int
    unshaped_return: float

    @property
    def valid(self) -> bool:
        """Whether the replay succeeded in exactly ``steps`` steps without contact and without any correction."""
        return self.success and self.length == self.steps and self.contact_steps == 0 and self.corrections == 0


def layout_signature(config: EnvConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """What the oracle of a layout depends on: the physics without ``d_bar``, and the geometry."""
    data = env_config_to_dict(config)
    physics = {key: value for key, value in data["physics"].items() if key != "d_bar"}
    return physics, data["geometry"]


@dataclass(frozen=True)
class OracleTable:
    """The oracle of every start of a grid on one layout.

    Attributes:
        physics: The physics the table was computed for, without ``d_bar`` (the oracle is undisturbed).
        geometry: The geometry the table was computed for.
        settings: The :class:`OracleSettings` used, as a dict.
        entries: One entry per start.
        source: Free text on how the table was computed (solver, version, date).
        reward: The reward the entries' returns were measured with, as a dict; recorded, not matched (the
            minimum times do not depend on it).
    """

    physics: dict[str, Any]
    geometry: dict[str, Any]
    settings: dict[str, Any]
    entries: tuple[OracleEntry, ...]
    source: str = ""
    reward: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def starts(self) -> FloatArray:
        """``(N, 2)`` starts of the entries."""
        result: FloatArray = np.array([entry.start for entry in self.entries], dtype=np.float64).reshape(-1, 2)
        return result

    @property
    def steps(self) -> IntArray:
        """``(N,)`` minimum times of the entries."""
        return np.array([entry.steps for entry in self.entries], dtype=np.int64)

    def matches(self, config: EnvConfig) -> bool:
        """Whether the table was computed for ``config``'s physics (apart from ``d_bar``) and geometry."""
        physics, geometry = layout_signature(config)
        return self.physics == physics and self.geometry == geometry

    def steps_for(self, starts: ArrayLike, tol: float = 1e-9) -> IntArray:
        """The minimum times of ``starts``, which must be the table's starts in its order.

        Raises:
            ValueError: If the starts are not those of the table.
        """
        points = np.atleast_2d(np.asarray(starts, dtype=np.float64))
        mine = self.starts
        if points.shape != mine.shape or not np.allclose(points, mine, atol=tol, rtol=0.0):
            raise ValueError("the starts are not those of the oracle table")
        return self.steps

    def to_dict(self) -> dict[str, Any]:
        """The table as plain JSON values."""
        return {
            "physics": dict(self.physics),
            "geometry": self.geometry,
            "settings": dict(self.settings),
            "reward": dict(self.reward),
            "source": self.source,
            "entries": [dataclasses.asdict(entry) for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OracleTable:
        """The inverse of :meth:`to_dict`.

        Raises:
            ValueError: If the data lack a key or an entry is malformed.
        """
        try:
            entries = tuple(
                OracleEntry(
                    start=(float(e["start"][0]), float(e["start"][1])),
                    lower_bound=int(e["lower_bound"]),
                    steps=int(e["steps"]),
                    effort=float(e["effort"]),
                    inputs=tuple((float(u[0]), float(u[1])) for u in e["inputs"]),
                    length=int(e["length"]),
                    success=bool(e["success"]),
                    contact_steps=int(e["contact_steps"]),
                    corrections=int(e["corrections"]),
                    unshaped_return=float(e["unshaped_return"]),
                )
                for e in data["entries"]
            )
            return cls(
                physics=dict(data["physics"]),
                geometry=data["geometry"],
                settings=dict(data["settings"]),
                entries=entries,
                source=str(data.get("source", "")),
                reward=dict(data.get("reward", {})),
            )
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError(f"malformed oracle table: {error!r}") from error

    def save(self, path: str | Path) -> None:
        """Write the table as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> OracleTable:
        """Read a table written by :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def find_oracle(
    config: EnvConfig, starts: ArrayLike, directory: str | Path | None = None
) -> tuple[Path, OracleTable] | None:
    """The table of ``config``'s layout for ``starts`` among the JSON files of ``directory``, by content.

    Files are tried in name order; one that is not a table is skipped. ``None``
    if no table matches the physics (apart from ``d_bar``), the geometry and the starts.
    """
    folder = ORACLE_DIR if directory is None else Path(directory)
    if not folder.is_dir():
        return None
    for path in sorted(folder.glob("*.json")):
        try:
            table = OracleTable.load(path)
        except (ValueError, json.JSONDecodeError, OSError):
            continue
        if not table.matches(config):
            continue
        try:
            table.steps_for(starts)
        except ValueError:
            continue
        return path, table
    return None


def oracle_record(found: tuple[Path, OracleTable] | None) -> str | None:
    """The name of a found table for a run's ``config.yaml``: relative to the repository for a file under
    :data:`ORACLE_DIR`, the full path otherwise; ``None`` without a table."""
    if found is None:
        return None
    path = found[0].resolve()
    try:
        return path.relative_to(ORACLE_DIR.parent.parent).as_posix()
    except ValueError:
        return str(path)


def oracle_line(found: tuple[Path, OracleTable] | None) -> str:
    """One line for the console at the start of a run: which table the oracle gaps are measured against."""
    if found is None:
        return "no oracle table for the layout and starts: the oracle gaps are NaN (D44)"
    return f"oracle table {oracle_record(found)} ({len(found[1].entries)} starts)"


def undisturbed(config: EnvConfig) -> EnvConfig:
    """``config`` with ``d_bar = 0``: the oracle plans and replays without disturbance."""
    return dataclasses.replace(config, physics=dataclasses.replace(config.physics, d_bar=0.0))


class OracleProblem:
    """The data of the oracle's problem for one layout: everything but the solver.

    Args:
        config: The layout and physics; ``d_bar`` is ignored.
        settings: Margins, gap and search limit.

    Raises:
        ValueError: If the margin does not cover :func:`hrlmpc.mpc_problem.face_slack`, the arena or an obstacle
            leaves no room for it, or the disk margin does not cover the solver's feasibility tolerance.
    """

    def __init__(self, config: EnvConfig, settings: OracleSettings | None = None) -> None:
        self.config = undisturbed(config)
        self.settings = settings if settings is not None else OracleSettings()
        self.params: PhysicsParams = self.config.physics
        self.layout = Layout.from_config(self.config.geometry)
        self.A, self.B = system_matrices(self.params.dt, self.params.gamma)
        rho = self.settings.rho
        arena = self.layout.arena
        self.arena_normals = arena.normals.copy()
        self.arena_bounds = arena.offsets - rho
        tight = tightened_arena(self.layout, rho)
        self.position_low = tight.vertices.min(axis=0)
        self.position_high = tight.vertices.max(axis=0)
        self.obstacles: list[Obstacle] = layout_obstacles(self.layout, rho, tight)
        self.face_slack = face_slack(self.obstacles)
        if not rho > self.face_slack:
            raise ValueError(
                f"the margin rho = {rho} m must exceed what a mixed-integer plan may violate a face by, "
                f"{self.face_slack:.2e} m"
            )
        eps, radius = self.settings.disk_margin, min(self.params.a_max, self.params.v_max)
        if not eps < radius or eps * (2.0 * radius - eps) < FEAS_TOL:  # (r - eps)^2 + FEAS_TOL <= r^2
            raise ValueError(
                f"the disk margin {eps} must lie in ({FEAS_TOL / (2.0 * radius):.2e}, {radius}) to cover the "
                f"feasibility tolerance {FEAS_TOL} on the squared norms"
            )
        self.a_max = self.params.a_max - eps
        """Radius of the input disk of the problem."""
        self.v_max = self.params.v_max - eps
        """Radius of the speed disk of the problem."""
        self.goal_x = float(self.layout.goal_x)

    def lower_bound(self, start: ArrayLike) -> int:
        """The bound of F7 on the arrival time from ``start``, at rest, without disturbance."""
        return int(arrival_lower_bound(self.layout, np.atleast_2d(start), self.params)[0])

    def check_start(self, start: ArrayLike) -> FloatArray:
        """``start`` as a position, if a plan may start there: in the tightened arena, beyond a face of every
        obstacle by the margin (so that the first segment has a face) and short of the goal line.

        Raises:
            ValueError: Otherwise.
        """
        point = np.asarray(start, dtype=np.float64).reshape(-1)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError(f"a start is a finite position (p_x, p_y), got {start!r}")
        if np.any(self.arena_normals @ point > self.arena_bounds):
            raise ValueError(f"the start {point.tolist()} is not within the margin of the arena")
        for index, obstacle in enumerate(self.obstacles):
            if np.all(obstacle.normals @ point < obstacle.bounds):
                raise ValueError(f"the start {point.tolist()} is not beyond a face of obstacle {index} by the margin")
        if point[0] >= self.goal_x:
            raise ValueError(f"the start {point.tolist()} lies in the goal region")
        return point

    def reach_boxes(self, start: ArrayLike, horizon: int) -> tuple[FloatArray, FloatArray]:
        """Bounds on the positions of every plan from ``start`` at rest: stage ``k`` lies within the distance that
        F7 allows in ``k`` steps of the start on each axis, clipped to the tightened arena's bounding box.

        Returns:
            ``(lo, hi)``, each ``(horizon + 1, 2)``.
        """
        p = np.asarray(start, dtype=np.float64)[:2]
        travel = progress_bound(0.0, horizon, self.params)[0][:, None]  # the distance coverable in k steps
        lo = np.maximum(p - travel, self.position_low)
        hi = np.minimum(p + travel, self.position_high)
        return lo, np.maximum(hi, lo)

    def fixed_binaries(self, start: ArrayLike, horizon: int) -> Bounds:
        """The binaries' bounds for a solve from ``start``: faces that every plan satisfies are fixed.

        As :meth:`hrlmpc.mpc_problem.MPCProblem.fixed_binaries`: a face is satisfied on segment ``k`` by every
        plan when the whole box of stage ``k + 1`` lies beyond it; its binary is fixed to 1 and the obstacle's
        others on that segment to 0, which drops inactive constraints only.
        """
        lo, hi = self.reach_boxes(start, horizon)
        centre, half = 0.5 * (lo + hi)[1:], 0.5 * (hi - lo)[1:]
        result: Bounds = []
        for obstacle in self.obstacles:
            lowest = centre @ obstacle.normals.T - half @ np.abs(obstacle.normals).T  # (horizon, m)
            certified = lowest >= obstacle.bounds
            lb = np.zeros((horizon, obstacle.num_faces))
            ub = np.ones((horizon, obstacle.num_faces))
            rows = np.flatnonzero(certified.any(axis=1))
            first = certified[rows].argmax(axis=1)
            ub[rows] = 0.0
            lb[rows, first] = 1.0
            ub[rows, first] = 1.0
            result.append((lb, ub))
        return result

    def minimum_time(self, start: ArrayLike, solve: Solve) -> tuple[int, FloatArray]:
        """The minimum time from ``start`` and its least-effort plan: the first feasible horizon from the bound
        of F7 upwards, within ``max_extra_steps`` of it.

        Args:
            start: The initial position, at rest.
            solve: :data:`Solve`, e.g. :meth:`OracleSolver.solve`.

        Raises:
            ValueError: If a plan may not start at ``start`` (:meth:`check_start`).
            RuntimeError: If no horizon within the search limit admits a plan.
        """
        point = self.check_start(start)
        bound = self.lower_bound(point)
        for horizon in range(bound, bound + self.settings.max_extra_steps + 1):
            inputs = solve(point, horizon)
            if inputs is not None:
                plan = np.asarray(inputs, dtype=np.float64)
                if plan.shape != (horizon, 2):
                    raise RuntimeError(f"solve returned inputs of shape {plan.shape}, expected ({horizon}, 2)")
                return horizon, plan
        raise RuntimeError(
            f"no plan from {point.tolist()} within {bound + self.settings.max_extra_steps} steps "
            f"(bound of F7: {bound}); the search limit max_extra_steps may be too small"
        )


def replay(config: EnvConfig, start: ArrayLike, inputs: ArrayLike) -> EpisodeStats | None:
    """Apply ``inputs`` from ``start`` at rest through the environment's physical step without disturbance.

    Returns:
        The statistics of the episode if it ended within the inputs, else ``None``.
    """
    plan = np.atleast_2d(np.asarray(inputs, dtype=np.float64))
    env = NavigationEnv(undisturbed(config), num_envs=1, seed=0, autoreset=False)
    env.reset(positions=np.atleast_2d(np.asarray(start, dtype=np.float64)))
    for u in plan:
        out = env.step_input(u[None, :])
        if out.episodes:
            return out.episodes[0]
    return None


def entry_for(problem: OracleProblem, start: ArrayLike, steps: int, inputs: ArrayLike) -> OracleEntry:
    """The entry of ``start`` for the plan ``inputs`` with minimum time ``steps``, replayed through the environment.

    Raises:
        RuntimeError: If the replay does not reach the goal region in exactly ``steps`` steps, without contact and
            without any correction of the input (the margins do not cover the solver's tolerances, or the plan is
            not the oracle's), or if ``effort_coef * steps >= time_penalty``, when a later arrival with less effort
            could beat the plan on the unshaped return (see the module).
    """
    point = np.asarray(start, dtype=np.float64)
    plan = np.atleast_2d(np.asarray(inputs, dtype=np.float64))
    stats = replay(problem.config, point, plan)
    entry = OracleEntry(
        start=(float(point[0]), float(point[1])),
        lower_bound=problem.lower_bound(point),
        steps=int(steps),
        effort=float(np.sum(plan**2)),
        inputs=tuple((float(u[0]), float(u[1])) for u in plan),
        length=int(stats.length) if stats is not None else int(len(plan)),
        success=bool(stats.success) if stats is not None else False,
        contact_steps=int(stats.contact_steps) if stats is not None else 0,
        corrections=int(stats.saturation_steps + stats.limiter_steps + stats.wall_steps) if stats is not None else 0,
        unshaped_return=float(stats.unshaped_return) if stats is not None else float("nan"),
    )
    if not entry.valid:
        raise RuntimeError(
            f"the plan from {point.tolist()} replays to length {entry.length}, success {entry.success}, "
            f"contact steps {entry.contact_steps}, corrected steps {entry.corrections}; expected success in "
            f"{steps} steps without contact or correction"
        )
    reward = problem.config.reward
    if not reward.effort_coef * entry.steps < reward.time_penalty:
        raise RuntimeError(
            f"effort_coef * steps = {reward.effort_coef} * {entry.steps} is not below time_penalty = "
            f"{reward.time_penalty}: the least-effort plan of the minimum time need not maximize the unshaped return"
        )
    return entry


def compute_oracle(
    problem: OracleProblem,
    starts: ArrayLike,
    solve: Solve,
    *,
    source: str = "",
    echo: Callable[[str], None] | None = None,
) -> OracleTable:
    """The table of ``starts`` on ``problem``'s layout, each plan replayed through the environment.

    Args:
        problem: The layout and settings.
        starts: ``(N, 2)`` initial positions, at rest, in the free space.
        solve: :data:`Solve`.
        source: Recorded in the table.
        echo: Receives one line per start.

    Raises:
        ValueError: If a plan may not start at one of the starts (:meth:`OracleProblem.check_start`).
        RuntimeError: As :meth:`OracleProblem.minimum_time` and :func:`entry_for`.
    """
    points = np.atleast_2d(np.asarray(starts, dtype=np.float64))
    for point in points:
        problem.check_start(point)  # every start, before the first solve
    entries = []
    for point in points:
        tick = time.perf_counter()
        steps, inputs = problem.minimum_time(point, solve)
        entry = entry_for(problem, point, steps, inputs)
        entries.append(entry)
        if echo is not None:
            echo(
                f"start ({point[0]:.3f}, {point[1]:.3f}): bound {entry.lower_bound}, minimum time {entry.steps}, "
                f"effort {entry.effort:.3f}, unshaped return {entry.unshaped_return:.2f} "
                f"({time.perf_counter() - tick:.1f} s)"
            )
    physics, geometry = layout_signature(problem.config)
    reward = env_config_to_dict(problem.config)["reward"]
    return OracleTable(physics, geometry, dataclasses.asdict(problem.settings), tuple(entries), source, reward)


GUROBI_RETRIES: tuple[dict[str, Any], ...] = (
    {},
    {"NumericFocus": 3, "BarHomogeneous": 1},
    {"NumericFocus": 3, "BarHomogeneous": 1, "ScaleFlag": 2, "Aggregate": 0, "Presolve": 0},
)
"""Gurobi's parameters tried in turn on a mixed-integer horizon: the defaults, then a numerically careful barrier
(a homogeneous barrier copes with near-degenerate relaxations, as at a horizon the plan can only just meet),
then that without presolve and with aggressive scaling. A horizon whose solve ends with a numerical status after
the last setting is an error."""

NUMERICAL_STATUSES = frozenset({12, 13})
"""Gurobi's ``NUMERIC`` and ``SUBOPTIMAL`` statuses: the solve did not end cleanly and is tried again."""


class OracleSolver:
    """Solves the oracle's problem: one model per start and horizon.

    A horizon whose binaries the reach boxes all fix (:meth:`OracleProblem.fixed_binaries`; every horizon
    without obstacles, as on the tunnel) is convex, a second-order cone program that Clarabel solves through
    cvxpy, as the Worker's exact problem (D34); the others go to Gurobi, with the settings of
    :data:`GUROBI_RETRIES` in turn when a solve ends with a numerical status.

    Args:
        problem: The layout and settings.

    Raises:
        ImportError: If gurobipy or cvxpy is not installed (the optional extra ``mpc``).
    """

    def __init__(self, problem: OracleProblem) -> None:
        import cvxpy as cp  # the optional extra ``mpc``
        import gurobipy as gp

        self.problem = problem
        self._cp = cp
        self._gp = gp
        self._env: Any = gp.Env(empty=True)
        self._env.setParam("OutputFlag", 0)
        self._env.start()
        self.convex_solves = 0
        """How many horizons Clarabel solved."""
        self.mixed_solves = 0
        """How many horizons Gurobi solved."""
        self.retries = 0
        """How many Gurobi solves needed another setting of :data:`GUROBI_RETRIES`."""

    def solve(self, start: ArrayLike, horizon: int) -> FloatArray | None:
        """The inputs ``(horizon, 2)`` of the least-effort plan with ``horizon`` steps from ``start`` at rest, or
        ``None`` if the problem is infeasible.

        Raises:
            ValueError: If the horizon is below 1.
            RuntimeError: If a solver ends with a status that is neither optimal nor infeasible.
            gurobipy.GurobiError: If Gurobi fails, e.g. for want of a licence that covers the model.
        """
        point = np.asarray(start, dtype=np.float64)[:2]
        N = int(horizon)
        if N < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        bounds = self.problem.fixed_binaries(point, N)
        if all(np.array_equal(lb, ub) for lb, ub in bounds):
            self.convex_solves += 1
            return self._solve_convex(point, N, [lb for lb, _ in bounds])
        self.mixed_solves += 1
        return self._solve_mixed(point, N, bounds)

    def _solve_convex(self, point: FloatArray, N: int, deltas: list[FloatArray]) -> FloatArray | None:
        """Clarabel on the problem with every binary fixed to ``deltas``: the selected faces hold on both ends."""
        cp, pr = self._cp, self.problem
        z, v = cp.Variable((N + 1, 4)), cp.Variable((N, 2))
        cons = [
            z[0] == np.array([point[0], point[1], 0.0, 0.0]),
            z[1:] == z[:-1] @ pr.A.T + v @ pr.B.T,
            cp.norm(v, 2, axis=1) <= pr.a_max,  # exact disks (D44), less the disk margin
            cp.norm(z[1:, 2:], 2, axis=1) <= pr.v_max,
            z[:, :2] @ pr.arena_normals.T <= np.tile(pr.arena_bounds, (N + 1, 1)),
            z[N, 0] >= pr.goal_x + pr.settings.goal_margin,
        ]
        for obstacle, delta in zip(pr.obstacles, deltas, strict=True):
            rhs = obstacle.bounds - obstacle.big_m * (1.0 - delta)  # (N, m): the face's bound where fixed to 1
            cons += [z[:N, :2] @ obstacle.normals.T >= rhs, z[1:, :2] @ obstacle.normals.T >= rhs]  # both ends
        prob = cp.Problem(cp.Minimize(cp.sum_squares(v)), cons)
        try:
            prob.solve(solver=cp.CLARABEL)
        except cp.SolverError as error:
            raise RuntimeError(f"Clarabel failed on the oracle's problem with horizon {N}: {error}") from error
        if prob.status == cp.OPTIMAL and v.value is not None:
            result: FloatArray = np.array(v.value, dtype=np.float64).reshape(N, 2)
            return result
        if prob.status == cp.INFEASIBLE:
            return None
        raise RuntimeError(f"Clarabel ended with status {prob.status!r} on the oracle's problem with horizon {N}")

    def _solve_mixed(self, point: FloatArray, N: int, bounds: Bounds) -> FloatArray | None:
        """Gurobi on the mixed-integer problem, with the settings of :data:`GUROBI_RETRIES` in turn."""
        gp, pr = self._gp, self.problem
        GRB = gp.GRB
        a_max, v_max = pr.a_max, pr.v_max  # the radii less the disk margin
        x_lb = np.full((N + 1, 4), -GRB.INFINITY)
        x_ub = np.full((N + 1, 4), GRB.INFINITY)
        x_lb[:, :2], x_ub[:, :2] = pr.position_low, pr.position_high
        x_lb[:, 2:], x_ub[:, 2:] = -v_max, v_max
        m = gp.Model(env=self._env)
        try:
            m.Params.Threads = 1
            m.Params.MIPGap = pr.settings.mip_gap
            x = m.addMVar((N + 1, 4), lb=x_lb, ub=x_ub, name="x")
            u = m.addMVar((N, 2), lb=-a_max, ub=a_max, name="u")
            m.addConstr(x[0] == np.array([point[0], point[1], 0.0, 0.0]), name="init")
            m.addConstr(x[1:] == x[:-1] @ pr.A.T + u @ pr.B.T, name="dynamics")
            xs, us = x.tolist(), u.tolist()  # the scalar variables, for the quadratic parts
            objective = gp.QuadExpr()
            for k in range(N):
                ux, uy = us[k]
                vx, vy = xs[k + 1][2], xs[k + 1][3]
                m.addQConstr(ux * ux + uy * uy <= a_max**2, name=f"input{k}")  # exact disks (D44)
                m.addQConstr(vx * vx + vy * vy <= v_max**2, name=f"speed{k + 1}")
                objective.add(ux * ux + uy * uy)
            m.addConstr(x[:, :2] @ pr.arena_normals.T <= np.tile(pr.arena_bounds, (N + 1, 1)), name="arena")
            for i, (obstacle, (lb, ub)) in enumerate(zip(pr.obstacles, bounds, strict=True)):
                d = m.addMVar((N, obstacle.num_faces), lb=lb, ub=ub, vtype=GRB.BINARY, name=f"delta{i}")
                rhs = obstacle.bounds - obstacle.big_m
                m.addConstr(x[:N, :2] @ obstacle.normals.T - d * obstacle.big_m >= rhs, name=f"first_end{i}")
                m.addConstr(x[1:, :2] @ obstacle.normals.T - d * obstacle.big_m >= rhs, name=f"second_end{i}")
                m.addConstr(d.sum(axis=1) >= 1, name=f"any{i}")
            m.addConstr(x[N, 0] >= pr.goal_x + pr.settings.goal_margin, name="goal")
            m.setObjective(objective, GRB.MINIMIZE)
            status = None
            for attempt, settings in enumerate(GUROBI_RETRIES):
                if attempt > 0:
                    self.retries += 1
                    m.reset(0)  # discard the last solve, so that the new setting starts afresh
                for name, value in settings.items():
                    m.setParam(name, value)
                m.optimize()
                status = m.Status
                if status not in NUMERICAL_STATUSES:
                    break
            if status == GRB.OPTIMAL:
                result: FloatArray = np.array(u.X, dtype=np.float64).reshape(N, 2)
                return result
            if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
                return None
            tried = f" after {len(GUROBI_RETRIES)} settings" if status in NUMERICAL_STATUSES else ""
            raise RuntimeError(
                f"Gurobi ended with status {status} ({gurobi_status_name(status)}) on the oracle's problem with "
                f"horizon {N}{tried}"
            )
        finally:
            m.dispose()

    def close(self) -> None:
        """Release the Gurobi environment; later calls do nothing."""
        env, self._env = self._env, None
        if env is not None:
            env.dispose()

    def __enter__(self) -> OracleSolver:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


_GUROBI_STATUS_NAMES = {
    1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED", 6: "CUTOFF",
    7: "ITERATION_LIMIT", 8: "NODE_LIMIT", 9: "TIME_LIMIT", 10: "SOLUTION_LIMIT", 11: "INTERRUPTED",
    12: "NUMERIC", 13: "SUBOPTIMAL", 14: "INPROGRESS", 15: "USER_OBJ_LIMIT", 16: "WORK_LIMIT", 17: "MEM_LIMIT",
}


def gurobi_status_name(status: int | None) -> str:
    """The name of a Gurobi status code, for messages."""
    return _GUROBI_STATUS_NAMES.get(int(status) if status is not None else -1, "unknown")


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: compute the oracle table of a layout and save it as JSON (needs Gurobi)."""
    parser = argparse.ArgumentParser(description="Compute the minimum-time oracle of a layout (D44).")
    parser.add_argument(
        "--env", type=Path, required=True, help="environment configuration, e.g. configs/env/slalom.yaml"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output JSON; by default configs/oracle/<env stem>.json"
    )
    parser.add_argument(
        "--grid", type=int, nargs=2, default=(5, 5), metavar=("NX", "NY"), help="starts on the spawn box (D20)"
    )
    parser.add_argument("--rho", type=float, default=OracleSettings.rho, help="margin from the walls [m]")
    parser.add_argument("--goal-margin", type=float, default=OracleSettings.goal_margin, help="margin at the goal [m]")
    parser.add_argument(
        "--disk-margin", type=float, default=OracleSettings.disk_margin, help="reduction of the disks' radii"
    )
    parser.add_argument("--mip-gap", type=float, default=OracleSettings.mip_gap, help="Gurobi's relative MIP gap")
    args = parser.parse_args(argv)
    config = load_env_config(args.env)
    settings = OracleSettings(
        rho=args.rho, goal_margin=args.goal_margin, disk_margin=args.disk_margin, mip_gap=args.mip_gap
    )
    problem = OracleProblem(config, settings)
    starts = spawn_grid(config.task.spawn, (int(args.grid[0]), int(args.grid[1])))
    out = ORACLE_DIR / f"{args.env.stem}.json" if args.out is None else args.out
    with OracleSolver(problem) as solver:
        import cvxpy as cp
        import gurobipy as gp

        source = (
            f"gurobipy {gp.gurobi.version()} and cvxpy {cp.__version__} with Clarabel on {time.strftime('%Y-%m-%d')}; "
            f"environment file {args.env}"
        )
        table = compute_oracle(problem, starts, solver.solve, source=source, echo=print)
        print(f"solves: {solver.convex_solves} convex (Clarabel), {solver.mixed_solves} mixed-integer (Gurobi), "
              f"{solver.retries} retried")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.save(out)
    print(f"written {out}: {len(table.entries)} starts, minimum times {table.steps.min()}-{table.steps.max()} steps")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
