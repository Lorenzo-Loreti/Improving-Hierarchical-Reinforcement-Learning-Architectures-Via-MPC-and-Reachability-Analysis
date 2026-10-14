"""Data of the MPC Worker's problem, without the solvers (decision log D31-D34, D37, D45, D46, D49).

The Worker of PPO_MPC and PPO_MPC_reachability (architectures 3 and 4) solves
at every step the robust tube-MPC problem of the tube-MPC note, specialised to
this plant and its polygonal map, with the cost of the waypoint Worker (D37)
and, with a tube, the regularizer of D46:

    min  q_g ||z_{k*,p} - g||^2 + r_e sum_{k<k*} ||v_k||^2 + eps ||x - z_0||^2,   k* the steps left in the segment
    s.t. z_{k+1} = A z_k + B v_k,                       k = 0, ..., N-1
         x - z_0 in Z                                   (the tube, D31)
         ||z_{k,v}|| <= v_bar,                           k = 1, ..., N-1   (D33)
         z_{N,v} = 0                                    (terminal set: at rest)
         ||v_k|| <= u_bar,                               k = 0, ..., N-1
         m_i' z_{k,p} <= b_i - r_p - rho,                every face i of the arena
         n_j' z_{k,p} >= c_j + r_p + rho - M_j (1 - delta_{k,j}),
         n_j' z_{k+1,p} >= c_j + r_p + rho - M_j (1 - delta_{k,j}),
         sum_j delta_{k,j} >= 1,                         every obstacle, k = 0, ..., N-1   (D32)

with ``x`` the measured state, ``g`` the target, the position at which the agent
should be when the segment ends, ``k* = H - j`` at step ``j`` of a segment of
``H`` steps, and the input ``u = v_0 + K (x - z_0)``. A reachable target is met
up to the trade-off with the effort; an unreachable one leaves the plan at the
reachable point nearest to it. After stage ``k*`` the plan only has to come to
rest by stage ``N``, so ``N >= H + n_b`` lets it pass the target at full speed
(:meth:`MPCProblem.check_segments`); without a horizon in the settings, ``N``
is that smallest one, ``H + n_b`` (D45). ``A``, ``B`` are the matrices of
:mod:`hrlmpc.model`; ``K`` the LQR gain of ``Q = diag(q_pos, q_pos, q_vel, q_vel)``,
``R = r I`` (D31); ``r_p`` the position radius of ``Z`` and ``rho`` a margin. The tightenings:

- ``v_bar = v_max - h_{A_K Z}(e_v)``: at the real state the input is nominally
  coherent, ``||(A x + B u)_v|| <= v_max``, so by F3 an action of the speed
  limiter is a disturbance ``d'`` in ``D`` and the error stays in ``Z`` (D33).
- ``u_bar = a_max - h_Z(K' e_u)``: ``||u|| <= a_max``, so the input saturation
  of the environment never acts.
- Positions by ``r_p + rho``: the real positions keep ``rho`` from every wall,
  less what the solvers' tolerances leave; ``rho`` must exceed the most a
  mixed-integer plan may violate a face by (:attr:`MPCProblem.face_slack`).
- One binary per obstacle face and *segment* ``[z_k, z_{k+1}]`` (D32): both
  ends lie beyond the same face of the enlarged obstacle, hence the whole
  nominal segment; the real segment ``[x_k, x_{k+1}]`` differs from it by
  convex combinations of errors in ``Z``, so it stays ``rho`` beyond the face of
  the obstacle itself, and the wall reaction of the environment never acts.

With a tube the nominal start ``z_0`` is free in ``x (+) (-Z)``, and the
first two terms see it only through the waypoint's position, so without the
third neither ``z_0`` nor the applied input need be unique. The third term
(D46) makes the cost strictly convex in ``z_0`` and in the inputs before the
waypoint, hence the optimal applied input unique for the binaries the search
settles on (faces may still tie). Without a tube ``z_0 = x``: the term is left
out and the problem is that of D37 (:attr:`MPCProblem.regularized`).

Every rest state is an equilibrium, so the shifted plan (drop the first step,
hold the last state at rest) is feasible at the next step whenever the
disturbance was in ``D``: theorem 4.1 of the note, under this terminal set.
Within a segment ``k*`` drops by one per step, so the shifted plan meets the
target at the same instant and costs its predecessor's cost minus the effort
``r_e ||v_0||^2`` of its first nominal input, plus the change of the third
term, ``eps (||x^+ - z_1||^2 - ||x - z_0||^2) <= eps (r_p^2 + r_v^2)`` while the
disturbance stays in ``D`` (``x^+ - z_1`` lies in ``Z``). Up to the MIP gap,
the optimal cost therefore does not grow within a segment without a tube, and
grows by at most ``eps (r_p^2 + r_v^2)`` per step with one (3.2e-3 at
``d_bar = 0.5``, 1.3e-2 at 1.0).

This module holds what does not need a solver: the matrices, the tube, the
tightened sets, the obstacles and their big-M constants, the binaries that the
reachable positions fix in advance, the shifted candidate, the cost and a
checker of plans (:meth:`MPCProblem.violations`, and :meth:`MPCProblem.plan_within`
with the tolerances of D49). :mod:`hrlmpc.mpc_worker` solves the problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrlmpc.geometry import ConvexPolygon, Layout
from hrlmpc.model import PhysicsParams, system_matrices
from hrlmpc.tube import TAIL_TOLERANCE, Tube, Weights, design_tube, is_rotation_invariant, lqr, rotation

FloatArray = NDArray[np.float64]

BIG_M_PAD = 1e-6
"""Padding of the big-M constants, so that a constant exactly at its bound does not bind."""

PLAN_TOL = 1e-4
"""How far a plan may violate a face and still count as satisfying it when its binaries are read
(:meth:`MPCProblem.binaries_of`, for the MIP start); above what the solvers leave on the slalom."""

INT_FEAS_TOL = 1e-5
"""Gurobi's default integrality tolerance: a binary at ``1 - 1e-5`` relaxes its face by ``M * 1e-5``."""

FEAS_TOL = 1e-6
"""Gurobi's default feasibility tolerance."""


@dataclass(frozen=True)
class MPCSettings:
    """Settings of the MPC Worker (D34, D37, D45, D46; provisional until Q13).

    Attributes:
        horizon: ``N``, the prediction horizon, fixed and receding; at least the
            segment's length plus the steps to stop (:meth:`MPCProblem.check_segments`).
            ``None`` sets it to that smallest horizon, which depends on the disturbance (D45).
        q_pos: Weight of each position error in the LQR of the ancillary gain ``K`` (D31).
        q_vel: Weight of each velocity in that LQR.
        r: Weight of each input in that LQR. ``K`` sets the tube, hence the
            tightened sets and the steps to stop, so the plan's effort has its own weight.
        target_weight: ``q_g``, weight of the squared distance from the target at the waypoint (D37).
        effort_weight: ``r_e``, weight of each nominal input until the waypoint (D37).
        start_weight: ``eps``, weight of ``||x - z_0||^2``, the distance of the plan's nominal start from the
            measured state, with a tube only (D46); 0 leaves the term out.
        rho: Safety margin [m] on the walls and the obstacles; it must exceed what
            a mixed-integer plan may violate a face by (checked by :class:`MPCProblem`).
        tail_tolerance: Bound on the tail of ``Z`` along the axes (D31) [m, m/s].
        mip_gap: Relative MIP gap of the mixed-integer solves, below 1.
        work_limit: Gurobi's deterministic work limit per solve, or ``None``.

    Raises:
        ValueError: If a setting is out of range.
    """

    horizon: int | None = None
    q_pos: float = 10.0
    q_vel: float = 1.0
    r: float = 0.1
    target_weight: float = 1e4
    effort_weight: float = 0.1
    start_weight: float = 0.1
    rho: float = 1e-3
    tail_tolerance: float = TAIL_TOLERANCE
    mip_gap: float = 1e-4
    work_limit: float | None = None

    def __post_init__(self) -> None:
        if self.horizon is not None and (
            isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1
        ):
            raise ValueError(f"horizon must be a positive integer or None, got {self.horizon!r}")
        Weights(self.q_pos, self.q_vel, self.r)  # validates the weights
        if isinstance(self.start_weight, bool) or not (np.isfinite(self.start_weight) and self.start_weight >= 0.0):
            raise ValueError(f"start_weight must be non-negative and finite, got {self.start_weight!r}")
        for name in ("target_weight", "effort_weight", "rho", "tail_tolerance", "mip_gap"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite, got {value!r}")
        if not self.mip_gap < 1.0:
            raise ValueError(f"mip_gap is a relative gap below 1, got {self.mip_gap!r}")
        if self.work_limit is not None and not (np.isfinite(self.work_limit) and self.work_limit > 0.0):
            raise ValueError(f"work_limit must be positive and finite or None, got {self.work_limit!r}")

    @property
    def weights(self) -> Weights:
        """The weights of the LQR that gives the ancillary gain."""
        return Weights(self.q_pos, self.q_vel, self.r)


@dataclass(frozen=True, eq=False)
class Obstacle:
    """One obstacle of the problem, in the position plane, with its live faces only.

    A face is live when some position of the tightened arena lies beyond it;
    the others, e.g. the faces that D16 pushed beyond a wall, can never
    separate a plan from the obstacle and get no binary.

    Attributes:
        normals: ``(m, 2)`` outward unit normals ``n_j`` of the live faces.
        offsets: ``(m,)`` offsets ``c_j``; the obstacle lies in ``{p : n_j' p <= c_j}``.
        bounds: ``(m,)`` the right-hand sides ``c_j + r_p + rho`` of the face constraints.
        big_m: ``(m,)`` constants ``M_j`` that deactivate a face anywhere in the arena.
    """

    normals: FloatArray
    offsets: FloatArray
    bounds: FloatArray
    big_m: FloatArray

    @property
    def num_faces(self) -> int:
        """The number of faces ``m``."""
        return int(self.normals.shape[0])


Binaries = list[FloatArray]
"""One ``(N, m)`` array of 0/1 per obstacle: the face each segment lies beyond."""

Bounds = list[tuple[FloatArray, FloatArray]]
"""One ``(lb, ub)`` pair of ``(N, m)`` arrays per obstacle: the binaries' bounds for one solve."""


def tightened_arena(layout: Layout, margin: float) -> ConvexPolygon:
    """The arena with every face pushed inwards by ``margin``: where the planned positions may lie.

    Raises:
        ValueError: If the arena leaves no room for the margin.
    """
    arena = layout.arena
    try:
        return arena.pushed_out(-margin * np.ones(len(arena.offsets)))
    except ValueError as error:
        raise ValueError(f"the arena leaves no room for a margin of {margin} m") from error


def layout_obstacles(layout: Layout, margin: float, tight: ConvexPolygon | None = None) -> list[Obstacle]:
    """The obstacles of a plan on ``layout`` whose positions keep ``margin`` from every face (D32): the blocked
    polygons of D16 with their live faces, the right-hand sides ``c_j + margin`` and the big-M constants.

    Shared by the Worker's problem and the oracle (D44).

    Args:
        layout: The map.
        margin: Distance [m] the planned positions keep from the faces.
        tight: :func:`tightened_arena` of ``layout`` and ``margin``, if already computed.

    Raises:
        ValueError: If an obstacle, enlarged by the margin, leaves no room in the arena.
    """
    arena = layout.arena
    tight = tightened_arena(layout, margin) if tight is None else tight
    result = []
    for index, polygon in enumerate(layout.blocked):
        bounds = polygon.offsets + margin
        live = (tight.vertices @ polygon.normals.T).max(axis=0) >= bounds
        if not live.any():
            raise ValueError(f"obstacle {index}, enlarged by the margin, leaves no room in the arena")
        normals, offsets, bounds = polygon.normals[live], polygon.offsets[live], bounds[live]
        lowest = (arena.vertices @ normals.T).min(axis=0)  # min over the arena of n_j' p
        big_m = np.maximum(0.0, bounds - lowest) + BIG_M_PAD
        result.append(Obstacle(normals.copy(), offsets.copy(), bounds, big_m))
    return result


def face_slack(obstacles: list[Obstacle]) -> float:
    """How far a mixed-integer plan may violate a face [m]: a binary within Gurobi's integrality tolerance of 1
    relaxes its face by ``M`` times it, plus the feasibility tolerance."""
    largest_m = max((float(obstacle.big_m.max()) for obstacle in obstacles), default=0.0)
    return INT_FEAS_TOL * largest_m + FEAS_TOL


class MPCProblem:
    """The offline data of the Worker's problem for one layout, plant and setting (D31-D34).

    Args:
        params: Physical parameters; ``d_bar`` is the disturbance the tube is designed for.
        layout: The map; the obstacles are its blocked polygons (D16).
        settings: Horizon, weights, margin and tolerances.
        segment_steps: ``H``, the steps of a segment; required when the settings
            leave the horizon to the problem, which is then ``H + n_b`` (D45).
            An explicit horizon is kept as it is; :meth:`check_segments` checks it.

    Raises:
        ValueError: If the closed loop is not rotation-invariant, the
            disturbance leaves no speed, thrust or room in the arena, the
            margin ``rho`` does not cover :attr:`face_slack`, or the horizon is
            left to the problem without a positive ``segment_steps``.
    """

    def __init__(
        self,
        params: PhysicsParams,
        layout: Layout,
        settings: MPCSettings | None = None,
        segment_steps: int | None = None,
    ) -> None:
        self.params = params
        self.layout = layout
        self.settings = settings if settings is not None else MPCSettings()
        if segment_steps is not None and (
            isinstance(segment_steps, bool) or not isinstance(segment_steps, int) or segment_steps < 1
        ):
            raise ValueError(f"segment_steps must be a positive integer, got {segment_steps!r}")
        if self.settings.horizon is None and segment_steps is None:
            raise ValueError("an automatic horizon (horizon None, D45) needs the length of the segments, segment_steps")
        self.dt = params.dt
        self.A, self.B = system_matrices(params.dt, params.gamma)
        weights = self.settings.weights
        self.Q, self.R = weights.Q, weights.R
        self.K, _ = lqr(self.A, self.B, self.Q, self.R)  # the ancillary gain (D31); the LQR's cost is not used (D37)
        self.tube: Tube = design_tube(self.A, self.B, self.K, params.d_bar, tolerance=self.settings.tail_tolerance)
        self._check_rotation_symmetry()

        self.v_bar = params.v_max - self.tube.next_speed_radius
        self.u_bar = params.a_max - self.tube.input_radius
        if not self.v_bar > 0.0:
            raise ValueError(f"the disturbance leaves no planned speed: v_bar = {self.v_bar}")
        if not self.u_bar > 0.0:
            raise ValueError(f"the disturbance leaves no planned thrust: u_bar = {self.u_bar}")
        if self.settings.horizon is not None:
            self.N: int = self.settings.horizon
        else:
            assert segment_steps is not None  # checked above
            self.N = segment_steps + self.stopping_steps()  # the smallest horizon of check_segments (D45)
        self.regularized = (not self.tube.is_zero) and self.settings.start_weight > 0.0
        """Whether the cost has the term ``eps ||x - z_0||^2`` of D46: with a tube and ``eps > 0``."""
        self.margin = self.tube.position_radius + self.settings.rho

        arena = layout.arena
        self.arena_normals = arena.normals.copy()
        self.arena_bounds = arena.offsets - self.margin
        tight = tightened_arena(layout, self.margin)
        self.tight_arena: ConvexPolygon = tight
        self.position_low = tight.vertices.min(axis=0)
        self.position_high = tight.vertices.max(axis=0)
        self.obstacles = layout_obstacles(layout, self.margin, tight)
        self.face_slack = face_slack(self.obstacles)
        """How far a mixed-integer plan may violate a face: a binary within Gurobi's integrality
        tolerance of 1 relaxes its face by ``M`` times it, plus the feasibility tolerance [m]."""
        if not self.settings.rho > self.face_slack:
            raise ValueError(
                f"the margin rho = {self.settings.rho} m must exceed what a mixed-integer plan may violate "
                f"a face by, M IntFeasTol + FeasibilityTol = {self.face_slack:.2e} m"
            )

    # ------------------------------------------------------------------ setup
    def _check_rotation_symmetry(self) -> None:
        """``A_K`` and ``K`` commute with the rotations of the plane, so the tightened sets are disks."""
        angles = (0.3, 1.1, 2.5)
        gain_commutes = all(np.allclose(self.K @ rotation(t), rotation(t)[:2, :2] @ self.K, atol=1e-10) for t in angles)
        if not (is_rotation_invariant(self.tube.closed_loop) and gain_commutes):
            raise ValueError(
                "the closed loop is not rotation-invariant: the plant and weights must treat both axes alike"
            )

    # ---------------------------------------------------------------- helpers
    def waypoint_stage(self, steps_left: int) -> int:
        """The stage ``k*`` of the plan at which the segment ends: ``steps_left``, checked to lie in ``1, ..., N``.

        Raises:
            ValueError: If ``steps_left`` is not an integer between 1 and ``N``.
        """
        if isinstance(steps_left, (bool, np.bool_)) or not isinstance(steps_left, (int, np.integer)):
            raise ValueError(f"steps_left must be an integer, got {steps_left!r}")
        if not 1 <= steps_left <= self.N:
            raise ValueError(f"steps_left must lie between 1 and the horizon N = {self.N}, got {steps_left}")
        return int(steps_left)

    def cost(
        self, z: ArrayLike, v: ArrayLike, target: ArrayLike, steps_left: int, x: ArrayLike | None = None
    ) -> float:
        """The cost of D37 and D46 of a plan ``(z, v)``, shapes ``(N+1, 4)`` and ``(N, 2)``, at the state ``x``.

        ``q_g ||z_{k*,p} - g||^2 + r_e sum_{k<k*} ||v_k||^2``, with ``k* = steps_left``,
        plus ``eps ||x - z_0||^2`` when :attr:`regularized`; ``x`` is ignored otherwise.

        Raises:
            ValueError: If ``steps_left`` does not lie in ``1, ..., N``, or the
                cost is regularized and ``x`` is missing.
        """
        k = self.waypoint_stage(steps_left)
        z_arr = np.asarray(z, dtype=np.float64)
        error = z_arr[k, :2] - np.asarray(target, dtype=np.float64).reshape(2)
        effort = np.asarray(v, dtype=np.float64)[:k]
        value = self.settings.target_weight * error @ error + self.settings.effort_weight * np.sum(effort**2)
        if self.regularized:
            if x is None:
                raise ValueError("the regularized cost (D46) needs the measured state x")
            start = np.asarray(x, dtype=np.float64).reshape(4) - z_arr[0]
            value += self.settings.start_weight * start @ start
        return float(value)

    def stage_speed_bounds(self) -> FloatArray:
        """``(N,)`` bounds on the nominal speed at stages ``1, ..., N`` that every plan meets.

        ``min(v_bar, w_(N-k))`` at stage ``k``, where ``w_n`` is the largest
        speed from which the nominal plan can rest after ``n`` steps: each step
        lowers the speed by at most ``(1 - beta) s + dt u_bar`` with
        ``beta = 1 - gamma dt``, so ``s_(k+n) >= beta^n s_k - dt u_bar (1 - beta^n) / (1 - beta)``
        and the terminal set requires ``s_N = 0``. Full braking along the
        velocity attains the bound, which is therefore exact.
        """
        beta = 1.0 - self.params.gamma * self.dt
        n = self.N - np.arange(1, self.N + 1)  # steps left to stop, from stage k = 1, ..., N
        stoppable = self.dt * self.u_bar * (1.0 - beta**n) / ((1.0 - beta) * beta**n)
        result: FloatArray = np.minimum(self.v_bar, stoppable)
        return result

    def reach_boxes(self, x: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Bounds on the nominal positions of every plan from the measured state ``x``.

        ``z_0`` lies in ``x (+) (-Z)``, so its position within ``r_p`` of ``x``'s.
        Afterwards ``p_{k+1} = p_k + dt v_{k+1}`` with the speed of stage ``k + 1``
        at most its bound of :meth:`stage_speed_bounds`, so stage ``k`` lies
        within ``r_p + dt (s_1 + ... + s_k)`` of ``x``'s position on each axis.
        The boxes are also clipped to the tightened arena's bounding box.

        Returns:
            ``(lo, hi)``, each ``(N + 1, 2)``.
        """
        p = np.asarray(x, dtype=np.float64)[:2]
        travel = np.concatenate([[0.0], np.cumsum(self.dt * self.stage_speed_bounds())])[:, None]
        reach = self.tube.position_radius + travel
        lo = np.maximum(p - reach, self.position_low)
        hi = np.minimum(p + reach, self.position_high)
        return lo, np.maximum(hi, lo)

    def fixed_binaries(self, x: ArrayLike) -> Bounds:
        """The binaries' bounds for a solve from ``x``: faces that every plan satisfies are fixed.

        A face is satisfied on segment ``k`` by every plan when the whole box of
        stage ``k + 1``, which contains the box of stage ``k``, lies beyond it.
        Its binary is then fixed to 1 and the obstacle's other faces on that
        segment to 0: inactive constraints, not wrong ones, so no feasible plan
        is lost (remark 3.12 of the note).
        """
        lo, hi = self.reach_boxes(x)
        centre, half = 0.5 * (lo + hi)[1:], 0.5 * (hi - lo)[1:]  # the boxes of stages 1..N, one per segment
        result: Bounds = []
        for obstacle in self.obstacles:
            lowest = centre @ obstacle.normals.T - half @ np.abs(obstacle.normals).T  # (N, m)
            certified = lowest >= obstacle.bounds
            lb = np.zeros((self.N, obstacle.num_faces))
            ub = np.ones((self.N, obstacle.num_faces))
            rows = np.flatnonzero(certified.any(axis=1))
            first = certified[rows].argmax(axis=1)
            ub[rows] = 0.0
            lb[rows, first] = 1.0
            ub[rows, first] = 1.0
            result.append((lb, ub))
        return result

    @staticmethod
    def free_binaries(bounds: Bounds) -> int:
        """How many binaries a solve leaves free."""
        return int(sum(np.sum(lb != ub) for lb, ub in bounds))

    def candidate(self, z: ArrayLike, v: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """The shifted plan: drop the first step and hold the final rest state with ``v = 0``."""
        z_arr = np.asarray(z, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        return np.vstack([z_arr[1:], z_arr[-1:]]), np.vstack([v_arr[1:], np.zeros((1, 2))])

    def binaries_of(self, z: ArrayLike, bounds: Bounds | None = None, tol: float = PLAN_TOL) -> Binaries | None:
        """The binaries of a plan: 1 on every face that both ends of a segment satisfy (to ``tol``).

        Fixed binaries of ``bounds`` keep their value. ``None`` if some segment
        lies beyond no face of some obstacle.
        """
        p = np.asarray(z, dtype=np.float64)[:, :2]
        result: Binaries = []
        for i, obstacle in enumerate(self.obstacles):
            beyond = (p @ obstacle.normals.T) >= obstacle.bounds - tol  # (N+1, m)
            deltas = (beyond[:-1] & beyond[1:]).astype(np.float64)
            if bounds is not None:
                lb, ub = bounds[i]
                fixed = lb == ub
                deltas[fixed] = lb[fixed]
            if not np.all(deltas.sum(axis=1) >= 1.0):
                return None
            result.append(deltas)
        return result

    def input(self, x: ArrayLike, z: ArrayLike, v: ArrayLike) -> FloatArray:
        """The applied input ``u = v_0 + K (x - z_0)``, projected onto ``U`` to absorb solver tolerances."""
        error = np.asarray(x, dtype=np.float64) - np.asarray(z, dtype=np.float64)[0]
        u = np.asarray(v, dtype=np.float64)[0] + self.K @ error
        return project_disk(u, self.params.a_max)

    def emergency_input(self, x: ArrayLike) -> FloatArray:
        """The brake outside every guarantee: the input that stops the agent in one step, projected onto ``U``."""
        velocity = np.asarray(x, dtype=np.float64)[2:]
        return project_disk(-(1.0 - self.params.gamma * self.dt) * velocity / self.dt, self.params.a_max)

    def _plan(self, z: ArrayLike, v: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """A plan as arrays, checked to have the shapes ``(N+1, 4)`` and ``(N, 2)``.

        Raises:
            ValueError: If it has other shapes.
        """
        z_arr = np.asarray(z, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        if z_arr.shape != (self.N + 1, 4) or v_arr.shape != (self.N, 2):
            raise ValueError(
                f"a plan has shapes ({self.N + 1}, 4) and ({self.N}, 2), got {z_arr.shape} and {v_arr.shape}"
            )
        return z_arr, v_arr

    def violations(self, x: ArrayLike, z: ArrayLike, v: ArrayLike) -> dict[str, float]:
        """How far a plan violates each group of constraints (0 when it satisfies them).

        The tube constraint ``x - z_0 in Z`` is not checked here, except that
        without a disturbance it reads ``z_0 = x``.

        Returns:
            The largest violation of: ``dynamics``, ``initial`` (only for
            ``Z = {0}``), ``terminal``, ``speed``, ``input``, ``arena`` and
            ``obstacles`` (per segment, the smallest violation over the faces
            of the larger one of its two ends).
        """
        x_arr = np.asarray(x, dtype=np.float64)
        z_arr, v_arr = self._plan(z, v)
        p = z_arr[:, :2]
        result = {
            "dynamics": float(np.abs(z_arr[1:] - z_arr[:-1] @ self.A.T - v_arr @ self.B.T).max()),
            "initial": float(np.abs(z_arr[0] - x_arr).max()) if self.tube.is_zero else 0.0,
            "terminal": float(np.abs(z_arr[-1, 2:]).max()),
            "speed": float(max(0.0, np.linalg.norm(z_arr[1:-1, 2:], axis=1).max(initial=0.0) - self.v_bar)),
            "input": float(max(0.0, np.linalg.norm(v_arr, axis=1).max() - self.u_bar)),
            "arena": float(max(0.0, (p @ self.arena_normals.T - self.arena_bounds).max())),
        }
        worst = 0.0
        for obstacle in self.obstacles:
            short = obstacle.bounds - p @ obstacle.normals.T  # (N+1, m): > 0 where a stage is short of a face
            per_face = np.maximum(short[:-1], short[1:])  # the worse end of each segment
            worst = max(worst, float(np.maximum(per_face.min(axis=1), 0.0).max()))
        result["obstacles"] = worst
        return result

    def plan_within(self, x: ArrayLike, z: ArrayLike, v: ArrayLike, tol: float = FEAS_TOL) -> bool:
        """Whether a plan meets every constraint but the tube's within the tolerances accepted for the relaxation's
        own plan (R2 of D49; module docstring of :mod:`hrlmpc.mpc_worker`).

        The dynamics, the initial state (without a tube), the terminal rest and
        the arena to ``tol``, Gurobi's feasibility tolerance; the speed and
        thrust disks to ``tol (1 + radius)``, what the cuts of the relaxation
        leave; every segment beyond some face of each obstacle to
        :attr:`face_slack`, which the margin ``rho`` covers. The tube
        constraint ``x - z_0 in Z`` needs the gauge of ``Z``, a solver: the
        Worker checks it.

        Raises:
            ValueError: If the plan's shapes are not ``(N+1, 4)`` and ``(N, 2)``.
        """
        z_arr, v_arr = self._plan(z, v)
        if not (np.all(np.isfinite(z_arr)) and np.all(np.isfinite(v_arr))):
            return False
        worst = self.violations(x, z_arr, v_arr)
        bounds = {
            "dynamics": tol,
            "initial": tol,
            "terminal": tol,
            "speed": tol * (1.0 + self.v_bar),
            "input": tol * (1.0 + self.u_bar),
            "arena": tol,
            "obstacles": self.face_slack,
        }
        return all(worst[key] <= bound for key, bound in bounds.items())

    def summary(self) -> str:
        """A one-paragraph description of the tube, the tightened sets and the cost."""
        t, s = self.tube, self.settings
        free = sum(o.num_faces for o in self.obstacles) * self.N
        return (
            f"tube MPC: d_bar {t.d_bar}, s = {t.num_terms} terms, tail {t.tail_radius:.1e}; "
            f"Z radii: position {t.position_radius:.4f} m, speed {t.speed_radius:.4f} m/s, "
            f"input {t.input_radius:.4f} m/s^2\n"
            f"  planned speed <= {self.v_bar:.3f} m/s (D33), planned thrust <= {self.u_bar:.3f} m/s^2, "
            f"position margin {self.margin:.4f} m; K = {np.round(self.K[0, [0, 2]], 3)} per axis\n"
            f"  {len(self.obstacles)} obstacles, {free} binaries before fixing; horizon N = {self.N}"
            f"{' (automatic, D45)' if s.horizon is None else ''}, stopping time {self.stopping_steps()} steps; "
            f"waypoint cost (D37): q_g = {s.target_weight:g}, r_e = {s.effort_weight:g}"
            f"{f', eps = {s.start_weight:g} (D46)' if self.regularized else ''}"
        )

    def stopping_steps(self) -> int:
        """Steps the nominal plan needs to stop from ``v_bar`` with ``u_bar`` (friction helps)."""
        speed, steps = self.v_bar, 0
        factor = 1.0 - self.params.gamma * self.dt
        while speed > 0.0:
            speed = factor * speed - self.dt * self.u_bar
            steps += 1
        return steps

    def check_segments(self, segment_steps: int) -> None:
        """Refuse a horizon too short for the waypoints of segments of ``H = segment_steps`` steps (D37).

        At a decision the waypoint lies at stage ``H``; a plan that passes it at
        the planned speed ``v_bar`` still needs :meth:`stopping_steps` steps to
        rest, so ``N >= H + n_b``.

        Raises:
            ValueError: If ``N < H + n_b``.
        """
        needed = segment_steps + self.stopping_steps()
        if self.N < needed:
            raise ValueError(
                f"the horizon N = {self.N} leaves no room to stop after a segment of {segment_steps} steps: "
                f"a plan needs {self.stopping_steps()} steps to stop from {self.v_bar:.3f} m/s, so N >= {needed} (D37)"
            )


def project_disk(u: ArrayLike, radius: float) -> FloatArray:
    """``u`` scaled radially onto ``||u|| <= radius`` where it lies outside."""
    u_arr = np.asarray(u, dtype=np.float64)
    norm = float(np.linalg.norm(u_arr))
    return u_arr if norm <= radius else u_arr * (radius / norm)
