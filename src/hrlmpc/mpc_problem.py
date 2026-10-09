"""Data of the MPC Worker's problem, without the solvers (decision log D31-D34).

The Worker of PPO_MPC (architecture 3) solves at every step the robust tube-MPC
problem of the tube-MPC note, specialised to this plant and its polygonal map:

    min  sum_{k<N} (z_k - x_r)' Q (z_k - x_r) + v_k' R v_k + (z_N - x_r)' P (z_N - x_r)
    s.t. z_{k+1} = A z_k + B v_k,                       k = 0, ..., N-1
         x - z_0 in Z                                   (the tube, D31)
         ||z_{k,v}|| <= v_bar,                           k = 1, ..., N-1   (D33)
         z_{N,v} = 0                                    (terminal set: at rest)
         ||v_k|| <= u_bar,                               k = 0, ..., N-1
         m_i' z_{k,p} <= b_i - r_p - rho,                every face i of the arena
         n_j' z_{k,p} >= c_j + r_p + rho - M_j (1 - delta_{k,j}),
         n_j' z_{k+1,p} >= c_j + r_p + rho - M_j (1 - delta_{k,j}),
         sum_j delta_{k,j} >= 1,                         every obstacle, k = 0, ..., N-1   (D32)

with ``x`` the measured state, ``x_r = (g, 0)`` the rest state at the target
``g`` and the input ``u = v_0 + K (x - z_0)``. ``A``, ``B`` are the matrices of
:mod:`hrlmpc.model`; ``K`` and ``P`` the LQR gain and cost of ``Q``, ``R``;
``r_p`` the position radius of ``Z`` and ``rho`` a margin. The tightenings:

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

Every rest state is an equilibrium, so the shifted plan (drop the first step,
hold the last state at rest) is feasible at the next step whenever the
disturbance was in ``D``: theorem 4.1 of the note, under this terminal set.

This module holds what does not need a solver: the matrices, the tube, the
tightened sets, the obstacles and their big-M constants, the binaries that the
reachable positions fix in advance, the shifted candidate, the cost and a
checker of plans. :mod:`hrlmpc.mpc_worker` solves the problem.
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
    """Settings of the MPC Worker (D34; provisional until Q13).

    Attributes:
        horizon: ``N``, the prediction horizon, fixed and receding.
        q_pos: Weight of each position error.
        q_vel: Weight of each velocity (the target is at rest).
        r: Weight of each nominal input.
        rho: Safety margin [m] on the walls and the obstacles; it must exceed what
            a mixed-integer plan may violate a face by (checked by :class:`MPCProblem`).
        tail_tolerance: Bound on the tail of ``Z`` along the axes (D31) [m, m/s].
        mip_gap: Relative MIP gap of the mixed-integer solves, below 1.
        work_limit: Gurobi's deterministic work limit per solve, or ``None``.

    Raises:
        ValueError: If a setting is out of range.
    """

    horizon: int = 10
    q_pos: float = 10.0
    q_vel: float = 1.0
    r: float = 0.1
    rho: float = 1e-3
    tail_tolerance: float = TAIL_TOLERANCE
    mip_gap: float = 1e-4
    work_limit: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError(f"horizon must be a positive integer, got {self.horizon!r}")
        Weights(self.q_pos, self.q_vel, self.r)  # validates the weights
        for name in ("rho", "tail_tolerance", "mip_gap"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite, got {value!r}")
        if not self.mip_gap < 1.0:
            raise ValueError(f"mip_gap is a relative gap below 1, got {self.mip_gap!r}")
        if self.work_limit is not None and not (np.isfinite(self.work_limit) and self.work_limit > 0.0):
            raise ValueError(f"work_limit must be positive and finite or None, got {self.work_limit!r}")

    @property
    def weights(self) -> Weights:
        """The stage weights."""
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


class MPCProblem:
    """The offline data of the Worker's problem for one layout, plant and setting (D31-D34).

    Args:
        params: Physical parameters; ``d_bar`` is the disturbance the tube is designed for.
        layout: The map; the obstacles are its blocked polygons (D16).
        settings: Horizon, weights, margin and tolerances.

    Raises:
        ValueError: If the closed loop is not rotation-invariant, the
            disturbance leaves no speed, thrust or room in the arena, or the
            margin ``rho`` does not cover :attr:`face_slack`.
    """

    def __init__(self, params: PhysicsParams, layout: Layout, settings: MPCSettings | None = None) -> None:
        self.params = params
        self.layout = layout
        self.settings = settings if settings is not None else MPCSettings()
        self.N = self.settings.horizon
        self.dt = params.dt
        self.A, self.B = system_matrices(params.dt, params.gamma)
        weights = self.settings.weights
        self.Q, self.R = weights.Q, weights.R
        self.K, self.P = lqr(self.A, self.B, self.Q, self.R)
        self.tube: Tube = design_tube(self.A, self.B, self.K, params.d_bar, tolerance=self.settings.tail_tolerance)
        self._check_rotation_symmetry()

        self.v_bar = params.v_max - self.tube.next_speed_radius
        self.u_bar = params.a_max - self.tube.input_radius
        if not self.v_bar > 0.0:
            raise ValueError(f"the disturbance leaves no planned speed: v_bar = {self.v_bar}")
        if not self.u_bar > 0.0:
            raise ValueError(f"the disturbance leaves no planned thrust: u_bar = {self.u_bar}")
        self.margin = self.tube.position_radius + self.settings.rho

        arena = layout.arena
        self.arena_normals = arena.normals.copy()
        self.arena_bounds = arena.offsets - self.margin
        try:
            tight = arena.pushed_out(-self.margin * np.ones(len(arena.offsets)))
        except ValueError as error:
            raise ValueError(f"the arena leaves no room for a margin of {self.margin} m") from error
        self.tight_arena: ConvexPolygon = tight
        self.position_low = tight.vertices.min(axis=0)
        self.position_high = tight.vertices.max(axis=0)
        self.obstacles = [self._obstacle(polygon, arena, tight, i) for i, polygon in enumerate(layout.blocked)]
        largest_m = max((float(obstacle.big_m.max()) for obstacle in self.obstacles), default=0.0)
        self.face_slack = INT_FEAS_TOL * largest_m + FEAS_TOL
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

    def _obstacle(self, polygon: ConvexPolygon, arena: ConvexPolygon, tight: ConvexPolygon, index: int) -> Obstacle:
        bounds = polygon.offsets + self.margin
        live = (tight.vertices @ polygon.normals.T).max(axis=0) >= bounds
        if not live.any():
            raise ValueError(f"obstacle {index}, enlarged by the margin, leaves no room in the arena")
        normals, offsets, bounds = polygon.normals[live], polygon.offsets[live], bounds[live]
        lowest = (arena.vertices @ normals.T).min(axis=0)  # min over the arena of n_j' p
        big_m = np.maximum(0.0, bounds - lowest) + BIG_M_PAD
        return Obstacle(normals.copy(), offsets.copy(), bounds, big_m)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def rest_state(target: ArrayLike) -> FloatArray:
        """``x_r = (g, 0)``: at rest at the target ``g``."""
        g = np.asarray(target, dtype=np.float64).reshape(2)
        return np.array([g[0], g[1], 0.0, 0.0])

    def cost(self, z: ArrayLike, v: ArrayLike, x_r: ArrayLike) -> float:
        """The cost ``J_N`` of a plan ``(z, v)``, shapes ``(N+1, 4)`` and ``(N, 2)``."""
        dz = np.asarray(z, dtype=np.float64) - np.asarray(x_r, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        return float(
            np.einsum("ki,ij,kj->", dz[:-1], self.Q, dz[:-1])
            + np.einsum("ki,ij,kj->", v_arr, self.R, v_arr)
            + dz[-1] @ self.P @ dz[-1]
        )

    def reach_boxes(self, x: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Bounds on the nominal positions of every plan from the measured state ``x``.

        ``z_0`` lies in ``x (+) (-Z)``, so its position within ``r_p`` of ``x``'s.
        Afterwards ``p_{k+1} = p_k + dt v_{k+1}`` with ``||v_{k+1}|| <= v_bar``
        for ``k + 1 <= N - 1`` and ``v_N = 0``, so stage ``k`` lies within
        ``r_p + min(k, N - 1) dt v_bar`` of ``x``'s position on each axis. The
        boxes are also clipped to the tightened arena's bounding box.

        Returns:
            ``(lo, hi)``, each ``(N + 1, 2)``.
        """
        p = np.asarray(x, dtype=np.float64)[:2]
        steps = np.minimum(np.arange(self.N + 1), self.N - 1)[:, None]
        reach = self.tube.position_radius + steps * self.dt * self.v_bar
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
        z_arr = np.asarray(z, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        if z_arr.shape != (self.N + 1, 4) or v_arr.shape != (self.N, 2):
            raise ValueError(
                f"a plan has shapes ({self.N + 1}, 4) and ({self.N}, 2), got {z_arr.shape} and {v_arr.shape}"
            )
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

    def summary(self) -> str:
        """A one-paragraph description of the tube and the tightened sets."""
        t = self.tube
        free = sum(o.num_faces for o in self.obstacles) * self.N
        return (
            f"tube MPC: d_bar {t.d_bar}, s = {t.num_terms} terms, tail {t.tail_radius:.1e}; "
            f"Z radii: position {t.position_radius:.4f} m, speed {t.speed_radius:.4f} m/s, "
            f"input {t.input_radius:.4f} m/s^2\n"
            f"  planned speed <= {self.v_bar:.3f} m/s (D33), planned thrust <= {self.u_bar:.3f} m/s^2, "
            f"position margin {self.margin:.4f} m; K = {np.round(self.K[0, [0, 2]], 3)} per axis\n"
            f"  {len(self.obstacles)} obstacles, {free} binaries before fixing; horizon N = {self.N}, "
            f"stopping time {self.stopping_steps()} steps"
        )

    def stopping_steps(self) -> int:
        """Steps the nominal plan needs to stop from ``v_bar`` with ``u_bar`` (friction helps)."""
        speed, steps = self.v_bar, 0
        factor = 1.0 - self.params.gamma * self.dt
        while speed > 0.0:
            speed = factor * speed - self.dt * self.u_bar
            steps += 1
        return steps


def project_disk(u: ArrayLike, radius: float) -> FloatArray:
    """``u`` scaled radially onto ``||u|| <= radius`` where it lies outside."""
    u_arr = np.asarray(u, dtype=np.float64)
    norm = float(np.linalg.norm(u_arr))
    return u_arr if norm <= radius else u_arr * (radius / norm)
