"""Targets of PPO_MPC_reachability: the reachable set and the Manager's action map (decision log D38-D40).

At a decision the Manager's action ``x`` in ``[-1, 1]^2`` sets a target in the
set ``R_H`` of the positions at stage ``H`` of the plans that the MPC Worker
may choose from the current state (D38): robust with respect to ``W`` in the
sense of the tube, contact-free, and able to stop by the end of the horizon.

1. Support points of the part of ``R_H`` without obstacles, in ``n`` fixed
   directions of the plane: one cone program each over the Worker's problem
   without obstacles (:class:`hrlmpc.mpc_worker.ReachSupport`). That part is
   convex, so the convex hull ``C`` of its support points is an inner
   approximation; between two directions it loses at most the fraction
   ``1 - cos(pi / n)`` of a disk's radius (F5).
2. ``q = c + ||m|| rho_C(m / ||m||) m / ||m||``, with ``m = m(x)`` the bijection
   of D15 onto the unit disk, ``c`` the centroid of ``C`` and ``rho_C(u)`` the
   distance from ``c`` to the boundary of ``C`` along ``u``: a bijection of the
   unit disk onto ``C``, radial about an interior point (:func:`radial_map`).
3. The target is the position at stage ``H`` of the Worker's plan for ``q``,
   solved as by a fresh Worker (:meth:`hrlmpc.mpc_worker.MPCWorker.project`):
   the point of ``R_H``, obstacles included, nearest to ``q`` up to the
   trade-off with the effort (D37: millimetres at the edge of the set), and
   reachable by construction (D39). If no plan is found the target is ``q``,
   and the failure is counted.

The Worker of architecture 3 treats an unreachable target in the same way
(D37), so architectures 3 and 4 differ only in the map of the Manager's
action: onto the disk of D24, or onto ``C``.

Everything here is NumPy; the solvers enter as callables.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrlmpc.action_map import square_to_disk
from hrlmpc.env import ACTION_TOL

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

HULL_TOL = 1e-9
"""Points closer than this [m], or turns flatter than this [m^2], are merged on the hull."""

AREA_TOL = 1e-6
"""A hull of smaller area [m^2] is degenerate: the target is then its points' mean, before the projection.
Reachable sets are four orders of magnitude larger; this happens only near an infeasible state."""

MOVED = 1e-2
"""A projection that moves the point of the hull by more than this [m] counts as moved by an obstacle or the
arena: the trade-off with the effort alone moves points on the edge of the set by millimetres (D37)."""

REACHED_MARGIN = 0.1
"""A target counts as reached when the segment ends within ``r_p + REACHED_MARGIN`` [m] of it (D40)."""

SupportFn = Callable[[FloatArray], FloatArray]
"""``(4,)`` state to ``(n, 2)`` support points in the fixed directions, a NaN row where a solve failed."""

ProjectFn = Callable[[FloatArray, FloatArray], tuple[FloatArray | None, int]]
"""``(4,)`` state and ``(2,)`` point to the reachable target nearest to it, or ``None`` if no plan was found, and
the solver failures the search survived or ended with."""

REACH_KEYS = (
    "support_ms",
    "project_ms",
    "hull_area",
    "projection",
    "support_failures",
    "hull_fallback",
    "projection_failed",
    "projection_errors",
)
"""Statistics of every decision: the solve times [ms]; the hull's area [m^2]; the distance [m] by which the
projection moved the point of the hull; the directions whose solve failed; whether the hull was degenerate,
so that the point was the support points' mean or the position; whether the projection found no plan; and the
solver failures during the projection."""


def support_directions(count: int) -> FloatArray:
    """``(count, 2)`` unit vectors at the angles ``2 pi i / count``, fixed in the plane.

    Raises:
        ValueError: If ``count`` is below 3.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 3:
        raise ValueError(f"at least 3 directions are needed, got {count!r}")
    angles = 2.0 * np.pi * np.arange(count) / count
    result: FloatArray = np.column_stack([np.cos(angles), np.sin(angles)])
    return result


def convex_hull(points: ArrayLike, tol: float = HULL_TOL) -> FloatArray:
    """The vertices of the convex hull, counter-clockwise from the lowest-leftmost, without repeats or flat turns.

    Andrew's monotone chain; fewer than three vertices for points on a line.

    Raises:
        ValueError: If ``points`` is not ``(n, 2)`` or not finite.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (n, 2), got {pts.shape}")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must be finite")
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    unique: list[FloatArray] = []
    for p in pts[order]:
        if not unique or float(np.max(np.abs(p - unique[-1]))) > tol:
            unique.append(p)
    if len(unique) < 3:
        return np.array(unique).reshape(-1, 2)

    def turn(o: FloatArray, a: FloatArray, b: FloatArray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    def chain(sequence: list[FloatArray]) -> list[FloatArray]:
        out: list[FloatArray] = []
        for p in sequence:
            while len(out) >= 2 and turn(out[-2], out[-1], p) <= tol:
                out.pop()
            out.append(p)
        return out

    lower, upper = chain(unique), chain(unique[::-1])
    return np.array(lower[:-1] + upper[:-1])


def polygon_area(vertices: ArrayLike) -> float:
    """The area of a polygon given counter-clockwise (shoelace formula); 0 below three vertices."""
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    if len(v) < 3:
        return 0.0
    x, y = v[:, 0] - v[0, 0], v[:, 1] - v[0, 1]  # relative to a vertex, against rounding far from the origin
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_centroid(vertices: ArrayLike) -> FloatArray:
    """The centroid of a polygon given counter-clockwise, interior to it when it is convex.

    Raises:
        ValueError: If the polygon has no area.
    """
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    area = polygon_area(v)
    if not area > 0.0:
        raise ValueError("a polygon without area has no centroid")
    origin = v[0].copy()  # relative coordinates, so that rounding does not grow with the distance from the origin
    x, y = v[:, 0] - origin[0], v[:, 1] - origin[1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    cx = float(np.sum((x + np.roll(x, -1)) * cross)) / (6.0 * area)
    cy = float(np.sum((y + np.roll(y, -1)) * cross)) / (6.0 * area)
    result: FloatArray = origin + np.array([cx, cy])
    return result


def radial_extent(vertices: ArrayLike, center: ArrayLike, direction: ArrayLike) -> float:
    """The distance from ``center``, inside the convex polygon, to its boundary along the unit ``direction``.

    Raises:
        ValueError: If the polygon has fewer than three vertices or ``center`` lies outside it.
    """
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    c = np.asarray(center, dtype=np.float64).reshape(2)
    u = np.asarray(direction, dtype=np.float64).reshape(2)
    if len(v) < 3:
        raise ValueError("a polygon needs three vertices")
    edges = np.roll(v, -1, axis=0) - v
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])  # outward for counter-clockwise vertices
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    slack = np.sum(normals * (v - c), axis=1)  # distance from c to each edge's line
    if np.any(slack < -HULL_TOL):
        raise ValueError("the centre lies outside the polygon")
    rate = normals @ u
    ahead = rate > 0.0
    return float(np.min(np.maximum(slack[ahead], 0.0) / rate[ahead]))


def radial_map(m: ArrayLike, vertices: ArrayLike, center: ArrayLike) -> FloatArray:
    """``c + ||m|| rho(m / ||m||) m / ||m||``: the point of the unit disk ``m`` mapped onto the convex polygon.

    A bijection of the unit disk onto the polygon that keeps rays from ``c``:
    ``m = 0`` goes to ``c`` and the unit circle onto the boundary.
    """
    m_arr = np.asarray(m, dtype=np.float64).reshape(2)
    c = np.asarray(center, dtype=np.float64).reshape(2)
    r = float(np.hypot(m_arr[0], m_arr[1]))
    if r == 0.0:
        return c.copy()
    u = m_arr / r
    result: FloatArray = c + r * radial_extent(vertices, c, u) * u
    return result


@dataclass(eq=False)
class ReachableTargets:
    """The target map of architecture 4 (D38, D39), with the :class:`~hrlmpc.hierarchy.TargetMap` interface.

    Attributes:
        support: The support points of the part of the reachable set without obstacles.
        project: The reachable target nearest to a point.
        radius: Length scale [m] of the target offsets, for the interface; ``v_max H T_s``.
        release: Releases the solvers behind ``support`` and ``project``; called by :meth:`close`.
    """

    support: SupportFn
    project: ProjectFn
    radius: float
    release: Callable[[], None] | None = None
    _stats: dict[str, list[float]] = field(default_factory=lambda: {key: [] for key in REACH_KEYS}, repr=False)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.radius) and self.radius > 0.0):
            raise ValueError(f"the radius must be positive and finite, got {self.radius}")

    def point(self, action: ArrayLike, state: ArrayLike) -> tuple[FloatArray, float, int, bool]:
        """The point ``q`` of the hull for one action at one state, the hull's area, the failed directions, and
        whether the hull was degenerate (then ``q`` is the support points' mean, or the position without any)."""
        x = np.asarray(state, dtype=np.float64).reshape(4)
        points = np.asarray(self.support(x), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"the support points must have shape (n, 2), got {points.shape}")
        valid = np.all(np.isfinite(points), axis=1)
        failures = int(np.sum(~valid))
        if not valid.any():
            return x[:2].copy(), 0.0, failures, True
        hull = convex_hull(points[valid])
        area = polygon_area(hull)
        if len(hull) >= 3 and area > AREA_TOL:
            m = square_to_disk(np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0), 1.0)
            try:
                return radial_map(m, hull, polygon_centroid(hull)), area, failures, False
            except ValueError:  # a sliver whose centroid rounding put outside it
                pass
        return np.mean(points[valid], axis=0), area, failures, True

    def __call__(self, actions: ArrayLike, positions: ArrayLike, velocities: ArrayLike) -> FloatArray:
        """The targets of ``(B, 2)`` actions in the square at ``(B, 2)`` positions and velocities.

        Raises:
            ValueError: If the shapes differ or are not ``(B, 2)``, a value is
                not finite, or an action lies outside the square.
        """
        x = np.asarray(actions, dtype=np.float64)
        p = np.asarray(positions, dtype=np.float64)
        v = np.asarray(velocities, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 2 or p.shape != x.shape or v.shape != x.shape:
            raise ValueError(f"actions, positions and velocities must share one shape (B, 2), got "
                             f"{x.shape}, {p.shape}, {v.shape}")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(p)) and np.all(np.isfinite(v))):
            raise ValueError("actions, positions and velocities must be finite")
        if np.any(np.abs(x) > 1.0 + ACTION_TOL):
            raise ValueError("the actions must lie within the square [-1, 1]^2")
        targets = np.empty_like(p)
        for i in range(len(x)):
            state = np.concatenate([p[i], v[i]])
            started = time.perf_counter()
            q, area, failures, fallback = self.point(x[i], state)
            between = time.perf_counter()
            projected, errors = self.project(state, q)
            ended = time.perf_counter()
            failed = projected is None
            targets[i] = q if projected is None else np.asarray(projected, dtype=np.float64).reshape(2)
            row = {
                "support_ms": 1e3 * (between - started),
                "project_ms": 1e3 * (ended - between),
                "hull_area": area,
                "projection": float(np.hypot(*(targets[i] - q))),
                "support_failures": float(failures),
                "hull_fallback": float(fallback),
                "projection_failed": float(failed),
                "projection_errors": float(errors),
            }
            for key in REACH_KEYS:
                self._stats[key].append(row[key])
        return targets

    def pop_stats(self) -> dict[str, FloatArray]:
        """The statistics of the decisions since the last call, each ``(decisions,)`` in call order, and clear them."""
        result = {key: np.array(values, dtype=np.float64) for key, values in self._stats.items()}
        self._stats = {key: [] for key in REACH_KEYS}
        return result

    def close(self) -> None:
        """Release the solvers, once; later calls do nothing."""
        release, self.release = self.release, None
        if release is not None:
            release()


def reached_metrics(distances: ArrayLike, tolerance: float, prefix: str) -> dict[str, float]:
    """How close the agents came to their targets at the end of full segments (D40), under ``<prefix>/``.

    ``reached_rate``: the fraction of segments that ended within ``tolerance``
    [m] of the target, ``r_p + REACHED_MARGIN``; ``end_distance_mean`` and
    ``end_distance_max`` [m]; ``full_segments``. Nothing without a segment.
    """
    d = np.asarray(distances, dtype=np.float64).reshape(-1)
    if d.size == 0:
        return {}
    return {
        f"{prefix}/reached_rate": float(np.mean(d <= tolerance)),
        f"{prefix}/end_distance_mean": float(d.mean()),
        f"{prefix}/end_distance_max": float(d.max()),
        f"{prefix}/full_segments": float(d.size),
    }


def reach_metrics(stats: dict[str, FloatArray], prefix: str, mask: ArrayLike | None = None) -> dict[str, float]:
    """Summaries of :meth:`ReachableTargets.pop_stats` under ``<prefix>/``.

    Times per decision, the hull's mean area, how far the projection moved
    the point of the hull (mean, largest, and the fraction of decisions moved by
    more than :data:`MOVED`, i.e. by an obstacle or the arena), and the counts
    of failed support directions, degenerate hulls, failed projections and
    solver failures during the projections.

    Args:
        stats: Arrays ``(decisions,)``.
        prefix: E.g. ``"reach"`` in training, ``"eval_reach"`` in an evaluation.
        mask: ``(decisions,)`` decisions to count; all by default.
    """
    count = len(stats["projection"])
    keep: BoolArray = np.ones(count, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).reshape(count)
    if not keep.any():
        return {}
    moved = stats["projection"][keep]
    return {
        f"{prefix}/support_ms_mean": float(stats["support_ms"][keep].mean()),
        f"{prefix}/project_ms_mean": float(stats["project_ms"][keep].mean()),
        f"{prefix}/hull_area_mean": float(stats["hull_area"][keep].mean()),
        f"{prefix}/projection_mean": float(moved.mean()),
        f"{prefix}/projection_max": float(moved.max()),
        f"{prefix}/projected_rate": float(np.mean(moved > MOVED)),
        f"{prefix}/support_failure_count": float(stats["support_failures"][keep].sum()),
        f"{prefix}/hull_fallback_count": float(stats["hull_fallback"][keep].sum()),
        f"{prefix}/projection_failure_count": float(stats["projection_failed"][keep].sum()),
        f"{prefix}/projection_error_count": float(stats["projection_errors"][keep].sum()),
        f"{prefix}/decisions": float(keep.sum()),
    }
