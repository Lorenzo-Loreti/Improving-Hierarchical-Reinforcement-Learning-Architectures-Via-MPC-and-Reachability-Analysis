"""Shortest paths in the free space, for the minimum-time bound F7 (decision log D16, D20).

The length of the shortest path in ``P_free`` from a point to the goal region
bounds from below the distance that any trajectory from that point must cover
(F7, :mod:`hrlmpc.evaluation`). Among polygonal obstacles a shortest path is a
taut string: it bends only at obstacle vertices, and its last segment meets
the goal line ``p_x = goal_x`` at a right angle. It is found by Dijkstra's
algorithm on the visibility graph of the start and of the vertices of the
blocked polygons that lie inside the arena (the visibility-graph method of
Lozano-Perez and Wesley, "An algorithm for planning collision-free paths among
polyhedral obstacles", Communications of the ACM 22(10), 1979).

A segment is free when it does not enter the interior of any blocked polygon;
it may touch a vertex or run along a face, as the agent may. The arena must be
an axis-aligned rectangle (every layout built from a configuration is), so the
last segment of a path never leaves it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrlmpc.geometry import TOL, Layout

FloatArray = NDArray[np.float64]


def segment_is_free(layout: Layout, start: ArrayLike, end: ArrayLike, tol: float = TOL) -> bool:
    """Whether the segment from ``start`` to ``end`` avoids the interior of every blocked polygon.

    The polygons are shrunk by ``tol`` for the test, so a segment that touches
    a vertex or runs along a face is free. Both end points are assumed to lie
    in the arena, which is convex, so the segment does too.
    """
    a = np.asarray(start, dtype=np.float64)
    d = np.asarray(end, dtype=np.float64) - a
    for polygon in layout.blocked:
        # a + t d lies inside the shrunk polygon iff n_k . (a + t d) < c_k - tol for every face k.
        room = polygon.offsets - tol - polygon.normals @ a
        rate = polygon.normals @ d
        lo, hi = 0.0, 1.0
        for r, s in zip(room, rate, strict=True):
            if s > 0.0:
                hi = min(hi, r / s)
            elif s < 0.0:
                lo = max(lo, r / s)
            elif r <= 0.0:
                lo, hi = 1.0, 0.0  # parallel to the face, outside its half-plane
            if lo >= hi:
                break
        if lo < hi:
            return False
    return True


def _check_arena(layout: Layout) -> None:
    normals = np.abs(layout.arena.normals)
    if not np.all(np.isclose(normals.max(axis=1), 1.0, rtol=0.0, atol=1e-12)):
        raise ValueError("shortest paths to the goal line need an axis-aligned rectangular arena")


def _goal_leg(layout: Layout, point: FloatArray) -> float:
    """Length of the horizontal segment from ``point`` to the goal line, or ``inf`` if it is blocked."""
    if point[0] >= layout.goal_x:
        return 0.0
    end = np.array([layout.goal_x, point[1]])
    return layout.goal_x - float(point[0]) if segment_is_free(layout, point, end) else float("inf")


def _vertices(layout: Layout) -> FloatArray:
    """Candidate bends: vertices of the blocked polygons inside the arena, free and before the goal line."""
    if not layout.blocked:
        return np.zeros((0, 2))
    points = np.concatenate([polygon.vertices for polygon in layout.blocked])
    keep = layout.arena.contains_interior(points) & layout.in_free_space(points) & (points[:, 0] < layout.goal_x)
    result: FloatArray = np.unique(points[keep], axis=0)
    return result


def _vertex_distances(layout: Layout, nodes: FloatArray) -> FloatArray:
    """Shortest free-path length from every vertex to the goal region (Dijkstra, all goal legs as sources)."""
    n = len(nodes)
    distance = np.array([_goal_leg(layout, node) for node in nodes])
    edges = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            if segment_is_free(layout, nodes[i], nodes[j]):
                edges[i, j] = edges[j, i] = float(np.hypot(*(nodes[j] - nodes[i])))
    done = np.zeros(n, dtype=bool)
    for _ in range(n):
        pending = np.where(done, np.inf, distance)
        i = int(np.argmin(pending))
        if not np.isfinite(pending[i]):
            break
        done[i] = True
        distance = np.minimum(distance, distance[i] + edges[i])
    return distance


def free_path_distance(layout: Layout, points: ArrayLike) -> FloatArray:
    """Length of the shortest path in ``P_free`` from each point to the goal region.

    In an arena without obstacles it is ``goal_x - p_x``.

    Args:
        layout: The layout.
        points: ``(N, 2)`` points in the free space.

    Returns:
        ``(N,)`` lengths.

    Raises:
        ValueError: If a point is not free, the goal region cannot be reached
            from it, or the arena is not an axis-aligned rectangle.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {pts.shape}")
    _check_arena(layout)
    if not layout.in_free_space(pts).all():
        raise ValueError("every point must lie in the free space")
    nodes = _vertices(layout)
    through = _vertex_distances(layout, nodes)
    result = np.empty(len(pts))
    for k, point in enumerate(pts):
        best = _goal_leg(layout, point)
        for node, rest in zip(nodes, through, strict=True):
            if np.isfinite(rest) and segment_is_free(layout, point, node):
                best = min(best, float(np.hypot(*(node - point))) + rest)
        if not np.isfinite(best):
            raise ValueError(f"the goal region cannot be reached from {point.tolist()}")
        result[k] = best
    return result
