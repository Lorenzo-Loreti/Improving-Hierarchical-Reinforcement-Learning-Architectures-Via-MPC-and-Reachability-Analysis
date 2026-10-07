"""Polygonal geometry of the navigation environment (decision log D11, D16).

The arena ``P`` and the obstacles ``O_i`` are convex polygons, stored both by
their vertices and in H-representation. The same layout serves the
environment, the MPC Worker, the oracle and the reachability analysis (D11).
A non-convex obstacle is a union of overlapping convex ones.

Free space (D16). The free positions form the closure of the interior of the
free region,

    P_free = cl( int P minus (O_1 u ... u O_m) ),

so the free space has no part of zero width. The set ``P minus U_i int O_i``
of the project instructions differs from it only where an obstacle touches a
wall or another obstacle without overlapping it: there it contains a segment
of zero width that a sliding agent could follow. To realize D16, every
obstacle that touches the arena boundary with a face is extended beyond that
wall (inside the arena nothing changes, see :attr:`Layout.blocked`), and
every other degenerate contact is rejected: two obstacles that touch without
overlapping, or an obstacle vertex on a wall.

Tolerances. Every geometric predicate uses ``TOL``: a point within ``TOL`` of
a face line counts as lying on it. A layout must not contain features between
``TOL`` and ``MIN_GAP``, i.e. gaps or obstacle widths so small that rounding
could decide whether the agent passes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrlmpc.config import Box, GeometryConfig

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

TOL = 1e-9
"""Tolerance of the geometric predicates [m]."""

MIN_GAP = 1e-6
"""Smallest gap between two walls, and smallest obstacle width, that a layout may have [m]."""

EXTENSION = 1.0
"""Distance [m] by which an obstacle touching a wall is pushed beyond it (D16), unless its shape allows less."""


def _frozen(array: ArrayLike, dtype: type = np.float64) -> NDArray:
    out = np.array(array, dtype=dtype)
    out.setflags(write=False)
    return out


def _points(points: ArrayLike) -> FloatArray:
    out = np.asarray(points, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 2:
        raise ValueError(f"points must be an (n, 2) array, got shape {out.shape}")
    return out


def dot(points: FloatArray, vectors: FloatArray) -> FloatArray:
    """``(n, k)`` dot products of ``(n, 2)`` points with ``(k, 2)`` vectors.

    Computed element by element, so the result for one point does not depend
    on the other points (a matrix product may round differently by batch size).
    """
    result: FloatArray = points[:, :1] * vectors[:, 0] + points[:, 1:] * vectors[:, 1]
    return result


@dataclass(frozen=True, eq=False)
class ConvexPolygon:
    """A strictly convex polygon.

    Attributes:
        vertices: ``(m, 2)`` vertices in counter-clockwise order. Edge ``k``
            runs from ``vertices[k]`` to ``vertices[k + 1]`` (cyclically).
        normals: ``(m, 2)`` outward unit normals of the edges.
        offsets: ``(m,)`` offsets; the polygon is ``{x : normals @ x <= offsets}``.
    """

    vertices: FloatArray
    normals: FloatArray
    offsets: FloatArray

    @classmethod
    def from_vertices(cls, vertices: ArrayLike) -> ConvexPolygon:
        """Build the polygon from its vertices, in either orientation.

        Raises:
            ValueError: If there are fewer than 3 vertices, a vertex is not
                finite or repeated, or the polygon is not strictly convex
                (collinear vertices, a reflex vertex, or more than one turn).
        """
        pts = np.array(vertices, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
            raise ValueError(f"a polygon needs at least 3 vertices as an (m, 2) array, got shape {pts.shape}")
        if not np.all(np.isfinite(pts)):
            raise ValueError("polygon vertices must be finite")
        nxt = np.roll(pts, -1, axis=0)
        if np.sum(pts[:, 0] * nxt[:, 1] - pts[:, 1] * nxt[:, 0]) < 0.0:  # clockwise
            pts = pts[::-1].copy()
        edges = np.roll(pts, -1, axis=0) - pts
        lengths = np.hypot(edges[:, 0], edges[:, 1])
        if np.any(lengths <= TOL):
            raise ValueError("polygon has repeated vertices")
        following = np.roll(edges, -1, axis=0)
        turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
        normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1) / lengths[:, None]
        offsets = np.sum(normals * pts, axis=1)
        violation = dot(pts, normals) - offsets  # every vertex must lie in every edge's half-plane
        if np.any(turns <= TOL * lengths * np.roll(lengths, -1)) or np.any(violation > TOL):
            raise ValueError("polygon is not strictly convex")
        return cls(_frozen(pts), _frozen(normals), _frozen(offsets))

    @classmethod
    def from_box(cls, box: Box) -> ConvexPolygon:
        """The axis-aligned rectangle ``box``."""
        (x0, x1), (y0, y1) = box.x, box.y
        return cls.from_vertices([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    def violation(self, points: ArrayLike) -> FloatArray:
        """``max_k (n_k . x - b_k)`` per point: negative inside, zero on the boundary, positive outside.

        It is a signed distance to the boundary only up to a factor; use it as a predicate.
        """
        result: FloatArray = np.max(dot(_points(points), self.normals) - self.offsets, axis=1)
        return result

    def contains(self, points: ArrayLike, tol: float = TOL) -> BoolArray:
        """Whether each point lies in the closed polygon, up to ``tol``."""
        result: BoolArray = self.violation(points) <= tol
        return result

    def contains_interior(self, points: ArrayLike, tol: float = TOL) -> BoolArray:
        """Whether each point lies in the interior, deeper than ``tol``."""
        result: BoolArray = self.violation(points) < -tol
        return result

    def width(self) -> float:
        """Smallest width of the polygon, attained across one of its faces."""
        depth = self.offsets[:, None] - dot(self.vertices, self.normals).T  # (faces, vertices)
        return float(np.min(np.max(depth, axis=1)))

    def room(self, face: int) -> float:
        """How far ``face`` can move outward before the lines of its two neighbours meet it (``inf`` if never)."""
        m = len(self.offsets)
        a, b = (face - 1) % m, (face + 1) % m
        (na_x, na_y), (nb_x, nb_y) = self.normals[a], self.normals[b]
        det = na_x * nb_y - na_y * nb_x
        if abs(det) <= 1e-12:  # parallel neighbours, as in a box: the face can move freely
            return float(np.inf)
        ba, bb = self.offsets[a], self.offsets[b]
        meet = np.array([ba * nb_y - na_y * bb, na_x * bb - ba * nb_x])  # Cramer's rule, times det
        gap = float(self.normals[face] @ meet / det - self.offsets[face])
        return gap if gap > 0.0 else float(np.inf)

    def pushed_out(self, shifts: ArrayLike) -> ConvexPolygon:
        """The polygon with the line of each face ``k`` moved outward by ``shifts[k]``.

        Raises:
            ValueError: If a shift reaches the point where the neighbours' lines meet
                (see :meth:`room`), so that the result is not a polygon with the same faces.
        """
        offsets = self.offsets + np.asarray(shifts, dtype=np.float64)
        # vertex k is the intersection of the lines of edges k-1 and k
        matrices = np.stack([np.roll(self.normals, 1, axis=0), self.normals], axis=1)
        rhs = np.stack([np.roll(offsets, 1), offsets], axis=1)
        return ConvexPolygon.from_vertices(np.linalg.solve(matrices, rhs[..., None])[..., 0])


def _separation(a: ConvexPolygon, b: ConvexPolygon) -> float:
    """Largest gap between the projections of ``a`` and ``b`` over their edge normals.

    Positive: the closed polygons are disjoint, at least this far apart along
    some direction. Negative: the interiors overlap. Zero: they touch
    (separating axis theorem).
    """
    gaps = [
        np.min(dot(b.vertices, a.normals), axis=0) - a.offsets,
        np.min(dot(a.vertices, b.normals), axis=0) - b.offsets,
    ]
    return float(np.max(np.concatenate(gaps)))


@dataclass(frozen=True, eq=False)
class Faces:
    """Every wall face of a layout, oriented toward the free space.

    Faces ``0 .. n_arena - 1`` belong to the arena; the faces of each blocked
    polygon follow, contiguous and in the order of :attr:`Layout.blocked`.

    Attributes:
        start, end: ``(F, 2)`` end points of each face.
        tangent: ``(F, 2)`` unit vector from ``start`` to ``end``.
        length: ``(F,)`` face lengths.
        normal: ``(F, 2)`` unit normal pointing into the free space.
        offset: ``(F,)``; ``normal . x - offset`` is the signed distance of
            ``x`` from the face line, positive on the free side.
        owner: ``(F,)`` -1 for the arena, ``i`` for blocked polygon ``i``.
        n_arena: Number of arena faces.
        group_starts: Index of the first face of each blocked polygon,
            counted from ``n_arena``.
        group_sizes: Number of faces of each blocked polygon.
    """

    start: FloatArray
    end: FloatArray
    tangent: FloatArray
    length: FloatArray
    normal: FloatArray
    offset: FloatArray
    owner: IntArray
    n_arena: int
    group_starts: IntArray
    group_sizes: IntArray

    @classmethod
    def of(cls, arena: ConvexPolygon, blocked: Sequence[ConvexPolygon]) -> Faces:
        """The faces of ``arena`` (normals inward) and of ``blocked`` (normals outward)."""
        starts, ends, normals, offsets, owners = [], [], [], [], []
        for index, (polygon, sign) in enumerate([(arena, -1.0)] + [(b, 1.0) for b in blocked]):
            starts.append(polygon.vertices)
            ends.append(np.roll(polygon.vertices, -1, axis=0))
            normals.append(sign * polygon.normals)
            offsets.append(sign * polygon.offsets)
            owners.append(np.full(len(polygon.offsets), index - 1))
        start, end = np.concatenate(starts), np.concatenate(ends)
        length = np.hypot(*(end - start).T)
        sizes = np.array([len(b.offsets) for b in blocked], dtype=np.int64)
        group_starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]) if len(sizes) else sizes
        return cls(
            start=_frozen(start),
            end=_frozen(end),
            tangent=_frozen((end - start) / length[:, None]),
            length=_frozen(length),
            normal=_frozen(np.concatenate(normals)),
            offset=_frozen(np.concatenate(offsets)),
            owner=_frozen(np.concatenate(owners), dtype=np.int64),
            n_arena=len(arena.offsets),
            group_starts=_frozen(group_starts, dtype=np.int64),
            group_sizes=_frozen(sizes, dtype=np.int64),
        )

    def signed_distances(self, points: ArrayLike) -> FloatArray:
        """``(n, F)`` signed distances of the points from the face lines, positive on the free side."""
        result: FloatArray = dot(_points(points), self.normal) - self.offset
        return result

    def on_face(self, points: ArrayLike, distances: FloatArray | None = None, tol: float = TOL) -> BoolArray:
        """``(n, F)`` whether each point lies on each face itself, not only on its line.

        Args:
            points: ``(n, 2)`` positions.
            distances: Their signed distances, if already computed.
            tol: Tolerance across and along the face.
        """
        pts = _points(points)
        if distances is None:
            distances = self.signed_distances(pts)
        along = (pts[:, :1] - self.start[:, 0]) * self.tangent[:, 0]
        along += (pts[:, 1:] - self.start[:, 1]) * self.tangent[:, 1]
        result: BoolArray = (np.abs(distances) <= tol) & (along >= -tol) & (along <= self.length + tol)
        return result


@dataclass(frozen=True, eq=False)
class Layout:
    """Arena, obstacles and goal region of one map (D11), with the free space of D16.

    Build it with :meth:`build` or :meth:`from_config`.

    Attributes:
        arena: The arena ``P``; every face is a wall.
        obstacles: The obstacles ``O_i`` as given, inside the arena.
        blocked: The obstacles as the physics sees them: an obstacle touching
            the arena boundary with a face is extended beyond that wall (D16),
            by :data:`EXTENSION` or, if its neighbouring faces converge
            outside the arena, by half the room they leave. Inside the arena
            ``blocked[i]`` and ``obstacles[i]`` coincide.
        goal_x: The goal region is ``{p in P_free : p_x >= goal_x}``.
        faces: All wall faces, oriented toward the free space.
    """

    arena: ConvexPolygon
    obstacles: tuple[ConvexPolygon, ...]
    blocked: tuple[ConvexPolygon, ...]
    goal_x: float
    faces: Faces

    @classmethod
    def build(cls, arena: ConvexPolygon, obstacles: Sequence[ConvexPolygon], goal_x: float) -> Layout:
        """Validate the layout and derive the blocked polygons and the faces.

        Raises:
            ValueError: If the goal line is not strictly inside the arena; an
                obstacle is not inside the arena, is thinner than
                :data:`MIN_GAP`, lies closer than :data:`MIN_GAP` to a wall
                it does not touch, or touches a wall with a vertex only; or
                two obstacles touch, or are closer than :data:`MIN_GAP`,
                without overlapping (D16).
        """
        x_lo, x_hi = float(arena.vertices[:, 0].min()), float(arena.vertices[:, 0].max())
        if not x_lo < goal_x < x_hi:
            raise ValueError(f"goal_x={goal_x} must lie strictly inside the arena's x-range [{x_lo}, {x_hi}]")
        blocked = [cls._extended(arena, obstacle, i) for i, obstacle in enumerate(obstacles)]
        for (i, a), (j, b) in combinations(enumerate(obstacles), 2):
            if -TOL <= _separation(a, b) < MIN_GAP:
                raise ValueError(
                    f"obstacles {i} and {j} touch or are closer than MIN_GAP; separate them or let them overlap (D16)"
                )
        return cls(arena, tuple(obstacles), tuple(blocked), float(goal_x), Faces.of(arena, blocked))

    @staticmethod
    def _extended(arena: ConvexPolygon, obstacle: ConvexPolygon, index: int) -> ConvexPolygon:
        """``obstacle`` with every face that lies on a wall pushed beyond it (D16)."""
        if obstacle.width() < MIN_GAP:
            raise ValueError(f"obstacle {index} is thinner than MIN_GAP")
        clearance = arena.offsets[:, None] - dot(obstacle.vertices, arena.normals).T  # (walls, vertices), >= 0 inside
        nearest = clearance.min(axis=1)
        if np.any(nearest < -TOL):
            raise ValueError(f"obstacle {index} is not inside the arena")
        if np.any((nearest > TOL) & (nearest < MIN_GAP)):
            raise ValueError(f"obstacle {index} is closer than MIN_GAP to a wall it does not touch")
        shifts = np.zeros(len(obstacle.offsets))
        for wall in np.flatnonzero(nearest <= TOL):
            on_wall = np.flatnonzero(
                (dot(obstacle.normals, arena.normals[wall : wall + 1])[:, 0] > 1.0 - 1e-12)
                & (np.abs(clearance[wall]) <= TOL)
                & (np.abs(np.roll(clearance[wall], -1)) <= TOL)
            )
            if on_wall.size == 0:
                raise ValueError(
                    f"obstacle {index} touches the arena boundary with a vertex only; move it off the wall (D16)"
                )
            for face in on_wall:
                shifts[face] = min(EXTENSION, 0.5 * obstacle.room(int(face)))
                if shifts[face] < MIN_GAP:
                    raise ValueError(
                        f"obstacle {index} meets the arena boundary at too shallow an angle to be extended (D16)"
                    )
        return obstacle.pushed_out(shifts) if shifts.any() else obstacle

    @classmethod
    def from_config(cls, config: GeometryConfig) -> Layout:
        """The layout of a geometry configuration (boxes)."""
        return cls.build(
            ConvexPolygon.from_box(config.arena),
            [ConvexPolygon.from_box(box) for box in config.obstacles],
            config.goal_x,
        )

    def in_free_space(self, points: ArrayLike, tol: float = TOL) -> BoolArray:
        """Whether each point lies in ``P_free`` (D16), up to ``tol``."""
        pts = _points(points)
        free = self.arena.contains(pts, tol)
        for polygon in self.blocked:
            free &= ~polygon.contains_interior(pts, tol)
        return free

    def in_goal(self, points: ArrayLike) -> BoolArray:
        """Whether each point lies in the goal half-plane ``p_x >= goal_x`` (the point is assumed free)."""
        result: BoolArray = _points(points)[:, 0] >= self.goal_x
        return result
