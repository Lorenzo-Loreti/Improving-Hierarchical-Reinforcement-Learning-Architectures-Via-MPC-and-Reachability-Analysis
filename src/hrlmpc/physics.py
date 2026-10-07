"""The physical step of the navigation environment (project instructions, "Setup"; rows E1-E7).

One pure function, :func:`physics_step`, advances a batch of agents by one
sampling period. Every architecture reaches the plant through it, and nothing
else in the code integrates the dynamics; a single environment is a batch of
one, with bit-for-bit the same result. For each agent with state
``x = (p, v)``, commanded acceleration ``u_cmd`` and disturbance ``d`` in ``D``:

1. saturation onto ``U``: ``u = Proj_U(u_cmd)`` (radial);
2. candidate velocity: ``v~ = (1 - gamma dt) v + dt (u + d)``;
3. speed limiter: ``v^ = Proj_V(v~)`` (radial);
4. wall reaction (decision log D17): starting from ``v = v^``, find the
   first face that the segment ``[p, p + dt v]`` crosses into a blocked
   region (the interior of an obstacle, or the outside of the arena); add it
   to the collected faces ``S``; set ``v`` to the Euclidean projection of
   ``v^`` onto the intersection of the half-planes
   ``{v : n_j . v >= -delta_j / dt}``, ``j in S``; repeat until the segment
   crosses no face. Here ``n_j`` is the unit normal of face ``j`` pointing
   into the free space and ``delta_j >= 0`` the distance of ``p`` from its
   line. Faces hit at the same instant are taken in order of the correction
   they need, smallest first, then by index;
5. wall friction: if the agent lies on a face, moves along it
   (``n_j . v = 0``) and is still on it at the end of the step,
   ``v <- (1 - gamma_w dt) v``;
6. position: ``p+ = p + dt v``.

By decision log F4 the wall reaction keeps ``||v||_2 <= v_max``.

Tolerances. The free space is the one of :mod:`hrlmpc.geometry` (D16). A
segment crosses into a blocked region only if it reaches more than
``geometry.TOL`` into it; a start that rounding has left slightly deeper is
treated as lying at that depth, so the agent can neither sink further nor be
stopped by a face it is not moving into. Input positions are accepted up to
``2 TOL`` outside the free space, which covers the rounding of the step's own
results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrlmpc.geometry import TOL, Faces, Layout, dot
from hrlmpc.model import PhysicsParams

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

INPUT_TOL = 2.0 * TOL
"""Distance [m] outside the free space up to which an input position is accepted."""

_FEASIBILITY_TOL = 1e-12
"""Slack [m/s] accepted on a half-plane when choosing the projection."""

_TINY = float(np.finfo(np.float64).tiny)


@dataclass(frozen=True, eq=False)
class StepResult:
    """Outcome of :func:`physics_step` for ``B`` agents and the ``F`` faces of the layout.

    Attributes:
        p: ``(B, 2)`` new positions ``p_(k+1)``.
        v: ``(B, 2)`` new velocities ``v_(k+1)``.
        u: ``(B, 2)`` inputs after the saturation onto ``U`` (stage 1).
        v_candidate: ``(B, 2)`` candidate velocities (stage 2).
        v_limited: ``(B, 2)`` velocities after the speed limiter (stage 3).
        v_wall: ``(B, 2)`` velocities after the wall reaction (stage 4),
            before the wall friction.
        wall_faces: ``(B, F)`` faces collected by the wall reaction.
        friction: ``(B,)`` whether the wall friction acted (stage 5).
        contact: ``(B,)`` contact in the sense of D14: a wall reaction, or the
            new position on a face.
    """

    p: FloatArray
    v: FloatArray
    u: FloatArray
    v_candidate: FloatArray
    v_limited: FloatArray
    v_wall: FloatArray
    wall_faces: BoolArray
    friction: BoolArray
    contact: BoolArray

    @property
    def dv_wall(self) -> FloatArray:
        """``(B, 2)`` velocity change of the wall reaction: the impulse per unit mass of D14."""
        result: FloatArray = self.v_wall - self.v_limited
        return result

    @property
    def wall_reaction(self) -> BoolArray:
        """``(B,)`` whether the wall reaction acted."""
        result: BoolArray = self.wall_faces.any(axis=1)
        return result


def physics_step(
    p: ArrayLike,
    v: ArrayLike,
    u_cmd: ArrayLike,
    d: ArrayLike,
    params: PhysicsParams,
    layout: Layout,
) -> StepResult:
    """Advance ``B`` agents by one step (stages 1-6 of the module docstring).

    The function is pure: it reads its arguments and returns new arrays. Each
    agent's result depends only on its own inputs.

    Args:
        p: ``(B, 2)`` positions, in the free space of ``layout``.
        v: ``(B, 2)`` velocities, with ``||v||_2 <= v_max``.
        u_cmd: ``(B, 2)`` commanded accelerations; any finite value.
        d: ``(B, 2)`` disturbances, with ``||d||_2 <= d_bar``.
        params: Physical parameters.
        layout: Arena and obstacles.

    Returns:
        The new state and the intermediate quantities of the step. The new
        state is again a valid input.

    Raises:
        ValueError: If the shapes differ or are not ``(B, 2)``, a value is
            not finite, or a state or disturbance violates its set.
    """
    p_arr, v_arr, u_arr, d_arr = (_batch(name, x) for name, x in (("p", p), ("v", v), ("u_cmd", u_cmd), ("d", d)))
    if not p_arr.shape == v_arr.shape == u_arr.shape == d_arr.shape:
        shapes = [x.shape for x in (p_arr, v_arr, u_arr, d_arr)]
        raise ValueError(f"p, v, u_cmd and d must have the same shape, got {shapes}")
    if not layout.in_free_space(p_arr, tol=INPUT_TOL).all():
        raise ValueError("every position must lie in the free space")
    if np.any(_norm(v_arr) > params.v_max + TOL):
        raise ValueError("every speed must be at most v_max")
    if np.any(_norm(d_arr) > params.d_bar + TOL):
        raise ValueError("every disturbance must lie in D (norm at most d_bar)")

    dt = params.dt
    faces = layout.faces
    u = _project_disk(u_arr, params.a_max)  # 1
    v_candidate = params.velocity_factor * v_arr + dt * (u + d_arr)  # 2
    v_limited = _project_disk(v_candidate, params.v_max)  # 3
    distances = faces.signed_distances(p_arr)
    v_wall, wall_faces = _wall_reaction(v_limited, distances, dt, faces)  # 4
    sliding = faces.on_face(p_arr, distances) & (np.abs(dot(v_wall, faces.normal)) <= TOL / dt)
    friction = np.any(sliding & faces.on_face(p_arr + dt * v_wall), axis=1)
    v_next = np.where(friction[:, None], (1.0 - params.gamma_w * dt) * v_wall, v_wall)  # 5
    p_next = p_arr + dt * v_next  # 6
    contact = wall_faces.any(axis=1) | faces.on_face(p_next).any(axis=1)
    return StepResult(p_next, v_next, u, v_candidate, v_limited, v_wall, wall_faces, friction, contact)


def _batch(name: str, x: ArrayLike) -> FloatArray:
    out = np.array(x, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 2 or out.shape[0] < 1:
        raise ValueError(f"{name} must have shape (B, 2), got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    return out


def _norm(x: FloatArray) -> FloatArray:
    """Row norms, without overflow for huge entries."""
    result: FloatArray = np.hypot(x[:, 0], x[:, 1])
    return result


def _project_disk(x: FloatArray, radius: float) -> FloatArray:
    """Radial projection of each row onto the disk of radius ``radius``."""
    scale = np.minimum(1.0, radius / np.maximum(_norm(x), _TINY))
    result: FloatArray = x * scale[:, None]
    return result


def _wall_reaction(
    v_hat: FloatArray, distances: FloatArray, dt: float, faces: Faces
) -> tuple[FloatArray, BoolArray]:
    """Stage 4: the wall velocity and the collected faces (D17)."""
    bound = -np.maximum(distances, 0.0) / dt  # half-plane of face j: n_j . v >= bound_j
    v = v_hat.copy()
    collected = np.zeros(distances.shape, dtype=bool)
    for _ in range(distances.shape[1] + 1):
        face = _first_impact(v, distances, bound, collected, dt, faces)
        hit = np.flatnonzero(face >= 0)
        if hit.size == 0:
            return v, collected
        collected[hit, face[hit]] = True
        v[hit] = _project(v_hat[hit], collected[hit], bound[hit], faces.normal)
    raise RuntimeError("the wall reaction did not settle")  # each pass collects a new face, so this is unreachable


def _first_impact(
    v: FloatArray, distances: FloatArray, bound: FloatArray, collected: BoolArray, dt: float, faces: Faces
) -> NDArray[np.int64]:
    """Index of the first face crossed by each segment ``[p, p + dt v]``, or -1.

    A segment crosses into a blocked region when it reaches more than ``TOL``
    into it, counted from the start if rounding has left the start deeper.
    The crossing time of a face is that of its line, ``t in [0, 1]`` along the
    segment. Faces crossed within ``TOL`` of the first crossing point count as
    simultaneous; among them the one needing the smallest correction
    ``bound_j - n_j . v`` is taken, then the lowest index.
    """
    normal_speed = dot(v, faces.normal)
    rate = dt * normal_speed  # change of each signed distance along the segment, per unit of t
    tie = TOL / np.maximum(dt * _norm(v), _TINY)[:, None]  # TOL along the segment, in units of t
    with np.errstate(divide="ignore", invalid="ignore"):
        line_time = np.where(rate < 0.0, -distances / rate, -np.inf)  # when the face line is crossed
    time = np.full(distances.shape, np.inf)

    # arena: the end point lies more than TOL beyond a face line, or beyond the start if that is deeper
    arena = slice(0, faces.n_arena)
    leaves = (np.maximum(distances[:, arena] + TOL, 0.0) + rate[:, arena] < 0.0) & ~collected[:, arena]
    time[:, arena] = np.where(leaves, np.maximum(line_time[:, arena], 0.0), np.inf)

    # blocked polygons: Cyrus-Beck clipping against the interior shrunk by TOL, or by more if the start lies deeper
    if len(faces.group_sizes):
        rest = slice(faces.n_arena, None)
        starts, sizes = faces.group_starts, faces.group_sizes
        shifted = distances[:, rest] + TOL
        start_depth = np.maximum(0.0, -np.maximum.reduceat(shifted, starts, axis=1))
        shifted = shifted + np.repeat(start_depth, sizes, axis=1)
        r = rate[:, rest]
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = -shifted / r
        enter = np.maximum(np.maximum.reduceat(np.where(r < 0.0, tau, -np.inf), starts, axis=1), 0.0)
        leave = np.minimum(np.minimum.reduceat(np.where(r > 0.0, tau, np.inf), starts, axis=1), 1.0)
        parallel_outside = np.logical_or.reduceat((r == 0.0) & (shifted >= 0.0), starts, axis=1)
        penetrates = np.repeat((enter < leave) & ~parallel_outside, sizes, axis=1)
        # entry face: the last line crossed among the faces the segment moves toward
        candidate = np.where((r < 0.0) & ~collected[:, rest], line_time[:, rest], -np.inf)
        last = np.repeat(np.maximum.reduceat(candidate, starts, axis=1), sizes, axis=1)
        entry = penetrates & np.isfinite(candidate) & (candidate >= last - tie)
        time[:, rest] = np.where(entry, np.maximum(last, 0.0), np.inf)

    first = time.min(axis=1, keepdims=True)
    simultaneous = np.isfinite(time) & (time <= first + tie)
    correction = np.where(simultaneous, bound - normal_speed, np.inf)
    face = np.argmin(correction, axis=1)
    result: NDArray[np.int64] = np.where(np.isfinite(first[:, 0]), face, -1)
    return result


def _project(v_hat: FloatArray, active: BoolArray, bound: FloatArray, normal: FloatArray) -> FloatArray:
    """Euclidean projection of each ``v_hat[r]`` onto ``{v : n_j . v >= bound[r, j], j active}``.

    Exact in the plane: the projection is ``v_hat`` itself, its projection
    onto one active line, or the intersection of two; the nearest feasible
    candidate is taken. Only the active faces of each row are used.
    """
    rows = np.arange(len(v_hat))
    k = int(active.sum(axis=1).max())
    order = np.argsort(~active, axis=1, kind="stable")[:, :k]  # active faces first, by index
    valid = np.take_along_axis(active, order, axis=1)  # (R, k)
    n = normal[order]  # (R, k, 2)
    b = np.take_along_axis(bound, order, axis=1)  # (R, k)
    gap = b - (v_hat[:, None, 0] * n[..., 0] + v_hat[:, None, 1] * n[..., 1])
    singles = v_hat[:, None, :] + gap[..., None] * n
    i, j = np.triu_indices(k, 1)
    det = n[:, i, 0] * n[:, j, 1] - n[:, i, 1] * n[:, j, 0]
    solvable = np.abs(det) > 1e-12
    det = np.where(solvable, det, 1.0)
    pairs = np.stack(
        [
            (b[:, i] * n[:, j, 1] - n[:, i, 1] * b[:, j]) / det,
            (n[:, i, 0] * b[:, j] - b[:, i] * n[:, j, 0]) / det,
        ],
        axis=-1,
    )
    candidates = np.concatenate([v_hat[:, None, :], singles, pairs], axis=1)
    usable = np.concatenate([np.ones((len(rows), 1), dtype=bool), valid, valid[:, i] & valid[:, j] & solvable], axis=1)
    slack = candidates[:, :, None, 0] * n[:, None, :, 0] + candidates[:, :, None, 1] * n[:, None, :, 1] - b[:, None, :]
    feasible = usable & np.all((slack >= -_FEASIBILITY_TOL) | ~valid[:, None, :], axis=2)
    distance = np.where(feasible, np.sum((candidates - v_hat[:, None, :]) ** 2, axis=2), np.inf)
    best = np.argmin(distance, axis=1)
    if not np.all(np.isfinite(distance[rows, best])):
        raise RuntimeError("no feasible projection candidate")  # the origin is always feasible, so unreachable
    result: FloatArray = candidates[rows, best]
    return result
