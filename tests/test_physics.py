"""Tests of the physical step of the environment (rows M1-M7, E1-E8; decision log D16, D17).

Several tests give the agent the velocity ``v_hat / FACTOR`` with zero input
and zero disturbance: the free motion multiplies the velocity by
``FACTOR = 1 - gamma dt``, so the limited velocity of the step is ``v_hat``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import nnls

from hrlmpc.config import Box, load_env_config
from hrlmpc.geometry import TOL, ConvexPolygon, Layout
from hrlmpc.model import PhysicsParams, free_step
from hrlmpc.physics import INPUT_TOL, StepResult, physics_step

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "env"
PARAMS = PhysicsParams(dt=0.1, a_max=2.5, v_max=1.2, gamma=0.5, gamma_w=1.0, d_bar=1.0)  # D9, largest d_bar
FACTOR = PARAMS.velocity_factor
SEED = 0

Array = NDArray[np.float64]


def _layout(name: str) -> Layout:
    return Layout.from_config(load_env_config(CONFIG_DIR / f"{name}.yaml").geometry)


def _box(x0: float, y0: float, x1: float, y1: float) -> ConvexPolygon:
    return ConvexPolygon.from_box(Box(x=(x0, x1), y=(y0, y1)))


def _face(layout: Layout, a: tuple[float, float], b: tuple[float, float]) -> int:
    """Index of the face with end points ``a`` and ``b``, in either order."""
    f = layout.faces
    ends = np.stack([f.start, f.end], axis=1)
    match = np.all(np.isclose(ends, [a, b]), axis=(1, 2)) | np.all(np.isclose(ends, [b, a]), axis=(1, 2))
    (index,) = np.flatnonzero(match)
    return int(index)


def _step(
    layout: Layout,
    p: ArrayLike,
    v: ArrayLike,
    u: ArrayLike | None = None,
    d: ArrayLike | None = None,
    params: PhysicsParams = PARAMS,
) -> StepResult:
    p_arr = np.array(p, dtype=np.float64)
    u_arr = np.zeros_like(p_arr) if u is None else np.array(u, dtype=np.float64)
    d_arr = np.zeros_like(p_arr) if d is None else np.array(d, dtype=np.float64)
    return physics_step(p_arr, np.array(v, dtype=np.float64), u_arr, d_arr, params, layout)


def _disk(rng: np.random.Generator, n: int, radius: float) -> Array:
    """``n`` points uniform in area on the disk of radius ``radius``."""
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    return np.stack([r * np.cos(angle), r * np.sin(angle)], axis=1)


def _circle(rng: np.random.Generator, n: int, radius: float) -> Array:
    """``n`` points uniform on the circle of radius ``radius``."""
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    return radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)


def _free_positions(rng: np.random.Generator, layout: Layout, n: int, on_walls: float = 0.3) -> Array:
    """``n`` free positions; about a fraction ``on_walls`` of them lie on a face."""
    lo, hi = layout.arena.vertices.min(axis=0), layout.arena.vertices.max(axis=0)
    faces = layout.faces
    chunks, count = [], 0
    while count < n:
        m = 4 * n
        points = rng.uniform(lo, hi, size=(m, 2))
        k = int(on_walls * m)
        j = rng.integers(len(faces.length), size=k)
        points[:k] = faces.start[j] + rng.uniform(0.0, 1.0, (k, 1)) * (faces.end[j] - faces.start[j])
        points = points[layout.in_free_space(points)]
        chunks.append(points)
        count += len(points)
    return rng.permutation(np.concatenate(chunks))[:n]


def _convex(rng: np.random.Generator, center: ArrayLike, radius: float, sides: int) -> Array:
    """Vertices of a random convex polygon inscribed in a circle."""
    jitter = rng.uniform(-0.3, 0.3, sides)
    angle = (np.arange(sides) + jitter) * 2.0 * np.pi / sides + rng.uniform(0.0, 2.0 * np.pi)
    return np.asarray(center, dtype=np.float64) + radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)


def _random_layout(rng: np.random.Generator) -> Layout:
    """A random valid layout: convex polygons in a convex arena, or boxes in a box, some against the walls."""
    while True:
        if rng.random() < 0.5:
            sides = int(rng.integers(4, 9))
            arena = ConvexPolygon.from_vertices(_convex(rng, (0.0, 0.0), rng.uniform(2.5, 3.5), sides))
            obstacles = [
                ConvexPolygon.from_vertices(
                    _convex(rng, rng.uniform(-1.5, 1.5, 2), rng.uniform(0.2, 0.7), int(rng.integers(3, 7)))
                )
                for _ in range(int(rng.integers(1, 5)))
            ]
            goal_x = 0.5 * float(arena.vertices[:, 0].max())
        else:
            arena = _box(0.0, 0.0, 6.0, 4.0)
            obstacles = []
            for _ in range(int(rng.integers(1, 6))):
                x0 = rng.uniform(0.3, 5.0)
                x1 = min(x0 + rng.uniform(0.3, 1.0), 6.0)
                kind = rng.integers(3)
                if kind == 0:  # on the floor
                    y0, y1 = 0.0, rng.uniform(0.5, 3.0)
                elif kind == 1:  # from the ceiling
                    y0, y1 = rng.uniform(1.0, 3.5), 4.0
                else:
                    y0 = rng.uniform(0.3, 2.5)
                    y1 = min(y0 + rng.uniform(0.3, 1.2), 4.0)
                obstacles.append(_box(x0, y0, x1, y1))
            goal_x = 5.5
        try:
            return Layout.build(arena, obstacles, goal_x=goal_x)
        except ValueError:  # an invalid layout, e.g. two obstacles that touch (D16)
            continue


def _enters(polygon: ConvexPolygon, p: Array, q: Array, depth: float) -> NDArray[np.bool_]:
    """Whether each segment ``[p_i, q_i]`` reaches deeper than ``depth`` into ``polygon`` (Liang-Barsky).

    Written independently of :mod:`hrlmpc.physics`, as a reference.
    """
    d = q - p
    lo, hi = np.zeros(len(p)), np.ones(len(p))
    parallel_outside = np.zeros(len(p), dtype=bool)
    for normal, offset in zip(polygon.normals, polygon.offsets):
        room = offset - depth - p @ normal  # inside this half-plane while t (d . n) < room
        speed = d @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = room / speed
        hi = np.where(speed > 0.0, np.minimum(hi, ratio), hi)
        lo = np.where(speed < 0.0, np.maximum(lo, ratio), lo)
        parallel_outside |= (speed == 0.0) & (room <= 0.0)
    return (lo < hi) & ~parallel_outside


def _segments_are_free(layout: Layout, p: Array, q: Array) -> NDArray[np.bool_]:
    """Whether every segment ``[p_i, q_i]`` stays in the arena and out of every obstacle, up to ``INPUT_TOL``.

    The obstacles are checked both as given and as extended by D16.
    """
    ok = layout.arena.contains(p, tol=INPUT_TOL) & layout.arena.contains(q, tol=INPUT_TOL)  # the arena is convex
    for polygon in layout.obstacles + layout.blocked:
        ok &= ~_enters(polygon, p, q, INPUT_TOL)
    return ok


def _assert_projection(layout: Layout, p: Array, out: StepResult, dt: float) -> None:
    """KKT conditions: the wall velocity is the projection of the limited velocity onto the collected half-planes."""
    distances = layout.faces.signed_distances(p)
    for i in np.flatnonzero(out.wall_faces.any(axis=1)):
        rows = np.flatnonzero(out.wall_faces[i])
        normals = layout.faces.normal[rows]
        bound = -np.maximum(distances[i, rows], 0.0) / dt
        slack = normals @ out.v_wall[i] - bound
        assert np.all(slack >= -1e-9), "the wall velocity violates a collected half-plane"
        active = slack <= 1e-9
        assert active.any(), "a collected face must constrain the wall velocity"
        _, residual = nnls(normals[active].T, out.v_wall[i] - out.v_limited[i])
        assert residual <= 1e-9, "the correction is not in the normal cone of the active half-planes"


def test_step_returns_batches_and_leaves_its_inputs_untouched() -> None:
    layout = _layout("slalom")
    rng = np.random.default_rng(SEED)
    p = np.array([[1.0, 0.0], [2.0, -2.0], [3.95, 0.3]])
    v, u, d = _disk(rng, 3, PARAMS.v_max), rng.uniform(-5.0, 5.0, (3, 2)), _disk(rng, 3, PARAMS.d_bar)
    copies = [x.copy() for x in (p, v, u, d)]
    out = physics_step(p, v, u, d, PARAMS, layout)
    for given, copy in zip((p, v, u, d), copies):
        np.testing.assert_array_equal(given, copy)
    for name in ("p", "v", "u", "v_candidate", "v_limited", "dv_wall", "v_wall"):
        assert getattr(out, name).shape == (3, 2)
    assert out.wall_faces.shape == (3, len(layout.faces.length))
    assert out.friction.shape == out.contact.shape == out.wall_reaction.shape == (3,)


@pytest.mark.parametrize(
    ("p", "v", "u", "d", "message"),
    [
        ([[4.5, -1.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], "free space"),  # inside O1
        ([[4.5, -2.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], "free space"),  # in the gap under O1 (D16)
        ([[1.0, 0.0]], [[1.3, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], "v_max"),
        ([[1.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[1.5, 0.0]], "disturbance"),
        ([[1.0, 0.0], [2.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], "shape"),
        ([1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], "shape"),
        ([[1.0, 0.0]], [[0.0, 0.0]], [[np.nan, 0.0]], [[0.0, 0.0]], "finite"),
    ],
)
def test_invalid_inputs_are_rejected(p: ArrayLike, v: ArrayLike, u: ArrayLike, d: ArrayLike, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _step(_layout("slalom"), p, v, u, d)


def test_step_equals_the_linear_model_when_no_correction_acts() -> None:
    """M2, M3, E2, E6: away from the walls and below the speed limit the step is x+ = A x + B u + B d."""
    rng = np.random.default_rng(SEED)
    n = 500
    p = rng.uniform([1.0, -1.0], [9.0, 1.0], size=(n, 2))  # at least 1 m from every wall of the tunnel
    v, u, d = _disk(rng, n, 0.5), _disk(rng, n, PARAMS.a_max), _disk(rng, n, PARAMS.d_bar)
    out = _step(_layout("tunnel"), p, v, u, d)  # speed <= 0.95 * 0.5 + 0.1 * 3.5 < v_max
    expected = free_step(np.hstack([p, v]), u, d, PARAMS.dt, PARAMS.gamma)
    np.testing.assert_allclose(np.hstack([out.p, out.v]), expected, rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(out.u, u)
    np.testing.assert_array_equal(out.v_limited, out.v_candidate)
    assert not (out.wall_reaction.any() or out.friction.any() or out.contact.any())


def test_disturbance_enters_like_an_acceleration() -> None:
    """M5: d is added to the saturated input, so it enters through B only, also at the walls."""
    layout = _layout("slalom")
    rng = np.random.default_rng(SEED)
    n = 2000
    p = _free_positions(rng, layout, n)
    v, d = _disk(rng, n, PARAMS.v_max), _disk(rng, n, PARAMS.d_bar)
    u = _disk(rng, n, PARAMS.a_max - PARAMS.d_bar)
    with_d = _step(layout, p, v, u, d)
    in_u = _step(layout, p, v, u + d, np.zeros_like(d))
    np.testing.assert_array_equal(with_d.p, in_u.p)
    np.testing.assert_array_equal(with_d.v, in_u.v)
    np.testing.assert_array_equal(with_d.wall_faces, in_u.wall_faces)
    assert with_d.wall_reaction.any()  # the walls were part of the test


def test_disturbance_is_not_saturated_with_the_input() -> None:
    """M5: the commanded action is saturated first, then d is added: u = (2.5, 0), u + d = (1.5, 0)."""
    out = _step(_layout("tunnel"), [[2.0, 0.0]], [[0.0, 0.0]], [[5.0, 0.0]], [[-1.0, 0.0]])
    np.testing.assert_allclose(out.v_candidate, [[0.15, 0.0]], atol=1e-15)


def test_commanded_action_is_saturated_radially() -> None:
    """E1."""
    u_cmd = np.array([[5.0, 0.0], [3.0, 4.0], [1.0, -1.0], [0.0, 0.0]])
    out = _step(_layout("tunnel"), np.tile([[2.0, 0.0]], (4, 1)), np.zeros((4, 2)), u_cmd)
    np.testing.assert_allclose(out.u, [[2.5, 0.0], [1.5, 2.0], [1.0, -1.0], [0.0, 0.0]], atol=1e-15)


def test_speed_limiter_is_radial_and_the_speed_limit_always_holds() -> None:
    """E3 and DL-F4: also with walls, saturated inputs and worst-case disturbances."""
    layout = _layout("slalom")
    rng = np.random.default_rng(SEED)
    n = 3000
    p = _free_positions(rng, layout, n)
    out = _step(layout, p, _disk(rng, n, PARAMS.v_max), rng.uniform(-10.0, 10.0, (n, 2)), _circle(rng, n, PARAMS.d_bar))
    speed = np.linalg.norm(out.v_candidate, axis=1, keepdims=True)
    expected = out.v_candidate * np.minimum(1.0, PARAMS.v_max / np.maximum(speed, 1e-300))
    np.testing.assert_allclose(out.v_limited, expected, rtol=0.0, atol=1e-15)
    assert np.all(np.linalg.norm(out.v, axis=1) <= PARAMS.v_max + 1e-12)
    assert (speed[:, 0] > PARAMS.v_max).any() and out.wall_reaction.any()


def test_speed_limiter_acts_before_the_wall_reaction() -> None:
    """E3, DL-A4: the candidate velocity would reach the floor in one step, the limited one does not."""
    diagonal = np.array([[1.0, -1.0]]) / np.sqrt(2.0)
    out = _step(_layout("tunnel"), [[2.0, -1.91]], PARAMS.v_max * diagonal, PARAMS.a_max * diagonal)
    assert np.linalg.norm(out.v_candidate) > PARAMS.v_max and PARAMS.dt * -out.v_candidate[0, 1] > 0.09
    assert not out.wall_reaction.any()
    np.testing.assert_allclose(out.v, PARAMS.v_max * diagonal, atol=1e-12)


def test_head_on_impact_stops_at_the_wall() -> None:
    """E4: the floor lets the agent travel 0.05 m, so v_y >= -0.05 / dt = -0.5."""
    out = _step(_layout("tunnel"), [[2.0, -1.95]], [[0.0, -1.2]])
    np.testing.assert_allclose(out.v_limited, [[0.0, -1.14]], atol=1e-12)
    np.testing.assert_allclose(out.v, [[0.0, -0.5]], atol=1e-12)
    np.testing.assert_allclose(out.p, [[2.0, -2.0]], atol=1e-12)
    np.testing.assert_allclose(out.dv_wall, [[0.0, 0.64]], atol=1e-12)
    assert out.contact.all() and out.wall_reaction.all() and not out.friction.any()


@pytest.mark.parametrize("gamma_w", [1.0, 0.0])
def test_sliding_along_a_wall_is_damped_by_wall_friction(gamma_w: float) -> None:
    """E5: tangential motion on the floor, and motion pushing into it, decay by (1 - gamma dt)(1 - gamma_w dt)."""
    params = PhysicsParams(dt=0.1, a_max=2.5, v_max=1.2, gamma=0.5, gamma_w=gamma_w, d_bar=1.0)
    out = _step(_layout("tunnel"), [[2.0, -2.0], [2.0, -2.0]], [[1.0, 0.0], [1.0, -0.5]], params=params)
    factor = 1.0 - gamma_w * params.dt
    np.testing.assert_allclose(out.v, [[0.95 * factor, 0.0], [0.95 * factor, 0.0]], atol=1e-12)
    np.testing.assert_allclose(out.p, [[2.0 + 0.095 * factor, -2.0]] * 2, atol=1e-12)
    assert out.friction.all() and out.contact.all()
    assert out.wall_reaction.tolist() == [False, True]


def test_no_wall_friction_when_leaving_a_wall_or_off_the_face() -> None:
    """E5: detaching from the floor; moving along the line of O1's top face, left of the obstacle."""
    out = _step(_layout("slalom"), [[2.0, -2.0], [3.9, 0.25]], [[1.0, 0.5], [-1.0, 0.0]])
    np.testing.assert_allclose(out.v, [[0.95, 0.475], [-0.95, 0.0]], atol=1e-12)
    assert not (out.friction.any() or out.wall_reaction.any() or out.contact.any())


@pytest.mark.parametrize("angle", [10.0, 30.0, 55.0, 80.0])
def test_sliding_on_a_slanted_face_goes_on_until_friction_stops_it(angle: float) -> None:
    """E4, E5: on a face that floating point cannot follow exactly, pushing along it and then coasting.

    The tangential speed obeys s+ = (1 - gamma_w dt) ((1 - gamma dt) s + dt a), with a = 2 m/s^2 for
    eight steps and 0 afterwards; the wall reaction may only collect the face itself.
    """
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    rotation = np.array([[c, -s], [s, c]])
    square = ConvexPolygon.from_vertices(np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]) @ rotation.T)
    layout = Layout.build(_box(-5.0, -5.0, 5.0, 5.0), [square], goal_x=4.0)
    tangent, normal = rotation @ np.array([1.0, 0.0]), rotation @ np.array([0.0, -1.0])
    (face,) = np.flatnonzero((layout.faces.owner == 0) & np.all(np.isclose(layout.faces.normal, normal), axis=1))
    p, v, speed = (rotation @ np.array([-0.95, -1.0]))[None, :], np.zeros((1, 2)), 0.0
    for k in range(20):
        thrust = 2.0 if k < 8 else 0.0
        out = _step(layout, p, v, (thrust * (tangent - 0.25 * normal))[None, :])  # along the face and into it
        speed = (1.0 - PARAMS.gamma_w * PARAMS.dt) * (FACTOR * speed + PARAMS.dt * thrust)
        assert np.flatnonzero(out.wall_faces[0]).tolist() in ([], [face]), f"step {k}"
        assert layout.faces.on_face(out.p)[0, face] and out.friction[0], f"step {k}"
        np.testing.assert_allclose(out.v[0], speed * tangent, atol=1e-12)
        p, v = out.p, out.v


def test_contact_flag() -> None:
    """D14: contact means a wall reaction, or the new position on a face."""
    out = _step(_layout("tunnel"), [[2.0, -2.0], [2.0, -1.5], [2.0, -1.95]], [[0.0, 0.0], [0.5, 0.0], [0.0, -1.2]])
    assert out.contact.tolist() == [True, False, True]
    assert out.wall_reaction.tolist() == [False, False, True]


def test_concave_corners_stop_the_agent() -> None:
    """E4, D17: the floor is hit first; the corrected motion then hits the second wall."""
    layout = _layout("slalom")
    floor = _face(layout, (-1, -2), (11, -2))
    left_wall = _face(layout, (-1, 2), (-1, -2))
    o1_left = _face(layout, (4, 0.25), (4, -3))
    v_hat = np.array([[-0.8, -0.8], [0.8, -0.8]])
    out = _step(layout, [[-0.97, -1.98], [3.97, -1.98]], v_hat / FACTOR)
    np.testing.assert_allclose(out.p, [[-1.0, -2.0], [4.0, -2.0]], atol=1e-12)
    np.testing.assert_allclose(out.v, [[-0.3, -0.2], [0.3, -0.2]], atol=1e-12)
    assert np.flatnonzero(out.wall_faces[0]).tolist() == sorted([floor, left_wall])
    assert np.flatnonzero(out.wall_faces[1]).tolist() == sorted([floor, o1_left])


def test_convex_corner_needs_one_face() -> None:
    """E4, D17: an obstacle is convex, so one face always suffices and the agent ends on it."""
    layout = _layout("slalom")
    top = _face(layout, (5, 0.25), (4, 0.25))
    left = _face(layout, (4, 0.25), (4, -3))
    v_hat = np.array([[0.8, -0.8], [0.8, -0.8]])
    # glancing: crosses the line x = 4 above the obstacle, then enters through the top face;
    # exactly through the vertex (4, 0.25)
    out = _step(layout, [[3.98, 0.30], [3.96, 0.29]], v_hat / FACTOR)
    assert np.flatnonzero(out.wall_faces[0]).tolist() == [top]
    np.testing.assert_allclose(out.p[0], [4.06, 0.25], atol=1e-12)
    assert out.wall_faces[1].sum() == 1 and out.wall_faces[1, [top, left]].any()
    assert layout.faces.on_face(out.p[1:])[0, [top, left]].any()


def test_at_a_vertex_the_face_needing_the_smaller_correction_goes_first() -> None:
    """D17: from O1's top-left vertex, moving right and slightly down, the agent slides on the top face."""
    layout = _layout("slalom")
    top = _face(layout, (5, 0.25), (4, 0.25))
    out = _step(layout, [[4.0, 0.25]], np.array([[0.8, -0.08]]) / FACTOR)
    assert np.flatnonzero(out.wall_faces[0]).tolist() == [top]
    np.testing.assert_allclose(out.v_wall, [[0.8, 0.0]], atol=1e-12)
    np.testing.assert_allclose(out.v, [[0.72, 0.0]], atol=1e-12)  # wall friction on the top face
    assert out.friction.all()


def test_no_phantom_contacts_near_a_vertex() -> None:
    """E4: passing above O1's corner, moving next to the extension of a face, ending exactly on the vertex."""
    v_hat = np.array([[0.8, -0.2], [-0.5, 0.3], [0.8, -0.8]])
    out = _step(_layout("slalom"), [[3.95, 0.30], [3.9, 0.2], [3.92, 0.33]], v_hat / FACTOR)
    assert not out.wall_reaction.any()
    np.testing.assert_allclose(out.v, v_hat, atol=1e-14)


def test_first_impact_ignores_faces_the_corrected_motion_misses() -> None:
    """D17: the original segment also enters B, but after sliding on A's top face the agent passes above B."""
    arena = _box(-1.0, -1.0, 3.0, 3.0)
    layout = Layout.build(arena, [_box(0.0, 0.0, 1.0, 1.0), _box(1.01, 0.9, 1.2, 0.99)], goal_x=2.5)
    out = _step(layout, [[0.95, 1.02]], np.array([[0.8, -0.48]]) / FACTOR)
    assert np.flatnonzero(out.wall_faces[0]).tolist() == [_face(layout, (1, 1), (0, 1))]
    np.testing.assert_allclose(out.v, [[0.8, -0.2]], atol=1e-12)


def test_sliding_along_the_floor_cannot_pass_under_an_obstacle() -> None:
    """D16: full thrust along the floor ends in the corner between the floor and O1."""
    layout = _layout("slalom")
    p, v = np.array([[3.0, -2.0]]), np.array([[1.0, 0.0]])
    for _ in range(100):
        out = _step(layout, p, v, [[PARAMS.a_max, 0.0]])
        p, v = out.p, out.v
        assert p[0, 0] <= 4.0 + TOL
    np.testing.assert_allclose(p, [[4.0, -2.0]], atol=1e-9)


def test_random_layouts_are_never_penetrated() -> None:
    """M6, E4: one step from random states, then long rollouts; the projection is checked against KKT."""
    rng = np.random.default_rng(SEED)
    reactions = 0
    for layout in [_layout("slalom")] + [_random_layout(rng) for _ in range(8)]:
        n = 400
        p = _free_positions(rng, layout, n)
        v, u, d = _disk(rng, n, PARAMS.v_max), rng.uniform(-5.0, 5.0, (n, 2)), _circle(rng, n, PARAMS.d_bar)
        out = _step(layout, p, v, u, d)
        assert layout.in_free_space(out.p, tol=INPUT_TOL).all()
        assert _segments_are_free(layout, p, out.p).all()
        _assert_projection(layout, p, out, PARAMS.dt)
        reactions += int(out.wall_reaction.sum())

        b = 16
        p, v = _free_positions(rng, layout, b, on_walls=0.0), np.zeros((b, 2))
        push = _circle(rng, b, 2.0 * PARAMS.a_max)
        for k in range(150):
            u = np.where(np.arange(b)[:, None] % 2 == 0, push, rng.uniform(-5.0, 5.0, (b, 2)))
            out = _step(layout, p, v, u, _disk(rng, b, PARAMS.d_bar))
            assert layout.in_free_space(out.p, tol=INPUT_TOL).all(), f"left the free space at step {k}"
            assert _segments_are_free(layout, p, out.p).all(), f"crossed a wall at step {k}"
            assert np.all(np.linalg.norm(out.v, axis=1) <= PARAMS.v_max + 1e-12)
            _assert_projection(layout, p, out, PARAMS.dt)
            reactions += int(out.wall_reaction.sum())
            p, v = out.p, out.v
    assert reactions > 1000  # the walls were exercised


def test_entry_face_is_the_last_line_crossed() -> None:
    """E4: the segment crosses the line of O1's top face left of the obstacle, then enters through its left face."""
    layout = _layout("slalom")
    left = _face(layout, (4, 0.25), (4, -3))
    out = _step(layout, [[3.95, 0.2501]], np.array([[1.0, -0.01]]) / FACTOR)
    assert np.flatnonzero(out.wall_faces[0]).tolist() == [left]
    np.testing.assert_allclose(out.v_wall, [[0.5, -0.01]], atol=1e-12)


def test_exact_ties_go_to_the_lowest_face_index() -> None:
    """D17: a segment exactly through O1's top-left vertex; dyadic numbers make the tie exact."""
    layout = _layout("slalom")
    top, left = _face(layout, (5, 0.25), (4, 0.25)), _face(layout, (4, 0.25), (4, -3))
    assert top < left
    out = _step(layout, [[3.96875, 0.28125]], np.array([[0.625, -0.625]]) / FACTOR)
    assert np.flatnonzero(out.wall_faces[0]).tolist() == [top]
    np.testing.assert_allclose(out.p, [[4.03125, 0.25]], atol=1e-12)


@pytest.mark.parametrize(("v_hat", "expected"), [((0.5, 0.0), (0.5, 0.0)), ((0.5, -0.001), (0.5, 0.0))])
def test_a_start_inside_the_tolerance_band_neither_sinks_nor_sticks(
    v_hat: tuple[float, float], expected: tuple[float, float]
) -> None:
    """E4: half a TOL inside O1's top face, sliding along it, then pushing into it."""
    layout = _layout("slalom")
    p = np.array([[4.5, 0.25 - 0.5 * TOL]])
    out = _step(layout, p, np.array([v_hat]) / FACTOR)
    np.testing.assert_allclose(out.v_wall, [expected], atol=1e-12)
    assert out.p[0, 1] >= p[0, 1] - 1e-15 and out.friction.all()


@pytest.mark.parametrize(("v_hat", "expected"), [((0.5, 0.0), (0.5, 0.0)), ((0.5, -0.001), (0.5, 0.0))])
def test_a_start_just_outside_the_arena_neither_sinks_nor_sticks(
    v_hat: tuple[float, float], expected: tuple[float, float]
) -> None:
    """E4: 1.5 TOL below the floor (an accepted input), sliding along it, then pushing out of the arena."""
    p = np.array([[2.0, -2.0 - 1.5 * TOL]])
    out = _step(_layout("tunnel"), p, np.array([v_hat]) / FACTOR)
    np.testing.assert_allclose(out.v_wall, [expected], atol=1e-12)
    assert out.wall_reaction.tolist() == [v_hat[1] < 0.0]
    assert out.p[0, 1] >= p[0, 1] - 1e-15


def test_every_output_is_a_valid_input_even_just_beyond_the_tolerance() -> None:
    """E4: rounding can end a step a little more than TOL inside O2; the next step accepts it and copes."""
    params = PhysicsParams(dt=0.1, a_max=2.5, v_max=1.2, gamma=0.5, gamma_w=1.0, d_bar=0.0)
    layout = _layout("slalom")
    first = _step(layout, [[4.5, 1.749999998]], [[0.0, 3.1578946773011966e-08]], params=params)
    assert TOL < first.p[0, 1] - 1.75 < INPUT_TOL  # the case this test is about
    deeper = _step(layout, first.p, first.v, params=params)  # still moving into O2
    assert deeper.p[0, 1] <= first.p[0, 1] + 1e-15
    along = _step(layout, first.p, [[1.0, 0.0]], params=params)  # moving along O2's bottom face
    assert not along.wall_reaction.any()
    np.testing.assert_allclose(along.v_wall, [[0.95, 0.0]], atol=1e-12)


@pytest.mark.parametrize(("overshoot", "reacts"), [(0.5 * TOL, False), (2.0 * TOL, True)])
def test_ending_within_tol_beyond_a_wall_is_no_impact(overshoot: float, reacts: bool) -> None:
    """E4: the tolerance applies to the arena walls too."""
    out = _step(_layout("tunnel"), [[2.0, -1.95]], [[0.0, -(0.05 + overshoot) / PARAMS.dt / FACTOR]])
    assert out.wall_reaction.tolist() == [reacts]


def test_no_wall_friction_when_leaving_a_face_at_its_vertex() -> None:
    """E5: from O1's top-left vertex straight up or to the left; from its top-right vertex to the right."""
    v_hat = np.array([[0.0, 0.8], [-0.8, 0.0], [0.8, 0.0]])
    out = _step(_layout("slalom"), [[4.0, 0.25], [4.0, 0.25], [5.0, 0.25]], v_hat / FACTOR)
    assert not (out.friction.any() or out.wall_reaction.any())
    np.testing.assert_allclose(out.v, v_hat, atol=1e-12)


def test_huge_commands_are_saturated_without_overflow() -> None:
    """E1: any finite command, however large, is saturated onto the boundary of U."""
    commands = [[1e155, 0.0], [-1e300, 1e300], [1.7e308, 1.7e308]]  # the last norm overflows
    out = _step(_layout("tunnel"), [[2.0, 0.0]] * 3, np.zeros((3, 2)), commands)
    r = 2.5 / np.sqrt(2.0)
    np.testing.assert_allclose(out.u, [[2.5, 0.0], [-r, r], [r, r]], atol=1e-12)


def test_batch_equals_single_agents_bit_for_bit() -> None:
    """E7: the scalar environment is a batch of one; a rollout gives exactly the same states either way."""
    layout = _layout("slalom")
    rng = np.random.default_rng(SEED)
    n, steps = 32, 40
    p0 = _free_positions(rng, layout, n)
    u = rng.uniform(-5.0, 5.0, (steps, n, 2))
    d = np.stack([_disk(rng, n, PARAMS.d_bar) for _ in range(steps)])
    batch, p, v = [], p0, np.zeros((n, 2))
    for k in range(steps):
        batch.append(_step(layout, p, v, u[k], d[k]))
        p, v = batch[-1].p, batch[-1].v
    assert sum(int(out.wall_reaction.sum()) for out in batch) > 0
    for i in range(n):
        p, v = p0[i : i + 1], np.zeros((1, 2))
        for k in range(steps):
            single = _step(layout, p, v, u[k, i : i + 1], d[k, i : i + 1])
            for name in ("p", "v", "v_wall", "wall_faces", "friction", "contact"):
                np.testing.assert_array_equal(getattr(single, name), getattr(batch[k], name)[i : i + 1])
            p, v = single.p, single.v
