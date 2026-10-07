"""Tests of the polygonal geometry (row M6; decision log D11, D16)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hrlmpc.config import Box, load_env_config
from hrlmpc.geometry import EXTENSION, MIN_GAP, TOL, ConvexPolygon, Layout

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "env"


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


def test_box_polygon_lists_vertices_counter_clockwise_with_outward_normals() -> None:
    poly = _box(0.0, 0.0, 2.0, 1.0)
    np.testing.assert_allclose(poly.vertices, [[0, 0], [2, 0], [2, 1], [0, 1]])
    np.testing.assert_allclose(poly.normals, [[0, -1], [1, 0], [0, 1], [-1, 0]], atol=1e-15)
    np.testing.assert_allclose(poly.offsets, [0, 2, 1, 0], atol=1e-15)
    points = np.array([[1.0, 0.5], [2.5, 0.5], [2.0, 0.3]])  # interior, outside, on a face
    assert poly.contains(points).tolist() == [True, False, True]
    assert poly.contains_interior(points).tolist() == [True, False, False]


def test_clockwise_vertices_give_the_same_polygon() -> None:
    clockwise = ConvexPolygon.from_vertices([[0, 0], [0, 1], [1, 1], [1, 0]])
    counter_clockwise = _box(0, 0, 1, 1)
    def rows(polygon: ConvexPolygon) -> set[tuple[float, ...]]:
        return {tuple(np.round(np.append(n, b), 12)) for n, b in zip(polygon.normals, polygon.offsets)}

    assert rows(clockwise) == rows(counter_clockwise)


@pytest.mark.parametrize(
    ("vertices", "message"),
    [
        ([[0, 0], [1, 0]], "at least 3"),
        ([[0, 0], [1, 0], [np.nan, 1]], "finite"),
        ([[0, 0], [1, 0], [1, 0], [0, 1]], "repeated"),
        ([[0, 0], [1, 0], [2, 0]], "convex"),
        ([[0, 0], [2, 0], [1, 0.2], [2, 1], [0, 1]], "convex"),
        # a pentagram turns left at every vertex but winds twice
        ([[np.cos(a), np.sin(a)] for a in np.deg2rad([0, 144, 288, 72, 216])], "convex"),
    ],
)
def test_invalid_polygons_are_rejected(vertices: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ConvexPolygon.from_vertices(vertices)


def test_pushing_a_face_out_moves_only_its_line() -> None:
    triangle = ConvexPolygon.from_vertices([[0, 0], [2, 0], [1, 1]])
    pushed = triangle.pushed_out([0.5, 0.0, 0.0])
    np.testing.assert_allclose(pushed.normals, triangle.normals, atol=1e-15)
    np.testing.assert_allclose(pushed.offsets, triangle.offsets + np.array([0.5, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(pushed.vertices, [[-0.5, -0.5], [2.5, -0.5], [1, 1]], atol=1e-12)


def test_room_and_width() -> None:
    trapezoid = ConvexPolygon.from_vertices([[4, 0], [5, 0], [6, 1], [3, 1]])  # widens away from its base
    assert trapezoid.room(0) == pytest.approx(0.5)  # the slanted sides meet 0.5 below the base
    assert trapezoid.room(2) == np.inf
    assert _box(0, 0, 3, 1).room(0) == np.inf
    assert _box(0, 0, 3, 1).width() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="convex"):
        trapezoid.pushed_out([0.6, 0.0, 0.0, 0.0])


def test_an_obstacle_widening_away_from_the_wall_is_extended_by_half_its_room() -> None:
    """D16 for a general convex obstacle: its base on the floor, its sides meeting 0.5 m below the floor."""
    trapezoid = ConvexPolygon.from_vertices([[4, 0], [5, 0], [6, 1], [3, 1]])
    layout = Layout.build(_box(0, 0, 10, 4), [trapezoid], goal_x=9.0)
    np.testing.assert_allclose(layout.blocked[0].vertices, [[4.25, -0.25], [4.75, -0.25], [6, 1], [3, 1]], atol=1e-12)
    points = np.random.default_rng(0).uniform([0, 0], [10, 4], size=(20000, 2))
    np.testing.assert_array_equal(trapezoid.contains_interior(points), layout.blocked[0].contains_interior(points))
    assert not layout.in_free_space(np.array([[4.5, 0.0]])).any()


def test_slalom_layout_matches_the_configuration() -> None:
    layout = _layout("slalom")
    np.testing.assert_allclose(layout.arena.vertices, [[-1, -2], [11, -2], [11, 2], [-1, 2]])
    np.testing.assert_allclose(layout.obstacles[0].vertices, [[4, -2], [5, -2], [5, 0.25], [4, 0.25]])
    assert len(layout.obstacles) == len(layout.blocked) == 4
    assert layout.goal_x == 10.0
    assert layout.faces.normal.shape == (20, 2)
    np.testing.assert_allclose(np.linalg.norm(layout.faces.normal, axis=1), 1.0)


def test_tunnel_layout_has_only_the_arena() -> None:
    layout = _layout("tunnel")
    assert layout.obstacles == () and layout.blocked == ()
    assert layout.faces.normal.shape == (4, 2)


def test_obstacles_touching_a_wall_are_extended_beyond_it() -> None:
    """D16: O1 and O3 rest on the floor, O2 and O4 hang from the ceiling."""
    layout = _layout("slalom")
    below, above = -2.0 - EXTENSION, 2.0 + EXTENSION
    np.testing.assert_allclose(layout.blocked[0].vertices, [[4, below], [5, below], [5, 0.25], [4, 0.25]])
    np.testing.assert_allclose(layout.blocked[1].vertices, [[4, 1.75], [5, 1.75], [5, above], [4, above]])
    np.testing.assert_allclose(layout.blocked[2].vertices, [[7, below], [8, below], [8, -1.75], [7, -1.75]])
    np.testing.assert_allclose(layout.blocked[3].vertices, [[7, -0.25], [8, -0.25], [8, above], [7, above]])


def test_extension_changes_nothing_inside_the_arena() -> None:
    layout = _layout("slalom")
    points = np.random.default_rng(0).uniform([-1, -2], [11, 2], size=(20000, 2))
    for original, blocked in zip(layout.obstacles, layout.blocked):
        np.testing.assert_array_equal(original.contains_interior(points), blocked.contains_interior(points))


def test_free_space_has_no_zero_width_gaps() -> None:
    """D16: the walls under and over the obstacles are not free."""
    layout = _layout("slalom")
    gaps = np.array([[4.5, -2.0], [7.5, -2.0], [4.5, 2.0], [7.5, 2.0], [4.0 + 1e-6, -2.0]])
    assert not layout.in_free_space(gaps).any()
    walls = np.array([[3.9, -2.0], [4.0, -2.0], [4.0, -1.0], [4.5, 0.25], [4.5, 1.0], [-1.0, 0.0], [11.0, 2.0]])
    assert layout.in_free_space(walls).all()


def test_faces_point_into_the_free_space() -> None:
    layout = _layout("slalom")
    faces = layout.faces
    fractions = np.array([0.05, 0.37, 0.95])
    points = (faces.start[:, None, :] + fractions[None, :, None] * (faces.end - faces.start)[:, None, :]).reshape(-1, 2)
    normals = np.repeat(faces.normal, len(fractions), axis=0)
    reachable = layout.in_free_space(points)  # faces of extended obstacles also run outside the arena
    assert reachable.reshape(-1, len(fractions)).any(axis=1).sum() == 16  # all but the 4 faces beyond the walls
    assert layout.in_free_space((points + 1e-3 * normals)[reachable]).all()
    assert not layout.in_free_space((points - 1e-3 * normals)[reachable]).any()


def test_contact_needs_the_point_on_the_face_not_on_its_line() -> None:
    layout = _layout("slalom")
    top = _face(layout, (5, 0.25), (4, 0.25))
    points = np.array([[4.5, 0.25], [4.0, 0.25], [3.9, 0.25], [4.5, 0.25 + 2 * TOL]])
    assert layout.faces.on_face(points)[:, top].tolist() == [True, True, False, False]


def test_signed_distances_are_positive_on_the_free_side() -> None:
    layout = _layout("slalom")
    top = _face(layout, (5, 0.25), (4, 0.25))
    floor = _face(layout, (-1, -2), (11, -2))
    distances = layout.faces.signed_distances(np.array([[4.5, 0.75], [4.5, 0.0]]))
    np.testing.assert_allclose(distances[:, top], [0.5, -0.25])
    np.testing.assert_allclose(distances[:, floor], [2.75, 2.0])


def test_goal_region() -> None:
    layout = _layout("slalom")
    assert layout.in_goal(np.array([[10.0, 0.0], [10.5, 1.9], [9.99, 0.0]])).tolist() == [True, True, False]


def test_overlapping_obstacles_are_allowed() -> None:
    arena = _box(0, 0, 10, 4)
    layout = Layout.build(arena, [_box(2, 1, 3.5, 2), _box(3, 1, 4, 3)], goal_x=9.0)
    assert not layout.in_free_space(np.array([[3.2, 1.5], [3.7, 2.5]])).any()
    assert layout.in_free_space(np.array([[2.5, 2.5], [4.0, 2.0], [2.0, 1.5]])).all()


@pytest.mark.parametrize(
    ("obstacles", "goal_x", "message"),
    [
        ([_box(2, 1, 3, 2), _box(3, 1, 4, 2)], 9.0, "touch"),  # shared face
        ([_box(2, 1, 3, 2), _box(3, 2, 4, 3)], 9.0, "touch"),  # shared vertex
        ([ConvexPolygon.from_vertices([[5, 0], [6, 1], [5, 2], [4, 1]])], 9.0, "vertex"),  # tip on the floor
        ([_box(9, 1, 11, 2)], 8.5, "inside the arena"),
        ([], 10.0, "goal"),
        ([_box(2, 1, 3, 1 + 0.5 * MIN_GAP)], 9.0, "thinner"),
        ([_box(2, 1.5e-9, 3, 1)], 9.0, "closer than MIN_GAP"),  # a passage 1.5 nm high under the obstacle
        ([_box(2, 1, 3, 2), _box(3 + 0.5 * MIN_GAP, 1, 4, 2)], 9.0, "closer than MIN_GAP"),
    ],
)
def test_degenerate_layouts_are_rejected(obstacles: list[ConvexPolygon], goal_x: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Layout.build(_box(0, 0, 10, 4), obstacles, goal_x=goal_x)
