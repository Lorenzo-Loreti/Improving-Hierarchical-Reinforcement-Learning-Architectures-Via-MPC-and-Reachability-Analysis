"""Tests of the shortest free paths to the goal region (decision log D16, F7)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from hrlmpc.config import Box, load_env_config
from hrlmpc.geometry import ConvexPolygon, Layout
from hrlmpc.paths import free_path_distance, segment_is_free

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "env"


def _layout(name: str) -> Layout:
    return Layout.from_config(load_env_config(CONFIGS / f"{name}.yaml").geometry)


def _box_layout(arena: Box, obstacles: list[Box], goal_x: float) -> Layout:
    return Layout.build(ConvexPolygon.from_box(arena), [ConvexPolygon.from_box(b) for b in obstacles], goal_x)


def test_without_obstacles_the_path_is_the_straight_run_to_the_goal_line() -> None:
    layout = _layout("tunnel")
    points = np.random.default_rng(0).uniform([-1.0, -2.0], [9.99, 2.0], (200, 2))
    np.testing.assert_array_equal(free_path_distance(layout, points), layout.goal_x - points[:, 0])


def test_a_single_obstacle_is_rounded_at_its_corners() -> None:
    layout = _box_layout(Box(x=(0.0, 10.0), y=(-5.0, 5.0)), [Box(x=(4.0, 6.0), y=(-1.0, 1.0))], goal_x=9.0)
    assert free_path_distance(layout, [[2.0, 0.0]])[0] == pytest.approx(math.sqrt(5.0) + 2.0 + 3.0, rel=1e-12)
    assert free_path_distance(layout, [[2.0, 1.0]])[0] == pytest.approx(7.0, rel=1e-12)  # along the top face
    assert free_path_distance(layout, [[7.0, 0.0]])[0] == pytest.approx(2.0, rel=1e-12)  # past the obstacle


def test_the_slalom_paths_wind_through_both_gates() -> None:
    """Taut strings over O_1 (top corners at y = 0.25) and under O_4 (bottom corners at y = -0.25)."""
    layout = _layout("slalom")
    gate_to_gate = math.hypot(2.0, 0.5)
    expected = {
        (2.0, -1.0): math.hypot(2.0, 1.25) + 1.0 + gate_to_gate + 1.0 + 2.0,
        (0.0, 1.0): math.hypot(5.0, 0.75) + gate_to_gate + 1.0 + 2.0,
        (6.0, 0.5): math.hypot(1.0, 0.75) + 1.0 + 2.0,  # between the gates, level with O_4
        (8.5, 1.5): 1.5,  # past both gates
    }
    distances = free_path_distance(layout, list(expected))
    np.testing.assert_allclose(distances, list(expected.values()), rtol=1e-12)
    straight = layout.goal_x - np.array(list(expected))[:, 0]
    assert np.all(distances >= straight)


def test_paths_never_pass_between_an_obstacle_and_the_wall_it_touches() -> None:
    """D16: O_1 touches the floor, so the floor is no gap under it."""
    layout = _layout("slalom")
    assert not segment_is_free(layout, [3.0, -2.0], [6.0, -2.0])
    over = math.hypot(1.0, 2.25) + 1.0 + math.hypot(2.0, 0.5) + 1.0 + 2.0
    assert free_path_distance(layout, [[3.0, -2.0]])[0] == pytest.approx(over, rel=1e-12)


@pytest.mark.parametrize(
    ("start", "end", "free"),
    [
        ((3.0, 0.25), (6.0, 0.25), True),  # along the top face of O_1
        ((3.0, -0.75), (5.0, 1.25), True),  # through the corner (4, 0.25) only
        ((3.0, -0.75), (5.0, 1.0), False),  # cuts the corner
        ((4.5, 0.3), (4.5, 1.7), True),  # across gate 1
        ((4.5, 0.3), (4.5, -0.5), False),  # into O_1
        ((6.0, 0.0), (9.0, 1.0), False),  # through O_4
        ((4.0, 0.25), (4.0, 0.25), True),  # a single point on the boundary
        ((4.5, 0.0), (4.5, 0.0), False),  # a single point inside
    ],
)
def test_segments_may_touch_the_obstacles_but_not_enter_them(
    start: tuple[float, float], end: tuple[float, float], free: bool
) -> None:
    assert segment_is_free(_layout("slalom"), start, end) is free


def test_an_unreachable_goal_and_bad_inputs_are_rejected() -> None:
    wall = _box_layout(Box(x=(0.0, 10.0), y=(-2.0, 2.0)), [Box(x=(4.0, 5.0), y=(-2.0, 2.0))], goal_x=9.0)
    with pytest.raises(ValueError, match="cannot be reached"):
        free_path_distance(wall, [[1.0, 0.0]])
    assert free_path_distance(wall, [[6.0, 0.0]])[0] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="free space"):
        free_path_distance(_layout("slalom"), [[4.5, -1.0]])
    with pytest.raises(ValueError, match="shape"):
        free_path_distance(_layout("slalom"), np.zeros((2, 3)))
    triangle = Layout.build(ConvexPolygon.from_vertices([[-1.0, -2.0], [11.0, -2.0], [5.0, 6.0]]), [], 5.0)
    with pytest.raises(ValueError, match="rectangular"):
        free_path_distance(triangle, [[0.0, -1.0]])


def test_the_distance_is_bounded_by_a_dense_sampled_path() -> None:
    """A path over a fine grid of free points is never shorter than the exact shortest path."""
    layout = _layout("slalom")
    xs, ys = np.linspace(-1.0, 10.0, 45), np.linspace(-2.0, 2.0, 33)
    grid = np.array([[x, y] for x in xs for y in ys])
    grid = grid[layout.in_free_space(grid)]
    start = np.array([1.0, -0.5])
    # Dijkstra on the 8-neighbourhood graph of the grid, with straight segments checked for freedom.
    nodes = np.vstack([start, grid])
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    best = np.full(len(nodes), np.inf)
    best[0] = 0.0
    done = np.zeros(len(nodes), dtype=bool)
    for _ in range(len(nodes)):
        i = int(np.argmin(np.where(done, np.inf, best)))
        if done[i] or not np.isfinite(best[i]):
            break
        done[i] = True
        offset = np.abs(nodes - nodes[i])
        near = np.flatnonzero((offset[:, 0] <= dx * 1.01) & (offset[:, 1] <= dy * 1.01))
        for j in near:
            if not done[j] and segment_is_free(layout, nodes[i], nodes[j]):
                best[j] = min(best[j], best[i] + float(np.hypot(*(nodes[j] - nodes[i]))))
    reached = nodes[:, 0] >= layout.goal_x - 1e-12
    sampled = float(best[reached].min())
    exact = free_path_distance(layout, [start])[0]
    assert exact <= sampled + 1e-12
    assert sampled < 1.08 * exact  # the grid path is close, so the exact one is not too short either
