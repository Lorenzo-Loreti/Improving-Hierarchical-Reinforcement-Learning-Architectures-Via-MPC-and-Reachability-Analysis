"""Tests of the targets of PPO_MPC_reachability without solvers (decision log D38-D40, D47)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from hrlmpc.action_map import disk_to_square
from hrlmpc.mpc_command import CONVERGED, ROUNDS, UNSOLVED, SearchReport
from hrlmpc.reachability import (
    AREA_TOL,
    MOVED,
    REACH_KEYS,
    REACHED_MARGIN,
    ReachableTargets,
    convex_hull,
    polygon_area,
    polygon_centroid,
    radial_extent,
    radial_map,
    reach_metrics,
    reached_metrics,
    support_directions,
)

SQUARE = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])


def test_the_support_directions_are_unit_vectors_at_equal_angles() -> None:
    d = support_directions(16)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0)
    np.testing.assert_allclose(d[0], [1.0, 0.0])
    np.testing.assert_allclose(d[4], [0.0, 1.0], atol=1e-15)
    for bad in (2, 0, True, 3.0):
        with pytest.raises(ValueError, match="3 directions"):
            support_directions(bad)  # type: ignore[arg-type]


def test_the_hull_is_convex_counter_clockwise_and_drops_inner_repeated_and_collinear_points() -> None:
    rng = np.random.default_rng(0)
    inner = rng.uniform(-0.9, 0.9, size=(50, 2))
    edges = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 1.0]])  # on the edges
    points = np.vstack([SQUARE, SQUARE + 1e-12, inner, edges])
    hull = convex_hull(rng.permutation(points))
    assert len(hull) == 4
    assert {tuple(np.round(v, 9)) for v in hull} == {tuple(v) for v in SQUARE}
    assert polygon_area(hull) == pytest.approx(4.0)  # counter-clockwise: positive area
    assert len(convex_hull([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])) == 2  # a segment
    assert len(convex_hull([[3.0, 1.0]] * 3)) == 1
    with pytest.raises(ValueError, match="finite"):
        convex_hull([[0.0, np.nan], [1.0, 1.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match="shape"):
        convex_hull(np.zeros((3, 3)))


def test_area_and_centroid_of_polygons() -> None:
    np.testing.assert_allclose(polygon_centroid(SQUARE + [2.0, -1.0]), [2.0, -1.0], atol=1e-15)
    tiny = 1e-3 * SQUARE + [10.7, -1.3]  # small and far from the origin: computed relative to a vertex
    np.testing.assert_allclose(polygon_centroid(tiny), [10.7, -1.3], atol=1e-15)
    triangle = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    assert polygon_area(triangle) == pytest.approx(4.5)
    np.testing.assert_allclose(polygon_centroid(triangle), [1.0, 1.0])
    assert polygon_area(triangle[:2]) == 0.0
    with pytest.raises(ValueError, match="area"):
        polygon_centroid(triangle[:2])


def test_the_radial_extent_reaches_the_boundary() -> None:
    assert radial_extent(SQUARE, [0.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    diagonal = np.array([1.0, 1.0]) / math.sqrt(2.0)
    assert radial_extent(SQUARE, [0.0, 0.0], diagonal) == pytest.approx(math.sqrt(2.0))
    assert radial_extent(SQUARE, [0.5, 0.0], [1.0, 0.0]) == pytest.approx(0.5)
    assert radial_extent(SQUARE, [0.5, 0.0], [-1.0, 0.0]) == pytest.approx(1.5)
    assert radial_extent(SQUARE, [1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)  # on the boundary, outward
    with pytest.raises(ValueError, match="outside"):
        radial_extent(SQUARE, [2.0, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="three"):
        radial_extent(SQUARE[:2], [0.0, 0.0], [1.0, 0.0])


def test_the_radial_map_is_a_bijection_of_the_disk_onto_the_polygon() -> None:
    """Rays from the centre are kept and stretched: the circle goes onto the boundary, 0 onto the centre, and
    distinct points stay distinct; every point of the polygon is the image of the point that undoes the scaling."""
    rng = np.random.default_rng(1)
    hexagon = np.array([[2.0, 0.0], [1.0, 1.5], [-1.0, 1.2], [-2.0, 0.0], [-1.0, -1.0], [1.5, -1.3]])
    hull = convex_hull(hexagon)
    c = polygon_centroid(hull)
    np.testing.assert_array_equal(radial_map([0.0, 0.0], hull, c), c)
    angles = rng.uniform(0.0, 2.0 * np.pi, 200)
    for a in angles:
        u = np.array([np.cos(a), np.sin(a)])
        edge = radial_map(u, hull, c)
        assert radial_extent(hull, c, u) == pytest.approx(np.linalg.norm(edge - c))  # on the boundary
        for r in (0.25, 0.5, 0.99):
            inside = radial_map(r * u, hull, c)
            np.testing.assert_allclose(inside, c + r * (edge - c), atol=1e-12)  # linear along the ray
    # the inverse: any point of the polygon comes from exactly one point of the disk
    for _ in range(200):
        w = rng.dirichlet(np.ones(len(hull)))
        point = w @ hull
        d = point - c
        dist = float(np.linalg.norm(d))
        if dist < 1e-12:
            continue
        m = (dist / radial_extent(hull, c, d / dist)) * d / dist
        assert np.linalg.norm(m) <= 1.0 + 1e-12
        np.testing.assert_allclose(radial_map(m, hull, c), point, atol=1e-12)


class _Disk:
    """Stand-in support points: the disk of radius ``r`` around the position shifted by ``drift`` times the velocity."""

    def __init__(self, radius: float = 1.0, drift: float = 0.5, fail: tuple[int, ...] = ()) -> None:
        self.radius, self.drift, self.fail = radius, drift, fail
        self.calls: list[np.ndarray] = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.calls.append(x.copy())
        points = x[:2] + self.drift * x[2:] + self.radius * support_directions(16)
        points[list(self.fail)] = np.nan
        return points


def _targets(support: Any = None, project: Any = None) -> ReachableTargets:
    return ReachableTargets(support=_Disk() if support is None else support,
                            project=(lambda x, q: (q.copy(), SearchReport())) if project is None else project,
                            radius=1.2)


def test_the_targets_map_the_square_onto_the_hull_about_its_centroid() -> None:
    """D38: x -> m(x) of D15 -> the radial map onto the hull of the support points; the identity as projection."""
    targets = _targets()
    p = np.array([[1.0, 0.5], [3.0, -1.0], [0.0, 0.0]])
    v = np.array([[0.4, 0.0], [0.0, -0.6], [0.0, 0.0]])
    centre = p + 0.5 * v
    between = disk_to_square([[math.cos(math.pi / 16), math.sin(math.pi / 16)]], 1.0)[0]  # half-way between vertices
    actions = np.array([[0.0, 0.0], [-1.0, 1.0], between])  # the centre, a vertex (135 degrees), between vertices
    g = targets(actions, p, v)
    np.testing.assert_allclose(g[0], centre[0], atol=1e-12)
    np.testing.assert_allclose(g[1], centre[1] + np.array([-1.0, 1.0]) / np.sqrt(2.0), atol=1e-12)  # on the circle
    assert np.linalg.norm(g[2] - centre[2]) == pytest.approx(math.cos(math.pi / 16), rel=1e-9)  # F5: the inner loss
    # every target lies in the hull, and the map is one-to-one on a grid of actions
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21)), axis=-1).reshape(-1, 2)
    many = targets(grid, np.zeros_like(grid), np.zeros_like(grid))
    assert np.all(np.linalg.norm(many, axis=1) <= 1.0 + 1e-12)
    assert len(np.unique(np.round(many, 9), axis=0)) == len(grid)
    # the action of a target is recovered through the inverse of D15
    m = many / np.array([radial_extent(convex_hull(support_directions(16)), [0.0, 0.0], t / max(np.linalg.norm(t),
                                                                                             1e-300))
                         if np.linalg.norm(t) > 0 else 1.0 for t in many])[:, None]
    np.testing.assert_allclose(disk_to_square(m, 1.0), grid, atol=1e-9)


def test_the_projection_gives_the_target_and_its_distance_is_recorded() -> None:
    def project(x: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, SearchReport]:
        # a wall at x = 0.5, a solver failure survived, an inaccurate exact solve and an uncertified exact plan on
        # the way (D47)
        report = SearchReport(errors=1, exit=CONVERGED, exact_inaccurate=1, exact_uncertified=1)
        return np.array([min(q[0], 0.5), q[1]]), report

    targets = _targets(project=project)
    g = targets(np.array([[1.0, 0.0], [-1.0, 0.0]]), np.zeros((2, 2)), np.zeros((2, 2)))
    np.testing.assert_allclose(g, [[0.5, 0.0], [-1.0, 0.0]], atol=1e-12)
    stats = targets.pop_stats()
    assert set(stats) == set(REACH_KEYS) and all(values.shape == (2,) for values in stats.values())
    np.testing.assert_allclose(stats["projection"], [0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(stats["hull_area"], 8 * math.sin(math.pi / 8))  # the 16-gon of radius 1
    np.testing.assert_array_equal(stats["projection_failed"], [0.0, 0.0])
    np.testing.assert_array_equal(stats["projection_errors"], [1.0, 1.0])
    np.testing.assert_array_equal(stats["projection_exit"], [CONVERGED, CONVERGED])
    np.testing.assert_array_equal(stats["projection_exact_inaccurate"], [1.0, 1.0])
    np.testing.assert_array_equal(stats["projection_exact_infeasible"], [0.0, 0.0])
    np.testing.assert_array_equal(stats["projection_exact_uncertified"], [1.0, 1.0])
    np.testing.assert_array_equal(stats["hull_fallback"], [0.0, 0.0])
    assert all(values.size == 0 for values in targets.pop_stats().values())


def test_failures_are_counted_and_fall_back() -> None:
    """A failed projection keeps the point of the hull; failed directions are skipped; with no direction left
    the position is held, and with too few for an area the points' mean is taken."""
    targets = _targets(support=_Disk(fail=(3, 7)),
                       project=lambda x, q: (None, SearchReport(errors=2, exit=UNSOLVED, exact_infeasible=1)))
    g = targets(np.array([[1.0, 0.0]]), np.zeros((1, 2)), np.zeros((1, 2)))
    left = np.delete(support_directions(16), [3, 7], axis=0)
    hull = convex_hull(left)
    np.testing.assert_allclose(g[0], radial_map([1.0, 0.0], hull, polygon_centroid(hull)), atol=1e-12)
    stats = targets.pop_stats()
    assert stats["support_failures"][0] == 2.0 and stats["projection_failed"][0] == 1.0
    assert stats["projection"][0] == 0.0 and stats["projection_errors"][0] == 2.0 and stats["hull_fallback"][0] == 0.0
    assert stats["projection_exit"][0] == UNSOLVED and stats["projection_exact_infeasible"][0] == 1.0
    nothing = _targets(support=_Disk(fail=tuple(range(16))))
    np.testing.assert_allclose(nothing(np.array([[1.0, 1.0]]), np.array([[2.0, 3.0]]), np.zeros((1, 2))), [[2.0, 3.0]])
    stats = nothing.pop_stats()
    assert stats["support_failures"][0] == 16.0 and stats["hull_fallback"][0] == 1.0
    line = _targets(support=lambda x: np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    np.testing.assert_allclose(line(np.array([[1.0, 0.0]]), np.zeros((1, 2)), np.zeros((1, 2))), [[1.0, 1.0]])
    assert line.pop_stats()["hull_fallback"][0] == 1.0
    # a sliver of area 1e-7 m^2 at arena coordinates is degenerate: the mean, without a centroid to round outside
    sliver = np.array([[7.0, 1.0], [7.5, 1.0 + 2e-7], [8.0, 1.0], [7.5, 1.0 + 4e-7]])
    assert 0.0 < polygon_area(convex_hull(sliver)) < AREA_TOL
    thin = _targets(support=lambda x: sliver)
    np.testing.assert_allclose(thin(np.array([[1.0, 0.0]]), np.zeros((1, 2)), np.zeros((1, 2)))[0], sliver.mean(axis=0))
    assert thin.pop_stats()["hull_fallback"][0] == 1.0


def test_invalid_inputs_are_refused_and_close_releases_once() -> None:
    released: list[int] = []
    targets = ReachableTargets(support=_Disk(), project=lambda x, q: (q, SearchReport()), radius=1.2,
                               release=lambda: released.append(1))
    with pytest.raises(ValueError, match="square"):
        targets(np.array([[1.5, 0.0]]), np.zeros((1, 2)), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="shape"):
        targets(np.zeros((2, 2)), np.zeros((3, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        targets(np.zeros((1, 2)), np.array([[np.nan, 0.0]]), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="support points"):
        _targets(support=lambda x: np.zeros(16))(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="radius"):
        ReachableTargets(support=_Disk(), project=lambda x, q: (q, SearchReport()), radius=0.0)
    targets.close()
    targets.close()
    assert released == [1]


def test_the_metrics_of_the_decisions() -> None:
    stats = {
        "support_ms": np.array([2.0, 4.0, 6.0]),
        "project_ms": np.array([1.0, 1.0, 4.0]),
        "hull_area": np.array([1.0, 2.0, 3.0]),
        "projection": np.array([0.0, 0.5, 5e-3]),  # 5 mm: the effort's trade-off, no obstacle
        "support_failures": np.array([0.0, 2.0, 1.0]),
        "hull_fallback": np.array([0.0, 0.0, 1.0]),
        "projection_failed": np.array([1.0, 0.0, 0.0]),
        "projection_errors": np.array([0.0, 3.0, 0.0]),
        "projection_exit": np.array([ROUNDS, CONVERGED, CONVERGED], dtype=np.float64),
        "projection_exact_inaccurate": np.array([2.0, 0.0, 1.0]),
        "projection_exact_infeasible": np.array([0.0, 1.0, 0.0]),
        "projection_exact_uncertified": np.array([4.0, 0.0, 1.0]),
    }
    assert MOVED == 1e-2
    assert reach_metrics(stats, "reach") == pytest.approx({
        "reach/support_ms_mean": 4.0, "reach/project_ms_mean": 2.0, "reach/hull_area_mean": 2.0,
        "reach/projection_mean": (0.5 + 5e-3) / 3, "reach/projection_max": 0.5, "reach/projected_rate": 1 / 3,
        "reach/support_failure_count": 3.0, "reach/hull_fallback_count": 1.0, "reach/projection_failure_count": 1.0,
        "reach/projection_error_count": 3.0, "reach/decisions": 3.0,
        "reach/projection_exit_converged_count": 2.0, "reach/projection_exit_relaxation_count": 0.0,
        "reach/projection_exit_fixed_count": 0.0, "reach/projection_exit_rounds_count": 1.0,
        "reach/projection_exit_empty_count": 0.0, "reach/projection_exit_unsolved_count": 0.0,
        "reach/projection_exit_error_count": 0.0, "reach/projection_exact_inaccurate_count": 3.0,
        "reach/projection_exact_infeasible_count": 1.0, "reach/projection_exact_uncertified_count": 5.0,
    })
    masked = reach_metrics(stats, "eval_reach", np.array([True, False, True]))
    assert masked["eval_reach/decisions"] == 2.0 and masked["eval_reach/projected_rate"] == 0.0
    assert masked["eval_reach/projection_exit_rounds_count"] == 1.0
    assert masked["eval_reach/projection_exact_infeasible_count"] == 0.0
    assert masked["eval_reach/projection_exact_uncertified_count"] == 5.0
    assert reach_metrics(stats, "eval_reach", np.zeros(3, dtype=bool)) == {}


def test_the_reached_metrics_of_the_segments() -> None:
    """D40: reached within r_p + 0.1 m at the end of a full segment."""
    assert REACHED_MARGIN == 0.1
    metrics = reached_metrics([0.0, 0.1, 0.16, 0.5], 0.15, "hierarchy")
    assert metrics == pytest.approx({"hierarchy/reached_rate": 0.5, "hierarchy/end_distance_mean": 0.19,
                                     "hierarchy/end_distance_max": 0.5, "hierarchy/full_segments": 4.0})
    assert reached_metrics([], 0.15, "hierarchy") == {}
