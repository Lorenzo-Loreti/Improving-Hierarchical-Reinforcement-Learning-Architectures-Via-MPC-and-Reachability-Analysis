"""Tests of the action map of decision log D15 (row AM1)."""

from __future__ import annotations

import numpy as np
import pytest

from hrlmpc.action_map import disk_to_square, square_to_disk

A_MAX = 2.5
SEED = 0


def test_corners_and_edges_of_the_square_reach_the_boundary_of_the_disk() -> None:
    x = np.array([[1.0, 1.0], [-1.0, 1.0], [1.0, 0.0], [0.0, -1.0], [1.0, -0.5], [0.0, 0.0]])
    u = square_to_disk(x, A_MAX)
    r = A_MAX / np.sqrt(2.0)
    np.testing.assert_allclose(u[:4], [[r, r], [-r, r], [A_MAX, 0.0], [0.0, -A_MAX]], atol=1e-15)
    assert np.linalg.norm(u[4]) == pytest.approx(A_MAX)
    np.testing.assert_array_equal(u[5], [0.0, 0.0])


def test_the_map_is_radial_and_scales_the_infinity_norm() -> None:
    """u = a_max (||x||_inf / ||x||_2) x: same direction as x, and ||u||_2 = a_max ||x||_inf."""
    x = np.random.default_rng(SEED).uniform(-1.0, 1.0, (5000, 2))
    u = square_to_disk(x, A_MAX)
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), A_MAX * np.max(np.abs(x), axis=1), rtol=1e-13)
    np.testing.assert_allclose(u[:, 0] * x[:, 1] - u[:, 1] * x[:, 0], 0.0, atol=1e-13)
    assert np.all(np.einsum("ij,ij->i", u, x) >= 0.0)


def test_the_map_is_a_bijection_with_the_given_inverse() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-1.0, 1.0, (5000, 2))
    np.testing.assert_allclose(disk_to_square(square_to_disk(x, A_MAX), A_MAX), x, atol=1e-13)
    angle, radius = rng.uniform(0.0, 2.0 * np.pi, 5000), A_MAX * np.sqrt(rng.uniform(0.0, 1.0, 5000))
    u = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    np.testing.assert_allclose(square_to_disk(disk_to_square(u, A_MAX), A_MAX), u, atol=1e-13)
    assert np.all(np.abs(disk_to_square(u, A_MAX)) <= 1.0 + 1e-13)


def test_the_area_distortion_lies_between_one_half_and_one_times_a_max_squared() -> None:
    """D15/AM1: the Jacobian determinant is a_max^2 cos^2(theta) with |theta| <= 45 degrees off the nearest axis."""
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-0.99, 0.99, (2000, 2))
    x = x[np.abs(np.abs(x[:, 0]) - np.abs(x[:, 1])) > 1e-3]  # away from the diagonals, where the map has a kink
    h = 1e-6
    columns = [(square_to_disk(x + h * e, A_MAX) - square_to_disk(x - h * e, A_MAX)) / (2.0 * h) for e in np.eye(2)]
    det = columns[0][:, 0] * columns[1][:, 1] - columns[0][:, 1] * columns[1][:, 0]
    angle = np.arctan2(np.minimum(np.abs(x[:, 0]), np.abs(x[:, 1])), np.maximum(np.abs(x[:, 0]), np.abs(x[:, 1])))
    np.testing.assert_allclose(det, A_MAX**2 * np.cos(angle) ** 2, rtol=1e-5)
    assert det.min() >= 0.5 * A_MAX**2 - 1e-6 and det.max() <= A_MAX**2 + 1e-6


def test_the_map_is_continuous_at_zero() -> None:
    x = np.array([[1e-12, -3e-12], [0.0, 1e-300]])
    assert np.all(np.linalg.norm(square_to_disk(x, A_MAX), axis=1) <= A_MAX * np.max(np.abs(x), axis=1) + 1e-300)
