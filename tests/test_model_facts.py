"""Numerical checks of the facts M and F1-F6 of the thesis decision log (code rule C6).

Ported from ``claude/verify_setup.py`` of the thesis project, now run against
:mod:`hrlmpc.model`. These are numerical checks, not proofs; the proofs belong
in the thesis. Every random draw comes from a seeded generator, so the tests
are deterministic.

    M   compact form x+ = A x + B u + B d of the update equations
    F1  the disk of radius a_max/gamma is invariant for every u in U (d = 0)
    F2  V is robustly control invariant iff d_bar <= min(a_max + gamma v_max, v_max/dt)
    F3  the speed-limiter saturation is absorbed into the disturbance disk
    F4  speed saturation and wall reaction are mutually compatible
    F5  the inner regular N-gon of a disk loses a fraction 1 - cos(pi/N) of the radius
    F6  isotropy: without state constraints, the positions reachable in H steps
        with a disk-shaped U form a disk
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from hrlmpc.model import PhysicsParams, free_step, position_gains, system_matrices

SEED = 0
I2 = np.eye(2)


def _sample_disk(rng: np.random.Generator, n: int, r: float) -> NDArray[np.float64]:
    """``n`` points drawn uniformly (in area) from the disk of radius ``r``."""
    ang = rng.uniform(0.0, 2.0 * np.pi, n)
    rad = r * np.sqrt(rng.uniform(0.0, 1.0, n))
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)


def _proj_disk(y: NDArray[np.float64], r: float) -> NDArray[np.float64]:
    """Euclidean projection onto the disk of radius ``r`` (radial saturation)."""
    nrm = float(np.linalg.norm(y))
    return y if nrm <= r else y * (r / nrm)


def _proj_halfplane(y: NDArray[np.float64], n: NDArray[np.float64], c: float) -> NDArray[np.float64]:
    """Euclidean projection onto ``{v : n^T v >= c}``, with ``n`` a unit vector."""
    s = float(n @ y)
    return y if s >= c else y + (c - s) * n


def _random_gamma_dt(rng: np.random.Generator, gamma_hi: float, dt_hi: float) -> tuple[float, float]:
    """Draw ``(gamma, dt)`` with ``gamma * dt < 1``."""
    while True:
        gamma, dt = rng.uniform(0.1, gamma_hi), rng.uniform(0.01, dt_hi)
        if gamma * dt < 1.0:
            return gamma, dt


def test_m_matrix_form_equals_recursion() -> None:
    rng = np.random.default_rng(SEED)
    for _ in range(200):
        gamma, dt = _random_gamma_dt(rng, 3.0, 0.3)
        A, B = system_matrices(dt, gamma)
        x, u, d = rng.normal(size=4), rng.normal(size=2), rng.normal(size=2)
        np.testing.assert_allclose(A @ x + B @ (u + d), free_step(x, u, d, dt, gamma), rtol=1e-12, atol=1e-12)


def test_m_free_step_is_batched() -> None:
    rng = np.random.default_rng(SEED)
    A, B = system_matrices(0.1, 0.5)
    x, u, d = rng.normal(size=(5, 4)), rng.normal(size=(5, 2)), rng.normal(size=(5, 2))
    np.testing.assert_allclose(free_step(x, u, d, 0.1, 0.5), x @ A.T + (u + d) @ B.T, rtol=1e-12, atol=1e-12)


def test_f1_invariant_speed_and_steady_state() -> None:
    rng = np.random.default_rng(SEED)
    for _ in range(100):
        gamma, dt = _random_gamma_dt(rng, 3.0, 0.3)
        a_max = rng.uniform(0.5, 5.0)
        r_inv = a_max / gamma
        v = _sample_disk(rng, 2000, r_inv)
        u = _sample_disk(rng, 2000, a_max)
        v_next = (1.0 - gamma * dt) * v + dt * u
        assert np.all(np.linalg.norm(v_next, axis=1) <= r_inv + 1e-9)
        speed = 0.0
        for _ in range(20000):
            speed = (1.0 - gamma * dt) * speed + dt * a_max
        assert abs(speed - r_inv) < 1e-6


def _rci_bruteforce(gamma: float, dt: float, a_max: float, v_max: float, d_bar: float) -> bool:
    """Grid test of: for every v in V there is u in U with v+ in V for every d in D.

    By isotropy it suffices to check the velocities on one ray.
    """
    a = 1.0 - gamma * dt
    th_u = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    rad_u = np.linspace(0.0, a_max, 25)
    u_grid = np.array([[r * np.cos(t), r * np.sin(t)] for r in rad_u for t in th_u])
    th_d = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    d_grid = d_bar * np.stack([np.cos(th_d), np.sin(th_d)], axis=1)
    worst = 0.0
    for r in np.linspace(0.0, v_max, 60):
        y = np.array([a * r, 0.0]) + dt * u_grid  # nominal successors
        worst_over_d = np.max(np.linalg.norm(y[:, None, :] + dt * d_grid[None, :, :], axis=2), axis=1)
        worst = max(worst, float(worst_over_d.min()))
    return worst <= v_max + 1e-6


def test_f2_closed_form_matches_brute_force() -> None:
    rng = np.random.default_rng(SEED)
    for _ in range(40):
        gamma, dt = _random_gamma_dt(rng, 2.0, 0.4)
        a_max, v_max = rng.uniform(0.5, 3.0), rng.uniform(0.5, 3.0)
        threshold = min(a_max + gamma * v_max, v_max / dt)
        d_bar = threshold * rng.choice([0.85, 1.15])  # away from the grid tolerance
        assert (d_bar <= threshold) == _rci_bruteforce(gamma, dt, a_max, v_max, d_bar)


def test_f3_limiter_saturation_is_absorbed_into_the_disturbance() -> None:
    """If y lies in V, then ||Proj_V(y + dt d) - y|| <= dt ||d||."""
    rng = np.random.default_rng(SEED)
    for _ in range(20000):
        v_max, dt = rng.uniform(0.5, 3.0), rng.uniform(0.01, 0.3)
        y = _sample_disk(rng, 1, v_max)[0]
        d = _sample_disk(rng, 1, rng.uniform(0.1, 10.0))[0]
        v_hat = _proj_disk(y + dt * d, v_max)
        assert float(np.linalg.norm(v_hat - y)) <= dt * float(np.linalg.norm(d)) + 1e-12


def test_f4_speed_saturation_and_wall_reaction_are_compatible() -> None:
    """Radial saturation keeps the half-planes; the half-plane projection keeps V."""
    rng = np.random.default_rng(SEED)
    for _ in range(20000):
        v_max, dt, delta = rng.uniform(0.5, 3.0), rng.uniform(0.01, 0.3), rng.uniform(0.0, 0.5)
        n = rng.normal(size=2)
        n /= np.linalg.norm(n)
        c = -delta / dt  # the half-plane {n^T v >= -delta/dt} contains the origin
        y = rng.normal(scale=3.0, size=2)
        assert float(n @ _proj_disk(_proj_halfplane(y, n, c), v_max)) >= c - 1e-12
        assert float(np.linalg.norm(_proj_halfplane(_proj_disk(y, v_max), n, c))) <= v_max + 1e-12


@pytest.mark.parametrize("sides", [8, 12, 16, 32])
def test_f5_inner_polygon_radius_loss(sides: int) -> None:
    """The inradius of the regular N-gon inscribed in the unit circle is cos(pi/N)."""
    t = 2.0 * np.pi * np.arange(sides) / sides
    verts = np.stack([np.cos(t), np.sin(t)], axis=1)
    edges = np.roll(verts, -1, axis=0) - verts
    cross = np.abs(verts[:, 0] * edges[:, 1] - verts[:, 1] * edges[:, 0])
    inradius = float(np.min(cross / np.linalg.norm(edges, axis=1)))
    assert abs(inradius - np.cos(np.pi / sides)) < 1e-12


def test_f5_radius_losses_quoted_in_the_decision_log() -> None:
    quoted = {8: 7.61, 12: 3.41, 16: 1.92, 32: 0.48}
    for sides, percent in quoted.items():
        assert round(100.0 * (1.0 - np.cos(np.pi / sides)), 2) == percent


def test_f6_blocks_of_powers_are_scalar_multiples_of_identity() -> None:
    gamma, dt, horizon = 0.8, 0.1, 15
    A, B = system_matrices(dt, gamma)
    gains = position_gains(dt, gamma, horizon)
    for i in range(horizon):
        block = np.linalg.matrix_power(A, horizon - 1 - i) @ B
        np.testing.assert_allclose(block[:2], gains[i] * I2, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(block[2:], block[2, 0] * I2, rtol=1e-12, atol=1e-15)


def test_f6_reachable_positions_form_a_disk() -> None:
    rng = np.random.default_rng(SEED)
    gamma, dt, a_max, horizon = 0.8, 0.1, 1.0, 15
    gains = position_gains(dt, gamma, horizon)
    radius = a_max * float(np.abs(gains).sum())
    sampled = max(float(np.linalg.norm(gains @ _sample_disk(rng, horizon, a_max))) for _ in range(20000))
    assert sampled <= radius + 1e-12
    for t in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
        e = np.array([np.cos(t), np.sin(t)])
        p_ext = (gains[:, None] * (a_max * np.sign(gains)[:, None] * e[None, :])).sum(axis=0)
        assert abs(float(np.linalg.norm(p_ext)) - radius) < 1e-12


def test_position_gains_rejects_an_empty_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        position_gains(0.1, 0.5, 0)


@pytest.mark.parametrize("d_bar", [0.0, 0.5, 1.0])
def test_d9_parameters_satisfy_f1_and_f2(d_bar: float) -> None:
    params = PhysicsParams(dt=0.1, a_max=2.5, v_max=1.2, gamma=0.5, gamma_w=1.0, d_bar=d_bar)
    assert params.velocity_factor == pytest.approx(0.95)
    assert params.invariant_speed == pytest.approx(5.0)
    assert params.max_robust_disturbance == pytest.approx(3.1)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"dt": 0.0}, "positive"),
        ({"gamma": 0.0}, r"gamma\*dt"),
        ({"gamma": 10.0}, r"gamma\*dt"),
        ({"gamma_w": -1.0}, r"gamma_w\*dt"),
        ({"gamma_w": 10.0}, r"gamma_w\*dt"),
        ({"d_bar": -0.1}, "d_bar"),
        ({"gamma": 2.5}, "F1"),
        ({"d_bar": 3.2}, "F2"),
    ],
)
def test_check_physics_rejects_violations(changes: dict[str, float], message: str) -> None:
    values = {"dt": 0.1, "a_max": 2.5, "v_max": 1.2, "gamma": 0.5, "gamma_w": 1.0, "d_bar": 0.0}
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        PhysicsParams(**values)
