"""Tests of the MPC Worker's problem data, without solvers (decision log D31-D34)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import hrlmpc.mpc_problem as mpc_problem
from hrlmpc.config import EnvConfig, load_env_config
from hrlmpc.geometry import Layout
from hrlmpc.mpc_problem import MPCProblem, MPCSettings, project_disk


def _setup(name: str, d_bar: float = 0.0, **settings: float) -> tuple[EnvConfig, Layout, MPCProblem]:
    cfg = load_env_config(f"configs/env/{name}.yaml")
    params = dataclasses.replace(cfg.physics, d_bar=d_bar)
    layout = Layout.from_config(cfg.geometry)
    return cfg, layout, MPCProblem(params, layout, MPCSettings(**settings))  # type: ignore[arg-type]


def _random_plan(problem: MPCProblem, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Positions of a nominal sequence that respects the speed bounds (not the dynamics)."""
    N = problem.N
    z = np.zeros((N + 1, 4))
    angle = rng.uniform(0, 2 * np.pi)
    z[0, :2] = x[:2] + rng.uniform(0, problem.tube.position_radius) * np.array([np.cos(angle), np.sin(angle)])
    for k in range(1, N + 1):
        if k < N:
            w = rng.normal(size=2)
            z[k, 2:] = rng.uniform(0, problem.v_bar) * w / np.linalg.norm(w)
        z[k, :2] = z[k - 1, :2] + problem.dt * z[k, 2:]
    return z, np.zeros((N, 2))


def test_settings_are_validated() -> None:
    MPCSettings()
    for bad in ({"horizon": 0}, {"horizon": 2.5}, {"horizon": True}, {"q_pos": 0.0}, {"r": -1.0}, {"rho": 0.0},
                {"tail_tolerance": float("nan")}, {"mip_gap": -1e-4}, {"mip_gap": 1.0}, {"work_limit": 0.0}):
        with pytest.raises(ValueError):
            MPCSettings(**bad)  # type: ignore[arg-type]


def test_the_tightened_sets_follow_the_tube() -> None:
    cfg, _, problem = _setup("slalom", d_bar=0.5)
    tube = problem.tube
    assert problem.v_bar == pytest.approx(cfg.physics.v_max - tube.next_speed_radius)
    assert problem.u_bar == pytest.approx(cfg.physics.a_max - tube.input_radius)
    assert problem.margin == pytest.approx(tube.position_radius + 1e-3)
    np.testing.assert_allclose(problem.arena_bounds, Layout.from_config(cfg.geometry).arena.offsets - problem.margin)
    _, _, calm = _setup("slalom", d_bar=0.0)
    assert calm.v_bar == cfg.physics.v_max and calm.u_bar == cfg.physics.a_max and calm.margin == pytest.approx(1e-3)


def test_a_disturbance_that_takes_the_whole_thrust_is_refused() -> None:
    with pytest.raises(ValueError, match="thrust"):
        _setup("tunnel", d_bar=2.0)


def test_an_asymmetric_closed_loop_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    real = mpc_problem.lqr

    def skewed(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        K, P = real(A, B, Q, R)
        K = K.copy()
        K[1] *= 1.5  # stiffer on y than on x
        return K, P

    monkeypatch.setattr(mpc_problem, "lqr", skewed)
    with pytest.raises(ValueError, match="rotation"):
        _setup("tunnel", d_bar=0.5)


def test_the_slalom_has_four_obstacles_with_three_live_faces_each() -> None:
    _, layout, problem = _setup("slalom", d_bar=0.5)
    assert len(problem.obstacles) == len(layout.blocked) == 4
    for obstacle, polygon in zip(problem.obstacles, layout.blocked, strict=True):
        assert obstacle.num_faces == 3  # the face that D16 pushed beyond the wall can never separate
        for n, c, b in zip(obstacle.normals, obstacle.offsets, obstacle.bounds, strict=True):
            j = int(np.argmin(np.linalg.norm(polygon.normals - n, axis=1)))
            assert polygon.offsets[j] == pytest.approx(c)
            assert b == pytest.approx(c + problem.margin)
    _, _, tunnel = _setup("tunnel")
    assert tunnel.obstacles == []


def test_big_m_deactivates_every_face_in_the_arena() -> None:
    _, layout, problem = _setup("slalom", d_bar=1.0)
    for obstacle in problem.obstacles:
        slack = layout.arena.vertices @ obstacle.normals.T - (obstacle.bounds - obstacle.big_m)
        assert np.all(slack > 0.0)  # with delta = 0 the constraint holds at every vertex, hence everywhere


def test_the_margin_covers_what_a_mixed_integer_plan_may_violate_a_face_by() -> None:
    _, _, problem = _setup("slalom", d_bar=1.0)
    largest = max(float(obstacle.big_m.max()) for obstacle in problem.obstacles)
    assert 9.0 < largest < 9.2  # about the arena's diagonal extent beyond a face
    assert problem.face_slack == pytest.approx(mpc_problem.INT_FEAS_TOL * largest + mpc_problem.FEAS_TOL)
    assert problem.face_slack < mpc_problem.PLAN_TOL < problem.settings.rho  # a plan's binaries are read back
    with pytest.raises(ValueError, match="rho"):
        _setup("slalom", d_bar=1.0, rho=0.5 * problem.face_slack)
    _, _, tunnel = _setup("tunnel")
    assert tunnel.face_slack == mpc_problem.FEAS_TOL  # no obstacle, no binary


@pytest.mark.parametrize("d_bar", [0.0, 0.5])
def test_the_reach_boxes_contain_every_plan(d_bar: float) -> None:
    _, _, problem = _setup("slalom", d_bar=d_bar)
    rng = np.random.default_rng(0)
    x = np.array([5.5, 0.5, 0.8, -0.3])  # far enough from the arena's walls for the boxes not to be clipped
    lo, hi = problem.reach_boxes(x)
    for _ in range(500):
        z, _ = _random_plan(problem, x, rng)
        assert np.all(z[:, :2] >= lo - 1e-12) and np.all(z[:, :2] <= hi + 1e-12)
    assert np.all(np.diff(hi - lo, axis=0) >= -1e-12)  # nested boxes
    lo_c, hi_c = problem.reach_boxes(np.array([-0.9, 1.9, 0.0, 0.0]))  # near a corner: clipped
    assert np.all(lo_c >= problem.position_low - 1e-12) and np.all(hi_c <= problem.position_high + 1e-12)


def test_fixed_binaries_are_satisfied_by_every_plan() -> None:
    _, _, problem = _setup("slalom", d_bar=0.5)
    rng = np.random.default_rng(1)
    total_fixed = 0
    for x in (np.array([1.0, 0.0, 0.0, 0.0]), np.array([3.6, 1.0, 1.0, 0.0]), np.array([6.0, -0.9, 0.9, 0.2])):
        bounds = problem.fixed_binaries(x)
        for _ in range(300):
            z, _ = _random_plan(problem, x, rng)
            p = z[:, :2]
            for obstacle, (lb, ub) in zip(problem.obstacles, bounds, strict=True):
                fixed_on = (lb == 1.0) & (ub == 1.0)
                beyond = p @ obstacle.normals.T >= obstacle.bounds - 1e-12
                both = beyond[:-1] & beyond[1:]
                assert np.all(both[fixed_on])  # a face fixed on holds for both ends of the segment
        for lb, ub in bounds:
            certified = np.all(lb == ub, axis=1)
            assert np.all(lb[certified].sum(axis=1) == 1.0)  # one face on per certified segment
            total_fixed += int(certified.sum())
    assert total_fixed > 0
    far = problem.fixed_binaries(np.array([1.0, 0.0, 0.0, 0.0]))
    assert problem.free_binaries(far) == 0  # every obstacle is out of reach from the spawn box


def test_the_shifted_plan_holds_the_final_rest_state() -> None:
    _, _, problem = _setup("tunnel")
    N = problem.N
    rng = np.random.default_rng(2)
    z = np.zeros((N + 1, 4))
    v = rng.uniform(-1, 1, size=(N, 2))
    v[-1] = -(1 - problem.params.gamma * problem.dt) * 0.0  # placeholder, fixed below
    z[0] = [1.0, 0.0, 0.3, 0.1]
    for k in range(N):
        z[k + 1] = problem.A @ z[k] + problem.B @ v[k]
    # make the last stage rest: choose v[N-1] that zeroes the velocity
    v[N - 1] = -(1 - problem.params.gamma * problem.dt) * z[N - 1, 2:] / problem.dt
    z[N] = problem.A @ z[N - 1] + problem.B @ v[N - 1]
    assert np.allclose(z[N, 2:], 0.0)
    zc, vc = problem.candidate(z, v)
    np.testing.assert_array_equal(zc[:-1], z[1:])
    np.testing.assert_array_equal(zc[-1], z[-1])
    np.testing.assert_array_equal(vc[-1], 0.0)
    np.testing.assert_allclose(zc[1:], zc[:-1] @ problem.A.T + vc @ problem.B.T, atol=1e-12)  # still a trajectory


def test_binaries_of_a_plan() -> None:
    _, _, problem = _setup("slalom", d_bar=0.0)
    N = problem.N
    # a plan at rest left of the first gate: beyond the left faces of the gate's obstacles
    z = np.tile([3.0, 1.0, 0.0, 0.0], (N + 1, 1))
    deltas = problem.binaries_of(z)
    assert deltas is not None
    for obstacle, d in zip(problem.obstacles, deltas, strict=True):
        left = int(np.argmin(np.linalg.norm(obstacle.normals - [-1.0, 0.0], axis=1)))
        assert np.all(d[:, left] == 1.0)
    # a segment that cuts the corner of O1 = [4, 5] x [-2, 0.25] diagonally: no single face for both ends
    corner = np.tile([3.9, 0.1, 0.0, 0.0], (N + 1, 1))  # beyond the left face only
    corner[1:, :2] = [4.1, 0.35]  # beyond the top face only; the segment crosses x = 4 at y = 0.225 < 0.25
    assert problem.binaries_of(corner) is None


def test_a_segment_beyond_one_face_never_enters_the_obstacle() -> None:
    """D32: both ends beyond the same face of the enlarged obstacle, errors in Z: the real segment stays out."""
    _, layout, problem = _setup("slalom", d_bar=1.0)
    rng = np.random.default_rng(3)
    r_p = problem.tube.position_radius
    for obstacle, polygon in zip(problem.obstacles, layout.blocked, strict=True):
        for j, (n, b) in enumerate(zip(obstacle.normals, obstacle.bounds, strict=True)):
            tangent = np.array([-n[1], n[0]])
            for _ in range(300):
                base = b + rng.uniform(0.0, 0.5, size=2)  # n' p >= b for both ends
                along = rng.uniform(-3.0, 3.0, size=2)
                ends = np.array([base[i] * n + along[i] * tangent for i in range(2)])
                errors = rng.normal(size=(2, 2))
                errors *= (r_p * rng.uniform(size=(2, 1))) / np.linalg.norm(errors, axis=1, keepdims=True)
                real = ends + errors
                lam = np.linspace(0.0, 1.0, 41)[:, None]
                points = lam * real[0] + (1.0 - lam) * real[1]
                assert not polygon.contains_interior(points, tol=-0.5e-3).any(), (j, ends)


def test_the_violations_of_plans() -> None:
    _, _, problem = _setup("slalom", d_bar=0.0)
    N = problem.N
    x = np.array([3.0, 1.0, 0.0, 0.0])
    z = np.tile(x, (N + 1, 1))
    v = np.zeros((N, 2))
    clean = problem.violations(x, z, v)
    assert max(clean.values()) == 0.0
    fast = z.copy()
    fast[3, 2] = problem.v_bar + 0.1
    assert problem.violations(x, fast, v)["speed"] == pytest.approx(0.1)
    assert problem.violations(x, fast, v)["dynamics"] > 0.0
    inside = z.copy()
    inside[4, :2] = [4.5, -1.0]  # inside O1
    assert problem.violations(x, inside, v)["obstacles"] > 0.4
    outside = z.copy()
    outside[2, :2] = [3.0, 2.5]
    assert problem.violations(x, outside, v)["arena"] == pytest.approx(0.5 + problem.margin)
    shifted = z.copy()
    shifted[0, 0] += 0.01
    assert problem.violations(x, shifted, v)["initial"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="shapes"):
        problem.violations(x, z[:-1], v)


def test_the_applied_inputs() -> None:
    _, _, problem = _setup("tunnel")
    x = np.array([1.0, 0.0, 0.2, 0.0])
    brake = problem.emergency_input(x)
    v_next = (1 - problem.params.gamma * problem.dt) * x[2:] + problem.dt * brake
    np.testing.assert_allclose(v_next, 0.0, atol=1e-12)  # stops in one step when U allows it
    fast = np.array([1.0, 0.0, 1.2, 0.0])
    assert np.linalg.norm(problem.emergency_input(fast)) == pytest.approx(problem.params.a_max)
    z = np.tile(x, (problem.N + 1, 1))
    v = np.zeros((problem.N, 2))
    v[0] = [3.0, 4.0]
    assert np.linalg.norm(problem.input(x, z, v)) == pytest.approx(problem.params.a_max)
    np.testing.assert_allclose(project_disk([0.3, 0.4], 1.0), [0.3, 0.4])


def test_the_summary_and_the_stopping_time() -> None:
    _, _, problem = _setup("slalom", d_bar=0.5)
    text = problem.summary()
    assert "planned speed <= 1.083" in text and "4 obstacles, 120 binaries" in text
    assert problem.stopping_steps() == 6
    _, _, calm = _setup("tunnel")
    assert calm.stopping_steps() == 5
