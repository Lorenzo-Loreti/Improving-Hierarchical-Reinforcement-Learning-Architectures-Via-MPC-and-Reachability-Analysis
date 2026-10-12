"""Tests of the MPC Worker's problem data, without solvers (decision log D31-D34, D37, D45, D46)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import hrlmpc.mpc_problem as mpc_problem
from hrlmpc.config import EnvConfig, load_env_config
from hrlmpc.geometry import Layout
from hrlmpc.mpc_problem import MPCProblem, MPCSettings, project_disk


def _setup(
    name: str, d_bar: float = 0.0, segment_steps: int | None = None, **settings: float | None
) -> tuple[EnvConfig, Layout, MPCProblem]:
    """The problem of a layout; the horizon is 15 unless given (``None``: automatic, D45)."""
    cfg = load_env_config(f"configs/env/{name}.yaml")
    params = dataclasses.replace(cfg.physics, d_bar=d_bar)
    layout = Layout.from_config(cfg.geometry)
    settings.setdefault("horizon", 15)
    problem = MPCProblem(params, layout, MPCSettings(**settings), segment_steps=segment_steps)  # type: ignore[arg-type]
    return cfg, layout, problem


def _random_plan(problem: MPCProblem, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Positions of a nominal sequence whose speeds respect the bounds of every plan (not the dynamics):
    the planned speed, and what still lets the plan rest by stage N."""
    N = problem.N
    z = np.zeros((N + 1, 4))
    angle = rng.uniform(0, 2 * np.pi)
    z[0, :2] = x[:2] + rng.uniform(0, problem.tube.position_radius) * np.array([np.cos(angle), np.sin(angle)])
    bounds = problem.stage_speed_bounds()
    for k in range(1, N + 1):
        w = rng.normal(size=2)
        z[k, 2:] = rng.uniform(0, bounds[k - 1]) * w / np.linalg.norm(w)
        z[k, :2] = z[k - 1, :2] + problem.dt * z[k, 2:]
    return z, np.zeros((N, 2))


def _brake(problem: MPCProblem, speed: float, steps: int) -> float:
    """The speed left after ``steps`` steps of full braking along the velocity, from ``speed``."""
    beta = 1.0 - problem.params.gamma * problem.dt
    for _ in range(steps):
        speed = max(0.0, beta * speed - problem.dt * problem.u_bar)
    return speed


def test_settings_are_validated() -> None:
    defaults = MPCSettings()
    assert (defaults.target_weight, defaults.effort_weight, defaults.r) == (1e4, 0.1, 0.1)  # D37
    assert defaults.horizon is None and defaults.start_weight == 0.1  # automatic horizon (D45), regularizer (D46)
    assert MPCSettings(start_weight=0.0).start_weight == 0.0  # no regularizer
    for bad in ({"horizon": 0}, {"horizon": 2.5}, {"horizon": True}, {"horizon": "auto"}, {"q_pos": 0.0},
                {"r": -1.0}, {"rho": 0.0}, {"target_weight": 0.0}, {"target_weight": float("inf")},
                {"effort_weight": -0.1}, {"effort_weight": 0.0}, {"start_weight": -0.1},
                {"start_weight": float("nan")}, {"start_weight": float("inf")}, {"start_weight": True},
                {"tail_tolerance": float("nan")}, {"mip_gap": -1e-4}, {"mip_gap": 1.0}, {"work_limit": 0.0}):
        with pytest.raises(ValueError):
            MPCSettings(**bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(("d_bar", "expected"), [(0.0, 15), (0.5, 16), (1.0, 17)])
def test_an_automatic_horizon_leaves_just_room_to_stop_after_a_segment(d_bar: float, expected: int) -> None:
    """D45: without a horizon in the settings, N = H + n_b, the smallest horizon that D37 accepts."""
    _, _, problem = _setup("slalom", d_bar=d_bar, segment_steps=10, horizon=None)
    assert problem.N == expected == 10 + problem.stopping_steps()
    assert f"horizon N = {expected} (automatic, D45)" in problem.summary()
    problem.check_segments(10)
    with pytest.raises(ValueError, match="N >= "):
        problem.check_segments(11)
    _, _, longer = _setup("tunnel", d_bar=d_bar, segment_steps=12, horizon=None)
    assert longer.N == expected + 2


def test_an_automatic_horizon_needs_the_length_of_the_segments() -> None:
    cfg = load_env_config("configs/env/tunnel.yaml")
    layout = Layout.from_config(cfg.geometry)
    with pytest.raises(ValueError, match="segment"):
        MPCProblem(cfg.physics, layout, MPCSettings())
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="segment"):
            MPCProblem(cfg.physics, layout, MPCSettings(), segment_steps=bad)  # type: ignore[arg-type]
    assert MPCProblem(cfg.physics, layout, MPCSettings(horizon=20), segment_steps=10).N == 20  # an explicit one stays
    assert MPCProblem(cfg.physics, layout, MPCSettings(horizon=20)).N == 20


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


def test_the_cost_is_the_distance_at_the_waypoint_and_the_effort_before_it() -> None:
    """D37: q_g ||z_{k*,p} - g||^2 + r_e sum_{k<k*} ||v_k||^2, with k* the steps left in the segment."""
    _, _, problem = _setup("tunnel", target_weight=100.0, effort_weight=0.5)
    N = problem.N
    rng = np.random.default_rng(4)
    z, v = rng.normal(size=(N + 1, 4)), rng.normal(size=(N, 2))
    g = np.array([3.0, -1.0])
    for k in (1, 4, N):
        expected = 100.0 * np.sum((z[k, :2] - g) ** 2) + 0.5 * np.sum(v[:k] ** 2)
        assert problem.cost(z, v, g, k) == pytest.approx(expected, rel=1e-12)
        assert problem.cost(z, v, g, np.int64(k)) == pytest.approx(expected, rel=1e-12)
    later = z.copy()
    later[3:, :2] += 5.0  # the stages after the waypoint and the inputs from it on cost nothing
    tail = v.copy()
    tail[2:] *= 7.0
    assert problem.cost(later, tail, g, 2) == pytest.approx(problem.cost(z, v, g, 2), rel=1e-12)
    for bad in (0, N + 1, -1, 2.0, True):
        with pytest.raises(ValueError, match="steps_left"):
            problem.cost(z, v, g, bad)  # type: ignore[arg-type]
    assert not problem.regularized
    assert problem.cost(z, v, g, 4, x=rng.normal(size=4)) == pytest.approx(problem.cost(z, v, g, 4), rel=1e-12)


def test_with_a_tube_the_cost_adds_the_distance_of_the_nominal_start() -> None:
    """D46: with a tube the cost gains eps ||x - z_0||^2, which needs the measured state x; without a tube
    (z_0 = x) or with eps = 0 the term is left out."""
    _, _, problem = _setup("tunnel", d_bar=0.5, target_weight=100.0, effort_weight=0.5, start_weight=2.0)
    N = problem.N
    rng = np.random.default_rng(5)
    z, v = rng.normal(size=(N + 1, 4)), rng.normal(size=(N, 2))
    g, x = np.array([3.0, -1.0]), rng.normal(size=4)
    base = 100.0 * np.sum((z[4, :2] - g) ** 2) + 0.5 * np.sum(v[:4] ** 2)
    assert problem.regularized
    assert problem.cost(z, v, g, 4, x=x) == pytest.approx(base + 2.0 * np.sum((x - z[0]) ** 2), rel=1e-12)
    with pytest.raises(ValueError, match="measured state"):
        problem.cost(z, v, g, 4)
    _, _, off = _setup("tunnel", d_bar=0.5, target_weight=100.0, effort_weight=0.5, start_weight=0.0)
    assert not off.regularized and off.cost(z, v, g, 4) == pytest.approx(base, rel=1e-12)


@pytest.mark.parametrize("d_bar", [0.0, 0.5, 1.0])
def test_the_speed_bounds_are_exactly_what_still_lets_a_plan_rest(d_bar: float) -> None:
    """Stage k of every plan: speed at most v_bar, and at most the speed from which full braking rests by N."""
    _, _, problem = _setup("tunnel", d_bar=d_bar)
    N, bounds = problem.N, problem.stage_speed_bounds()
    assert bounds.shape == (N,) and bounds[-1] == 0.0
    assert np.all(np.diff(bounds) <= 1e-12) and np.all(bounds <= problem.v_bar)
    capped = int(np.sum(bounds < problem.v_bar))
    assert capped == problem.stopping_steps()  # only the last n_b stages are capped by the stop
    for k in range(1, N + 1):
        n, s = N - k, bounds[k - 1]
        assert _brake(problem, s, n) <= 1e-9  # full braking from the bound rests by stage N
        if s < problem.v_bar:  # the bound is exact: from slightly faster no plan rests in time
            assert _brake(problem, s * (1.0 + 1e-6) + 1e-9, n) > 0.0


def _dynamic_plan(problem: MPCProblem, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """A plan that follows the dynamics from a start in the tube: random inputs in the thrust disk, then full
    braking along the velocity; ``True`` if it meets the speed bound and rests at stage N, as every plan must."""
    N, beta, dt = problem.N, 1.0 - problem.params.gamma * problem.dt, problem.dt
    z = np.zeros((N + 1, 4))
    w = rng.normal(size=4)
    z[0] = x + rng.uniform() * w / np.linalg.norm(w) * np.array([problem.tube.position_radius] * 2
                                                                 + [problem.tube.speed_radius] * 2)
    brake_from = int(rng.integers(0, N))
    heading = rng.normal(size=2)
    for k in range(N):
        speed = float(np.linalg.norm(z[k, 2:]))
        if k >= brake_from:  # the input that lowers the speed most, without reversing
            u = -min(problem.u_bar, beta * speed / dt) * z[k, 2:] / max(speed, 1e-300)
        else:
            u = problem.u_bar * rng.uniform() * heading / np.linalg.norm(heading)
        z[k + 1, 2:] = beta * z[k, 2:] + dt * u
        z[k + 1, :2] = z[k, :2] + dt * z[k + 1, 2:]
    speeds = np.linalg.norm(z[1:N, 2:], axis=1)
    return z, bool(np.all(speeds <= problem.v_bar) and np.linalg.norm(z[N, 2:]) <= 1e-12)


@pytest.mark.parametrize(
    ("d_bar", "horizon", "speed"), [(0.0, 3, 0.25), (0.0, 15, 0.95), (0.5, 5, 0.6), (0.5, 16, 0.95), (1.0, 17, 0.9)]
)
def test_the_reach_boxes_contain_every_plan_that_follows_the_dynamics(d_bar: float, horizon: int,
                                                                       speed: float) -> None:
    """Not circular: the plans obey the dynamics, the thrust disk, the speed bound and the terminal rest, and the
    boxes built from :meth:`stage_speed_bounds` contain every one of them. The start is slow enough to rest by N."""
    _, _, problem = _setup("slalom", d_bar=d_bar, horizon=horizon)
    rng = np.random.default_rng(6)
    x = np.array([5.5, 0.0, 0.9 * speed / 0.95, 0.3 * speed / 0.95])
    lo, hi = problem.reach_boxes(x)
    accepted = 0
    for _ in range(3000):
        z, ok = _dynamic_plan(problem, x, rng)
        if ok:
            accepted += 1
            assert np.all(z[:, :2] >= lo - 1e-12) and np.all(z[:, :2] <= hi + 1e-12)
    assert accepted > 300


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
    half = 0.5 * (hi - lo)[:, 0]
    travel = np.concatenate([[0.0], np.cumsum(problem.stage_speed_bounds())])
    expected = problem.tube.position_radius + problem.dt * travel
    np.testing.assert_allclose(half, expected, rtol=1e-12)
    assert half[-1] < problem.tube.position_radius + (problem.N - 1) * problem.dt * problem.v_bar - 0.1  # tighter
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
    _, _, problem = _setup("slalom", d_bar=0.5, horizon=16)
    text = problem.summary()
    assert "planned speed <= 1.083" in text and "4 obstacles, 192 binaries" in text and "q_g = 10000" in text
    assert "eps = 0.1" in text  # the regularizer of D46, with a tube only
    assert problem.stopping_steps() == 6
    _, _, calm = _setup("tunnel")
    assert calm.stopping_steps() == 5 and "eps =" not in calm.summary()  # no tube, no regularizer
    assert "automatic" not in calm.summary()  # an explicit horizon
    _, _, rough = _setup("tunnel", d_bar=1.0, horizon=17)
    assert rough.stopping_steps() == 7


def test_the_horizon_must_leave_room_to_stop_after_a_segment() -> None:
    """D37: the waypoint lies at stage H at a decision, so N >= H + n_b: 15 at d_bar = 0, 16 at 0.5, 17 at 1.0."""
    for d_bar, needed in ((0.0, 15), (0.5, 16), (1.0, 17)):
        _, _, enough = _setup("tunnel", d_bar=d_bar, horizon=needed)
        enough.check_segments(10)
        _, _, short = _setup("tunnel", d_bar=d_bar, horizon=needed - 1)
        with pytest.raises(ValueError, match=f"N >= {needed}"):
            short.check_segments(10)
    _, _, problem = _setup("tunnel")
    problem.check_segments(3)
    with pytest.raises(ValueError, match="N >= 16"):
        problem.check_segments(11)
