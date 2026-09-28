"""Tests for the tube-MPC worker (algorithms/tube_mpc.py).

Three layers: the set computations of the note's chapter 2 against brute
force or their defining properties; the offline sets of the worker (the
obstacles against the environment's own contact rule, the big-M constants
against their definition); and the closed loop, where the claims that
matter live -- theorem 4.1's candidate is feasible after any disturbance in
W, the real state never touches a wall, and fixing unreachable binaries
changes no solution.

Every model here stays within the size-limited Gurobi licence that comes
with `pip install gurobipy` (200 variables for a quadratic model): the
disturbed worker, whose lifted representation of Z adds 4 * s ~ 70
variables, runs at a short horizon.

The shared test directory only puts the tunnel's `envs` on sys.path, so the
slalom's gates are rebuilt below from the same WidthSegments inside a
TunnelEnv, whose step() is the slalom's.
"""

import numpy as np
import pytest
from scipy.optimize import linprog

from envs.config import TunnelEnvConfig
from envs.tunnel_env import TunnelEnv
from envs.width_profile import WidthProfile, WidthSegment
from tube_mpc import (
    CANDIDATE, EMERGENCY, SOLVER, Polytope, RPIOuterApprox, TubeMPCWorker,
    corridor_obstacles, dlqr, minkowski_outer, pontryagin_diff,
)

NOISE = (0.005, 0.05)   # the level chosen for the disturbed experiments


def slalom_profile():
    """scenarios/slalom/envs/width_profile.py's slalom_profile() at its defaults."""
    return WidthProfile([
        WidthSegment(-np.inf, 4.0, 2.0, 0.0),
        WidthSegment(4.0, 5.0, 0.75, 1.0),
        WidthSegment(5.0, 7.0, 2.0, 0.0),
        WidthSegment(7.0, 8.0, 0.75, -1.0),
        WidthSegment(8.0, np.inf, 2.0, 0.0),
    ])


def slalom_env(noise=(0.0, 0.0), **overrides):
    return TunnelEnv(config=TunnelEnvConfig(width_profile=slalom_profile(), noise_bound_p=noise[0],
                                            noise_bound_v=noise[1], **overrides))


def gate_target(p):
    """A hand-written manager: aim through each gate's opening, then at the line."""
    y = 1.0 if p[0] < 5.0 else (-1.0 if p[0] < 8.0 else 0.0)
    return np.array([p[0] + 1.8, y])


def _plant(dt=0.1):
    A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    B = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
    return A, B


# ----------------------------------------------------------------------------
# Chapter 2: sets and support functions
# ----------------------------------------------------------------------------

def test_box_support_closed_form_matches_the_lp():
    box = Polytope.box([-1.0, 0.5, -2.0], [3.0, 1.5, 0.0])
    generic = Polytope(box.H, box.h)            # same set, no closed form
    rng = np.random.default_rng(0)
    for a in rng.standard_normal((20, 3)):
        assert box.support(a) == pytest.approx(generic.support(a), abs=1e-9)


def test_pontryagin_difference_of_a_box_is_the_shrunk_box():
    P = Polytope.box([0.0, 0.0], [4.0, 2.0])
    S = Polytope.box([-0.5, -0.1], [0.5, 0.3])
    D = pontryagin_diff(P, S.support)
    lo, hi = D.bounds
    np.testing.assert_allclose(lo, [0.5, 0.1])
    np.testing.assert_allclose(hi, [3.5, 1.7])


def test_minkowski_outer_keeps_the_normals_and_contains_the_sum():
    P = Polytope(np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]), np.array([1.0, 0.0, 0.0]))  # a triangle
    S = Polytope.box([-0.2, -0.2], [0.2, 0.2])
    O = minkowski_outer(P, S.support)
    np.testing.assert_array_equal(O.H, P.H)
    for vertex in ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]):
        for corner in ([-0.2, -0.2], [0.2, 0.2], [-0.2, 0.2], [0.2, -0.2]):
            assert np.all(O.H @ (np.array(vertex) + corner) <= O.h + 1e-12)


def _worker_ak(q=(10.0, 1.0), r=0.1):
    A, B = _plant()
    K, _ = dlqr(A, B, np.diag([q[0], q[0], q[1], q[1]]), r * np.eye(2))
    return A + B @ K


def test_rpi_set_is_robustly_invariant_in_sampled_directions():
    """Section 5.6 (i): h_Z(A_K^T a) + h_W(a) <= h_Z(a) for many directions a,
    i.e. A_K Z (+) W in Z (proposition 2.3(c)). Sampling cannot prove it --
    proposition 2.8 does -- but it catches a wrong power, transpose or alpha."""
    AK = _worker_ak()
    b = np.array([NOISE[0], NOISE[0], NOISE[1], NOISE[1]])
    W = Polytope.box(-b, b)
    Z = RPIOuterApprox(AK, W, eps=1e-2)
    rng = np.random.default_rng(0)
    gaps = [Z.support(AK.T @ a) + W.support(a) - Z.support(a) for a in rng.standard_normal((500, 4))]
    assert max(gaps) <= 1e-9


def test_rpi_set_is_within_eps_of_the_minimal_one():
    """F_inf in Z in F_inf (+) B_inf(eps) (proposition 2.8). F_inf is
    approximated by F_300, whose missing tail A_K^300 W is ~1e-35 here."""
    AK = _worker_ak()
    b = np.array([NOISE[0], NOISE[0], NOISE[1], NOISE[1]])
    W = Polytope.box(-b, b)
    eps = 1e-2
    Z = RPIOuterApprox(AK, W, eps=eps)
    powers = [np.linalg.matrix_power(AK, l) for l in range(300)]
    for a in np.vstack([np.eye(4), -np.eye(4)]):
        h_inf = sum(W.support(Ai.T @ a) for Ai in powers)
        assert h_inf - 1e-12 <= Z.support(a) <= h_inf + eps


def test_no_disturbance_gives_the_zero_tube():
    Z = RPIOuterApprox(_worker_ak(), Polytope.box(np.zeros(4), np.zeros(4)))
    assert Z.s == 0 and Z.lifted_matrices() == []
    assert Z.support(np.ones(4)) == 0.0


def test_a_flat_disturbance_box_is_rejected():
    """Algorithm 1 needs 0 in the interior of W (assumption 2)."""
    with pytest.raises(ValueError, match="interior of W"):
        RPIOuterApprox(_worker_ak(), Polytope.box([0, 0, -0.05, -0.05], [0, 0, 0.05, 0.05]))


# ----------------------------------------------------------------------------
# The worker's offline sets
# ----------------------------------------------------------------------------

def test_the_slalom_is_four_three_faced_obstacles_and_the_tunnel_none():
    obstacles = corridor_obstacles(slalom_profile())
    assert len(obstacles) == 4
    assert all(O.H.shape == (3, 4) for O in obstacles)
    assert np.all(np.stack([O.H for O in obstacles])[:, :, 2:] == 0)   # cylinders: no velocity
    assert corridor_obstacles(WidthProfile([WidthSegment(-np.inf, np.inf, 2.0, 0.0)])) == []


def test_obstacles_are_exactly_where_the_environment_counts_a_contact():
    """At random positions away from every boundary, a position is inside an
    obstacle (or past the envelope, which is X0's wall) exactly when the
    environment's width profile puts it outside the corridor."""
    profile = slalom_profile()
    obstacles = corridor_obstacles(profile)
    env_lo, env_hi = profile.envelope()
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-1, 11, 4000), rng.uniform(-2.2, 2.2, 4000)])
    boundaries_x = np.array([4.0, 5.0, 7.0, 8.0])
    boundaries_y = np.array([-2.0, -1.75, -0.25, 0.25, 1.75, 2.0])
    far = (np.min(np.abs(pts[:, :1] - boundaries_x), axis=1) > 1e-3) & \
          (np.min(np.abs(pts[:, 1:] - boundaries_y), axis=1) > 1e-3)
    for p in pts[far]:
        x = np.array([p[0], p[1], 0.0, 0.0])
        in_obstacle = any(np.all(O.H @ x <= O.h) for O in obstacles) or not (env_lo < p[1] < env_hi)
        y_lo, y_hi = profile.bounds_at(p[0])
        assert in_obstacle == (not (y_lo < p[1] < y_hi)), p


def test_big_m_constants_deactivate_every_face_over_the_tightened_box():
    """(3.16): M_ij >= max over X0 (-) Z of h~_ij - H_ij z, by LP."""
    w = TubeMPCWorker.from_env(slalom_env(), horizon=2)
    lo, hi = w.Xbar0.bounds
    for Ob, M in zip(w.Obar, w.M):
        for row, h, m in zip(Ob.H, Ob.h, M):
            res = linprog(row, bounds=list(zip(lo, hi)), method="highs")   # min H z
            assert m >= h - res.fun - 1e-9


def test_tightening_matches_the_note():
    """X0 (-) Z (3.9) and U (-) KZ (3.6) by their support functions, and the
    margin rho on the walls only."""
    rho = 1e-3
    w = TubeMPCWorker.from_env(slalom_env(NOISE), horizon=2, rho=rho)
    env = slalom_env()
    e = np.eye(4)
    lo, hi = w.Xbar0.bounds
    for j in range(4):
        margin = rho if j == 1 else 0.0
        assert hi[j] == pytest.approx(env.x_max[j] - w.Z.support(e[j]) - margin)
        assert lo[j] == pytest.approx(env.x_min[j] + w.Z.support(-e[j]) + margin)
    assert w.V.bounds[1][0] == pytest.approx(env.u_max - w.Z.support(w.K.T @ np.array([1.0, 0.0])))
    # The level chosen for the disturbed experiments, as the flag's help quotes it.
    assert w.Z.support(e[0]) == pytest.approx(0.083, abs=1e-3)
    assert hi[2] == pytest.approx(0.989, abs=1e-3)


# ----------------------------------------------------------------------------
# The closed loop
# ----------------------------------------------------------------------------

def _run(worker, env, target_fn, steps=200, seed=1, vertex_noise=None):
    """Roll the worker out; `vertex_noise` adds w on a random vertex of that
    box after every step (the environment's own disturbance is uniform, and
    the vertices are the worst case; section 5.6 (iii))."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    worker.reset()
    outcomes, ratios, walls = [], [], 0
    for t in range(steps):
        u, info = worker.act(obs, target_fn(obs))
        outcomes.append(info["outcome"])
        ratios.append(info["tube_ratio"])
        obs, _, terminated, truncated, step_info = env.step(u.astype(np.float32))
        if terminated or truncated:
            break
        if vertex_noise is not None:
            b = np.array([vertex_noise[0], vertex_noise[0], vertex_noise[1], vertex_noise[1]])
            env.state = (env.state + b * rng.choice([-1.0, 1.0], size=4)).astype(np.float32)
            obs = env.state.copy()
            y_lo, y_hi = env.width_profile.bounds_at(obs[0])
            walls += int(not (y_lo < obs[1] < y_hi))
    return dict(steps=t + 1, success=step_info["is_success"], contacts=step_info["collision_count"] + walls,
                outcomes=np.array(outcomes), ratios=np.array(ratios), final=obs)


def test_a_manager_that_aims_at_the_openings_threads_the_slalom():
    """Without a disturbance, goals in each gate's opening take the worker
    through both gates without a contact, near the oracle's time (the
    oracle's mean over the spawn grid is ~1013 at ~77 steps; measured here
    2026-09-28: 78 steps, return 1013.7)."""
    w = TubeMPCWorker.from_env(slalom_env())
    out = _run(w, slalom_env(), gate_target)
    assert out["success"] and out["contacts"] == 0
    assert out["steps"] <= 80
    assert np.all(out["outcomes"] == SOLVER)


def test_a_goal_straight_ahead_leaves_the_worker_stuck_at_a_gate():
    """The note's remark 4.6: the worker minimises the distance to its goal
    over one horizon, so a goal straight past a wall is a local minimum it
    cannot leave by itself. Held 1.8 m ahead, it slips through gate 1 and
    stops in front of gate 2 -- safely, which is the point of the tube; the
    manager's job is to not ask for this."""
    w = TubeMPCWorker.from_env(slalom_env())
    out = _run(w, slalom_env(), lambda p: np.array([p[0] + 1.8, p[1]]))
    assert not out["success"] and out["contacts"] == 0
    assert 6.5 < out["final"][0] < 7.0


def test_the_tube_holds_under_worst_case_disturbances():
    """Theorem 4.1 in closed loop: with w on a random vertex of W at every
    step, the real state never touches a wall, the error stays in Z (its
    bounding box, which is what the tube ratio measures), and every step has
    a plan. A short horizon keeps the model inside the size-limited licence
    and makes the run slow, not unsafe."""
    env = slalom_env()                      # the disturbance is injected by _run
    w = TubeMPCWorker.from_env(env, horizon=4, noise_bound_p=NOISE[0], noise_bound_v=NOISE[1])
    for seed in (0, 1):
        out = _run(w, env, gate_target, seed=seed, vertex_noise=NOISE)
        assert out["success"] and out["contacts"] == 0
        assert np.nanmax(out["ratios"]) <= 1.0 + 1e-6
        assert not np.any(out["outcomes"] == EMERGENCY)


def test_the_uniform_environment_disturbance_is_what_the_tube_is_built_for():
    """The same through the environment's own disturbance, which is what
    training sees."""
    env = slalom_env(NOISE)
    w = TubeMPCWorker.from_env(env, horizon=4)
    out = _run(w, env, gate_target)
    assert out["success"] and out["contacts"] == 0
    assert np.nanmax(out["ratios"]) <= 1.0 + 1e-6


def test_the_shifted_candidate_is_feasible_after_a_disturbance():
    """Theorem 4.1's candidate, checked constraint by constraint after a
    disturbance on a vertex of W: dynamics, the tightened boxes, the terminal
    rest state, at least one free face of every obstacle at every stage, and
    x+ - z~_0 in Z through the lifted representation (2.8), by LP."""
    env = slalom_env()
    w = TubeMPCWorker.from_env(env, horizon=4, noise_bound_p=NOISE[0], noise_bound_v=NOISE[1])
    b = np.array([NOISE[0], NOISE[0], NOISE[1], NOISE[1]])
    rng = np.random.default_rng(3)
    x = np.array([3.6, 0.4, 0.8, 0.3])       # just before gate 1
    for _ in range(12):
        target = gate_target(x)
        u, _ = w.act(x, target)
        x = w.A @ x + w.B @ u + b * rng.choice([-1.0, 1.0], size=4)
        cand = w._candidate(0, np.array([*target, 0.0, 0.0]))
        z, v = cand["z"], cand["v"]
        np.testing.assert_allclose(z[1:], z[:-1] @ w.A.T + v @ w.B.T, atol=1e-6)
        lo, hi = w.Xbar0.bounds
        assert np.all(z >= lo - 1e-6) and np.all(z <= hi + 1e-6)
        assert np.all(np.abs(v) <= w.V.bounds[1] + 1e-6)
        np.testing.assert_allclose(z[-1, 2:], 0.0, atol=1e-9)
        for Ob in w.Obar:
            assert np.all(np.any(z @ Ob.H.T >= Ob.h - 1e-4, axis=1))
        # x - z_0 = sum_l L_l w_l with every w_l in W: a feasibility LP.
        L = w.Z.lifted_matrices()
        A_eq = np.hstack(L)
        res = linprog(np.zeros(A_eq.shape[1]), A_eq=A_eq, b_eq=x - z[0],
                      bounds=[(-bi, bi) for _ in L for bi in b], method="highs")
        assert res.status == 0


def test_fixing_unreachable_binaries_changes_no_solution():
    """Remark 3.12's pruning is exact: with every binary free the optimal cost
    is the same, at states near each gate."""
    env = slalom_env()
    fixed = TubeMPCWorker.from_env(env)
    free = TubeMPCWorker.from_env(env)
    free._fix_unreachable = lambda x: [(np.zeros((free.N + 1, Ob.H.shape[0])), np.ones((free.N + 1, Ob.H.shape[0])))
                                       for Ob in free.Obar]
    for x in ([3.2, 0.0, 1.0, 0.0], [4.6, 1.0, 1.1, -0.2], [6.0, 0.5, 1.0, -0.8], [7.5, -1.0, 1.2, 0.0]):
        x = np.array(x)
        target = gate_target(x)
        _, info_fixed = fixed.act(x, target)
        _, info_free = free.act(x, target)
        fixed.reset()
        free.reset()
        assert info_fixed["free_binaries"] < info_free["free_binaries"]
        assert info_fixed["J"] == pytest.approx(info_free["J"], rel=1e-3, abs=1e-6)


def test_a_state_outside_the_feasible_set_brakes_as_an_emergency():
    """A first state no plan can start from -- here inside a gate's wall --
    has no candidate either: the worker reports EMERGENCY and brakes."""
    w = TubeMPCWorker.from_env(slalom_env())
    u, info = w.act(np.array([4.5, -1.0, 1.0, -0.5]), np.array([6.0, 1.0]))
    assert info["outcome"] == EMERGENCY
    np.testing.assert_allclose(u, [-2.5, 2.5])


def test_the_candidate_takes_over_when_the_solver_finds_nothing():
    """Algorithm 3's fallback: when a solve returns no plan (a work limit
    cutting it short, say), the worker applies the shifted candidate,
    u = v~_0 + K (x - z~_0), and keeps it as its plan."""
    env = slalom_env()
    w = TubeMPCWorker.from_env(env)
    obs, _ = env.reset(seed=0)
    u, first = w.act(obs, gate_target(obs))
    obs, *_ = env.step(u.astype(np.float32))
    expected = w._candidate(0, np.array([*gate_target(obs), 0.0, 0.0]))
    w._optimize = lambda mdl: None
    u, second = w.act(obs, gate_target(obs))
    assert first["outcome"] == SOLVER and second["outcome"] == CANDIDATE
    np.testing.assert_allclose(u, expected["v"][0] + w.K @ (obs - expected["z"][0]))
    np.testing.assert_array_equal(w._prev[0][0], expected["z"])


def test_a_solver_plan_worse_than_the_candidate_is_rejected():
    """Algorithm 3, line 8: a plan costlier than the candidate is discarded."""
    env = slalom_env()
    w = TubeMPCWorker.from_env(env)
    obs, _ = env.reset(seed=0)
    u, _ = w.act(obs, gate_target(obs))
    obs, *_ = env.step(u.astype(np.float32))
    cand = w._candidate(0, np.array([*gate_target(obs), 0.0, 0.0]))
    stay = np.tile(obs.astype(float), (w.N + 1, 1))          # a plan that ignores the goal
    w._optimize = lambda mdl: (stay, np.zeros((w.N, 2)))
    _, info = w.act(obs, gate_target(obs))
    assert w.cost(stay, np.zeros((w.N, 2)), np.array([*gate_target(obs), 0.0, 0.0])) > cand["J"]
    assert info["outcome"] == CANDIDATE


def test_the_worker_is_deterministic_and_reset_forgets_the_plan():
    env = slalom_env()
    a, b = TubeMPCWorker.from_env(env), TubeMPCWorker.from_env(env)
    x = np.array([3.0, 0.2, 0.9, 0.1])
    ua, _ = a.act(x, gate_target(x))
    ub, _ = b.act(x, gate_target(x))
    np.testing.assert_array_equal(ua, ub)
    assert a._prev[0] is not None
    a.reset(0)
    assert a._prev[0] is None


def test_slots_do_not_share_plans():
    w = TubeMPCWorker.from_env(slalom_env(), num_slots=2)
    X = np.array([[1.0, 0.0, 0.0, 0.0], [2.0, 0.5, 0.5, 0.0]])
    U, stats = w.act_batch(X, X[:, :2] + [1.8, 0.5])
    assert U.shape == (2, 2) and np.all(stats["outcome"] == SOLVER)
    np.testing.assert_allclose(w._prev[1][0][0], X[1])   # Z = {0}: z_0 = x
    w.reset(1)
    assert w._prev[0] is not None and w._prev[1] is None


def test_the_tunnel_is_a_plain_qp_that_reaches_the_line():
    env = TunnelEnv(config=TunnelEnvConfig())
    w = TubeMPCWorker.from_env(env)
    assert w.Obar == []
    out = _run(w, env, lambda p: np.array([p[0] + 1.8, 0.0]))
    assert out["success"] and out["contacts"] == 0


def test_from_env_takes_the_environments_disturbance_unless_told_otherwise():
    env = slalom_env(NOISE)
    assert TubeMPCWorker.from_env(env, horizon=2).settings["noise_bound_v"] == NOISE[1]
    assert TubeMPCWorker.from_env(env, horizon=2, noise_bound_p=0.0,
                                  noise_bound_v=0.0).settings["noise_bound_v"] == 0.0
