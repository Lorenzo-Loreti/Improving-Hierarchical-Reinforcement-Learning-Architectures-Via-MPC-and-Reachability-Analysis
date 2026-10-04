"""Tests for the tube-MPC worker (algorithms/tube_mpc.py).

Three layers: the set computations of the note's chapter 2 against brute
force or their defining properties (for the polytopes the note uses and for
the disks the environments use since 2026-10-04); the offline sets of the
worker (the obstacles against the environment's own contact rule, the
big-M constants against their definition, the tightened disks); and the
closed loop, where the claims that matter live -- theorem 4.1's candidate is
feasible after any disturbance in W, the real state never touches a wall
nor leaves the speed and thrust disks, the plan is the exact optimum
whatever relaxation found it, and fixing unreachable binaries changes no
solution.

The disturbed worker runs at a short horizon in a few tests, which keeps
them fast; the Gurobi model no longer carries Z's lifted representation
(it lives in the Clarabel problem, see the module docstring), so model size
is not what limits it.

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
import tube_mpc
from tube_mpc import (
    CANDIDATE, EMERGENCY, SOLVER, DiskProduct, Polytope, RPIOuterApprox, TubeMPCWorker,
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


def test_disk_product_support_and_scaling_match_brute_force():
    """W = {||w_p|| <= b_p, ||w_v|| <= b_v}: h_W(a) = b_p ||a_p|| + b_v ||a_v||
    against the max over many points of W, and `scaling` -- the smallest
    alpha with M W in alpha W -- against brute force, exact for a matrix whose
    2x2 blocks are multiples of rotations (every power of A_K is) and an upper
    bound for any other."""
    W = DiskProduct(0.3, 0.7)
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, (20000, 2))
    pts = np.column_stack([0.3 * np.cos(th[:, 0]), 0.3 * np.sin(th[:, 0]),
                           0.7 * np.cos(th[:, 1]), 0.7 * np.sin(th[:, 1])])   # W's extreme points
    for a in rng.standard_normal((10, 4)):
        assert W.support(a) == pytest.approx(np.max(pts @ a), rel=1e-3)
        assert W.support(a) >= np.max(pts @ a) - 1e-12

    def brute(M):
        img = pts @ M.T
        return max(np.linalg.norm(img[:, :2], axis=1).max() / 0.3, np.linalg.norm(img[:, 2:], axis=1).max() / 0.7)

    AK = _worker_ak()
    for M in (AK, np.linalg.matrix_power(AK, 5)):
        assert W.scaling(M) == pytest.approx(brute(M), rel=1e-3)
    generic = rng.standard_normal((4, 4))
    assert W.scaling(generic) >= brute(generic) - 1e-9


def test_rpi_set_of_the_disks_is_robustly_invariant_and_rotation_invariant():
    """The same two checks as for a box W, for the disks: A_K Z (+) W in Z in
    sampled directions (proposition 2.3(c)), and Z within eps of the minimal
    RPI set. And the property the worker's exact disks rest on: Z's support
    is invariant under rotating the plane, diag(R, R) on [p; v]."""
    AK = _worker_ak()
    W = DiskProduct(*NOISE)
    Z = RPIOuterApprox(AK, W, eps=1e-2)
    rng = np.random.default_rng(0)
    gaps = [Z.support(AK.T @ a) + W.support(a) - Z.support(a) for a in rng.standard_normal((500, 4))]
    assert max(gaps) <= 1e-9
    powers = [np.linalg.matrix_power(AK, l) for l in range(300)]
    for a in rng.standard_normal((8, 4)):
        h_inf = sum(W.support(Ai.T @ a) for Ai in powers)
        assert h_inf - 1e-12 <= Z.support(a) <= h_inf + 1e-2 * np.abs(a).sum()
    for a in rng.standard_normal((20, 4)):
        theta = rng.uniform(0, 2 * np.pi)
        Rt = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        assert Z.support(np.kron(np.eye(2), Rt) @ a) == pytest.approx(Z.support(a), rel=1e-12)


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


def test_the_tightened_speed_and_thrust_sets_are_exact_disks():
    """X0 (-) Z on the velocity and U (-) KZ are disks (module docstring):
    the support of Z's velocity part and of K Z is the same in every
    direction, so the tightened radii v_bar = v_max - r_v and u_bar = u_max -
    r_u hold in every direction, not only along the axes."""
    env = slalom_env(NOISE)
    w = TubeMPCWorker.from_env(env, horizon=2)
    for theta in np.linspace(0.0, 2.0 * np.pi, 13):
        n = np.array([np.cos(theta), np.sin(theta)])
        assert env.v_max - w.Z.support(np.concatenate([[0.0, 0.0], n])) == pytest.approx(w.v_bar, abs=1e-12)
        assert env.u_max - w.Z.support(w.K.T @ n) == pytest.approx(w.u_bar, abs=1e-12)
    assert w.u_bar == pytest.approx(1.692, abs=1e-3)


def test_an_asymmetric_closed_loop_is_refused():
    """The disks are exact only if rotating the plane commutes with the
    closed loop; a gain that treats the axes differently must fail loudly."""
    w = TubeMPCWorker.from_env(slalom_env(), horizon=2)
    w.K = w.K * np.array([[1.0], [0.9]])
    w.AK = w.A + w.B @ w.K
    with pytest.raises(ValueError, match="rotation-symmetric"):
        w._check_rotation_symmetry()


def test_the_gauge_oracle_separates_z():
    """gamma_Z(e) <= 1 for points of Z (built from its lifted representation
    with every omega_l in W), > 1 past them, and the returned y has
    h_Z(y) = 1, so y^T e <= 1 is a valid cut that the point violates."""
    w = TubeMPCWorker.from_env(slalom_env(NOISE), horizon=2)
    L = w.Z.lifted_matrices()
    rng = np.random.default_rng(0)
    for _ in range(10):
        th = rng.uniform(0, 2 * np.pi, (len(L), 2))
        r = np.sqrt(rng.uniform(size=(len(L), 2)))
        om = np.column_stack([NOISE[0] * r[:, 0] * np.cos(th[:, 0]), NOISE[0] * r[:, 0] * np.sin(th[:, 0]),
                              NOISE[1] * r[:, 1] * np.cos(th[:, 1]), NOISE[1] * r[:, 1] * np.sin(th[:, 1])])
        e = sum(Ll @ o for Ll, o in zip(L, om))
        gamma, y = w._gauge(e)
        assert gamma <= 1.0 + 1e-7
        assert w.Z.support(y) == pytest.approx(1.0, abs=1e-12)
        gamma2, y2 = w._gauge(1.5 * e / gamma)
        assert gamma2 == pytest.approx(1.5, rel=1e-5) and y2 @ (1.5 * e / gamma) > 1.0


# ----------------------------------------------------------------------------
# The closed loop
# ----------------------------------------------------------------------------

def _edge_of_w(rng, noise):
    """A random extreme point of W: on the edge of both disks, at independent
    angles. W's extreme points are the worst case (section 5.6 (iii)); since
    2026-10-04 they are the disks' circles, no longer a box's vertices, which
    lie outside the disks."""
    th = rng.uniform(0.0, 2.0 * np.pi, 2)
    return np.array([noise[0] * np.cos(th[0]), noise[0] * np.sin(th[0]),
                     noise[1] * np.cos(th[1]), noise[1] * np.sin(th[1])])


def _run(worker, env, target_fn, steps=200, seed=1, edge_noise=None):
    """Roll the worker out; `edge_noise` adds w at a random extreme point of
    W (`_edge_of_w`) after every step (the environment's own disturbance is
    uniform on W). Also tracks the largest real speed and input."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    worker.reset()
    outcomes, ratios, walls, speed, thrust = [], [], 0, 0.0, 0.0
    for t in range(steps):
        u, info = worker.act(obs, target_fn(obs))
        outcomes.append(info["outcome"])
        ratios.append(info["tube_ratio"])
        thrust = max(thrust, float(np.linalg.norm(u)))
        obs, _, terminated, truncated, step_info = env.step(u.astype(np.float32))
        speed = max(speed, float(np.linalg.norm(obs[2:])))
        if terminated or truncated:
            break
        if edge_noise is not None:
            env.state = (env.state + _edge_of_w(rng, edge_noise)).astype(np.float32)
            obs = env.state.copy()
            speed = max(speed, float(np.linalg.norm(obs[2:])))
            y_lo, y_hi = env.width_profile.bounds_at(obs[0])
            walls += int(not (y_lo < obs[1] < y_hi))
    return dict(steps=t + 1, success=step_info["is_success"], contacts=step_info["collision_count"] + walls,
                outcomes=np.array(outcomes), ratios=np.array(ratios), final=obs, speed=speed, thrust=thrust)


def test_a_manager_that_aims_at_the_openings_threads_the_slalom():
    """Without a disturbance, goals in each gate's opening take the worker
    through both gates without a contact, near the oracle's time, at the
    speed limit and inside the thrust disk. Measured 2026-10-04 with the
    disks: 82 steps from this start (seed 1), where the oracle needs 79.
    With the per-axis limits, on 2026-09-28: 78 steps, the oracle's mean
    over the spawn grid ~77 -- steering then cost nothing."""
    from optimal_solver import MinTimeSolver
    env = slalom_env()
    w = TubeMPCWorker.from_env(env)
    out = _run(w, env, gate_target)
    assert out["success"] and out["contacts"] == 0
    env.reset(seed=1)
    assert out["steps"] <= MinTimeSolver().solve(env).length + 4
    assert np.all(out["outcomes"] == SOLVER)
    assert env.v_max - 1e-5 < out["speed"] <= env.v_max + 1e-5
    assert out["thrust"] <= env.u_max + 1e-9


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
    """Theorem 4.1 in closed loop: with w at a random extreme point of W at
    every step, the real state never touches a wall nor exceeds the speed
    limit, the input stays in the thrust disk, the error stays in Z (its
    projections, which is what the tube ratio measures), and every step has
    a plan. A short horizon keeps the test fast and makes the run slow, not
    unsafe."""
    env = slalom_env()                      # the disturbance is injected by _run
    w = TubeMPCWorker.from_env(env, horizon=4, noise_bound_p=NOISE[0], noise_bound_v=NOISE[1])
    for seed in (0, 1):
        out = _run(w, env, gate_target, seed=seed, edge_noise=NOISE)
        assert out["success"] and out["contacts"] == 0
        assert np.nanmax(out["ratios"]) <= 1.0 + 1e-6
        assert not np.any(out["outcomes"] == EMERGENCY)
        assert out["speed"] <= env.v_max + 1e-6 and out["thrust"] <= env.u_max + 1e-9


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
    disturbance at an extreme point of W: dynamics, the position box, the
    speed disk on stages 1..N-1, the thrust disk, the terminal rest state, at
    least one free face of every obstacle at every stage, and x+ - z~_0 in Z
    by its gauge (the separation oracle)."""
    env = slalom_env()
    w = TubeMPCWorker.from_env(env, horizon=4, noise_bound_p=NOISE[0], noise_bound_v=NOISE[1])
    rng = np.random.default_rng(3)
    x = np.array([3.6, 0.4, 0.8, 0.3])       # just before gate 1
    for _ in range(12):
        target = gate_target(x)
        u, _ = w.act(x, target)
        x = w.A @ x + w.B @ u + _edge_of_w(rng, NOISE)
        cand = w._candidate(0, np.array([*target, 0.0, 0.0]))
        z, v = cand["z"], cand["v"]
        np.testing.assert_allclose(z[1:], z[:-1] @ w.A.T + v @ w.B.T, atol=1e-6)
        lo, hi = w.Xbar0.bounds
        assert np.all(z[:, :2] >= lo[:2] - 1e-6) and np.all(z[:, :2] <= hi[:2] + 1e-6)
        assert np.all(np.linalg.norm(z[1:w.N, 2:], axis=1) <= w.v_bar + 1e-6)
        assert np.all(np.linalg.norm(v, axis=1) <= w.u_bar + 1e-6)
        np.testing.assert_allclose(z[-1, 2:], 0.0, atol=1e-9)
        for Ob in w.Obar:
            assert np.all(np.any(z @ Ob.H.T >= Ob.h - 1e-4, axis=1))
        gamma, _ = w._gauge(x - z[0])
        assert gamma <= 1.0 + 1e-6


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
    has no candidate either: the worker reports EMERGENCY and brakes, at full
    thrust straight against the velocity."""
    w = TubeMPCWorker.from_env(slalom_env())
    u, info = w.act(np.array([4.5, -1.0, 1.0, -0.5]), np.array([6.0, 1.0]))
    assert info["outcome"] == EMERGENCY
    np.testing.assert_allclose(u, 2.5 * np.array([-1.0, 0.5]) / np.linalg.norm([1.0, 0.5]))


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
    # Up to the final projection onto the thrust disk, which only absorbs
    # the solvers' tolerances.
    np.testing.assert_allclose(u, expected["v"][0] + w.K @ (obs - expected["z"][0]), rtol=1e-6)
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
    np.testing.assert_allclose(w._prev[1][0][0], X[1], atol=1e-9)   # Z = {0}: z_0 = x
    w.reset(1)
    assert w._prev[0] is not None and w._prev[1] is None


def test_the_tunnel_is_convex_and_reaches_the_line():
    """No obstacles, so no binaries: every step is the convex problem alone,
    solved by Clarabel without a mixed-integer round."""
    env = TunnelEnv(config=TunnelEnvConfig())
    w = TubeMPCWorker.from_env(env)
    assert w.Obar == []
    obs, _ = env.reset(seed=0)
    _, info = w.act(obs, np.array([obs[0] + 1.8, 0.0]))
    assert info["outcome"] == SOLVER and info["cut_rounds"] == 0
    out = _run(w, env, lambda p: np.array([p[0] + 1.8, 0.0]))
    assert out["success"] and out["contacts"] == 0


def test_the_plan_is_the_exact_optimum_whatever_relaxation_found_it():
    """The relaxation only proposes binaries and a lower bound; the plan is
    the exact problem's optimum (module docstring). So a much coarser or
    finer starting polygon -- 4 sides, a square 41% past the disk, or 64 --
    gives the same cost, at states near each gate where binaries are free
    and the speed or thrust disk binds; and the plan satisfies the disks to
    the solver's precision, not to the polygon's."""
    env = slalom_env(NOISE)
    states = ([3.4, 0.6, 1.0, 0.25], [4.6, 1.0, 0.95, -0.3], [6.4, 0.2, 0.8, -0.7], [7.5, -1.0, 0.98, 0.1])
    costs = {}
    for sides in (4, 16, 64):
        polygon = np.array([[np.cos(t), np.sin(t)] for t in 2.0 * np.pi * np.arange(sides) / sides])
        old = tube_mpc._POLYGON
        tube_mpc._POLYGON = polygon
        try:
            w = TubeMPCWorker.from_env(env)
        finally:
            tube_mpc._POLYGON = old
        costs[sides] = []
        for x in states:
            x = np.array(x)
            u, info = w.act(x, gate_target(x))
            w.reset()
            assert info["outcome"] == SOLVER
            assert np.all(np.linalg.norm(info["plan"][1:w.N, 2:], axis=1) <= w.v_bar + 1e-6)
            assert np.linalg.norm(u) <= env.u_max + 1e-9
            costs[sides].append(info["J"])
    np.testing.assert_allclose(costs[4], costs[16], rtol=2e-4)
    np.testing.assert_allclose(costs[64], costs[16], rtol=2e-4)


def test_from_env_takes_the_environments_disturbance_unless_told_otherwise():
    env = slalom_env(NOISE)
    assert TubeMPCWorker.from_env(env, horizon=2).settings["noise_bound_v"] == NOISE[1]
    assert TubeMPCWorker.from_env(env, horizon=2, noise_bound_p=0.0,
                                  noise_bound_v=0.0).settings["noise_bound_v"] == 0.0
