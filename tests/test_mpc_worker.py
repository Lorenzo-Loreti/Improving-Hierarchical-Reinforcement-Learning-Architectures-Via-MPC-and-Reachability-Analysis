"""Tests of the tube-MPC Worker with its solvers (decision log D31-D35, D37-D39, D45-D47); they need cvxpy, Clarabel
and Gurobi.

The closed loops run on the environment's own physics (:func:`hrlmpc.physics.physics_step`),
with scripted targets in place of a Manager, one per segment of ``H = 10``
steps, and with chosen disturbances, so the claims of D31-D33 and D37 are
checked where they matter: the real state never touches a wall, the error to
the plan stays in the tube (by its gauge), the input never needs the
saturation, the shifted plan stays feasible, the plans are optimal, a
reachable target is passed when the segment ends and an unreachable one is
projected onto the reachable set. Every search reports how it ended (D47),
consistently with its outcome; the solvers minimise the problem's cost, the
regularizer of D46 included. Gurobi's pip licence is size-limited: a test
whose model exceeds it is skipped with that reason.

The tolerances follow the solvers': a plan from Clarabel meets its constraints
to about 1e-8, the relaxation's own plan (module docstring of
:mod:`hrlmpc.mpc_worker`) the disks and ``Z`` to 1e-6 and the faces to
``face_slack``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")
gp = pytest.importorskip("gurobipy")
pytest.importorskip("clarabel")

from hrlmpc.config import EnvConfig, load_env_config  # noqa: E402
from hrlmpc.env import ObservationMap  # noqa: E402
from hrlmpc.geometry import Layout  # noqa: E402
from hrlmpc.mpc_command import (  # noqa: E402
    CANDIDATE,
    CONVERGED,
    EMERGENCY,
    EMPTY,
    ERROR,
    FIXED,
    FOUND,
    KEPT,
    RELAXATION,
    ROUNDS,
    SOLVER,
    UNSOLVED,
    MPCCommand,
)
from hrlmpc.mpc_problem import MPCProblem, MPCSettings  # noqa: E402
from hrlmpc.mpc_worker import MPCWorker, ReachSupport, _GaugeOracle, reachable_targets  # noqa: E402
from hrlmpc.physics import physics_step  # noqa: E402
from hrlmpc.reachability import support_directions  # noqa: E402

GRB = gp.GRB
TOL = 1e-5
"""Violations and gauge excess accepted in the closed loops (the solvers' tolerances; module docstring)."""

H = 10
"""Steps per segment, as in the configurations (D27): the scripted targets change every ``H`` steps."""


@contextlib.contextmanager
def _licence() -> Iterator[None]:
    """Skip, rather than fail, when Gurobi's size-limited pip licence refuses the model."""
    try:
        yield
    except gp.GurobiError as error:
        if "size-limited" in str(error) or getattr(error, "errno", None) == 10010:
            pytest.skip(f"Gurobi's size-limited licence: {error}")
        raise


def _setup(name: str, d_bar: float = 0.0, **settings: Any) -> tuple[EnvConfig, Layout, MPCProblem]:
    """The problem of a layout; by default with the horizon training uses, the automatic one of D45 (``H`` plus
    the steps to stop: 15 at ``d_bar = 0``, 16 at 0.5, 17 at 1.0), and the regularizer of D46."""
    cfg = load_env_config(f"configs/env/{name}.yaml")
    cfg = dataclasses.replace(cfg, physics=dataclasses.replace(cfg.physics, d_bar=d_bar))
    layout = Layout.from_config(cfg.geometry)
    return cfg, layout, MPCProblem(cfg.physics, layout, MPCSettings(**settings), segment_steps=H)


WAYPOINTS = np.array([[3.5, 1.0], [5.5, 1.0], [6.5, -1.0], [8.5, -1.0], [10.6, -1.0]])
"""Targets through the openings of the slalom's gates (open for 0.25 <= y <= 1.75, then for -1.75 <= y <= -0.25)."""


def _through_the_gates(p: np.ndarray) -> np.ndarray:
    ahead = WAYPOINTS[WAYPOINTS[:, 0] > p[0] + 0.3]
    return ahead[0] if len(ahead) else WAYPOINTS[-1]


def _straight_ahead(p: np.ndarray) -> np.ndarray:
    return p + np.array([1.8, 0.0])


def _toward_the_walls(layout: Layout, p: np.ndarray, d_bar: float, k: int, rng: np.random.Generator) -> np.ndarray:
    """A disturbance pushing toward the nearest point of a wall on most steps, random on the others."""
    if k % 4 == 3:
        w = rng.normal(size=2)
        return d_bar * rng.uniform() * w / np.linalg.norm(w)
    f = layout.faces
    along = np.clip(((p - f.start) * f.tangent).sum(axis=1), 0.0, f.length)
    nearest = f.start + along[:, None] * f.tangent
    distance = np.linalg.norm(nearest - p, axis=1)
    j = int(np.argmin(distance))
    return d_bar * (nearest[j] - p) / max(float(distance[j]), 1e-12)


def _run(
    problem: MPCProblem,
    layout: Layout,
    cfg: EnvConfig,
    start: tuple[float, float],
    targets: Callable[[np.ndarray], np.ndarray],
    *,
    steps: int = 200,
    disturbance: Callable[[np.ndarray, int], np.ndarray] | None = None,
    worker: MPCWorker | None = None,
) -> dict[str, Any]:
    """A closed loop with a target set every ``H`` steps; returns the trajectory and every check.

    Per step: what the Worker reported (outcome, tube ratio, free binaries,
    cut rounds, invalid shifted plans, solver errors, cost, the search's exit and
    its exact solves without a plan); the violations of the
    applied plan; the norm of the input before its projection onto
    ``U``, at most ``a_max`` by D33 so that the saturation never acts; contact
    and the speed limiter in the physics; and, with a tube, the gauge of the
    error to the plan's prediction after the step, at most 1 when it stayed in ``Z``.
    """
    worker = MPCWorker(problem) if worker is None else worker
    p, v = np.array(start, dtype=np.float64), np.zeros(2)
    target = targets(p)
    names = ("outcome", "ratio", "free", "rounds", "invalid", "errors", "cost", "exit", "inaccurate", "infeasible",
             "uncertified", "violations", "thrust", "contact", "limited", "gauge")
    log: dict[str, Any] = {name: [] for name in names}
    log.update(p=[p.copy()], success=False)
    with _licence():
        for k in range(steps):
            if k % H == 0:
                target = targets(p)
            x = np.concatenate([p, v])
            u, info = worker.act(x, target, H - k % H)
            for name, value in (("outcome", info.outcome), ("ratio", info.tube_ratio), ("free", info.free_binaries),
                                ("rounds", info.cut_rounds), ("invalid", info.invalid_candidate),
                                ("errors", info.solver_errors), ("cost", info.cost), ("exit", info.exit),
                                ("inaccurate", info.exact_inaccurate), ("infeasible", info.exact_infeasible),
                                ("uncertified", info.exact_uncertified)):
                log[name].append(value)
            plan = worker._slots[0].plan  # the applied plan; None after a brake
            if plan is not None:
                log["violations"].append(problem.violations(x, *plan))
                log["thrust"].append(float(np.linalg.norm(plan[1][0] + problem.K @ (x - plan[0][0]))))
            d = np.zeros(2) if disturbance is None else disturbance(p, k)
            step = physics_step(p[None], v[None], u[None], d[None], cfg.physics, layout)
            log["contact"].append(bool(step.contact[0] or step.wall_reaction[0]))
            log["limited"].append(bool(np.linalg.norm(step.v_limited[0] - step.v_candidate[0]) > 1e-9))
            p, v = step.p[0], step.v[0]
            log["p"].append(p.copy())
            if plan is not None and worker._gauge is not None:
                log["gauge"].append(worker._gauge(np.concatenate([p, v]) - plan[0][1])[0])
            if p[0] >= cfg.geometry.goal_x:
                log["success"] = True
                break
    log["p"] = np.array(log["p"])
    return log


def _worst(log: dict[str, Any]) -> dict[str, float]:
    """The largest violation of each group of constraints over the applied plans."""
    return {key: max(v[key] for v in log["violations"]) for key in log["violations"][0]}


def _consistent(log: dict[str, Any]) -> None:
    """D47: every step's exit agrees with its free binaries, its outcome, its solver failures, its rounds and its
    exact solves."""
    for free, end, outcome, errors, rounds, exact in zip(
        log["free"], log["exit"], log["outcome"], log["errors"], log["rounds"],
        zip(log["inaccurate"], log["infeasible"], log["uncertified"], strict=True), strict=True
    ):
        assert end in ((FIXED, UNSOLVED, ERROR) if free == 0 else (CONVERGED, RELAXATION, ROUNDS, EMPTY, ERROR))
        assert (end in FOUND) == (outcome in (SOLVER, KEPT))  # a plan from the solvers, applied or benchmarked
        assert end != ERROR or errors > 0
        assert (rounds == 0) == (free == 0)  # Gurobi solves exactly when some binary is free
        if free == 0:  # one exact solve, certified by none: a plan, or a status
            assert exact[2] == 0 and sum(exact) == (0 if end == FIXED else int(end == UNSOLVED))
        else:  # at most one exact solve per round
            assert sum(exact) <= rounds


def _nominal(log: dict[str, Any], problem: MPCProblem) -> None:
    """What every closed loop with its disturbances in ``D`` must show."""
    _consistent(log)
    assert not any(log["contact"])  # the wall reaction never acted, and no state lay on a face
    assert EMERGENCY not in log["outcome"]
    assert sum(log["errors"]) == 0 and sum(log["invalid"]) == 0
    worst = _worst(log)
    assert worst.pop("obstacles") <= problem.face_slack and max(worst.values()) <= TOL
    assert max(log["thrust"]) <= problem.params.a_max * (1.0 + TOL)  # the saturation of U never acts (D33)


def test_the_tunnel_is_convex_and_reaches_the_goal() -> None:
    cfg, layout, problem = _setup("tunnel")
    log = _run(problem, layout, cfg, (1.0, 0.0), _straight_ahead)
    assert log["success"]
    assert set(log["outcome"]) == {SOLVER}
    assert max(log["free"]) == 0 and max(log["rounds"]) == 0  # no binaries: Clarabel alone
    assert set(log["exit"]) == {FIXED} and sum(log["inaccurate"]) == sum(log["infeasible"]) == 0  # D47
    assert sum(log["uncertified"]) == 0
    _nominal(log, problem)
    assert max(_worst(log).values()) <= 1e-6  # Clarabel's plans alone
    assert len(log["outcome"]) == 78  # the bound of F7 from (1, 0): the waypoint Worker drives at full speed (D37)


def test_a_manager_that_aims_at_the_openings_threads_the_slalom() -> None:
    cfg, layout, problem = _setup("slalom")
    log = _run(problem, layout, cfg, (1.0, 0.0), _through_the_gates)
    assert log["success"]
    _nominal(log, problem)
    assert max(log["free"]) > 0  # the gates' binaries were live
    assert {CONVERGED, RELAXATION} & set(log["exit"])  # and Gurobi's searches found plans (D47)
    costs = np.array(log["cost"])
    for start in range(0, len(costs) - 1, H):  # D37: without disturbance the cost does not grow within a segment
        segment = costs[start : start + H]
        assert np.all(np.diff(segment) <= 1e-6 * (1.0 + segment[:-1])), start


def _segment(worker: MPCWorker, problem: MPCProblem, cfg: EnvConfig, layout: Layout, x: np.ndarray,
             target: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """One segment of ``H`` steps toward ``target``, with random disturbances on the edge of ``D`` if ``rng``."""
    with _licence():
        for j in range(H):
            u, info = worker.act(x, target, H - j)
            assert info.outcome == SOLVER and info.solver_errors == 0
            w = np.zeros(2) if rng is None else rng.normal(size=2)
            d = problem.params.d_bar * w / max(float(np.linalg.norm(w)), 1e-12)
            step = physics_step(x[None, :2], x[None, 2:], u[None], d[None], cfg.physics, layout)
            assert not (step.contact[0] or step.wall_reaction[0])
            x = np.concatenate([step.p[0], step.v[0]])
    return x


@pytest.mark.parametrize("d_bar", [0.0, 0.5])
def test_a_reachable_target_is_passed_when_the_segment_ends(d_bar: float) -> None:
    """D37: the agent is at the target after H steps, up to the tube, and still moving: a waypoint, not a rest."""
    cfg, layout, problem = _setup("tunnel", d_bar=d_bar)
    target = np.array([1.5, 0.2])  # 0.54 m from the start: reachable in 10 steps from rest (0.95 m)
    rng = None if d_bar == 0.0 else np.random.default_rng(5)
    x = _segment(MPCWorker(problem), problem, cfg, layout, np.array([1.0, 0.0, 0.0, 0.0]), target, rng)
    assert np.linalg.norm(x[:2] - target) <= problem.tube.position_radius + 2e-3
    assert np.linalg.norm(x[2:]) > 0.3  # it passes the target (about 0.69 m/s without disturbance)


def test_an_unreachable_target_is_projected_onto_the_reachable_set() -> None:
    """D37: a target beyond reach leaves the agent at the reachable point nearest to it, here the farthest point
    straight ahead: full thrust from rest covers the displacement of the bound of F7 in 10 steps."""
    cfg, layout, problem = _setup("tunnel")
    x = _segment(MPCWorker(problem), problem, cfg, layout, np.array([1.0, 0.0, 0.0, 0.0]), np.array([4.0, 0.0]))
    speed, travel = 0.0, 0.0
    for _ in range(H):
        speed = min((1.0 - problem.params.gamma * problem.dt) * speed + problem.dt * problem.params.a_max,
                    problem.params.v_max)
        travel += problem.dt * speed
    assert x[0] - 1.0 == pytest.approx(travel, abs=1e-4) and abs(x[1]) <= 1e-6


def test_a_target_straight_ahead_leaves_the_worker_in_front_of_the_second_gate() -> None:
    cfg, layout, problem = _setup("slalom")
    log = _run(problem, layout, cfg, (1.0, 1.0), _straight_ahead)
    assert not log["success"]
    _nominal(log, problem)
    assert 6.0 < log["p"][-1, 0] < 7.0 - problem.margin + 1e-6  # stopped before O4, at rest


@pytest.mark.parametrize("d_bar", [0.5, 1.0])
def test_the_tube_holds_under_worst_case_disturbances(d_bar: float) -> None:
    cfg, layout, problem = _setup("slalom", d_bar=d_bar)
    rng = np.random.default_rng(1)
    log = _run(problem, layout, cfg, (1.0, 0.0), _through_the_gates,
               disturbance=lambda p, k: _toward_the_walls(layout, p, d_bar, k, rng))
    _nominal(log, problem)
    assert max(log["gauge"]) <= 1.0 + TOL  # every error to the plan's prediction stayed in Z
    assert np.nanmax(np.array(log["ratio"], dtype=float)) <= 1.0 + 1e-6


def test_the_tube_holds_with_the_disturbance_along_the_velocity() -> None:
    """D33: cruising at the tightened speed, pushed mostly along the velocity, the tube holds without contact."""
    cfg, layout, problem = _setup("tunnel", d_bar=1.0)
    rng = np.random.default_rng(2)

    def along(p: np.ndarray, k: int) -> np.ndarray:
        return 1.0 * (np.array([1.0, 0.0]) if rng.uniform() < 0.8 else -np.array([1.0, 0.0]))

    log = _run(problem, layout, cfg, (0.0, 0.0), _straight_ahead, disturbance=along)
    assert log["success"]
    _nominal(log, problem)
    assert max(log["gauge"]) <= 1.0 + TOL


@pytest.mark.parametrize("d_bar", [0.5, 1.0])
def test_the_shifted_plan_is_feasible_after_every_disturbance_in_d(d_bar: float) -> None:
    """Theorem 4.1, constraint by constraint: after a disturbance on the edge of ``D``, the shift of the applied
    plan meets every constraint at the new state, the tube constraint by its gauge."""
    cfg, layout, problem = _setup("slalom", d_bar=d_bar)
    worker = MPCWorker(problem)
    rng = np.random.default_rng(3)
    p, v = np.array([1.0, 0.0]), np.zeros(2)
    free = []
    with _licence():
        for k in range(80):  # through the first gate
            x = np.concatenate([p, v])
            u, info = worker.act(x, _through_the_gates(p), H - k % H)
            assert info.outcome != EMERGENCY and info.invalid_candidate == 0 and info.solver_errors == 0
            free.append(info.free_binaries)
            w = rng.normal(size=2)
            step = physics_step(p[None], v[None], u[None], (d_bar * w / np.linalg.norm(w))[None], cfg.physics, layout)
            p, v = step.p[0], step.v[0]
            z_c, v_c = problem.candidate(*worker._slots[0].plan)
            violations = problem.violations(np.concatenate([p, v]), z_c, v_c)
            assert violations.pop("obstacles") <= problem.face_slack and max(violations.values()) <= TOL
            assert worker._gauge(np.concatenate([p, v]) - z_c[0])[0] <= 1.0 + TOL
    assert max(free) > 0  # the gate's binaries were live


@pytest.mark.parametrize("d_bar", [0.5, 1.0])
def test_the_gauge_oracle_separates_z(d_bar: float) -> None:
    """``gamma_Z(e) <= 1`` on points of ``Z`` built from its lifted representation, ``> 1`` beyond them, and
    the returned ``y`` has ``h_Z(y) = 1``, so ``y' e <= 1`` is a valid cut that a point beyond violates."""
    _, _, problem = _setup("tunnel", d_bar=d_bar)
    tube, oracle = problem.tube, _GaugeOracle(problem)
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 4))
    for extreme in (False, True):
        for _ in range(5):
            d = rng.normal(size=(tube.num_terms, 2))
            d *= tube.d_bar / np.linalg.norm(d, axis=1, keepdims=True)
            eta = rng.normal(size=4)
            eta /= np.linalg.norm(eta)
            if not extreme:
                d *= rng.uniform(size=(tube.num_terms, 1))
                eta *= rng.uniform()
            e = tube.lifted(d, eta)
            gamma, y = oracle(e)
            assert gamma <= 1.0 + 1e-7
            assert gamma >= np.max(a @ e / tube.support(a)) - 1e-6  # every direction bounds the gauge from below
            assert float(tube.support(y)) == pytest.approx(1.0, abs=1e-12)
            beyond = 1.5 * e / gamma
            gamma_beyond, y_beyond = oracle(beyond)
            assert gamma_beyond == pytest.approx(1.5, rel=1e-5) and y_beyond @ beyond > 1.0
    gamma, y = oracle(np.zeros(4))
    assert gamma == 0.0 and float(tube.support(y)) == pytest.approx(1.0, abs=1e-12)  # no solve for a zero error


def test_a_tube_of_its_tail_alone_is_solved() -> None:
    """For a tiny ``d_bar`` the tube has no disturbance term (``s = 0``): the problem keeps its tail alone."""
    cfg, layout, problem = _setup("tunnel", d_bar=1e-4)
    assert problem.tube.num_terms == 0 and not problem.tube.is_zero
    log = _run(problem, layout, cfg, (1.0, 0.0), _straight_ahead, steps=30,
               disturbance=lambda p, k: 1e-4 * np.array([np.cos(k), np.sin(k)]))
    assert set(log["outcome"]) == {SOLVER} and sum(log["errors"]) == 0 and not any(log["contact"])


def _exhaustive_cost(problem: MPCProblem, x: np.ndarray, target: np.ndarray, steps_left: int) -> float:
    """The optimum over every choice of one face per obstacle and segment, each a convex problem (cvxpy, Clarabel)."""
    N, tube = problem.N, problem.tube
    bounds = problem.fixed_binaries(x)
    options = []
    for (lb, ub), obstacle in zip(bounds, problem.obstacles, strict=True):
        for k in range(N):
            if np.array_equal(lb[k], ub[k]):
                options.append([int(np.argmax(lb[k]))])
            else:
                options.append(list(range(obstacle.num_faces)))
    weights = problem.settings
    best = np.inf
    for choice in itertools.product(*options):
        z, v = cp.Variable((N + 1, 4)), cp.Variable((N, 2))
        cons = [z[1:] == z[:-1] @ problem.A.T + v @ problem.B.T, z[N, 2:] == 0,
                z[:, :2] @ problem.arena_normals.T <= np.tile(problem.arena_bounds, (N + 1, 1)),
                cp.norm(v, 2, axis=1) <= problem.u_bar, cp.norm(z[1:N, 2:], 2, axis=1) <= problem.v_bar]
        if tube.is_zero:
            cons.append(z[0] == x)
        else:
            d, eta = cp.Variable((tube.num_terms, 2)), cp.Variable(4)
            cons += [z[0] + sum(tube.terms[lag] @ d[lag] for lag in range(tube.num_terms)) + tube.tail @ eta == x,
                     cp.norm(d, 2, axis=1) <= tube.d_bar, cp.norm(eta, 2) <= 1.0]
        it = iter(choice)
        for obstacle in problem.obstacles:
            for k in range(N):
                j = next(it)
                n, b = obstacle.normals[j], obstacle.bounds[j]
                cons += [z[k, :2] @ n >= b, z[k + 1, :2] @ n >= b]
        cost = weights.target_weight * cp.sum_squares(z[steps_left, :2] - target)  # D37
        cost = cost + weights.effort_weight * cp.sum_squares(v[:steps_left])
        if problem.regularized:  # D46
            cost = cost + weights.start_weight * cp.sum_squares(z[0] - x)
        prob = cp.Problem(cp.Minimize(cost), cons)
        prob.solve(solver=cp.CLARABEL)
        if prob.status == "optimal":
            best = min(best, float(prob.value))
    return best


OPTIMALITY_STATES = {
    0.0: (([3.95, 0.3, 0.3, 0.0], [5.4, 1.2], 2), ([6.95, -0.2, 0.3, -0.3], [7.5, -1.2], 3)),
    0.5: (([3.9, 0.3, 0.3, 0.0], [5.4, 1.2], 2), ([6.85, -0.2, 0.3, -0.3], [7.5, -1.2], 3)),
}
"""States slow enough to rest within ``N = 3`` steps and so close to a corner of a gate that the stopping-limited
reach boxes leave 9 binaries free; certified feasible (with ``z_0 = x`` and inscribed 32-gons) by a mixed-integer
linear program in SciPy. The waypoint lies before the end of the horizon, then at its end."""


@pytest.mark.parametrize("d_bar", [0.0, 0.5])
def test_the_plan_is_the_optimum_over_every_choice_of_faces(d_bar: float) -> None:
    _, _, problem = _setup("slalom", d_bar=d_bar, horizon=3)
    for x, target, left in OPTIMALITY_STATES[d_bar]:
        x_arr = np.array(x)
        assert problem.free_binaries(problem.fixed_binaries(x_arr)) > 0
        with _licence():
            u, info = MPCWorker(problem).act(x_arr, target, left)
        assert info.outcome == SOLVER and info.free_binaries > 0 and info.solver_errors == 0
        best = _exhaustive_cost(problem, x_arr, np.array(target), left)
        assert info.cost <= best + 1e-4 * abs(best) + 1e-6
        assert info.cost >= best - 1e-6 * (1.0 + abs(best))


def test_the_plan_does_not_depend_on_the_relaxations_polygon(monkeypatch: pytest.MonkeyPatch) -> None:
    """The relaxation only proposes binaries and a bound; the plan is the exact optimum. So a polygon of 4 sides
    (a square 41% past the disks), 16 or 64 gives the same cost near the gates, where binaries are free, and the
    plan meets the disks to the solvers' precision, not to the polygon's."""
    import hrlmpc.mpc_worker as module

    _, _, problem = _setup("slalom", d_bar=0.5)
    states = ([3.4, 1.0, 1.0, 0.0], [4.6, 1.0, 1.0, -0.2], [6.4, 0.2, 0.8, -0.7], [7.5, -1.0, 0.98, 0.1])
    costs: dict[int, list[float]] = {}
    for sides in (4, 16, 64):
        angles = 2.0 * np.pi * np.arange(sides) / sides
        monkeypatch.setattr(module, "_POLYGON", np.column_stack([np.cos(angles), np.sin(angles)]))
        costs[sides] = []
        with _licence():
            worker = MPCWorker(problem)
            for x in states:
                _, info = worker.act(np.array(x), _through_the_gates(np.array(x[:2])), H)
                z, v = worker._slots[0].plan
                worker.reset()
                assert info.outcome == SOLVER and info.free_binaries > 0 and info.solver_errors == 0
                assert np.linalg.norm(z[1 : problem.N, 2:], axis=1).max() <= problem.v_bar * (1.0 + TOL)
                assert np.linalg.norm(v, axis=1).max() <= problem.u_bar * (1.0 + TOL)
                costs[sides].append(info.cost)
    np.testing.assert_allclose(costs[4], costs[16], rtol=2e-4)
    np.testing.assert_allclose(costs[64], costs[16], rtol=2e-4)


NEAR_THE_GATE = np.array([3.75, 0.45, 1.0, 0.0])
"""A state at ``d_bar = 0.5`` with ``N = 5`` whose stopping-limited reach boxes leave 12 binaries of the first gate
free, so that every solve calls Gurobi; certified feasible like :data:`OPTIMALITY_STATES`."""


def test_fixing_unreachable_binaries_changes_no_solution(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, problem = _setup("slalom", d_bar=0.5, horizon=5)
    states = [(NEAR_THE_GATE, [5.4, 1.2]), (np.array([1.0, 0.0, 0.0, 0.0]), [2.8, 0.0])]
    assert problem.free_binaries(problem.fixed_binaries(states[0][0])) > 0  # some fixed, some free
    assert problem.free_binaries(problem.fixed_binaries(states[1][0])) == 0  # every one fixed
    with _licence():
        fixed = [MPCWorker(problem).act(np.array(x), g, 3)[1].cost for x, g in states]

        def all_free(x: Any) -> list[tuple[np.ndarray, np.ndarray]]:
            return [(np.zeros((problem.N, o.num_faces)), np.ones((problem.N, o.num_faces))) for o in problem.obstacles]

        monkeypatch.setattr(problem, "fixed_binaries", all_free)
        free = [MPCWorker(problem).act(np.array(x), g, 3)[1].cost for x, g in states]
    np.testing.assert_allclose(free, fixed, rtol=2e-4)


def test_a_state_that_cannot_avoid_a_wall_brakes_as_an_emergency() -> None:
    _, _, problem = _setup("slalom")
    x = np.array([3.95, -1.0, 1.2, 0.0])  # 5 cm from O1 at full speed: no plan stops in time
    with _licence():
        u, info = MPCWorker(problem).act(x, [6.0, -1.0], H)
    assert info.outcome == EMERGENCY and info.plan is None and np.isnan(info.cost)
    assert info.solver_errors == 0  # infeasible is no failure
    # D47: O1's binaries are free, and the first relaxation is already infeasible, so no exact solve follows
    assert info.free_binaries > 0 and info.exit == EMPTY and info.cut_rounds == 1
    assert info.exact_inaccurate == info.exact_infeasible == info.exact_uncertified == 0
    np.testing.assert_allclose(u, problem.emergency_input(x))


def test_a_state_that_cannot_avoid_the_tunnels_wall_has_no_plan() -> None:
    """D47: without binaries Clarabel alone finds the problem infeasible (certified or almost): no plan, no
    failure, one infeasible exact solve. 5 cm below the upper wall at full speed upward, the next position lies
    3.9 cm past it at least (``0.114 m`` of drift against ``0.025 m`` of full thrust)."""
    _, _, problem = _setup("tunnel")
    x = np.array([5.0, 1.95, 0.0, 1.2])
    with _licence():
        u, info = MPCWorker(problem).act(x, [6.0, 1.0], H)
    assert info.outcome == EMERGENCY and info.free_binaries == 0 and info.cut_rounds == 0
    assert info.exit == UNSOLVED and info.exact_infeasible == 1 and info.exact_inaccurate == 0
    assert info.solver_errors == 0
    np.testing.assert_allclose(u, problem.emergency_input(x))


class _Status:
    """Stands in for a cvxpy problem whose solve ends with a given status and without a plan."""

    def __init__(self, status: str) -> None:
        self.status = status

    def solve(self, **kwargs: Any) -> None:
        pass


@pytest.mark.parametrize(
    ("status", "inaccurate"),
    [("optimal_inaccurate", 1), ("user_limit", 1), ("infeasible", 0), ("infeasible_inaccurate", 0)],
)
def test_an_exact_solve_without_a_plan_is_counted_by_its_status(status: str, inaccurate: int,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """D47: an exact solve that ends with an inaccurate optimum or at Clarabel's iteration limit (cvxpy's
    ``user_limit``), where a plan may exist, counts as inaccurate; one that finds the binaries infeasible,
    certified or almost, as infeasible; neither is a solver failure."""
    _, _, problem = _setup("tunnel")
    worker = MPCWorker(problem)
    monkeypatch.setattr(worker._exact, "prob", _Status(status))
    with _licence():
        _, info = worker.act(np.array([1.0, 0.0, 0.3, 0.0]), [2.8, 0.0], H)
    assert info.outcome == EMERGENCY and info.exit == UNSOLVED and info.solver_errors == 0
    assert (info.exact_inaccurate, info.exact_infeasible, info.exact_uncertified) == (inaccurate, 1 - inaccurate, 0)
    worker.close()


def test_a_search_out_of_rounds_ends_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    """D47: with no round of cuts allowed, a search with free binaries ends with its rounds exhausted, before
    any mixed-integer solve."""
    import hrlmpc.mpc_worker as module

    _, _, problem = _setup("slalom", d_bar=0.5, horizon=5)
    monkeypatch.setattr(module, "MAX_CUT_ROUNDS", 0)
    with _licence():
        _, info = MPCWorker(problem).act(NEAR_THE_GATE, [5.4, 1.2], 5)
    assert info.free_binaries > 0 and info.outcome == EMERGENCY  # a fresh Worker: no shifted plan
    assert info.exit == ROUNDS and info.cut_rounds == 0 and info.solver_errors == 0


def test_the_shifted_plan_takes_over_when_the_solvers_find_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, problem = _setup("tunnel")
    worker = MPCWorker(problem)
    x = np.array([1.0, 0.0, 0.0, 0.0])
    with _licence():
        worker.act(x, [2.8, 0.0], H)
    plan = worker._slots[0].plan
    assert plan is not None
    monkeypatch.setattr(worker, "_optimize", lambda *args: None)
    x_next = plan[0][1]  # exactly where the plan said: no disturbance
    u, info = worker.act(x_next, [2.8, 0.0], H - 1)
    assert info.outcome == CANDIDATE and info.invalid_candidate == 0
    z_c, v_c = problem.candidate(*plan)
    np.testing.assert_allclose(u, problem.input(x_next, z_c, v_c))


def _costlier(problem: MPCProblem, worker: MPCWorker, target: list[float]) -> tuple[Any, Any, Any]:
    """After the first step of a segment, from rest: the plan, its shift and a plan one metre off the target,
    costlier than the shift at the next step."""
    with _licence():
        worker.act(np.array([1.0, 0.0, 0.0, 0.0]), target, H)
    plan = worker._slots[0].plan
    assert plan is not None
    z_c, v_c = problem.candidate(*plan)
    return plan, (z_c, v_c), (z_c + np.array([0.0, 1.0, 0.0, 0.0]), v_c)


@pytest.mark.parametrize("d_bar", [0.0, 0.5])
def test_a_feasible_shifted_plan_cheaper_than_the_solvers_plan_is_kept(d_bar: float,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, problem = _setup("tunnel", d_bar=d_bar)
    worker, target = MPCWorker(problem), [2.8, 0.0]
    plan, shifted, costlier = _costlier(problem, worker, target)
    monkeypatch.setattr(worker, "_optimize", lambda *args: costlier)
    # where the plan said, or (with a tube) a small disturbance away, well inside Z (gauge about 0.03)
    x_next = plan[0][1] + (np.array([0.0, 0.0, 0.0, 1e-3]) if d_bar > 0.0 else 0.0)
    u, info = worker.act(x_next, target, H - 1)
    assert info.outcome == KEPT and info.invalid_candidate == 0 and info.solver_errors == 0
    assert info.cost == pytest.approx(problem.cost(*shifted, target, H - 1, x=x_next))  # with a tube, D46's term
    np.testing.assert_allclose(u, problem.input(x_next, *shifted))


@pytest.mark.parametrize("d_bar", [0.0, 0.5])
def test_a_shifted_plan_outside_its_tube_is_only_the_last_resort(d_bar: float,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """After a disturbance outside ``D`` the shifted plan is infeasible: no benchmark, only the fallback before
    the brake, and reported."""
    _, _, problem = _setup("tunnel", d_bar=d_bar)
    worker, target = MPCWorker(problem), [2.8, 0.0]
    plan, shifted, costlier = _costlier(problem, worker, target)
    x_next = plan[0][1] + np.array([0.0, 0.0, 0.0, 0.5])  # pushed sideways at 0.5 m/s: r_v is 0.17 m/s at most
    monkeypatch.setattr(worker, "_optimize", lambda *args: costlier)
    u, info = worker.act(x_next, target, H - 1)
    assert info.outcome == SOLVER and info.invalid_candidate == 1  # applied, though costlier
    np.testing.assert_allclose(u, problem.input(x_next, *costlier))
    worker._slots[0].plan = plan  # the same step, now without a plan from the solvers
    monkeypatch.setattr(worker, "_optimize", lambda *args: None)
    u, info = worker.act(x_next, target, H - 1)
    assert info.outcome == CANDIDATE and info.invalid_candidate == 1
    np.testing.assert_allclose(u, problem.input(x_next, *shifted))


class _FailingProblem:
    """Stands in for a cvxpy problem whose solver fails."""

    status = None

    def solve(self, **kwargs: Any) -> None:
        raise cp.SolverError("Solver 'CLARABEL' failed.")


def _no_solution(slot: Any) -> None:
    """A relaxation that fails with a genuine Gurobi error: there is no solution to read (errno 10005)."""
    slot.m.reset(0)
    _ = slot.z.X


def test_a_gurobi_failure_is_counted_and_the_shifted_plan_takes_over(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, problem = _setup("slalom", d_bar=0.5, horizon=5)
    worker, target = MPCWorker(problem), [5.4, 1.2]
    with _licence():
        _, first = worker.act(NEAR_THE_GATE, target, 5)
        plan = worker._slots[0].plan
        monkeypatch.setattr(worker, "_relax", _no_solution)
        u, info = worker.act(plan[0][1], target, 4)
        fresh = MPCWorker(problem)
        monkeypatch.setattr(fresh, "_relax", _no_solution)
        _, alone = fresh.act(NEAR_THE_GATE, target, 5)
    assert first.outcome == SOLVER and first.solver_errors == 0 and first.exit in (CONVERGED, RELAXATION)
    assert info.free_binaries > 0 and info.outcome == CANDIDATE
    assert info.solver_errors == 1 and info.invalid_candidate == 0
    assert info.exit == ERROR and info.cut_rounds == 1  # D47: the failure ended the search in its first round
    assert alone.outcome == EMERGENCY and alone.solver_errors == 1  # no shifted plan in a fresh episode
    assert alone.exit == ERROR and alone.cut_rounds == 1


def test_gurobis_licence_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    import hrlmpc.mpc_worker as module

    assert set(module.LICENCE_ERRORS) == {GRB.ERROR_NO_LICENSE, GRB.ERROR_SIZE_LIMIT_EXCEEDED} == {10009, 10010}
    _, _, problem = _setup("slalom", d_bar=0.5, horizon=5)
    worker = MPCWorker(problem)
    with pytest.raises(gp.GurobiError) as failure:
        _no_solution(worker._slots[0])
    monkeypatch.setattr(module, "LICENCE_ERRORS", (failure.value.errno,))  # as if it were a licence error
    monkeypatch.setattr(worker, "_relax", _no_solution)
    assert problem.free_binaries(problem.fixed_binaries(NEAR_THE_GATE)) > 0  # so Gurobi is called
    with pytest.raises(gp.GurobiError):
        worker.act(NEAR_THE_GATE, [5.4, 1.2], 5)


def test_a_clarabel_failure_is_counted_and_brakes_without_a_shifted_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, problem = _setup("tunnel")
    worker = MPCWorker(problem)
    monkeypatch.setattr(worker._exact, "prob", _FailingProblem())
    x = np.array([1.0, 0.0, 0.3, 0.0])
    with _licence():
        u, info = worker.act(x, [2.8, 0.0], H)
    assert info.outcome == EMERGENCY and info.solver_errors == 1
    assert info.exit == ERROR and info.exact_inaccurate == info.exact_infeasible == 0  # D47: a failure, no status
    assert info.exact_uncertified == 0
    np.testing.assert_allclose(u, problem.emergency_input(x))


def test_a_failed_check_of_the_shifted_plan_counts_as_infeasible(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, problem = _setup("tunnel", d_bar=0.5)
    worker, target = MPCWorker(problem), [2.8, 0.0]
    plan, _, costlier = _costlier(problem, worker, target)
    monkeypatch.setattr(worker, "_optimize", lambda *args: costlier)
    monkeypatch.setattr(worker._gauge, "prob", _FailingProblem())
    _, info = worker.act(plan[0][1] + np.array([0.0, 0.0, 0.0, 1e-3]), target, H - 1)
    assert info.outcome == SOLVER and info.invalid_candidate == 1 and info.solver_errors == 1


def test_the_worker_is_deterministic_and_reset_forgets_the_plan() -> None:
    _, _, problem = _setup("slalom", d_bar=0.5, horizon=5)
    states = [NEAR_THE_GATE, np.array([3.85, 0.47, 1.0, 0.05])]  # 12 free binaries each: Gurobi's determinism too
    assert all(problem.free_binaries(problem.fixed_binaries(x)) > 0 for x in states)
    with _licence():
        a, b = MPCWorker(problem), MPCWorker(problem)
        for left, x in zip((5, 4), states, strict=True):
            ua, _ = a.act(x, [5.4, 1.2], left)
            ub, _ = b.act(x, [5.4, 1.2], left)
            np.testing.assert_array_equal(ua, ub)
        a.reset()
        _, info = a.act(states[1], [5.4, 1.2], 4)
    assert np.isnan(info.tube_ratio) and info.outcome == SOLVER  # no previous plan, so no candidate either


def test_a_reset_slot_solves_as_a_fresh_worker() -> None:
    """D29, D34: a reset forgets the plan, the cuts of ``Z`` and Gurobi's last solution, so an evaluation does
    not depend on the earlier ones, and a saved Manager evaluated with a new Worker reproduces it."""
    _, _, problem = _setup("slalom", d_bar=0.5, horizon=5)
    x, target = NEAR_THE_GATE, [5.4, 1.2]
    with _licence():
        used = MPCWorker(problem)
        for left, state in zip((5, 4, 3), ([3.75, 0.45, 1.0, 0.0], [3.85, 0.47, 1.0, 0.05], [1.0, 0.0, 0.0, 0.0]),
                               strict=True):
            used.act(np.array(state), target, left)
        used.reset(0)
        u_used, info_used = used.act(x, target, 5)
        u_fresh, info_fresh = MPCWorker(problem).act(x, target, 5)
    assert info_used.free_binaries > 0 and info_used.outcome == info_fresh.outcome == SOLVER
    assert np.isnan(info_used.tube_ratio)
    np.testing.assert_allclose(u_used, u_fresh, rtol=0.0, atol=1e-9)
    assert info_used.cost == pytest.approx(info_fresh.cost, rel=1e-9)


def test_the_conic_problems_are_compiled_once() -> None:
    """Every parameter enters affinely (DPP), so cvxpy compiles the exact problem and the gauge oracle once,
    when the Worker is built, and only updates their data at each step."""
    _, _, problem = _setup("slalom", d_bar=0.5)
    assert problem.regularized  # the regularizer of D46 is affine in the parameters too
    with _licence():
        worker = MPCWorker(problem)
    assert worker._exact.prob.is_dpp() and worker._gauge is not None and worker._gauge.prob.is_dpp()
    assert ReachSupport(problem, H, support_directions(4)).prob.is_dpp()
    worker.close()


def _relaxation_objective(worker: MPCWorker, z: np.ndarray, v: np.ndarray) -> float:
    """The objective that Gurobi minimises in slot 0, evaluated at the plan ``(z, v)``."""
    slot = worker._slots[0]
    slot.m.update()
    values = np.zeros(slot.m.NumVars)
    for variables, plan in ((slot.z, z), (slot.v, v)):
        flat = [var for row in variables.tolist() for var in row]  # row by row, as ravel orders the plan
        for var, value in zip(flat, plan.ravel(), strict=True):
            values[var.index] = value
    objective = slot.m.getObjective()
    linear = objective.getLinExpr()
    total = linear.getConstant() + sum(linear.getCoeff(i) * values[linear.getVar(i).index]
                                       for i in range(linear.size()))
    return float(total + sum(objective.getCoeff(i) * values[objective.getVar1(i).index]
                             * values[objective.getVar2(i).index] for i in range(objective.size())))


@pytest.mark.parametrize("d_bar", [0.0, 0.5])
def test_both_solvers_minimise_the_cost_of_the_problem(d_bar: float) -> None:
    """D37, D46: at any plan, the objective of Gurobi's relaxation and that of Clarabel's exact problem equal the
    problem's cost, the regularizer included with a tube. So the relaxation's bound bounds the exact optimum, and
    the costs of the solvers' plan and of the shifted plan compare."""
    _, _, problem = _setup("slalom", d_bar=d_bar)
    assert problem.regularized == (d_bar > 0.0)
    rng = np.random.default_rng(4)
    ones = [np.ones((problem.N, obstacle.num_faces)) for obstacle in problem.obstacles]
    with _licence():
        worker = MPCWorker(problem)
        for k in (1, 4, H):
            z, v = rng.normal(size=(problem.N + 1, 4)), rng.normal(size=(problem.N, 2))
            x, g = rng.normal(size=4), rng.normal(size=2)  # x far from z_0: the regularizer weighs
            expected = problem.cost(z, v, g, k, x=x)
            worker._set_objective(worker._slots[0], g, k, x)
            assert _relaxation_objective(worker, z, v) == pytest.approx(expected, rel=1e-9)
            worker._exact.assign(x, g, k, ones)
            worker._exact.z.value, worker._exact.v.value = z, v
            assert float(worker._exact.prob.objective.value) == pytest.approx(expected, rel=1e-9)
        worker.close()


def test_the_regularizer_keeps_the_nominal_start_nearer_the_state() -> None:
    """D46: among the plans that are optimal without it, the regularizer takes one whose nominal start is no
    farther from the state than any of them, at an extra cost of D37 of at most ``eps`` times that distance
    squared. On the tunnel, Clarabel alone, with reachable targets, so that the costs are small and the solvers'
    tolerances do not blur the distances.

    A strong regularizer must pull the start to the state, which the solvers do only if they minimise it:
    ``eps d^2 <= J(x) - J(z_0) <= J_c``, with ``J_c`` the cost of D37 of the plan that coasts from ``z_0 = x``
    to the waypoint and brakes afterwards (feasible: below ``v_bar``, at rest by ``N``). In the third case the
    coasting plan misses the target by 5.9 cm, which the tube absorbs (the reachable shifts of the waypoint form
    a disk of 7.1 cm): without the regularizer the optimum costs nothing and starts at least 5.5 cm from ``x``,
    with ``eps = 1e6`` within ``sqrt(J_c / eps) = 5.9 mm``."""
    _, _, problem = _setup("tunnel", d_bar=0.5)
    _, _, plain = _setup("tunnel", d_bar=0.5, start_weight=0.0)
    _, _, strong = _setup("tunnel", d_bar=0.5, start_weight=1e6)
    assert problem.regularized and not plain.regularized
    eps = problem.settings.start_weight
    cases = (([1.0, 0.0, 0.0, 0.0], [1.5, 0.2], H), ([2.0, 0.5, 0.8, -0.3], [2.5, 0.4], 6),
             ([3.0, -1.0, 1.0, 0.4], [3.35, -0.8], 4))
    distances = []
    with _licence():
        for x_list, target, left in cases:
            x = np.array(x_list)
            regularized, unregularized = MPCWorker(problem), MPCWorker(plain)
            _, info_r = regularized.act(x, target, left)
            _, info_u = unregularized.act(x, target, left)
            assert info_r.outcome == info_u.outcome == SOLVER and info_r.exit == info_u.exit == FIXED
            (z_r, v_r), (z_u, _) = regularized._slots[0].plan, unregularized._slots[0].plan
            d_r, d_u = np.linalg.norm(x - z_r[0]), np.linalg.norm(x - z_u[0])
            distances.append(d_u)
            # eps d_r^2 <= eps d_u^2 + (the unregularized plan's excess over the optimum, 1e-6 at most)
            assert d_r**2 <= d_u**2 + 1e-6 / eps
            optimum = info_u.cost  # D37's optimum, without the regularizer
            assert optimum - 1e-6 <= plain.cost(z_r, v_r, target, left) <= optimum + eps * d_u**2 + 1e-6
            assert info_r.cost == pytest.approx(plain.cost(z_r, v_r, target, left) + eps * d_r**2, rel=1e-9)
        x, target, left = np.array(cases[2][0]), np.array(cases[2][1]), cases[2][2]
        coast = np.linalg.matrix_power(problem.A, left) @ x
        bound = float(np.sqrt(problem.settings.target_weight * np.sum((coast[:2] - target) ** 2) / 1e6))
        pulled = MPCWorker(strong)
        _, info_s = pulled.act(x, target, left)
        z_s = pulled._slots[0].plan[0]
    assert bound == pytest.approx(5.9e-3, abs=1e-4) and distances[2] >= 0.05
    assert info_s.outcome == SOLVER and np.linalg.norm(x - z_s[0]) <= bound + 1e-6


def test_slots_do_not_share_plans() -> None:
    _, _, problem = _setup("tunnel")
    with _licence():
        worker = MPCWorker(problem, num_slots=2)
        worker.act(np.array([1.0, 0.0, 0.0, 0.0]), [2.8, 0.0], H, slot=0)
    assert worker._slots[0].plan is not None and worker._slots[1].plan is None
    worker.reset(0)
    assert worker._slots[0].plan is None
    worker.close()
    worker.close()  # a second close does nothing


def test_the_command_adapter_steers_every_agent() -> None:
    cfg, _, problem = _setup("tunnel")
    obs_map = ObservationMap.from_config(cfg)
    command = MPCCommand(MPCWorker(problem, num_slots=3), obs_map)
    p = np.array([[1.0, 0.0], [1.5, -0.5], [0.5, 0.8]])
    v = np.zeros((3, 2))
    obs = obs_map.observe(p, v)
    with _licence():
        u = command.command(obs, p, p + np.array([1.8, 0.0]), np.full(3, H))
    assert u.shape == (3, 2) and np.all(u[:, 0] > 0.0)
    stats = command.pop_stats()
    assert stats["outcome"].shape == (1, 3) and np.all(stats["outcome"] == SOLVER)
    assert command.pop_stats()["outcome"].size == 0
    with pytest.raises(ValueError, match="slots"):
        command.command(np.vstack([obs, obs]), np.vstack([p, p]), np.vstack([p, p]), np.full(6, H))
    with pytest.raises(ValueError, match="steps_left"):
        MPCWorker(problem).act(np.array([1.0, 0.0, 0.0, 0.0]), [2.8, 0.0], problem.N + 1)


# --------------------------------------------------------------------------
# The targets of architecture 4 (D38, D39)
# --------------------------------------------------------------------------


def _f7_travel(problem: MPCProblem, speed: float, steps: int) -> float:
    """The displacement of full thrust along a line in ``steps`` steps from ``speed`` (the recursion of F7)."""
    travel = 0.0
    for _ in range(steps):
        speed = min((1.0 - problem.params.gamma * problem.dt) * speed + problem.dt * problem.u_bar, problem.v_bar)
        travel += problem.dt * speed
    return travel


def test_the_support_points_from_rest_lie_on_the_circle_of_f7() -> None:
    """D38 and F6: away from the walls and at rest, the reachable set is a disk, whose radius is the displacement
    of full thrust in H steps, and its 16 support points lie on it, in their directions."""
    _, _, problem = _setup("tunnel")
    support = ReachSupport(problem, H, support_directions(16))
    x = np.array([5.0, 0.0, 0.0, 0.0])
    points = support(x)
    radius = _f7_travel(problem, 0.0, H)
    assert radius == pytest.approx(0.95092, abs=1e-5)
    np.testing.assert_allclose(points - x[:2], radius * support_directions(16), atol=1e-5)
    assert support.failures == 0


def test_the_forward_support_point_at_full_speed_is_a_segment_at_full_speed() -> None:
    """At the planned speed the set reaches H T_s v_bar ahead, and it is symmetric about the velocity."""
    _, _, problem = _setup("tunnel")
    support = ReachSupport(problem, H, support_directions(16))
    x = np.array([2.0, 0.0, problem.v_bar, 0.0])
    points = support(x)
    np.testing.assert_allclose(points[0], [2.0 + H * problem.dt * problem.v_bar, 0.0], atol=1e-5)
    np.testing.assert_allclose(points[4], points[12] * np.array([1.0, -1.0]), atol=1e-5)  # up and down mirror


def _effort_bound(problem: MPCProblem) -> float:
    """How far the Worker's plan may end from a reachable target (D37): the plan that reaches it costs at most its
    effort, ``r_e H u_bar^2``, plus with a tube the regularizer of D46, at most ``eps (r_p^2 + r_v^2)`` since
    ``x - z_0`` lies in ``Z``; and the plan's cost exceeds ``q_g`` times the squared miss. 2.5 cm without a tube,
    1.8 cm at ``d_bar = 0.5``. At the edge of the set the miss is a few millimetres, inside it a fraction of one."""
    s, tube = problem.settings, problem.tube
    start = s.start_weight * (tube.position_radius**2 + tube.speed_radius**2) if problem.regularized else 0.0
    return float(np.sqrt((s.effort_weight * H * problem.u_bar**2 + start) / s.target_weight))


def test_every_support_point_is_a_target_the_worker_reaches() -> None:
    """The support points and the Worker share one problem, so the projection of a support point is itself, up
    to the trade-off with the effort (D37)."""
    _, _, problem = _setup("tunnel")
    support = ReachSupport(problem, H, support_directions(8))
    x = np.array([2.0, 0.5, 1.0, 0.3])
    assert _effort_bound(problem) == pytest.approx(0.025)
    with _licence():
        worker = MPCWorker(problem)
        for point in support(x):
            reached, report = worker.project(x, point, H)
            assert reached is not None and report.errors == 0 and report.exit == FIXED  # D47: no binaries
            assert np.linalg.norm(reached - point) <= _effort_bound(problem)


def test_a_projection_solves_as_a_fresh_worker() -> None:
    """D39: a projection neither uses nor leaves a plan or cuts of Z, near a gate, where Gurobi and the cuts work."""
    _, _, problem = _setup("slalom", d_bar=0.5)
    x, q = np.array([3.75, 0.45, 1.0, 0.0]), np.array([5.4, 1.2])
    assert problem.free_binaries(problem.fixed_binaries(x)) > 0
    with _licence():
        used = MPCWorker(problem)
        used.act(x, q, H)  # leaves a plan, and possibly cuts of Z
        used.project(np.array([3.4, 1.0, 1.0, 0.0]), np.array([4.6, 0.5]), H)
        first, _ = used.project(x, q, H)
        fresh, _ = MPCWorker(problem).project(x, q, H)
    assert first is not None and fresh is not None
    np.testing.assert_allclose(first, fresh, rtol=0.0, atol=1e-9)
    assert used._slots[0].plan is None and used._slots[0].z_seed == []


def test_a_target_inside_an_obstacle_is_projected_next_to_its_face() -> None:
    """D39: in front of O1 = [4, 5] x [-2, 0.25] at 0.8 m/s, a point inside it goes to the nearest reachable point,
    next to the enlarged left face; the gate above is out of reach within the segment. The point stops a few
    millimetres short of the face: the plan must rest there by the end of the horizon, and braking within the
    segment costs effort (D37)."""
    _, _, problem = _setup("slalom")
    x, q = np.array([3.0, -1.0, 0.8, 0.0]), np.array([4.5, -1.0])
    face = 4.0 - problem.margin
    with _licence():
        worker = MPCWorker(problem)
        g, report = worker.project(x, q, H)
        assert g is not None and report.errors == 0 and report.exit in FOUND
        assert face - 2e-2 <= g[0] <= face + 1e-6 and abs(g[1] + 1.0) <= 1e-2
        again, _ = worker.project(x, g, H)
    assert again is not None and np.linalg.norm(again - g) <= _effort_bound(problem)  # reachable


@pytest.mark.parametrize("d_bar", [0.0, 0.5])
def test_the_reachable_targets_are_reachable_and_contact_free(d_bar: float) -> None:
    """D38, D39 end to end on the slalom: from states near the gates, the targets of random actions lie outside
    every enlarged obstacle, and a fresh projection leaves them where they are."""
    _, _, problem = _setup("slalom", d_bar=d_bar)
    rng = np.random.default_rng(8)
    states = np.array([[3.4, 1.0, 1.0, 0.0], [6.4, 0.2, 0.8, -0.7], [1.0, 0.0, 0.0, 0.0]])
    with _licence():
        targets = reachable_targets(problem, H, 16)
        checker = MPCWorker(problem)
        actions = rng.uniform(-1.0, 1.0, size=(len(states), 2))
        actions[0] = [1.0, -0.2]  # toward the obstacle below the first gate
        g = targets(actions, states[:, :2], states[:, 2:])
        for x, target in zip(states, g, strict=True):
            for obstacle in problem.obstacles:  # beyond one face, to what a mixed-integer plan may leave
                assert np.max(obstacle.normals @ target - obstacle.bounds) >= -problem.face_slack - 1e-9
            again, _ = checker.project(x, target, H)
            assert again is not None and np.linalg.norm(again - target) <= _effort_bound(problem)
        stats = targets.pop_stats()
        targets.close()
        targets.close()
    assert stats["projection_failed"].sum() == 0 and stats["support_failures"].sum() == 0
    assert stats["hull_fallback"].sum() == 0 and np.all(stats["hull_area"] > 0.5)
    assert np.all(np.isin(stats["projection_exit"], FOUND))  # D47: every projection found its plan
