"""Tests of the tube-MPC Worker with its solvers (decision log D31-D35); they need cvxpy, Clarabel and Gurobi.

The closed loops run on the environment's own physics (:func:`hrlmpc.physics.physics_step`),
with scripted targets in place of a Manager and with chosen disturbances, so
the claims of D31-D33 are checked where they matter: the real state never
touches a wall, the error to the plan stays in the tube (by its gauge), the
input never needs the saturation, the shifted plan stays feasible, and the
plans are optimal. Gurobi's pip licence is size-limited: a test whose model
exceeds it is skipped with that reason.

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
from hrlmpc.mpc_command import CANDIDATE, EMERGENCY, KEPT, SOLVER, MPCCommand  # noqa: E402
from hrlmpc.mpc_problem import MPCProblem, MPCSettings  # noqa: E402
from hrlmpc.mpc_worker import MPCWorker, _GaugeOracle  # noqa: E402
from hrlmpc.physics import physics_step  # noqa: E402

GRB = gp.GRB
TOL = 1e-5
"""Violations and gauge excess accepted in the closed loops (the solvers' tolerances; module docstring)."""


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
    cfg = load_env_config(f"configs/env/{name}.yaml")
    cfg = dataclasses.replace(cfg, physics=dataclasses.replace(cfg.physics, d_bar=d_bar))
    layout = Layout.from_config(cfg.geometry)
    return cfg, layout, MPCProblem(cfg.physics, layout, MPCSettings(**settings))


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
    """A closed loop with a target set every 10 steps; returns the trajectory and every check.

    Per step: what the Worker reported (outcome, tube ratio, free binaries,
    cut rounds, invalid shifted plans, solver errors); the violations of the
    applied plan; the norm of the input before its projection onto
    ``U``, at most ``a_max`` by D33 so that the saturation never acts; contact
    and the speed limiter in the physics; and, with a tube, the gauge of the
    error to the plan's prediction after the step, at most 1 when it stayed in ``Z``.
    """
    worker = MPCWorker(problem) if worker is None else worker
    p, v = np.array(start, dtype=np.float64), np.zeros(2)
    target = targets(p)
    names = ("outcome", "ratio", "free", "rounds", "invalid", "errors", "violations", "thrust", "contact", "limited",
             "gauge")
    log: dict[str, Any] = {name: [] for name in names}
    log.update(p=[p.copy()], success=False)
    with _licence():
        for k in range(steps):
            if k % 10 == 0:
                target = targets(p)
            x = np.concatenate([p, v])
            u, info = worker.act(x, target)
            for name, value in (("outcome", info.outcome), ("ratio", info.tube_ratio), ("free", info.free_binaries),
                                ("rounds", info.cut_rounds), ("invalid", info.invalid_candidate),
                                ("errors", info.solver_errors)):
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


def _nominal(log: dict[str, Any], problem: MPCProblem) -> None:
    """What every closed loop with its disturbances in ``D`` must show."""
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
    _nominal(log, problem)
    assert max(_worst(log).values()) <= 1e-6  # Clarabel's plans alone
    assert len(log["outcome"]) <= 81  # the bound of F7 from (1, 0) is 78 steps: the gap of D20 is at most 5%


def test_a_manager_that_aims_at_the_openings_threads_the_slalom() -> None:
    cfg, layout, problem = _setup("slalom")
    log = _run(problem, layout, cfg, (1.0, 0.0), _through_the_gates)
    assert log["success"]
    _nominal(log, problem)
    assert max(log["free"]) > 0  # the gates' binaries were live


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
        for _ in range(80):  # through the first gate
            x = np.concatenate([p, v])
            u, info = worker.act(x, _through_the_gates(p))
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


def _exhaustive_cost(problem: MPCProblem, x: np.ndarray, target: np.ndarray) -> float:
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
    x_r = problem.rest_state(target)
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
        cost = sum(cp.quad_form(z[k] - x_r, problem.Q) + cp.quad_form(v[k], problem.R) for k in range(N))
        cost = cost + cp.quad_form(z[N] - x_r, problem.P)
        prob = cp.Problem(cp.Minimize(cost), cons)
        prob.solve(solver=cp.CLARABEL)
        if prob.status == "optimal":
            best = min(best, float(prob.value))
    return best


@pytest.mark.parametrize("d_bar", [0.0, 0.5])
def test_the_plan_is_the_optimum_over_every_choice_of_faces(d_bar: float) -> None:
    _, _, problem = _setup("slalom", d_bar=d_bar, horizon=3)
    # slow enough to come to rest within N = 3 steps, and close enough to a gate for its binaries to be live
    for x, target in (([3.9, 0.3, 0.3, 0.0], [5.4, 1.2]), ([6.85, -0.2, 0.3, -0.3], [7.5, -1.2])):
        x_arr = np.array(x)
        with _licence():
            u, info = MPCWorker(problem).act(x_arr, target)
        assert info.outcome == SOLVER and info.free_binaries > 0 and info.solver_errors == 0
        best = _exhaustive_cost(problem, x_arr, np.array(target))
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
                _, info = worker.act(np.array(x), _through_the_gates(np.array(x[:2])))
                z, v = worker._slots[0].plan
                worker.reset()
                assert info.outcome == SOLVER and info.free_binaries > 0 and info.solver_errors == 0
                assert np.linalg.norm(z[1 : problem.N, 2:], axis=1).max() <= problem.v_bar * (1.0 + TOL)
                assert np.linalg.norm(v, axis=1).max() <= problem.u_bar * (1.0 + TOL)
                costs[sides].append(info.cost)
    np.testing.assert_allclose(costs[4], costs[16], rtol=2e-4)
    np.testing.assert_allclose(costs[64], costs[16], rtol=2e-4)


def test_fixing_unreachable_binaries_changes_no_solution(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, problem = _setup("slalom", d_bar=0.5, horizon=5)
    states = [([3.6, 0.6, 1.0, 0.0], [5.4, 1.2]), ([1.0, 0.0, 0.0, 0.0], [2.8, 0.0])]
    with _licence():
        fixed = [MPCWorker(problem).act(np.array(x), g)[1].cost for x, g in states]

        def all_free(x: Any) -> list[tuple[np.ndarray, np.ndarray]]:
            return [(np.zeros((problem.N, o.num_faces)), np.ones((problem.N, o.num_faces))) for o in problem.obstacles]

        monkeypatch.setattr(problem, "fixed_binaries", all_free)
        free = [MPCWorker(problem).act(np.array(x), g)[1].cost for x, g in states]
    np.testing.assert_allclose(free, fixed, rtol=2e-4)


def test_a_state_that_cannot_avoid_a_wall_brakes_as_an_emergency() -> None:
    _, _, problem = _setup("slalom")
    x = np.array([3.95, -1.0, 1.2, 0.0])  # 5 cm from O1 at full speed: no plan stops in time
    with _licence():
        u, info = MPCWorker(problem).act(x, [6.0, -1.0])
    assert info.outcome == EMERGENCY and info.plan is None and np.isnan(info.cost)
    assert info.solver_errors == 0  # infeasible is no failure
    np.testing.assert_allclose(u, problem.emergency_input(x))


def test_the_shifted_plan_takes_over_when_the_solvers_find_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, problem = _setup("tunnel")
    worker = MPCWorker(problem)
    x = np.array([1.0, 0.0, 0.0, 0.0])
    with _licence():
        worker.act(x, [2.8, 0.0])
    plan = worker._slots[0].plan
    assert plan is not None
    monkeypatch.setattr(worker, "_optimize", lambda *args: None)
    x_next = plan[0][1]  # exactly where the plan said: no disturbance
    u, info = worker.act(x_next, [2.8, 0.0])
    assert info.outcome == CANDIDATE and info.invalid_candidate == 0
    z_c, v_c = problem.candidate(*plan)
    np.testing.assert_allclose(u, problem.input(x_next, z_c, v_c))


def _costlier(problem: MPCProblem, worker: MPCWorker, target: list[float]) -> tuple[Any, Any, Any]:
    """After one step from rest: the plan, its shift and a plan one metre off the target, costlier than the shift."""
    with _licence():
        worker.act(np.array([1.0, 0.0, 0.0, 0.0]), target)
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
    u, info = worker.act(x_next, target)
    assert info.outcome == KEPT and info.invalid_candidate == 0 and info.solver_errors == 0
    assert info.cost == pytest.approx(problem.cost(*shifted, problem.rest_state(target)))
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
    u, info = worker.act(x_next, target)
    assert info.outcome == SOLVER and info.invalid_candidate == 1  # applied, though costlier
    np.testing.assert_allclose(u, problem.input(x_next, *costlier))
    worker._slots[0].plan = plan  # the same step, now without a plan from the solvers
    monkeypatch.setattr(worker, "_optimize", lambda *args: None)
    u, info = worker.act(x_next, target)
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
        _, first = worker.act(np.array([3.6, 0.6, 1.0, 0.0]), target)
        plan = worker._slots[0].plan
        monkeypatch.setattr(worker, "_relax", _no_solution)
        u, info = worker.act(plan[0][1], target)
        fresh = MPCWorker(problem)
        monkeypatch.setattr(fresh, "_relax", _no_solution)
        _, alone = fresh.act(np.array([3.6, 0.6, 1.0, 0.0]), target)
    assert first.outcome == SOLVER and first.solver_errors == 0
    assert info.free_binaries > 0 and info.outcome == CANDIDATE
    assert info.solver_errors == 1 and info.invalid_candidate == 0
    assert alone.outcome == EMERGENCY and alone.solver_errors == 1  # no shifted plan in a fresh episode


def test_gurobis_licence_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    import hrlmpc.mpc_worker as module

    assert set(module.LICENCE_ERRORS) == {GRB.ERROR_NO_LICENSE, GRB.ERROR_SIZE_LIMIT_EXCEEDED} == {10009, 10010}
    _, _, problem = _setup("slalom", d_bar=0.5, horizon=5)
    worker = MPCWorker(problem)
    with pytest.raises(gp.GurobiError) as failure:
        _no_solution(worker._slots[0])
    monkeypatch.setattr(module, "LICENCE_ERRORS", (failure.value.errno,))  # as if it were a licence error
    monkeypatch.setattr(worker, "_relax", _no_solution)
    with pytest.raises(gp.GurobiError):
        worker.act(np.array([3.6, 0.6, 1.0, 0.0]), [5.4, 1.2])


def test_a_clarabel_failure_is_counted_and_brakes_without_a_shifted_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, problem = _setup("tunnel")
    worker = MPCWorker(problem)
    monkeypatch.setattr(worker._exact, "prob", _FailingProblem())
    x = np.array([1.0, 0.0, 0.3, 0.0])
    with _licence():
        u, info = worker.act(x, [2.8, 0.0])
    assert info.outcome == EMERGENCY and info.solver_errors == 1
    np.testing.assert_allclose(u, problem.emergency_input(x))


def test_a_failed_check_of_the_shifted_plan_counts_as_infeasible(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, problem = _setup("tunnel", d_bar=0.5)
    worker, target = MPCWorker(problem), [2.8, 0.0]
    plan, _, costlier = _costlier(problem, worker, target)
    monkeypatch.setattr(worker, "_optimize", lambda *args: costlier)
    monkeypatch.setattr(worker._gauge, "prob", _FailingProblem())
    _, info = worker.act(plan[0][1] + np.array([0.0, 0.0, 0.0, 1e-3]), target)
    assert info.outcome == SOLVER and info.invalid_candidate == 1 and info.solver_errors == 1


def test_the_worker_is_deterministic_and_reset_forgets_the_plan() -> None:
    _, _, problem = _setup("slalom", d_bar=0.5, horizon=5)
    states = [np.array([3.6, 0.6, 1.0, 0.0]), np.array([3.7, 0.62, 1.0, 0.05])]
    with _licence():
        a, b = MPCWorker(problem), MPCWorker(problem)
        for x in states:
            ua, _ = a.act(x, [5.4, 1.2])
            ub, _ = b.act(x, [5.4, 1.2])
            np.testing.assert_array_equal(ua, ub)
        a.reset()
        _, info = a.act(states[1], [5.4, 1.2])
    assert np.isnan(info.tube_ratio) and info.outcome == SOLVER  # no previous plan, so no candidate either


def test_a_reset_slot_solves_as_a_fresh_worker() -> None:
    """D29, D34: a reset forgets the plan, the cuts of ``Z`` and Gurobi's last solution, so an evaluation does
    not depend on the earlier ones, and a saved Manager evaluated with a new Worker reproduces it."""
    _, _, problem = _setup("slalom", d_bar=0.5, horizon=5)
    x, target = np.array([3.6, 0.6, 1.0, 0.0]), [5.4, 1.2]
    with _licence():
        used = MPCWorker(problem)
        for state in ([3.6, 0.6, 1.0, 0.0], [3.7, 0.62, 1.0, 0.05], [1.0, 0.0, 0.0, 0.0]):
            used.act(np.array(state), target)
        used.reset(0)
        u_used, info_used = used.act(x, target)
        u_fresh, info_fresh = MPCWorker(problem).act(x, target)
    assert info_used.free_binaries > 0 and info_used.outcome == info_fresh.outcome == SOLVER
    assert np.isnan(info_used.tube_ratio)
    np.testing.assert_allclose(u_used, u_fresh, rtol=0.0, atol=1e-9)
    assert info_used.cost == pytest.approx(info_fresh.cost, rel=1e-9)


def test_slots_do_not_share_plans() -> None:
    _, _, problem = _setup("tunnel")
    with _licence():
        worker = MPCWorker(problem, num_slots=2)
        worker.act(np.array([1.0, 0.0, 0.0, 0.0]), [2.8, 0.0], slot=0)
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
        u = command.command(obs, p, p + np.array([1.8, 0.0]))
    assert u.shape == (3, 2) and np.all(u[:, 0] > 0.0)
    stats = command.pop_stats()
    assert stats["outcome"].shape == (1, 3) and np.all(stats["outcome"] == SOLVER)
    assert command.pop_stats()["outcome"].size == 0
    with pytest.raises(ValueError, match="slots"):
        command.command(np.vstack([obs, obs]), np.vstack([p, p]), np.vstack([p, p]))
