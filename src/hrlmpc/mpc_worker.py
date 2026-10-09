"""The MPC Worker of PPO_MPC: the tube MPC of D31-D34, solved by Gurobi and Clarabel.

:mod:`hrlmpc.mpc_problem` defines the problem; this module solves it, at every
step and for every agent, with the scheme of the pre-alignment Worker
(``algorithms/tube_mpc.py``, D34):

- Gurobi solves a relaxation, a mixed-integer QP: the speed and thrust disks
  replaced by the regular polygon of :data:`BASE_SIDES` sides circumscribed
  around them, and the tube constraint ``x - z_0 in Z`` by the half-spaces
  ``a' (x - z_0) <= h_Z(a)`` along :func:`z_directions`. The relaxation contains
  the exact problem, so its lower bound (``ObjBound``) bounds the exact optimum.
- Clarabel, through cvxpy, solves the exact problem with Gurobi's binaries
  fixed: a second-order cone program with the disks and the lifted
  representation of ``Z`` (``x - z_0 = sum_l A_K^l B d_l + T eta``,
  ``||d_l|| <= d_bar``, ``||eta|| <= 1``).

If the exact cost is within the MIP gap of the lower bound, the exact plan is
optimal for the exact problem to that gap. Otherwise the relaxation is
tightened where it was loose (tangents of the disks at the plans' points on or
past their edges, and the supporting half-space of ``Z`` that the separation
oracle :class:`_GaugeOracle` returns), and both are solved again, at most
:data:`MAX_CUT_ROUNDS` times. A round that adds no cut returns the
relaxation's own plan, as the legacy Worker did: it meets the disks and ``Z``
to :data:`CUT_TOL` and the faces to :attr:`~hrlmpc.mpc_problem.MPCProblem.face_slack`,
which ``rho`` covers, so the speed limiter may act about 1e-6 m/s beyond what
F3 covers. When the reachable positions fix every binary
(:meth:`~hrlmpc.mpc_problem.MPCProblem.fixed_binaries`), as on the tunnel, the
problem is convex and Clarabel solves it alone.

The shifted previous plan (algorithm 3 of the tube-MPC note) is the benchmark
of the solvers' plan and the fallback. Theorem 4.1 keeps it feasible after
every disturbance in ``D``; of its constraints, only the tube constraint
depends on the last step's disturbance, and it is checked (to
:data:`CANDIDATE_TOL`) whenever the shifted plan would be applied:

- the solvers' plan is applied, unless the shifted plan is cheaper and
  feasible: then the shifted plan is kept (:data:`~hrlmpc.mpc_command.KEPT`);
- if the solvers find nothing, the shifted plan is applied
  (:data:`~hrlmpc.mpc_command.CANDIDATE`), even outside its tube (after a
  disturbance outside ``D``) as the last resort; that case is reported in
  ``invalid_candidate``;
- without a shifted plan (a fresh episode) and without a solution, the agent
  brakes: an emergency outside every guarantee.

A solver failure (an exception of Clarabel or Gurobi, or an unusable result
of the gauge oracle) ends the step's search as if nothing was found and is
counted in ``solver_errors``; Gurobi's licence errors propagate, since every
step would repeat them.

Every solve depends only on its inputs and on the slot's earlier solves in the
same episode (the shifted plan, and the cuts of ``Z`` it keeps): Gurobi runs
single-threaded, so solves are deterministic without a work limit, and
:meth:`MPCWorker.reset` also discards the solution Gurobi keeps, so a reset
slot solves as a fresh Worker would. The conic problems are compiled when the
Worker is built, so that cvxpy's compilation does not count in the first
step's solve time.

Gurobi (academic licence: the pip licence is size-limited) and cvxpy with
Clarabel are the extra ``mpc`` (D5); importing this module without them fails.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from numpy.typing import ArrayLike, NDArray

from hrlmpc.mpc_command import CANDIDATE, EMERGENCY, KEPT, SOLVER, WorkerStep
from hrlmpc.mpc_problem import Binaries, Bounds, MPCProblem

FloatArray = NDArray[np.float64]
Plan = tuple[FloatArray, FloatArray]
"""A plan ``(z, v)``: nominal states ``(N+1, 4)`` and nominal inputs ``(N, 2)``."""

BASE_SIDES = 16
"""Sides of the polygons that relax the speed and thrust disks (circumscribed: 2% larger at most)."""

CUT_TOL = 1e-6
"""How far past the edge of a disk, or outside ``Z``, a plan's point must lie to get a cut."""

MAX_CUT_ROUNDS = 25
"""Largest number of mixed-integer solves per step before the shifted plan takes over."""

Z_SEED_KEEP = 32
"""How many of the slot's latest cuts of ``Z`` each solve starts with."""

ACCEPT_TOL = 1e-6
"""Algorithm 3's acceptance: the solvers' plan is applied unless the shifted plan is cheaper by more
than ``ACCEPT_TOL (1 + |J|)`` of its cost ``J``."""

CANDIDATE_TOL = 1e-4
"""How far the shifted plan may miss the measured state and still count as feasible: the gauge of
``x - z_0`` at most ``1 + CANDIDATE_TOL`` or, without a tube, ``z_0`` within ``CANDIDATE_TOL`` of ``x``
(infinity norm). Above what the solvers' tolerances leave, well inside the margin ``rho``."""

GAUGE_ZERO = 1e-12
"""Errors this small (infinity norm) get the gauge 0 without a solve: they lie deep inside ``Z``."""

LICENCE_ERRORS = (GRB.ERROR_NO_LICENSE, GRB.ERROR_SIZE_LIMIT_EXCEEDED)
"""Gurobi's errors that propagate: no licence, or a model beyond the licence's size."""

_POLYGON = np.array([[np.cos(t), np.sin(t)] for t in 2.0 * np.pi * np.arange(BASE_SIDES) / BASE_SIDES])


def z_directions() -> FloatArray:
    """The 64 directions ``a = (cos(psi) n, sin(psi) n)`` that start the relaxation of ``Z``.

    Eight headings ``n`` and eight mixing angles ``psi``: the directions in
    which a position error and a velocity error point the same way, the shape
    of every support function ``Z``'s construction evaluates; rotation-symmetric like ``Z``.
    """
    dirs = []
    for phi in 2.0 * np.pi * np.arange(8) / 8:
        n = np.array([np.cos(phi), np.sin(phi)])
        for psi in 2.0 * np.pi * np.arange(8) / 8:
            dirs.append(np.concatenate([np.cos(psi) * n, np.sin(psi) * n]))
    return np.array(dirs)


class SolverFailure(RuntimeError):
    """Clarabel failed on a problem the step cannot do without: an exception, or an unusable result."""


class _ExactProblem:
    """The problem with the binaries fixed: convex, with the disks and ``Z`` exact (Clarabel).

    Built once with the measured state, the target and the binaries as parameters.
    """

    def __init__(self, problem: MPCProblem) -> None:
        N, tube = problem.N, problem.tube
        self.problem = problem
        self.x = cp.Parameter(4)
        self.x_r = cp.Parameter(4)
        self.z = cp.Variable((N + 1, 4))
        self.v = cp.Variable((N, 2))
        z, v = self.z, self.v
        cons = [
            z[1:] == z[:-1] @ problem.A.T + v @ problem.B.T,
            z[N, 2:] == 0,  # the terminal set: at rest
            z[:, :2] @ problem.arena_normals.T <= np.tile(problem.arena_bounds, (N + 1, 1)),
            cp.norm(v, 2, axis=1) <= problem.u_bar,
        ]
        if N > 1:  # stage 0 free, stage N at rest
            cons.append(cp.norm(z[1:N, 2:], 2, axis=1) <= problem.v_bar)
        if tube.is_zero:
            cons.append(z[0] == self.x)
        else:
            eta = cp.Variable(4)
            lifted = tube.tail @ eta
            cons.append(cp.norm(eta, 2) <= 1.0)
            if tube.num_terms > 0:  # none for a tiny d_bar, whose tail alone is within the tolerance
                d = cp.Variable((tube.num_terms, 2))
                lifted = lifted + sum(tube.terms[lag] @ d[lag] for lag in range(tube.num_terms))
                cons.append(cp.norm(d, 2, axis=1) <= tube.d_bar)
            cons.append(z[0] + lifted == self.x)
        self.deltas: list[cp.Parameter] = []
        for obstacle in problem.obstacles:
            delta = cp.Parameter((N, obstacle.num_faces))
            big_m = np.tile(obstacle.big_m, (N, 1))
            rhs = np.tile(obstacle.bounds - obstacle.big_m, (N, 1)) + cp.multiply(delta, big_m)
            cons += [z[:N, :2] @ obstacle.normals.T >= rhs, z[1:, :2] @ obstacle.normals.T >= rhs]  # both ends (D32)
            self.deltas.append(delta)
        Qh, Rh, Ph = (np.linalg.cholesky(S).T for S in (problem.Q, problem.R, problem.P))
        x_r_row = cp.reshape(self.x_r, (1, 4), order="C")
        objective = (
            cp.sum_squares((z[:N] - x_r_row) @ Qh.T) + cp.sum_squares(v @ Rh.T) + cp.sum_squares(Ph @ (z[N] - self.x_r))
        )
        self.prob = cp.Problem(cp.Minimize(objective), cons)

    def __call__(self, x: FloatArray, x_r: FloatArray, deltas: Binaries) -> tuple[FloatArray, FloatArray, float] | None:
        """``(z, v, J)`` of the exact optimum for these binaries, or ``None``.

        ``None`` when the binaries leave the exact problem infeasible, or
        Clarabel reaches only an inaccurate solution: a plan whose constraints
        are not met to the solver's tolerance has no place in a robust plan,
        and the step falls back on the relaxation's cuts or on the shifted plan.

        Raises:
            SolverFailure: If Clarabel fails (cvxpy's ``SolverError``).
        """
        self.x.value = np.asarray(x, dtype=np.float64)
        self.x_r.value = np.asarray(x_r, dtype=np.float64)
        for parameter, value in zip(self.deltas, deltas, strict=True):
            parameter.value = value
        try:
            with warnings.catch_warnings():  # inaccurate solves are handled below
                warnings.simplefilter("ignore")
                self.prob.solve(solver=cp.CLARABEL)
        except cp.SolverError as error:
            raise SolverFailure(f"Clarabel failed on the exact problem: {error}") from error
        if self.prob.status != "optimal" or self.z.value is None or self.v.value is None:
            return None
        z, v = np.array(self.z.value), np.array(self.v.value)
        return z, v, self.problem.cost(z, v, x_r)


class _GaugeOracle:
    """``gamma_Z(e) = max{y' e : h_Z(y) <= 1}`` and the maximiser ``y``, a cut of ``Z`` when ``gamma > 1``.

    ``h_Z(y) = d_bar sum_l ||(A_K^l B)' y|| + ||T y||`` from the lifted
    representation, so this is a second-order cone program in ``y``, built
    once and solved by Clarabel with ``e`` as a parameter.
    """

    def __init__(self, problem: MPCProblem) -> None:
        tube = problem.tube
        self.tube = tube
        self.e = cp.Parameter(4)
        self.y = cp.Variable(4)
        support = sum(tube.d_bar * cp.norm(tube.terms[lag].T @ self.y, 2) for lag in range(tube.num_terms))
        support = support + cp.norm(tube.tail.T @ self.y, 2)
        self.prob = cp.Problem(cp.Maximize(self.e @ self.y), [support <= 1.0])
        axis = np.array([1.0, 0.0, 0.0, 0.0])
        self._axis = axis / float(tube.support(axis))  # h_Z = 1, returned with the gauge of a zero error

    def __call__(self, e: ArrayLike) -> tuple[float, FloatArray]:
        """``(gamma, y)`` with ``y`` scaled so that ``h_Z(y) = 1`` exactly.

        Then ``y' e' <= 1`` holds on ``Z`` whatever the solver's tolerance.
        ``gamma`` is ``+inf`` when Clarabel is inaccurate, so the caller cuts
        (the cut stays valid) rather than trusts a point it could not verify;
        it is 0 without a solve for an error within :data:`GAUGE_ZERO` of 0.

        Raises:
            SolverFailure: If Clarabel fails or returns no usable direction.
        """
        error = np.asarray(e, dtype=np.float64).reshape(4)
        if np.max(np.abs(error)) <= GAUGE_ZERO:
            return 0.0, self._axis.copy()
        self.e.value = error
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.prob.solve(solver=cp.CLARABEL)
        except cp.SolverError as failure:
            raise SolverFailure(f"Clarabel failed on the gauge of Z: {failure}") from failure
        if self.prob.status not in ("optimal", "optimal_inaccurate") or self.y.value is None:
            raise SolverFailure(f"the gauge oracle ended with status {self.prob.status}")
        y = np.array(self.y.value)
        scale = float(self.tube.support(y))
        if not (np.isfinite(scale) and scale > 0.0):
            raise SolverFailure("the gauge oracle returned no direction")
        y = y / scale
        gamma = float(y @ error) if self.prob.status == "optimal" else np.inf
        return gamma, y


@dataclass
class _Slot:
    """One agent's Gurobi model and memory."""

    m: Any
    z: Any
    v: Any
    init: Any
    z_base: Any
    deltas: list[Any]
    disks: list[tuple[Any, float]]
    z_seed: list[FloatArray] = field(default_factory=list)
    disk_seed: list[tuple[int, int, FloatArray]] = field(default_factory=list)
    plan: Plan | None = None
    cut_rounds: int = 0


class MPCWorker:
    """The tube MPC of D31-D34, one Gurobi model per agent (slot).

    ``act(x, target, slot)`` returns the input that steers toward rest at the
    target. Slots are independent agents, each with its own model and last
    plan; :meth:`reset` forgets a slot's plan and must be called when its
    episode ends, so the next step does not start from another episode's plan.

    Args:
        problem: The problem's data.
        num_slots: Number of agents.
    """

    def __init__(self, problem: MPCProblem, num_slots: int = 1) -> None:
        if num_slots < 1:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        self.problem = problem
        self.num_slots = int(num_slots)
        self._errors = 0
        self._z_dirs = z_directions()
        self._z_h = problem.tube.support(self._z_dirs)
        self._gauge = None if problem.tube.is_zero else _GaugeOracle(problem)
        self._exact = _ExactProblem(problem)
        self._warm_up()
        self._slots: list[_Slot] = []
        self._env: Any = gp.Env(empty=True)
        try:
            self._env.setParam("OutputFlag", 0)
            self._env.start()
            for _ in range(self.num_slots):
                self._slots.append(self._build_slot())
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------ setup
    def _warm_up(self) -> None:
        """Solve each conic problem once, so that cvxpy compiles it now and not within the first step's time.

        The instance is arbitrary (with every face on, it is infeasible), and
        a failure is ignored: only the compilation, done before the solver
        runs, matters. Clarabel keeps nothing between solves, so later solves
        do not depend on this one.
        """
        pr = self.problem
        x = pr.rest_state(0.5 * (pr.position_low + pr.position_high))
        try:
            self._exact(x, x, [np.ones((pr.N, obstacle.num_faces)) for obstacle in pr.obstacles])
        except SolverFailure:
            pass
        if self._gauge is not None:
            try:
                self._gauge(np.ones(4))
            except SolverFailure:
                pass

    def _build_slot(self) -> _Slot:
        """The relaxation as one Gurobi model; per solve only the measured state,
        the target, the binaries' bounds, the MIP start and the cuts change."""
        pr = self.problem
        N = pr.N
        z_lb = np.full((N + 1, 4), -GRB.INFINITY)
        z_ub = np.full((N + 1, 4), GRB.INFINITY)
        z_lb[:, :2], z_ub[:, :2] = pr.position_low, pr.position_high
        z_lb[1:N, 2:], z_ub[1:N, 2:] = -pr.v_bar, pr.v_bar  # stage 0 free
        z_lb[N, 2:], z_ub[N, 2:] = 0.0, 0.0  # terminal set: at rest
        m = gp.Model(env=self._env)
        m.Params.Threads = 1
        m.Params.MIPGap = pr.settings.mip_gap
        if pr.settings.work_limit is not None:
            m.Params.WorkLimit = pr.settings.work_limit
        z = m.addMVar((N + 1, 4), lb=z_lb, ub=z_ub, name="z")
        v = m.addMVar((N, 2), lb=-pr.u_bar, ub=pr.u_bar, name="v")
        disks: list[tuple[Any, float]] = [(z[1:N, 2:], pr.v_bar), (v, pr.u_bar)]
        for i, (rows, radius) in enumerate(disks):
            if rows.shape[0] > 0:
                m.addConstr(rows @ _POLYGON.T <= radius, name=f"polygon{i}")
        if pr.tube.is_zero:
            init = m.addConstr(z[0] == np.zeros(4), name="init")
            z_base = None
        else:
            init = None
            z_base = m.addConstr(-(z[0] @ self._z_dirs.T) <= self._z_h, name="z_base")
        m.addConstr(z[1:] == z[:-1] @ pr.A.T + v @ pr.B.T, name="dynamics")
        m.addConstr(z[:, :2] @ pr.arena_normals.T <= np.tile(pr.arena_bounds, (N + 1, 1)), name="arena")
        deltas = []
        for i, obstacle in enumerate(pr.obstacles):
            d = m.addMVar((N, obstacle.num_faces), vtype=GRB.BINARY, name=f"delta{i}")
            rhs = obstacle.bounds - obstacle.big_m
            m.addConstr(z[:N, :2] @ obstacle.normals.T - d * obstacle.big_m >= rhs, name=f"first_end{i}")
            m.addConstr(z[1:, :2] @ obstacle.normals.T - d * obstacle.big_m >= rhs, name=f"second_end{i}")
            m.addConstr(d.sum(axis=1) >= 1, name=f"any{i}")
            deltas.append(d)
        objective = sum(z[k] @ pr.Q @ z[k] + v[k] @ pr.R @ v[k] for k in range(N)) + z[N] @ pr.P @ z[N]
        m.setObjective(objective, GRB.MINIMIZE)
        m.update()
        return _Slot(m=m, z=z, v=v, init=init, z_base=z_base, deltas=deltas, disks=disks)

    def close(self) -> None:
        """Release the Gurobi models and environment; later calls do nothing."""
        for slot in self._slots:
            slot.m.dispose()
        self._slots = []
        if self._env is not None:
            self._env.dispose()
            self._env = None

    # ---------------------------------------------------------------- helpers
    def _set_target(self, slot: _Slot, x_r: FloatArray) -> None:
        """The linear and constant parts of the cost for the rest state ``x_r``."""
        pr = self.problem
        c = np.tile(-2.0 * pr.Q @ x_r, (pr.N + 1, 1))
        c[pr.N] = -2.0 * pr.P @ x_r
        slot.z.Obj = c
        slot.m.ObjCon = float(pr.N * x_r @ pr.Q @ x_r + x_r @ pr.P @ x_r)

    def _tube_ratio(self, slot: _Slot, x: FloatArray) -> float:
        tube = self.problem.tube
        if slot.plan is None or tube.is_zero:
            return float("nan")
        e = x - slot.plan[0][1]
        return float(max(np.linalg.norm(e[:2]) / tube.position_radius, np.linalg.norm(e[2:]) / tube.speed_radius))

    def _one_face_each(self, slot: _Slot) -> Binaries:
        """The relaxation's binaries with one face per obstacle and segment: of the faces it
        switched on (all satisfied by its plan), the one both ends clear by the most."""
        p = slot.z.X[:, :2]
        out = []
        for obstacle, d in zip(self.problem.obstacles, slot.deltas, strict=True):
            on = np.round(d.X)
            clearance = p @ obstacle.normals.T - obstacle.bounds  # (N+1, m)
            margin = np.where(on > 0, np.minimum(clearance[:-1], clearance[1:]), -np.inf)
            one = np.zeros_like(on)
            one[np.arange(on.shape[0]), margin.argmax(axis=1)] = 1.0
            out.append(one)
        return out

    def _relax(self, slot: _Slot) -> None:
        """Solve the slot's relaxation (Gurobi)."""
        slot.m.optimize()

    def _exact_plan(
        self, x: FloatArray, x_r: FloatArray, deltas: Binaries
    ) -> tuple[FloatArray, FloatArray, float] | None:
        """The exact problem for these binaries; a Clarabel failure counts as no plan, and as an error."""
        try:
            return self._exact(x, x_r, deltas)
        except SolverFailure:
            self._errors += 1
            return None

    def _optimize(self, slot: _Slot, x: FloatArray, x_r: FloatArray, bounds: Bounds) -> Plan | None:
        """The optimal plan, or ``None`` if there is none, the cuts did not converge or a solver failed.

        Raises:
            gurobipy.GurobiError: Gurobi's licence errors (:data:`LICENCE_ERRORS`).
        """
        try:
            if all(np.array_equal(lb, ub) for lb, ub in bounds):
                exact = self._exact_plan(x, x_r, [lb for lb, _ in bounds])
                return None if exact is None else exact[:2]
            return self._cut_loop(slot, x, x_r)
        except SolverFailure:
            self._errors += 1
            return None
        except gp.GurobiError as error:
            if getattr(error, "errno", None) in LICENCE_ERRORS:
                raise
            self._errors += 1
            return None

    def _cut_loop(self, slot: _Slot, x: FloatArray, x_r: FloatArray) -> Plan | None:
        """Relaxation, exact problem and cuts, until the exact plan is optimal to the gap (module docstring)."""
        m = slot.m
        cuts: list[Any] = []
        gap = self.problem.settings.mip_gap

        def disk_cut(which: int, i: int, n: FloatArray) -> None:
            rows, radius = slot.disks[which]
            cuts.append(m.addConstr(rows[i] @ n <= radius))

        def z_cut(y: FloatArray) -> None:
            cuts.append(m.addConstr(-(slot.z[0] @ y) <= 1.0 - y @ x))
            slot.z_seed.append(y)

        for which, i, n in slot.disk_seed:
            disk_cut(which, i, n)
        if self._gauge is not None:
            slot.z_seed = slot.z_seed[-Z_SEED_KEEP:]
            for y in list(slot.z_seed):
                cuts.append(m.addConstr(-(slot.z[0] @ y) <= 1.0 - y @ x))
        try:
            for rnd in range(MAX_CUT_ROUNDS):
                self._relax(slot)
                if m.SolCount == 0:
                    return None
                deltas = self._one_face_each(slot)
                bound = m.ObjBound
                exact = self._exact_plan(x, x_r, deltas)
                if exact is not None and exact[2] <= bound + gap * abs(exact[2]) + 1e-9:
                    slot.cut_rounds = rnd + 1
                    return exact[0], exact[1]
                added = len(cuts)
                plans: list[tuple[FloatArray, FloatArray, float]] = [(slot.z.X, slot.v.X, CUT_TOL)]
                if exact is not None:
                    plans.append((exact[0], exact[1], -1e-4))
                for z, v, slack in plans:
                    for which, Y in enumerate((z[1 : self.problem.N, 2:], v)):
                        radius = slot.disks[which][1]
                        norms = np.linalg.norm(Y, axis=1)
                        past = norms > radius * (1.0 + slack) + (CUT_TOL if slack > 0 else 0.0)
                        for row in np.flatnonzero(past):
                            disk_cut(which, int(row), Y[row] / norms[row])
                    if self._gauge is not None:
                        gamma, y = self._gauge(x - z[0])
                        if gamma > 1.0 + slack:
                            z_cut(y)
                if len(cuts) == added:  # the relaxation's plan already satisfies the exact sets
                    slot.cut_rounds = rnd + 1
                    return slot.z.X.copy(), slot.v.X.copy()
            return None
        finally:
            if cuts:
                m.remove(cuts)
                m.update()

    def _shift_is_feasible(self, x: FloatArray, z0: FloatArray) -> bool:
        """Whether the shifted plan, which starts at ``z0``, is feasible at the measured state ``x``.

        The shift of a feasible plan meets every other constraint by
        construction (theorem 4.1: every rest state is an equilibrium), so
        only the tube constraint ``x - z0 in Z``, which the last step's
        disturbance moved, is checked, to :data:`CANDIDATE_TOL`; without a
        tube it reads ``z0 = x``. A failed check counts as infeasible, and as an error.
        """
        error = x - z0
        if self._gauge is None:
            return bool(np.max(np.abs(error)) <= CANDIDATE_TOL)
        try:
            gamma, _ = self._gauge(error)
        except SolverFailure:
            self._errors += 1
            return False
        return gamma <= 1.0 + CANDIDATE_TOL

    def _choose(
        self, x: FloatArray, x_r: FloatArray, solution: Plan | None, candidate: Plan | None, candidate_cost: float
    ) -> tuple[Plan | None, int, int]:
        """Algorithm 3's choice: ``(plan, outcome, invalid_candidate)`` (module docstring)."""
        if solution is not None:
            cost = self.problem.cost(solution[0], solution[1], x_r)
            if candidate is None or cost <= candidate_cost + ACCEPT_TOL * (1.0 + abs(candidate_cost)):
                return solution, SOLVER, 0
            if self._shift_is_feasible(x, candidate[0][0]):
                return candidate, KEPT, 0
            return solution, SOLVER, 1  # cheaper, but not a feasible plan: no benchmark
        if candidate is not None:
            return candidate, CANDIDATE, int(not self._shift_is_feasible(x, candidate[0][0]))
        return None, EMERGENCY, 0

    # ----------------------------------------------------------------- online
    def reset(self, slot: int | None = None) -> None:
        """Forget the last plan, the cuts of ``Z`` and Gurobi's last solution of ``slot``, or of every slot.

        The slot's next step then solves as a fresh Worker would.
        """
        for s in range(self.num_slots) if slot is None else (slot,):
            state = self._slots[s]
            state.plan = None
            state.z_seed = []
            state.m.reset(0)

    def act(self, x: ArrayLike, target: ArrayLike, slot: int = 0) -> tuple[FloatArray, WorkerStep]:
        """One step for ``slot``: the input at the measured state ``x = (p, v)`` toward rest at ``target``.

        Raises:
            gurobipy.GurobiError: Gurobi's licence errors (:data:`LICENCE_ERRORS`).
        """
        pr = self.problem
        x = np.asarray(x, dtype=np.float64).reshape(4)
        x_r = pr.rest_state(target)
        s = self._slots[slot]
        started = time.perf_counter()
        self._errors = 0
        tube_ratio = self._tube_ratio(s, x)
        if s.init is not None:
            s.init.RHS = x
        else:
            s.z_base.RHS = self._z_h - self._z_dirs @ x
        self._set_target(s, x_r)
        bounds = pr.fixed_binaries(x)
        for d, (lb, ub) in zip(s.deltas, bounds, strict=True):
            d.LB = lb
            d.UB = ub
        free = pr.free_binaries(bounds)

        candidate = None if s.plan is None else pr.candidate(*s.plan)
        candidate_cost = np.inf if candidate is None else pr.cost(candidate[0], candidate[1], x_r)
        candidate_deltas = None if candidate is None else pr.binaries_of(candidate[0], bounds)
        s.disk_seed = []
        if candidate is not None:
            for which, Y in enumerate((candidate[0][1 : pr.N, 2:], candidate[1])):
                radius = s.disks[which][1]
                norms = np.linalg.norm(Y, axis=1)
                for i in np.flatnonzero(norms > 0.5 * radius):
                    s.disk_seed.append((which, int(i), Y[i] / norms[i]))
            s.z.Start = candidate[0]
            s.v.Start = candidate[1]
        else:
            s.z.Start = np.full((pr.N + 1, 4), GRB.UNDEFINED)
            s.v.Start = np.full((pr.N, 2), GRB.UNDEFINED)
        starts: list[FloatArray | None] = [None] * len(s.deltas) if candidate_deltas is None else list(candidate_deltas)
        for d, dv in zip(s.deltas, starts, strict=True):
            d.Start = np.full(d.shape, GRB.UNDEFINED) if dv is None else dv

        s.cut_rounds = 0
        solution = self._optimize(s, x, x_r, bounds)
        plan, outcome, invalid = self._choose(x, x_r, solution, candidate, candidate_cost)
        if plan is None:
            u = pr.emergency_input(x)
            s.plan = None
            cost = float("nan")
        else:
            u = pr.input(x, plan[0], plan[1])
            s.plan = (plan[0], plan[1])
            cost = pr.cost(plan[0], plan[1], x_r)
        info = WorkerStep(
            outcome=outcome,
            solve_ms=1e3 * (time.perf_counter() - started),
            free_binaries=free,
            tube_ratio=tube_ratio,
            cost=cost,
            cut_rounds=s.cut_rounds,
            plan=None if plan is None else plan[0],
            invalid_candidate=invalid,
            solver_errors=self._errors,
        )
        return u, info
