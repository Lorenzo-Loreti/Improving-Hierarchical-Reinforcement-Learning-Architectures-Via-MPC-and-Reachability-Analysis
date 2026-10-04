"""Robust tube MPC with non-convex state constraints: PPO+MPC's worker.

The controller of the tube-MPC note, "Robust Tube-Based Model Predictive
Control for Linear Systems with Non-Convex State Constraints" (kept outside
this repo, like the thesis itself; every section, equation, algorithm and
remark number below is the note's), specialised to this project's plant and
corridors.

Why it replaced MPCWorker (algorithms/mpc_worker.py) as PPO+MPC's worker on
2026-09-28:

- MPCWorker knew the corridor only at the *current* p_x. It looked the width
  up there once per solve and held it for the whole horizon, so it could not
  see a gate until it was already inside one -- which is how the old PPO+MPC
  crossed the slalom's gates "at whatever lateral phase it happened to be
  in" (see the goal-box block in ppo_mpc/ppo_mpc_train.py). This worker knows
  every state it may not enter, over the whole horizon: each gate's blocked
  bands are obstacles (1.5), derived from the scenario's width profile, and
  the non-convex free space they leave is encoded exactly with binaries
  (3.15). Each solve is therefore a mixed-integer program, with the speed
  and thrust disks (below) enforced by cutting planes.
- The environment can now be disturbed, by a bounded uniform w in W (see
  noise_bound_p in the scenarios' envs/config.py). This worker is robust
  to it: the real state stays in a tube x in z (+) Z around a nominal plan z,
  and z is planned against constraints tightened by Z, so no disturbance in
  W can push the real state into a wall (theorem 4.1). With W = {0}, the
  default environment, Z = {0} and this is a nominal MPC with z_0 = x.

MPCWorker stays in the repo because ppo_mpc_reach still uses it.

Disks, exactly (since 2026-10-04). The environment limits ||u|| <= u_max
and ||v|| <= v_max, and its disturbance is W = {||w_p|| <= b_p,
||w_v|| <= b_v}, two disks (envs/actuation.py; until then all three were
boxes, and so was every set here). The note's sets are polytopes; with disks
they stay exact, with no polygonal approximation anywhere, because of a
symmetry: rotating the plane by any angle, R = diag(R_theta, R_theta) on
[p; v], commutes with the closed loop (A_K = A1 (x) I_2, one 2x2 block A1 per
axis, since Q and R weigh both axes alike) and with the gain (K R =
R_theta K), and maps W onto itself. So it maps Z = (1 - alpha)^-1 (+)_l
A_K^l W onto itself, and so every projection of Z the tightening needs is
rotation-invariant: Z's velocity part and K Z are disks, of radii
r_v = h_Z(e_vx) and r_u = h_Z(K^T e_x). The tightened sets are then the
disks ||z_v|| <= v_max - r_v and ||v|| <= u_max - r_u, exactly (a disk minus
a disk is a disk); the walls are half-planes, tightened by Z's position
radius as before; and z_0's constraint x - z_0 in Z keeps its lifted form
(2.8), with every omega_l in the two disks of W. `TubeMPCWorker.__init__`
checks the commutation, so a change that broke it (different weights per
axis, say) fails loudly instead of quietly making the disks approximate. The
radii equal the half-widths the box W gave along the axes, since every
support function the algorithm evaluates is along a direction of the form
(a_p n, a_v n), where the two sets agree.

How it is solved (since 2026-10-04). The problem is mixed-integer, and its
continuous part is convex but not polyhedral: the speed and thrust disks,
and x - z_0 in Z. Two solvers split it:

- Gurobi solves a relaxation, as a mixed-integer QP: every disk replaced by
  the regular polygon of `BASE_SIDES` sides circumscribed around it (one
  side tangent at the +x axis, so a straight cruise along the corridor is
  exact), and Z by the half-spaces a^T (x - z_0) <= h_Z(a) along
  `_z_directions` (its support function is exact and cheap). Because the
  relaxation contains the exact problem, its lower bound (ObjBound) is a
  lower bound on the exact optimum, whatever binaries attain it.
- Clarabel, an interior-point conic solver, solves the exact problem with
  Gurobi's binaries fixed: a convex program with the disks and Z's lifted
  representation (2.8), x - z_0 = sum_l L_l omega_l with every omega_l in
  W's disks, as second-order cones (`_ExactProblem`, ~3-5 ms).

If the exact cost is within the MIP gap of the lower bound, the exact plan
is optimal for the exact problem to that gap -- the guarantee the worker
always had. Otherwise the relaxation is tightened where it was loose and
both are solved again: tangents at the exact plan's points on the edge of
their disks, Z's supporting half-space at its x - z_0, and, where the
relaxation's own plan left a disk or Z, the tangent there, Z's from the
separation oracle `_GaugeOracle`. When `_fix_unreachable` has fixed every
binary -- most steps, and every step on the tunnel -- the problem is already
convex and Clarabel solves it alone. Each solve starts with extra cuts that
make the relaxation tight where the plan is likely to be (tangents at the
shifted candidate's points (4.1), and the Z cuts of the slot's previous
solves); all are removed after the solve, and the Z cuts are remembered
until `reset(slot)`, so a solve depends on its inputs and on earlier solves
of the same episode only, as it already did through the candidate.

Two simpler designs were tried first, on 2026-10-04, and dropped:

- Gurobi alone, with the disks as quadratic constraints (an MIQCP). Its
  relaxations go through the barrier, which ran into "numerical trouble" on
  about one solve in ten on the slalom -- on instances Clarabel solves in
  milliseconds, even with all the binaries fixed -- and then stalled until
  the time limit. NumericFocus, BarHomogeneous, ScaleFlag, Aggregate and the
  outer-approximation MIQCPMethod each fixed some instances and not others.
- Gurobi alone, with cutting planes only (Kelley): tangents added where the
  plan leaves a disk, until it leaves none by more than 1e-6. Robust, but
  the violation fell only ~4x per round, so a step took ~8 MIQP solves
  (~60 ms) without a disturbance; and on Z's lifted representation, whose
  2s omegas the cost does not depend on, the solver moved them to a new
  vertex every round and often ran out of rounds.

Offline, once per worker (the note's table 5.1):
  1. K, P: the LQR gain and cost of the stage weights (3.1).
  2. Z: an RPI outer approximation of the error's minimal RPI set
     (algorithm 1), kept only as a support function and a lifted
     representation (remark 2.9), never as facets.
  3. The tightened sets: X0 (-) Z (3.9), V = U (-) KZ (3.6), and every
     obstacle enlarged to an outer approximation of O (+) (-Z) plus a margin
     rho (3.8).
  4. The terminal set -- not the note's; see "Terminal set" below.
  5. The big-M constants (3.16).
Online, every step and every environment: the mixed-integer problem (3.17)
with that terminal set, solved by Gurobi -- with the disks by cutting planes
(above) -- with the shifted candidate (4.1) as MIP start and as fallback
(algorithm 3).

Deliberate departures from the note, and why:

- Terminal set: "stop safely", X_f = {z in X-bar : velocity = 0}, instead of
  the note's x_r (+) Omega_inf (section 3.3). The note's terminal set and its
  guarantees are built around one fixed equilibrium target x_r. Here the
  target is the manager's goal, which moves every manager_freq steps: with
  X_f tied to it, a new goal can make P_N(x) infeasible from the current
  state and every such switch needs a fallback. (The note's remark 4.6
  describes exactly this role for a "global planner that provides a
  sequence of intermediate equilibria", which is what the manager is.) Every
  state at rest is an equilibrium of the nominal system under v = 0, and
  0 is in V, so X_f is positively invariant and the candidate (4.1) becomes
  z~_N = z_N, v~_{N-1} = 0 (see `_candidate`). Theorem 4.1 -- recursive
  feasibility and robust constraint satisfaction -- then carries over
  unchanged, and since no constraint depends on the goal it holds across
  goal changes too. Theorem 4.2 (convergence to x_r (+) Z) does not: on X_f
  the terminal cost does not decrease, so V_f(z+) - V_f(z) <= -l fails.
  Converging anywhere is the manager's job, not the worker's. Chosen on
  2026-09-28, over the note's X_f (which also needs goals reachable within
  N steps) and over no terminal constraint at all (no guarantee; remark 4.6).
  Consequence for the horizon: every plan has to be able to stop within N
  steps, so N below the nominal stopping time caps the planned speed.
  `summary()` prints that stopping time.
- A margin rho also on the walls, not only on the obstacles (3.8). The
  environments count a state *on* a wall as a contact (`p_y <= y_lo`),
  where the note's admissible set (1.6) removes only the interior of the
  obstacles and keeps the boundary. The same rho covers both, and it must
  exceed M * IntFeasTol (section 5.5): see `rho`.
- Obstacles that no nominal state can reach at a given prediction stage have
  their binaries fixed, per stage and per solve (remark 3.12, "excluding
  obstacles that cannot be reached within the horizon"). The reachable box
  is exact -- every nominal plan stays inside it -- so the fixing removes no
  feasible plan; see `_fix_unreachable`.
- No inter-sample constraints (remark 3.10). The environments check contact
  only at the sampling instants, so only those are constrained, as in the
  note's default formulation.
- The note's X0 is a compact polytope; here it is the environment's own
  position box (x_min, x_max), which is what env.step clips the position
  to, and its speed disk. Keeping the real state strictly inside both
  matters beyond the walls: the speed limit (envs/actuation.py) or a
  saturated input would make the plant nonlinear and the error dynamics
  (3.5) would no longer hold. That is why the planned speed is tightened
  too, to ||z_v|| <= v_max - r_v. The tube then even keeps the environment's
  speed limit from ever acting: v + u dt, the velocity before the step's
  disturbance, is z_v' + (A_K e)_v, whose norm is at most (v_max - r_v) +
  (r_v - b_v) < v_max, since A_K Z (+) W lies in Z.
  The price is smaller than that cap suggests, because z_0 is a decision
  variable (remark 3.8): the plan can put its nominal state behind the real
  one, anywhere inside the tube, so the real state rides at the front of
  it. Measured on the disturbed slalom with the per-axis limits and the box
  W of the time (|w_p| <= 0.005, |w_v| <= 0.05, 2026-09-28): the plan's v_x
  never exceeded its cap of 0.989 m/s, while the real v_x averaged 1.08 m/s
  and peaked at 1.195, still inside v_max as the tightening guaranteed.
  Episodes then took ~4.5 steps (~6%) longer than flat PPO's and hPPO's,
  where the cap alone would cost ~21% at cruise speed (1.2 / 0.989). The
  disks have the same radii, 0.989 m/s included.
- The obstacles are enlarged by Z with their own normals (proposition 2.5),
  an outer approximation of O (+) (-Z), as before. With Z's position part
  a disk the exact sum would have rounded corners, so the enlarged gate
  walls keep square corners up to (sqrt(2) - 1) * r_p (3.4 cm at the
  disturbed level) beyond them: conservative, on the safe side, and the
  only place where a set is larger than the exact one. (The exact rounded
  corner would need a non-convex constraint, ||p - corner|| >= r.)
"""

import time
import warnings

import numpy as np
import scipy.linalg as sla
from scipy.optimize import linprog

import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB

FREE = (None, None)  # an unbounded LP variable

# The outcome of one solve, as `act` reports it:
#   SOLVER     the solver's plan, no worse than the candidate (algorithm 3, line 9);
#   CANDIDATE  the shifted candidate, because the solver failed or did worse (line 11);
#   EMERGENCY  no plan at all: the solver failed and there was no candidate,
#              i.e. the first state of an episode lies outside X_N. The
#              action is then a plain brake, outside every guarantee.
SOLVER, CANDIDATE, EMERGENCY = 0, 1, 2


# The relaxation and its refinement (module docstring): the sides of the
# disks' starting polygon -- 16 sides exceed the disk by at most
# 1/cos(pi/16) - 1 = 2% -- how close to its edge a point counts as on it, or
# past it, when cuts are placed, how many rounds one step may take before
# giving up on the solver's plan (algorithm 3's candidate then covers it), and
# how many of the slot's latest Z cuts each solve starts with.
BASE_SIDES = 16
CUT_TOL = 1e-6
MAX_CUT_ROUNDS = 25
Z_SEED_KEEP = 32
_POLYGON = np.array([[np.cos(t), np.sin(t)] for t in 2.0 * np.pi * np.arange(BASE_SIDES) / BASE_SIDES])


def _z_directions():
    """The directions Z's starting outer approximation is cut along: every
    a = (cos(psi) n, sin(psi) n) for 8 headings n and 8 mixing angles psi --
    the directions in which a position error and a velocity error point the
    same way, the shape of every support function Z's construction
    evaluates (module docstring). 64 half-spaces, rotation-symmetric like Z."""
    dirs = []
    for phi in 2.0 * np.pi * np.arange(8) / 8:
        n = np.array([np.cos(phi), np.sin(phi)])
        for psi in 2.0 * np.pi * np.arange(8) / 8:
            dirs.append(np.concatenate([np.cos(psi) * n, np.sin(psi) * n]))
    return np.array(dirs)


class _ExactProblem:
    """The worker's problem (3.17) with the binaries fixed: convex, with the
    speed and thrust disks and x - z_0 in Z exact, as second-order cones,
    built once with cvxpy (the measured x, the target and the binaries as
    parameters) and solved by Clarabel (module docstring)."""

    def __init__(self, w):
        N = w.N
        self.w = w
        self.x = cp.Parameter(4)
        self.x_r = cp.Parameter(4)
        self.z = cp.Variable((N + 1, 4))
        self.v = cp.Variable((N, 2))
        z, v = self.z, self.v
        lo, hi = w.Xbar0.bounds
        cons = [z[1:] == z[:-1] @ w.A.T + v @ w.B.T,
                z[N, 2:] == 0,                                       # the terminal set
                z[:, :2] >= np.tile(lo[:2], (N + 1, 1)),
                z[:, :2] <= np.tile(hi[:2], (N + 1, 1)),
                cp.norm(z[1:N, 2:], 2, axis=1) <= w.v_bar,           # stage 0 free, as in the MIQP
                cp.norm(v, 2, axis=1) <= w.u_bar]
        L = w.Z.lifted_matrices()
        if L:
            om = cp.Variable((len(L), 4))
            b_p, b_v = w.W.radii
            cons += [z[0] + sum(L[l] @ om[l] for l in range(len(L))) == self.x,
                     cp.norm(om[:, :2], 2, axis=1) <= b_p,
                     cp.norm(om[:, 2:], 2, axis=1) <= b_v]
        else:
            cons.append(z[0] == self.x)
        # (3.15b) with the binaries as parameters; one face per obstacle and
        # stage is on, as the MIQP chose.
        self.d = []
        for Ob, M in zip(w.Obar, w.M):
            d = cp.Parameter((N + 1, Ob.H.shape[0]))
            cons.append(z @ Ob.H.T >= np.tile(Ob.h - M, (N + 1, 1)) + cp.multiply(d, np.tile(M, (N + 1, 1))))
            self.d.append(d)
        Qh, Rh, Ph = (np.linalg.cholesky(S).T for S in (w.Q, w.R, w.P))
        obj = (cp.sum_squares((z[:N] - cp.reshape(self.x_r, (1, 4), order="C")) @ Qh.T)
               + cp.sum_squares(v @ Rh.T) + cp.sum_squares(Ph @ (z[N] - self.x_r)))
        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def __call__(self, x, x_r, deltas):
        """(z, v, J) of the exact optimum for these binaries, or None if they
        leave the exact problem infeasible, or Clarabel fails or reaches only
        an inaccurate solution: one whose constraints are not met to its
        tolerance has no place in a robust plan, and None sends the step to
        the relaxation's cuts or to algorithm 3's candidate instead. Seen on
        early PPO+MPC training, rarely (0 of ~3000 solves under random goals,
        2026-10-04)."""
        self.x.value = np.asarray(x, dtype=float)
        self.x_r.value = np.asarray(x_r, dtype=float)
        for d, dv in zip(self.d, deltas):
            d.value = dv
        try:
            # Scoped: cvxpy warns on every inaccurate solve, which is handled
            # here, and must not flood a training log (see MPCWorker's
            # _solve_cvxpy for why not the process-global filter).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.prob.solve(solver=cp.CLARABEL)
        except cp.SolverError:
            return None
        if self.prob.status != "optimal" or self.z.value is None:
            return None
        z, v = self.z.value.copy(), self.v.value.copy()
        return z, v, self.w.cost(z, v, x_r)


class _GaugeOracle:
    """gamma_Z(e) = min{t >= 0 : e in t Z} = max{y^T e : h_Z(y) <= 1}, and the
    maximiser y: a separating half-space y^T e' <= 1 of Z whenever
    gamma_Z(e) > 1 (module docstring). h_Z(y) = sum_l h_W(L_l^T y), from the
    lifted representation (2.8), with h_W in closed form, so this is a
    second-order cone program in y alone, built once and re-solved by
    Clarabel with e as a parameter."""

    def __init__(self, Z, W):
        self.Z = Z
        b_p, b_v = W.radii
        self.e = cp.Parameter(4)
        self.y = cp.Variable(4)
        h = sum(b_p * cp.norm(Ll.T[:2] @ self.y) + b_v * cp.norm(Ll.T[2:] @ self.y)
                for Ll in Z.lifted_matrices())
        self.prob = cp.Problem(cp.Maximize(self.e @ self.y), [h <= 1])

    def __call__(self, e):
        """(gamma, y) with y scaled so that h_Z(y) = 1 exactly, which makes
        y^T e' <= 1 valid for Z whatever the solver's tolerance, and gamma =
        y^T e <= gamma_Z(e). They are equal when Clarabel solves to its
        tolerance; when it reaches only an inaccurate solution, gamma is
        returned as +inf instead, so the caller cuts (the cut stays valid)
        rather than trusts a point it could not verify."""
        self.e.value = np.asarray(e, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.prob.solve(solver=cp.CLARABEL)
        if self.prob.status not in ("optimal", "optimal_inaccurate") or self.y.value is None:
            raise RuntimeError(f"gauge oracle failed: {self.prob.status}")
        y = self.y.value / self.Z.support(self.y.value)
        gamma = float(y @ e) if self.prob.status == "optimal" else np.inf
        return gamma, y


def _project_disk(u, radius):
    """u scaled radially onto ||u|| <= radius where it lies outside."""
    n = np.linalg.norm(u)
    return u if n <= radius else u * (radius / n)


# =============================================================================
# Polytopes and support functions (section 2.1)
# =============================================================================
class Polytope:
    """P = {x : H x <= h}. A polytope built with `box` also keeps its bounds,
    so its support function has a closed form (proposition 2.3(d)); any other
    support function costs one LP."""

    def __init__(self, H, h):
        self.H = np.atleast_2d(np.asarray(H, dtype=float))
        self.h = np.asarray(h, dtype=float).ravel()
        if self.H.shape[0] != self.h.size:
            raise ValueError("H and h have incompatible dimensions")
        self._box = None  # (lb, ub) when P is a box

    @property
    def dim(self):
        return self.H.shape[1]

    @staticmethod
    def box(lb, ub):
        lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
        if np.any(lb > ub):
            raise ValueError(f"empty box: lb={lb}, ub={ub}")
        n = lb.size
        P = Polytope(np.vstack([np.eye(n), -np.eye(n)]), np.concatenate([ub, -lb]))
        P._box = (lb, ub)
        return P

    @property
    def bounds(self):
        """(lb, ub) of a box; an error for any other polytope."""
        if self._box is None:
            raise ValueError("not a box")
        return self._box

    def support(self, a):
        """h_P(a) = max_{x in P} a^T x (definition 2.2)."""
        a = np.asarray(a, dtype=float).ravel()
        if self._box is not None:
            lb, ub = self._box
            return float(np.sum(np.where(a >= 0, a * ub, a * lb)))
        res = linprog(-a, A_ub=self.H, b_ub=self.h, bounds=[FREE] * self.dim, method="highs")
        if res.status == 3:
            return np.inf
        if res.status != 0:
            raise RuntimeError(f"support-function LP failed: {res.message}")
        return float(-res.fun)

    @property
    def is_zero(self):
        return bool(np.all(self.h == 0))

    @property
    def has_interior_origin(self):
        return bool(np.all(self.h > 0))

    def scaling(self, M):
        """The smallest alpha with M P in alpha P, by P's facets (2.5)."""
        return max(self.support(M.T @ g) / hg for g, hg in zip(self.H, self.h))


class DiskProduct:
    """W = {w in R^4 : ||w[:2]|| <= b_p, ||w[2:]|| <= b_v}: the product of a
    disk on the positions and a disk on the velocities, the environments'
    disturbance set since 2026-10-04 (envs/actuation.py). Not a polytope, so
    it carries the two operations algorithm 1 and the online problem need,
    in closed form."""

    def __init__(self, b_p, b_v):
        self.radii = np.array([b_p, b_v], dtype=float)
        if np.any(self.radii < 0):
            raise ValueError(f"radii must be >= 0, got {self.radii}")

    @property
    def is_zero(self):
        return bool(np.all(self.radii == 0))

    @property
    def has_interior_origin(self):
        return bool(np.all(self.radii > 0))

    @property
    def bounds(self):
        """(lb, ub) of W's bounding box."""
        b = np.repeat(self.radii, 2)
        return -b, b

    def support(self, a):
        """h_W(a) = b_p ||a_p|| + b_v ||a_v||."""
        a = np.asarray(a, dtype=float).ravel()
        return float(self.radii[0] * np.linalg.norm(a[:2]) + self.radii[1] * np.linalg.norm(a[2:]))

    def scaling(self, M):
        """The smallest alpha with M W in alpha W: max over the output blocks
        i of sum_j ||M_ij|| b_j / b_i, M_ij the 2x2 blocks and ||.|| the
        spectral norm. Exact when every block is a multiple of a rotation, as
        every power of A_K is here (each block is a multiple of I_2): the
        sup of ||sum_j M_ij w_j|| over the disks is then reached by turning
        every term the same way. An upper bound otherwise, which still makes
        M W in alpha W true, so algorithm 1 still returns an RPI set."""
        blocks = [[np.linalg.norm(M[2 * i:2 * i + 2, 2 * j:2 * j + 2], 2) for j in range(2)] for i in range(2)]
        return max(sum(blocks[i][j] * self.radii[j] for j in range(2)) / self.radii[i] for i in range(2))


def pontryagin_diff(P, support_S):
    """P (-) S = {x : H x <= h - h_S(H^T)}, exact (proposition 2.4). It keeps
    P's normals, so the difference of a box is a box again."""
    h = P.h - np.array([support_S(row) for row in P.H])
    if P._box is None:
        return Polytope(P.H, h)
    n = P.dim
    return Polytope.box(-h[n:], h[:n])


def minkowski_outer(P, support_S):
    """An outer approximation of P (+) S with P's normals (proposition 2.5).
    For an obstacle, being larger than the exact sum is the safe side, and
    keeping the normals keeps the logical structure (1.7)."""
    return Polytope(P.H, P.h + np.array([support_S(row) for row in P.H]))


def dlqr(A, B, Q, R):
    """Discrete-time LQR (3.1), with the convention u = K x."""
    P = sla.solve_discrete_are(A, B, Q, R)
    P = 0.5 * (P + P.T)
    K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


# =============================================================================
# Robust invariant set (section 2.2)
# =============================================================================
class RPIOuterApprox:
    """Z = (1 - alpha)^-1 (+)_{l<s} A_K^l W with A_K^s W in alpha W: a
    compact RPI set for e+ = A_K e + w that contains the minimal one and lies
    within eps of it in the infinity norm (proposition 2.8, algorithm 1).

    Only the support function (2.7) and the lifted representation (2.8) are
    kept (remark 2.9): computing Z's facets means a Minkowski sum of s
    polytopes, and nothing here needs them.

    W is anything with `support`, `scaling`, `is_zero` and
    `has_interior_origin`: a Polytope, or the DiskProduct the worker uses.

    W = {0} (the deterministic environment) gives Z = {0}: s = 0, no powers,
    a support function that is identically 0 and no lifted variables, so the
    online problem's first constraint becomes z_0 = x. Any other W needs 0 in
    its interior (assumption 2): for the worker's disks, both radii > 0.
    """

    def __init__(self, AK, W, eps=1e-2, s_max=300):
        if np.max(np.abs(np.linalg.eigvals(AK))) >= 1:
            raise ValueError("A_K must be Schur")
        self.AK, self.W, self.eps = AK, W, eps
        if W.is_zero:
            self.s, self.alpha, self.powers = 0, 0.0, []
            return
        if not W.has_interior_origin:
            raise ValueError(
                "0 must lie in the interior of W (assumption 2): both disk radii (every "
                "half-width, for a box) must be > 0, or all of them 0 for the "
                "deterministic environment. A disturbance on the velocities alone makes W "
                "flat, and algorithm 1's test A_K^s W in alpha W can then never pass; "
                "bound the positions by a small positive number instead")
        n = AK.shape[0]
        powers = [np.eye(n)]
        Mp, Mm = np.zeros(n), np.zeros(n)
        for s in range(1, s_max + 1):
            Ai = powers[-1]                                     # A_K^{s-1}
            Mp += [W.support(Ai[j]) for j in range(n)]          # (2.6)
            Mm += [W.support(-Ai[j]) for j in range(n)]
            As = AK @ Ai                                        # A_K^s
            alpha = W.scaling(As)                               # (2.5)
            if alpha < 1 and alpha <= eps / (eps + max(Mp.max(), Mm.max())):
                break
            powers.append(As)
        else:
            raise RuntimeError("algorithm 1 did not stop within s_max: raise s_max or eps")
        self.s, self.alpha, self.powers = s, float(alpha), powers

    def support(self, a):
        """h_Z(a) (2.7)."""
        if not self.powers:
            return 0.0
        a = np.asarray(a, dtype=float)
        return sum(self.W.support(Ai.T @ a) for Ai in self.powers) / (1.0 - self.alpha)

    def lifted_matrices(self):
        """L_l = A_K^l / (1 - alpha): e in Z <=> e = sum_l L_l w_l, w_l in W (2.8)."""
        return [Ai / (1.0 - self.alpha) for Ai in self.powers]


# =============================================================================
# The corridor as obstacles (section 1.2)
# =============================================================================
def corridor_obstacles(width_profile):
    """The states a width profile (envs/width_profile.py) forbids, as the
    note's polytopic obstacles (1.5) in the state space [p_x, p_y, v_x, v_y].

    A segment narrower than the profile's envelope blocks a band below its
    opening, {x_start <= p_x <= x_end, p_y <= y_lo}, and/or one above it,
    {x_start <= p_x <= x_end, p_y >= y_hi}. Each is a *cylinder* in the
    note's sense: it constrains only the position, so it extends over every
    velocity. It is also unbounded toward the wall it leans on, which the
    note allows (section 1.2): the wall itself is X0's, so the obstacle needs
    no face there, and dropping it saves a binary per stage. An infinite
    segment end contributes no face either. On the slalom this gives four
    obstacles of three faces each, two per gate; the tunnel's constant
    profile gives none, and the MIQP is then a plain QP.

    The environment's segment covers x_start <= p_x < x_end, and an
    obstacle's closure also takes p_x = x_end: the safe side.
    """
    env_lo, env_hi = width_profile.envelope()
    obstacles = []
    for seg in width_profile.segments:
        y_lo = seg.center_y - seg.half_width
        y_hi = seg.center_y + seg.half_width
        rows, offsets = [], []
        if np.isfinite(seg.x_start):
            rows.append([-1.0, 0.0, 0.0, 0.0])
            offsets.append(-seg.x_start)
        if np.isfinite(seg.x_end):
            rows.append([1.0, 0.0, 0.0, 0.0])
            offsets.append(seg.x_end)
        if y_lo > env_lo:
            obstacles.append(Polytope(rows + [[0.0, 1.0, 0.0, 0.0]], offsets + [y_lo]))
        if y_hi < env_hi:
            obstacles.append(Polytope(rows + [[0.0, -1.0, 0.0, 0.0]], offsets + [-y_hi]))
    return obstacles


# =============================================================================
# The worker
# =============================================================================
class TubeMPCWorker:
    """Tube MPC for the planar double integrator, with the corridor as
    obstacles; one Gurobi model per environment slot.

    `act(x, target, slot)` returns the input u = v_0* + K (x - z_0*) (3.14)
    that steers toward the equilibrium x_r = (target, 0, 0): the manager's
    goal is a point to reach, and the worker plans to come to rest at it or
    as close as the constraints allow. The target may lie anywhere, inside an
    obstacle or past a wall included: it enters only the cost, never a
    constraint.

    Slots are independent environments. Each keeps its own Gurobi model and
    its own last plan, from which the next step's candidate (4.1) is built;
    `reset(slot)` forgets that plan, and must be called when the slot's
    episode ends. Sharing one slot between two episodes, or between a
    rollout and an evaluation, would hand one of them the other's candidate
    -- a plan for a different state, whose feasibility nothing guarantees.

    Parameters, besides the plant's:
      horizon     N, the prediction horizon, fixed and receding.
      q_pos, q_vel, r
                  the stage weights Q = diag(q_pos, q_pos, q_vel, q_vel),
                  R = r I, which also give the ancillary gain K and the
                  terminal cost P (3.1). One gain for both roles, as in the
                  note's code (remark 3.7 allows two). The defaults are the
                  old MPCWorker's.
      noise_bound_p, noise_bound_v
                  the radii of the disks of W the tube is designed for: the
                  worker's *model* of the disturbance, which `from_env` takes
                  from the environment it will run in.
      rho         the safety margin (3.8), in metres, applied to the
                  obstacles and to the walls. It must exceed M * IntFeasTol,
                  about 13 * 1e-5 here, plus the feasibility tolerance and
                  the environment's float32 rounding (section 5.5); 1e-3 is
                  the note's value and clears all three several times over.
      rpi_eps     algorithm 1's tolerance: Z lies within this distance of the
                  minimal RPI set, in the infinity norm.
      mip_gap     Gurobi's relative MIP gap.
      work_limit  Gurobi's WorkLimit, in work units, or None for none. A
                  bound on deterministic work rather than on wall clock
                  (the note's T_max), so a limited solve is reproducible.
                  Algorithm 3's candidate covers a solve it cuts short.
    Gurobi runs single-threaded, which with no time limit makes every solve
    deterministic, and keeps many training runs from oversubscribing the CPU.
    """

    SETTINGS = ("horizon", "q_pos", "q_vel", "r", "noise_bound_p", "noise_bound_v",
                "rho", "rpi_eps", "mip_gap", "work_limit")

    def __init__(self, dt, v_max, u_max, x_min, x_max, width_profile,
                 horizon=10, q_pos=10.0, q_vel=1.0, r=0.1,
                 noise_bound_p=0.0, noise_bound_v=0.0,
                 rho=1e-3, rpi_eps=1e-2, mip_gap=1e-4, work_limit=None, num_slots=1):
        self.dt, self.N, self.num_slots = float(dt), int(horizon), int(num_slots)
        if self.N < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.settings = dict(horizon=self.N, q_pos=q_pos, q_vel=q_vel, r=r,
                             noise_bound_p=noise_bound_p, noise_bound_v=noise_bound_v,
                             rho=rho, rpi_eps=rpi_eps, mip_gap=mip_gap, work_limit=work_limit)

        # Plant: the environments' double integrator, block-diagonal per axis.
        self.A = np.array([[1.0, 0.0, dt, 0.0],
                           [0.0, 1.0, 0.0, dt],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
        self.B = np.array([[0.5 * dt ** 2, 0.0],
                           [0.0, 0.5 * dt ** 2],
                           [dt, 0.0],
                           [0.0, dt]])
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(float)
        self.R = r * np.eye(2)

        # 1) Ancillary and terminal gain, terminal cost (3.1).
        self.K, self.P = dlqr(self.A, self.B, self.Q, self.R)
        self.AK = self.A + self.B @ self.K

        # The rotation symmetry every disk below rests on (module docstring).
        self._check_rotation_symmetry()

        # 2) The error's RPI set. X0 and U are kept as boxes -- the position
        # box and the bounding boxes of the speed and thrust disks -- for
        # everything that wants bounds (the variables' bounds, big-M, the
        # reachable box); the disks themselves enter by cutting planes.
        self.u_max, self.v_max = float(u_max), float(v_max)
        # The velocity rows from v_max itself: the environment's x_min/x_max
        # are float32, whose 1.2 is 1.2000000477.
        x_min, x_max = np.array(x_min, dtype=float), np.array(x_max, dtype=float)
        x_min[2:], x_max[2:] = -self.v_max, self.v_max
        self.X0 = Polytope.box(x_min, x_max)
        self.U = Polytope.box([-u_max, -u_max], [u_max, u_max])
        self.W = DiskProduct(noise_bound_p, noise_bound_v)
        self.Z = RPIOuterApprox(self.AK, self.W, eps=rpi_eps)

        # 3) Tightened constraints (3.6), (3.9), and the margin on the walls
        # (see the module docstring): only the p_y rows, the walls; p_x's
        # bounds are the environment's clip box, not a contact. Every row is
        # tightened by Z's support along it, which on the velocity rows is
        # Z's velocity radius r_v and on V's rows K Z's radius r_u, so
        # these boxes are exactly the bounding boxes of the tightened disks,
        # whose radii are `v_bar` and `u_bar`.
        Xbar0 = pontryagin_diff(self.X0, self.Z.support)
        lo, hi = (b.copy() for b in Xbar0.bounds)
        lo[1] += rho
        hi[1] -= rho
        self.Xbar0 = Polytope.box(lo, hi)
        self.V = pontryagin_diff(self.U, lambda a: self.Z.support(self.K.T @ a))
        v_lo, v_hi = self.V.bounds
        if np.any(v_lo >= 0) or np.any(v_hi <= 0):
            raise ValueError(
                f"V = U (-) KZ = [{v_lo}, {v_hi}] leaves no room around 0: the "
                "disturbance takes the whole input range")
        if lo[2] >= 0 or hi[2] <= 0 or lo[3] >= 0 or hi[3] <= 0:
            raise ValueError("X0 (-) Z has no velocity left around 0: the disturbance is too large")
        self.v_bar = float(hi[2])          # ||z_v|| <= v_max - r_v
        self.u_bar = float(v_hi[0])        # ||v||   <= u_max - r_u
        # Z's cutting planes (module docstring): the starting directions,
        # each with its support, and the separation oracle. None of it for
        # Z = {0}, where z_0 = x.
        if self.Z.powers:
            self._z_dirs = _z_directions()
            self._z_h = np.array([self.Z.support(a) for a in self._z_dirs])
            self._gauge = _GaugeOracle(self.Z, self.W)

        # ...and the enlarged obstacles (3.8).
        self.obstacles = corridor_obstacles(width_profile)
        self.Obar = []
        for O in self.obstacles:
            Ob = minkowski_outer(O, lambda a: self.Z.support(-a))
            Ob.h = Ob.h + rho * np.linalg.norm(Ob.H, axis=1)
            self.Obar.append(Ob)

        # 5) Big-M (3.16): the smallest constant that makes a face's
        # constraint inactive anywhere in X0 (-) Z. A little padding keeps a
        # constant that comes out exactly at the bound from being active.
        self.M = [np.maximum(0.0, Ob.h + np.array([self.Xbar0.support(-row) for row in Ob.H])) + 1e-6
                  for Ob in self.Obar]

        self._build_models()
        self._exact = _ExactProblem(self)
        self._prev = [None] * self.num_slots   # each slot's last plan, (z, v)

    def _check_rotation_symmetry(self):
        """A_K R = R A_K and K R = R_theta K for rotations R = diag(R_theta,
        R_theta) of the plane (module docstring): what makes Z
        rotation-invariant, and so every tightened set below an exact disk.
        True for the per-axis double integrator with Q and R weighing both
        axes alike, which is all this class builds; checked rather than
        assumed, at a few angles."""
        for theta in (0.3, 1.1, 2.5):
            c, s_ = np.cos(theta), np.sin(theta)
            Rt = np.array([[c, -s_], [s_, c]])
            R = np.kron(np.eye(2), Rt)          # diag(R_theta, R_theta) on [p; v]
            if not (np.allclose(self.AK @ R, R @ self.AK, atol=1e-10)
                    and np.allclose(self.K @ R, Rt @ self.K, atol=1e-10)):
                raise ValueError(
                    "the closed loop is not rotation-symmetric, so the tightened speed and "
                    "thrust sets would not be disks: the plant and weights must treat both "
                    "axes alike")

    @classmethod
    def from_env(cls, env, num_slots=1, **settings):
        """A worker for `env`'s plant and corridor. The disturbance model
        defaults to the environment's own W; a reloaded controller passes the
        one it was trained with instead (see ppo_mpc_train.load_agent)."""
        settings.setdefault("noise_bound_p", float(env.noise_bound_p))
        settings.setdefault("noise_bound_v", float(env.noise_bound_v))
        return cls(dt=env.dt, v_max=env.v_max, u_max=env.u_max,
                   x_min=np.asarray(env.x_min, dtype=float), x_max=np.asarray(env.x_max, dtype=float),
                   width_profile=env.width_profile, num_slots=num_slots, **settings)

    # ------------------------------------------------------------------ setup
    def _build_models(self):
        """The mixed-integer problem (3.17), once per slot. Only what changes
        between solves is touched afterwards: the right-hand side of the
        initial constraint (the measured x), the linear objective (the
        target), the binaries' bounds (`_fix_unreachable`), the MIP start,
        and the cuts `_optimize` adds and removes within a solve.

        The tightened speed and thrust disks start as their circumscribed
        polygons and are listed in the slot's "disks" as (rows, radius),
        `rows` an (n, 2) MVar of the points one disk holds; Z starts as
        `_z_directions`' half-spaces (module docstring). "z_seed" keeps the
        directions of the Z cuts of the slot's earlier solves this episode,
        "disk_seed" the tangents `act` asks for from the candidate."""
        self._gp_env = gp.Env(empty=True)
        self._gp_env.setParam("OutputFlag", 0)
        self._gp_env.start()
        N = self.N
        z_lb = np.tile(self.Xbar0.bounds[0], (N + 1, 1))
        z_ub = np.tile(self.Xbar0.bounds[1], (N + 1, 1))
        # Terminal set: at rest (see the module docstring). Inside the
        # tightened velocity box, which contains 0 (checked above).
        z_lb[N, 2:] = 0.0
        z_ub[N, 2:] = 0.0
        # No speed constraint on stage 0, neither the box nor the disk below.
        # It is redundant: the measured state is what it is, and every later
        # real velocity is covered by stages 1..N (x_k+1 = z_1 + e_1 with
        # e_1 in Z), so theorem 4.1 does not use it. And it is harmful: on
        # the deterministic environment z_0 = x, and a vehicle cruising at
        # the speed limit has ||x_v|| within float32 rounding of v_max,
        # sometimes just above it. The disk on a fixed point at its edge has
        # no interior, and Gurobi's barrier then stalled on the root
        # relaxation (2026-10-04: 18 of the 19 solves of one slalom episode
        # that ran into a 2 s limit had ||x_v|| within 1e-7 of 1.2).
        z_lb[0, 2:] = -GRB.INFINITY
        z_ub[0, 2:] = GRB.INFINITY
        v_lb = np.tile(self.V.bounds[0], (N, 1))
        v_ub = np.tile(self.V.bounds[1], (N, 1))

        self._models = []
        for _ in range(self.num_slots):
            m = gp.Model(env=self._gp_env)
            m.Params.Threads = 1
            m.Params.MIPGap = self.settings["mip_gap"]
            if self.settings["work_limit"] is not None:
                m.Params.WorkLimit = self.settings["work_limit"]

            z = m.addMVar((N + 1, 4), lb=z_lb, ub=z_ub, name="z")
            v = m.addMVar((N, 2), lb=v_lb, ub=v_ub, name="v")
            # The tightened disks (3.6), (3.9), each as its circumscribed
            # polygon, a relaxation the cuts refine. The speed disk binds
            # stages 1 to N - 1: stage N's velocity is fixed at 0 by the
            # terminal set, and stage 0's is left free (see z_lb above).
            disks = [(z[1:N, 2:], self.v_bar), (v, self.u_bar)]
            for i, (rows, radius) in enumerate(disks):
                if rows.shape[0] > 0:
                    m.addConstr(rows @ _POLYGON.T <= radius, name=f"polygon{i}")
            # (3.13b): x - z_0 in Z. With Z = {0}, z_0 = x; otherwise Z's
            # starting half-spaces a^T (x - z_0) <= h_Z(a), whose right-hand
            # sides `act` sets from the measured x, refined by cuts.
            if self.Z.powers:
                init = None
                z_base = m.addConstr(-(z[0] @ self._z_dirs.T) <= self._z_h, name="z_base")
            else:
                init = m.addConstr(z[0] == np.zeros(4), name="init")
                z_base = None
            # (3.13c) nominal dynamics.
            m.addConstr(z[1:] == z[:-1] @ self.A.T + v @ self.B.T, name="dyn")
            # (3.15b-d) for k = 0..N: the terminal stage needs them too,
            # since this X_f is not inside a single free cell the way the
            # note's is (its (3.11)).
            deltas = []
            for i, (Ob, M) in enumerate(zip(self.Obar, self.M)):
                d = m.addMVar((N + 1, Ob.H.shape[0]), vtype=GRB.BINARY, name=f"delta{i}")
                m.addConstr(z @ Ob.H.T - d * M >= Ob.h - M, name=f"face{i}")
                m.addConstr(d.sum(axis=1) >= 1, name=f"any{i}")
                deltas.append(d)
            # (3.13a): the quadratic part is fixed; the linear part carries
            # the target and is rewritten every solve (`_set_target`).
            obj = sum(z[k] @ self.Q @ z[k] + v[k] @ self.R @ v[k] for k in range(N))
            obj = obj + z[N] @ self.P @ z[N]
            m.setObjective(obj, GRB.MINIMIZE)
            m.update()
            self._models.append(dict(m=m, z=z, v=v, init=init, z_base=z_base, deltas=deltas,
                                     disks=disks, z_seed=[], x=None, disk_seed=[]))

    # ---------------------------------------------------------------- helpers
    def cost(self, z, v, x_r):
        """J_N (3.13a) of a plan, with u_r = 0."""
        dz = z - x_r
        return float(np.einsum("ki,ij,kj->", dz[:-1], self.Q, dz[:-1])
                     + np.einsum("ki,ij,kj->", v, self.R, v)
                     + dz[-1] @ self.P @ dz[-1])

    def _set_target(self, mdl, x_r):
        """The linear and constant parts of (3.13a) for target x_r."""
        c = np.tile(-2.0 * self.Q @ x_r, (self.N + 1, 1))
        c[self.N] = -2.0 * self.P @ x_r
        mdl["z"].Obj = c
        mdl["m"].ObjCon = float(self.N * x_r @ self.Q @ x_r + x_r @ self.P @ x_r)

    def _reachable_box(self, x):
        """Bounds on every nominal state at each stage k = 0..N, valid for
        every plan (3.17) admits from the measured x.

        z_0 lies in x (+) (-Z), so each of its components within h_Z of x's.
        After that each position moves by p_{k+1} = p_k + dt (v_k + v_{k+1}) / 2
        under the double integrator, with every velocity from stage 1 on
        bounded by v_bar, the tightened speed limit, and stage 0's -- which
        the speed disk leaves free -- by |x_v| + h_Z on each axis. So stage
        k >= 1 lies within dt (|v_0| + v_bar) / 2 + (k - 1) dt v_bar of z_0.
        Everything is also inside X0 (-) Z. Returns (lo, hi), each
        (N+1, 4)."""
        z_lo, z_hi = self.Xbar0.bounds
        e = np.eye(4)
        z0_lo = np.maximum(x - np.array([self.Z.support(e[j]) for j in range(4)]), z_lo)
        z0_hi = np.minimum(x + np.array([self.Z.support(-e[j]) for j in range(4)]), z_hi)
        z0_lo[2:] = x[2:] - np.array([self.Z.support(e[j]) for j in (2, 3)])
        z0_hi[2:] = x[2:] + np.array([self.Z.support(-e[j]) for j in (2, 3)])
        v_bar = np.maximum(np.abs(z_lo[2:]), np.abs(z_hi[2:]))
        v_0 = np.maximum(np.abs(z0_lo[2:]), np.abs(z0_hi[2:]))
        k = np.arange(self.N + 1)[:, None]
        reach = np.where(k >= 1, self.dt * (v_0 + v_bar) / 2.0 + (k - 1) * self.dt * v_bar, 0.0)
        lo = np.tile(z_lo, (self.N + 1, 1))
        hi = np.tile(z_hi, (self.N + 1, 1))
        lo[:, :2] = np.maximum(z0_lo[:2] - reach, z_lo[:2])
        hi[:, :2] = np.minimum(z0_hi[:2] + reach, z_hi[:2])
        lo[0, 2:], hi[0, 2:] = z0_lo[2:], z0_hi[2:]
        return lo, hi

    def _fix_unreachable(self, x):
        """Per obstacle, the (lb, ub) of its binaries for this solve (remark
        3.12).

        At stage k, face j of obstacle i holds for every plan when the whole
        reachable box lies on its outer side: min over the box of H_ij z,
        which is H_ij c - |H_ij| r for a box of centre c and half-widths r,
        is >= h~_ij. Its binary is then fixed to 1 and the obstacle's other
        faces at that stage to 0 -- inactive constraints, not wrong ones. No
        feasible plan is lost, so the problem is the same one, with fewer
        binaries for branch-and-bound. A slalom gate is 1 m long and the two
        are 2 m apart, against a reach of N * dt * v_bar ~ 1 m per horizon, so
        at most one gate's two obstacles are ever live."""
        lo, hi = self._reachable_box(x)
        c, rad = (lo + hi) / 2.0, (hi - lo) / 2.0
        fixes = []
        for Ob in self.Obar:
            min_face = c @ Ob.H.T - rad @ np.abs(Ob.H).T        # (N+1, q)
            certified = min_face >= Ob.h
            lb = np.zeros_like(min_face)
            ub = np.ones_like(min_face)
            rows = np.flatnonzero(certified.any(axis=1))
            first = certified[rows].argmax(axis=1)
            ub[rows] = 0.0
            lb[rows, first] = 1.0
            ub[rows, first] = 1.0
            fixes.append((lb, ub))
        return fixes

    def _candidate(self, slot, x_r):
        """The shifted plan (4.1) under this file's terminal set: drop the
        first step, and hold the final rest state with v = 0 -- feasible for
        P_N at the new x by theorem 4.1 whenever the last step's disturbance
        was in W. None when the slot has no plan (a fresh episode)."""
        prev = self._prev[slot]
        if prev is None:
            return None
        z, v = prev
        z_c = np.vstack([z[1:], z[-1:]])
        v_c = np.vstack([v[1:], np.zeros((1, 2))])
        return dict(z=z_c, v=v_c, J=self.cost(z_c, v_c, x_r))

    def _candidate_deltas(self, z, fixes):
        """Binaries for a MIP start from plan z: 1 on every face z satisfies,
        or the fixed value where `_fix_unreachable` fixed one. None if some
        stage satisfies no face of some obstacle, which a candidate from
        theorem 4.1 never does.

        "Satisfies" up to 1e-4: above what a solved plan can violate a face
        by (FeasibilityTol 1e-6, plus M * IntFeasTol ~ 1e-4 through a binary
        at 1 - 1e-5), and well inside the margin rho that absorbs it."""
        out = []
        for Ob, (lb, ub) in zip(self.Obar, fixes):
            d = (z @ Ob.H.T >= Ob.h - 1e-4).astype(float)
            fixed = lb == ub
            d[fixed] = lb[fixed]
            if not np.all(d.sum(axis=1) >= 1):
                return None
            out.append(d)
        return out

    def _tube_ratio(self, slot, x):
        """How far the real state strayed from the nominal plan's prediction,
        as a fraction of the tube: the larger of ||e_p|| / r_p and
        ||e_v|| / r_v, with e = x - z_1 of the slot's last plan and r_p, r_v
        the radii of Z's position and velocity disks. Theorem 4.1 keeps e in
        Z, so this stays <= 1 while the disturbance respects W; it is only a
        necessary condition for e in Z (Z's projections, not Z itself), and
        NaN without a plan or with Z = {0}. Until 2026-10-04 it was the same
        per axis, against Z's bounding box."""
        prev = self._prev[slot]
        if prev is None or not self.Z.powers:
            return float("nan")
        e = x - prev[0][1]
        r_p, r_v = self.Z.support(np.eye(4)[0]), self.Z.support(np.eye(4)[2])
        return float(max(np.linalg.norm(e[:2]) / r_p, np.linalg.norm(e[2:]) / r_v))

    def _one_face_each(self, mdl):
        """The relaxation's binaries for the exact problem, with one face on
        per obstacle and stage: where the relaxation switched on several
        (all satisfied by its plan), the one its plan clears by the most.
        Every face left on is a constraint of the exact problem, so keeping
        one leaves it the largest convex piece the relaxation vouched for."""
        z = mdl["z"].X
        out = []
        for Ob, d in zip(self.Obar, mdl["deltas"]):
            on = np.round(d.X)
            margin = np.where(on > 0, z @ Ob.H.T - Ob.h, -np.inf)
            one = np.zeros_like(on)
            one[np.arange(on.shape[0]), margin.argmax(axis=1)] = 1.0
            out.append(one)
        return out

    def _optimize(self, mdl):
        """Solve one slot's problem as set up by `act` (module docstring):
        the optimal plan (z, v), or None if there is none (infeasible, or the
        relaxation stopped by the work limit before finding a plan) or the
        refinement did not converge within `MAX_CUT_ROUNDS` rounds, which
        algorithm 3's candidate then covers. Sets mdl["cut_rounds"] to the
        number of mixed-integer solves it took, 0 when the binaries were all
        fixed and Clarabel solved alone. A method of its own so a test can
        stand in a failed solve and exercise that fallback."""
        x, x_r, fixes = mdl["x"], mdl["x_r"], mdl["fixes"]
        if all(np.array_equal(lb, ub) for lb, ub in fixes):
            exact = self._exact(x, x_r, [lb for lb, _ in fixes])
            return None if exact is None else exact[:2]

        m = mdl["m"]
        cuts = []

        def disk_cut(d, i, n):
            rows, radius = mdl["disks"][d]
            cuts.append(m.addConstr(rows[i] @ n <= radius))

        def z_cut(y):
            cuts.append(m.addConstr(-(mdl["z"][0] @ y) <= 1.0 - y @ x))
            mdl["z_seed"].append(y)

        for d, i, n in mdl["disk_seed"]:
            disk_cut(d, i, n)
        if self.Z.powers:
            # The most recent Z cuts of this episode; older ones, which the
            # state has moved away from, would only grow the model.
            mdl["z_seed"] = mdl["z_seed"][-Z_SEED_KEEP:]
            for y in list(mdl["z_seed"]):
                cuts.append(m.addConstr(-(mdl["z"][0] @ y) <= 1.0 - y @ x))
        try:
            for rnd in range(MAX_CUT_ROUNDS):
                m.optimize()
                if m.SolCount == 0:
                    return None
                deltas = self._one_face_each(mdl)
                bound = m.ObjBound
                exact = self._exact(x, x_r, deltas)
                if exact is not None and exact[2] <= bound + self.settings["mip_gap"] * abs(exact[2]) + 1e-9:
                    mdl["cut_rounds"] = rnd + 1
                    return exact[:2]
                # Tighten the relaxation where it was loose: at the exact
                # plan's points on the edge of their sets, and where the
                # relaxation's own plan left them.
                added = len(cuts)
                plans = [(mdl["z"].X, mdl["v"].X, CUT_TOL)]
                if exact is not None:
                    plans.append((exact[0], exact[1], -1e-4))
                for z, v, slack in plans:
                    for d, Y in enumerate((z[1:self.N, 2:], v)):
                        radius = mdl["disks"][d][1]
                        norms = np.linalg.norm(Y, axis=1)
                        for i in np.flatnonzero(norms > radius * (1.0 + slack) + (CUT_TOL if slack > 0 else 0.0)):
                            disk_cut(d, i, Y[i] / norms[i])
                    if self.Z.powers:
                        gamma, y = self._gauge(x - z[0])
                        if gamma > 1.0 + slack:
                            z_cut(y)
                if len(cuts) == added:
                    # Nothing left to cut: the relaxation's plan already
                    # satisfies the exact sets, so it is feasible, and optimal
                    # to the gap.
                    mdl["cut_rounds"] = rnd + 1
                    return mdl["z"].X.copy(), mdl["v"].X.copy()
            return None
        finally:
            if cuts:
                m.remove(cuts)
                m.update()

    # ----------------------------------------------------------------- online
    def reset(self, slot=None):
        """Forget the last plan of `slot`, or of every slot, and the
        directions of its Z cuts: at an episode's end, so the next step
        solves from scratch rather than from a candidate for another
        episode's state."""
        if slot is None:
            self._prev = [None] * self.num_slots
            for mdl in self._models:
                mdl["z_seed"] = []
        else:
            self._prev[slot] = None
            self._models[slot]["z_seed"] = []

    def act(self, x, target, slot=0):
        """One step of algorithm 3 for `slot`: the input to apply at the
        measured state `x` ([p_x, p_y, v_x, v_y]), toward rest at `target`
        ([p_x, p_y]), and an info dict (outcome, solve time in ms, number of
        free binaries, tube ratio, how many solves the disks' cutting planes
        took -- 0 when none produced the plan -- and the plan)."""
        x = np.asarray(x, dtype=float)
        x_r = np.array([target[0], target[1], 0.0, 0.0], dtype=float)
        mdl = self._models[slot]
        t0 = time.perf_counter()
        tube_ratio = self._tube_ratio(slot, x)

        mdl["x"], mdl["x_r"] = x, x_r
        if mdl["init"] is not None:
            mdl["init"].RHS = x
        else:
            mdl["z_base"].RHS = self._z_h - self._z_dirs @ x
        self._set_target(mdl, x_r)
        fixes = self._fix_unreachable(x)
        mdl["fixes"] = fixes
        free = 0
        for d, (lb, ub) in zip(mdl["deltas"], fixes):
            d.LB = lb
            d.UB = ub
            free += int(np.sum(lb != ub))

        cand = self._candidate(slot, x_r)
        cand_deltas = None if cand is None else self._candidate_deltas(cand["z"], fixes)
        # Tangents at the candidate's points near the edge of their disks,
        # which the new plan is likely to lean on too (module docstring).
        mdl["disk_seed"] = []
        if cand is not None:
            for d, Y in enumerate((cand["z"][1:self.N, 2:], cand["v"])):
                radius = mdl["disks"][d][1]
                norms = np.linalg.norm(Y, axis=1)
                for i in np.flatnonzero(norms > 0.5 * radius):
                    mdl["disk_seed"].append((d, i, Y[i] / norms[i]))
        # MIP start: the candidate (algorithm 3, line 7), omega left for
        # Gurobi to complete. Cleared when there is none, so a start from the
        # previous solve never leaks into this one.
        if cand is not None:
            mdl["z"].Start = cand["z"]
            mdl["v"].Start = cand["v"]
        else:
            mdl["z"].Start = np.full((self.N + 1, 4), GRB.UNDEFINED)
            mdl["v"].Start = np.full((self.N, 2), GRB.UNDEFINED)
        for d, dv in zip(mdl["deltas"], cand_deltas or [None] * len(mdl["deltas"])):
            d.Start = np.full(d.shape, GRB.UNDEFINED) if dv is None else dv

        plan = None
        mdl["cut_rounds"] = 0
        solution = self._optimize(mdl)
        if solution is not None:
            z, v = solution
            J = self.cost(z, v, x_r)
            # Algorithm 3, line 8: accept only if no worse than the candidate.
            if cand is None or J <= cand["J"] + 1e-6 * (1.0 + abs(cand["J"])):
                plan, outcome = (z, v), SOLVER
        if plan is None and cand is not None:
            plan, outcome = (cand["z"], cand["v"]), CANDIDATE

        if plan is None:
            # Outside X_N: no guarantee applies. Brake as hard as U allows,
            # straight against the velocity.
            u = _project_disk(-x[2:] / self.dt, self.u_max)
            self._prev[slot] = None
            outcome, J = EMERGENCY, float("nan")
        else:
            z, v = plan
            u = v[0] + self.K @ (x - z[0])                          # (3.14)
            # Inside U by construction (V (+) KZ in U); the projection only
            # absorbs the solver's tolerances.
            u = _project_disk(u, self.u_max)
            self._prev[slot] = (z, v)
            J = self.cost(z, v, x_r)
        return u, dict(outcome=outcome, solve_ms=1e3 * (time.perf_counter() - t0),
                       free_binaries=free, tube_ratio=tube_ratio, J=J,
                       cut_rounds=mdl["cut_rounds"], plan=None if plan is None else plan[0])

    def act_batch(self, X, targets):
        """`act` for slots 0..len(X)-1 in turn. Returns the (n, 2) inputs and
        a dict of per-slot arrays (outcome, solve_ms, free_binaries,
        tube_ratio)."""
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if n > self.num_slots:
            raise ValueError(f"{n} environments but only {self.num_slots} slots")
        U = np.empty((n, 2))
        stats = {k: np.empty(n) for k in ("outcome", "solve_ms", "free_binaries", "tube_ratio")}
        for i in range(n):
            U[i], info = self.act(X[i], targets[i], slot=i)
            for k in stats:
                stats[k][i] = info[k]
        return U, stats

    def close(self):
        """Release the Gurobi models and environment."""
        for mdl in self._models:
            mdl["m"].dispose()
        self._models = []
        self._gp_env.dispose()

    # ---------------------------------------------------------------- summary
    def stopping_steps(self):
        """Steps the nominal plan needs to stop from its top speed at its
        largest deceleration: the horizon below which the terminal set caps
        the planned speed (module docstring)."""
        v_bar = float(self.Xbar0.bounds[1][2])
        a_bar = float(self.V.bounds[1][0])
        return int(np.ceil(v_bar / (a_bar * self.dt) - 1e-9))

    def summary(self):
        e = np.eye(4)
        ext = [self.Z.support(e[j]) for j in range(4)]
        n_bin = sum(Ob.H.shape[0] for Ob in self.Obar) * (self.N + 1)
        lo, hi = self.Xbar0.bounds
        return (f"tube MPC: W disks of radii (p, v) {self.W.radii}, RPI s={self.Z.s} alpha={self.Z.alpha:.3g}; "
                f"Z radii (p, v) = {np.round([ext[0], ext[2]], 4)}\n"
                f"  X0 (-) Z: p_x [{lo[0]:.3f}, {hi[0]:.3f}], p_y [{lo[1]:.3f}, {hi[1]:.3f}], "
                f"||v|| <= {self.v_bar:.3f}; V = U (-) KZ: ||u|| <= {self.u_bar:.3f}; "
                f"K = {np.round(self.K[0, [0, 2]], 3)} per axis\n"
                f"  {len(self.Obar)} obstacles, {n_bin} binaries before fixing, "
                f"max big-M {max((M.max() for M in self.M), default=0.0):.2f}; "
                f"horizon N={self.N}, stopping time {self.stopping_steps()} steps")
