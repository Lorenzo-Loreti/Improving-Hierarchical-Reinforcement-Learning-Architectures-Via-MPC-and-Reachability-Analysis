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
  (3.15). Each solve is therefore a mixed-integer QP.
- The environment can now be disturbed, by a bounded uniform w in the box W
  (see noise_bound_p in the scenarios' envs/config.py). This worker is robust
  to it: the real state stays in a tube x in z (+) Z around a nominal plan z,
  and z is planned against constraints tightened by Z, so no disturbance in
  W can push the real state into a wall (theorem 4.1). With W = {0}, the
  default environment, Z = {0} and this is a nominal MPC with z_0 = x.

MPCWorker stays in the repo because ppo_mpc_reach still uses it.

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
Online, every step and every environment: the MIQP (3.17) with that
terminal set, solved by Gurobi with the shifted candidate (4.1) as MIP start
and as fallback (algorithm 3).

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
  state box (x_min, x_max), which is what env.step clips to. Keeping the
  real state strictly inside it matters beyond the walls: a clip on the
  velocity (the speed limit, see SlalomEnv.step) or on the input would make
  the plant nonlinear and the error dynamics (3.5) would no longer hold.
  That is why the planned speed is tightened too, to v_max - h_Z(e_v): the
  price of robustness is paid in top speed as well as in clearance.
"""

import time

import numpy as np
import scipy.linalg as sla
from scipy.optimize import linprog

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

    W = {0} (the deterministic environment) gives Z = {0}: s = 0, no powers,
    a support function that is identically 0 and no lifted variables, so the
    online problem's first constraint becomes z_0 = x. Any other W needs 0 in
    its interior (assumption 2), which for a box means every half-width > 0.
    """

    def __init__(self, AK, W, eps=1e-2, s_max=300):
        if np.max(np.abs(np.linalg.eigvals(AK))) >= 1:
            raise ValueError("A_K must be Schur")
        self.AK, self.W, self.eps = AK, W, eps
        if np.all(W.h == 0):
            self.s, self.alpha, self.powers = 0, 0.0, []
            return
        if np.any(W.h <= 0):
            raise ValueError(
                "0 must lie in the interior of W (assumption 2): with a box W, every "
                "half-width must be > 0, or all of them 0 for the deterministic "
                "environment. A disturbance on the velocities alone makes W flat, and "
                "algorithm 1's test A_K^s W in alpha W can then never pass; bound the "
                "positions by a small positive number instead")
        n = AK.shape[0]
        powers = [np.eye(n)]
        Mp, Mm = np.zeros(n), np.zeros(n)
        for s in range(1, s_max + 1):
            Ai = powers[-1]                                     # A_K^{s-1}
            Mp += [W.support(Ai[j]) for j in range(n)]          # (2.6)
            Mm += [W.support(-Ai[j]) for j in range(n)]
            As = AK @ Ai                                        # A_K^s
            alpha = max(W.support(As.T @ g) / hg for g, hg in zip(W.H, W.h))  # (2.5)
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
                  the half-widths of the box W the tube is designed for: the
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

        # 2) The error's RPI set.
        self.X0 = Polytope.box(x_min, x_max)
        self.U = Polytope.box([-u_max, -u_max], [u_max, u_max])
        w = np.array([noise_bound_p, noise_bound_p, noise_bound_v, noise_bound_v], dtype=float)
        self.W = Polytope.box(-w, w)
        self.Z = RPIOuterApprox(self.AK, self.W, eps=rpi_eps)

        # 3) Tightened constraints (3.6), (3.9), and the margin on the walls
        # (see the module docstring): only the p_y rows, the walls; p_x's
        # bounds are the environment's clip box, not a contact, and the
        # velocity bounds are the speed limit.
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
        self._prev = [None] * self.num_slots   # each slot's last plan, (z, v)

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
        """The MIQP (3.17), once per slot. Only what changes between solves is
        touched afterwards: the right-hand side of the initial constraint
        (the measured x), the linear objective (the target), the binaries'
        bounds (`_fix_unreachable`) and the MIP start."""
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
        v_lb = np.tile(self.V.bounds[0], (N, 1))
        v_ub = np.tile(self.V.bounds[1], (N, 1))
        L = self.Z.lifted_matrices()

        self._models = []
        for _ in range(self.num_slots):
            m = gp.Model(env=self._gp_env)
            m.Params.Threads = 1
            m.Params.MIPGap = self.settings["mip_gap"]
            if self.settings["work_limit"] is not None:
                m.Params.WorkLimit = self.settings["work_limit"]

            z = m.addMVar((N + 1, 4), lb=z_lb, ub=z_ub, name="z")
            v = m.addMVar((N, 2), lb=v_lb, ub=v_ub, name="v")
            # (3.13b) through (2.8): x - z_0 = sum_l L_l omega_l, omega_l in W.
            if L:
                w_lb, w_ub = self.W.bounds
                om = m.addMVar((len(L), 4), lb=np.tile(w_lb, (len(L), 1)),
                               ub=np.tile(w_ub, (len(L), 1)), name="omega")
                lhs = z[0] + sum(L[l] @ om[l] for l in range(len(L)))
            else:
                om = None
                lhs = z[0] + 0.0
            init = m.addConstr(lhs == np.zeros(4), name="init")
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
            self._models.append(dict(m=m, z=z, v=v, om=om, init=init, deltas=deltas))

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
        After that each position moves by at most dt * v_bar per step, v_bar
        the tightened speed limit: p_{k+1} = p_k + dt (v_k + v_{k+1}) / 2
        under the double integrator, and both velocities are bounded by
        v_bar. Everything is also inside X0 (-) Z. Returns (lo, hi), each
        (N+1, 4)."""
        z_lo, z_hi = self.Xbar0.bounds
        e = np.eye(4)
        z0_lo = np.maximum(x - np.array([self.Z.support(e[j]) for j in range(4)]), z_lo)
        z0_hi = np.minimum(x + np.array([self.Z.support(-e[j]) for j in range(4)]), z_hi)
        v_bar = np.maximum(np.abs(z_lo[2:]), np.abs(z_hi[2:]))
        k = np.arange(self.N + 1)[:, None]
        lo = np.tile(z_lo, (self.N + 1, 1))
        hi = np.tile(z_hi, (self.N + 1, 1))
        lo[:, :2] = np.maximum(z0_lo[:2] - k * self.dt * v_bar, z_lo[:2])
        hi[:, :2] = np.minimum(z0_hi[:2] + k * self.dt * v_bar, z_hi[:2])
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
        as a fraction of the tube: max over the axes of |e_r| / h_Z(+-e_r),
        with e = x - z_1 of the slot's last plan. Theorem 4.1 keeps e in Z, so
        this stays <= 1 while the disturbance respects W; it is only a
        necessary condition for e in Z (Z's bounding box, not Z itself), and
        NaN without a plan or with Z = {0}."""
        prev = self._prev[slot]
        if prev is None or not self.Z.powers:
            return float("nan")
        e = x - prev[0][1]
        eye = np.eye(4)
        widths = np.array([self.Z.support(np.sign(e[j]) * eye[j]) if e[j] != 0 else 1.0
                           for j in range(4)])
        return float(np.max(np.abs(e) / widths))

    def _optimize(self, mdl):
        """Solve one slot's MIQP as set up by `act`: the best plan Gurobi
        found, (z, v), or None if it found none (infeasible, or stopped by
        the work limit first). A method of its own so a test can stand in a
        failed solve and exercise algorithm 3's fallback."""
        mdl["m"].optimize()
        if mdl["m"].SolCount == 0:
            return None
        return mdl["z"].X.copy(), mdl["v"].X.copy()

    # ----------------------------------------------------------------- online
    def reset(self, slot=None):
        """Forget the last plan of `slot`, or of every slot: at an episode's
        end, so the next step solves from scratch rather than from a
        candidate for another episode's state."""
        if slot is None:
            self._prev = [None] * self.num_slots
        else:
            self._prev[slot] = None

    def act(self, x, target, slot=0):
        """One step of algorithm 3 for `slot`: the input to apply at the
        measured state `x` ([p_x, p_y, v_x, v_y]), toward rest at `target`
        ([p_x, p_y]), and an info dict (outcome, solve time in ms, number of
        free binaries, tube ratio, the plan)."""
        x = np.asarray(x, dtype=float)
        x_r = np.array([target[0], target[1], 0.0, 0.0], dtype=float)
        mdl = self._models[slot]
        t0 = time.perf_counter()
        tube_ratio = self._tube_ratio(slot, x)

        mdl["init"].RHS = x
        self._set_target(mdl, x_r)
        fixes = self._fix_unreachable(x)
        free = 0
        for d, (lb, ub) in zip(mdl["deltas"], fixes):
            d.LB = lb
            d.UB = ub
            free += int(np.sum(lb != ub))

        cand = self._candidate(slot, x_r)
        cand_deltas = None if cand is None else self._candidate_deltas(cand["z"], fixes)
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
        if mdl["om"] is not None:
            mdl["om"].Start = np.full(mdl["om"].shape, GRB.UNDEFINED)

        plan = None
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
            # Outside X_N: no guarantee applies. Brake as hard as U allows.
            u = np.clip(-x[2:] / self.dt, self.U.bounds[0], self.U.bounds[1])
            self._prev[slot] = None
            outcome, J = EMERGENCY, float("nan")
        else:
            z, v = plan
            u = v[0] + self.K @ (x - z[0])                          # (3.14)
            # Inside U by construction (V (+) KZ in U); the clip only absorbs
            # the solver's tolerances.
            u = np.clip(u, self.U.bounds[0], self.U.bounds[1])
            self._prev[slot] = (z, v)
            J = self.cost(z, v, x_r)
        return u, dict(outcome=outcome, solve_ms=1e3 * (time.perf_counter() - t0),
                       free_binaries=free, tube_ratio=tube_ratio, J=J,
                       plan=None if plan is None else plan[0])

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
        return (f"tube MPC: W half-widths {self.W.bounds[1]}, RPI s={self.Z.s} alpha={self.Z.alpha:.3g}; "
                f"Z half-widths (p_x, p_y, v_x, v_y) = {np.round(ext, 4)}\n"
                f"  X0 (-) Z: p_x [{lo[0]:.3f}, {hi[0]:.3f}], p_y [{lo[1]:.3f}, {hi[1]:.3f}], "
                f"|v| <= {hi[2]:.3f}; V = U (-) KZ: |u| <= {self.V.bounds[1][0]:.3f}; "
                f"K = {np.round(self.K[0, [0, 2]], 3)} per axis\n"
                f"  {len(self.Obar)} obstacles, {n_bin} binaries before fixing, "
                f"max big-M {max((M.max() for M in self.M), default=0.0):.2f}; "
                f"horizon N={self.N}, stopping time {self.stopping_steps()} steps")
