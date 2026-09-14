import warnings

import numpy as np
import scipy.sparse as sp

import cvxpy as cp
import osqp

# `envs` is scenario-specific (scenarios/<scenario>/envs), so this shared
# module does not add it to sys.path itself -- the caller (a scenario's
# training script, or this package's own tests/conftest.py) is responsible
# for putting the right scenario root on sys.path before importing this
# module. `WidthSegment`/`WidthProfile`/`constant_profile` are identical
# across scenarios, so it never matters which scenario's copy resolves here.
from envs.width_profile import constant_profile


class MPCWorker:
    """The fixed (unlearned) low-level controller: a constrained LQR-style MPC
    that drives the vehicle toward the manager's goal state.

    Two backends solve the identical problem.

    `"osqp"` (the default) assembles the QP once per horizon length and then
    only rewrites the linear cost `q` and the initial-state bounds on each
    call, which is all that actually changes between solves. `"cvxpy"` is the
    original formulation, kept as the readable reference the OSQP assembly is
    tested against (tests/test_mpc_worker.py).

    The default changed because the solver was the whole cost of a PPO_MPC
    rollout, and almost none of it was the QP. Measured on this problem at
    horizon 10: cvxpy 26.3 ms per call, of which OSQP's own reported solve time
    was 0.21 ms -- 99% of the wall clock was cvxpy's Python-side canonicalisation
    and solution retrieval, repeated every step even though the problem is DPP
    and the sparsity pattern never changes. The direct assembly runs the same
    solve in 0.32 ms, an 81x speedup, and agrees with cvxpy to within 7e-5 on
    the returned action -- below OSQP's own 1e-5 convergence tolerance. For
    comparison an environment step is under a microsecond, so this, not the
    environment, is what a PPO_MPC rollout spends its time on.

    Warm starting is per environment. Each of `num_envs` slots keeps its own
    solver instances, so a batched rollout does not have environment i's solve
    warm-started from environment i-1's solution -- which is what sharing one
    problem object across a vectorized rollout would silently do.
    """

    def __init__(self, dt=0.1, L=10.0, W=4.0, v_max=2.0, u_max=1.0, horizon=10,
                 Q=None, R=None, backend="osqp", num_envs=1, width_profile=None):
        self.dt = dt
        self.L = L
        self.W = W
        self.v_max = v_max
        self.u_max = u_max
        self.N = horizon
        self.num_envs = num_envs

        # Piecewise-constant lateral width profile (see envs/width_profile.py):
        # x_min[1]/x_max[1] below become the *global envelope* -- the widest
        # bound this profile ever allows -- while the position-specific,
        # possibly tighter bound is looked up and applied fresh on every
        # solve (`_solve_osqp`/`_solve_cvxpy`), never baked in statically.
        # See the scenario's ENV.md section 3 for why this is piecewise-constant
        # rather than a continuous taper: p_x is itself a decision variable
        # inside the horizon, so the bound applied to a whole solve can only
        # be as fresh as "whichever segment the *current, measured* p_x is
        # in" -- valid as long as no segment is shorter than the worst-case
        # travel within one horizon (v_max * horizon * dt).
        self.width_profile = width_profile if width_profile is not None else constant_profile(self.W)

        if backend not in ("osqp", "cvxpy"):
            raise ValueError(f"backend must be 'osqp' or 'cvxpy', got {backend!r}")
        self.backend = backend

        # Linear dynamics matrices (Discrete-time double integrator)
        self.A = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.B = np.array([
            [0.5 * self.dt**2, 0.0],
            [0.0, 0.5 * self.dt**2],
            [self.dt, 0.0],
            [0.0, self.dt]
        ])

        # State and action bounds. y is the profile's global envelope here;
        # a solve's actually-enforced y bound is narrower whenever the
        # current position sits in a non-full-width segment (see above).
        y_lo, y_hi = self.width_profile.envelope()
        self.x_min = np.array([-1.0, y_lo, -self.v_max, -self.v_max])
        self.x_max = np.array([self.L + 1.0, y_hi, self.v_max, self.v_max])

        self.u_min = np.array([-self.u_max, -self.u_max])
        self.u_max_arr = np.array([self.u_max, self.u_max])

        # Cost matrices. Defaults deliberately match what each scenario's
        # script_ppo_mpc.py passes (Q also penalizes velocity error,
        # since the manager's goal includes a target velocity), so a worker
        # constructed directly behaves like the trained configuration instead
        # of silently differing from it.
        if Q is None:
            self.Q = np.diag([10.0, 10.0, 1.0, 1.0])
        else:
            self.Q = Q

        if R is None:
            self.R = np.diag([0.1, 0.1])
        else:
            self.R = R

        self.nx = 4
        self.nu = 2

        if self.backend == "cvxpy":
            self._setup_cvxpy()
        else:
            self._setup_osqp()

    # --- backend setup ----------------------------------------------------

    def _setup_cvxpy(self):
        """The original formulation. One problem per horizon length, since a
        segment's last few steps re-plan over a shrinking horizon.

        The lateral (y) state bound is two scalar `cp.Parameter`s
        (`y_lo_params[n]`/`y_hi_params[n]`) rather than a baked-in constant,
        so `_solve_cvxpy` can refresh it every solve from `width_profile`
        exactly like `_solve_osqp` does -- both backends read the identical
        `width_profile.bounds_at` call, which is what keeps them numerically
        comparable (see `MPCWorker`'s class docstring).
        """
        self.probs = {}
        self.x_inits = {}
        self.x_targets = {}
        self.u_vars = {}
        self.y_lo_params = {}
        self.y_hi_params = {}

        for n in range(1, self.N + 1):
            x = cp.Variable((4, n + 1))
            u = cp.Variable((2, n))

            x_init = cp.Parameter(4)
            x_target = cp.Parameter(4)
            y_lo_param = cp.Parameter()
            y_hi_param = cp.Parameter()
            lo_vec = cp.hstack([self.x_min[0], y_lo_param, self.x_min[2], self.x_min[3]])
            hi_vec = cp.hstack([self.x_max[0], y_hi_param, self.x_max[2], self.x_max[3]])

            cost = 0
            constraints = []

            # Initial condition constraint
            constraints += [x[:, 0] == x_init]

            for k in range(n):
                # Stage cost
                cost += cp.quad_form(x[:, k] - x_target, self.Q) + cp.quad_form(u[:, k], self.R)

                # Dynamics constraint
                constraints += [x[:, k+1] == self.A @ x[:, k] + self.B @ u[:, k]]

                # State and input constraints
                constraints += [x[:, k] >= lo_vec]
                constraints += [x[:, k] <= hi_vec]
                constraints += [u[:, k] >= self.u_min]
                constraints += [u[:, k] <= self.u_max_arr]

            # Terminal cost and constraints
            cost += cp.quad_form(x[:, n] - x_target, self.Q)
            constraints += [x[:, n] >= lo_vec]
            constraints += [x[:, n] <= hi_vec]

            # Define the problem
            self.probs[n] = cp.Problem(cp.Minimize(cost), constraints)
            self.x_inits[n] = x_init
            self.x_targets[n] = x_target
            self.u_vars[n] = u
            self.y_lo_params[n] = y_lo_param
            self.y_hi_params[n] = y_hi_param

    def _build_qp(self, n):
        """Assemble the horizon-n QP in OSQP's form.

        Decision vector z = [x_0 .. x_n, u_0 .. u_{n-1}], so the state
        trajectory is an explicit variable and the dynamics are equality
        constraints (the same "sparse"/simultaneous transcription cvxpy is
        given above, not a condensed one -- keeping them identical is what
        makes the two backends comparable term by term).

        OSQP minimises 1/2 z'Pz + q'z, so the quadratic blocks carry a factor
        of two against the cost written as a sum of quadratic forms. The
        dropped constant x_target'Q x_target does not move the argmin.

        Also returns `y_idx`, the flat position within the tiled state-box
        block of every stage's y-row -- `_solve_osqp` overwrites exactly
        these entries every solve from `width_profile`, leaving the rest of
        the box (and P/A_c themselves) untouched after setup.
        """
        nx, nu = self.nx, self.nu
        nz = nx * (n + 1) + nu * n
        ux0 = nx * (n + 1)  # column offset of the u block

        P = sp.block_diag(
            [sp.kron(sp.eye(n + 1), 2 * self.Q), sp.kron(sp.eye(n), 2 * self.R)],
            format="csc")

        # Rows: initial state (nx) + dynamics (nx*n) + a box on every entry of z
        Ad = np.zeros((nx + nx * n + nz, nz))
        Ad[:nx, :nx] = np.eye(nx)
        for k in range(n):
            r = nx + k * nx
            Ad[r:r + nx, k * nx:(k + 1) * nx] = -self.A
            Ad[r:r + nx, (k + 1) * nx:(k + 2) * nx] = np.eye(nx)
            Ad[r:r + nx, ux0 + k * nu: ux0 + (k + 1) * nu] = -self.B
        Ad[nx + nx * n:, :] = np.eye(nz)
        A_c = sp.csc_matrix(Ad)

        # The initial-state rows are rewritten per solve; the rest are fixed.
        lo = np.concatenate([np.zeros(nx), np.zeros(nx * n),
                             np.tile(self.x_min, n + 1), np.tile(self.u_min, n)])
        hi = np.concatenate([np.zeros(nx), np.zeros(nx * n),
                             np.tile(self.x_max, n + 1), np.tile(self.u_max_arr, n)])

        box0 = nx + nx * n  # flat offset where the tiled state box begins
        y_idx = box0 + np.arange(n + 1) * nx + 1  # the y-row of every stage
        return P, A_c, lo, hi, nz, y_idx

    def _setup_osqp(self):
        """One solver per (horizon length, environment slot).

        Per environment because OSQP carries its warm start inside the solver
        object; sharing one across a vectorized rollout would have each
        environment resume from a neighbour's solution.
        """
        self._qp = {}
        self._solvers = {}
        for n in range(1, self.N + 1):
            P, A_c, lo, hi, nz, y_idx = self._build_qp(n)
            self._qp[n] = (lo, hi, nz, y_idx)
            for i in range(self.num_envs):
                m = osqp.OSQP()
                m.setup(P=P, q=np.zeros(nz), A=A_c, l=lo, u=hi,
                        verbose=False, warm_starting=True,
                        eps_abs=1e-5, eps_rel=1e-5, polishing=True)
                self._solvers[(n, i)] = m

    # --- solve ------------------------------------------------------------

    def _braking_action(self, current_state):
        """Safe fallback when the QP does not return a usable solution."""
        return np.clip(-np.asarray(current_state)[2:] / self.dt, -self.u_max, self.u_max)

    def _solve_cvxpy(self, current_state, target_state, steps_left):
        self.x_inits[steps_left].value = current_state
        self.x_targets[steps_left].value = target_state
        y_lo, y_hi = self.width_profile.bounds_at(current_state[0])
        self.y_lo_params[steps_left].value = y_lo
        self.y_hi_params[steps_left].value = y_hi
        try:
            prob = self.probs[steps_left]
            # Scoped to this solve only: an unscoped `warnings.filterwarnings`
            # call here would mutate the *global* filter on every step for the
            # rest of the process, silently swallowing unrelated warnings
            # (numpy/torch deprecations, etc.) everywhere else too.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob.solve(solver=cp.OSQP, warm_start=True)
            u_var = self.u_vars[steps_left]
            if prob.status not in ["infeasible", "unbounded"] and u_var[:, 0].value is not None:
                return u_var[:, 0].value
        except Exception:
            pass
        return self._braking_action(current_state)

    def _solve_osqp(self, current_state, target_state, steps_left, env_index):
        nx, nu = self.nx, self.nu
        n = steps_left
        lo, hi, nz, y_idx = self._qp[n]
        try:
            q = np.zeros(nz)
            # d/dx of (x - x_t)'Q(x - x_t) contributes -2 Q x_t to the linear
            # term, on every state block including the terminal one. The input
            # blocks have no linear term.
            q[:nx * (n + 1)] = np.tile(-2.0 * (self.Q @ target_state), n + 1)
            lo = lo.copy()
            hi = hi.copy()
            lo[:nx] = current_state
            hi[:nx] = current_state

            # The y-bound is looked up fresh from width_profile every solve
            # (see MPCWorker.__init__ and _build_qp) -- applied uniformly
            # across every stage of this horizon, not just the initial state.
            y_lo, y_hi = self.width_profile.bounds_at(current_state[0])
            lo[y_idx] = y_lo
            hi[y_idx] = y_hi

            m = self._solvers[(n, env_index)]
            m.update(q=q, l=lo, u=hi)
            # Explicit because OSQP's default is scheduled to flip to
            # raise_error=True. Either way a failed solve ends at the braking
            # fallback below -- via the status check now, via the except then --
            # but pinning it keeps which of the two paths runs from changing
            # under a library upgrade.
            res = m.solve(raise_error=False)
            status = str(res.info.status).lower()
            # "solved" and "solved inaccurate" are both usable; "primal
            # infeasible" / "maximum iterations reached" are not.
            if "solved" in status and res.x is not None and np.all(np.isfinite(res.x)):
                return np.asarray(res.x[nx * (n + 1): nx * (n + 1) + nu])
        except Exception:
            pass
        return self._braking_action(current_state)

    def get_action(self, current_state, goal_phys, steps_left, env_index=0):
        """
        current_state: [p_x, p_y, v_x, v_y]
        goal_phys: [delta_x, delta_y, delta_vx, delta_vy] - The relative target state
        steps_left: The remaining horizon length (1 to self.N)
        env_index: which environment's warm-start slot to solve in
        Returns the optimal control action u_0
        """
        current_state = np.asarray(current_state, dtype=float)
        # Define absolute target state
        target_state = current_state + np.asarray(goal_phys, dtype=float)

        # Clamp steps_left just in case
        steps_left = max(1, min(int(steps_left), self.N))

        if self.backend == "cvxpy":
            return self._solve_cvxpy(current_state, target_state, steps_left)
        return self._solve_osqp(current_state, target_state, steps_left, env_index)

    def get_actions(self, states, goals, steps_left):
        """Batched `get_action` over `num_envs` environments.

        `states` is (num_envs, 4), `goals` is (num_envs, 4), `steps_left` a
        (num_envs,) integer array -- the environments run independent episodes,
        so they sit at different points in their manager segment and re-plan
        over different horizon lengths.

        Still one QP per environment: the horizons differ, so there is no
        single batched solve to make, and the win over the serial rollout comes
        from the per-solve cost rather than from fusing them. Each environment
        keeps its own warm start.
        """
        states = np.asarray(states, dtype=float)
        goals = np.asarray(goals, dtype=float)
        steps_left = np.asarray(steps_left).astype(int).reshape(-1)
        n_env = states.shape[0]
        if n_env > self.num_envs:
            raise ValueError(
                f"got {n_env} environments but this worker was built with "
                f"num_envs={self.num_envs}; each needs its own warm-start slot")
        out = np.empty((n_env, self.nu), dtype=float)
        for i in range(n_env):
            out[i] = self.get_action(states[i], goals[i], steps_left[i], env_index=i)
        return out
