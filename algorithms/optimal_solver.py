"""Offline, scenario-agnostic minimum-time / maximum-return oracle.

Both TunnelEnv and SlalomEnv share the same reward shape: a -1 per-step
penalty, a +goal_reward on arrival (replacing that step's penalty), a
per-contact penalty, and a progress term `progress_reward_coef*(p_x'-p_x)`
that telescopes over an episode to `progress_reward_coef*(p_x_final -
p_x_initial)` -- a constant for any successful path with the same
endpoints. So for a *successful, contact-free* episode of length N:

    return = goal_reward - (N - 1) + progress_reward_coef * (L - p_x0)

Maximizing return is therefore equivalent to minimizing the arrival time N
(then avoiding wall contact), which is a well-posed minimum-time optimal
control problem: both envs are a discrete-time double integrator with
independent per-axis dynamics (the A/B matrices are block-diagonal per
axis), constrained only through the corridor, whose lateral (y) bound is a
function of the current x position -- itself a decision variable. That
position-dependent bound is what makes the problem non-convex; everything
else here is a convex QP.

This solves it via SCP (sequential convex programming): for a candidate
horizon length N, repeatedly (a) freeze the corridor's y-bound at every
stage using the *previous* iterate's x-trajectory, (b) solve the resulting
convex multi-stage QP (the same "state trajectory as an explicit decision
variable, dynamics as equality constraints" transcription `MPCWorker`
already uses in `_setup_cvxpy`, but full-horizon rather than receding), and
(c) stop once the segment assignment implied by the new solution matches
the one used to build it (a fixed point, hence a genuinely feasible
trajectory for that N -- not just for its own linearization). The smallest
N for which this converges is taken as the arrival time.

This is a heuristic, not a certified global optimum: SCP can in principle
get stuck in a locally-feasible-looking loop that never reaches a fixed
point even though a solution exists. Each candidate N is retried from a
few different initial guesses to mitigate that (see `_initial_guess`).

Once a trajectory is found, its action sequence is replayed through the
*real* env (`env.step`), so every reported number (return, success,
collision count) is exactly what a policy executing these actions would
receive -- never a value computed from this module's own model of the
dynamics/reward.

Performance note: like `MPCWorker` (see its class docstring), a QP rebuilt
from scratch on every solve pays cvxpy's Python-side canonicalization cost
every time even though the problem's sparsity pattern is fixed for a given
horizon length N. `_HorizonQP` builds the N-stage problem once, with the
per-stage y-bound and the initial state as `cp.Parameter`s, so every SCP
iteration/restart/episode that reuses the same N only pays for a parameter
update and a warm-started re-solve. `MinTimeSolver` keeps one `_HorizonQP`
per N it has ever needed, across `solve()` calls -- built for exactly the
CLI scripts' use case of solving many episodes of the same scenario config.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import cvxpy as cp

# The QP models dynamics in float64; the real envs store state in float32
# and clip every step, so a trajectory the QP resolves to exactly p_x==L
# can come back from `env.step` a few 1e-6 short (observed: ~7e-6 over a
# ~70-step Tunnel episode) -- enough to miss the env's `p_x >= L` check by
# a hair and have the replay run past `horizon_used` without terminating.
# Solving for `p_x_N >= L + _TERMINAL_MARGIN` instead absorbs that, with
# plenty of room to spare relative to a single step's travel (~0.1-0.12
# units near cruise speed).
_TERMINAL_MARGIN = 1e-3

# Both envs flag wall contact with an *inclusive* p_y <= y_lo / p_y >= y_hi
# check, so a QP solution that (validly, at the boundary) touches y_lo/y_hi
# exactly would count as a contact on replay even in exact arithmetic --
# and float32 replay noise can tip a near-boundary point over the line
# either way. Shrinking the QP's own y-box inward by this much keeps the
# solved trajectory strictly clear of both. Negligible next to any gate
# this project actually uses (slalom_profile's default gate_half_width is
# 0.75 -- a 750x margin).
_LATERAL_MARGIN = 1e-3


@dataclass
class OptimalResult:
    actions: np.ndarray       # (T, 2) -- the action replayed at each step
    states: np.ndarray        # (T+1, 4) -- ground-truth states from the env
    rewards: np.ndarray       # (T,) -- ground-truth per-step rewards
    total_return: float
    length: int
    success: bool
    collision_count: int
    horizon_used: int         # the SCP horizon N whose solution was replayed


def _x_only_min_time_steps(p_x0, L, v_max, u_max, dt):
    """Closed-form bang-cruise minimum time for the x-axis alone (ignoring
    the y-corridor entirely), rounded up to a step count.

    A valid lower bound on the true minimum time for the full problem
    (dropping a constraint can only make it easier), and the exact answer
    whenever the corridor never actually restricts y along the way -- e.g.
    every TunnelEnv episode, since `constant_profile` never binds tighter
    than the spawn box the agent already starts inside.
    """
    d = max(L - p_x0, 0.0)
    if d <= 1e-9:
        return 1
    t_to_vmax = v_max / u_max
    d_to_vmax = 0.5 * u_max * t_to_vmax ** 2
    if d <= d_to_vmax:
        t_star = np.sqrt(2.0 * d / u_max)
    else:
        t_star = t_to_vmax + (d - d_to_vmax) / v_max
    # -1e-9 guards against ceil'ing a value that is only an integer up to
    # floating-point noise.
    return max(1, int(np.ceil(t_star / dt - 1e-9)))


def _initial_guess(x0, L, N, width_profile, variant, lead):
    """A starting `p_x`/`p_y` trajectory for the first SCP iteration.

    `variant`/`lead` give a handful of qualitatively different guesses (see
    `MinTimeSolver.solve`'s restarts) so a bad first linearization at one
    candidate N doesn't have to be the only thing tried before giving up.
    """
    p_x0 = float(x0[0])
    p_x_guess = np.linspace(p_x0, max(L, p_x0), N + 1)

    if variant == "straight":
        p_y_guess = np.full(N + 1, float(x0[1]))
        return p_x_guess, p_y_guess

    # "centered": aim for the midpoint of whichever segment applies at each
    # stage's guessed p_x, optionally looked up `lead` units ahead so the
    # guess starts repositioning before it strictly has to.
    lookup_x = p_x_guess + lead
    y_lo, y_hi = width_profile.bounds_at(lookup_x)
    p_y_guess = (np.asarray(y_lo) + np.asarray(y_hi)) / 2.0
    p_y_guess[0] = float(x0[1])
    return p_x_guess, p_y_guess


def _bounds_converged(width_profile, p_x_used, p_x_new):
    """Whether the corridor bounds implied by the new solution's own
    p_x-trajectory match the (frozen) bounds it was solved against -- the
    SCP fixed point. Compares the bound *values*, not a segment index, so
    this never reaches into WidthProfile's internals."""
    y_lo_used, y_hi_used = width_profile.bounds_at(p_x_used)
    y_lo_new, y_hi_new = width_profile.bounds_at(p_x_new)
    return np.array_equal(y_lo_used, y_lo_new) and np.array_equal(y_hi_used, y_hi_new)


class _HorizonQP:
    """The N-stage QP, built once and re-solved with updated Parameters.

    `A`/`B`/`u_max`/`v_max`/`x_min[0]`/`x_max[0]`/`L` are baked in as plain
    constants: for one `MinTimeSolver` (one scenario config) they never
    change across episodes or SCP iterations. Only the initial state and
    the per-stage y-bound do, so those alone are `cp.Parameter`s.
    """

    def __init__(self, N, A, B, u_max, v_max, x_min0, x_max0, L):
        nx, nu = 4, 2
        self.N = N
        x = cp.Variable((nx, N + 1))
        u = cp.Variable((nu, N))
        x0_param = cp.Parameter(nx)
        y_lo_param = cp.Parameter(N + 1)
        y_hi_param = cp.Parameter(N + 1)

        constraints = [x[:, 0] == x0_param]
        for k in range(N):
            constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k]]
            constraints += [u[:, k] >= -u_max, u[:, k] <= u_max]
        constraints += [x[0, :] >= x_min0, x[0, :] <= x_max0]
        constraints += [x[1, :] >= y_lo_param, x[1, :] <= y_hi_param]
        constraints += [x[2, :] >= -v_max, x[2, :] <= v_max]
        constraints += [x[3, :] >= -v_max, x[3, :] <= v_max]
        constraints += [x[0, N] >= L + _TERMINAL_MARGIN]

        self.x, self.u = x, u
        self.x0_param, self.y_lo_param, self.y_hi_param = x0_param, y_lo_param, y_hi_param
        self.prob = cp.Problem(cp.Minimize(cp.sum_squares(u)), constraints)

    def solve(self, x0, y_lo, y_hi):
        y_lo = np.asarray(y_lo, dtype=float) + _LATERAL_MARGIN
        y_hi = np.asarray(y_hi, dtype=float) - _LATERAL_MARGIN
        self.x0_param.value = x0
        self.y_lo_param.value = y_lo
        self.y_hi_param.value = y_hi
        try:
            # Scoped: see MPCWorker._solve_cvxpy for why this must not be
            # the bare, process-global `warnings.filterwarnings` call.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.prob.solve(solver=cp.OSQP, warm_start=True)
        except Exception:
            return None
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        if self.x.value is None or self.u.value is None:
            return None
        return self.x.value, self.u.value


def _scp_solve(qp, x0, width_profile, p_x_guess, max_iters):
    for _ in range(max_iters):
        y_lo, y_hi = width_profile.bounds_at(p_x_guess)
        solved = qp.solve(x0, y_lo, y_hi)
        if solved is None:
            return None
        x_sol, u_sol = solved
        p_x_new = x_sol[0]
        if _bounds_converged(width_profile, p_x_guess, p_x_new):
            return x_sol, u_sol
        p_x_guess = p_x_new
    return None


# A handful of qualitatively different first guesses, tried in order for
# each candidate horizon before moving on to the next N.
_RESTART_VARIANTS = [
    dict(variant="centered", lead=0.0),
    dict(variant="straight", lead=0.0),
    dict(variant="centered", lead=1.0),
]


class MinTimeSolver:
    """Solves `solve_min_time`'s problem for many episodes of one scenario
    config, caching the per-horizon-length QP (`_HorizonQP`) across calls --
    built for the CLI scripts' Monte-Carlo-over-initial-conditions use case,
    where the same handful of horizon lengths recur across episodes.
    """

    def __init__(self, max_scp_iters=20, restarts=3):
        self.max_scp_iters = max_scp_iters
        self.restarts = restarts
        self._qp_cache = {}  # (A/B/u_max/v_max/x_min0/x_max0/L) -> {N: _HorizonQP}

    def _get_qp(self, key, N, A, B, u_max, v_max, x_min0, x_max0, L):
        cache = self._qp_cache.setdefault(key, {})
        qp = cache.get(N)
        if qp is None:
            qp = _HorizonQP(N, A, B, u_max, v_max, x_min0, x_max0, L)
            cache[N] = qp
        return qp

    def solve(self, env):
        """Compute and replay the maximum-return trajectory for `env`'s
        *current* state (i.e. call `env.reset(...)` first) and its exact
        dynamics/corridor/reward.

        Returns an `OptimalResult` built entirely from replaying the solved
        action sequence through `env.step`, not from this module's own
        model of the reward.
        """
        if env.state is None:
            raise ValueError("env must be reset() before calling solve()")

        x0 = np.asarray(env.state, dtype=float).copy()
        A = np.asarray(env.A, dtype=float)
        B = np.asarray(env.B, dtype=float)
        L = float(env.L)
        v_max = float(env.v_max)
        u_max = float(env.u_max)
        x_min0 = float(env.x_min[0])
        x_max0 = float(env.x_max[0])
        width_profile = env.width_profile
        max_steps = int(env.max_steps)
        dt = float(env.dt)

        # QPs are cached per (dt, L, v_max, u_max, x bounds) -- everything
        # a `_HorizonQP` bakes in as a constant. Not per width_profile: the
        # profile only ever enters through the per-solve y_lo/y_hi
        # Parameters, so the same cached QP is valid across different
        # corridors sharing the same scenario physics.
        cache_key = (dt, L, v_max, u_max, x_min0, x_max0)

        n_min = _x_only_min_time_steps(x0[0], L, v_max, u_max, dt)

        x_sol = u_sol = horizon_used = None
        for N in range(n_min, max_steps + 1):
            qp = self._get_qp(cache_key, N, A, B, u_max, v_max, x_min0, x_max0, L)
            for kwargs in _RESTART_VARIANTS[: self.restarts]:
                p_x_guess, _ = _initial_guess(x0, L, N, width_profile, **kwargs)
                solved = _scp_solve(qp, x0, width_profile, p_x_guess, self.max_scp_iters)
                if solved is not None:
                    x_sol, u_sol = solved
                    horizon_used = N
                    break
            if horizon_used is not None:
                break

        if horizon_used is None:
            raise RuntimeError(
                f"optimal_solver: no feasible minimum-time trajectory found up "
                f"to max_steps={max_steps} for initial state {x0.tolist()} "
                f"under this width_profile. Either the corridor is genuinely "
                f"infeasible in time from this start (e.g. a gate offset the "
                f"spawn box can't reach in time), or SCP failed to converge -- "
                f"try increasing `restarts`."
            )

        actions = np.asarray(u_sol).T  # (N, 2)

        states = [x0.copy()]
        rewards = []
        info = {}
        for k in range(horizon_used):
            obs, reward, terminated, truncated, info = env.step(actions[k])
            states.append(np.asarray(obs, dtype=float))
            rewards.append(float(reward))
            if terminated or truncated:
                break
        else:
            raise RuntimeError(
                f"optimal_solver: replaying the solved {horizon_used}-step "
                f"action sequence through the real env never terminated -- "
                f"the solver's model of the dynamics/bounds has drifted from "
                f"the env's."
            )

        rewards = np.asarray(rewards)
        return OptimalResult(
            actions=actions[: len(rewards)],
            states=np.asarray(states),
            rewards=rewards,
            total_return=float(rewards.sum()),
            length=len(rewards),
            success=bool(info.get("is_success", False)),
            collision_count=int(info.get("collision_count", 0)),
            horizon_used=horizon_used,
        )


def solve_min_time(env, max_scp_iters=20, restarts=3):
    """One-shot convenience wrapper around `MinTimeSolver` for a single
    episode. Solving many episodes of the same scenario should instead
    construct one `MinTimeSolver` and call `.solve(env)` repeatedly, so the
    per-horizon QPs are cached across episodes instead of rebuilt each time.
    """
    return MinTimeSolver(max_scp_iters=max_scp_iters, restarts=restarts).solve(env)


def spawn_grid(env, n_x=5, n_y=5):
    """An evenly-spaced grid of initial conditions covering the spawn box
    `env.reset()` otherwise draws *randomly* from: `p_x0` in [0, 2], `p_y0`
    in [-W/4, W/4], velocity always 0. These bounds are duplicated from
    `reset()`'s own hardcoded spawn box rather than read off it (it isn't
    exposed via config) -- the same duplication `slalom_profile()`'s
    `spawn_end = 2.0` comment already documents and accepts.

    A *fixed* grid (rather than more random draws) is the point: solving the
    oracle for each point once (see `precompute_optimal_grid`) and reusing
    those same points on every later evaluation pass is what makes an
    "is this policy within tolerance of optimal everywhere" check cheap to
    run repeatedly during training.

    Returns an `(n_x*n_y, 4)` float32 array of `[p_x0, p_y0, 0, 0]`, `p_x0`
    varying slowest (row-major over the `(n_x, n_y)` grid).
    """
    p_x_vals = np.linspace(0.0, 2.0, n_x)
    p_y_vals = np.linspace(-env.W / 4.0, env.W / 4.0, n_y)
    grid = np.array(
        [[p_x, p_y, 0.0, 0.0] for p_x in p_x_vals for p_y in p_y_vals],
        dtype=np.float32,
    )
    return grid


def precompute_optimal_grid(env, grid, solver=None):
    """Solve the oracle once for every point in `grid` (see `spawn_grid`),
    reusing one `MinTimeSolver` (and so its per-horizon-length QP cache)
    across all of them unless `solver` is given.

    `env` is driven through `env.reset(options={"init_state": point})` for
    each point -- see the `init_state` support added to
    `TunnelEnv.reset`/`SlalomEnv.reset` -- and left at whichever point was
    solved last.

    Any `MinTimeSolver.solve` infeasibility is *not* caught here: it should
    surface immediately (e.g. at training-script startup, before any
    training happens) rather than be discovered mid-training the first time
    that grid point comes up.

    Returns a list of `(init_state, OptimalResult)` pairs, in `grid`'s order.
    """
    solver = solver if solver is not None else MinTimeSolver()
    results = []
    for point in grid:
        env.reset(options={"init_state": point})
        result = solver.solve(env)
        results.append((np.asarray(point, dtype=np.float32).copy(), result))
    return results
