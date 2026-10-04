"""Offline, scenario-agnostic maximum-return oracle (minimum time, then
minimum effort).

Both TunnelEnv and SlalomEnv share the same reward shape: a -1 per-step
penalty, a +goal_reward on arrival (replacing that step's penalty), a
per-contact penalty, a progress term `progress_reward_coef * (min(p_x', L) -
p_x)` that telescopes over a successful episode to `progress_reward_coef *
(L - p_x0)`, and an effort term `effort_penalty * (||u_k|| / u_max)^2` on
every step's delivered acceleration. So for a *successful, contact-free*
episode that crosses the goal line on step N:

    return = goal_reward + step_penalty * (N - 1) + progress_reward_coef * (L - p_x0)
             + (effort_penalty / u_max^2) * Sum_{k<N} ||u_k||^2

(A contact costs 50 steps' worth and failing the episode costs the goal
reward, so the best episode is always successful and contact-free.) For a
fixed N the return is maximised by the trajectory that crosses on step N with
the least Sum ||u_k||^2; over N, a later arrival can only win if it saves
more effort than the steps it adds. That is a convex problem per N and a
one-dimensional search over N:

- Per N, minimise Sum ||u_k||^2 over the discrete-time double integrator with
  ||u_k|| <= u_max and ||v_k|| <= v_max (second-order cones, the environment's
  own limits: envs/actuation.py), the corridor's lateral bound at every
  stage, p_x_k < L before step N and p_x_N >= L. The lateral bound is a
  function of the current x position -- itself a decision variable -- and is
  what makes the problem non-convex; everything else is convex.
- Over N, from a lower bound upward (`_x_only_min_time_steps`): stop at the
  first N whose return, even with zero effort, could not beat the best one
  found so far. With the canonical effort_penalty = -0.01 a whole episode's
  effort is worth less than one step, so the first feasible N is always the
  answer: the oracle is the minimum-time trajectory that spends the least
  effort among the minimum-time ones.

The oracle and the reward therefore maximise the same thing, by
construction, since 2026-10-04. Before, the oracle minimised arrival time and
used Sum ||u||^2 only to pick one of the many min-time trajectories, while the
reward knew nothing about effort and paid the crossing step's overshoot past
L; it was then the min-time ceiling, but not the maximum return, and a policy
could exceed its return by up to 1.2 on the same arrival step (see the
envs' config.py, progress_reward_coef). Also until 2026-10-04 the limits were
per axis, |u_i| <= u_max and |v_i| <= v_max, and each stage a QP solved by
OSQP; the cones need Clarabel (an interior-point conic solver; OSQP solves QPs
only).

The corridor is handled by SCP (sequential convex programming): for a
candidate horizon length N, repeatedly (a) freeze the corridor's y-bound at
every stage using the *previous* iterate's x-trajectory, (b) solve the
resulting convex multi-stage problem (the same "state trajectory as an
explicit decision variable, dynamics as equality constraints" transcription
`MPCWorker` uses in `_setup_cvxpy`, but full-horizon rather than receding),
and (c) stop once the segment assignment implied by the new solution matches
the one used to build it (a fixed point, hence a genuinely feasible
trajectory for that N -- not just for its own linearization).

This is a heuristic, not a certified global optimum: SCP can in principle
get stuck in a locally-feasible-looking loop that never reaches a fixed
point even though a solution exists. Each candidate N is retried from a
few different initial guesses to mitigate that (see `_initial_guess`).

Once a trajectory is found, its action sequence is replayed through the
*real* env (`env.step`), so every reported number (return, success,
collision count) is exactly what a policy executing these actions would
receive -- never a value computed from this module's own model of the
dynamics/reward.

Performance note: like `MPCWorker` (see its class docstring), a problem
rebuilt from scratch on every solve pays cvxpy's Python-side
canonicalization cost every time even though its sparsity pattern is fixed
for a given horizon length N. `_HorizonSOCP` builds the N-stage problem once,
with the per-stage y-bound and the initial state as `cp.Parameter`s, so
every SCP iteration/restart/episode that reuses the same N only pays for a
parameter update and a re-solve. `MinTimeSolver` keeps one `_HorizonSOCP`
per N it has ever needed, across `solve()` calls -- built for exactly the
CLI scripts' use case of solving many episodes of the same scenario config.
The 25-start grid takes ~4 s on the slalom.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import cvxpy as cp

# The solver models dynamics in float64; the real envs store state in
# float32 and clip every step, so a trajectory solved to exactly p_x==L can
# come back from `env.step` a few 1e-6 short (observed: ~7e-6 over a
# ~70-step Tunnel episode) -- enough to miss the env's `p_x >= L` check by a
# hair and have the replay run past `horizon_used` without terminating.
# Solving for `p_x_N >= L + _TERMINAL_MARGIN` instead absorbs that, with
# plenty of room to spare relative to a single step's travel (~0.1-0.12
# units near cruise speed). The same margin on the other side, `p_x_k <= L -
# _TERMINAL_MARGIN` for k < N, keeps the replay from crossing a step early.
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
    predicted_return: float   # the return formula's value for the solved plan,
                              # before the replay (the module docstring's)


def _x_only_min_time_steps(p_x0, L, v_max, u_max, dt):
    """Closed-form bang-cruise minimum time for the x-axis alone (ignoring
    the y-corridor entirely), rounded up to a step count.

    A valid lower bound on the true minimum time for the full problem:
    dropping the corridor, the y-axis and the sampling (a continuous-time
    input can do anything a piecewise-constant one can) only makes it
    easier, and ||u|| <= u_max, ||v|| <= v_max imply |u_x| <= u_max,
    |v_x| <= v_max. The exact answer whenever the corridor never makes the
    vehicle steer -- e.g. every TunnelEnv episode, since `constant_profile`
    never binds tighter than the spawn box the agent already starts inside.
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


class _HorizonSOCP:
    """The N-stage problem, built once and re-solved with updated Parameters:
    min Sum ||u_k||^2 subject to the dynamics, ||u_k|| <= u_max,
    ||v_k|| <= v_max, the p_x clip box, the frozen per-stage lateral bound,
    p_x_k <= L - margin for k < N and p_x_N >= L + margin.

    `A`/`B`/`u_max`/`v_max`/`x_min[0]`/`x_max[0]`/`L` are baked in as plain
    constants: for one `MinTimeSolver` (one scenario config) they never
    change across episodes or SCP iterations. Only the initial state and
    the per-stage y-bound do, so those alone are `cp.Parameter`s.

    The objective is the reward's own effort term up to its (negative)
    scale, effort_penalty / u_max^2, which does not move the minimiser. The
    initial velocity is a parameter, not a constraint, so an initial state
    outside the speed disk makes the problem infeasible rather than quietly
    pulling it inside.
    """

    def __init__(self, N, A, B, u_max, v_max, x_min0, x_max0, L):
        nx, nu = 4, 2
        self.N = N
        x = cp.Variable((nx, N + 1))
        u = cp.Variable((nu, N))
        x0_param = cp.Parameter(nx)
        y_lo_param = cp.Parameter(N + 1)
        y_hi_param = cp.Parameter(N + 1)

        constraints = [x[:, 0] == x0_param,
                       x[:, 1:] == A @ x[:, :-1] + B @ u,
                       cp.norm(u, 2, axis=0) <= u_max,
                       cp.norm(x[2:4, :], 2, axis=0) <= v_max,
                       x[0, :] >= x_min0, x[0, :] <= x_max0,
                       x[1, :] >= y_lo_param, x[1, :] <= y_hi_param,
                       x[0, N] >= L + _TERMINAL_MARGIN]
        if N > 1:
            constraints.append(x[0, 1:N] <= L - _TERMINAL_MARGIN)

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
                self.prob.solve(solver=cp.CLARABEL)
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
    config, caching the per-horizon-length problem (`_HorizonSOCP`) across
    calls -- built for the CLI scripts' Monte-Carlo-over-initial-conditions
    use case, where the same handful of horizon lengths recur across
    episodes.

    The name is from when the oracle minimised arrival time alone. It now
    maximises the return (the module docstring), which with the canonical
    effort_penalty is still the minimum-time trajectory.
    """

    def __init__(self, max_scp_iters=20, restarts=3):
        self.max_scp_iters = max_scp_iters
        self.restarts = restarts
        self._qp_cache = {}  # (A/B/u_max/v_max/x_min0/x_max0/L) -> {N: _HorizonSOCP}

    def _get_qp(self, key, N, A, B, u_max, v_max, x_min0, x_max0, L):
        cache = self._qp_cache.setdefault(key, {})
        qp = cache.get(N)
        if qp is None:
            qp = _HorizonSOCP(N, A, B, u_max, v_max, x_min0, x_max0, L)
            cache[N] = qp
        return qp

    def solve(self, env):
        """Compute and replay the maximum-return trajectory for `env`'s
        *current* state (i.e. call `env.reset(...)` first) and its exact
        dynamics/corridor/reward.

        Returns an `OptimalResult` built entirely from replaying the solved
        action sequence through `env.step`, not from this module's own
        model of the reward; the model's value is kept alongside, as
        `predicted_return`, and the two agree to float32 rounding.
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

        # Problems are cached per (dt, L, v_max, u_max, x bounds) --
        # everything a `_HorizonSOCP` bakes in as a constant. Not per
        # width_profile: the profile only ever enters through the per-solve
        # y_lo/y_hi Parameters, so the same cached problem is valid across
        # different corridors sharing the same scenario physics. Not per
        # reward either: the reward only enters `return_of` below.
        cache_key = (dt, L, v_max, u_max, x_min0, x_max0)

        # The module docstring's return formula, for a plan crossing on step
        # N with Sum ||u||^2 = effort.
        cfg = env.config
        base = cfg.goal_reward + cfg.progress_reward_coef * (L - x0[0])
        effort_scale = cfg.effort_penalty / u_max ** 2

        def return_of(N, effort=0.0):
            return base + cfg.step_penalty * (N - 1) + effort_scale * effort

        n_min = _x_only_min_time_steps(x0[0], L, v_max, u_max, dt)

        x_sol = u_sol = horizon_used = None
        best = -np.inf
        for N in range(n_min, max_steps + 1):
            # Even with no effort at all, no N from here on can beat the
            # best plan found: stop. (The return falls by |step_penalty| per
            # step; effort only lowers it further.)
            if return_of(N) <= best:
                break
            qp = self._get_qp(cache_key, N, A, B, u_max, v_max, x_min0, x_max0, L)
            for kwargs in _RESTART_VARIANTS[: self.restarts]:
                p_x_guess, _ = _initial_guess(x0, L, N, width_profile, **kwargs)
                solved = _scp_solve(qp, x0, width_profile, p_x_guess, self.max_scp_iters)
                if solved is not None:
                    value = return_of(N, float(np.sum(solved[1] ** 2)))
                    if value > best:
                        best = value
                        x_sol, u_sol = solved
                        horizon_used = N
                    break

        if horizon_used is None:
            raise RuntimeError(
                f"optimal_solver: no feasible trajectory found up "
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
            predicted_return=float(best),
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
    in [-W/4, W/4], velocity always 0. The bounds are read off the env
    (`env.spawn_low`/`env.spawn_high`, from the scenario's
    envs/spawn_sampler.py). Until 2026-10-03 they were duplicated here from
    `reset()`'s own hardcoded box, which was not exposed; `slalom_profile()`'s
    `spawn_end = 2.0` still duplicates the upper p_x0 bound.

    A *fixed* grid (rather than more random draws) is the point: solving the
    oracle for each point once (see `precompute_optimal_grid`) and reusing
    those same points on every later evaluation pass is what makes an
    "is this policy within tolerance of optimal everywhere" check cheap to
    run repeatedly during training.

    Returns an `(n_x*n_y, 4)` float32 array of `[p_x0, p_y0, 0, 0]`, `p_x0`
    varying slowest (row-major over the `(n_x, n_y)` grid).
    """
    p_x_vals = np.linspace(env.spawn_low[0], env.spawn_high[0], n_x)
    p_y_vals = np.linspace(env.spawn_low[1], env.spawn_high[1], n_y)
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
