"""Deterministic "is this policy essentially optimal everywhere" check.

Pairs with `algorithms/optimal_solver.py`'s `spawn_grid`/
`precompute_optimal_grid`: given the oracle's already-solved return for a
fixed grid of initial conditions, rolls the *current* policy out from every
one of those same points and checks whether it comes within `tolerance`
return of optimal at every single one -- the criterion a training script
uses to decide the task is solved and stop early, rather than running a
fixed timestep budget regardless of how close to optimal the policy already
is.

Deliberately agent-framework-agnostic: takes a `make_policy_fn() ->
policy_fn(obs) -> action` factory rather than an agent object, so the same
function works whether the caller is PPO, hPPO, PPO+MPC, or anything else -- the
caller is responsible for making the returned callable deterministic (e.g.
this project's agents already expose a `deterministic=True` action mode for
evaluation; see script_ppo.py's existing eval loop), since a stochastic
policy_fn would make "solved" flicker between eval passes for no reason
related to the policy actually improving.

Why a *factory* rather than a plain `policy_fn`: flat PPO is memoryless at
eval time (one obs in, one action out), but hPPO and the PPO+MPC variants are
not -- their
eval-time policy is stateful (a manager goal set once per episode, or every
`manager_freq` steps, decaying/carrying in between). `check_solved` reuses
whatever `make_policy_fn` returns across many sequential episodes (one per
grid point) with no episode-boundary signal in between, so a *shared*
stateful closure would carry the wrong replan-timer/goal state into every
episode after the first (episodes end at arbitrary step counts, not
multiples of `manager_freq`). Calling `make_policy_fn()` fresh right after
each grid point's `env.reset()` gives every episode a clean closure, exactly
mirroring how each training script's own eval loop already re-initializes
this state right after its own `env.reset()`. A memoryless caller just does
`check_solved(lambda: policy_fn, ...)`.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SolvedCheckResult:
    solved: bool
    gaps: np.ndarray            # (num_points,) optimal_return - agent_return
    agent_returns: np.ndarray
    optimal_returns: np.ndarray
    worst_gap: float
    worst_point: np.ndarray     # the init_state with the largest gap


def check_solved(make_policy_fn, env, optimal_grid, tolerance):
    """`optimal_grid` is `precompute_optimal_grid`'s output: a list of
    `(init_state, OptimalResult)` pairs. Resets `env` to each `init_state`
    in turn (via the same `options={"init_state": ...}` support the oracle
    itself uses), builds a fresh `policy_fn` for that one episode via
    `make_policy_fn()`, and rolls it out to termination.
    """
    agent_returns = []
    optimal_returns = []
    for init_state, optimal_result in optimal_grid:
        obs, _ = env.reset(options={"init_state": init_state})
        policy_fn = make_policy_fn()
        total_reward = 0.0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        agent_returns.append(total_reward)
        optimal_returns.append(optimal_result.total_return)

    agent_returns = np.asarray(agent_returns, dtype=float)
    optimal_returns = np.asarray(optimal_returns, dtype=float)
    gaps = optimal_returns - agent_returns
    worst_idx = int(np.argmax(gaps))

    return SolvedCheckResult(
        solved=bool(np.all(gaps <= tolerance)),
        gaps=gaps,
        agent_returns=agent_returns,
        optimal_returns=optimal_returns,
        worst_gap=float(gaps[worst_idx]),
        worst_point=optimal_grid[worst_idx][0],
    )
