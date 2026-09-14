"""Regression tests for the MPC worker.

Pins down the one non-obvious bug this module had: `get_action` used to call
the bare `warnings.filterwarnings("ignore")`, which mutates the *global*
warnings filter on every single call, silently suppressing unrelated warnings
(numpy/torch deprecations, etc.) for the rest of the process. It must be
scoped to the solve alone.

Since the default backend became the direct OSQP assembly, the other job here
is to hold that assembly to the cvxpy formulation it replaced. The two solve
the same QP by construction, but "by construction" is exactly the claim a
transcription bug invalidates -- a sign error in the linear cost term, or a
misplaced dynamics block, still returns a plausible bounded action.
"""

import warnings

import numpy as np
import pytest

from mpc_worker import MPCWorker
from envs.width_profile import WidthSegment, WidthProfile

BACKENDS = ["osqp", "cvxpy"]


def _gate_profile():
    """A minimal multi-segment profile with one narrow, offset gate -- just
    enough to exercise position-dependent bounds."""
    return WidthProfile([
        WidthSegment(-np.inf, 2.0, 2.0, 0.0),
        WidthSegment(2.0, 5.0, 0.75, 1.0),
        WidthSegment(5.0, np.inf, 2.0, 0.0),
    ])


@pytest.mark.parametrize("backend", BACKENDS)
def test_get_action_does_not_leak_the_warnings_filter(backend):
    worker = MPCWorker(horizon=5, backend=backend)
    warnings.resetwarnings()
    before = warnings.filters.copy()

    state = np.array([1.0, 0.0, 0.0, 0.0])
    goal = np.array([2.0, 0.0, 0.0, 0.0])
    worker.get_action(state, goal, steps_left=3)

    assert warnings.filters == before


def test_default_Q_matches_the_training_script():
    """Q used to default to diag([10, 10, 0, 0]) -- no velocity tracking --
    while each scenario's script_ppo_mpc.py always passed diag([10, 10, 1, 1]). A
    worker built directly (a notebook, a test) silently behaved differently
    from the trained configuration."""
    worker = MPCWorker()
    assert np.allclose(worker.Q, np.diag([10.0, 10.0, 1.0, 1.0]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_get_action_drives_toward_a_reachable_goal(backend):
    """A basic sanity check on the QP: commanding acceleration toward a goal
    ahead of the agent should point the chosen action forward, not backward."""
    worker = MPCWorker(horizon=10, backend=backend)
    state = np.array([1.0, 0.0, 0.0, 0.0])
    goal = np.array([5.0, 0.0, 0.0, 0.0])  # target 5m further along +x
    action = worker.get_action(state, goal, steps_left=10)
    assert action[0] > 0  # accelerates toward +x
    assert abs(action[1]) < 1e-6  # no reason to move laterally


@pytest.mark.parametrize("backend", BACKENDS)
def test_get_action_respects_action_bounds(backend):
    worker = MPCWorker(u_max=1.0, horizon=10, backend=backend)
    state = np.array([1.0, 0.0, 0.0, 0.0])
    goal = np.array([100.0, 0.0, 0.0, 0.0])  # unreachable in the horizon
    action = worker.get_action(state, goal, steps_left=10)
    assert np.all(np.abs(action) <= 1.0 + 1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_steps_left_is_clamped_into_the_precomputed_range(backend):
    """steps_left is clamped to [1, N]; anything outside would KeyError on
    the per-horizon problems, since only n=1..N are built."""
    worker = MPCWorker(horizon=5, backend=backend)
    state = np.array([1.0, 0.0, 0.0, 0.0])
    goal = np.array([2.0, 0.0, 0.0, 0.0])
    # Should not raise despite out-of-range steps_left.
    worker.get_action(state, goal, steps_left=0)
    worker.get_action(state, goal, steps_left=999)


def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        MPCWorker(horizon=3, backend="ipopt")


# --------------------------------------------------------------------------
# Position-dependent width profile
# --------------------------------------------------------------------------

def test_no_profile_reproduces_the_historical_constant_width_envelope():
    worker = MPCWorker(W=4.0)
    assert worker.width_profile.envelope() == (-2.0, 2.0)
    assert worker.x_min[1] == -2.0 and worker.x_max[1] == 2.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_solve_respects_the_position_dependent_gate_bound(backend):
    """A single `get_action` call's u_0 alone is a weak signal here: the QP
    is free to front-load acceleration and only ease off at a *later* stage
    in the horizon, so u_0 stays nearly identical regardless of how close
    the wall is (confirmed empirically -- comparing single actions was tried
    first and does not discriminate). Driving the same worker closed-loop
    toward an aggressive, either-way-unreachable target does: the box
    constraint the QP actually enforces at every stage of every solve
    accumulates over repeated calls into a trajectory that saturates at
    each segment's *own* bound, not a shared/frozen one.
    """
    profile = _gate_profile()
    worker = MPCWorker(horizon=10, backend=backend, width_profile=profile)
    goal = np.array([0.0, 5.0, 0.0, 0.0])  # aggressive, unreachable-either-way +y push

    def rollout(x0, steps=15):
        state = np.array([x0, 1.0, 0.0, 0.0])
        for _ in range(steps):
            action = worker.get_action(state, goal, steps_left=10)
            state = worker.A @ state + worker.B @ np.clip(action, -worker.u_max, worker.u_max)
            y_lo, y_hi = profile.bounds_at(state[0])
            assert y_lo - 1e-6 <= state[1] <= y_hi + 1e-6, (
                f"x={state[0]}: y={state[1]} violated this segment's own "
                f"bound [{y_lo}, {y_hi}]")
        return state[1]

    final_y_gate = rollout(x0=3.0)  # gate 1: y in [0.25, 1.75]
    final_y_open = rollout(x0=0.0)  # full-width entry segment: y in [-2, 2]

    # Each trajectory saturates at its own segment's bound (checked point-by
    # -point above); the gate's is materially tighter, so it ends up closer
    # to its own wall than the open trajectory gets to its (further) one.
    assert final_y_gate < final_y_open - 0.05


# --------------------------------------------------------------------------
# The OSQP assembly against the cvxpy reference
# --------------------------------------------------------------------------

def test_osqp_backend_matches_the_cvxpy_reference():
    """The claim that makes the 81x speedup safe to take: same QP, same action.

    Tolerance is 1e-3 on the action, two orders looser than the ~7e-5 worst
    case observed over these states, because both backends stop at their own
    1e-5 convergence tolerance rather than at the exact optimum.
    """
    width_profile = None
    ref = MPCWorker(horizon=10, backend="cvxpy", width_profile=width_profile)
    fast = MPCWorker(horizon=10, backend="osqp", width_profile=width_profile)
    rng = np.random.default_rng(0)

    worst = 0.0
    for _ in range(30):
        state = np.array([rng.uniform(0, 10), rng.uniform(-1.8, 1.8),
                          rng.uniform(-2, 2), rng.uniform(-2, 2)])
        goal = np.array([rng.uniform(-10, 10), rng.uniform(-10, 10),
                         rng.uniform(-2, 2), rng.uniform(-2, 2)])
        steps_left = int(rng.integers(1, 11))
        a_ref = ref.get_action(state, goal, steps_left)
        a_fast = fast.get_action(state, goal, steps_left)
        worst = max(worst, float(np.max(np.abs(a_ref - a_fast))))
    assert worst < 1e-3, f"worst |cvxpy - osqp| = {worst:.2e}"


def test_osqp_backend_returns_the_braking_fallback_on_a_failed_solve(monkeypatch):
    """The fallback path is the one that only runs when something has already
    gone wrong, so it is the one that rots unnoticed."""
    worker = MPCWorker(horizon=5, backend="osqp")
    state = np.array([1.0, 0.0, 2.0, -1.0])

    class _Boom:
        def update(self, **kwargs):
            raise RuntimeError("solver exploded")

    monkeypatch.setitem(worker._solvers, (5, 0), _Boom())
    action = worker.get_action(state, np.zeros(4), steps_left=5)
    assert np.allclose(action, np.clip(-state[2:] / worker.dt, -1.0, 1.0))


# --------------------------------------------------------------------------
# Batched solving
# --------------------------------------------------------------------------

def test_get_actions_matches_per_environment_get_action():
    worker = MPCWorker(horizon=10, backend="osqp", num_envs=4)
    single = MPCWorker(horizon=10, backend="osqp", num_envs=1)
    rng = np.random.default_rng(1)
    states = rng.uniform(-1, 3, (4, 4))
    goals = rng.uniform(-5, 5, (4, 4))
    steps_left = np.array([10, 7, 1, 4])

    batched = worker.get_actions(states, goals, steps_left)

    for i in range(4):
        expected = single.get_action(states[i], goals[i], steps_left[i])
        assert np.allclose(batched[i], expected, atol=1e-3), f"env {i}"


def test_get_actions_rejects_more_environments_than_warm_start_slots():
    """Silently reusing slot 0 for every environment would make each solve
    warm-start from a different environment's solution."""
    worker = MPCWorker(horizon=5, backend="osqp", num_envs=2)
    with pytest.raises(ValueError, match="warm-start slot"):
        worker.get_actions(np.zeros((3, 4)), np.zeros((3, 4)), np.ones(3, dtype=int))


def test_each_environment_slot_has_its_own_solver():
    worker = MPCWorker(horizon=3, backend="osqp", num_envs=3)
    solvers = [worker._solvers[(3, i)] for i in range(3)]
    assert len({id(s) for s in solvers}) == 3
