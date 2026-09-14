"""Regression tests for the PPO+MPC-Reachability manager implementation.

Each test pins down a property that is either non-obvious from the source or
easy to break silently. The set mirrors algorithms/ppo_mpc/tests/test_ppo_mpc.py
for everything this module still shares with it -- the GAE indexing convention,
the value standardisation round trip, the derived manager discount, the
metric definitions -- and adds a dedicated section for what this fork changes:
`scale_goal` no longer applies a fixed diagonal box, it maps the manager's
raw [-1,1]^4 action through the plant's exact reachable set (`reachable_goal`
in ppo_mpc_reach.py; PPO_MPC_Reachability_explanation.md section 3). The tests below
under "Reachability-aware goal scaling" verify that claim independently of
`reachable_goal`'s own algebra, via a from-scratch linear-program feasibility
check, rather than just re-deriving the same formula and comparing.
"""

import os
import tempfile

import numpy as np
import pytest
import torch
from scipy.optimize import linprog

from ppo_mpc_reach import (
    ManagerActor,
    ManagerRolloutBuffer,
    PPOMPCAgent,
    RunningMeanStd,
    ScaledBeta,
    normalize_obs,
    reachable_goal,
)
from envs.width_profile import WidthSegment, WidthProfile


def _gate_profile():
    """A minimal multi-segment profile with one narrow, offset gate -- just
    enough to exercise position-dependent bounds."""
    return WidthProfile([
        WidthSegment(-np.inf, 2.0, 2.0, 0.0),
        WidthSegment(2.0, 5.0, 0.75, 1.0),
        WidthSegment(5.0, np.inf, 2.0, 0.0),
    ])

OBS_DIM, GOAL_DIM = 4, 4
OBS_LOW = [-1.0, -2.0, -2.0, -2.0]
OBS_HIGH = [11.0, 2.0, 2.0, 2.0]
# The environment's state bounds coincide with its observation bounds (position and
# velocity are both fully observed and both hard-enforced), so the same pair
# doubles as the reachability layer's state_min/state_max in these tests.
STATE_MIN, STATE_MAX = OBS_LOW, OBS_HIGH
DT, U_MAX = 0.1, 1.0


def _agent(**kwargs):
    kwargs.setdefault("obs_low", OBS_LOW)
    kwargs.setdefault("obs_high", OBS_HIGH)
    kwargs.setdefault("dt", DT)
    kwargs.setdefault("u_max", U_MAX)
    kwargs.setdefault("state_min", STATE_MIN)
    kwargs.setdefault("state_max", STATE_MAX)
    return PPOMPCAgent(OBS_DIM, GOAL_DIM, device="cpu", **kwargs)


def _filled_buffer(num_steps=16, obs_dim=OBS_DIM, goal_dim=GOAL_DIM, seed=0, fill=None):
    """A buffer with plausible, reproducible contents, filled to `fill` steps.

    Partial filling is the norm here, not an edge case: the manager writes one
    transition per c worker steps into a buffer sized for the worker's step
    budget.
    """
    torch.manual_seed(seed)
    buf = ManagerRolloutBuffer(num_steps, obs_dim, goal_dim, "cpu")
    buf.states = torch.randn(num_steps, obs_dim)
    buf.actions = torch.rand(num_steps, goal_dim) * 2 - 1
    buf.logprobs = torch.randn(num_steps)
    buf.rewards = torch.randn(num_steps) * 100
    buf.values = torch.randn(num_steps) * 100
    buf.dones = (torch.rand(num_steps) < 0.3).float()
    buf.step = num_steps if fill is None else fill
    return buf


# --------------------------------------------------------------------------
# Observation map
# --------------------------------------------------------------------------

def test_normalize_obs_maps_bounds_to_plus_minus_one():
    low = np.array(OBS_LOW, dtype=np.float32)
    high = np.array(OBS_HIGH, dtype=np.float32)
    assert normalize_obs(low, low, high) == pytest.approx([-1.0] * 4)
    assert normalize_obs(high, low, high) == pytest.approx([1.0] * 4)
    mid = (low + high) / 2
    assert normalize_obs(mid, low, high) == pytest.approx([0.0] * 4)


def test_agent_without_bounds_refuses_to_normalize():
    """Silently feeding raw physical units (p_x in [-1, 11]) to a policy
    trained on [-1, 1] is the failure this guard exists to make loud."""
    agent = PPOMPCAgent(OBS_DIM, GOAL_DIM, device="cpu")
    with pytest.raises(ValueError, match="observation bounds"):
        agent.normalize_obs(np.zeros(OBS_DIM, dtype=np.float32))


def test_agent_without_reachability_geometry_refuses_to_scale():
    agent = PPOMPCAgent(OBS_DIM, GOAL_DIM, device="cpu")
    with pytest.raises(ValueError, match="reachability geometry"):
        agent.scale_goal(np.zeros(GOAL_DIM, dtype=np.float32), np.zeros(OBS_DIM, dtype=np.float32))


# --------------------------------------------------------------------------
# Reachability-aware goal scaling (reachable_goal / PPOMPCAgent.scale_goal)
# --------------------------------------------------------------------------

def _is_exactly_reachable(dp, dv, manager_freq, dt, u_max, tol=1e-6):
    """From-scratch feasibility check, independent of `reachable_goal`'s own
    derivation: does there exist a control sequence u in [-u_max, u_max]^N
    (N = manager_freq) whose double-integrator response over one segment is
    *exactly* (dp, dv)? `dp` excludes the free v0*T drift term -- it is the
    control-induced position change alone, matching what g1/g2 span.

    v_N - v0 = dt * sum(u_k);  p_N - p0 - N*dt*v0 = dt**2 * sum(w_k * u_k),
    w_k = N - k - 0.5 (k=0..N-1) -- the same weights `PPO_MPC_Reachability_
    explanation.md` section 3 derives. This is a pure feasibility LP (zero
    objective): `linprog` reports success iff some point in the box satisfies
    both linear equalities, i.e. iff (dp, dv) is in the true reachable set.
    """
    k = np.arange(manager_freq)
    w = manager_freq - k - 0.5
    A_eq = np.array([np.full(manager_freq, dt), dt**2 * w])
    b_eq = np.array([dv, dp])
    res = linprog(c=np.zeros(manager_freq), A_eq=A_eq, b_eq=b_eq,
                   bounds=[(-u_max, u_max)] * manager_freq, method="highs")
    return res.success


@pytest.mark.parametrize("manager_freq", [10, 9, 11, 1, 2, 3])
def test_scale_goal_outputs_are_exactly_reachable(manager_freq):
    """The literal claim this whole fork exists for: every goal `scale_goal`
    returns is one an MPCWorker with the same dt/u_max/manager_freq can hit
    *exactly* in one segment -- checked via `_is_exactly_reachable`'s
    independent LP, not by recomputing reachable_goal's own formula.

    Parametrized over odd *and* even manager_freq: `g2`'s continuous-time
    closed form (reachable_goal's docstring, section 3.2 of the explanation
    doc) is only the exact discrete optimum for even N -- an earlier version
    of this function used it unconditionally and was a hair outside the true
    reachable set at odd N (confirmed against this same LP oracle, by
    exactly `u_max*dt**2/4`, before the parity correction was added). 10 is
    the only manager_freq this repo's scripts ever use, but the guarantee
    should not silently depend on that."""
    agent = _agent(manager_freq=manager_freq)
    rng = np.random.default_rng(0)
    for _ in range(30):
        state = rng.uniform(STATE_MIN, STATE_MAX)
        z = rng.uniform(-1.0, 1.0, size=4)
        goal = agent.scale_goal(z, state)
        for p_idx, v_idx in ((0, 2), (1, 3)):
            dp_ctrl = goal[p_idx] - agent.manager_freq * agent.dt * state[v_idx]
            assert _is_exactly_reachable(dp_ctrl, goal[v_idx], agent.manager_freq,
                                         agent.dt, agent.u_max), (state, z, goal)


@pytest.mark.parametrize("accel_split", [0.0, 0.3, 0.7, 1.0])
def test_scale_goal_outputs_are_exactly_reachable_across_accel_split(accel_split):
    """`--accel-split` is an exposed, user-facing tuning knob (section 9.2 of
    the explanation doc measures its cost at the default and recommends
    sweeping it) -- the reachability guarantee must hold at every value in
    its valid range, including both extremes (0: all budget to the turn
    generator g2, g1 contributes nothing; 1: the reverse), not only at the
    untested default of 0.5."""
    agent = _agent(manager_freq=10, accel_split=accel_split)
    rng = np.random.default_rng(0)
    for _ in range(30):
        state = rng.uniform(STATE_MIN, STATE_MAX)
        z = rng.uniform(-1.0, 1.0, size=4)
        goal = agent.scale_goal(z, state)
        for p_idx, v_idx in ((0, 2), (1, 3)):
            dp_ctrl = goal[p_idx] - agent.manager_freq * agent.dt * state[v_idx]
            assert _is_exactly_reachable(dp_ctrl, goal[v_idx], agent.manager_freq,
                                         agent.dt, agent.u_max), (state, z, goal)


def test_scale_goal_rejects_an_out_of_range_accel_split():
    """Caught a real gap while reviewing this module: accel_split=1.5 (e.g. a
    typo) silently returned a goal requesting dv=1.5 at u_max=T=1 -- a target
    outside the true reachable set, with no warning that the whole point of
    this function (an exact reachability guarantee) had quietly stopped
    holding. Should fail loudly instead."""
    agent = _agent(manager_freq=10, accel_split=1.5)
    with pytest.raises(ValueError, match="accel_split"):
        agent.scale_goal(np.array([1.0, 0.0, 1.0, 0.0]), np.array([5.0, 0.0, 0.0, 0.0]))


def test_scale_goal_respects_hard_bounds_when_the_state_allows_it():
    """The hard-bound clip is best-effort *beyond* the kinodynamic guarantee,
    not a second unconditional one: from a state already moving at speed
    toward a nearby wall, braking for the *entire* segment can still be too
    little too late (u_max=1, v_max=2 means stopping from top speed takes 2s
    = 20 steps, twice manager_freq=10's 1s segment) -- no target, reachability
    -aware or not, changes that, and MPCWorker's own QP is equally infeasible
    from such a state (falls back to `_braking_action`, `PPO_MPC_explanation.
    md` section 8.4). What *is* unconditional: whenever coasting alone (the
    alpha=0 floor) already satisfies both bounds, scale_goal's output must
    too -- filtered to exactly that recoverable regime here."""
    agent = _agent(manager_freq=10)
    rng = np.random.default_rng(1)
    lo, hi = np.array(STATE_MIN), np.array(STATE_MAX)
    T = agent.manager_freq * agent.dt
    n_checked = 0
    for _ in range(500):
        state = rng.uniform(lo, hi)
        coast = state.copy()
        coast[0] += T * state[2]
        coast[1] += T * state[3]
        if not (np.all(coast >= lo - 1e-9) and np.all(coast <= hi + 1e-9)):
            continue  # unrecoverable from this state regardless of the target
        n_checked += 1
        z = rng.uniform(-1.0, 1.0, size=4)
        target = state + agent.scale_goal(z, state)
        assert np.all(target >= lo - 1e-6)
        assert np.all(target <= hi + 1e-6)
    assert n_checked > 50  # sanity: the filter isn't discarding almost everything


def test_scale_goal_at_zero_action_is_pure_coast():
    """z=0 on both slots of an axis requests neither acceleration nor turn,
    so the goal must reduce to exactly the free (uncontrolled) drift term."""
    agent = _agent(manager_freq=10)
    state = np.array([5.0, 0.0, 1.0, -0.5])
    goal = agent.scale_goal(np.zeros(4), state)
    T = agent.manager_freq * agent.dt
    assert goal == pytest.approx([T * 1.0, T * -0.5, 0.0, 0.0])


def test_scale_goal_applies_maximum_correction_when_unrecoverable():
    """From a state where even coasting already violates a bound (see the
    previous test's docstring), shrinking the request toward coast makes
    things worse, not better -- alpha must pick the *most* corrective value
    available (1, using the full braking request) rather than collapsing
    toward the broken alpha=0 floor `_headroom_ratio`'s ordinary branch uses."""
    agent = _agent(manager_freq=10)
    state = np.array([5.0, 2.0, 0.0, 2.0])  # p_y, v_y both already at the wall
    z_full_brake = np.array([0.0, -1.0, 0.0, -1.0])
    z_no_brake = np.array([0.0, 1.0, 0.0, 1.0])
    braked = agent.scale_goal(z_full_brake, state)[1]
    unbraked = agent.scale_goal(z_no_brake, state)[1]
    assert braked < unbraked  # more braking must still move the target down


def test_scale_goal_saturates_gracefully_at_top_speed():
    """At v0 == v_max, requesting further positive acceleration must not push
    v_target past v_max -- the segment can only coast, not accelerate more."""
    agent = _agent(manager_freq=10)
    state = np.array([5.0, 0.0, 2.0, 0.0])  # v_x already at v_max
    goal = agent.scale_goal(np.array([1.0, 0.0, 1.0, 0.0]), state)
    assert goal[2] == pytest.approx(0.0, abs=1e-6)


def test_reachable_goal_matches_agent_scale_goal():
    """PPOMPCAgent.scale_goal is a thin dispatch onto the free function --
    pinned so the two cannot silently diverge."""
    agent = _agent(manager_freq=10)
    state = np.array([4.0, -1.0, 0.5, 1.0])
    z = np.array([0.3, -0.6, 0.9, -0.2])
    direct = reachable_goal(z, state, agent.manager_freq, agent.dt, agent.u_max,
                            agent.state_min, agent.state_max, agent.accel_split)
    assert agent.scale_goal(z, state) == pytest.approx(direct)


# --------------------------------------------------------------------------
# Position-dependent width profile
# --------------------------------------------------------------------------

def test_width_profile_none_matches_the_historical_frozen_bound():
    """A caller passing no width_profile (every caller before this feature
    existed) must get exactly today's behaviour: the y-axis hard bound is
    state_min[1]/state_max[1], not looked up anywhere."""
    state = np.array([4.0, -1.0, 0.5, 1.0])
    z = np.array([0.3, -0.6, 0.9, -0.2])
    no_profile = reachable_goal(z, state, 10, DT, U_MAX, STATE_MIN, STATE_MAX,
                                accel_split=0.5, width_profile=None)

    # A single-segment profile spanning the same bound must reproduce it
    # exactly -- proving the width_profile path, when it degenerates to the
    # frozen case, is not a different code path with a different answer.
    single_segment = WidthProfile([WidthSegment(-np.inf, np.inf, (STATE_MAX[1] - STATE_MIN[1]) / 2.0,
                                                 (STATE_MAX[1] + STATE_MIN[1]) / 2.0)])
    with_profile = reachable_goal(z, state, 10, DT, U_MAX, STATE_MIN, STATE_MAX,
                                  accel_split=0.5, width_profile=single_segment)
    assert with_profile == pytest.approx(no_profile)


def test_scale_goal_y_clip_differs_between_gate_and_open_segment():
    """The whole point of threading width_profile through: an aggressive
    lateral request must be clipped harder from inside a narrow gate than
    from a full-width segment, using the position the manager actually
    observed (state_phys[..., 0]), not a frozen bound."""
    profile = _gate_profile()
    agent = _agent(manager_freq=10, width_profile=profile)
    z = np.array([0.0, 1.0, 0.0, 1.0])  # full turn+accel budget toward +y

    # y0=1.5: gate 1's headroom to its own wall (1.75 - 1.5 = 0.25) is less
    # than accel_split=0.5's max reachable |dp| (0.375), so the clip actually
    # binds there; the open segment's headroom (2.0 - 1.5 = 0.5) does not,
    # so its target is the unclipped request -- the two must therefore differ.
    state_in_gate = np.array([3.0, 1.5, 0.0, 0.0])  # gate 1: y in [0.25, 1.75]
    state_open = np.array([0.0, 1.5, 0.0, 0.0])  # entry segment: y in [-2, 2]

    goal_gate = agent.scale_goal(z, state_in_gate)
    goal_open = agent.scale_goal(z, state_open)

    target_y_gate = state_in_gate[1] + goal_gate[1]
    target_y_open = state_open[1] + goal_open[1]

    gate_lo, gate_hi = profile.bounds_at(state_in_gate[0])
    open_lo, open_hi = profile.bounds_at(state_open[0])
    assert target_y_gate <= gate_hi + 1e-6
    assert target_y_open <= open_hi + 1e-6
    # the gate's own bound is what actually did the clipping, not the wider
    # open-segment bound (or the wider still global envelope, +-2.0 either
    # way here -- so this also confirms the lookup is position-specific).
    assert target_y_gate < target_y_open - 0.1


def test_scale_goal_batched_rows_use_their_own_segment():
    """One batched call with different rows in different segments must clip
    each row against its own row's position (width_profile.bounds_at's
    vectorized lookup), not a single shared bound broadcast over the batch."""
    profile = _gate_profile()
    agent = _agent(manager_freq=10, width_profile=profile)
    z = np.tile(np.array([0.0, 1.0, 0.0, 1.0]), (2, 1))
    states = np.array([
        [3.0, 1.5, 0.0, 0.0],  # gate 1: this request's clip actually binds here
        [0.0, 1.5, 0.0, 0.0],  # open entry segment: same request, unclipped
    ])

    goals = agent.scale_goal(z, states)  # one batched call, not a loop
    assert goals.shape == (2, 4)
    targets_y = states[:, 1] + goals[:, 1]

    gate_hi = profile.bounds_at(states[0, 0])[1]
    open_hi = profile.bounds_at(states[1, 0])[1]
    assert targets_y[0] <= gate_hi + 1e-6
    assert targets_y[1] <= open_hi + 1e-6
    assert targets_y[0] < targets_y[1] - 0.1


# --------------------------------------------------------------------------
# The manager's derived segment-level discount
# --------------------------------------------------------------------------

def test_gamma_manager_is_derived_from_manager_freq():
    """One manager step spans `manager_freq` env steps (SMDP/options
    treatment). A previous version of this class had no manager_freq at all,
    so the script had to independently compute gamma**c and pass it into
    compute_returns_and_advantage by hand -- exactly the kind of duplicated
    discount that drifted apart in HRL/hPPO's history."""
    agent = _agent(gamma=0.99, manager_freq=10)
    assert agent.gamma_manager == pytest.approx(0.99 ** 10)


def test_loading_a_different_manager_freq_updates_the_derived_discount():
    agent = _agent(gamma=0.99, manager_freq=10)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        agent.save(path)

        reloaded = _agent(gamma=0.99, manager_freq=999)  # deliberately wrong
        reloaded.load(path)

    assert reloaded.manager_freq == 10
    assert reloaded.gamma_manager == pytest.approx(0.99 ** 10)


# --------------------------------------------------------------------------
# GAE
# --------------------------------------------------------------------------

def test_gae_matches_transition_indexed_reference():
    """dones[t] means 'transition t ended an episode', so the mask is
    1 - dones[t]. CleanRL stores the other convention and masks with
    dones[t+1]; this test pins ours down so the difference is never 'fixed'
    by mistake."""
    buf = _filled_buffer()
    next_value, next_done = torch.randn(1).item(), 1.0
    gamma, lam = 0.99 ** 10, 0.95
    buf.compute_returns_and_advantage(next_value, next_done, gamma, lam)

    expected = torch.zeros(buf.step)
    last = 0.0
    for t in reversed(range(buf.step)):
        if t == buf.step - 1:
            nonterminal, next_v = 1.0 - next_done, next_value
        else:
            nonterminal, next_v = 1.0 - buf.dones[t].item(), buf.values[t + 1].item()
        delta = buf.rewards[t].item() + gamma * next_v * nonterminal - buf.values[t].item()
        last = delta + gamma * lam * nonterminal * last
        expected[t] = last
    assert torch.allclose(buf.advantages[:buf.step], expected, atol=1e-4)


def test_gae_returns_identity_holds():
    buf = _filled_buffer()
    buf.compute_returns_and_advantage(0.0, 0.0, 0.99 ** 10, 0.95)
    assert torch.allclose(buf.returns[:buf.step], buf.advantages[:buf.step] + buf.values[:buf.step], atol=1e-4)


def test_gae_stops_bootstrapping_at_a_done():
    buf = ManagerRolloutBuffer(2, OBS_DIM, GOAL_DIM, "cpu")
    buf.add(np.zeros(OBS_DIM), np.zeros(GOAL_DIM), 0.0, 5.0, 0.0, 1.0)  # terminal segment
    buf.add(np.zeros(OBS_DIM), np.zeros(GOAL_DIM), 0.0, 7.0, 0.0, 0.0)
    buf.compute_returns_and_advantage(100.0, 0.0, 0.99 ** 10, 0.95)
    assert buf.advantages[0].item() == pytest.approx(5.0, abs=1e-4)


def test_gae_parameters_are_required():
    """They used to default to 0.99/0.95 while the manager is discounted at
    gamma**manager_freq -- a caller who omitted them got silently different
    discounting from the trained configuration."""
    buf = _filled_buffer()
    with pytest.raises(TypeError):
        buf.compute_returns_and_advantage(0.0, 0.0)


def test_gae_handles_a_partially_filled_buffer():
    """The manager writes far fewer transitions than the buffer's step budget
    (one per c worker steps); GAE and get() must only touch what was filled."""
    buf = _filled_buffer(num_steps=64, fill=5)
    buf.compute_returns_and_advantage(0.0, 0.0, 0.99 ** 10, 0.95)
    states, actions, logprobs, returns, advantages = buf.get()
    assert states.shape[0] == 5
    assert returns.shape[0] == 5
    assert advantages.shape[0] == 5


def test_agent_is_the_single_source_of_the_discount():
    agent = _agent(gamma=0.5, manager_freq=2, gae_lambda=0.5)  # gamma_manager = 0.25
    buf_a, buf_b = _filled_buffer(), _filled_buffer()
    agent.compute_manager_returns_and_advantage(buf_a, 0.0, 0.0)
    buf_b.compute_returns_and_advantage(0.0, 0.0, 0.25, 0.5)
    assert torch.allclose(buf_a.advantages, buf_b.advantages, atol=1e-6)

    agent.gamma_manager = 0.999
    agent.compute_manager_returns_and_advantage(buf_a, 0.0, 0.0)
    assert not torch.allclose(buf_a.advantages, buf_b.advantages, atol=1e-6)


# --------------------------------------------------------------------------
# Value-target standardisation
# --------------------------------------------------------------------------

def test_get_manager_value_inverts_the_standardisation():
    agent = _agent()
    agent.manager_ret_rms.mean, agent.manager_ret_rms.var = 350.0, 53570.0
    states = torch.randn(8, OBS_DIM)
    raw = agent.manager_critic(states).squeeze(-1)
    assert torch.allclose(
        agent.get_manager_value(states),
        raw * agent.manager_ret_rms.std + agent.manager_ret_rms.mean,
        atol=1e-4,
    )


def test_running_mean_std_tracks_batch_statistics():
    rms = RunningMeanStd()
    x = torch.randn(4096) * 30 + 400
    rms.update(x)
    assert rms.mean == pytest.approx(float(x.mean()), rel=1e-3)
    assert rms.std == pytest.approx(float(x.std(unbiased=False)), rel=1e-2)


# --------------------------------------------------------------------------
# Checkpoint round trip
# --------------------------------------------------------------------------

def test_checkpoint_is_self_describing():
    """ret_rms, the observation bounds, the reachability geometry (dt, u_max,
    state_min, state_max, accel_split) AND manager_freq must all survive a
    save/load round trip, or a reloaded manager's inputs, outputs, discount,
    and goal-reachability guarantee are all wrong."""
    agent = _agent(gamma=0.99, manager_freq=10, width_profile=_gate_profile())
    agent.manager_ret_rms.mean, agent.manager_ret_rms.var = 350.0, 53570.0
    states = torch.randn(8, OBS_DIM)
    before = agent.get_manager_value(states)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        agent.save(path)
        reloaded = PPOMPCAgent(OBS_DIM, GOAL_DIM, gamma=0.99, manager_freq=10, device="cpu")
        reloaded.load(path)  # must not need weights_only=False

    assert reloaded.obs_low == pytest.approx(OBS_LOW)
    assert reloaded.obs_high == pytest.approx(OBS_HIGH)
    assert reloaded.dt == pytest.approx(DT)
    assert reloaded.u_max == pytest.approx(U_MAX)
    assert reloaded.state_min == pytest.approx(STATE_MIN)
    assert reloaded.state_max == pytest.approx(STATE_MAX)
    assert reloaded.accel_split == pytest.approx(0.5)
    assert reloaded.manager_freq == 10
    assert torch.allclose(reloaded.get_manager_value(states), before, atol=1e-5)
    assert reloaded.normalize_obs(np.array(OBS_HIGH, dtype=np.float32)) == pytest.approx([1.0] * 4)

    assert reloaded.width_profile is not None
    assert [(s.x_start, s.x_end, s.half_width, s.center_y) for s in reloaded.width_profile.segments] == \
        [(s.x_start, s.x_end, s.half_width, s.center_y) for s in agent.width_profile.segments]

    probe_state = np.array([5.0, 0.0, 0.3, -0.3], dtype=np.float32)
    probe_z = np.array([0.4, -0.7, 0.6, 0.1], dtype=np.float32)
    assert reloaded.scale_goal(probe_z, probe_state) == pytest.approx(agent.scale_goal(probe_z, probe_state))

    # A state inside one of the reloaded profile's gates must still clip
    # the way it did before saving -- not just that the segment list matches,
    # but that reloaded.scale_goal actually consults it.
    gate_state = np.array([3.0, 1.0, 0.0, 0.0], dtype=np.float32)
    gate_z = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    assert reloaded.scale_goal(gate_z, gate_state) == pytest.approx(agent.scale_goal(gate_z, gate_state))


def test_load_tolerates_a_single_group_optimiser_state(capsys):
    """Checkpoints written before actor/critic had separate parameter groups
    hold a one-group optimiser state. Adam rejects it; `load` must warn and
    keep the weights rather than raise."""
    import torch.optim as optim

    old = _agent()
    old.manager_optimizer = optim.Adam(
        list(old.manager_actor.parameters()) + list(old.manager_critic.parameters()),
        lr=3e-4, eps=1e-5,
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.pt")
        old.save(path)

        new = _agent(critic_lr_mult=3.0)
        new.load(path)  # must not raise

    assert "predates the two-group optimiser" in capsys.readouterr().out
    for a, b in zip(old.manager_actor.parameters(), new.manager_actor.parameters()):
        assert torch.allclose(a, b)
    for a, b in zip(old.manager_critic.parameters(), new.manager_critic.parameters()):
        assert torch.allclose(a, b)


# --------------------------------------------------------------------------
# The two-group optimiser
# --------------------------------------------------------------------------

def test_optimiser_splits_actor_and_critic_into_separate_groups():
    agent = _agent(lr_manager=3e-4, critic_lr_mult=3.0)
    groups = agent.manager_optimizer.param_groups
    assert len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(3e-4)
    assert groups[1]["lr"] == pytest.approx(9e-4)

    actor_ids = {id(p) for p in agent.manager_actor.parameters()}
    critic_ids = {id(p) for p in agent.manager_critic.parameters()}
    assert {id(p) for p in groups[0]["params"]} == actor_ids
    assert {id(p) for p in groups[1]["params"]} == critic_ids


def test_critic_lr_mult_of_one_restores_a_single_rate():
    agent = _agent(lr_manager=3e-4, critic_lr_mult=1.0)
    assert [g["lr"] for g in agent.manager_optimizer.param_groups] == pytest.approx([3e-4, 3e-4])


def test_annealing_every_group_keeps_the_critic_ratio():
    agent = _agent(lr_manager=3e-4, critic_lr_mult=3.0)
    base_lrs = [g["lr"] for g in agent.manager_optimizer.param_groups]
    for group, base_lr in zip(agent.manager_optimizer.param_groups, base_lrs):
        group["lr"] = 0.5 * base_lr
    lrs = [g["lr"] for g in agent.manager_optimizer.param_groups]
    assert lrs == pytest.approx([1.5e-4, 4.5e-4])
    assert lrs[1] / lrs[0] == pytest.approx(3.0)


# --------------------------------------------------------------------------
# update_manager()
# --------------------------------------------------------------------------

def _prepared_agent_and_buffer():
    agent = _agent()
    buf = _filled_buffer(num_steps=8)
    with torch.no_grad():
        _, logprobs, _ = agent.manager_policy_forward(buf.states, buf.actions)
    buf.logprobs = logprobs
    agent.compute_manager_returns_and_advantage(buf, 0.0, 0.0)
    return agent, buf


def test_explained_variance_uses_pre_update_values():
    """Scoring the critic after its own update epochs on the batch measures
    training-set fit, not prediction -- the metric must use the values that
    actually produced the advantages. A previous version recomputed the
    critic's output on `states` *after* the update loop, which is optimistically
    biased and inconsistent with RL/PPO's and HRL/hPPO's definition."""
    agent, buf = _prepared_agent_and_buffer()
    returns = buf.returns[:buf.step].numpy()
    values = buf.values[:buf.step].numpy()
    expected = 1 - np.var(returns - values) / np.var(returns)

    metrics = agent.update_manager(buf, minibatch_size=8, update_epochs=10)
    assert metrics["manager/explained_variance"] == pytest.approx(expected, abs=1e-4)
    assert metrics["manager/value_bias"] == pytest.approx(float(np.mean(values - returns)), abs=1e-2)


def test_update_manager_reports_the_expected_metrics():
    agent, buf = _prepared_agent_and_buffer()
    metrics = agent.update_manager(buf, minibatch_size=8, update_epochs=2)
    assert set(metrics) == {
        "manager/loss_policy", "manager/loss_value", "manager/entropy",
        "manager/approx_kl", "manager/clipfrac", "manager/value_bias",
        "manager/value_target_mean", "manager/value_target_std",
        "manager/explained_variance",
    }
    assert all(isinstance(v, float) for v in metrics.values())


def test_update_manager_changes_both_actor_and_critic():
    agent, buf = _prepared_agent_and_buffer()
    actor_before = [p.clone() for p in agent.manager_actor.parameters()]
    critic_before = [p.clone() for p in agent.manager_critic.parameters()]
    agent.update_manager(buf, minibatch_size=8, update_epochs=2)
    assert any(not torch.allclose(a, b) for a, b in zip(actor_before, agent.manager_actor.parameters()))
    assert any(not torch.allclose(a, b) for a, b in zip(critic_before, agent.manager_critic.parameters()))


def test_update_manager_handles_a_partially_filled_buffer():
    """The common case: the manager's buffer ends an update ~c times emptier
    than the worker's step budget."""
    agent = _agent()
    buf = _filled_buffer(num_steps=64, fill=5)
    with torch.no_grad():
        _, logprobs, _ = agent.manager_policy_forward(buf.states[:5], buf.actions[:5])
    buf.logprobs[:5] = logprobs
    agent.compute_manager_returns_and_advantage(buf, 0.0, 0.0)
    metrics = agent.update_manager(buf, minibatch_size=8, update_epochs=2)
    assert np.isfinite(metrics["manager/loss_value"])


# --------------------------------------------------------------------------
# Defaults match the training script (script_ppo_mpc_reach.py)
# --------------------------------------------------------------------------

def test_agent_defaults_match_the_training_script():
    """ent_coef_manager used to default to 0.0 while the script always passed
    0.01 -- an agent built directly (e.g. in a notebook) silently behaved
    differently from every trained checkpoint."""
    agent = PPOMPCAgent(OBS_DIM, GOAL_DIM, device="cpu")
    assert agent.gamma == 0.99
    assert agent.manager_freq == 10
    assert agent.ent_coef_manager == pytest.approx(0.01)


# --------------------------------------------------------------------------
# ScaledBeta sanity (shared with RL/PPO and HRL/hPPO; a quick smoke check)
# --------------------------------------------------------------------------

def test_scaled_beta_samples_are_inside_the_action_box():
    dist = ScaledBeta(torch.full((2000,), 1.7), torch.full((2000,), 3.1), low=torch.tensor(-1.0), high=torch.tensor(1.0))
    samples = dist.sample()
    assert samples.min() >= -1.0 and samples.max() <= 1.0


def test_manager_actor_output_shapes():
    actor = ManagerActor(OBS_DIM, GOAL_DIM)
    alpha, beta = actor(torch.randn(5, OBS_DIM))
    assert alpha.shape == (5, GOAL_DIM)
    assert beta.shape == (5, GOAL_DIM)
    assert (alpha >= 1.0).all() and (beta >= 1.0).all()
