import numpy as np
import pytest

from envs.config import TunnelEnvConfig
from envs.tunnel_env import TunnelEnv
from envs.width_profile import WidthSegment, WidthProfile


def make(**overrides):
    return TunnelEnv(**overrides)


def test_reset_seeded_is_reproducible():
    env1 = make()
    env2 = make()
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)


def test_reset_initial_state_bounds():
    env = make()
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        p_x, p_y, v_x, v_y = obs
        assert 0.0 <= p_x <= 2.0
        assert -env.W / 4.0 <= p_y <= env.W / 4.0
        assert v_x == 0.0 and v_y == 0.0
        assert info["is_success"] is False
        assert info["collision"] is False
        assert info["collision_count"] == 0
        assert info["collision_impacts"] == []
        assert info["distance_to_goal"] == pytest.approx(env.L - p_x)


def test_reset_with_init_state_option_overrides_the_random_draw():
    env = make()
    # Deliberately outside the random spawn box (p_x in [0,2], p_y in
    # [-W/4,W/4]), to prove this isn't silently clamped to it.
    forced = np.array([7.5, -1.9, 0.3, -0.2], dtype=np.float32)
    obs, info = env.reset(seed=0, options={"init_state": forced})
    np.testing.assert_array_equal(obs, forced)
    np.testing.assert_array_equal(env.state, forced)
    assert info["distance_to_goal"] == pytest.approx(max(env.L - forced[0], 0.0))


def test_reset_without_options_is_unaffected_by_init_state_support():
    env1 = make()
    env2 = make()
    obs1, _ = env1.reset(seed=7)
    obs2, _ = env2.reset(seed=7, options=None)
    np.testing.assert_array_equal(obs1, obs2)
    p_x, p_y, v_x, v_y = obs1
    assert 0.0 <= p_x <= 2.0
    assert -env1.W / 4.0 <= p_y <= env1.W / 4.0
    assert v_x == 0.0 and v_y == 0.0


def test_step_dynamics_zero_noise_zero_action():
    env = make(sigma_p=0.0, sigma_v=0.0)
    obs, _ = env.reset(seed=0)
    action = np.zeros(2, dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)
    expected = env.A @ obs
    np.testing.assert_allclose(next_obs, expected, atol=1e-5)
    assert reward == env.config.step_penalty
    assert not terminated and not truncated


def test_step_dynamics_matches_A_B_with_action():
    env = make(sigma_p=0.0, sigma_v=0.0)
    obs, _ = env.reset(seed=0)
    action = np.array([0.5, -0.3], dtype=np.float32)
    next_obs, *_ = env.step(action)
    expected = env.A @ obs + env.B @ action
    expected = np.clip(expected, env.x_min, env.x_max)
    np.testing.assert_allclose(next_obs, expected, atol=1e-5)


def test_action_is_clipped_before_applied():
    env = make(sigma_p=0.0, sigma_v=0.0, u_max=1.0)
    obs, _ = env.reset(seed=0)
    oversized_action = np.array([5.0, -5.0], dtype=np.float32)
    next_obs, *_ = env.step(oversized_action)
    clipped_action = np.array([1.0, -1.0], dtype=np.float32)
    expected = np.clip(env.A @ obs + env.B @ clipped_action, env.x_min, env.x_max)
    np.testing.assert_allclose(next_obs, expected, atol=1e-5)


def test_collision_at_wall_boundary():
    """Wall contact is a fully inelastic bounce, not a terminal event: p_y is
    clamped to the wall, v_y is absorbed to zero, and the episode continues
    with a per-step penalty instead of ending."""
    env = make(sigma_p=0.0, sigma_v=0.0, tunnel_width=4.0)
    env.reset(seed=0)
    env.state = np.array([1.0, env.W / 2.0 - 1e-4, 0.0, env.v_max], dtype=np.float32)
    action = np.array([0.0, env.u_max], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is False
    assert info["is_success"] is False
    assert info["collision"] is True
    assert info["collision_count"] == 1
    assert len(info["collision_impacts"]) == 1

    assert obs[1] == pytest.approx(env.W / 2.0)  # clamped to the wall
    assert obs[3] == 0.0  # v_y absorbed by the inelastic bounce

    expected = env.config.step_penalty + env.config.contact_penalty
    assert reward == pytest.approx(expected)
    assert info["collision_impacts"][0] == pytest.approx(expected - env.config.step_penalty)


def test_goal_reached_at_boundary():
    env = make(sigma_p=0.0, sigma_v=0.0, tunnel_length=10.0)
    env.reset(seed=0)
    p_x_prev = 10.0 - 1e-4
    env.state = np.array([p_x_prev, 0.0, env.v_max, 0.0], dtype=np.float32)
    action = np.array([env.u_max, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated is True
    # goal_reward plus the same progress-shaping term every other step gets,
    # on the actual (post-clip) p_x delta -- not a bare goal_reward compare.
    expected = env.config.goal_reward + env.config.progress_reward_coef * (obs[0] - p_x_prev)
    assert reward == pytest.approx(expected)
    assert info["is_success"] is True
    assert info["collision"] is False


def test_truncation_at_max_steps():
    env = make(sigma_p=0.0, sigma_v=0.0, max_steps=3, tunnel_length=1000.0, tunnel_width=1000.0)
    env.reset(seed=0)
    action = np.zeros(2, dtype=np.float32)
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(action)
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(action)
    assert not terminated
    assert truncated is True


def test_observation_stays_within_space_after_out_of_bounds_step():
    env = make(sigma_p=0.0, sigma_v=0.0)
    env.reset(seed=0)
    env.state = np.array([env.L, 0.0, env.v_max, env.v_max], dtype=np.float32)
    action = np.array([env.u_max, env.u_max], dtype=np.float32)
    obs, *_ = env.step(action)
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"tunnel_length": -1.0},
        {"tunnel_width": 0.0},
        {"v_max": -1.0},
        {"u_max": 0.0},
        {"sigma_p": -0.1},
        {"sigma_v": -0.1},
        {"max_steps": 0},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        TunnelEnvConfig(**kwargs)


def test_default_config_pins_current_tuned_values():
    """Pins the reward-shaping defaults from the stall-optimum fix (see
    config.py's Historical note): a change here should be a deliberate
    retune, not silent drift."""
    config = TunnelEnvConfig()
    assert config.dt == 0.1
    assert config.tunnel_length == 10.0
    assert config.tunnel_width == 4.0
    assert config.v_max == 1.2
    assert config.u_max == 2.5
    assert config.sigma_p == 0.0
    assert config.sigma_v == 0.0
    assert config.max_steps == 200
    assert config.step_penalty == -1.0
    assert config.contact_penalty == -50.0
    assert config.goal_reward == 200.0
    assert config.progress_reward_coef == 10.0


def test_overrides_replace_config_fields():
    env = TunnelEnv(tunnel_length=20.0, sigma_p=0.5)
    assert env.L == 20.0
    assert env.sigma_p == 0.5
    assert env.config.tunnel_width == 4.0  # untouched default


def test_default_config_has_no_width_profile():
    assert TunnelEnvConfig().width_profile is None


def test_env_derives_constant_profile_when_unset():
    env = make()
    y_lo, y_hi = env.width_profile.envelope()
    assert (y_lo, y_hi) == (-env.W / 2.0, env.W / 2.0)


def test_replace_config_rederives_width_profile_not_a_stale_one():
    """The one correctness trap width_profile's docstring calls out: TunnelEnv
    applies `overrides` via `dataclasses.replace(config, **overrides)`, which
    resupplies every *current* field as a kwarg to the new config. If
    width_profile were baked in eagerly at TunnelEnvConfig construction, a
    later override of tunnel_width would silently carry the stale profile
    through instead of one matching the new width."""
    env = TunnelEnv(tunnel_width=10.0)
    y_lo, y_hi = env.width_profile.envelope()
    assert (y_lo, y_hi) == (-5.0, 5.0)


def _gate_profile():
    """A minimal multi-segment profile with one narrow, offset gate -- just
    enough to exercise position-dependent bounds."""
    return WidthProfile([
        WidthSegment(-np.inf, 2.0, 2.0, 0.0),
        WidthSegment(2.0, 5.0, 0.75, 1.0),
        WidthSegment(5.0, np.inf, 2.0, 0.0),
    ])


def test_collision_is_position_dependent_under_a_width_profile():
    profile = _gate_profile()

    # y=1.9 is safe under the global envelope (+-2.0), but at x=3.0 -- inside
    # the gate, x in (2.0, 5.0], opening y in [0.25, 1.75] -- it is a wall touch.
    env = make(sigma_p=0.0, sigma_v=0.0, width_profile=profile)
    env.reset(seed=0)
    env.state = np.array([3.0, 1.9, 0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is False
    assert info["collision"] is True
    y_lo, y_hi = profile.bounds_at(3.0)
    assert obs[1] == pytest.approx(y_hi)  # clamped to the gate's upper edge
    assert obs[3] == 0.0
    expected = env.config.step_penalty + env.config.contact_penalty
    assert reward == pytest.approx(expected)

    # The identical p_y in a full-width segment (x=0.0) is safe.
    env2 = make(sigma_p=0.0, sigma_v=0.0, width_profile=profile)
    env2.reset(seed=0)
    env2.state = np.array([0.0, 1.9, 0.0, 0.0], dtype=np.float32)
    _, _, terminated2, _, info2 = env2.step(np.zeros(2, dtype=np.float32))
    assert terminated2 is False
    assert info2["collision"] is False


def test_observation_space_matches_profile_envelope():
    profile = _gate_profile()
    env = make(width_profile=profile)
    y_lo, y_hi = profile.envelope()
    assert env.observation_space.low[1] == pytest.approx(y_lo)
    assert env.observation_space.high[1] == pytest.approx(y_hi)


def test_progress_reward_coef_zero_is_noop():
    """progress_reward_coef=0.0 must not perturb the reward even when p_x
    actually changes step to step -- the shaping term is additive and must
    vanish cleanly when disabled. Passed explicitly since the default is no
    longer 0.0 (see config.py's Historical note)."""
    env = make(sigma_p=0.0, sigma_v=0.0, progress_reward_coef=0.0)
    env.reset(seed=0)
    action = np.array([env.u_max, 0.0], dtype=np.float32)
    _, reward, terminated, truncated, _ = env.step(action)
    assert not terminated and not truncated
    assert reward == env.config.step_penalty


def test_progress_shaping_rewards_forward_and_penalizes_backward_motion():
    coef = 2.0

    env = make(sigma_p=0.0, sigma_v=0.0, progress_reward_coef=coef)
    env.reset(seed=0)
    env.state = np.array([5.0, 0.0, 1.0, 0.0], dtype=np.float32)
    p_x_prev = float(env.state[0])
    next_obs, reward, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    assert not terminated and not truncated
    assert next_obs[0] > p_x_prev
    expected = env.config.step_penalty + coef * (next_obs[0] - p_x_prev)
    assert reward == pytest.approx(expected)
    assert reward > env.config.step_penalty

    env2 = make(sigma_p=0.0, sigma_v=0.0, progress_reward_coef=coef)
    env2.reset(seed=0)
    env2.state = np.array([5.0, 0.0, -1.0, 0.0], dtype=np.float32)
    p_x_prev2 = float(env2.state[0])
    next_obs2, reward2, terminated2, truncated2, _ = env2.step(np.zeros(2, dtype=np.float32))
    assert not terminated2 and not truncated2
    assert next_obs2[0] < p_x_prev2
    expected2 = env2.config.step_penalty + coef * (next_obs2[0] - p_x_prev2)
    assert reward2 == pytest.approx(expected2)
    assert reward2 < env2.config.step_penalty


def test_progress_shaping_additive_on_collision():
    coef = 3.0
    env = make(sigma_p=0.0, sigma_v=0.0, tunnel_width=4.0, progress_reward_coef=coef)
    env.reset(seed=0)
    p_x_prev = 1.0
    env.state = np.array([p_x_prev, env.W / 2.0 - 1e-4, 0.0, env.v_max], dtype=np.float32)
    action = np.array([0.0, env.u_max], dtype=np.float32)
    next_obs, reward, terminated, _, info = env.step(action)
    assert terminated is False
    assert info["collision"] is True
    expected = (
        env.config.step_penalty
        + env.config.contact_penalty
        + coef * (next_obs[0] - p_x_prev)
    )
    assert reward == pytest.approx(expected)


def test_progress_shaping_additive_on_goal():
    coef = 3.0
    env = make(sigma_p=0.0, sigma_v=0.0, tunnel_length=10.0, progress_reward_coef=coef)
    env.reset(seed=0)
    p_x_prev = 10.0 - 1e-4
    env.state = np.array([p_x_prev, 0.0, env.v_max, 0.0], dtype=np.float32)
    action = np.array([env.u_max, 0.0], dtype=np.float32)
    next_obs, reward, terminated, _, info = env.step(action)
    assert terminated is True
    assert info["is_success"] is True
    expected = env.config.goal_reward + coef * (next_obs[0] - p_x_prev)
    assert reward == pytest.approx(expected)


def test_progress_shaping_telescopes_over_episode():
    """The sum of per-step shaping increments over a multi-step rollout must
    equal coef * (p_x_final - p_x_initial), regardless of the path taken
    step to step -- the defining property of potential-based shaping."""
    coef = 1.5
    env = make(
        sigma_p=0.0, sigma_v=0.0, max_steps=3,
        tunnel_length=1000.0, tunnel_width=1000.0,
        progress_reward_coef=coef,
    )
    obs0, _ = env.reset(seed=0)
    p_x_initial = float(obs0[0])
    action = np.array([0.3, 0.0], dtype=np.float32)

    total_shaping = 0.0
    p_x_final = p_x_initial
    for _ in range(3):
        next_obs, reward, terminated, truncated, _ = env.step(action)
        assert not terminated
        total_shaping += reward - env.config.step_penalty
        p_x_final = float(next_obs[0])

    expected_total = coef * (p_x_final - p_x_initial)
    assert total_shaping == pytest.approx(expected_total, rel=1e-4)


def test_collision_penalty_matches_contact_formula():
    """Direct check of the per-contact reward formula: step_penalty +
    contact_penalty (flat, regardless of impact speed), plus progress
    shaping on top. No more refund mechanism -- see
    test_collision_penalty_independent_of_timestep for the property that
    used to require it."""
    coef = 3.0
    env = make(
        sigma_p=0.0, sigma_v=0.0, tunnel_width=4.0,
        progress_reward_coef=coef,
    )
    env.reset(seed=0)
    env.state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # A few safe steps first, purely to exercise that the contact formula
    # does not depend on how many steps preceded it.
    safe_action = np.zeros(2, dtype=np.float32)
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(safe_action)
        assert not terminated and not truncated
    assert env.steps == 3

    p_x_prev = float(env.state[0])
    env.state = np.array([p_x_prev, env.W / 2.0 - 1e-4, 0.0, 1.5], dtype=np.float32)
    next_obs, reward, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is False
    assert info["collision"] is True
    assert env.steps == 4

    expected = (
        env.config.step_penalty
        + env.config.contact_penalty
        + coef * (next_obs[0] - p_x_prev)
    )
    assert reward == pytest.approx(expected)


def test_collision_penalty_independent_of_timestep():
    """The reward contribution of a single contact (reward minus step_penalty
    minus progress shaping) must be identical whether the contact happens on
    the first step or after several safe steps. True by construction now
    that a contact is additive rather than a refunded terminal event, but
    worth locking in as an explicit regression test."""
    common = dict(sigma_p=0.0, sigma_v=0.0, tunnel_width=4.0, progress_reward_coef=0.0)

    env_early = make(**common)
    env_early.reset(seed=0)
    env_early.state = np.array([1.0, env_early.W / 2.0 - 1e-4, 0.0, 1.5], dtype=np.float32)
    _, reward_early, terminated_early, _, info_early = env_early.step(np.zeros(2, dtype=np.float32))
    assert not terminated_early and info_early["collision"]
    contact_delta_early = reward_early - env_early.config.step_penalty

    env_late = make(**common)
    env_late.reset(seed=0)
    env_late.state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    safe_action = np.zeros(2, dtype=np.float32)
    for _ in range(10):
        _, _, terminated, truncated, _ = env_late.step(safe_action)
        assert not terminated and not truncated
    env_late.state = np.array([float(env_late.state[0]), env_late.W / 2.0 - 1e-4, 0.0, 1.5], dtype=np.float32)
    _, reward_late, terminated_late, _, info_late = env_late.step(np.zeros(2, dtype=np.float32))
    assert not terminated_late and info_late["collision"]
    contact_delta_late = reward_late - env_late.config.step_penalty

    assert contact_delta_late == pytest.approx(contact_delta_early)


def test_collision_penalty_independent_of_impact_speed():
    """The contact penalty must be flat regardless of |v_y| at the moment of
    impact -- unlike the old impact_penalty_coef design, a hard hit and a
    soft graze cost exactly the same."""
    common = dict(sigma_p=0.0, sigma_v=0.0, tunnel_width=4.0, progress_reward_coef=0.0)

    env_slow = make(**common)
    env_slow.reset(seed=0)
    env_slow.state = np.array([1.0, env_slow.W / 2.0 - 1e-4, 0.0, 0.5], dtype=np.float32)
    _, reward_slow, terminated_slow, _, info_slow = env_slow.step(np.zeros(2, dtype=np.float32))
    assert not terminated_slow and info_slow["collision"]

    env_fast = make(**common)
    env_fast.reset(seed=0)
    env_fast.state = np.array([1.0, env_fast.W / 2.0 - 1e-4, 0.0, env_fast.v_max], dtype=np.float32)
    _, reward_fast, terminated_fast, _, info_fast = env_fast.step(np.zeros(2, dtype=np.float32))
    assert not terminated_fast and info_fast["collision"]

    assert reward_fast == pytest.approx(reward_slow)


def test_collision_count_and_impacts_accumulate_over_episode():
    """collision_count and collision_impacts must track *every* contact in
    the episode, not just the most recent one -- and collision_impacts holds
    each contact's own penalty, not a running cumulative sum."""
    env = make(sigma_p=0.0, sigma_v=0.0, tunnel_width=4.0, max_steps=5)
    env.reset(seed=0)
    action = np.array([0.0, env.u_max], dtype=np.float32)

    env.state = np.array([1.0, env.W / 2.0 - 1e-4, 0.0, 1.0], dtype=np.float32)
    _, _, _, _, info1 = env.step(action)
    assert info1["collision_count"] == 1
    assert len(info1["collision_impacts"]) == 1

    # v_y was zeroed by the first bounce -- nudge the agent back into the
    # wall for a second, independent contact.
    env.state[1] = env.W / 2.0 - 1e-4
    env.state[3] = 1.0
    _, _, _, _, info2 = env.step(action)
    assert info2["collision_count"] == 2
    assert len(info2["collision_impacts"]) == 2
    # Individual, not cumulative: the first entry is unchanged by the second contact.
    assert info2["collision_impacts"][0] == pytest.approx(info1["collision_impacts"][0])


def test_collision_bookkeeping_resets_on_reset():
    env = make(sigma_p=0.0, sigma_v=0.0, tunnel_width=4.0)
    env.reset(seed=0)
    env.state = np.array([1.0, env.W / 2.0 - 1e-4, 0.0, 1.0], dtype=np.float32)
    _, _, _, _, info = env.step(np.array([0.0, env.u_max], dtype=np.float32))
    assert info["collision_count"] == 1

    _, info2 = env.reset(seed=1)
    assert info2["collision_count"] == 0
    assert info2["collision_impacts"] == []


def test_render_rgb_array_returns_frame():
    env = TunnelEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None
    assert frame.ndim == 3 and frame.shape[2] == 3
    env.close()


def test_a_step_at_top_speed_covers_v_max_dt_however_hard_the_thrust():
    """The speed limit acts on the delivered acceleration. Before 2026-09-24
    it was a clip of the state after the full LTI update, so thrusting at top
    speed covered v_max*dt + u_max*dt**2/2 per step, 10% more, and trained
    agents used it to beat the min-time oracle (see SlalomEnv.step and TunnelEnv.step)."""
    env = make(sigma_p=0.0, sigma_v=0.0)
    env.reset(seed=0)
    for u in (0.0, env.u_max):
        env.state = np.array([5.0, 0.0, env.v_max, -env.v_max], dtype=np.float32)
        obs, *_ = env.step(np.array([u, -u], dtype=np.float32))
        assert obs[0] - 5.0 == pytest.approx(env.v_max * env.dt, abs=1e-5)
        assert obs[1] == pytest.approx(-env.v_max * env.dt, abs=1e-5)
        assert obs[2] == pytest.approx(env.v_max) and obs[3] == pytest.approx(-env.v_max)


def test_reaching_the_speed_limit_gives_the_in_bounds_actions_state():
    """A step that hits the limit lands exactly where the in-bounds action
    (v_max - v) / dt lands: the plant the oracle and MPCWorker model."""
    env = make(sigma_p=0.0, sigma_v=0.0)
    env.reset(seed=0)
    start = np.array([5.0, 0.0, 1.0, -1.1], dtype=np.float32)
    env.state = start.copy()
    obs, *_ = env.step(np.array([env.u_max, -env.u_max], dtype=np.float32))
    in_bounds = np.array([(env.v_max - 1.0) / env.dt, (-env.v_max + 1.1) / env.dt], dtype=np.float32)
    np.testing.assert_allclose(obs, env.A @ start + env.B @ in_bounds, atol=1e-5)
