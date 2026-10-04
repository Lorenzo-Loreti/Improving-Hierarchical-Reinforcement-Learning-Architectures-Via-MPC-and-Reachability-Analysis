import numpy as np
import pytest

from envs.config import SlalomEnvConfig
from envs.slalom_env import SlalomEnv
from envs.width_profile import slalom_profile


def make(**overrides):
    return SlalomEnv(**overrides)


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
    env = make(noise_bound_p=0.0, noise_bound_v=0.0)
    obs, _ = env.reset(seed=0)
    action = np.zeros(2, dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)
    expected = env.A @ obs
    np.testing.assert_allclose(next_obs, expected, atol=1e-5)
    assert reward == env.config.step_penalty
    assert not terminated and not truncated


def test_step_dynamics_matches_A_B_with_action():
    env = make(noise_bound_p=0.0, noise_bound_v=0.0)
    obs, _ = env.reset(seed=0)
    action = np.array([0.5, -0.3], dtype=np.float32)
    next_obs, *_ = env.step(action)
    expected = env.A @ obs + env.B @ action
    expected = np.clip(expected, env.x_min, env.x_max)
    np.testing.assert_allclose(next_obs, expected, atol=1e-5)


def test_action_is_saturated_radially_before_applied():
    """The thrust limit is ||u|| <= u_max (envs/actuation.py): an oversized
    command keeps its direction and is scaled to the disk's edge -- not
    clipped per axis, which would turn (5, 1) toward the diagonal and let
    (5, -5) through at sqrt(2) u_max."""
    for command in ([5.0, -5.0], [5.0, 1.0], [0.0, -3.0]):
        env = make(noise_bound_p=0.0, noise_bound_v=0.0, u_max=1.0)
        obs, _ = env.reset(seed=0)
        command = np.array(command, dtype=np.float32)
        next_obs, *_ = env.step(command)
        saturated = command / np.linalg.norm(command) * env.u_max
        np.testing.assert_allclose(next_obs, env.A @ obs + env.B @ saturated, atol=1e-5)


def test_collision_at_wall_boundary():
    """Wall contact is a fully inelastic bounce, not a terminal event: p_y is
    clamped to the wall, v_y is absorbed to zero, and the episode continues
    with a per-step penalty instead of ending."""
    env = make(noise_bound_p=0.0, noise_bound_v=0.0, tunnel_width=4.0)
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
    env = make(noise_bound_p=0.0, noise_bound_v=0.0, tunnel_length=10.0)
    env.reset(seed=0)
    p_x_prev = 10.0 - 1e-4
    env.state = np.array([p_x_prev, 0.0, env.v_max, 0.0], dtype=np.float32)
    action = np.array([env.u_max, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated is True
    # goal_reward plus the same progress-shaping term every other step gets,
    # up to the goal line where the potential is cut -- not a bare
    # goal_reward compare. Thrusting along v at top speed delivers nothing,
    # so no effort is charged.
    assert info["effort_penalty"] == 0.0
    expected = env.config.goal_reward + env.config.progress_reward_coef * (env.L - p_x_prev)
    assert reward == pytest.approx(expected)
    assert info["is_success"] is True
    assert info["collision"] is False


def test_truncation_at_max_steps():
    env = make(noise_bound_p=0.0, noise_bound_v=0.0, max_steps=3, tunnel_length=1000.0, tunnel_width=1000.0)
    env.reset(seed=0)
    action = np.zeros(2, dtype=np.float32)
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(action)
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(action)
    assert not terminated
    assert truncated is True


def test_observation_stays_within_space_after_out_of_bounds_step():
    env = make(noise_bound_p=0.0, noise_bound_v=0.0)
    env.reset(seed=0)
    diagonal = env.v_max / np.sqrt(2.0)
    env.state = np.array([env.L, 0.0, diagonal, diagonal], dtype=np.float32)
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
        {"noise_bound_p": -0.1},
        {"noise_bound_v": -0.1},
        {"max_steps": 0},
        {"effort_penalty": 0.01},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        SlalomEnvConfig(**kwargs)


def test_default_config_pins_current_tuned_values():
    """Pins the reward-shaping defaults from the stall-optimum fix (see
    config.py's Historical note): a change here should be a deliberate
    retune, not silent drift."""
    config = SlalomEnvConfig()
    assert config.dt == 0.1
    assert config.tunnel_length == 10.0
    assert config.tunnel_width == 4.0
    assert config.v_max == 1.2
    assert config.u_max == 2.5
    assert config.noise_bound_p == 0.0
    assert config.noise_bound_v == 0.0
    assert config.max_steps == 200
    assert config.step_penalty == -1.0
    assert config.contact_penalty == -50.0
    assert config.goal_reward == 1000.0
    assert config.progress_reward_coef == 10.0
    assert config.effort_penalty == -0.01


def test_overrides_replace_config_fields():
    env = SlalomEnv(tunnel_length=20.0, noise_bound_p=0.5)
    assert env.L == 20.0
    assert env.noise_bound_p == 0.5
    assert env.config.tunnel_width == 4.0  # untouched default


def test_default_config_has_no_width_profile():
    assert SlalomEnvConfig().width_profile is None


def test_env_derives_constant_profile_when_unset():
    env = make()
    y_lo, y_hi = env.width_profile.envelope()
    assert (y_lo, y_hi) == (-env.W / 2.0, env.W / 2.0)


def test_replace_config_rederives_width_profile_not_a_stale_one():
    """The one correctness trap width_profile's docstring calls out: SlalomEnv
    applies `overrides` via `dataclasses.replace(config, **overrides)`, which
    resupplies every *current* field as a kwarg to the new config. If
    width_profile were baked in eagerly at SlalomEnvConfig construction, a
    later override of tunnel_width would silently carry the stale profile
    through instead of one matching the new width."""
    env = SlalomEnv(tunnel_width=10.0)
    y_lo, y_hi = env.width_profile.envelope()
    assert (y_lo, y_hi) == (-5.0, 5.0)


def test_collision_is_position_dependent_under_a_width_profile():
    profile = slalom_profile()

    # y=1.9 is safe under the global envelope (+-2.0), but at x=4.5 -- inside
    # gate 1, x in (4.0, 5.0], opening y in [0.25, 1.75] -- it is a wall touch.
    env = make(noise_bound_p=0.0, noise_bound_v=0.0, width_profile=profile)
    env.reset(seed=0)
    env.state = np.array([4.5, 1.9, 0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is False
    assert info["collision"] is True
    y_lo, y_hi = profile.bounds_at(4.5)
    assert obs[1] == pytest.approx(y_hi)  # clamped to gate 1's upper edge
    assert obs[3] == 0.0
    expected = env.config.step_penalty + env.config.contact_penalty
    assert reward == pytest.approx(expected)

    # The identical p_y in a full-width segment (x=0.0) is safe.
    env2 = make(noise_bound_p=0.0, noise_bound_v=0.0, width_profile=profile)
    env2.reset(seed=0)
    env2.state = np.array([0.0, 1.9, 0.0, 0.0], dtype=np.float32)
    _, _, terminated2, _, info2 = env2.step(np.zeros(2, dtype=np.float32))
    assert terminated2 is False
    assert info2["collision"] is False


def test_observation_space_matches_profile_envelope():
    profile = slalom_profile()
    env = make(width_profile=profile)
    y_lo, y_hi = profile.envelope()
    assert env.observation_space.low[1] == pytest.approx(y_lo)
    assert env.observation_space.high[1] == pytest.approx(y_hi)


def test_progress_reward_coef_zero_is_noop():
    """progress_reward_coef=0.0 must not perturb the reward even when p_x
    actually changes step to step -- the shaping term is additive and must
    vanish cleanly when disabled. Passed explicitly since the default is no
    longer 0.0 (see config.py's Historical note)."""
    env = make(noise_bound_p=0.0, noise_bound_v=0.0, progress_reward_coef=0.0, effort_penalty=0.0)
    env.reset(seed=0)
    action = np.array([env.u_max, 0.0], dtype=np.float32)
    _, reward, terminated, truncated, _ = env.step(action)
    assert not terminated and not truncated
    assert reward == env.config.step_penalty


def test_progress_shaping_rewards_forward_and_penalizes_backward_motion():
    coef = 2.0

    env = make(noise_bound_p=0.0, noise_bound_v=0.0, progress_reward_coef=coef)
    env.reset(seed=0)
    env.state = np.array([5.0, 0.0, 1.0, 0.0], dtype=np.float32)
    p_x_prev = float(env.state[0])
    next_obs, reward, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    assert not terminated and not truncated
    assert next_obs[0] > p_x_prev
    expected = env.config.step_penalty + coef * (next_obs[0] - p_x_prev)
    assert reward == pytest.approx(expected)
    assert reward > env.config.step_penalty

    env2 = make(noise_bound_p=0.0, noise_bound_v=0.0, progress_reward_coef=coef)
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
    env = make(noise_bound_p=0.0, noise_bound_v=0.0, tunnel_width=4.0, progress_reward_coef=coef)
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
    env = make(noise_bound_p=0.0, noise_bound_v=0.0, tunnel_length=10.0, progress_reward_coef=coef)
    env.reset(seed=0)
    p_x_prev = 10.0 - 1e-4
    env.state = np.array([p_x_prev, 0.0, env.v_max, 0.0], dtype=np.float32)
    action = np.array([env.u_max, 0.0], dtype=np.float32)
    next_obs, reward, terminated, _, info = env.step(action)
    assert terminated is True
    assert info["is_success"] is True
    assert next_obs[0] > env.L
    expected = env.config.goal_reward + coef * (env.L - p_x_prev)
    assert reward == pytest.approx(expected)


def test_progress_is_cut_at_the_goal_line():
    """The crossing step's progress is paid up to L, however far past it the
    vehicle lands: the return of a successful episode depends on its length,
    effort and contacts alone, as the oracle assumes (see the config's
    progress_reward_coef). Two crossings from the same state at different
    speeds land at different p_x and earn the same progress."""
    rewards = []
    for v_x in (0.6, 1.2):
        env = make(noise_bound_p=0.0, noise_bound_v=0.0, effort_penalty=0.0)
        env.reset(seed=0)
        env.state = np.array([env.L - 0.05, 0.0, v_x, 0.0], dtype=np.float32)
        obs, reward, terminated, *_ = env.step(np.zeros(2, dtype=np.float32))
        assert terminated and obs[0] > env.L
        rewards.append(reward)
    assert rewards[0] == pytest.approx(rewards[1])
    assert rewards[0] == pytest.approx(env.config.goal_reward + env.config.progress_reward_coef * 0.05, abs=1e-4)


def test_progress_shaping_telescopes_over_episode():
    """The sum of per-step shaping increments over a multi-step rollout must
    equal coef * (p_x_final - p_x_initial), regardless of the path taken
    step to step -- the defining property of potential-based shaping."""
    coef = 1.5
    env = make(
        noise_bound_p=0.0, noise_bound_v=0.0, max_steps=3,
        tunnel_length=1000.0, tunnel_width=1000.0,
        progress_reward_coef=coef, effort_penalty=0.0,
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
        noise_bound_p=0.0, noise_bound_v=0.0, tunnel_width=4.0,
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
    env.state = np.array([p_x_prev, env.W / 2.0 - 1e-4, 0.0, 1.0], dtype=np.float32)
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
    common = dict(noise_bound_p=0.0, noise_bound_v=0.0, tunnel_width=4.0, progress_reward_coef=0.0)

    env_early = make(**common)
    env_early.reset(seed=0)
    env_early.state = np.array([1.0, env_early.W / 2.0 - 1e-4, 0.0, 1.0], dtype=np.float32)
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
    env_late.state = np.array([float(env_late.state[0]), env_late.W / 2.0 - 1e-4, 0.0, 1.0], dtype=np.float32)
    _, reward_late, terminated_late, _, info_late = env_late.step(np.zeros(2, dtype=np.float32))
    assert not terminated_late and info_late["collision"]
    contact_delta_late = reward_late - env_late.config.step_penalty

    assert contact_delta_late == pytest.approx(contact_delta_early)


def test_collision_penalty_independent_of_impact_speed():
    """The contact penalty must be flat regardless of |v_y| at the moment of
    impact -- unlike the old impact_penalty_coef design, a hard hit and a
    soft graze cost exactly the same."""
    common = dict(noise_bound_p=0.0, noise_bound_v=0.0, tunnel_width=4.0, progress_reward_coef=0.0)

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
    env = make(noise_bound_p=0.0, noise_bound_v=0.0, tunnel_width=4.0, max_steps=5)
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
    env = make(noise_bound_p=0.0, noise_bound_v=0.0, tunnel_width=4.0)
    env.reset(seed=0)
    env.state = np.array([1.0, env.W / 2.0 - 1e-4, 0.0, 1.0], dtype=np.float32)
    _, _, _, _, info = env.step(np.array([0.0, env.u_max], dtype=np.float32))
    assert info["collision_count"] == 1

    _, info2 = env.reset(seed=1)
    assert info2["collision_count"] == 0
    assert info2["collision_impacts"] == []


def test_render_rgb_array_returns_frame():
    env = SlalomEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None
    assert frame.ndim == 3 and frame.shape[2] == 3
    env.close()


def test_a_step_at_top_speed_covers_v_max_dt_however_hard_the_thrust():
    """The speed limit acts on the delivered acceleration. Before 2026-09-24
    it was a clip of the state after the full LTI update, so thrusting at top
    speed covered v_max*dt + u_max*dt**2/2 per step, 10% more, and trained
    agents used it to beat the min-time oracle (see envs/actuation.py). Since
    2026-10-04 the limit is on ||v||, so this holds for a heading off the
    axes too, and for a thrust in any direction."""
    heading = np.array([np.cos(-0.5), np.sin(-0.5)])
    for angle in np.linspace(0.0, 2.0 * np.pi, 13):
        env = make(noise_bound_p=0.0, noise_bound_v=0.0)
        env.reset(seed=0)
        env.state = np.array([5.0, 0.0, *(env.v_max * heading)], dtype=np.float32)
        thrust = env.u_max * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        obs, *_ = env.step(thrust)
        assert np.linalg.norm(obs[:2] - [5.0, 0.0]) <= env.v_max * env.dt + 1e-6
        assert np.linalg.norm(obs[2:]) <= env.v_max + 1e-6
    # Thrust along the heading: exactly v_max * dt along it, at v_max.
    env = make(noise_bound_p=0.0, noise_bound_v=0.0)
    env.reset(seed=0)
    env.state = np.array([5.0, 0.0, *(env.v_max * heading)], dtype=np.float32)
    obs, _, _, _, info = env.step((env.u_max * heading).astype(np.float32))
    np.testing.assert_allclose(obs[:2] - [5.0, 0.0], env.v_max * env.dt * heading, atol=1e-5)
    np.testing.assert_allclose(obs[2:], env.v_max * heading, atol=1e-5)
    assert info["effort_penalty"] == pytest.approx(0.0, abs=1e-7)


def test_steering_at_top_speed_costs_forward_speed():
    """What the per-axis limits got wrong: there, a lateral thrust at
    v_x = v_max left v_x alone, so steering was free. On the speed disk the
    velocity turns but its magnitude stays at v_max, so v_x drops."""
    env = make(noise_bound_p=0.0, noise_bound_v=0.0)
    env.reset(seed=0)
    env.state = np.array([5.0, 0.0, env.v_max, 0.0], dtype=np.float32)
    obs, *_ = env.step(np.array([0.0, env.u_max], dtype=np.float32))
    assert np.linalg.norm(obs[2:]) == pytest.approx(env.v_max, abs=1e-6)
    assert obs[3] > 0.2
    assert obs[2] < env.v_max - 0.01


def test_reaching_the_speed_limit_gives_the_in_bounds_actions_state():
    """A step that hits the limit lands exactly where the in-bounds action
    (v' - v) / dt lands, v' the projection of v + u dt onto the speed disk --
    and that action is admissible, ||u|| <= u_max: the plant the oracle and
    the tube MPC model (envs/actuation.py)."""
    env = make(noise_bound_p=0.0, noise_bound_v=0.0)
    env.reset(seed=0)
    start = np.array([5.0, 0.0, 1.0, -0.5], dtype=np.float32)
    env.state = start.copy()
    command = np.array([env.u_max, -env.u_max], dtype=np.float32)
    obs, *_ = env.step(command)
    saturated = command / np.linalg.norm(command) * env.u_max
    v_next = start[2:] + saturated * env.dt
    assert np.linalg.norm(v_next) > env.v_max
    in_bounds = (v_next / np.linalg.norm(v_next) * env.v_max - start[2:]) / env.dt
    assert np.linalg.norm(in_bounds) <= env.u_max
    np.testing.assert_allclose(obs, env.A @ start + env.B @ in_bounds, atol=1e-5)


def test_delivered_input_is_admissible_and_untouched_inside_the_limits():
    """envs/actuation.deliver over random velocities in the speed disk and
    random commands, many far outside the thrust disk: the delivered u is
    always admissible (||u|| <= u_max, ||v + u dt|| <= v_max, up to float32
    rounding), and a command that breaks neither limit comes back bit for
    bit."""
    from envs.actuation import deliver
    env = make()
    rng = np.random.default_rng(0)
    n = 20000
    radius = env.v_max * np.sqrt(rng.uniform(size=n))
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    v = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1).astype(np.float32)
    u = rng.uniform(-3.0 * env.u_max, 3.0 * env.u_max, size=(n, 2)).astype(np.float32)
    delivered = deliver(u, v, env.u_max, env.v_max, env.dt)
    assert np.all(np.linalg.norm(delivered, axis=1) <= env.u_max * (1 + 1e-5))
    assert np.all(np.linalg.norm(v + delivered * np.float32(env.dt), axis=1) <= env.v_max * (1 + 1e-6))
    inside = (np.linalg.norm(u, axis=1) <= env.u_max) & \
             (np.linalg.norm(v + u * np.float32(env.dt), axis=1) <= env.v_max)
    assert inside.sum() > 100
    np.testing.assert_array_equal(delivered[inside], u[inside])


def test_effort_penalty_is_charged_on_the_delivered_input():
    """effort_penalty * (||u|| / u_max)^2 per step, on what the plant
    delivered: a full-thrust step from rest costs effort_penalty, whatever
    its direction or how far past u_max the command was; half thrust costs a
    quarter of it; and a thrust the speed limit absorbs entirely costs
    nothing. It adds to the rest of the reward."""
    env = make(noise_bound_p=0.0, noise_bound_v=0.0, tunnel_width=1000.0)
    c = env.config.effort_penalty
    for command, expected in (([env.u_max, 0.0], c), ([0.0, -0.5 * env.u_max], 0.25 * c),
                              ([0.6 * env.u_max, 0.8 * env.u_max], c), ([4.0 * env.u_max, 0.0], c)):
        env.reset(seed=0, options={"init_state": np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32)})
        obs, reward, _, _, info = env.step(np.array(command, dtype=np.float32))
        assert info["effort_penalty"] == pytest.approx(expected, rel=1e-5)
        progress = env.config.progress_reward_coef * (obs[0] - 3.0)
        assert reward == pytest.approx(env.config.step_penalty + progress + expected, abs=1e-5)
    env.reset(seed=0, options={"init_state": np.array([3.0, 0.0, env.v_max, 0.0], dtype=np.float32)})
    _, _, _, _, info = env.step(np.array([env.u_max, 0.0], dtype=np.float32))
    assert info["effort_penalty"] == 0.0


def test_noise_is_uniform_on_the_bounded_disks():
    """The disturbance is drawn uniformly from W = {||w_p|| <= noise_bound_p,
    ||w_v|| <= noise_bound_v} (see the config's noise_bound_p): every step's
    w stays inside both disks, fills them out to the edge off the axes too
    (isotropic, not a box), and is uniform in area -- a quarter of the draws
    inside half the radius."""
    bound_p, bound_v = 0.01, 0.05
    env = make(noise_bound_p=bound_p, noise_bound_v=bound_v, tunnel_length=1000.0, max_steps=10000)
    start = np.array([1.0, 0.0, 0.3, 0.0], dtype=np.float32)
    env.reset(seed=0, options={"init_state": start})
    ws = []
    for _ in range(4000):
        # Restarted from the same mid-corridor state every step, so no
        # random walk ever reaches a wall or the speed limit, whose guards
        # would be mistaken for the disturbance.
        env.state = start.copy()
        predicted = env.A.astype(np.float64) @ env.state.astype(np.float64)
        env.step(np.zeros(2, dtype=np.float32))
        ws.append(env.state.astype(np.float64) - predicted)
    ws = np.array(ws)
    for block, bound in ((ws[:, :2], bound_p), (ws[:, 2:], bound_v)):
        radius = np.linalg.norm(block, axis=1)
        assert np.all(radius <= bound * (1 + 1e-4) + 1e-6)
        assert radius.max() > 0.98 * bound
        assert 0.22 < np.mean(radius <= bound / 2) < 0.28
        diagonal = np.abs(block @ np.array([1.0, 1.0]) / np.sqrt(2.0))
        assert diagonal.max() > 0.95 * bound


def test_zero_noise_draws_no_randomness():
    """With W = {0} (the default) step() consumes nothing from the env's
    generator, which is what keeps every run from before the bounded
    disturbance existed bit-identical."""
    env = make()
    env.reset(seed=3)
    rng_state = env.np_random.bit_generator.state
    env.step(np.array([1.0, -1.0], dtype=np.float32))
    assert env.np_random.bit_generator.state == rng_state
