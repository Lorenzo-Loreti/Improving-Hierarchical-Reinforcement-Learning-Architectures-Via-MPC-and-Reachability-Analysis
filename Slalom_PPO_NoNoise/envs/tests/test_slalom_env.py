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
        assert info["distance_to_goal"] == pytest.approx(env.L - p_x)


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
    env = make(sigma_p=0.0, sigma_v=0.0, tunnel_width=4.0)
    env.reset(seed=0)
    env.state = np.array([1.0, env.W / 2.0 - 1e-4, 0.0, env.v_max], dtype=np.float32)
    action = np.array([0.0, env.u_max], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated is True
    assert reward == env.config.collision_reward
    assert info["collision"] is True
    assert info["is_success"] is False


def test_goal_reached_at_boundary():
    env = make(sigma_p=0.0, sigma_v=0.0, tunnel_length=10.0)
    env.reset(seed=0)
    env.state = np.array([10.0 - 1e-4, 0.0, env.v_max, 0.0], dtype=np.float32)
    action = np.array([env.u_max, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated is True
    assert reward == env.config.goal_reward
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
        SlalomEnvConfig(**kwargs)


def test_default_config_matches_historical_hardcoded_values():
    config = SlalomEnvConfig()
    assert config.dt == 0.1
    assert config.tunnel_length == 10.0
    assert config.tunnel_width == 4.0
    assert config.v_max == 2.0
    assert config.u_max == 1.0
    assert config.sigma_p == 0.0
    assert config.sigma_v == 0.0
    assert config.max_steps == 500
    assert config.step_penalty == -1.0
    assert config.collision_reward == -150.0
    assert config.goal_reward == 500.0
    assert config.progress_reward_coef == 0.0


def test_overrides_replace_config_fields():
    env = SlalomEnv(tunnel_length=20.0, sigma_p=0.5)
    assert env.L == 20.0
    assert env.sigma_p == 0.5
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
    env = make(sigma_p=0.0, sigma_v=0.0, width_profile=profile)
    env.reset(seed=0)
    env.state = np.array([4.5, 1.9, 0.0, 0.0], dtype=np.float32)
    _, reward, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is True
    assert info["collision"] is True
    assert reward == env.config.collision_reward

    # The identical p_y in a full-width segment (x=0.0) is safe.
    env2 = make(sigma_p=0.0, sigma_v=0.0, width_profile=profile)
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


def test_progress_reward_coef_default_is_noop():
    """Default progress_reward_coef=0.0 must not perturb the reward even when
    p_x actually changes step to step -- guards the no-op claim for every
    existing checkpoint/result in this repository."""
    env = make(sigma_p=0.0, sigma_v=0.0)
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
    assert terminated is True
    assert info["collision"] is True
    expected = env.config.collision_reward + coef * (next_obs[0] - p_x_prev)
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


def test_collision_reward_matches_refund_and_impact_formula():
    """Direct check of the collision reward formula: collision_reward -
    impact_penalty_coef * |v_y| + refund of the step_penalty already
    accumulated before the crash, plus progress shaping on top."""
    coef = 3.0
    impact_coef = 7.0
    env = make(
        sigma_p=0.0, sigma_v=0.0, tunnel_width=4.0,
        progress_reward_coef=coef, impact_penalty_coef=impact_coef,
    )
    env.reset(seed=0)
    env.state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Three safe steps (no collision, no goal) before forcing the crash, so
    # step_penalty has already accumulated by the time it happens.
    safe_action = np.zeros(2, dtype=np.float32)
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(safe_action)
        assert not terminated and not truncated
    assert env.steps == 3

    p_x_prev = float(env.state[0])
    env.state = np.array([p_x_prev, env.W / 2.0 - 1e-4, 0.0, 0.0], dtype=np.float32)
    action = np.array([1.0, env.u_max], dtype=np.float32)
    next_obs, reward, terminated, _, info = env.step(action)
    assert terminated is True
    assert info["collision"] is True
    assert env.steps == 4

    impact_speed = abs(float(next_obs[3]))
    time_penalty_refund = -env.config.step_penalty * (env.steps - 1)
    expected = (
        env.config.collision_reward
        - impact_coef * impact_speed
        + time_penalty_refund
        + coef * (next_obs[0] - p_x_prev)
    )
    assert reward == pytest.approx(expected)


def test_collision_reward_independent_of_timestep_when_impact_matches():
    """Same lateral impact speed and progress_reward_coef=0: the *cumulative
    episode return* (safe per-step penalties already paid out, plus the
    terminal collision reward) must be the same whether the crash happens on
    the first step or after several safe steps -- the whole point of
    refunding the accumulated step_penalty. The terminal-step reward alone
    differs (it carries the refund), which is exactly what makes the totals
    match."""
    common = dict(sigma_p=0.0, sigma_v=0.0, tunnel_width=4.0, progress_reward_coef=0.0)

    env_early = make(**common)
    env_early.reset(seed=0)
    env_early.state = np.array([1.0, env_early.W / 2.0 - 1e-4, 0.0, 0.0], dtype=np.float32)
    _, reward_early, terminated_early, _, info_early = env_early.step(
        np.array([0.0, env_early.u_max], dtype=np.float32)
    )
    assert terminated_early and info_early["collision"]
    total_return_early = reward_early

    env_late = make(**common)
    env_late.reset(seed=0)
    env_late.state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    safe_action = np.zeros(2, dtype=np.float32)
    total_return_late = 0.0
    for _ in range(10):
        _, reward, terminated, truncated, _ = env_late.step(safe_action)
        assert not terminated and not truncated
        total_return_late += reward
    env_late.state = np.array([float(env_late.state[0]), env_late.W / 2.0 - 1e-4, 0.0, 0.0], dtype=np.float32)
    _, reward_late, terminated_late, _, info_late = env_late.step(
        np.array([0.0, env_late.u_max], dtype=np.float32)
    )
    assert terminated_late and info_late["collision"]
    total_return_late += reward_late

    assert total_return_late == pytest.approx(total_return_early)


def test_impact_penalty_scales_with_lateral_impact_speed():
    """Higher |v_y| at the moment of impact must yield a more negative
    reward when impact_penalty_coef > 0, at the same timestep."""
    common = dict(
        sigma_p=0.0, sigma_v=0.0, tunnel_width=4.0,
        progress_reward_coef=0.0, impact_penalty_coef=5.0,
    )

    env_slow = make(**common)
    env_slow.reset(seed=0)
    env_slow.state = np.array([1.0, env_slow.W / 2.0 - 1e-4, 0.0, 0.5], dtype=np.float32)
    _, reward_slow, terminated_slow, _, info_slow = env_slow.step(np.zeros(2, dtype=np.float32))
    assert terminated_slow and info_slow["collision"]

    env_fast = make(**common)
    env_fast.reset(seed=0)
    env_fast.state = np.array([1.0, env_fast.W / 2.0 - 1e-4, 0.0, env_fast.v_max], dtype=np.float32)
    _, reward_fast, terminated_fast, _, info_fast = env_fast.step(np.zeros(2, dtype=np.float32))
    assert terminated_fast and info_fast["collision"]

    assert reward_fast < reward_slow


def test_render_rgb_array_returns_frame():
    env = SlalomEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None
    assert frame.ndim == 3 and frame.shape[2] == 3
    env.close()
