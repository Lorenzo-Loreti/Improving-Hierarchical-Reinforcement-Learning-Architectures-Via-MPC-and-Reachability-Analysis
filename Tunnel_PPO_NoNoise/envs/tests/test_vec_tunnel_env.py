import numpy as np
import pytest

from envs.config import TunnelEnvConfig
from envs.tunnel_env import TunnelEnv
from envs.vec_tunnel_env import TunnelVecEnv
from envs.width_profile import WidthSegment, WidthProfile


def _gate_profile():
    """A minimal multi-segment profile with one narrow, offset gate -- just
    enough to exercise position-dependent bounds."""
    return WidthProfile([
        WidthSegment(-np.inf, 2.0, 2.0, 0.0),
        WidthSegment(2.0, 5.0, 0.75, 1.0),
        WidthSegment(5.0, np.inf, 2.0, 0.0),
    ])


def test_vec_env_matches_single_env_dynamics_zero_noise():
    config = TunnelEnvConfig(sigma_p=0.0, sigma_v=0.0, width_profile=None)
    num_envs = 5

    vec_env = TunnelVecEnv(num_envs=num_envs, config=config)
    vec_env.seed(123)
    vec_obs, _ = vec_env.reset()

    single_envs = [TunnelEnv(config=config) for _ in range(num_envs)]
    single_obs = []
    for i, env in enumerate(single_envs):
        obs, _ = env.reset(seed=None)
        env.state = vec_env.states[i].copy()
        single_obs.append(env.state.copy())
    single_obs = np.array(single_obs)

    np.testing.assert_allclose(vec_obs, single_obs, atol=1e-5)

    rng = np.random.default_rng(0)
    for _ in range(20):
        actions = rng.uniform(-1.0, 1.0, size=(num_envs, 2)).astype(np.float32)
        vec_obs, vec_rewards, vec_term, vec_trunc, vec_info = vec_env.step(actions)

        for i, env in enumerate(single_envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            if terminated or truncated:
                assert vec_term[i] == terminated
                assert vec_trunc[i] == truncated
                assert vec_rewards[i] == reward
                np.testing.assert_allclose(
                    vec_info["final_observation"][i], obs, atol=1e-5
                )
                env.reset(seed=None)
                env.state = vec_env.states[i].copy()
            else:
                assert not vec_term[i] and not vec_trunc[i]
                np.testing.assert_allclose(vec_obs[i], obs, atol=1e-5)
                assert vec_rewards[i] == reward


def test_vec_env_matches_single_env_dynamics_with_progress_shaping():
    """Same equivalence check as
    test_vec_env_matches_single_env_dynamics_zero_noise, with
    progress_reward_coef != 0 -- the shaping term must agree between the
    batched and single-env implementations row by row."""
    config = TunnelEnvConfig(sigma_p=0.0, sigma_v=0.0, width_profile=None, progress_reward_coef=2.0)
    num_envs = 5

    vec_env = TunnelVecEnv(num_envs=num_envs, config=config)
    vec_env.seed(123)
    vec_env.reset()

    single_envs = [TunnelEnv(config=config) for _ in range(num_envs)]
    for i, env in enumerate(single_envs):
        env.reset(seed=None)
        env.state = vec_env.states[i].copy()

    rng = np.random.default_rng(0)
    for _ in range(20):
        actions = rng.uniform(-1.0, 1.0, size=(num_envs, 2)).astype(np.float32)
        _, vec_rewards, vec_term, vec_trunc, _ = vec_env.step(actions)

        for i, env in enumerate(single_envs):
            _, reward, terminated, truncated, _ = env.step(actions[i])
            assert vec_rewards[i] == pytest.approx(reward, abs=1e-4)
            if terminated or truncated:
                assert vec_term[i] == terminated
                assert vec_trunc[i] == truncated
                env.reset(seed=None)
                env.state = vec_env.states[i].copy()


def test_vec_env_matches_single_env_collision_impact_and_refund():
    """Same equivalence check as
    test_vec_env_matches_single_env_dynamics_zero_noise, with
    impact_penalty_coef != 0 -- the batched collision reward (impact term +
    refund of the already-accumulated step_penalty) must agree with the
    single-env implementation row by row, including for rows that collide
    several steps into the rollout."""
    config = TunnelEnvConfig(sigma_p=0.0, sigma_v=0.0, width_profile=None, impact_penalty_coef=4.0)
    num_envs = 5

    vec_env = TunnelVecEnv(num_envs=num_envs, config=config)
    vec_env.seed(123)
    vec_env.reset()

    single_envs = [TunnelEnv(config=config) for _ in range(num_envs)]
    for i, env in enumerate(single_envs):
        env.reset(seed=None)
        env.state = vec_env.states[i].copy()

    rng = np.random.default_rng(0)
    saw_collision = False
    for _ in range(30):
        actions = rng.uniform(-1.0, 1.0, size=(num_envs, 2)).astype(np.float32)
        _, vec_rewards, vec_term, vec_trunc, vec_info = vec_env.step(actions)

        for i, env in enumerate(single_envs):
            _, reward, terminated, truncated, _ = env.step(actions[i])
            assert vec_rewards[i] == pytest.approx(reward, abs=1e-4)
            if terminated or truncated:
                assert vec_term[i] == terminated
                assert vec_trunc[i] == truncated
                saw_collision = saw_collision or bool(vec_info["final_info"]["collision"][i])
                env.reset(seed=None)
                env.state = vec_env.states[i].copy()

    assert saw_collision, "test setup never exercised the collision branch"


def test_vec_env_progress_shaping_uses_terminal_px_not_post_reset_px():
    """Regression guard for the exact aliasing trap already documented in
    TUNNEL_ENV.md Sec 14.3 (info["distance_to_goal"] previously leaked the
    post-reset row's value on a done row): the progress-shaping term must be
    computed from each row's terminal p_x, not the freshly-sampled
    next-episode p_x that _sample_initial writes in place after done rows
    are identified."""
    coef = 10.0
    config = TunnelEnvConfig(
        sigma_p=0.0, sigma_v=0.0, tunnel_width=1000.0,
        tunnel_length=2.0, max_steps=1000, progress_reward_coef=coef,
    )
    vec_env = TunnelVecEnv(num_envs=2, config=config)
    vec_env.reset(seed=0)

    # Row 0 is one step from the goal; row 1 stays far from termination.
    vec_env.states = np.array(
        [[2.0 - 1e-4, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32
    )
    p_x_prev_row0 = float(vec_env.states[0, 0])

    actions = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    obs, rewards, terminated, _, info = vec_env.step(actions)

    assert terminated[0] and info["final_info"]["is_success"][0]
    terminal_p_x_row0 = float(info["final_observation"][0, 0])
    expected_reward_row0 = config.goal_reward + coef * (terminal_p_x_row0 - p_x_prev_row0)
    assert rewards[0] == pytest.approx(expected_reward_row0)

    # The bug this guards against: computing the shaping term from the
    # post-reset row instead would use obs[0, 0] (freshly sampled in [0, 2))
    # rather than the terminal p_x (~2.0+) -- a materially different delta.
    post_reset_p_x_row0 = float(obs[0, 0])
    wrong_reward_row0 = config.goal_reward + coef * (post_reset_p_x_row0 - p_x_prev_row0)
    assert rewards[0] != pytest.approx(wrong_reward_row0)


def test_vec_env_reset_shape_and_bounds():
    config = TunnelEnvConfig()
    vec_env = TunnelVecEnv(num_envs=8, config=config)
    obs, _ = vec_env.reset(seed=0)
    assert obs.shape == (8, 4)
    assert np.all(obs[:, 0] >= 0.0) and np.all(obs[:, 0] <= 2.0)
    assert np.all(np.abs(obs[:, 1]) <= config.tunnel_width / 4.0)
    assert np.all(obs[:, 2] == 0.0) and np.all(obs[:, 3] == 0.0)


def test_vec_env_autoresets_on_done():
    config = TunnelEnvConfig(tunnel_length=1.0, max_steps=1000, sigma_p=0.0, sigma_v=0.0)
    vec_env = TunnelVecEnv(num_envs=4, config=config)
    vec_env.reset(seed=0)
    actions = np.ones((4, 2), dtype=np.float32)
    for _ in range(50):
        obs, rewards, terminated, truncated, info = vec_env.step(actions)
        if np.any(terminated | truncated):
            done_mask = terminated | truncated
            assert np.all(obs[done_mask][:, 0] <= 2.0)
            return
    raise AssertionError("expected at least one env to terminate within 50 steps")


def test_vec_env_collision_is_position_dependent_per_row():
    """Different rows of one step() straddling different width-profile
    segments must each be checked against their own segment's bound, not one
    shared constant -- the batched analogue of
    test_tunnel_env.py::test_collision_is_position_dependent_under_a_width_profile."""
    config = TunnelEnvConfig(sigma_p=0.0, sigma_v=0.0, width_profile=_gate_profile())
    vec_env = TunnelVecEnv(num_envs=2, config=config)
    vec_env.reset(seed=0)

    # Row 0: x=3.0, inside the gate (opening y in [0.25, 1.75]) -- y=1.9 collides.
    # Row 1: x=0.0, full-width entry segment -- the same y=1.9 is safe.
    vec_env.states = np.array([[3.0, 1.9, 0.0, 0.0], [0.0, 1.9, 0.0, 0.0]], dtype=np.float32)
    _, rewards, terminated, _, info = vec_env.step(np.zeros((2, 2), dtype=np.float32))

    assert terminated[0] and info["final_info"]["collision"][0]
    assert rewards[0] == config.collision_reward
    assert not terminated[1] and not info["final_info"]["collision"][1]


def test_final_observation_reconstructs_a_physically_continuous_trajectory():
    """The contract both HRL rollouts depend on, and the trap they would fall
    into without it.

    hPPO and PPO_MPC decrement the manager's goal by the distance the vehicle
    actually moved, `term_obs - obs`. Under auto-reset `obs[i]` for a finished
    environment is already the *next* episode's start, so taking the delta from
    it measures a teleport from wherever the episode ended back to the tunnel
    entrance -- tens of times a real step, on every episode boundary, silently
    corrupting the worker's progress reward and the manager's goal. The
    pre-reset state has to come from `info["final_observation"]`.

    Asserted as a physical bound: one step cannot move the vehicle further than
    v_max * dt (plus a noise allowance), whatever the policy does.
    """
    config = TunnelEnvConfig(sigma_p=0.0, sigma_v=0.0)
    num_envs = 8
    vec_env = TunnelVecEnv(num_envs=num_envs, config=config)
    obs, _ = vec_env.reset(seed=0)
    rng = np.random.default_rng(0)
    bound = config.v_max * config.dt + 1e-6

    worst_reconstructed = 0.0
    worst_naive = 0.0
    done_events = 0
    for _ in range(2000):
        actions = rng.uniform(-1, 1, (num_envs, 2)).astype(np.float32)
        next_obs, _, terminated, truncated, info = vec_env.step(actions)
        done = terminated | truncated

        term_obs = next_obs.copy()
        if np.any(done):
            term_obs[done] = info["final_observation"][done]

        worst_reconstructed = max(worst_reconstructed, float(
            np.linalg.norm(term_obs[:, :2] - obs[:, :2], axis=1).max()))
        worst_naive = max(worst_naive, float(
            np.linalg.norm(next_obs[:, :2] - obs[:, :2], axis=1).max()))
        done_events += int(done.sum())
        obs = next_obs

    assert done_events > 0, "no episode ended, so the reconstruction was never exercised"
    assert worst_reconstructed <= bound, (
        f"final_observation did not reconstruct the pre-reset state: a step "
        f"moved {worst_reconstructed:.4f} m against a bound of {bound:.4f} m")
    # And the naive version really is broken, so this test cannot pass vacuously.
    assert worst_naive > bound
