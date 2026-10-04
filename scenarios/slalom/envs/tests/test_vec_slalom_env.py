import numpy as np
import pytest

from envs.config import SlalomEnvConfig
from envs.slalom_env import SlalomEnv
from envs.vec_slalom_env import SlalomVecEnv
from envs.width_profile import slalom_profile


def test_vec_env_matches_single_env_dynamics_zero_noise():
    config = SlalomEnvConfig(noise_bound_p=0.0, noise_bound_v=0.0, width_profile=slalom_profile())
    num_envs = 5

    vec_env = SlalomVecEnv(num_envs=num_envs, config=config)
    vec_env.seed(123)
    vec_obs, _ = vec_env.reset()

    single_envs = [SlalomEnv(config=config) for _ in range(num_envs)]
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
    config = SlalomEnvConfig(
        noise_bound_p=0.0, noise_bound_v=0.0, width_profile=slalom_profile(), progress_reward_coef=2.0
    )
    num_envs = 5

    vec_env = SlalomVecEnv(num_envs=num_envs, config=config)
    vec_env.seed(123)
    vec_env.reset()

    single_envs = [SlalomEnv(config=config) for _ in range(num_envs)]
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


def test_vec_env_matches_single_env_collision_impact():
    """Same equivalence check as
    test_vec_env_matches_single_env_dynamics_zero_noise, focused on wall
    contact -- the batched per-contact reward (contact_penalty) must agree
    with the single-env implementation row by row, including rows that
    bounce off the wall several times over the rollout (a contact no longer
    ends the episode, so both implementations must also keep applying the
    same clamp-and-absorb physics in lockstep without an explicit reset)."""
    config = SlalomEnvConfig(
        noise_bound_p=0.0, noise_bound_v=0.0, width_profile=slalom_profile()
    )
    num_envs = 5

    vec_env = SlalomVecEnv(num_envs=num_envs, config=config)
    vec_env.seed(123)
    vec_env.reset()

    single_envs = [SlalomEnv(config=config) for _ in range(num_envs)]
    for i, env in enumerate(single_envs):
        env.reset(seed=None)
        env.state = vec_env.states[i].copy()

    rng = np.random.default_rng(0)
    saw_collision = False
    for _ in range(30):
        actions = rng.uniform(-1.0, 1.0, size=(num_envs, 2)).astype(np.float32)
        _, vec_rewards, vec_term, vec_trunc, _ = vec_env.step(actions)

        for i, env in enumerate(single_envs):
            _, reward, terminated, truncated, info = env.step(actions[i])
            assert vec_rewards[i] == pytest.approx(reward, abs=1e-4)
            saw_collision = saw_collision or bool(info["collision"])
            if terminated or truncated:
                assert vec_term[i] == terminated
                assert vec_trunc[i] == truncated
                env.reset(seed=None)
                env.state = vec_env.states[i].copy()

    assert saw_collision, "test setup never exercised the collision branch"


def test_vec_env_progress_shaping_uses_terminal_px_not_post_reset_px():
    """Regression guard for an aliasing trap (info["distance_to_goal"]
    previously leaked the post-reset row's value on a done row): the
    progress-shaping term must be
    computed from each row's terminal p_x, not the freshly-sampled
    next-episode p_x that _sample_initial writes in place after done rows
    are identified."""
    coef = 10.0
    config = SlalomEnvConfig(
        noise_bound_p=0.0, noise_bound_v=0.0, tunnel_width=1000.0,
        tunnel_length=2.0, max_steps=1000, progress_reward_coef=coef,
    )
    vec_env = SlalomVecEnv(num_envs=2, config=config)
    vec_env.reset(seed=0)

    # Row 0 is one step from the goal; row 1 stays far from termination.
    vec_env.states = np.array(
        [[2.0 - 1e-4, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32
    )
    p_x_prev_row0 = float(vec_env.states[0, 0])

    actions = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    obs, rewards, terminated, _, info = vec_env.step(actions)

    assert terminated[0] and info["final_info"]["is_success"][0]
    terminal_p_x_row0 = float(info["final_observation"][0, 0])
    assert terminal_p_x_row0 > config.tunnel_length
    # The potential is cut at L, so the terminal row earns the distance up
    # to the line, plus its effort term.
    expected_reward_row0 = (config.goal_reward + coef * (config.tunnel_length - p_x_prev_row0)
                            + info["effort_penalty"][0])
    assert info["effort_penalty"][0] < 0
    assert rewards[0] == pytest.approx(expected_reward_row0)

    # The bug this guards against: computing the shaping term from the
    # post-reset row instead would use obs[0, 0] (freshly sampled in [0, 2))
    # rather than the terminal p_x (~2.0+) -- a materially different delta.
    post_reset_p_x_row0 = float(obs[0, 0])
    wrong_reward_row0 = (config.goal_reward + coef * (post_reset_p_x_row0 - p_x_prev_row0)
                         + info["effort_penalty"][0])
    assert rewards[0] != pytest.approx(wrong_reward_row0)


def test_vec_env_reset_shape_and_bounds():
    config = SlalomEnvConfig()
    vec_env = SlalomVecEnv(num_envs=8, config=config)
    obs, _ = vec_env.reset(seed=0)
    assert obs.shape == (8, 4)
    assert np.all(obs[:, 0] >= 0.0) and np.all(obs[:, 0] <= 2.0)
    assert np.all(np.abs(obs[:, 1]) <= config.tunnel_width / 4.0)
    assert np.all(obs[:, 2] == 0.0) and np.all(obs[:, 3] == 0.0)


def test_vec_env_autoresets_on_done():
    config = SlalomEnvConfig(tunnel_length=1.0, max_steps=1000, noise_bound_p=0.0, noise_bound_v=0.0)
    vec_env = SlalomVecEnv(num_envs=4, config=config)
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
    test_slalom_env.py::test_collision_is_position_dependent_under_a_width_profile."""
    config = SlalomEnvConfig(noise_bound_p=0.0, noise_bound_v=0.0, width_profile=slalom_profile())
    vec_env = SlalomVecEnv(num_envs=2, config=config)
    vec_env.reset(seed=0)

    # Row 0: x=4.5, inside gate 1 (opening y in [0.25, 1.75]) -- y=1.9 collides.
    # Row 1: x=0.0, full-width entry segment -- the same y=1.9 is safe.
    vec_env.states = np.array([[4.5, 1.9, 0.0, 0.0], [0.0, 1.9, 0.0, 0.0]], dtype=np.float32)
    obs, rewards, terminated, _, info = vec_env.step(np.zeros((2, 2), dtype=np.float32))

    assert not terminated[0] and info["final_info"]["collision"][0]
    y_lo, y_hi = config.width_profile.bounds_at(4.5)
    assert obs[0, 1] == pytest.approx(y_hi)
    assert obs[0, 3] == 0.0
    expected0 = config.step_penalty + config.contact_penalty  # v_y starts at 0
    assert rewards[0] == pytest.approx(expected0)

    assert not terminated[1] and not info["final_info"]["collision"][1]


def test_final_observation_reconstructs_a_physically_continuous_trajectory():
    """The contract both HRL rollouts depend on, and the trap they would fall
    into without it.

    hPPO and PPO_MPC decrement the manager's goal by the distance the vehicle
    actually moved, `term_obs - obs`. Under auto-reset `obs[i]` for a finished
    environment is already the *next* episode's start, so taking the delta from
    it measures a teleport from wherever the episode ended back to the slalom
    entrance -- tens of times a real step, on every episode boundary, silently
    corrupting the worker's progress reward and the manager's goal. The
    pre-reset state has to come from `info["final_observation"]`.

    Asserted as a physical bound: one step cannot move the vehicle further than
    v_max * dt, whatever the policy does -- the position integrates over the
    average of the velocities before and after the step, both inside the
    speed disk (envs/actuation.py). Until 2026-10-04 the limits were per
    axis and the bound here was sqrt(2) times a per-axis one. A wall contact
    does not loosen this bound either -- it only clamps p_y back to the
    boundary, never past it.
    """
    config = SlalomEnvConfig(noise_bound_p=0.0, noise_bound_v=0.0)
    num_envs = 8
    vec_env = SlalomVecEnv(num_envs=num_envs, config=config)
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


def test_vec_env_collision_count_and_impacts_track_each_contact():
    """Per-row analogue of
    test_slalom_env.py::test_collision_count_and_impacts_accumulate_over_episode:
    collision_count and collision_impacts[i] must track every contact for row
    i independently, not just the most recent one, and collision_impacts
    holds each contact's own penalty rather than a running sum."""
    config = SlalomEnvConfig(noise_bound_p=0.0, noise_bound_v=0.0, tunnel_width=4.0, max_steps=5)
    vec_env = SlalomVecEnv(num_envs=2, config=config)
    vec_env.reset(seed=0)

    vec_env.states = np.array(
        [[1.0, config.tunnel_width / 2.0 - 1e-4, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    actions = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    _, _, _, _, info1 = vec_env.step(actions)
    assert info1["final_info"]["collision_count"][0] == 1
    assert len(info1["final_info"]["collision_impacts"][0]) == 1
    assert info1["final_info"]["collision_count"][1] == 0
    assert info1["final_info"]["collision_impacts"][1] == []

    # v_y in row 0 was zeroed by the first bounce -- nudge it back into the
    # wall for a second, independent contact.
    vec_env.states[0, 1] = config.tunnel_width / 2.0 - 1e-4
    vec_env.states[0, 3] = 1.0
    _, _, _, _, info2 = vec_env.step(actions)
    assert info2["final_info"]["collision_count"][0] == 2
    assert len(info2["final_info"]["collision_impacts"][0]) == 2
    assert info2["final_info"]["collision_impacts"][0][0] == pytest.approx(
        info1["final_info"]["collision_impacts"][0][0]
    )


def test_vec_env_collision_bookkeeping_resets_per_row_on_autoreset():
    """The per-row count/impacts must be snapshotted into final_info *before*
    _sample_initial clears them for the row's next episode -- so the same
    step()'s final_info still shows the finished episode's contacts, while
    the internal counters are already reset for what comes next."""
    config = SlalomEnvConfig(noise_bound_p=0.0, noise_bound_v=0.0, tunnel_length=10.0, tunnel_width=4.0, max_steps=1000)
    vec_env = SlalomVecEnv(num_envs=2, config=config)
    vec_env.reset(seed=0)

    # Row 0 takes a wall contact, then a few steps later reaches the goal and
    # auto-resets; row 1 is inert throughout.
    vec_env.states = np.array(
        [[1.0, config.tunnel_width / 2.0 - 1e-4, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    contact_action = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    _, _, _, _, info = vec_env.step(contact_action)
    assert info["final_info"]["collision_count"][0] == 1

    vec_env.states[0] = [10.0 - 1e-4, 0.0, vec_env.v_max, 0.0]
    goal_action = np.array([[vec_env.u_max, 0.0], [0.0, 0.0]], dtype=np.float32)
    _, _, terminated, _, info = vec_env.step(goal_action)
    assert terminated[0]
    # The terminal snapshot still reflects the just-finished episode...
    assert info["final_info"]["collision_count"][0] == 1

    # ...but the auto-reset already ran inside that same step() call, so the
    # counters for row 0's fresh episode read cleared right away.
    assert vec_env._collision_counts[0] == 0
    assert vec_env._collision_impacts[0] == []


def test_vec_env_applies_the_speed_limit_like_the_single_env():
    """Rows at or near top speed, thrusting into the limit or across it, and
    one command outside the thrust disk (see envs/actuation.py)."""
    config = SlalomEnvConfig(noise_bound_p=0.0, noise_bound_v=0.0, width_profile=slalom_profile())
    starts = np.array([[5.0, 0.0, 0.85, -0.85], [5.0, 0.0, 1.0, 0.3], [2.0, 0.5, -0.4, 1.1]], dtype=np.float32)
    actions = np.array([[2.5, -2.5], [0.0, 2.5], [-9.0, 2.5]], dtype=np.float32)
    vec_env = SlalomVecEnv(num_envs=3, config=config)
    vec_env.reset(seed=0)
    vec_env.states[:] = starts
    vec_obs, *_ = vec_env.step(actions)
    for i in range(3):
        env = SlalomEnv(config=config)
        env.reset(options={"init_state": starts[i]})
        obs, *_ = env.step(actions[i])
        np.testing.assert_array_equal(vec_obs[i], obs)


def test_vec_env_noise_is_uniform_on_the_bounded_disks():
    """As the scalar env's test of the same name: every draw stays inside both
    disks, reaches their edge, and is uniform in area."""
    bound_p, bound_v = 0.01, 0.05
    config = SlalomEnvConfig(noise_bound_p=bound_p, noise_bound_v=bound_v, tunnel_length=1000.0, max_steps=10000)
    vec_env = SlalomVecEnv(num_envs=4, config=config)
    vec_env.reset(seed=0)
    ws = []
    for _ in range(1000):
        # Restarted every step; see the scalar test.
        vec_env.states[:] = np.array([1.0, 0.0, 0.3, 0.0], dtype=np.float32)
        predicted = vec_env.states.astype(np.float64) @ vec_env.A.T.astype(np.float64)
        vec_env.step(np.zeros((4, 2), dtype=np.float32))
        ws.append(vec_env.states.astype(np.float64) - predicted)
    ws = np.concatenate(ws)
    for block, bound in ((ws[:, :2], bound_p), (ws[:, 2:], bound_v)):
        radius = np.linalg.norm(block, axis=1)
        assert np.all(radius <= bound * (1 + 1e-4) + 1e-6)
        assert radius.max() > 0.98 * bound
        assert 0.22 < np.mean(radius <= bound / 2) < 0.28


def test_vec_env_reports_the_effort_term_it_charged():
    """info["effort_penalty"] is each row's effort term for this step, the one
    already inside `rewards` (hPPO's worker reads it): effort_penalty for a
    full-thrust row, 0 for an idle one."""
    config = SlalomEnvConfig(tunnel_width=1000.0)
    vec_env = SlalomVecEnv(num_envs=3, config=config)
    vec_env.reset(seed=0)
    vec_env.states[:] = np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32)
    actions = np.array([[config.u_max, 0.0], [0.0, 0.0], [0.0, -0.5 * config.u_max]], dtype=np.float32)
    obs, rewards, *_, info = vec_env.step(actions)
    np.testing.assert_allclose(info["effort_penalty"], config.effort_penalty * np.array([1.0, 0.0, 0.25]), rtol=1e-5)
    progress = config.progress_reward_coef * (obs[:, 0] - 3.0)
    np.testing.assert_allclose(rewards, config.step_penalty + progress + info["effort_penalty"], atol=1e-5)


def test_vec_env_zero_noise_draws_no_randomness():
    vec_env = SlalomVecEnv(num_envs=3, config=SlalomEnvConfig())
    vec_env.reset(seed=3)
    rng_state = vec_env._np_random.bit_generator.state
    vec_env.step(np.ones((3, 2), dtype=np.float32))
    assert vec_env._np_random.bit_generator.state == rng_state
