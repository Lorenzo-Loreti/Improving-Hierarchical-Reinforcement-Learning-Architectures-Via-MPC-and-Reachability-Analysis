"""Tests of the navigation MDP (rows M1, M5, E7, R1-R5, AM1, X5; decision log D6, D10, D13-D15, D18)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from hrlmpc.action_map import square_to_disk
from hrlmpc.config import EnvConfig, env_config_from_dict
from hrlmpc.env import NavigationEnv
from hrlmpc.physics import physics_step

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "env"
SEED = 0


def _config(name: str = "slalom", **sections: dict[str, Any]) -> EnvConfig:
    """The YAML configuration ``name`` with some values replaced, e.g. ``physics={"d_bar": 0.0}``."""
    data = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    data = copy.deepcopy(data)
    for section, values in sections.items():
        data[section].update(values)
    return env_config_from_dict(data)


def _phi(x: np.ndarray, config: EnvConfig) -> np.ndarray:
    """Shaping potential of D13."""
    return -config.reward.progress_coef * np.maximum(config.geometry.goal_x - x, 0.0)


def test_reset_starts_every_agent_at_rest_in_the_spawn_box() -> None:
    config = _config()
    env = NavigationEnv(config, num_envs=64, seed=SEED)
    obs = env.reset()
    spawn = config.task.spawn
    p, v = env.positions, env.velocities
    assert np.all((p[:, 0] >= spawn.x[0]) & (p[:, 0] <= spawn.x[1]) & (p[:, 1] >= spawn.y[0]) & (p[:, 1] <= spawn.y[1]))
    np.testing.assert_array_equal(v, 0.0)
    assert obs.shape == (64, 4)
    np.testing.assert_allclose(obs, env.observe(p, v))


def test_observations_are_normalized_with_fixed_bounds() -> None:
    """D13: (p, v) mapped to [-1, 1] with the arena's bounding box and v_max."""
    env = NavigationEnv(_config(), num_envs=1, seed=SEED)
    p = np.array([[-1.0, -2.0], [11.0, 2.0], [5.0, 0.0]])
    v = np.array([[1.2, 0.0], [0.0, -1.2], [0.6, 0.6]])
    np.testing.assert_allclose(env.observe(p, v), [[-1, -1, 1, 0], [1, 1, 0, -1], [0, 0, 0.5, 0.5]], atol=1e-15)


def test_actions_are_mapped_onto_u_by_the_bijection_of_d15() -> None:
    env = NavigationEnv(_config(physics={"d_bar": 0.0}), num_envs=3, seed=SEED)
    env.reset()
    x = np.array([[1.0, 1.0], [-0.5, 0.25], [0.0, 0.0]])
    out = env.step(x)
    np.testing.assert_allclose(out.info.u_cmd, square_to_disk(x, 2.5), atol=1e-15)
    np.testing.assert_allclose(out.info.u, out.info.u_cmd, atol=1e-15)
    np.testing.assert_allclose(out.info.saturation, 0.0, atol=1e-12)  # the bijection lands in U


@pytest.mark.parametrize(
    ("action", "message"),
    [
        ([[1.5, 0.0]], "within"),
        ([[np.nan, 0.0]], "finite"),
        ([[0.0, 0.0], [0.0, 0.0]], "shape"),
        ([0.0, 0.0], "shape"),
    ],
)
def test_invalid_actions_are_rejected(action: Any, message: str) -> None:
    env = NavigationEnv(_config(), num_envs=1, seed=SEED)
    env.reset()
    with pytest.raises(ValueError, match=message):
        env.step(action)


def test_commanded_accelerations_are_saturated_and_the_correction_recorded() -> None:
    """The MPC Worker's path: any finite command, saturated onto U by the physics (R4)."""
    env = NavigationEnv(_config(physics={"d_bar": 0.0}), num_envs=2, seed=SEED)
    env.reset()
    out = env.step_input([[5.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(out.info.u, [[2.5, 0.0], [1.0, 1.0]], atol=1e-15)
    np.testing.assert_allclose(out.info.saturation, [2.5, 0.0], atol=1e-15)


def test_the_environment_steps_through_physics_step() -> None:
    """E7, M5: the transition is the one of the single physics module, with the sampled disturbance."""
    config = _config()
    env = NavigationEnv(config, num_envs=16, seed=SEED)
    env.reset()
    rng = np.random.default_rng(SEED)
    for _ in range(30):
        p, v = env.positions, env.velocities
        u = rng.uniform(-4.0, 4.0, (16, 2))
        out = env.step_input(u)
        expected = physics_step(p, v, u, out.info.disturbance, config.physics, env.layout)
        alive = ~(out.terminated | out.truncated)
        np.testing.assert_array_equal(env.positions[alive], expected.p[alive])
        np.testing.assert_array_equal(env.velocities[alive], expected.v[alive])
        np.testing.assert_array_equal(out.info.impulse, np.hypot(*expected.dv_wall.T))


def test_disturbances_are_iid_uniform_on_the_disk() -> None:
    """D10: uniform in area on D, independent of the state and of the actions."""
    config = _config(physics={"d_bar": 1.0})
    a, b = NavigationEnv(config, num_envs=8, seed=SEED), NavigationEnv(config, num_envs=8, seed=SEED)
    a.reset()
    b.reset()
    rng = np.random.default_rng(1)
    draws = []
    for _ in range(2500):
        da = a.step(np.zeros((8, 2))).info.disturbance
        db = b.step(rng.uniform(-1.0, 1.0, (8, 2))).info.disturbance
        np.testing.assert_array_equal(da, db)
        draws.append(da)
    d = np.concatenate(draws)
    radius2 = np.sum(d**2, axis=1)
    assert radius2.max() <= 1.0 + 1e-12
    assert abs(radius2.mean() - 0.5) < 0.01  # E||d||^2 = d_bar^2 / 2 for the uniform disk
    assert np.all(np.abs(d.mean(axis=0)) < 0.01)


def test_disturbance_levels_share_their_random_numbers() -> None:
    """Paired comparisons: the same seed draws the same unit disturbances at every level of d_bar."""
    runs = {}
    for d_bar in (0.0, 0.5, 1.0):
        env = NavigationEnv(_config(physics={"d_bar": d_bar}), num_envs=4, seed=SEED)
        env.reset()
        runs[d_bar] = np.stack([env.step(np.zeros((4, 2))).info.disturbance for _ in range(20)])
    np.testing.assert_array_equal(runs[0.0], 0.0)
    np.testing.assert_allclose(runs[0.5], 0.5 * runs[1.0], rtol=1e-15)


def test_seeding_is_reproducible_and_independent_of_the_batch() -> None:
    """Agent 0 of a batch of 4 follows exactly the trajectory of a single agent with the same seed."""
    config = _config(physics={"d_bar": 1.0}, task={"horizon": 25})
    single, batch = NavigationEnv(config, num_envs=1, seed=3), NavigationEnv(config, num_envs=4, seed=3)
    np.testing.assert_array_equal(single.reset(), batch.reset()[:1])
    rng = np.random.default_rng(SEED)
    for _ in range(80):  # more than three episodes, so automatic resets are included
        x = rng.uniform(-1.0, 1.0, (4, 2))
        one, many = single.step(x[:1]), batch.step(x)
        for name in ("obs", "reward", "terminated", "truncated", "final_obs"):
            np.testing.assert_array_equal(getattr(one, name), getattr(many, name)[:1])
    other = NavigationEnv(config, num_envs=1, seed=4)
    assert not np.array_equal(other.reset(), NavigationEnv(config, num_envs=1, seed=3).reset())


@pytest.mark.parametrize("s", [1.0, 0.99])
def test_reward_of_a_head_on_impact(s: float) -> None:
    """D13, D14, D22: r = -1 - c_n ||dv_wall|| - c_s + (s Phi(p+) - Phi(p)) - effort, with p_x unchanged."""
    config = _config(physics={"d_bar": 0.0}, reward={"shaping_discount": s})
    env = NavigationEnv(config, num_envs=1, seed=SEED)
    env.reset(positions=[[2.0, -1.95]], velocities=[[0.0, -1.2]])
    out = env.step([[0.0, 0.0]])
    np.testing.assert_allclose(out.info.impulse, [0.64], atol=1e-12)
    shaping = s * -80.0 + 80.0
    np.testing.assert_allclose(out.reward, [-1.0 - 50.0 * 0.64 - 1.0 + shaping], atol=1e-10)
    np.testing.assert_allclose(out.info.reward_unshaped, [-1.0 - 50.0 * 0.64 - 1.0], atol=1e-10)
    assert out.info.contact.all()


@pytest.mark.parametrize("s", [1.0, 0.99])
def test_reward_of_free_motion_with_full_thrust(s: float) -> None:
    """D13, D22: effort -0.01 ||u||^2 / a_max^2 on the saturated input, progress shaping on p_x."""
    config = _config(physics={"d_bar": 0.0}, reward={"shaping_discount": s})
    env = NavigationEnv(config, num_envs=1, seed=SEED)
    env.reset(positions=[[2.0, 0.0]], velocities=[[0.0, 0.0]])
    out = env.step([[1.0, 0.0]])
    x_next = 2.0 + 0.1 * 0.25  # v+ = 0.1 * 2.5, p+ = p + 0.1 v+
    shaping = s * _phi(np.array(x_next), config) - _phi(np.array(2.0), config)
    np.testing.assert_allclose(out.reward, [-1.0 - 0.01 + shaping], atol=1e-10)
    np.testing.assert_allclose(out.info.reward_unshaped, [-1.0 - 0.01], atol=1e-10)
    assert not out.info.contact.any()


def test_effort_is_charged_on_the_saturated_input() -> None:
    """D13: a command of 5 m/s^2 is saturated to a_max, so the effort term is 0.01, not 0.04."""
    config = _config(physics={"d_bar": 0.0})
    env = NavigationEnv(config, num_envs=1, seed=SEED)
    env.reset(positions=[[2.0, 0.0]], velocities=[[0.0, 0.0]])
    out = env.step_input([[5.0, 0.0]])
    np.testing.assert_allclose(out.info.reward_unshaped, [-1.0 - 0.01], atol=1e-10)


def test_success_terminates_and_adds_the_bonus_to_the_other_terms() -> None:
    """D13 correction 1: the bonus adds to the step's other terms; Phi vanishes on the goal region."""
    config = _config(physics={"d_bar": 0.0})
    env = NavigationEnv(config, num_envs=1, seed=SEED)
    env.reset(positions=[[9.95, 0.0]], velocities=[[1.0, 0.0]])
    out = env.step([[0.0, 0.0]])
    assert out.terminated.tolist() == [True] and out.truncated.tolist() == [False]
    np.testing.assert_allclose(out.reward, [-1.0 + 1000.0 + 0.5], atol=1e-10)  # shaping s * 0 - (-0.5), any s
    np.testing.assert_allclose(out.final_obs[0, 0], (10.045 + 1.0) / 6.0 - 1.0, atol=1e-12)
    (episode,) = out.episodes
    assert episode.success and episode.length == 1 and episode.env == 0
    assert episode.unshaped_return == pytest.approx(999.0)
    spawn = config.task.spawn
    assert spawn.x[0] <= env.positions[0, 0] <= spawn.x[1]  # automatic reset into the spawn box


def test_truncation_after_the_horizon() -> None:
    env = NavigationEnv(_config(task={"horizon": 5}), num_envs=2, seed=SEED)
    env.reset()
    for k in range(5):
        out = env.step(np.zeros((2, 2)))
        assert out.truncated.tolist() == [k == 4] * 2
        assert not out.terminated.any()
    assert [e.length for e in out.episodes] == [5, 5] and not any(e.success for e in out.episodes)
    np.testing.assert_array_equal(env.velocities, 0.0)  # new episodes at rest
    assert not np.array_equal(out.obs, out.final_obs)


def test_shaping_telescopes_without_discount() -> None:
    """D22: with shaping discount 1 the progress terms of an episode sum to Phi(p_T) - Phi(p_0)."""
    config = _config(reward={"shaping_discount": 1.0}, task={"horizon": 60})
    env = NavigationEnv(config, num_envs=4, seed=SEED)
    env.reset()
    start = env.positions[:, 0].copy()
    rng = np.random.default_rng(SEED)
    shaping = np.zeros(4)
    for _ in range(60):
        out = env.step(rng.uniform(-0.2, 1.0, (4, 2)))
        shaping += out.reward - out.info.reward_unshaped
        end = np.where(out.terminated | out.truncated, (out.final_obs[:, 0] + 1.0) * 6.0 - 1.0, np.nan)
    np.testing.assert_allclose(shaping, _phi(end, config) - _phi(start, config), atol=1e-9)


def test_contact_metrics_count_events_duration_and_impulse() -> None:
    """R3: a head-on impact, then two steps at rest on the floor, a lift-off and a free step.

    The impact takes two steps: the first reaction lets the agent reach the floor with 0.5 m/s,
    the second cancels that speed (0.475 m/s after the ambient friction).
    """
    env = NavigationEnv(_config(physics={"d_bar": 0.0}, task={"horizon": 6}), num_envs=1, seed=SEED)
    env.reset(positions=[[2.0, -1.95]], velocities=[[0.0, -1.2]])
    contacts = [env.step([[0.0, 0.0]]).info.contact[0] for _ in range(4)]
    contacts.append(env.step([[0.0, 1.0]]).info.contact[0])  # lifts off the floor
    last = env.step([[0.0, 0.0]])
    contacts.append(last.info.contact[0])
    assert contacts == [True, True, True, True, False, False]
    (episode,) = last.episodes
    assert (episode.contact_steps, episode.contact_events, episode.wall_steps) == (4, 1, 2)
    assert episode.impulse == pytest.approx(0.64 + 0.475)


def test_correction_metrics_of_the_speed_limiter() -> None:
    """R4: full thrust at top speed; the limiter removes 0.19 m/s, i.e. 1.9 m/s^2 of the commanded action."""
    env = NavigationEnv(_config(physics={"d_bar": 0.0}, task={"horizon": 1}), num_envs=1, seed=SEED)
    env.reset(positions=[[2.0, 0.0]], velocities=[[1.2, 0.0]])
    out = env.step([[1.0, 0.0]])
    np.testing.assert_allclose(out.info.limiter, [1.9], atol=1e-12)
    (episode,) = out.episodes
    assert episode.limiter_steps == 1 and episode.limiter_total == pytest.approx(1.9)
    assert episode.saturation_steps == 0 and episode.wall_steps == 0


def test_samples_are_physical_steps_summed_over_the_batch() -> None:
    """D6: resets are not samples."""
    env = NavigationEnv(_config(task={"horizon": 3}), num_envs=3, seed=SEED)
    env.reset()
    for _ in range(10):
        env.step(np.zeros((3, 2)))
    env.reset()
    assert (env.counter.env_steps, env.counter.manager_decisions) == (30, 0)


def test_without_automatic_reset_a_finished_episode_must_be_reset() -> None:
    env = NavigationEnv(_config(task={"horizon": 2}), num_envs=2, seed=SEED, autoreset=False)
    env.reset()
    env.step(np.zeros((2, 2)))
    out = env.step(np.zeros((2, 2)))
    np.testing.assert_array_equal(out.obs, out.final_obs)  # no automatic reset
    with pytest.raises(ValueError, match="reset"):
        env.step(np.zeros((2, 2)))
    env.reset(mask=[True, False])
    with pytest.raises(ValueError, match="reset"):
        env.step(np.zeros((2, 2)))
    env.reset(mask=[False, True])
    env.step(np.zeros((2, 2)))


def test_reset_with_given_states_is_validated() -> None:
    env = NavigationEnv(_config(), num_envs=2, seed=SEED)
    with pytest.raises(ValueError, match="free space"):
        env.reset(positions=[[4.5, -1.0], [1.0, 0.0]])  # inside O1
    with pytest.raises(ValueError, match="v_max"):
        env.reset(positions=[[1.0, 0.0], [1.0, 0.5]], velocities=[[2.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="goal region"):
        env.reset(positions=[[10.5, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="boolean"):
        env.reset(mask=[0, 1])
    env.reset()
    before = env.positions.copy()
    env.reset(positions=[[3.0, 1.0]], mask=[False, True])
    np.testing.assert_array_equal(env.positions, [before[0], [3.0, 1.0]])


@pytest.mark.parametrize("name", ["slalom", "tunnel"])
def test_shipped_configs_meet_the_criterion_of_the_contact_penalty(name: str) -> None:
    """D18, D22: braking against a wall does not pay off.

    One saved step is worth at most c_t + (1 - gamma_RL) B + c_e, plus, when
    the shaping discount s exceeds gamma_RL, the step's share (s - gamma_RL)
    c_p d of the progress reward, with d at most the arena's length to the
    goal line; c_n must exceed that worth divided by a_max dt.
    """
    config = _config(name)
    r, p = config.reward, config.physics
    agent = yaml.safe_load((CONFIG_DIR.parent / "agent" / "ppo.yaml").read_text(encoding="utf-8"))
    gamma = agent["update"]["discount"]
    d_max = config.geometry.goal_x - config.geometry.arena.x[0]
    step = r.time_penalty + (1.0 - gamma) * r.success_bonus + r.effort_coef
    step += max(r.shaping_discount - gamma, 0.0) * r.progress_coef * d_max
    threshold = step / (p.a_max * p.dt)
    assert threshold == pytest.approx(48.44)  # 44.04 with exact shaping (D18)
    assert r.contact_impulse_coef >= threshold
    assert r.contact_step_coef > 0.0


def test_a_failed_reset_changes_nothing() -> None:
    """A reset that raises draws no random number: the next reset equals that of a fresh environment."""
    env, fresh = NavigationEnv(_config(), num_envs=2, seed=SEED), NavigationEnv(_config(), num_envs=2, seed=SEED)
    with pytest.raises(ValueError):
        env.reset(velocities=[[2.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(env.reset(), fresh.reset())


def test_actions_a_rounding_error_outside_the_square_are_clipped_into_it() -> None:
    env = NavigationEnv(_config(physics={"d_bar": 0.0}), num_envs=1, seed=SEED)
    env.reset()
    out = env.step([[1.0 + 5e-13, 0.0]])
    assert np.hypot(*out.info.u_cmd[0]) <= 2.5


def test_success_at_the_horizon_is_a_termination_not_a_truncation() -> None:
    env = NavigationEnv(_config(physics={"d_bar": 0.0}, task={"horizon": 1}), num_envs=2, seed=SEED)
    env.reset(positions=[[9.95, 0.0], [2.0, 0.0]], velocities=[[1.0, 0.0], [1.0, 0.0]])
    out = env.step(np.zeros((2, 2)))
    assert out.terminated.tolist() == [True, False] and out.truncated.tolist() == [False, True]


def test_final_observations_differ_from_the_next_ones_only_for_finished_episodes() -> None:
    env = NavigationEnv(_config(physics={"d_bar": 0.0}), num_envs=2, seed=SEED)
    env.reset(positions=[[9.95, 0.0], [2.0, 0.0]], velocities=[[1.0, 0.0], [1.0, 0.0]])
    out = env.step(np.zeros((2, 2)))
    assert out.terminated.tolist() == [True, False]
    np.testing.assert_array_equal(out.final_obs[1], out.obs[1])
    assert not np.array_equal(out.final_obs[0], out.obs[0])


def test_episode_statistics_are_the_sums_of_the_step_quantities() -> None:
    """R3, R4: over many episodes with resets, walls and disturbances, every statistic adds up the steps."""
    env = NavigationEnv(_config(physics={"d_bar": 1.0}, task={"horizon": 40}), num_envs=8, seed=SEED)
    env.reset()
    rng = np.random.default_rng(SEED)
    keys = ("episode_return", "unshaped_return", "contact_steps", "contact_events", "impulse", "wall_steps",
            "limiter_steps", "limiter_total", "saturation_steps", "saturation_total")
    sums = {key: np.zeros(8) for key in keys}
    previous_contact = np.zeros(8, dtype=bool)
    checked = 0
    for _ in range(400):
        u = np.where(rng.random((8, 1)) < 0.5, rng.uniform(-6.0, 6.0, (8, 2)), [[2.5, 2.5]])
        out = env.step_input(u)
        i = out.info
        limiter, saturation = i.limiter > 1e-6, i.saturation > 1e-6
        step = {
            "episode_return": out.reward, "unshaped_return": i.reward_unshaped, "contact_steps": i.contact,
            "contact_events": i.contact & ~previous_contact, "impulse": i.impulse, "wall_steps": i.wall_reaction,
            "limiter_steps": limiter, "limiter_total": np.where(limiter, i.limiter, 0.0),
            "saturation_steps": saturation, "saturation_total": np.where(saturation, i.saturation, 0.0),
        }
        for key in keys:
            sums[key] += step[key]
        previous_contact = i.contact.copy()
        for episode in out.episodes:
            for key in keys:
                assert getattr(episode, key) == pytest.approx(sums[key][episode.env], rel=1e-12, abs=1e-9), key
                sums[key][episode.env] = 0.0
            previous_contact[episode.env] = False
            checked += 1
    assert checked >= 60
    assert sum(sums["contact_events"]) >= 0  # the loop above checked every finished episode


def test_discounted_shaping_telescopes() -> None:
    """D13: with a shaping discount gamma the discounted terms of an episode sum to gamma^T Phi(p_T) - Phi(p_0)."""
    config = _config(physics={"d_bar": 1.0}, task={"horizon": 30}, reward={"shaping_discount": 0.99})
    gamma = config.reward.shaping_discount
    env = NavigationEnv(config, num_envs=4, seed=SEED)
    env.reset()
    start = env.positions[:, 0].copy()
    rng = np.random.default_rng(SEED)
    shaping = np.zeros(4)
    for k in range(30):
        out = env.step(rng.uniform(-0.2, 1.0, (4, 2)))
        shaping += gamma**k * (out.reward - out.info.reward_unshaped)
    end = (out.final_obs[:, 0] + 1.0) * 6.0 - 1.0
    np.testing.assert_allclose(shaping, gamma**30 * _phi(end, config) - _phi(start, config), atol=1e-9)
