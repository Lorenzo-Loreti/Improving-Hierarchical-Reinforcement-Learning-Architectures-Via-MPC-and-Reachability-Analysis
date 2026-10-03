"""Tests for the spawn box and the start samplers (envs/spawn_sampler.py), and
for how SlalomVecEnv uses them. scenarios/tunnel/envs/tests/
test_spawn_sampler.py is the same suite for the tunnel."""

import warnings

import numpy as np
import pytest
from scipy.stats import kstest

from envs.config import SlalomEnvConfig
from envs.slalom_env import SlalomEnv
from envs.spawn_sampler import SobolSpawnStream, spawn_box
from envs.vec_slalom_env import SlalomVecEnv
from envs.width_profile import slalom_profile


def _config(**overrides):
    return SlalomEnvConfig(width_profile=slalom_profile(), **overrides)


# --------------------------------------------------------------------------
# the spawn box
# --------------------------------------------------------------------------

def test_spawn_box_is_px_0_to_2_and_py_within_a_quarter_width():
    low, high = spawn_box(4.0)
    np.testing.assert_array_equal(low, [0.0, -1.0])
    np.testing.assert_array_equal(high, [2.0, 1.0])


def test_every_env_reads_the_same_spawn_box():
    config = _config()
    env, vec_env = SlalomEnv(config=config), SlalomVecEnv(num_envs=2, config=config)
    for e in (env, vec_env):
        np.testing.assert_array_equal(e.spawn_low, spawn_box(config.tunnel_width)[0])
        np.testing.assert_array_equal(e.spawn_high, spawn_box(config.tunnel_width)[1])


# --------------------------------------------------------------------------
# SobolSpawnStream
# --------------------------------------------------------------------------

def test_sobol_stream_is_fixed_by_its_seed():
    a = SobolSpawnStream(seed=3).take(64)
    b = SobolSpawnStream(seed=3).take(64)
    c = SobolSpawnStream(seed=4).take(64)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


@pytest.mark.parametrize("start", [0, 32, 96])
def test_every_aligned_block_of_32_sobol_points_is_a_0_5_2_net(start):
    """The (0, 2)-sequence property the sampler is chosen for: every aligned
    block of 2^5 consecutive points has exactly one point in every dyadic
    rectangle of area 2^-5, i.e. in every cell of the 1 x 32, 2 x 16, 4 x 8,
    8 x 4, 16 x 2 and 32 x 1 grids. Scrambling must preserve it."""
    stream = SobolSpawnStream(seed=11)
    stream.take(start)
    block = stream.take(32)
    for i in range(6):
        nx, ny = 2 ** i, 2 ** (5 - i)
        cells = set(zip((block[:, 0] * nx).astype(int), (block[:, 1] * ny).astype(int)))
        assert len(cells) == 32, f"{nx} x {ny} grid: {32 - len(cells)} cells empty"


def test_each_sobol_point_is_uniform_over_the_scrambling():
    """What keeps the sampler unbiased: over the random scrambling, any one
    point of the stream (here the first and the fifth) is uniform on the unit
    square. One-sample Kolmogorov-Smirnov test per coordinate over 2 000
    seeds."""
    for index in (0, 4):
        points = np.array([SobolSpawnStream(seed=s, first_block=8).take(index + 1)[index]
                           for s in range(2000)])
        for coord in range(2):
            assert kstest(points[:, coord], "uniform").pvalue > 1e-3


def test_sobol_stream_continues_past_its_first_block_without_warnings():
    """Draws past the first block extend the same sequence, doubling the
    total so it stays a power of 2 (scipy warns otherwise)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        small = SobolSpawnStream(seed=5, first_block=4)
        chunks = [small.take(3), small.take(10), small.take(1), small.take(20)]
    np.testing.assert_array_equal(np.concatenate(chunks), SobolSpawnStream(seed=5).take(34))


def test_sobol_stream_first_block_must_be_a_power_of_two():
    with pytest.raises(ValueError):
        SobolSpawnStream(seed=0, first_block=12)


# --------------------------------------------------------------------------
# the config and the environments
# --------------------------------------------------------------------------

def test_config_defaults_to_independent_draws_and_rejects_unknown_samplers():
    assert SlalomEnvConfig().init_sampler == "uniform"
    with pytest.raises(ValueError):
        SlalomEnvConfig(init_sampler="halton")


def test_vec_env_uniform_sampler_is_the_original_independent_draw():
    """The default reproduces the draw every run before 2026-10-03 made:
    all p_x0 from the seeded generator, then all p_y0."""
    vec_env = SlalomVecEnv(num_envs=8, config=_config())
    obs, _ = vec_env.reset(seed=3)
    rng = np.random.default_rng(3)
    np.testing.assert_array_equal(obs[:, 0], rng.uniform(0.0, 2.0, size=8).astype(np.float32))
    np.testing.assert_array_equal(obs[:, 1], rng.uniform(-1.0, 1.0, size=8).astype(np.float32))


def test_vec_env_sobol_starts_follow_the_stream_in_env_index_order():
    """reset() takes the stream's first num_envs points; every auto-reset
    takes the next ones, the finished environments in ascending index. With
    max_steps=1 every environment finishes on every step."""
    num_envs = 8
    vec_env = SlalomVecEnv(num_envs=num_envs, config=_config(init_sampler="sobol", max_steps=1))
    obs, _ = vec_env.reset(seed=7)
    expected = SobolSpawnStream(seed=7).take(4 * num_envs)
    low, high = vec_env.spawn_low, vec_env.spawn_high

    def mapped(u):
        return np.stack([(low[0] + u[:, 0] * (high[0] - low[0])).astype(np.float32),
                         (low[1] + u[:, 1] * (high[1] - low[1])).astype(np.float32)], axis=1)

    np.testing.assert_array_equal(obs[:, :2], mapped(expected[:num_envs]))
    np.testing.assert_array_equal(obs[:, 2:], 0.0)
    for k in range(1, 4):
        obs, _, terminated, truncated, _ = vec_env.step(np.zeros((num_envs, 2), dtype=np.float32))
        assert np.all(terminated | truncated)
        np.testing.assert_array_equal(obs[:, :2], mapped(expected[k * num_envs:(k + 1) * num_envs]))


def test_vec_env_sobol_starts_stay_in_the_box_at_rest():
    vec_env = SlalomVecEnv(num_envs=8, config=_config(init_sampler="sobol", max_steps=1))
    obs, _ = vec_env.reset(seed=0)
    starts = [obs]
    for _ in range(200):
        obs, *_ = vec_env.step(np.zeros((8, 2), dtype=np.float32))
        starts.append(obs)
    starts = np.concatenate(starts)
    assert np.all(starts[:, :2] >= vec_env.spawn_low.astype(np.float32))
    assert np.all(starts[:, :2] <= vec_env.spawn_high.astype(np.float32))
    np.testing.assert_array_equal(starts[:, 2:], 0.0)


def test_vec_env_sobol_sampler_never_draws_from_the_disturbance_generator():
    """Under "sobol" a reset leaves the generator the disturbance comes from
    untouched, so that generator holds the disturbance sequence alone."""
    vec_env = SlalomVecEnv(num_envs=8, config=_config(init_sampler="sobol"))
    vec_env.reset(seed=1)
    before = vec_env._np_random.bit_generator.state
    vec_env._sample_initial(np.ones(8, dtype=bool))
    assert vec_env._np_random.bit_generator.state == before


def test_plain_env_ignores_the_sampler():
    """Evaluation's reseeded resets draw the same independent starts under
    either sampler, so both arms of a comparison are evaluated alike."""
    for seed in range(5):
        a, _ = SlalomEnv(config=_config()).reset(seed=seed)
        b, _ = SlalomEnv(config=_config(init_sampler="sobol")).reset(seed=seed)
        np.testing.assert_array_equal(a, b)
