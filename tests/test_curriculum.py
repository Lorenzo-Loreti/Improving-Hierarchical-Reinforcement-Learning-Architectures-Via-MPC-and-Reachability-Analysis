"""Tests of the reverse spawn curriculum (decision log D62): the bands, the sampling in the free space, the
advancement rule, the metrics, and its place in the environment."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from hrlmpc.config import Box, CurriculumConfig, EnvConfig, load_env_config
from hrlmpc.curriculum import SpawnCurriculum
from hrlmpc.env import NavigationEnv
from hrlmpc.geometry import Layout

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "env"


def _config(name: str = "slalom_sparse") -> EnvConfig:
    return load_env_config(CONFIG_DIR / f"{name}.yaml")


def _curriculum(name: str = "slalom_sparse", **changes: float) -> SpawnCurriculum:
    config = _config(name)
    settings = config.task.curriculum
    assert settings is not None
    settings = dataclasses.replace(settings, **changes)
    return SpawnCurriculum(settings, config.task.spawn, Layout.from_config(config.geometry))


# ------------------------------------------------------------------------------- bands


def test_the_bands_slide_from_the_goal_to_the_spawn_box() -> None:
    """D62: x_start 8, step 1 m and the spawn box [0, 2] x [-1, 1] give nine bands, the last the spawn box."""
    cur = _curriculum()
    assert cur.stages == 9 and cur.stage == 0 and not cur.final
    assert cur.band == Box(x=(8.0, 10.0), y=(-1.0, 1.0))
    bands = []
    for _ in range(cur.stages - 1):
        cur.advance()
        bands.append(cur.band)
    assert cur.final and cur.stage == 8
    assert [b.x for b in bands] == [(7.0, 9.0), (6.0, 8.0), (5.0, 7.0), (4.0, 6.0), (3.0, 5.0), (2.0, 4.0),
                                    (1.0, 3.0), (0.0, 2.0)]
    assert cur.band == _config().task.spawn
    cur.advance()  # never beyond the last stage
    assert cur.stage == 8 and cur.band == _config().task.spawn


def test_a_step_that_does_not_divide_the_distance_clamps_the_last_band() -> None:
    cur = _curriculum(x_start=7.5, step=2.0)  # 7.5, 5.5, 3.5, 1.5, then 0
    assert cur.stages == 5
    xs = []
    for _ in range(cur.stages):
        xs.append(cur.band.x)
        cur.advance()
    assert xs == [(7.5, 9.5), (5.5, 7.5), (3.5, 5.5), (1.5, 3.5), (0.0, 2.0)]


def test_a_curriculum_that_starts_at_the_spawn_box_has_one_stage() -> None:
    cur = _curriculum(x_start=0.0)
    assert cur.stages == 1 and cur.final and cur.band == _config().task.spawn


def test_a_band_with_too_few_free_points_is_refused() -> None:
    """Every band must be free for at least 1 % of its area: a spawn box as wide as the lower block of gate 1 and a
    band on it (free fraction 0), or one whose free part is a sliver of 0.5 % (x in [3.995, 4])."""
    config = _config()
    layout = Layout.from_config(config.geometry)
    spawn = Box(x=(0.0, 1.0), y=(-1.9, -0.5))
    with pytest.raises(ValueError, match="free fraction"):
        SpawnCurriculum(CurriculumConfig(x_start=4.0, step=1.0, window=50, threshold=0.8), spawn, layout)
    with pytest.raises(ValueError, match="free fraction"):
        SpawnCurriculum(CurriculumConfig(x_start=3.995, step=1.0, window=50, threshold=0.8), spawn, layout)
    cur = SpawnCurriculum(CurriculumConfig(x_start=3.9, step=1.0, window=50, threshold=0.8), spawn, layout)
    assert 0.09 < cur.free_fraction(cur.band) < 0.11  # the sliver x in [3.9, 4]: 10 % of the band
    for band in (Box(x=(8.0, 10.0), y=(-1.0, 1.0)), Box(x=(0.0, 2.0), y=(-1.0, 1.0))):
        assert _curriculum().free_fraction(band) == 1.0
    assert 0.66 < _curriculum().free_fraction(Box(x=(7.0, 9.0), y=(-1.0, 1.0))) < 0.72  # gate 2 blocks 5/16


def test_the_last_band_is_the_spawn_box_exactly_despite_rounding() -> None:
    """x_start 7.9 and step 0.3 do not divide: the clamp alone would leave the last band at 0.1 + 1 ulp."""
    config = _config()
    layout = Layout.from_config(config.geometry)
    spawn = Box(x=(0.1, 2.1), y=(-1.0, 1.0))
    cur = SpawnCurriculum(CurriculumConfig(x_start=7.9, step=0.3, window=50, threshold=0.8), spawn, layout)
    assert cur.stages == 27
    for _ in range(cur.stages - 1):
        cur.advance()
    assert cur.band == spawn and cur.band.x == (0.1, 2.1)


# ---------------------------------------------------------------------------- sampling


def test_samples_are_free_within_the_band_and_short_of_the_goal() -> None:
    cur = _curriculum()
    layout = Layout.from_config(_config().geometry)
    rng = np.random.default_rng(1)
    for _ in range(cur.stages):
        points = cur.sample(rng, 400)
        band = cur.band
        assert points.shape == (400, 2)
        assert np.all(points[:, 0] >= band.x[0]) and np.all(points[:, 0] < band.x[1])
        assert np.all(points[:, 1] >= band.y[0]) and np.all(points[:, 1] <= band.y[1])
        assert layout.in_free_space(points).all() and not layout.in_goal(points).any()
        cur.advance()


def test_sampling_rejects_the_blocked_part_of_a_band() -> None:
    """The band [7, 9] crosses the upper block of gate 2 (x in [7, 8], y >= -0.25): those points never appear,
    and the free part is covered."""
    cur = _curriculum()
    cur.advance()
    assert cur.band.x == (7.0, 9.0)
    points = cur.sample(np.random.default_rng(2), 2000)
    in_gate = (points[:, 0] >= 7.0) & (points[:, 0] <= 8.0)
    assert in_gate.any() and np.all(points[in_gate, 1] < -0.25)
    assert np.any(points[in_gate, 1] < -0.9) and np.any(~in_gate & (points[:, 1] > 0.9))


def test_excluded_regions_are_never_sampled_and_lower_the_free_fraction() -> None:
    """D69: the starts skip the excluded boxes (the cavity of the trap's U); the free fraction of a band counts
    them as blocked, and the last band is still the spawn box."""
    cur = _curriculum("trap_sparse")
    config = _config("trap_sparse")
    assert config.task.curriculum is not None and config.task.curriculum.exclude == (
        Box(x=(3.75, 5.25), y=(-0.8, 0.8)),
    )
    layout = Layout.from_config(config.geometry)
    rng = np.random.default_rng(5)
    for _ in range(cur.stages):
        points = cur.sample(rng, 600)
        assert layout.in_free_space(points).all()
        # between x = 3.75 and 5.75 the band's y-range [-1, 1] is cavity (excluded), arm or back wall
        assert not ((points[:, 0] > 3.75) & (points[:, 0] < 5.75)).any()
        cur.advance()
    assert cur.final and cur.band == config.task.spawn
    fresh = _curriculum("trap_sparse")
    assert 0.08 < fresh.free_fraction(Box(x=(4.0, 6.0), y=(-1.0, 1.0))) < 0.16  # x in [5.75, 6] only
    assert 0.33 < fresh.free_fraction(Box(x=(3.0, 5.0), y=(-1.0, 1.0))) < 0.45  # x in [3, 3.75] only
    without = _curriculum("trap_sparse", exclude=())
    assert 0.57 < without.free_fraction(Box(x=(4.0, 6.0), y=(-1.0, 1.0))) < 0.68  # the cavity counts as free


def test_a_band_that_is_all_excluded_is_refused_unless_it_is_the_spawn_box() -> None:
    """The band [4, 6] of the trap is entirely excluded; the construction refuses it, as it refuses a band of
    too few free points, unless it is the last band (the spawn box, never excluded)."""
    config = _config("trap_sparse")
    assert config.task.curriculum is not None
    layout = Layout.from_config(config.geometry)
    settings = dataclasses.replace(config.task.curriculum, x_start=3.75, step=3.75)  # bands [3.75, 5.75], [0, 2]
    with pytest.raises(ValueError, match="free fraction"):
        SpawnCurriculum(settings, config.task.spawn, layout)


def test_sampling_is_deterministic_in_the_generator() -> None:
    a, b = _curriculum(), _curriculum()
    np.testing.assert_array_equal(a.sample(np.random.default_rng(7), 20), b.sample(np.random.default_rng(7), 20))
    assert a.sample(np.random.default_rng(7), 0).shape == (0, 2)
    with pytest.raises(ValueError, match="count"):
        a.sample(np.random.default_rng(7), -1)


def test_sampling_gives_up_after_the_draw_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A band whose free part vanishes after construction (here: a layout swapped under the curriculum) ends in an
    error, not an endless loop."""
    import hrlmpc.curriculum as module

    cur = _curriculum()
    monkeypatch.setattr(module, "MAX_DRAWS", 64)
    monkeypatch.setattr(cur, "_free", lambda points: np.zeros(len(points), dtype=bool))
    with pytest.raises(RuntimeError, match="after 64 draws"):
        cur.sample(np.random.default_rng(0), 1)


# -------------------------------------------------------------------------- advancement


def test_the_stage_advances_when_a_block_meets_the_threshold() -> None:
    """Window 50, threshold 0.8: a block of 50 episodes of the stage with 40 successes advances it once."""
    cur = _curriculum()
    for _ in range(10):
        cur.record(0, False)
    for _ in range(39):
        cur.record(0, True)
    assert cur.stage == 0 and cur.window_success == pytest.approx(39 / 49)
    cur.record(0, True)  # the 50th episode: 40 of 50
    assert cur.stage == 1 and math.isnan(cur.window_success)  # the block restarts with the stage


def test_a_block_below_the_threshold_starts_a_new_block() -> None:
    """The outcomes do not slide: 11 failures then 39 successes make a block of 50 at 0.78, which fails, and the
    next block starts empty; a sliding window would have advanced at the 51st episode."""
    cur = _curriculum()
    for _ in range(11):
        cur.record(0, False)
    for _ in range(39):
        cur.record(0, True)
    assert cur.stage == 0 and math.isnan(cur.window_success)
    cur.record(0, True)
    assert cur.stage == 0 and cur.window_success == 1.0
    for _ in range(49):
        cur.record(0, True)
    assert cur.stage == 1


def test_episodes_of_earlier_stages_and_explicit_starts_do_not_count() -> None:
    cur = _curriculum()
    for _ in range(50):
        cur.record(0, True)
    assert cur.stage == 1
    for _ in range(50):
        cur.record(0, True)  # episodes started in stage 0 that finish now
    for _ in range(50):
        cur.record(-1, True)  # episodes started from explicit positions (an evaluation)
    assert cur.stage == 1 and math.isnan(cur.window_success)


def test_the_stage_never_regresses_or_passes_the_last() -> None:
    cur = _curriculum()
    for stage in range(cur.stages):
        for _ in range(50):
            cur.record(stage, True)
    assert cur.final and cur.stage == cur.stages - 1
    for _ in range(100):
        cur.record(cur.stage, False)
    assert cur.stage == cur.stages - 1


def test_in_the_last_stage_the_window_slides_for_the_metric() -> None:
    """From the spawn box nothing advances; ``window_success`` is the success rate of the last 50 episodes and
    never restarts (the training success rate from the spawn box, for the logs)."""
    cur = _curriculum()
    for _ in range(8):
        cur.advance()
    assert cur.final
    for _ in range(120):
        cur.record(cur.stage, True)
    assert cur.window_success == 1.0
    for _ in range(25):
        cur.record(cur.stage, False)
    assert cur.window_success == pytest.approx(0.5) and cur.stage == 8


def test_metrics_describe_the_stage() -> None:
    cur = _curriculum()
    m = cur.metrics()
    assert m["curriculum/stage"] == 0.0 and m["curriculum/stages"] == 9.0 and m["curriculum/x_min"] == 8.0
    assert m["curriculum/final"] == 0.0 and math.isnan(m["curriculum/window_success"])
    cur.record(0, True)
    assert cur.metrics()["curriculum/window_success"] == 1.0
    for _ in range(8):
        cur.advance()
    m = cur.metrics()
    assert m["curriculum/stage"] == 8.0 and m["curriculum/x_min"] == 0.0 and m["curriculum/final"] == 1.0


# -------------------------------------------------------------------- the environment


def test_the_environment_starts_from_the_band_and_feeds_the_curriculum() -> None:
    """With the curriculum, resets draw from the current band; finished episodes advance it; later resets come
    from the next band. A scripted thrust along +x from [8, 10] succeeds at once."""
    config = _config()
    env = NavigationEnv(config, num_envs=8, seed=3)
    assert env.curriculum is not None and env.curriculum.stage == 0
    env.reset()
    assert np.all(env.positions[:, 0] >= 8.0) and np.all(env.positions[:, 0] < 10.0)
    thrust = np.tile([config.physics.a_max, 0.0], (8, 1))
    finished = 0
    while finished < 50:
        out = env.step_input(thrust)
        finished += len(out.episodes)
        assert all(ep.success for ep in out.episodes)
    assert env.curriculum.stage == 1
    env.reset()
    assert np.all(env.positions[:, 0] >= 7.0) and np.all(env.positions[:, 0] < 9.0)
    assert env.curriculum_metrics()["curriculum/stage"] == 1.0


def test_the_environment_without_a_curriculum_is_unchanged() -> None:
    env = NavigationEnv(_config("slalom"), num_envs=4, seed=0)
    assert env.curriculum is None and env.curriculum_metrics() == {}
    env.reset()
    assert np.all(env.positions[:, 0] <= 2.0)


def test_explicit_starts_bypass_the_curriculum() -> None:
    """Explicit positions are not drawn from the band and their episodes do not count for the stage."""
    config = _config()
    env = NavigationEnv(config, num_envs=2, seed=0, autoreset=False)
    env.reset(positions=np.array([[9.5, 0.0], [9.6, 0.0]]))
    thrust = np.tile([config.physics.a_max, 0.0], (2, 1))
    out = env.step_input(thrust)
    while not out.episodes:
        out = env.step_input(thrust)
    assert env.curriculum is not None and env.curriculum.stage == 0 and math.isnan(env.curriculum.window_success)


def test_the_evaluation_ignores_the_curriculum() -> None:
    """``evaluate`` resets with explicit starts and automatic resets: its environment has no curriculum, so the
    agents that finish early restart in the spawn box, as before D62, and nothing is recorded."""
    from hrlmpc.evaluation import evaluate

    config = _config("tunnel_sparse")
    starts = np.array([[9.5, 0.0], [1.0, 0.0]])  # the first finishes within a few steps, the second much later

    class Thrust:
        inputs = True

        def __call__(self, obs: np.ndarray) -> np.ndarray:
            return np.tile([config.physics.a_max, 0.0], (len(obs), 1))

    result = evaluate(Thrust(), config, starts, seed=0)
    assert [e.success for e in result.episodes] == [True, True]
    assert result.episodes[0].length < result.episodes[1].length
    env = NavigationEnv(dataclasses.replace(config, task=dataclasses.replace(config.task, curriculum=None)), 2, seed=0)
    assert env.curriculum is None
    plain_env = NavigationEnv(config, num_envs=2, seed=0, autoreset=True)
    plain_env.reset(positions=starts)
    thrust = np.tile([config.physics.a_max, 0.0], (2, 1))
    out = plain_env.step_input(thrust)
    while not out.episodes:
        out = plain_env.step_input(thrust)
    assert plain_env.curriculum is not None and plain_env.positions[0, 0] >= 8.0  # the training env would use the band


def test_the_plain_sampling_is_unchanged_without_a_curriculum() -> None:
    """Without a curriculum the starts are the spawn box's uniform draws of each agent's own generator, as before."""
    config = _config("slalom")
    env = NavigationEnv(config, num_envs=3, seed=5)
    env.reset()
    spawn = config.task.spawn
    lo, hi = np.array([spawn.x[0], spawn.y[0]]), np.array([spawn.x[1], spawn.y[1]])
    expected = []
    for child in np.random.SeedSequence(5).spawn(3):
        first, _ = child.spawn(2)
        expected.append(np.random.default_rng(first).uniform(lo, hi))
    np.testing.assert_array_equal(env.positions, np.array(expected))


def test_starts_with_a_curriculum_do_not_depend_on_the_batch_size() -> None:
    small, large = NavigationEnv(_config(), num_envs=2, seed=4), NavigationEnv(_config(), num_envs=6, seed=4)
    small.reset()
    large.reset()
    np.testing.assert_array_equal(small.positions, large.positions[:2])


def test_starts_are_reproducible_for_a_seed() -> None:
    a, b = NavigationEnv(_config(), num_envs=4, seed=11), NavigationEnv(_config(), num_envs=4, seed=11)
    a.reset()
    b.reset()
    np.testing.assert_array_equal(a.positions, b.positions)
    c = NavigationEnv(_config(), num_envs=4, seed=12)
    c.reset()
    assert not np.array_equal(a.positions, c.positions)
