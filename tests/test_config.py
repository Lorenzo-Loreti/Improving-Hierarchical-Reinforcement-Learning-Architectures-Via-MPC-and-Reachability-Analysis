"""Tests of the YAML environment configuration (code rule C2; decision log D9, D11-D14, D45, D51)."""

from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from hrlmpc.config import Box, CurriculumConfig, env_config_from_dict, env_config_to_dict, load_env_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "env"


def _slalom_dict() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((CONFIG_DIR / "slalom.yaml").read_text(encoding="utf-8"))
    return data


def test_slalom_config_matches_the_decision_log() -> None:
    cfg = load_env_config(CONFIG_DIR / "slalom.yaml")
    p = cfg.physics
    assert (p.dt, p.a_max, p.v_max, p.gamma, p.gamma_w, p.d_bar) == (0.1, 2.5, 1.2, 0.5, 1.0, 0.0)
    assert cfg.geometry.arena == Box(x=(-1.0, 11.0), y=(-2.0, 2.0))
    assert cfg.geometry.obstacles == (
        Box(x=(4.0, 5.0), y=(-2.0, 0.25)),
        Box(x=(4.0, 5.0), y=(1.75, 2.0)),
        Box(x=(7.0, 8.0), y=(-2.0, -1.75)),
        Box(x=(7.0, 8.0), y=(-0.25, 2.0)),
    )
    assert cfg.geometry.goal_x == 10.0
    assert cfg.task.spawn == Box(x=(0.0, 2.0), y=(-1.0, 1.0))
    assert cfg.task.horizon == 200
    r = cfg.reward
    assert (
        r.time_penalty,
        r.success_bonus,
        r.progress_coef,
        r.shaping_discount,
        r.effort_coef,
        r.contact_impulse_coef,
        r.contact_step_coef,
    ) == (1.0, 1000.0, 10.0, 1.0, 0.01, 50.0, 1.0)  # shaping discount 1: D22


def test_tunnel_config_is_the_slalom_without_obstacles() -> None:
    slalom = load_env_config(CONFIG_DIR / "slalom.yaml")
    tunnel = load_env_config(CONFIG_DIR / "tunnel.yaml")
    assert tunnel.geometry.obstacles == ()
    assert tunnel.geometry.arena == slalom.geometry.arena
    assert tunnel.geometry.goal_x == slalom.geometry.goal_x
    assert (tunnel.physics, tunnel.task, tunnel.reward) == (slalom.physics, slalom.task, slalom.reward)


@pytest.mark.parametrize(
    ("layout", "level", "d_bar"), [("slalom", "d05", 0.5), ("slalom", "d10", 1.0), ("tunnel", "d05", 0.5),
                                   ("tunnel", "d10", 1.0)]
)
def test_each_disturbance_level_has_its_own_files(layout: str, level: str, d_bar: float) -> None:
    """D45: the file of a level is its layout's, with only the radius of the disturbance set changed."""
    base = load_env_config(CONFIG_DIR / f"{layout}.yaml")
    cfg = load_env_config(CONFIG_DIR / f"{layout}_{level}.yaml")
    assert base.physics.d_bar == 0.0
    assert cfg == dataclasses.replace(base, physics=dataclasses.replace(base.physics, d_bar=d_bar))


@pytest.mark.parametrize(
    ("layout", "level"),
    [("slalom", ""), ("slalom", "_d05"), ("slalom", "_d10"), ("tunnel", ""), ("tunnel", "_d05"), ("tunnel", "_d10")],
)
def test_sparse_reward_files_drop_the_progress_term_and_add_the_curriculum(layout: str, level: str) -> None:
    """D51, D62: the sparse file of a layout and level is its shaped file with the progress coefficient set to 0
    and the reverse spawn curriculum of D62; nothing else differs."""
    shaped = load_env_config(CONFIG_DIR / f"{layout}{level}.yaml")
    sparse = load_env_config(CONFIG_DIR / f"{layout}_sparse{level}.yaml")
    assert shaped.reward.progress_coef == 10.0 and shaped.task.curriculum is None
    curriculum = CurriculumConfig(x_start=8.0, step=1.0, window=50, threshold=0.8)
    expected = dataclasses.replace(
        shaped,
        reward=dataclasses.replace(shaped.reward, progress_coef=0.0),
        task=dataclasses.replace(shaped.task, curriculum=curriculum),
    )
    assert sparse == expected


# ------------------------------------------------------------ the spawn curriculum (D62)


def test_the_curriculum_is_optional_and_round_trips() -> None:
    data = _slalom_dict()
    assert env_config_from_dict(data).task.curriculum is None
    data["task"]["curriculum"] = None
    assert env_config_from_dict(data).task.curriculum is None
    data["task"]["curriculum"] = {"x_start": 8.0, "step": 1.0, "window": 50, "threshold": 0.8}
    cfg = env_config_from_dict(data)
    assert cfg.task.curriculum == CurriculumConfig(8.0, 1.0, 50, 0.8)
    assert cfg.task.curriculum.exclude == ()  # no excluded region unless listed (D69)
    plain = env_config_to_dict(cfg)
    assert plain["task"]["curriculum"] == {"x_start": 8.0, "step": 1.0, "window": 50, "threshold": 0.8,
                                           "exclude": [], "y": None}
    assert env_config_from_dict(plain) == cfg
    assert env_config_to_dict(load_env_config(CONFIG_DIR / "slalom.yaml"))["task"]["curriculum"] is None


def test_the_curriculum_may_exclude_regions_from_the_training_starts() -> None:
    """D69: `exclude` lists boxes (dead ends) from which no training start is drawn; they round-trip, may not
    overlap the spawn box (the last band and the evaluation's starts) and must be well-formed."""
    data = _slalom_dict()
    data["task"]["curriculum"] = {"x_start": 8.0, "step": 1.0, "window": 50, "threshold": 0.8,
                                  "exclude": [{"x": [3.5, 5.5], "y": [-0.8, 0.8]}, {"x": [6.0, 6.5], "y": [0.0, 1.0]}]}
    cfg = env_config_from_dict(data)
    assert cfg.task.curriculum is not None
    assert cfg.task.curriculum.exclude == (Box(x=(3.5, 5.5), y=(-0.8, 0.8)), Box(x=(6.0, 6.5), y=(0.0, 1.0)))
    plain = env_config_to_dict(cfg)
    assert plain["task"]["curriculum"]["exclude"] == [{"x": [3.5, 5.5], "y": [-0.8, 0.8]},
                                                      {"x": [6.0, 6.5], "y": [0.0, 1.0]}]
    assert env_config_from_dict(plain) == cfg
    data["task"]["curriculum"]["exclude"] = []
    assert env_config_from_dict(data).task.curriculum.exclude == ()
    for bad, message in (
        ([{"x": [1.5, 3.0], "y": [-0.5, 0.5]}], "spawn"),  # overlaps the spawn box [0, 2] x [-1, 1]
        ([{"x": [3.5, 3.5], "y": [-0.8, 0.8]}], "x"),  # degenerate
        ([{"x": [3.5, 5.5]}], "missing"),
        ({"x": [3.5, 5.5], "y": [-0.8, 0.8]}, "list"),  # not a list of boxes
        ([{"x": [12.0, 13.0], "y": [-0.8, 0.8]}], "arena"),  # outside the arena
    ):
        data["task"]["curriculum"]["exclude"] = bad
        with pytest.raises(ValueError, match=message):
            env_config_from_dict(data)


def test_the_curriculum_may_set_the_bands_height() -> None:
    """D69: `y` gives the bands' y-range (the spawn box's by default); it must lie inside the arena and contain
    the spawn box's y-range, so that the last band, the spawn box, is no surprise."""
    data = _slalom_dict()
    data["task"]["curriculum"] = {"x_start": 8.0, "step": 1.0, "window": 50, "threshold": 0.8, "y": [-2.0, 2.0]}
    cfg = env_config_from_dict(data)
    assert cfg.task.curriculum is not None and cfg.task.curriculum.y == (-2.0, 2.0)
    plain = env_config_to_dict(cfg)
    assert plain["task"]["curriculum"]["y"] == [-2.0, 2.0] and env_config_from_dict(plain) == cfg
    del data["task"]["curriculum"]["y"]
    cfg = env_config_from_dict(data)
    assert cfg.task.curriculum is not None and cfg.task.curriculum.y is None
    assert env_config_to_dict(cfg)["task"]["curriculum"]["y"] is None
    for bad, message in (
        ([-2.5, 2.0], "arena"),
        ([-0.5, 2.0], "spawn"),  # does not contain the spawn box's [-1, 1]
        ([2.0, -2.0], "y"),
        ("tall", "list"),
    ):
        data["task"]["curriculum"]["y"] = bad
        with pytest.raises(ValueError, match=message):
            env_config_from_dict(data)


@pytest.mark.parametrize(
    "curriculum, message",
    [
        ({"x_start": 8.0, "step": 0.0, "window": 50, "threshold": 0.8}, "step"),
        ({"x_start": 8.0, "step": 1.0, "window": 0, "threshold": 0.8}, "window"),
        ({"x_start": 8.0, "step": 1.0, "window": 50, "threshold": 0.0}, "threshold"),
        ({"x_start": 8.0, "step": 1.0, "window": 50, "threshold": 1.5}, "threshold"),
        ({"x_start": 8.0, "step": 1.0, "window": 50.0, "threshold": 0.8}, "integer"),
        ({"x_start": 8.5, "step": 1.0, "window": 50, "threshold": 0.8}, "goal"),  # first band [8.5, 10.5]
        ({"x_start": 10.5, "step": 1.0, "window": 50, "threshold": 0.8}, "goal"),  # beyond the arena too
        ({"x_start": -0.5, "step": 1.0, "window": 50, "threshold": 0.8}, "spawn"),  # behind the spawn box
        ({"x_start": 8.0, "step": 2.5, "window": 50, "threshold": 0.8}, "gap"),  # step above the width of 2
        ({"x_start": 8.0, "step": 1.0, "threshold": 0.8}, "missing"),
        ({"x_start": 8.0, "step": 1.0, "window": 50, "threshold": 0.8, "extra": 1}, "unknown"),
    ],
)
def test_invalid_curricula_are_rejected(curriculum: dict[str, Any], message: str) -> None:
    data = _slalom_dict()
    data["task"]["curriculum"] = curriculum
    with pytest.raises(ValueError, match=message):
        env_config_from_dict(data)


def test_the_first_band_may_touch_the_goal_line_and_the_step_may_equal_the_width() -> None:
    data = _slalom_dict()
    data["task"]["curriculum"] = {"x_start": 8.0, "step": 2.0, "window": 50, "threshold": 0.8}
    cfg = env_config_from_dict(data)  # [8, 10] ends on the goal line; bands [8,10], [6,8], ... touch
    assert cfg.task.curriculum is not None and cfg.task.curriculum.step == 2.0
    data["task"]["curriculum"]["x_start"] = 0.0  # a one-stage curriculum: the spawn box itself
    assert env_config_from_dict(data).task.curriculum is not None


def test_a_yaml_file_with_curriculum_null_loads_without_one(tmp_path: Path) -> None:
    data = _slalom_dict()
    data["task"]["curriculum"] = None
    path = tmp_path / "env.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_env_config(path).task.curriculum is None
    assert load_env_config(path) == load_env_config(CONFIG_DIR / "slalom.yaml")


@pytest.mark.parametrize("d_bar", [0.0, 0.5, 1.0])
def test_every_disturbance_level_of_d9_is_valid(d_bar: float) -> None:
    data = _slalom_dict()
    data["physics"]["d_bar"] = d_bar
    assert env_config_from_dict(data).physics.d_bar == d_bar


def test_round_trip_through_a_dict_and_yaml() -> None:
    cfg = load_env_config(CONFIG_DIR / "slalom.yaml")
    as_dict = env_config_to_dict(cfg)
    assert env_config_from_dict(as_dict) == cfg
    assert env_config_from_dict(yaml.safe_load(yaml.safe_dump(as_dict))) == cfg


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("physics", "gamma", 10.0, r"gamma\*dt"),
        ("physics", "gamma", 2.5, "F1"),
        ("physics", "d_bar", 3.2, "F2"),
        ("physics", "dt", True, "must be a number"),
        ("reward", "shaping_discount", 1.5, "shaping_discount"),
        ("reward", "effort_coef", -0.01, "non-negative"),
        ("task", "horizon", 0, "horizon"),
        ("task", "horizon", 200.5, "must be an integer"),
        ("geometry", "goal_x", 11.0, "goal_x"),
    ],
)
def test_invalid_values_are_rejected(section: str, key: str, value: Any, message: str) -> None:
    data = _slalom_dict()
    data[section][key] = value
    with pytest.raises(ValueError, match=message):
        env_config_from_dict(data)


def test_unknown_key_is_rejected() -> None:
    data = _slalom_dict()
    data["physics"]["gama"] = 0.5
    with pytest.raises(ValueError, match="unknown keys"):
        env_config_from_dict(data)


def test_missing_key_is_rejected() -> None:
    data = _slalom_dict()
    del data["reward"]["time_penalty"]
    with pytest.raises(ValueError, match="missing keys"):
        env_config_from_dict(data)


def test_obstacle_outside_the_arena_is_rejected() -> None:
    data = _slalom_dict()
    data["geometry"]["obstacles"].append({"x": [10.5, 12.0], "y": [0.0, 1.0]})
    with pytest.raises(ValueError, match="not inside the arena"):
        env_config_from_dict(data)


def test_spawn_overlapping_an_obstacle_is_rejected() -> None:
    data = _slalom_dict()
    data["geometry"]["obstacles"].append({"x": [1.0, 1.5], "y": [0.0, 0.5]})
    with pytest.raises(ValueError, match="overlaps obstacle"):
        env_config_from_dict(data)


def test_spawn_reaching_the_goal_is_rejected() -> None:
    data = _slalom_dict()
    data["task"]["spawn"] = {"x": [9.0, 10.5], "y": [-1.0, 1.0]}
    with pytest.raises(ValueError, match="goal region"):
        env_config_from_dict(data)


def test_degenerate_box_is_rejected() -> None:
    data = copy.deepcopy(_slalom_dict())
    data["geometry"]["arena"] = {"x": [1.0, 1.0], "y": [-2.0, 2.0]}
    with pytest.raises(ValueError, match="lo < hi"):
        env_config_from_dict(data)


@pytest.mark.parametrize(("section", "key"), [("physics", "dt"), ("reward", "time_penalty"), ("geometry", "goal_x")])
@pytest.mark.parametrize("value", [math.inf, math.nan])
def test_non_finite_values_are_rejected(section: str, key: str, value: float) -> None:
    data = _slalom_dict()
    data[section][key] = value
    with pytest.raises(ValueError, match="finite"):
        env_config_from_dict(data)


def test_a_repeated_key_is_rejected(tmp_path: Path) -> None:
    """YAML parsers keep the last of two equal keys; the configuration loader refuses the file instead."""
    text = (CONFIG_DIR / "slalom.yaml").read_text(encoding="utf-8")
    path = tmp_path / "twice.yaml"
    path.write_text(text.replace("physics:\n", "physics:\n  dt: 0.2\n", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key 'dt'"):
        load_env_config(path)



# ------------------------------------------------------------- the trap layout (D69)


TRAP_OBSTACLES = (
    Box(x=(5.25, 5.75), y=(-1.2, 1.2)),  # the U: back wall
    Box(x=(3.75, 5.75), y=(0.8, 1.2)),  # the U: upper arm
    Box(x=(3.75, 5.75), y=(-1.2, -0.8)),  # the U: lower arm
    Box(x=(8.0, 8.5), y=(-2.0, -0.375)),  # the gate: lower block
    Box(x=(8.0, 8.5), y=(0.375, 2.0)),  # the gate: upper block
)
TRAP_CAVITY = Box(x=(3.75, 5.25), y=(-0.8, 0.8))


def test_the_trap_layout_is_the_sparse_slalom_with_its_obstacles_and_bands_of_the_arenas_height() -> None:
    """D69 (amended twice): the U-trap opening towards the spawn box on the straight route, the corridors of
    0.8 m around it, the gate of 0.75 m after it; the sparse reward and the curriculum of the slalom, with bands
    of the arena's height and no excluded region: the cavity of the U carries training starts like any other
    free region, so that every stage extends the one before (the starts in front of the mouth lead into it)."""
    base = load_env_config(CONFIG_DIR / "slalom_sparse.yaml")
    trap = load_env_config(CONFIG_DIR / "trap_sparse.yaml")
    assert base.task.curriculum is not None and trap.task.curriculum is not None
    expected = dataclasses.replace(
        base,
        geometry=dataclasses.replace(base.geometry, obstacles=TRAP_OBSTACLES),
        task=dataclasses.replace(
            base.task, curriculum=dataclasses.replace(base.task.curriculum, exclude=(), y=(-2.0, 2.0)),
        ),
    )
    assert trap == expected  # the bands span the arena's height: the corridors carry starts
    assert trap.task.curriculum.exclude == ()  # the cavity is not excluded (the second amendment of D69)
    back, upper, lower, gate_lower, gate_upper = TRAP_OBSTACLES
    assert upper.y[0] - lower.y[1] == pytest.approx(1.6) and back.x[0] - upper.x[0] == pytest.approx(1.5)  # cavity
    assert back.x[1] - upper.x[0] == pytest.approx(2.0)  # the U's footprint: 2 m, so that no band of D62 is covered
    assert 2.0 - upper.y[1] == pytest.approx(0.8) and lower.y[0] + 2.0 == pytest.approx(0.8)  # the corridors
    assert gate_upper.y[0] - gate_lower.y[1] == pytest.approx(0.75)  # the gate, centred on y = 0
    assert TRAP_CAVITY == Box(x=(upper.x[0], back.x[0]), y=(lower.y[1], upper.y[0]))
    assert trap.reward.progress_coef == 0.0


def test_the_exploration_setting_of_the_trap_is_the_trap_without_the_curriculum() -> None:
    main = load_env_config(CONFIG_DIR / "trap_sparse.yaml")
    plain = load_env_config(CONFIG_DIR / "trap_sparse_nocur.yaml")
    assert plain.task.curriculum is None
    assert plain == dataclasses.replace(main, task=dataclasses.replace(main.task, curriculum=None))


def test_the_dense_setting_of_the_trap_is_the_trap_exploration_file_with_the_progress_term() -> None:
    """D71: `trap.yaml`, the dense (shaped) setting of the trap restored on 2026-10-20, is `trap_sparse_nocur.yaml`
    with the progress term of D22 added back (coefficient 10, undiscounted); nothing else differs, as for the
    shaped files of the other layouts."""
    plain = load_env_config(CONFIG_DIR / "trap_sparse_nocur.yaml")
    dense = load_env_config(CONFIG_DIR / "trap.yaml")
    assert dense.task.curriculum is None and dense.reward.shaping_discount == 1.0
    assert dense == dataclasses.replace(plain, reward=dataclasses.replace(plain.reward, progress_coef=10.0))


def test_the_narrow_gate_files_are_gone() -> None:
    """D61 is withdrawn: the trap layout contains the narrow gate."""
    assert not list(CONFIG_DIR.glob("slalom_narrow*"))


@pytest.mark.parametrize("layout", ["slalom", "tunnel"])
def test_the_exploration_setting_is_the_sparse_layout_without_the_curriculum(layout: str) -> None:
    """D63, D68: `<layout>_sparse_nocur.yaml` differs from `<layout>_sparse.yaml` in the curriculum only (none);
    the tunnel's exploration file is the control of D68."""
    main = load_env_config(CONFIG_DIR / f"{layout}_sparse.yaml")
    plain = load_env_config(CONFIG_DIR / f"{layout}_sparse_nocur.yaml")
    assert main.task.curriculum is not None and plain.task.curriculum is None
    assert plain == dataclasses.replace(main, task=dataclasses.replace(main.task, curriculum=None))
