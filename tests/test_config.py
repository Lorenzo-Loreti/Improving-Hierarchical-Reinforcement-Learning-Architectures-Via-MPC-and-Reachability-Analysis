"""Tests of the YAML environment configuration (code rule C2; decision log D9, D11-D14)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from hrlmpc.config import Box, env_config_from_dict, env_config_to_dict, load_env_config

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
    ) == (1.0, 1000.0, 10.0, 0.99, 0.01, 50.0, 1.0)


def test_tunnel_config_is_the_slalom_without_obstacles() -> None:
    slalom = load_env_config(CONFIG_DIR / "slalom.yaml")
    tunnel = load_env_config(CONFIG_DIR / "tunnel.yaml")
    assert tunnel.geometry.obstacles == ()
    assert tunnel.geometry.arena == slalom.geometry.arena
    assert tunnel.geometry.goal_x == slalom.geometry.goal_x
    assert (tunnel.physics, tunnel.task, tunnel.reward) == (slalom.physics, slalom.task, slalom.reward)


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
