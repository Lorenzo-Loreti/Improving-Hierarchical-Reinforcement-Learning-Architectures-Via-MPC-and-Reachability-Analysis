"""Tests of the hPPO configuration (code rule C2, decision log D24, D27, D28)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from hrlmpc.hppo_config import HPPOConfig, hppo_config_from_dict, hppo_config_to_dict, load_hppo_config

SHIPPED = Path(__file__).resolve().parents[1] / "configs" / "agent" / "hppo.yaml"


def _data() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))
    return data


def _with(path: tuple[str, ...], value: Any) -> dict[str, Any]:
    data = copy.deepcopy(_data())
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return data


def test_the_shipped_configuration_is_the_pre_alignment_tuning_of_hppo() -> None:
    """D24, D27, D28."""
    config = load_hppo_config(SHIPPED)
    assert (config.hierarchy.segment_steps, config.hierarchy.target_reach_factor) == (10, 1.5)
    assert (config.rollout.num_envs, config.rollout.num_steps) == (8, 256)
    for level, minibatches in ((config.manager, 4), (config.worker, 8)):
        assert level.network.hidden_sizes == (64, 64)
        u = level.update
        assert (u.discount, u.epochs, u.minibatches, u.learning_rate, u.anneal_learning_rate) == (
            0.99,
            10,
            minibatches,
            3e-4,
            False,
        )
        assert (u.critic_lr_mult, u.adam_eps, u.gae_lambda, u.clip_coef) == (3.0, 1e-5, 0.95, 0.2)
        assert (u.entropy_coef, u.max_grad_norm, u.value_norm_horizon) == (0.01, 0.5, 10)
    assert (config.evaluation.every, config.evaluation.grid, config.evaluation.seed) == (10240, (5, 5), 12345)
    assert (config.runtime.device, config.runtime.torch_threads) == ("cpu", 1)


def test_derived_sizes() -> None:
    config = load_hppo_config(SHIPPED)
    assert config.batch_size == 2048
    assert config.guaranteed_segments == 200  # every agent closes at least 256 // 10 segments per rollout


def test_the_configuration_round_trips() -> None:
    config = load_hppo_config(SHIPPED)
    assert isinstance(config, HPPOConfig)
    assert hppo_config_from_dict(hppo_config_to_dict(config)) == config
    assert yaml.safe_load(yaml.safe_dump(hppo_config_to_dict(config))) == hppo_config_to_dict(config)


@pytest.mark.parametrize(
    "path",
    [
        ("hierarchy",),
        ("manager",),
        ("worker", "update"),
        ("manager", "network"),
        ("worker", "update", "discount"),
        ("hierarchy", "target_reach_factor"),
    ],
)
def test_missing_keys_are_rejected(path: tuple[str, ...]) -> None:
    data = copy.deepcopy(_data())
    node = data
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    with pytest.raises(ValueError, match="missing"):
        hppo_config_from_dict(data)


@pytest.mark.parametrize(
    "path", [("extra",), ("hierarchy", "extra"), ("manager", "extra"), ("worker", "update", "extra")]
)
def test_unknown_keys_are_rejected(path: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="unknown"):
        hppo_config_from_dict(_with(path, 1))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("hierarchy", "segment_steps"), 0, "segment_steps"),
        (("hierarchy", "segment_steps"), 2.5, "integer"),
        (("hierarchy", "target_reach_factor"), 0.0, "target_reach_factor"),
        (("hierarchy", "target_reach_factor"), True, "number"),
        (("rollout", "num_steps"), 9, "segment"),
        (("manager", "update", "minibatches"), 201, "minibatches"),
        (("worker", "update", "minibatches"), 4096, "minibatches"),
        (("manager", "update", "discount"), 0.0, "discount"),
        (("worker", "network", "hidden_sizes"), [], "hidden_sizes"),
        (("evaluation", "every"), 3000, "multiple"),
        (("runtime", "torch_threads"), 0, "torch_threads"),
    ],
)
def test_invalid_values_are_rejected(path: tuple[str, ...], value: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        hppo_config_from_dict(_with(path, value))


def test_each_update_gets_at_least_two_samples() -> None:
    """PPOAgent.update needs two samples: one agent may close a single segment in a rollout of up to 2H - 1 steps."""
    data = copy.deepcopy(_data())
    data["rollout"].update(num_envs=1, num_steps=19)
    data["manager"]["update"]["minibatches"] = data["worker"]["update"]["minibatches"] = 1
    data["evaluation"]["every"] = 19
    with pytest.raises(ValueError, match="at least 2 segments"):
        hppo_config_from_dict(data)
    data["rollout"]["num_steps"], data["evaluation"]["every"] = 20, 20
    assert hppo_config_from_dict(data).guaranteed_segments == 2
    data["hierarchy"]["segment_steps"], data["rollout"]["num_envs"], data["rollout"]["num_steps"] = 1, 1, 1
    data["evaluation"]["every"] = 1
    with pytest.raises(ValueError, match="at least 2 segments"):  # one segment, and one Worker sample
        hppo_config_from_dict(data)


def test_repeated_keys_are_rejected(tmp_path: Path) -> None:
    shipped = SHIPPED.read_text(encoding="utf-8")
    text = shipped.replace("  segment_steps: 10", "  segment_steps: 10\n  segment_steps: 5", 1)
    assert text != shipped
    path = tmp_path / "hppo.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="segment_steps"):
        load_hppo_config(path)
