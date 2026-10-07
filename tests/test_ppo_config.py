"""Tests of the PPO configuration and its YAML loader (code rule C2, decision log D19, D20)."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from hrlmpc.ppo_config import PPOConfig, load_ppo_config, ppo_config_from_dict, ppo_config_to_dict

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "agent" / "ppo.yaml"


def _data() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return copy.deepcopy(data)


def _with(section: str, key: str, value: Any) -> dict[str, Any]:
    data = _data()
    data[section][key] = value
    return data


def test_the_shipped_configuration_is_the_pre_alignment_tuning() -> None:
    """D19: the values of the pre-alignment flat PPO (algorithms/ppo/ppo_train.py at tag pre-alignment)."""
    config = load_ppo_config(CONFIG)
    assert config.network.hidden_sizes == (64, 64)
    assert (config.rollout.num_envs, config.rollout.num_steps) == (8, 128)
    assert config.batch_size == 1024 and config.minibatch_size == 256
    u = config.update
    assert (u.epochs, u.minibatches, u.learning_rate, u.anneal_learning_rate) == (10, 4, 3e-4, True)
    assert (u.critic_lr_mult, u.adam_eps, u.gae_lambda, u.clip_coef) == (3.0, 1e-5, 0.95, 0.2)
    assert (u.entropy_coef, u.max_grad_norm, u.value_norm_horizon) == (0.01, 0.5, 10)
    assert (config.evaluation.every, config.evaluation.grid) == (10240, (5, 5))
    assert config.evaluation.seed not in range(1, 100)  # distinct from the usual training seeds
    assert (config.runtime.device, config.runtime.torch_threads) == ("cpu", 1)


def test_the_configuration_round_trips() -> None:
    config = load_ppo_config(CONFIG)
    assert ppo_config_from_dict(ppo_config_to_dict(config)) == config
    assert yaml.safe_load(yaml.safe_dump(ppo_config_to_dict(config))) == ppo_config_to_dict(config)


def test_unknown_and_missing_keys_are_rejected() -> None:
    data = _data()
    data["update"]["learning_rat"] = data["update"].pop("learning_rate")
    with pytest.raises(ValueError, match="missing keys \\['learning_rate'\\], unknown keys \\['learning_rat'\\]"):
        ppo_config_from_dict(data)
    data = _data()
    data["extra"] = {}
    with pytest.raises(ValueError, match="unknown keys \\['extra'\\]"):
        ppo_config_from_dict(data)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("network", "hidden_sizes", [], "hidden_sizes"),
        ("network", "hidden_sizes", [64, 0], "hidden_sizes"),
        ("network", "hidden_sizes", [64.0], "integer"),
        ("rollout", "num_envs", 0, "num_envs"),
        ("rollout", "num_steps", True, "integer"),
        ("update", "epochs", 0, "epochs"),
        ("update", "minibatches", 2048, "minibatches"),
        ("update", "learning_rate", 0.0, "learning_rate"),
        ("update", "learning_rate", "3e-4", "number"),
        ("update", "anneal_learning_rate", 1, "boolean"),
        ("update", "critic_lr_mult", -1.0, "critic_lr_mult"),
        ("update", "adam_eps", 0.0, "adam_eps"),
        ("update", "gae_lambda", 1.5, "gae_lambda"),
        ("update", "clip_coef", 0.0, "clip_coef"),
        ("update", "entropy_coef", -0.01, "entropy_coef"),
        ("update", "max_grad_norm", 0.0, "max_grad_norm"),
        ("update", "value_norm_horizon", 0, "value_norm_horizon"),
        ("evaluation", "every", 1000, "multiple"),
        ("evaluation", "grid", [5], "grid"),
        ("evaluation", "grid", [5, 0], "grid"),
        ("evaluation", "seed", -1, "seed"),
        ("runtime", "device", 3, "device"),
        ("runtime", "device", "", "device"),
        ("runtime", "torch_threads", 0, "torch_threads"),
        ("update", "learning_rate", True, "number"),
        ("update", "learning_rate", math.inf, "finite"),
        ("update", "entropy_coef", math.nan, "finite"),
        ("network", "hidden_sizes", 64, "list"),
    ],
)
def test_invalid_values_are_rejected(section: str, key: str, value: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ppo_config_from_dict(_with(section, key, value))


def test_the_batch_sizes_follow_from_the_rollout() -> None:
    data = _with("rollout", "num_envs", 4)
    data["evaluation"]["every"] = 4 * 128 * 3
    config = ppo_config_from_dict(data)
    assert isinstance(config, PPOConfig)
    assert config.batch_size == 512 and config.minibatch_size == 128



@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("update", "entropy_coef", 0.0),
        ("update", "gae_lambda", 0.0),
        ("update", "gae_lambda", 1.0),
        ("update", "minibatches", 1024),
        ("evaluation", "seed", 0),
    ],
)
def test_boundary_values_are_accepted(section: str, key: str, value: Any) -> None:
    config = ppo_config_from_dict(_with(section, key, value))
    assert getattr(getattr(config, section), key) == value


def test_a_repeated_key_is_rejected(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    path = tmp_path / "twice.yaml"
    path.write_text(text.replace("update:\n", "update:\n  learning_rate: 1.0e-2\n", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key 'learning_rate'"):
        load_ppo_config(path)
