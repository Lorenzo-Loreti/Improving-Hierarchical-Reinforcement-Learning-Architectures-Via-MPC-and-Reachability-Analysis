"""Tests of the hPPO configuration (code rule C2, decision log D24, D27, D28)."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import pytest
import yaml

from hrlmpc.hppo_config import (
    SHAPED_PROGRESS_COEF,
    HPPOConfig,
    MapFeaturesConfig,
    hppo_config_from_dict,
    hppo_config_to_dict,
    load_hppo_config,
    with_worker_price,
)

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


def test_the_shipped_configuration_is_the_pre_alignment_tuning_with_the_selection_of_d65() -> None:
    """D24, D27, D28; the Manager's entropy coefficient 0.03 since the tuning of D65 (configuration c3,
    2026-10-14)."""
    config = load_hppo_config(SHIPPED)
    assert (config.hierarchy.segment_steps, config.hierarchy.target_reach_factor) == (10, 1.5)
    assert config.hierarchy.worker_progress_coef == 10.0
    assert (config.rollout.num_envs, config.rollout.num_steps) == (8, 256)
    for level, minibatches, entropy in ((config.manager, 4, 0.03), (config.worker, 8, 0.01)):
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
        assert (u.entropy_coef, u.max_grad_norm, u.value_norm_horizon) == (entropy, 0.5, 10)
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
        ("hierarchy", "worker_progress_coef"),
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
        (("hierarchy", "worker_progress_coef"), 0.0, "worker_progress_coef"),
        (("hierarchy", "worker_progress_coef"), -1.0, "worker_progress_coef"),
        (("hierarchy", "worker_progress_coef"), "10", "number"),
        (("hierarchy", "worker_progress_coef"), float("inf"), "finite"),
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


@pytest.mark.parametrize("segment_steps", [6, 4])
def test_the_sweep_files_differ_in_the_segment_length_only(segment_steps: int) -> None:
    """D60: the agility sweep changes H alone; the target radius follows from it."""
    base, sweep = load_hppo_config(SHIPPED), load_hppo_config(SHIPPED.with_name(f"hppo_h{segment_steps}.yaml"))
    assert sweep == dataclasses.replace(
        base, hierarchy=dataclasses.replace(base.hierarchy, segment_steps=segment_steps)
    )
    assert sweep.hierarchy.segment_steps == segment_steps and base.hierarchy.segment_steps == 10


@pytest.mark.parametrize("name", ["hppo.yaml", "hppo_h6.yaml", "hppo_h4.yaml"])
def test_the_workers_progress_is_priced_as_in_the_shaped_reward(name: str) -> None:
    """D66: the Worker's progress coefficient is the hierarchy's, 10 s/m as the shaped reward's ``progress_coef``
    (D13), so the Worker keeps its dense reward of D25 under the sparse reward of D51 (whose ``progress_coef`` is 0)."""
    config = load_hppo_config(SHIPPED.with_name(name))
    shaped = yaml.safe_load((SHIPPED.parents[1] / "env" / "slalom.yaml").read_text(encoding="utf-8"))
    sparse = yaml.safe_load((SHIPPED.parents[1] / "env" / "slalom_sparse.yaml").read_text(encoding="utf-8"))
    assert config.hierarchy.worker_progress_coef == shaped["reward"]["progress_coef"] == SHAPED_PROGRESS_COEF == 10.0
    assert sparse["reward"]["progress_coef"] == 0.0


def test_a_configuration_from_before_the_workers_price_gets_the_shaped_one() -> None:
    """D66: ``with_worker_price`` fills a missing ``hierarchy.worker_progress_coef`` with the shaped price, for the
    reload of a run from before; it leaves a present key, its input and anything but a mapping alone."""
    data = _data()
    del data["hierarchy"]["worker_progress_coef"]
    before = copy.deepcopy(data)
    with pytest.raises(ValueError, match="missing"):
        hppo_config_from_dict(data)
    filled = with_worker_price(data)
    assert data == before  # not mutated
    assert filled["hierarchy"]["worker_progress_coef"] == SHAPED_PROGRESS_COEF
    assert hppo_config_from_dict(filled) == load_hppo_config(SHIPPED)
    present = _data()
    assert with_worker_price(present) is present
    assert with_worker_price("text") == "text" and with_worker_price({"rollout": {}}) == {"rollout": {}}


# --------------------------------------------------------------------------
# The Worker's local map features (D75)
# --------------------------------------------------------------------------


def test_the_map_features_are_optional_and_absent_from_the_shipped_file() -> None:
    """D75: ``hierarchy.worker_map_features`` may be absent or null (the Worker of D26), or give the rays and the
    range; the shipped file has none, and the loaded configuration round-trips with the key written as null."""
    config = load_hppo_config(SHIPPED)
    assert config.hierarchy.worker_map_features is None
    assert "worker_map_features" not in _data()["hierarchy"]
    assert hppo_config_to_dict(config)["hierarchy"]["worker_map_features"] is None
    assert hppo_config_from_dict(_with(("hierarchy", "worker_map_features"), None)) == config
    with_features = hppo_config_from_dict(_with(("hierarchy", "worker_map_features"), {"rays": 8, "range": 1.5}))
    assert with_features.hierarchy.worker_map_features == MapFeaturesConfig(rays=8, range=1.5)
    assert with_features != config
    assert hppo_config_from_dict(hppo_config_to_dict(with_features)) == with_features


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"rays": 0, "range": 2.0}, "rays"),
        ({"rays": 8.0, "range": 2.0}, "integer"),
        ({"rays": True, "range": 2.0}, "integer"),
        ({"rays": 8, "range": 0.0}, "range"),
        ({"rays": 8, "range": "2"}, "number"),
        ({"rays": 8}, "missing"),
        ({"rays": 8, "range": 2.0, "extra": 1}, "unknown"),
        (3, "mapping"),
    ],
)
def test_invalid_map_features_are_rejected(value: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        hppo_config_from_dict(_with(("hierarchy", "worker_map_features"), value))


def test_the_map_file_is_the_shipped_file_with_the_workers_map_features() -> None:
    """D75: ``hppo_map.yaml`` differs from ``hppo.yaml`` in the Worker's map features only: 16 rays, range 2 m, which
    covers the targets' radius of 1.8 m."""
    base, ablation = load_hppo_config(SHIPPED), load_hppo_config(SHIPPED.with_name("hppo_map.yaml"))
    features = ablation.hierarchy.worker_map_features
    assert features == MapFeaturesConfig(rays=16, range=2.0)
    hierarchy = dataclasses.replace(base.hierarchy, worker_map_features=features)
    assert ablation == dataclasses.replace(base, hierarchy=hierarchy)
    assert features.range >= base.hierarchy.target_reach_factor * 1.2 * base.hierarchy.segment_steps * 0.1  # R = 1.8 m
