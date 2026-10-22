"""Typed configuration of hPPO, loaded from YAML (code rule C2, decision log D24, D27, D28).

``configs/agent/hppo.yaml`` holds the hierarchy (segment length ``H``, the
targets' radius as a multiple of a segment's reach, the price of the
Worker's progress, D66, and optionally the Worker's local map features,
D75), the rollout, one network
and one PPO update for each level, the evaluation protocol and the runtime. The
sections shared with flat PPO are validated by :mod:`hrlmpc.ppo_config`. Each
level has its own discount: the Worker's per step, the Manager's per step
within a segment and ``gamma^tau`` between decisions (D27). Unknown, missing or
repeated keys are errors. This module does not import PyTorch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hrlmpc.ppo_config import (
    EvaluationConfig,
    NetworkConfig,
    RolloutConfig,
    RuntimeConfig,
    UpdateConfig,
    check_keys,
    parse_section,
)
from hrlmpc.utils.strict_yaml import load_yaml


@dataclass(frozen=True)
class MapFeaturesConfig:
    """The learned Worker's local map features (D75): ``rays`` distances to the first wall along as many fixed
    directions, clipped at ``range`` metres and scaled by it (:class:`hrlmpc.hierarchy.MapFeatures`).

    Raises:
        ValueError: If ``rays`` is not positive or ``range`` is not positive and finite.
    """

    rays: int
    range: float

    def __post_init__(self) -> None:
        if self.rays < 1:
            raise ValueError(f"rays must be positive, got {self.rays}")
        if not (math.isfinite(self.range) and self.range > 0.0):
            raise ValueError(f"range must be positive and finite, got {self.range}")


@dataclass(frozen=True)
class HierarchyConfig:
    """Segments, targets, the Worker's price and the Worker's map features (D24, D25, D27, D66, D75).

    Attributes:
        segment_steps: ``H``, the steps of a full segment.
        target_reach_factor: The targets' radius as a multiple of the longest
            displacement in a segment, ``v_max H T_s``.
        worker_progress_coef: ``c_p`` of the Worker's reward (D25): the price
            of the decrease of the distance to the target, in seconds per
            metre. It belongs to the hierarchy, not to the environment: the
            sparse reward of D51 has no progress term (``progress_coef`` 0),
            and the Worker keeps its dense reward (D66). The shipped value is
            the shaped reward's, 10.
        worker_map_features: The local map features appended to the Worker's
            input (D75), or ``None`` for the Worker of D26, which knows no
            walls; absent from the file means ``None``.
    """

    segment_steps: int
    target_reach_factor: float
    worker_progress_coef: float
    worker_map_features: MapFeaturesConfig | None = None

    def __post_init__(self) -> None:
        if self.segment_steps < 1:
            raise ValueError(f"segment_steps must be positive, got {self.segment_steps}")
        if not self.target_reach_factor > 0.0:
            raise ValueError(f"target_reach_factor must be positive, got {self.target_reach_factor}")
        if not (math.isfinite(self.worker_progress_coef) and self.worker_progress_coef > 0.0):
            raise ValueError(f"worker_progress_coef must be positive and finite, got {self.worker_progress_coef}")


@dataclass(frozen=True)
class LevelConfig:
    """The network and the PPO update of one level of the hierarchy."""

    network: NetworkConfig
    update: UpdateConfig


@dataclass(frozen=True)
class HPPOConfig:
    """Complete configuration of hPPO (D28).

    Raises:
        ValueError: If a rollout cannot close a segment per agent, if the
            segments a rollout is sure to close or the Worker's samples are
            fewer than two or than the level's minibatches (the update needs
            both), or if the evaluation interval is not a multiple of the
            samples per update.
    """

    hierarchy: HierarchyConfig
    rollout: RolloutConfig
    manager: LevelConfig
    worker: LevelConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        if self.rollout.num_steps < self.hierarchy.segment_steps:
            raise ValueError(
                f"a rollout of {self.rollout.num_steps} steps per agent may close no segment of "
                f"{self.hierarchy.segment_steps} steps"
            )
        if self.guaranteed_segments < max(2, self.manager.update.minibatches):
            raise ValueError(
                f"the Manager's update needs at least 2 segments and minibatches={self.manager.update.minibatches}; "
                f"a rollout is sure to close only {self.guaranteed_segments}"
            )
        if self.batch_size < max(2, self.worker.update.minibatches):
            raise ValueError(
                f"the Worker's update needs at least 2 samples and minibatches={self.worker.update.minibatches}; "
                f"the rollout gives {self.batch_size}"
            )
        if self.evaluation.every % self.batch_size:
            raise ValueError(
                f"evaluation.every={self.evaluation.every} must be a multiple of the "
                f"{self.batch_size} samples per update"
            )

    @property
    def batch_size(self) -> int:
        """Samples per update, ``num_envs * num_steps``: the Worker's batch."""
        return self.rollout.num_envs * self.rollout.num_steps

    @property
    def guaranteed_segments(self) -> int:
        """Segments every rollout closes at least, ``num_envs * (num_steps // H)``: the Manager's smallest batch.

        A segment in flight when a rollout starts has run ``s < H`` steps, so
        an agent closes at least ``(num_steps + s) // H`` segments; more if
        episodes end.
        """
        return self.rollout.num_envs * (self.rollout.num_steps // self.hierarchy.segment_steps)


SHAPED_PROGRESS_COEF = 10.0
"""The shaped reward's ``progress_coef`` (D13), the shipped price of the Worker's progress (D66)."""

_TOP = ("hierarchy", "rollout", "manager", "worker", "evaluation", "runtime")
_LEVEL = ("network", "update")
_HIERARCHY = ("segment_steps", "target_reach_factor", "worker_progress_coef", "worker_map_features")
_OPTIONAL_HIERARCHY = ("worker_map_features",)
_MAP_FEATURES = ("rays", "range")


def _map_features(data: Any) -> MapFeaturesConfig | None:
    if data is None:
        return None
    raw = check_keys(data, _MAP_FEATURES, "hierarchy.worker_map_features")
    rays = raw["rays"]
    if isinstance(rays, bool) or not isinstance(rays, int):
        raise ValueError(f"hierarchy.worker_map_features.rays must be an integer, got {rays!r}")
    span = raw["range"]
    if isinstance(span, bool) or not isinstance(span, (int, float)) or not math.isfinite(span):
        raise ValueError(f"hierarchy.worker_map_features.range must be a finite number, got {span!r}")
    return MapFeaturesConfig(rays=rays, range=float(span))


def _hierarchy(data: Any) -> HierarchyConfig:
    if isinstance(data, Mapping):
        data = {**{key: None for key in _OPTIONAL_HIERARCHY}, **data}  # an absent optional key reads as None
    raw = check_keys(data, _HIERARCHY, "hierarchy")
    steps = raw["segment_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise ValueError(f"hierarchy.segment_steps must be an integer, got {steps!r}")
    for key in ("target_reach_factor", "worker_progress_coef"):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"hierarchy.{key} must be a number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"hierarchy.{key} must be finite, got {value!r}")
    return HierarchyConfig(
        segment_steps=steps,
        target_reach_factor=float(raw["target_reach_factor"]),
        worker_progress_coef=float(raw["worker_progress_coef"]),
        worker_map_features=_map_features(raw["worker_map_features"]),
    )


def with_worker_price(agent: Any) -> Any:
    """``agent`` with ``hierarchy.worker_progress_coef`` set to :data:`SHAPED_PROGRESS_COEF` where it is missing:
    the ``config.yaml`` of a run from before D66, whose Worker was paid the environment's ``progress_coef``. For
    reloading a run's policies for evaluation, where the price is immaterial."""
    hierarchy = agent.get("hierarchy") if isinstance(agent, dict) else None
    if isinstance(hierarchy, dict) and "worker_progress_coef" not in hierarchy:
        return {**agent, "hierarchy": {**hierarchy, "worker_progress_coef": SHAPED_PROGRESS_COEF}}
    return agent


def _level(data: Any, name: str) -> LevelConfig:
    raw = check_keys(data, _LEVEL, name)
    return LevelConfig(
        network=NetworkConfig(**parse_section(raw["network"], "network", f"{name}.network")),
        update=UpdateConfig(**parse_section(raw["update"], "update", f"{name}.update")),
    )


def hppo_config_from_dict(data: Mapping[str, Any]) -> HPPOConfig:
    """Build and validate an :class:`HPPOConfig` from a nested mapping, as in ``configs/agent/hppo.yaml``.

    Raises:
        ValueError: If a key is missing or unknown, or a value is invalid.
    """
    top = check_keys(data, _TOP, "config")
    return HPPOConfig(
        hierarchy=_hierarchy(top["hierarchy"]),
        rollout=RolloutConfig(**parse_section(top["rollout"], "rollout")),
        manager=_level(top["manager"], "manager"),
        worker=_level(top["worker"], "worker"),
        evaluation=EvaluationConfig(**parse_section(top["evaluation"], "evaluation")),
        runtime=RuntimeConfig(**parse_section(top["runtime"], "runtime")),
    )


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def hppo_config_to_dict(config: HPPOConfig) -> dict[str, Any]:
    """The configuration as plain YAML values; it round-trips through :func:`hppo_config_from_dict`."""
    result: dict[str, Any] = _plain(asdict(config))
    return result


def load_hppo_config(path: str | Path) -> HPPOConfig:
    """Load and validate an hPPO configuration from a YAML file.

    Raises:
        ValueError: If the file repeats a key or does not describe a valid configuration.
    """
    return hppo_config_from_dict(load_yaml(path))
