"""Typed configuration of the in-house PPO, loaded from YAML (code rule C2, decision log D19, D20).

A file under ``configs/agent/`` holds the hyperparameters of one learner:
the networks, the rollout, the PPO update, the evaluation protocol and the
runtime. The discount is deliberately absent: the training loops read it from
the environment's ``reward.shaping_discount``, so that the potential-based
shaping stays exact (D13). Loading validates every value; unknown or missing
keys are errors. This module does not import PyTorch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hrlmpc.utils.strict_yaml import load_yaml


@dataclass(frozen=True)
class NetworkConfig:
    """Actor and critic: separate tanh MLPs with these hidden sizes."""

    hidden_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.hidden_sizes or any(h < 1 for h in self.hidden_sizes):
            raise ValueError(f"hidden_sizes must be a non-empty list of positive sizes, got {self.hidden_sizes}")


@dataclass(frozen=True)
class RolloutConfig:
    """Parallel agents and steps per agent collected before each update."""

    num_envs: int
    num_steps: int

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")


@dataclass(frozen=True)
class UpdateConfig:
    """Hyperparameters of the PPO update (D19).

    Attributes:
        epochs: Passes over the batch per update.
        minibatches: Minibatches per epoch.
        learning_rate: Adam learning rate of the actor.
        anneal_learning_rate: Whether the learning rates decay linearly to zero over the budget.
        critic_lr_mult: Learning rate of the critic, as a multiple of ``learning_rate``.
        adam_eps: Adam's epsilon.
        gae_lambda: GAE parameter.
        clip_coef: Clipping range of the probability ratio.
        entropy_coef: Weight of the entropy bonus.
        max_grad_norm: Gradient-norm clip, applied to each network separately.
        value_norm_horizon: Updates over which the running statistics of the
            critic's regression targets are averaged.
    """

    epochs: int
    minibatches: int
    learning_rate: float
    anneal_learning_rate: bool
    critic_lr_mult: float
    adam_eps: float
    gae_lambda: float
    clip_coef: float
    entropy_coef: float
    max_grad_norm: float
    value_norm_horizon: int

    def __post_init__(self) -> None:
        positive = ("epochs", "minibatches", "learning_rate", "critic_lr_mult", "adam_eps", "clip_coef")
        for name in (*positive, "max_grad_norm", "value_norm_horizon"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must lie in [0, 1], got {self.gae_lambda}")
        if self.entropy_coef < 0.0:
            raise ValueError(f"entropy_coef must be non-negative, got {self.entropy_coef}")


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation protocol (D20).

    Attributes:
        every: Samples between two evaluations; also the first one is at zero samples.
        grid: Points of the start grid on the spawn box, along ``x`` and ``y``.
        seed: Seed of the evaluation environment, the same for every run and
            distinct from the training seeds.
    """

    every: int
    grid: tuple[int, int]
    seed: int

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError(f"every must be positive, got {self.every}")
        if len(self.grid) != 2 or any(n < 1 for n in self.grid):
            raise ValueError(f"grid must be two positive sizes, got {self.grid}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")


@dataclass(frozen=True)
class RuntimeConfig:
    """PyTorch device and number of intra-op threads."""

    device: str
    torch_threads: int

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device:
            raise ValueError(f"device must be a non-empty string, got {self.device!r}")
        if self.torch_threads < 1:
            raise ValueError(f"torch_threads must be positive, got {self.torch_threads}")


@dataclass(frozen=True)
class PPOConfig:
    """Complete configuration of a flat PPO learner.

    Raises:
        ValueError: If there are more minibatches than samples per update, or
            the evaluation interval is not a multiple of the samples per update.
    """

    network: NetworkConfig
    rollout: RolloutConfig
    update: UpdateConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        if self.update.minibatches > self.batch_size:
            raise ValueError(f"minibatches={self.update.minibatches} exceeds the {self.batch_size} samples per update")
        if self.evaluation.every % self.batch_size:
            raise ValueError(
                f"evaluation.every={self.evaluation.every} must be a multiple of the "
                f"{self.batch_size} samples per update"
            )

    @property
    def batch_size(self) -> int:
        """Samples per update: ``num_envs * num_steps``."""
        return self.rollout.num_envs * self.rollout.num_steps

    @property
    def minibatch_size(self) -> int:
        """Samples per minibatch, rounded down when the batch does not split evenly."""
        return self.batch_size // self.update.minibatches


_SECTIONS: dict[str, tuple[str, ...]] = {
    "network": ("hidden_sizes",),
    "rollout": ("num_envs", "num_steps"),
    "update": (
        "epochs",
        "minibatches",
        "learning_rate",
        "anneal_learning_rate",
        "critic_lr_mult",
        "adam_eps",
        "gae_lambda",
        "clip_coef",
        "entropy_coef",
        "max_grad_norm",
        "value_norm_horizon",
    ),
    "evaluation": ("every", "grid", "seed"),
    "runtime": ("device", "torch_threads"),
}
_INTS = {"num_envs", "num_steps", "epochs", "minibatches", "value_norm_horizon", "every", "seed", "torch_threads"}
_BOOLS = {"anneal_learning_rate"}
_INT_LISTS = {"hidden_sizes", "grid"}
_STRINGS = {"device"}


def _section(data: Any, keys: tuple[str, ...], where: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError(f"{where} must be a mapping, got {type(data).__name__}")
    missing = [k for k in keys if k not in data]
    unknown = [k for k in data if k not in keys]
    if missing or unknown:
        raise ValueError(f"{where}: missing keys {missing}, unknown keys {unknown}")
    return dict(data)


def _value(key: str, value: Any, where: str) -> Any:
    if key in _BOOLS:
        if not isinstance(value, bool):
            raise ValueError(f"{where} must be a boolean, got {value!r}")
        return value
    if key in _STRINGS:
        if not isinstance(value, str):
            raise ValueError(f"{where} (device) must be a string, got {value!r}")
        return value
    if key in _INT_LISTS:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where} must be a list of integers, got {value!r}")
        return tuple(_value("_int", v, f"{where}[{i}]") for i, v in enumerate(value))
    if key in _INTS or key == "_int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where} must be an integer, got {value!r}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{where} must be finite, got {value!r}")
    return float(value)


def ppo_config_from_dict(data: Mapping[str, Any]) -> PPOConfig:
    """Build and validate a :class:`PPOConfig` from a nested mapping, as in ``configs/agent/ppo.yaml``.

    Raises:
        ValueError: If a key is missing or unknown, or a value is invalid.
    """
    top = _section(data, tuple(_SECTIONS), "config")
    parsed: dict[str, dict[str, Any]] = {}
    for name, keys in _SECTIONS.items():
        raw = _section(top[name], keys, name)
        parsed[name] = {k: _value(k, raw[k], f"{name}.{k}") for k in keys}
    return PPOConfig(
        network=NetworkConfig(**parsed["network"]),
        rollout=RolloutConfig(**parsed["rollout"]),
        update=UpdateConfig(**parsed["update"]),
        evaluation=EvaluationConfig(**parsed["evaluation"]),
        runtime=RuntimeConfig(**parsed["runtime"]),
    )


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def ppo_config_to_dict(config: PPOConfig) -> dict[str, Any]:
    """The configuration as plain YAML values; it round-trips through :func:`ppo_config_from_dict`."""
    result: dict[str, Any] = _plain(asdict(config))
    return result


def load_ppo_config(path: str | Path) -> PPOConfig:
    """Load and validate a PPO configuration from a YAML file.

    Raises:
        ValueError: If the file repeats a key or does not describe a valid configuration.
    """
    return ppo_config_from_dict(load_yaml(path))
