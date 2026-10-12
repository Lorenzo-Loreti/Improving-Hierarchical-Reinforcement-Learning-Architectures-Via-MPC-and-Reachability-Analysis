"""Typed configuration of PPO_MPC and PPO_MPC_reachability, loaded from YAML (code rule C2, decision log D24,
D27, D28, D34, D37, D38).

``configs/agent/ppo_mpc.yaml`` (architecture 3) and
``configs/agent/ppo_mpc_reachability.yaml`` (architecture 4) hold the hierarchy
(segment length ``H`` and the radius of the targets of D24), the rollout, the
Manager's network and PPO update, the MPC Worker's settings, the targets of
architecture 4 (``reachability``: ``null`` for architecture 3, else the number
of support directions of D38), the evaluation protocol and the runtime. The
Manager and the hierarchy are hPPO's; the Worker is the tube MPC of D31-D34
with the waypoint cost of D37, designed for the disturbance of the
environment's configuration. The two architectures differ only in the
targets. Unknown, missing or repeated keys are errors. Whether the
horizon leaves room to stop after a segment depends on the environment's
physics, so the training checks it (:meth:`hrlmpc.mpc_problem.MPCProblem.check_segments`).
This module does not import PyTorch or the solvers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hrlmpc.hppo_config import HierarchyConfig, LevelConfig, _hierarchy, _level
from hrlmpc.mpc_problem import MPCSettings
from hrlmpc.ppo_config import EvaluationConfig, RolloutConfig, RuntimeConfig, check_keys, parse_section
from hrlmpc.utils.strict_yaml import load_yaml


@dataclass(frozen=True)
class ReachabilityConfig:
    """The targets of architecture 4 (D38).

    Attributes:
        directions: Support directions of the inner approximation of the reachable set, at least 3.
    """

    directions: int

    def __post_init__(self) -> None:
        if isinstance(self.directions, bool) or not isinstance(self.directions, int) or self.directions < 3:
            raise ValueError(f"reachability.directions must be an integer of at least 3, got {self.directions!r}")


@dataclass(frozen=True)
class PPOMPCConfig:
    """Complete configuration of PPO_MPC (``reachability`` is ``None``) or PPO_MPC_reachability.

    Raises:
        ValueError: If a rollout cannot close a segment per agent, if the
            segments a rollout is sure to close are fewer than two or than the
            Manager's minibatches, or if the evaluation interval is not a
            multiple of the samples per update.
    """

    hierarchy: HierarchyConfig
    rollout: RolloutConfig
    manager: LevelConfig
    mpc: MPCSettings
    reachability: ReachabilityConfig | None
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
        if self.evaluation.every % self.batch_size:
            raise ValueError(
                f"evaluation.every={self.evaluation.every} must be a multiple of the "
                f"{self.batch_size} samples per update"
            )

    @property
    def algorithm(self) -> str:
        """``ppo_mpc_reachability`` with the reachable targets of D38, ``ppo_mpc`` otherwise."""
        return "ppo_mpc" if self.reachability is None else "ppo_mpc_reachability"

    @property
    def batch_size(self) -> int:
        """Samples per update, ``num_envs * num_steps``."""
        return self.rollout.num_envs * self.rollout.num_steps

    @property
    def guaranteed_segments(self) -> int:
        """Segments every rollout closes at least, ``num_envs * (num_steps // H)``: the Manager's smallest batch."""
        return self.rollout.num_envs * (self.rollout.num_steps // self.hierarchy.segment_steps)


_TOP = ("hierarchy", "rollout", "manager", "mpc", "reachability", "evaluation", "runtime")
_MPC = (
    "horizon", "q_pos", "q_vel", "r", "target_weight", "effort_weight", "rho", "tail_tolerance", "mip_gap", "work_limit"
)


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{where} must be finite, got {value!r}")
    return float(value)


def _mpc(data: Any) -> MPCSettings:
    raw = check_keys(data, _MPC, "mpc")
    horizon = raw["horizon"]
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError(f"mpc.horizon must be an integer, got {horizon!r}")
    limit = raw["work_limit"]
    return MPCSettings(
        horizon=horizon,
        q_pos=_number(raw["q_pos"], "mpc.q_pos"),
        q_vel=_number(raw["q_vel"], "mpc.q_vel"),
        r=_number(raw["r"], "mpc.r"),
        target_weight=_number(raw["target_weight"], "mpc.target_weight"),
        effort_weight=_number(raw["effort_weight"], "mpc.effort_weight"),
        rho=_number(raw["rho"], "mpc.rho"),
        tail_tolerance=_number(raw["tail_tolerance"], "mpc.tail_tolerance"),
        mip_gap=_number(raw["mip_gap"], "mpc.mip_gap"),
        work_limit=None if limit is None else _number(limit, "mpc.work_limit"),
    )


def _reachability(data: Any) -> ReachabilityConfig | None:
    if data is None:
        return None
    raw = check_keys(data, ("directions",), "reachability")
    return ReachabilityConfig(directions=raw["directions"])


def ppo_mpc_config_from_dict(data: Mapping[str, Any]) -> PPOMPCConfig:
    """Build and validate a :class:`PPOMPCConfig` from a nested mapping, as in ``configs/agent/ppo_mpc.yaml``.

    Raises:
        ValueError: If a key is missing or unknown, or a value is invalid.
    """
    top = check_keys(data, _TOP, "config")
    return PPOMPCConfig(
        hierarchy=_hierarchy(top["hierarchy"]),
        rollout=RolloutConfig(**parse_section(top["rollout"], "rollout")),
        manager=_level(top["manager"], "manager"),
        mpc=_mpc(top["mpc"]),
        reachability=_reachability(top["reachability"]),
        evaluation=EvaluationConfig(**parse_section(top["evaluation"], "evaluation")),
        runtime=RuntimeConfig(**parse_section(top["runtime"], "runtime")),
    )


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def ppo_mpc_config_to_dict(config: PPOMPCConfig) -> dict[str, Any]:
    """The configuration as plain YAML values; it round-trips through :func:`ppo_mpc_config_from_dict`."""
    result: dict[str, Any] = _plain(asdict(config))
    return result


def load_ppo_mpc_config(path: str | Path) -> PPOMPCConfig:
    """Load and validate a PPO_MPC configuration from a YAML file.

    Raises:
        ValueError: If the file repeats a key or does not describe a valid configuration.
    """
    return ppo_mpc_config_from_dict(load_yaml(path))
