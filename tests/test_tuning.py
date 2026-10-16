"""Tests of the tuning configurations (decision log D55, D65): eight files per architecture that cover the
2 x 2 x 2 design, nothing but the three knobs changed; the shipped file of an architecture is its selected
configuration."""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path
from typing import Any

import pytest

from hrlmpc.hppo_config import HPPOConfig, load_hppo_config
from hrlmpc.ppo_config import PPOConfig, load_ppo_config
from hrlmpc.ppo_mpc_config import PPOMPCConfig, load_ppo_mpc_config

AGENTS = Path(__file__).resolve().parents[1] / "configs" / "agent"
LEARNING_RATES = (3.0e-4, 1.0e-3)
ENTROPIES = (0.01, 0.03)
SELECTED = {"ppo": 5, "hppo": 3, "ppo_mpc": 7, "ppo_mpc_reachability": 7}
"""The configuration each shipped file equals: the selections of 2026-10-14 (PPO c5, hPPO c3) and of
2026-10-16 (both MPC architectures c7: the Manager's learning rate 1e-3 and entropy 0.03, the Worker's effort
weight 0.1, the same for both by the tie-break of D65 among configurations within the noise)."""


def _design(third: tuple[Any, Any]) -> list[tuple[float, float, Any]]:
    """The eight points in the order of the files: c1 = (lr low, entropy low, third low), the third fastest."""
    return list(itertools.product(LEARNING_RATES, ENTROPIES, third))


def _ppo_variant(base: PPOConfig, lr: float, entropy: float, num_steps: int) -> PPOConfig:
    return dataclasses.replace(
        base,
        rollout=dataclasses.replace(base.rollout, num_steps=num_steps),
        update=dataclasses.replace(base.update, learning_rate=lr, entropy_coef=entropy),
    )


def _hppo_variant(base: HPPOConfig, lr: float, entropy: float, worker_lr: float) -> HPPOConfig:
    return dataclasses.replace(
        base,
        manager=dataclasses.replace(
            base.manager, update=dataclasses.replace(base.manager.update, learning_rate=lr, entropy_coef=entropy)
        ),
        worker=dataclasses.replace(
            base.worker, update=dataclasses.replace(base.worker.update, learning_rate=worker_lr)
        ),
    )


def _mpc_variant(base: PPOMPCConfig, lr: float, entropy: float, effort: float) -> PPOMPCConfig:
    return dataclasses.replace(
        base,
        manager=dataclasses.replace(
            base.manager, update=dataclasses.replace(base.manager.update, learning_rate=lr, entropy_coef=entropy)
        ),
        mpc=dataclasses.replace(base.mpc, effort_weight=effort),
    )


def test_the_ppo_tuning_files_cover_the_design() -> None:
    """Every file is the shipped file with its three knobs replaced: the shipped file differs from the pre-tuning
    one in those knobs only, so this holds before and after the selection."""
    base = load_ppo_config(AGENTS / "ppo.yaml")
    for index, (lr, entropy, num_steps) in enumerate(_design((128, 256)), start=1):
        config = load_ppo_config(AGENTS / "tuning" / "ppo" / f"c{index}.yaml")
        assert config == _ppo_variant(base, lr, entropy, num_steps), index
    assert load_ppo_config(AGENTS / "tuning" / "ppo" / f"c{SELECTED['ppo']}.yaml") == base
    assert base.update.learning_rate == 1.0e-3 and base.rollout.num_steps == 128  # c5 (2026-10-14)


def test_the_hppo_tuning_files_cover_the_design() -> None:
    base = load_hppo_config(AGENTS / "hppo.yaml")
    for index, (lr, entropy, worker_lr) in enumerate(_design((3.0e-4, 1.0e-3)), start=1):
        config = load_hppo_config(AGENTS / "tuning" / "hppo" / f"c{index}.yaml")
        assert config == _hppo_variant(base, lr, entropy, worker_lr), index
    assert load_hppo_config(AGENTS / "tuning" / "hppo" / f"c{SELECTED['hppo']}.yaml") == base
    assert base.manager.update.entropy_coef == 0.03 and base.worker.update.learning_rate == 3.0e-4  # c3


@pytest.mark.parametrize("architecture", ["ppo_mpc", "ppo_mpc_reachability"])
def test_the_mpc_tuning_files_cover_the_design(architecture: str) -> None:
    """Both MPC architectures use the same space: the Manager's learning rate and entropy, the Worker's effort."""
    base = load_ppo_mpc_config(AGENTS / f"{architecture}.yaml")
    for index, (lr, entropy, effort) in enumerate(_design((0.1, 1.0)), start=1):
        config = load_ppo_mpc_config(AGENTS / "tuning" / architecture / f"c{index}.yaml")
        assert config == _mpc_variant(base, lr, entropy, effort), index
        assert config.algorithm == base.algorithm
    assert load_ppo_mpc_config(AGENTS / "tuning" / architecture / f"c{SELECTED[architecture]}.yaml") == base
    assert base.manager.update.learning_rate == 1.0e-3 and base.manager.update.entropy_coef == 0.03  # c7
    assert base.mpc.effort_weight == 0.1


def test_every_architecture_has_exactly_eight_files() -> None:
    for architecture in ("ppo", "hppo", "ppo_mpc", "ppo_mpc_reachability"):
        names = sorted(p.name for p in (AGENTS / "tuning" / architecture).glob("*.yaml"))
        assert names == [f"c{i}.yaml" for i in range(1, 9)], architecture
