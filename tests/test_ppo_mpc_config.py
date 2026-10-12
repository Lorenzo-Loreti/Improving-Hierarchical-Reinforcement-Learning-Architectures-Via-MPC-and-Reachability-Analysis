"""Tests of the PPO_MPC and PPO_MPC_reachability configurations (code rule C2, decision log D24, D27, D28, D34,
D37, D38, D45, D46)."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import pytest
import yaml

from hrlmpc.hppo_config import load_hppo_config
from hrlmpc.mpc_problem import MPCSettings
from hrlmpc.ppo_mpc_config import (
    PPOMPCConfig,
    ReachabilityConfig,
    load_ppo_mpc_config,
    ppo_mpc_config_from_dict,
    ppo_mpc_config_to_dict,
)

REPO = Path(__file__).resolve().parents[1]
SHIPPED = REPO / "configs" / "agent" / "ppo_mpc.yaml"
REACHABILITY = REPO / "configs" / "agent" / "ppo_mpc_reachability.yaml"


def _data() -> dict[str, Any]:
    return copy.deepcopy(yaml.safe_load(SHIPPED.read_text(encoding="utf-8")))


def test_the_shipped_configuration_is_the_decided_one() -> None:
    config = load_ppo_mpc_config(SHIPPED)
    hppo = load_hppo_config(REPO / "configs" / "agent" / "hppo.yaml")
    # the Manager, the hierarchy, the rollout and the evaluation are hPPO's (D34)
    assert config.manager == hppo.manager
    assert config.hierarchy == hppo.hierarchy
    assert config.rollout == hppo.rollout
    assert config.evaluation == hppo.evaluation
    assert config.runtime == hppo.runtime
    # the Worker's settings of D34, with the waypoint cost of D37, the automatic horizon of D45 and the
    # regularizer of D46
    assert config.mpc == MPCSettings(
        horizon=None, q_pos=10.0, q_vel=1.0, r=0.1, target_weight=1e4, effort_weight=0.1, start_weight=0.1,
        rho=1e-3, mip_gap=1e-4, work_limit=None,
    )
    assert config.mpc == MPCSettings()  # the defaults are the decided values
    assert config.batch_size == 2048 and config.guaranteed_segments == 8 * 25
    assert config.reachability is None and config.algorithm == "ppo_mpc"


def test_architecture_4_differs_only_in_its_targets() -> None:
    """D38: the shipped configuration of PPO_MPC_reachability is PPO_MPC's with the reachable targets."""
    three, four = load_ppo_mpc_config(SHIPPED), load_ppo_mpc_config(REACHABILITY)
    assert four.reachability == ReachabilityConfig(directions=16) and four.algorithm == "ppo_mpc_reachability"
    assert four == dataclasses.replace(three, reachability=four.reachability)
    assert ppo_mpc_config_from_dict(ppo_mpc_config_to_dict(four)) == four


def test_the_configuration_round_trips() -> None:
    config = load_ppo_mpc_config(SHIPPED)
    assert ppo_mpc_config_from_dict(ppo_mpc_config_to_dict(config)) == config
    data = _data()
    data["mpc"]["work_limit"] = 50.0
    assert ppo_mpc_config_from_dict(ppo_mpc_config_to_dict(ppo_mpc_config_from_dict(data))).mpc.work_limit == 50.0
    data["mpc"]["horizon"] = 16  # an explicit horizon is kept (D45: null sets it automatically)
    assert ppo_mpc_config_from_dict(ppo_mpc_config_to_dict(ppo_mpc_config_from_dict(data))).mpc.horizon == 16


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.pop("mpc"),
        lambda d: d.update(worker={}),
        lambda d: d["mpc"].pop("horizon"),
        lambda d: d["mpc"].update(solver="gurobi"),
        lambda d: d["mpc"].update(horizon=10.0),
        lambda d: d["mpc"].update(horizon=True),
        lambda d: d["mpc"].update(horizon=0),
        lambda d: d["mpc"].update(horizon="auto"),
        lambda d: d["mpc"].pop("start_weight"),
        lambda d: d["mpc"].update(start_weight=-0.1),
        lambda d: d["mpc"].update(start_weight="0.1"),
        lambda d: d["mpc"].update(start_weight=float("nan")),
        lambda d: d["mpc"].update(q_pos="10"),
        lambda d: d["mpc"].update(r=False),
        lambda d: d["mpc"].pop("target_weight"),
        lambda d: d["mpc"].update(target_weight=0.0),
        lambda d: d["mpc"].update(target_weight="1e4"),
        lambda d: d["mpc"].pop("effort_weight"),
        lambda d: d["mpc"].update(effort_weight=-0.1),
        lambda d: d["mpc"].update(rho=float("inf")),
        lambda d: d["mpc"].update(rho=0.0),
        lambda d: d["mpc"].update(work_limit=-1.0),
        lambda d: d["mpc"].update(work_limit=True),
        lambda d: d["mpc"].update(work_limit="50"),
        lambda d: d["mpc"].update(mip_gap=float("nan")),
        lambda d: d["mpc"].update(mip_gap=1.0),
        lambda d: d["mpc"].update(tail_tolerance=0.0),
        lambda d: d["mpc"].update(tail_tolerance=-1e-3),
        lambda d: d.update(mpc=[10, 10.0]),
        lambda d: d.pop("reachability"),
        lambda d: d.update(reachability={}),
        lambda d: d.update(reachability={"directions": 2}),
        lambda d: d.update(reachability={"directions": 16.0}),
        lambda d: d.update(reachability={"directions": True}),
        lambda d: d.update(reachability={"directions": 16, "anchor": "centroid"}),
        lambda d: d.update(reachability=16),
    ],
)
def test_invalid_settings_are_refused(change: Any) -> None:
    data = _data()
    change(data)
    with pytest.raises(ValueError):
        ppo_mpc_config_from_dict(data)


def test_the_batch_sizes_are_validated() -> None:
    data = _data()
    data["rollout"].update(num_steps=5)
    with pytest.raises(ValueError, match="no segment"):
        ppo_mpc_config_from_dict(data)
    data = _data()
    data["rollout"].update(num_envs=1, num_steps=10)
    with pytest.raises(ValueError, match="2 segments"):
        ppo_mpc_config_from_dict(data)
    data = _data()
    data["evaluation"].update(every=1000)
    with pytest.raises(ValueError, match="multiple"):
        ppo_mpc_config_from_dict(data)


def test_a_repeated_key_is_refused(tmp_path: Path) -> None:
    text = SHIPPED.read_text(encoding="utf-8").replace("  horizon: null", "  horizon: null\n  horizon: 12", 1)
    assert text != SHIPPED.read_text(encoding="utf-8")
    path = tmp_path / "ppo_mpc.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_ppo_mpc_config(path)
    assert isinstance(load_ppo_mpc_config(SHIPPED), PPOMPCConfig)
