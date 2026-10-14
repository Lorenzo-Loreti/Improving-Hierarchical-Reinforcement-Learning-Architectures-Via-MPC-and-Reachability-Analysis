"""Tests of the re-evaluation of trained policies from explicit starts (decision log D70): the escape from the
trap; they need PyTorch. The MPC architectures use the stand-ins of ``tests/test_train_ppo_mpc.py``."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from hrlmpc.config import Box, EnvConfig, env_config_from_dict  # noqa: E402
from hrlmpc.evaluation import evaluate, spawn_grid  # noqa: E402
from hrlmpc.hppo_config import hppo_config_from_dict  # noqa: E402
from hrlmpc.mpc_command import SOLVER, SearchReport, WorkerStep  # noqa: E402
from hrlmpc.mpc_problem import MPCProblem, project_disk  # noqa: E402
from hrlmpc.ppo_config import ppo_config_from_dict  # noqa: E402
from hrlmpc.ppo_mpc_config import PPOMPCConfig, ppo_mpc_config_from_dict  # noqa: E402
from hrlmpc.reachability import ReachableTargets, support_directions  # noqa: E402
from hrlmpc.reevaluate import (  # noqa: E402
    REEVALUATIONS,
    load_controller,
    main,
    reevaluate_run,
    summarize_reevaluations,
)
from hrlmpc.train_hppo import train as train_hppo  # noqa: E402
from hrlmpc.train_ppo import load_agent  # noqa: E402
from hrlmpc.train_ppo import train as train_ppo  # noqa: E402
from hrlmpc.train_ppo_mpc import train as train_ppo_mpc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BOX = Box(x=(3.0, 4.0), y=(-0.5, 0.5))


class PDWorker:
    """A stand-in Worker (as in ``tests/test_train_ppo_mpc.py``): ``u = 2 (g - p) - 2 v`` on the disk of the
    plant; every step reports the solver's outcome."""

    instances: list[PDWorker] = []

    def __init__(self, problem: MPCProblem, num_slots: int) -> None:
        self.problem = problem
        self.num_slots = num_slots
        self.closed = False
        PDWorker.instances.append(self)

    def act(self, x: Any, target: Any, steps_left: int, slot: int = 0) -> tuple[np.ndarray, WorkerStep]:
        self.problem.waypoint_stage(steps_left)
        state, goal = np.asarray(x, dtype=np.float64), np.asarray(target, dtype=np.float64)
        u = project_disk(2.0 * (goal - state[:2]) - 2.0 * state[2:], self.problem.params.a_max)
        info = WorkerStep(outcome=SOLVER, solve_ms=0.5, free_binaries=0, tube_ratio=float("nan"), cost=1.0,
                          cut_rounds=1, plan=None, invalid_candidate=0, solver_errors=0)
        return u, info

    def reset(self, slot: int | None = None) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class DiskTargetsStandIn:
    """Stand-in targets of architecture 4: support points on a disk of 0.8 m around the free drift of the state,
    a projection that moves every point up by 1 cm; ``release`` marks the targets closed."""

    instances: list[ReachableTargets] = []

    @classmethod
    def build(cls, problem: MPCProblem, config: PPOMPCConfig) -> ReachableTargets:
        assert config.reachability is not None
        directions = support_directions(config.reachability.directions)

        def support(x: np.ndarray) -> np.ndarray:
            return x[:2] + 0.5 * x[2:] + 0.8 * directions

        def project(x: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, SearchReport]:
            return q + np.array([0.0, 0.01]), SearchReport()

        targets = ReachableTargets(support=support, project=project, radius=1.2,
                                   release=lambda: setattr(targets, "released", True))
        cls.instances.append(targets)
        return targets


def _env(horizon: int = 12) -> EnvConfig:
    data = copy.deepcopy(yaml.safe_load((REPO / "configs" / "env" / "tunnel.yaml").read_text(encoding="utf-8")))
    data["task"]["horizon"] = horizon
    return env_config_from_dict(data)


def _agent(name: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((REPO / "configs" / "agent" / f"{name}.yaml").read_text(encoding="utf-8"))
    data["evaluation"].update(every=40, grid=[2, 2], seed=99)
    return data


def _runs(tmp_path: Path) -> dict[str, Path]:
    """One tiny run per architecture, under ``tmp_path/<algorithm>``."""
    ppo = _agent("ppo")
    ppo["rollout"].update(num_envs=2, num_steps=20)
    ppo["update"].update(epochs=2, minibatches=2)
    hppo = _agent("hppo")
    hppo["rollout"].update(num_envs=2, num_steps=20)
    for level in ("manager", "worker"):
        hppo[level]["update"].update(epochs=2, minibatches=2)
    mpc = _agent("ppo_mpc")
    mpc["rollout"].update(num_envs=2, num_steps=20)
    mpc["manager"]["update"].update(epochs=2, minibatches=2)
    reach = _agent("ppo_mpc_reachability")
    reach["rollout"].update(num_envs=2, num_steps=20)
    reach["manager"]["update"].update(epochs=2, minibatches=2)
    env = _env()
    provenance = {"env_file": "configs/env/tunnel.yaml"}
    runs = {
        "ppo": train_ppo(env, ppo_config_from_dict(ppo), budget=40, seed=1, run_dir=tmp_path / "ppo", echo=None,
                         provenance=provenance),
        "hppo": train_hppo(env, hppo_config_from_dict(hppo), budget=40, seed=1, run_dir=tmp_path / "hppo",
                           echo=None, provenance=provenance),
        "ppo_mpc": train_ppo_mpc(env, ppo_mpc_config_from_dict(mpc), budget=40, seed=1,
                                 run_dir=tmp_path / "ppo_mpc", worker_factory=PDWorker, echo=None,
                                 provenance=provenance),
        "ppo_mpc_reachability": train_ppo_mpc(env, ppo_mpc_config_from_dict(reach), budget=40, seed=1,
                                              run_dir=tmp_path / "reach", worker_factory=PDWorker,
                                              targets_factory=DiskTargetsStandIn.build, echo=None,
                                              provenance=provenance),
    }
    return runs


def _lines(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / REEVALUATIONS).read_text(encoding="utf-8").splitlines()]


def test_every_architecture_is_reevaluated_from_a_grid_in_a_box(tmp_path: Path) -> None:
    """D70: the final policy of a run, with the mode (D57), from the grid of starts over a box; one line per
    re-evaluation in ``reevaluations.jsonl`` with its name, the starts and the per-start results."""
    runs = _runs(tmp_path)
    starts = spawn_grid(BOX, (2, 2))
    for name, run_dir in runs.items():
        line = reevaluate_run(run_dir, BOX, (2, 2), name="escape", worker_factory=PDWorker,
                              targets_factory=DiskTargetsStandIn.build)
        assert line["name"] == "escape" and line["action"] == "mode" and line["seed"] == 99
        assert line["box"] == {"x": [3.0, 4.0], "y": [-0.5, 0.5]} and line["grid"] == [2, 2]
        np.testing.assert_allclose(line["starts"], starts)
        assert len(line["length"]) == 4 and len(line["success"]) == 4 and len(line["lower_bound"]) == 4
        assert set(line["metrics"]) >= {"eval/success_rate", "eval/length", "eval/contact_fraction",
                                        "eval/unshaped_return", "eval/arrival_gap_mean"}
        assert _lines(run_dir) == [line], name
    # PPO: the same episodes as a direct evaluation of the loaded agent with the mode
    agent = load_agent(runs["ppo"])
    direct = evaluate(agent.mode_action, _env(), starts, seed=99)
    assert _lines(runs["ppo"])[0]["length"] == [episode.length for episode in direct.episodes]
    # the mean is also available, and lines accumulate
    line = reevaluate_run(runs["ppo"], BOX, (2, 2), name="escape_mean", action="mean")
    assert line["action"] == "mean" and [ln["name"] for ln in _lines(runs["ppo"])] == ["escape", "escape_mean"]


def test_the_controllers_of_the_mpc_runs_are_closed(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    before = len(PDWorker.instances)
    controller, env, close = load_controller(runs["ppo_mpc"], "mode", 4, worker_factory=PDWorker)
    assert controller.inputs and env.task.horizon == 12
    worker = PDWorker.instances[-1]
    assert len(PDWorker.instances) == before + 1 and worker.num_slots == 4 and not worker.closed
    close()
    assert worker.closed
    DiskTargetsStandIn.instances.clear()
    _, _, close = load_controller(runs["ppo_mpc_reachability"], "mode", 4, worker_factory=PDWorker,
                                  targets_factory=DiskTargetsStandIn.build)
    assert len(DiskTargetsStandIn.instances) == 1
    close()
    assert getattr(DiskTargetsStandIn.instances[0], "released", False)


def test_starts_must_be_admissible(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    with pytest.raises(ValueError):
        reevaluate_run(runs["ppo"], Box(x=(10.5, 10.8), y=(-0.5, 0.5)), (2, 2), name="goal")  # in the goal region


def test_the_summary_groups_the_runs_and_the_command_line_runs_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = _runs(tmp_path)
    for run_dir in (runs["ppo"], runs["hppo"]):
        reevaluate_run(run_dir, BOX, (2, 2), name="escape")
    text = summarize_reevaluations([runs["ppo"], runs["hppo"]], "escape")
    assert "escape" in text and "ppo tunnel" in text and "hppo tunnel" in text and "success" in text
    assert main(["--name", "cli", "--box", "3.0", "4.0", "-0.5", "0.5", "--grid", "2", "2", str(runs["ppo"]),
                 str(runs["hppo"])]) == 0
    out = capsys.readouterr().out
    assert "cli" in out and "ppo tunnel" in out
    assert [ln["name"] for ln in _lines(runs["ppo"])] == ["escape", "cli"]
