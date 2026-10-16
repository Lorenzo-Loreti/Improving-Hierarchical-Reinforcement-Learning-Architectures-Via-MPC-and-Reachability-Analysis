"""Tests of the PPO_MPC training loop and its command line (decision log D6, D24-D35, D37-D40, D43, D45, D47,
D49, D50); they need PyTorch.

The Worker is a NumPy stand-in with the interface of :class:`hrlmpc.mpc_worker.MPCWorker`
(a PD law toward the target), so the loop is tested without the solvers;
``tests/test_mpc_worker.py`` tests the tube MPC itself.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

import hrlmpc.train_ppo_mpc as train_ppo_mpc  # noqa: E402
from hrlmpc.config import EnvConfig, env_config_from_dict  # noqa: E402
from hrlmpc.evaluation import EVAL_ACTIONS  # noqa: E402
from hrlmpc.mpc_command import CANDIDATE, EMERGENCY, EXITS, KEPT, SOLVER, SearchReport, WorkerStep  # noqa: E402
from hrlmpc.mpc_problem import MPCProblem, project_disk  # noqa: E402
from hrlmpc.ppo_mpc_config import PPOMPCConfig, ppo_mpc_config_from_dict  # noqa: E402
from hrlmpc.reachability import ReachableTargets, support_directions  # noqa: E402
from hrlmpc.train_ppo_mpc import load_manager, main, train  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class PDWorker:
    """A stand-in Worker: ``u = 2 (g - p) - 2 v`` toward the target, on the disk of the plant.

    Its outcomes cycle through every kind; a shifted plan outside its tube
    comes with every fallback and a solver error with every brake. The tube
    ratio is NaN on slot 0 and ``slot (t + 1) / 100`` on the others, ``t`` the
    slot's steps since its last reset. ``steps`` records ``(slot, outcome, ratio)``
    of every call, so that tests can recompute the statistics, and ``left`` the
    steps left in the segment that each call was given (D37).
    """

    instances: list[PDWorker] = []
    CYCLE = (SOLVER, SOLVER, CANDIDATE, SOLVER, KEPT, SOLVER, EMERGENCY)

    def __init__(self, problem: MPCProblem, num_slots: int) -> None:
        self.problem = problem
        self.num_slots = num_slots
        self.resets: list[int | None] = []
        self.steps: list[tuple[int, int, float]] = []
        self.left: list[int] = []
        self.targets: list[np.ndarray] = []
        self.counts = [0] * num_slots
        self.closed = False
        PDWorker.instances.append(self)

    def input(self, x: np.ndarray, target: np.ndarray) -> np.ndarray:
        return project_disk(2.0 * (target - x[:2]) - 2.0 * x[2:], self.problem.params.a_max)

    def act(self, x: Any, target: Any, steps_left: int, slot: int = 0) -> tuple[np.ndarray, WorkerStep]:
        self.problem.waypoint_stage(steps_left)  # the real Worker's check
        self.left.append(steps_left)
        self.targets.append(np.array(target, dtype=np.float64))
        u = self.input(np.asarray(x, dtype=np.float64), np.asarray(target, dtype=np.float64))
        outcome = self.CYCLE[len(self.steps) % len(self.CYCLE)]
        self.counts[slot] += 1
        ratio = float("nan") if slot == 0 else slot * self.counts[slot] / 100.0
        self.steps.append((slot, outcome, ratio))
        info = WorkerStep(outcome=outcome, solve_ms=0.5, free_binaries=slot, tube_ratio=ratio, cost=1.0,
                          cut_rounds=1, plan=None, invalid_candidate=int(outcome == CANDIDATE),
                          solver_errors=int(outcome == EMERGENCY))
        return u, info

    def reset(self, slot: int | None = None) -> None:
        self.resets.append(slot)
        for s in range(self.num_slots) if slot is None else (slot,):
            self.counts[s] = 0

    def close(self) -> None:
        self.closed = True


class DiskTargetsStandIn:
    """Stand-in targets of architecture 4: support points on a disk of 0.8 m around the free drift of the state,
    and a projection that moves every point up, recording what it returns; ``release`` marks it closed.

    The shift is 1 cm, or, with ``by_side``, 2 cm above the axis ``y = 0`` and none below it, so that decisions
    of different agents differ."""

    instances: list[ReachableTargets] = []
    projected: list[np.ndarray] = []
    by_side = False

    @classmethod
    def build(cls, problem: MPCProblem, config: PPOMPCConfig) -> ReachableTargets:
        assert config.reachability is not None
        directions = support_directions(config.reachability.directions)

        def support(x: np.ndarray) -> np.ndarray:
            return x[:2] + 0.5 * x[2:] + 0.8 * directions

        def project(x: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, SearchReport]:
            shift = (0.02 if x[1] > 0.0 else 0.0) if cls.by_side else 0.01
            g = q + np.array([0.0, shift])
            cls.projected.append(g.copy())
            return g, SearchReport()

        targets = ReachableTargets(support=support, project=project, radius=1.2,
                                   release=lambda: setattr(targets, "released", True))
        cls.instances.append(targets)
        return targets


class PushWorker(PDWorker):
    """Full thrust along +x, whatever the target: the agents nearer the goal finish first."""

    def input(self, x: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.array([self.problem.params.a_max, 0.0])


def _env(**sections: dict[str, Any]) -> EnvConfig:
    data = copy.deepcopy(yaml.safe_load((REPO / "configs" / "env" / "tunnel.yaml").read_text(encoding="utf-8")))
    data["task"]["horizon"] = 15  # episodes end inside the short rollouts, some in the middle of a segment
    for section, values in sections.items():
        data[section].update(values)
    return env_config_from_dict(data)


def _config(**changes: Any) -> PPOMPCConfig:
    """A tiny hierarchy: 2 agents x 20 steps = 40 samples and at least 4 segments per update."""
    data = yaml.safe_load((REPO / "configs" / "agent" / "ppo_mpc.yaml").read_text(encoding="utf-8"))
    data["rollout"].update(num_envs=2, num_steps=20)
    data["manager"]["update"].update(epochs=2, minibatches=2)
    data["evaluation"].update(every=40, grid=[2, 2], seed=99)
    for path, value in changes.items():
        node = data
        keys = path.split("__")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return ppo_mpc_config_from_dict(data)


def _records(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]


def _details(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()]


def _without_time(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The records without the wall-clock times: under ``time/``, and the solvers' times ``*_ms_*``."""
    return [{k: v for k, v in r.items() if not k.startswith("time/") and "_ms_" not in k} for r in records]


def _recomputed(steps: list[tuple[int, int, float]], agents: int, prefix: str, lengths: Any = None) -> dict[str, Any]:
    """The statistics of the logged calls, rows of ``agents`` calls, over each agent's first ``lengths`` rows."""
    slots = np.array([s for s, _, _ in steps]).reshape(-1, agents)
    assert np.all(slots == np.arange(agents))  # one call per agent and step, in the agents' order
    stats = {
        "outcome": np.array([o for _, o, _ in steps], dtype=float).reshape(-1, agents),
        "tube_ratio": np.array([r for _, _, r in steps], dtype=float).reshape(-1, agents),
    }
    keep = np.ones(stats["outcome"].shape, dtype=bool)
    if lengths is not None:
        keep = np.arange(len(keep))[:, None] < np.asarray(lengths)[None, :]
    outcome, ratio = stats["outcome"][keep], stats["tube_ratio"][keep]
    return {
        f"{prefix}/steps": float(keep.sum()),
        f"{prefix}/candidate_rate": float(np.mean(outcome == CANDIDATE)),
        f"{prefix}/kept_rate": float(np.mean(outcome == KEPT)),
        f"{prefix}/emergency_count": float(np.sum(outcome == EMERGENCY)),
        f"{prefix}/invalid_candidate_count": float(np.sum(outcome == CANDIDATE)),
        f"{prefix}/solver_error_count": float(np.sum(outcome == EMERGENCY)),
        f"{prefix}/tube_ratio_max": float(np.nanmax(ratio)),
    }


@pytest.mark.parametrize(("d_bar", "expected"), [(0.0, 15), (0.5, 16), (1.0, 17)])
def test_the_horizon_follows_the_disturbance(d_bar: float, expected: int) -> None:
    """D45: the shipped configuration leaves the horizon to the problem, H + the steps to stop; an explicit one
    stays."""
    env = _env(physics={"d_bar": d_bar})
    assert train_ppo_mpc.problem_of(env, _config()).N == expected
    assert train_ppo_mpc.problem_of(env, _config(mpc__horizon=20)).N == 20
    with pytest.raises(ValueError, match="N >= "):
        train_ppo_mpc.problem_of(env, _config(mpc__horizon=expected - 1))


def test_a_short_run_writes_its_directory(tmp_path: Path) -> None:
    PDWorker.instances.clear()
    run_dir = train(_env(), _config(), budget=80, seed=1, run_dir=tmp_path / "run", worker_factory=PDWorker, echo=None)
    names = sorted(p.name for p in run_dir.iterdir())
    assert names == ["config.yaml", "evaluations.jsonl", "final.pt", "meta.json", "metrics.jsonl"]
    records = _records(run_dir)
    updates = [r for r in records if "train/updates" in r]
    evaluations = [r for r in records if "eval/success_rate" in r]
    assert [r["env_steps"] for r in updates] == [40, 80]
    assert [r["env_steps"] for r in evaluations] == [0, 40, 80]
    rollout_worker, eval_worker = PDWorker.instances
    for k, record in enumerate(updates):
        assert {"mpc/solve_ms_mean", "mpc/solve_ms_max", "mpc/free_binaries_mean", "mpc/cut_rounds_mean",
                "manager/policy_loss", "manager/learning_rate", "hierarchy/segments",
                "worker/reward_progress"} <= set(record)
        expected = _recomputed(rollout_worker.steps[40 * k : 40 * (k + 1)], 2, "mpc")
        assert {key: record[key] for key in expected} == pytest.approx(expected)
        assert record["mpc/steps"] == 40.0  # every sample of the rollout, one Worker step each
        # D47: how each search ended, here always the default of the stand-in
        assert record["mpc/exit_converged_count"] == 40.0
        assert all(record[f"mpc/exit_{name}_count"] == 0.0 for name in EXITS[1:])
        assert record["mpc/exact_inaccurate_count"] == record["mpc/exact_infeasible_count"] == 0.0
        assert record["mpc/exact_uncertified_count"] == record["mpc/exact_accepted_count"] == 0.0  # D49
        assert record["mpc/best_applied_count"] == record["mpc/accepted_applied_count"] == 0.0
        assert not any(k.startswith("worker/") and not k.startswith("worker/reward") for k in record)  # no learning
        assert record["time/rollout_s"] > 0.0 and record["time/update_s"] > 0.0  # D50
    np.testing.assert_allclose(
        [r["time/train_s"] for r in updates], np.cumsum([r["time/rollout_s"] + r["time/update_s"] for r in updates]),
        rtol=1e-12,
    )
    for record in evaluations:  # D43, D50: the evaluation with the mode under its own names, and the compute
        assert {"eval_mode/success_rate", "eval_mode_mpc/steps", "eval_mode_mpc/solve_ms_mean",
                "eval_mode_mpc/exit_best_count", "eval_mode_hierarchy/reached_rate"} <= set(record)
        assert record["eval_mpc/steps"] == record["eval_mode_mpc/steps"] == 60.0  # 4 agents, episodes of 15 steps
        for name in ("eval", "eval_mode"):
            assert record[f"time/{name}_s"] > 0.0
            assert record[f"time/{name}_step_ms_max"] >= record[f"time/{name}_step_ms_mean"] > 0.0
    assert [(d["env_steps"], d["action"]) for d in _details(run_dir)] == [
        (n, a) for n in (0, 40, 80) for a in EVAL_ACTIONS
    ]
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert rollout_worker.problem.N == 15 == eval_worker.problem.N  # the automatic horizon at d_bar = 0 (D45)
    assert config["algorithm"] == "ppo_mpc" and config["agent"]["mpc"]["horizon"] == 15  # stated in the run
    assert rollout_worker.num_slots == 2 and eval_worker.num_slots == 4
    assert rollout_worker.closed and eval_worker.closed
    assert any(slot is not None for slot in rollout_worker.resets)  # episodes ended and their slots were reset
    assert set(torch.load(run_dir / "final.pt", weights_only=True)) == {"manager"}


def test_the_evaluation_counts_only_the_steps_of_the_evaluated_episodes(tmp_path: Path) -> None:
    """D29: agents that finished keep being stepped; their steps are no evaluation statistics."""
    PDWorker.instances.clear()
    env = _env(geometry={"goal_x": 3.0})  # the starts at x = 2 arrive, those at x = 0 do not
    run_dir = train(env, _config(), budget=40, seed=2, run_dir=tmp_path / "run", worker_factory=PushWorker, echo=None)
    evaluations = [r for r in _records(run_dir) if "eval/success_rate" in r]
    details = _details(run_dir)
    assert [d["action"] for d in details] == list(EVAL_ACTIONS) * len(evaluations)
    eval_worker = PDWorker.instances[1]
    calls = 0
    for k, record in enumerate(evaluations):  # the mean's steps, then the mode's (D43)
        for prefix, detail in (("eval_mpc", details[2 * k]), ("eval_mode_mpc", details[2 * k + 1])):
            lengths = detail["length"]
            assert lengths == [15, 15, 11, 11]
            steps = eval_worker.steps[calls : calls + 4 * max(lengths)]
            calls += len(steps)
            expected = _recomputed(steps, 4, prefix, lengths)
            assert {key: record[key] for key in expected} == pytest.approx(expected)
            assert record[f"{prefix}/steps"] == float(sum(lengths)) < 4 * max(lengths)
            assert record[f"{prefix}/tube_ratio_max"] == pytest.approx(0.33)  # slot 3 for its 11 steps, not 15
    assert calls == len(eval_worker.steps)


def test_every_evaluation_runs_the_mean_and_then_the_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D43: the Manager's mean and then its mode, with the same Worker, every slot of which is reset before each."""
    seen: list[str] = []
    original = train_ppo_mpc.evaluate

    def spy(controller: Any, config: Any, starts: Any, **kwargs: Any) -> Any:
        seen.append(controller.manager.__name__)
        return original(controller, config, starts, **kwargs)

    monkeypatch.setattr(train_ppo_mpc, "evaluate", spy)
    PDWorker.instances.clear()
    train(_env(), _config(), budget=40, seed=1, run_dir=tmp_path / "run", worker_factory=PDWorker, echo=None)
    assert seen == ["deterministic_action", "mode_action"] * 2
    assert PDWorker.instances[1].resets == [0, 1, 2, 3] * 4  # two points, two actions


def test_a_run_is_repeatable_and_depends_on_its_seed(tmp_path: Path) -> None:
    env = _env(physics={"d_bar": 0.5})
    runs = [
        train(env, _config(mpc__horizon=16), budget=80, seed=s, run_dir=tmp_path / name, worker_factory=PDWorker,
              echo=None)
        for s, name in ((3, "a"), (3, "b"), (4, "c"))
    ]
    a, b, c = (_without_time(_records(r)) for r in runs)
    assert a == b
    assert a != c
    weights = [torch.load(r / "final.pt", weights_only=True) for r in runs[:2]]
    for network in ("actor", "critic"):
        for key, value in weights[0]["manager"][network].items():
            assert torch.equal(value, weights[1]["manager"][network][key])


@pytest.mark.parametrize("budget", [0, 60, 120])
def test_the_budget_must_end_on_an_update_and_an_evaluation(tmp_path: Path, budget: int) -> None:
    with pytest.raises(ValueError, match="multiple"):
        train(_env(), _config(evaluation__every=80), budget=budget, seed=1, run_dir=tmp_path / "run",
              worker_factory=PDWorker, echo=None)
    assert not (tmp_path / "run").exists()


def test_a_horizon_without_room_to_stop_after_a_segment_is_refused(tmp_path: Path) -> None:
    """D37: an explicit horizon must satisfy N >= H + the steps to stop from the planned speed, 15 at d_bar = 0
    and 16 at 0.5 (the automatic one of D45 always does); checked before any Worker is built or anything is
    written."""
    PDWorker.instances.clear()
    with pytest.raises(ValueError, match="N >= 15"):
        train(_env(), _config(mpc__horizon=14), budget=40, seed=1, run_dir=tmp_path / "a", worker_factory=PDWorker,
              echo=None)
    with pytest.raises(ValueError, match="N >= 16"):
        train(_env(physics={"d_bar": 0.5}), _config(mpc__horizon=15), budget=40, seed=1, run_dir=tmp_path / "b",
              worker_factory=PDWorker, echo=None)
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists() and not PDWorker.instances


def test_the_worker_gets_the_steps_left_in_each_segment(tmp_path: Path) -> None:
    """D37: in training and in the evaluations, H - j at step j of a segment; the evaluated agents start together."""
    PDWorker.instances.clear()
    train(_env(), _config(), budget=40, seed=1, run_dir=tmp_path / "run", worker_factory=PDWorker, echo=None)
    rollout_worker, eval_worker = PDWorker.instances
    assert set(rollout_worker.left) == set(range(1, 11))
    first = np.array(rollout_worker.left[:20]).reshape(10, 2)  # both agents decide at the start: 10, 9, ..., 1
    np.testing.assert_array_equal(first, np.repeat(np.arange(10, 0, -1)[:, None], 2, axis=1))
    calls = np.array(eval_worker.left).reshape(-1, 4)  # 4 agents per call
    assert np.all(calls == calls[:, :1]) and len(calls) == 4 * 15  # two points, two actions, episodes of 15 steps
    np.testing.assert_array_equal(calls[:15, 0], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6])


def test_inconsistent_settings_are_refused_before_anything_is_written(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evaluation seed"):
        train(_env(), _config(), budget=40, seed=99, run_dir=tmp_path / "a", worker_factory=PDWorker, echo=None)
    with pytest.raises(ValueError, match="provenance"):
        train(_env(), _config(), budget=40, seed=1, run_dir=tmp_path / "b", provenance={"seed": 5},
              worker_factory=PDWorker, echo=None)
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_the_workers_are_closed_when_training_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    PDWorker.instances.clear()

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("rollout failed")

    monkeypatch.setattr(train_ppo_mpc, "collect_hierarchical_rollout", broken)
    with pytest.raises(RuntimeError, match="rollout failed"):
        train(_env(), _config(), budget=40, seed=1, run_dir=tmp_path / "run", worker_factory=PDWorker, echo=None)
    assert len(PDWorker.instances) == 2 and all(w.closed for w in PDWorker.instances)


def test_a_worker_that_cannot_be_built_leaves_nothing_on_disk(tmp_path: Path) -> None:
    """A missing solver or licence fails before the run directory exists, which would otherwise break the analysis."""
    PDWorker.instances.clear()

    def factory(problem: MPCProblem, num_slots: int) -> PDWorker:
        if PDWorker.instances:
            raise RuntimeError("no licence for the evaluation Worker")
        return PDWorker(problem, num_slots)

    with pytest.raises(RuntimeError, match="no licence"):
        train(_env(), _config(), budget=40, seed=1, run_dir=tmp_path / "run", worker_factory=factory, echo=None)
    assert not (tmp_path / "run").exists()
    assert len(PDWorker.instances) == 1 and PDWorker.instances[0].closed  # the one built was released


def test_every_worker_is_closed_even_if_one_fails_to(tmp_path: Path) -> None:
    PDWorker.instances.clear()

    class FailsToClose(PDWorker):
        def close(self) -> None:
            super().close()
            raise OSError("could not release the solver")

    def factory(problem: MPCProblem, num_slots: int) -> PDWorker:
        return (PDWorker if PDWorker.instances else FailsToClose)(problem, num_slots)

    with pytest.raises(OSError, match="release"):
        train(_env(), _config(), budget=40, seed=1, run_dir=tmp_path / "run", worker_factory=factory, echo=None)
    assert len(PDWorker.instances) == 2 and all(w.closed for w in PDWorker.instances)


def test_the_saved_manager_reproduces_the_last_evaluation(tmp_path: Path) -> None:
    env = _env(physics={"d_bar": 0.5})
    run_dir = train(env, _config(), budget=80, seed=5, run_dir=tmp_path / "run", worker_factory=PDWorker, echo=None)
    manager, hierarchy, problem = load_manager(run_dir)
    assert hierarchy.segment_steps == 10 and problem.N == 16 and problem.tube.d_bar == 0.5  # the run's horizon
    from hrlmpc.env import ObservationMap
    from hrlmpc.evaluation import evaluate, spawn_grid
    from hrlmpc.hierarchy import HierarchicalController
    from hrlmpc.mpc_command import MPCCommand

    command = MPCCommand(PDWorker(problem, 4), ObservationMap.from_config(env))
    details = _details(run_dir)
    for action in EVAL_ACTIONS:  # D43
        controller = HierarchicalController(hierarchy, manager.deterministic_policy(action), command.as_worker())
        result = evaluate(controller, env, spawn_grid(env.task.spawn, (2, 2)), seed=99)
        last = [d for d in details if d["action"] == action][-1]
        assert [ep.length for ep in result.episodes] == last["length"]
        np.testing.assert_allclose([ep.episode_return for ep in result.episodes], last["episode_return"])


def test_a_run_from_before_the_regularizer_is_loaded_without_it(tmp_path: Path) -> None:
    """D46: the ``config.yaml`` of a run of steps 9 and 10 has no ``mpc.start_weight``; its Worker had no
    regularizer, so :func:`load_manager` rebuilds it with ``start_weight = 0``, at any disturbance."""
    run_dir = train(_env(physics={"d_bar": 0.5}), _config(), budget=40, seed=2, run_dir=tmp_path / "run",
                    worker_factory=PDWorker, echo=None)
    path = run_dir / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["agent"]["mpc"]["start_weight"] == 0.1 and raw["agent"]["mpc"]["horizon"] == 16
    del raw["agent"]["mpc"]["start_weight"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    _, _, problem = load_manager(run_dir)
    assert problem.settings.start_weight == 0.0 and not problem.regularized and problem.N == 16


def test_a_run_from_before_the_workers_price_is_loaded_at_the_shaped_price(tmp_path: Path) -> None:
    """D66: the ``config.yaml`` of a run from before has no ``hierarchy.worker_progress_coef``; its reload, which
    only evaluates the deterministic policy, takes the shaped reward's 10 (the price is immaterial there)."""
    run_dir = train(_env(), _config(), budget=40, seed=2, run_dir=tmp_path / "run", worker_factory=PDWorker,
                    echo=None)
    path = run_dir / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["agent"]["hierarchy"]["worker_progress_coef"] == 10.0
    del raw["agent"]["hierarchy"]["worker_progress_coef"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    _, hierarchy, _ = load_manager(run_dir)
    assert hierarchy.progress_coef == 10.0


def test_the_learning_rates_are_constant_unless_annealed(tmp_path: Path) -> None:
    """Pinned to a learning rate of 3e-4 (the shipped file carries the selection of D65, 1e-3)."""
    constant = _config(manager__update__learning_rate=3e-4)
    run_dir = train(_env(), constant, budget=160, seed=1, run_dir=tmp_path / "constant", worker_factory=PDWorker,
                    echo=None)
    records = [r for r in _records(run_dir) if "train/updates" in r]
    np.testing.assert_allclose([r["manager/learning_rate"] for r in records], [3e-4] * 4, rtol=1e-12)
    np.testing.assert_allclose([r["manager/critic_learning_rate"] for r in records], [9e-4] * 4, rtol=1e-12)
    annealed = _config(manager__update__learning_rate=3e-4, manager__update__anneal_learning_rate=True)
    run_dir = train(_env(), annealed, budget=160, seed=1, run_dir=tmp_path / "annealed", worker_factory=PDWorker,
                    echo=None)
    records = [r for r in _records(run_dir) if "train/updates" in r]
    np.testing.assert_allclose([r["manager/learning_rate"] for r in records], [3e-4, 2.25e-4, 1.5e-4, 0.75e-4],
                               rtol=1e-9)


def test_the_manager_is_updated_on_its_segments_at_its_own_discount(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """D27: the Manager's batch is the rollout's segments with the GAE at gamma^tau and the configured lambda."""
    import hrlmpc.ppo as ppo_module
    import hrlmpc.train_hppo as train_hppo
    from hrlmpc.hierarchy import segment_gae

    rollouts: list[Any] = []
    batches: list[Any] = []
    lambdas: list[float] = []
    collect, update = train_ppo_mpc.collect_hierarchical_rollout, ppo_module.PPOAgent.update
    gae = train_hppo.segment_gae

    def spy_collect(*args: Any, **kwargs: Any) -> Any:
        result = collect(*args, **kwargs)
        rollouts.append(result[0])
        return result

    def spy_update(self: Any, batch: Any) -> Any:
        batches.append(batch)
        return update(self, batch)

    def spy_gae(segments: Any, gae_lambda: float) -> Any:
        lambdas.append(gae_lambda)
        return gae(segments, gae_lambda)

    monkeypatch.setattr(train_ppo_mpc, "collect_hierarchical_rollout", spy_collect)
    monkeypatch.setattr(ppo_module.PPOAgent, "update", spy_update)
    monkeypatch.setattr(train_hppo, "segment_gae", spy_gae)
    config = _config(manager__update__discount=0.9, manager__update__gae_lambda=0.7, mpc__horizon=16)
    train(_env(physics={"d_bar": 0.5}), config, budget=80, seed=0, run_dir=tmp_path / "run",
          worker_factory=PDWorker, echo=None)
    assert len(rollouts) == len(batches) == 2 and lambdas == [0.7, 0.7]

    def same(tensor: Any, values: Any) -> None:
        np.testing.assert_array_equal(tensor.cpu().numpy(), np.asarray(values, dtype=np.float32))

    for rollout, batch in zip(rollouts, batches, strict=True):
        seg = rollout.segments
        np.testing.assert_allclose(seg.discounts, 0.9**seg.steps, rtol=1e-12)
        assert seg.steps.max() == 10 and seg.steps.min() < 10  # episodes of 15 steps cut segments
        advantages, returns = segment_gae(seg, 0.7)
        for name in ("obs", "actions", "log_probs", "values"):
            same(getattr(batch, name), getattr(seg, name))
        same(batch.advantages, advantages)
        same(batch.returns, returns)


def test_evaluations_do_not_change_the_training(tmp_path: Path) -> None:
    """D6, D20: with evaluations every 40 or every 80 samples, training follows the same trajectory."""
    env = _env(physics={"d_bar": 0.5})
    often = train(env, _config(mpc__horizon=16), budget=160, seed=5, run_dir=tmp_path / "often",
                  worker_factory=PDWorker, echo=None)
    rarely = train(env, _config(evaluation__every=80, mpc__horizon=16), budget=160, seed=5,
                   run_dir=tmp_path / "rarely", worker_factory=PDWorker, echo=None)

    def training(run_dir: Path) -> list[dict[str, Any]]:
        return [r for r in _without_time(_records(run_dir)) if "train/updates" in r]

    assert len(training(often)) == 4 and training(often) == training(rarely)
    assert [r["env_steps"] for r in _records(rarely) if "eval/success_rate" in r] == [0, 80, 160]
    weights = [torch.load(run / "final.pt", weights_only=True) for run in (often, rarely)]
    for network in ("actor", "critic"):
        for key, value in weights[0]["manager"][network].items():
            assert torch.equal(value, weights[1]["manager"][network][key])


def test_training_leaves_the_global_torch_settings_as_they_were(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(2)
    try:
        train(_env(), _config(), budget=40, seed=1, run_dir=tmp_path / "run", worker_factory=PDWorker, echo=None)
        assert not torch.are_deterministic_algorithms_enabled() and torch.get_num_threads() == 2

        def broken(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("rollout failed")

        monkeypatch.setattr(train_ppo_mpc, "collect_hierarchical_rollout", broken)
        with pytest.raises(RuntimeError, match="rollout failed"):
            train(_env(), _config(), budget=40, seed=1, run_dir=tmp_path / "failed", worker_factory=PDWorker,
                  echo=None)
        assert not torch.are_deterministic_algorithms_enabled() and torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(threads)


def test_the_command_line_trains_every_seed(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_ppo_mpc, "solver_worker", PDWorker)
    agent = tmp_path / "agent.yaml"
    data = yaml.safe_load((REPO / "configs" / "agent" / "ppo_mpc.yaml").read_text(encoding="utf-8"))
    data["rollout"].update(num_envs=2, num_steps=20)
    data["manager"]["update"].update(epochs=1, minibatches=2)
    data["evaluation"].update(every=40, grid=[2, 2])
    agent.write_text(yaml.safe_dump(data), encoding="utf-8")
    env = tmp_path / "tunnel.yaml"
    env_data = yaml.safe_load((REPO / "configs" / "env" / "tunnel.yaml").read_text(encoding="utf-8"))
    env_data["task"]["horizon"] = 15
    env.write_text(yaml.safe_dump(env_data), encoding="utf-8")
    assert main(["--env", str(env), "--agent", str(agent), "--budget", "40", "--seeds", "1", "2",
                 "--runs-dir", str(tmp_path / "runs")]) == 0
    runs = sorted((tmp_path / "runs" / "tunnel" / "ppo_mpc").iterdir())
    assert [p.name.split("-")[0] for p in runs] == ["seed1", "seed2"]
    config = yaml.safe_load((runs[0] / "config.yaml").read_text(encoding="utf-8"))
    assert config["env_file"] == str(env) and config["agent_file"] == str(agent)
    out = capsys.readouterr().out
    assert "run written to" in out and "mean/mode: success" in out and "| solve" in out  # D50
    with pytest.raises(SystemExit):
        main(["--env", str(env), "--agent", str(agent), "--budget", "40", "--seeds", "1", "1"])
    with pytest.raises(SystemExit):
        main(["--env", str(env), "--agent", str(agent), "--budget", "40", "--seeds", "12345"])


# --------------------------------------------------------------------------
# Architecture 4: the reachable targets (D38-D40)
# --------------------------------------------------------------------------


def _reach_config(**changes: Any) -> PPOMPCConfig:
    return _config(reachability={"directions": 8}, **changes)


def test_architecture_4_trains_on_its_projected_targets(tmp_path: Path) -> None:
    """D38, D39: the Worker gets the projected targets, and the decisions are logged under reach/ and eval_reach/."""
    PDWorker.instances.clear()
    DiskTargetsStandIn.instances.clear()
    DiskTargetsStandIn.projected.clear()
    run_dir = train(_env(), _reach_config(), budget=80, seed=1, run_dir=tmp_path / "run", worker_factory=PDWorker,
                    targets_factory=DiskTargetsStandIn.build, echo=None)
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert config["algorithm"] == "ppo_mpc_reachability" and config["agent"]["reachability"] == {"directions": 8}
    (targets,) = DiskTargetsStandIn.instances
    assert getattr(targets, "released", False)  # closed with the run
    records = _records(run_dir)
    updates = [r for r in records if "train/updates" in r]
    evaluations = [r for r in records if "eval/success_rate" in r]
    assert all({"reach/decisions", "reach/support_ms_mean", "reach/projection_mean", "reach/hull_area_mean",
                "reach/projection_failure_count", "hierarchy/reached_rate"} <= set(r) for r in updates)
    assert all({"eval_reach/decisions", "eval_hierarchy/reached_rate", "eval_mode_reach/decisions",
                "eval_mode_hierarchy/reached_rate", "eval_mode_reach/projection_exit_best_count"} <= set(r)
               for r in evaluations)  # D43, D50
    for record in updates:
        assert record["reach/projection_mean"] == pytest.approx(0.01)  # the stand-in's shift
        assert record["reach/projection_failure_count"] == 0.0
        assert record["reach/projection_exit_converged_count"] == record["reach/decisions"]  # D47
        assert record["reach/projection_exact_accepted_count"] == 0.0  # D49
    # the decisions of training: one per segment, those of the first update included
    assert sum(r["reach/decisions"] for r in updates) == sum(r["hierarchy/segments"] for r in updates) + 2
    # every target the Worker steered to was a projected one, held for its whole segment
    rollout_worker = PDWorker.instances[0]
    projected = {tuple(np.round(g, 12)) for g in DiskTargetsStandIn.projected}
    assert all(tuple(np.round(g, 12)) in projected for g in rollout_worker.targets)
    assert all(r[f"{prefix}/projection_mean"] == pytest.approx(0.01)
               for r in evaluations for prefix in ("eval_reach", "eval_mode_reach"))


@pytest.mark.parametrize("d_bar", [0.0, 0.5])
def test_the_reached_rate_counts_the_full_segments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   d_bar: float) -> None:
    """D40: within r_p + 0.1 m of the target at the end of a segment of H steps; r_p = 0.065 m at d_bar = 0.5."""
    rollouts: list[Any] = []
    collect = train_ppo_mpc.collect_hierarchical_rollout

    def spy(*args: Any, **kwargs: Any) -> Any:
        result = collect(*args, **kwargs)
        rollouts.append(result[0])
        return result

    monkeypatch.setattr(train_ppo_mpc, "collect_hierarchical_rollout", spy)
    PDWorker.instances.clear()
    run_dir = train(_env(physics={"d_bar": d_bar}), _config(mpc__horizon=16), budget=80, seed=1,
                    run_dir=tmp_path / "run", worker_factory=PDWorker, echo=None)
    tolerance = PDWorker.instances[0].problem.tube.position_radius + 0.1
    assert tolerance == pytest.approx(0.1 if d_bar == 0.0 else 0.165, abs=1e-3)
    updates = [r for r in _records(run_dir) if "train/updates" in r]
    for rollout, record in zip(rollouts, updates, strict=True):
        full = rollout.segments.steps == 10
        distances = rollout.segments.end_distances[full]
        assert record["hierarchy/full_segments"] == float(full.sum()) > 0
        assert record["hierarchy/reached_rate"] == pytest.approx(float(np.mean(distances <= tolerance)))
        assert record["hierarchy/end_distance_mean"] == pytest.approx(float(distances.mean()))


def test_the_evaluation_counts_the_segments_and_decisions_of_the_evaluated_episodes(tmp_path: Path) -> None:
    """D29, D40: the agents from x = 2 reach the goal at x = 2.9 in exactly 10 steps, so their episodes end with
    their first segment: their second decision and their first segment's end, both at step 10, do not count."""
    PDWorker.instances.clear()
    DiskTargetsStandIn.by_side = True  # agents at y = 1 are shifted 2 cm, those at y = -1 not at all
    try:
        env = _env(geometry={"goal_x": 2.9})
        run_dir = train(env, _reach_config(), budget=40, seed=2, run_dir=tmp_path / "run", worker_factory=PushWorker,
                        targets_factory=DiskTargetsStandIn.build, echo=None)
    finally:
        DiskTargetsStandIn.by_side = False
    evaluations = [r for r in _records(run_dir) if "eval/success_rate" in r]
    details = _details(run_dir)
    assert len(details) == 2 * len(evaluations)
    for k, record in enumerate(evaluations):  # both actions (D43)
        for prefix, detail in (("eval", details[2 * k]), ("eval_mode", details[2 * k + 1])):
            assert detail["length"] == [15, 15, 10, 10]  # starts (0, -1), (0, 1), (2, -1), (2, 1)
            assert record[f"{prefix}_reach/decisions"] == 4 + 2  # all four at step 0, the two from x = 0 at step 10
            assert record[f"{prefix}_reach/projection_mean"] == pytest.approx(0.01)  # (0 + 2 + 0 + 2 + 0 + 2) cm / 6
            assert record[f"{prefix}_hierarchy/full_segments"] == 2.0  # the agents from x = 0, ended at step 10


def test_targets_that_cannot_be_built_leave_nothing_on_disk(tmp_path: Path) -> None:
    PDWorker.instances.clear()

    def broken(problem: MPCProblem, config: PPOMPCConfig) -> ReachableTargets:
        raise RuntimeError("no licence for the projection")

    with pytest.raises(RuntimeError, match="no licence"):
        train(_env(), _reach_config(), budget=40, seed=1, run_dir=tmp_path / "run", worker_factory=PDWorker,
              targets_factory=broken, echo=None)
    assert not (tmp_path / "run").exists()
    assert len(PDWorker.instances) == 2 and all(w.closed for w in PDWorker.instances)


def test_the_saved_manager_of_architecture_4_comes_with_its_targets(tmp_path: Path) -> None:
    DiskTargetsStandIn.instances.clear()
    run_dir = train(_env(), _reach_config(), budget=40, seed=3, run_dir=tmp_path / "run", worker_factory=PDWorker,
                    targets_factory=DiskTargetsStandIn.build, echo=None)
    manager, hierarchy, problem = load_manager(run_dir, targets_factory=DiskTargetsStandIn.build)
    assert hierarchy.targets is DiskTargetsStandIn.instances[-1] and len(DiskTargetsStandIn.instances) == 2
    assert problem.N == 15 and hierarchy.segment_steps == 10
