"""Tests of the hPPO training loop and its command line (decision log D6, D24-D29); they need PyTorch."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from hrlmpc.config import EnvConfig, env_config_from_dict, env_config_to_dict  # noqa: E402
from hrlmpc.evaluation import evaluate, spawn_grid  # noqa: E402
from hrlmpc.hppo_config import HPPOConfig, hppo_config_from_dict, hppo_config_to_dict  # noqa: E402
from hrlmpc.train_hppo import deterministic_controller, load_hierarchy, main, train  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _env(**sections: dict[str, Any]) -> EnvConfig:
    data = copy.deepcopy(yaml.safe_load((REPO / "configs" / "env" / "tunnel.yaml").read_text(encoding="utf-8")))
    data["task"]["horizon"] = 15  # episodes end inside the short rollouts, some in the middle of a segment
    for section, values in sections.items():
        data[section].update(values)
    return env_config_from_dict(data)


def _hppo(**changes: Any) -> HPPOConfig:
    """A tiny hierarchy: 2 agents x 20 steps = 40 samples and at least 4 segments per update."""
    data = yaml.safe_load((REPO / "configs" / "agent" / "hppo.yaml").read_text(encoding="utf-8"))
    data["rollout"].update(num_envs=2, num_steps=20)
    data["manager"]["update"].update(epochs=2, minibatches=2)
    data["worker"]["update"].update(epochs=2, minibatches=2)
    data["evaluation"].update(every=40, grid=[2, 2], seed=99)
    for path, value in changes.items():
        node = data
        keys = path.split("__")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return hppo_config_from_dict(data)


def _records(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]


def _without_time(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if not k.startswith("time/")} for r in records]


def test_a_short_run_writes_its_directory(tmp_path: Path) -> None:
    run_dir = train(_env(), _hppo(), budget=80, seed=1, run_dir=tmp_path / "run")
    names = sorted(p.name for p in run_dir.iterdir())
    assert names == ["config.yaml", "evaluations.jsonl", "final.pt", "meta.json", "metrics.jsonl"]
    records = _records(run_dir)
    evaluations = [r for r in records if "eval/success_rate" in r]
    updates = [r for r in records if "train/updates" in r]
    assert [r["env_steps"] for r in evaluations] == [0, 40, 80]
    assert [r["env_steps"] for r in updates] == [40, 80]
    assert [r["train/updates"] for r in updates] == [1, 2]
    decisions = [r["manager_decisions"] for r in updates]
    assert decisions[0] >= 2 + 4 and decisions[1] - decisions[0] >= 4  # the start, then >= 2 segments per agent
    assert evaluations[0]["manager_decisions"] == 0
    for record in updates:
        assert record["hierarchy/segments"] >= 4 and 1.0 <= record["hierarchy/segment_steps"] <= 10.0
        for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "explained_variance"):
            assert f"manager/{name}" in record and f"worker/{name}" in record
        for name in ("reward", "reward_progress", "reward_contact", "reward_effort"):
            assert f"worker/{name}" in record
        assert "hierarchy/target_distance" in record and record["rollout/episodes"] >= 1
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert (config["algorithm"], config["budget"], config["seed"]) == ("hppo", 80, 1)
    assert env_config_from_dict(config["env"]) == _env()
    assert hppo_config_from_dict(config["agent"]) == _hppo()
    details = [json.loads(line) for line in (run_dir / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [d["env_steps"] for d in details] == [0, 40, 80]
    assert details[0]["starts"] == spawn_grid(_env().task.spawn, (2, 2)).tolist()


def test_the_learning_rates_are_constant_unless_annealed(tmp_path: Path) -> None:
    run_dir = train(_env(), _hppo(), budget=160, seed=1, run_dir=tmp_path / "constant")
    records = [r for r in _records(run_dir) if "train/updates" in r]
    for level in ("manager", "worker"):
        np.testing.assert_allclose([r[f"{level}/learning_rate"] for r in records], [3e-4] * 4, rtol=1e-12)
        np.testing.assert_allclose([r[f"{level}/critic_learning_rate"] for r in records], [9e-4] * 4, rtol=1e-12)
    annealed = _hppo(worker__update__anneal_learning_rate=True)  # the Worker only
    run_dir = train(_env(), annealed, budget=160, seed=1, run_dir=tmp_path / "annealed")
    records = [r for r in _records(run_dir) if "train/updates" in r]
    expected = [3e-4, 2.25e-4, 1.5e-4, 0.75e-4]
    np.testing.assert_allclose([r["worker/learning_rate"] for r in records], expected, rtol=1e-9)
    np.testing.assert_allclose([r["manager/learning_rate"] for r in records], [3e-4] * 4, rtol=1e-12)


def test_a_run_is_repeatable_and_depends_on_its_seed(tmp_path: Path) -> None:
    env = _env(physics={"d_bar": 0.5})
    a = train(env, _hppo(), budget=80, seed=3, run_dir=tmp_path / "a")
    b = train(env, _hppo(), budget=80, seed=3, run_dir=tmp_path / "b")
    c = train(env, _hppo(), budget=80, seed=4, run_dir=tmp_path / "c")
    assert _without_time(_records(a)) == _without_time(_records(b))
    assert _without_time(_records(a)) != _without_time(_records(c))
    weights_a = torch.load(a / "final.pt", weights_only=True)
    weights_b = torch.load(b / "final.pt", weights_only=True)
    for level in ("manager", "worker"):
        for network in ("actor", "critic"):
            for key, value in weights_a[level][network].items():
                assert torch.equal(value, weights_b[level][network][key])


@pytest.mark.parametrize(("budget", "every"), [(0, 40), (60, 40), (120, 80)])
def test_the_budget_must_end_on_an_update_and_an_evaluation(tmp_path: Path, budget: int, every: int) -> None:
    with pytest.raises(ValueError, match="budget"):
        train(_env(), _hppo(evaluation__every=every), budget=budget, seed=0, run_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_each_level_uses_its_own_discount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D25, D27: the Worker's GAE at its discount without terminal states, the Manager's at gamma^tau."""
    import hrlmpc.train_hppo as module

    worker_calls: list[tuple[Any, ...]] = []
    manager_calls: list[Any] = []
    original_gae, original_segment_gae = module.compute_gae, module.segment_gae

    def spy_gae(*args: Any, **kwargs: Any) -> Any:
        worker_calls.append((kwargs["discount"], kwargs["gae_lambda"], bool(np.asarray(args[3]).any())))
        return original_gae(*args, **kwargs)

    def spy_segment_gae(segments: Any, gae_lambda: float) -> Any:
        manager_calls.append((segments.discounts.copy(), segments.steps.copy(), gae_lambda))
        return original_segment_gae(segments, gae_lambda)

    monkeypatch.setattr(module, "compute_gae", spy_gae)
    monkeypatch.setattr(module, "segment_gae", spy_segment_gae)
    config = _hppo(manager__update__discount=0.9, worker__update__discount=0.8, manager__update__gae_lambda=0.7)
    train(_env(), config, budget=80, seed=0, run_dir=tmp_path / "run")
    assert worker_calls == [(0.8, 0.95, False), (0.8, 0.95, False)]
    assert len(manager_calls) == 2
    for discounts, steps, lam in manager_calls:
        np.testing.assert_allclose(discounts, 0.9**steps, rtol=1e-12)
        assert lam == 0.7 and steps.max() == 10 and steps.min() < 10  # episodes of 15 steps cut segments


def test_the_updates_get_the_rollouts_transitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D25, D27: each level is updated on its own transitions and on the advantages of its own GAE."""
    import hrlmpc.ppo as ppo_module
    import hrlmpc.train_hppo as module
    from hrlmpc.hierarchy import segment_gae
    from hrlmpc.rollout import compute_gae

    rollouts: list[Any] = []
    batches: list[tuple[int, Any]] = []
    collect, update = module.collect_hierarchical_rollout, ppo_module.PPOAgent.update

    def spy_collect(*args: Any, **kwargs: Any) -> Any:
        result = collect(*args, **kwargs)
        rollouts.append(result[0])
        return result

    def spy_update(self: Any, batch: Any) -> Any:
        batches.append((self.obs_size, batch))
        return update(self, batch)

    monkeypatch.setattr(module, "collect_hierarchical_rollout", spy_collect)
    monkeypatch.setattr(ppo_module.PPOAgent, "update", spy_update)
    config = _hppo(manager__update__discount=0.9, worker__update__discount=0.8, manager__update__gae_lambda=0.7)
    train(_env(physics={"d_bar": 0.5}), config, budget=80, seed=0, run_dir=tmp_path / "run")
    assert [size for size, _ in batches] == [6, 4, 6, 4]  # the Worker first, then the Manager

    def same(tensor: Any, values: Any) -> None:
        np.testing.assert_array_equal(tensor.cpu().numpy(), np.asarray(values, dtype=np.float32))

    for k, rollout in enumerate(rollouts):
        worker_batch, manager_batch = batches[2 * k][1], batches[2 * k + 1][1]
        w = rollout.worker
        advantages, returns = compute_gae(
            w.rewards, w.values, w.next_values, w.terminated, w.done, discount=0.8, gae_lambda=0.95
        )
        for name, values in (("obs", w.obs.reshape(-1, 6)), ("actions", w.actions.reshape(-1, 2))):
            same(getattr(worker_batch, name), values)
        for name, values in (("log_probs", w.log_probs), ("values", w.values)):
            same(getattr(worker_batch, name), values.reshape(-1))
        same(worker_batch.advantages, advantages.reshape(-1))
        same(worker_batch.returns, returns.reshape(-1))
        seg = rollout.segments
        advantages, returns = segment_gae(seg, 0.7)
        np.testing.assert_allclose(seg.discounts, 0.9**seg.steps, rtol=1e-12)
        assert seg.done.any() and not seg.done.all()
        for name in ("obs", "actions", "log_probs", "values"):
            same(getattr(manager_batch, name), getattr(seg, name))
        same(manager_batch.advantages, advantages)
        same(manager_batch.returns, returns)


def test_the_final_checkpoint_reproduces_the_last_evaluation(tmp_path: Path) -> None:
    env, hppo = _env(physics={"d_bar": 0.5}), _hppo()
    run_dir = train(env, hppo, budget=80, seed=2, run_dir=tmp_path / "run")
    manager, worker, hierarchy = load_hierarchy(run_dir)
    controller = deterministic_controller(manager, worker, hierarchy)
    result = evaluate(controller, env, spawn_grid(env.task.spawn, (2, 2)), seed=hppo.evaluation.seed)
    last = json.loads((run_dir / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert result.details()["length"] == last["length"]
    assert result.details()["unshaped_return"] == last["unshaped_return"]


def test_the_command_line_trains_every_seed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env_file, agent_file = tmp_path / "tiny.yaml", tmp_path / "hppo.yaml"
    env_file.write_text(yaml.safe_dump(env_config_to_dict(_env())), encoding="utf-8")
    agent_file.write_text(yaml.safe_dump(hppo_config_to_dict(_hppo())), encoding="utf-8")
    argv = ["--env", str(env_file), "--agent", str(agent_file), "--budget", "40", "--seeds", "1", "2"]
    assert main([*argv, "--runs-dir", str(tmp_path / "runs")]) == 0
    runs = sorted((tmp_path / "runs" / "tiny" / "hppo").iterdir())
    assert [p.name.split("-")[0] for p in runs] == ["seed1", "seed2"]
    config = yaml.safe_load((runs[0] / "config.yaml").read_text(encoding="utf-8"))
    assert config["env_file"] == str(env_file) and config["agent_file"] == str(agent_file)
    assert str(runs[1]) in capsys.readouterr().out


def test_evaluations_do_not_change_the_training(tmp_path: Path) -> None:
    """D6, D20: with evaluations every 40 or every 80 samples, training follows the same trajectory."""
    env = _env(physics={"d_bar": 0.5})
    often = train(env, _hppo(), budget=160, seed=5, run_dir=tmp_path / "often")
    rarely = train(env, _hppo(evaluation__every=80), budget=160, seed=5, run_dir=tmp_path / "rarely")

    def training(run_dir: Path) -> list[dict[str, Any]]:
        return [r for r in _without_time(_records(run_dir)) if "train/updates" in r]

    assert len(training(often)) == 4 and training(often) == training(rarely)
    evaluated = [r["env_steps"] for r in _records(rarely) if "eval/success_rate" in r]
    assert evaluated == [0, 80, 160]
    weights = [torch.load(run / "final.pt", weights_only=True) for run in (often, rarely)]
    for level in ("manager", "worker"):
        for network in ("actor", "critic"):
            for key, value in weights[0][level][network].items():
                assert torch.equal(value, weights[1][level][network][key])


def test_inconsistent_settings_are_refused_before_anything_is_written(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evaluation seed"):
        train(_env(), _hppo(), budget=40, seed=99, run_dir=tmp_path / "run")  # _hppo() evaluates with seed 99
    with pytest.raises(ValueError, match="overwrite"):
        train(_env(), _hppo(), budget=40, seed=1, run_dir=tmp_path / "run", provenance={"seed": 5})
    assert not (tmp_path / "run").exists()
    with pytest.raises(SystemExit):
        main(["--env", "x.yaml", "--agent", "y.yaml", "--budget", "40", "--seeds", "1", "1"])
    env_file, agent_file = tmp_path / "tiny.yaml", tmp_path / "hppo.yaml"
    env_file.write_text(yaml.safe_dump(env_config_to_dict(_env())), encoding="utf-8")
    agent_file.write_text(yaml.safe_dump(hppo_config_to_dict(_hppo())), encoding="utf-8")
    argv = ["--env", str(env_file), "--agent", str(agent_file), "--budget", "40", "--seeds", "1", "99"]
    with pytest.raises(SystemExit):  # 99 is the evaluation seed: refused before seed 1 trains
        main([*argv, "--runs-dir", str(tmp_path / "runs")])
    assert not (tmp_path / "runs").exists()


def test_training_leaves_the_global_torch_settings_as_they_were(tmp_path: Path) -> None:
    threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(2)
    try:
        train(_env(), _hppo(), budget=40, seed=1, run_dir=tmp_path / "run", echo=None)
        assert not torch.are_deterministic_algorithms_enabled() and torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(threads)
