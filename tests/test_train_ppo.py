"""Tests of the flat PPO training loop and its command line (decision log D6, D13, D19, D20); they need PyTorch."""

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
from hrlmpc.ppo_config import PPOConfig, ppo_config_from_dict, ppo_config_to_dict  # noqa: E402
from hrlmpc.train_ppo import load_agent, main, train  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _env(**sections: dict[str, Any]) -> EnvConfig:
    data = copy.deepcopy(yaml.safe_load((REPO / "configs" / "env" / "tunnel.yaml").read_text(encoding="utf-8")))
    data["task"]["horizon"] = 10  # episodes end inside the short rollouts
    for section, values in sections.items():
        data[section].update(values)
    return env_config_from_dict(data)


def _ppo(**sections: dict[str, Any]) -> PPOConfig:
    """A tiny learner: 2 agents x 8 steps = 16 samples per update, evaluations every update."""
    data = yaml.safe_load((REPO / "configs" / "agent" / "ppo.yaml").read_text(encoding="utf-8"))
    data["rollout"].update(num_envs=2, num_steps=8)
    data["update"].update(epochs=2, minibatches=2)
    data["evaluation"].update(every=16, grid=[2, 2], seed=99)
    for section, values in sections.items():
        data[section].update(values)
    return ppo_config_from_dict(data)


def _records(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]


def _without_time(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if not k.startswith("time/")} for r in records]


def test_a_short_run_writes_its_directory(tmp_path: Path) -> None:
    run_dir = train(_env(), _ppo(), budget=32, seed=1, run_dir=tmp_path / "run")
    names = sorted(p.name for p in run_dir.iterdir())
    assert names == ["config.yaml", "evaluations.jsonl", "final.pt", "meta.json", "metrics.jsonl"]
    records = _records(run_dir)
    evaluations = [r for r in records if "eval/success_rate" in r]
    updates = [r for r in records if "train/policy_loss" in r]
    assert [r["env_steps"] for r in evaluations] == [0, 16, 32]
    assert [r["env_steps"] for r in updates] == [16, 32]
    assert all(r["manager_decisions"] == 0 for r in records)
    assert updates[0]["train/updates"] == 1 and updates[1]["train/updates"] == 2
    assert updates[-1]["rollout/episodes"] >= 1  # the horizon of 10 ends episodes inside the rollouts
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert (config["algorithm"], config["budget"], config["seed"], config["discount"]) == ("ppo", 32, 1, 0.99)
    assert config["env"]["reward"]["shaping_discount"] == 1.0  # D22
    assert env_config_from_dict(config["env"]) == _env()
    assert ppo_config_from_dict(config["agent"]) == _ppo()
    details = [json.loads(line) for line in (run_dir / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [d["env_steps"] for d in details] == [0, 16, 32]
    assert details[0]["starts"] == spawn_grid(_env().task.spawn, (2, 2)).tolist()


def test_the_learning_rates_decay_linearly_over_the_budget(tmp_path: Path) -> None:
    run_dir = train(_env(), _ppo(), budget=64, seed=1, run_dir=tmp_path / "run")
    rates = [r["train/learning_rate"] for r in _records(run_dir) if "train/learning_rate" in r]
    np.testing.assert_allclose(rates, [3e-4, 2.25e-4, 1.5e-4, 0.75e-4], rtol=1e-9)
    critic = [r["train/critic_learning_rate"] for r in _records(run_dir) if "train/critic_learning_rate" in r]
    np.testing.assert_allclose(critic, [9e-4, 6.75e-4, 4.5e-4, 2.25e-4], rtol=1e-9)


def test_a_run_is_repeatable_and_depends_on_its_seed(tmp_path: Path) -> None:
    a = train(_env(physics={"d_bar": 0.5}), _ppo(), budget=32, seed=3, run_dir=tmp_path / "a")
    b = train(_env(physics={"d_bar": 0.5}), _ppo(), budget=32, seed=3, run_dir=tmp_path / "b")
    c = train(_env(physics={"d_bar": 0.5}), _ppo(), budget=32, seed=4, run_dir=tmp_path / "c")
    assert _without_time(_records(a)) == _without_time(_records(b))
    assert _without_time(_records(a)) != _without_time(_records(c))
    weights_a = torch.load(a / "final.pt", weights_only=True)
    weights_b = torch.load(b / "final.pt", weights_only=True)
    for network in ("actor", "critic"):
        for key, value in weights_a[network].items():
            assert torch.equal(value, weights_b[network][key])


@pytest.mark.parametrize(("budget", "every"), [(0, 16), (24, 16), (48, 32)])
def test_the_budget_must_end_on_an_update_and_an_evaluation(tmp_path: Path, budget: int, every: int) -> None:
    with pytest.raises(ValueError, match="budget"):
        train(_env(), _ppo(evaluation={"every": every}), budget=budget, seed=0, run_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_the_discount_is_the_agents_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D22: GAE uses the agent configuration's discount, whatever the environment's shaping discount."""
    import hrlmpc.train_ppo as module

    seen: list[Any] = []
    original = module.compute_gae

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append((kwargs["discount"], kwargs["gae_lambda"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "compute_gae", spy)
    env = _env(reward={"shaping_discount": 0.95})
    run_dir = train(env, _ppo(update={"discount": 0.9}), budget=32, seed=0, run_dir=tmp_path / "run")
    assert seen == [(0.9, 0.95), (0.9, 0.95)]
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert config["discount"] == 0.9 and config["env"]["reward"]["shaping_discount"] == 0.95


def test_the_final_checkpoint_reproduces_the_last_evaluation(tmp_path: Path) -> None:
    env, ppo = _env(physics={"d_bar": 0.5}), _ppo()
    run_dir = train(env, ppo, budget=32, seed=2, run_dir=tmp_path / "run")
    agent = load_agent(run_dir)
    result = evaluate(agent.deterministic_action, env, spawn_grid(env.task.spawn, (2, 2)), seed=ppo.evaluation.seed)
    last = json.loads((run_dir / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert result.details()["length"] == last["length"]
    assert result.details()["unshaped_return"] == last["unshaped_return"]


def test_the_command_line_trains_every_seed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env_file, agent_file = tmp_path / "tiny.yaml", tmp_path / "ppo.yaml"
    env_file.write_text(yaml.safe_dump(env_config_to_dict(_env())), encoding="utf-8")
    agent_file.write_text(yaml.safe_dump(ppo_config_to_dict(_ppo())), encoding="utf-8")
    argv = ["--env", str(env_file), "--agent", str(agent_file), "--budget", "32", "--seeds", "1", "2"]
    assert main([*argv, "--runs-dir", str(tmp_path / "runs")]) == 0
    runs = sorted((tmp_path / "runs" / "tiny" / "ppo").iterdir())
    assert [p.name.split("-")[0] for p in runs] == ["seed1", "seed2"]
    config = yaml.safe_load((runs[0] / "config.yaml").read_text(encoding="utf-8"))
    assert config["env_file"] == str(env_file) and config["agent_file"] == str(agent_file)
    assert str(runs[1]) in capsys.readouterr().out


def test_evaluations_do_not_change_the_training(tmp_path: Path) -> None:
    """D6, D20: with evaluations every 16 or every 32 samples, training follows the same trajectory."""
    env = _env(physics={"d_bar": 0.5})
    often = train(env, _ppo(), budget=64, seed=5, run_dir=tmp_path / "often")
    rarely = train(env, _ppo(evaluation={"every": 32}), budget=64, seed=5, run_dir=tmp_path / "rarely")

    def training(run_dir: Path) -> list[dict[str, Any]]:
        return [r for r in _without_time(_records(run_dir)) if "train/policy_loss" in r]

    assert len(training(often)) == 4 and training(often) == training(rarely)
    evaluated = [r["env_steps"] for r in _records(rarely) if "eval/success_rate" in r]
    assert evaluated == [0, 32, 64]
    weights = [torch.load(run / "final.pt", weights_only=True) for run in (often, rarely)]
    for network in ("actor", "critic"):
        for key, value in weights[0][network].items():
            assert torch.equal(value, weights[1][network][key])


def test_inconsistent_settings_are_refused_before_anything_is_written(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evaluation seed"):
        train(_env(), _ppo(), budget=32, seed=99, run_dir=tmp_path / "run")  # _ppo() evaluates with seed 99
    with pytest.raises(ValueError, match="overwrite"):
        train(_env(), _ppo(), budget=32, seed=1, run_dir=tmp_path / "run", provenance={"seed": 5})
    assert not (tmp_path / "run").exists()
    with pytest.raises(SystemExit):
        main(["--env", "x.yaml", "--agent", "y.yaml", "--budget", "32", "--seeds", "1", "1"])


def test_training_leaves_the_global_torch_settings_as_they_were(tmp_path: Path) -> None:
    threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(2)
    try:
        train(_env(), _ppo(), budget=16, seed=1, run_dir=tmp_path / "run", echo=None)
        assert not torch.are_deterministic_algorithms_enabled() and torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(threads)

