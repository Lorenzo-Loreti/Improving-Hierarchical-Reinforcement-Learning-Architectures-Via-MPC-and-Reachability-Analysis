"""Training of hPPO on the navigation environment (architecture 2; decision log D6, D20, D24-D29).

Usage, from the repository root::

    python -m hrlmpc.train_hppo --env configs/env/tunnel.yaml --agent configs/agent/hppo.yaml \
        --budget 204800 --seeds 1 2 3 4 5

A PPO Manager sets a target every ``H`` steps on the common loop of
:mod:`hrlmpc.hierarchy`; a PPO Worker, conditioned on the target, acts at
every step. Both levels are the in-house PPO of D19. After each rollout the
Worker is updated on its own transitions (GAE at its discount, bootstrapping at
every end of an episode, D25), then the Manager on the segments that ended
(GAE at ``gamma^tau``, D27).

Each seed writes the run directory ``<runs-dir>/<layout>/hppo/seed<seed>-<UTC time>`` with:

    config.yaml        resolved configuration: environment, agent, budget, seed
    meta.json          provenance (hrlmpc.utils.runlog)
    metrics.jsonl      one record per update and per evaluation, keyed by the sample counters
    evaluations.jsonl  per-start results of every evaluation
    final.pt           both levels' weights, optimizer states and target statistics

The budget counts samples, physical steps during training summed over the
agents (D6); the Manager's decisions are counted alongside. It must be a whole
number of updates and of evaluation intervals. Evaluations run the
deterministic hierarchy in an environment of their own, draw no random number
of the training and are not samples (D20, D29).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from hrlmpc.config import EnvConfig, env_config_from_dict, env_config_to_dict, load_env_config
from hrlmpc.env import NavigationEnv
from hrlmpc.evaluation import evaluate, spawn_grid
from hrlmpc.hierarchy import (
    HierarchicalController,
    HierarchicalRollout,
    Hierarchy,
    LearnedWorker,
    collect_hierarchical_rollout,
    disk_hierarchy,
    segment_gae,
    start_segments,
)
from hrlmpc.hppo_config import HPPOConfig, hppo_config_from_dict, hppo_config_to_dict, load_hppo_config
from hrlmpc.ppo import Batch, PPOAgent, make_batch
from hrlmpc.rollout import compute_gae, episode_metrics
from hrlmpc.utils.runlog import RunLogger
from hrlmpc.utils.seeding import seed_everything

WORKER_INPUT_SIZE = 6
"""The Worker's input: the observation and the target offset (D26)."""


def hierarchy_of(env_config: EnvConfig, hppo_config: HPPOConfig) -> Hierarchy:
    """Targets and segments of hPPO (D24, D27), with the Manager's discount."""
    return disk_hierarchy(
        env_config,
        segment_steps=hppo_config.hierarchy.segment_steps,
        reach_factor=hppo_config.hierarchy.target_reach_factor,
        discount=hppo_config.manager.update.discount,
    )


def deterministic_controller(manager: PPOAgent, worker: PPOAgent, hierarchy: Hierarchy) -> HierarchicalController:
    """The hierarchy with the means of both policies, for the evaluation (D29); fresh for each evaluation."""
    return HierarchicalController(hierarchy, manager.deterministic_action, worker.deterministic_action)


def _renamed(metrics: Mapping[str, float], level: str) -> dict[str, float]:
    """``train/<name>`` of an update as ``<level>/<name>``."""
    return {f"{level}/{key.split('/', 1)[1]}": value for key, value in metrics.items()}


def _manager_batch(rollout: HierarchicalRollout, gae_lambda: float, device: torch.device) -> Batch:
    seg = rollout.segments
    advantages, returns = segment_gae(seg, gae_lambda)

    def tensor(values: Any) -> torch.Tensor:
        return torch.tensor(np.asarray(values, dtype=np.float32), device=device)

    return Batch(
        obs=tensor(seg.obs),
        actions=tensor(seg.actions),
        log_probs=tensor(seg.log_probs),
        values=tensor(seg.values),
        returns=tensor(returns),
        advantages=tensor(advantages),
    )


def _hierarchy_metrics(rollout: HierarchicalRollout) -> dict[str, float]:
    seg, reward = rollout.segments, rollout.worker_reward
    result = {
        "hierarchy/segments": float(len(seg)),  # also the decisions of the rollout
        "worker/reward": float(reward.total.mean()),
        "worker/reward_progress": float(reward.progress.mean()),
        "worker/reward_contact": float(reward.contact.mean()),
        "worker/reward_effort": float(reward.effort.mean()),
    }
    if len(seg):
        result["hierarchy/segment_steps"] = float(seg.steps.mean())
        result["hierarchy/target_distance"] = float(seg.end_distances.mean())
    return result


def _format_gap(value: float) -> str:
    return "  -  " if np.isnan(value) else f"{value:5.3f}"


def train(
    env_config: EnvConfig,
    hppo_config: HPPOConfig,
    *,
    budget: int,
    seed: int,
    run_dir: str | Path,
    provenance: Mapping[str, Any] | None = None,
    repo_dir: str | Path | None = None,
    echo: Callable[[str], None] | None = print,
) -> Path:
    """Train hPPO for ``budget`` samples and write the run directory.

    Args:
        env_config: Layout, physics, task and reward.
        hppo_config: Hierarchy, both levels, rollout, evaluation and runtime.
        budget: Samples (D6); a positive multiple of the samples per update
            and of the evaluation interval.
        seed: Seed of PyTorch, NumPy and the training environment.
        run_dir: Directory of the run; it must not exist.
        provenance: Extra entries for ``config.yaml``, e.g. the source files.
        repo_dir: Directory inside the git checkout, for ``meta.json``; by
            default the package's own directory.
        echo: Receives one line per evaluation; ``None`` keeps quiet.

    Returns:
        The run directory.

    Raises:
        ValueError: If the budget is not a positive multiple of the update and
            evaluation sizes, the seed is the evaluation's seed, a provenance
            entry would overwrite a setting, or a CUDA device lacks the
            ``CUBLAS_WORKSPACE_CONFIG`` that deterministic algorithms need;
            nothing is written then.
    """
    per_update, every = hppo_config.batch_size, hppo_config.evaluation.every
    if budget < 1 or budget % per_update or budget % every:
        raise ValueError(
            f"the budget must be a positive multiple of the {per_update} samples per update and of the "
            f"evaluation interval of {every} samples, got {budget}"
        )
    if seed == hppo_config.evaluation.seed:
        raise ValueError(f"the training seed {seed} must differ from the evaluation seed")
    reserved = {"algorithm", "seed", "budget", "env", "agent"}
    if reserved & set(provenance or {}):
        raise ValueError(f"provenance entries may not overwrite {sorted(reserved & set(provenance or {}))}")
    if hppo_config.runtime.device.startswith("cuda") and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        raise ValueError("deterministic training on CUDA needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment")
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads, benchmark = torch.get_num_threads(), torch.backends.cudnn.benchmark
    try:
        return _train(env_config, hppo_config, budget, seed, Path(run_dir), provenance, repo_dir, echo)
    finally:  # leave the process's global PyTorch settings as they were
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)
        torch.backends.cudnn.benchmark = benchmark


def _train(
    env_config: EnvConfig,
    hppo_config: HPPOConfig,
    budget: int,
    seed: int,
    path: Path,
    provenance: Mapping[str, Any] | None,
    repo_dir: str | Path | None,
    echo: Callable[[str], None] | None,
) -> Path:
    per_update, every = hppo_config.batch_size, hppo_config.evaluation.every
    seed_everything(seed)
    torch.set_num_threads(hppo_config.runtime.torch_threads)
    device = hppo_config.runtime.device
    env = NavigationEnv(env_config, hppo_config.rollout.num_envs, seed=seed)
    hierarchy = hierarchy_of(env_config, hppo_config)
    m_cfg, w_cfg = hppo_config.manager, hppo_config.worker
    manager = PPOAgent(env.observation_size, env.action_size, m_cfg.network, m_cfg.update, device=device)
    worker = PPOAgent(WORKER_INPUT_SIZE, env.action_size, w_cfg.network, w_cfg.update, device=device)
    starts = spawn_grid(env_config.task.spawn, hppo_config.evaluation.grid)
    config: dict[str, Any] = {
        "algorithm": "hppo",
        "seed": seed,
        "budget": budget,
        **dict(provenance or {}),
        "env": env_config_to_dict(env_config),
        "agent": hppo_config_to_dict(hppo_config),
    }
    num_updates = budget // per_update
    source = Path(__file__).resolve().parent if repo_dir is None else repo_dir
    with RunLogger(path, config, seed=seed, repo_dir=source) as logger:

        def evaluate_now() -> None:
            controller = deterministic_controller(manager, worker, hierarchy)
            result = evaluate(controller, env_config, starts, seed=hppo_config.evaluation.seed)
            metrics = result.metrics()
            logger.log(metrics, env.counter)
            with (path / "evaluations.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"env_steps": env.counter.env_steps, **result.details()}) + "\n")
            if echo is not None:
                echo(
                    f"{path.name}: {env.counter.env_steps:>8} samples | success {metrics['eval/success_rate']:4.2f}"
                    f" | unshaped return {metrics['eval/unshaped_return']:8.1f}"
                    f" | gap {_format_gap(metrics['eval/arrival_gap_mean'])}"
                    f" | contact {metrics['eval/contact_fraction']:4.2f}"
                )

        evaluate_now()
        state = start_segments(env, hierarchy, manager.act, manager.value)
        learned = LearnedWorker(policy=worker.act, value=worker.value)
        started = time.perf_counter()
        training_time = 0.0
        for k in range(num_updates):
            tick = time.perf_counter()
            for agent, settings in ((manager, m_cfg), (worker, w_cfg)):
                if settings.update.anneal_learning_rate:
                    agent.set_learning_rate_fraction(1.0 - k / num_updates)
            rates = {"manager": manager.learning_rates, "worker": worker.learning_rates}
            rollout, state = collect_hierarchical_rollout(
                env, state, hppo_config.rollout.num_steps, hierarchy, manager.act, manager.value, learned
            )
            transitions = rollout.worker
            assert transitions is not None  # a LearnedWorker records its transitions
            advantages, returns = compute_gae(
                transitions.rewards,
                transitions.values,
                transitions.next_values,
                transitions.terminated,
                transitions.done,
                discount=w_cfg.update.discount,
                gae_lambda=w_cfg.update.gae_lambda,
            )
            worker_batch = make_batch(transitions, advantages, returns, worker.device)
            metrics: dict[str, Any] = _renamed(worker.update(worker_batch), "worker")
            batch = _manager_batch(rollout, m_cfg.update.gae_lambda, manager.device)
            metrics.update(_renamed(manager.update(batch), "manager"))
            training_time += time.perf_counter() - tick
            metrics.update(episode_metrics(rollout.episodes))
            metrics.update(_hierarchy_metrics(rollout))
            for level, (actor_rate, critic_rate) in rates.items():
                metrics[f"{level}/learning_rate"] = actor_rate
                metrics[f"{level}/critic_learning_rate"] = critic_rate
            metrics.update(
                {
                    "train/updates": k + 1,
                    "time/train_s": training_time,
                    "time/elapsed_s": time.perf_counter() - started,
                    "time/sps": env.counter.env_steps / training_time,
                }
            )
            logger.log(metrics, env.counter)
            if env.counter.env_steps % every == 0:
                evaluate_now()
        torch.save({"manager": manager.state_dict(), "worker": worker.state_dict()}, path / "final.pt")
    return path


def load_hierarchy(run_dir: str | Path, checkpoint: str = "final.pt") -> tuple[PPOAgent, PPOAgent, Hierarchy]:
    """The Manager, the Worker and the hierarchy of a run, rebuilt from its ``config.yaml`` and a checkpoint."""
    path = Path(run_dir)
    config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    env_config = env_config_from_dict(config["env"])
    hppo_config = hppo_config_from_dict(config["agent"])
    device = hppo_config.runtime.device
    m_cfg, w_cfg = hppo_config.manager, hppo_config.worker
    manager = PPOAgent(NavigationEnv.observation_size, NavigationEnv.action_size, m_cfg.network, m_cfg.update, device)
    worker = PPOAgent(WORKER_INPUT_SIZE, NavigationEnv.action_size, w_cfg.network, w_cfg.update, device)
    state = torch.load(path / checkpoint, map_location=manager.device, weights_only=True)
    manager.load_state_dict(state["manager"])
    worker.load_state_dict(state["worker"])
    return manager, worker, hierarchy_of(env_config, hppo_config)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: one training run per seed."""
    parser = argparse.ArgumentParser(description="Train hPPO on a layout of the navigation environment.")
    parser.add_argument(
        "--env", type=Path, required=True, help="environment configuration, e.g. configs/env/tunnel.yaml"
    )
    parser.add_argument("--agent", type=Path, required=True, help="hPPO configuration, e.g. configs/agent/hppo.yaml")
    parser.add_argument(
        "--budget", type=int, required=True, help="samples per run: physical steps summed over the agents (D6)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="training seeds, one run each")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="root of the run directories")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) < len(args.seeds):
        parser.error(f"every seed may appear once, got {args.seeds}")
    env_config = load_env_config(args.env)
    hppo_config = load_hppo_config(args.agent)
    bad = [seed for seed in args.seeds if seed < 0 or seed == hppo_config.evaluation.seed]
    if bad:  # checked before the first run, not when its turn comes
        parser.error(f"training seeds must be non-negative and differ from the evaluation seed, got {bad}")
    for seed in args.seeds:
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.runs_dir / args.env.stem / "hppo" / f"seed{seed}-{stamp}"
        train(
            env_config,
            hppo_config,
            budget=args.budget,
            seed=seed,
            run_dir=run_dir,
            provenance={"env_file": str(args.env), "agent_file": str(args.agent)},
        )
        print(f"run written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
