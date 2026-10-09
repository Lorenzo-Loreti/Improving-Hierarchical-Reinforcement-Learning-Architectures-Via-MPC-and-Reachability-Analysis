"""Training of PPO_MPC on the navigation environment (architecture 3; decision log D6, D20, D24-D35).

Usage, from the repository root::

    python -m hrlmpc.train_ppo_mpc --env configs/env/tunnel.yaml --agent configs/agent/ppo_mpc.yaml \
        --budget 204800 --seeds 1 2 3 4 5

A PPO Manager sets a target every ``H`` steps on the common loop of
:mod:`hrlmpc.hierarchy`, exactly as in hPPO; the Worker is the tube MPC of
D31-D34 (:mod:`hrlmpc.mpc_worker`), which knows the model, the constraints and
the obstacles, plans contact-free and does not learn. After each rollout the
Manager is updated on the segments that ended (GAE at ``gamma^tau``, D27).

Each seed writes the run directory ``<runs-dir>/<layout>/ppo_mpc/seed<seed>-<UTC time>`` with:

    config.yaml        resolved configuration: environment, agent, budget, seed
    meta.json          provenance (hrlmpc.utils.runlog)
    metrics.jsonl      one record per update and per evaluation, keyed by the sample counters
    evaluations.jsonl  per-start results of every evaluation
    final.pt           the Manager's weights, optimizer state and target statistics

Besides the Manager's and the episodes' metrics, every update logs the
Worker's steps under ``mpc/`` and every evaluation under ``eval_mpc/``
(:func:`hrlmpc.mpc_command.mpc_metrics`): mean and largest solve time, the
rates of the steps on the shifted plan (the fallback when the solvers found
nothing, or kept because cheaper), the counts of emergencies, of shifted plans
outside their tube and of solver failures, the free binaries, the cut rounds
and, with a tube, the largest tube ratio. An evaluation counts the steps of the
evaluated episodes only: agents that finished first keep being stepped, in
ignored episodes (D29), and their solves cost time but no metric. The budget
counts samples, physical steps during training summed over the agents (D6);
the Worker's predictions are not samples, their cost is the solve time.

The Worker needs cvxpy, Clarabel and Gurobi (the extra ``mpc``); tests pass a
stand-in through ``worker_factory``. The Workers are built before the run
directory, so a Worker that cannot be built leaves nothing on disk.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from hrlmpc.config import (
    EnvConfig,
    env_config_from_dict,
    env_config_to_dict,
    load_env_config,
)
from hrlmpc.env import NavigationEnv, ObservationMap
from hrlmpc.evaluation import evaluate, spawn_grid
from hrlmpc.geometry import Layout
from hrlmpc.hierarchy import (
    HierarchicalController,
    Hierarchy,
    collect_hierarchical_rollout,
    disk_hierarchy,
    start_segments,
)
from hrlmpc.mpc_command import MPCCommand, StepActor, episode_mask, mpc_metrics
from hrlmpc.mpc_problem import MPCProblem
from hrlmpc.ppo import PPOAgent
from hrlmpc.ppo_mpc_config import (
    PPOMPCConfig,
    load_ppo_mpc_config,
    ppo_mpc_config_from_dict,
    ppo_mpc_config_to_dict,
)
from hrlmpc.rollout import episode_metrics
from hrlmpc.train_hppo import _format_gap, _hierarchy_metrics, _manager_batch, _renamed
from hrlmpc.utils.runlog import RunLogger
from hrlmpc.utils.seeding import seed_everything

WorkerFactory = Callable[[MPCProblem, int], StepActor]
"""Builds a Worker for a problem and a number of agents."""


def solver_worker(problem: MPCProblem, num_slots: int) -> StepActor:
    """The tube MPC of :mod:`hrlmpc.mpc_worker` (imported here: it needs the solvers)."""
    from hrlmpc.mpc_worker import MPCWorker

    return MPCWorker(problem, num_slots)


def hierarchy_of(env_config: EnvConfig, config: PPOMPCConfig) -> Hierarchy:
    """Targets and segments (D24, D27), as hPPO's, with the Manager's discount."""
    return disk_hierarchy(
        env_config,
        segment_steps=config.hierarchy.segment_steps,
        reach_factor=config.hierarchy.target_reach_factor,
        discount=config.manager.update.discount,
    )


def problem_of(env_config: EnvConfig, config: PPOMPCConfig) -> MPCProblem:
    """The Worker's problem: the environment's plant, map and disturbance, and the settings of D34."""
    return MPCProblem(env_config.physics, Layout.from_config(env_config.geometry), config.mpc)


def _close(worker: StepActor) -> None:
    close = getattr(worker, "close", None)
    if callable(close):
        close()


def train(
    env_config: EnvConfig,
    config: PPOMPCConfig,
    *,
    budget: int,
    seed: int,
    run_dir: str | Path,
    provenance: Mapping[str, Any] | None = None,
    repo_dir: str | Path | None = None,
    echo: Callable[[str], None] | None = print,
    worker_factory: WorkerFactory | None = None,
) -> Path:
    """Train PPO_MPC for ``budget`` samples and write the run directory.

    Args:
        env_config: Layout, physics, task and reward.
        config: Hierarchy, Manager, Worker, rollout, evaluation and runtime.
        budget: Samples (D6); a positive multiple of the samples per update
            and of the evaluation interval.
        seed: Seed of PyTorch, NumPy and the training environment.
        run_dir: Directory of the run; it must not exist.
        provenance: Extra entries for ``config.yaml``, e.g. the source files.
        repo_dir: Directory inside the git checkout, for ``meta.json``; by
            default the package's own directory.
        echo: Receives the Worker's summary and one line per evaluation; ``None`` keeps quiet.
        worker_factory: Builds the Workers of the rollouts and of the evaluations;
            :func:`solver_worker` by default.

    Returns:
        The run directory.

    Raises:
        ValueError: If the budget is not a positive multiple of the update and
            evaluation sizes, the seed is the evaluation's seed, a provenance
            entry would overwrite a setting, or a CUDA device lacks the
            ``CUBLAS_WORKSPACE_CONFIG`` that deterministic algorithms need;
            nothing is written then, nor when a Worker cannot be built.
    """
    per_update, every = config.batch_size, config.evaluation.every
    if budget < 1 or budget % per_update or budget % every:
        raise ValueError(
            f"the budget must be a positive multiple of the {per_update} samples per update and of the "
            f"evaluation interval of {every} samples, got {budget}"
        )
    if seed == config.evaluation.seed:
        raise ValueError(f"the training seed {seed} must differ from the evaluation seed")
    reserved = {"algorithm", "seed", "budget", "env", "agent"}
    if reserved & set(provenance or {}):
        raise ValueError(f"provenance entries may not overwrite {sorted(reserved & set(provenance or {}))}")
    if config.runtime.device.startswith("cuda") and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        raise ValueError("deterministic training on CUDA needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment")
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads, benchmark = torch.get_num_threads(), torch.backends.cudnn.benchmark
    try:
        factory = solver_worker if worker_factory is None else worker_factory
        return _train(env_config, config, budget, seed, Path(run_dir), provenance, repo_dir, echo, factory)
    finally:  # leave the process's global PyTorch settings as they were
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)
        torch.backends.cudnn.benchmark = benchmark


def _train(
    env_config: EnvConfig,
    config: PPOMPCConfig,
    budget: int,
    seed: int,
    path: Path,
    provenance: Mapping[str, Any] | None,
    repo_dir: str | Path | None,
    echo: Callable[[str], None] | None,
    worker_factory: WorkerFactory,
) -> Path:
    per_update, every = config.batch_size, config.evaluation.every
    seed_everything(seed)
    torch.set_num_threads(config.runtime.torch_threads)
    env = NavigationEnv(env_config, config.rollout.num_envs, seed=seed)
    hierarchy = hierarchy_of(env_config, config)
    problem = problem_of(env_config, config)
    m_cfg = config.manager
    manager = PPOAgent(env.observation_size, env.action_size, m_cfg.network, m_cfg.update, device=config.runtime.device)
    starts = spawn_grid(env_config.task.spawn, config.evaluation.grid)
    obs_map = ObservationMap.from_config(env_config)
    run_config: dict[str, Any] = {
        "algorithm": "ppo_mpc",
        "seed": seed,
        "budget": budget,
        **dict(provenance or {}),
        "env": env_config_to_dict(env_config),
        "agent": ppo_mpc_config_to_dict(config),
    }
    num_updates = budget // per_update
    source = Path(__file__).resolve().parent if repo_dir is None else repo_dir
    with contextlib.ExitStack() as stack:  # closes every Worker, even when one fails to close
        rollout_worker = worker_factory(problem, config.rollout.num_envs)
        stack.callback(_close, rollout_worker)
        eval_worker = worker_factory(problem, len(starts))
        stack.callback(_close, eval_worker)
        command = MPCCommand(rollout_worker, obs_map)
        eval_command = MPCCommand(eval_worker, obs_map)
        logger = stack.enter_context(RunLogger(path, run_config, seed=seed, repo_dir=source))
        if echo is not None:
            echo(problem.summary())

        def evaluate_now() -> None:
            controller = HierarchicalController(hierarchy, manager.deterministic_action, eval_command.as_worker())
            result = evaluate(controller, env_config, starts, seed=config.evaluation.seed)
            metrics: dict[str, Any] = result.metrics()
            stats = eval_command.pop_stats()
            lengths = [episode.length for episode in result.episodes]
            if len(stats["outcome"]) != max(lengths):  # one Worker call per step until the last episode ends
                raise RuntimeError(f"{len(stats['outcome'])} Worker steps for episodes of up to {max(lengths)} steps")
            metrics.update(mpc_metrics(stats, "eval_mpc", episode_mask(lengths, len(stats["outcome"]))))
            logger.log(metrics, env.counter)
            with (path / "evaluations.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"env_steps": env.counter.env_steps, **result.details()}) + "\n")
            if echo is not None:
                echo(
                    f"{path.name}: {env.counter.env_steps:>8} samples | success {metrics['eval/success_rate']:4.2f}"
                    f" | unshaped return {metrics['eval/unshaped_return']:8.1f}"
                    f" | gap {_format_gap(metrics['eval/arrival_gap_mean'])}"
                    f" | contact {metrics['eval/contact_fraction']:4.2f}"
                    f" | solve {metrics.get('eval_mpc/solve_ms_mean', float('nan')):6.1f} ms"
                )

        evaluate_now()
        state = start_segments(env, hierarchy, manager.act, manager.value)
        started = time.perf_counter()
        training_time = 0.0
        for k in range(num_updates):
            tick = time.perf_counter()
            if m_cfg.update.anneal_learning_rate:
                manager.set_learning_rate_fraction(1.0 - k / num_updates)
            actor_rate, critic_rate = manager.learning_rates
            rollout, state = collect_hierarchical_rollout(
                env, state, config.rollout.num_steps, hierarchy, manager.act, manager.value, command.as_worker()
            )
            batch = _manager_batch(rollout, m_cfg.update.gae_lambda, manager.device)
            metrics: dict[str, Any] = _renamed(manager.update(batch), "manager")
            training_time += time.perf_counter() - tick
            metrics.update(episode_metrics(rollout.episodes))
            metrics.update(_hierarchy_metrics(rollout))
            metrics.update(mpc_metrics(command.pop_stats(), "mpc"))
            metrics.update(
                {
                    "manager/learning_rate": actor_rate,
                    "manager/critic_learning_rate": critic_rate,
                    "train/updates": k + 1,
                    "time/train_s": training_time,
                    "time/elapsed_s": time.perf_counter() - started,
                    "time/sps": env.counter.env_steps / training_time,
                }
            )
            logger.log(metrics, env.counter)
            if env.counter.env_steps % every == 0:
                evaluate_now()
        torch.save({"manager": manager.state_dict()}, path / "final.pt")
    return path


def load_manager(run_dir: str | Path, checkpoint: str = "final.pt") -> tuple[PPOAgent, Hierarchy, MPCProblem]:
    """The Manager, the hierarchy and the Worker's problem of a run, rebuilt from ``config.yaml`` and a checkpoint.

    The Worker itself is built from the problem by the caller (it needs the solvers).
    """
    path = Path(run_dir)
    raw = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    env_config = env_config_from_dict(raw["env"])
    config = ppo_mpc_config_from_dict(raw["agent"])
    m_cfg = config.manager
    manager = PPOAgent(
        NavigationEnv.observation_size, NavigationEnv.action_size, m_cfg.network, m_cfg.update, config.runtime.device
    )
    state = torch.load(path / checkpoint, map_location=manager.device, weights_only=True)
    manager.load_state_dict(state["manager"])
    return manager, hierarchy_of(env_config, config), problem_of(env_config, config)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: one training run per seed."""
    parser = argparse.ArgumentParser(description="Train PPO_MPC on a layout of the navigation environment.")
    parser.add_argument(
        "--env", type=Path, required=True, help="environment configuration, e.g. configs/env/tunnel.yaml"
    )
    parser.add_argument(
        "--agent", type=Path, required=True, help="PPO_MPC configuration, e.g. configs/agent/ppo_mpc.yaml"
    )
    parser.add_argument(
        "--budget", type=int, required=True, help="samples per run: physical steps summed over the agents (D6)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="training seeds, one run each")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="root of the run directories")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) < len(args.seeds):
        parser.error(f"every seed may appear once, got {args.seeds}")
    env_config = load_env_config(args.env)
    config = load_ppo_mpc_config(args.agent)
    bad = [seed for seed in args.seeds if seed < 0 or seed == config.evaluation.seed]
    if bad:  # checked before the first run, not when its turn comes
        parser.error(f"training seeds must be non-negative and differ from the evaluation seed, got {bad}")
    for seed in args.seeds:
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.runs_dir / args.env.stem / "ppo_mpc" / f"seed{seed}-{stamp}"
        train(
            env_config,
            config,
            budget=args.budget,
            seed=seed,
            run_dir=run_dir,
            provenance={"env_file": str(args.env), "agent_file": str(args.agent)},
        )
        print(f"run written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
