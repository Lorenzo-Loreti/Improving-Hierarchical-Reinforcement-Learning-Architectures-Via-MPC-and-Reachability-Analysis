"""Re-evaluation of trained policies from explicit starts (decision log D70): the escape from the trap.

The final policies of a run (its ``final.pt``) are evaluated once more, with
the mode of the Beta (D57) by default, from the grid of starts over a box
given on the command line instead of the spawn box: for the trap layout of
D69, a grid inside the cavity of the U, where no training start was ever
drawn in the main setting. Nothing is trained; the results go to one line of
``reevaluations.jsonl`` in the run directory, with a name, and the summary
groups the runs as :mod:`hrlmpc.analysis` does.

Usage::

    python -m hrlmpc.reevaluate runs/trap_sparse runs/trap_sparse_nocur \\
        --name escape --box 4.0 5.0 -0.4 0.4 --grid 3 3 [--action mode] [--seed 12345]

The MPC architectures need the solvers (the Worker is rebuilt for the
starts; the targets of architecture 4 too). PyTorch is needed to load the
policies: the modules that need it are imported inside the functions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hrlmpc.analysis import group_runs, load_run
from hrlmpc.config import Box, EnvConfig, env_config_from_dict
from hrlmpc.env import ObservationMap
from hrlmpc.evaluation import EVAL_ACTIONS, Controller, evaluate, evaluation_name, spawn_grid

REEVALUATIONS = "reevaluations.jsonl"
"""The file of a run directory that collects its re-evaluations, one JSON line each."""

Closer = Callable[[], None]


def _config_of(run_dir: Path) -> dict[str, Any]:
    config: dict[str, Any] = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    return config


def load_controller(
    run_dir: str | Path,
    action: str,
    num_agents: int,
    *,
    checkpoint: str = "final.pt",
    worker_factory: Any = None,
    targets_factory: Any = None,
) -> tuple[Controller, EnvConfig, Closer]:
    """The deterministic controller of a run's final policies, its environment configuration and a function that
    releases the solvers behind it (a no-op for PPO and hPPO).

    Args:
        run_dir: The run directory (``config.yaml`` and the checkpoint).
        action: ``"mean"`` or ``"mode"`` (D43).
        num_agents: How many starts the controller will be evaluated from (the slots of an MPC Worker).
        checkpoint: The checkpoint file in the run directory.
        worker_factory: For the MPC architectures, builds the Worker from the problem and the number of slots;
            the tube MPC with the solvers by default.
        targets_factory: For architecture 4, builds the reachable targets; the solvers' by default.

    Raises:
        ValueError: If the run's algorithm is unknown or the action is not one of :data:`EVAL_ACTIONS`.
    """
    evaluation_name(action)  # checks the action
    path = Path(run_dir)
    config = _config_of(path)
    env_config = env_config_from_dict(config["env"])
    algorithm = config.get("algorithm")
    if algorithm == "ppo":
        from hrlmpc.train_ppo import load_agent

        agent = load_agent(path, checkpoint)
        return agent.deterministic_policy(action), env_config, lambda: None
    if algorithm == "hppo":
        from hrlmpc.train_hppo import deterministic_controller, load_hierarchy

        manager, learned_worker, hierarchy = load_hierarchy(path, checkpoint)
        return deterministic_controller(manager, learned_worker, hierarchy, action), env_config, lambda: None
    if algorithm in ("ppo_mpc", "ppo_mpc_reachability"):
        from hrlmpc.hierarchy import HierarchicalController
        from hrlmpc.mpc_command import MPCCommand, StepActor
        from hrlmpc.train_ppo_mpc import load_manager, solver_worker

        mpc_manager, mpc_hierarchy, problem = load_manager(path, checkpoint, targets_factory=targets_factory)
        factory = solver_worker if worker_factory is None else worker_factory
        actor: StepActor = factory(problem, num_agents)
        command = MPCCommand(actor, ObservationMap.from_config(env_config))
        policy = mpc_manager.deterministic_policy(action)
        controller = HierarchicalController(mpc_hierarchy, policy, command.as_worker())

        def close() -> None:
            for thing in (actor, mpc_hierarchy.targets):
                release = getattr(thing, "close", None)
                if callable(release):
                    release()

        return controller, env_config, close
    raise ValueError(f"unknown algorithm {algorithm!r} in {path / 'config.yaml'}")


def reevaluate_run(
    run_dir: str | Path,
    box: Box,
    grid: tuple[int, int],
    *,
    name: str,
    action: str = "mode",
    seed: int | None = None,
    checkpoint: str = "final.pt",
    worker_factory: Any = None,
    targets_factory: Any = None,
) -> dict[str, Any]:
    """Evaluate the final policies of a run from the grid of starts over ``box`` and append the results, as one
    JSON line, to the run's :data:`REEVALUATIONS`.

    Args:
        run_dir: The run directory.
        box: The box of the starts, in the free space and short of the goal line.
        grid: The grid's shape over the box, as :func:`~hrlmpc.evaluation.spawn_grid`.
        name: The name of the re-evaluation (``"escape"`` for the starts inside the trap).
        action: ``"mean"`` or ``"mode"``.
        seed: The evaluation's seed; by default the run's evaluation seed.
        checkpoint: The checkpoint file.
        worker_factory: As in :func:`load_controller`.
        targets_factory: As in :func:`load_controller`.

    Returns:
        The line written: the name, the action, the seed, the box, the grid, the metrics of the evaluation (under
        ``eval/``, whatever the action) and the per-start details of
        :meth:`~hrlmpc.evaluation.Evaluation.details`.

    Raises:
        ValueError: If a start is not admissible (:func:`~hrlmpc.evaluation.evaluate`).
    """
    path = Path(run_dir)
    if seed is None:
        seed = int(_config_of(path)["agent"]["evaluation"]["seed"])
    starts = spawn_grid(box, grid)
    controller, env_config, close = load_controller(
        path, action, len(starts), checkpoint=checkpoint, worker_factory=worker_factory,
        targets_factory=targets_factory,
    )
    try:
        result = evaluate(controller, env_config, starts, seed=seed)
    finally:
        close()
    metrics = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in result.metrics("eval/").items()}
    line: dict[str, Any] = {
        "name": name,
        "action": action,
        "seed": seed,
        "checkpoint": checkpoint,
        "box": {"x": list(box.x), "y": list(box.y)},
        "grid": list(grid),
        "metrics": metrics,  # NaN written as null, as the run logger does
        **result.details(),
    }
    with (path / REEVALUATIONS).open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    return line


def read_reevaluations(run_dir: str | Path, name: str) -> dict[str, Any] | None:
    """The last re-evaluation called ``name`` of a run, or ``None``."""
    path = Path(run_dir) / REEVALUATIONS
    if not path.exists():
        return None
    found = None
    for text in path.read_text(encoding="utf-8").splitlines():
        if text.strip():
            line = json.loads(text)
            if line.get("name") == name:
                found = line
    return found


_COLUMNS = ("eval/success_rate", "eval/length", "eval/contact_fraction", "eval/arrival_gap_mean",
            "eval/unshaped_return")


def summarize_reevaluations(run_dirs: Sequence[str | Path], name: str) -> str:
    """The re-evaluation ``name`` of the runs, grouped as in :mod:`hrlmpc.analysis`: per run and as means over the
    group; a run without it is listed as missing."""
    runs = [load_run(p) for p in run_dirs]
    lines: list[str] = [f"re-evaluation {name!r} (D70): the final policies from the given starts"]
    header = (
        f"{'run':<28} {'success':>8} {'length':>8} {'contact':>8} {'gap':>8} {'return':>9}"
    )
    for label, members in group_runs(runs).items():
        lines += ["", f"== {label} ({len(members)} runs)", header, "-" * len(header)]
        rows: list[list[float]] = []
        for run in members:
            line = read_reevaluations(run.path, name)
            if line is None:
                lines.append(f"{run.name:<28} {'missing':>8}")
                continue
            m = line["metrics"]
            values = [math.nan if m.get(key) is None else float(m[key]) for key in _COLUMNS]
            rows.append(values)
            lines.append(
                f"{run.name:<28} {values[0]:>8.2f} {values[1]:>8.1f} {values[2]:>8.2f} "
                f"{values[3]:>8.3f} {values[4]:>9.1f}"
            )
        if rows:
            table = np.array(rows)
            means = np.array([np.mean(col[~np.isnan(col)]) if (~np.isnan(col)).any() else math.nan for col in table.T])
            lines.append(
                f"{'mean over ' + str(len(rows)) + ' runs':<28} {means[0]:>8.2f} {means[1]:>8.1f} "
                f"{means[2]:>8.2f} {means[3]:>8.3f} {means[4]:>9.1f}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: re-evaluate every run below the given directories from a grid of starts and print the summary."""
    parser = argparse.ArgumentParser(description="Re-evaluate trained policies from a grid of starts (D70).")
    parser.add_argument("paths", nargs="+", help="run directories or directories containing runs")
    parser.add_argument("--name", required=True, help="name of the re-evaluation, e.g. escape")
    parser.add_argument(
        "--box", type=float, nargs=4, required=True, metavar=("X_LO", "X_HI", "Y_LO", "Y_HI"),
        help="box of the starts [m]",
    )
    parser.add_argument("--grid", type=int, nargs=2, default=(3, 3), metavar=("NX", "NY"), help="grid over the box")
    parser.add_argument("--action", choices=EVAL_ACTIONS, default="mode", help="the deterministic action (D43)")
    parser.add_argument("--seed", type=int, default=None, help="evaluation seed; by default the run's")
    parser.add_argument("--checkpoint", default="final.pt", help="checkpoint file in the run directory")
    args = parser.parse_args(argv)
    box = Box(x=(args.box[0], args.box[1]), y=(args.box[2], args.box[3]))
    grid = (int(args.grid[0]), int(args.grid[1]))
    from hrlmpc.analysis import find_runs

    runs = find_runs(args.paths)
    for run in runs:
        line = reevaluate_run(run.path, box, grid, name=args.name, action=args.action, seed=args.seed,
                              checkpoint=args.checkpoint)
        m = line["metrics"]
        print(f"{run.path}: success {m['eval/success_rate']:.2f}, length {m['eval/length']:.1f}")
    print()
    print(summarize_reevaluations([run.path for run in runs], args.name))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
