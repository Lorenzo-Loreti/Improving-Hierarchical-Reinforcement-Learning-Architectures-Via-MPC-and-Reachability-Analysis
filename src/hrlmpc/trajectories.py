"""Replays of trained policies (decision log D74): the evaluation episodes of a run recorded step by step, for the
figures of the learned behaviour.

The final policies of a run (its ``final.pt``) are evaluated once more
exactly as the training loop evaluates them — the mode of the Beta (D57),
the grid of starts on the spawn box, the run's evaluation seed, so that the
episodes are those of the last evaluation point of the run — and every
episode is recorded: the positions and velocities at every step, whether the
step made contact and, for the hierarchies, the target in force. A box and a
grid of starts can be given instead, as for the escape evaluation of D70.
Nothing is trained and nothing is scored here: the outcomes (success, length,
contact) are those of :func:`hrlmpc.evaluation.evaluate`, which
:func:`record_episodes` mirrors step by step. The replay goes to
``trajectories_<name>.npz`` in the run directory (:func:`save_replay`), a
compressed NumPy archive that a figure script reads with
:func:`load_replay`; the figures are drawn from these files, never from
numbers typed by hand.

Usage::

    python -m hrlmpc.trajectories runs/final/slalom_sparse/ppo/seed1-... [more runs or directories of runs]
        [--name eval] [--box X_LO X_HI Y_LO Y_HI --grid NX NY] [--action mode] [--seed S] [--checkpoint final.pt]

The MPC architectures need the solvers; PyTorch is needed to load the
policies (imported inside :func:`hrlmpc.reevaluate.load_controller`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray

from hrlmpc.config import Box, EnvConfig
from hrlmpc.env import NavigationEnv
from hrlmpc.evaluation import EVAL_ACTIONS, Controller, evaluation_name, spawn_grid

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

DEFAULT_NAME = "eval"
"""The name of the replay from the run's own evaluation grid."""


@dataclass(frozen=True, eq=False)
class Recording:
    """The episodes of a controller from ``N`` starts, step by step; arrays padded with NaN beyond each episode.

    Attributes:
        starts: ``(N, 2)`` initial positions, at rest.
        positions: ``(N, L + 1, 2)`` positions before every step and after the last, ``L`` the longest episode.
        velocities: ``(N, L + 1, 2)`` velocities, likewise.
        contact: ``(N, L)`` whether each step made contact (D14); ``False`` beyond the episode.
        length: ``(N,)`` lengths of the episodes.
        success: ``(N,)`` whether each episode reached the goal region.
        targets: ``(N, L, 2)`` targets in force at every step for a hierarchy, ``None`` for a flat policy.
    """

    starts: FloatArray
    positions: FloatArray
    velocities: FloatArray
    contact: BoolArray
    length: IntArray
    success: BoolArray
    targets: FloatArray | None


@dataclass(frozen=True, eq=False)
class Replay(Recording):
    """A :class:`Recording` of a run's final policies, with what identifies it.

    Attributes:
        run: The name of the run directory.
        algorithm: The run's algorithm (``ppo``, ``hppo``, ``ppo_mpc``, ``ppo_mpc_reachability``).
        name: The replay's name: :data:`DEFAULT_NAME` from the evaluation grid, the given one from a box.
        action: ``"mean"`` or ``"mode"`` (D43).
        seed: The seed of the evaluation environment.
        checkpoint: The checkpoint file replayed.
        env: The run's environment configuration, as in its ``config.yaml`` (the layout for the figures).
        segment_steps: ``H`` of the hierarchy, 0 for a flat policy.
        target_radius: The length scale of the targets [m] (D24), NaN for a flat policy.
    """

    run: str
    algorithm: str
    name: str
    action: str
    seed: int
    checkpoint: str
    env: dict[str, Any]
    segment_steps: int
    target_radius: float


def _padded(rows: Sequence[Sequence[Any]], width: int, trailing: tuple[int, ...], fill: Any, dtype: Any) -> Any:
    result = np.full((len(rows), width, *trailing), fill, dtype=dtype)
    for i, row in enumerate(rows):
        if row:
            result[i, : len(row)] = np.asarray(row, dtype=dtype)
    return result


def record_episodes(
    controller: Controller, config: EnvConfig, starts: ArrayLike, *, seed: int, inputs: bool | None = None
) -> Recording:
    """Run one episode of ``controller`` from each start, at rest, and record it step by step.

    The loop is that of :func:`hrlmpc.evaluation.evaluate` — one agent per
    start in a fresh environment seeded by ``seed``, the controller reset
    before the first step, every agent stepped until the last first episode
    ends — so the episodes are the evaluation's; only the per-step record is
    added. The states reached are read back from the observations, hence
    exact up to rounding. A controller with a ``targets`` attribute (the
    hierarchies, :class:`hrlmpc.hierarchy.HierarchicalController`) has the
    targets in force recorded at every step.

    Args:
        controller: Deterministic map from observations to commands.
        config: The layout; its curriculum, if any, plays no part (explicit starts).
        starts: ``(N, 2)`` initial positions in the free space, outside the goal region.
        seed: Seed of the evaluation environment.
        inputs: Whether the commands are accelerations; by default the controller's ``inputs`` attribute.

    Raises:
        ValueError: As :func:`hrlmpc.evaluation.evaluate`: a bad shape, an inadmissible start, or ``inputs``
            contradicting the controller.
    """
    points = np.asarray(starts, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError(f"starts must have shape (N, 2) with N >= 1, got {points.shape}")
    declared = getattr(controller, "inputs", None)
    if inputs is None:
        inputs = bool(declared)
    elif declared is not None and bool(declared) != inputs:
        raise ValueError(f"inputs={inputs} contradicts the controller's inputs={declared}")
    plain = dataclasses.replace(config, task=dataclasses.replace(config.task, curriculum=None))
    n = len(points)
    env = NavigationEnv(plain, num_envs=n, seed=seed, autoreset=True)
    obs = env.reset(positions=points)
    reset = getattr(controller, "reset", None)
    if callable(reset):
        reset(n)
    hierarchical = hasattr(controller, "targets")
    positions: list[list[FloatArray]] = [[env.positions[i]] for i in range(n)]
    velocities: list[list[FloatArray]] = [[env.velocities[i]] for i in range(n)]
    contact: list[list[bool]] = [[] for _ in range(n)]
    targets: list[list[FloatArray]] = [[] for _ in range(n)]
    done = np.zeros(n, dtype=bool)
    length = np.zeros(n, dtype=np.int64)
    success = np.zeros(n, dtype=bool)
    while not done.all():
        commands = controller(obs)
        current = np.asarray(controller.targets, dtype=np.float64) if hierarchical else None  # type: ignore[attr-defined]
        out = env.step_input(commands) if inputs else env.step(commands)
        reached_p, reached_v = env.obs_map.positions(out.final_obs), env.obs_map.velocities(out.final_obs)
        for i in np.flatnonzero(~done):
            positions[i].append(reached_p[i])
            velocities[i].append(reached_v[i])
            contact[i].append(bool(out.info.contact[i]))
            if current is not None:
                targets[i].append(current[i])
        for episode in out.episodes:
            if not done[episode.env]:
                done[episode.env] = True
                length[episode.env] = episode.length
                success[episode.env] = episode.success
        obs = out.obs
    longest = int(length.max())
    return Recording(
        starts=points.copy(),
        positions=_padded(positions, longest + 1, (2,), math.nan, np.float64),
        velocities=_padded(velocities, longest + 1, (2,), math.nan, np.float64),
        contact=_padded(contact, longest, (), False, np.bool_),
        length=length,
        success=success,
        targets=_padded(targets, longest, (2,), math.nan, np.float64) if hierarchical else None,
    )


def _config_of(run_dir: Path) -> dict[str, Any]:
    config: dict[str, Any] = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    return config


def replay_run(
    run_dir: str | Path,
    *,
    box: Box | None = None,
    grid: tuple[int, int] | None = None,
    name: str | None = None,
    action: str = "mode",
    seed: int | None = None,
    checkpoint: str = "final.pt",
    worker_factory: Any = None,
    targets_factory: Any = None,
) -> Replay:
    """Record the evaluation episodes of a run's final policies.

    Args:
        run_dir: The run directory (``config.yaml`` and the checkpoint).
        box: The box of the starts; by default the run's spawn box.
        grid: The grid's shape over the box; by default the run's evaluation grid.
        name: The replay's name; :data:`DEFAULT_NAME` by default, which a box must not use.
        action: ``"mean"`` or ``"mode"`` (D43).
        seed: The evaluation's seed; by default the run's evaluation seed.
        checkpoint: The checkpoint file.
        worker_factory: As in :func:`hrlmpc.reevaluate.load_controller`.
        targets_factory: As in :func:`hrlmpc.reevaluate.load_controller`.

    Raises:
        ValueError: If a box is given under the default name, the action is unknown, or a start is not
            admissible.
    """
    from hrlmpc.reevaluate import load_controller

    evaluation_name(action)  # checks the action
    path = Path(run_dir)
    config = _config_of(path)
    if name is None:
        name = DEFAULT_NAME
    if box is not None and name == DEFAULT_NAME:
        raise ValueError(f"a replay from a box needs a name other than {DEFAULT_NAME!r}")
    if seed is None:
        seed = int(config["agent"]["evaluation"]["seed"])
    if grid is None:
        grid = (int(config["agent"]["evaluation"]["grid"][0]), int(config["agent"]["evaluation"]["grid"][1]))
    controller, env_config, close = load_controller(
        path, action, int(grid[0] * grid[1]), checkpoint=checkpoint, worker_factory=worker_factory,
        targets_factory=targets_factory,
    )
    starts = spawn_grid(env_config.task.spawn if box is None else box, grid)
    try:
        recording = record_episodes(controller, env_config, starts, seed=seed)
    finally:
        close()
    hierarchy = getattr(controller, "hierarchy", None)
    return Replay(
        **vars(recording),
        run=path.name,
        algorithm=str(config["algorithm"]),
        name=name,
        action=action,
        seed=seed,
        checkpoint=checkpoint,
        env=json.loads(json.dumps(config["env"])),
        segment_steps=0 if hierarchy is None else int(hierarchy.segment_steps),
        target_radius=math.nan if hierarchy is None else float(hierarchy.targets.radius),
    )


def replay_file(run_dir: str | Path, name: str) -> Path:
    """The file of the replay ``name`` of a run: ``trajectories_<name>.npz`` in the run directory."""
    return Path(run_dir) / f"trajectories_{name}.npz"


_META = ("run", "algorithm", "name", "action", "seed", "checkpoint", "env", "segment_steps", "target_radius")


def save_replay(replay: Replay, run_dir: str | Path) -> Path:
    """Write a replay to :func:`replay_file` as a compressed NumPy archive (the metadata as one JSON string) and
    return the path."""
    path = replay_file(run_dir, replay.name)
    meta = {key: getattr(replay, key) for key in _META}
    meta["target_radius"] = None if math.isnan(replay.target_radius) else replay.target_radius
    arrays: dict[str, Any] = {
        "meta": json.dumps(meta),
        "starts": replay.starts,
        "positions": replay.positions,
        "velocities": replay.velocities,
        "contact": replay.contact,
        "length": replay.length,
        "success": replay.success,
    }
    if replay.targets is not None:
        arrays["targets"] = replay.targets
    np.savez_compressed(path, **arrays)
    return path


def load_replay(path: str | Path) -> Replay:
    """Read a replay written by :func:`save_replay`."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        arrays = {key: data[key] for key in data.files if key != "meta"}
    radius = meta["target_radius"]
    return Replay(
        starts=arrays["starts"],
        positions=arrays["positions"],
        velocities=arrays["velocities"],
        contact=arrays["contact"],
        length=arrays["length"],
        success=arrays["success"],
        targets=arrays.get("targets"),
        run=str(meta["run"]),
        algorithm=str(meta["algorithm"]),
        name=str(meta["name"]),
        action=str(meta["action"]),
        seed=int(meta["seed"]),
        checkpoint=str(meta["checkpoint"]),
        env=meta["env"],
        segment_steps=int(meta["segment_steps"]),
        target_radius=math.nan if radius is None else float(radius),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: replay the final policies of every run below the given paths and write the files."""
    parser = argparse.ArgumentParser(description="Record the evaluation episodes of trained policies (D74).")
    parser.add_argument("paths", nargs="+", help="run directories or directories containing runs")
    parser.add_argument("--name", default=None, help=f"name of the replay ({DEFAULT_NAME!r} from the evaluation grid)")
    parser.add_argument(
        "--box", type=float, nargs=4, default=None, metavar=("X_LO", "X_HI", "Y_LO", "Y_HI"),
        help="box of the starts [m] instead of the spawn box (needs --name)",
    )
    parser.add_argument("--grid", type=int, nargs=2, default=None, metavar=("NX", "NY"),
                        help="grid over the box; by default the run's evaluation grid")
    parser.add_argument("--action", choices=EVAL_ACTIONS, default="mode", help="the deterministic action (D43)")
    parser.add_argument("--seed", type=int, default=None, help="evaluation seed; by default the run's")
    parser.add_argument("--checkpoint", default="final.pt", help="checkpoint file in the run directory")
    args = parser.parse_args(argv)
    box = None if args.box is None else Box(x=(args.box[0], args.box[1]), y=(args.box[2], args.box[3]))
    grid = None if args.grid is None else (int(args.grid[0]), int(args.grid[1]))
    if box is not None and (args.name is None or args.name == DEFAULT_NAME):
        parser.error(f"--box needs a --name other than {DEFAULT_NAME!r}")
    from hrlmpc.analysis import find_runs

    for run in find_runs(args.paths):
        replay = replay_run(run.path, box=box, grid=grid, name=args.name, action=args.action, seed=args.seed,
                            checkpoint=args.checkpoint)
        path = save_replay(replay, run.path)
        print(f"{run.path}: {len(replay.starts)} starts, success {replay.success.mean():.2f}, "
              f"contact {np.mean(replay.contact.any(axis=1)):.2f} -> {path.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
