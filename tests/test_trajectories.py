"""Tests of the replays of trained policies (decision log D74): the evaluation episodes of a run recorded step by
step, for the figures of the learned behaviour. The recording loop is tested with hand-made controllers; the
run-level functions need PyTorch and, for the MPC architecture, use a stand-in Worker as ``tests/test_reevaluate.py``
does."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from hrlmpc.config import Box, EnvConfig, env_config_from_dict
from hrlmpc.env import ObservationMap
from hrlmpc.evaluation import evaluate, spawn_grid
from hrlmpc.geometry import Layout
from hrlmpc.hierarchy import CommandWorker, HierarchicalController, disk_hierarchy
from hrlmpc.trajectories import (
    DEFAULT_NAME,
    Recording,
    Replay,
    load_replay,
    main,
    record_episodes,
    replay_file,
    replay_run,
    save_replay,
)

REPO = Path(__file__).resolve().parents[1]
H = 5


def _env(horizon: int = 200, **physics: float) -> EnvConfig:
    data = copy.deepcopy(yaml.safe_load((REPO / "configs" / "env" / "tunnel.yaml").read_text(encoding="utf-8")))
    data["task"]["horizon"] = horizon
    data["physics"].update(physics)
    return env_config_from_dict(data)


def _disk(u: np.ndarray, radius: float) -> np.ndarray:
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    return np.where(norms > radius, u * (radius / np.maximum(norms, 1e-12)), u)


class _FlatTracker:
    """A flat controller of accelerations: ``u = 2 (g - p) - 2 v`` towards a fixed point ``g``, on the disk."""

    inputs = True

    def __init__(self, config: EnvConfig, goal: tuple[float, float]) -> None:
        self.obs_map = ObservationMap.from_config(config)
        self.goal = np.array(goal)
        self.a_max = config.physics.a_max

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        p, v = self.obs_map.positions(obs), self.obs_map.velocities(obs)
        return _disk(2.0 * (self.goal - p) - 2.0 * v, self.a_max)


def _hierarchical(config: EnvConfig) -> HierarchicalController:
    """A hierarchy whose Manager always aims at the farthest point ahead and whose Worker tracks the target."""
    hierarchy = disk_hierarchy(config, segment_steps=H, reach_factor=2.5, discount=0.99, progress_coef=1.0)
    a_max = config.physics.a_max

    def command(obs: np.ndarray, p: np.ndarray, g: np.ndarray, left: np.ndarray) -> np.ndarray:
        return _disk(4.0 * (g - p) - 2.0 * hierarchy.obs_map.velocities(obs), a_max)

    return HierarchicalController(hierarchy, lambda o: np.tile([1.0, 0.0], (len(o), 1)), CommandWorker(command))


def _check_geometry(recording: Recording, env: EnvConfig) -> None:
    """The recorded positions are a trajectory of the environment: in the free space, one step apart by at most
    ``T_s v_max``, ending in the goal region exactly when the episode succeeded; NaN beyond the episode's length."""
    layout = Layout.from_config(env.geometry)
    params = env.physics
    n = len(recording.length)
    longest = int(recording.length.max())
    assert recording.positions.shape == (n, longest + 1, 2) and recording.velocities.shape == (n, longest + 1, 2)
    assert recording.contact.shape == (n, longest) and recording.contact.dtype == np.bool_
    assert recording.length.dtype == np.int64 and recording.success.dtype == np.bool_
    for i, length in enumerate(recording.length):
        p = recording.positions[i, : length + 1]
        assert np.isfinite(p).all() and np.isnan(recording.positions[i, length + 1 :]).all()
        assert layout.in_free_space(p).all()
        assert np.all(np.linalg.norm(np.diff(p, axis=0), axis=1) <= params.dt * params.v_max + 1e-9)
        assert bool(layout.in_goal(p[-1:])[0]) == bool(recording.success[i])
        assert not layout.in_goal(p[:-1]).any()
        v = recording.velocities[i, : length + 1]
        assert np.isfinite(v).all() and np.isnan(recording.velocities[i, length + 1 :]).all()
        assert np.all(np.linalg.norm(v, axis=1) <= params.v_max + 1e-9)
        assert not recording.contact[i, length:].any()


# --------------------------------------------------------------------------
# The recording loop (no PyTorch)
# --------------------------------------------------------------------------


def test_the_recording_holds_the_episodes_of_the_evaluation_step_by_step() -> None:
    """D74: one episode per start, at rest, the same as :func:`hrlmpc.evaluation.evaluate` runs them; the positions
    before every step and after the last, the velocities, the contact flags; NaN beyond each episode."""
    env = _env(d_bar=1.0)
    starts = spawn_grid(env.task.spawn, (3, 2))
    controller = _FlatTracker(env, goal=(10.5, 0.0))
    recording = record_episodes(controller, env, starts, seed=3)
    np.testing.assert_allclose(recording.starts, starts)
    np.testing.assert_allclose(recording.positions[:, 0], starts)
    assert np.all(recording.velocities[:, 0] == 0.0) and recording.targets is None
    _check_geometry(recording, env)
    assert recording.success.all() and len(set(recording.length.tolist())) > 1  # episodes of different lengths
    direct = evaluate(controller, env, starts, seed=3)
    assert recording.length.tolist() == [episode.length for episode in direct.episodes]
    assert recording.success.tolist() == [episode.success for episode in direct.episodes]
    assert recording.contact.sum(axis=1).tolist() == [episode.contact_steps for episode in direct.episodes]
    # the disturbances are those of the seed: another seed gives other trajectories, the same seed the same
    again = record_episodes(controller, env, starts, seed=3)
    np.testing.assert_array_equal(again.positions, recording.positions)
    other = record_episodes(controller, env, starts, seed=4)
    assert not np.array_equal(other.positions[:, 1:], recording.positions[:, 1:], equal_nan=True)


def test_a_truncated_episode_and_a_contact_are_recorded() -> None:
    """An episode that hits the horizon has that length and no success; a wall contact shows in ``contact``."""
    env = _env(horizon=15)
    starts = np.array([[0.5, 1.5], [1.0, 0.0]])
    controller = _FlatTracker(env, goal=(0.5, 3.0))  # aims beyond the upper wall: contact, no success
    recording = record_episodes(controller, env, starts, seed=0)
    _check_geometry(recording, env)
    assert recording.length.tolist() == [15, 15] and not recording.success.any()
    assert recording.contact[0].any()
    assert recording.positions[0, -1, 1] == pytest.approx(2.0, abs=1e-9)  # on the wall at the end


def test_the_targets_of_a_hierarchy_are_recorded_per_step() -> None:
    """The targets in force at every step of a hierarchy: constant over a segment of ``H`` steps, NaN beyond the
    episode; the trajectory of the first ``H`` steps follows the first target."""
    env = _env()
    starts = spawn_grid(env.task.spawn, (2, 2))
    controller = _hierarchical(env)
    recording = record_episodes(controller, env, starts, seed=7)
    _check_geometry(recording, env)
    assert recording.success.all()
    assert recording.targets is not None and recording.targets.shape == recording.contact.shape + (2,)
    radius = controller.hierarchy.targets.radius
    for i, length in enumerate(recording.length):
        t = recording.targets[i, :length]
        assert np.isfinite(t).all() and np.isnan(recording.targets[i, length:]).all()
        for start in range(0, int(length), H):
            np.testing.assert_allclose(t[start : start + H], np.broadcast_to(t[start], t[start : start + H].shape))
        np.testing.assert_allclose(t[0], starts[i] + [radius, 0.0])  # the Manager's action (1, 0) at the start
        # a decision after H steps aims from where the agent is then
        if length > H:
            np.testing.assert_allclose(t[H], recording.positions[i, H] + [radius, 0.0])


def test_the_starts_must_be_admissible_and_inputs_consistent() -> None:
    env = _env()
    controller = _FlatTracker(env, goal=(10.5, 0.0))
    with pytest.raises(ValueError):
        record_episodes(controller, env, np.array([[10.5, 0.0]]), seed=0)  # in the goal region
    with pytest.raises(ValueError, match="contradicts"):
        record_episodes(controller, env, np.array([[1.0, 0.0]]), seed=0, inputs=False)
    with pytest.raises(ValueError):
        record_episodes(controller, env, np.zeros((0, 2)), seed=0)


def test_a_replay_is_saved_and_loaded_without_loss(tmp_path: Path) -> None:
    env = _env()
    starts = spawn_grid(env.task.spawn, (2, 1))
    for controller, algorithm, steps, radius in (
        (_FlatTracker(env, goal=(10.5, 0.0)), "ppo", 0, float("nan")),
        (_hierarchical(env), "hppo", H, 2.5 * 0.6),
    ):
        recording = record_episodes(controller, env, starts, seed=1)
        replay = Replay(**vars(recording), run="seed1-x", algorithm=algorithm, name=DEFAULT_NAME, action="mode",
                        seed=1, checkpoint="final.pt", env=json.loads(json.dumps(yaml.safe_load(
                            (REPO / "configs" / "env" / "tunnel.yaml").read_text(encoding="utf-8")))),
                        segment_steps=steps, target_radius=radius)
        path = save_replay(replay, tmp_path)
        assert path == replay_file(tmp_path, DEFAULT_NAME) and path.name == "trajectories_eval.npz"
        loaded = load_replay(path)
        for key in ("run", "algorithm", "name", "action", "seed", "checkpoint", "segment_steps", "env"):
            assert getattr(loaded, key) == getattr(replay, key), key
        assert loaded.target_radius == radius or (np.isnan(loaded.target_radius) and np.isnan(radius))
        for key in ("starts", "positions", "velocities", "contact", "length", "success"):
            np.testing.assert_array_equal(getattr(loaded, key), getattr(replay, key), err_msg=key)
        if replay.targets is None:
            assert loaded.targets is None
        else:
            np.testing.assert_array_equal(loaded.targets, replay.targets)


# --------------------------------------------------------------------------
# The replays of runs (PyTorch)
# --------------------------------------------------------------------------


def _agent(name: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((REPO / "configs" / "agent" / f"{name}.yaml").read_text(encoding="utf-8"))
    data["evaluation"].update(every=40, grid=[2, 2], seed=99)
    return data


class _PDWorker:
    """A stand-in MPC Worker (as in ``tests/test_reevaluate.py``): ``u = 2 (g - p) - 2 v`` on the disk."""

    def __init__(self, problem: Any, num_slots: int) -> None:
        self.problem = problem
        self.num_slots = num_slots

    def act(self, x: Any, target: Any, steps_left: int, slot: int = 0) -> tuple[np.ndarray, Any]:
        from hrlmpc.mpc_command import SOLVER, WorkerStep
        from hrlmpc.mpc_problem import project_disk

        self.problem.waypoint_stage(steps_left)
        state, goal = np.asarray(x, dtype=np.float64), np.asarray(target, dtype=np.float64)
        u = project_disk(2.0 * (goal - state[:2]) - 2.0 * state[2:], self.problem.params.a_max)
        info = WorkerStep(outcome=SOLVER, solve_ms=0.5, free_binaries=0, tube_ratio=float("nan"), cost=1.0,
                          cut_rounds=1, plan=None, invalid_candidate=0, solver_errors=0)
        return u, info

    def reset(self, slot: int | None = None) -> None:
        pass

    def close(self) -> None:
        pass


def _runs(tmp_path: Path) -> dict[str, Path]:
    """One tiny run of PPO, hPPO and PPO_MPC each, under ``tmp_path/<algorithm>``."""
    pytest.importorskip("torch")
    from hrlmpc.hppo_config import hppo_config_from_dict
    from hrlmpc.ppo_config import ppo_config_from_dict
    from hrlmpc.ppo_mpc_config import ppo_mpc_config_from_dict
    from hrlmpc.train_hppo import train as train_hppo
    from hrlmpc.train_ppo import train as train_ppo
    from hrlmpc.train_ppo_mpc import train as train_ppo_mpc

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
    env = _env(horizon=12)
    provenance = {"env_file": "configs/env/tunnel.yaml"}
    return {
        "ppo": train_ppo(env, ppo_config_from_dict(ppo), budget=40, seed=1, run_dir=tmp_path / "ppo", echo=None,
                         provenance=provenance),
        "hppo": train_hppo(env, hppo_config_from_dict(hppo), budget=40, seed=1, run_dir=tmp_path / "hppo",
                           echo=None, provenance=provenance),
        "ppo_mpc": train_ppo_mpc(env, ppo_mpc_config_from_dict(mpc), budget=40, seed=1,
                                 run_dir=tmp_path / "ppo_mpc", worker_factory=_PDWorker, echo=None,
                                 provenance=provenance),
    }


def test_the_replay_of_a_run_is_its_evaluation_recorded(tmp_path: Path) -> None:
    """D74: the final policy of a run, with the mode (D57), from the run's evaluation grid and with its evaluation
    seed; the outcomes are those of the evaluation of the loaded controller."""
    from hrlmpc.reevaluate import load_controller

    runs = _runs(tmp_path)
    env = _env(horizon=12)
    starts = spawn_grid(env.task.spawn, (2, 2))
    for name, run_dir in runs.items():
        replay = replay_run(run_dir, worker_factory=_PDWorker)
        assert replay.algorithm == name and replay.action == "mode" and replay.seed == 99
        assert replay.name == DEFAULT_NAME and replay.run == run_dir.name and replay.checkpoint == "final.pt"
        np.testing.assert_allclose(replay.starts, starts)
        _check_geometry(replay, env)
        assert replay.env == json.loads(json.dumps(yaml.safe_load((run_dir / "config.yaml").read_text())["env"]))
        controller, env_config, close = load_controller(run_dir, "mode", len(starts), worker_factory=_PDWorker)
        try:
            direct = evaluate(controller, env_config, starts, seed=99)
        finally:
            close()
        assert replay.length.tolist() == [episode.length for episode in direct.episodes], name
        assert replay.success.tolist() == [episode.success for episode in direct.episodes], name
        assert replay.contact.sum(axis=1).tolist() == [episode.contact_steps for episode in direct.episodes], name
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        if name == "ppo":
            assert replay.targets is None and replay.segment_steps == 0 and np.isnan(replay.target_radius)
        else:
            h = int(config["agent"]["hierarchy"]["segment_steps"])
            assert replay.segment_steps == h and replay.target_radius > 0.0
            assert replay.targets is not None and replay.targets.shape == replay.contact.shape + (2,)
            for i, length in enumerate(replay.length):
                for start in range(0, int(length), h):
                    segment = replay.targets[i, start : start + h]
                    np.testing.assert_allclose(segment, np.broadcast_to(segment[0], segment.shape))


def test_a_replay_from_a_box_takes_its_name_and_grid(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    box = Box(x=(3.0, 4.0), y=(-0.5, 0.5))
    replay = replay_run(runs["ppo"], box=box, grid=(3, 1), name="escape", action="mean", seed=7)
    assert replay.name == "escape" and replay.action == "mean" and replay.seed == 7
    np.testing.assert_allclose(replay.starts, spawn_grid(box, (3, 1)))
    with pytest.raises(ValueError, match="name"):
        replay_run(runs["ppo"], box=box)  # a box under the default name
    with pytest.raises(ValueError):
        replay_run(runs["ppo"], box=Box(x=(10.5, 10.8), y=(-0.5, 0.5)), grid=(2, 2), name="goal")


def test_the_command_line_replays_every_run_below_the_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = _runs(tmp_path)
    assert main([str(runs["ppo"]), str(runs["hppo"])]) == 0
    for name in ("ppo", "hppo"):
        assert load_replay(replay_file(runs[name], DEFAULT_NAME)).algorithm == name
    assert "trajectories_eval.npz" in capsys.readouterr().out
    assert main(["--name", "mouth", "--box", "3.0", "4.0", "-0.5", "0.5", "--grid", "2", "1", str(runs["ppo"])]) == 0
    loaded = load_replay(replay_file(runs["ppo"], "mouth"))
    assert loaded.name == "mouth" and loaded.starts.shape == (2, 2)
    with pytest.raises(SystemExit):
        main(["--box", "3.0", "4.0", "-0.5", "0.5", str(runs["ppo"])])  # a box without a name
