"""Tests of the minimum-time oracle (decision log D44): the tables, their lookup by content, the replay, the
search over the horizon and the data of the problem. The solves need Gurobi and run in the author's environment
(the last section); everything else runs without a solver."""

from __future__ import annotations

import contextlib
import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from hrlmpc.config import EnvConfig, env_config_to_dict, load_env_config
from hrlmpc.evaluation import arrival_lower_bound, spawn_grid
from hrlmpc.model import system_matrices
from hrlmpc.mpc_problem import FEAS_TOL
from hrlmpc.oracle import (
    OracleEntry,
    OracleProblem,
    OracleSettings,
    OracleTable,
    compute_oracle,
    entry_for,
    find_oracle,
    layout_signature,
    oracle_line,
    oracle_record,
    replay,
    undisturbed,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "env"


def _config(name: str = "slalom") -> EnvConfig:
    return load_env_config(CONFIG_DIR / f"{name}.yaml")


def _thrust_plan(config: EnvConfig, start: tuple[float, float]) -> tuple[int, np.ndarray]:
    """Full thrust along +x until the speed limit, then the thrust that holds it: the bound of F7, which is the
    minimum time in the tunnel, by a coherent plan (the replay needs no correction)."""
    problem = OracleProblem(config)
    steps = problem.lower_bound(np.array(start))
    ph = config.physics
    speed, inputs = 0.0, []
    for _ in range(steps):
        u = min(ph.a_max, (ph.v_max - (1.0 - ph.gamma * ph.dt) * speed) / ph.dt)
        speed = (1.0 - ph.gamma * ph.dt) * speed + ph.dt * u
        inputs.append([u, 0.0])
    return steps, np.array(inputs)


def _positions(config: EnvConfig, start: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    """``(N + 1, 2)`` positions of a plan from ``start`` at rest under the model of the specification."""
    A, B = system_matrices(config.physics.dt, config.physics.gamma)
    x = np.array([start[0], start[1], 0.0, 0.0])
    out = [x[:2].copy()]
    for u in inputs:
        x = A @ x + B @ u
        out.append(x[:2].copy())
    return np.array(out)


def _table(config: EnvConfig, starts: np.ndarray, steps: list[int]) -> OracleTable:
    physics, geometry = layout_signature(config)
    entries = tuple(
        OracleEntry(
            start=(float(p[0]), float(p[1])),
            lower_bound=int(n),
            steps=int(n),
            effort=1.0,
            inputs=((2.5, 0.0),) * int(n),
            length=int(n),
            success=True,
            contact_steps=0,
            corrections=0,
            unshaped_return=1000.0 - 1.01 * n,
        )
        for p, n in zip(starts, steps, strict=True)
    )
    reward = env_config_to_dict(config)["reward"]
    return OracleTable(physics, geometry, dataclasses.asdict(OracleSettings()), entries, source="test", reward=reward)


# ----------------------------------------------------------------------------- tables


def test_layout_signature_ignores_the_disturbance_and_the_reward() -> None:
    """One table serves every reward regime and disturbance level of a layout (D44, D51)."""
    slalom = layout_signature(_config("slalom"))
    assert layout_signature(_config("slalom_d05")) == slalom
    assert layout_signature(_config("slalom_sparse_d10")) == slalom
    assert layout_signature(_config("tunnel")) != slalom
    assert "d_bar" not in slalom[0] and slalom[0]["a_max"] == 2.5


def test_table_round_trips_through_json(tmp_path: Path) -> None:
    config = _config()
    starts = spawn_grid(config.task.spawn, (2, 2))
    table = _table(config, starts, [70, 71, 72, 73])
    table.save(tmp_path / "t.json")
    loaded = OracleTable.load(tmp_path / "t.json")
    assert loaded == table
    np.testing.assert_array_equal(loaded.steps, [70, 71, 72, 73])
    np.testing.assert_array_equal(loaded.starts, starts)
    data = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    assert data["source"] == "test" and data["reward"]["effort_coef"] == 0.01 and data["entries"][0]["corrections"] == 0
    assert OracleTable.from_dict({**data, "reward": {"time_penalty": 2.0}}).matches(config)  # recorded, not matched


def test_malformed_tables_are_rejected() -> None:
    with pytest.raises(ValueError, match="malformed"):
        OracleTable.from_dict({"physics": {}, "geometry": {}, "settings": {}, "entries": [{"start": [0.0]}]})
    with pytest.raises(ValueError, match="malformed"):
        OracleTable.from_dict({"physics": {}})


def test_steps_for_requires_the_tables_starts() -> None:
    config = _config()
    starts = spawn_grid(config.task.spawn, (2, 2))
    table = _table(config, starts, [70, 71, 72, 73])
    np.testing.assert_array_equal(table.steps_for(starts), [70, 71, 72, 73])
    with pytest.raises(ValueError, match="starts"):
        table.steps_for(starts[::-1])
    with pytest.raises(ValueError, match="starts"):
        table.steps_for(starts[:3])


def test_find_oracle_matches_by_content(tmp_path: Path) -> None:
    """The table of a layout is found for every file of that layout, and only for its own starts."""
    config = _config()
    starts = spawn_grid(config.task.spawn, (5, 5))
    (tmp_path / "a_not_a_table.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / "b_broken.json").write_text("{", encoding="utf-8")
    _table(config, starts, [75] * 25).save(tmp_path / "slalom.json")
    found = find_oracle(_config("slalom_sparse_d05"), starts, tmp_path)
    assert found is not None
    path, table = found
    assert path == tmp_path / "slalom.json" and table.steps.tolist() == [75] * 25
    assert find_oracle(_config("tunnel"), starts, tmp_path) is None
    assert find_oracle(config, spawn_grid(config.task.spawn, (3, 3)), tmp_path) is None
    assert find_oracle(config, starts, tmp_path / "missing") is None
    assert oracle_record(found) == str(path.resolve()) and oracle_record(None) is None  # outside the repository
    assert oracle_line(found) == f"oracle table {path.resolve()} (25 starts)" and "NaN" in oracle_line(None)


# ------------------------------------------------------------------------------ replay


def test_thrust_on_the_tunnel_is_a_valid_entry_at_the_bound_of_f7() -> None:
    """F7: in the tunnel, from rest, thrusting along +x arrives at the bound; the replay measures the entry."""
    config = _config("tunnel")
    steps, inputs = _thrust_plan(config, (1.0, 0.0))
    assert steps == 78
    problem = OracleProblem(config)
    entry = entry_for(problem, np.array([1.0, 0.0]), steps, inputs)
    assert entry.valid and entry.lower_bound == 78 and entry.length == 78 and entry.corrections == 0
    assert entry.effort == pytest.approx(float(np.sum(inputs**2))) and entry.effort < 78 * 2.5**2
    assert entry.unshaped_return == pytest.approx(1000.0 - 78 - 0.01 * entry.effort / 2.5**2)  # bonus, time, effort


def test_a_plan_the_environment_corrects_is_not_an_entry() -> None:
    """Full thrust throughout arrives at the bound too, but only because the speed limiter clips it (R4): the oracle's
    plan must be coherent, so that the environment's trajectory is the model's."""
    config = _config("tunnel")
    problem = OracleProblem(config)
    full = np.tile([config.physics.a_max, 0.0], (78, 1))
    stats = replay(config, [1.0, 0.0], full)
    assert stats is not None and stats.success and stats.length == 78 and stats.limiter_steps > 0
    with pytest.raises(RuntimeError, match=f"corrected steps {stats.limiter_steps}"):
        entry_for(problem, np.array([1.0, 0.0]), 78, full)


def test_the_effort_term_must_be_worth_less_than_a_step() -> None:
    """The least-effort plan of the minimum time maximizes the unshaped return only if effort_coef * N < time_penalty
    (the module docstring)."""
    config = _config("tunnel")
    steps, inputs = _thrust_plan(config, (1.0, 0.0))
    heavy = dataclasses.replace(config, reward=dataclasses.replace(config.reward, effort_coef=1.0 / steps))
    with pytest.raises(RuntimeError, match="effort_coef"):
        entry_for(OracleProblem(heavy), np.array([1.0, 0.0]), steps, inputs)
    light = dataclasses.replace(config, reward=dataclasses.replace(config.reward, effort_coef=0.99 / steps))
    assert entry_for(OracleProblem(light), np.array([1.0, 0.0]), steps, inputs).valid


def test_replay_ignores_the_disturbance_of_the_configuration() -> None:
    """The oracle is undisturbed: the replay of a level file equals that of its layout's file."""
    steps, inputs = _thrust_plan(_config("tunnel"), (0.0, 0.0))
    disturbed, base = replay(_config("tunnel_d10"), [0.0, 0.0], inputs), replay(_config("tunnel"), [0.0, 0.0], inputs)
    assert disturbed is not None and base is not None
    assert (disturbed.length, disturbed.success) == (base.length, base.success) == (steps, True)
    assert undisturbed(_config("tunnel_d10")).physics.d_bar == 0.0


def test_a_plan_that_does_not_arrive_is_not_an_entry() -> None:
    problem = OracleProblem(_config("tunnel"))
    assert replay(problem.config, [1.0, 0.0], np.zeros((10, 2))) is None
    with pytest.raises(RuntimeError, match="replays to length 10, success False"):
        entry_for(problem, np.array([1.0, 0.0]), 10, np.zeros((10, 2)))


def test_a_plan_that_arrives_late_or_touches_a_wall_is_not_an_entry() -> None:
    config = _config("tunnel")
    steps, inputs = _thrust_plan(config, (1.0, 0.0))
    problem = OracleProblem(config)
    with pytest.raises(RuntimeError, match="expected success in 77 steps"):
        entry_for(problem, np.array([1.0, 0.0]), steps - 1, inputs)  # arrives one step after the claimed time
    sideways = np.tile([1.0, 1.0], (200, 1))  # up into the ceiling, then along it
    stats = replay(config, [1.0, 0.0], sideways)
    assert stats is not None and stats.contact_steps > 0 and stats.wall_steps > 0
    with pytest.raises(RuntimeError, match=f"contact steps {stats.contact_steps}, corrected steps"):
        entry_for(problem, np.array([1.0, 0.0]), stats.length, sideways)


# ------------------------------------------------------------------------- the search


def test_minimum_time_is_the_first_feasible_horizon_from_the_bound() -> None:
    problem = OracleProblem(_config())
    start = np.array([2.0, -1.0])
    bound = problem.lower_bound(start)
    assert bound == 73  # F7 on the slalom from (2, -1)
    asked: list[int] = []

    def solve(point: np.ndarray, horizon: int) -> np.ndarray | None:
        asked.append(horizon)
        np.testing.assert_array_equal(point, start)
        return None if horizon < bound + 3 else np.zeros((horizon, 2))

    steps, plan = problem.minimum_time(start, solve)
    assert steps == bound + 3 and plan.shape == (bound + 3, 2)
    assert asked == [bound, bound + 1, bound + 2, bound + 3]


def test_the_search_gives_up_after_the_limit() -> None:
    problem = OracleProblem(_config(), OracleSettings(max_extra_steps=2))
    with pytest.raises(RuntimeError, match="no plan"):
        problem.minimum_time(np.array([2.0, -1.0]), lambda point, horizon: None)
    with pytest.raises(RuntimeError, match="shape"):
        problem.minimum_time(np.array([2.0, -1.0]), lambda point, horizon: np.zeros((horizon + 1, 2)))


def test_starts_are_checked_before_the_search() -> None:
    """A plan may start only within the margins of the arena and the obstacles and short of the goal line."""
    problem = OracleProblem(_config())
    never = []

    def solve(point: np.ndarray, horizon: int) -> np.ndarray | None:
        never.append(horizon)
        return None

    np.testing.assert_array_equal(problem.check_start([2.0, -1.0]), [2.0, -1.0])
    for start, reason in [
        ([-1.0, 0.0], "arena"),  # on the left wall, inside the margin
        ([2.0, 1.9995], "arena"),
        ([4.5, -1.0], "obstacle 0"),  # inside the lower block of gate 1 (D16: its live faces only)
        ([4.5, 0.2505], "obstacle 0"),  # above it, within the margin of its top face
        ([10.0, 0.0], "goal"),
        ([np.nan, 0.0], "finite"),
        ([1.0, 2.0, 3.0], "finite"),
    ]:
        with pytest.raises(ValueError, match=reason):
            problem.minimum_time(np.array(start), solve)
        with pytest.raises(ValueError, match=reason):
            compute_oracle(problem, np.array([[2.0, -1.0], start[:2]]) if len(start) == 2 else [start], solve)
    assert never == []
    assert OracleProblem(_config("tunnel")).check_start([0.0, 0.0]).tolist() == [0.0, 0.0]


def test_compute_oracle_replays_every_start_and_records_the_signature() -> None:
    """With a stand-in solve that returns full thrust, the tunnel's table is the bound of F7 at every start."""
    config = _config("tunnel")
    problem = OracleProblem(config)
    starts = spawn_grid(config.task.spawn, (3, 2))

    def solve(point: np.ndarray, horizon: int) -> np.ndarray | None:
        steps, inputs = _thrust_plan(config, (float(point[0]), float(point[1])))
        return inputs if horizon >= steps else None

    lines: list[str] = []
    table = compute_oracle(problem, starts, solve, source="stand-in", echo=lines.append)
    np.testing.assert_array_equal(table.steps, arrival_lower_bound(problem.layout, starts, problem.params))
    assert all(entry.valid for entry in table.entries) and len(lines) == 6
    assert table.matches(_config("tunnel_sparse")) and not table.matches(_config("slalom"))
    assert table.settings == dataclasses.asdict(OracleSettings()) and table.source == "stand-in"
    assert table.reward == env_config_to_dict(config)["reward"] and table.reward["time_penalty"] == 1.0


# ------------------------------------------------------------------------ the problem


def test_settings_are_validated() -> None:
    with pytest.raises(ValueError):
        OracleSettings(rho=0.0)
    with pytest.raises(ValueError):
        OracleSettings(disk_margin=0.0)
    with pytest.raises(ValueError):
        OracleSettings(mip_gap=1.0)
    with pytest.raises(ValueError):
        OracleSettings(max_extra_steps=-1)
    assert OracleSettings() == OracleSettings(1e-3, 1e-5, 1e-6, 1e-4, 60)


def test_the_disk_margin_must_cover_the_feasibility_tolerance() -> None:
    """The solver may violate ``||u||^2 <= r^2`` by FEAS_TOL; the margin keeps the replay's disks respected."""
    problem = OracleProblem(_config())
    assert problem.a_max == 2.5 - 1e-6 and problem.v_max == 1.2 - 1e-6
    assert (problem.v_max) ** 2 + FEAS_TOL < 1.2**2
    with pytest.raises(ValueError, match="disk margin"):
        OracleProblem(_config(), OracleSettings(disk_margin=FEAS_TOL / 2.5))  # below FEAS_TOL / (2 v_max)
    with pytest.raises(ValueError, match="disk margin"):
        OracleProblem(_config(), OracleSettings(disk_margin=1.2))


def test_the_margin_must_cover_the_face_slack() -> None:
    problem = OracleProblem(_config())
    assert 0.0 < problem.face_slack < problem.settings.rho
    with pytest.raises(ValueError, match="face"):
        OracleProblem(_config(), OracleSettings(rho=problem.face_slack / 2))
    assert OracleProblem(_config("tunnel")).obstacles == []


def test_reach_boxes_contain_every_plan() -> None:
    """Plans of the model with ||u|| <= a_max and the speed clipped to v_max stay inside the boxes while they
    stay inside the tightened arena."""
    config = _config()
    problem = OracleProblem(config)
    params = problem.params
    rng = np.random.default_rng(3)
    start = np.array([1.5, 0.5])
    horizon = 30
    lo, hi = problem.reach_boxes(start, horizon)
    assert lo.shape == hi.shape == (horizon + 1, 2)
    np.testing.assert_array_equal(lo[0], start)
    for _ in range(50):
        p, v = start.copy(), np.zeros(2)
        for k in range(1, horizon + 1):
            angle = rng.uniform(0.0, 2.0 * np.pi)
            u = params.a_max * rng.uniform(0.0, 1.0) ** 0.5 * np.array([np.cos(angle), np.sin(angle)])
            v = (1.0 - params.gamma * params.dt) * v + params.dt * u
            speed = float(np.hypot(*v))
            if speed > params.v_max:
                v *= params.v_max / speed
            p = p + params.dt * v
            if np.any(p < problem.position_low) or np.any(p > problem.position_high):
                break  # the plan left the tightened arena: no longer a plan of the problem
            assert np.all(p >= lo[k] - 1e-9) and np.all(p <= hi[k] + 1e-9)


def test_binaries_of_unreachable_faces_are_fixed() -> None:
    """From the spawn box the gates are out of reach in 10 steps (every binary fixed), not in 40."""
    problem = OracleProblem(_config())
    short = problem.fixed_binaries(np.array([1.0, 0.0]), 10)
    assert all(np.array_equal(lb, ub) and np.all(lb.sum(axis=1) == 1) for lb, ub in short)
    long = problem.fixed_binaries(np.array([1.0, 0.0]), 40)
    free = sum(int(np.sum(lb != ub)) for lb, ub in long)
    assert free > 0
    for lb, ub in long:
        fixed_rows = np.flatnonzero((lb == ub).all(axis=1))
        assert np.all(lb[fixed_rows].sum(axis=1) == 1)  # one face selected on every fixed segment
        assert np.all(lb <= ub)


def test_the_problem_uses_the_model_of_the_specification() -> None:
    problem = OracleProblem(_config("slalom_d10"))
    A, B = system_matrices(0.1, 0.5)
    np.testing.assert_array_equal(problem.A, A)
    np.testing.assert_array_equal(problem.B, B)
    assert problem.params.d_bar == 0.0 and problem.goal_x == 10.0
    assert len(problem.obstacles) == 4 and all(o.num_faces == 3 for o in problem.obstacles)
    np.testing.assert_allclose(problem.position_low, [-1.0 + 1e-3, -2.0 + 1e-3])
    np.testing.assert_allclose(problem.position_high, [11.0 - 1e-3, 2.0 - 1e-3])


# --------------------------------------------------- with the solvers (author's machine)


@contextlib.contextmanager
def _licence() -> Iterator[None]:
    """Skip, rather than fail, when Gurobi's size-limited pip licence refuses the model."""
    gp = pytest.importorskip("gurobipy")
    try:
        yield
    except gp.GurobiError as error:
        if "size-limited" in str(error) or getattr(error, "errno", None) == 10010:
            pytest.skip(f"Gurobi's size-limited licence: {error}")
        raise


def test_gurobi_statuses_have_names() -> None:
    from hrlmpc.oracle import gurobi_status_name

    assert gurobi_status_name(2) == "OPTIMAL" and gurobi_status_name(12) == "NUMERIC"
    assert gurobi_status_name(None) == "unknown" and gurobi_status_name(99) == "unknown"


@pytest.mark.parametrize("start", [(0.0, 0.0), (1.0, 1.0), (2.0, -1.0)])
def test_the_oracle_attains_the_bound_of_f7_on_the_tunnel(start: tuple[float, float]) -> None:
    """Without obstacles every binary is fixed, Clarabel solves the convex problem, the minimum time is the bound
    of F7 (from (2, -1) with 3 cm to spare, a horizon the plan can only just meet) and the plan replays without
    any correction."""
    pytest.importorskip("cvxpy")
    pytest.importorskip("gurobipy")
    from hrlmpc.oracle import OracleSolver

    config = _config("tunnel")
    problem = OracleProblem(config)
    with _licence(), OracleSolver(problem) as solver:
        steps, inputs = problem.minimum_time(np.array(start), solver.solve)
        assert solver.convex_solves == 1 and solver.mixed_solves == 0
    assert steps == problem.lower_bound(np.array(start))
    entry = entry_for(problem, np.array(start), steps, inputs)
    stats = replay(config, start, inputs)
    assert entry.valid and entry.corrections == 0 and stats is not None
    assert stats.limiter_steps == 0 and stats.saturation_steps == 0 and stats.wall_steps == 0
    assert np.all(np.hypot(inputs[:, 0], inputs[:, 1]) <= config.physics.a_max)  # strictly within the disk


def test_the_oracle_on_the_slalom_threads_the_gates_in_the_minimum_time() -> None:
    """From (2, -1) the slalom's bound of F7 is 73 steps, the shortest free path (8.42 m, round the corners of
    the gates) at full speed; the plan of the oracle attains it or needs a few steps more (turning at 1.2 m/s with
    2.5 m/s^2 lengthens the path by about a centimetre per corner), threads both gates and replays contact-free."""
    pytest.importorskip("cvxpy")
    pytest.importorskip("gurobipy")
    from hrlmpc.oracle import OracleSolver

    config = _config("slalom")
    problem = OracleProblem(config)
    start = np.array([2.0, -1.0])
    assert problem.lower_bound(start) == 73
    with _licence(), OracleSolver(problem) as solver:
        steps, inputs = problem.minimum_time(start, solver.solve)
        assert solver.mixed_solves >= 1  # the gates are within reach: free binaries, Gurobi
    assert 73 <= steps <= 80
    entry = entry_for(problem, start, steps, inputs)
    stats = replay(config, start, inputs)
    assert entry.valid and entry.corrections == 0 and stats is not None
    assert stats.wall_steps == 0 and stats.limiter_steps == 0 and stats.saturation_steps == 0
    assert entry.unshaped_return == pytest.approx(1000.0 - steps - 0.01 * entry.effort / 2.5**2, abs=1e-6)
    positions = _positions(config, start, inputs)
    in_gate_1 = (positions[:, 0] >= 4.0) & (positions[:, 0] <= 5.0)
    in_gate_2 = (positions[:, 0] >= 7.0) & (positions[:, 0] <= 8.0)
    assert in_gate_1.any() and in_gate_2.any()
    assert np.all(positions[in_gate_1, 1] >= 0.25) and np.all(positions[in_gate_2, 1] <= -0.25)
    assert positions[-1, 0] >= 10.0
