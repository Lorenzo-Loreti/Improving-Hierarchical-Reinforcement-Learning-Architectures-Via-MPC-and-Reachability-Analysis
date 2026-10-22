"""Tests of the results of the experiment matrix (decision log D64, D67, D68, D71–D75): the cells, the statistics
of censored times, the tables, the figures and the command line, on synthetic runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hrlmpc.analysis import Run, load_run
from hrlmpc.config import env_config_to_dict, load_env_config
from hrlmpc.reevaluate import REEVALUATIONS
from hrlmpc.results import (
    ARCHS,
    BUDGET,
    Cell,
    Summary,
    ablation_table,
    cells_of,
    ci,
    classify,
    compute_table,
    disturbance_table,
    escape_rates,
    escape_table,
    fisher,
    k,
    km,
    logrank,
    logrank_table,
    main,
    mean_pm,
    pfmt,
    results_table,
    run_metrics,
    summarize,
    sweep_table,
    worker_table,
    write_figures,
    write_report,
    write_tables,
)
from hrlmpc.utils.runlog import RunLogger, SampleCounter

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "env"
EVERY = 10240


def _env(name: str, **physics: float) -> dict[str, Any]:
    data = env_config_to_dict(load_env_config(CONFIG_DIR / f"{name}.yaml"))
    data["physics"].update(physics)
    return data


def _fake_run(algorithm: str, env_file: str, agent_file: str | None, env: dict[str, Any], seed: int = 1) -> Run:
    config: dict[str, Any] = {"algorithm": algorithm, "seed": seed, "env_file": env_file, "env": env}
    if agent_file is not None:
        config["agent_file"] = agent_file
        config["agent"] = {"hierarchy": {"segment_steps": 10}}
    return Run(Path("none"), config, {}, ())


def _write_run(
    directory: Path,
    *,
    algorithm: str,
    env_name: str,
    seed: int,
    successes: list[float],
    gaps: list[float] | None = None,
    contacts: list[float] | None = None,
    agent_file: str | None = None,
    d_bar: float = 0.0,
    mpc: bool = False,
    reach: bool = False,
    hierarchy: bool = False,
    curriculum_stages: list[int] | None = None,
    elapsed_s: float = 600.0,
) -> Path:
    """A synthetic run: one evaluation per ``successes`` entry, an update between them with the training keys the
    results read; the mode's success, gap and contact as given (the mean's copied)."""
    env = _env(env_name, d_bar=d_bar)
    run_dir = directory / env_name / algorithm / f"seed{seed}"
    config: dict[str, Any] = {
        "algorithm": algorithm, "seed": seed, "budget": BUDGET, "env_file": f"configs/env/{env_name}.yaml",
        "agent_file": agent_file or f"configs/agent/{algorithm}.yaml", "env": env,
        "agent": {"hierarchy": {"segment_steps": 10}} if algorithm != "ppo" else {},
    }
    gaps = gaps or [0.05] * len(successes)
    contacts = contacts or [0.0] * len(successes)
    with RunLogger(run_dir, config, seed=seed) as logger:
        counter = SampleCounter()
        for i, success in enumerate(successes):
            if i > 0:
                counter.add_env_steps(EVERY)
                update: dict[str, Any] = {
                    "train/policy_loss": 0.1, "rollout/episodes": 40.0, "rollout/contact_fraction": contacts[i],
                    "rollout/impulse": 0.5 * contacts[i], "time/elapsed_s": elapsed_s * i / (len(successes) - 1),
                    "time/sps": 1000.0,
                }
                if curriculum_stages is not None:
                    stage = curriculum_stages[min(i, len(curriculum_stages) - 1)]
                    update.update({"curriculum/stage": stage, "curriculum/stages": 9,
                                   "curriculum/final": 1.0 if stage == 8 else 0.0})
                if hierarchy:
                    update.update({"hierarchy/target_distance": 0.6, "hierarchy/segments": 100.0})
                if mpc:
                    update.update({
                        "mpc/steps": 2048.0, "mpc/emergency_count": 0.0, "mpc/solver_error_count": 2.0,
                        "mpc/exit_converged_count": 1500.0, "mpc/exit_fixed_count": 540.0,
                        "mpc/exit_relaxation_count": 2.0, "mpc/exit_best_count": 2.0, "mpc/exit_rounds_count": 2.0,
                        "mpc/exit_empty_count": 1.0, "mpc/exit_unsolved_count": 1.0, "mpc/exit_error_count": 0.0,
                        "mpc/exact_accepted_count": 0.0, "mpc/best_applied_count": 0.0,
                        "mpc/accepted_applied_count": 0.0, "mpc/invalid_candidate_count": 0.0,
                        "mpc/candidate_rate": 0.001, "mpc/kept_rate": 0.001, "mpc/solve_ms_mean": 40.0,
                        "mpc/solve_ms_max": 300.0, "mpc/cut_rounds_mean": 1.5,
                        "hierarchy/reached_rate": 0.5,
                    })
                if reach:
                    update.update({
                        "reach/decisions": 200.0, "reach/hull_fallback_count": 0.0,
                        "reach/projection_failure_count": 0.0, "reach/support_failure_count": 1.0,
                        "reach/projection_error_count": 0.0, "reach/projected_rate": 0.15,
                        "reach/project_ms_mean": 30.0, "reach/support_ms_mean": 50.0, "reach/hull_area_mean": 2.0,
                    })
                logger.log(update, counter)
            record = {}
            for prefix in ("eval", "eval_mode"):
                record.update({
                    f"{prefix}/success_rate": success, f"{prefix}/oracle_gap_max": gaps[i],
                    f"{prefix}/oracle_gap_mean": gaps[i] / 2, f"{prefix}/arrival_gap_max": gaps[i],
                    f"{prefix}/contact_fraction": contacts[i], f"{prefix}/unshaped_return": 800.0 * success - 100.0,
                    f"{prefix}/length": 60.0,
                })
            if mpc:
                record.update({"eval_mode_mpc/solve_ms_mean": 40.0, "eval_mode_hierarchy/reached_rate": 0.5,
                               "eval_mode_hierarchy/end_distance_mean": 0.6})
            elif hierarchy:
                record["eval_mode_hierarchy/end_distance_mean"] = 0.6
            logger.log(record, counter)
    return run_dir


def _matrix(tmp_path: Path) -> Path:
    """A small matrix: the slalom in the main and exploration settings, the tunnel in the dense setting, two seeds
    each; PPO solves late with contact, hPPO never, PPO_MPC early, the reachable one early; a disturbed cell, a sweep
    cell and the map variant on the slalom's main setting."""
    root = tmp_path / "runs"
    curves = {
        "ppo": ([0.0, 0.2, 0.6, 1.0, 1.0, 1.0], [0.3, 0.2, 0.1, 0.1, 0.1, 0.1]),
        "hppo": ([0.0, 0.0, 0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        "ppo_mpc": ([0.0, 0.5, 1.0, 1.0, 1.0, 1.0], [0.0] * 6),
        "ppo_mpc_reachability": ([0.0, 0.8, 1.0, 1.0, 1.0, 1.0], [0.0] * 6),
    }
    for env_name in ("slalom_sparse", "slalom_sparse_nocur", "tunnel"):
        for arch, (successes, contacts) in curves.items():
            for seed in (1, 2):
                shifted = successes if seed == 1 else [0.0] + successes[:-1]
                _write_run(root / "final", algorithm=arch, env_name=env_name, seed=seed, successes=shifted,
                           contacts=contacts, mpc=arch.startswith("ppo_mpc"),
                           reach=arch == "ppo_mpc_reachability", hierarchy=arch == "hppo",
                           curriculum_stages=[0, 2, 4, 6, 8, 8] if env_name == "slalom_sparse" else None)
    for arch, (successes, contacts) in curves.items():
        _write_run(root / "final", algorithm=arch, env_name="slalom_sparse_d10", seed=1, successes=successes,
                   contacts=contacts, d_bar=1.0, mpc=arch.startswith("ppo_mpc"),
                   reach=arch == "ppo_mpc_reachability", hierarchy=arch == "hppo")
    for arch in ("hppo", "ppo_mpc"):
        _write_run(root / "final_h6", algorithm=arch, env_name="slalom_sparse", seed=1, successes=curves[arch][0],
                   agent_file=f"configs/agent/{arch}_h6.yaml", mpc=arch == "ppo_mpc", hierarchy=arch == "hppo")
    _write_run(root / "final_map", algorithm="hppo", env_name="slalom_sparse", seed=1,
               successes=[0.0, 0.5, 1.0, 1.0, 1.0, 1.0], agent_file="configs/agent/hppo_map.yaml", hierarchy=True)
    _write_run(root / "final_map", algorithm="hppo", env_name="slalom_sparse", seed=2,
               successes=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0], agent_file="configs/agent/hppo_map.yaml", hierarchy=True)
    return root


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------


def test_a_run_is_classified_from_its_configuration() -> None:
    """D71, D62, D63, D45, D60, D75: the setting from the reward and the curriculum, the level from the physics,
    the variant from the agent file's name, whichever separator the path uses."""
    slalom, sparse, nocur = _env("slalom"), _env("slalom_sparse"), _env("slalom_sparse_nocur")
    assert classify(_fake_run("ppo", "configs/env/slalom_sparse.yaml", "configs/agent/ppo.yaml", sparse)) == Cell(
        "slalom", "main", 0.0, "", "ppo")
    assert classify(_fake_run("hppo", "configs\\env\\slalom_sparse_nocur.yaml", "configs\\agent\\hppo.yaml", nocur)
                    ) == Cell("slalom", "exploration", 0.0, "", "hppo")
    assert classify(_fake_run("ppo_mpc", "configs/env/slalom.yaml", "configs/agent/ppo_mpc.yaml", slalom)) == Cell(
        "slalom", "dense", 0.0, "", "ppo_mpc")
    disturbed = _env("trap_sparse", d_bar=1.0)
    assert classify(_fake_run("ppo_mpc_reachability", "configs/env/trap_sparse_d10.yaml",
                              "configs/agent/ppo_mpc_reachability.yaml", disturbed)) == Cell(
        "trap", "main", 1.0, "", "ppo_mpc_reachability")
    assert classify(_fake_run("hppo", "configs/env/slalom_sparse.yaml", "configs/agent/hppo_h6.yaml", sparse)
                    ).variant == "h6"
    assert classify(_fake_run("hppo", "configs/env/slalom_sparse.yaml", "configs/agent/hppo_map.yaml", sparse)
                    ).variant == "map"
    assert classify(_fake_run("ppo", "configs/env/tunnel.yaml", None, _env("tunnel"))) == Cell(
        "tunnel", "dense", 0.0, "", "ppo")
    cell = Cell("slalom", "main", 0.5, "", "ppo")
    assert cell.key == "slalom/main/d0.5/base/ppo" and not cell.base
    assert Cell("slalom", "main", 0.0, "", "ppo").base and not Cell("slalom", "main", 0.0, "h6", "ppo").base
    with pytest.raises(ValueError, match="lacks"):
        classify(Run(Path("none"), {"algorithm": "ppo"}, {}, ()))


def test_runs_are_grouped_by_cell_and_sorted_by_seed(tmp_path: Path) -> None:
    root = _matrix(tmp_path)
    from hrlmpc.analysis import find_runs

    cells = cells_of(find_runs([root]))
    assert len(cells) == 3 * 4 + 4 + 2 + 1
    main_ppo = cells[Cell("slalom", "main", 0.0, "", "ppo")]
    assert [int(r.seed) for r in main_ppo] == [1, 2]
    assert Cell("slalom", "main", 1.0, "", "ppo") in cells and Cell("slalom", "main", 0.0, "h6", "hppo") in cells
    assert Cell("slalom", "main", 0.0, "map", "hppo") in cells
    with pytest.raises(ValueError, match="twice"):
        cells_of(main_ppo + [main_ppo[0]])


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_kaplan_meier_fraction_solved_and_median() -> None:
    """Censored times (None) stay at risk until the budget and never count as solved."""
    xs, ys, median = km([10, 20, None, 20], budget=100)
    np.testing.assert_array_equal(xs, [0, 10, 20, 100])
    np.testing.assert_allclose(ys, [0.0, 0.25, 0.75, 0.75])
    assert median == 20
    xs, ys, median = km([None, None], budget=100)
    np.testing.assert_array_equal(xs, [0, 100]) and np.testing.assert_array_equal(ys, [0.0, 0.0])
    assert median is None
    _, _, median = km([30, None, None, None])
    assert median is None  # a quarter solved: the fraction never reaches one half
    with pytest.raises(ValueError):
        km([150], budget=100)


def test_fisher_and_log_rank() -> None:
    assert fisher(5, 5, 0, 5) == pytest.approx(0.00793650793650794)
    assert fisher(3, 5, 3, 5) == 1.0
    assert logrank([None] * 3, [None] * 3) is None
    assert logrank([10240] * 5, [10240] * 5) == 1.0  # identical times: no evidence of a difference
    early, late = [10240, 20480, 20480, 30720, 30720], [102400, 112640, None, None, None]
    p = logrank(early, late)
    assert p is not None and p < 0.01
    assert logrank(early, late) == logrank(late, early)
    assert logrank(early, early) == pytest.approx(1.0)


def test_confidence_intervals_skip_missing_values() -> None:
    assert ci([1.0, None, 3.0, math.nan]) == {"mean": 2.0, "low": pytest.approx(2.0 - 12.706204736174698),
                                             "high": pytest.approx(2.0 + 12.706204736174698), "n": 2}
    assert ci([2.0]) == {"mean": 2.0, "low": None, "high": None, "n": 1}
    assert ci([None]) == {"mean": None, "low": None, "high": None, "n": 0}
    assert mean_pm(ci([1.0, None, 3.0]), 1) == "2.0 $\\pm$ 12.7" and mean_pm(ci([2.0]), 1) == "2.0"
    assert mean_pm(ci([]), 1) == "--"
    assert k(None) == "--" and k(40960) == "41k" and k(512) == "512"
    assert pfmt(None) == "--" and pfmt(0.0004) == "$<0.001$" and pfmt(0.0051) == "0.005" and pfmt(0.25) == "0.25"


# --------------------------------------------------------------------------
# Metrics and summary
# --------------------------------------------------------------------------


def test_run_metrics_read_the_times_the_final_policy_and_the_counters(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, algorithm="ppo_mpc", env_name="slalom_sparse", seed=3,
                         successes=[0.0, 0.5, 1.0, 1.0, 1.0], gaps=[0.2, 0.2, 0.12, 0.04, 0.04],
                         contacts=[0.0, 0.0, 0.0, 0.0, 0.0], mpc=True, curriculum_stages=[0, 3, 8, 8, 8])
    m = run_metrics(load_run(run_dir))
    assert m["seed"] == 3 and m["complete"] is False and m["horizon"] == 200  # the budget is not reached
    assert m["t_success"] == 2 * EVERY and m["t_gap10"] == 3 * EVERY and m["t_strict"] == 3 * EVERY
    assert m["t_contact_free"] == 2 * EVERY
    assert m["final_success"] == 1.0 and m["final_gap_max"] == 0.04 and m["final_contact_fraction"] == 0.0
    assert m["final_return"] == 700.0 and m["train_contact_fraction"] == 0.0
    assert m["curriculum_stage"] == 8 and m["curriculum_stages"] == 9 and m["curriculum_last_stage_at"] == 2 * EVERY
    assert m["mpc_steps"] == 4 * 2048.0 and m["mpc_solver_error_count"] == 8.0 and m["mpc_solve_ms_max"] == 300.0
    assert m["reached_rate_eval_final"] == 0.5 and m["end_distance_eval_final"] == 0.6
    assert m["auc"] == pytest.approx((0.25 + 0.75 + 1.0 + 1.0) * EVERY / BUDGET)
    with pytest.raises(ValueError, match="evaluation"):
        run_metrics(Run(tmp_path, {"env": {"task": {"horizon": 200}}, "seed": 1}, {}, ()))


def test_group_stats_and_the_summary(tmp_path: Path) -> None:
    root = _matrix(tmp_path)
    from hrlmpc.analysis import find_runs

    summary = summarize(find_runs([root]))
    assert isinstance(summary, Summary)
    ppo = summary.stats[Cell("slalom", "main", 0.0, "", "ppo")]
    assert ppo["n"] == 2 and ppo["seeds"] == [1, 2] and ppo["t_success"]["solved"] == 2
    assert ppo["t_success"]["times"] == [3 * EVERY, 4 * EVERY] and ppo["t_success"]["km_median"] == 3 * EVERY
    assert ppo["t_success"]["min"] == 3 * EVERY and ppo["t_success"]["max"] == 4 * EVERY
    assert ppo["t_contact_free"]["solved"] == 0 and ppo["final_contact_fraction"]["mean"] == pytest.approx(0.1)
    assert ppo["curriculum_final_stage"] == [8, 8]
    hppo = summary.stats[Cell("slalom", "main", 0.0, "", "hppo")]
    assert hppo["t_success"]["solved"] == 0 and hppo["t_success"]["km_median"] is None
    assert hppo["final_gap_max_solved"]["n"] == 0 and "end_distance_eval_final" in hppo
    mpc = summary.stats[Cell("slalom", "main", 0.0, "", "ppo_mpc")]
    assert mpc["mpc_steps"] == 2 * 5 * 2048.0 and mpc["mpc_exit_converged_count_rate"] == pytest.approx(1500 / 2048)
    assert mpc["mpc_emergency_episode_equivalents"] == 0.0 and "reached_rate_eval_final" in mpc
    reach = summary.stats[Cell("slalom", "main", 0.0, "", "ppo_mpc_reachability")]
    assert reach["reach_decisions"] == 2 * 5 * 200.0 and reach["reach_support_failure_count"] == 10.0
    tests = summary.tests["slalom/main/d0/base"]
    assert set(tests) == {f"{a} vs {b}" for i, a in enumerate(ARCHS) for b in ARCHS[i + 1:]}
    pair = tests["ppo vs ppo_mpc"]
    assert pair["fisher_success"] == 1.0 and 0.0 < pair["logrank_success"] <= 1.0
    assert tests["ppo vs hppo"]["fisher_success"] == pytest.approx(1 / 3)
    assert "slalom/main/d1/base" in summary.tests and "slalom/main/d0/h6" in summary.tests
    assert summary.ablations["slalom/main/d0/map/hppo"]["base"] == "slalom/main/d0/base/hppo"
    assert summary.ablations["slalom/main/d0/h6/ppo_mpc"]["base"] == "slalom/main/d0/base/ppo_mpc"
    assert summary.cell("slalom", "main", "ppo") == Cell("slalom", "main", 0.0, "", "ppo")
    assert summary.cell("trap", "main", "ppo") is None
    as_json = summary.as_json()
    assert set(as_json) == {"cells", "tests", "ablations"} and "slalom/main/d0/base/ppo" in as_json["cells"]
    json.dumps(as_json, default=float)


# --------------------------------------------------------------------------
# Tables, figures, report, command line
# --------------------------------------------------------------------------


def _escape(run_dir: Path, name: str, success: float, contact: float) -> None:
    line = {"name": name, "action": "mode", "seed": 12345, "checkpoint": "final.pt",
            "box": {"x": [4.0, 5.0], "y": [-0.4, 0.4]}, "grid": [3, 3],
            "metrics": {"eval/success_rate": success, "eval/contact_fraction": contact, "eval/length": 150.0}}
    with (run_dir / REEVALUATIONS).open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def test_the_tables_are_tabular_bodies_with_one_row_per_cell(tmp_path: Path) -> None:
    root = _matrix(tmp_path)
    from hrlmpc.analysis import find_runs

    summary = summarize(find_runs([root]))
    table = results_table(summary)
    assert table.startswith("\\begin{tabular}") and table.rstrip().endswith("\\end{tabular}")
    assert table.count("\\emph{") == 3  # slalom main, slalom exploration, tunnel dense
    assert table.count("PPO\\textsubscript{MPC} &") == 3 and "9/10" not in table
    assert "PPO & 2/2 & 31k & 31k--41k & 0/2 &" in table  # the slalom's main setting: solved late, with contact
    assert "hPPO & 0/2 & -- & -- & 0/2 & -- &" in table
    assert results_table(summary, ("tunnel",)).count("\\emph{") == 1
    assert results_table(summary, ("trap",)).count("\\emph{") == 0
    tests = logrank_table(summary)
    assert tests.count(" \\\\") == 1 + 3  # the header and three contexts
    assert "Slalom, main (curriculum) & 1.00 &" in tests or "Slalom, main (curriculum) & 0." in tests
    sweep = sweep_table(summary)
    assert "$H = 10$" in sweep and "$H = 6$" in sweep and "$H = 4$" not in sweep
    disturbed = disturbance_table(summary)
    assert "$\\bar d = 0$" in disturbed and "$\\bar d = 1$" in disturbed and disturbed.count("\\emph{") == 2
    ablation = ablation_table(summary)
    assert "hPPO (map) &" in ablation and ablation.count("\\emph{") == 1 and "& log-rank" in ablation
    worker = worker_table(summary)
    assert "Slalom, $\\bar d = 0$ & PPO\\textsubscript{MPC} &" in worker and "All layouts, $\\bar d > 0$" in worker
    assert "73.2" in worker  # converged 1500 of 2048 steps
    compute = compute_table(summary)
    assert "Slalom and trap, $\\bar d = 0$ & PPO &" in compute and "Tunnel, $\\bar d = 0$" in compute
    assert "50 + 30" in compute  # support + projection of the reachable targets
    written = write_tables(summary, tmp_path / "tables")
    assert {p.name for p in written} >= {"results_all.tex", "tests.tex", "escape.tex", "worker_solves.tex",
                                         "compute.tex", "disturbance.tex", "sweep.tex", "ablation_map.tex"}


def test_the_escape_evaluation_is_read_from_the_runs(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    for arch in ("ppo", "ppo_mpc"):
        for seed in (1, 2):
            run_dir = _write_run(root, algorithm=arch, env_name="trap_sparse", seed=seed,
                                 successes=[0.0, 1.0, 1.0], mpc=arch == "ppo_mpc")
            _escape(run_dir, "escape", 0.0 if arch == "ppo" else 0.5 * seed, 1.0)
            _escape(run_dir, "mouth", 0.2, 0.5)
    from hrlmpc.analysis import find_runs

    summary = summarize(find_runs([root]))
    rates = escape_rates(summary)
    assert rates["trap/main/d0/base/ppo_mpc/escape"]["success"]["mean"] == pytest.approx(0.75)
    assert rates["trap/main/d0/base/ppo_mpc/escape"]["success_per_seed"] == [0.5, 1.0]
    assert rates["trap/main/d0/base/ppo/mouth"]["contact"]["mean"] == 0.5
    table = escape_table(summary)
    assert "PPO\\textsubscript{MPC} & 75 & 20 & -- & -- & -- & -- \\\\" in table
    assert "PPO & 0 & 20 & -- & -- & -- & -- \\\\" in table and "hPPO" not in table
    assert escape_table(summarize(find_runs([root / "trap_sparse" / "ppo"])), layout="slalom").count("\\\\") == 2
    assert escape_rates(summary, layout="slalom") == {}


def test_the_figures_and_the_report_are_written(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    root = _matrix(tmp_path)
    from hrlmpc.analysis import find_runs

    summary = summarize(find_runs([root]))
    figures = write_figures(summary, tmp_path / "figures")
    names = {p.name for p in figures}
    assert names >= {"slalom_success", "tunnel_success", "slalom_return", "time_to_success", "levels_sweep",
                     "training_contact", "curriculum_stage", "hppo_map", "time_to_success_d1"}
    assert "trap_success" in names  # drawn empty: the layout has no runs here
    assert "trajectories_main" not in names  # no replay below the roots
    for p in figures:
        assert p.with_suffix(".pdf").exists() and p.with_suffix(".png").exists()
    write_report(summary, tmp_path / "report.md")
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## slalom/main/d0/base/ppo" in report and "INCOMPLETE" in report  # the synthetic runs stop early
    assert "## Tests (two-sided)" in report and "## Variants against their bases" in report


def test_the_trajectories_figure_draws_the_replays(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from hrlmpc.results import trajectories_figure
    from hrlmpc.trajectories import Replay

    env = _env("slalom")
    starts = np.array([[0.0, -1.0], [1.0, 0.0]])
    positions = np.full((2, 4, 2), np.nan)
    positions[0, :3] = [[0.0, -1.0], [0.1, -1.0], [0.2, -0.9]]
    positions[1, :4] = [[1.0, 0.0], [1.1, 0.0], [1.2, 0.0], [1.3, 0.1]]
    contact = np.zeros((2, 3), dtype=bool)
    contact[0, 0] = True
    targets = np.full((2, 3, 2), np.nan)
    targets[1] = [[2.0, 0.0]] * 3
    targets[0, :2] = [[1.5, -1.0]] * 2
    replays = {}
    for arch, with_targets in (("ppo", False), ("hppo", True)):
        replays[arch] = Replay(starts=starts, positions=positions, velocities=np.zeros_like(positions),
                               contact=contact, length=np.array([2, 3]), success=np.array([False, True]),
                               targets=targets if with_targets else None, run="seed1", algorithm=arch, name="eval",
                               action="mode", seed=12345, checkpoint="final.pt", env=env,
                               segment_steps=10 if with_targets else 0, target_radius=1.8 if with_targets else math.nan)
    trajectories_figure(replays, tmp_path / "trajectories", targets_of=1)
    assert (tmp_path / "trajectories.pdf").exists()
    # two groups: one column each, a row per architecture present in any of them
    trajectories_figure([("slalom, main", replays), ("slalom, exploration", {"ppo": replays["ppo"]})],
                        tmp_path / "trajectories_main", targets_of=1)
    assert (tmp_path / "trajectories_main.pdf").exists()


def test_the_command_line_writes_everything(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("matplotlib")
    root = _matrix(tmp_path)
    out = tmp_path / "out"
    assert main([str(root / "final"), str(root / "final_h6"), str(root / "final_map"), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "cells" in text and "INCOMPLETE" in text and "figures under" in text
    assert (out / "summary.json").exists() and (out / "report.md").exists()
    assert (out / "tables" / "results_all.tex").exists() and (out / "figures" / "slalom_success.pdf").exists()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert "slalom/main/d0/base/ppo" in summary["cells"]
    assert main([str(root / "final"), "--out", str(tmp_path / "out2"), "--no-figures"]) == 0
    assert not (tmp_path / "out2" / "figures").exists()
