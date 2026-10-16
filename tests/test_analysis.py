"""Tests of the run analysis: learning curves over seeds with confidence intervals (code rule C4, decision log D20,
D43, D50)."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from scipy import stats

from hrlmpc.analysis import (
    CURVE_KEYS,
    GAP_LENIENT,
    SOLVED_LEVELS,
    accepted,
    curve_keys,
    find_runs,
    first_accepted,
    group_runs,
    has_oracle,
    inventory,
    load_run,
    main,
    mean_curve,
    series,
    solved,
    success_auc,
    summarize,
    t_interval,
    time_to_solve,
    uses_oracle,
)
from hrlmpc.config import env_config_to_dict, load_env_config
from hrlmpc.utils.runlog import RunLogger, SampleCounter

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "env"
TUNNEL: dict[str, Any] = {"geometry": {"obstacles": []}}
SLALOM: dict[str, Any] = {"geometry": {"obstacles": [{"x": [4.0, 5.0], "y": [-2.0, 0.25]}]}}


def _eval(success: float, gap: float, contact: float, ret: float, mode: tuple[float, ...] | None = None
          ) -> dict[str, Any]:
    """An evaluation record; with ``mode``, also that of the mode (D43): its success, gap, contact and return."""
    record = {
        "eval/success_rate": success,
        "eval/arrival_gap_max": gap,
        "eval/arrival_gap_mean": gap / 2,
        "eval/contact_fraction": contact,
        "eval/unshaped_return": ret,
    }
    if mode is not None:
        record.update({
            "eval_mode/success_rate": mode[0],
            "eval_mode/arrival_gap_max": mode[1],
            "eval_mode/arrival_gap_mean": mode[1] / 2,
            "eval_mode/contact_fraction": mode[2],
            "eval_mode/unshaped_return": mode[3],
        })
    return record


def _make_run(
    directory: Path,
    seed: int,
    evaluations: list[dict[str, Any]],
    every: int = 10240,
    env: dict[str, Any] | None = None,
    name: str | None = None,
    agent: tuple[str, dict[str, Any]] | None = None,
    budget: int | None = None,
) -> Path:
    run_dir = directory / (name or f"seed{seed}")
    config = {"algorithm": "ppo", "seed": seed, "env_file": "configs/env/tunnel.yaml", "env": env or TUNNEL}
    if budget is not None:
        config["budget"] = budget
    if agent is not None:
        config["agent_file"] = agent[0]
        config["agent"] = agent[1]
    with RunLogger(run_dir, config, seed=seed) as logger:
        counter = SampleCounter()
        for i, record in enumerate(evaluations):
            if i > 0:
                counter.add_env_steps(every)
                logger.log({"train/policy_loss": 0.1 * i}, counter)
            logger.log(record, counter)
    return run_dir


@pytest.fixture
def three_runs(tmp_path: Path) -> list[Path]:
    start = _eval(0.0, math.nan, 0.0, -200.0)
    return [
        _make_run(tmp_path, 1, [start, _eval(0.8, 0.10, 0.04, 700.0), _eval(1.0, 0.03, 0.0, 905.0)]),
        _make_run(tmp_path, 2, [start, _eval(1.0, 0.04, 0.0, 900.0), _eval(1.0, 0.02, 0.0, 906.0)]),
        _make_run(tmp_path, 3, [start, _eval(0.4, 0.20, 0.0, 300.0), _eval(1.0, 0.07, 0.0, 899.0)]),
    ]


def test_the_t_interval_is_students() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    mean, low, high = t_interval(values)
    half = stats.t.ppf(0.975, 4) * values.std(ddof=1) / math.sqrt(5)
    assert mean == pytest.approx(3.2) and low == pytest.approx(3.2 - half) and high == pytest.approx(3.2 + half)
    half_90 = stats.t.ppf(0.95, 4) * values.std(ddof=1) / math.sqrt(5)
    assert t_interval(values, confidence=0.9)[2] == pytest.approx(3.2 + half_90)
    assert t_interval(np.array([2.0, 2.0]))[1:] == (2.0, 2.0)
    single = t_interval(np.array([5.0]))
    assert single[0] == 5.0 and math.isnan(single[1]) and math.isnan(single[2])
    with pytest.raises(ValueError, match="confidence"):
        t_interval(values, confidence=95.0)


def test_a_run_directory_is_read_back(three_runs: list[Path]) -> None:
    run = load_run(three_runs[0])
    assert run.config["seed"] == 1 and run.meta["seed"] == 1 and run.seed == 1
    assert not run.has_obstacles
    assert len(run.records) == 5 and run.records[0]["env_steps"] == 0
    steps, values = series(run, "eval/success_rate")
    np.testing.assert_array_equal(steps, [0, 10240, 20480])
    np.testing.assert_array_equal(values, [0.0, 0.8, 1.0])
    _, gaps = series(run, "eval/arrival_gap_max")
    assert math.isnan(gaps[0])  # written as null


def test_runs_are_found_below_a_directory_once(tmp_path: Path, three_runs: list[Path]) -> None:
    (tmp_path / "not-a-run").mkdir()
    found = find_runs([tmp_path, three_runs[0], three_runs[0] / ".." / three_runs[0].name])
    assert [r.path for r in found] == sorted(p.resolve() for p in three_runs)
    assert [r.path for r in find_runs([three_runs[1]])] == [three_runs[1].resolve()]
    with pytest.raises(ValueError, match="no run"):
        find_runs([tmp_path / "not-a-run"])


def test_group_labels_name_the_layout_on_any_platform(tmp_path: Path) -> None:
    run = load_run(_make_run(tmp_path, 1, [_eval(1.0, 0.01, 0.0, 900.0)]))
    run.config["env_file"] = "configs\\env\\slalom.yaml"  # written on Windows
    assert list(group_runs([run])) == ["ppo slalom"]


def test_runs_are_grouped_by_their_configuration(tmp_path: Path) -> None:
    record = [_eval(1.0, 0.01, 0.0, 900.0)]
    for seed in (1, 2):
        _make_run(tmp_path / "tunnel", seed, record)
        _make_run(tmp_path / "slalom", seed, record, env=SLALOM)
    groups = group_runs(find_runs([tmp_path]))
    assert len(groups) == 2 and all(len(members) == 2 for members in groups.values())
    assert {members[0].has_obstacles for members in groups.values()} == {True, False}
    assert all(label.startswith("ppo tunnel") for label in groups)  # same file stem, told apart by a hash
    _make_run(tmp_path / "again", 1, record)
    with pytest.raises(ValueError, match="seed twice"):
        group_runs(find_runs([tmp_path]))


def test_a_run_written_before_an_optional_key_existed_shares_the_group(tmp_path: Path) -> None:
    """A recorded configuration that lacks an optional key of today's schema (the bands' height of D69, say)
    fingerprints as one that records its default: the groups follow the configuration, not the schema's age."""
    env = env_config_to_dict(load_env_config(CONFIG_DIR / "slalom_sparse.yaml"))
    record = [_eval(1.0, 0.01, 0.0, 900.0)]
    _make_run(tmp_path, 1, record, env=env)
    older = copy.deepcopy(env)
    del older["task"]["curriculum"]["y"]  # the key did not exist when the run was written
    _make_run(tmp_path, 2, record, env=older)
    groups = group_runs(find_runs([tmp_path]))
    assert list(groups) == ["ppo tunnel"] and len(groups["ppo tunnel"]) == 2
    other = copy.deepcopy(env)
    other["task"]["curriculum"]["y"] = [-2.0, 2.0]  # a different configuration still tells
    _make_run(tmp_path / "other", 1, record, env=other)
    assert len(group_runs(find_runs([tmp_path]))) == 2


def test_an_incomplete_run_is_flagged_and_its_auc_counts_the_area_it_reached(tmp_path: Path) -> None:
    """A run killed before its budget (a restart of the machine, say) keeps its records: the summary says so, and
    its AUC is the area over the budget, not over the shorter range it reached; a run without a recorded budget
    is taken as complete, with the AUC over its last evaluation (older runs)."""
    curve = [_eval(0.0, math.nan, 0.0, -200.0), _eval(0.5, 0.1, 0.0, 400.0), _eval(1.0, 0.02, 0.0, 905.0)]
    full = load_run(_make_run(tmp_path / "full", 1, curve, budget=20480))
    cut = load_run(_make_run(tmp_path / "cut", 1, curve, budget=40960))
    old = load_run(_make_run(tmp_path / "old", 1, curve))
    assert full.complete and old.complete and not cut.complete
    assert full.budget == 20480 and old.budget is None
    assert success_auc(full) == pytest.approx(0.5) and success_auc(old) == pytest.approx(0.5)
    assert success_auc(cut) == pytest.approx(0.25)  # the same area over twice the samples
    text = summarize(find_runs([tmp_path / "cut"]))
    assert "INCOMPLETE: seed1 stopped at 20480 of 40960 samples" in text
    assert "INCOMPLETE" not in summarize(find_runs([tmp_path / "full"]))
    listing = inventory(find_runs([tmp_path / "cut", tmp_path / "full"]))
    assert listing.count("\n") == 1 and "2 runs" not in listing  # two groups (the budgets differ), one run each
    assert "INCOMPLETE: seed1 (20480 of 40960)" in listing
    _make_run(tmp_path / "full", 2, curve, budget=20480)
    listing = inventory(find_runs([tmp_path / "full"]))
    assert "2 runs  seeds 1, 2" in listing and "INCOMPLETE" not in listing
    assert main([str(tmp_path / "full"), "--inventory"]) == 0


def test_the_mean_curve_has_a_confidence_band(three_runs: list[Path]) -> None:
    runs = [load_run(p) for p in three_runs]
    curve = mean_curve(runs, "eval/unshaped_return")
    np.testing.assert_array_equal(curve.env_steps, [0, 10240, 20480])
    assert curve.n == 3
    np.testing.assert_array_equal(curve.count, [3, 3, 3])
    np.testing.assert_allclose(curve.mean, [-200.0, 633.33333333, 903.33333333])
    _, low, high = t_interval(np.array([905.0, 906.0, 899.0]))
    assert curve.low[-1] == pytest.approx(low) and curve.high[-1] == pytest.approx(high)


def test_values_missing_in_some_runs_are_left_out_point_by_point(tmp_path: Path) -> None:
    a = load_run(_make_run(tmp_path, 1, [_eval(0.0, math.nan, 0.0, 0.0), _eval(1.0, 0.04, 0.0, 0.0)]))
    b = load_run(_make_run(tmp_path, 2, [_eval(1.0, 0.20, 0.0, 0.0), _eval(1.0, 0.08, 0.0, 0.0)]))
    curve = mean_curve([a, b], "eval/arrival_gap_max")
    np.testing.assert_array_equal(curve.count, [1, 2])
    np.testing.assert_allclose(curve.mean, [0.20, 0.06])
    assert math.isnan(curve.low[0]) and not math.isnan(curve.low[1])


def test_curves_need_the_same_evaluation_points(tmp_path: Path) -> None:
    a = load_run(_make_run(tmp_path, 1, [_eval(1.0, 0.0, 0.0, 1.0)] * 3))
    b = load_run(_make_run(tmp_path, 2, [_eval(1.0, 0.0, 0.0, 1.0)] * 3, every=2048))
    with pytest.raises(ValueError, match="evaluation points"):
        mean_curve([a, b], "eval/success_rate")


def test_the_acceptance_criterion_of_step_7() -> None:
    """D20: success from every start, arrival within 5% of the bound, no contact."""
    assert accepted(_eval(1.0, 0.05, 0.0, 0.0), max_gap=0.05)
    assert not accepted(_eval(1.0, 0.051, 0.0, 0.0), max_gap=0.05)
    assert not accepted(_eval(0.96, 0.01, 0.0, 0.0), max_gap=0.05)
    assert not accepted(_eval(1.0, 0.01, 0.04, 0.0), max_gap=0.05)
    assert not accepted(_eval(0.0, math.nan, 0.0, 0.0), max_gap=0.05)
    assert not accepted({"train/policy_loss": 0.0}, max_gap=0.05)


def test_the_first_accepted_evaluation(three_runs: list[Path]) -> None:
    runs = [load_run(p) for p in three_runs]
    assert [first_accepted(r, max_gap=0.05) for r in runs] == [20480, 10240, None]
    assert [first_accepted(r, max_gap=0.025) for r in runs] == [None, 20480, None]


def _rows(text: str, action: str = "mean") -> dict[str, list[str]]:
    """The rows of the runs in the table of ``action`` (before the time-to-solve block)."""
    mean, _, mode = text.partition("time to solve")[0].partition("action: mode")
    table = mean if action == "mean" else mode
    return {line.split()[0]: line.split() for line in table.splitlines() if line.startswith("seed")}


def _solve_rows(text: str) -> dict[str, list[str]]:
    """The rows of the runs in the time-to-solve block (D67)."""
    block = text.partition("time to solve")[2].partition("solved:")[0]
    return {line.split()[0]: line.split() for line in block.splitlines() if line.startswith("seed")}


def test_the_summary_reports_every_run(three_runs: list[Path]) -> None:
    text = summarize([load_run(p) for p in three_runs], max_gap=0.05)
    rows = _rows(text)
    assert [rows[f"seed{s}"][7] for s in (1, 2, 3)] == ["yes", "yes", "no"]
    assert [rows[f"seed{s}"][8] for s in (1, 2, 3)] == ["20480", "10240", "-"]
    assert "== ppo tunnel (3 runs)" in text and "accepted at the end: 2/3 (max gap 0.05, on the gap of F7)" in text
    strict = summarize([load_run(p) for p in three_runs], max_gap=0.025)
    assert [_rows(strict)[f"seed{s}"][7] for s in (1, 2, 3)] == ["no", "yes", "no"]
    assert "accepted at the end: 1/3" in strict


def test_acceptance_does_not_apply_with_obstacles_and_no_oracle(tmp_path: Path) -> None:
    """With obstacles F7 is only a lower bound, so without an oracle table the gap criterion is not applied."""
    run = load_run(_make_run(tmp_path, 1, [_eval(1.0, 0.01, 0.0, 900.0)], env=SLALOM))
    text = summarize([run])
    assert _rows(text)["seed1"][7] == "n/a" and "acceptance not applicable" in text


def test_the_acceptance_uses_the_oracle_gap_where_it_is_logged(tmp_path: Path) -> None:
    """D59: with an oracle table the criterion of D20 applies to the slalom too, on the oracle gap; the gap of F7
    stays reported."""
    first = _eval(1.0, 0.20, 0.0, 850.0)
    first.update({"eval/oracle_gap_max": 0.08, "eval/oracle_gap_mean": 0.04})
    last = _eval(1.0, 0.12, 0.0, 900.0)
    last.update({"eval/oracle_gap_max": 0.03, "eval/oracle_gap_mean": 0.01})
    run = load_run(_make_run(tmp_path, 1, [first, last], env=SLALOM))
    text = summarize([run], max_gap=0.05)
    assert _rows(text)["seed1"][7] == "yes" and _rows(text)["seed1"][8] == "10240"
    assert "accepted at the end: 1/1 (max gap 0.05, on the oracle gap, D59)" in text
    assert "eval/oracle_gap_max        mean 0.030" in text
    assert accepted(last, max_gap=0.05) and not accepted(first, max_gap=0.05)
    assert accepted(last, max_gap=0.05, oracle=False) is False  # on the gap of F7 alone, 0.12 fails
    assert first_accepted(run, max_gap=0.05) == 10240


def test_a_run_without_arrivals_at_the_end_is_rejected_not_exempted(tmp_path: Path) -> None:
    """An evaluation without arrivals carries no oracle gap; the run still measures it (the table is named in the
    configuration, or an earlier evaluation carries the gap), so its verdict is "no", not "n/a"."""
    first = _eval(1.0, 0.20, 0.0, 850.0)
    first.update({"eval/oracle_gap_max": 0.08, "eval/oracle_gap_mean": 0.04})
    failed = _eval(0.0, math.nan, 0.0, -200.0)
    run = load_run(_make_run(tmp_path, 1, [first, failed], env=SLALOM))
    text = summarize([run], max_gap=0.05)
    assert _rows(text)["seed1"][7] == "no" and "accepted at the end: 0/1" in text
    assert uses_oracle(run) and not has_oracle(failed)
    named = _make_run(tmp_path / "named", 2, [failed], env=SLALOM)
    config = yaml.safe_load((named / "config.yaml").read_text(encoding="utf-8"))
    config["oracle_file"] = "configs/oracle/slalom.json"
    (named / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    text = summarize([load_run(named)], max_gap=0.05)
    assert _rows(text)["seed2"][7] == "no" and "on the oracle gap, D59" in text
    assert "acceptance not applicable" in summarize([load_run(_make_run(tmp_path / "bare", 3, [failed], env=SLALOM))])


def test_the_acceptance_is_decided_on_the_mode_where_it_is_logged(tmp_path: Path) -> None:
    """D57: from the final runs on, the mode is the primary action: the acceptance is decided on its evaluation when
    it is logged, and on the mean's otherwise (the runs of steps 7-10)."""
    run = load_run(_make_run(tmp_path, 1, [_eval(1.0, 0.04, 0.0, 905.0, mode=(1.0, 0.10, 0.0, 880.0))]))
    text = summarize([run], max_gap=0.05)
    assert "acceptance on the mode (D57)" in text and "accepted at the end: 0/1" in text
    assert _rows(text, "mode")["seed1"][7] == "no"
    lenient = summarize([run], max_gap=0.10)
    assert "accepted at the end: 1/1" in lenient
    old = load_run(_make_run(tmp_path / "old", 1, [_eval(1.0, 0.04, 0.0, 905.0)]))
    text = summarize([old], max_gap=0.05)
    assert "acceptance on the mean (D20)" in text and "accepted at the end: 1/1" in text


def test_the_curriculum_is_summarized(tmp_path: Path) -> None:
    """D62: runs with a curriculum report the stage at the end and the samples at which the last stage was
    reached; runs without one say nothing about it."""
    env = {"geometry": {"obstacles": []}, "task": {"curriculum": {"x_start": 8.0, "step": 1.0, "window": 50,
                                                                   "threshold": 0.8}}}
    run_dir = tmp_path / "seed1"
    config = {"algorithm": "ppo", "seed": 1, "env_file": "x.yaml", "env": env}
    with RunLogger(run_dir, config, seed=1) as logger:
        counter = SampleCounter()
        logger.log(_eval(0.0, math.nan, 0.0, -200.0), counter)
        stages = [(2048, 3.0), (4096, 7.0), (6144, 8.0), (8192, 8.0)]
        for steps, stage in stages:
            counter.add_env_steps(steps - counter.env_steps)
            logger.log({"train/policy_loss": 0.1, "curriculum/stage": stage, "curriculum/stages": 9.0,
                        "curriculum/x_min": 8.0 - stage, "curriculum/final": float(stage == 8.0),
                        "curriculum/window_success": 0.5}, counter)
            if steps == 8192:
                logger.log(_eval(1.0, 0.02, 0.0, 900.0), counter)
    text = summarize([load_run(run_dir)])
    assert "curriculum (D62): stage at the end and samples to the last stage" in text
    assert "seed1" in text and " 8/8 " in text and " 6144" in text
    plain = summarize([load_run(_make_run(tmp_path, 2, [_eval(1.0, 0.02, 0.0, 900.0)]))])
    assert "curriculum" not in plain


def test_the_command_line_prints_and_plots(
    three_runs: list[Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("matplotlib")
    figure = tmp_path / "curves.png"
    assert main([str(tmp_path), "--plot", str(figure)]) == 0
    assert "accepted at the end: 2/3" in capsys.readouterr().out
    assert figure.stat().st_size > 0


def test_the_summary_reports_the_mode_beside_the_mean(tmp_path: Path) -> None:
    """D43, D50, D57: a table for each action; where the mode is logged the criterion of D20 is decided on it."""
    run = load_run(_make_run(tmp_path, 1, [_eval(1.0, 0.04, 0.0, 905.0, mode=(0.75, 0.01, 0.25, 640.0))]))
    text = summarize([run], max_gap=0.05)
    assert _rows(text)["seed1"][2:7] == ["1.00", "905.0", "0.020", "0.040", "0.00"]
    assert _rows(text, "mode")["seed1"][2:8] == ["0.75", "640.0", "0.005", "0.010", "0.25", "no"]
    assert "accepted at the end: 0/1 (max gap 0.05, on the gap of F7)" in text  # on the mode, whatever the mean did
    assert "eval_mode/success_rate     mean 0.750" in text


def test_runs_without_the_mode_say_so(three_runs: list[Path]) -> None:
    """The runs of steps 7-10 were evaluated with the mean only."""
    text = summarize([load_run(p) for p in three_runs])
    assert "action: mode (D43): not evaluated in these runs" in text and not _rows(text, "mode")


def test_the_curves_of_either_action(three_runs: list[Path], tmp_path: Path) -> None:
    assert curve_keys("mean") == CURVE_KEYS
    assert curve_keys("mode") == tuple(key.replace("eval/", "eval_mode/") for key in CURVE_KEYS)
    with pytest.raises(ValueError, match="action"):
        curve_keys("median")
    pytest.importorskip("matplotlib")
    _make_run(tmp_path / "mixed", 4, [_eval(0.5, 0.1, 0.0, 400.0, mode=(0.5, 0.2, 0.0, 380.0))] * 2)
    _make_run(tmp_path / "mixed", 5, [_eval(0.5, 0.1, 0.0, 400.0)] * 2)  # before D43: the mean only
    figure = tmp_path / "mode.png"
    assert main([str(tmp_path / "mixed"), "--plot", str(figure), "--action", "mode"]) == 0
    assert figure.stat().st_size > 0



# ------------------------------------------------------------------- tuning (D55, D65)


def test_the_success_auc_is_the_normalized_area_under_the_curve(tmp_path: Path) -> None:
    """Evaluations at 0, 10 240 and 20 480 with success 0, 0.5 and 1 (trapezoids) give 0.5 of the budget."""
    run = load_run(_make_run(tmp_path, 1, [_eval(0.0, math.nan, 0.0, -200.0), _eval(0.5, 0.2, 0.0, 300.0),
                                           _eval(1.0, 0.05, 0.0, 900.0)]))
    assert success_auc(run, "mean") == pytest.approx(0.5)
    assert math.isnan(success_auc(run, "mode"))  # not logged
    single = load_run(_make_run(tmp_path / "one", 1, [_eval(1.0, 0.05, 0.0, 900.0)]))
    assert math.isnan(success_auc(single, "mean"))  # one point has no area


def test_group_labels_name_the_agent_file_when_it_is_not_the_algorithms(tmp_path: Path) -> None:
    """Tuning and sweep files (c3.yaml, hppo_h6.yaml) tell the groups apart by name, not by a hash."""
    a = _make_run(tmp_path / "a", 1, [_eval(1.0, 0.05, 0.0, 900.0)], agent=("configs/agent/tuning/ppo/c3.yaml",
                                                                             {"lr": 1.0}))
    b = _make_run(tmp_path / "b", 1, [_eval(1.0, 0.05, 0.0, 900.0)], agent=("configs/agent/tuning/ppo/c4.yaml",
                                                                             {"lr": 2.0}))
    c = _make_run(tmp_path / "c", 1, [_eval(1.0, 0.05, 0.0, 900.0)], agent=("configs\\agent\\ppo.yaml", {"lr": 3.0}))
    groups = group_runs([load_run(a), load_run(b), load_run(c)])
    assert sorted(groups) == ["ppo tunnel", "ppo tunnel c3", "ppo tunnel c4"]


def test_the_summary_ranks_the_groups_by_the_auc(tmp_path: Path) -> None:
    """D65: with several groups the summary ends with the ranking by the mean AUC of the success curve of the
    primary action (the mode where logged), the selection rule of the tuning protocol."""
    low = [_eval(0.0, math.nan, 0.0, -200.0), _eval(0.0, math.nan, 0.0, -200.0), _eval(1.0, 0.05, 0.0, 900.0)]
    high = [_eval(0.0, math.nan, 0.0, -200.0), _eval(1.0, 0.05, 0.0, 850.0), _eval(1.0, 0.05, 0.0, 890.0)]
    runs = []
    for seed in (101, 102):
        runs.append(load_run(_make_run(tmp_path / "c1", seed, low, agent=("tuning/ppo/c1.yaml", {"k": 1}))))
        runs.append(load_run(_make_run(tmp_path / "c2", seed, high, agent=("tuning/ppo/c2.yaml", {"k": 2}))))
    text = summarize(runs)
    assert "auc of the success curve (mean)" in text
    ranking = text.split("groups ranked by the AUC of the success curve")[1]
    lines = [line for line in ranking.splitlines() if line.strip().startswith(("1.", "2."))]
    assert lines[0].split()[1:4] == ["ppo", "tunnel", "c2"] and "auc 0.750" in lines[0] and "n=2" in lines[0]
    assert lines[1].split()[1:4] == ["ppo", "tunnel", "c1"] and "auc 0.250" in lines[1]
    assert "final return 890.0" in lines[0]
    single = summarize([runs[0]])
    assert "groups ranked" not in single


def test_the_time_to_solve_at_the_three_thresholds(tmp_path: Path) -> None:
    """D67: the first evaluation with full success from the grid (primary), with full success and the gap at most
    10 %, and meeting the strict criterion of D20 (gap at most 5 %, no contact); the contact-free full success is
    reported beside them. On the primary action, the gap of the oracle where logged, else of F7."""
    assert SOLVED_LEVELS == ("success", "gap_10", "strict", "contact_free") and GAP_LENIENT == 0.10
    evaluations = [
        _eval(0.0, math.nan, 0.0, -200.0, mode=(0.0, math.nan, 0.0, -200.0)),
        _eval(1.0, 0.20, 0.04, 800.0, mode=(0.8, 0.20, 0.04, 700.0)),  # 10240: the mean arrives, the mode not yet
        _eval(1.0, 0.08, 0.04, 850.0, mode=(1.0, 0.20, 0.04, 800.0)),  # 20480: full success of the mode, contact
        _eval(1.0, 0.08, 0.00, 870.0, mode=(1.0, 0.08, 0.04, 850.0)),  # 30720: gap 8 %, still a contact
        _eval(1.0, 0.03, 0.00, 900.0, mode=(1.0, 0.08, 0.00, 870.0)),  # 40960: contact-free
        _eval(1.0, 0.03, 0.00, 900.0, mode=(1.0, 0.03, 0.00, 900.0)),  # 51200: strict
    ]
    run = load_run(_make_run(tmp_path, 1, evaluations))
    assert time_to_solve(run, "mode") == {"success": 20480, "gap_10": 30720, "strict": 51200, "contact_free": 40960}
    assert time_to_solve(run, "mean") == {"success": 10240, "gap_10": 20480, "strict": 40960, "contact_free": 30720}
    assert time_to_solve(run, "mode", max_gap=0.02) == {"success": 20480, "gap_10": 30720, "strict": None,
                                                        "contact_free": 40960}
    record = evaluations[3]
    assert solved(record, "success", "mode") and solved(record, "gap_10", "mode")
    assert not solved(record, "strict", "mode") and not solved(record, "contact_free", "mode")
    with pytest.raises(ValueError, match="level"):
        solved(record, "gap_20", "mode")
    text = summarize([run], max_gap=0.05)
    assert "time to solve (D67)" in text
    assert _solve_rows(text)["seed1"][1:] == ["20480", "30720", "51200", "40960"]
    assert "solved: full success 1/1 (20480), gap <= 10% 1/1 (30720), strict 1/1 (51200), contact-free 1/1 (40960)" \
        in text


def test_the_time_to_solve_is_censored_and_needs_the_oracle_for_the_gaps(tmp_path: Path) -> None:
    """A run that never meets a level is censored ("-"); on an obstacle layout without an oracle table the gap
    levels are not applicable, the success levels still are."""
    evaluations = [_eval(0.0, math.nan, 0.0, -200.0), _eval(1.0, 0.03, 0.0, 900.0)]
    text = summarize([load_run(_make_run(tmp_path, 1, evaluations, env=SLALOM))], max_gap=0.05)
    assert _solve_rows(text)["seed1"][1:] == ["10240", "n/a", "n/a", "10240"]
    assert "solved: full success 1/1 (10240), gap <= 10% n/a, strict n/a, contact-free 1/1 (10240)" in text
    stuck = [_eval(0.0, math.nan, 0.0, -200.0), _eval(0.5, 0.30, 0.2, 300.0)]
    runs = [load_run(_make_run(tmp_path / "a", 2, stuck)), load_run(_make_run(tmp_path / "b", 3, evaluations))]
    text = summarize(runs, max_gap=0.05)
    rows = _solve_rows(text)
    assert rows["seed2"][1:] == ["-", "-", "-", "-"] and rows["seed3"][1:] == ["10240", "10240", "10240", "10240"]
    assert "solved: full success 1/2 (10240), gap <= 10% 1/2 (10240), strict 1/2 (10240), contact-free 1/2 (10240)" \
        in text
    assert time_to_solve(runs[0]) == {level: None for level in SOLVED_LEVELS}
