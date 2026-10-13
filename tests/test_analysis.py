"""Tests of the run analysis: learning curves over seeds with confidence intervals (code rule C4, decision log D20,
D43, D50)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats

from hrlmpc.analysis import (
    CURVE_KEYS,
    accepted,
    curve_keys,
    find_runs,
    first_accepted,
    group_runs,
    load_run,
    main,
    mean_curve,
    series,
    summarize,
    t_interval,
)
from hrlmpc.utils.runlog import RunLogger, SampleCounter

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
) -> Path:
    run_dir = directory / (name or f"seed{seed}")
    config = {"algorithm": "ppo", "seed": seed, "env_file": "configs/env/tunnel.yaml", "env": env or TUNNEL}
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
    """The rows of the runs in the table of ``action``."""
    mean, _, mode = text.partition("action: mode")
    table = mean if action == "mean" else mode
    return {line.split()[0]: line.split() for line in table.splitlines() if line.startswith("seed")}


def test_the_summary_reports_every_run(three_runs: list[Path]) -> None:
    text = summarize([load_run(p) for p in three_runs], max_gap=0.05)
    rows = _rows(text)
    assert [rows[f"seed{s}"][7] for s in (1, 2, 3)] == ["yes", "yes", "no"]
    assert [rows[f"seed{s}"][8] for s in (1, 2, 3)] == ["20480", "10240", "-"]
    assert "== ppo tunnel (3 runs)" in text and "accepted at the end: 2/3 (max gap 0.05)" in text
    strict = summarize([load_run(p) for p in three_runs], max_gap=0.025)
    assert [_rows(strict)[f"seed{s}"][7] for s in (1, 2, 3)] == ["no", "yes", "no"]
    assert "accepted at the end: 1/3" in strict


def test_acceptance_does_not_apply_with_obstacles(tmp_path: Path) -> None:
    """With obstacles F7 is only a lower bound, so the gap criterion of D20 is not applied."""
    run = load_run(_make_run(tmp_path, 1, [_eval(1.0, 0.01, 0.0, 900.0)], env=SLALOM))
    text = summarize([run])
    assert _rows(text)["seed1"][7] == "n/a" and "acceptance not applicable" in text


def test_the_command_line_prints_and_plots(
    three_runs: list[Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("matplotlib")
    figure = tmp_path / "curves.png"
    assert main([str(tmp_path), "--plot", str(figure)]) == 0
    assert "accepted at the end: 2/3" in capsys.readouterr().out
    assert figure.stat().st_size > 0


def test_the_summary_reports_the_mode_beside_the_mean(tmp_path: Path) -> None:
    """D43, D50: a table for each action; the criterion of D20 stays on the mean."""
    run = load_run(_make_run(tmp_path, 1, [_eval(1.0, 0.04, 0.0, 905.0, mode=(0.75, 0.01, 0.25, 640.0))]))
    text = summarize([run], max_gap=0.05)
    assert _rows(text)["seed1"][2:8] == ["1.00", "905.0", "0.020", "0.040", "0.00", "yes"]
    assert _rows(text, "mode")["seed1"][2:7] == ["0.75", "640.0", "0.005", "0.010", "0.25"]
    assert "accepted at the end: 1/1 (max gap 0.05)" in text  # on the mean, whatever the mode did
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

