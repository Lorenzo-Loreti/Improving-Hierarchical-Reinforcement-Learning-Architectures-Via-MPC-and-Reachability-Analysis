"""Tests for algorithms/metrics_log.py, the local copy of a run's W&B log."""

import json

import numpy as np

from metrics_log import MetricsLog, read_config, read_metrics, series


def test_records_round_trip_with_the_step_and_numpy_scalars_cast(tmp_path):
    log = MetricsLog(str(tmp_path), config={"seed": 3, "total_timesteps": 1024})
    log.log({"eval/episodic_return": np.float32(12.5), "charts/SPS": 900}, step=1024)
    log.log({"solved/is_solved": 1.0}, step=2048)
    log.close()
    assert read_config(str(tmp_path)) == {"seed": 3, "total_timesteps": 1024}
    records = read_metrics(str(tmp_path))
    assert records == [
        {"global_step": 1024, "eval/episodic_return": 12.5, "charts/SPS": 900.0},
        {"global_step": 2048, "solved/is_solved": 1.0},
    ]


def test_each_record_is_on_disk_before_the_log_is_closed(tmp_path):
    """An interrupted run keeps everything it logged."""
    log = MetricsLog(str(tmp_path))
    log.log({"x": 1.0}, step=1)
    lines = open(tmp_path / "metrics.jsonl").read().splitlines()
    assert [json.loads(line) for line in lines] == [{"global_step": 1, "x": 1.0}]
    log.close()


def test_series_keeps_only_the_records_that_carry_the_key():
    records = [{"global_step": 1, "a": 1.0}, {"global_step": 2, "b": 5.0}, {"global_step": 3, "a": 2.0}]
    assert series(records, "a") == ([1, 3], [1.0, 2.0])
    assert series(records, "missing") == ([], [])
