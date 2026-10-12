"""Tests of the run directory and the learning-curve log (code rules C3, C4; decision log D6, D50)."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import numpy as np
import pytest
import yaml

from hrlmpc.utils.runlog import SLURM_VARIABLES, RunLogger, SampleCounter, machine_info, package_versions


def test_counter_accumulates() -> None:
    counter = SampleCounter()
    counter.add_env_steps(8)
    counter.add_env_steps(8)
    counter.add_manager_decisions(1)
    assert (counter.env_steps, counter.manager_decisions) == (16, 1)


def test_counter_rejects_negative_increments() -> None:
    counter = SampleCounter()
    with pytest.raises(ValueError):
        counter.add_env_steps(-1)
    with pytest.raises(ValueError):
        counter.add_manager_decisions(-1)


def test_run_directory_layout(tmp_path: Path) -> None:
    config = {"env": {"physics": {"dt": 0.1}}, "algorithm": {"name": "ppo", "lr": 0.0003}}
    run_dir = tmp_path / "run"
    counter = SampleCounter()
    with RunLogger(run_dir, config, seed=7, repo_dir=tmp_path) as log:
        counter.add_env_steps(8)
        log.log({"return": np.float32(1.5), "success": np.bool_(True), "episodes": np.int64(3)}, counter)
        counter.add_env_steps(8)
        counter.add_manager_decisions(1)
        log.log({"return": float("nan")}, counter)

    assert yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) == config

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 7
    assert meta["git"] == {"commit": None, "dirty": None}  # tmp_path is not a git checkout
    assert meta["packages"]["numpy"] == np.__version__
    assert meta["machine"]["host"] == platform.node()  # D50

    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records == [
        {"env_steps": 8, "manager_decisions": 0, "return": 1.5, "success": True, "episodes": 3},
        {"env_steps": 16, "manager_decisions": 1, "return": None},
    ]


def test_existing_run_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    with pytest.raises(FileExistsError):
        RunLogger(tmp_path / "run", {}, seed=0, repo_dir=tmp_path)


def test_reserved_keys_are_refused(tmp_path: Path) -> None:
    with RunLogger(tmp_path / "run", {}, seed=0, repo_dir=tmp_path) as log:
        with pytest.raises(ValueError, match="reserved"):
            log.log({"env_steps": 3}, SampleCounter())


def test_non_scalar_metric_is_refused(tmp_path: Path) -> None:
    with RunLogger(tmp_path / "run", {}, seed=0, repo_dir=tmp_path) as log:
        with pytest.raises(TypeError, match="unsupported type"):
            log.log({"returns": [1.0, 2.0]}, SampleCounter())


def test_logging_after_close_fails(tmp_path: Path) -> None:
    log = RunLogger(tmp_path / "run", {}, seed=0, repo_dir=tmp_path)
    log.close()
    with pytest.raises(ValueError, match="closed"):
        log.log({"return": 1.0}, SampleCounter())


def test_package_versions_reports_missing_packages() -> None:
    versions = package_versions(["numpy", "surely-not-an-installed-distribution"])
    assert versions["numpy"] == np.__version__
    assert versions["surely-not-an-installed-distribution"] is None


def test_the_machine_is_described(monkeypatch: pytest.MonkeyPatch) -> None:
    """D50: processor, cores, host, the load when the run starts and the threads' variables; the Slurm job only
    inside one."""
    for name in SLURM_VARIABLES.values():
        monkeypatch.delenv(name, raising=False)
    machine = machine_info()
    assert set(machine) == {"host", "processor", "cores", "usable_cores", "load", "threads", "slurm"}
    assert machine["host"] == platform.node() and machine["cores"] == os.cpu_count()
    assert machine["processor"] is None or (isinstance(machine["processor"], str) and machine["processor"])
    assert 1 <= machine["usable_cores"] <= machine["cores"]
    assert machine["load"] is None or (len(machine["load"]) == 3 and min(machine["load"]) >= 0.0)
    assert set(machine["threads"]) == {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
    assert machine["slurm"] is None
    assert json.loads(json.dumps(machine)) == machine  # plain JSON
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "cpu")
    monkeypatch.setenv("SLURMD_NODENAME", "wolfgang-cpu03")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    machine = machine_info()
    assert machine["slurm"] == {
        "job_id": "4242", "array_job_id": None, "array_task_id": None, "partition": "cpu",
        "node": "wolfgang-cpu03", "cpus_per_task": None,
    }
    assert machine["threads"]["OMP_NUM_THREADS"] == "1"

