"""Run directories and learning-curve logging (code rules C3 and C4, decision log D6).

Every training run writes one directory:

    config.yaml      the resolved configuration of the run
    meta.json        provenance: seed, time, Python and package versions, git commit
    metrics.jsonl    one JSON object per call of :meth:`RunLogger.log`

Each metrics record carries the two interaction counters of decision D6.
``env_steps`` counts the calls of the environment's physical step during
training, summed over the parallel environments: it is the sample unit and
the x-axis of every learning curve. ``manager_decisions`` is a secondary
counter for the hierarchical architectures.
"""

from __future__ import annotations

import datetime
import json
import math
import numbers
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

import numpy as np
import yaml

TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "PyYAML",
    "torch",
    "gymnasium",
    "cvxpy",
    "clarabel",
    "gurobipy",
    "matplotlib",
)
"""Packages whose versions are recorded in ``meta.json``."""

RESERVED_KEYS: tuple[str, ...] = ("env_steps", "manager_decisions")
"""Keys written by the logger itself; metrics may not use them."""


@dataclass
class SampleCounter:
    """Interaction counters of one training run (decision log D6).

    Attributes:
        env_steps: Calls of the environment's physical step during training,
            summed over the parallel environments.
        manager_decisions: Decisions taken by the Manager of a hierarchical
            architecture; always zero for flat PPO.
    """

    env_steps: int = 0
    manager_decisions: int = 0

    def add_env_steps(self, n: int) -> None:
        """Count ``n`` more environment steps.

        Raises:
            ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError(f"cannot add a negative number of environment steps: {n}")
        self.env_steps += n

    def add_manager_decisions(self, n: int) -> None:
        """Count ``n`` more Manager decisions.

        Raises:
            ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError(f"cannot add a negative number of Manager decisions: {n}")
        self.manager_decisions += n


def package_versions(names: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed versions of the given distributions, ``None`` for missing ones."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git(args: list[str], repo_dir: Path) -> str | None:
    """Output of a git command in ``repo_dir``, or ``None`` if it fails."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_state(repo_dir: str | Path | None = None) -> dict[str, Any]:
    """Commit and dirty flag of the git checkout containing ``repo_dir``.

    Args:
        repo_dir: A directory inside the checkout; the current directory if
            ``None``.

    Returns:
        ``{"commit": <hash or None>, "dirty": <bool or None>}``. ``dirty``
        refers to tracked files only. Both are ``None`` outside a git checkout.
    """
    directory = Path.cwd() if repo_dir is None else Path(repo_dir)
    commit = _git(["rev-parse", "HEAD"], directory)
    if commit is None:
        return {"commit": None, "dirty": None}
    status = _git(["status", "--porcelain", "--untracked-files=no"], directory)
    return {"commit": commit, "dirty": None if status is None else status != ""}


def run_metadata(seed: int, repo_dir: str | Path | None = None) -> dict[str, Any]:
    """Provenance of a run: seed, UTC start time, platform, package versions and git state."""
    return {
        "seed": seed,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "git": git_state(repo_dir),
    }


def _jsonable(value: Any, key: str) -> bool | int | float | str | None:
    """Convert a metric value to a JSON scalar; non-finite floats become ``None``."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    raise TypeError(f"metric {key!r} has unsupported type {type(value).__name__}")


class RunLogger:
    """Writes the directory of one training run.

    Args:
        run_dir: Directory of the run. It must not exist yet, so that two
            runs can never be mixed in one directory.
        config: Resolved configuration of the run, as plain YAML values
            (e.g. the result of :func:`hrlmpc.config.env_config_to_dict`
            together with the algorithm's settings).
        seed: Seed of the run.
        repo_dir: Directory inside the git checkout of the code; the current
            directory if ``None``.

    Raises:
        FileExistsError: If ``run_dir`` already exists.
    """

    def __init__(
        self,
        run_dir: str | Path,
        config: Mapping[str, Any],
        *,
        seed: int,
        repo_dir: str | Path | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        with (self.run_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(dict(config), f, sort_keys=False)
        with (self.run_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(run_metadata(seed, repo_dir), f, indent=2)
        self._metrics: TextIO | None = (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8")

    def log(self, metrics: Mapping[str, Any], counter: SampleCounter) -> None:
        """Append one record with the counters and the given metrics.

        Args:
            metrics: Scalar metrics. NumPy scalars are converted; non-finite
                floats are written as ``null``.
            counter: Interaction counters of the run.

        Raises:
            ValueError: If a metric uses a reserved key or the logger is closed.
            TypeError: If a metric is not a scalar.
        """
        if self._metrics is None:
            raise ValueError("the logger is closed")
        clash = [k for k in metrics if k in RESERVED_KEYS]
        if clash:
            raise ValueError(f"metric keys {clash} are reserved for the sample counters")
        record: dict[str, Any] = {
            "env_steps": counter.env_steps,
            "manager_decisions": counter.manager_decisions,
        }
        for key, value in metrics.items():
            record[key] = _jsonable(value, key)
        self._metrics.write(json.dumps(record) + "\n")
        self._metrics.flush()

    def close(self) -> None:
        """Close the metrics file; further calls of :meth:`log` fail."""
        if self._metrics is not None:
            self._metrics.close()
            self._metrics = None

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
