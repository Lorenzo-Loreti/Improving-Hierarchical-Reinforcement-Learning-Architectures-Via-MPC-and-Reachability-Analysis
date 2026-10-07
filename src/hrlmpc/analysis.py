"""Learning curves and summaries of training runs (code rule C4 and the analysis rules; decision log D6, D20).

Reads the run directories written by :class:`hrlmpc.utils.runlog.RunLogger`
and aggregates them over seeds: means with Student-t confidence intervals,
against the sample counter ``env_steps`` (D6). Runs are pooled only within a
group of runs that share their whole configuration except the seed.

It also checks the acceptance criterion of step 7 (D20): success from every
start of the evaluation grid, arrival within a relative gap of the bound of
F7, and no contact. The criterion applies to layouts without obstacles only,
where the bound of F7 is the minimum time; with obstacles the gap is reported
but measures the distance from a lower bound, not from the optimum.

Usage::

    python -m hrlmpc.analysis runs/tunnel/ppo runs/slalom/ppo [--plot curves.png]

The statistical comparison between architectures belongs to step 11 (Q13).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]

CURVE_KEYS: tuple[str, ...] = (
    "eval/success_rate",
    "eval/unshaped_return",
    "eval/arrival_gap_mean",
    "eval/contact_fraction",
)
"""Evaluation metrics plotted by the command line."""

_FRACTIONS = {"eval/success_rate", "eval/contact_fraction"}
_PROVENANCE = {"seed", "env_file", "agent_file"}


@dataclass(frozen=True)
class Run:
    """A training run read back from its directory."""

    path: Path
    config: dict[str, Any]
    meta: dict[str, Any]
    records: tuple[dict[str, Any], ...]

    @property
    def name(self) -> str:
        """Name of the run directory."""
        return self.path.name

    @property
    def seed(self) -> Any:
        """Training seed, from the configuration or else from the provenance."""
        return self.config.get("seed", self.meta.get("seed"))

    @property
    def has_obstacles(self) -> bool:
        """Whether the layout of the run has obstacles (unknown layouts count as having them)."""
        obstacles = self.config.get("env", {}).get("geometry", {}).get("obstacles")
        return bool(obstacles != [])


def load_run(path: str | Path) -> Run:
    """Read ``config.yaml``, ``meta.json`` and ``metrics.jsonl`` of one run directory."""
    run_dir = Path(path)
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    records = tuple(json.loads(line) for line in lines if line.strip())
    return Run(run_dir, config, meta, records)


def find_runs(paths: Iterable[str | Path]) -> list[Run]:
    """All runs at or below the given directories, each once, sorted by path.

    Raises:
        ValueError: If no run is found.
    """
    roots = [Path(p) for p in paths]
    found = {p.parent.resolve() for root in roots for p in root.rglob("metrics.jsonl")}
    if not found:
        raise ValueError(f"no run directory (with a metrics.jsonl) below {[str(p) for p in roots]}")
    return [load_run(p) for p in sorted(found)]


def _fingerprint(config: Mapping[str, Any]) -> str:
    shared = {k: v for k, v in config.items() if k not in _PROVENANCE}
    return hashlib.sha256(json.dumps(shared, sort_keys=True, default=str).encode()).hexdigest()[:8]


def group_runs(runs: Sequence[Run]) -> dict[str, list[Run]]:
    """Runs grouped by their configuration without the seed and the source files.

    The label of a group names the algorithm and the environment file; a
    short hash of the configuration tells apart groups that share them.

    Raises:
        ValueError: If a seed appears twice in a group.
    """
    by_print: dict[str, list[Run]] = {}
    for run in runs:
        by_print.setdefault(_fingerprint(run.config), []).append(run)
    labels: dict[str, str] = {}
    for fingerprint, members in by_print.items():
        config = members[0].config
        env_file = config.get("env_file")
        labels[fingerprint] = f"{config.get('algorithm', 'run')} {Path(env_file).stem if env_file else 'env'}"
    groups: dict[str, list[Run]] = {}
    for fingerprint, members in by_print.items():
        label = labels[fingerprint]
        if list(labels.values()).count(label) > 1:
            label = f"{label} #{fingerprint}"
        seeds = [run.seed for run in members]
        if len(set(map(str, seeds))) < len(seeds):
            raise ValueError(f"group {label!r} contains a seed twice: {sorted(map(str, seeds))}")
        groups[label] = members
    return groups


def series(run: Run, key: str) -> tuple[NDArray[np.int64], FloatArray]:
    """``env_steps`` and values of the records that contain ``key``; ``null`` becomes NaN."""
    rows = [r for r in run.records if key in r]
    steps = np.array([r["env_steps"] for r in rows], dtype=np.int64)
    values = np.array([math.nan if r[key] is None else r[key] for r in rows], dtype=np.float64)
    return steps, values


def t_interval(values: FloatArray, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and two-sided Student-t confidence interval of independent values (NaN bounds for one value).

    Raises:
        ValueError: If ``confidence`` is not in ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean())
    if x.size < 2:
        return mean, math.nan, math.nan
    half = float(stats.t.ppf(0.5 + confidence / 2.0, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size))
    return mean, mean - half, mean + half


@dataclass(frozen=True)
class Curve:
    """Mean over runs of one metric, with a pointwise confidence interval.

    Attributes:
        env_steps: Evaluation points.
        mean: Mean over the runs with a value at each point.
        low: Lower end of the interval.
        high: Upper end of the interval.
        count: Number of runs with a value at each point; NaN values (e.g.
            the arrival gap when no episode succeeded) are left out.
        n: Number of runs.
    """

    env_steps: NDArray[np.int64]
    mean: FloatArray
    low: FloatArray
    high: FloatArray
    count: NDArray[np.int64]
    n: int


def mean_curve(runs: Sequence[Run], key: str, confidence: float = 0.95) -> Curve:
    """Pointwise mean and Student-t interval over runs.

    Raises:
        ValueError: If the runs do not share their evaluation points.
    """
    columns = [series(run, key) for run in runs]
    steps = columns[0][0]
    if any(not np.array_equal(s, steps) for s, _ in columns):
        raise ValueError(f"the runs do not share their evaluation points of {key!r}")
    table = np.stack([v for _, v in columns])
    mean, low, high = (np.full(len(steps), math.nan) for _ in range(3))
    count = np.sum(~np.isnan(table), axis=0).astype(np.int64)
    for i in range(len(steps)):
        column = table[:, i][~np.isnan(table[:, i])]
        if column.size:
            mean[i], low[i], high[i] = t_interval(column, confidence)
    return Curve(steps, mean, low, high, count, len(runs))


def accepted(record: Mapping[str, Any], max_gap: float) -> bool:
    """Acceptance criterion of step 7 (D20) on one evaluation record."""
    keys = ("eval/success_rate", "eval/arrival_gap_max", "eval/contact_fraction")
    success, gap, contact = (record.get(k) for k in keys)
    if success is None or gap is None or contact is None:
        return False
    return bool(success == 1.0 and gap <= max_gap and contact == 0.0)


def first_accepted(run: Run, max_gap: float) -> int | None:
    """``env_steps`` of the first evaluation that meets the criterion, or ``None``."""
    for record in run.records:
        if "eval/success_rate" in record and accepted(record, max_gap):
            return int(record["env_steps"])
    return None


def _last_evaluation(run: Run) -> dict[str, Any]:
    evaluations = [r for r in run.records if "eval/success_rate" in r]
    if not evaluations:
        raise ValueError(f"run {run.path} has no evaluation")
    return evaluations[-1]


def _fmt(value: Any, digits: int = 3) -> str:
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def _summarize_group(label: str, runs: Sequence[Run], max_gap: float) -> list[str]:
    applicable = not any(run.has_obstacles for run in runs)
    header = (
        f"{'run':<28} {'samples':>8} {'success':>8} {'return':>9} {'gap mean':>9} {'gap max':>8}"
        f" {'contact':>8} {'accepted':>8} {'first':>8}"
    )
    lines = [f"== {label} ({len(runs)} runs)", header, "-" * len(header)]
    finals = [_last_evaluation(run) for run in runs]
    for run, final in zip(runs, finals, strict=True):
        first = first_accepted(run, max_gap) if applicable else None
        verdict = ("yes" if accepted(final, max_gap) else "no") if applicable else "n/a"
        lines.append(
            f"{run.name:<28} {final['env_steps']:>8} {_fmt(final['eval/success_rate'], 2):>8} "
            f"{_fmt(final['eval/unshaped_return'], 1):>9} {_fmt(final['eval/arrival_gap_mean']):>9} "
            f"{_fmt(final['eval/arrival_gap_max']):>8} {_fmt(final['eval/contact_fraction'], 2):>8} "
            f"{verdict:>8} {'-' if first is None else first:>8}"
        )
    lines.append("")
    for key in CURVE_KEYS:
        values = np.array([math.nan if f[key] is None else f[key] for f in finals], dtype=np.float64)
        values = values[~np.isnan(values)]
        if values.size:
            mean, low, high = t_interval(values)
            lines.append(f"{key:<24} mean {mean:.3f}  95% CI [{_fmt(low)}, {_fmt(high)}]  (n={values.size})")
    if applicable:
        count = sum(accepted(f, max_gap) for f in finals)
        lines.append(f"accepted at the end: {count}/{len(runs)} (max gap {max_gap:g})")
    else:
        lines.append("acceptance not applicable: the layout has obstacles, so the bound of F7 is not the minimum time")
    return lines


def summarize(runs: Sequence[Run], max_gap: float = 0.05) -> str:
    """For every group: the last evaluation of each run, means over runs and the acceptance status (D20)."""
    lines: list[str] = []
    for label, members in group_runs(runs).items():
        lines += _summarize_group(label, members, max_gap)
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def plot_curves(runs: Sequence[Run], path: str | Path, keys: Sequence[str] = CURVE_KEYS) -> None:
    """Plot the mean curves of every group with their 95% intervals into an image file (needs matplotlib).

    Points where some runs have no value (NaN) are drawn dashed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = group_runs(runs)
    fig, axes = plt.subplots(len(keys), 1, figsize=(6.5, 2.3 * len(keys)), sharex=True, squeeze=False)
    for ax, key in zip(axes[:, 0], keys, strict=True):
        for color, (label, members) in enumerate(groups.items()):
            curve = mean_curve(members, key)
            low, high = curve.low, curve.high
            if key in _FRACTIONS:
                low, high = np.clip(low, 0.0, 1.0), np.clip(high, 0.0, 1.0)
            full = curve.count == curve.n
            ax.plot(curve.env_steps, np.where(full, curve.mean, np.nan), color=f"C{color}", lw=1.5, label=label)
            ax.plot(curve.env_steps, np.where(full, np.nan, curve.mean), color=f"C{color}", lw=1.0, ls="--")
            ax.fill_between(curve.env_steps, low, high, color=f"C{color}", alpha=0.2, lw=0)
        ax.set_ylabel(key.removeprefix("eval/").replace("_", " "))
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("samples (environment steps, D6)")
    axes[0, 0].set_title("mean and 95% t-interval over seeds; dashed: some runs without a value")
    axes[0, 0].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: print the summary of the runs below the given directories, optionally plot them."""
    parser = argparse.ArgumentParser(description="Summarize training runs: final evaluation and learning curves.")
    parser.add_argument("paths", nargs="+", type=Path, help="run directories or directories containing runs")
    parser.add_argument("--max-gap", type=float, default=0.05, help="largest accepted arrival gap (D20)")
    parser.add_argument("--plot", type=Path, default=None, help="image file for the learning curves")
    args = parser.parse_args(argv)
    runs = find_runs(args.paths)
    print(summarize(runs, args.max_gap))
    if args.plot is not None:
        plot_curves(runs, args.plot)
        print(f"curves written to {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
