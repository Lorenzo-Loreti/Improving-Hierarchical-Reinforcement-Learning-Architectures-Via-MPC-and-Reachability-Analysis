"""The results of the experiment matrix for the thesis: cells, statistics, tables and figures (decision log D64,
D67, D68, D71–D75).

Every number and every figure of the results chapter comes from the run
directories through this module, never from values typed by hand: the author
regenerates them with::

    python -m hrlmpc.results runs/final runs/final_h6 runs/final_h4 runs/final_map --out ../6_THESIS/figures/results

which writes ``summary.json`` (every number), ``report.md`` (the numbers,
readable), ``tables/*.tex`` (``tabular`` bodies to ``\\input``) and
``figures/*.pdf`` (with ``.png`` previews) under the output directory.

Cells. A run belongs to the cell (layout, setting, disturbance level, variant,
architecture) read from its configuration (:func:`classify`): the layout
from the name of its environment file, the setting from the reward and the
curriculum of its environment section (the dense reward of D71; the
reverse spawn curriculum of D62 marks the main setting; neither marks the
exploration setting of D63), the disturbance level from the physics, the
variant from the name of its agent file (the agility sweep of D60,
``_h6``/``_h4``; the map-aware hPPO Worker of D75, ``_map``).

Times. The time to a level of "solved" (D67) is the sample counter at the
first evaluation at which the mode of the policy met it (D57), censored at
the budget when the run never did. Groups of such times are summarised by
the Kaplan–Meier estimate of the fraction solved and its median, compared by
the two-sided log-rank test and, for the counts of solving seeds, by Fisher's
exact test; means over seeds carry Student-t 95 % intervals (chapter 5 of
the thesis).

Figures. The curves show the mean over seeds with its 95 % band; bars show
the median time to full success over the solving seeds with min–max
whiskers, hatched at the budget where no seed solved and labelled ``k/n``
where some did; one colour and one line style per architecture everywhere
(a palette checked for colour-vision deficiencies). Matplotlib is imported
only when figures are drawn.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from hrlmpc.analysis import Run, find_runs, mean_curve, series, success_auc, t_interval, time_to_solve
from hrlmpc.reevaluate import read_reevaluations

FloatArray = NDArray[np.float64]

ARCHS: tuple[str, ...] = ("ppo", "hppo", "ppo_mpc", "ppo_mpc_reachability")
"""The architectures, in the order of the thesis."""

LABELS = {
    "ppo": "PPO",
    "hppo": "hPPO",
    "ppo_mpc": "PPO$_{\\mathrm{MPC}}$",
    "ppo_mpc_reachability": "PPO$_{\\mathrm{MPC,reach}}$",
}
"""The architectures' names in the figures (matplotlib's mathtext)."""

LABELS_TEX = {
    "ppo": "PPO",
    "hppo": "hPPO",
    "ppo_mpc": "PPO\\textsubscript{MPC}",
    "ppo_mpc_reachability": "PPO\\textsubscript{MPC,reach}",
}
"""The architectures' names in the tables, as the thesis writes them in text mode."""

LABELS_TXT = {"ppo": "PPO", "hppo": "hPPO", "ppo_mpc": "PPO_MPC", "ppo_mpc_reachability": "PPO_MPC_reach"}
"""The architectures' names in plain text (the report)."""

VARIANT_LABELS = {"": "", "map": " (map)", "h6": " ($H = 6$)", "h4": " ($H = 4$)", "b2": " (twice the budget)"}

COLORS = {"ppo": "#2a78d6", "hppo": "#c98500", "ppo_mpc": "#008300", "ppo_mpc_reachability": "#e34948"}
"""One colour per architecture: blue, amber, green, red (validated for colour-vision deficiencies)."""

STYLES = {"ppo": "-", "hppo": "--", "ppo_mpc": "-.", "ppo_mpc_reachability": ":"}
MARKERS = {"ppo": "o", "hppo": "s", "ppo_mpc": "^", "ppo_mpc_reachability": "D"}

SETTINGS: tuple[str, ...] = ("main", "exploration", "dense")
SETTING_LABELS = {"main": "main (curriculum)", "exploration": "exploration", "dense": "dense reward"}
LAYOUTS: tuple[str, ...] = ("slalom", "trap", "tunnel")
LEVELS: tuple[str, ...] = ("success", "gap_10", "strict", "contact_free")

BUDGET = 204_800
"""The budget of the final matrix (D64), in samples; the censoring time of the times to solve."""

TEXT_WIDTH_IN = 10.55 / 2.54
"""The width of the text block of the thesis (A5), in inches."""

_MPC_COUNTS = (
    "emergency_count",
    "solver_error_count",
    "exit_converged_count",
    "exit_fixed_count",
    "exit_relaxation_count",
    "exit_best_count",
    "exit_rounds_count",
    "exit_empty_count",
    "exit_unsolved_count",
    "exit_error_count",
    "exact_accepted_count",
    "best_applied_count",
    "accepted_applied_count",
    "invalid_candidate_count",
)
_REACH_COUNTS = ("hull_fallback_count", "projection_failure_count", "support_failure_count", "projection_error_count")


# --------------------------------------------------------------------------- cells


@dataclass(frozen=True, order=True)
class Cell:
    """A cell of the experiment matrix.

    Attributes:
        layout: ``slalom``, ``trap`` or ``tunnel`` (the first word of the environment file's name).
        setting: ``main`` (the sparse reward with the curriculum), ``exploration`` (without), ``dense``.
        d_bar: The radius of the disturbance set.
        variant: ``""`` for the shipped agent file, ``h6``/``h4`` for the sweep, ``map`` for the ablation.
        arch: The algorithm.
    """

    layout: str
    setting: str
    d_bar: float
    variant: str
    arch: str

    @property
    def key(self) -> str:
        """The cell as one string, for JSON."""
        level = f"{self.d_bar:g}"
        return "/".join([self.layout, self.setting, f"d{level}", self.variant or "base", self.arch])

    @property
    def base(self) -> bool:
        """Whether the cell belongs to the matrix proper: no disturbance, the shipped agent files."""
        return self.d_bar == 0.0 and self.variant == ""


def classify(run: Run) -> Cell:
    """The cell of a run, from its configuration (see the module docstring).

    Raises:
        ValueError: If the configuration lacks the environment file, the algorithm or the environment section.
    """
    config = run.config
    try:
        env_file, algorithm, env = str(config["env_file"]), str(config["algorithm"]), config["env"]
    except KeyError as error:
        raise ValueError(f"run {run.path} lacks {error} in its config.yaml") from error
    layout = PureWindowsPath(env_file).stem.split("_")[0]
    dense = float(env["reward"].get("progress_coef", 0.0)) > 0.0
    curriculum = env["task"].get("curriculum") is not None
    setting = "dense" if dense else ("main" if curriculum else "exploration")
    d_bar = float(env["physics"].get("d_bar", 0.0))
    agent_file = config.get("agent_file")
    stem = PureWindowsPath(str(agent_file)).stem if agent_file else algorithm
    variant = stem[len(algorithm) + 1 :] if stem.startswith(algorithm + "_") else ""
    budget = run.budget
    if budget is not None and budget != BUDGET:  # a budget other than the matrix's is a variant of its own (D79)
        variant = (variant + "_" if variant else "") + f"b{budget / BUDGET:g}"
    return Cell(layout, setting, d_bar, variant, algorithm)


def cells_of(runs: Iterable[Run]) -> dict[Cell, list[Run]]:
    """The runs grouped by cell, each group sorted by seed.

    Raises:
        ValueError: If a seed appears twice in a cell.
    """
    groups: dict[Cell, list[Run]] = defaultdict(list)
    for run in runs:
        groups[classify(run)].append(run)
    for cell, members in groups.items():
        members.sort(key=lambda r: int(r.seed))
        seeds = [int(r.seed) for r in members]
        if len(set(seeds)) < len(seeds):
            raise ValueError(f"cell {cell.key} holds a seed twice: {seeds}")
    return dict(sorted(groups.items()))


# --------------------------------------------------------------------------- per-run metrics


def _evaluations(run: Run) -> list[dict[str, Any]]:
    return [r for r in run.records if "eval/success_rate" in r]


def _updates(run: Run) -> list[dict[str, Any]]:
    return [r for r in run.records if "eval/success_rate" not in r]


def _total(records: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(sum(float(r.get(key) or 0.0) for r in records))


def _nanmean(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = np.array([math.nan if r.get(key) is None else float(r[key]) for r in records], dtype=np.float64)
    good = values[~np.isnan(values)]
    return float(good.mean()) if good.size else None


def _weighted(records: Sequence[Mapping[str, Any]], key: str, weight: str) -> float | None:
    """The mean of ``key`` over the records weighted by ``weight`` (the episodes of each update)."""
    w = np.array([float(r.get(weight) or 0.0) for r in records], dtype=np.float64)
    v = np.array([math.nan if r.get(key) is None else float(r[key]) for r in records], dtype=np.float64)
    good = ~np.isnan(v) & (w > 0)
    return float((v[good] * w[good]).sum() / w[good].sum()) if good.any() else None


def run_metrics(run: Run) -> dict[str, Any]:
    """The numbers of one run: the times to each level of D67 (mode, oracle gap), the area under the success
    curve, the final policy's success, gaps, contact and return, the training contact, the curriculum, the
    hierarchy's targets, the MPC Worker's and the reachable targets' counters, the compute.

    Raises:
        ValueError: If the run has no evaluation.
    """
    evaluations = _evaluations(run)
    if not evaluations:
        raise ValueError(f"run {run.path} has no evaluation")
    last, ups = evaluations[-1], _updates(run)
    times = time_to_solve(run, action="mode", max_gap=0.05, oracle=True)
    m: dict[str, Any] = {
        "seed": int(run.seed),
        "run": run.name,
        "complete": run.complete,
        "budget": BUDGET if run.budget is None else int(run.budget),
        "horizon": int(run.config["env"]["task"]["horizon"]),
        "t_success": times["success"],
        "t_gap10": times["gap_10"],
        "t_strict": times["strict"],
        "t_contact_free": times["contact_free"],
        "auc": success_auc(run, "mode"),
        "final_success": last.get("eval_mode/success_rate"),
        "final_gap_max": last.get("eval_mode/oracle_gap_max"),
        "final_gap_mean": last.get("eval_mode/oracle_gap_mean"),
        "final_contact_fraction": last.get("eval_mode/contact_fraction"),
        "final_return": last.get("eval_mode/unshaped_return"),
        "final_length": last.get("eval_mode/length"),
        "elapsed_s": ups[-1].get("time/elapsed_s") if ups else None,
        "sps": _nanmean(ups, "time/sps"),
        "train_contact_fraction": _weighted(ups, "rollout/contact_fraction", "rollout/episodes"),
        "train_impulse_per_episode": _weighted(ups, "rollout/impulse", "rollout/episodes"),
    }
    staged = [r for r in ups if "curriculum/stage" in r]
    if staged:
        m["curriculum_stage"] = int(staged[-1]["curriculum/stage"])
        m["curriculum_stages"] = int(staged[-1]["curriculum/stages"])
        m["curriculum_last_stage_at"] = next(
            (int(r["env_steps"]) for r in staged if r.get("curriculum/final") == 1.0), None
        )
    if any("hierarchy/reached_rate" in r for r in ups):
        tail = [r for r in ups if "hierarchy/reached_rate" in r][-10:]
        m["reached_rate_train_tail"] = _nanmean(tail, "hierarchy/reached_rate")
        m["reached_rate_eval_final"] = last.get("eval_mode_hierarchy/reached_rate")
    if any("hierarchy/target_distance" in r for r in ups):
        tail = [r for r in ups if "hierarchy/target_distance" in r][-10:]
        m["target_distance_train_tail"] = _nanmean(tail, "hierarchy/target_distance")
    if "eval_mode_hierarchy/end_distance_mean" in last:
        m["end_distance_eval_final"] = last.get("eval_mode_hierarchy/end_distance_mean")
    if any("mpc/steps" in r for r in ups):
        m["mpc_steps"] = _total(ups, "mpc/steps")
        for key in _MPC_COUNTS:
            m["mpc_" + key] = _total(ups, "mpc/" + key)
        m["mpc_candidate_rate"] = _nanmean(ups, "mpc/candidate_rate")
        m["mpc_kept_rate"] = _nanmean(ups, "mpc/kept_rate")
        m["mpc_solve_ms_mean"] = _nanmean(ups, "mpc/solve_ms_mean")
        maxima = [float(r["mpc/solve_ms_max"]) for r in ups if r.get("mpc/solve_ms_max") is not None]
        m["mpc_solve_ms_max"] = max(maxima) if maxima else None
        m["mpc_cut_rounds_mean"] = _nanmean(ups, "mpc/cut_rounds_mean")
        m["eval_mpc_solve_ms_mean"] = last.get("eval_mode_mpc/solve_ms_mean")
    if any("reach/decisions" in r for r in ups):
        m["reach_decisions"] = _total(ups, "reach/decisions")
        for key in _REACH_COUNTS:
            m["reach_" + key] = _total(ups, "reach/" + key)
        for key in ("projected_rate", "project_ms_mean", "support_ms_mean", "hull_area_mean"):
            m["reach_" + key] = _nanmean(ups, "reach/" + key)
    return m


# --------------------------------------------------------------------------- statistics


def km(times: Sequence[int | None], budget: int = BUDGET) -> tuple[FloatArray, FloatArray, float | None]:
    """The Kaplan–Meier estimate of the fraction solved by the samples: the steps of the estimate, the fraction
    solved after each, and the median time (the first step at which the fraction reaches one half, ``None`` if
    it never does). Times are censored at ``budget``.

    Raises:
        ValueError: If a time is negative or beyond the budget.
    """
    t = np.array([budget if x is None else x for x in times], dtype=np.float64)
    if np.any(t < 0.0) or np.any(t > budget):
        raise ValueError(f"every time must lie in [0, {budget}], got {list(times)}")
    e = np.array([x is not None for x in times])
    survival, xs, ys = 1.0, [0.0], [0.0]
    for ti in np.unique(t):
        at_risk = int(np.sum(t >= ti))
        solved_now = int(np.sum((t == ti) & e))
        if solved_now:
            survival *= 1.0 - solved_now / at_risk
            xs.append(float(ti))
            ys.append(1.0 - survival)
    xs.append(float(budget))
    ys.append(ys[-1])
    median = next((x for x, y in zip(xs, ys, strict=True) if y >= 0.5), None)
    return np.array(xs), np.array(ys), median


def fisher(solved_a: int, n_a: int, solved_b: int, n_b: int) -> float:
    """The two-sided p-value of Fisher's exact test on the counts of solving seeds."""
    return float(stats.fisher_exact([[solved_a, n_a - solved_a], [solved_b, n_b - solved_b]])[1])


def logrank(
    times_a: Sequence[int | None],
    times_b: Sequence[int | None],
    budget: int = BUDGET,
    budget_b: int | None = None,
) -> float | None:
    """The two-sided p-value of the log-rank test between two groups of times to solve, censored at ``budget``
    (group b at ``budget_b`` when given); ``None`` when no seed of either group solved (the test is undefined),
    1 when every time of both groups is the same (the statistic is 0/0: no evidence of a difference).

    Raises:
        ValueError: If a time lies beyond its group's budget.
    """

    def censored(times: Sequence[int | None], at: int) -> Any:
        uncensored = np.array([x for x in times if x is not None], dtype=np.float64)
        if np.any(uncensored > at):
            raise ValueError(f"every time must lie within its budget {at}, got {list(times)}")
        right = np.array([at for x in times if x is None], dtype=np.float64)
        return stats.CensoredData(uncensored=uncensored, right=right)

    if all(x is None for x in times_a) and all(x is None for x in times_b):
        return None
    with np.errstate(invalid="ignore"):
        p = float(
            stats.logrank(censored(times_a, budget), censored(times_b, budget if budget_b is None else budget_b)).pvalue
        )
    return 1.0 if math.isnan(p) else p  # NaN when every time of both groups coincides: no evidence of a difference


def ci(values: Iterable[float | None]) -> dict[str, Any]:
    """The mean and the Student-t 95 % interval of the values that are not ``None`` or NaN, with their count."""
    x = np.array(
        [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=np.float64
    )
    if x.size == 0:
        return {"mean": None, "low": None, "high": None, "n": 0}
    mean, low, high = t_interval(x)
    return {
        "mean": float(mean),
        "low": None if math.isnan(low) else float(low),
        "high": None if math.isnan(high) else float(high),
        "n": int(x.size),
    }


def group_stats(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The statistics of one cell from its runs' metrics: for each level the times, the count of solving seeds,
    the Kaplan–Meier median and the range; means with intervals of the rest; the Worker's and the reachable
    targets' counters summed over the runs, with their rates."""
    g: dict[str, Any] = {
        "n": len(metrics),
        "seeds": [int(m["seed"]) for m in metrics],
        "complete": all(bool(m["complete"]) for m in metrics),
    }
    budget = max(int(m["budget"]) for m in metrics)
    g["budget"] = budget
    for level in ("t_success", "t_gap10", "t_strict", "t_contact_free"):
        ts = [m[level] for m in metrics]
        _, _, median = km(ts, budget)
        done = [t for t in ts if t is not None]
        g[level] = {
            "times": ts,
            "solved": len(done),
            "km_median": median,
            "min": min(done) if done else None,
            "max": max(done) if done else None,
        }
    for key in (
        "auc",
        "final_success",
        "final_contact_fraction",
        "final_return",
        "train_contact_fraction",
        "train_impulse_per_episode",
        "sps",
    ):
        g[key] = ci([m.get(key) for m in metrics])
    g["final_gap_max_solved"] = ci([m["final_gap_max"] for m in metrics if m["final_success"] == 1.0])
    g["final_gap_mean_solved"] = ci([m["final_gap_mean"] for m in metrics if m["final_success"] == 1.0])
    g["elapsed_min"] = ci([None if m["elapsed_s"] is None else m["elapsed_s"] / 60.0 for m in metrics])
    if "curriculum_stage" in metrics[0]:
        g["curriculum_final_stage"] = [m["curriculum_stage"] for m in metrics]
        g["curriculum_stages"] = int(metrics[0]["curriculum_stages"])
        g["curriculum_last_stage_at"] = [m["curriculum_last_stage_at"] for m in metrics]
    for key in (
        "reached_rate_eval_final",
        "reached_rate_train_tail",
        "target_distance_train_tail",
        "end_distance_eval_final",
    ):
        if key in metrics[0]:
            g[key] = ci([m.get(key) for m in metrics])
    if "mpc_steps" in metrics[0]:
        steps = sum(float(m["mpc_steps"]) for m in metrics)
        g["mpc_steps"] = steps
        for key in _MPC_COUNTS:
            g["mpc_" + key] = sum(float(m["mpc_" + key]) for m in metrics)
            g["mpc_" + key + "_rate"] = g["mpc_" + key] / steps if steps else None
        # the brake holds until the episode ends (a start within the tightening margin), so the emergency steps
        # divided by the horizon count the episodes spent braking
        g["mpc_emergency_episode_equivalents"] = g["mpc_emergency_count"] / float(metrics[0]["horizon"])
        for key in ("mpc_solve_ms_mean", "mpc_cut_rounds_mean", "mpc_candidate_rate", "mpc_kept_rate"):
            g[key] = ci([m.get(key) for m in metrics])
        maxima = [m["mpc_solve_ms_max"] for m in metrics if m.get("mpc_solve_ms_max") is not None]
        g["mpc_solve_ms_max"] = max(maxima) if maxima else None
    if "reach_decisions" in metrics[0]:
        decisions = sum(float(m["reach_decisions"]) for m in metrics)
        g["reach_decisions"] = decisions
        for key in _REACH_COUNTS:
            g["reach_" + key] = sum(float(m["reach_" + key]) for m in metrics)
            g["reach_" + key + "_rate"] = g["reach_" + key] / decisions if decisions else None
        for key in ("reach_projected_rate", "reach_project_ms_mean", "reach_support_ms_mean", "reach_hull_area_mean"):
            g[key] = ci([m.get(key) for m in metrics])
    return g


def pairwise_tests(metrics: Mapping[Cell, Sequence[Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    """For every (layout, setting, level, variant) with at least two architectures, the tests between each pair:
    Fisher on the counts of seeds at full success, the log-rank test on the times to full success, Fisher on the
    counts of seeds at a contact-free full success."""
    by_context: dict[tuple[str, str, float, str], list[Cell]] = defaultdict(list)
    for cell in metrics:
        by_context[(cell.layout, cell.setting, cell.d_bar, cell.variant)].append(cell)
    tests: dict[str, dict[str, Any]] = {}
    for (layout, setting, d_bar, variant), cells in sorted(by_context.items()):
        present = [c for a in ARCHS for c in cells if c.arch == a]
        if len(present) < 2:
            continue
        context = "/".join([layout, setting, f"d{d_bar:g}", variant or "base"])
        t: dict[str, Any] = {}
        for i, a in enumerate(present):
            for b in present[i + 1 :]:
                ma, mb = metrics[a], metrics[b]
                ta, tb = [m["t_success"] for m in ma], [m["t_success"] for m in mb]
                ca, cb = [m["t_contact_free"] for m in ma], [m["t_contact_free"] for m in mb]
                t[f"{a.arch} vs {b.arch}"] = {
                    "fisher_success": fisher(
                        sum(x is not None for x in ta), len(ta), sum(x is not None for x in tb), len(tb)
                    ),
                    "logrank_success": logrank(ta, tb, _budget(ma), _budget(mb)),
                    "fisher_contact_free": fisher(
                        sum(x is not None for x in ca), len(ca), sum(x is not None for x in cb), len(cb)
                    ),
                }
        tests[context] = t
    return tests


def _budget(metrics: Sequence[Mapping[str, Any]]) -> int:
    """The censoring time of a group of runs: the largest budget among them."""
    return max(int(m.get("budget", BUDGET)) for m in metrics)


def ablation_tests(metrics: Mapping[Cell, Sequence[Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    """The tests between a variant of an architecture and its base in the same layout, setting and level
    (the map-aware hPPO Worker of D75 against hPPO; the sweep's H against H = 10)."""
    tests: dict[str, dict[str, Any]] = {}
    for cell, ms in metrics.items():
        if not cell.variant:
            continue
        base = dataclasses.replace(cell, variant="")
        if base not in metrics:
            continue
        ta, tb = [m["t_success"] for m in metrics[base]], [m["t_success"] for m in ms]
        tests[cell.key] = {
            "base": base.key,
            "fisher_success": fisher(sum(x is not None for x in tb), len(tb), sum(x is not None for x in ta), len(ta)),
            "logrank_success": logrank(tb, ta, _budget(ms), _budget(metrics[base])),
        }
    return tests


# --------------------------------------------------------------------------- the summary


@dataclass(frozen=True)
class Summary:
    """Everything the tables, the figures and the report are made of.

    Attributes:
        cells: The runs of every cell.
        metrics: The metrics of every run, by cell.
        stats: The statistics of every cell.
        tests: The pairwise tests between architectures, by context.
        ablations: The tests between the variants and their bases.
    """

    cells: dict[Cell, list[Run]]
    metrics: dict[Cell, list[dict[str, Any]]]
    stats: dict[Cell, dict[str, Any]]
    tests: dict[str, dict[str, Any]]
    ablations: dict[str, dict[str, Any]]

    def cell(self, layout: str, setting: str, arch: str, d_bar: float = 0.0, variant: str = "") -> Cell | None:
        """The cell with these coordinates, if it has runs."""
        cell = Cell(layout, setting, d_bar, variant, arch)
        return cell if cell in self.cells else None

    def as_json(self) -> dict[str, Any]:
        """The metrics, the statistics and the tests with the cells as strings."""
        return {
            "cells": {c.key: {"runs": self.metrics[c], "stats": self.stats[c]} for c in self.cells},
            "tests": self.tests,
            "ablations": self.ablations,
        }


def summarize(runs: Iterable[Run]) -> Summary:
    """The :class:`Summary` of the runs."""
    cells = cells_of(runs)
    metrics = {cell: [run_metrics(run) for run in members] for cell, members in cells.items()}
    stats_ = {cell: group_stats(ms) for cell, ms in metrics.items()}
    return Summary(cells, metrics, stats_, pairwise_tests(metrics), ablation_tests(metrics))


# --------------------------------------------------------------------------- formatting


def k(x: float | int | None) -> str:
    """Samples in thousands (``41k``), ``--`` for none."""
    if x is None:
        return "--"
    return f"{x / 1000:.0f}k" if x >= 1000 else str(int(x))


def pct(x: float | None, digits: int = 0) -> str:
    """A fraction as a percentage without the sign, ``--`` for none."""
    return "--" if x is None else f"{100.0 * x:.{digits}f}"


def mean_ci(d: Mapping[str, Any], digits: int = 2, scale: float = 1.0) -> str:
    """``mean [low, high]`` of a :func:`ci` result, the mean alone for a single value, ``--`` for none."""
    if d["mean"] is None:
        return "--"
    if d["low"] is None:
        return f"{scale * d['mean']:.{digits}f}"
    return f"{scale * d['mean']:.{digits}f} [{scale * d['low']:.{digits}f}, {scale * d['high']:.{digits}f}]"


def mean_pm(d: Mapping[str, Any], digits: int = 2, scale: float = 1.0) -> str:
    """``mean ± half-width`` of a :func:`ci` result, for tables; the mean alone for a single value."""
    if d["mean"] is None:
        return "--"
    if d["low"] is None:
        return f"{scale * d['mean']:.{digits}f}"
    half = scale * (d["high"] - d["low"]) / 2.0
    return f"{scale * d['mean']:.{digits}f} $\\pm$ {half:.{digits}f}"


def pfmt(p: float | None) -> str:
    """A p-value for a table: three decimals below 0.01, two otherwise, ``$<0.001$`` below that."""
    if p is None:
        return "--"
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


# --------------------------------------------------------------------------- tables


def _row_of(summary: Summary, cell: Cell, label: str) -> str:
    g = summary.stats[cell]
    s = g["t_success"]
    rng = f"{k(s['min'])}--{k(s['max'])}" if s["solved"] else "--"
    return (
        f"{label} & {s['solved']}/{g['n']} & {k(s['km_median'])} & {rng} & {g['t_contact_free']['solved']}/{g['n']} & "
        f"{mean_pm(g['final_gap_max_solved'], 0, 100.0)} & {mean_pm(g['final_contact_fraction'], 0, 100.0)} \\\\"
    )


_RESULTS_HEADER = "Architecture & solved & median & range & no contact & final gap [\\%] & contact [\\%] \\\\"


def _tabular(spec: str) -> str:
    return f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}"


def results_table(summary: Summary, layouts: Sequence[str] = LAYOUTS, settings: Sequence[str] = SETTINGS) -> str:
    """The main table: per layout and setting of the matrix (no disturbance, the shipped agent files) and per
    architecture, the seeds at full success, the Kaplan–Meier median and the range of the samples to it, the seeds
    at a contact-free full success, and the final policy's largest oracle gap (mean over the solving seeds) and
    contact fraction (mean over the seeds), both as percentages with the half-width of the 95 % interval."""
    lines = [_tabular("lcccccc"), "\\toprule", _RESULTS_HEADER]
    for layout in layouts:
        for setting in settings:
            cells = [c for a in ARCHS if (c := summary.cell(layout, setting, a)) is not None]
            if not cells:
                continue
            lines.append("\\midrule")
            lines.append(
                f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{layout.capitalize()}, {SETTING_LABELS[setting]}}}}} \\\\"
            )
            lines.extend(_row_of(summary, c, LABELS_TEX[c.arch]) for c in cells)
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


_PAIRS = (("ppo", "ppo_mpc"), ("ppo", "ppo_mpc_reachability"), ("ppo_mpc", "ppo_mpc_reachability"), ("hppo", "ppo_mpc"))


def logrank_table(summary: Summary) -> str:
    """The log-rank p-values of the times to full success for the pairs of architectures that the thesis
    discusses, per layout and setting of the matrix; Fisher's p-values on the counts are in the report."""
    head = " & ".join(f"{LABELS_TEX[a]} vs {LABELS_TEX[b]}" for a, b in _PAIRS)
    lines = [_tabular("l" + "c" * len(_PAIRS)), "\\toprule", f"Layout, setting & {head} \\\\", "\\midrule"]
    for layout in LAYOUTS:
        for setting in SETTINGS:
            context = f"{layout}/{setting}/d0/base"
            t = summary.tests.get(context)
            if not t:
                continue
            values = [pfmt(t[f"{a} vs {b}"]["logrank_success"]) if f"{a} vs {b}" in t else "--" for a, b in _PAIRS]
            lines.append(f"{layout.capitalize()}, {SETTING_LABELS[setting]} & " + " & ".join(values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


ESCAPE_NAMES = {"escape": "cavity", "mouth": "mouth"}
"""The re-evaluations of D70 by name: inside the cavity of the trap and in front of its mouth."""


def escape_rates(summary: Summary, layout: str = "trap") -> dict[str, dict[str, Any]]:
    """Per cell of the layout and re-evaluation name, the success rate and the contact fraction over the starts,
    mean over the seeds with their interval, and the per-seed success rates; cells without re-evaluations are
    left out."""
    result: dict[str, dict[str, Any]] = {}
    for cell, runs in summary.cells.items():
        if cell.layout != layout or not cell.base:
            continue
        for name in ESCAPE_NAMES:
            lines = [read_reevaluations(run.path, name) for run in runs]
            found = [ln for ln in lines if ln is not None]
            if not found:
                continue
            success = [float(ln["metrics"]["eval/success_rate"]) for ln in found]
            contact = [float(ln["metrics"]["eval/contact_fraction"]) for ln in found]
            result[f"{cell.key}/{name}"] = {
                "cell": cell.key,
                "name": name,
                "box": found[0]["box"],
                "grid": found[0]["grid"],
                "n": len(found),
                "success": ci(success),
                "contact": ci(contact),
                "success_per_seed": success,
            }
    return result


def escape_table(summary: Summary, layout: str = "trap") -> str:
    """The escape evaluation of D70 on the trap: per architecture and setting, the success rate (per cent, mean
    over the seeds) from the starts inside the cavity and from those in front of the mouth."""
    rates = escape_rates(summary, layout)
    lines = [
        _tabular("l" + "cc" * len(SETTINGS)),
        "\\toprule",
        "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SETTING_LABELS[s]}}}" for s in SETTINGS) + " \\\\",
        "Architecture & " + " & ".join("cavity & mouth" for _ in SETTINGS) + " \\\\",
        "\\midrule",
    ]
    for arch in ARCHS:
        values = []
        for setting in SETTINGS:
            cell = summary.cell(layout, setting, arch)
            for name in ESCAPE_NAMES:
                entry = rates.get(f"{cell.key}/{name}") if cell is not None else None
                values.append("--" if entry is None else pct(entry["success"]["mean"], 0))
        if any(v != "--" for v in values):
            lines.append(f"{LABELS_TEX[arch]} & " + " & ".join(values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _pooled_worker(summary: Summary, cells: Sequence[Cell]) -> dict[str, float | None]:
    """The Worker's exit counts of several cells pooled, as rates of their steps."""
    steps = sum(summary.stats[c]["mpc_steps"] for c in cells)
    pooled: dict[str, float | None] = {"steps": steps}
    for key in _MPC_COUNTS:
        total = sum(summary.stats[c]["mpc_" + key] for c in cells)
        pooled[key] = total / steps if steps else None
    runs = sum(summary.stats[c]["n"] for c in cells)
    horizon = float(summary.metrics[cells[0]][0]["horizon"])
    pooled["emergency_per_run"] = (
        sum(summary.stats[c]["mpc_emergency_count"] for c in cells) / horizon / runs if runs else None
    )
    return pooled


def worker_table(summary: Summary) -> str:
    """The MPC Worker's solves, pooled over the settings of each layout without disturbance and over every cell
    with disturbance: the fraction of steps whose search converged, ended with every binary fixed, applied a
    fallback plan (the relaxation's or the best uncertified one), or found no plan (the brake), the solver
    errors per step and the braking episodes per run (emergency steps over the horizon)."""
    lines = [
        _tabular("llrrrrrrr"),
        "\\toprule",
        "Layout & Architecture & steps & converged & fixed & fallback & no plan & errors & braking \\\\",
        "\\midrule",
    ]
    blocks: list[tuple[str, Callable[[Cell], bool]]] = [
        *(
            (
                lay.capitalize() + ", $\\bar d = 0$",
                (lambda c, lay=lay: c.layout == lay and c.d_bar == 0.0 and c.variant == ""),
            )
            for lay in LAYOUTS
        ),
        ("All layouts, $\\bar d > 0$", lambda c: c.d_bar > 0.0 and c.variant == ""),
    ]
    for label, keep in blocks:
        for arch in ("ppo_mpc", "ppo_mpc_reachability"):
            cells = [c for c in summary.cells if c.arch == arch and keep(c) and "mpc_steps" in summary.stats[c]]
            if not cells:
                continue
            w = _pooled_worker(summary, cells)
            fallback = (w["exit_relaxation_count"] or 0.0) + (w["exit_best_count"] or 0.0)
            no_plan = (
                (w["exit_unsolved_count"] or 0.0) + (w["exit_empty_count"] or 0.0) + (w["exit_rounds_count"] or 0.0)
            )
            lines.append(
                f"{label} & {LABELS_TEX[arch]} & {w['steps'] / 1e6:.2f}M & {pct(w['exit_converged_count'], 1)} & "
                f"{pct(w['exit_fixed_count'], 1)} & {pct(fallback, 2)} & {pct(no_plan, 2)} & "
                f"{pct(w['solver_error_count'], 2)} & {w['emergency_per_run']:.1f} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def compute_table(summary: Summary) -> str:
    """The compute per architecture: wall-clock minutes per run (mean over the runs), the Worker's solve time per
    step and the reachable targets' time per decision (support points plus projection), pooled over the matrix's
    cells of the obstacle layouts without disturbance, of the tunnel, and of every cell with disturbance."""
    lines = [
        _tabular("llrrr"),
        "\\toprule",
        "Cells & Architecture & run [min] & Worker [ms/step] & targets [ms/decision] \\\\",
        "\\midrule",
    ]
    blocks: list[tuple[str, Callable[[Cell], bool]]] = [
        ("Slalom and trap, $\\bar d = 0$", lambda c: c.layout != "tunnel" and c.d_bar == 0.0 and c.variant == ""),
        ("Tunnel, $\\bar d = 0$", lambda c: c.layout == "tunnel" and c.d_bar == 0.0 and c.variant == ""),
        ("All layouts, $\\bar d > 0$", lambda c: c.d_bar > 0.0 and c.variant == ""),
    ]
    for label, keep in blocks:
        for arch in ARCHS:
            ms = [m for c in summary.cells if c.arch == arch and keep(c) for m in summary.metrics[c]]
            if not ms:
                continue
            minutes = ci([None if m["elapsed_s"] is None else m["elapsed_s"] / 60.0 for m in ms])
            solve = ci([m.get("mpc_solve_ms_mean") for m in ms])
            support = ci([m.get("reach_support_ms_mean") for m in ms])
            project = ci([m.get("reach_project_ms_mean") for m in ms])
            targets = "--" if support["mean"] is None else f"{support['mean']:.0f} + {project['mean']:.0f}"
            lines.append(
                f"{label} & {LABELS_TEX[arch]} & {mean_pm(minutes, 0)} & "
                f"{'--' if solve['mean'] is None else mean_pm(solve, 0)} & {targets} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def disturbance_table(summary: Summary) -> str:
    """The cells with disturbance beside their cells without: per layout, setting and level, the rows of
    :func:`results_table`."""
    lines = [_tabular("lcccccc"), "\\toprule", _RESULTS_HEADER]
    for layout in LAYOUTS:
        for setting in SETTINGS:
            levels = sorted(
                {c.d_bar for c in summary.cells if c.layout == layout and c.setting == setting and c.variant == ""}
            )
            if len(levels) < 2:
                continue
            for d_bar in levels:
                cells = [c for a in ARCHS if (c := summary.cell(layout, setting, a, d_bar)) is not None]
                lines.append("\\midrule")
                lines.append(
                    f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{layout.capitalize()}, {SETTING_LABELS[setting]}, "
                    f"$\\bar d = {d_bar:g}$}}}} \\\\"
                )
                lines.extend(_row_of(summary, c, LABELS_TEX[c.arch]) for c in cells)
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def sweep_table(summary: Summary, layout: str = "slalom", setting: str = "main") -> str:
    """The agility sweep of D60: the hierarchies at ``H = 10`` (the shipped files) and at the sweep's segment
    lengths, the rows of :func:`results_table`."""
    lines = [_tabular("lcccccc"), "\\toprule", _RESULTS_HEADER]
    for variant, label in (("", "$H = 10$"), ("h6", "$H = 6$"), ("h4", "$H = 4$")):
        cells = [c for a in ARCHS[1:] if (c := summary.cell(layout, setting, a, 0.0, variant)) is not None]
        if not cells:
            continue
        lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{layout.capitalize()}, {SETTING_LABELS[setting]}, {label}}}}} \\\\"
        )
        lines.extend(_row_of(summary, c, LABELS_TEX[c.arch]) for c in cells)
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def ablation_table(summary: Summary, variant: str = "map", arch: str = "hppo") -> str:
    """A variant of an architecture beside its base, per layout and setting: the rows of :func:`results_table`
    and the log-rank p-value of the times to full success between the two."""
    lines = [_tabular("lccccccc"), "\\toprule", _RESULTS_HEADER.replace(" \\\\", " & log-rank \\\\")]
    for layout in LAYOUTS:
        for setting in SETTINGS:
            base, cell = summary.cell(layout, setting, arch), summary.cell(layout, setting, arch, 0.0, variant)
            if base is None or cell is None:
                continue
            lines.append("\\midrule")
            lines.append(
                f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{layout.capitalize()}, {SETTING_LABELS[setting]}}}}} \\\\"
            )
            lines.append(_row_of(summary, base, LABELS_TEX[arch]).replace(" \\\\", " & \\\\"))
            p = summary.ablations.get(cell.key, {}).get("logrank_success")
            lines.append(
                _row_of(summary, cell, LABELS_TEX[arch] + VARIANT_LABELS.get(variant, f" ({variant})")).replace(
                    " \\\\", f" & {pfmt(p)} \\\\"
                )
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_tables(summary: Summary, directory: Path) -> list[Path]:
    """Write every table as a ``tabular`` body under ``directory`` and return the paths."""
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "results_all": results_table(summary),
        "results_slalom": results_table(summary, ("slalom",)),
        "results_trap": results_table(summary, ("trap",)),
        "results_tunnel": results_table(summary, ("tunnel",)),
        "tests": logrank_table(summary),
        "escape": escape_table(summary),
        "worker_solves": worker_table(summary),
        "compute": compute_table(summary),
        "disturbance": disturbance_table(summary),
        "sweep": sweep_table(summary),
        "sweep_exploration": sweep_table(summary, "slalom", "exploration"),
        "ablation_map": ablation_table(summary),
    }
    written = []
    for name, text in tables.items():
        path = directory / f"{name}.tex"
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


# --------------------------------------------------------------------------- figures


def _plt() -> Any:
    """Matplotlib's pyplot with the thesis' style (imported here, so the statistics need no matplotlib)."""
    import logging

    import matplotlib

    logging.getLogger("fontTools").setLevel(logging.ERROR)  # the embedded fonts' timestamps are not our concern
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.2,
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.3,
            "grid.color": "#dddddd",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "figure.dpi": 150,
            "font.family": "serif",
            "mathtext.fontset": "cm",
        }
    )
    return plt


def _save(fig: Any, path: Path) -> None:
    """Write the figure as PDF (without creation date, so that the same data gives the same file) and PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=200)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _legend(fig: Any, ax: Any, ncol: int = 4, bottom: float = 0.12, handlelength: float = 2.4) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        handlelength=handlelength,
        columnspacing=1.2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(w_pad=0.6, rect=(0, bottom, 1, 1))


def learning_curves_figure(
    summary: Summary,
    layout: str | Sequence[str],
    settings: Sequence[str],
    key: str,
    ylabel: str,
    path: Path,
    *,
    ylim: tuple[float, float] | None = (-0.02, 1.02),
    height: float = 1.6,
    d_bar: float = 0.0,
    variant: str = "",
) -> None:
    """One panel per setting of a layout, or per (layout, setting) when several layouts are given, in one row:
    the mean over seeds of an evaluation metric of the mode against the samples, with the 95 % Student-t band,
    one line per architecture."""
    layouts = [layout] if isinstance(layout, str) else list(layout)
    panels = [(lay, s) for lay in layouts for s in settings]
    plt = _plt()
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXT_WIDTH_IN, height), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (lay, setting) in zip(axes, panels, strict=True):
        for arch in ARCHS:
            cell = summary.cell(lay, setting, arch, d_bar, variant)
            if cell is None:
                continue
            curve = mean_curve(summary.cells[cell], key)
            x = curve.env_steps / 1000
            ax.plot(x, curve.mean, STYLES[arch], color=COLORS[arch], label=LABELS[arch])
            low = np.where(np.isnan(curve.low), curve.mean, curve.low)
            high = np.where(np.isnan(curve.high), curve.mean, curve.high)
            ax.fill_between(x, low, high, color=COLORS[arch], alpha=0.15, linewidth=0)
        ax.set_title(SETTING_LABELS[setting] if len(layouts) == 1 else f"{lay}, {SETTING_LABELS[setting]}")
        ax.set_xlabel("samples [$\\times 10^3$]")
        ax.set_xlim(0, BUDGET / 1000)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    _legend(fig, axes[0])
    _save(fig, path)


def _bars(
    ax: Any,
    cells: Sequence[tuple[str, Callable[[str], Cell | None]]],
    summary: Summary,
    ylabel: str,
    archs: Sequence[str] = ARCHS,
    width: float = 0.19,
) -> None:
    """Grouped bars of the samples to full success: one group per cell label, one bar per architecture (the
    median over the solving seeds, min–max whiskers), hatched at the budget where no seed solved, labelled
    ``k/n`` where some seeds solved."""
    labelled: set[str] = set()
    for i, (_label, cell_of) in enumerate(cells):
        for j, arch in enumerate(archs):
            cell = cell_of(arch)
            if cell is None:
                continue
            ts = [m["t_success"] for m in summary.metrics[cell]]
            done = [t / 1000 for t in ts if t is not None]
            x = i + (j - (len(archs) - 1) / 2) * width
            label = None if arch in labelled else LABELS[arch]
            labelled.add(arch)
            if done:
                med = float(np.median(done))
                ax.bar(x, med, width * 0.92, color=COLORS[arch], label=label, linewidth=0)
                ax.errorbar(
                    x,
                    med,
                    yerr=[[med - min(done)], [max(done) - med]],
                    fmt="none",
                    ecolor="#333333",
                    elinewidth=0.6,
                    capsize=1.5,
                )
                if len(done) < len(ts):
                    ax.annotate(
                        f"{len(done)}/{len(ts)}",
                        (x, max(done)),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                        color="#333333",
                        bbox={"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                    )
            else:
                ax.bar(
                    x,
                    BUDGET / 1000,
                    width * 0.92,
                    facecolor="none",
                    edgecolor=COLORS[arch],
                    hatch="////",
                    linewidth=0.5,
                    label=label,
                )
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c[0] for c in cells])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, BUDGET / 1000 + 25)
    ax.axhline(BUDGET / 1000, color="#999999", linewidth=0.5, linestyle="--")
    ax.grid(axis="x", visible=False)


def time_to_success_figure(summary: Summary, path: Path, d_bar: float = 0.0) -> bool:
    """The samples to full success per layout and setting of the matrix at one disturbance level, as grouped bars
    (:func:`_bars`); returns whether any cell at that level exists (nothing is drawn otherwise)."""
    panels = [("slalom", SETTINGS), ("trap", SETTINGS), ("tunnel", ("exploration", "dense"))]
    if all(summary.cell(lay, s, a, d_bar) is None for lay, ss in panels for s in ss for a in ARCHS):
        return False
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.0), sharey=True, gridspec_kw={"width_ratios": [3, 3, 2]})
    short = {"main": "main", "exploration": "explor.", "dense": "dense"}
    for ax, (layout, settings) in zip(axes, panels, strict=True):
        cells = [(short[s], (lambda a, s=s, lay=layout: summary.cell(lay, s, a, d_bar))) for s in settings]
        _bars(ax, cells, summary, "samples to full success [$\\times 10^3$]" if layout == "slalom" else "")
        ax.set_title(layout if d_bar == 0 else f"{layout}, $\\bar d = {d_bar:g}$")
    _legend(fig, axes[0], bottom=0.1, handlelength=1.4)
    _save(fig, path)
    return True


def levels_figure(summary: Summary, path: Path, layout: str = "slalom", setting: str = "main") -> bool:
    """The disturbance levels and the agility sweep on one layout and setting: the samples to full success as
    grouped bars; returns whether the setting has more than one level or a sweep variant (nothing is drawn
    otherwise)."""
    levels = sorted({c.d_bar for c in summary.cells if c.layout == layout and c.setting == setting and c.variant == ""})
    swept = any(c.layout == layout and c.setting == setting and c.variant in ("h6", "h4") for c in summary.cells)
    if len(levels) < 2 and not swept:
        return False
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 1.95))
    cells_d = [(f"$\\bar d = {d:g}$", (lambda a, d=d: summary.cell(layout, setting, a, d))) for d in levels]
    cells_h = [
        (label, (lambda a, v=v: summary.cell(layout, setting, a, 0.0, v)))
        for v, label in (("", "$H = 10$"), ("h6", "$H = 6$"), ("h4", "$H = 4$"))
    ]
    _bars(axes[0], cells_d, summary, "samples to full success [$\\times 10^3$]")
    _bars(axes[1], cells_h, summary, "", archs=ARCHS[1:])
    axes[0].set_title(f"disturbance ({layout}, {setting})")
    axes[1].set_title(f"segment length ({layout}, {setting})")
    _legend(fig, axes[0], bottom=0.1, handlelength=1.4)
    _save(fig, path)
    return True


def training_curve_figure(
    summary: Summary,
    key: str,
    ylabel: str,
    path: Path,
    cells: Sequence[tuple[str, str]] = (("slalom", "main"), ("trap", "main")),
    *,
    smooth: int = 5,
    ylim: tuple[float, float] | None = None,
    variant: str = "",
) -> None:
    """A statistic of the training updates against the samples, smoothed over ``smooth`` updates, the mean over
    seeds with the 95 % band; one panel per (layout, setting)."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(cells), figsize=(TEXT_WIDTH_IN, 1.75), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (layout, setting) in zip(axes, cells, strict=True):
        for arch in ARCHS:
            cell = summary.cell(layout, setting, arch, 0.0, variant)
            if cell is None or not any(key in r for r in summary.cells[cell][0].records):
                continue
            curves = [series(r, key) for r in summary.cells[cell]]
            steps = curves[0][0]
            table = np.stack([v for _, v in curves])
            kernel = np.ones(smooth) / smooth
            sm = np.stack([np.convolve(np.nan_to_num(row), kernel, mode="same") for row in table])
            mean = sm.mean(axis=0)
            half = (
                stats.t.ppf(0.975, sm.shape[0] - 1) * sm.std(axis=0, ddof=1) / np.sqrt(sm.shape[0])
                if sm.shape[0] > 1
                else np.zeros_like(mean)
            )
            x = steps / 1000
            ax.plot(x, mean, STYLES[arch], color=COLORS[arch], label=LABELS[arch])
            ax.fill_between(x, mean - half, mean + half, color=COLORS[arch], alpha=0.15, linewidth=0)
        ax.set_title(f"{layout}, {SETTING_LABELS[setting]}")
        ax.set_xlabel("samples [$\\times 10^3$]")
        ax.set_xlim(0, BUDGET / 1000)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    _legend(fig, axes[0])
    _save(fig, path)


def ablation_figure(
    summary: Summary,
    path: Path,
    variant: str = "map",
    arch: str = "hppo",
    layouts: Sequence[str] = ("slalom", "trap"),
    settings: Sequence[str] = SETTINGS,
) -> bool:
    """The success curves of a variant of an architecture beside its base, one row of panels per layout and one
    column per setting, for the cells in which both exist; returns whether any such cell exists (nothing is drawn
    otherwise)."""
    rows = [
        lay
        for lay in layouts
        if any(
            summary.cell(lay, s, arch, 0.0, variant) is not None and summary.cell(lay, s, arch) is not None
            for s in settings
        )
    ]
    if not rows:
        return False
    plt = _plt()
    fig, axes = plt.subplots(
        len(rows), len(settings), figsize=(TEXT_WIDTH_IN, 0.75 + 1.05 * len(rows)), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes).reshape(len(rows), len(settings))
    for i, layout in enumerate(rows):
        for j, setting in enumerate(settings):
            ax = axes[i, j]
            for v, style, label in (
                ("", "--", LABELS[arch]),
                (variant, "-", LABELS[arch] + VARIANT_LABELS.get(variant, "")),
            ):
                cell = summary.cell(layout, setting, arch, 0.0, v)
                if cell is None:
                    continue
                curve = mean_curve(summary.cells[cell], "eval_mode/success_rate")
                x = curve.env_steps / 1000
                ax.plot(x, curve.mean, style, color=COLORS[arch], label=label)
                low = np.where(np.isnan(curve.low), curve.mean, curve.low)
                high = np.where(np.isnan(curve.high), curve.mean, curve.high)
                ax.fill_between(x, low, high, color=COLORS[arch], alpha=0.15, linewidth=0)
            ax.set_title(f"{layout}, {SETTING_LABELS[setting].split(' ')[0]}")
            ax.set_xlim(0, BUDGET / 1000)
            ax.set_ylim(-0.02, 1.02)
            if i == len(rows) - 1:
                ax.set_xlabel("samples [$\\times 10^3$]")
            if j == 0:
                ax.set_ylabel("success rate")
    _legend(fig, axes[0, 0], ncol=2, bottom=0.1 if len(rows) > 1 else 0.14)
    _save(fig, path)
    return True


def _draw_layout(ax: Any, env: Mapping[str, Any]) -> None:
    """The arena, the obstacles and the goal line of an environment section, as the background of a panel."""
    from matplotlib.patches import Rectangle

    geometry = env["geometry"]
    (x0, x1), (y0, y1) = geometry["arena"]["x"], geometry["arena"]["y"]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="white", edgecolor="#444444", linewidth=0.6))
    for box in geometry["obstacles"]:
        (bx0, bx1), (by0, by1) = box["x"], box["y"]
        ax.add_patch(
            Rectangle((bx0, by0), bx1 - bx0, by1 - by0, facecolor="#bbbbbb", edgecolor="#444444", linewidth=0.5)
        )
    goal = geometry["goal_x"]
    ax.axvspan(goal, x1, facecolor="#e8f0e8", edgecolor="none", zorder=0)
    ax.plot([goal, goal], [y0, y1], color="#2c7a2c", linewidth=0.6, linestyle="--")
    spawn = env["task"]["spawn"]
    ax.add_patch(
        Rectangle(
            (spawn["x"][0], spawn["y"][0]),
            spawn["x"][1] - spawn["x"][0],
            spawn["y"][1] - spawn["y"][0],
            facecolor="none",
            edgecolor="#888888",
            linewidth=0.5,
            linestyle=":",
        )
    )
    ax.set_xlim(x0 - 0.1, x1 + 0.1)
    ax.set_ylim(y0 - 0.1, y1 + 0.1)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_replay(ax: Any, arch: str, replay: Any, targets_of: int | None, wide: bool) -> None:
    """One panel of :func:`trajectories_figure`: the layout, every episode's path, the contacts, the cuts, the
    starts and the targets of one start; the title counts the episodes at the goal and those with a contact."""
    _draw_layout(ax, replay.env)
    color = COLORS[arch]
    for i, length in enumerate(replay.length):
        p = replay.positions[i, : length + 1]
        ax.plot(p[:, 0], p[:, 1], color=color, linewidth=0.6 if wide else 0.45, alpha=0.75, solid_capstyle="round")
        touching = np.flatnonzero(replay.contact[i, :length]) + 1  # the step's end lies on the wall
        if touching.size:
            ax.plot(p[touching, 0], p[touching, 1], ".", color="#222222", markersize=2.2 if wide else 1.6, alpha=0.9)
        if not replay.success[i]:
            ax.plot(p[-1, 0], p[-1, 1], "x", color="#222222", markersize=3.5 if wide else 2.6, markeredgewidth=0.7)
    ax.plot(
        replay.starts[:, 0],
        replay.starts[:, 1],
        "o",
        color="#555555",
        markersize=1.6 if wide else 1.1,
        markeredgewidth=0,
    )
    if targets_of is not None and replay.targets is not None and targets_of < len(replay.length):
        n = int(replay.length[targets_of])
        t = replay.targets[targets_of, : n : max(1, replay.segment_steps)]
        ax.plot(
            t[:, 0],
            t[:, 1],
            "o",
            markerfacecolor="none",
            markeredgecolor="#222222",
            markersize=3.2 if wide else 2.4,
            markeredgewidth=0.6,
        )
    solved, n_episodes = int(replay.success.sum()), len(replay.length)
    touched = int(replay.contact.any(axis=1).sum())
    if wide:
        ax.set_title(f"{LABELS[arch]}: {solved}/{n_episodes} reach the goal, {touched} touch a wall", fontsize=7, pad=2)
    else:
        ax.set_title(f"{LABELS[arch]}: goal {solved}/{n_episodes}, contact {touched}", fontsize=6.5, pad=1.5)


def trajectories_figure(
    replays: Mapping[str, Any] | Sequence[tuple[str, Mapping[str, Any]]],
    path: Path,
    *,
    targets_of: int | None = 12,
    archs: Sequence[str] = ARCHS,
) -> None:
    """The recorded episodes of one final policy per architecture (:mod:`hrlmpc.trajectories`): one panel per
    architecture with the layout, every episode's path (thin line; the steps in contact with a wall as dots; a
    cross where an episode was cut by the horizon) and, for the hierarchies, the targets of one start as small
    circles. With several groups of replays, one column per group (its title above) and one row per architecture.

    Args:
        replays: A :class:`hrlmpc.trajectories.Replay` per architecture to draw, or a sequence of
            ``(title, replays)`` groups, one column each.
        path: The figure file (``.pdf`` and ``.png``).
        targets_of: The index of the start whose targets are drawn, or ``None`` for none.
        archs: The rows, in order; architectures without a replay in any group are skipped.
    """
    plt = _plt()
    groups: list[tuple[str, Mapping[str, Any]]] = (
        [("", replays)] if isinstance(replays, Mapping) else [(t, r) for t, r in replays]
    )
    present = [a for a in archs if any(a in r for _, r in groups)]
    wide = len(groups) == 1
    height = (1.45 if wide else 0.9) * len(present) + (0 if wide else 0.15)
    fig, axes = plt.subplots(len(present), len(groups), figsize=(TEXT_WIDTH_IN, height), squeeze=False)
    for j, (title, group) in enumerate(groups):
        for i, arch in enumerate(present):
            ax = axes[i, j]
            if arch in group:
                _draw_replay(ax, arch, group[arch], targets_of, wide)
            else:
                ax.set_axis_off()
        if title:
            axes[0, j].annotate(
                title,
                xy=(0.5, 1.0),
                xytext=(0, 11),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout(h_pad=0.3, w_pad=0.5, rect=(0, 0, 1, 1 if wide else 0.97))
    _save(fig, path)


def load_replays(paths: Iterable[str | Path], layout: str, setting: str, name: str = "eval") -> dict[str, Any]:
    """The replays ``trajectories_<name>.npz`` found below the given paths whose run belongs to the layout and
    setting (no disturbance, shipped agent files), one per architecture (the first found, by path)."""
    from hrlmpc.analysis import load_run
    from hrlmpc.trajectories import load_replay

    found: dict[str, Any] = {}
    for root in paths:
        for file in sorted(Path(root).rglob(f"trajectories_{name}.npz")):
            cell = classify(load_run(file.parent))
            if cell.layout == layout and cell.setting == setting and cell.base and cell.arch not in found:
                found[cell.arch] = load_replay(file)
    return found


def write_figures(summary: Summary, directory: Path, replay_roots: Sequence[str | Path] = ()) -> list[Path]:
    """Draw every figure of the results chapter under ``directory`` and return the paths (without suffixes)."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def draw(name: str, draw_it: Callable[[Path], Any]) -> None:
        target = directory / name
        if draw_it(target) is not False:
            written.append(target)

    two = ("exploration", "dense")
    draw(
        "slalom_success",
        lambda p: learning_curves_figure(summary, "slalom", SETTINGS, "eval_mode/success_rate", "success rate", p),
    )
    draw(
        "trap_success",
        lambda p: learning_curves_figure(summary, "trap", SETTINGS, "eval_mode/success_rate", "success rate", p),
    )
    draw(
        "tunnel_success",
        lambda p: learning_curves_figure(summary, "tunnel", two, "eval_mode/success_rate", "success rate", p),
    )
    draw(
        "learning_curves_main",
        lambda p: learning_curves_figure(
            summary, ("slalom", "trap"), ("main",), "eval_mode/success_rate", "success rate", p, height=1.7
        ),
    )
    draw(
        "slalom_return",
        lambda p: learning_curves_figure(
            summary, "slalom", SETTINGS, "eval_mode/unshaped_return", "return", p, ylim=None
        ),
    )
    draw(
        "trap_return",
        lambda p: learning_curves_figure(
            summary, "trap", SETTINGS, "eval_mode/unshaped_return", "return", p, ylim=None
        ),
    )
    draw("time_to_success", lambda p: time_to_success_figure(summary, p))
    draw("time_to_success_d1", lambda p: time_to_success_figure(summary, p, 1.0))
    draw("levels_sweep", lambda p: levels_figure(summary, p))
    draw("levels_sweep_exploration", lambda p: levels_figure(summary, p, "slalom", "exploration"))
    draw(
        "training_contact",
        lambda p: training_curve_figure(summary, "rollout/contact_fraction", "episodes with contact", p, ylim=(0, 1)),
    )
    draw(
        "curriculum_stage",
        lambda p: training_curve_figure(summary, "curriculum/stage", "curriculum stage", p, smooth=1, ylim=(0, 8.3)),
    )
    draw("hppo_map", lambda p: ablation_figure(summary, p))
    for setting in ("main", "exploration"):
        groups = []
        for layout in ("slalom", "trap"):
            replays = load_replays(replay_roots, layout, setting) if replay_roots else {}
            if replays:
                draw(f"trajectories_{layout}_{setting}", lambda p, r=replays: trajectories_figure(r, p))
                groups.append((f"{layout}, {SETTING_LABELS[setting]}", replays))
        if len(groups) > 1:
            draw(f"trajectories_{setting}", lambda p, g=groups: trajectories_figure(g, p))
    return written


# --------------------------------------------------------------------------- report and command line


def write_report(summary: Summary, path: Path) -> None:
    """The numbers of every cell, readable, with the per-seed times and the tests."""
    lines = [
        "# Results of the experiment matrix",
        "",
        "Times: the samples at the first evaluation (every 10 240) at which the mode policy met the level; "
        "median: Kaplan–Meier (None if fewer than half the seeds solved); AUC: area under the success curve "
        "over the budget; final gap: the largest oracle gap of the final policy, mean over the seeds at full "
        "success; contact: fraction of the evaluation starts with a contact (final policy); CI: Student-t 95 %.",
        "",
    ]
    for cell in summary.cells:
        g = summary.stats[cell]
        s = g["t_success"]
        rng = f"{k(s['min'])}–{k(s['max'])}" if s["solved"] else "–"
        lines.append(f"## {cell.key}" + ("" if g["complete"] else "  (INCOMPLETE RUNS)"))
        lines.append("")
        lines.append(
            f"- n = {g['n']} (seeds {g['seeds']}); solved {s['solved']}, KM median {k(s['km_median'])}, "
            f"range {rng}; gap ≤ 10 %: {g['t_gap10']['solved']}, strict: {g['t_strict']['solved']}, "
            f"contact-free: {g['t_contact_free']['solved']}; AUC {mean_ci(g['auc'])}"
        )
        lines.append(
            f"- final: success {mean_ci(g['final_success'], 2)}, largest gap (solved) "
            f"{mean_ci(g['final_gap_max_solved'], 3)}, mean gap (solved) {mean_ci(g['final_gap_mean_solved'], 3)}, "
            f"contact {mean_ci(g['final_contact_fraction'], 2)}, return {mean_ci(g['final_return'], 1)}"
        )
        lines.append(
            f"- training: contact fraction {mean_ci(g['train_contact_fraction'], 2)}, impulse per episode "
            f"{mean_ci(g['train_impulse_per_episode'], 3)}; wall time {mean_ci(g['elapsed_min'], 0)} min, "
            f"{mean_ci(g['sps'], 0)} samples/s"
        )
        lines.append(
            f"- times to full success per seed: {[k(t) for t in s['times']]}; contact-free: "
            f"{[k(t) for t in g['t_contact_free']['times']]}"
        )
        if "curriculum_final_stage" in g:
            lines.append(
                f"- curriculum: final stage {g['curriculum_final_stage']} of {g['curriculum_stages']}, last "
                f"stage reached at {[k(t) for t in g['curriculum_last_stage_at']]}"
            )
        if "reached_rate_eval_final" in g:
            lines.append(
                f"- targets reached (final evaluation) {mean_ci(g['reached_rate_eval_final'], 2)}, training "
                f"tail {mean_ci(g['reached_rate_train_tail'], 2)}"
            )
        if "end_distance_eval_final" in g:
            lines.append(
                f"- distance from the target at the segments' end (final evaluation) "
                f"{mean_ci(g['end_distance_eval_final'], 2)} m"
            )
        if "mpc_steps" in g:
            exits = ", ".join(
                f"{name} {pct(g[f'mpc_exit_{name}_count_rate'], 3)} %"
                for name in ("converged", "fixed", "relaxation", "best", "rounds", "empty", "unsolved", "error")
            )
            lines.append(
                f"- Worker: {g['mpc_steps']:.0f} steps; exits {exits}; solver errors "
                f"{pct(g['mpc_solver_error_count_rate'], 2)} % of the steps; emergency steps "
                f"{g['mpc_emergency_count']:.0f} (≈ {g['mpc_emergency_episode_equivalents']:.1f} episodes); "
                f"shifted plan kept {mean_ci(g['mpc_kept_rate'], 3)}; solve {mean_ci(g['mpc_solve_ms_mean'], 1)} "
                f"ms/step (max {g['mpc_solve_ms_max']:.0f}); cut rounds {mean_ci(g['mpc_cut_rounds_mean'], 2)}"
            )
        if "reach_decisions" in g:
            counts = ", ".join(
                f"{name} {g[f'reach_{name}_count']:.0f}"
                for name in ("hull_fallback", "projection_failure", "support_failure", "projection_error")
            )
            lines.append(
                f"- reachable targets: {g['reach_decisions']:.0f} decisions; projected rate "
                f"{mean_ci(g['reach_projected_rate'], 3)}; {counts}; support "
                f"{mean_ci(g['reach_support_ms_mean'], 1)} ms, projection {mean_ci(g['reach_project_ms_mean'], 1)} ms "
                f"per decision; hull area "
                f"{mean_ci(g['reach_hull_area_mean'], 2)} m²"
            )
        lines.append("")
    lines.append("## Tests (two-sided)")
    lines.append("")
    for context, t in summary.tests.items():
        lines.append(
            f"- {context}: "
            + "; ".join(
                f"{pair}: Fisher p={pfmt(v['fisher_success'])}, log-rank p={pfmt(v['logrank_success'])}, "
                f"Fisher no-contact p={pfmt(v['fisher_contact_free'])}"
                for pair, v in t.items()
            )
        )
    if summary.ablations:
        lines += ["", "## Variants against their bases", ""]
        for key, v in summary.ablations.items():
            lines.append(
                f"- {key} vs {v['base']}: Fisher p={pfmt(v['fisher_success'])}, log-rank p={pfmt(v['logrank_success'])}"
            )
    escape = escape_rates(summary)
    if escape:
        lines += ["", "## Escape evaluation (D70)", ""]
        for key, v in escape.items():
            per_seed = [round(x, 2) for x in v["success_per_seed"]]
            lines.append(
                f"- {key}: success {mean_ci(v['success'], 2)} (per seed {per_seed}), "
                f"contact {mean_ci(v['contact'], 2)}; box {v['box']}, grid {v['grid']}, n = {v['n']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: summarise the runs below the given paths and write the tables, the figures and the report."""
    parser = argparse.ArgumentParser(description="Tables and figures of the results chapter from the run directories.")
    parser.add_argument("paths", nargs="+", help="directories containing runs (e.g. runs/final runs/final_h6)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--no-figures", action="store_true", help="write the tables and the report only")
    parser.add_argument(
        "--replays",
        nargs="*",
        default=None,
        help="directories searched for trajectories_*.npz (by default the run directories)",
    )
    args = parser.parse_args(argv)
    summary = summarize(find_runs(args.paths))
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary.as_json(), indent=1, default=float), encoding="utf-8")
    write_report(summary, out / "report.md")
    tables = write_tables(summary, out / "tables")
    print(
        f"{len(summary.cells)} cells, {sum(len(r) for r in summary.cells.values())} runs; "
        f"{len(tables)} tables under {out / 'tables'}"
    )
    incomplete = [c.key for c in summary.cells if not summary.stats[c]["complete"]]
    if incomplete:
        print(f"INCOMPLETE runs in {incomplete}")
    if not args.no_figures:
        roots = args.paths if args.replays is None else args.replays
        figures = write_figures(summary, out / "figures", roots)
        print(f"{len(figures)} figures under {out / 'figures'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
