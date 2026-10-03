"""Independent vs Sobol' training starts on the slalom, for every algorithm:
does spreading the training episodes' starts evenly over the spawn box (the
env config's init_sampler, scenarios/slalom/envs/spawn_sampler.py) change how
fast the algorithms learn, or how precisely they end up?

    python scenarios/slalom/scripts/study_init_sampler.py            # train every arm, then analyze and plot
    python scenarios/slalom/scripts/study_init_sampler.py train      # training only
    python scenarios/slalom/scripts/study_init_sampler.py analyze    # after training: recompute and plot
    python scenarios/slalom/scripts/study_init_sampler.py plot       # redraw the figures only
    python scenarios/slalom/scripts/study_init_sampler.py --algos ppo,hppo --seeds 1-20

Each arm, one algorithm under one sampler, is an ordinary seed study
(algorithms/study.py): 20 seeds, 204 800 steps, every early stop off, so its
directory, scenarios/slalom/studies/init_sampler/<algo>_<sampler>/, has the
usual runs/, figures/, analysis.pkl and summary.md, and study_compare.py can
read it. On top of the arms this script writes compare_<algo>/ (each
algorithm's two arms in the study's comparison figures, with a Mann-Whitney
test on the first solves), and summary.md and figures/ for the experiment as a
whole. Training is resumable (finished runs are skipped) and runs every arm
from one queue. Any argument this script does not know is passed through to
every training run alike. git ignores all of it.

The protocol and why
--------------------
The only difference between the two arms of an algorithm is --init-sampler.
Both arms train from the same seeds, the same code and the same environment,
and are evaluated from the same starts: evaluation and the solved-check do
not use the sampler (see spawn_sampler.py). The default sampler, "uniform",
reproduces earlier runs bit for bit, so the uniform arms are the baseline
re-measured on today's code; they are trained here rather than read from the
2026-09-24 studies, which predate several hPPO changes.

"sobol" is not bit-exact against "uniform": it changes every start, and runs
here are chaotic (docs/benchmark.md), so a difference shows only across many
seeds. Hence 20 seeds per arm, as in the seed studies. The comparison looks at
sample efficiency (the first solve of the strict solved-check, and the mean
evaluation return over the run, an area under the learning curve), at late
precision (the share of solved-checks passed after the first solve, and
whether the last one passes) and at the final policy (grid arrival steps and
contacts against the oracle). Each training run also logs how evenly its
latest 32 starts covered the box (spawn/*, algorithms/spawn_coverage.py), so
the experiment records the mechanism next to its effect.

Result (2026-10-03): seeds 1-20, 204 800 steps, no disturbance; p values are
Mann-Whitney tests on the first-solve steps, Sobol' against independent.

                                   flat PPO         hPPO             PPO+MPC
  first solve, median (both arms)  51k              72k              72k
  first solve, mean                53k / 53k        79k / 79k        69k / 69k
  p                                0.90             0.88             0.72
  solved-checks passed after the
    first solve                    97% / 98%        86% / 89%        100% / 100%
  spawn discrepancy                0.012 / 0.001 in every algorithm

- The sampler spread the starts tenfold more evenly and changed nothing
  measurable, for any algorithm. hPPO's one nominal difference (grid arrival
  steps, p = 0.038) did not replicate on seeds 21-40 (p = 0.95); over 40 seeds
  per arm its first solve is 72k against 72k (p = 0.50).
- The independent arms reproduce the earlier PPO, hPPO and PPO+MPC seed
  studies seed for seed.
- The default stays "uniform". The full reading, and why it makes no
  difference (probe_start_variance.py), is in docs/init-sampler.md.
"""
import argparse
import os
import pickle
import re
import sys
from dataclasses import replace
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
# Scenario root, for `envs`; algorithms/, for the flat `study` modules. Each
# study_<algo> module adds its own algorithm's directory on import.
sys.path.append(os.path.abspath(os.path.join(HERE, '..')))
sys.path.append(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'algorithms')))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import study_hppo
import study_ppo
import study_ppo_mpc
import study_plots
from envs.config import SlalomEnvConfig
from envs.visualize_env import plot_env
from envs.vec_slalom_env import SlalomVecEnv
from envs.width_profile import slalom_profile
from metrics_log import read_config, read_metrics, series
from study import (DEFAULT_TOTAL_TIMESTEPS, PHASES, TrainTask, _first_solve, _parse_seeds,
                   analyze_phase, find_run, load_analysis, parse_known_args_with_phase,
                   seed_table, train_runs, write_summary)

ALGOS = {"ppo": study_ppo.STUDY, "hppo": study_hppo.STUDY, "ppo_mpc": study_ppo_mpc.STUDY}
SAMPLERS = ("uniform", "sobol")
SAMPLER_LABEL = {"uniform": "uniform", "sobol": "Sobol'"}
# Within an algorithm, the baseline is grey and the Sobol' arm takes the
# algorithm's own color (study_plots: a color means an algorithm).
UNIFORM_COLOR = "#9a9893"

DEFAULT_OUT = os.path.abspath(os.path.join(HERE, '..', 'studies', 'init_sampler'))


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------

def arm_study(out, algo, sampler):
    """The seed study of one arm: the algorithm's own Study, relabelled and
    moved to its own directory."""
    base = ALGOS[algo]
    return replace(base,
                   label=f"{base.label}, {SAMPLER_LABEL[sampler]}",
                   color=base.color if sampler == "sobol" else UNIFORM_COLOR,
                   out_dir=os.path.join(out, f"{algo}_{sampler}"))


def find_arm_run(study, sampler, seed, total_timesteps):
    """find_run, and the run's own config agrees on the sampler (runs from
    before the flag existed have none, and were uniform)."""
    run_dir = find_run(os.path.join(study.out_dir, "runs"), study, seed, total_timesteps)
    if run_dir is not None and read_config(run_dir).get("init_sampler", "uniform") != sampler:
        sys.exit(f"{run_dir} was not trained with --init-sampler {sampler}")
    return run_dir


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Independent vs Sobol' training starts on the slalom, for every algorithm. Any "
                    "argument not listed here is passed through to every training run.")
    parser.add_argument("phase", nargs="?", default="all", choices=PHASES,
        help="must come first. train: run every arm (skips finished runs); analyze: compute "
             "every arm's study and the comparison, then plot; plot: redraw the figures; all "
             "(default): train, then analyze")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT,
        help="experiment directory: one sub-directory per arm, compare_<algo>/, figures/, "
             "analysis.pkl and summary.md")
    parser.add_argument("--algos", type=str, default="ppo,hppo,ppo_mpc",
        help="comma-separated algorithms, from " + ", ".join(ALGOS) + "; trained in this order")
    parser.add_argument("--seeds", type=str, default="1-20",
        help="seeds of every arm (see algorithms/study.py's --seeds for why 20)")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS,
        help="training budget of every run")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
        help="runs trained in parallel, each pinned to one thread")
    args, passthrough = parse_known_args_with_phase(parser)
    args.seed_list = _parse_seeds(args.seeds)
    args.algo_list = [a.strip() for a in args.algos.split(",") if a.strip()]
    unknown = [a for a in args.algo_list if a not in ALGOS]
    if unknown:
        parser.error(f"unknown algorithms {unknown}; choose from {list(ALGOS)}")
    if passthrough and args.phase not in ("all", "train"):
        parser.error(f"unrecognized arguments for the {args.phase} phase: {' '.join(passthrough)}")
    return args, passthrough


def main():
    args, passthrough = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.phase in ("all", "train"):
        train_phase(args, passthrough)
    if args.phase in ("all", "analyze"):
        experiment = analyze(args)
        plot(experiment, args.out)
    if args.phase == "plot":
        plot(load_experiment(args.out), args.out)


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------

def train_phase(args, passthrough):
    """One queue for every arm, algorithm by algorithm in --algos order and
    seed-major within one, so the faster algorithms' results come first and
    an interrupted algorithm leaves both of its arms with the same seeds."""
    tasks, skipped = [], 0
    for algo in args.algo_list:
        for seed in args.seed_list:
            for sampler in SAMPLERS:
                study = arm_study(args.out, algo, sampler)
                if find_arm_run(study, sampler, seed, args.total_timesteps) is not None:
                    skipped += 1
                    continue
                tasks.append(TrainTask(study, study.out_dir, seed, args.total_timesteps,
                                       ("--init-sampler", sampler, *passthrough)))
    if skipped:
        print(f"[init_sampler] {skipped} runs already finished at {args.total_timesteps} steps, skipped")
    train_runs(tasks, args.jobs)


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------

def _late_precision(records):
    """Share of the strict solved-checks passed *after* the first solve, or
    None if the run never solved (as in sweep_extrinsic_coef.py)."""
    first = _first_solve(records)
    if first is None:
        return None
    steps, solved = series(records, "solved/is_solved")
    after = [s for step, s in zip(steps, solved) if step > first]
    return float(np.mean(after)) if after else None


# The solved-check's line in every training loop's log: the start with the
# largest return gap to the oracle at that evaluation. Only the log has it.
SOLVED_CHECK_LINE = re.compile(r"Solved-check at update \d+: solved=(True|False) worst_gap=\S+ at "
                               r"p_x0=([-\d.]+) p_y0=([-\d.]+)")


def failed_check_worst_starts(log_path):
    """(p_x0, p_y0) of the worst start of every solved-check that failed,
    in the order of the run."""
    with open(log_path, encoding="utf-8", errors="replace") as f:
        return [(float(x), float(y)) for solved, x, y in SOLVED_CHECK_LINE.findall(f.read())
                if solved == "False"]


def run_extras(run_dir, log_path):
    """What the arm's analysis.pkl does not already hold. From the run's
    metrics.jsonl: late precision, the mean evaluation return over the run,
    the training contacts, and the spawn/* coverage series. From its log: the
    worst start of every failed solved-check."""
    records = read_metrics(run_dir)
    _, eval_return = series(records, "eval/episodic_return")
    _, contacts = series(records, "charts/collision_count_mean")
    extras = {
        "late_precision": _late_precision(records),
        "eval_return_mean": float(np.mean(eval_return)),
        "train_contacts": float(np.mean(contacts)) if contacts else None,
        "failed_worst_starts": failed_check_worst_starts(log_path),
    }
    for key in ("spawn/empty_cells", "spawn/discrepancy"):
        steps, values = series(records, key)
        extras[key] = (np.asarray(steps, dtype=float), np.asarray(values, dtype=float))
    return extras


def analyze(args):
    arms = {}
    for algo in args.algo_list:
        for sampler in SAMPLERS:
            study = arm_study(args.out, algo, sampler)
            runs = {s: find_arm_run(study, sampler, s, args.total_timesteps) for s in args.seed_list}
            seeds = [s for s, r in runs.items() if r is not None]
            if not seeds:
                print(f"[init_sampler] {study.label}: no finished runs, left out")
                continue
            arm_args = SimpleNamespace(out=study.out_dir, seed_list=seeds,
                                       total_timesteps=args.total_timesteps, checkpoint="final")
            analysis = analyze_phase(study, arm_args)
            study_plots.plot_study(study, analysis, study.out_dir, animate=False)
            write_summary(analysis, study.out_dir)
            arms[(algo, sampler)] = {
                "study_dir": study.out_dir,
                "extras": {s: run_extras(analysis["runs"][s],
                                         os.path.join(study.out_dir, "logs", f"seed_{s}.log"))
                           for s in analysis["seeds"]},
            }
    experiment = {"algos": args.algo_list, "arms": arms, "total_timesteps": args.total_timesteps}
    with open(os.path.join(args.out, "analysis.pkl"), "wb") as f:
        pickle.dump(experiment, f)
    return experiment


def load_experiment(out):
    path = os.path.join(out, "analysis.pkl")
    if not os.path.exists(path):
        sys.exit(f"no analysis.pkl in {out}: run the analyze phase first")
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------

def _p(a, b):
    """Two-sided Mann-Whitney U p value, or None without two samples."""
    if len(a) < 2 or len(b) < 2:
        return None
    return float(mannwhitneyu(a, b).pvalue)


def _median_shift(a, b, n_boot=10_000):
    """median(b) - median(a), with a 95 % bootstrap interval (seeds resampled
    within each arm): how large a difference the seeds leave room for. Seeds
    that never solved count as infinite, so the shift can be infinite too."""
    rng = np.random.default_rng(0)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    boot = (np.median(rng.choice(b, (n_boot, len(b))), axis=1)
            - np.median(rng.choice(a, (n_boot, len(a))), axis=1))
    return float(np.median(b) - np.median(a)), *np.percentile(boot, [2.5, 97.5])


def arm_row(analysis, extras):
    """One arm's numbers. A seed that never solved ranks after every seed that
    did (its first solve counts as infinite in the tests)."""
    rows = seed_table(analysis)
    seeds = analysis["seeds"]
    firsts = [analysis["first_solve"][s] for s in seeds]
    solved = [f for f in firsts if f is not None]
    late = [extras[s]["late_precision"] for s in seeds if extras[s]["late_precision"] is not None]
    worst = np.array([p for s in seeds for p in extras[s]["failed_worst_starts"]]).reshape(-1, 2)
    on_corner = (np.isin(worst[:, 0], [0.0, 2.0]) & np.isin(np.abs(worst[:, 1]), [1.0]))
    extra_steps = np.array([[r["length"] - o["length"] for r, o in zip(analysis["agent_grid"][s],
                                                                         analysis["oracle_grid"])]
                            for s in seeds])
    spawn = {key: float(np.mean([np.mean(extras[s][key][1]) for s in seeds if len(extras[s][key][1])]))
             for key in ("spawn/empty_cells", "spawn/discrepancy")}
    return {
        "label": analysis["label"],
        "n": len(seeds),
        "solved": len(solved),
        "first_ranked": [np.inf if f is None else f for f in firsts],
        "first_median": float(np.median(solved)) if solved else None,
        "first_mean": float(np.mean(solved)) if solved else None,
        "first_range": (min(solved), max(solved)) if solved else None,
        "auc": [extras[s]["eval_return_mean"] for s in seeds],
        "late": late,
        "late_precision": float(np.mean(late)) if late else None,
        "solved_at_end": sum(r["solved_at_end"] for r in rows),
        "final_return": float(np.mean([r["final_eval_return"] for r in rows])),
        "pct_of_oracle": float(np.mean([r["pct_of_oracle"] for r in rows])),
        "grid_on_oracle_step": float(np.mean(extra_steps <= 0)),
        "grid_extra_steps": float(extra_steps.mean()),
        "grid_extra_steps_per_seed": list(extra_steps.mean(axis=1)),
        "failed_checks": len(worst),
        "corner_share": float(on_corner.mean()) if len(worst) else None,
        "grid_contacts": sum(r["grid_contacts"] for r in rows),
        "train_contacts": float(np.mean([extras[s]["train_contacts"] for s in seeds])),
        **spawn,
    }


def write_experiment_summary(experiment, analyses, out):
    k = lambda v: "-" if v is None else f"{v / 1e3:.0f}k"
    fmt_p = lambda p: "-" if p is None else f"{p:.2g}"
    pct = lambda v: "-" if v is None else f"{100 * v:.0f} %"

    def shift(vs, row):
        if vs is None:
            return "-"
        d, lo, hi = _median_shift(vs["first_ranked"], row["first_ranked"])
        sign = lambda v: "inf" if np.isinf(v) else f"{v / 1e3:+.0f}k"
        return f"{sign(d)} ({sign(lo)} to {sign(hi)})"
    learning = [
        "| arm | solved | first solve, median / mean (range) | p | median shift (95 % CI) "
        "| return AUC, median | p | late precision | p | solved at the end |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    policy = [
        "| arm | final eval return (% of oracle) | grid: on the oracle's step | grid: mean extra steps "
        "| p | grid contacts | training contacts/ep | failed checks: worst start at a corner "
        "| spawn: empty 5x5 cells | spawn: discrepancy |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for algo in experiment["algos"]:
        base = None
        for sampler in SAMPLERS:
            key = (algo, sampler)
            if key not in analyses:
                continue
            row = arm_row(analyses[key], experiment["arms"][key]["extras"])
            if sampler == "uniform":
                base = row
            vs = base if sampler != "uniform" else None
            p = lambda field: fmt_p(_p(vs[field], row[field])) if vs else "-"
            rng = row["first_range"]
            learning.append(
                f"| {row['label']} | {row['solved']}/{row['n']} "
                f"| {k(row['first_median'])} / {k(row['first_mean'])}"
                + (f" ({k(rng[0])}-{k(rng[1])})" if rng else "")
                + f" | {p('first_ranked')} | {shift(vs, row)} | {np.median(row['auc']):.1f} | {p('auc')} "
                f"| {pct(row['late_precision'])} | {p('late')} | {row['solved_at_end']}/{row['n']} |")
            policy.append(
                f"| {row['label']} | {row['final_return']:.1f} ({row['pct_of_oracle']:.1f} %) "
                f"| {pct(row['grid_on_oracle_step'])} | {row['grid_extra_steps']:+.2f} "
                f"| {p('grid_extra_steps_per_seed')} | {row['grid_contacts']} | {row['train_contacts']:.2f} "
                f"| {pct(row['corner_share'])} of {row['failed_checks']} "
                f"| {row['spawn/empty_cells']:.2f} | {row['spawn/discrepancy']:.4f} |")
    lines = [
        "# Independent vs Sobol' training starts on the slalom",
        "",
        f"Every arm: the same seeds, {experiment['total_timesteps']} steps, every early stop off; "
        "the arms of an algorithm differ only in --init-sampler. p: two-sided Mann-Whitney U over "
        "the seeds, against the same algorithm's uniform arm (a seed that never solved ranks after "
        "every seed that did).",
        "",
        "## Learning",
        "",
        "First solve: of the strict solved-check, held for two evaluations. Median shift: the "
        "Sobol' arm's median first solve minus the uniform arm's, with a 95 % bootstrap interval "
        "over seeds. Return AUC: the "
        "evaluation return averaged over the whole run, per seed. Late precision: share of the "
        "solved-checks passed after the first solve, averaged over the seeds that solved.",
        "",
        *learning,
        "",
        "## The final policy, and the mechanism",
        "",
        "Grid: the final policy from the 25 solved-check starts, against the oracle. Failed "
        "checks: over every solved-check that failed in any seed, the share whose worst start is "
        "one of the box's 4 corners (4 of the 25 grid starts, 16 %). Spawn: how evenly each run's "
        "latest 32 training starts covered the box (algorithms/spawn_coverage.py), averaged over "
        "the run; for reference, 32 independent starts leave 6.8 cells empty, at a discrepancy of "
        "0.012.",
        "",
        *policy,
        "",
        "Per-seed numbers: each arm's own summary.md. Figures: figures/ and compare_<algo>/figures/.",
    ]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[init_sampler] wrote {os.path.join(out, 'summary.md')}")


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_starts(fig_dir, n=64, seed=1):
    """The first n starts under each sampler, on the spawn box with the 5 x 5
    partition of the solved-check's resolution. Drawn from a vector env whose
    8 episodes all end on every step (max_steps=1), so it shows the samplers'
    own sequences, not a particular run's (whose order of episode ends varies)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    for ax, sampler in zip(axes, SAMPLERS):
        config = SlalomEnvConfig(width_profile=slalom_profile(), init_sampler=sampler, max_steps=1)
        vec_env = SlalomVecEnv(num_envs=8, config=config)
        starts = [vec_env.reset(seed=seed)[0]]
        while sum(len(s) for s in starts) < n:
            starts.append(vec_env.step(np.zeros((8, 2), dtype=np.float32))[0])
        starts = np.concatenate(starts)[:n, :2]
        low, high = vec_env.spawn_low, vec_env.spawn_high
        for edge in np.linspace(0.0, 1.0, 6):
            ax.axvline(low[0] + edge * (high[0] - low[0]), color=study_plots.GRID, lw=0.8, zorder=0)
            ax.axhline(low[1] + edge * (high[1] - low[1]), color=study_plots.GRID, lw=0.8, zorder=0)
        cells = np.minimum(((starts - low) / (high - low) * 5).astype(int), 4)
        empty = 25 - len(set(map(tuple, cells)))
        color = study_plots.INK if sampler == "sobol" else UNIFORM_COLOR
        ax.scatter(starts[:32, 0], starts[:32, 1], s=16, color=color, zorder=3, label="starts 1-32")
        ax.scatter(starts[32:, 0], starts[32:, 1], s=16, facecolor="white", edgecolor=color, zorder=3,
                   label=f"starts 33-{n}")
        empty32 = 25 - len(set(map(tuple, cells[:32])))
        ax.set_title(f"{SAMPLER_LABEL[sampler]}\n{empty32} empty cells after 32 starts, {empty} after {n}")
        ax.add_patch(plt.Rectangle(tuple(low), *(high - low), fill=False, edgecolor=study_plots.AXIS, lw=0.8))
        margin = 0.04 * (high - low)
        ax.set_xlim(low[0] - margin[0], high[0] + margin[0])
        ax.set_ylim(low[1] - margin[1], high[1] + margin[1])
        ax.set_aspect("equal")
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.grid(False)
        ax.set_xlabel("p_x0 [m]")
    axes[0].set_ylabel("p_y0 [m]")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    study_plots._save(fig, fig_dir, "starts")


def fig_spawn_coverage(experiment, fig_dir):
    """spawn/* over training, median and interquartile band over the seeds,
    one row per algorithm."""
    algos = [a for a in experiment["algos"] if any((a, s) in experiment["arms"] for s in SAMPLERS)]
    keys = [("spawn/empty_cells", "empty 5x5 cells, last 32 starts"),
            ("spawn/discrepancy", "centred L2-discrepancy, last 32 starts")]
    fig, axes = plt.subplots(len(algos), 2, figsize=(7.2, 2.2 * len(algos)), squeeze=False)
    for row, algo in enumerate(algos):
        for col, (key, title) in enumerate(keys):
            ax = axes[row, col]
            for sampler in SAMPLERS:
                arm = experiment["arms"].get((algo, sampler))
                if arm is None:
                    continue
                curves = [e[key] for e in arm["extras"].values() if len(e[key][1])]
                n = min(len(c[1]) for c in curves)
                steps = curves[0][0][:n]
                values = np.vstack([c[1][:n] for c in curves])
                color = ALGOS[algo].color if sampler == "sobol" else UNIFORM_COLOR
                ax.plot(steps, np.median(values, axis=0), color=color, lw=1.2,
                        label=SAMPLER_LABEL[sampler])
                ax.fill_between(steps, *np.percentile(values, [25, 75], axis=0), color=color,
                                alpha=0.2, lw=0)
            ax.set_title(f"{ALGOS[algo].label}: {title}")
            ax.xaxis.set_major_formatter(study_plots.STEPS_FMT)
            if row == len(algos) - 1:
                ax.set_xlabel("environment steps")
            if col == 0:
                ax.set_ylim(bottom=0)
                ax.legend(loc="upper right")
    fig.tight_layout()
    study_plots._save(fig, fig_dir, "spawn_coverage")


def fig_worst_starts(experiment, fig_dir):
    """Where the policy is furthest from the oracle while it has not solved
    yet: for every failed solved-check of every seed, the grid start with the
    largest return gap, as a share of the arm's failed checks. One row per
    algorithm, one column per sampler. If the corners were hard because
    independent sampling visits them too rarely, the Sobol' column would
    spread the mass out; it does not."""
    algos = [a for a in experiment["algos"] if all((a, s) in experiment["arms"] for s in SAMPLERS)]
    xs, ys = np.linspace(0.0, 2.0, 5), np.linspace(-1.0, 1.0, 5)   # spawn_grid's 5 x 5 points
    fig, axes = plt.subplots(len(algos), 2, figsize=(5.6, 2.6 * len(algos)), squeeze=False)
    for row, algo in enumerate(algos):
        for col, sampler in enumerate(SAMPLERS):
            ax = axes[row, col]
            worst = [p for e in experiment["arms"][(algo, sampler)]["extras"].values()
                     for p in e["failed_worst_starts"]]
            share = np.zeros((5, 5))
            for x, y in worst:
                share[int(np.argmin(np.abs(ys - y))), int(np.argmin(np.abs(xs - x)))] += 1
            share /= max(1, len(worst))
            ax.imshow(share, origin="lower", cmap="Greys", vmin=0.0, vmax=0.5,
                      extent=(-0.25, 2.25, -1.25, 1.25), aspect="equal")
            for i, y in enumerate(ys):
                for j, x in enumerate(xs):
                    if share[i, j] >= 0.005:
                        ax.text(x, y, f"{100 * share[i, j]:.0f}", ha="center", va="center", fontsize=7,
                                color="white" if share[i, j] > 0.25 else study_plots.INK)
            ax.set_title(f"{ALGOS[algo].label}, {SAMPLER_LABEL[sampler]}\n{len(worst)} failed checks")
            ax.set_xticks(xs)
            ax.set_yticks(ys)
            ax.grid(False)
            if row == len(algos) - 1:
                ax.set_xlabel("p_x0 [m]")
            if col == 0:
                ax.set_ylabel("p_y0 [m]")
    fig.suptitle("Worst start of each failed solved-check, % of the arm's failed checks",
                 x=0.02, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    study_plots._save(fig, fig_dir, "worst_starts")


def fig_first_solves(experiment, analyses, fig_dir):
    """Every seed's first solve, per arm, with the median: the experiment's
    sample-efficiency result in one panel."""
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ticks, labels = [], []
    rng = np.random.default_rng(0)  # horizontal jitter only
    for i, algo in enumerate(experiment["algos"]):
        for j, sampler in enumerate(SAMPLERS):
            analysis = analyses.get((algo, sampler))
            if analysis is None:
                continue
            x = 3 * i + j
            firsts = [f for f in analysis["first_solve"].values() if f is not None]
            color = ALGOS[algo].color if sampler == "sobol" else UNIFORM_COLOR
            ax.scatter(x + rng.uniform(-0.18, 0.18, len(firsts)), firsts, s=12, color=color, zorder=3)
            if firsts:
                ax.hlines(np.median(firsts), x - 0.3, x + 0.3, color=study_plots.INK, lw=1.5, zorder=4)
            ticks.append(x)
            labels.append(f"{ALGOS[algo].label}\n{SAMPLER_LABEL[sampler]}")
    ax.set_xticks(ticks, labels)
    ax.set_ylim(bottom=0)
    top = ax.get_ylim()[1]
    for x, (algo, sampler) in zip(ticks, [(a, s) for a in experiment["algos"] for s in SAMPLERS
                                           if (a, s) in analyses]):
        firsts = analyses[(algo, sampler)]["first_solve"].values()
        ax.text(x, top, f"{sum(f is not None for f in firsts)}/{len(firsts)} solved",
                ha="center", va="bottom", fontsize=7, color=study_plots.INK_2)
    ax.yaxis.set_major_formatter(study_plots.STEPS_FMT)
    ax.set_ylabel("first solve [steps]")
    ax.set_title("First solve of the strict solved-check, every seed (bar: median)", pad=14)
    ax.grid(axis="x", visible=False)
    study_plots._save(fig, fig_dir, "first_solves")


def plot(experiment, out):
    analyses = {key: load_analysis(arm["study_dir"]) for key, arm in experiment["arms"].items()}
    fig_dir = os.path.join(out, "figures")
    print("[init_sampler] drawing figures")
    with plt.rc_context(study_plots.STYLE):
        fig_starts(fig_dir)
        fig_spawn_coverage(experiment, fig_dir)
        fig_worst_starts(experiment, fig_dir)
        fig_first_solves(experiment, analyses, fig_dir)
    for algo in experiment["algos"]:
        pair = [analyses.get((algo, s)) for s in SAMPLERS]
        if all(a is not None for a in pair):
            study_plots.plot_comparison(pair, os.path.join(out, f"compare_{algo}"), plot_env, animate=False)
    write_experiment_summary(experiment, analyses, out)


if __name__ == "__main__":
    main()
