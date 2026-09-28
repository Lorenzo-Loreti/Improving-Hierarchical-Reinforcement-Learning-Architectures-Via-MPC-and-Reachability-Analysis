"""Seed studies: train one algorithm on many seeds, then show what it learned.

A study is the figure-producing counterpart of the benchmark protocol
(docs/benchmark.md). It trains N seeds of one algorithm on one scenario with
every early stop disabled, so all seeds share one x-axis. It then reads the
learning curves back from each run's metrics.jsonl (algorithms/metrics_log.py),
replays the trained controllers from fixed starts, and sets them against the
min-time oracle (algorithms/optimal_solver.py) from the same starts. Each
scenario's `scripts/study_<algo>.py` describes the algorithm in a `Study` and
hands it to `main`; `scripts/study_compare.py` overlays two finished studies
through `compare_main`. The figures themselves are in study_plots.py.

Three phases, so a figure can be redrawn without retraining:

    train     launch the seeds in parallel, one thread each. Resumable: a seed
              with a finished run of the same budget is skipped.
    analyze   curves, oracle, rollouts -> <out>/analysis.pkl, then plot.
    plot      redraw every figure from analysis.pkl (seconds).
    all       (default) train, then analyze.

What an oracle comparison can and cannot say. On the canonical slalom the
gates cost no time at all: the oracle reaches the lower bound of an
unobstructed straight run from every start (docs/benchmark.md). Many
trajectories therefore share the minimum arrival time, and the oracle returns
one of them: the one with the least control effort, sum of |u|^2. An agent
whose path differs from the oracle's but arrives on the same step is just as
optimal. The figures compare arrival steps and returns, and present the
oracle's path as *one* optimal solution, not *the* optimal solution.

The first studies found that the oracle could be beaten, and this is why
the environment's speed limit changed. The oracle is optimal under its own
model of the plant, in which |v| <= v_max is a hard constraint on the state.
The environment used to integrate the position first and clip the velocity
afterwards, so an agent that kept thrusting at v_max covered
v_max*dt + u_max*dt^2/2 = 0.1325 m per step instead of 0.12 (+10%).
Measured 2026-09-24 (slalom, seeds 1-13, 163 840 steps): from every one of
the 25 grid starts, every PPO seed arrived 5-6 steps before the oracle and
every hPPO seed 3-5 steps before it, with no wall contact. Constant full
thrust down an open corridor from (1, 0) arrived in 71 steps against the
oracle's 78, and PPO's representative seed spent 66 of its 72 steps at v_max
with a mean a_x of 2.2 (u_max = 2.5). That, not the difference between the
grid and random starts, is why evaluation returns sat above 100% of the
oracle. The environment now applies the limit to the acceleration (see
SlalomEnv.step), full thrust arrives on the oracle's step exactly, and the
oracle is an upper bound again. The figures still let a negative gap show,
should one reappear: the grid heatmap's scale is centred on zero, and the
solved-check panel of the learning curves is symmetric-log.

One residual, much smaller, that is not a defect: the oracle is exactly
min-time, but not exactly max-return. The progress term telescopes to
progress_reward_coef * (p_x_final - p_x0), and p_x_final is where the goal
step lands *past* the line, which the oracle, aiming at L plus a 1 mm margin,
does not maximise. At the same arrival step an agent can therefore out-score
it by up to progress_reward_coef * v_max * dt = 1.2, and one step later it can
still come out up to 0.2 ahead. Measured in the 20-seed studies on the fixed
environment (2026-09-24, 204 800 steps): where PPO arrived on the oracle's
step it crossed 3.6 cm past the line on average against the oracle's 0.1 cm,
for return gaps down to -0.73. This is well inside the solved-check's
tolerance of 5, and it is why a trained agent may settle one step behind the
oracle's time at almost no cost in return (hPPO does so from most starts).

Imported by bare name once algorithms/ is on sys.path; the study scripts put it
there.
"""

import argparse
import glob
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from metrics_log import read_config, read_metrics, series
from optimal_solver import MinTimeSolver, spawn_grid


# The default training budget, the same for both algorithms so their curves
# share an x-axis: about 200k steps, exactly 204 800 = 20 x 10 240. Both
# algorithms evaluate every 10 240 steps at their defaults (flat PPO every 10
# updates of 1 024, hPPO every 5 of 2 048), so the last evaluation lands on the
# last training step; a round 200 000 would leave the last 5k unevaluated.
#
# 500k (docs/benchmark.md's budget) spends nine tenths of the axis on the
# plateau. In the per-seed logs of the 2026-09-24 full-budget measurement
# (seeds 1-13, early stops off) the eval return was within 5% of the oracle
# for good by 41k steps for both algorithms, and the strict solved-check first
# passed at 40-72k (PPO) and 51-154k (hPPO). The first studies ran at 163 840
# and reproduced that (PPO 13/13 by 133k, hPPO by 154k). 200k keeps a margin
# over it, which matters because those numbers all predate the speed-limit
# fix (see the note on the oracle above), which changes the dynamics every
# run trains on.
#
# How a shorter budget relates to a 500k run differs by algorithm. hPPO
# trains at a constant learning rate, so its run is a line-for-line prefix of
# the 500k one (checked on all 13 seeds of the 163 840 study). Flat PPO anneals
# its learning rate to 0 over the budget, so a shorter budget is a different
# schedule, not a prefix.
DEFAULT_TOTAL_TIMESTEPS = 204_800

# Starts the trajectory figures use, as (p_x0, p_y0), all at rest. All three
# are points of the 5x5 solved-check grid (optimal_solver.spawn_grid), so their
# rollouts are shared with the grid figures. CENTER_START is the one
# single-start figures use: the middle of the spawn box.
SHOWCASE_STARTS = [(0.0, -1.0), (1.0, 0.0), (2.0, 1.0)]
CENTER_START = (1.0, 0.0)

# The progression figure shows at most this many checkpoints.
MAX_PROGRESSION_PANELS = 6


@dataclass(frozen=True)
class Study:
    """Everything a study needs to know about one algorithm on one scenario."""
    label: str                  # "PPO" / "hPPO": legends and titles
    key: str                    # "ppo" / "hppo": the run-directory prefix
    scenario: str               # "slalom"
    train_script: str           # absolute path of the scenario's script_<key>.py
    make_env: Callable          # a run's config.json (dict) -> plain env it trained on
    plot_env: Callable          # the scenario's plot_env(config, ax, overlay=True)
    load_policy: Callable       # (checkpoint path, env) -> factory(goal_trace=None) -> policy_fn
    hierarchical: bool          # has a manager whose goals the figures can draw
    color: str                  # the algorithm's color in every figure
    out_dir: str                # default output directory

    @property
    def exp_name(self):
        return f"{self.key}_{self.scenario}"


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

PHASES = ["all", "train", "analyze", "plot"]


def parse_known_args_with_phase(parser, argv=None):
    """`parser.parse_known_args()` for a script whose optional first
    positional is the phase and whose unknown arguments are passed through to
    a training script.

    The phase is read off the first argument by hand. Left to argparse, an
    optional positional takes the first free-standing value it meets, which
    here is the value of the first pass-through flag. `--anneal-lr true`
    then fails as "invalid choice: 'true'", and so does `--contact-penalty
    -10`, since a parser with no negative-number options reads -10 as a
    positional. Both ran fine only when the phase was written out first.
    The parser still declares `phase`, so that --help lists it."""
    argv = list(sys.argv[1:] if argv is None else argv)
    phase = argv.pop(0) if argv and argv[0] in PHASES else "all"
    return parser.parse_known_args([phase] + argv)


def _parse_seeds(text):
    """'1-13' / '1,4,7' / '1-3,9' -> sorted list of ints."""
    seeds = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.update(range(int(lo), int(hi) + 1))
        elif part:
            seeds.add(int(part))
    return sorted(seeds)


def parse_args(study):
    parser = argparse.ArgumentParser(
        description=f"{study.label} seed study on the {study.scenario}: train the seeds, "
                    f"then plot learning curves and trajectories against the oracle. Any "
                    f"argument not listed here is passed through to {os.path.basename(study.train_script)}.")
    parser.add_argument("phase", nargs="?", default="all", choices=PHASES,
        help="must come first. train: run the seeds (skips finished ones); analyze: compute "
             "curves, oracle and rollouts, then plot; plot: redraw the figures from "
             "analysis.pkl; all (default): train, then analyze")
    parser.add_argument("--out", type=str, default=study.out_dir,
        help="study directory: runs/, logs/, figures/, analysis.pkl and summary.md go here")
    parser.add_argument("--seeds", type=str, default="1-20",
        help="seeds to train and analyze, e.g. '1-20' or '1,2,5'. On this task fewer than "
             "~10 cannot tell a sample-efficiency difference from seed noise "
             "(docs/benchmark.md, whose tables use seeds 1-13); 20 tightens the "
             "interquartile bands and the first-solve distribution")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS,
        help="training budget of every seed; see DEFAULT_TOTAL_TIMESTEPS in algorithms/study.py "
             "for how the default was chosen")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
        help="seeds trained in parallel, each pinned to one thread")
    parser.add_argument("--checkpoint", type=str, default="final", choices=["final", "solved", "best"],
        help="which checkpoint of each seed the trajectory figures replay")
    parser.add_argument("--plot-max-steps", type=int, default=None,
        help="crop the learning-curve x-axis here (default: the whole budget)")
    parser.add_argument("--no-animation", action="store_true",
        help="skip the GIF/MP4 animation, the slowest figure")
    args, passthrough = parse_known_args_with_phase(parser)
    args.seed_list = _parse_seeds(args.seeds)
    if passthrough and args.phase not in ("all", "train"):
        parser.error(f"unrecognized arguments for the {args.phase} phase: {' '.join(passthrough)}")
    return args, passthrough


def main(study):
    args, passthrough = parse_args(study)
    os.makedirs(args.out, exist_ok=True)
    if args.phase in ("all", "train"):
        train_phase(study, args, passthrough)
    if args.phase in ("all", "analyze"):
        analysis = analyze_phase(study, args)
        _plot(study, analysis, args)
    if args.phase == "plot":
        _plot(study, load_analysis(args.out), args)


def _plot(study, analysis, args):
    import study_plots
    study_plots.plot_study(study, analysis, args.out,
                           plot_max_steps=args.plot_max_steps, animate=not args.no_animation)
    write_summary(analysis, args.out)


# --------------------------------------------------------------------------
# train phase
# --------------------------------------------------------------------------

def find_run(runs_dir, study, seed, total_timesteps):
    """The newest *finished* run of `seed` at this budget in `runs_dir`, or
    None. Finished means final.pt exists, which the loop writes last."""
    pattern = os.path.join(runs_dir, f"{study.exp_name}_{seed}_*")
    # Run directories end in a Unix timestamp, so name order is age order.
    for run_dir in sorted(glob.glob(pattern), reverse=True):
        if not (os.path.exists(os.path.join(run_dir, "final.pt"))
                and os.path.exists(os.path.join(run_dir, "config.json"))):
            continue
        if read_config(run_dir).get("total_timesteps") == total_timesteps:
            return run_dir
    return None


def train_phase(study, args, passthrough):
    runs_dir = os.path.join(args.out, "runs")
    logs_dir = os.path.join(args.out, "logs")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    todo = [s for s in args.seed_list if find_run(runs_dir, study, s, args.total_timesteps) is None]
    done = [s for s in args.seed_list if s not in todo]
    if done:
        print(f"[{study.label}] already trained at {args.total_timesteps} steps, skipped: seeds {done}")
    if not todo:
        return

    # One thread per trainer: without the cap every process starts PyTorch's
    # full-core thread pool and a parallel batch oversubscribes the CPU.
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    jobs = max(1, min(args.jobs, len(todo)))
    print(f"[{study.label}] training seeds {todo} at {args.total_timesteps} steps, {jobs} at a time; "
          f"logs in {logs_dir}")

    pending = list(todo)
    running = {}  # seed -> (Popen, log file, start time)
    failed = []
    t0 = time.time()
    try:
        while pending or running:
            while pending and len(running) < jobs:
                seed = pending.pop(0)
                cmd = [sys.executable, "-u", study.train_script,
                       "--seed", str(seed),
                       "--exp-name", study.exp_name,
                       "--total-timesteps", str(args.total_timesteps),
                       # Every seed runs the whole budget, so all curves share
                       # an x-axis; the first solve is still logged, and saved
                       # as solved.pt.
                       "--solved-early-stop", "false",
                       "--save-eval-checkpoints", "true",
                       "--checkpoint-dir", os.path.abspath(runs_dir),
                       *passthrough]
                log_file = open(os.path.join(logs_dir, f"seed_{seed}.log"), "w")
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env,
                                        cwd=os.path.dirname(study.train_script))
                running[seed] = (proc, log_file, time.time())
            for seed, (proc, log_file, started) in list(running.items()):
                if proc.poll() is None:
                    continue
                log_file.close()
                del running[seed]
                minutes = (time.time() - started) / 60
                finished = len(todo) - len(pending) - len(running)
                if proc.returncode != 0:
                    failed.append(seed)
                    print(f"[{study.label}] seed {seed} FAILED (exit {proc.returncode}) after "
                          f"{minutes:.1f} min, see logs/seed_{seed}.log")
                    continue
                run_dir = find_run(runs_dir, study, seed, args.total_timesteps)
                first = _first_solve(read_metrics(run_dir)) if run_dir else None
                print(f"[{study.label}] seed {seed} done in {minutes:.1f} min "
                      f"({finished}/{len(todo)}), first solve: "
                      f"{'%.0fk steps' % (first / 1e3) if first else 'not within budget'}")
            time.sleep(1.0)
    finally:
        for proc, log_file, _ in running.values():
            proc.terminate()
            log_file.close()
    print(f"[{study.label}] training finished in {(time.time() - t0) / 60:.1f} min"
          + (f"; failed seeds: {failed}" if failed else ""))


# --------------------------------------------------------------------------
# analyze phase
# --------------------------------------------------------------------------

def _first_solve(records):
    steps, _ = series(records, "solved/first_solved_step")
    return steps[0] if steps else None


def rollout(env, make_policy_fn, init_state):
    """Run one deterministic episode from `init_state` and record it.

    `make_policy_fn(goal_trace)` builds a fresh controller for the episode; a
    hierarchical one appends its manager's goals to `goal_trace` (see
    make_policy_fn in algorithms/hppo/hppo_train.py), any other ignores it.
    """
    obs, _ = env.reset(options={"init_state": np.asarray(init_state, dtype=np.float32)})
    goal_trace = []
    policy_fn = make_policy_fn(goal_trace)
    states, actions, rewards, contacts = [np.asarray(obs, dtype=float)], [], [], []
    done, info = False, {}
    while not done:
        action = np.asarray(policy_fn(obs), dtype=float)
        obs, reward, terminated, truncated, info = env.step(action)
        # What the plant actually received: the env clips out-of-range actions.
        actions.append(np.clip(action, env.action_space.low, env.action_space.high))
        states.append(np.asarray(obs, dtype=float))
        rewards.append(float(reward))
        contacts.append(bool(info["collision"]))
        done = terminated or truncated
    return {
        "states": np.asarray(states),           # (T+1, 4): p_x, p_y, v_x, v_y
        "actions": np.asarray(actions),         # (T, 2)
        "rewards": np.asarray(rewards),         # (T,)
        "contacts": np.asarray(contacts),       # (T,) wall contact on that step
        "success": bool(info.get("is_success", False)),
        "return": float(np.sum(rewards)),
        "length": len(rewards),
        "replanned": np.asarray([r for r, _ in goal_trace]) if goal_trace else None,  # (T,)
        "goals": np.asarray([g for _, g in goal_trace]) if goal_trace else None,      # (T, 2)
    }


def _replay_factory(actions):
    """A controller that plays back a fixed action sequence (the oracle's)."""
    def make_policy_fn(goal_trace=None):
        step = {"k": 0}

        def policy_fn(obs):
            action = actions[step["k"]]
            step["k"] += 1
            return action
        return policy_fn
    return make_policy_fn


def oracle_rollout(env, solver, init_state):
    """The oracle's trajectory from `init_state`, replayed through `rollout`
    so it is recorded exactly like an agent's (per-step contacts included)."""
    env.reset(options={"init_state": np.asarray(init_state, dtype=np.float32)})
    result = solver.solve(env)
    return rollout(env, _replay_factory(result.actions), init_state)


def _checkpoint_path(run_dir, which):
    path = os.path.join(run_dir, f"{which}.pt")
    if not os.path.exists(path):
        print(f"  {os.path.basename(run_dir)}: no {which}.pt (never solved?), using final.pt")
        path = os.path.join(run_dir, "final.pt")
    return path


def _choose_progression(eval_steps, first_solve):
    """At most MAX_PROGRESSION_PANELS evaluation steps that tell the learning
    story: the first evaluation, three points through the transition, the
    first solve (or the best guess at one), and the end of the run."""
    eval_steps = sorted(eval_steps)
    anchor = first_solve if first_solve is not None else eval_steps[len(eval_steps) // 2]
    targets = [eval_steps[0], 0.25 * anchor, 0.5 * anchor, 0.75 * anchor, anchor, eval_steps[-1]]
    chosen = []
    for t in targets:
        nearest = min(eval_steps, key=lambda s: abs(s - t))
        if nearest not in chosen:
            chosen.append(nearest)
    return sorted(chosen)[:MAX_PROGRESSION_PANELS]


def analyze_phase(study, args):
    runs_dir = os.path.join(args.out, "runs")
    runs = {}
    for seed in args.seed_list:
        run_dir = find_run(runs_dir, study, seed, args.total_timesteps)
        if run_dir is None:
            print(f"[{study.label}] seed {seed}: no finished run at {args.total_timesteps} steps, left out")
        else:
            runs[seed] = run_dir
    if not runs:
        sys.exit(f"[{study.label}] nothing to analyze in {runs_dir} at {args.total_timesteps} steps "
                 f"(pass the --total-timesteps the seeds were trained with)")
    seeds = sorted(runs)
    config = read_config(runs[seeds[0]])
    env = study.make_env(config)
    solver = MinTimeSolver()
    print(f"[{study.label}] analyzing seeds {seeds}")

    # Learning curves, straight from each run's metrics.jsonl.
    curves, first_solve = {}, {}
    for seed in seeds:
        records = read_metrics(runs[seed])
        curve = {}
        for name, key in [("return", "eval/episodic_return"), ("success", "eval/success_rate"),
                          ("contacts", "eval/collision_count_mean"), ("length", "eval/episodic_length"),
                          ("worst_gap", "solved/worst_gap"), ("mean_gap", "solved/mean_gap"),
                          ("is_solved", "solved/is_solved")]:
            steps, values = series(records, key)
            curve[name] = np.asarray(values, dtype=float)
            curve["step"] = np.asarray(steps, dtype=float)  # identical for every eval-time key
        curves[seed] = curve
        first_solve[seed] = _first_solve(records)

    # The oracle on each seed's own evaluation starts. The loops evaluate from
    # env.reset(seed=run_seed + i), i < eval_episodes, so every seed has its own
    # start set, and its own optimum: this is what its eval return is a
    # percentage of. Adjacent seeds share most draws, so the solves are cached.
    print(f"[{study.label}] solving the oracle on every seed's evaluation starts")
    oracle_cache = {}

    def oracle_on_draw(draw_seed):
        if draw_seed not in oracle_cache:
            env.reset(seed=draw_seed)
            result = solver.solve(env)
            oracle_cache[draw_seed] = (result.total_return, result.length)
        return oracle_cache[draw_seed]

    eval_episodes = int(config["eval_episodes"])
    oracle_eval = {}
    for seed in seeds:
        draws = [oracle_on_draw(seed + i) for i in range(eval_episodes)]
        oracle_eval[seed] = {"return": float(np.mean([d[0] for d in draws])),
                             "length": float(np.mean([d[1] for d in draws]))}

    # The representative seed is the one with the median first solve, so the
    # seed the trajectory figures show is not a hand-picked best case. A seed
    # that never solved sorts last.
    order = sorted(seeds, key=lambda s: (first_solve[s] if first_solve[s] is not None else np.inf, s))
    rep_seed = order[(len(order) - 1) // 2]
    print(f"[{study.label}] representative seed (median first solve): {rep_seed}")

    # Every seed's chosen checkpoint, from every point of the solved-check grid.
    grid = spawn_grid(env, int(config["solved_grid_nx"]), int(config["solved_grid_ny"]))
    print(f"[{study.label}] oracle on the {len(grid)}-point start grid")
    oracle_grid = [oracle_rollout(env, solver, point) for point in grid]
    print(f"[{study.label}] replaying {args.checkpoint}.pt of every seed from the grid")
    agent_grid = {}
    for seed in seeds:
        make_policy_fn = study.load_policy(_checkpoint_path(runs[seed], args.checkpoint), env)
        agent_grid[seed] = [rollout(env, make_policy_fn, point) for point in grid]

    def grid_index(start):
        return int(np.argmin([np.hypot(p[0] - start[0], p[1] - start[1]) for p in grid]))

    showcase = [grid_index(s) for s in SHOWCASE_STARTS]
    center = grid_index(CENTER_START)

    # The representative seed's policy through training, from the center start.
    eval_ckpts = sorted(glob.glob(os.path.join(runs[rep_seed], "eval_*.pt")))
    progression = []
    if eval_ckpts:
        by_step = {int(os.path.basename(p)[5:-3]): p for p in eval_ckpts}
        for step in _choose_progression(list(by_step), first_solve[rep_seed]):
            make_policy_fn = study.load_policy(by_step[step], env)
            progression.append((step, rollout(env, make_policy_fn, grid[center])))
    else:
        print(f"[{study.label}] seed {rep_seed} has no eval_*.pt checkpoints; no progression figure")

    analysis = {
        "label": study.label, "key": study.key, "scenario": study.scenario,
        "hierarchical": study.hierarchical, "color": study.color,
        "config": config, "env_config": env.config, "checkpoint": args.checkpoint,
        "total_timesteps": args.total_timesteps,
        "seeds": seeds, "runs": runs, "rep_seed": rep_seed,
        "curves": curves, "first_solve": first_solve, "oracle_eval": oracle_eval,
        "grid": grid, "oracle_grid": oracle_grid, "agent_grid": agent_grid,
        "showcase": showcase, "center": center, "progression": progression,
    }
    path = os.path.join(args.out, "analysis.pkl")
    with open(path, "wb") as f:
        pickle.dump(analysis, f)
    print(f"[{study.label}] saved {path}")
    return analysis


def load_analysis(out_dir):
    path = os.path.join(out_dir, "analysis.pkl")
    if not os.path.exists(path):
        sys.exit(f"no analysis.pkl in {out_dir}: run the analyze phase first")
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# per-seed numbers
# --------------------------------------------------------------------------

def seed_table(analysis):
    """One row per seed: the numbers summary.md prints and the figures quote."""
    rows = []
    for seed in analysis["seeds"]:
        curve = analysis["curves"][seed]
        agent, oracle = analysis["agent_grid"][seed], analysis["oracle_grid"]
        delta = [a["length"] - o["length"] for a, o in zip(agent, oracle)]
        gap = [o["return"] - a["return"] for a, o in zip(agent, oracle)]
        rows.append({
            "seed": seed,
            "first_solve": analysis["first_solve"][seed],
            "final_eval_return": float(curve["return"][-1]),
            "pct_of_oracle": 100.0 * float(curve["return"][-1]) / analysis["oracle_eval"][seed]["return"],
            "final_eval_contacts": float(curve["contacts"][-1]),
            "solved_at_end": bool(curve["is_solved"][-1]),
            "grid_success": int(sum(a["success"] for a in agent)),
            "grid_mean_extra_steps": float(np.mean(delta)),
            "grid_worst_extra_steps": int(np.max(delta)),
            "grid_worst_return_gap": float(np.max(gap)),
            "grid_contacts": int(sum(a["contacts"].sum() for a in agent)),
        })
    return rows


def write_summary(analysis, out_dir):
    rows = seed_table(analysis)
    n = len(rows)
    firsts = [r["first_solve"] for r in rows if r["first_solve"] is not None]
    k = lambda v: "-" if v is None else f"{v / 1e3:.0f}k"
    lines = [
        f"# {analysis['label']} on the {analysis['scenario']}: seed study",
        "",
        f"{n} seeds, {analysis['total_timesteps']} steps each, early stops off. Trajectories "
        f"replay `{analysis['checkpoint']}.pt`. Representative seed (median first solve): "
        f"**{analysis['rep_seed']}**.",
        "",
        f"- Solved (strict solved-check, held twice) within the budget: **{len(firsts)}/{n}**"
        + (f", first solve median {np.median(firsts) / 1e3:.0f}k / mean {np.mean(firsts) / 1e3:.0f}k steps"
           f" (range {min(firsts) / 1e3:.0f}k-{max(firsts) / 1e3:.0f}k)" if firsts else ""),
        f"- Solved at the last evaluation: {sum(r['solved_at_end'] for r in rows)}/{n}",
        f"- Final eval return: mean {np.mean([r['final_eval_return'] for r in rows]):.1f}, "
        f"{np.mean([r['pct_of_oracle'] for r in rows]):.1f} % of the oracle on the same starts",
        "",
        "Grid columns replay the checkpoint from the 25 points of the solved-check grid; "
        "extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, "
        "negative = the agent arrives first). A return gap slightly below 0 (down to about "
        "-1.2) is not an agent beating the oracle's time: the goal step's progress term pays "
        "for how far past the line the agent crosses, which the oracle does not maximise. "
        "See the notes on the oracle in algorithms/study.py.",
        "",
        "| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end "
        "| grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']}{' (rep.)' if r['seed'] == analysis['rep_seed'] else ''} | {k(r['first_solve'])} "
            f"| {r['final_eval_return']:.1f} | {r['pct_of_oracle']:.1f} % | {r['final_eval_contacts']:.2f} "
            f"| {'yes' if r['solved_at_end'] else 'no'} | {r['grid_success']}/25 "
            f"| {r['grid_mean_extra_steps']:.2f} / {r['grid_worst_extra_steps']} "
            f"| {r['grid_worst_return_gap']:.1f} | {r['grid_contacts']} |")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[{analysis['label']}] wrote {os.path.join(out_dir, 'summary.md')}")


# --------------------------------------------------------------------------
# comparison of two finished studies
# --------------------------------------------------------------------------

def compare_main(default_dirs, default_out, plot_env):
    parser = argparse.ArgumentParser(
        description="Overlay finished seed studies (their analysis.pkl) in shared figures.")
    parser.add_argument("--studies", nargs="+", default=default_dirs,
        help="study directories to compare, each already analyzed")
    parser.add_argument("--out", type=str, default=default_out)
    parser.add_argument("--plot-max-steps", type=int, default=None,
        help="crop the learning-curve x-axis here (default: the shortest study's budget)")
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()
    analyses = [load_analysis(d) for d in args.studies]
    os.makedirs(args.out, exist_ok=True)
    import study_plots
    study_plots.plot_comparison(analyses, args.out, plot_env,
                                plot_max_steps=args.plot_max_steps, animate=not args.no_animation)
