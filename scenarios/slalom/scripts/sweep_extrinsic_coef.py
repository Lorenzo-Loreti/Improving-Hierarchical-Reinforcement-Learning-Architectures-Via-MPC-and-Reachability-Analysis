"""hPPO on the slalom across --worker-extrinsic-coef, from the default 0.02
down to 0: how much of the environment's reward the worker needs, and whether
the hierarchy is still a hierarchy at the amount it gets.

    python scenarios/slalom/scripts/sweep_extrinsic_coef.py           # train every (coef, seed), then analyze and plot
    python scenarios/slalom/scripts/sweep_extrinsic_coef.py train     # training only
    python scenarios/slalom/scripts/sweep_extrinsic_coef.py analyze   # after training: recompute and plot
    python scenarios/slalom/scripts/sweep_extrinsic_coef.py plot      # redraw the figures only
    python scenarios/slalom/scripts/sweep_extrinsic_coef.py --coefs 0.02,0.01,0 --seeds 1-5
    python scenarios/slalom/scripts/sweep_extrinsic_coef.py --help

Training is resumable (finished runs are skipped), and any argument this
script does not know is passed through to script_hppo.py, for every arm alike.
Everything goes to scenarios/slalom/studies/extrinsic_sweep/, which git
ignores: one coef_<value>/ directory per arm (its runs/ and logs/), plus
figures/, analysis.pkl and summary.md.

Why sweep it
------------
0.02 was a scale argument that worked on the first value tried, not the
optimum of a sweep (docs/worker-termination-avoidance.md, section 10). The
term exists to fix the worker's termination avoidance (the comment above the
flag in algorithms/hppo/hppo_train.py): with a purely intrinsic reward,
stalling in front of the goal line is worth more to the worker than crossing
it. It also has a cost that the sweep has to measure, not only the benefit.
The worker sees the full state (p_x, p_y, v_x, v_y) as well as its goal, so
once the environment's reward reaches it, nothing stops it from learning the
slalom by itself and leaving the manager with nothing to decide. That is the
feudal hierarchy degenerating into one agent with an ornamental second head.

Which coefficients
------------------
The coefficient has a floor, which can be derived. When the worker orbits in
front of the line, it earns at most v_max*dt of goal progress per step plus
coef * step_penalty; the progress term of the environment's reward sums to
zero over an orbit. When it crosses, it earns coef * goal_reward once. So

    V(stall) ~ (v_max*dt - coef*|step_penalty|) / (1 - gamma)
    V(cross) ~ coef * goal_reward

and the terminal only wins above

    coef* = (v_max*dt / (1 - gamma)) / (goal_reward + |step_penalty| / (1 - gamma))
          = 12 / (1000 + 100) ~ 0.011 on the slalom   (0.04 on the tunnel)

The default arms bracket that floor. 0.02 is the default, with a margin of 2x
(+20 against ~10). 0.015 still clears it. 0.01 sits just below it, where
stalling beats crossing by only ~1. 0.005 is well below it, and 0 is the
pre-fix reward, which collapses (0/2 clean solves in the original ablation).
The floor is an estimate: a real orbit rarely earns the full v_max*dt, and the
policy has to find the orbit before it pays. So the prediction is that the
collapse returns below the floor, and later the nearer the arm is to it. It is
also why the budget is the full 500k with every early stop off (the
full-budget protocol of docs/benchmark.md). The pathology binds late, and
often only after a first solve. A run stopped at its first solve would score a
delayed collapse as a clean one.

(Result of the first sweep, 2026-09-24, 10 seeds per arm: the prediction was
too conservative. 0.01 and 0.005 did not collapse on any seed. The collapsed
runs at 0 harvest ~0.055 m of goal progress per step, not 0.12, and with that
rate in place of v_max*dt the floor is ~0.005 on the slalom and ~0.018 on the
tunnel. v_max*dt is not a hard ceiling on the progress either: the
environment limits speed per axis, so diagonal motion covers slightly more,
and runs log up to ~0.126.)

The coefficient scales a second effect independently of the floor. A wall
contact costs the worker coef * contact_penalty: -1.0 at 0.02, -0.25 at 0.005.
That is the worker's own reason to avoid walls. The success-bonus arm had no
such reason: it gave the same final policy as the extrinsic mix at 3-5x the
samples (266k-399k steps against 71k-81k, docs/worker-termination-avoidance.md
section 5.3). So sample efficiency should also degrade toward that as the
coefficient falls, even above the floor. (The first sweep did not show it:
first-solve medians were 72k / 67k / 72k / 82k from 0.02 down to 0.005, p >=
0.46 against 0.02, with zero contacts at the end in every arm. A contact
penalty of -0.25 is still enough.)

Does the extrinsic term swamp the intrinsic one?
-------------------------------------------------
Magnitude alone does not answer it. On a clean run the dense extrinsic terms
almost cancel: a step costs coef*1 and full-speed progress pays
coef*10*v_max*dt, about coef*1.2. What remains is sparse: the terminal (+20
at 0.02) and the contacts. The dense per-step signal is still the intrinsic
one. The relevant question is behavioural, so each arm reports three probes.
Each is measured on every seed's final.pt, from the 25 starts of the
solved-check grid, under the deterministic controller that evaluation uses
(make_policy_fn in algorithms/hppo/hppo_train.py):

- Reward composition. The extrinsic share of the worker's reward,
  sum|coef*r_env| / (sum|r_int| + sum|coef*r_env|), over the final policy's
  steps. The same quantity is logged by the training loop at every update
  (worker/extrinsic_share, next to worker/reward_intrinsic, the worker's goal
  progress per step), so the learning-curve figure shows how it moves over
  training.
- Goal share of the worker's action variance. At every state the final
  policy visits, the worker's deterministic action is evaluated under
  GOAL_SHARE_GOALS goals drawn uniformly from the manager's box. By the law of
  total variance the action variance splits into a part the goal explains
  (the mean over states of the variance across goals) and a part the state
  explains (the variance over states of the mean across goals). The goal
  share is the first part's fraction: 1 is a worker driven only by its goal,
  0 is a worker that ignores its goal. The box includes goals the manager
  never emits, such as backward ones. That is deliberate: a worker that
  ignores its goals ignores them everywhere, and sampling from the manager's
  own repertoire would confound a narrow manager with a deaf worker.
- Manager knock-outs. The same grid replayed with the manager's goal
  replaced at every re-plan:
    forward:  a fixed (max_goal_bound, 0), straight down the corridor. Does
              the worker still need its manager to thread the gates?
    mirrored: the manager's own goal with goal_y negated, which is the one
              decision the slalom's offset gates turn on. Does the manager
              still steer the worker?
  Each is scored as the mean return gap to the oracle over the grid, the
  same figure as the hierarchy's own.

Reading them together: a worker that has made its manager ornamental scores a
knock-out gap close to the hierarchy's own and a goal share near 0. A feudal
worker loses badly under both knock-outs. Some resistance to the mirrored
goal is expected even from a healthy worker, because avoiding walls on its
own is exactly what the extrinsic term teaches it. What matters is how that
resistance moves across the arms.

Color in these figures encodes the coefficient: an ordinal blue ramp, darkest
at the default. Unlike algorithms/study_plots.py, it does not encode the
algorithm, since every line here is hPPO. The ramp is the reference
palette's validated sequential blue, used from step 250 up, as that palette
prescribes for ordinal marks on a light surface.
"""
import argparse
import os
import pickle
import subprocess
import sys
import time
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# Scenario root, for `envs`; then algorithms/hppo and algorithms/, for the flat
# modules imported by bare name -- as in study_hppo.py, whose Study this reuses.
sys.path.append(os.path.abspath(os.path.join(HERE, '..')))
sys.path.append(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'algorithms', 'hppo')))
sys.path.append(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'algorithms')))
from study_hppo import STUDY
from hppo_train import load_agent, make_policy_fn, worker_input
from study import find_run, rollout, oracle_rollout, _first_solve, _parse_seeds
from metrics_log import read_config, read_metrics, series
from optimal_solver import MinTimeSolver, spawn_grid


DEFAULT_COEFS = "0.02,0.015,0.01,0.005,0"
DEFAULT_COEF = 0.02          # script_hppo.py's default: the reference arm
# The full-budget protocol (docs/benchmark.md): the collapse this sweep looks
# for binds late, so every run gets the whole budget with every early stop off.
DEFAULT_TOTAL_TIMESTEPS = 500_000
DEFAULT_OUT = os.path.abspath(os.path.join(HERE, '..', 'studies', 'extrinsic_sweep'))

# Goals per visited state for the goal share, drawn once from a fixed seed so
# every checkpoint is probed with the same goals.
GOAL_SHARE_GOALS = 32
GOAL_SHARE_SEED = 0

# Keys read back from each run's metrics.jsonl: the eval-time ones are logged
# every --eval-freq updates, the update-time ones every update.
EVAL_KEYS = ["eval/episodic_return", "eval/success_rate", "eval/collision_count_mean",
             "eval/episodic_length", "solved/mean_gap", "solved/is_solved"]
UPDATE_KEYS = ["worker/reward_intrinsic", "worker/reward_extrinsic", "worker/extrinsic_share",
               "charts/worker_ax_near_goal"]


def arm_name(coef):
    return f"coef_{coef:g}"


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="hPPO on the slalom across --worker-extrinsic-coef: train every (coef, seed) "
                    "with early stops off, then plot performance and the hierarchy probes. Any "
                    "argument not listed here is passed through to script_hppo.py.")
    parser.add_argument("phase", nargs="?", default="all", choices=["all", "train", "analyze", "plot"],
        help="train: run every (coef, seed) (skips finished ones); analyze: read the curves, "
             "probe every final.pt, then plot; plot: redraw from analysis.pkl; "
             "all (default): train, then analyze")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT,
        help="sweep directory: coef_<value>/, figures/, analysis.pkl and summary.md go here")
    parser.add_argument("--coefs", type=str, default=DEFAULT_COEFS,
        help="comma-separated --worker-extrinsic-coef values, one arm each. The default brackets "
             "the derived floor (~0.011 on the slalom; see this file's docstring)")
    parser.add_argument("--seeds", type=str, default="1-10",
        help="seeds per arm, e.g. '1-10' or '1,2,5'. Fewer than ~10 cannot tell a "
             "sample-efficiency difference from seed noise on this task (docs/benchmark.md); "
             "whether an arm collapses at all shows with fewer")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS,
        help="training budget of every run")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
        help="runs trained in parallel, each pinned to one thread")
    args, passthrough = parser.parse_known_args()
    args.coef_list = sorted({float(c) for c in args.coefs.split(",") if c.strip()}, reverse=True)
    args.seed_list = _parse_seeds(args.seeds)
    if passthrough and args.phase not in ("all", "train"):
        parser.error(f"unrecognized arguments for the {args.phase} phase: {' '.join(passthrough)}")
    if any(a.startswith("--worker-extrinsic-coef") for a in passthrough):
        parser.error("the coefficient is this sweep's variable: pass it through --coefs")
    return args, passthrough


def main():
    args, passthrough = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.phase in ("all", "train"):
        train_phase(args, passthrough)
    if args.phase in ("all", "analyze"):
        analysis = analyze_phase(args)
        plot_sweep(analysis, args.out)
    if args.phase == "plot":
        plot_sweep(load_analysis(args.out), args.out)


# --------------------------------------------------------------------------
# train phase
# --------------------------------------------------------------------------

def find_arm_run(out, coef, seed, total_timesteps):
    """The newest finished run of `seed` in `coef`'s arm, or None. The
    coefficient is checked against the run's own config, not only trusted to
    the directory it sits in."""
    run_dir = find_run(os.path.join(out, arm_name(coef), "runs"), STUDY, seed, total_timesteps)
    if run_dir is not None and float(read_config(run_dir)["worker_extrinsic_coef"]) != coef:
        sys.exit(f"{run_dir} was trained at --worker-extrinsic-coef "
                 f"{read_config(run_dir)['worker_extrinsic_coef']}, not {coef}: remove it or fix --out")
    return run_dir


def train_phase(args, passthrough):
    """study.train_phase over (coef, seed) pairs instead of seeds: one queue for
    the whole sweep, so no arm's last runs leave cores idle while the next arm
    waits. Seed-major order, so a sweep stopped part-way leaves every arm with
    the same seeds."""
    tasks = [(coef, seed) for seed in args.seed_list for coef in args.coef_list]
    todo = [t for t in tasks if find_arm_run(args.out, *t, args.total_timesteps) is None]
    if len(todo) < len(tasks):
        print(f"[sweep] {len(tasks) - len(todo)} of {len(tasks)} runs already finished at "
              f"{args.total_timesteps} steps, skipped")
    if not todo:
        return

    # One thread per trainer: without the cap every process starts PyTorch's
    # full-core thread pool and a parallel batch oversubscribes the CPU.
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    jobs = max(1, min(args.jobs, len(todo)))
    print(f"[sweep] training {len(todo)} runs (coefs {args.coef_list}, seeds {args.seed_list}) at "
          f"{args.total_timesteps} steps, {jobs} at a time")

    pending = list(todo)
    running = {}  # (coef, seed) -> (Popen, log file, start time)
    failed = []
    t0 = time.time()
    try:
        while pending or running:
            while pending and len(running) < jobs:
                coef, seed = pending.pop(0)
                arm_dir = os.path.join(args.out, arm_name(coef))
                os.makedirs(os.path.join(arm_dir, "logs"), exist_ok=True)
                cmd = [sys.executable, "-u", STUDY.train_script,
                       "--seed", str(seed),
                       "--exp-name", STUDY.exp_name,
                       "--total-timesteps", str(args.total_timesteps),
                       "--worker-extrinsic-coef", repr(coef),
                       # Every run takes the whole budget (see the docstring);
                       # the first solve is still logged, and saved as solved.pt.
                       "--solved-early-stop", "false",
                       "--checkpoint-dir", os.path.join(arm_dir, "runs"),
                       *passthrough]
                log_file = open(os.path.join(arm_dir, "logs", f"seed_{seed}.log"), "w")
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env,
                                        cwd=os.path.dirname(STUDY.train_script))
                running[(coef, seed)] = (proc, log_file, time.time())
            for (coef, seed), (proc, log_file, started) in list(running.items()):
                if proc.poll() is None:
                    continue
                log_file.close()
                del running[(coef, seed)]
                minutes = (time.time() - started) / 60
                finished = len(todo) - len(pending) - len(running)
                if proc.returncode != 0:
                    failed.append((coef, seed))
                    print(f"[sweep] coef {coef:g} seed {seed} FAILED (exit {proc.returncode}) after "
                          f"{minutes:.1f} min, see {arm_name(coef)}/logs/seed_{seed}.log")
                    continue
                run_dir = find_arm_run(args.out, coef, seed, args.total_timesteps)
                first = _first_solve(read_metrics(run_dir)) if run_dir else None
                print(f"[sweep] coef {coef:g} seed {seed} done in {minutes:.1f} min "
                      f"({finished}/{len(todo)}), first solve: "
                      f"{'%.0fk steps' % (first / 1e3) if first else 'not within budget'}")
            time.sleep(1.0)
    finally:
        for proc, log_file, _ in running.values():
            proc.terminate()
            log_file.close()
    print(f"[sweep] training finished in {(time.time() - t0) / 60:.1f} min"
          + (f"; failed: {failed}" if failed else ""))


# --------------------------------------------------------------------------
# analyze phase: learning curves
# --------------------------------------------------------------------------

def _late_precision(records):
    """Share of the strict solved-checks passed *after* the first solve (the
    full-budget protocol's late-run precision), or None if never solved."""
    first = _first_solve(records)
    if first is None:
        return None
    steps, solved = series(records, "solved/is_solved")
    after = [s for step, s in zip(steps, solved) if step > first]
    return float(np.mean(after)) if after else None


def read_curves(records):
    """key -> (steps, values) for every key this sweep plots; a key the run
    never logged (the worker/ reward keys predate 2026-09-24) is left out."""
    curves = {}
    for key in EVAL_KEYS + UPDATE_KEYS:
        steps, values = series(records, key)
        if steps:
            curves[key] = (np.asarray(steps, dtype=float), np.asarray(values, dtype=float))
    return curves


# --------------------------------------------------------------------------
# analyze phase: probes of a final checkpoint
# --------------------------------------------------------------------------

def _grid_score(rollouts, oracle_grid):
    return {
        "gap": float(np.mean([o["return"] - r["return"] for r, o in zip(rollouts, oracle_grid)])),
        "success": int(sum(r["success"] for r in rollouts)),
        "contacts": float(np.mean([r["contacts"].sum() for r in rollouts])),
        "length": float(np.mean([r["length"] for r in rollouts])),
    }


def reward_composition(rollouts, coef, gamma):
    """What the worker's reward was made of along these rollouts: the
    intrinsic goal-closing term against coef * the environment's reward.

    The intrinsic term is rebuilt exactly as the training loop computes it,
    from the goal the worker acted on at each step and the displacement that
    step produced: ||g_t|| - ||g_t - (p_{t+1} - p_t)||."""
    r_int_all, r_ext_all, g_int, g_ext = [], [], [], []
    for ro in rollouts:
        goals = ro["goals"]
        step = np.diff(ro["states"][:, :2], axis=0)
        r_int = np.linalg.norm(goals, axis=-1) - np.linalg.norm(goals - step, axis=-1)
        r_ext = coef * ro["rewards"]
        discount = gamma ** np.arange(len(r_int))
        r_int_all.append(r_int)
        r_ext_all.append(r_ext)
        g_int.append(float(discount @ r_int))
        g_ext.append(float(discount @ r_ext))
    r_int_all, r_ext_all = np.concatenate(r_int_all), np.concatenate(r_ext_all)
    abs_int, abs_ext = np.abs(r_int_all).sum(), np.abs(r_ext_all).sum()
    return {
        "intrinsic_per_step": float(r_int_all.mean()),
        "extrinsic_per_step": float(r_ext_all.mean()),
        "extrinsic_share": float(abs_ext / max(abs_int + abs_ext, 1e-12)),
        "return_intrinsic": float(np.mean(g_int)),   # discounted, from the start state
        "return_extrinsic": float(np.mean(g_ext)),
    }


def goal_share(agent, rollouts, goals):
    """Fraction of the worker's action variance, over the states `rollouts`
    visited and the given `goals`, that the goal explains; see the docstring."""
    states = np.concatenate([ro["states"][:-1] for ro in rollouts])
    obs_norm = agent.normalize_obs(states)
    n, k = len(states), len(goals)
    obs_rep = np.repeat(obs_norm, k, axis=0)            # (n*k, obs_dim), state-major
    goal_rep = np.tile(goals, (n, 1))                   # (n*k, 2)
    with torch.no_grad():
        actions = agent.worker_act(worker_input(agent, obs_rep, goal_rep)).cpu().numpy()
    actions = actions.reshape(n, k, -1)
    within = actions.var(axis=1).mean(axis=0).sum()     # E_s[Var_g a]
    between = actions.mean(axis=1).var(axis=0).sum()    # Var_s[E_g a]
    return float(within / max(within + between, 1e-12))


def probe_checkpoint(path, env, grid, oracle_grid, coef, gamma):
    agent = load_agent(path, env)
    bound = agent.max_goal_bound
    knockouts = {
        "manager": None,
        "forward": lambda g: np.array([bound, 0.0]),
        "mirrored": lambda g: g * np.array([1.0, -1.0]),
    }
    probe, manager_rollouts = {}, None
    for mode, replan_goal in knockouts.items():
        def factory(goal_trace=None, replan_goal=replan_goal):
            return make_policy_fn(agent, goal_trace, replan_goal=replan_goal)
        rollouts = [rollout(env, factory, point) for point in grid]
        probe[mode] = _grid_score(rollouts, oracle_grid)
        if mode == "manager":
            manager_rollouts = rollouts
    probe["reward"] = reward_composition(manager_rollouts, coef, gamma)
    rng = np.random.default_rng(GOAL_SHARE_SEED)
    probe["goal_share"] = goal_share(
        agent, manager_rollouts, rng.uniform(-bound, bound, size=(GOAL_SHARE_GOALS, 2)))
    return probe


# --------------------------------------------------------------------------
# analyze phase
# --------------------------------------------------------------------------

def extrinsic_floor(env_config, gamma):
    """The derived coefficient floor of the docstring, for this environment."""
    farm = env_config.v_max * env_config.dt / (1.0 - gamma)
    return farm / (env_config.goal_reward + abs(env_config.step_penalty) / (1.0 - gamma))


def analyze_phase(args):
    arms = {}
    for coef in args.coef_list:
        runs = {s: find_arm_run(args.out, coef, s, args.total_timesteps) for s in args.seed_list}
        missing = [s for s, r in runs.items() if r is None]
        if missing:
            print(f"[sweep] coef {coef:g}: no finished run at {args.total_timesteps} steps for "
                  f"seeds {missing}, left out")
        runs = {s: r for s, r in runs.items() if r is not None}
        if runs:
            arms[coef] = runs
    if not arms:
        sys.exit(f"[sweep] nothing to analyze in {args.out} at {args.total_timesteps} steps "
                 f"(pass the --total-timesteps the runs were trained with)")

    config = read_config(next(iter(next(iter(arms.values())).values())))
    env = STUDY.make_env(config)
    gamma = float(config["gamma"])
    solver = MinTimeSolver()
    grid = spawn_grid(env, int(config["solved_grid_nx"]), int(config["solved_grid_ny"]))
    print(f"[sweep] oracle on the {len(grid)}-point start grid")
    oracle_grid = [oracle_rollout(env, solver, point) for point in grid]

    results = {}
    for coef, runs in arms.items():
        print(f"[sweep] coef {coef:g}: reading curves and probing final.pt of seeds {sorted(runs)}")
        results[coef] = {}
        for seed, run_dir in sorted(runs.items()):
            records = read_metrics(run_dir)
            curves = read_curves(records)
            results[coef][seed] = {
                "run": run_dir,
                "curves": curves,
                "first_solve": _first_solve(records),
                "late_precision": _late_precision(records),
                "solved_at_end": bool(curves["solved/is_solved"][1][-1]) if "solved/is_solved" in curves else False,
                "probe": probe_checkpoint(os.path.join(run_dir, "final.pt"), env, grid, oracle_grid,
                                          coef, gamma),
            }

    analysis = {
        "coefs": sorted(arms, reverse=True),
        "results": results,
        "total_timesteps": args.total_timesteps,
        "config": config,
        "env_config": env.config,
        "gamma": gamma,
        "floor": extrinsic_floor(env.config, gamma),
        "oracle_grid_return": float(np.mean([o["return"] for o in oracle_grid])),
        "max_goal_bound": float(config["max_goal_bound"]),
    }
    path = os.path.join(args.out, "analysis.pkl")
    with open(path, "wb") as f:
        pickle.dump(analysis, f)
    print(f"[sweep] saved {path}")
    return analysis


def load_analysis(out_dir):
    path = os.path.join(out_dir, "analysis.pkl")
    if not os.path.exists(path):
        sys.exit(f"no analysis.pkl in {out_dir}: run the analyze phase first")
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def _reference_coef(analysis):
    coefs = analysis["coefs"]
    return DEFAULT_COEF if DEFAULT_COEF in coefs else max(coefs)


def _first_solve_p_value(analysis, coef):
    """Two-sided Mann-Whitney U on first-solve steps against the reference arm,
    a seed that never solved ranking after every one that did."""
    from scipy.stats import mannwhitneyu
    never = analysis["total_timesteps"] + 1
    def steps(c):
        return [r["first_solve"] if r["first_solve"] is not None else never
                for r in analysis["results"][c].values()]
    a, b = steps(coef), steps(_reference_coef(analysis))
    if len(a) < 2 or len(b) < 2 or len(set(a + b)) == 1:
        return None
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def arm_rows(analysis):
    rows = []
    cfg = analysis["env_config"]
    for coef in analysis["coefs"]:
        seeds = analysis["results"][coef]
        firsts = [r["first_solve"] for r in seeds.values() if r["first_solve"] is not None]
        late = [r["late_precision"] for r in seeds.values() if r["late_precision"] is not None]
        probe = lambda f: float(np.median([f(r["probe"]) for r in seeds.values()]))
        rows.append({
            "coef": coef,
            "terminal": coef * cfg.goal_reward,
            "contact": coef * cfg.contact_penalty,
            "n": len(seeds),
            "solved": len(firsts),
            "first_median": float(np.median(firsts)) if firsts else None,
            "first_range": (min(firsts), max(firsts)) if firsts else None,
            "p_value": None if coef == _reference_coef(analysis) else _first_solve_p_value(analysis, coef),
            "late_precision": float(np.mean(late)) if late else None,
            "solved_at_end": sum(r["solved_at_end"] for r in seeds.values()),
            "gap": probe(lambda p: p["manager"]["gap"]),
            "contacts": probe(lambda p: p["manager"]["contacts"]),
            "reached": probe(lambda p: p["manager"]["success"]),
            "extrinsic_share": probe(lambda p: p["reward"]["extrinsic_share"]),
            "goal_share": probe(lambda p: p["goal_share"]),
            "gap_forward": probe(lambda p: p["forward"]["gap"]),
            "gap_mirrored": probe(lambda p: p["mirrored"]["gap"]),
        })
    return rows


def write_summary(analysis, out_dir):
    k = lambda v: "-" if v is None else f"{v / 1e3:.0f}k"
    pct = lambda v: "-" if v is None else f"{100 * v:.0f} %"
    ref = _reference_coef(analysis)
    lines = [
        "# hPPO on the slalom across --worker-extrinsic-coef",
        "",
        f"{analysis['total_timesteps']} steps per run, early stops off. Conservative floor of the "
        f"coefficient (terminal = value of stalling): **{analysis['floor']:.4f}**; see the "
        f"docstring of scenarios/slalom/scripts/sweep_extrinsic_coef.py. Probes replay each "
        f"seed's `final.pt` from the 25 points of the solved-check grid (oracle mean return "
        f"{analysis['oracle_grid_return']:.1f}); a gap is oracle return - agent return, "
        f"averaged over the grid. Probe columns are medians over seeds. p: two-sided "
        f"Mann-Whitney U on first-solve steps against coef {ref:g}, unsolved seeds ranked last.",
        "",
        "| coef | terminal / contact (worker units) | solved | first solve, median of solved seeds (range) | p "
        "| solved-checks passed after first solve | solved at end | grid gap | contacts/ep "
        "| extrinsic share of reward | goal share of action var. | gap, forward goal "
        "| gap, mirrored goal |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in arm_rows(analysis):
        rng = "" if r["first_range"] is None else f" ({k(r['first_range'][0])}-{k(r['first_range'][1])})"
        p = "-" if r["p_value"] is None else f"{r['p_value']:.2f}"
        lines.append(
            f"| {r['coef']:g}{' (default)' if r['coef'] == DEFAULT_COEF else ''} "
            f"| +{r['terminal']:.1f} / {r['contact']:.2f} | {r['solved']}/{r['n']} "
            f"| {k(r['first_median'])}{rng} | {p} "
            f"| {pct(r['late_precision'])} | {r['solved_at_end']}/{r['n']} | {r['gap']:.1f} "
            f"| {r['contacts']:.2f} | {pct(r['extrinsic_share'])} | {pct(r['goal_share'])} "
            f"| {r['gap_forward']:.1f} | {r['gap_mirrored']:.1f} |")
    lines += ["", "## Per seed", "",
              "| coef | seed | first solve | checks passed after | solved at end | grid gap | reached goal "
              "| contacts/ep | extrinsic share | goal share | gap, forward | gap, mirrored |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for coef in analysis["coefs"]:
        for seed, r in sorted(analysis["results"][coef].items()):
            p = r["probe"]
            lines.append(
                f"| {coef:g} | {seed} | {k(r['first_solve'])} | {pct(r['late_precision'])} "
                f"| {'yes' if r['solved_at_end'] else 'no'} | {p['manager']['gap']:.1f} "
                f"| {p['manager']['success']}/25 | {p['manager']['contacts']:.2f} "
                f"| {pct(p['reward']['extrinsic_share'])} | {pct(p['goal_share'])} "
                f"| {p['forward']['gap']:.1f} | {p['mirrored']['gap']:.1f} |")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[sweep] wrote {os.path.join(out_dir, 'summary.md')}")


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

# The reference palette's sequential blue, steps 250-650: the ordinal range on
# a light surface (see the docstring). Arms take evenly spaced steps from it.
ORDINAL_BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                "#256abf", "#1c5cab", "#184f95", "#104281"]

# Per-update series are smoothed over this many updates before plotting.
SMOOTH_UPDATES = 5


def arm_colors(coefs):
    """coef -> color, lightest for the smallest coefficient."""
    order = sorted(coefs)
    if len(order) > len(ORDINAL_BLUE):
        raise ValueError(f"at most {len(ORDINAL_BLUE)} arms can be told apart on one ramp")
    if len(order) == 1:
        return {order[0]: ORDINAL_BLUE[-1]}
    idx = np.round(np.linspace(0, len(ORDINAL_BLUE) - 1, len(order))).astype(int)
    return {c: ORDINAL_BLUE[i] for c, i in zip(order, idx)}


def _arm_label(coef):
    return f"coef = {coef:g}" + (" (default)" if coef == DEFAULT_COEF else "")


def _band(seed_results, key):
    """(steps, median, q25, q75) over seeds of one logged series, on the union
    of the steps any seed logged it at; None if no seed logged it."""
    logged = [r["curves"][key] for r in seed_results if key in r["curves"]]
    if not logged:
        return None
    steps = np.unique(np.concatenate([s for s, _ in logged]))
    values = np.full((len(logged), len(steps)), np.nan)
    for i, (s, v) in enumerate(logged):
        values[i, np.searchsorted(steps, s)] = v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # a step no seed logged
        median = np.nanmedian(values, axis=0)
        q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
    return steps, median, q25, q75


def _smooth(y, window):
    """Centred moving average that skips NaNs (updates without a sample)."""
    if window <= 1:
        return y
    kernel = np.ones(window)
    valid = ~np.isnan(y)
    num = np.convolve(np.where(valid, y, 0.0), kernel, mode="same")
    den = np.convolve(valid.astype(float), kernel, mode="same")
    return np.where(den > 0, num / np.maximum(den, 1.0), np.nan)


def fig_learning_curves(analysis, fig_dir):
    import matplotlib.pyplot as plt
    from study_plots import STYLE, INK, MUTED, STEPS_FMT, GAP_LINTHRESH, _save
    colors = arm_colors(analysis["coefs"])
    v_step = analysis["env_config"].v_max * analysis["env_config"].dt
    panels = [
        # key, title, y label, symlog, smoothing window
        ("eval/episodic_return", "(a) Evaluation return", "return", False, 1),
        ("solved/mean_gap", "(b) Gap to the oracle, solved-check grid", "mean return gap", True, 1),
        ("eval/success_rate", "(c) Evaluation success rate", "success rate", False, 1),
        ("eval/collision_count_mean", "(d) Wall contacts per evaluation episode", "contacts / episode", False, 1),
        ("worker/reward_intrinsic", "(e) Worker's goal progress per step", "intrinsic reward (m)", False, SMOOTH_UPDATES),
        ("worker/extrinsic_share", "(f) Extrinsic share of the worker's reward", "share of |reward|", False, SMOOTH_UPDATES),
    ]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.6), sharex=True)
        for ax, (key, title, ylabel, symlog, window) in zip(axes.flat, panels):
            drawn = False
            for coef in sorted(analysis["coefs"]):
                band = _band(analysis["results"][coef].values(), key)
                if band is None:
                    continue
                steps, median, q25, q75 = band
                median, q25, q75 = (_smooth(y, window) for y in (median, q25, q75))
                ax.fill_between(steps, q25, q75, color=colors[coef], alpha=0.15, lw=0)
                ax.plot(steps, median, color=colors[coef], lw=2 if coef == DEFAULT_COEF else 1.4,
                        label=_arm_label(coef))
                drawn = True
            if not drawn:
                ax.text(0.5, 0.5, "not logged by these runs\n(predates 2026-09-24)", ha="center",
                        va="center", color=MUTED, transform=ax.transAxes)
            if symlog:
                ax.set_yscale("symlog", linthresh=GAP_LINTHRESH)
                ax.axhline(0.0, color=INK, lw=0.8, ls="--")
            if key == "worker/reward_intrinsic":
                ax.axhline(v_step, color=MUTED, lw=0.8, ls="--")
                ax.text(0.99, v_step, "$v_{max}\\,dt$ (per axis)", color=MUTED, ha="right", va="bottom",
                        transform=ax.get_yaxis_transform(), fontsize=8)
            if key in ("eval/success_rate", "worker/extrinsic_share"):
                ax.set_ylim(-0.03, 1.03)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_formatter(STEPS_FMT)
        for ax in axes[-1]:
            ax.set_xlabel("environment steps")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout()
        _save(fig, fig_dir, "sweep_learning_curves")


def fig_dose_response(analysis, fig_dir):
    """Per-seed dots and the median (a black tick) per arm, the arms in
    ascending order of the coefficient, the derived floor as a dashed line."""
    import matplotlib.pyplot as plt
    from study_plots import STYLE, INK, MUTED, STEPS_FMT, GAP_LINTHRESH, _save
    coefs = sorted(analysis["coefs"])
    colors = arm_colors(coefs)
    budget = analysis["total_timesteps"]
    never = budget * 1.08
    panels = [
        # value of one seed's result, title, y label, kind
        (lambda r: r["first_solve"] if r["first_solve"] is not None else never,
         "(a) First solve", "environment steps", "steps"),
        (lambda r: None if r["late_precision"] is None else 100 * r["late_precision"],
         "(b) Solved-checks passed after it", "%", "pct"),
        (lambda r: r["probe"]["manager"]["gap"], "(c) Gap to the oracle", "mean return gap", "gap"),
        (lambda r: r["probe"]["manager"]["contacts"], "(d) Wall contacts", "contacts / episode", "plain"),
        (lambda r: 100 * r["probe"]["reward"]["extrinsic_share"],
         "(e) Extrinsic share of reward", "% of the worker's |reward|", "pct"),
        (lambda r: 100 * r["probe"]["goal_share"], "(f) Goal share of action var.", "% of the worker's action variance", "pct"),
        (lambda r: r["probe"]["forward"]["gap"], "(g) Knock-out: forward goal", "mean return gap", "gap"),
        (lambda r: r["probe"]["mirrored"]["gap"], "(h) Knock-out: mirrored goal", "mean return gap", "gap"),
    ]
    # Where the floor falls between the categorical arm positions.
    floor_x = None
    if coefs[0] < analysis["floor"] < coefs[-1]:
        floor_x = float(np.interp(analysis["floor"], coefs, np.arange(len(coefs))))
    gaps = [f(r) for f, _, _, kind in panels if kind == "gap"
            for c in coefs for r in analysis["results"][c].values()]
    gap_lim = (min(-GAP_LINTHRESH, min(gaps) * 1.3), max(GAP_LINTHRESH, max(gaps) * 1.3))

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(12.8, 5.8), sharex=True)
        for ax, (value, title, ylabel, kind) in zip(axes.flat, panels):
            if all(value(r) is None for c in coefs for r in analysis["results"][c].values()):
                ax.text(0.5, 0.5, "no seed solved", ha="center", va="center", color=MUTED,
                        transform=ax.transAxes)
            for x, coef in enumerate(coefs):
                seeds = sorted(analysis["results"][coef])
                vals = [value(analysis["results"][coef][s]) for s in seeds]
                jitter = np.linspace(-0.18, 0.18, len(vals)) if len(vals) > 1 else [0.0]
                for dx, v in zip(jitter, vals):
                    if v is None:
                        continue
                    hollow = kind == "steps" and v == never
                    ax.plot(x + dx, v, "o", ms=5, mec=colors[coef] if hollow else "white",
                            mfc="white" if hollow else colors[coef], mew=1.2, zorder=3)
                # For the first solve an unsolved seed counts, as beyond the
                # budget: a median over the solved seeds alone would flatter
                # an arm that mostly failed.
                kept = [v for v in vals if v is not None]
                if kept:
                    ax.plot([x - 0.28, x + 0.28], [np.median(kept)] * 2, color=INK, lw=2, zorder=4)
            if floor_x is not None:
                ax.axvline(floor_x, color=MUTED, lw=0.8, ls="--", zorder=1)
            if kind == "steps":
                ax.axhline(budget, color=MUTED, lw=0.8)
                ax.text(0.01, budget, "not solved above", color=MUTED, va="bottom", fontsize=7,
                        transform=ax.get_yaxis_transform())
                ax.yaxis.set_major_formatter(STEPS_FMT)
                ax.set_ylim(0, budget * 1.15)
            elif kind == "pct":
                ax.set_ylim(-3, 103)
            elif kind == "gap":
                ax.set_yscale("symlog", linthresh=GAP_LINTHRESH)
                ax.set_ylim(*gap_lim)
                ax.axhline(0.0, color=INK, lw=0.8, ls="--")
            ax.set_title(title)
            ax.set_ylabel(ylabel)
        for ax in axes[-1]:
            ax.set_xticks(range(len(coefs)))
            ax.set_xticklabels([f"{c:g}" + ("\ndefault" if c == DEFAULT_COEF else "") for c in coefs])
            ax.set_xlabel("--worker-extrinsic-coef")
        if floor_x is not None:
            axes.flat[0].text(floor_x, 0.99, f" conservative floor ~ {analysis['floor']:.3f}", color=MUTED, fontsize=7,
                              va="top", transform=axes.flat[0].get_xaxis_transform())
        fig.tight_layout()
        _save(fig, fig_dir, "sweep_dose_response")


def plot_sweep(analysis, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    fig_learning_curves(analysis, fig_dir)
    fig_dose_response(analysis, fig_dir)
    write_summary(analysis, out_dir)


if __name__ == "__main__":
    main()
