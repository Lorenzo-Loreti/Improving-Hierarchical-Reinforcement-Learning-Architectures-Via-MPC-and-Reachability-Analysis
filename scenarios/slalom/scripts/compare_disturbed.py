"""PPO+MPC, hPPO and flat PPO on the *disturbed* slalom: the same seeds, the
same bounded disturbance, the full budget.

    python scenarios/slalom/scripts/compare_disturbed.py            # train every (algorithm, seed), then summarize
    python scenarios/slalom/scripts/compare_disturbed.py train      # training only (resumable)
    python scenarios/slalom/scripts/compare_disturbed.py summary    # after training: the table
    python scenarios/slalom/scripts/compare_disturbed.py --seeds 1-3 --algos ppo_mpc,hppo

Everything goes to scenarios/slalom/studies/disturbed_<p>_<v>/, which git
ignores: one <algorithm>/ directory per arm (its runs/ and logs/), plus
summary.md.

The question
------------
Every result in this repo before 2026-09-28 is on the deterministic slalom.
The PPO+MPC rebuilt that day (algorithms/ppo_mpc, algorithms/tube_mpc.py)
has a worker designed for a bounded disturbance w in W: its tube keeps the
real state off every wall for any w in W, at the price of a slower, wider
plan (top planned speed 0.99 m/s instead of 1.2 at the level used here).
The two learned policies have no such guarantee: they were tuned on an
environment where the shortest path grazes the gates (the oracle takes gate
1 about 4 cm from its edge), and a disturbance can push that path into a
wall. So the comparison is safety against speed:

- contacts, in evaluation and -- the part a tube should change most -- during
  training, where the manager's goals are still being sampled;
- the return, and its gap to the undisturbed oracle, which no controller
  facing the disturbance should be expected to close;
- how soon each reaches a clean evaluation: every episode at the goal, no
  contact.

Protocol: the full-budget one of docs/benchmark.md -- 500k steps with every
early stop off, since the solved criterion is measured against the
undisturbed oracle, which a robust controller was expected to miss.
The disturbance is the level chosen on 2026-09-28, |w_p| <= 0.005 m and
|w_v| <= 0.05 m/s per step, uniform, passed to every algorithm through the
same --noise-bound-p/-v flags. Evaluation runs on the disturbed environment
with fixed seeds, so every evaluation of every run faces the same draws.

This reads each run's metrics.jsonl directly. algorithms/study.py cannot be
used: it replays the oracle and the policies on one environment, and refuses
disturbed runs for that reason.

Result (2026-09-28, seeds 1-10 each, 500k steps; summary.md has the rest)
-------------------------------------------------------------------------
                              PPO+MPC          hPPO                    PPO
  eval return, last 100k      1008.3           1012.5  (p = 2e-4)      1012.8  (p = 2e-4)
  mean grid gap to oracle     4.6              0.3                     -0.1
  eval contacts/episode       0                0                       0 (one seed 0.01)
  training contacts/episode   0 (whole run)    0.55 (whole run)        0.31 (whole run)
                              0 (last 100k)    0.11 (last 100k)        0.02 (last 100k)
  first clean evaluation      46k              61k     (p = 0.004)     41k     (p = 0.10)
  strict solved-check         never            10/10, median 72k       10/10, median 51k

- The tube holds, and not only in evaluation. PPO+MPC had no contact in
  any training episode of any seed, across 500k steps of sampled manager
  goals, with no candidate fallback and no emergency in the whole study.
  The learned policies averaged 0.31 (PPO) and 0.55 (hPPO) wall contacts
  per training episode over the run, with a tail of 0.02 and 0.11 per
  episode in its last 100k steps.
- The prediction that the disturbance would push the learned policies into
  the gates is refuted, in evaluation. Trained on the disturbed
  environment, flat PPO and hPPO learned margins of their own: no contact
  in evaluation, a gap to the undisturbed oracle of ~0, and first solves at
  the same medians as on the deterministic slalom (51k and 72k,
  docs/benchmark.md). At this level of disturbance a learned policy pays
  for robustness during training, in contacts, and not at the end.
- PPO+MPC pays at the end instead. It is 4.5 reward units (~4.5 steps,
  82.0 against ~77.5 per episode) behind the other two on every seed; the
  spread across seeds is 0.1. Its worst grid gap settled at 5.8-6.0,
  just past the solved tolerance of 5. The cost (~6%) is well below what
  the plan's speed cap alone would imply (1.2 / 0.989, ~21% at cruise):
  the real state rides at the front of the tube (see the end of
  tube_mpc.py's docstring).
- It reaches a clean evaluation sooner than hPPO (46k against 61k, p =
  0.004) and no slower than flat PPO (41k, p = 0.10).
- Wall clock: ~121 min per PPO+MPC run with 14 runs in parallel (~10 ms
  per solve under that load, ~4.6 ms alone), against ~14 min for hPPO and
  ~10 min for PPO.

Twice the disturbance (2026-09-29, --noise-bound-p 0.01 --noise-bound-v 0.1)
---------------------------------------------------------------------------
This asks whether the learned policies' own margin stops being enough.
The tube is then +-0.165 m and +-0.42 m/s, the plan is held to 0.78 m/s
and 0.90 m/s^2, and a stop takes 9 of the 10 horizon steps. Three times
the first level (0.015 / 0.15) is out of reach with this ancillary gain:
KZ takes all but 0.107 of the input's 2.5, a stop needs 54 steps, and
even goals placed by hand in the gate openings take 189 steps.

                              PPO+MPC          hPPO                    PPO
  eval return, last 100k      1002.3           1012.0  (p = 2e-4)      1012.4  (p = 2e-4)
  mean grid gap to oracle     10.6             0.7                     0.3
  eval episode length         87.9             78.4                    77.9
  eval contacts/episode       0                0                       0
  training contacts/episode   0 (whole run)    0.61 (whole run)        0.31 (whole run)
  first clean evaluation      56k              67k     (p = 0.08)      46k     (p = 0.03)
  strict solved-check         never            10/10, median 82k       10/10, median 56k

- The learned margins still suffice, against uniform noise. PPO and hPPO
  evaluate without a contact and within ~1 of the undisturbed oracle.
  They solve at 56k and 82k, a little later than at the first level (51k,
  72k), and their training contacts are unchanged (0.31 and 0.61 per
  episode against 0.31 and 0.55).
- The tube's price roughly doubles, from ~4.5 to ~10 steps per episode
  (~13%), and its benefit is the same as before: no contact in training.
  The worker fell back to the candidate on 6 steps out of 5 million, with
  no emergency.
- The evaluation draws w uniformly from W, and the tube is designed for
  the worst w in W. Neither study tests the learned policies against that
  worst case, e.g. w on W's vertices or pushed toward the nearest wall,
  which is the case the guarantee is for.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'algorithms')))
from metrics_log import read_config, read_metrics, series

ALGOS = {
    # name: (training script, exp-name it writes runs under)
    "ppo_mpc": ("script_ppo_mpc.py", "ppo_mpc_slalom"),
    "hppo": ("script_hppo.py", "hppo_slalom"),
    "ppo": ("script_ppo.py", "ppo_slalom"),
}
REFERENCE = "ppo_mpc"
DEFAULT_TOTAL_TIMESTEPS = 500_000
LATE_WINDOW = 100_000      # "the end of the run": its last 100k steps


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", nargs="?", default="all", choices=("all", "train", "summary"),
        help="train: run every (algorithm, seed), skipping finished ones; summary: the table; "
             "all (default): both")
    parser.add_argument("--algos", type=str, default=",".join(ALGOS),
        help="comma-separated algorithms, from " + ", ".join(ALGOS))
    parser.add_argument("--seeds", type=str, default="1-10",
        help="seeds per algorithm, e.g. '1-10' or '1,2,5'")
    parser.add_argument("--noise-bound-p", type=float, default=0.005)
    parser.add_argument("--noise-bound-v", type=float, default=0.05)
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
        help="runs trained in parallel, each pinned to one thread")
    parser.add_argument("--out", type=str, default=None,
        help="study directory (default: scenarios/slalom/studies/disturbed_<p>_<v>)")
    args = parser.parse_args()
    args.algo_list = [a for a in args.algos.split(",") if a]
    unknown = set(args.algo_list) - set(ALGOS)
    if unknown:
        parser.error(f"unknown algorithms: {sorted(unknown)}")
    seeds = []
    for part in args.seeds.split(","):
        lo, _, hi = part.partition("-")
        seeds += list(range(int(lo), int(hi or lo) + 1))
    args.seed_list = seeds
    if args.out is None:
        args.out = os.path.join(HERE, "..", "studies",
                                f"disturbed_{args.noise_bound_p:g}_{args.noise_bound_v:g}")
    # Absolute: the trainers run from this directory and would resolve a
    # relative --checkpoint-dir against it.
    args.out = os.path.abspath(args.out)
    return args


def find_run(args, algo, seed):
    """The newest finished run of `algo` and `seed` at this budget and this
    disturbance, or None. The disturbance is read back from the run's own
    config, not trusted to the directory it sits in."""
    _, exp_name = ALGOS[algo]
    for run_dir in sorted(glob.glob(os.path.join(args.out, algo, "runs", f"{exp_name}_{seed}_*")), reverse=True):
        if not os.path.exists(os.path.join(run_dir, "final.pt")):
            continue
        config = read_config(run_dir)
        if (config.get("total_timesteps") == args.total_timesteps
                and config.get("noise_bound_p") == args.noise_bound_p
                and config.get("noise_bound_v") == args.noise_bound_v):
            return run_dir
    return None


def train_phase(args):
    # Longest first: PPO+MPC solves an MIQP per step and runs ~10x slower
    # than the learned hierarchies, so it starts first and the fast runs fill
    # the cores around it.
    tasks = [(algo, seed) for algo in args.algo_list for seed in args.seed_list]
    todo = [t for t in tasks if find_run(args, *t) is None]
    if len(todo) < len(tasks):
        print(f"[disturbed] {len(tasks) - len(todo)} of {len(tasks)} runs already finished, skipped")
    if not todo:
        return
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    jobs = max(1, min(args.jobs, len(todo)))
    print(f"[disturbed] training {len(todo)} runs at {args.total_timesteps} steps, "
          f"|w_p| <= {args.noise_bound_p}, |w_v| <= {args.noise_bound_v}, {jobs} at a time")
    pending, running, failed = list(todo), {}, []
    t0 = time.time()
    try:
        while pending or running:
            while pending and len(running) < jobs:
                algo, seed = pending.pop(0)
                script, exp_name = ALGOS[algo]
                arm_dir = os.path.join(args.out, algo)
                os.makedirs(os.path.join(arm_dir, "logs"), exist_ok=True)
                cmd = [sys.executable, "-u", os.path.join(HERE, script),
                       "--seed", str(seed), "--exp-name", exp_name,
                       "--total-timesteps", str(args.total_timesteps),
                       "--noise-bound-p", repr(args.noise_bound_p),
                       "--noise-bound-v", repr(args.noise_bound_v),
                       "--solved-early-stop", "false",
                       "--checkpoint-dir", os.path.join(arm_dir, "runs")]
                log_file = open(os.path.join(arm_dir, "logs", f"seed_{seed}.log"), "w")
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env, cwd=HERE)
                running[(algo, seed)] = (proc, log_file, time.time())
            for key, (proc, log_file, started) in list(running.items()):
                if proc.poll() is None:
                    continue
                log_file.close()
                del running[key]
                minutes = (time.time() - started) / 60
                done = len(todo) - len(pending) - len(running)
                if proc.returncode != 0:
                    failed.append(key)
                    print(f"[disturbed] {key[0]} seed {key[1]} FAILED (exit {proc.returncode}) after "
                          f"{minutes:.1f} min, see {key[0]}/logs/seed_{key[1]}.log")
                else:
                    print(f"[disturbed] {key[0]} seed {key[1]} done in {minutes:.1f} min ({done}/{len(todo)})")
            time.sleep(2.0)
    finally:
        for proc, log_file, _ in running.values():
            proc.terminate()
            log_file.close()
    print(f"[disturbed] training finished in {(time.time() - t0) / 60:.1f} min"
          + (f"; failed: {failed}" if failed else ""))


def run_summary(run_dir, total_timesteps):
    """One run's numbers: the end of the run (its last LATE_WINDOW steps),
    the whole run's training contacts, and the first clean evaluation."""
    records = read_metrics(run_dir)
    late_from = total_timesteps - LATE_WINDOW

    def late_mean(key):
        steps, values = series(records, key)
        v = [x for s, x in zip(steps, values) if s > late_from]
        return float(np.mean(v)) if v else float("nan")

    eval_steps, success = series(records, "eval/success_rate")
    _, contacts = series(records, "eval/collision_count_mean")
    first_clean = next((s for s, sr, c in zip(eval_steps, success, contacts) if sr == 1.0 and c == 0.0), None)
    _, train_contacts = series(records, "charts/collision_count_mean")
    out = {
        "return_end": late_mean("eval/episodic_return"),
        "success_end": late_mean("eval/success_rate"),
        "contacts_end": late_mean("eval/collision_count_mean"),
        "gap_end": late_mean("solved/mean_gap"),
        "train_contacts_end": late_mean("charts/collision_count_mean"),
        "train_contacts_all": float(np.mean(train_contacts)) if train_contacts else float("nan"),
        "first_clean": first_clean,
    }
    if series(records, "mpc/candidate_rate")[0]:
        out["candidate_rate"] = float(np.mean(series(records, "mpc/candidate_rate")[1]))
        out["emergencies"] = float(np.sum(series(records, "mpc/emergency_count")[1]))
        out["solve_ms"] = float(np.mean(series(records, "mpc/solve_ms_mean")[1]))
    return out


ROWS = [
    # key, label, format, lower is better
    ("return_end", "eval return, last 100k", "{:.1f}", False),
    ("gap_end", "grid gap to undisturbed oracle, last 100k", "{:.1f}", True),
    ("success_end", "eval success rate, last 100k", "{:.2f}", False),
    ("contacts_end", "eval contacts/episode, last 100k", "{:.2f}", True),
    ("train_contacts_end", "training contacts/episode, last 100k", "{:.2f}", True),
    ("train_contacts_all", "training contacts/episode, whole run", "{:.2f}", True),
    ("first_clean", "first clean evaluation (k steps)", "{:.0f}", True),
]


def summary_phase(args):
    results = {}
    for algo in args.algo_list:
        results[algo] = {}
        for seed in args.seed_list:
            run_dir = find_run(args, algo, seed)
            if run_dir is not None:
                results[algo][seed] = run_summary(run_dir, args.total_timesteps)
    lines = [f"# Disturbed slalom: |w_p| <= {args.noise_bound_p}, |w_v| <= {args.noise_bound_v}",
             "",
             f"{args.total_timesteps} steps per run, every early stop off; medians over seeds "
             f"[min, max], and a two-sided Mann-Whitney p against {REFERENCE}. A clean "
             "evaluation: every episode at the goal, no contact (seeds that never have one "
             "count as the budget).",
             "",
             "| | " + " | ".join(f"{a} (n={len(results[a])})" for a in args.algo_list) + " |",
             "|---|" + "---|" * len(args.algo_list)]
    for key, label, fmt, _ in ROWS:
        cells = []
        for algo in args.algo_list:
            vals = [r[key] for r in results[algo].values()]
            if key == "first_clean":
                vals = [args.total_timesteps if v is None else v for v in vals]
                vals = [v / 1e3 for v in vals]
            vals = np.array([v for v in vals if v is not None and not np.isnan(v)])
            if vals.size == 0:
                cells.append("--")
                continue
            cell = f"{fmt.format(np.median(vals))} [{fmt.format(vals.min())}, {fmt.format(vals.max())}]"
            if key == "first_clean":
                never = sum(r[key] is None for r in results[algo].values())
                cell += f" (never: {never})" if never else ""
            ref = [r[key] for r in results.get(REFERENCE, {}).values()]
            if key == "first_clean":
                ref = [(args.total_timesteps if v is None else v) / 1e3 for v in ref]
            ref = np.array([v for v in ref if v is not None and not np.isnan(v)])
            if algo != REFERENCE and ref.size and vals.size and not (np.all(ref == ref[0]) and np.all(vals == ref[0])):
                cell += f", p={mannwhitneyu(vals, ref, alternative='two-sided').pvalue:.2g}"
            cells.append(cell)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if REFERENCE in results and results[REFERENCE]:
        r = list(results[REFERENCE].values())
        lines += ["", f"{REFERENCE}'s worker over every run: mean solve "
                      f"{np.mean([x.get('solve_ms', np.nan) for x in r]):.1f} ms, candidate fallbacks "
                      f"{np.mean([x.get('candidate_rate', np.nan) for x in r]):.2%} of steps, "
                      f"{int(np.nansum([x.get('emergencies', np.nan) for x in r]))} emergencies."]
    lines += ["", "Per seed:", ""]
    for algo in args.algo_list:
        for seed, r in sorted(results[algo].items()):
            fc = "never" if r["first_clean"] is None else f"{r['first_clean'] / 1e3:.0f}k"
            lines.append(f"- {algo} seed {seed}: return {r['return_end']:.1f}, gap {r['gap_end']:.1f}, "
                         f"eval contacts {r['contacts_end']:.2f}, training contacts "
                         f"{r['train_contacts_end']:.2f} (whole run {r['train_contacts_all']:.2f}), "
                         f"first clean {fc}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.phase in ("all", "train"):
        train_phase(args)
    if args.phase in ("all", "summary"):
        summary_phase(args)


if __name__ == "__main__":
    main()
