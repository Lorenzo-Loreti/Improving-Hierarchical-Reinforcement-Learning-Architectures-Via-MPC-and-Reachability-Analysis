"""The disturbed studies' final policies against harder disturbances in the
same box W: does the margin flat PPO and hPPO learn hold against a worst
case, the case PPO+MPC's tube is designed for?

    python scenarios/slalom/scripts/stress_disturbed.py
    python scenarios/slalom/scripts/stress_disturbed.py --studies disturbed_0.01_0.1 --modes adversary

It trains nothing. It replays every seed's final.pt from each finished
compare_disturbed.py study, from the 25 starts of the solved-check grid,
under three disturbances, all inside the W the policy was trained with:

    uniform    w drawn uniformly from W, as in training;
    vertex     w on a random vertex of W (the note's section 5.6 (iii));
    adversary  w on the vertex that pushes the state toward the forbidden
               point nearest to where it is heading: sign(q - p') on each
               axis, applied to the position and the velocity alike, where
               p' is the next position without disturbance (the adversary
               knows the action) and q the nearest point of the walls or a
               gate's blocked bands. Greedy, not the worst case -- that would
               take solving a game -- but aimed.

The tube's guarantee (theorem 4.1) holds for every sequence in W, so
PPO+MPC must stay contact-free under all three; nothing guarantees the
learned policies anything. The disturbance is injected through the
environment's own generator, so it enters env.step before the contact
check, exactly where the training disturbance did.

The summary (scenarios/slalom/studies/stress/summary.md, which git
ignores) reports, per study, algorithm and disturbance, the median over
seeds of each seed's mean over the grid: contacts per episode, the share
of episodes with a contact, success, return, the gap to the undisturbed
oracle, and the smallest clearance to a wall reached in an episode.

Result (2026-09-29): seeds 1-10 of every arm of both disturbed studies. The
figures are medians over seeds, with [min, max] where the spread matters.

  contacts per episode           uniform   vertex   adversary
  |w| <= 0.005 / 0.05  PPO+MPC     0         0        0
                       hPPO        0         0        0.06 [0, 0.32]   (2% of episodes)
                       PPO         0         0        0.18 [0.04, 1.2] (10%)
  |w| <= 0.01 / 0.1    PPO+MPC     0         0        0
                       hPPO        0         0        11.6 [4.9, 16.2] (56%), return 434
                       PPO         0         0        9.4 [1.6, 38.4]  (82%), return 553

- The learned margins hold against random disturbances, uniform or on
  W's vertices, and fail against an aimed one. At the first level the
  failure is mild. At twice the level both learned policies still reach
  the goal, but through the walls: ~10 contacts per episode, with returns
  of ~400-550 against ~1012.
- PPO+MPC has no contact under any disturbance at either level, as
  theorem 4.1 says. The guarantee is tight, not padded: under the
  adversary its real state came within 1-3 mm of a wall (the margin rho is
  1 mm), where hPPO and PPO keep 0.15-0.37 m and 0.06-0.13 m against random
  disturbances. The tube spends its whole width, and nothing more.
- Safety is guaranteed, progress is not. Under the adversary at twice the
  level, one PPO+MPC seed (7) did not reach the goal from 13 of the 25
  starts. It came to rest at (6.83, -2.0), in the corner below gate 2,
  just in front of that gate's enlarged obstacle, with no contact. Every
  solve there succeeded (no candidate fallback, no emergency). This is the
  note's remark 4.6: the adversary pushes the state where the manager,
  trained under uniform noise, never went, and the goal it emits there is
  a local minimum the worker cannot leave by itself. The terminal set
  gives recursive feasibility, not convergence (see tube_mpc.py); the
  other nine seeds' managers get out.
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
for sub in ("scenarios/slalom", "algorithms/ppo", "algorithms/hppo", "algorithms/ppo_mpc", "algorithms"):
    sys.path.append(os.path.join(ROOT, *sub.split("/")))
from envs.config import SlalomEnvConfig
from envs.slalom_env import SlalomEnv
from envs.width_profile import slalom_profile
from metrics_log import read_config
from optimal_solver import MinTimeSolver, spawn_grid, precompute_optimal_grid

STUDIES_DIR = os.path.join(ROOT, "scenarios", "slalom", "studies")
DEFAULT_STUDIES = ("disturbed_0.005_0.05", "disturbed_0.01_0.1")
ALGOS = ("ppo_mpc", "hppo", "ppo")
MODES = ("uniform", "vertex", "adversary")


# --------------------------------------------------------------------------
# the disturbances
# --------------------------------------------------------------------------

def nearest_forbidden_point(profile, p):
    """The point nearest to position p of the set the width profile forbids,
    {y <= y_lo(x)} and {y >= y_hi(x)} over each segment's closed x-range,
    and its distance (0 when p is already in it)."""
    best, best_d = None, np.inf
    for seg in profile.segments:
        x = float(np.clip(p[0], seg.x_start, seg.x_end))
        y_lo, y_hi = seg.center_y - seg.half_width, seg.center_y + seg.half_width
        for q in ((x, min(p[1], y_lo)), (x, max(p[1], y_hi))):
            d = float(np.hypot(q[0] - p[0], q[1] - p[1]))
            if d < best_d:
                best, best_d = np.array(q), d
    return best, best_d


class Disturbance:
    """Stands in for the environment's generator: SlalomEnv.step draws its
    disturbance as `self.np_random.uniform(low=-b, high=b)`, and this
    answers that call with the chosen mode's w."""

    def __init__(self, env, mode, rng):
        self.env, self.mode, self.rng = env, mode, rng

    def uniform(self, low, high):
        b = np.asarray(high, dtype=float)
        if self.mode == "uniform":
            return self.rng.uniform(low, high)
        if self.mode == "vertex":
            return b * self.rng.choice([-1.0, 1.0], size=b.shape)
        env = self.env
        s = env.state.astype(float)
        u = np.clip(env.last_action, -env.u_max, env.u_max)
        u = np.clip(u, (-env.v_max - s[2:]) / env.dt, (env.v_max - s[2:]) / env.dt)   # the speed limit, as step() applies it
        p_next = s[:2] + env.dt * s[2:] + 0.5 * env.dt ** 2 * u
        q, _ = nearest_forbidden_point(env.width_profile, p_next)
        direction = np.sign(np.where(np.abs(q - p_next) > 1e-9, q - p_next, 0.0))
        return np.array([direction[0] * b[0], direction[1] * b[1], direction[0] * b[2], direction[1] * b[3]])


class StressedSlalomEnv(SlalomEnv):
    """The slalom with its disturbance drawn by a `Disturbance`."""

    def __init__(self, config, mode):
        super().__init__(config=config)
        self.mode = mode
        self.last_action = np.zeros(2)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.np_random = Disturbance(self, self.mode, np.random.default_rng(seed))
        return obs, info

    def step(self, action):
        self.last_action = np.asarray(action, dtype=float)
        return super().step(action)


# --------------------------------------------------------------------------
# replaying one seed
# --------------------------------------------------------------------------

def load_factory(algo, path, env):
    """A factory of fresh per-episode controllers for a checkpoint, as each
    algorithm's own evaluation builds them."""
    if algo == "ppo":
        import ppo_train
        agent = ppo_train.load_agent(path, env)
        return lambda: ppo_train.make_policy_fn(agent)
    if algo == "hppo":
        import hppo_train
        agent = hppo_train.load_agent(path, env)
        return lambda: hppo_train.make_policy_fn(agent)
    import ppo_mpc_train
    agent, worker = ppo_mpc_train.load_agent(path, env)
    return lambda: ppo_mpc_train.make_policy_fn(agent, worker)


def replay_seed(task):
    """Every mode, every grid start, for one (study, algorithm, seed)."""
    study, algo, seed, path, noise, modes, grid = task
    import torch
    torch.set_num_threads(1)
    config = SlalomEnvConfig(width_profile=slalom_profile(), noise_bound_p=noise[0], noise_bound_v=noise[1])
    out = {}
    for mode in modes:
        env = StressedSlalomEnv(config, mode)
        make_policy = load_factory(algo, path, env)
        rows = []
        for i, start in enumerate(grid):
            obs, _ = env.reset(seed=1000 * seed + i, options={"init_state": start})
            policy_fn = make_policy()
            total, clearance, done = 0.0, np.inf, False
            while not done:
                obs, reward, terminated, truncated, info = env.step(policy_fn(obs))
                total += reward
                clearance = min(clearance, nearest_forbidden_point(env.width_profile, obs[:2])[1])
                done = terminated or truncated
            rows.append((total, info["collision_count"], bool(info["is_success"]), clearance))
        out[mode] = np.array(rows, dtype=float)
    return study, algo, seed, out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def find_final(study, algo, seed):
    runs = sorted(glob.glob(os.path.join(STUDIES_DIR, study, algo, "runs", f"*_{seed}_*")), reverse=True)
    for run in runs:
        if os.path.exists(os.path.join(run, "final.pt")):
            return run
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--studies", nargs="+", default=list(DEFAULT_STUDIES),
        help="compare_disturbed.py study directories under scenarios/slalom/studies")
    parser.add_argument("--algos", nargs="+", default=list(ALGOS))
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out", type=str, default=os.path.join(STUDIES_DIR, "stress"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # The oracle, once, on the undisturbed environment: the reference every
    # gap is measured against, as in training.
    oracle_env = SlalomEnv(config=SlalomEnvConfig(width_profile=slalom_profile()))
    grid = spawn_grid(oracle_env, 5, 5)
    oracle = np.array([r.total_return for _, r in precompute_optimal_grid(oracle_env, grid, MinTimeSolver())])

    tasks = []
    for study in args.studies:
        for algo in args.algos:
            for seed in args.seeds:
                run = find_final(study, algo, seed)
                if run is None:
                    print(f"[stress] {study} {algo} seed {seed}: no final.pt, left out")
                    continue
                cfg = read_config(run)
                tasks.append((study, algo, seed, os.path.join(run, "final.pt"),
                              (cfg["noise_bound_p"], cfg["noise_bound_v"]), args.modes, grid))
    print(f"[stress] replaying {len(tasks)} checkpoints x {len(args.modes)} disturbances x {len(grid)} starts, "
          f"{args.jobs} at a time")
    results = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for study, algo, seed, out in pool.map(replay_seed, tasks):
            results[(study, algo, seed)] = out

    def cell(values, fmt):
        v = np.asarray(values, dtype=float)
        return f"{fmt.format(np.median(v))} [{fmt.format(v.min())}, {fmt.format(v.max())}]"

    lines = ["# Final policies of the disturbed studies under harder disturbances in W", "",
             "Each seed's final.pt from the 25 solved-check starts; medians over seeds [min, max] of "
             "each seed's mean over the starts. Clearance: the smallest distance to a wall or a "
             "gate's blocked band an episode reached (0 = contact). Gap: to the undisturbed oracle.",
             ""]
    for study in args.studies:
        lines += [f"## {study}", "",
                  "| algorithm | disturbance | contacts/episode | episodes with a contact | success "
                  "| return | gap to oracle | min clearance (m) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for algo in args.algos:
            seeds = [s for s in args.seeds if (study, algo, s) in results]
            if not seeds:
                continue
            for mode in args.modes:
                per = [results[(study, algo, s)][mode] for s in seeds]
                lines.append(
                    f"| {algo} (n={len(seeds)}) | {mode} "
                    f"| {cell([r[:, 1].mean() for r in per], '{:.2f}')} "
                    f"| {cell([100 * (r[:, 1] > 0).mean() for r in per], '{:.0f}%')} "
                    f"| {cell([r[:, 2].mean() for r in per], '{:.2f}')} "
                    f"| {cell([r[:, 0].mean() for r in per], '{:.1f}')} "
                    f"| {cell([(oracle - r[:, 0]).mean() for r in per], '{:.1f}')} "
                    f"| {cell([r[:, 3].min() for r in per], '{:.3f}')} |")
        lines.append("")
    text = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
