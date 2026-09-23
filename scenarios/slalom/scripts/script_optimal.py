"""Maximum-return (minimum-time) oracle for SlalomEnv.

Solves the exact minimum-time, wall-contact-free trajectory (see
algorithms/optimal_solver.py) for a batch of randomly-drawn initial
conditions -- the same distribution `SlalomEnv.reset()` draws from during
training -- and reports the same metrics script_ppo.py logs for a trained
agent's eval pass (episodic_return, success_rate, collision_count_mean), so
the two can be compared directly as a ceiling against actual performance.

Usage:
    python scenarios/slalom/scripts/script_optimal.py
    python scenarios/slalom/scripts/script_optimal.py --episodes 200 --output optimal_metrics.json --plot optimal_trajectory.png
"""

import os
import sys
import argparse
import json

import numpy as np

# Scenario root (this script's parent), for `envs`; then algorithms/, for
# the flat `optimal_solver` module it imports by bare name.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithms')))
from envs.config import SlalomEnvConfig
from envs.slalom_env import SlalomEnv
from envs.width_profile import slalom_profile
from envs.visualize_env import plot_env
from optimal_solver import MinTimeSolver


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the maximum-return oracle trajectory for SlalomEnv.")
    parser.add_argument("--episodes", type=int, default=100,
        help="number of random initial conditions to solve and average over")
    parser.add_argument("--seed", type=int, default=1,
        help="seed for the initial-condition draws (episode i uses seed + i)")
    parser.add_argument("--output", type=str, default=None,
        help="path to save the aggregate metrics as JSON")
    parser.add_argument("--plot", type=str, default=None,
        help="path to save a plot of one representative optimal trajectory over the track layout")
    return parser.parse_args()


def main():
    args = parse_args()

    config = SlalomEnvConfig(width_profile=slalom_profile())
    env = SlalomEnv(config=config)
    solver = MinTimeSolver()

    returns, lengths, successes, collisions = [], [], [], []
    representative = None
    for i in range(args.episodes):
        env.reset(seed=args.seed + i)
        result = solver.solve(env)
        returns.append(result.total_return)
        lengths.append(result.length)
        successes.append(result.success)
        collisions.append(result.collision_count)
        if representative is None:
            representative = result

    returns = np.asarray(returns, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    successes = np.asarray(successes, dtype=float)
    collisions = np.asarray(collisions, dtype=float)

    metrics = {
        "scenario": "slalom",
        "num_episodes": args.episodes,
        "seed": args.seed,
        "episodic_return_mean": float(returns.mean()),
        "episodic_return_std": float(returns.std()),
        "episodic_return_min": float(returns.min()),
        "episodic_return_max": float(returns.max()),
        "episode_length_mean": float(lengths.mean()),
        "success_rate": float(successes.mean()),
        "collision_count_mean": float(collisions.mean()),
    }

    print(
        f"optimal[slalom]: episodes={args.episodes} "
        f"return_mean={metrics['episodic_return_mean']:.2f} "
        f"return_std={metrics['episodic_return_std']:.2f} "
        f"return_min={metrics['episodic_return_min']:.2f} "
        f"return_max={metrics['episodic_return_max']:.2f} "
        f"length_mean={metrics['episode_length_mean']:.1f} "
        f"success_rate={metrics['success_rate']:.2f} "
        f"collisions/ep={metrics['collision_count_mean']:.2f}"
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {args.output}")

    if args.plot:
        import matplotlib.pyplot as plt
        ax = plot_env(config)
        traj = representative.states
        ax.plot(traj[:, 0], traj[:, 1], color="tab:red", linewidth=2, label="optimal trajectory")
        ax.legend(loc="upper right", fontsize=8)
        plt.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
