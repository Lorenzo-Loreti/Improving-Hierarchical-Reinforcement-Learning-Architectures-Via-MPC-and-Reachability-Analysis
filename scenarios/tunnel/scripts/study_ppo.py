"""Flat-PPO seed study on the tunnel: train the seeds, then plot learning curves
and the learned trajectories against the min-time oracle. The study itself
(phases, defaults, figures) lives in algorithms/study.py and
algorithms/study_plots.py; this file only says which algorithm and scenario.

    python scenarios/tunnel/scripts/study_ppo.py            # train 20 seeds, then analyze and plot
    python scenarios/tunnel/scripts/study_ppo.py analyze    # after training: recompute and plot
    python scenarios/tunnel/scripts/study_ppo.py plot       # redraw the figures only
    python scenarios/tunnel/scripts/study_ppo.py --help

Training is resumable (finished seeds are skipped). Everything goes to
scenarios/tunnel/studies/ppo/, which git ignores. Run study_hppo.py too, then
study_compare.py for the shared figures.

The tunnel's seed studies were added on 2026-10-04, with the disk limits and
the effort penalty (scenarios/tunnel/envs/actuation.py, envs/config.py);
before that the tunnel had only the 3-seed benchmark of 2026-09-23
(docs/benchmark.md).
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
# Scenario root, for `envs`; then algorithms/ppo and algorithms/, for the flat
# modules imported by bare name. This script's own directory (for
# `script_ppo`) is already on sys.path as the running script's.
sys.path.append(os.path.abspath(os.path.join(HERE, '..')))
sys.path.append(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'algorithms', 'ppo')))
sys.path.append(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'algorithms')))
from envs.tunnel_env import TunnelEnv
from envs.visualize_env import plot_env
from script_ppo import make_env_config
from ppo_train import load_agent, make_policy_fn
from study import Study, main


def make_env(run_config):
    """The environment a run trained on, from its config.json."""
    return TunnelEnv(config=make_env_config(run_config.get("env_u_max")))


def load_policy(path, env):
    agent = load_agent(path, env)
    # Memoryless: the goal trace a hierarchical controller fills stays empty.
    return lambda goal_trace=None: make_policy_fn(agent)


STUDY = Study(
    label="PPO",
    key="ppo",
    scenario="tunnel",
    train_script=os.path.join(HERE, "script_ppo.py"),
    make_env=make_env,
    plot_env=plot_env,
    load_policy=load_policy,
    hierarchical=False,
    color="#2a78d6",
    out_dir=os.path.abspath(os.path.join(HERE, '..', 'studies', 'ppo')),
)

if __name__ == "__main__":
    main(STUDY)
