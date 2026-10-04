"""hPPO seed study on the tunnel: train the seeds, then plot learning curves,
the learned trajectories against the min-time oracle, and the manager's
subgoals. The study itself (phases, defaults, figures) lives in
algorithms/study.py and algorithms/study_plots.py; this file only says which
algorithm and scenario.

    python scenarios/tunnel/scripts/study_hppo.py           # train 20 seeds, then analyze and plot
    python scenarios/tunnel/scripts/study_hppo.py analyze   # after training: recompute and plot
    python scenarios/tunnel/scripts/study_hppo.py plot      # redraw the figures only
    python scenarios/tunnel/scripts/study_hppo.py --help

Training is resumable (finished seeds are skipped). Everything goes to
scenarios/tunnel/studies/hppo/, which git ignores. Run study_ppo.py too, then
study_compare.py for the shared figures.

The tunnel's seed studies were added on 2026-10-04, with the disk limits and
the effort penalty (scenarios/tunnel/envs/actuation.py, envs/config.py);
before that the tunnel had only the 3-seed benchmark of 2026-09-23
(docs/benchmark.md).
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
# Scenario root, for `envs`; then algorithms/hppo and algorithms/, for the flat
# modules imported by bare name. This script's own directory (for
# `script_hppo`) is already on sys.path as the running script's.
sys.path.append(os.path.abspath(os.path.join(HERE, '..')))
sys.path.append(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'algorithms', 'hppo')))
sys.path.append(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'algorithms')))
from envs.tunnel_env import TunnelEnv
from envs.visualize_env import plot_env
from script_hppo import make_env_config
from hppo_train import load_agent, make_policy_fn
from study import Study, main


def make_env(run_config):
    """The environment a run trained on, from its config.json: hPPO's script
    can override the contact penalty as well as u_max."""
    overrides = {"contact_penalty": run_config["contact_penalty"]}
    if run_config.get("env_u_max") is not None:
        overrides["u_max"] = run_config["env_u_max"]
    return TunnelEnv(config=make_env_config(**overrides))


def load_policy(path, env):
    agent = load_agent(path, env)
    return lambda goal_trace=None: make_policy_fn(agent, goal_trace)


STUDY = Study(
    label="hPPO",
    key="hppo",
    scenario="tunnel",
    train_script=os.path.join(HERE, "script_hppo.py"),
    make_env=make_env,
    plot_env=plot_env,
    load_policy=load_policy,
    hierarchical=True,
    color="#eb6834",
    out_dir=os.path.abspath(os.path.join(HERE, '..', 'studies', 'hppo')),
)

if __name__ == "__main__":
    main(STUDY)
