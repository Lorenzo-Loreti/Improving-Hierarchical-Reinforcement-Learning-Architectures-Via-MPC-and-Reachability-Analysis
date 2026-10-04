"""PPO+MPC seed study on the tunnel: train the seeds, then plot what they
learned against the oracle. The phases and flags are algorithms/study.py's,
the same protocol as study_ppo.py and study_hppo.py, so the three studies
overlay in study_compare.py.

    python scenarios/tunnel/scripts/study_ppo_mpc.py --seeds 1-10
    python scenarios/tunnel/scripts/study_ppo_mpc.py analyze --seeds 1-10

Output goes to scenarios/tunnel/studies/ppo_mpc/, which git ignores. The
controller is the pair the checkpoint describes: the manager and a tube-MPC
worker rebuilt with the settings it was trained with (ppo_mpc_train.load_agent).
Runs trained with a disturbance are refused by study.py.

The tunnel's seed studies were added on 2026-10-04, with the disk limits and
the effort penalty (scenarios/tunnel/envs/actuation.py, envs/config.py);
before that the tunnel had only the 3-seed benchmark of 2026-09-23
(docs/benchmark.md).
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
# Scenario root, for `envs`; then algorithms/ppo_mpc and algorithms/, for the
# flat `ppo_mpc_train`/`ppo_mpc` and shared modules imported by bare name. This
# script's own directory (for `script_ppo_mpc`) is already on sys.path as the
# running script's.
sys.path.append(os.path.abspath(os.path.join(HERE, '..')))
sys.path.append(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'algorithms', 'ppo_mpc')))
sys.path.append(os.path.abspath(os.path.join(HERE, '..', '..', '..', 'algorithms')))
from envs.tunnel_env import TunnelEnv
from envs.visualize_env import plot_env
from script_ppo_mpc import make_env_config
from ppo_mpc_train import load_agent, make_policy_fn
from study import Study, main


def make_env(run_config):
    """The environment a run trained on, from its config.json: the contact
    penalty and u_max can be overridden, as in hPPO's script."""
    overrides = {"contact_penalty": run_config["contact_penalty"]}
    if run_config.get("env_u_max") is not None:
        overrides["u_max"] = run_config["env_u_max"]
    return TunnelEnv(config=make_env_config(**overrides))


def load_policy(path, env):
    """One worker per checkpoint, shared by every episode replayed from it:
    make_policy_fn resets its slot at the start of each episode, and study.py
    replays episodes one at a time."""
    agent, worker = load_agent(path, env)
    return lambda goal_trace=None: make_policy_fn(agent, worker, goal_trace=goal_trace)


STUDY = Study(
    label="PPO+MPC",
    key="ppo_mpc",
    scenario="tunnel",
    train_script=os.path.join(HERE, "script_ppo_mpc.py"),
    make_env=make_env,
    plot_env=plot_env,
    load_policy=load_policy,
    hierarchical=True,
    # The reference palette's third categorical slot (aqua), after PPO's blue
    # and hPPO's orange, in its fixed order.
    color="#1baf7a",
    out_dir=os.path.abspath(os.path.join(HERE, '..', 'studies', 'ppo_mpc')),
)

if __name__ == "__main__":
    main(STUDY)
