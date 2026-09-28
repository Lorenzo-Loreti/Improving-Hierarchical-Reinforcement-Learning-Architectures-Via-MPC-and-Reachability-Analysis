"""PPO+MPC on the slalom: hPPO's manager over the tube-MPC worker. The
training loop, its flags and their defaults live in algorithms/ppo_mpc/
ppo_mpc_train.py, the worker in algorithms/tube_mpc.py; this file only says
which environment to build, the slalom's own defaults and its arm of the
regime study.

    python scenarios/slalom/scripts/script_ppo_mpc.py --seed 1 [--track]
    python scenarios/slalom/scripts/script_ppo_mpc.py --seed 1 --noise-bound-p 0.005 --noise-bound-v 0.05

Until 2026-09-28 this was a ~900-line script with its own copy of the loop
and the old MPCWorker; see ppo_mpc_train.py's docstring for what changed.
"""
import os
import sys
# Scenario root (this script's parent), for `envs`; then algorithms/ppo_mpc
# and algorithms/, for the flat `ppo_mpc_train`/`ppo_mpc` and shared modules
# (`common`, `tube_mpc`, `optimal_solver`, `solved_check`) imported by bare
# name.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithms', 'ppo_mpc')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithms')))
from envs.config import SlalomEnvConfig
from envs.slalom_env import SlalomEnv
from envs.vec_slalom_env import SlalomVecEnv
from envs.width_profile import slalom_profile
from ppo_mpc_train import Scenario, main


def make_env_config(**overrides):
    return SlalomEnvConfig(width_profile=slalom_profile(), **overrides)


SLALOM = Scenario(
    name="slalom",
    env_cls=SlalomEnv,
    vec_env_cls=SlalomVecEnv,
    make_env_config=make_env_config,
    total_timesteps=500000,
    wandb_project="Slalom-PPO-MPC-NoNoise",
    env_u_max_help=(
        "REGIME STUDY ONLY. Override the environment's acceleration "
        "limit u_max (default None = SlalomEnvConfig's own 2.5, i.e. the "
        "canonical environment every architecture in this repo is "
        "compared on). Lowering it makes the plant sluggish, which "
        "narrows and off-centres the per-segment reachable set until no "
        "static goal box can cover it -- the axis along which the "
        "reachability ablation is swept. See docs/goal-box-saturation.md "
        "and docs/reachability-regime-study.md"),
    script_dir=os.path.dirname(os.path.abspath(__file__)),
)

if __name__ == "__main__":
    main(SLALOM)
