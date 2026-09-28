"""Flat PPO on the slalom. The training loop, its flags and their defaults
live in algorithms/ppo/ppo_train.py; this file only says which environment
to build, the slalom's own defaults and its arm of the regime study.

    python scenarios/slalom/scripts/script_ppo.py --seed 1 [--track]
"""
import os
import sys
# Scenario root (this script's parent), for `envs`; then algorithms/ppo and
# algorithms/, for the flat `ppo_train`/`ppo` and shared modules (`common`,
# `optimal_solver`, `solved_check`) imported by bare name.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithms', 'ppo')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithms')))
from envs.config import SlalomEnvConfig
from envs.slalom_env import SlalomEnv
from envs.vec_slalom_env import SlalomVecEnv
from envs.width_profile import slalom_profile
from ppo_train import Scenario, main


def make_env_config(u_max=None, **overrides):
    return SlalomEnvConfig(
        width_profile=slalom_profile(),
        **({} if u_max is None else {"u_max": u_max}),
        **overrides,
    )


SLALOM = Scenario(
    name="slalom",
    env_cls=SlalomEnv,
    vec_env_cls=SlalomVecEnv,
    make_env_config=make_env_config,
    total_timesteps=500000,
    wandb_project="Slalom-PPO-NoNoise",
    env_u_max_help=(
        "REGIME STUDY ONLY. Override the environment's acceleration "
        "limit u_max (default None = the canonical 2.5). Mirrors the "
        "flag of the same name in script_ppo_mpc.py so flat PPO can be "
        "run as the control arm of the reachability regime study; see "
        "docs/reachability-regime-study.md"),
    script_dir=os.path.dirname(os.path.abspath(__file__)),
)

if __name__ == "__main__":
    main(SLALOM)
