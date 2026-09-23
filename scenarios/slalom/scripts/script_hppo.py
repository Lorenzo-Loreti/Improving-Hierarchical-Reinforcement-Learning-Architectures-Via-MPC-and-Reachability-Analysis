"""hPPO on the slalom. The training loop, its flags and their defaults live
in algorithms/hppo/hppo_train.py; this file only says which environment to
build, the slalom's own defaults and its arm of the regime study.

    python scenarios/slalom/scripts/script_hppo.py --seed 1 [--track]
"""
import os
import sys
# Scenario root (this script's parent), for `envs`; then algorithms/hppo and
# algorithms/, for the flat `hppo_train`/`hppo` and shared modules (`common`,
# `optimal_solver`, `solved_check`) imported by bare name.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithms', 'hppo')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithms')))
from envs.config import SlalomEnvConfig
from envs.slalom_env import SlalomEnv
from envs.vec_slalom_env import SlalomVecEnv
from envs.width_profile import slalom_profile
from hppo_train import Scenario, main


def make_env_config(**overrides):
    return SlalomEnvConfig(width_profile=slalom_profile(), **overrides)


SLALOM = Scenario(
    name="slalom",
    env_cls=SlalomEnv,
    vec_env_cls=SlalomVecEnv,
    make_env_config=make_env_config,
    total_timesteps=500000,
    wandb_project="Slalom-hPPO-NoNoise",
    env_u_max_help=(
        "REGIME STUDY ONLY. Override the environment's acceleration "
        "limit u_max (default None = SlalomEnvConfig's own 2.5, i.e. the "
        "canonical environment every architecture in this repo is "
        "compared on). Lowering it makes the plant sluggish, raising the "
        "agility ratio rho = v_max/(u_max*manager_freq*dt). Mirrors the "
        "identically-named flag in the other three scripts; hPPO is the "
        "*learned*-worker arm of that sweep, the one control that says "
        "whether robustness to plant sluggishness comes from hierarchy "
        "as such or specifically from the model-based worker. See "
        "docs/reachability-regime-study.md"),
    script_dir=os.path.dirname(os.path.abspath(__file__)),
)

if __name__ == "__main__":
    main(SLALOM)
