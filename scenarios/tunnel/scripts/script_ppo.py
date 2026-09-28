"""Flat PPO on the tunnel. The training loop, its flags and their defaults
live in algorithms/ppo/ppo_train.py; this file only says which environment
to build, the tunnel's own defaults and its arm of the regime study.

    python scenarios/tunnel/scripts/script_ppo.py --seed 1 [--track]
"""
import os
import sys
# Scenario root (this script's parent), for `envs`; then algorithms/ppo and
# algorithms/, for the flat `ppo_train`/`ppo` and shared modules (`common`,
# `optimal_solver`, `solved_check`) imported by bare name.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithms', 'ppo')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithms')))
from envs.config import TunnelEnvConfig
from envs.tunnel_env import TunnelEnv
from envs.vec_tunnel_env import TunnelVecEnv
from ppo_train import Scenario, main


def make_env_config(u_max=None, **overrides):
    return TunnelEnvConfig(**({} if u_max is None else {"u_max": u_max}), **overrides)


TUNNEL = Scenario(
    name="tunnel",
    env_cls=TunnelEnv,
    vec_env_cls=TunnelVecEnv,
    make_env_config=make_env_config,
    total_timesteps=200000,
    wandb_project="Tunnel-PPO-NoNoise",
    env_u_max_help=(
        "REGIME STUDY ONLY. Override the environment's acceleration "
        "limit u_max (default None = TunnelEnvConfig's own 2.5, i.e. the "
        "canonical environment every architecture in this repo is "
        "compared on). Lowering it makes the plant sluggish, raising the "
        "agility ratio rho = v_max/(u_max*manager_freq*dt). Mirrors the "
        "identically-named flag in scenarios/slalom, and exists so the "
        "constant-width corridor can act as that sweep's control: it "
        "isolates whether a rho effect comes from plant sluggishness "
        "alone or needs the slalom's lateral precision demand. See "
        "docs/reachability-regime-study.md"),
    script_dir=os.path.dirname(os.path.abspath(__file__)),
)

if __name__ == "__main__":
    main(TUNNEL)
