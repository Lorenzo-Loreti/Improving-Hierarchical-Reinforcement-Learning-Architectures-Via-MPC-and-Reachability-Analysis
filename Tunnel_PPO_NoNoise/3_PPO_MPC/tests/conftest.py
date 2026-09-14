import os
import sys

# Repository root, for `envs`; then 3_PPO_MPC, for the flat `ppo_mpc` /
# `mpc_worker` modules the training script imports by bare name.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
