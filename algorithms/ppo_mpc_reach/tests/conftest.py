import os
import sys

# algorithms/ppo_mpc_reach, for the flat `ppo_mpc_reach` module this test imports
# by bare name.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# `ppo_mpc_reach.py` needs `envs.width_profile` (WidthSegment/WidthProfile), which
# is identical across scenarios, so either scenario's `envs` package works
# here -- tunnel is picked arbitrarily.
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "scenarios", "tunnel")))
