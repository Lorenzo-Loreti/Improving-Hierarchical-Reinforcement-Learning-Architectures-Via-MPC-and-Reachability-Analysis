import os
import sys

HERE = os.path.dirname(__file__)

# algorithms/ppo_mpc, for the flat `ppo_mpc`/`ppo_mpc_train` modules these
# tests import by bare name, and this directory, for the helpers one test
# file borrows from another.
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(HERE))

# algorithms/, for the flat `common`, `tube_mpc`, `optimal_solver`, ...
# modules ppo_mpc imports.
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

# The tunnel's `envs`, for the end-to-end and controller tests. Its constant
# corridor has no obstacles, so its MIQP has no binaries and stays within the
# size-limited Gurobi licence even with the disturbance's tube on.
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "scenarios", "tunnel")))
