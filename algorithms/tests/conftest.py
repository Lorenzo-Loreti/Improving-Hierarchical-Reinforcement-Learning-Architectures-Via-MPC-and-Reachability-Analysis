import os
import sys

# algorithms/, for the flat legacy modules these tests import by bare name
# (optimal_solver, solved_check, study, metrics_log, spawn_coverage).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# The oracle's tests need a scenario's `envs` package (the environment and
# `envs.width_profile`), which is identical across scenarios, so either
# works here -- tunnel is picked arbitrarily.
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "scenarios", "tunnel")))
