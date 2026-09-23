import os
import sys

# algorithms/ppo_mpc, for the flat `ppo_mpc` module this test imports by bare
# name.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# algorithms/, for the flat `common` module `ppo_mpc.py` itself imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
