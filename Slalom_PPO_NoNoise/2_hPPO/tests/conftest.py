import os
import sys

# Repository root, for `envs`; then 2_hPPO, for the flat `hppo` module the
# training script imports by bare name.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
