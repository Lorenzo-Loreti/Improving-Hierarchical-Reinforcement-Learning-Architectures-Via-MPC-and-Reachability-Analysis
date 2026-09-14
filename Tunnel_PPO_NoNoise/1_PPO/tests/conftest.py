import os
import sys

# Repository root, for `envs`; then 1_PPO, for the flat `ppo` module the
# training script imports by bare name.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
