import os
import sys

# algorithms/hppo, for the flat `hppo` module this test imports by bare name.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
