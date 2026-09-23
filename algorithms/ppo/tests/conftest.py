import os
import sys

# algorithms/ppo, for the flat `ppo` module this test imports by bare name.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# algorithms/, for the flat `common` module `ppo.py` itself imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
