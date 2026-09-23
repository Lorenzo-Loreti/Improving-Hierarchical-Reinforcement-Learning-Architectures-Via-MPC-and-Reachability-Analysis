import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# scenarios/slalom and scenarios/tunnel both name their env package `envs`.
# If tunnel's suite already ran in this process, `envs`/`envs.*` are cached
# in sys.modules pointing at tunnel's files, and the sys.path.insert above
# would be silently ignored -- Python checks sys.modules before sys.path.
# Scrubbing the cache here forces a fresh import from *this* scenario's path.
for _name in list(sys.modules):
    if _name == "envs" or _name.startswith("envs."):
        del sys.modules[_name]
