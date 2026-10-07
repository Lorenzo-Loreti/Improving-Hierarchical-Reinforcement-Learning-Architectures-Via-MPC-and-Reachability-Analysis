"""Settings shared by every test directory.

Tests render off-screen with matplotlib's non-interactive Agg backend, so no
test needs a display or a working Tcl/Tk installation. Left to choose,
matplotlib may pick its Tk backend, which creates a Tk window for every pyplot
figure, even one only read back as an ``rgb_array`` frame, and fails where
Tcl/Tk is missing or broken.

pytest loads this file before any test module, hence before matplotlib is
imported. A backend chosen by the user through ``MPLBACKEND`` takes precedence.

In continuous integration (``CI=true``, as GitHub Actions sets it) every
optional dependency of the tests must be installed: a missing one would turn
whole test modules into silent skips, and the CI would stay green without
running them.
"""

import importlib.util
import os

os.environ.setdefault("MPLBACKEND", "Agg")

if os.environ.get("CI") == "true":
    _missing = [name for name in ("torch", "gymnasium", "matplotlib") if importlib.util.find_spec(name) is None]
    if _missing:
        raise RuntimeError(f"the CI must install {_missing} (extras rl and plots); see .github/workflows/tests.yml")
