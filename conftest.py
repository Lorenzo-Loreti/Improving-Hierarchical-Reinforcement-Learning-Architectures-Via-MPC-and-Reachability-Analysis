"""Settings shared by every test directory.

Tests render off-screen with matplotlib's non-interactive Agg backend, so no
test needs a display or a working Tcl/Tk installation. Left to choose,
matplotlib may pick its Tk backend, which creates a Tk window for every pyplot
figure, even one only read back as an ``rgb_array`` frame, and fails where
Tcl/Tk is missing or broken.

pytest loads this file before any test module, hence before matplotlib is
imported. A backend chosen by the user through ``MPLBACKEND`` takes precedence.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
