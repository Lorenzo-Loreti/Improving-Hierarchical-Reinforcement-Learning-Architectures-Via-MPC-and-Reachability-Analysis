"""Where each episode starts: the spawn box, and how training draws from it.

Every episode starts at rest at a point (p_x0, p_y0) of the spawn box,
p_x0 in [0, 2] and p_y0 in [-W/4, W/4]; the initial-state distribution is
uniform on it. The vector env draws the starts of its training episodes
either independently ("uniform", the default) or from a scrambled Sobol'
sequence ("sobol", randomized quasi-Monte Carlo): every start is still
uniform on the box, so the objective is unchanged, but successive starts
cover it evenly instead of clustering. The plain env's reset, which
evaluation reseeds per episode, always draws independently.

The same module as scenarios/slalom/envs/spawn_sampler.py, whose docstring
has the full rationale: the measured coverage, the (0, 2)-sequence property,
why scrambling keeps the sampler unbiased, how the stream is seeded and
consumed, and the references. On the slalom "sobol" changed nothing
measurable for any algorithm (docs/init-sampler.md), so "uniform" stays the
default.
"""

import numpy as np
from scipy.stats import qmc

INIT_SAMPLERS = ("uniform", "sobol")

# 2^14 = 16 384 starts, more than any run here needs; a longer run draws
# more, doubling the total each time (see `take`).
SOBOL_FIRST_BLOCK = 2 ** 14


def spawn_box(W):
    """The spawn box as (low, high) arrays over (p_x0, p_y0): p_x0 in [0, 2],
    p_y0 in [-W/4, W/4]. The one place these bounds are written; the envs'
    resets, the vector env's samplers and the solved-check grid
    (optimal_solver.spawn_grid) all read them from here."""
    return np.array([0.0, -W / 4.0]), np.array([2.0, W / 4.0])


class SobolSpawnStream:
    """Scrambled Sobol' points in [0, 1)^2, handed out in sequence order.

    `seed` fixes the scrambling (None: fresh entropy, like `default_rng()`).
    The scrambling generator is a child of `seed`, so the stream is
    independent of any `default_rng(seed)` the caller also draws from."""

    def __init__(self, seed=None, first_block=SOBOL_FIRST_BLOCK):
        if first_block <= 0 or first_block & (first_block - 1):
            raise ValueError(f"first_block must be a power of 2, got {first_block}")
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
        self._sobol = qmc.Sobol(d=2, scramble=True, rng=rng)
        self._points = self._sobol.random(first_block)
        self._next = 0

    def take(self, n):
        """The next `n` points, an (n, 2) array."""
        while self._next + n > len(self._points):
            # As many again as drawn so far, so the total stays a power of 2.
            self._points = np.concatenate([self._points, self._sobol.random(self._sobol.num_generated)])
        points = self._points[self._next:self._next + n]
        self._next += n
        return points
