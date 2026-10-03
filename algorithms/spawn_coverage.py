"""How evenly the latest training episodes' starts cover the spawn box.

The training loops log this every update (the `spawn/*` keys), so a run
records whether the env's start sampler (the env config's `init_sampler`, see
scenarios/slalom/envs/spawn_sampler.py) actually spread the starts out, next
to whatever effect that had on learning. It is measured on the starts the
policy really trained on, the observations the vector env hands back after
each reset, not on the sampler in isolation.

Two numbers over the last `window` starts, both mapped onto the unit square:

    spawn/discrepancy   the centred L2-discrepancy (Hickernell 1998,
                        scipy.stats.qmc.discrepancy): how far the empirical
                        distribution of the starts is from uniform, over all
                        sub-boxes at once. Lower is more even.
    spawn/empty_cells   how many cells of a 5 x 5 partition of the box, the
                        resolution of the solved-check grid
                        (optimal_solver.spawn_grid), no start fell in.

The window of 32 starts is a few updates' worth (flat PPO finishes about 5-12
episodes per update, hPPO about 10-25), and a power of 2, the length at which
a Sobol' block is exactly balanced; the window slides, so it is usually not
aligned to a block. Reference values on the unit square, 2 000 repetitions:
32 independent starts have a discrepancy of 0.012 and leave 6.8 of the 25
cells empty on average; 32 consecutive points of a scrambled Sobol' sequence
starting at its beginning, 0.0006 and 3.2.

Pure bookkeeping: it reads observations the loop already has and draws no
random number, so a run that logs it is otherwise unchanged.

Imported by bare name once algorithms/ is on sys.path, like solved_check.
"""

from collections import deque

import numpy as np
from scipy.stats import qmc

WINDOW = 32
CELLS = 5


class SpawnCoverage:
    def __init__(self, low, high, window=WINDOW, cells=CELLS):
        """`low`/`high`: the spawn box over (p_x0, p_y0), e.g. a vector
        env's spawn_low/spawn_high."""
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.cells = cells
        self._recent = deque(maxlen=window)

    def add(self, starts):
        """Record the starts of new episodes: an (n, >= 2) array of
        observations whose first two columns are (p_x0, p_y0). n may be 0."""
        starts = np.asarray(starts, dtype=np.float64).reshape(-1, np.shape(starts)[-1])
        # Clipped because a float32 start can round onto the box's upper edge.
        unit = np.clip((starts[:, :2] - self.low) / (self.high - self.low), 0.0, 1.0)
        self._recent.extend(unit)

    def metrics(self):
        """The `spawn/*` metrics over the last `window` starts, or {} until
        that many have been recorded."""
        if len(self._recent) < self._recent.maxlen:
            return {}
        unit = np.array(self._recent)
        cell = np.minimum((unit * self.cells).astype(int), self.cells - 1)
        occupied = len(set(map(tuple, cell)))
        return {
            "spawn/discrepancy": float(qmc.discrepancy(unit, method="CD")),
            "spawn/empty_cells": float(self.cells ** 2 - occupied),
        }
