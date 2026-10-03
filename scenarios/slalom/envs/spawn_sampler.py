"""Where each episode starts: the spawn box, and how training draws from it.

Every episode starts at rest at a point (p_x0, p_y0) of the spawn box,
p_x0 in [0, 2] and p_y0 in [-W/4, W/4] (the box `spawn_box` returns). The
initial-state distribution rho_0 of the MDP is the *uniform* distribution on
that box, under both samplers below. What the sampler changes is only how the
starts of successive training episodes are drawn from it:

    "uniform"  independent draws (plain Monte Carlo). The only sampler until
               2026-10-03, and still the default.
    "sobol"    a scrambled Sobol' sequence (randomized quasi-Monte Carlo,
               RQMC): every start is still uniform on the box, but successive
               starts are spread out over it instead of independent.

Because rho_0 is the same, so is the objective the agent maximises,
J(pi) = E_{s_0 ~ rho_0}[V^pi(s_0)]: the oracle comparison, the solved-check
and the comparison between algorithms keep their meaning. "sobol" is a
variance-reduction device, not a curriculum (a curriculum, e.g. reverse
curriculum generation or prioritized level replay, would change rho_0 and so
the problem being solved).

Why independent draws are a poor fit here. A policy update sees few episodes:
flat PPO's batch is 8 x 128 = 1 024 steps and hPPO's 2 048, i.e. about 5
(PPO) to 10 (hPPO) episodes early on, when episodes run out their 200 steps,
and 12-25 once they last about 80. A few independent points cluster and leave
holes. Measured on the unit square, 2 000 repetitions, counting the cells of
a 5 x 5 partition (the resolution of the solved-check grid,
optimal_solver.spawn_grid) that no start falls in:

    starts   independent        Sobol' (scrambled)
    25       9.0 empty (max 14)  6.0 empty (max 12)
    32       6.8 empty (max 12)  3.2 empty (max 9)
    100      0.41                0.08

and the centred L2-discrepancy of 25 starts drops from 0.016 to 0.002. (A
5 x 5 partition is not dyadic, so even Sobol' leaves some of its cells empty;
on the dyadic partitions, 4 x 8, 8 x 4, 2 x 16 ..., 32 Sobol' starts leave
none, see below.)

Why Sobol', and why scrambled. In two dimensions the Sobol' sequence is a
(0, 2)-sequence in base 2 (Sobol' 1967): every aligned block of 2^m
consecutive points puts exactly one point in every dyadic rectangle
[a/2^i, (a+1)/2^i) x [b/2^j, (b+1)/2^j) with i + j = m. In words: 32
consecutive episodes start once in each cell of a 4 x 8 grid, and of an 8 x 4,
a 2 x 16, ... grid, at every resolution at once, with no resolution to
choose. A stratified sampler over a fixed grid (e.g. one start per cell of the
5 x 5 grid, in random order) has that property only at its own resolution.
The plain sequence is deterministic and starts at the corner (0, 0);
scipy's scrambling (a random linear matrix scramble plus a digital shift,
Matousek 1998, in the family of Owen's 1995 scrambled nets) keeps the
(0, 2)-property and makes every single point uniform on the box. That last
part is what keeps the sampler unbiased: any average over the starts is an
unbiased estimate of its expectation under rho_0, with lower variance than
under independent draws (Owen 1997; for policy-gradient RL, Arnold, L'Ecuyer
et al., "Policy Learning and Evaluation with Randomized Quasi-Monte Carlo",
AISTATS 2022).

How the stream is used. The vector env owns one stream for all its
sub-environments: the 8 starts of `reset()` are its first 8 points, and every
auto-reset afterwards takes the next one, environments that finish on the same
step in ascending index order. The stream is fixed by the env's seed, so a run
stays bit-reproducible. Its scrambling draws from a child of that seed (a
`SeedSequence.spawn`), not from the generator the disturbance comes from, so
neither stream shifts the other. Points are drawn ahead in blocks whose total
stays a power of 2: scipy warns when the number of Sobol' points drawn is not
one, since only then is the balance above exact.

Only training rollouts (the vector env) use the sampler. The plain env's
`reset(seed=...)` stays an independent uniform draw: evaluation reseeds it
for every episode (env.reset(seed=run_seed + i)), so it would only ever see a
stream's first point, and keeping it independent gives both samplers exactly
the same evaluation starts.

What it did (2026-10-03, docs/init-sampler.md): on the slalom, 20 seeds per
arm for flat PPO, hPPO and PPO+MPC, "sobol" made the training starts ten times
more even (centred L2-discrepancy 0.012 to 0.001) and changed nothing
measurable in learning or in the final policy. The starts that policies find
hardest are the box's corners nearest to or furthest from the gates'
openings under both samplers, and the starts are only ~3 % (at most ~11 %)
of flat PPO's policy-gradient variance. So "uniform" stays the default.

scenarios/tunnel/envs/spawn_sampler.py is the same module for the tunnel.
"""

import numpy as np
from scipy.stats import qmc

INIT_SAMPLERS = ("uniform", "sobol")

# 2^14 = 16 384 starts, more than any run here needs: 500k steps at the
# fastest episodes the task allows (~70 steps) is ~7 200 episodes. A longer
# run draws more, doubling the total each time (see `take`).
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
