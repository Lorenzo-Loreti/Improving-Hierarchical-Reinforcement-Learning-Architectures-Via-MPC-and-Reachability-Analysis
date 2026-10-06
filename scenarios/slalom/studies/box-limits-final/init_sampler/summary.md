# Independent vs Sobol' training starts on the slalom

Every arm: the same seeds, 204800 steps, every early stop off; the arms of an algorithm differ only in --init-sampler. p: two-sided Mann-Whitney U over the seeds, against the same algorithm's uniform arm (a seed that never solved ranks after every seed that did).

## Learning

First solve: of the strict solved-check, held for two evaluations. Median shift: the Sobol' arm's median first solve minus the uniform arm's, with a 95 % bootstrap interval over seeds. Return AUC: the evaluation return averaged over the whole run, per seed. Late precision: share of the solved-checks passed after the first solve, averaged over the seeds that solved.

| arm | solved | first solve, median / mean (range) | p | median shift (95 % CI) | return AUC, median | p | late precision | p | solved at the end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO, uniform | 20/20 | 51k / 53k (41k-133k) | - | - | 986.8 | - | 97 % | - | 19/20 |
| PPO, Sobol' | 20/20 | 51k / 53k (41k-123k) | 0.9 | +0k (-10k to +10k) | 981.2 | 0.69 | 98 % | 1 | 20/20 |
| hPPO, uniform | 20/20 | 72k / 79k (51k-133k) | - | - | 963.4 | - | 86 % | - | 19/20 |
| hPPO, Sobol' | 20/20 | 72k / 79k (61k-154k) | 0.88 | +0k (-20k to +15k) | 957.9 | 0.76 | 89 % | 0.79 | 18/20 |
| PPO+MPC, uniform | 20/20 | 72k / 69k (61k-82k) | - | - | 838.2 | - | 100 % | - | 20/20 |
| PPO+MPC, Sobol' | 20/20 | 72k / 69k (61k-82k) | 0.72 | +0k (-10k to +5k) | 850.2 | 0.49 | 100 % | 1 | 20/20 |

## The final policy, and the mechanism

Grid: the final policy from the 25 solved-check starts, against the oracle. Failed checks: over every solved-check that failed in any seed, the share whose worst start is one of the box's 4 corners (4 of the 25 grid starts, 16 %). Spawn: how evenly each run's latest 32 training starts covered the box (algorithms/spawn_coverage.py), averaged over the run; for reference, 32 independent starts leave 6.8 cells empty, at a discrepancy of 0.012.

| arm | final eval return (% of oracle) | grid: on the oracle's step | grid: mean extra steps | p | grid contacts | training contacts/ep | failed checks: worst start at a corner | spawn: empty 5x5 cells | spawn: discrepancy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO, uniform | 1013.3 (100.0 %) | 79 % | +0.21 | - | 1 | 0.61 | 61 % of 71 | 6.84 | 0.0123 |
| PPO, Sobol' | 1013.3 (100.0 %) | 80 % | +0.20 | 0.3 | 0 | 0.63 | 65 % of 65 | 4.12 | 0.0012 |
| hPPO, uniform | 1012.7 (100.0 %) | 8 % | +0.96 | - | 1 | 1.09 | 65 % of 143 | 6.83 | 0.0124 |
| hPPO, Sobol' | 1012.8 (100.0 %) | 17 % | +0.85 | 0.038 | 3 | 1.13 | 69 % of 137 | 4.12 | 0.0012 |
| PPO+MPC, uniform | 1013.3 (100.0 %) | 80 % | +0.20 | - | 0 | 0.00 | 72 % of 95 | 6.81 | 0.0124 |
| PPO+MPC, Sobol' | 1013.3 (100.0 %) | 80 % | +0.20 | 1 | 0 | 0.00 | 76 % of 94 | 4.13 | 0.0011 |

Per-seed numbers: each arm's own summary.md. Figures: figures/ and compare_<algo>/figures/.
