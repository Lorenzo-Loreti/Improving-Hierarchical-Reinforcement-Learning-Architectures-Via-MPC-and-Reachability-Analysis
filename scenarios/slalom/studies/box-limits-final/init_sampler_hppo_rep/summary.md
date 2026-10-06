# Independent vs Sobol' training starts on the slalom

Every arm: the same seeds, 204800 steps, every early stop off; the arms of an algorithm differ only in --init-sampler. p: two-sided Mann-Whitney U over the seeds, against the same algorithm's uniform arm (a seed that never solved ranks after every seed that did).

## Learning

First solve: of the strict solved-check, held for two evaluations. Median shift: the Sobol' arm's median first solve minus the uniform arm's, with a 95 % bootstrap interval over seeds. Return AUC: the evaluation return averaged over the whole run, per seed. Late precision: share of the solved-checks passed after the first solve, averaged over the seeds that solved.

| arm | solved | first solve, median / mean (range) | p | median shift (95 % CI) | return AUC, median | p | late precision | p | solved at the end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hPPO, uniform | 20/20 | 72k / 71k (51k-102k) | - | - | 968.7 | - | 87 % | - | 18/20 |
| hPPO, Sobol' | 20/20 | 61k / 67k (51k-113k) | 0.19 | -10k (-20k to +5k) | 960.6 | 0.46 | 90 % | 0.55 | 19/20 |

## The final policy, and the mechanism

Grid: the final policy from the 25 solved-check starts, against the oracle. Failed checks: over every solved-check that failed in any seed, the share whose worst start is one of the box's 4 corners (4 of the 25 grid starts, 16 %). Spawn: how evenly each run's latest 32 training starts covered the box (algorithms/spawn_coverage.py), averaged over the run; for reference, 32 independent starts leave 6.8 cells empty, at a discrepancy of 0.012.

| arm | final eval return (% of oracle) | grid: on the oracle's step | grid: mean extra steps | p | grid contacts | training contacts/ep | failed checks: worst start at a corner | spawn: empty 5x5 cells | spawn: discrepancy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hPPO, uniform | 1012.4 (99.9 %) | 13 % | +0.94 | - | 3 | 1.09 | 68 % of 127 | 6.81 | 0.0123 |
| hPPO, Sobol' | 1012.6 (100.0 %) | 11 % | +0.98 | 0.95 | 1 | 1.13 | 60 % of 113 | 4.03 | 0.0012 |

Per-seed numbers: each arm's own summary.md. Figures: figures/ and compare_<algo>/figures/.
