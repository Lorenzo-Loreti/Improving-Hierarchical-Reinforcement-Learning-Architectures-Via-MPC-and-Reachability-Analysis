# Independent vs Sobol' training starts on the slalom

Every arm: the same seeds, ppo 1024000, ppo_mpc 204800 steps, every early stop off; the arms of an algorithm differ only in --init-sampler. p: two-sided Mann-Whitney U over the seeds, against the same algorithm's uniform arm (a seed that never solved ranks after every seed that did).

## Learning

First solve: of the strict solved-check, held for two evaluations. Median shift: the Sobol' arm's median first solve minus the uniform arm's, with a 95 % bootstrap interval over seeds. Return AUC: the evaluation return averaged over the whole run, per seed. Late precision: share of the solved-checks passed after the first solve, averaged over the seeds that solved.

| arm | solved | first solve, median / mean (range) | p | median shift (95 % CI) | return AUC, median | p | late precision | p | solved at the end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO, uniform | 9/20 | 748k / 701k (338k-973k) | - | - | 287.8 | - | 52 % | - | 6/20 |
| PPO, Sobol' | 7/20 | 553k / 546k (225k-983k) | 0.8 | -- (-- to --) | -143.2 | 0.31 | 57 % | 0.96 | 6/20 |
| PPO+MPC, uniform | 18/20 | 108k / 113k (92k-195k) | - | - | 811.5 | - | 60 % | - | 11/20 |
| PPO+MPC, Sobol' | 18/20 | 108k / 119k (82k-184k) | 0.65 | +0k (-15k to +36k) | 817.0 | 0.29 | 65 % | 0.75 | 13/20 |

## The final policy, and the mechanism

Grid: the final policy from the 25 solved-check starts, against the oracle. Failed checks: over every solved-check that failed in any seed, the share whose worst start is one of the box's 4 corners (4 of the 25 grid starts, 16 %). Spawn: how evenly each run's latest 32 training starts covered the box (algorithms/spawn_coverage.py), averaged over the run; for reference, 32 independent starts leave 6.8 cells empty, at a discrepancy of 0.012.

| arm | final eval return (% of oracle) | grid: on the oracle's step | grid: mean extra steps | p | grid contacts | training contacts/ep | failed checks: worst start at a corner | spawn: empty 5x5 cells | spawn: discrepancy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO, uniform | 663.3 (65.5 %) | 1 % | +37.91 | - | 15 | 0.28 | 44 % of 1814 | 6.81 | 0.0122 |
| PPO, Sobol' | 377.0 (37.3 %) | 1 % | +67.66 | 0.43 | 189 | 0.25 | 39 % of 1803 | 4.11 | 0.0010 |
| PPO+MPC, uniform | 1010.7 (99.9 %) | 8 % | +1.34 | - | 0 | 0.00 | 53 % of 244 | 6.82 | 0.0124 |
| PPO+MPC, Sobol' | 1010.7 (99.9 %) | 9 % | +1.30 | 0.6 | 0 | 0.00 | 57 % of 248 | 4.09 | 0.0011 |

Per-seed numbers: each arm's own summary.md. Figures: figures/ and compare_<algo>/figures/.
