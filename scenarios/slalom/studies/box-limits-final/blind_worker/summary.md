# hPPO on the slalom across --worker-extrinsic-coef

Worker observation: `velocity`; goal box 10 m. 1000000 steps per run, early stops off. Conservative floor of the coefficient (terminal = value of stalling): **0.0109**; see the docstring of scenarios/slalom/scripts/sweep_extrinsic_coef.py. Probes replay each seed's `final.pt` from the 25 points of the solved-check grid (oracle mean return 1013.0); a gap is oracle return - agent return, averaged over the grid. Probe columns are medians over seeds. p: two-sided Mann-Whitney U on first-solve steps against coef 0.02, unsolved seeds ranked last.

| coef | terminal / contact (worker units) | solved | first solve, median of solved seeds (range) | p | solved-checks passed after first solve | solved at end | grid gap | contacts/ep | extrinsic share of reward | goal share of action var. | gap, forward goal | gap, mirrored goal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.02 (default) | +20.0 / -1.00 | 8/10 | 461k (113k-840k) | - | 89 % | 7/10 | 0.4 | 0.00 | 72 % | 64 % | 436.9 | 999.5 |
| 0 | +0.0 / -0.00 | 9/10 | 195k (92k-410k) | 0.07 | 51 % | 5/10 | 6.4 | 0.00 | 0 % | 96 % | 1801.3 | 4631.0 |

## Per seed

| coef | seed | first solve | checks passed after | solved at end | grid gap | reached goal | contacts/ep | extrinsic share | goal share | gap, forward | gap, mirrored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.02 | 1 | 143k | 92 % | no | 0.1 | 25/25 | 0.00 | 69 % | 64 % | 412.7 | 1137.5 |
| 0.02 | 2 | 451k | 92 % | yes | 0.8 | 25/25 | 0.00 | 71 % | 81 % | 475.5 | 1512.1 |
| 0.02 | 3 | 113k | 74 % | yes | 0.2 | 25/25 | 0.00 | 73 % | 59 % | 382.7 | 843.0 |
| 0.02 | 4 | 666k | 94 % | yes | 0.5 | 25/25 | 0.00 | 72 % | 44 % | 415.2 | 50.7 |
| 0.02 | 5 | 471k | 80 % | yes | 0.4 | 25/25 | 0.00 | 72 % | 63 % | 461.1 | 1321.1 |
| 0.02 | 6 | 328k | 92 % | yes | 0.0 | 25/25 | 0.00 | 71 % | 51 % | 453.1 | 137.0 |
| 0.02 | 7 | - | - | no | 46.5 | 25/25 | 0.92 | 73 % | 84 % | 1651.7 | 4958.6 |
| 0.02 | 8 | 840k | 100 % | yes | 0.4 | 25/25 | 0.00 | 68 % | 48 % | 420.8 | 861.5 |
| 0.02 | 9 | - | - | no | 3.0 | 25/25 | 0.04 | 81 % | 81 % | 217.9 | 836.0 |
| 0.02 | 10 | 666k | 88 % | yes | 0.5 | 25/25 | 0.00 | 81 % | 84 % | 1905.2 | 1953.7 |
| 0 | 1 | 92k | 49 % | no | 7.3 | 25/25 | 0.00 | 0 % | 94 % | 1852.4 | 1859.5 |
| 0 | 2 | 174k | 72 % | yes | 1.6 | 25/25 | 0.00 | 0 % | 97 % | 2234.5 | 10225.0 |
| 0 | 3 | 174k | 59 % | yes | 2.0 | 25/25 | 0.00 | 0 % | 96 % | 1732.7 | 1831.0 |
| 0 | 4 | 410k | 32 % | no | 56.9 | 25/25 | 0.96 | 0 % | 86 % | 1750.2 | 4019.5 |
| 0 | 5 | 297k | 38 % | yes | 3.1 | 25/25 | 0.00 | 0 % | 95 % | 1144.2 | 1452.8 |
| 0 | 6 | 195k | 46 % | no | 7.5 | 25/25 | 0.04 | 0 % | 98 % | 1540.9 | 1507.3 |
| 0 | 7 | 184k | 56 % | yes | 7.6 | 25/25 | 0.08 | 0 % | 99 % | 4528.2 | 10223.0 |
| 0 | 8 | 317k | 83 % | no | 5.6 | 25/25 | 0.00 | 0 % | 96 % | 1928.2 | 5242.5 |
| 0 | 9 | 195k | 22 % | yes | 3.5 | 25/25 | 0.00 | 0 % | 95 % | 2335.6 | 10177.6 |
| 0 | 10 | - | - | no | 1160.5 | 0/25 | 0.00 | 0 % | 74 % | 541.6 | 8549.1 |
