# hPPO on the slalom across --worker-extrinsic-coef

Worker observation: `velocity`; goal box 10 m in x, 2.5 m in y. 1000000 steps per run, early stops off. Conservative floor of the coefficient (terminal = value of stalling): **0.0109**; see the docstring of scenarios/slalom/scripts/sweep_extrinsic_coef.py. Probes replay each seed's `final.pt` from the 25 points of the solved-check grid (oracle mean return 1013.0); a gap is oracle return - agent return, averaged over the grid. Probe columns are medians over seeds. p: two-sided Mann-Whitney U on first-solve steps against coef 0, unsolved seeds ranked last.

| coef | terminal / contact (worker units) | solved | first solve, median of solved seeds (range) | p | solved-checks passed after first solve | solved at end | grid gap | contacts/ep | extrinsic share of reward | goal share of action var. | gap, forward goal | gap, mirrored goal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | +0.0 / -0.00 | 7/10 | 358k (195k-666k) | - | 45 % | 2/10 | 37.2 | 0.66 | 0 % | 94 % | 2574.9 | 10129.8 |

## Per seed

| coef | seed | first solve | checks passed after | solved at end | grid gap | reached goal | contacts/ep | extrinsic share | goal share | gap, forward | gap, mirrored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 358k | 68 % | yes | 0.9 | 25/25 | 0.00 | 0 % | 95 % | 3180.6 | 10233.0 |
| 0 | 2 | 666k | 44 % | no | 8.3 | 25/25 | 0.00 | 0 % | 96 % | 4030.5 | 10213.0 |
| 0 | 3 | - | - | no | 56.2 | 25/25 | 1.00 | 0 % | 94 % | 10173.0 | 9321.0 |
| 0 | 4 | - | - | no | 159.6 | 25/25 | 3.16 | 0 % | 96 % | 441.2 | 541.8 |
| 0 | 5 | 492k | 31 % | no | 167.2 | 25/25 | 3.28 | 0 % | 91 % | 4965.5 | 10178.1 |
| 0 | 6 | 297k | 44 % | no | 1.4 | 25/25 | 0.00 | 0 % | 96 % | 1760.3 | 10081.5 |
| 0 | 7 | 430k | 7 % | no | 107.2 | 25/25 | 1.72 | 0 % | 82 % | 1485.4 | 9229.9 |
| 0 | 8 | 195k | 47 % | no | 18.1 | 25/25 | 0.32 | 0 % | 97 % | 10263.0 | 10263.0 |
| 0 | 9 | - | - | no | 87.5 | 25/25 | 1.80 | 0 % | 91 % | 1969.2 | 9863.0 |
| 0 | 10 | 307k | 73 % | yes | 3.6 | 25/25 | 0.04 | 0 % | 94 % | 1843.8 | 10201.4 |
