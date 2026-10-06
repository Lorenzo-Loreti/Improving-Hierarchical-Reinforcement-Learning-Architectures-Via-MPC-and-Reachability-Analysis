# hPPO on the slalom across --worker-extrinsic-coef

Worker observation: `velocity`; goal box 10 m. 1000000 steps per run, early stops off. Conservative floor of the coefficient (terminal = value of stalling): **0.0109**; see the docstring of scenarios/slalom/scripts/sweep_extrinsic_coef.py. Probes replay each seed's `final.pt` from the 25 points of the solved-check grid (oracle mean return 1013.0); a gap is oracle return - agent return, averaged over the grid. Probe columns are medians over seeds. p: two-sided Mann-Whitney U on first-solve steps against coef 0, unsolved seeds ranked last.

| coef | terminal / contact (worker units) | solved | first solve, median of solved seeds (range) | p | solved-checks passed after first solve | solved at end | grid gap | contacts/ep | extrinsic share of reward | goal share of action var. | gap, forward goal | gap, mirrored goal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | +0.0 / -0.00 | 6/10 | 118k (82k-532k) | - | 21 % | 2/10 | 50.4 | 0.04 | 0 % | 83 % | 494.2 | 5851.8 |

## Per seed

| coef | seed | first solve | checks passed after | solved at end | grid gap | reached goal | contacts/ep | extrinsic share | goal share | gap, forward | gap, mirrored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 133k | 5 % | no | 11.3 | 25/25 | 0.20 | 0 % | 87 % | 1494.7 | 1186.8 |
| 0 | 2 | 82k | 27 % | yes | 0.6 | 25/25 | 0.00 | 0 % | 90 % | 532.6 | 8423.8 |
| 0 | 3 | 184k | 1 % | no | 56.6 | 25/25 | 1.12 | 0 % | 77 % | 432.2 | 698.4 |
| 0 | 4 | 532k | 33 % | no | 5.5 | 25/25 | 0.08 | 0 % | 71 % | 440.9 | 5037.8 |
| 0 | 5 | - | - | no | 1159.6 | 0/25 | 0.04 | 0 % | 82 % | 1174.0 | 9045.5 |
| 0 | 6 | - | - | no | 1156.1 | 0/25 | 0.00 | 0 % | 86 % | 565.8 | 8513.3 |
| 0 | 7 | - | - | no | 1157.8 | 0/25 | 0.00 | 0 % | 69 % | 376.8 | 6665.8 |
| 0 | 8 | - | - | no | 1153.5 | 0/25 | 0.04 | 0 % | 84 % | 455.7 | 8530.6 |
| 0 | 9 | 82k | 28 % | yes | 1.2 | 25/25 | 0.00 | 0 % | 86 % | 1299.6 | 1092.5 |
| 0 | 10 | 102k | 29 % | no | 44.2 | 25/25 | 0.84 | 0 % | 74 % | 442.9 | 663.6 |
