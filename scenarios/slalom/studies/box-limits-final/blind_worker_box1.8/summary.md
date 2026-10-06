# hPPO on the slalom across --worker-extrinsic-coef

Worker observation: `velocity`; goal box 1.8 m. 1000000 steps per run, early stops off. Conservative floor of the coefficient (terminal = value of stalling): **0.0109**; see the docstring of scenarios/slalom/scripts/sweep_extrinsic_coef.py. Probes replay each seed's `final.pt` from the 25 points of the solved-check grid (oracle mean return 1013.0); a gap is oracle return - agent return, averaged over the grid. Probe columns are medians over seeds. p: two-sided Mann-Whitney U on first-solve steps against coef 0, unsolved seeds ranked last.

| coef | terminal / contact (worker units) | solved | first solve, median of solved seeds (range) | p | solved-checks passed after first solve | solved at end | grid gap | contacts/ep | extrinsic share of reward | goal share of action var. | gap, forward goal | gap, mirrored goal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | +0.0 / -0.00 | 0/10 | - | - | - | 0/10 | 1160.7 | 0.00 | 0 % | 82 % | 584.7 | 8324.9 |

## Per seed

| coef | seed | first solve | checks passed after | solved at end | grid gap | reached goal | contacts/ep | extrinsic share | goal share | gap, forward | gap, mirrored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | - | - | no | 1160.9 | 0/25 | 0.00 | 0 % | 93 % | 622.6 | 8757.5 |
| 0 | 2 | - | - | no | 1161.5 | 0/25 | 0.00 | 0 % | 87 % | 537.9 | 8014.7 |
| 0 | 3 | - | - | no | 1160.4 | 0/25 | 0.00 | 0 % | 81 % | 548.2 | 8832.7 |
| 0 | 4 | - | - | no | 1162.2 | 0/25 | 0.00 | 0 % | 85 % | 583.6 | 6432.7 |
| 0 | 5 | - | - | no | 1160.3 | 0/25 | 0.00 | 0 % | 77 % | 585.9 | 8769.5 |
| 0 | 6 | - | - | no | 1159.8 | 0/25 | 0.00 | 0 % | 81 % | 513.3 | 8914.8 |
| 0 | 7 | - | - | no | 1160.3 | 0/25 | 0.00 | 0 % | 82 % | 604.7 | 8635.2 |
| 0 | 8 | - | - | no | 1161.3 | 0/25 | 0.00 | 0 % | 83 % | 461.4 | 3680.4 |
| 0 | 9 | - | - | no | 1161.5 | 0/25 | 0.00 | 0 % | 91 % | 612.1 | 6898.1 |
| 0 | 10 | - | - | no | 1159.2 | 0/25 | 0.00 | 0 % | 73 % | 603.8 | 5664.4 |
