# hPPO, uniform on the slalom: seed study

20 seeds, 204800 steps each, early stops off. Trajectories replay `final.pt`. Representative seed (median first solve): **26**.

- Solved (strict solved-check, held twice) within the budget: **20/20**, first solve median 72k / mean 71k steps (range 51k-102k)
- Solved at the last evaluation: 18/20
- Final eval return: mean 1012.4, 99.9 % of the oracle on the same starts

Grid columns replay the checkpoint from the 25 points of the solved-check grid; extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, negative = the agent arrives first). A return gap slightly below 0 (down to about -1.2) is not an agent beating the oracle's time: the goal step's progress term pays for how far past the line the agent crosses, which the oracle does not maximise. See the notes on the oracle in algorithms/study.py.

| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end | grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 21 | 51k | 1013.2 | 100.0 % | 0.00 | yes | 25/25 | 0.56 / 1 | 0.3 | 0 |
| 22 | 82k | 1012.9 | 100.0 % | 0.00 | yes | 25/25 | 0.92 / 1 | 0.6 | 0 |
| 23 | 51k | 1012.9 | 100.0 % | 0.00 | yes | 25/25 | 0.88 / 1 | 0.6 | 0 |
| 24 | 72k | 1012.5 | 99.9 % | 0.00 | yes | 25/25 | 1.08 / 2 | 1.0 | 0 |
| 25 | 72k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 1.0 | 0 |
| 26 (rep.) | 72k | 1012.5 | 99.9 % | 0.00 | yes | 25/25 | 1.24 / 2 | 1.1 | 0 |
| 27 | 51k | 1012.7 | 100.0 % | 0.00 | yes | 25/25 | 0.92 / 1 | 0.6 | 0 |
| 28 | 82k | 1011.9 | 99.9 % | 0.00 | yes | 25/25 | 1.68 / 2 | 1.7 | 0 |
| 29 | 61k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 0.92 / 1 | 0.7 | 0 |
| 30 | 51k | 1013.1 | 100.0 % | 0.00 | yes | 25/25 | 0.56 / 1 | 0.3 | 0 |
| 31 | 72k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 0.76 / 1 | 0.4 | 0 |
| 32 | 102k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.8 | 0 |
| 33 | 72k | 1010.3 | 99.8 % | 0.05 | no | 25/25 | 0.64 / 1 | 50.3 | 1 |
| 34 | 82k | 1009.6 | 99.7 % | 0.05 | no | 25/25 | 1.32 / 2 | 51.3 | 2 |
| 35 | 82k | 1012.7 | 100.0 % | 0.00 | yes | 25/25 | 0.84 / 1 | 0.5 | 0 |
| 36 | 92k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.88 / 1 | 0.6 | 0 |
| 37 | 61k | 1012.4 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.6 | 0 |
| 38 | 82k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.72 / 1 | 0.5 | 0 |
| 39 | 61k | 1012.5 | 100.0 % | 0.00 | yes | 25/25 | 0.88 / 1 | 0.6 | 0 |
| 40 | 72k | 1012.4 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.7 | 0 |
