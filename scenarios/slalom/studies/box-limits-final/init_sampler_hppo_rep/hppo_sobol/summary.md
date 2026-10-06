# hPPO, Sobol' on the slalom: seed study

20 seeds, 204800 steps each, early stops off. Trajectories replay `final.pt`. Representative seed (median first solve): **34**.

- Solved (strict solved-check, held twice) within the budget: **20/20**, first solve median 61k / mean 67k steps (range 51k-113k)
- Solved at the last evaluation: 19/20
- Final eval return: mean 1012.6, 100.0 % of the oracle on the same starts

Grid columns replay the checkpoint from the 25 points of the solved-check grid; extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, negative = the agent arrives first). A return gap slightly below 0 (down to about -1.2) is not an agent beating the oracle's time: the goal step's progress term pays for how far past the line the agent crosses, which the oracle does not maximise. See the notes on the oracle in algorithms/study.py.

| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end | grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 21 | 72k | 1013.0 | 100.0 % | 0.00 | yes | 25/25 | 0.88 / 1 | 0.6 | 0 |
| 22 | 61k | 1012.4 | 99.9 % | 0.00 | yes | 25/25 | 1.40 / 2 | 1.3 | 0 |
| 23 | 61k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.6 | 0 |
| 24 | 72k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.7 | 0 |
| 25 | 72k | 1012.9 | 100.0 % | 0.00 | no | 25/25 | 0.84 / 1 | 50.5 | 1 |
| 26 | 113k | 1012.9 | 100.0 % | 0.00 | yes | 25/25 | 0.88 / 1 | 0.6 | 0 |
| 27 | 72k | 1013.0 | 100.0 % | 0.00 | yes | 25/25 | 0.60 / 1 | 0.3 | 0 |
| 28 | 51k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 0.84 / 1 | 0.6 | 0 |
| 29 | 61k | 1013.0 | 100.0 % | 0.00 | yes | 25/25 | 0.76 / 1 | 0.4 | 0 |
| 30 | 51k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 0.88 / 1 | 0.8 | 0 |
| 31 | 51k | 1012.5 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.8 | 0 |
| 32 | 61k | 1012.7 | 100.0 % | 0.00 | yes | 25/25 | 0.88 / 1 | 0.6 | 0 |
| 33 | 51k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.88 / 1 | 0.8 | 0 |
| 34 (rep.) | 61k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.96 / 1 | 0.7 | 0 |
| 35 | 72k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.8 | 0 |
| 36 | 102k | 1011.0 | 99.8 % | 0.00 | yes | 25/25 | 2.40 / 3 | 2.8 | 0 |
| 37 | 72k | 1012.0 | 99.9 % | 0.00 | yes | 25/25 | 1.12 / 2 | 1.0 | 0 |
| 38 | 72k | 1012.4 | 100.0 % | 0.00 | yes | 25/25 | 0.92 / 1 | 0.8 | 0 |
| 39 | 61k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.80 / 1 | 0.5 | 0 |
| 40 | 51k | 1012.7 | 100.0 % | 0.00 | yes | 25/25 | 0.64 / 1 | 0.3 | 0 |
