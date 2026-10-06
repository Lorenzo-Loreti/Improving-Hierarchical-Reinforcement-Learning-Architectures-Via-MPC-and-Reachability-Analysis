# hPPO, Sobol' on the slalom: seed study

20 seeds, 204800 steps each, early stops off. Trajectories replay `final.pt`. Representative seed (median first solve): **16**.

- Solved (strict solved-check, held twice) within the budget: **20/20**, first solve median 72k / mean 79k steps (range 61k-154k)
- Solved at the last evaluation: 18/20
- Final eval return: mean 1012.8, 100.0 % of the oracle on the same starts

Grid columns replay the checkpoint from the 25 points of the solved-check grid; extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, negative = the agent arrives first). A return gap slightly below 0 (down to about -1.2) is not an agent beating the oracle's time: the goal step's progress term pays for how far past the line the agent crosses, which the oracle does not maximise. See the notes on the oracle in algorithms/study.py.

| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end | grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 61k | 1012.7 | 100.0 % | 0.00 | yes | 25/25 | 0.84 / 1 | 0.5 | 0 |
| 2 | 61k | 1012.8 | 100.0 % | 0.00 | no | 25/25 | 0.64 / 1 | 50.3 | 2 |
| 3 | 72k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 0.60 / 1 | 0.3 | 0 |
| 4 | 61k | 1012.4 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.7 | 0 |
| 5 | 61k | 1012.4 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.7 | 0 |
| 6 | 92k | 1012.4 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.8 | 0 |
| 7 | 82k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 0.76 / 1 | 0.5 | 0 |
| 8 | 102k | 1012.4 | 100.0 % | 0.00 | no | 25/25 | 1.20 / 2 | 50.9 | 1 |
| 9 | 82k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.76 / 1 | 0.5 | 0 |
| 10 | 61k | 1012.7 | 100.0 % | 0.00 | yes | 25/25 | 0.88 / 1 | 0.6 | 0 |
| 11 | 61k | 1012.9 | 100.0 % | 0.00 | yes | 25/25 | 0.76 / 1 | 0.6 | 0 |
| 12 | 72k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 0.76 / 1 | 0.5 | 0 |
| 13 | 154k | 1012.2 | 99.9 % | 0.00 | yes | 25/25 | 1.20 / 2 | 1.0 | 0 |
| 14 | 82k | 1012.9 | 100.0 % | 0.00 | yes | 25/25 | 0.76 / 1 | 0.6 | 0 |
| 15 | 113k | 1012.9 | 100.0 % | 0.00 | yes | 25/25 | 0.88 / 1 | 0.8 | 0 |
| 16 (rep.) | 72k | 1013.1 | 100.0 % | 0.00 | yes | 25/25 | 0.72 / 1 | 0.5 | 0 |
| 17 | 61k | 1013.0 | 100.0 % | 0.00 | yes | 25/25 | 0.88 / 1 | 0.8 | 0 |
| 18 | 82k | 1013.1 | 100.0 % | 0.00 | yes | 25/25 | 0.84 / 1 | 0.6 | 0 |
| 19 | 72k | 1013.1 | 100.0 % | 0.00 | yes | 25/25 | 0.88 / 1 | 0.6 | 0 |
| 20 | 72k | 1013.2 | 100.0 % | 0.00 | yes | 25/25 | 0.72 / 1 | 0.5 | 0 |
