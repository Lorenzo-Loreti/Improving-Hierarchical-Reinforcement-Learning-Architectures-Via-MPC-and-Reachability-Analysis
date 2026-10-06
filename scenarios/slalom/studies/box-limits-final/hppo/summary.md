# hPPO on the slalom: seed study

20 seeds, 204800 steps each, early stops off. Trajectories replay `final.pt`. Representative seed (median first solve): **6**.

- Solved (strict solved-check, held twice) within the budget: **20/20**, first solve median 72k / mean 79k steps (range 51k-133k)
- Solved at the last evaluation: 19/20
- Final eval return: mean 1012.7, 100.0 % of the oracle on the same starts

Grid columns replay the checkpoint from the 25 points of the solved-check grid; extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, negative = the agent arrives first). A return gap slightly below 0 (down to about -1.2) is not an agent beating the oracle's time: the goal step's progress term pays for how far past the line the agent crosses, which the oracle does not maximise. See the notes on the oracle in algorithms/study.py.

| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end | grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 51k | 1012.5 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.7 | 0 |
| 2 | 61k | 1011.8 | 99.9 % | 0.00 | yes | 25/25 | 1.56 / 2 | 1.2 | 0 |
| 3 | 133k | 1012.5 | 100.0 % | 0.00 | yes | 25/25 | 0.88 / 1 | 0.7 | 0 |
| 4 | 72k | 1012.1 | 99.9 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.9 | 0 |
| 5 | 92k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.84 / 1 | 0.5 | 0 |
| 6 (rep.) | 72k | 1012.5 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.8 | 0 |
| 7 | 61k | 1012.5 | 100.0 % | 0.00 | yes | 25/25 | 0.92 / 1 | 0.8 | 0 |
| 8 | 92k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 0.72 / 1 | 0.4 | 0 |
| 9 | 61k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.88 / 1 | 0.6 | 0 |
| 10 | 113k | 1012.5 | 100.0 % | 0.00 | no | 25/25 | 1.04 / 2 | 51.0 | 1 |
| 11 | 82k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 0.88 / 1 | 0.6 | 0 |
| 12 | 72k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 0.76 / 1 | 0.5 | 0 |
| 13 | 92k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.92 / 1 | 0.7 | 0 |
| 14 | 113k | 1012.7 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.7 | 0 |
| 15 | 72k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.8 | 0 |
| 16 | 51k | 1012.7 | 100.0 % | 0.00 | yes | 25/25 | 1.20 / 2 | 1.3 | 0 |
| 17 | 61k | 1012.9 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.8 | 0 |
| 18 | 61k | 1013.2 | 100.0 % | 0.00 | yes | 25/25 | 0.68 / 1 | 0.4 | 0 |
| 19 | 123k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 1.00 / 1 | 0.8 | 0 |
| 20 | 51k | 1013.0 | 100.0 % | 0.00 | yes | 25/25 | 0.88 / 1 | 0.7 | 0 |
