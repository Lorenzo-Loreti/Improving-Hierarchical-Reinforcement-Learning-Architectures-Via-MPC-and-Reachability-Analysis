# PPO on the slalom: seed study

20 seeds, 204800 steps each, early stops off. Trajectories replay `final.pt`. Representative seed (median first solve): **6**.

- Solved (strict solved-check, held twice) within the budget: **20/20**, first solve median 51k / mean 53k steps (range 41k-133k)
- Solved at the last evaluation: 19/20
- Final eval return: mean 1013.3, 100.0 % of the oracle on the same starts

Grid columns replay the checkpoint from the 25 points of the solved-check grid; extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, negative = the agent arrives first). A return gap slightly below 0 (down to about -1.2) is not an agent beating the oracle's time: the goal step's progress term pays for how far past the line the agent crosses, which the oracle does not maximise. See the notes on the oracle in algorithms/study.py.

| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end | grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 61k | 1013.2 | 100.0 % | 0.00 | no | 25/25 | 0.20 / 1 | 49.9 | 1 |
| 2 | 41k | 1013.1 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | -0.0 | 0 |
| 3 | 41k | 1013.1 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | -0.1 | 0 |
| 4 | 51k | 1012.9 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | -0.0 | 0 |
| 5 | 41k | 1013.1 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | -0.1 | 0 |
| 6 (rep.) | 51k | 1013.1 | 100.0 % | 0.00 | yes | 25/25 | 0.24 / 1 | -0.0 | 0 |
| 7 | 51k | 1013.2 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | -0.0 | 0 |
| 8 | 41k | 1013.2 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | -0.0 | 0 |
| 9 | 41k | 1013.1 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | -0.0 | 0 |
| 10 | 51k | 1013.2 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | -0.0 | 0 |
| 11 | 41k | 1013.3 | 100.0 % | 0.00 | yes | 25/25 | 0.24 / 1 | -0.0 | 0 |
| 12 | 61k | 1013.2 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | -0.0 | 0 |
| 13 | 51k | 1013.2 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 0.0 | 0 |
| 14 | 51k | 1013.3 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | -0.0 | 0 |
| 15 | 51k | 1013.5 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | -0.0 | 0 |
| 16 | 41k | 1013.5 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | -0.1 | 0 |
| 17 | 51k | 1013.6 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | -0.0 | 0 |
| 18 | 133k | 1013.5 | 100.0 % | 0.00 | yes | 25/25 | 0.40 / 1 | 0.1 | 0 |
| 19 | 61k | 1013.6 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | -0.0 | 0 |
| 20 | 41k | 1013.6 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | -0.0 | 0 |
