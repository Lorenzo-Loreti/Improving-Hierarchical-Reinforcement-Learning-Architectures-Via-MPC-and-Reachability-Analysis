# hPPO on the slalom: seed study

19 seeds, 204800 steps each, early stops off. Trajectories replay `final.pt`. Representative seed (median first solve): **11**.

- Solved (strict solved-check, held twice) within the budget: **0/19**
- Solved at the last evaluation: 0/19
- Final eval return: mean 971.8, 96.0 % of the oracle on the same starts

Grid columns replay the checkpoint from the 25 points of the solved-check grid; extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, negative = the agent arrives first). Effort: the episode's effort term (the env's effort_penalty * sum of (||u||/u_max)^2), mean over the grid; the oracle's is -0.040. See the notes on the oracle in algorithms/study.py.

| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end | grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts | grid: mean effort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | - | 959.8 | 94.9 % | 1.00 | no | 25/25 | 2.08 / 4 | 99.0 | 26 | -0.052 |
| 2 | - | 959.0 | 94.8 % | 1.00 | no | 25/25 | 3.00 / 5 | 100.0 | 26 | -0.046 |
| 3 | - | 959.9 | 94.9 % | 1.00 | no | 25/25 | 1.92 / 4 | 99.0 | 26 | -0.052 |
| 4 | - | 955.3 | 94.4 % | 1.10 | no | 25/25 | 1.40 / 3 | 100.0 | 27 | -0.050 |
| 5 | - | 1008.4 | 99.7 % | 0.00 | no | 25/25 | 3.88 / 8 | 54.0 | 2 | -0.052 |
| 6 | - | 960.3 | 94.9 % | 1.00 | no | 25/25 | 1.64 / 3 | 100.0 | 27 | -0.049 |
| 7 | - | 960.0 | 94.9 % | 1.00 | no | 25/25 | 1.92 / 4 | 99.0 | 26 | -0.050 |
| 9 | - | 959.1 | 94.8 % | 1.00 | no | 25/25 | 2.60 / 5 | 100.0 | 26 | -0.051 |
| 10 | - | 958.8 | 94.8 % | 1.00 | no | 25/25 | 2.96 / 4 | 104.0 | 28 | -0.046 |
| 11 (rep.) | - | 959.9 | 94.8 % | 1.00 | no | 25/25 | 2.12 / 4 | 100.0 | 26 | -0.049 |
| 12 | - | 959.7 | 94.8 % | 1.00 | no | 25/25 | 2.20 / 4 | 100.0 | 26 | -0.046 |
| 13 | - | 957.6 | 94.6 % | 1.05 | no | 25/25 | 1.68 / 3 | 100.0 | 27 | -0.046 |
| 14 | - | 958.9 | 94.8 % | 1.00 | no | 25/25 | 2.96 / 4 | 200.0 | 28 | -0.048 |
| 15 | - | 957.6 | 94.6 % | 1.05 | no | 25/25 | 1.72 / 3 | 100.0 | 27 | -0.052 |
| 16 | - | 1008.1 | 99.6 % | 0.00 | no | 25/25 | 3.72 / 6 | 6.0 | 0 | -0.051 |
| 17 | - | 1008.8 | 99.7 % | 0.00 | no | 25/25 | 3.40 / 5 | 5.0 | 0 | -0.049 |
| 18 | - | 1007.9 | 99.6 % | 0.00 | no | 25/25 | 4.16 / 7 | 7.0 | 0 | -0.055 |
| 19 | - | 1007.9 | 99.6 % | 0.00 | no | 25/25 | 3.96 / 6 | 50.0 | 1 | -0.056 |
| 20 | - | 957.7 | 94.6 % | 1.05 | no | 25/25 | 1.72 / 3 | 100.0 | 27 | -0.048 |
