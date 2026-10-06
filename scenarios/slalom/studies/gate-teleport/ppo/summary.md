# PPO on the slalom: seed study

20 seeds, 204800 steps each, early stops off. Trajectories replay `final.pt`. Representative seed (median first solve): **10**.

- Solved (strict solved-check, held twice) within the budget: **0/20**
- Solved at the last evaluation: 0/20
- Final eval return: mean 955.9, 94.5 % of the oracle on the same starts

Grid columns replay the checkpoint from the 25 points of the solved-check grid; extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, negative = the agent arrives first). Effort: the episode's effort term (the env's effort_penalty * sum of (||u||/u_max)^2), mean over the grid; the oracle's is -0.040. See the notes on the oracle in algorithms/study.py.

| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end | grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts | grid: mean effort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | - | 953.2 | 94.2 % | 1.15 | no | 25/25 | 1.08 / 2 | 101.0 | 31 | -0.052 |
| 2 | - | 955.4 | 94.4 % | 1.10 | no | 25/25 | 1.32 / 3 | 101.0 | 29 | -0.052 |
| 3 | - | 955.6 | 94.4 % | 1.10 | no | 25/25 | 1.20 / 3 | 101.0 | 29 | -0.051 |
| 4 | - | 955.4 | 94.4 % | 1.10 | no | 25/25 | 1.32 / 3 | 101.0 | 28 | -0.051 |
| 5 | - | 955.5 | 94.4 % | 1.10 | no | 25/25 | 1.28 / 3 | 101.0 | 28 | -0.052 |
| 6 | - | 955.2 | 94.4 % | 1.10 | no | 25/25 | 1.56 / 3 | 101.0 | 29 | -0.051 |
| 7 | - | 955.6 | 94.4 % | 1.10 | no | 25/25 | 1.00 / 2 | 101.0 | 29 | -0.051 |
| 8 | - | 955.4 | 94.4 % | 1.10 | no | 25/25 | 1.28 / 3 | 102.0 | 29 | -0.050 |
| 9 | - | 955.3 | 94.4 % | 1.10 | no | 25/25 | 1.40 / 3 | 101.0 | 29 | -0.053 |
| 10 (rep.) | - | 955.4 | 94.4 % | 1.10 | no | 25/25 | 1.44 / 3 | 101.0 | 29 | -0.054 |
| 11 | - | 958.2 | 94.7 % | 1.05 | no | 25/25 | 1.20 / 3 | 101.0 | 28 | -0.049 |
| 12 | - | 955.4 | 94.4 % | 1.10 | no | 25/25 | 1.36 / 3 | 101.0 | 29 | -0.052 |
| 13 | - | 955.2 | 94.4 % | 1.10 | no | 25/25 | 1.52 / 3 | 102.0 | 29 | -0.052 |
| 14 | - | 955.7 | 94.4 % | 1.10 | no | 25/25 | 0.96 / 2 | 101.0 | 29 | -0.050 |
| 15 | - | 955.8 | 94.4 % | 1.10 | no | 25/25 | 1.00 / 2 | 100.0 | 29 | -0.051 |
| 16 | - | 955.6 | 94.4 % | 1.10 | no | 25/25 | 1.48 / 3 | 101.0 | 28 | -0.054 |
| 17 | - | 955.7 | 94.4 % | 1.10 | no | 25/25 | 1.44 / 3 | 101.0 | 29 | -0.051 |
| 18 | - | 958.3 | 94.7 % | 1.05 | no | 25/25 | 1.40 / 3 | 101.0 | 29 | -0.049 |
| 19 | - | 958.3 | 94.7 % | 1.05 | no | 25/25 | 1.28 / 3 | 101.0 | 29 | -0.052 |
| 20 | - | 958.2 | 94.7 % | 1.05 | no | 25/25 | 1.24 / 3 | 102.0 | 30 | -0.054 |
