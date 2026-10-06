# PPO on the slalom: seed study

20 seeds, 1024000 steps each, early stops off. Trajectories replay `final.pt`. Representative seed (median first solve): **2**.

- Solved (strict solved-check, held twice) within the budget: **9/20**, first solve median 748k / mean 701k steps (range 338k-973k)
- Solved at the last evaluation: 6/20
- Final eval return: mean 663.3, 65.5 % of the oracle on the same starts

Grid columns replay the checkpoint from the 25 points of the solved-check grid; extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, negative = the agent arrives first). Effort: the episode's effort term (the env's effort_penalty * sum of (||u||/u_max)^2), mean over the grid; the oracle's is -0.040. See the notes on the oracle in algorithms/study.py.

| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end | grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts | grid: mean effort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 338k | 1011.0 | 99.9 % | 0.00 | yes | 25/25 | 1.04 / 2 | 2.0 | 0 | -0.052 |
| 2 (rep.) | - | -128.8 | -12.7 % | 0.00 | no | 0/25 | 120.80 / 130 | 1244.3 | 2 | -0.058 |
| 3 | - | 1008.9 | 99.7 % | 0.00 | no | 25/25 | 3.04 / 5 | 5.0 | 0 | -0.049 |
| 4 | - | -144.4 | -14.3 % | 0.00 | no | 0/25 | 120.80 / 130 | 1163.2 | 0 | -0.056 |
| 5 | - | 1008.6 | 99.7 % | 0.00 | no | 25/25 | 3.44 / 6 | 6.0 | 0 | -0.050 |
| 6 | - | 1007.8 | 99.6 % | 0.00 | no | 25/25 | 4.28 / 8 | 208.1 | 4 | -0.052 |
| 7 | 614k | 1010.3 | 99.9 % | 0.00 | yes | 25/25 | 1.60 / 4 | 4.0 | 0 | -0.051 |
| 8 | - | -172.1 | -17.0 % | 0.00 | no | 0/25 | 120.80 / 130 | 1192.1 | 0 | -0.056 |
| 9 | - | -143.1 | -14.1 % | 0.00 | no | 0/25 | 120.80 / 130 | 1162.6 | 0 | -0.056 |
| 10 | 922k | 1009.5 | 99.8 % | 0.00 | yes | 25/25 | 2.36 / 4 | 4.0 | 0 | -0.049 |
| 11 | - | -141.4 | -14.0 % | 0.00 | no | 0/25 | 120.80 / 130 | 1162.5 | 0 | -0.058 |
| 12 | 358k | 1010.6 | 99.9 % | 0.00 | no | 25/25 | 1.44 / 4 | 54.1 | 1 | -0.052 |
| 13 | - | 1007.7 | 99.6 % | 0.00 | no | 25/25 | 4.32 / 6 | 156.1 | 3 | -0.054 |
| 14 | 840k | 1010.3 | 99.8 % | 0.00 | no | 25/25 | 1.72 / 4 | 54.1 | 1 | -0.050 |
| 15 | - | 1010.2 | 99.8 % | 0.00 | no | 25/25 | 2.04 / 4 | 104.1 | 2 | -0.050 |
| 16 | 707k | 1010.2 | 99.8 % | 0.00 | yes | 25/25 | 1.92 / 4 | 4.0 | 0 | -0.048 |
| 17 | 748k | 1010.2 | 99.8 % | 0.00 | yes | 25/25 | 1.88 / 4 | 4.0 | 0 | -0.049 |
| 18 | 809k | 1010.2 | 99.8 % | 0.00 | yes | 25/25 | 1.96 / 4 | 4.0 | 0 | -0.048 |
| 19 | 973k | 1010.0 | 99.8 % | 0.00 | no | 25/25 | 2.32 / 5 | 105.1 | 2 | -0.050 |
| 20 | - | -140.1 | -13.8 % | 0.00 | no | 0/25 | 120.80 / 130 | 1162.4 | 0 | -0.062 |
