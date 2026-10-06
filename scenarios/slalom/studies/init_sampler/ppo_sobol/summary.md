# PPO, Sobol' on the slalom: seed study

20 seeds, 1024000 steps each, early stops off. Trajectories replay `final.pt`. Representative seed (median first solve): **7**.

- Solved (strict solved-check, held twice) within the budget: **7/20**, first solve median 553k / mean 546k steps (range 225k-983k)
- Solved at the last evaluation: 6/20
- Final eval return: mean 377.0, 37.3 % of the oracle on the same starts

Grid columns replay the checkpoint from the 25 points of the solved-check grid; extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, negative = the agent arrives first). Effort: the episode's effort term (the env's effort_penalty * sum of (||u||/u_max)^2), mean over the grid; the oracle's is -0.040. See the notes on the oracle in algorithms/study.py.

| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end | grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts | grid: mean effort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | - | -113.1 | -11.2 % | 0.00 | no | 0/25 | 120.80 / 130 | 1278.9 | 3 | -0.057 |
| 2 | - | -134.0 | -13.2 % | 0.00 | no | 0/25 | 120.80 / 130 | 1299.6 | 3 | -0.066 |
| 3 | 983k | 1009.7 | 99.8 % | 0.00 | yes | 25/25 | 2.32 / 4 | 4.0 | 0 | -0.048 |
| 4 | 614k | 1010.6 | 99.9 % | 0.00 | no | 25/25 | 1.52 / 3 | 53.1 | 1 | -0.049 |
| 5 | 246k | 1010.9 | 99.9 % | 0.00 | yes | 25/25 | 1.04 / 2 | 2.0 | 0 | -0.050 |
| 6 | 553k | 1010.6 | 99.9 % | 0.00 | yes | 25/25 | 1.32 / 3 | 3.0 | 0 | -0.048 |
| 7 (rep.) | - | -142.9 | -14.1 % | 0.00 | no | 0/25 | 120.80 / 130 | 1163.0 | 0 | -0.063 |
| 8 | - | -143.2 | -14.2 % | 0.00 | no | 0/25 | 120.80 / 130 | 1163.2 | 0 | -0.055 |
| 9 | - | -131.2 | -13.0 % | 0.00 | no | 0/25 | 120.80 / 130 | 10088.4 | 178 | -0.112 |
| 10 | 225k | 1010.8 | 99.9 % | 0.00 | yes | 25/25 | 1.12 / 2 | 2.0 | 0 | -0.051 |
| 11 | - | -114.2 | -11.3 % | 0.00 | no | 0/25 | 120.80 / 130 | 1181.3 | 1 | -0.056 |
| 12 | - | 1000.4 | 98.9 % | 0.00 | no | 25/25 | 11.60 / 15 | 165.1 | 3 | -0.058 |
| 13 | - | -142.7 | -14.1 % | 0.00 | no | 0/25 | 120.80 / 130 | 1163.0 | 0 | -0.060 |
| 14 | 328k | 1010.9 | 99.9 % | 0.00 | yes | 25/25 | 1.04 / 2 | 2.0 | 0 | -0.050 |
| 15 | - | -170.6 | -16.9 % | 0.00 | no | 0/25 | 120.80 / 130 | 1192.2 | 0 | -0.058 |
| 16 | - | -170.4 | -16.8 % | 0.00 | no | 0/25 | 120.80 / 130 | 1192.3 | 0 | -0.058 |
| 17 | - | -140.2 | -13.9 % | 0.00 | no | 0/25 | 120.80 / 130 | 1162.6 | 0 | -0.056 |
| 18 | 870k | 1010.5 | 99.8 % | 0.00 | yes | 25/25 | 1.84 / 4 | 4.0 | 0 | -0.048 |
| 19 | - | 1009.6 | 99.7 % | 0.00 | no | 25/25 | 2.52 / 5 | 5.0 | 0 | -0.052 |
| 20 | - | -140.5 | -13.9 % | 0.00 | no | 0/25 | 120.80 / 130 | 1162.8 | 0 | -0.057 |
