# PPO on the tunnel: seed study

20 seeds, 204800 steps each, early stops off. Trajectories replay `final.pt`. Representative seed (median first solve): **19**.

- Solved (strict solved-check, held twice) within the budget: **20/20**, first solve median 26k / mean 26k steps (range 20k-31k)
- Solved at the last evaluation: 20/20
- Final eval return: mean 1012.6, 100.0 % of the oracle on the same starts

Grid columns replay the checkpoint from the 25 points of the solved-check grid; extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, negative = the agent arrives first). Effort: the episode's effort term (the env's effort_penalty * sum of (||u||/u_max)^2), mean over the grid; the oracle's is -0.035. See the notes on the oracle in algorithms/study.py.

| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end | grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts | grid: mean effort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 31k | 1012.5 | 100.0 % | 0.00 | yes | 25/25 | 0.28 / 1 | 1.0 | 0 | -0.041 |
| 2 | 20k | 1012.4 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.041 |
| 3 | 20k | 1012.4 | 100.0 % | 0.00 | yes | 25/25 | 0.24 / 1 | 1.0 | 0 | -0.041 |
| 4 | 20k | 1012.2 | 100.0 % | 0.00 | yes | 25/25 | 0.36 / 1 | 1.0 | 0 | -0.041 |
| 5 | 20k | 1012.3 | 100.0 % | 0.00 | yes | 25/25 | 0.32 / 1 | 1.0 | 0 | -0.041 |
| 6 | 31k | 1012.5 | 100.0 % | 0.00 | yes | 25/25 | 0.28 / 1 | 1.0 | 0 | -0.042 |
| 7 | 20k | 1012.5 | 100.0 % | 0.00 | yes | 25/25 | 0.28 / 1 | 1.0 | 0 | -0.041 |
| 8 | 31k | 1012.5 | 100.0 % | 0.00 | yes | 25/25 | 0.40 / 1 | 1.0 | 0 | -0.041 |
| 9 | 20k | 1012.4 | 100.0 % | 0.00 | yes | 25/25 | 0.28 / 1 | 1.0 | 0 | -0.041 |
| 10 | 31k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.041 |
| 11 | 31k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.44 / 1 | 1.0 | 0 | -0.041 |
| 12 | 31k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.24 / 1 | 1.0 | 0 | -0.041 |
| 13 | 20k | 1012.4 | 100.0 % | 0.00 | yes | 25/25 | 0.40 / 1 | 1.0 | 0 | -0.042 |
| 14 | 20k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.28 / 1 | 1.0 | 0 | -0.042 |
| 15 | 20k | 1012.7 | 100.0 % | 0.00 | yes | 25/25 | 0.24 / 1 | 1.0 | 0 | -0.041 |
| 16 | 31k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 0.28 / 1 | 1.0 | 0 | -0.042 |
| 17 | 31k | 1012.7 | 100.0 % | 0.00 | yes | 25/25 | 0.44 / 1 | 1.0 | 0 | -0.041 |
| 18 | 31k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 0.28 / 1 | 1.0 | 0 | -0.041 |
| 19 (rep.) | 20k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 0.28 / 1 | 1.0 | 0 | -0.041 |
| 20 | 31k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 0.24 / 1 | 1.0 | 0 | -0.041 |
