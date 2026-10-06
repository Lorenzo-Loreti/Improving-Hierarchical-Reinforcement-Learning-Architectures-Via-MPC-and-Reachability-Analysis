# hPPO on the tunnel: seed study

20 seeds, 204800 steps each, early stops off. Trajectories replay `final.pt`. Representative seed (median first solve): **11**.

- Solved (strict solved-check, held twice) within the budget: **20/20**, first solve median 31k / mean 33k steps (range 31k-41k)
- Solved at the last evaluation: 20/20
- Final eval return: mean 1012.3, 99.9 % of the oracle on the same starts

Grid columns replay the checkpoint from the 25 points of the solved-check grid; extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, negative = the agent arrives first). Effort: the episode's effort term (the env's effort_penalty * sum of (||u||/u_max)^2), mean over the grid; the oracle's is -0.035. See the notes on the oracle in algorithms/study.py.

| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end | grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts | grid: mean effort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 31k | 1011.7 | 99.9 % | 0.00 | yes | 25/25 | 1.08 / 2 | 2.0 | 0 | -0.037 |
| 2 | 31k | 1011.9 | 99.9 % | 0.00 | yes | 25/25 | 0.80 / 1 | 1.0 | 0 | -0.036 |
| 3 | 31k | 1012.4 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.040 |
| 4 | 31k | 1012.3 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.040 |
| 5 | 31k | 1011.9 | 99.9 % | 0.00 | yes | 25/25 | 0.80 / 1 | 1.0 | 0 | -0.036 |
| 6 | 31k | 1012.5 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.040 |
| 7 | 31k | 1012.4 | 100.0 % | 0.00 | yes | 25/25 | 0.56 / 1 | 1.0 | 0 | -0.038 |
| 8 | 41k | 1012.4 | 100.0 % | 0.00 | yes | 25/25 | 0.48 / 1 | 1.0 | 0 | -0.039 |
| 9 | 31k | 1012.1 | 99.9 % | 0.00 | yes | 25/25 | 0.68 / 1 | 1.0 | 0 | -0.038 |
| 10 | 31k | 1012.4 | 99.9 % | 0.00 | yes | 25/25 | 0.40 / 1 | 1.0 | 0 | -0.039 |
| 11 (rep.) | 31k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.40 / 1 | 1.0 | 0 | -0.040 |
| 12 | 31k | 1012.3 | 99.9 % | 0.00 | yes | 25/25 | 0.44 / 1 | 1.0 | 0 | -0.039 |
| 13 | 41k | 1012.5 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.040 |
| 14 | 31k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.40 / 1 | 1.0 | 0 | -0.039 |
| 15 | 41k | 1011.6 | 99.9 % | 0.00 | yes | 25/25 | 1.60 / 2 | 2.0 | 0 | -0.029 |
| 16 | 31k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.040 |
| 17 | 31k | 1012.3 | 99.9 % | 0.00 | yes | 25/25 | 1.00 / 1 | 1.0 | 0 | -0.036 |
| 18 | 41k | 1012.8 | 100.0 % | 0.00 | yes | 25/25 | 0.40 / 1 | 1.0 | 0 | -0.039 |
| 19 | 41k | 1012.6 | 99.9 % | 0.00 | yes | 25/25 | 0.68 / 1 | 1.0 | 0 | -0.037 |
| 20 | 31k | 1012.5 | 99.9 % | 0.00 | yes | 25/25 | 0.80 / 1 | 1.0 | 0 | -0.036 |
