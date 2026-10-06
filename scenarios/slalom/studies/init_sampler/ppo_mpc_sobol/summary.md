# PPO+MPC, Sobol' on the slalom: seed study

20 seeds, 204800 steps each, early stops off. Trajectories replay `final.pt`. Representative seed (median first solve): **12**.

- Solved (strict solved-check, held twice) within the budget: **18/20**, first solve median 108k / mean 119k steps (range 82k-184k)
- Solved at the last evaluation: 13/20
- Final eval return: mean 1010.7, 99.9 % of the oracle on the same starts

Grid columns replay the checkpoint from the 25 points of the solved-check grid; extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, negative = the agent arrives first). Effort: the episode's effort term (the env's effort_penalty * sum of (||u||/u_max)^2), mean over the grid; the oracle's is -0.040. See the notes on the oracle in algorithms/study.py.

| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end | grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts | grid: mean effort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 102k | 1010.7 | 99.9 % | 0.00 | yes | 25/25 | 1.28 / 3 | 3.0 | 0 | -0.073 |
| 2 | 164k | 1010.6 | 99.9 % | 0.00 | yes | 25/25 | 1.28 / 2 | 2.0 | 0 | -0.075 |
| 3 | 123k | 1010.7 | 99.9 % | 0.00 | yes | 25/25 | 1.32 / 4 | 4.1 | 0 | -0.070 |
| 4 | 184k | 1010.3 | 99.9 % | 0.00 | yes | 25/25 | 1.52 / 3 | 3.0 | 0 | -0.071 |
| 5 | 82k | 1010.8 | 99.9 % | 0.00 | no | 25/25 | 1.32 / 5 | 5.1 | 0 | -0.072 |
| 6 | 123k | 1010.5 | 99.9 % | 0.00 | yes | 25/25 | 1.44 / 3 | 3.0 | 0 | -0.074 |
| 7 | - | 1010.6 | 99.9 % | 0.00 | no | 25/25 | 1.28 / 5 | 5.1 | 0 | -0.076 |
| 8 | 92k | 1010.6 | 99.9 % | 0.00 | yes | 25/25 | 1.24 / 2 | 2.0 | 0 | -0.072 |
| 9 | 102k | 1010.5 | 99.9 % | 0.00 | yes | 25/25 | 1.24 / 2 | 2.0 | 0 | -0.072 |
| 10 | 123k | 1010.7 | 99.9 % | 0.00 | yes | 25/25 | 1.24 / 4 | 4.1 | 0 | -0.072 |
| 11 | - | 1010.9 | 99.9 % | 0.00 | yes | 25/25 | 1.12 / 2 | 2.0 | 0 | -0.068 |
| 12 (rep.) | 113k | 1010.5 | 99.9 % | 0.00 | no | 25/25 | 1.36 / 5 | 5.1 | 0 | -0.071 |
| 13 | 174k | 1010.4 | 99.9 % | 0.00 | no | 25/25 | 1.60 / 5 | 5.1 | 0 | -0.078 |
| 14 | 102k | 1010.5 | 99.9 % | 0.00 | yes | 25/25 | 1.36 / 4 | 4.1 | 0 | -0.071 |
| 15 | 92k | 1010.5 | 99.8 % | 0.00 | yes | 25/25 | 1.52 / 3 | 3.0 | 0 | -0.070 |
| 16 | 102k | 1011.3 | 99.9 % | 0.00 | no | 25/25 | 1.08 / 5 | 5.1 | 0 | -0.071 |
| 17 | 102k | 1010.8 | 99.9 % | 0.00 | no | 25/25 | 1.32 / 5 | 5.1 | 0 | -0.070 |
| 18 | 113k | 1010.8 | 99.8 % | 0.00 | no | 25/25 | 1.28 / 5 | 5.2 | 0 | -0.080 |
| 19 | 164k | 1010.9 | 99.9 % | 0.00 | yes | 25/25 | 1.04 / 2 | 2.0 | 0 | -0.070 |
| 20 | 92k | 1010.8 | 99.8 % | 0.00 | yes | 25/25 | 1.16 / 4 | 4.2 | 0 | -0.075 |
