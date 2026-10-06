# PPO+MPC, uniform on the slalom: seed study

20 seeds, 204800 steps each, early stops off. Trajectories replay `final.pt`. Representative seed (median first solve): **4**.

- Solved (strict solved-check, held twice) within the budget: **18/20**, first solve median 108k / mean 113k steps (range 92k-195k)
- Solved at the last evaluation: 11/20
- Final eval return: mean 1010.7, 99.9 % of the oracle on the same starts

Grid columns replay the checkpoint from the 25 points of the solved-check grid; extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, negative = the agent arrives first). Effort: the episode's effort term (the env's effort_penalty * sum of (||u||/u_max)^2), mean over the grid; the oracle's is -0.040. See the notes on the oracle in algorithms/study.py.

| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end | grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts | grid: mean effort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 123k | 1010.5 | 99.9 % | 0.00 | yes | 25/25 | 1.48 / 3 | 3.0 | 0 | -0.071 |
| 2 | 195k | 1010.9 | 99.9 % | 0.00 | no | 25/25 | 1.16 / 5 | 5.2 | 0 | -0.072 |
| 3 | - | 1010.4 | 99.9 % | 0.00 | no | 25/25 | 1.56 / 5 | 5.1 | 0 | -0.071 |
| 4 (rep.) | 113k | 1010.4 | 99.9 % | 0.00 | yes | 25/25 | 1.28 / 2 | 2.0 | 0 | -0.076 |
| 5 | 113k | 1010.8 | 99.9 % | 0.00 | yes | 25/25 | 1.28 / 4 | 4.1 | 0 | -0.074 |
| 6 | 102k | 1010.8 | 99.9 % | 0.00 | no | 25/25 | 1.32 / 5 | 5.1 | 0 | -0.073 |
| 7 | - | 1010.7 | 99.9 % | 0.00 | no | 25/25 | 1.28 / 5 | 5.1 | 0 | -0.071 |
| 8 | 123k | 1010.5 | 99.9 % | 0.00 | no | 25/25 | 1.48 / 5 | 5.2 | 0 | -0.076 |
| 9 | 123k | 1010.4 | 99.9 % | 0.00 | no | 25/25 | 1.60 / 5 | 5.2 | 0 | -0.074 |
| 10 | 113k | 1010.7 | 99.9 % | 0.00 | yes | 25/25 | 1.16 / 2 | 2.0 | 0 | -0.069 |
| 11 | 113k | 1010.9 | 99.9 % | 0.00 | yes | 25/25 | 0.96 / 2 | 2.0 | 0 | -0.068 |
| 12 | 164k | 1010.5 | 99.9 % | 0.00 | no | 25/25 | 1.56 / 5 | 5.1 | 0 | -0.072 |
| 13 | 102k | 1010.6 | 99.9 % | 0.00 | yes | 25/25 | 1.20 / 2 | 2.0 | 0 | -0.071 |
| 14 | 92k | 1010.5 | 99.9 % | 0.00 | yes | 25/25 | 1.40 / 4 | 4.1 | 0 | -0.076 |
| 15 | 92k | 1010.7 | 99.9 % | 0.00 | yes | 25/25 | 1.40 / 3 | 3.0 | 0 | -0.069 |
| 16 | 92k | 1010.7 | 99.9 % | 0.00 | no | 25/25 | 1.48 / 5 | 5.1 | 0 | -0.073 |
| 17 | 102k | 1010.8 | 99.9 % | 0.00 | yes | 25/25 | 1.12 / 2 | 2.0 | 0 | -0.068 |
| 18 | 92k | 1010.7 | 99.8 % | 0.00 | no | 25/25 | 1.72 / 5 | 5.1 | 0 | -0.069 |
| 19 | 92k | 1010.9 | 99.9 % | 0.00 | yes | 25/25 | 1.12 / 2 | 2.0 | 0 | -0.068 |
| 20 | 92k | 1010.8 | 99.8 % | 0.00 | yes | 25/25 | 1.20 / 2 | 2.0 | 0 | -0.069 |
