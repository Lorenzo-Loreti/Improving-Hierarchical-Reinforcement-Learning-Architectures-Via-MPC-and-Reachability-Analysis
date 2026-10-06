# PPO+MPC on the slalom: seed study

10 seeds, 204800 steps each, early stops off. Trajectories replay `final.pt`. Representative seed (median first solve): **1**.

- Solved (strict solved-check, held twice) within the budget: **8/10**, first solve median 118k / mean 125k steps (range 102k-195k)
- Solved at the last evaluation: 4/10
- Final eval return: mean 1010.6, 99.9 % of the oracle on the same starts

Grid columns replay the checkpoint from the 25 points of the solved-check grid; extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, negative = the agent arrives first). Effort: the episode's effort term (the env's effort_penalty * sum of (||u||/u_max)^2), mean over the grid; the oracle's is -0.040. See the notes on the oracle in algorithms/study.py.

| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end | grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts | grid: mean effort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (rep.) | 123k | 1010.5 | 99.9 % | 0.00 | yes | 25/25 | 1.48 / 3 | 3.0 | 0 | -0.071 |
| 2 | 195k | 1010.9 | 99.9 % | 0.00 | no | 25/25 | 1.16 / 5 | 5.2 | 0 | -0.072 |
| 3 | - | 1010.4 | 99.9 % | 0.00 | no | 25/25 | 1.56 / 5 | 5.1 | 0 | -0.071 |
| 4 | 113k | 1010.4 | 99.9 % | 0.00 | yes | 25/25 | 1.28 / 2 | 2.0 | 0 | -0.076 |
| 5 | 113k | 1010.8 | 99.9 % | 0.00 | yes | 25/25 | 1.28 / 4 | 4.1 | 0 | -0.074 |
| 6 | 102k | 1010.8 | 99.9 % | 0.00 | no | 25/25 | 1.32 / 5 | 5.1 | 0 | -0.073 |
| 7 | - | 1010.7 | 99.9 % | 0.00 | no | 25/25 | 1.28 / 5 | 5.1 | 0 | -0.071 |
| 8 | 123k | 1010.5 | 99.9 % | 0.00 | no | 25/25 | 1.48 / 5 | 5.2 | 0 | -0.076 |
| 9 | 123k | 1010.4 | 99.9 % | 0.00 | no | 25/25 | 1.60 / 5 | 5.2 | 0 | -0.074 |
| 10 | 113k | 1010.7 | 99.9 % | 0.00 | yes | 25/25 | 1.16 / 2 | 2.0 | 0 | -0.069 |
