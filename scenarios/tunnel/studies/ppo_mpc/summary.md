# PPO+MPC on the tunnel: seed study

10 seeds, 204800 steps each, early stops off. Trajectories replay `final.pt`. Representative seed (median first solve): **2**.

- Solved (strict solved-check, held twice) within the budget: **10/10**, first solve median 72k / mean 67k steps (range 51k-82k)
- Solved at the last evaluation: 10/10
- Final eval return: mean 1012.6, 100.0 % of the oracle on the same starts

Grid columns replay the checkpoint from the 25 points of the solved-check grid; extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, negative = the agent arrives first). Effort: the episode's effort term (the env's effort_penalty * sum of (||u||/u_max)^2), mean over the grid; the oracle's is -0.035. See the notes on the oracle in algorithms/study.py.

| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end | grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts | grid: mean effort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 82k | 1012.5 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.052 |
| 2 (rep.) | 72k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.051 |
| 3 | 82k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.051 |
| 4 | 61k | 1012.5 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.051 |
| 5 | 51k | 1012.5 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.051 |
| 6 | 51k | 1012.7 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.051 |
| 7 | 51k | 1012.7 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.051 |
| 8 | 72k | 1012.7 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.051 |
| 9 | 72k | 1012.6 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.051 |
| 10 | 72k | 1012.7 | 100.0 % | 0.00 | yes | 25/25 | 0.20 / 1 | 1.0 | 0 | -0.051 |
