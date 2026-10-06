# hPPO on the slalom: seed study

20 seeds, 1024000 steps each, early stops off. Trajectories replay `final.pt`. Representative seed (median first solve): **10**.

- Solved (strict solved-check, held twice) within the budget: **0/20**
- Solved at the last evaluation: 0/20
- Final eval return: mean -149.8, -14.8 % of the oracle on the same starts

Grid columns replay the checkpoint from the 25 points of the solved-check grid; extra steps = agent arrival step - oracle arrival step (0 = as fast as the oracle, negative = the agent arrives first). Effort: the episode's effort term (the env's effort_penalty * sum of (||u||/u_max)^2), mean over the grid; the oracle's is -0.040. See the notes on the oracle in algorithms/study.py.

| seed | first solve | final eval return | % of oracle | eval contacts/ep | solved at end | grid: reached goal | grid: mean / worst extra steps | grid: worst return gap | grid: contacts | grid: mean effort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | - | -150.5 | -14.9 % | 0.00 | no | 0/25 | 120.80 / 130 | 1170.3 | 0 | -0.051 |
| 2 | - | -147.2 | -14.5 % | 0.00 | no | 0/25 | 120.80 / 130 | 1165.7 | 0 | -0.647 |
| 3 | - | -147.9 | -14.6 % | 0.00 | no | 0/25 | 120.80 / 130 | 1167.9 | 0 | -0.319 |
| 4 | - | -147.0 | -14.5 % | 0.00 | no | 0/25 | 120.80 / 130 | 1165.0 | 0 | -0.464 |
| 5 | - | -146.8 | -14.5 % | 0.00 | no | 0/25 | 120.80 / 130 | 1166.1 | 0 | -0.099 |
| 6 | - | -166.0 | -16.4 % | 0.00 | no | 0/25 | 120.80 / 130 | 1187.8 | 0 | -0.579 |
| 7 | - | -176.5 | -17.4 % | 0.00 | no | 0/25 | 120.80 / 130 | 1195.2 | 0 | -0.828 |
| 8 | - | -145.7 | -14.4 % | 0.00 | no | 0/25 | 120.80 / 130 | 1165.6 | 0 | -0.454 |
| 9 | - | -176.1 | -17.4 % | 0.00 | no | 0/25 | 120.80 / 130 | 1194.1 | 0 | -0.533 |
| 10 (rep.) | - | -145.6 | -14.4 % | 0.00 | no | 0/25 | 120.80 / 130 | 1166.8 | 0 | -0.580 |
| 11 | - | -146.6 | -14.5 % | 0.00 | no | 0/25 | 120.80 / 130 | 1168.7 | 0 | -0.251 |
| 12 | - | -145.8 | -14.4 % | 0.00 | no | 0/25 | 120.80 / 130 | 1165.2 | 0 | -0.230 |
| 13 | - | -144.8 | -14.3 % | 0.00 | no | 0/25 | 120.80 / 130 | 1166.5 | 0 | -0.665 |
| 14 | - | -144.4 | -14.3 % | 0.00 | no | 0/25 | 120.80 / 130 | 1165.5 | 0 | -0.190 |
| 15 | - | -144.6 | -14.3 % | 0.00 | no | 0/25 | 120.80 / 130 | 1166.7 | 0 | -0.099 |
| 16 | - | -143.7 | -14.2 % | 0.00 | no | 0/25 | 120.80 / 130 | 1165.6 | 0 | -0.369 |
| 17 | - | -145.3 | -14.4 % | 0.00 | no | 0/25 | 120.80 / 130 | 1167.3 | 0 | -0.482 |
| 18 | - | -142.7 | -14.1 % | 0.00 | no | 0/25 | 120.80 / 130 | 1168.1 | 0 | -0.542 |
| 19 | - | -143.9 | -14.2 % | 0.00 | no | 0/25 | 120.80 / 130 | 1166.6 | 0 | -0.399 |
| 20 | - | -144.8 | -14.3 % | 0.00 | no | 0/25 | 120.80 / 130 | 1168.1 | 0 | -0.189 |
