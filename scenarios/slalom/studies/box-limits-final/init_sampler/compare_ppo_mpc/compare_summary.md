# Seed-study comparison

| algorithm | seeds | solved within budget | first solve median / mean | final eval return | % of oracle | grid: starts reached on the oracle's step | grid: mean extra steps | grid: contacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO+MPC, uniform | 20 | 20/20 | 72k / 69k | 1013.3 | 100.0 % | 80 % of 500 | +0.20 | 0 |
| PPO+MPC, Sobol' | 20 | 20/20 | 72k / 69k | 1013.3 | 100.0 % | 80 % of 500 | +0.20 | 0 |

First-solve steps, Mann-Whitney U (two-sided): p = 0.722. A seed that never solved ranks after every seed that did.
