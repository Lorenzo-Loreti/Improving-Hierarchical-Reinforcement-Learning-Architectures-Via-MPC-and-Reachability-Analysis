# Seed-study comparison

| algorithm | seeds | solved within budget | first solve median / mean | final eval return | % of oracle | grid: starts reached on the oracle's step | grid: mean extra steps | grid: contacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO, uniform | 20 | 20/20 | 51k / 53k | 1013.3 | 100.0 % | 79 % of 500 | +0.21 | 1 |
| PPO, Sobol' | 20 | 20/20 | 51k / 53k | 1013.3 | 100.0 % | 80 % of 500 | +0.20 | 0 |

First-solve steps, Mann-Whitney U (two-sided): p = 0.897. A seed that never solved ranks after every seed that did.
