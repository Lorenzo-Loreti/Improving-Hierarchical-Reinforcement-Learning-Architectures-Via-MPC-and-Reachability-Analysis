# Seed-study comparison

| algorithm | seeds | solved within budget | first solve median / mean | final eval return | % of oracle | grid: starts reached on the oracle's step | grid: mean extra steps | grid: contacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO+MPC, uniform | 20 | 18/20 | 108k / 113k | 1010.7 | 99.9 % | 8 % of 500 | +1.34 | 0 |
| PPO+MPC, Sobol' | 20 | 18/20 | 108k / 119k | 1010.7 | 99.9 % | 9 % of 500 | +1.30 | 0 |

First-solve steps, Mann-Whitney U (two-sided): p = 0.651. A seed that never solved ranks after every seed that did.
