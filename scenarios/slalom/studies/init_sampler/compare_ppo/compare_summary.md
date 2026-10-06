# Seed-study comparison

| algorithm | seeds | solved within budget | first solve median / mean | final eval return | % of oracle | grid: starts reached on the oracle's step | grid: mean extra steps | grid: contacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO, uniform | 20 | 9/20 | 748k / 701k | 663.3 | 65.5 % | 1 % of 500 | +37.91 | 15 |
| PPO, Sobol' | 20 | 7/20 | 553k / 546k | 377.0 | 37.3 % | 1 % of 500 | +67.66 | 189 |

First-solve steps, Mann-Whitney U (two-sided): p = 0.795. A seed that never solved ranks after every seed that did.
