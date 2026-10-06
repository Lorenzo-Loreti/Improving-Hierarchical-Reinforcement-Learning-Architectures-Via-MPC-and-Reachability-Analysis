# Seed-study comparison

| algorithm | seeds | solved within budget | first solve median / mean | final eval return | % of oracle | grid: starts reached on the oracle's step | grid: mean extra steps | grid: contacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hPPO, uniform | 20 | 20/20 | 72k / 79k | 1012.7 | 100.0 % | 8 % of 500 | +0.96 | 1 |
| hPPO, Sobol' | 20 | 20/20 | 72k / 79k | 1012.8 | 100.0 % | 17 % of 500 | +0.85 | 3 |

First-solve steps, Mann-Whitney U (two-sided): p = 0.879. A seed that never solved ranks after every seed that did.
