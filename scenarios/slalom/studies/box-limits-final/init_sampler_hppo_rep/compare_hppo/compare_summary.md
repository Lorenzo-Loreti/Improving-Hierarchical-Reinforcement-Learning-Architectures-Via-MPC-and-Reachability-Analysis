# Seed-study comparison

| algorithm | seeds | solved within budget | first solve median / mean | final eval return | % of oracle | grid: starts reached on the oracle's step | grid: mean extra steps | grid: contacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hPPO, uniform | 20 | 20/20 | 72k / 71k | 1012.4 | 99.9 % | 13 % of 500 | +0.94 | 3 |
| hPPO, Sobol' | 20 | 20/20 | 61k / 67k | 1012.6 | 100.0 % | 11 % of 500 | +0.98 | 1 |

First-solve steps, Mann-Whitney U (two-sided): p = 0.190. A seed that never solved ranks after every seed that did.
