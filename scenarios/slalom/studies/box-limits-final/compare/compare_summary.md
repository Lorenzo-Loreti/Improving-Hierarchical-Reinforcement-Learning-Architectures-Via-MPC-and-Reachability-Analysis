# Seed-study comparison

| algorithm | seeds | solved within budget | first solve median / mean | final eval return | % of oracle | grid: starts reached on the oracle's step | grid: mean extra steps | grid: contacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 20 | 20/20 | 51k / 53k | 1013.3 | 100.0 % | 79 % of 500 | +0.21 | 1 |
| hPPO | 20 | 20/20 | 72k / 79k | 1012.7 | 100.0 % | 8 % of 500 | +0.96 | 1 |
| PPO+MPC | 10 | 10/10 | 72k / 68k | 1013.1 | 100.0 % | 80 % of 250 | +0.20 | 0 |
