# Seed-study comparison

| algorithm | seeds | solved within budget | first solve median / mean | final eval return | % of oracle | grid: starts reached on the oracle's step | grid: mean extra steps | grid: contacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 20 | 20/20 | 26k / 26k | 1012.6 | 100.0 % | 70 % of 500 | +0.30 | 0 |
| hPPO | 20 | 20/20 | 31k / 33k | 1012.3 | 99.9 % | 46 % of 500 | +0.58 | 0 |
| PPO+MPC | 10 | 10/10 | 72k / 67k | 1012.6 | 100.0 % | 80 % of 250 | +0.20 | 0 |
