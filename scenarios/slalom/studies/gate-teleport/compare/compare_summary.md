# Seed-study comparison

| algorithm | seeds | solved within budget | first solve median / mean | final eval return | % of oracle | grid: starts reached on the oracle's step | grid: mean extra steps | grid: contacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 20 | 0/20 | - | 955.9 | 94.5 % | 14 % of 500 | +1.29 | 579 |
| hPPO | 19 | 0/19 | - | 971.8 | 96.0 % | 5 % of 475 | +2.58 | 376 |
| PPO+MPC | 10 | 8/10 | 118k / 125k | 1010.6 | 99.9 % | 8 % of 250 | +1.36 | 0 |
