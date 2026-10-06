# Seed-study comparison

| algorithm | seeds | solved within budget | first solve median / mean | final eval return | % of oracle | grid: starts reached on the oracle's step | grid: mean extra steps | grid: contacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 20 | 9/20 | 748k / 701k | 663.3 | 65.5 % | 1 % of 500 | +37.91 | 15 |
| hPPO | 20 | 0/20 | - | -149.8 | -14.8 % | 0 % of 500 | +120.80 | 0 |
| PPO+MPC | 10 | 8/10 | 118k / 125k | 1010.6 | 99.9 % | 8 % of 250 | +1.36 | 0 |
