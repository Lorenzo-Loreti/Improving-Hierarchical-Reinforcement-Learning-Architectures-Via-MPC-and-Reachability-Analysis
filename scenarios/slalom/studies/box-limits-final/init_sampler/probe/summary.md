# Where flat PPO's policy-gradient noise comes from: the starts or the rest

128 batches per condition and checkpoint (one batch = 8 x 128 steps, as in training). tr Cov: total variance of the surrogate's actor gradient across batches. Ratios are against the independent starts, with 95 % bootstrap intervals over batches. Start share = 1 - fixed / independent: the part of the variance due to which starts a batch drew, the most any start sampler could remove.

| seed | checkpoint | tr Cov, independent | Sobol' / independent | fixed / independent | start share | noise-to-signal, independent |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10k | 6.610e-03 | 0.93 (0.79-1.11) | 1.03 (0.87-1.23) | -0.03 | 1.3 |
| 1 | 31k | 1.685e-02 | 0.81 (0.60-1.06) | 0.94 (0.73-1.21) | +0.06 | 1.3 |
| 1 | 51k | 4.038e-02 | 0.96 (0.71-1.30) | 0.83 (0.61-1.12) | +0.17 | 1.8 |
| 1 | 205k | 1.031e-01 | 1.07 (0.79-1.45) | 1.22 (0.91-1.62) | -0.22 | 105.4 |
| 2 | 10k | 8.097e-03 | 0.87 (0.74-1.04) | 0.89 (0.74-1.07) | +0.11 | 0.3 |
| 2 | 31k | 2.019e-02 | 1.07 (0.85-1.34) | 0.86 (0.67-1.09) | +0.14 | 0.6 |
| 2 | 51k | 4.161e-02 | 0.97 (0.70-1.35) | 1.06 (0.74-1.46) | -0.06 | 0.9 |
| 2 | 205k | 7.387e-02 | 0.97 (0.70-1.30) | 0.96 (0.68-1.34) | +0.04 | 16.9 |
| 3 | 10k | 8.631e-03 | 0.90 (0.73-1.10) | 0.93 (0.73-1.18) | +0.07 | 0.3 |
| 3 | 31k | 2.562e-02 | 1.05 (0.80-1.35) | 0.87 (0.68-1.09) | +0.13 | 0.7 |
| 3 | 51k | 5.702e-02 | 0.89 (0.67-1.18) | 1.21 (0.92-1.60) | -0.21 | 2.0 |
| 3 | 205k | 1.042e-01 | 0.87 (0.65-1.15) | 0.93 (0.69-1.24) | +0.07 | 34.3 |
| **all 12** | | | **0.94** (0.88-1.02) | **0.97** (0.89-1.05) | **+0.03** (-0.05 to +0.11) | |
