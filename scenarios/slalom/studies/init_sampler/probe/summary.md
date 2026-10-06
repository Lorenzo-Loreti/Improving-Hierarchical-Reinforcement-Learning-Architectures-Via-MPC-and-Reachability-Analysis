# Where flat PPO's policy-gradient noise comes from: the starts or the rest

128 batches per condition and checkpoint (one batch = 8 x 128 steps, as in training). tr Cov: total variance of the surrogate's actor gradient across batches. Ratios are against the independent starts, with 95 % bootstrap intervals over batches. Start share = 1 - fixed / independent: the part of the variance due to which starts a batch drew, the most any start sampler could remove.

| seed | checkpoint | tr Cov, independent | Sobol' / independent | fixed / independent | start share | noise-to-signal, independent |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 102k | 4.548e-02 | 0.96 (0.68-1.35) | 0.87 (0.61-1.23) | +0.13 | 4.8 |
| 1 | 307k | 6.323e-02 | 0.82 (0.60-1.11) | 0.68 (0.50-0.94) | +0.32 | 17.8 |
| 1 | 614k | 7.157e-02 | 1.00 (0.74-1.33) | 1.06 (0.78-1.45) | -0.06 | 72.4 |
| 1 | 1024k | 1.630e-01 | 0.71 (0.52-0.96) | 0.80 (0.56-1.12) | +0.20 | 110.1 |
| 2 | 102k | 2.248e-02 | 1.02 (0.69-1.48) | 1.00 (0.70-1.52) | -0.00 | 0.3 |
| 2 | 307k | 2.598e-01 | 0.92 (0.60-1.44) | 1.07 (0.63-1.66) | -0.07 | 5.6 |
| 2 | 614k | 8.749e-02 | 0.95 (0.71-1.28) | 1.17 (0.87-1.55) | -0.17 | 4.5 |
| 2 | 1024k | 4.564e-01 | 1.17 (0.88-1.57) | 1.09 (0.82-1.49) | -0.09 | 47.4 |
| 3 | 102k | 3.365e-02 | 1.04 (0.91-1.21) | 0.99 (0.86-1.15) | +0.01 | 3.6 |
| 3 | 307k | 6.622e-02 | 0.97 (0.77-1.20) | 0.79 (0.63-0.98) | +0.21 | 0.9 |
| 3 | 614k | 8.073e-02 | 0.82 (0.67-1.03) | 0.94 (0.74-1.22) | +0.06 | 7.3 |
| 3 | 1024k | 1.139e-01 | 0.94 (0.70-1.27) | 1.17 (0.85-1.58) | -0.17 | 43.1 |
| **all 12** | | | **0.94** (0.85-1.02) | **0.96** (0.87-1.05) | **+0.04** (-0.05 to +0.13) | |
