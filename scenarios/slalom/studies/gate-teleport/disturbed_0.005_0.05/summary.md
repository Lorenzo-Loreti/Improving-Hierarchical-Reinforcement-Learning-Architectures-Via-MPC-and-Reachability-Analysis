# Disturbed slalom: |w_p| <= 0.005, |w_v| <= 0.05

ppo 500000, hppo 500000 steps per run, every early stop off; medians over seeds [min, max], and a two-sided Mann-Whitney p against ppo_mpc. A clean evaluation: every episode at the goal, no contact (seeds that never have one count as the budget).

| | ppo (n=10) | hppo (n=10) |
|---|---|---|
| eval return, last 100k | 955.5 [953.2, 1009.5] | 1008.1 [959.2, 1009.4] |
| grid gap to undisturbed oracle, last 100k | 58.0 [8.8, 62.4] | 8.7 [3.7, 56.5] |
| eval success rate, last 100k | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| eval contacts/episode, last 100k | 1.10 [0.00, 1.15] | 0.01 [0.00, 1.01] |
| training contacts/episode, last 100k | 1.13 [0.09, 1.23] | 0.33 [0.13, 1.21] |
| training contacts/episode, whole run | 1.33 [0.43, 1.39] | 0.96 [0.72, 1.59] |
| first clean evaluation (k steps) | 500 [61, 500] (never: 8) | 230 [133, 500] (never: 4) |

Per seed:

- ppo seed 1: return 1007.3, gap 8.8, eval contacts 0.02, training contacts 0.09 (whole run 0.86), first clean 276k
- ppo seed 2: return 955.4, gap 59.3, eval contacts 1.10, training contacts 1.13 (whole run 1.30), first clean never
- ppo seed 3: return 955.3, gap 58.0, eval contacts 1.10, training contacts 1.15 (whole run 1.37), first clean never
- ppo seed 4: return 955.0, gap 59.0, eval contacts 1.11, training contacts 1.18 (whole run 1.31), first clean never
- ppo seed 5: return 953.2, gap 62.4, eval contacts 1.15, training contacts 1.23 (whole run 1.35), first clean never
- ppo seed 6: return 957.8, gap 56.6, eval contacts 1.04, training contacts 1.08 (whole run 1.30), first clean never
- ppo seed 7: return 956.1, gap 57.1, eval contacts 1.09, training contacts 1.12 (whole run 1.39), first clean never
- ppo seed 8: return 1009.5, gap 16.3, eval contacts 0.00, training contacts 0.14 (whole run 0.43), first clean 61k
- ppo seed 9: return 955.5, gap 59.1, eval contacts 1.10, training contacts 1.18 (whole run 1.38), first clean never
- ppo seed 10: return 955.6, gap 58.1, eval contacts 1.10, training contacts 1.11 (whole run 1.34), first clean never
- hppo seed 1: return 1009.3, gap 3.7, eval contacts 0.00, training contacts 0.19 (whole run 0.72), first clean 143k
- hppo seed 2: return 1008.9, gap 6.5, eval contacts 0.00, training contacts 0.25 (whole run 0.86), first clean 184k
- hppo seed 3: return 1009.4, gap 4.4, eval contacts 0.00, training contacts 0.15 (whole run 0.90), first clean 205k
- hppo seed 4: return 960.2, gap 54.1, eval contacts 1.00, training contacts 1.19 (whole run 1.47), first clean never
- hppo seed 5: return 1008.8, gap 5.1, eval contacts 0.00, training contacts 0.13 (whole run 0.82), first clean 133k
- hppo seed 6: return 960.1, gap 55.5, eval contacts 1.00, training contacts 1.21 (whole run 1.59), first clean never
- hppo seed 7: return 959.7, gap 56.5, eval contacts 1.01, training contacts 1.16 (whole run 1.49), first clean never
- hppo seed 8: return 1008.1, gap 10.9, eval contacts 0.00, training contacts 0.18 (whole run 1.03), first clean 256k
- hppo seed 9: return 959.2, gap 54.4, eval contacts 1.01, training contacts 1.14 (whole run 1.51), first clean never
- hppo seed 10: return 1008.1, gap 5.1, eval contacts 0.02, training contacts 0.40 (whole run 0.81), first clean 143k
