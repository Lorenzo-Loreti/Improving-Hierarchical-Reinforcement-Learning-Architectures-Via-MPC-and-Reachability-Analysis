# Disturbed slalom: |w_p| <= 0.01, |w_v| <= 0.1

500000 steps per run, every early stop off; medians over seeds [min, max], and a two-sided Mann-Whitney p against ppo_mpc. A clean evaluation: every episode at the goal, no contact (seeds that never have one count as the budget).

| | ppo_mpc (n=10) | hppo (n=10) | ppo (n=10) |
|---|---|---|---|
| eval return, last 100k | 1002.3 [1002.2, 1002.4] | 1012.0 [1011.6, 1012.2], p=0.00018 | 1012.4 [1012.2, 1012.5], p=0.00018 |
| grid gap to undisturbed oracle, last 100k | 10.6 [10.5, 10.7] | 0.7 [0.5, 1.3], p=0.00018 | 0.3 [0.3, 0.5], p=0.00018 |
| eval success rate, last 100k | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| eval contacts/episode, last 100k | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| training contacts/episode, last 100k | 0.00 [0.00, 0.00] | 0.11 [0.08, 0.24], p=6.4e-05 | 0.02 [0.00, 0.04], p=6.4e-05 |
| training contacts/episode, whole run | 0.00 [0.00, 0.00] | 0.61 [0.55, 0.74], p=6.4e-05 | 0.31 [0.23, 0.49], p=6.4e-05 |
| first clean evaluation (k steps) | 56 [41, 61] | 67 [41, 113], p=0.079 | 46 [31, 51], p=0.025 |

ppo_mpc's worker over every run: mean solve 10.3 ms, candidate fallbacks 0.00% of steps, 0 emergencies.

Per seed:

- ppo_mpc seed 1: return 1002.3, gap 10.7, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 41k
- ppo_mpc seed 2: return 1002.4, gap 10.6, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 51k
- ppo_mpc seed 3: return 1002.3, gap 10.6, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 61k
- ppo_mpc seed 4: return 1002.3, gap 10.5, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 61k
- ppo_mpc seed 5: return 1002.2, gap 10.6, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 51k
- ppo_mpc seed 6: return 1002.3, gap 10.6, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 61k
- ppo_mpc seed 7: return 1002.3, gap 10.6, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 51k
- ppo_mpc seed 8: return 1002.4, gap 10.6, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 41k
- ppo_mpc seed 9: return 1002.4, gap 10.7, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 61k
- ppo_mpc seed 10: return 1002.4, gap 10.7, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 61k
- hppo seed 1: return 1011.8, gap 0.8, eval contacts 0.00, training contacts 0.08 (whole run 0.60), first clean 72k
- hppo seed 2: return 1012.0, gap 0.6, eval contacts 0.00, training contacts 0.24 (whole run 0.70), first clean 51k
- hppo seed 3: return 1012.1, gap 0.5, eval contacts 0.00, training contacts 0.17 (whole run 0.56), first clean 113k
- hppo seed 4: return 1011.8, gap 0.7, eval contacts 0.00, training contacts 0.10 (whole run 0.70), first clean 92k
- hppo seed 5: return 1012.1, gap 0.5, eval contacts 0.00, training contacts 0.09 (whole run 0.57), first clean 92k
- hppo seed 6: return 1012.1, gap 0.7, eval contacts 0.00, training contacts 0.15 (whole run 0.55), first clean 61k
- hppo seed 7: return 1011.7, gap 1.1, eval contacts 0.00, training contacts 0.16 (whole run 0.73), first clean 61k
- hppo seed 8: return 1011.6, gap 1.3, eval contacts 0.00, training contacts 0.12 (whole run 0.74), first clean 51k
- hppo seed 9: return 1012.1, gap 1.2, eval contacts 0.00, training contacts 0.09 (whole run 0.61), first clean 72k
- hppo seed 10: return 1012.2, gap 1.0, eval contacts 0.00, training contacts 0.08 (whole run 0.62), first clean 41k
- ppo seed 1: return 1012.3, gap 0.3, eval contacts 0.00, training contacts 0.03 (whole run 0.36), first clean 41k
- ppo seed 2: return 1012.4, gap 0.3, eval contacts 0.00, training contacts 0.01 (whole run 0.49), first clean 51k
- ppo seed 3: return 1012.3, gap 0.3, eval contacts 0.00, training contacts 0.01 (whole run 0.27), first clean 41k
- ppo seed 4: return 1012.2, gap 0.3, eval contacts 0.00, training contacts 0.03 (whole run 0.30), first clean 51k
- ppo seed 5: return 1012.3, gap 0.5, eval contacts 0.00, training contacts 0.01 (whole run 0.23), first clean 31k
- ppo seed 6: return 1012.4, gap 0.3, eval contacts 0.00, training contacts 0.02 (whole run 0.28), first clean 41k
- ppo seed 7: return 1012.5, gap 0.3, eval contacts 0.00, training contacts 0.04 (whole run 0.37), first clean 51k
- ppo seed 8: return 1012.5, gap 0.4, eval contacts 0.00, training contacts 0.00 (whole run 0.27), first clean 41k
- ppo seed 9: return 1012.4, gap 0.3, eval contacts 0.00, training contacts 0.02 (whole run 0.32), first clean 51k
- ppo seed 10: return 1012.5, gap 0.4, eval contacts 0.00, training contacts 0.01 (whole run 0.45), first clean 51k
