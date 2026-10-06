# Disturbed slalom: |w_p| <= 0.005, |w_v| <= 0.05

500000 steps per run, every early stop off; medians over seeds [min, max], and a two-sided Mann-Whitney p against ppo_mpc. A clean evaluation: every episode at the goal, no contact (seeds that never have one count as the budget).

| | ppo_mpc (n=10) | hppo (n=10) | ppo (n=10) |
|---|---|---|---|
| eval return, last 100k | 1008.3 [1008.2, 1008.3] | 1012.5 [1012.0, 1012.7], p=0.00018 | 1012.8 [1012.7, 1013.0], p=0.00018 |
| grid gap to undisturbed oracle, last 100k | 4.6 [4.6, 4.6] | 0.3 [0.2, 0.8], p=0.00018 | -0.1 [-0.1, 1.0], p=0.00018 |
| eval success rate, last 100k | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| eval contacts/episode, last 100k | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01], p=0.37 |
| training contacts/episode, last 100k | 0.00 [0.00, 0.00] | 0.11 [0.07, 0.17], p=6.4e-05 | 0.02 [0.00, 0.20], p=6.4e-05 |
| training contacts/episode, whole run | 0.00 [0.00, 0.00] | 0.55 [0.44, 0.59], p=6.4e-05 | 0.31 [0.27, 0.45], p=6.4e-05 |
| first clean evaluation (k steps) | 46 [41, 61] | 61 [51, 82], p=0.0037 | 41 [31, 72], p=0.096 |

ppo_mpc's worker over every run: mean solve 10.0 ms, candidate fallbacks 0.00% of steps, 0 emergencies.

Per seed:

- ppo_mpc seed 1: return 1008.2, gap 4.6, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 41k
- ppo_mpc seed 2: return 1008.2, gap 4.6, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 41k
- ppo_mpc seed 3: return 1008.2, gap 4.6, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 51k
- ppo_mpc seed 4: return 1008.2, gap 4.6, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 61k
- ppo_mpc seed 5: return 1008.2, gap 4.6, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 51k
- ppo_mpc seed 6: return 1008.3, gap 4.6, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 41k
- ppo_mpc seed 7: return 1008.3, gap 4.6, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 61k
- ppo_mpc seed 8: return 1008.3, gap 4.6, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 41k
- ppo_mpc seed 9: return 1008.3, gap 4.6, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 51k
- ppo_mpc seed 10: return 1008.3, gap 4.6, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 41k
- hppo seed 1: return 1012.4, gap 0.3, eval contacts 0.00, training contacts 0.10 (whole run 0.59), first clean 82k
- hppo seed 2: return 1012.5, gap 0.2, eval contacts 0.00, training contacts 0.16 (whole run 0.48), first clean 61k
- hppo seed 3: return 1012.5, gap 0.2, eval contacts 0.00, training contacts 0.15 (whole run 0.56), first clean 61k
- hppo seed 4: return 1012.2, gap 0.4, eval contacts 0.00, training contacts 0.07 (whole run 0.44), first clean 51k
- hppo seed 5: return 1012.0, gap 0.6, eval contacts 0.00, training contacts 0.10 (whole run 0.51), first clean 61k
- hppo seed 6: return 1012.6, gap 0.2, eval contacts 0.00, training contacts 0.10 (whole run 0.56), first clean 51k
- hppo seed 7: return 1012.6, gap 0.2, eval contacts 0.00, training contacts 0.08 (whole run 0.56), first clean 82k
- hppo seed 8: return 1012.7, gap 0.8, eval contacts 0.00, training contacts 0.13 (whole run 0.50), first clean 61k
- hppo seed 9: return 1012.3, gap 0.5, eval contacts 0.00, training contacts 0.17 (whole run 0.54), first clean 61k
- hppo seed 10: return 1012.6, gap 0.2, eval contacts 0.00, training contacts 0.12 (whole run 0.57), first clean 61k
- ppo seed 1: return 1012.8, gap -0.1, eval contacts 0.00, training contacts 0.03 (whole run 0.31), first clean 31k
- ppo seed 2: return 1012.8, gap 1.0, eval contacts 0.00, training contacts 0.04 (whole run 0.31), first clean 31k
- ppo seed 3: return 1012.7, gap -0.1, eval contacts 0.00, training contacts 0.01 (whole run 0.33), first clean 41k
- ppo seed 4: return 1012.7, gap -0.1, eval contacts 0.01, training contacts 0.20 (whole run 0.45), first clean 72k
- ppo seed 5: return 1012.8, gap -0.1, eval contacts 0.00, training contacts 0.00 (whole run 0.27), first clean 41k
- ppo seed 6: return 1012.9, gap -0.0, eval contacts 0.00, training contacts 0.01 (whole run 0.31), first clean 51k
- ppo seed 7: return 1012.9, gap -0.1, eval contacts 0.00, training contacts 0.06 (whole run 0.30), first clean 41k
- ppo seed 8: return 1012.9, gap -0.1, eval contacts 0.00, training contacts 0.01 (whole run 0.29), first clean 41k
- ppo seed 9: return 1012.8, gap -0.1, eval contacts 0.00, training contacts 0.05 (whole run 0.31), first clean 41k
- ppo seed 10: return 1013.0, gap -0.1, eval contacts 0.00, training contacts 0.01 (whole run 0.30), first clean 31k
