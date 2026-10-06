# Disturbed slalom: |w_p| <= 0.005, |w_v| <= 0.05

ppo_mpc 500000, hppo 1024000, ppo 1024000 steps per run, every early stop off; medians over seeds [min, max], and a two-sided Mann-Whitney p against ppo_mpc. A clean evaluation: every episode at the goal, no contact (seeds that never have one count as the budget).

| | ppo_mpc (n=10) | hppo (n=10) | ppo (n=10) |
|---|---|---|---|
| eval return, last 100k | 1001.8 [1001.7, 1001.9] | -146.7 [-157.7, -144.2], p=0.00018 | 1008.3 [-162.9, 1009.8], p=0.47 |
| grid gap to undisturbed oracle, last 100k | 10.3 [10.2, 10.3] | 1157.0 [1155.1, 1167.8], p=0.00018 | 28.7 [2.0, 1176.5], p=1 |
| eval success rate, last 100k | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00], p=1.6e-05 | 1.00 [0.00, 1.00], p=0.034 |
| eval contacts/episode, last 100k | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| training contacts/episode, last 100k | 0.00 [0.00, 0.00] | 0.21 [0.10, 0.29], p=6.4e-05 | 0.08 [0.02, 0.36], p=6.4e-05 |
| training contacts/episode, whole run | 0.00 [0.00, 0.00] | 0.31 [0.27, 0.33], p=6.4e-05 | 0.34 [0.09, 0.78], p=6.4e-05 |
| first clean evaluation (k steps) | 51 [41, 61] | 1024 [1024, 1024] (never: 10), p=5.3e-05 | 532 [307, 1024] (never: 4), p=0.00015 |

ppo_mpc's worker over every run: mean solve 81.0 ms, candidate fallbacks 0.00% of steps, 0 emergencies.

Per seed:

- ppo_mpc seed 1: return 1001.7, gap 10.3, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 51k
- ppo_mpc seed 2: return 1001.8, gap 10.3, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 41k
- ppo_mpc seed 3: return 1001.8, gap 10.3, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 51k
- ppo_mpc seed 4: return 1001.8, gap 10.3, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 61k
- ppo_mpc seed 5: return 1001.9, gap 10.2, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 41k
- ppo_mpc seed 6: return 1001.7, gap 10.2, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 51k
- ppo_mpc seed 7: return 1001.7, gap 10.3, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 61k
- ppo_mpc seed 8: return 1001.7, gap 10.3, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 51k
- ppo_mpc seed 9: return 1001.7, gap 10.2, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 61k
- ppo_mpc seed 10: return 1001.8, gap 10.2, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 51k
- hppo seed 1: return -145.7, gap 1156.3, eval contacts 0.00, training contacts 0.10 (whole run 0.28), first clean never
- hppo seed 2: return -146.7, gap 1156.9, eval contacts 0.00, training contacts 0.21 (whole run 0.30), first clean never
- hppo seed 3: return -157.7, gap 1167.8, eval contacts 0.00, training contacts 0.10 (whole run 0.27), first clean never
- hppo seed 4: return -147.2, gap 1156.8, eval contacts 0.00, training contacts 0.24 (whole run 0.33), first clean never
- hppo seed 5: return -147.2, gap 1157.3, eval contacts 0.00, training contacts 0.18 (whole run 0.31), first clean never
- hppo seed 6: return -145.8, gap 1156.4, eval contacts 0.00, training contacts 0.21 (whole run 0.31), first clean never
- hppo seed 7: return -146.4, gap 1157.1, eval contacts 0.00, training contacts 0.26 (whole run 0.30), first clean never
- hppo seed 8: return -146.6, gap 1157.6, eval contacts 0.00, training contacts 0.13 (whole run 0.31), first clean never
- hppo seed 9: return -147.7, gap 1157.7, eval contacts 0.00, training contacts 0.24 (whole run 0.28), first clean never
- hppo seed 10: return -144.2, gap 1155.1, eval contacts 0.00, training contacts 0.29 (whole run 0.33), first clean never
- ppo seed 1: return 1007.9, gap 5.6, eval contacts 0.00, training contacts 0.31 (whole run 0.40), first clean 522k
- ppo seed 2: return 1008.9, gap 8.7, eval contacts 0.00, training contacts 0.36 (whole run 0.78), first clean 461k
- ppo seed 3: return 1009.8, gap 2.0, eval contacts 0.00, training contacts 0.05 (whole run 0.31), first clean 307k
- ppo seed 4: return 1009.0, gap 2.9, eval contacts 0.00, training contacts 0.06 (whole run 0.38), first clean 532k
- ppo seed 5: return -126.7, gap 1138.5, eval contacts 0.00, training contacts 0.02 (whole run 0.12), first clean never
- ppo seed 6: return -162.9, gap 1176.5, eval contacts 0.00, training contacts 0.10 (whole run 0.09), first clean never
- ppo seed 7: return 1009.6, gap 2.1, eval contacts 0.00, training contacts 0.06 (whole run 0.45), first clean 502k
- ppo seed 8: return -143.1, gap 1156.7, eval contacts 0.00, training contacts 0.13 (whole run 0.20), first clean never
- ppo seed 9: return 1008.7, gap 48.7, eval contacts 0.00, training contacts 0.34 (whole run 0.56), first clean 532k
- ppo seed 10: return -142.0, gap 1153.1, eval contacts 0.00, training contacts 0.02 (whole run 0.11), first clean never
