# Disturbed slalom: |w_p| <= 0.01, |w_v| <= 0.1

ppo_mpc 500000, hppo 1024000, ppo 1024000 steps per run, every early stop off; medians over seeds [min, max], and a two-sided Mann-Whitney p against ppo_mpc. A clean evaluation: every episode at the goal, no contact (seeds that never have one count as the budget).

| | ppo_mpc (n=10) | hppo (n=10) | ppo (n=10) |
|---|---|---|---|
| eval return, last 100k | 991.3 [991.1, 991.4] | -147.6 [-148.4, -146.2], p=0.00018 | -142.7 [-173.1, 1008.4], p=0.14 |
| grid gap to undisturbed oracle, last 100k | 20.9 [20.9, 21.1] | 1157.5 [1156.8, 1159.5], p=0.00018 | 1153.4 [3.4, 1183.7], p=0.14 |
| eval success rate, last 100k | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00], p=1.6e-05 | 0.00 [0.00, 1.00], p=0.0018 |
| eval contacts/episode, last 100k | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.02], p=0.37 | 0.00 [0.00, 0.03], p=0.078 |
| training contacts/episode, last 100k | 0.00 [0.00, 0.00] | 0.19 [0.14, 0.74], p=6.4e-05 | 0.02 [0.01, 0.91], p=6.4e-05 |
| training contacts/episode, whole run | 0.00 [0.00, 0.00] | 0.30 [0.25, 0.39], p=6.4e-05 | 0.16 [0.11, 0.42], p=6.4e-05 |
| first clean evaluation (k steps) | 61 [41, 92] | 1024 [1024, 1024] (never: 10), p=5.9e-05 | 1024 [195, 1024] (never: 6), p=0.00014 |

ppo_mpc's worker over every run: mean solve 95.6 ms, candidate fallbacks 0.00% of steps, 0 emergencies.

Per seed:

- ppo_mpc seed 1: return 991.2, gap 21.1, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 61k
- ppo_mpc seed 2: return 991.3, gap 20.9, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 51k
- ppo_mpc seed 3: return 991.3, gap 21.0, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 72k
- ppo_mpc seed 4: return 991.4, gap 20.9, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 92k
- ppo_mpc seed 5: return 991.3, gap 20.9, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 61k
- ppo_mpc seed 6: return 991.2, gap 20.9, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 61k
- ppo_mpc seed 7: return 991.1, gap 21.1, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 82k
- ppo_mpc seed 8: return 991.2, gap 20.9, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 61k
- ppo_mpc seed 9: return 991.4, gap 20.9, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 41k
- ppo_mpc seed 10: return 991.4, gap 20.9, eval contacts 0.00, training contacts 0.00 (whole run 0.00), first clean 51k
- hppo seed 1: return -147.8, gap 1158.7, eval contacts 0.00, training contacts 0.74 (whole run 0.39), first clean never
- hppo seed 2: return -147.9, gap 1157.3, eval contacts 0.02, training contacts 0.18 (whole run 0.32), first clean never
- hppo seed 3: return -147.3, gap 1157.6, eval contacts 0.00, training contacts 0.22 (whole run 0.30), first clean never
- hppo seed 4: return -147.7, gap 1157.4, eval contacts 0.00, training contacts 0.18 (whole run 0.25), first clean never
- hppo seed 5: return -147.0, gap 1157.1, eval contacts 0.00, training contacts 0.16 (whole run 0.30), first clean never
- hppo seed 6: return -146.2, gap 1156.8, eval contacts 0.00, training contacts 0.14 (whole run 0.26), first clean never
- hppo seed 7: return -147.8, gap 1158.9, eval contacts 0.00, training contacts 0.16 (whole run 0.28), first clean never
- hppo seed 8: return -146.5, gap 1157.3, eval contacts 0.00, training contacts 0.20 (whole run 0.32), first clean never
- hppo seed 9: return -147.4, gap 1157.8, eval contacts 0.00, training contacts 0.27 (whole run 0.35), first clean never
- hppo seed 10: return -148.4, gap 1159.5, eval contacts 0.00, training contacts 0.30 (whole run 0.29), first clean never
- ppo seed 1: return -142.7, gap 1153.3, eval contacts 0.00, training contacts 0.01 (whole run 0.17), first clean never
- ppo seed 2: return 1006.8, gap 15.8, eval contacts 0.03, training contacts 0.13 (whole run 0.18), first clean 573k
- ppo seed 3: return 1006.6, gap 5.3, eval contacts 0.02, training contacts 0.47 (whole run 0.30), first clean 819k
- ppo seed 4: return -144.1, gap 1153.7, eval contacts 0.00, training contacts 0.01 (whole run 0.11), first clean never
- ppo seed 5: return -143.4, gap 1153.6, eval contacts 0.00, training contacts 0.01 (whole run 0.15), first clean never
- ppo seed 6: return -142.8, gap 1153.6, eval contacts 0.00, training contacts 0.01 (whole run 0.13), first clean never
- ppo seed 7: return 1008.4, gap 3.4, eval contacts 0.01, training contacts 0.05 (whole run 0.21), first clean 195k
- ppo seed 8: return 429.9, gap 964.1, eval contacts 0.00, training contacts 0.91 (whole run 0.42), first clean 983k
- ppo seed 9: return -173.1, gap 1183.4, eval contacts 0.00, training contacts 0.01 (whole run 0.14), first clean never
- ppo seed 10: return -172.6, gap 1183.7, eval contacts 0.00, training contacts 0.02 (whole run 0.12), first clean never
