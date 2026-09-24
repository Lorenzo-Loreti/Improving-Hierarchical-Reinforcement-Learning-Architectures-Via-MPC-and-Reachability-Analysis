# Cross-algorithm benchmark — slalom and tunnel

**Run:** 2026-09-23, 4 algorithms × 3 seeds × 2 scenarios, 24 runs.
**Companion:** [`worker-termination-avoidance.md`](worker-termination-avoidance.md),
which is why the hierarchical numbers here differ from anything measured before
that fix.

## Protocol

Identical for every algorithm **except where noted below**, so the columns are
comparable with one documented caveat:

- `--total-timesteps 500000` (environment steps).
- `--solved-early-stop` at its default — the one stopping criterion that exists
  in all four scripts. A run stops once it is within `--solved-tolerance` of
  the oracle's optimal return at **all 25 points** of the fixed evaluation grid,
  for 2 consecutive evaluation passes.
- `--early-stop-success-rate 2.0` wherever that flag exists, to disable the
  looser stop and leave exactly one criterion in play everywhere.
- Seeds 1, 2, 3. Every trainer pinned to one thread (`OMP_NUM_THREADS=1`), run
  in parallel.
- Oracle = `algorithms/optimal_solver.py`'s min-time solution, precomputed per
  grid point at startup.

**Reading `% of oracle`:** the oracle mean is over the 25-point fixed grid,
while the evaluation return is over random initial conditions (`--eval-episodes`:
20 for PPO and hPPO, 10 for the two MPC arms — see the caveat below). They are
different distributions, so values slightly above 100 % mean the random draws
were marginally easier than the grid — not that the agent beat the optimum.

### Is the oracle actually optimal?

`optimal_solver.py` is explicit that its SCP scheme is a heuristic, not a
certified global optimum, so "% of oracle" is only meaningful if the
denominator is right. It can be checked against a bound the solver already
carries: `_x_only_min_time_steps` is the closed-form bang-cruise arrival time
for the x-axis *ignoring the corridor entirely*, which is a relaxation of the
real problem and therefore a valid lower bound on arrival time.

| scenario | `rho` | grid points where oracle == lower bound | mean excess (steps) |
| --- | --- | --- | --- |
| tunnel | 0.48 | **25/25** | 0.00 |
| slalom | 0.48 | **25/25** | 0.00 |
| slalom | 2.40 | 19/25 | 0.76 (max 14) |

On both canonical scenarios the oracle **attains the relaxation's lower bound at
every grid point**, so it is exactly optimal there — not approximately. Every
`% of oracle` figure in this document therefore has a denominator that is the
true optimum, and values above 100 % can only be the grid-vs-random
distribution gap described above, never the agent beating a weak oracle.

Two further things follow, and the second is a statement about the task:

- The check stops being a proof at `rho = 2.40`, where 6 of 25 points exceed
  the bound (mean 0.76 steps, one by 14). There the oracle is an upper bound on
  achievable return and could in principle be beaten slightly; the
  `reachability-regime-study.md` figures at that regime should be read with
  that slack in mind. No arm comes near it — the best is 98.7 % — so it changes
  no conclusion.
- On the **canonical slalom the two gates cost exactly zero extra time**. The
  optimal slalom trajectory arrives as fast as an unobstructed straight run.
  That is a precise statement of what makes this task discriminating: at
  `rho = 0.48` the gates are purely a *control-precision* demand, with no
  time/safety trade-off to balance. The trade-off only appears as the plant
  gets sluggish — which is exactly the axis
  [`reachability-regime-study.md`](reachability-regime-study.md) sweeps.

### The one place the protocol is *not* identical

Two evaluation-side defaults were never unified across the four scripts, and
both land on the `steps to solve` column:

| | `--eval-freq` | steps/update | evaluated every | `solved` holds over | `--eval-episodes` |
| --- | --- | --- | --- | --- | --- |
| PPO (flat) | 10 | 1 024 | 10 240 steps | 20 480 steps | 20 |
| hPPO | 5 | 2 048 | 10 240 steps | 20 480 steps | 20 |
| PPO+MPC | 1 | 2 000 | **2 000 steps** | **4 000 steps** | 10 |
| PPO+MPC-reach | 1 | 2 000 | **2 000 steps** | **4 000 steps** | 10 |

So the two MPC arms are checked on a 5× finer grid and have to hold the
criterion for a 5× shorter window. **The bias runs in favour of the MPC arms**:
they can report the first qualifying moment, while PPO and hPPO can only report
at 10 240-step granularity and must sustain the criterion five times longer.

This does not disturb any conclusion drawn here, and on one it cuts the useful
way: flat PPO is reported as the *fastest* arm on the slalom (51k–81k) while
carrying the coarser grid and the longer hold, so its sample-efficiency
advantage over the hierarchies is if anything understated. The reach-vs-fixed-box
head-to-head in §5 is unaffected — both arms are the same script family at
identical `--eval-freq 1` / `--eval-episodes 10`.

The defaults are deliberately left as they are so the numbers already measured
stay reproducible; the reproduction block below pins both flags explicitly for
anyone re-running the cross-algorithm comparison.

## Slalom (oracle mean optimal return = 1013.0)

| algorithm | solved | steps to solve | final return | % of oracle | contacts/ep |
| --- | --- | --- | --- | --- | --- |
| PPO (flat) | 3/3 | 81k / 51k / 51k | 1019 / 1018 / 1018 | 100.5 % | 0 / 0 / 0 |
| PPO+MPC | **0/3** | — | 948 / 943 / 923 | 92.6 % | 1.3 / 1.4 / 1.8 |
| PPO+MPC-reach | 3/3 | 126k / 68k / 68k | 1012 / 1012 / 1006 | 99.7 % | 0 / 0 / 0.1 |
| hPPO | 3/3 | **71k / 81k / 81k** | 1015 / 1016 / 1016 | 100.3 % | 0 / 0 / 0 |

## Tunnel (oracle mean optimal return = 213.0)

| algorithm | solved | steps to solve | final return | % of oracle | contacts/ep |
| --- | --- | --- | --- | --- | --- |
| PPO (flat) | 3/3 | 20k / 20k / 20k | 217 / 217 / 216 | 101.7 % | 0 / 0 / 0 |
| PPO+MPC | 3/3 | 12k / 16k / **8k** | 213 / 214 / 213 | 100.2 % | 0 / 0 / 0 |
| PPO+MPC-reach | 3/3 | 24k / 22k / 22k | 209 / 209 / 209 | 98.3 % | 0 / 0 / 0 |
| hPPO | 3/3 | 30k / 40k / 40k | 212 / 213 / 215 | 100.2 % | 0 / 0 / 0 |

### Since measured: flat PPO and hPPO simplified (2026-09-23)

The PPO and hPPO rows above were measured before both were simplified. The
simplification removed options that were never enabled and changed behaviour
in three ways:

- the entropy autotuner, inert at its learning rate, was replaced by a
  fixed `ent_coef = 0.01`, which is where it had stayed anyway;
- `vf_coef` was dropped, since each actor/critic pair shares no parameters;
- hPPO's manager no longer re-plans on a wall contact (`--replan-on-collision`).

See the class comments of `PPOAgent` and `HPPOAgent`. The rows are left as
measured. Re-running under this same protocol, with 13 seeds on the slalom
because 3 cannot resolve a sample-efficiency change on this task:

| slalom, 13 seeds | solved | steps to solve, median / mean | final return | contacts/ep |
| --- | --- | --- | --- | --- |
| PPO, before | 13/13 | 51k / 59k | 1018.2 | 0 |
| PPO, after | 13/13 | 51k / 56k | 1018.0 | 0 |
| hPPO, before | 13/13 | 82k / 85k | 1015.6 | 0 |
| hPPO, no collision re-plan | 13/13 | 92k / 95k | 1016.0 | 0 |
| hPPO, after (also no autotuner, no `vf_coef`) | 13/13 | 82k / 96k | 1016.1 | 0 |

None of the differences in steps is significant (Mann–Whitney, hPPO after vs
before: p = 0.71; the collision re-plan alone: p = 0.15). The mean moves because
a few seeds sit on a plateau for longer (hPPO after: 184k on seed 7, 133k on
seeds 2 and 3). Final quality is unchanged throughout. On the tunnel both are
unchanged: PPO 20k on all 3 seeds before and after, hPPO 31k / 41k / 41k →
41k / 31k / 31k. The "before" seeds 1–3 reproduce the rows above
(hPPO slalom 72k / 82k / 82k against 71k / 81k / 81k, within one evaluation
interval), so these rows and the table are one protocol.

### Since measured: hPPO's manager minibatches evened out (2026-09-24)

A review of hPPO against flat PPO found one place where the update did not do
what PPO does. The manager's batch is ragged (201–222 transitions per update,
median 216, over the first 100 updates of slalom seed 1), and it was sliced
into a fixed minibatch size of `total // 4`, so whenever the total was not a
multiple of 4 — 73 updates in 100 — each epoch ended on an extra minibatch of
1–3 samples, still a full Adam step. Every update now takes the minibatch
*count* and splits the batch into that many near-equal parts. Flat PPO and the
worker, whose batches divide evenly, are unchanged bit for bit (checked on
seeds 1–3 of both scenarios); the manager is not, so it was re-measured under
the same protocol. Alongside it, as a separate arm on the unchanged code,
`--clip-vloss false`, the one remaining algorithmic difference between an
hPPO head's update and flat PPO's:

| hPPO, slalom, 13 seeds | solved | steps to solve, median / mean | Mann–Whitney vs HEAD | final return | contacts/ep |
| --- | --- | --- | --- | --- | --- |
| before (the "after" row above, re-run) | 13/13 | 82k / 96k | — | 1016.1 | 0 |
| even manager minibatches | 13/13 | 92k / 97k | p = 0.47 | 1016.0 | 0 |
| before, `--clip-vloss false` | 13/13 | 72k / 80k | p = 0.10 | 1015.5 | 0 |

The re-run reproduces the "after" row exactly, seed for seed. Neither change
moves sample efficiency measurably, and final quality is the same in all three.
The minibatch split is kept because it is what PPO's update means, not because
it is faster. `--clip-vloss` stays on: this protocol stops at "solved" and so
cannot see the late-run stability it was adopted for, which has not been
re-tested since the collapse was traced to the worker's reward. On the tunnel
all three arms solve 3/3 in 31k–41k.

## What the numbers say

**1. The tunnel does not discriminate.** Every algorithm solves it 3/3, within
2 % of the oracle, with no wall contacts, inside 8k–40k steps. It is a
sanity check, not a comparison; nothing in the thesis's headline claims should
rest on it.

**2. On the slalom, three of four solve it 3/3 at essentially the oracle.** The
interesting axis is therefore sample efficiency, not final quality:

```
PPO (flat)  51-81k
PPO+MPC-r   68-126k
hPPO        71-81k
PPO+MPC     never
```

**3. The hierarchy is not paying for its hierarchy — once its worker's reward
is correct.** `hPPO` solves the slalom in 71–81k steps against flat PPO's
51–81k: the same order, not a penalty.

This is worth stating carefully because it inverts a conclusion drawn one round
earlier. With `--worker-success-bonus 20` as the default, `hPPO` also solved
slalom 3/3 at the same final quality — but needed **266k / 296k / 399k** steps,
which reads as a clean 4–6× hierarchy tax over flat PPO. Switching the default
to `--worker-extrinsic-coef 0.02` moved that to 71k / 81k / 81k. The tax was
never the hierarchy; it was a worker that had to discover wall avoidance
indirectly, through a manager acting once every 10 steps, instead of being told
about it. See §5.3 of the companion document.

**4. `PPO+MPC` is the only algorithm that never solves the slalom.** It gets to
92.6 % of the oracle and stalls there with 1.3–1.8 contacts per episode on every
seed. The same algorithm solves the tunnel fastest in the table (8k–16k).

**This has since been diagnosed and fixed**; see
[goal-box-saturation.md](goal-box-saturation.md). The cause was not the
hierarchy or the QP but a single inherited constant: `max_goal_bound = 10.0`,
copied from `script_hppo.py`, where it is genuinely free (hPPO re-normalizes
the goal by the same constant before the worker network sees it, so only the
direction survives). An MPC worker tracks the goal as a hard QP setpoint, and
over one macro-step the plant cannot displace further than
`v_max * manager_freq * dt` = 1.20 m. Past that radius the worker's response
is *exactly flat* — so 88 % of the manager's action range mapped to one
identical full-speed lunge, and the nominally 4-D goal collapsed to "which
direction". Sizing the box from the plant instead takes the slalom from 0/3 to
**9/9 seeds at the oracle return**, eight of them with zero contacts. The rows
above are left as measured, at the old default.

**5. Reachability-aware goals are what `PPO+MPC` needs.** `PPO+MPC-reach` goes
from 0/3 to 3/3 and from 92.6 % to 99.7 %. Constraining goals to the reachable
set helps a worker that must *hit* its goal — which is exactly what a QP
tracking a hard setpoint does, and exactly what a learned worker fed a
re-normalized goal does not.

This conclusion is **overturned** by §4's fix, and the direction matters.
Re-running both arms against each other on the fixed code path — 9 seeds each,
slalom, 300k, identical protocol — leaves nothing for `reachable_goal` to
explain:

| | solved | steps to solve (median, range) | final return | contacts/ep |
| --- | --- | --- | --- | --- |
| `PPO+MPC`, fixed box | 9/9 | 116k, 100k–168k | **1012.7** | 0.011 |
| `PPO+MPC-reach` | 9/9 | 102k, 66k–248k | 1010.2 | 0.033 |

Sample efficiency is indistinguishable (Mann–Whitney p = 0.45), and final
quality is *slightly worse* for reach (p = 0.004) — it arrives ~1.4 steps later
per episode, which at −1/step is the entire gap, and which is what
`accel_split=0.5`'s deliberately conservative inner approximation of the
reachable set predicts. `PPO+MPC-reach`'s original 0/3 → 3/3 was the goal box's
*size*, not its shape.

So the row above should be read as: reachability-aware goals were compensating
for an oversized box. That was a sound fix, but not a necessary one, and it is
not evidence that a worker which must *hit* its goal needs a reachability-aware
goal set — only that it needs a correctly-sized one. Where the *shape* does
start to earn its keep is on a less agile plant; see
[goal-box-saturation.md](goal-box-saturation.md) §8 and
[reachability-regime-study.md](reachability-regime-study.md).

## Caveats

- **Three seeds.** Enough to separate 0/3 from 3/3; not enough to rank
  71k against 81k. Treat the sample-efficiency ordering as coarse.
- **`solved` is a strict criterion** (all 25 grid points within tolerance, held
  twice). `PPO+MPC` reaching 92.6 % without ever passing it is a real failure to
  solve, but "0/3" overstates how far off it is — read the return column too.
- **Early stopping is on**, so this table says nothing about late-run stability.
  That question is the companion document's, and it is why every ablation there
  ran with early stopping disabled instead.
- **Wall-clock is not comparable** and is not reported: runs were executed
  several at a time on a 12-core CPU, so per-run timings reflect contention, not
  the algorithms.

## Reproducing

```bash
# one scenario, one seed, all four algorithms
cd scenarios/slalom/scripts
T="--total-timesteps 500000"; K="--early-stop-success-rate 2.0"
# pin the two flags whose defaults differ across the scripts, so the
# `steps to solve` column really is measured the same way everywhere
P="--eval-freq 1 --eval-episodes 20"
python script_ppo.py           --seed 1 $T $P
python script_ppo_mpc.py       --seed 1 $T $K $P
python script_ppo_mpc_reach.py --seed 1 $T $K $P
python script_hppo.py          --seed 1 $T $K $P
```

The table above was measured at the scripts' own defaults, i.e. *without* `$P`;
pinning it is the stricter protocol, not the one those rows came from.

`script_hppo.py` picks up `--worker-extrinsic-coef 0.02` from its defaults. To
reproduce the pre-fix numbers instead, add `--worker-extrinsic-coef 0` (and
`--worker-success-bonus 0`).
