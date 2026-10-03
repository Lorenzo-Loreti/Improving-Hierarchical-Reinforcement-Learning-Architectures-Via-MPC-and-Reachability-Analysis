# Training starts: independent vs Sobol' sampling of the spawn box

**What this document is.** The record of one change to how training episodes
start (2026-10-03): why it was made, what exactly it changes and what it
deliberately leaves alone, how it was checked, and what a 20-seed experiment
on every algorithm says about its effect. It is written so that the
corresponding section of the thesis can be lifted from it.

**Companion documents**

- [`benchmark.md`](benchmark.md) — the evaluation protocol (the min-time
  oracle, the strict solved-check on the 5 x 5 grid) every number here uses.
- [`progression.md`](progression.md) — the algorithms being compared.

**Code**

- `scenarios/slalom/envs/spawn_sampler.py` (and its tunnel copy) — the spawn
  box and the Sobol' stream, with the full rationale in its docstring.
- `scenarios/{slalom,tunnel}/envs/vec_*_env.py` — `_sample_initial`, where the
  sampler is used.
- `algorithms/spawn_coverage.py` — the `spawn/*` metrics every training run
  logs.
- `scenarios/slalom/scripts/study_init_sampler.py` — the experiment.

---

## 1. Summary

**The change.** `--init-sampler sobol` (the environment configs'
`init_sampler`) draws the starts of training episodes from a scrambled Sobol'
sequence instead of independently. Every start is still uniform on the spawn
box, so the objective is the same, but successive starts are spread evenly
over it (randomized quasi-Monte Carlo). Evaluation is unchanged, and the
default, `"uniform"`, reproduces every earlier run bit for bit.

**The experiment.** Flat PPO, hPPO and PPO+MPC on the slalom, both samplers,
20 seeds per arm, 204 800 steps, every early stop off (120 runs); hPPO again
on 20 new seeds; a coarser check on the tunnel.

**The result.** The sampler did its job: the discrepancy of the starts the
policies trained on fell tenfold (0.012 to 0.001). It changed nothing
measurable. The median first solve is the same under both samplers for every
algorithm (flat PPO 51k, hPPO 72k, PPO+MPC 72k steps; p = 0.90, 0.88, 0.72;
hPPO over 40 seeds per arm p = 0.50), and so are the learning curves, the
late precision and the final policy. The one nominal difference (hPPO's
arrival steps, p = 0.038) did not replicate on new seeds.

**Why.** Two measurements explain it. The starts from which the policies
struggle are the corners of the box nearest to or furthest from the gates'
openings, and they stay the hardest under both samplers: they are hard
because of where the gates are, not because they are sampled too rarely.
And the starts are a small part of the policy gradient's noise, at most
about a tenth of its variance (point estimate 3 %); the rest comes from the
sampled actions.

**The decision.** The default stays `"uniform"`. `"sobol"` remains available,
tested and documented.

---

## 2. The question

Every episode starts at rest at a point `(p_x0, p_y0)` of the **spawn box**,
`p_x0 ∈ [0, 2]`, `p_y0 ∈ [-W/4, W/4]` (= `[-1, 1]` for the 4 m corridors).
The initial-state distribution `ρ0` of the MDP is uniform on that box. Until
this change, the vector environment that collects training rollouts drew
every start independently from it: plain Monte Carlo.

The proposal was to make the starts a training run actually sees *more
complete*: to make sure successive episodes start from different parts of the
box instead of, by chance, from the same part several times in a row.

Independent draws do cluster, and in this setting the effect is not
negligible, because a policy update sees few episodes. Flat PPO's batch is
8 x 128 = 1 024 steps and hPPO's 2 048. Early in training, when episodes run
out their 200 steps, that is about 5 (PPO) to 10 (hPPO) episodes per update.
Once episodes last about 80 steps it is about 12 to 25. Counting, over 2 000
repetitions, how many cells of a 5 x 5 partition of the box (the resolution
of the solved-check grid) receive no start at all:

| starts | independent | Sobol' (scrambled) |
| --- | --- | --- |
| 25 | 9.0 empty (max 14) | 6.0 empty (max 12) |
| 32 | 6.8 empty (max 12) | 3.2 empty (max 9) |
| 100 | 0.41 | 0.08 |

With 25 independent starts, more than a third of the box is, on average,
not visited at all. The centred L2-discrepancy (Hickernell 1998), a
resolution-free measure of distance from uniform, drops from 0.016 to 0.002
for 25 starts. (5 x 5 is not a dyadic partition, which is why Sobol' still
leaves some of its cells empty; on the dyadic ones it leaves none, section
4.1.)

The hypothesis to test: a training signal that covers the box evenly at every
point of training gives a lower-variance policy-gradient estimate of the same
objective, and should show up as faster or more reliable convergence to the
oracle, in particular at the corners of the box, which the strict
solved-check requires as much as the centre.

---

## 3. What changes, and what deliberately does not

**`ρ0` does not change.** Under both samplers every single start is uniform
on the box. The objective the agent maximises,
`J(π) = E_{s0 ~ ρ0}[V^π(s0)]`, is therefore the same, and so is the meaning
of everything measured against it: the oracle's optimal returns, the
solved-check, the comparison between algorithms. What changes is only the
*joint* distribution of successive starts: they are no longer independent,
they are spread out. In Monte Carlo terms, this is a variance-reduction
device (randomized quasi-Monte Carlo, RQMC), not a change of problem.

Two alternatives were considered and rejected for that reason:

- **A curriculum or prioritized starts** — reverse curriculum generation
  (Florensa et al., CoRL 2017), prioritized level replay (Jiang et al.,
  ICML 2021) — would oversample starts the agent finds hard. That changes
  `ρ0`, hence the objective, and would make the comparison with the oracle
  and with earlier runs a comparison between different problems.
- **Training on a fixed deterministic grid** is biased (the policy never sees
  the points between grid nodes), and training on the solved-check's own
  grid would let the evaluation points leak into training.

**Evaluation does not change either.** Only the vector environment, which
collects training rollouts, uses the sampler. The plain environment's
`reset(seed=...)` stays an independent draw: evaluation reseeds it for every
episode (`env.reset(seed=run_seed + i)`), so a stream there would only ever
yield its first point, and keeping it independent means both samplers are
evaluated from exactly the same starts. The solved-check uses its fixed grid
as before.

**The default does not change behaviour.** `init_sampler="uniform"` is the
default of both environment configs and of the `--init-sampler` flag, and it
reproduces every earlier run bit for bit (section 6.1).

---

## 4. The sampler

### 4.1 Why Sobol', and why scrambled

The Sobol' sequence (Sobol' 1967) is a low-discrepancy sequence. In two
dimensions it is a **(0, 2)-sequence in base 2**: every aligned block of
`2^m` consecutive points puts exactly one point in every dyadic rectangle
`[a/2^i, (a+1)/2^i) x [b/2^j, (b+1)/2^j)` with `i + j = m`. In words: 32
consecutive episodes start exactly once in each cell of a 4 x 8 grid of the
box, *and* of an 8 x 4, a 2 x 16, a 16 x 2 grid, at every resolution at once.

That property is the reason to prefer it over the simpler alternatives:

- **Stratified (jittered) sampling** over a fixed grid, e.g. one start per
  cell of the 5 x 5 grid in random order, balances the starts only at that
  one resolution, which has to be chosen, and choosing the solved-check's own
  resolution would look tailored to the evaluation.
- **Latin hypercube sampling** (McKay et al. 1979) balances each coordinate
  separately but not the two jointly, and needs a fixed block size.
- **Halton** (Halton 1960) would work equally well in two dimensions;
  Sobol' is the more standard choice in the RQMC literature and has the exact
  net property above.

The plain Sobol' sequence is deterministic and starts at the corner `(0, 0)`.
**Scrambling** randomizes it while keeping the net property: scipy applies a
random linear matrix scramble plus a digital shift (Matoušek 1998), from the
family of Owen's scrambled nets (Owen 1995). After scrambling, every single
point is uniform on the unit square. That is what keeps the sampler unbiased:
any average over the starts is an unbiased estimate of its expectation under
`ρ0`. For Owen's nested scrambling the variance of such an average is provably
never more than a small constant times that of independent sampling, and much
lower for smooth integrands (Owen 1997a, 1997b); the cheaper linear scramble
used here shares the two properties the argument rests on, the net structure
and the uniform marginals. RQMC has been applied to policy-gradient RL
directly, with lower-variance gradient estimates as the result (Arnold,
L'Ecuyer et al., AISTATS 2022).

### 4.2 How it is wired

- `init_sampler: str = "uniform"` in `SlalomEnvConfig` and `TunnelEnvConfig`
  (`"uniform"` or `"sobol"`, validated), exposed as `--init-sampler` by the
  flat PPO, hPPO and PPO+MPC training loops, with the config's value as
  default. The run's `config.json` records it.
- **One stream per vector environment**, shared by its 8 sub-environments.
  `reset(seed)` takes the stream's first 8 points; every auto-reset afterwards
  takes the next one. Environments that finish on the same step take theirs
  in ascending index order (`self.states[mask] = ...` fills rows that way).
  Whichever environment it lands in, the sequence of starts the run sees is
  the Sobol' sequence in order.
- **Seeding.** The stream is fixed by the env's seed, so runs stay
  bit-reproducible. Its scrambling draws from a child of that seed
  (`np.random.SeedSequence(seed).spawn(1)[0]`), not from the generator the
  disturbance comes from. Under `"sobol"` that generator therefore holds the
  disturbance alone; under `"uniform"` starts and disturbance share it,
  interleaved in the order episodes end, as before.
- **Blocks.** Points are drawn ahead, 2^14 = 16 384 at first (more than any
  run here needs: 500k steps at ~70 steps per episode is ~7 200 episodes),
  then doubling, so the number drawn is always a power of 2 — the only case
  in which the balance is exact, and the only one in which scipy does not
  warn.
- **The spawn box has one definition**, `spawn_box(W)` in
  `envs/spawn_sampler.py`. Both environments' resets, the vector
  environments' samplers, the solved-check grid (`optimal_solver.spawn_grid`)
  and the track figure (`visualize_env.plot_env`) read it from there; before,
  it was written out by hand in each of them.

### 4.3 What the starts look like

`figures/starts.png` in the experiment's directory
(`scenarios/slalom/studies/init_sampler/`, redrawn by its `plot` phase; git
ignores it) shows the first 64 starts of a vector environment seeded 1, under
each sampler, on the 5 x 5 partition. Independent: 9 cells still empty after
32 starts and 2 after 64, with visible clusters and gaps. Sobol': 4 after 32
and 0 after 64, with no two starts close together.

---

## 5. Instrumentation: the `spawn/*` metrics

Following the project's practice of instrumenting the mechanism alongside the
change (so that the experiment says *why*, not only *whether*), every training
loop now logs, at every update, how evenly the latest 32 training starts
covered the box (`algorithms/spawn_coverage.py`):

- `spawn/discrepancy` — the centred L2-discrepancy of those starts mapped to
  the unit square (`scipy.stats.qmc.discrepancy`, method `"CD"`). Lower is
  more even.
- `spawn/empty_cells` — how many cells of the 5 x 5 partition none of them
  fell in.

The 32 starts are a few updates' worth, and a power of 2. The window slides,
so it is usually not aligned to a Sobol' block. Reference values for 32
points: independent 0.012 / 6.8 empty cells; Sobol' from the start of the
sequence 0.0006 / 3.2. The metrics are computed from the observations the
vector environment hands back after each reset — the starts the policy really
trained on — and draw no random number, so logging them changes nothing else
in a run.

---

## 6. Checking the implementation

### 6.1 The default is bit-identical

Seven short runs (seed 1) were made before and after the change, all at the
default sampler: flat PPO and hPPO on the slalom and on the tunnel (30 720
steps), flat PPO on the slalom under the disturbance `|w_p| ≤ 0.005`,
`|w_v| ≤ 0.05` (30 720 steps), and PPO+MPC on the slalom (4 096 steps). In
every one, every logged value and every tensor of `final.pt` is identical;
the only differences are the new `spawn/*` keys and, for PPO+MPC, the
wall-clock MIQP solve times. The `"uniform"` arms of the experiment below are
therefore the same algorithms as before the change.

### 6.2 Tests

`scenarios/{slalom,tunnel}/envs/tests/test_spawn_sampler.py` (15 tests each):

- the spawn box, and that every environment reads the same one;
- the stream is fixed by its seed and differs across seeds;
- **every aligned block of 32 points is a (0, 5, 2)-net** — exactly one point
  in every cell of the 1 x 32, 2 x 16, 4 x 8, 8 x 4, 16 x 2 and 32 x 1 grids —
  for the first, second and fourth block;
- **each point is uniform over the scrambling** (Kolmogorov-Smirnov on the
  first and fifth point over 2 000 seeds) — the unbiasedness argument;
- the stream continues past its first block without warnings, as one
  sequence;
- the config defaults to `"uniform"` and rejects unknown samplers;
- `"uniform"` is exactly the original independent draw;
- under `"sobol"`, the vector environment's starts follow the stream in
  environment-index order across auto-resets, stay in the box and are at
  rest, and never touch the disturbance generator;
- the plain environment ignores the sampler.

`algorithms/tests/test_spawn_coverage.py` (5 tests) covers the `spawn/*`
metrics. The full suite passes.

---

## 7. The experiment

### 7.1 Protocol

`scenarios/slalom/scripts/study_init_sampler.py`. Every algorithm the thesis
compares on the slalom — flat PPO, hPPO and PPO+MPC (hPPO's manager over the
robust tube-MPC worker) — is trained twice, once per sampler, on seeds 1-20.
Each of the six arms is an ordinary seed study (`algorithms/study.py`): the
canonical slalom, no disturbance, 204 800 environment steps, every early stop
off, so every run spans the same x-axis and the first solve is read off a
complete run. The two arms of an algorithm differ only in `--init-sampler`;
they share seeds, code, environment, and evaluation starts.

Outcomes, decided before looking at the results:

- **sample efficiency** (primary): the step of the first solve of the strict
  solved-check (every one of the 25 grid starts within 5 reward units of the
  oracle, for two evaluations in a row), and the evaluation return averaged
  over the whole run (an area under the learning curve);
- **late precision**: the share of solved-checks passed after the first
  solve, and whether the last one passes;
- **the final policy**: arrival steps against the oracle from the 25 grid
  starts, and wall contacts;
- **the mechanism**: the `spawn/*` coverage metrics, and where on the grid
  the policy is worst while it has not solved yet.

Comparisons are two-sided Mann-Whitney U tests over seeds, Sobol' against
independent within each algorithm. Around a dozen are made, so a single
p ≈ 0.04 is not evidence on its own; the one such result was replicated on 20
new seeds (section 7.6). First solves are quantised by the evaluation
interval (10 240 steps), which is why several seeds share a value.

Wall clock, 14 runs in parallel on 16 logical cores: ~4.5 min per flat-PPO
run, ~6 min per hPPO run, ~40 min per PPO+MPC run (one MIQP per step).

### 7.2 The sampler did what it was designed to do

In every arm, the training starts were spread as intended, for the whole run:

| | empty 5 x 5 cells, last 32 starts | centred L2-discrepancy, last 32 starts |
| --- | --- | --- |
| independent (all three algorithms) | 6.8 | 0.0123 |
| Sobol' (all three algorithms) | 4.1 | 0.0012 |

(Run averages over the seeds. Per algorithm, independent / Sobol': flat PPO
6.84 / 4.12 empty cells, hPPO 6.83 / 4.12, PPO+MPC 6.81 / 4.13; the
discrepancy is 0.0123-0.0124 / 0.0011-0.0012 in all three. Over time:
`figures/spawn_coverage.png`.) The discrepancy of the starts the policy
actually trained on is ten times lower under Sobol'. Whatever the result
below, it is not because the sampler failed to change the training
distribution.

### 7.3 Learning: no difference

| arm | solved | first solve, median / mean (range) | p | median shift (95 % CI) | return AUC, median | p | late precision | p | solved at the end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO, independent | 20/20 | 51k / 53k (41k-133k) | | | 986.8 | | 97 % | | 19/20 |
| PPO, Sobol' | 20/20 | 51k / 53k (41k-123k) | 0.90 | +0k (-10k to +10k) | 981.2 | 0.69 | 98 % | 1.0 | 20/20 |
| hPPO, independent | 20/20 | 72k / 79k (51k-133k) | | | 963.4 | | 86 % | | 19/20 |
| hPPO, Sobol' | 20/20 | 72k / 79k (61k-154k) | 0.88 | +0k (-20k to +15k) | 957.9 | 0.76 | 89 % | 0.79 | 18/20 |
| PPO+MPC, independent | 20/20 | 72k / 69k (61k-82k) | | | 838.2 | | 100 % | | 20/20 |
| PPO+MPC, Sobol' | 20/20 | 72k / 69k (61k-82k) | 0.72 | +0k (-10k to +5k) | 850.2 | 0.49 | 100 % | 1.0 | 20/20 |

The two samplers give the same median and the same mean first solve for
every algorithm, the same learning curves within their interquartile bands
(`compare_<algo>/figures/compare_learning_curves.png`), and the same late
precision. The bootstrap interval of the median shift bounds what the seeds
leave room for: for flat PPO, an advantage of more than one evaluation
interval (10k steps, ~20 % of the time to solve) is excluded; for hPPO, more
than 20k on seeds 1-20 and more than 10k over all 40 seeds (section 7.6); for
PPO+MPC, more than 10k.

### 7.4 The final policy: no difference

| arm | final eval return (% of oracle) | grid: arrives on the oracle's step | grid: mean extra steps | p | grid contacts | training contacts/ep |
| --- | --- | --- | --- | --- | --- | --- |
| PPO, independent | 1013.3 (100.0 %) | 79 % | +0.21 | | 1 | 0.61 |
| PPO, Sobol' | 1013.3 (100.0 %) | 80 % | +0.20 | 0.30 | 0 | 0.63 |
| hPPO, independent | 1012.7 (100.0 %) | 8 % | +0.96 | | 1 | 1.09 |
| hPPO, Sobol' | 1012.8 (100.0 %) | 17 % | +0.85 | 0.038 | 3 | 1.13 |
| PPO+MPC, independent | 1013.3 (100.0 %) | 80 % | +0.20 | | 0 | 0.00 |
| PPO+MPC, Sobol' | 1013.3 (100.0 %) | 80 % | +0.20 | 1.0 | 0 | 0.00 |

(Grid: the final checkpoint of every seed replayed from the 25 solved-check
starts, 500 rollouts per arm; extra steps = agent arrival step - oracle
arrival step. A seed whose last solved-check fails has, in every case here,
one wall contact on one grid start — a return gap of ~50.)

### 7.5 The independent arms reproduce the earlier studies, seed for seed

The independent arms were retrained rather than read from the earlier seed
studies, which predate several hPPO changes. They reproduce them exactly: the
first solve of every one of the 20 flat-PPO seeds and the 20 hPPO seeds
equals that of the 2026-09-24 studies, and the first solves of PPO+MPC's
seeds 1-10 equal those of the 2026-09-29 study, which had only those ten.
This confirms, at the scale of the whole experiment, that the default changed
nothing (section 6.1), and that the baseline here is the one the rest of the
thesis reports.

### 7.6 The one nominal difference, replicated

One secondary outcome of one algorithm crossed p < 0.05: hPPO's final policy
arrived on the oracle's step from 17 % of the grid starts under Sobol'
against 8 %, mean extra steps +0.85 against +0.96 (p = 0.038). It is a
difference of 0.11 steps (0.011 s), worth at most ~0.1 reward units, and one
test among about a dozen. hPPO was therefore trained again on 20 new seeds
(21-40), both arms, same protocol:

| hPPO | first solve, median / mean (range) | p | grid: arrives on the oracle's step | grid: mean extra steps | p |
| --- | --- | --- | --- | --- | --- |
| seeds 21-40, independent | 72k / 71k (51k-102k) | | 13 % | +0.94 | |
| seeds 21-40, Sobol' | 61k / 67k (51k-113k) | 0.19 | 11 % | +0.98 | 0.95 |
| seeds 1-40, independent | 72k / 75k | | | +0.95 | |
| seeds 1-40, Sobol' | 72k / 73k | 0.50 | | +0.92 | 0.13 |

The grid difference did not replicate. On the new seeds Sobol' arrives on
the oracle's step from 11 % of the grid starts against 13 %, with +0.98
against +0.94 extra steps (p = 0.95); over all 40 seeds per arm, +0.92
against +0.95 (p = 0.13). The p = 0.038 was the kind of false positive a
dozen tests make likely. The new seeds lean the other way on the first solve
(median 61k against 72k, p = 0.19), which does not hold over the 40 seeds
either: 72k against 72k, p = 0.50, a median shift of 0k with a 95 %
bootstrap interval of -10k to +10k; the return AUC gives p = 0.41.
(`scenarios/slalom/studies/init_sampler_hppo_rep/summary.md`.)

### 7.7 The tunnel

A coarser check on the tunnel (`script_ppo.py`/`script_hppo.py` at their
defaults, which stop at the first solve; seeds 1-10, both samplers): every
run solved, flat PPO at 20k or 31k steps (7 vs 8 of 10 seeds at 20k, p = 0.65)
and hPPO at 31k on all 20 runs (p = 1.0). The tunnel is solved within two or
three evaluations by every arm, so it cannot resolve a difference in sample
efficiency; it confirms that the Sobol' path works in the second scenario.

---

## 8. Why it makes no difference

The sampler worked (7.2), and learning did not change (7.3-7.7). Two
measurements explain why.

### 8.1 The hard starts are hard because of the track, not because they are rarely sampled

The motivation was that independent sampling could leave parts of the box,
the corners in particular, under-trained. The solved-check logs, at every
evaluation, the grid start where the policy is furthest from the oracle.
Over every failed check of every seed:

| arm | failed checks | worst start at one of the 4 corners |
| --- | --- | --- |
| PPO, independent | 71 | 61 % |
| PPO, Sobol' | 65 | 65 % |
| hPPO, independent | 143 | 65 % |
| hPPO, Sobol' | 137 | 69 % |
| PPO+MPC, independent | 95 | 72 % |
| PPO+MPC, Sobol' | 94 | 76 % |

The corners are 4 of the 25 grid starts (16 %), yet they are the worst start
in 61-76 % of the failed checks, under both samplers alike, and the map of
where the policies fail looks the same under both (`figures/worst_starts.png`).
For every algorithm, most of the mass is on `(2, -1)`: the start nearest to
gate 1 and furthest from its opening, which spans `p_y ∈ [0.25, 1.75]`, so
that the agent has 2 m of forward travel in which to climb at least 1.25 m.
For flat PPO and hPPO, `(0, +1)`, the start furthest from the goal, is the
other; PPO+MPC's worker handles it, and its failures concentrate on `(2, -1)`
and its neighbour `(2, -0.5)` alone. What makes these starts hard is where
they lie relative to the gates, not how often they are sampled: spreading
the starts out does not make them any easier.

There is also less to fix than the per-update counts suggest. A policy
changes little from one update to the next, and over the ~10 updates in
which it changes noticeably it sees 100-250 episodes. 100 independent starts
leave on average 0.41 of the 25 cells empty: on the time scale that matters
for learning, independent sampling already covers the box.

### 8.2 The starts are a small part of the policy gradient's noise

A start sampler can only reduce the part of the gradient's variance that the
randomness of the starts causes. `scenarios/slalom/scripts/probe_start_variance.py`
measures that part directly for flat PPO. At four checkpoints of three
independent-arm seeds (10k, 31k, 51k ≈ the median first solve, and 205k
steps), it collects 128 training batches (8 x 128 steps, as in training) under
three conditions — independent starts, Sobol' starts, and the very same start
sequence in every batch — with the same action-noise seeds, and estimates the
total variance tr Cov(g) of the surrogate's actor gradient across batches.
By the law of total variance,

    Var(g) = Var_starts( E[g | starts] ) + E_starts( Var(g | starts) ),

and the fixed-starts condition removes the first term entirely, so it is the
floor no start sampler can go below:

| | Sobol' / independent | fixed starts / independent | share of the variance due to the starts |
| --- | --- | --- | --- |
| all 12 checkpoints (geometric mean, 95 % bootstrap CI) | 0.94 (0.88-1.02) | 0.97 (0.89-1.05) | +3 % (-5 % to +11 %) |

No single checkpoint's interval excludes 1
(`scenarios/slalom/studies/init_sampler/probe/summary.md`,
`probe/figures/gradient_variance.png`). The starts account for at most about
a tenth of the gradient's variance; the rest comes from the sampled actions
and the states they lead to. Removing all of the starts' share would be
worth at most what a ~10 % larger batch is worth, which is well inside the
seed-to-seed spread of the first solve. This is consistent with the
geometry: the spawn box is 2 m x 2 m at the entrance of a 10 m track, every
trajectory has to pass the same two gates, and trajectories from different
starts merge before gate 1, so most of every batch is spent in states that
do not depend on where the episode began.

(Caveats of the probe: one algorithm, flat PPO; the floor uses one fixed
start sequence rather than an average over sequences; the variance is a sum
over all actor parameters, which weights them equally.)

---

## 9. Decision

**The default stays `"uniform"`.** The criterion agreed before the experiment
was to switch only on a measured improvement, and there is none: not in
sample efficiency, not in late precision, not in the final policy, for any of
the three algorithms, and the one nominal difference did not replicate.
Switching would also make every result recorded so far (all trained with
independent starts) no longer bit-reproducible at the defaults, for no
gain.

`--init-sampler sobol` stays available. It costs nothing at run time, keeps
the objective unchanged, and is tested. It could matter where the starts are
a larger share of the gradient's noise than here: a spawn box that is large
relative to the task (e.g. starts anywhere in the corridor), much smaller
batches, or tasks whose trajectories from different starts do not merge.

For the thesis, the result is a clean negative one with its mechanism
measured: on this task, independent sampling already covers the
initial-state distribution well enough, because the difficulty of a start
comes from the track geometry and the starts are a small part of the
gradient's noise.

---

## 10. What this does not cover

- One scenario at full power. The tunnel check cannot resolve a difference
  in sample efficiency (7.7).
- 20 seeds per arm (40 for hPPO) exclude large effects only: the bootstrap
  intervals in 7.3 are the honest statement of what was ruled out.
- Undisturbed environment only. Under the bounded disturbance the starts are
  an even smaller share of the noise, so no different outcome is expected,
  but it was not measured.
- Other samplers (stratified, Latin hypercube, Halton) were not trained. The
  probe bounds what any start sampler could do here: at most the starts'
  share of the gradient's variance, about a tenth.

---

## 11. Changes, file by file

| file | change |
| --- | --- |
| `scenarios/{slalom,tunnel}/envs/spawn_sampler.py` | new: `spawn_box(W)`, `SobolSpawnStream`, `INIT_SAMPLERS`; the rationale in the slalom copy's docstring |
| `scenarios/{slalom,tunnel}/envs/config.py` | `init_sampler: str = "uniform"`, validated |
| `scenarios/{slalom,tunnel}/envs/vec_*_env.py` | `spawn_low/high`; `seed()` builds the Sobol' stream; `_sample_initial` draws from either sampler |
| `scenarios/{slalom,tunnel}/envs/*_env.py` | `spawn_low/high` from `spawn_box`; the reset still draws independently |
| `scenarios/{slalom,tunnel}/envs/visualize_env.py`, `slalom/envs/width_profile.py`, `algorithms/optimal_solver.py` | read the spawn box from `spawn_box` / the env instead of restating it |
| `algorithms/{ppo,hppo,ppo_mpc}/*_train.py` | `--init-sampler`; `spawn/*` logged every update |
| `algorithms/spawn_coverage.py` | new: the `spawn/*` metrics |
| `algorithms/study.py` | the training pool extracted into `TrainTask`/`train_runs`, so one queue can train several studies; `train_phase` behaves as before |
| `scenarios/slalom/scripts/study_init_sampler.py` | new: the experiment |
| `scenarios/slalom/scripts/probe_start_variance.py` | new: the gradient-variance probe |
| `scenarios/{slalom,tunnel}/envs/tests/test_spawn_sampler.py`, `algorithms/tests/test_spawn_coverage.py` | new tests (35) |
| `scenarios/{slalom,tunnel}/ENVIRONMENT.md` | the option, in plain words |

---

## 12. Reproducing

```bash
# the experiment: 6 arms x 20 seeds, then the analysis and every figure
python scenarios/slalom/scripts/study_init_sampler.py
# the hPPO replication on new seeds
python scenarios/slalom/scripts/study_init_sampler.py --algos hppo --seeds 21-40 \
    --out scenarios/slalom/studies/init_sampler_hppo_rep
# the gradient-variance probe (needs the experiment's independent PPO arm)
python scenarios/slalom/scripts/probe_start_variance.py
# one training run with Sobol' starts
python scenarios/slalom/scripts/script_ppo.py --seed 1 --init-sampler sobol
# the tunnel check (stops at the first solve)
python scenarios/tunnel/scripts/script_ppo.py --seed 1 --init-sampler sobol --total-timesteps 204800
```

Outputs go to `scenarios/slalom/studies/init_sampler/` (git ignores it):
`summary.md` (the tables of section 7), `figures/` (`starts`,
`spawn_coverage`, `worst_starts`, `first_solves`), `compare_<algo>/` (learning
curves, first-solve distributions, trajectories), `probe/`, and one ordinary
seed-study directory per arm.

---

## References

- Arnold, S. M. R., L'Ecuyer, P., Chen, L., Chen, Y., Sha, F. (2022). Policy
  learning and evaluation with randomized quasi-Monte Carlo. *AISTATS*, PMLR
  151, 1041–1061.
- Florensa, C., Held, D., Wulfmeier, M., Zhang, M., Abbeel, P. (2017).
  Reverse curriculum generation for reinforcement learning. *CoRL*.
- Halton, J. H. (1960). On the efficiency of certain quasi-random sequences
  of points in evaluating multi-dimensional integrals. *Numerische
  Mathematik* 2, 84–90.
- Hickernell, F. J. (1998). A generalized discrepancy and quadrature error
  bound. *Mathematics of Computation* 67, 299–322.
- Jiang, M., Grefenstette, E., Rocktäschel, T. (2021). Prioritized level
  replay. *ICML*.
- Matoušek, J. (1998). On the L2-discrepancy for anchored boxes. *Journal of
  Complexity* 14, 527–556.
- McKay, M. D., Beckman, R. J., Conover, W. J. (1979). A comparison of three
  methods for selecting values of input variables in the analysis of output
  from a computer code. *Technometrics* 21, 239–245.
- Owen, A. B. (1995). Randomly permuted (t, m, s)-nets and (t, s)-sequences.
  In *Monte Carlo and Quasi-Monte Carlo Methods in Scientific Computing*,
  Lecture Notes in Statistics 106, 299–317.
- Owen, A. B. (1997a). Monte Carlo variance of scrambled net quadrature.
  *SIAM Journal on Numerical Analysis* 34, 1884–1910.
- Owen, A. B. (1997b). Scrambled net variance for integrals of smooth
  functions. *Annals of Statistics* 25, 1541–1562.
- Sobol', I. M. (1967). On the distribution of points in a cube and the
  approximate evaluation of integrals. *USSR Computational Mathematics and
  Mathematical Physics* 7, 86–112.
