# From flat PPO to reachability-aware hierarchical MPC

**What this document is.** The spine of the experimental argument: what changes
as control is moved out of the policy and into a model-based worker, what
breaks at each step, and what the evidence says about why. It is written to be
read before the two detail documents it points at, not after.

**Companion documents**

- [`goal-box-saturation.md`](goal-box-saturation.md) — why `PPO+MPC` failed,
  in full, with the diagnosis and the fix.
- [`reachability-regime-study.md`](reachability-regime-study.md) — when
  reachability-aware goals pay, when they cost, and the quantity that predicts
  which.
- [`benchmark.md`](benchmark.md) — the four-algorithm cross-comparison these
  results revise.
- [`worker-termination-avoidance.md`](worker-termination-avoidance.md) — the
  earlier fix to the *learned* hierarchies' worker reward.

---

## 1. The progression, and what each step is actually changing

Three architectures, on the same environment, differing only in where the
control authority lives.

| | who chooses the acceleration | what is learned | what is given |
| --- | --- | --- | --- |
| **flat PPO** | the policy, every step | the entire controller | nothing |
| **PPO+MPC** | a fixed constrained QP | *where to go*, every `c` steps | the controller |
| **PPO+MPC-reach** | the same QP | *where to go*, within a set computed from the plant | the controller **and** the goal set |

Read left to right, this is a progression in how much model knowledge is
injected. Flat PPO learns everything from reward. `PPO+MPC` keeps the low-level
controller fixed and optimal-by-construction, and asks the policy only for a
setpoint. `PPO+MPC-reach` additionally uses the plant model a second time — to
decide *which setpoints are physically askable*.

The thesis question is whether that second use of the model is worth anything.
It is a narrower question than "does hierarchy help", and it has a cleaner
answer.

### The object that matters: the goal space

Everything in this document turns on one design object, so it is worth naming
precisely. The manager emits a normalized action in `[-1, 1]^4`. Something must
map that to a physical goal `(Δp_x, Δp_y, Δv_x, Δv_y)` the worker consumes. That
map is the **goal space**, and it is the entire interface between the learned
layer and the model-based one.

The central finding of this work is that **the goal space is not a
hyperparameter, it is part of the controller**, and that getting it wrong
degrades the architecture more than any of the RL-side choices that were
ablated before it (entropy autotuning, value clipping, advantage
normalization, KL targets — see `benchmark.md` and
`worker-termination-avoidance.md`).

---

## 2. Why the slalom is the discriminating task

Both scenarios in this repo are the same discrete-time double integrator. They
differ only in the corridor.

- **Tunnel** — constant width. The optimal policy is "full speed ahead". Every
  algorithm tested solves it 3/3 within 8k–40k steps, with no wall contacts.
  It does not discriminate, and no claim here rests on it.
- **Slalom** — two narrow, laterally-offset gates (`x ∈ (4,5]` requires
  `y ∈ [0.25, 1.75]`; `x ∈ (7,8]` requires `y ∈ [-1.75, -0.25]`), full width
  elsewhere.

The slalom discriminates because it demands *graded lateral control*: the agent
must arrive at a specific lateral band at a specific longitudinal position, and
then hold it. The tunnel demands only saturation. That distinction is what makes
the goal space's resolution measurable at all — and it is why a defect that is
invisible on the tunnel is fatal on the slalom.

One structural property of the worker matters throughout: `MPCWorker` looks its
corridor bound up at the **current** `p_x` and holds it constant across the
horizon. It therefore cannot see a gate until the agent is already inside it.
Gate avoidance is not the worker's job — it rests entirely on the manager
placing the agent correctly *before* the gate. This is what makes the goal
space load-bearing rather than merely convenient.

---

## 3. Step 1 → 2: adding the MPC worker, and what it cost

On canonical slalom, the original comparison read:

| | solved | steps | % of oracle | contacts/ep |
| --- | --- | --- | --- | --- |
| flat PPO | 3/3 | 51k–81k | 100.5 % | 0 |
| `PPO+MPC` | **0/3** | — | 92.6 % | 1.3–1.8 |

Replacing a learned controller with an optimal one made the system *worse*.
That is the result the rest of this work explains, and the explanation is not
about hierarchy.

**The cause was a single inherited constant.** `script_ppo_mpc.py` set its goal
box to `max_goal_bound = 10.0`, copied from `script_hppo.py`. In hPPO that
constant is free: the worker is a network that receives the goal re-normalized
by the same constant, so it cancels exactly and an unreachable goal is simply a
direction. An MPC worker tracks the goal as a **hard QP setpoint**, and over one
macro-step the plant cannot displace further than `v_max · c · dt` = 1.20 m.
Beyond that radius the worker's response is *exactly flat* — measured, not
argued:

| `Δy` commanded | +0.40 | +0.90 | +1.20 | +2.00 | +5.00 | +10.00 |
| --- | --- | --- | --- | --- | --- | --- |
| end-of-segment `p_y` | +0.352 | +0.709 | +0.847 | +0.906 | +0.910 | +0.910 |

So 88 % of the manager's action range mapped to one identical full-speed lunge.
A nominally 4-D goal collapsed to roughly two bits per segment: the sign of each
axis. The trained policies confirm it — 100 % of their `Δx` goals and 75–90 % of
their `Δy` goals were outside the reachable set, with mean `|Δy|` of 2.5–4.6 m
against a corridor half-width of 2.0 m.

**Two controls establish the worker was never the limitation.** The oracle
solves the slalom collision-free at return 1013. Feeding *the same* `MPCWorker`
oracle-derived goals at the same cadence returns 1013.3 with zero contacts,
using only `Δy ∈ [-0.37, +0.13]` — about 2 % of the old action range. The QP
could always thread the gates; it was never asked to.

**Fixing the goal box alone closes the gap**, sizing it from the plant
(`1.5 × v_max · c · dt`) rather than by inheritance:

| | solved | steps | % of oracle | contacts/ep |
| --- | --- | --- | --- | --- |
| `PPO+MPC`, old box | 0/3 | — | 91.6 % | 1.70 |
| `PPO+MPC`, plant-derived box | **9/9** | 100k–168k | **100.0 %** | 0 (8/9) |

So the honest reading of step 1 → 2 is: **the MPC worker was never the problem,
and the hierarchy was never the problem. The interface between them was.**
Full diagnosis in [`goal-box-saturation.md`](goal-box-saturation.md).

### What this costs, and it is worth stating

Under the saturated box, *every* goal was unreachable, so "stop" was literally
inexpressible — the agent always advanced, and always clipped walls. A reachable
box makes graded control possible, which necessarily also makes idling
expressible. At the tightest box tested (`1.0 ×` reach) one seed in nine found
exactly the risk-averse idle optimum the environment has always had
(`SlalomEnvConfig`'s own historical note documents it): parked before a gate,
contacts driven to 0.3/ep by never advancing. The adopted `1.5 ×` default does
not show this in nine seeds, but it is a real property of the change, not
something the fix removed.

The tunnel also gets slower (24k–30k against 8k–16k). Expected: on a
constant-width corridor the saturated policy *is* optimal, so the oversized box
was handing the answer away for free.

---

## 4. Step 2 → 3: does reachability analysis add anything?

With the box fixed, `PPO+MPC` and `PPO+MPC-reach` were re-run head to head,
9 seeds each, canonical slalom, identical protocol:

| | solved | median steps | return | contacts/ep |
| --- | --- | --- | --- | --- |
| `PPO+MPC`, fixed box | 9/9 | 116k | **1012.7** | 0.011 |
| `PPO+MPC-reach` | 9/9 | 102k | 1010.2 | 0.033 |

Sample efficiency indistinguishable (Mann–Whitney p = 0.45). Final quality
*slightly worse* for reach (p = 0.004), traceable to 1.4 extra steps per episode.

**On the canonical plant, reachability analysis buys nothing.** The original
`benchmark.md` claim — that `PPO+MPC-reach`'s 0/3 → 3/3 demonstrated the value
of reachability-aware goals — was measuring the goal box's *size*, not its
shape. Once the size is right, the shape is redundant **on this plant**.

That last qualifier is the whole of section 5.

---

## 5. The axis that organizes everything: plant agility

A null result on one plant is not a result about the method. The right question
is under what conditions the mechanism is exercised, and for this method the
condition is identifiable in closed form.

`reachable_goal`'s one structural advantage over any fixed box is that it
centres the reachable displacement set on the **free-drift outcome** `T · v₀`
instead of on the current position. That is invisible when the actuator can
reverse the velocity within a macro-step, and dominant when it cannot. The
governing dimensionless quantity is the **agility ratio**

```
rho = v_max / (u_max * manager_freq * dt)
```

— how much of `v_max` the actuator can add or remove inside one macro-step. As
`rho` grows the reachable set both narrows and re-centres, and no
state-independent box can track it:

| `u_max` | `rho` | reachable `Δp` from `v₀ = +1.2` | live fraction of a ±1.8 box |
| --- | --- | --- | --- |
| 2.5 (canonical) | 0.48 | `[-0.17, +1.20]` | 38 % |
| 1.0 | 1.20 | `[+0.65, +1.20]` | 15 % |
| 0.5 | 2.40 | `[+0.92, +1.20]` | 8 % |

At `rho = 2.4` the reachable interval from cruise is *entirely positive* while
the box is symmetric about zero — a failure of shape, which no resizing fixes.

### Sweeping it reveals two crossovers, not one

Percent of oracle return, and wall contacts per episode. The oracle is
recomputed per regime, so these are comparable **down** a column, not across
one. Flat PPO and the hierarchies are 5 seeds per regime (canonical flat PPO is
3, from `benchmark.md`).

| `rho` | flat PPO | `PPO+MPC` (fixed box) | `PPO+MPC-reach` (tight) |
| --- | --- | --- | --- |
| 0.48 | **100.5 %** · 0.00 | 100.0 % · 0.01 | 99.8 % · 0.02 |
| 1.20 | 97.3 % · 0.60 | 99.4 % · 0.12 | **99.8 % · 0.02** |
| 2.40 | 95.1 % · 1.00 | 97.6 % · 0.50 | **98.7 % · 0.22** |

Solve rates over the same sweep (`rho` = 0.48 / 0.80 / 1.20 / 2.40):

| arm | solve rate across `rho` |
| --- | --- |
| flat PPO | 3/3 · — · 2/5 · 0/5 |
| `PPO+MPC`, fixed box | 9/9 · 5/5 · 3/5 · 0/5 |
| `PPO+MPC-reach`, tight | 5/5 · 5/5 · 5/5 · 1/5 |

**Crossover 1 — flat RL stops being sufficient (`rho ≈ 1`).** Below it, flat PPO
matches the hierarchies *and is faster* (51k–81k against 100k–180k), so the
hierarchy is pure overhead. Above it flat PPO degrades steadily on every measure
at once — solve rate 3/3 → 2/5 → 0/5, return 100.5 → 97.3 → 95.1 %, contacts
0.00 → 0.60 → 1.00 — and is the worst arm on both return and contacts at
`rho = 2.4`. A fixed MPC worker already knows the plant; flat RL has to
discover drift-dominated consequences from reward alone.

The effect is real and monotone, but it is a *degradation*, not a collapse:
flat PPO still reaches 95 % of oracle at `rho = 2.4`. The gap that matters is
the safety one — twice the wall contacts of the fixed-box hierarchy and nearly
five times those of the reachability-aware one.

**Crossover 2 — a correctly-sized fixed box stops being sufficient
(`rho ≈ 0.8`).** The fixed box degrades in `rho` (9/9 → 5/5 → 3/5 → 0/5) while
tight reachability holds at 5/5 through `rho = 1.2`. At `rho = 0.8` reach-tight
is 5/5 at 100.0 % of oracle with **zero** contacts against 0.04.

At canonical `rho` the two are statistically indistinguishable: reach-tight's
68k median solve time against the fixed box's 116k is p = 0.42, on a
heavy-tailed `[64k, 64k, 68k, 132k, 270k]`. The fixed box actually has the
tightest distribution and the best final return there. The honest reading is a
**tie below `rho ≈ 0.8` and a widening reach advantage above it**, cleanest on
wall contacts, where the ordering holds at every regime tested.

So the hierarchy's value is not sample efficiency and not final quality on an
easy plant. It is **robustness to plant sluggishness** — and reachability
analysis is what preserves that robustness once the goal space itself starts to
misrepresent the plant.

### The finding that inverts the naive expectation

Reachability-aware goals are **not monotonically safe**. At `rho = 2.4` the
repo's shipped configuration (`accel_split = 0.5`) is dramatically *worse* than
the fixed box — return 853.5 against 978.7, 2.80 contacts against 0.50. The
cause is coverage: that setting spans only 68 % of the true reachable width.
Raising it to `accel_split = 1.0` (91 % coverage) reverses the result.

The mechanism is that an oversized box produces **saturation**, and on a
drift-dominated plant saturation is accidentally *useful* — continuous
full-authority lateral acceleration approximates min-time repositioning. A goal
set that is correctly centred but 32 % too narrow removes that free bang-bang
without yet supplying enough precision to replace it. Below a coverage
threshold, reachability analysis is worse than no analysis at all.

Full sweep, coverage measurements and caveats:
[`reachability-regime-study.md`](reachability-regime-study.md).

---

## 6. What to claim, and what not to

**Supported.**

1. The goal space is part of the controller. Sizing it by inheritance rather
   than from the plant cost `PPO+MPC` the entire task, and no RL-side
   mitigation recovered it.
2. A zeroth-order reachability bound (`v_max · c · dt`) is sufficient on an
   agile plant and recovers full oracle performance.
3. The model-based worker is what makes the controller robust to plant
   sluggishness. Flat PPO degrades monotonically in `rho` on solve rate, return
   and contacts simultaneously, ending at `rho = 2.4` with twice the wall
   contacts of the fixed-box hierarchy and five times those of the
   reachability-aware one.
4. Exact reachability analysis pays once `rho ≳ 0.8`, improving solve rate,
   return and contacts simultaneously.
5. A conservative inner approximation can invert the sign of the effect.
   Coverage, not the presence of "reachability analysis", is what decides.

**Not supported, and should not be claimed.**

- That reachability analysis improves hierarchical MPC *in general*. On the
  canonical plant it is neutral-to-slightly-negative across 9 seeds.
- That hierarchy beats flat control in general. At `rho = 0.48` flat PPO is
  faster and marginally better, and even at `rho = 2.4` it still reaches 95 %
  of oracle — the hierarchy's advantage there is in contacts and solve rate,
  not in a collapse of the flat baseline.
- That `accel_split = 1.0` is optimal — it is the best of three values at one
  `rho`, and coverage is non-monotone in it at canonical `rho`.
- That the ±1.8 box is the best *possible* fixed box in the hard regimes. It is
  the best on the canonical plant; no per-regime box sweep was run.
- Any fine ranking from 5-seed arms.

**The strongest honest framing** is not "our method wins". It is: *the
learned/model-based interface has a correctness condition; `rho` is the
computable quantity that determines it; below one threshold the cheap bound
suffices, above another the full analysis becomes necessary, and a loose
approximation is worse than none.* That claim survives scrutiny precisely
because it comes with regimes where the method loses and a mechanism for why.

---

## 7. Reproducing

```bash
cd scenarios/slalom/scripts
T="--total-timesteps 400000 --early-stop-success-rate 2.0"

# canonical plant
python script_ppo.py            --seed 1 $T
python script_ppo_mpc.py        --seed 1 $T
python script_ppo_mpc_reach.py  --seed 1 $T --accel-split 1.0

# the old, saturated goal box (pre-fix PPO+MPC)
python script_ppo_mpc.py        --seed 1 $T --max-goal-bound 10 --max-goal-vel 2

# drift-dominated plant (rho = 2.4)
python script_ppo_mpc.py        --seed 1 $T --env-u-max 0.5
python script_ppo_mpc_reach.py  --seed 1 $T --env-u-max 0.5 --accel-split 1.0
```

`--env-u-max` and `--accel-split` default to the canonical values, so omitting
them reproduces the shared environment and the repo's shipped configuration.
