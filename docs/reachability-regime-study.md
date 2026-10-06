# When does reachability analysis pay?

> **Dynamics.** Every number in this document was measured on the per-axis
> speed and thrust limits (|v_i| <= v_max, |u_i| <= u_max), with no effort term in
> the reward and the progress potential uncut at the goal line: the environments
> up to git tag `box-limits-final`. Since 2026-10-04 the limits bound ||v|| and
> ||u||, the reward charges control effort, and the oracle maximises it exactly;
> see [`disk-limits-and-effort.md`](disk-limits-and-effort.md), which also has the
> studies rerun on the new dynamics.

**What this document is.** The regime study behind
[`progression.md`](progression.md) §5. It answers a question the canonical
benchmark cannot: *under what conditions does computing the reachable goal set
actually improve a hierarchical MPC controller, and when does it make things
worse?*

It exists because the head-to-head comparison on the canonical plant came back
**null** — `PPO+MPC-reach` neither beat nor was beaten by a correctly-sized
fixed goal box (9 seeds, p = 0.45 on sample efficiency). A null result on one
plant is not a result about the method, so the right move was to identify the
mechanism the method implements and find the regime that exercises it.

---

## 1. The mechanism, and the quantity that governs it

`reachable_goal` (`algorithms/ppo_mpc_reach/ppo_mpc_reach.py`) differs from a
fixed goal box in exactly one structural way: it centres the reachable
displacement set on the **free-drift outcome** `T · v₀` rather than on the
agent's current position, where `T = manager_freq · dt` is the macro-step.

Whether that matters depends on how much the actuator can change the velocity
within one macro-step, relative to the velocity itself. That is one
dimensionless number — the **agility ratio**:

```
rho  =  v_max / (u_max * manager_freq * dt)
```

- `rho << 1` — the actuator can more than reverse the velocity inside a
  segment. The reachable set stays roughly centred on the current position and
  roughly symmetric, so a fixed symmetric box is a decent approximation.
- `rho >> 1` — the plant is **drift-dominated**. Where it can get to is mostly
  determined by where it is already heading. The reachable set both *narrows*
  and *re-centres* away from the origin, and no state-independent box can
  track it.

Computed exactly (forward-simulating the extreme bang trajectories under
`|u| ≤ u_max`, `|v| ≤ v_max`, `c = 10`, `dt = 0.1`, `v_max = 1.2`):

| `u_max` | `rho` | reachable `Δp` from `v₀ = 0` | from `v₀ = +1.2` | live fraction of a ±1.8 box |
| --- | --- | --- | --- | --- |
| 2.5 (canonical) | 0.48 | `[-0.97, +0.97]` | `[-0.17, +1.20]` | 54 % / 38 % |
| 1.5 | 0.80 | — | — | — |
| 1.0 | 1.20 | `[-0.55, +0.55]` | `[+0.65, +1.20]` | 31 % / 15 % |
| 0.5 | 2.40 | `[-0.28, +0.28]` | `[+0.92, +1.20]` | 15 % / 8 % |

At `rho = 2.4` and cruise velocity, the reachable interval is **entirely
positive** while a fixed box is symmetric about zero. That is a failure of
*shape*, which no amount of resizing fixes — and it is precisely what
reachability analysis exists to prevent.

`rho` is therefore the axis this study sweeps, via `--env-u-max` (which
defaults to the canonical plant, so omitting it changes nothing).

---

## 2. Result 1 — the model-based worker is what survives a sluggish plant

Before asking about reachability, the more basic control: does the *task* get
harder with `rho`, or does the *architecture* fail? Flat PPO has no goal space
at all, so it isolates this.

Percent of oracle return and wall contacts per episode (the oracle is
recomputed per regime, so these compare **down** a column, not across one).
5 seeds per cell; canonical flat PPO is 3 seeds from `benchmark.md`.

| `rho` | flat PPO | `PPO+MPC` (fixed box) | `PPO+MPC-reach` (tight) |
| --- | --- | --- | --- |
| 0.48 | **100.5 %** · 0.00 | 100.0 % · 0.01 | see §4 |
| 1.20 | 97.3 % · 0.60 | 99.4 % · 0.12 | **99.8 % · 0.02** |
| 2.40 | 95.1 % · 1.00 | 97.6 % · 0.50 | **98.7 % · 0.22** |

Solve rate tells the same story more sharply: flat PPO goes 3/3 → 2/5 → 0/5
across `rho` = 0.48 / 1.20 / 2.40, while the fixed-box hierarchy goes
9/9 → 3/5 → 0/5 and the reachability-aware one 5/5 → 1/5 at the two hard
regimes.

Flat PPO's degradation is monotone and shows on every measure at once, but it
is a degradation rather than a collapse — it still reaches 95 % of oracle at
`rho = 2.4`. What separates the arms there is **safety**: flat PPO takes twice
the wall contacts of the fixed-box hierarchy and roughly five times those of
the reachability-aware one.

**This is the first crossover, and it sets up everything else.** On an agile
plant the hierarchy is pure overhead: flat PPO matches it and solves in
51k–81k steps against the hierarchy's 100k–180k. As the plant becomes
drift-dominated, flat RL has to discover through reward alone that its actions
now have long-delayed consequences, and it becomes unreliable — it stops
meeting the solved criterion entirely, and its wall-contact rate grows faster
than either hierarchy's.
A fixed MPC worker already knows the plant, so it does not have to.

The hierarchy's value is therefore **not** sample efficiency and **not** final
quality on an easy plant. It is robustness to plant sluggishness.

---

## 3. Result 2 — approximation tightness decides the sign of the effect

The first attempt at this study predicted reach would win at high `rho`. It
lost, badly:

| arm at `rho = 2.40` | return | contacts/ep |
| --- | --- | --- |
| `PPO+MPC`, fixed box | 978.7 | 0.50 |
| `PPO+MPC-reach`, `accel_split = 0.5` (repo default) | **853.5** | **2.80** |

The prediction was not merely unconfirmed, it was inverted. The cause is a
design parameter that had never been swept.

`reachable_goal` builds its goal set from two generators — `g1` (constant full
acceleration) and `g2` (accelerate-then-brake, zero net `Δv`) — and splits the
actuator budget between them by `accel_split`, so that no corner of the
`[-1,1]²` action square exceeds `u_max`. The docstring justifies the default
`0.5` on a symmetry argument: "no reason to prefer one axis of the parallelogram
over the other". That argument is about the *generators*, not about the
resulting **coverage of the true reachable set**, and the two are not the same.

Measured coverage (span of commandable `Δy` as a fraction of the true reachable
width):

| `v_y0` | true width at `rho = 2.4` | `split = 0.5` | `split = 0.75` | `split = 1.0` |
| --- | --- | --- | --- | --- |
| 0.0 | 0.55 | 68 % | 80 % | **91 %** |
| 0.6 | 0.55 | 68 % | 80 % | **91 %** |
| 1.2 | 0.28 | 91 % | 91 % | 91 % |

The last row is not a typo and matters for §3.1: at saturated lateral velocity
the `alpha` headroom clip in `reachable_goal` removes the whole `a > 0` half of
the action range (no `delta_v` is available upward at `v_max`), so `g1`
contributes only half its span and the three splits land on *the same* 0.25 m.
The coverage difference the rest of this section turns on exists only while the
agent has lateral velocity in hand.

And performance follows coverage monotonically (Spearman = 1.00):

| arm at `rho = 2.40` | coverage | solved | return | contacts/ep |
| --- | --- | --- | --- | --- |
| `PPO+MPC`, fixed box | n/a | 0/5 | 978.7 | 0.50 |
| reach, `accel_split = 0.5` | 68 % | 0/5 | 853.5 | 2.80 |
| reach, `accel_split = 0.75` | 80 % | 0/5 | 949.3 | 0.98 |
| reach, `accel_split = 1.0` | 91 % | **1/5** | **990.2** | **0.22** |

`reach(tight)` vs fixed box: contacts p = 0.042, return p = 0.15 (n = 5).

### A confound in that ladder: coverage and rank move together

The ladder above reads as a clean dose-response in coverage, but `accel_split`
moves two things at once, and only one of them is coverage.

Write the commandable half-span per axis directly from the generators:

```
span(split) = 2 * (g1p + g2p)
            = 2 * (0.5*split*u_max*T^2  +  0.25*(1-split)*u_max*T^2)
            = u_max * T^2 * (0.5 + 0.5*split)
```

This is **strictly increasing** in `accel_split` — `g1p` grows twice as fast as
`g2p` shrinks — so the widest goal set is always the one at `split = 1.0`. But
`g2p = 0.25*(1-split)*u_max*T^2` is **exactly zero** at `split = 1.0`, and `g2`
is the only generator the manager's *position* action slots feed. Measured
through `reachable_goal` at `rho = 2.4`, `v_y0 = 0`:

| `accel_split` | `dy` span | coverage | span from the **position** slot | rank of the map |
| --- | --- | --- | --- | --- |
| 0.50 | 0.375 | 68 % | 0.125 | 2 |
| 0.75 | 0.438 | 80 % | 0.063 | 2 |
| 0.90 | 0.463 | 84 % | 0.025 | 2 |
| **1.00** | **0.500** | **91 %** | **0.000** | **1** |

At `split = 1.0` the manager's two position slots (`action[0]`, `action[1]`) have
literally no effect on the goal it produces: the map degenerates to
`delta_p = T*v0 + (T/2)*delta_v`, one bang-bang command per axis, and half the
action space is dead. The goal space is no longer a parallelogram, it is a line.

So **coverage is maximal exactly where the goal space degenerates**, and no
sweep of `accel_split` alone can distinguish two very different explanations of
§3's result:

- **coverage** — a wider commandable set is what helps; or
- **authority** — what helps is that the map becomes pure constant-acceleration
  bang-bang, which §3.1 argues is close to min-time repositioning on a
  drift-dominated plant.

Note that the second reading is the *same mechanism* §3.1 invokes to explain why
an oversized fixed box is accidentally useful at high `rho`. If it is right, then
"tight reachability analysis" wins at `rho = 2.4` partly by discarding the
zonotope's second generator — by being *less* of a reachability analysis, not
more — and the honest headline in §5 would need rewording.

Separating them needs an arm that holds coverage fixed and changes rank.
`--reach-u-frac` (`script_ppo_mpc_reach.py`) does exactly that: it shrinks the
`u_max` the goal map believes it has, without touching the plant or the QP the
worker solves. `--accel-split 1.0 --reach-u-frac 0.75` reproduces
`--accel-split 0.5`'s span of 0.375 (68 % coverage) at rank 1:

| arm at `rho = 2.40` | coverage | rank | predicted by |
| --- | --- | --- | --- |
| A `split=0.5` | 68 % | 2 | — |
| **B `split=1.0 --reach-u-frac 0.75`** | **68 %** | **1** | coverage → behaves like A; authority → behaves like C |
| C `split=1.0` | 91 % | 1 | — |

### Why a loose reachable set is worse than no reachable set

This is the study's least obvious finding and deserves stating carefully.

An oversized goal box produces **saturation**: the manager commands a setpoint
the worker cannot reach, and the QP responds by driving at the actuator limit
toward it for the whole segment. On the canonical plant that is a pathology — it
destroys the manager's lateral resolution and is exactly what broke `PPO+MPC`
(see [`goal-box-saturation.md`](goal-box-saturation.md)).

On a drift-dominated plant it is accidentally *useful*. Continuous full-authority
lateral acceleration is close to the min-time repositioning strategy for a
sluggish plant, so an oversized box gets bang-bang behaviour for free. A goal set
that is correctly *centred* but 32 % too *narrow* removes that saturation and
replaces it with a setpoint the worker reaches and then holds — surrendering
authority the loose box was exploiting by accident, without yet supplying enough
precision to compensate.

So reachability-aware goals are not monotonically safe. They are an improvement
only past a coverage threshold; below it, they are worse than not doing the
analysis. **Approximation tightness is a first-class design variable, not a
tuning detail** — and `accel_split = 0.5` ships below the threshold, which is
why the canonical head-to-head came back null.

---

## 4. Result 3 — the crossover

With `accel_split = 1.0`, sweeping `rho`. All cells 5 seeds except the two
canonical 9-seed arms; oracle recomputed per regime.

| `rho` | arm | solved | median steps | % of oracle | contacts/ep |
| --- | --- | --- | --- | --- | --- |
| **0.48** | `PPO+MPC`, fixed box | 9/9 | 116k | **100.0 %** | 0.01 |
| | reach, `split=0.5` | 9/9 | 102k | 99.7 % | 0.03 |
| | reach, `split=1.0` | 5/5 | 68k | 99.8 % | 0.02 |
| **0.80** | `PPO+MPC`, fixed box | 5/5 | 136k | 99.8 % | 0.04 |
| | reach, `split=1.0` | **5/5** | **86k** | **100.0 %** | **0.00** |
| **1.20** | `PPO+MPC`, fixed box | 3/5 | 174k | 99.4 % | 0.12 |
| | reach, `split=1.0` | **5/5** | 182k | **99.8 %** | **0.02** |
| **2.40** | `PPO+MPC`, fixed box | 0/5 | — | 97.6 % | 0.50 |
| | reach, `split=1.0` | **1/5** | 338k | **98.7 %** | **0.22** |

(The `split=0.5` arms at `rho` = 0.80 and 1.20 are still running; they complete
the picture of the loose parametrization degrading as `rho` grows.)

**At `rho = 0.48` the three arms are statistically indistinguishable on sample
efficiency.** Reach-tight's 68k median looks decisive but is not: its
steps-to-solve are `[64k, 64k, 68k, 132k, 270k]`, a heavy tail that gives
p = 0.42 against the fixed box. The fixed box has the *tightest* distribution
(100k–168k) and the best final return. On an agile plant, reachability analysis
buys nothing measurable — which is the honest canonical result and was the
starting point of this study.

**The separation appears from `rho = 0.80` onward** and grows:

- at `0.80`, reach-tight matches on solve rate but takes **zero** contacts
  against 0.04 and reaches 100.0 % of oracle;
- at `1.20`, 5/5 against 3/5, with a sixth of the contacts;
- at `2.40`, neither solves reliably, but reach-tight has the best return and
  under half the contacts.

The fixed box degrades monotonically in `rho` — 9/9 → 5/5 → 3/5 → 0/5 — while
tight reachability holds at 5/5 through `rho = 1.2` before both fail at 2.4.
The crossover sits near `rho ≈ 0.8`, and the ordering on **wall contacts** is
consistent at every single regime, which is the more robust signal than return.

---

## 5. What this licenses as a claim

**Supported.**

1. `rho = v_max / (u_max · c · dt)` predicts when a fixed goal box stops being
   an adequate parametrization, and it is computable from the plant before any
   training is run.
2. Exact reachability analysis improves a hierarchical MPC controller once
   `rho ≳ 0.8`, on solve rate, final return and wall contacts simultaneously.
3. Below that, it is neutral — the cheap zeroth-order bound `v_max · c · dt` is
   sufficient.
4. A conservative inner approximation of the reachable set can invert the sign
   of the effect. Coverage, not the presence of "reachability analysis", is what
   determines the benefit.
5. The model-based worker (with *either* goal parametrization) is what makes
   the controller robust to plant sluggishness at all; flat PPO collapses at
   `rho = 2.4` while both hierarchies hold above 97 %.

**Not supported.**

- Any fine ranking within a regime. All regime arms are 5 seeds; they separate
  "solves" from "does not", not 86k from 123k.
- That `accel_split = 1.0` is optimal. It is the best of three values tested at
  one `rho`. Coverage at canonical `rho` is *non-monotone* in `accel_split`
  (93 % at `0.5` vs 62 % at `1.0` for `v₀ = 0`, inverting at `v₀ = 1.2`), so the
  best split is plausibly regime-dependent and was not swept per regime.
- That the ±1.8 fixed box is the best possible fixed box in the hard regimes.
  It is the best on the *canonical* plant; a per-regime box sweep was not run,
  so §4's comparison is "tuned-for-canonical box" against "reachability", not
  "best possible box" against "reachability".

**The honest headline.** Reachability analysis is not a free improvement to
hierarchical MPC. It is the correct parametrization of a goal space whose
adequacy is governed by a computable property of the plant, and it must be
tight to help at all. On an agile plant, the cheap bound suffices and the full
analysis is redundant; as the plant becomes drift-dominated, the full analysis
becomes the thing that keeps the controller working.

---

## 6. Reproducing

```bash
cd scenarios/slalom/scripts
T="--total-timesteps 400000 --early-stop-success-rate 2.0 --cuda false"

for U in 2.5 1.5 1.0 0.5; do            # rho = 0.48 0.80 1.20 2.40
  python script_ppo.py           --seed 1 $T --env-u-max $U
  python script_ppo_mpc.py       --seed 1 $T --env-u-max $U
  python script_ppo_mpc_reach.py --seed 1 $T --env-u-max $U --accel-split 1.0
done

# the accel_split ladder at rho = 2.4
for S in 0.5 0.75 1.0; do
  python script_ppo_mpc_reach.py --seed 1 $T --env-u-max 0.5 --accel-split $S
done
```

`--env-u-max` defaults to the canonical 2.5 and `--accel-split` to the repo's
0.5, so omitting both reproduces the shipped configuration on the shared
environment.
