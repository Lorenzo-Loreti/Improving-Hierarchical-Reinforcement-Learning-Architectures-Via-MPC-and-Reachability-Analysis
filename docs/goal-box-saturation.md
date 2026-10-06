# The goal box, and why `PPO+MPC` could not solve the slalom

> **Dynamics.** Every number in this document was measured on the per-axis
> speed and thrust limits (|v_i| <= v_max, |u_i| <= u_max), with no effort term in
> the reward and the progress potential uncut at the goal line: the environments
> up to git tag `box-limits-final`. Since 2026-10-04 the limits bound ||v|| and
> ||u||, the reward charges control effort, and the oracle maximises it exactly;
> see [`disk-limits-and-effort.md`](disk-limits-and-effort.md), which also has the
> studies rerun on the new dynamics.

**Date:** 2026-09-23.
**Companion:** [`benchmark.md`](benchmark.md) §4–5, whose `PPO+MPC` row this
explains and fixes.

`PPO+MPC` was the only algorithm in the cross-algorithm benchmark that never
solved the slalom: 0/3 seeds, stalling at 92.6 % of the oracle with 1.3–1.8
wall contacts per episode. It was simultaneously the *fastest* algorithm on the
tunnel (8k–16k steps). This document is the diagnosis of that split and the
change that closes it.

## 1. The one-line cause

`script_ppo_mpc.py` set its goal box by copying the constant out of
`script_hppo.py`:

```python
max_goal_bound = 10.0
v_max = 2.0          # note: the plant's v_max is 1.2
goal_scale = np.array([max_goal_bound, max_goal_bound, v_max, v_max])
```

That constant is free in hPPO and fatal here, because the two workers consume a
goal in completely different ways.

- **hPPO's worker is a network.** It receives the goal *re-normalized by the
  same `max_goal_bound`* before it reaches the first layer (`worker_input` in
  `algorithms/hppo/hppo_train.py`); `normalize_goal` and `scale_goal` are exact inverses
  (`algorithms/common.py`, `algorithms/hppo/hppo.py`), so the constant cancels
  and the worker network literally sees the manager's raw `[-1, 1]` action. An
  unreachable goal is simply a direction. The one place it does not cancel is
  the worker's intrinsic reward, which is computed on the *physical* goal
  (`norm(current_goal) - norm(next_goal)`) — but that quantity is the
  projection of a step's displacement onto the goal direction, bounded by
  physics rather than by the goal's magnitude, so it too is near
  scale-invariant. Which is why hPPO trains perfectly well at 10.0.
- **`PPO+MPC`'s worker is a QP.** `MPCWorker` tracks the goal as a hard
  setpoint, penalized at *every stage* of its horizon
  (`_solve_osqp` / `_setup_cvxpy`). The goal's absolute magnitude is
  load-bearing.

Over one macro-step of `manager_freq` steps the plant cannot displace further
than `v_max * manager_freq * dt` — 1.20 m at the defaults — because that is a
hard box constraint inside the worker's own QP. A setpoint past that radius is
unreachable *by construction*, and the quadratic tracking cost is then minimized
by driving at the actuator/velocity limit toward it for the whole segment.

## 2. The saturation, measured

One 10-step segment from `[p_x=3.0, p_y=0.0, v_x=1.2, v_y=0.0]`, sweeping the
commanded `delta_y` and recording where the segment actually ends:

| `delta_y` commanded | +0.10 | +0.40 | +0.90 | +1.20 | +2.00 | +3.00 | +5.00 | +10.00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| end `p_y` | +0.086 | +0.352 | +0.709 | +0.847 | +0.906 | +0.909 | +0.910 | +0.910 |
| end `v_y` | +0.000 | +0.235 | +0.627 | +0.870 | +1.133 | +1.186 | +1.200 | +1.200 |

Beyond the reachable radius the response is **exactly flat**. With a ±10 box,
88 % of the manager's per-axis action range maps to one identical saturated
segment.

## 3. What that did to the trained policies

Rolling the three benchmark checkpoints over a 25-point spawn grid and
recording every goal the manager emitted:

| seed | return | contacts/ep | `|dx|` unreachable | `|dy|` unreachable | mean `|dy|` |
| --- | --- | --- | --- | --- | --- |
| 1 | 941 | 1.44 | 100 % | 90 % | 4.59 m |
| 2 | 926 | 1.76 | 100 % | 75 % | 2.53 m |
| 3 | 921 | 1.84 | 100 % | 80 % | 3.71 m |

Mean commanded `|dy|` is 2.5–4.6 m — past the reachable radius *and* past the
corridor's own half-width of 2.0 m. A nominally 4-D goal had degenerated into
roughly two bits per segment: the sign of each axis, at full speed. The
resulting trajectory is a bang-bang zig-zag, re-aimed once every 10 steps.

This is also why the failure is slalom-specific. On the tunnel, "full speed
ahead" *is* the optimal policy, so a degenerate direction-only goal costs
nothing — hence `PPO+MPC` topping that table. The slalom needs a graded lateral
command, which the saturated box could not express.

A second property compounds it: `MPCWorker` looks its corridor bound up at the
*current* `p_x` and holds it across the horizon, so the worker cannot see a gate
until the agent is already inside it. Gate avoidance therefore rests entirely on
the manager — which, saturated, had no way to say "be at y = +0.9 and hold".

Contacts land where that predicts. Across the three seeds they cluster on the
**full-width wall between the two gates** (14 / 16 / 20 of them), not in the
gates.

## 4. Two controls: the worker was never the limitation

- The oracle (`algorithms/optimal_solver.py`) solves the slalom collision-free
  at return 1013.
- Feeding **the same `MPCWorker`** goals derived from that oracle trajectory, at
  the same `c = 10` cadence, gives return 1013.3 with **zero contacts** — and
  those goals need only `delta_y` in `[-0.37, +0.13]`, about 2 % of the old
  action range.

So the QP could always thread the gates. It was never being asked to.

## 5. The fix

Size the goal box from the plant rather than inheriting hPPO's constant:

```python
reach_per_segment = env_config.v_max * args.manager_freq * env_config.dt   # 1.20 m
max_goal_bound = args.max_goal_bound or args.goal_bound_slack * reach_per_segment
max_goal_vel   = args.max_goal_vel   or env_config.v_max                  # 1.2, not 2.0
```

Exposed as `--max-goal-bound` / `--max-goal-vel` / `--goal-bound-slack`.
`--max-goal-bound 10 --max-goal-vel 2` reproduces the old behaviour exactly.

### Result — slalom, 300k budget, benchmark protocol

| config | solved | steps to solve | return | contacts/ep |
| --- | --- | --- | --- | --- |
| old box (`10` / `2`) | 0/3 | — | 928 / 928 / 928 | 1.70 / 1.70 / 1.70 |
| plant-derived box (default) | **9/9** | 100k–168k (median 116k) | 1013.3 ± 0.1 | **0** on 8/9 |

The old-box arm reproduces the benchmark's original numbers on the current code
path, so the comparison is clean. The oracle mean is 1013.0.

The old-box arm is 3 seeds (the benchmark's own count, and it is unanimous); the
fixed arm is 9. Per-seed solve steps: 124k / 168k / 100k / 100k / 116k / 110k /
156k / 114k / 122k.

One qualification on the contacts column. `solved` is judged on the 25-point
fixed grid, while the reported `collisions/ep` comes from the 10-random-initial-
condition evaluation (`script_ppo_mpc.py`'s `--eval-episodes` default is 10, not
the 20 the flat-PPO and hPPO scripts use) — different distributions, as
`benchmark.md` notes. Eight
seeds end at exactly 0 contacts there; seed 8 ends at 0.10 (one contact across
ten episodes, return 1008.2) despite passing the strict grid criterion twice.
So: 9/9 solved, 8/9 provably contact-free on both measures.

The manager now emits exactly the graded command the old box could not. One
episode, seed 1: `delta_x` held at ≈ +1.0 throughout while `delta_y` sweeps
`+0.74 → +0.16 → −0.38 → −0.46 → −0.36 → −0.27 → −0.24` across the two gates.

## 6. Why `--goal-bound-slack`, and why 1.5

The two axes want opposite things from the box. On **y** the manager needs
resolution, which argues for the tightest possible box. On **x** saturation is
the right answer — the optimal policy is full speed ahead — and a box pinned at
exactly the reachable radius makes "full speed" require an action of exactly
1.0, the very edge of the Beta's support.

Swept on both scenarios, 3 seeds each:

(The tunnel column was measured at the tunnel's former `goal_reward` of 200. It
has used the slalom's 1000 since 2026-09-24, which puts the oracle at 1013.0.)

| box | slalom solved | slalom contacts/ep | tunnel solved at | tunnel return (oracle 213.0) |
| --- | --- | --- | --- | --- |
| 10.0 (old) | 0/3 | 1.70 | 8k–16k | 213 / 214 / 213 |
| 1.0 × reach (1.2) | 8/9 | 0 | 30k / 36k / 34k | 209.7 / 210.5 / 210.1 |
| **1.5 × reach (1.8)** | **9/9** | **0 on 8/9** | 24k / 30k / 24k | 212.7 / 212.2 / 212.4 |
| 2.0 × reach (2.4) | 2/3 | 0.40 on the failing seed | 22k / 22k / 20k | 212.4 / 212.2 / 212.4 |

(Slalom columns: 9 seeds at 1.0x and 1.5x, 3 at the others. Tunnel columns:
3 seeds throughout, 100k budget.)

1.5 is the middle of that curve, not a fitted value: 1.0 costs the tunnel its
x-axis headroom, 2.0 starts letting the saturation back in. At 1.5, 67 % of the
manager's range is still reachable, against 12 % under the old box.

Notably, the one seed that failed at 1.0 × reach (see §7) solves at 1.5 ×, in
168k steps — the slowest of the nine, but a clean solve at the oracle return.

## 7. What the fix costs, honestly

**At 1.0 × reach, seed 2 of 9 failed in a new way** — worth recording because it
is the failure mode this change makes *possible*, even though the adopted 1.5 ×
default does not exhibit it. That run did not collapse: its critic stayed
healthy (`explained_variance` 0.4–0.75, no value drift, no KL spike). It drove
contacts from 3.5/ep to 0.3/ep by *never advancing* — `success_rate` 0.00 and
`length` 200 for all 150 updates, parked at `p_x ≈ 4.2–5.0` wobbling laterally,
commanding `delta_x → 0` and negative `delta_v_x`.

That is the risk-averse idle optimum `SlalomEnvConfig`'s own historical note
already documents for this environment. The trade is worth stating plainly:
under the old box *every* goal saturated, so "stop" was literally
inexpressible — the agent always barrelled forward and always clipped walls. A
reachable box makes graded control possible, which necessarily also makes "stop"
expressible.

**The tunnel is still slower than it was**: 24k–30k against the old 8k–16k, at
212.2–212.7 against ~213. Expected, and cheap: on a constant-width corridor the
optimal policy *is* the saturated one, so the old box handed it the answer for
free. The tunnel is the benchmark's own declared non-discriminating scenario,
and the criterion it is judged by still passes 3/3 with zero contacts.

## 8. Relationship to `PPO+MPC-reach`

This is deliberately a *scale* fix and not the one
`algorithms/ppo_mpc_reach/ppo_mpc_reach.py` makes. `reachable_goal` there
reshapes the goal set into the velocity-dependent zonotope actually reachable
from the current state, and clips it against the corridor segment the manager is
standing in — strictly more than getting a fixed box's size right.

Keeping them in separate scripts is what preserves the ablation, and that
ablation has now been run: both arms, 9 seeds, slalom, 300k, identical
protocol.

| | solved | steps to solve (median, range) | final return | contacts/ep |
| --- | --- | --- | --- | --- |
| `PPO+MPC`, fixed box | 9/9 | 116k, 100k–168k | **1012.73** | 0.011 |
| `PPO+MPC-reach` | 9/9 | 102k, 66k–248k | 1010.18 | 0.033 |

**Sample efficiency is indistinguishable** (Mann–Whitney U on steps-to-solve,
p = 0.45; means 123.3k vs 122.9k). Reach has the better median but a much wider
spread — its best seed is 66k and its worst 248k, against a 100k–168k band for
the fixed baseline.

**Final policy quality is slightly but consistently *worse* for reach**
(Mann–Whitney U on final return, p = 0.004). Eight of the fixed baseline's nine
seeds land in 1013.1–1013.5, essentially on the oracle's 1013.0; no reach seed
reaches 1013.1, and its spread runs down to 1001.8.

The mechanism is visible in the episode lengths: 76.21 steps mean for the fixed
box against 77.62 for reach, distributions that barely overlap (fixed max 77.1,
reach min 76.9). Reach arrives ~1.4 steps later, and at −1 reward per step that
is the whole return gap. This is what `reachable_goal`'s own docstring predicts:
`accel_split=0.5` splits the actuator budget between the two zonotope
generators, making the goal set a deliberately *conservative inner*
approximation of the truly reachable one. The reach manager therefore cannot
command the most aggressive goals plain `PPO+MPC` can, and pays for it in
arrival time.

**So `benchmark.md` §5's conclusion is overturned, not merely unsupported.**
`PPO+MPC-reach`'s 0/3 → 3/3 was entirely the box *size*; once the box is sized
from the plant, the zonotope reshaping buys no sample efficiency and costs a
little final quality. Reachability-aware goals were compensating for an
oversized box, and nothing else.

Two caveats against over-reading this. It does not retro-justify the old
default: reshaping was a *sound* way to rescue a saturating worker, it simply
turns out not to be a necessary one, and `accel_split` was never swept — a
less conservative inner approximation might close the return gap. And it is a
result about *this* plant: on a less agile one no fixed box tracks the reachable
set, whose shape and centre both move with the state, and the analysis starts
earning its keep again — see
[`reachability-regime-study.md`](reachability-regime-study.md).

## Reproducing

```bash
cd scenarios/slalom/scripts
# fixed (new default: 1.5x the segment-reachable displacement)
python script_ppo_mpc.py --seed 1 --total-timesteps 300000 --early-stop-success-rate 2.0
# the old, saturated box
python script_ppo_mpc.py --seed 1 --total-timesteps 300000 --early-stop-success-rate 2.0 \
    --max-goal-bound 10 --max-goal-vel 2
# the reach arm of section 8, same protocol
python script_ppo_mpc_reach.py --seed 1 --total-timesteps 300000 --early-stop-success-rate 2.0
```

All tables here are 9 seeds (1–9) except the tunnel columns in §6 and the
old-box arm in §5, which are 3.
