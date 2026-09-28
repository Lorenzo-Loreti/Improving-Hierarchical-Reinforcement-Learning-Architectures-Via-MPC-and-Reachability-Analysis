# Termination avoidance in the hierarchical workers

**Status:** diagnosed, fixed and validated, 2026-09-22.
**Scope:** the one agent in this tree with a *learned* low-level worker —
`hppo`, on both the `slalom` and `tunnel` scenarios. `ppo` (flat) and
`ppo_mpc`/`ppo_mpc_reach` (QP worker) are structurally immune; why, and how
that was used as a control, is in
[Audit](#audit-which-algorithms-are-affected).

This document exists because the finding invalidates an earlier, published-in-
comments diagnosis of the same symptom ("the hPPO manager collapse"). The
canonical short version lives in code — the worker termination-avoidance block in
`algorithms/hppo/hppo_train.py` (written in `scenarios/slalom/scripts/script_hppo.py`,
before both scenarios' hPPO loops were merged there) and the dated addendum at the end of
`ManagerActor`'s docstring in `algorithms/common.py`. This is the long version:
what the defect is, how it was pinned down, what changed, and what the evidence
for each change actually is.

---

## 1. Summary

The hierarchical agents train their worker on a purely **intrinsic** reward —
the per-step reduction in the distance to the manager's goal:

```python
worker_reward = ||current_goal|| - ||next_goal||
```

That reward stream has no terminal term. In the worker's own MDP, reaching the
environment's goal line is therefore an absorbing state of value **exactly 0**,
while *not* reaching it keeps paying about `v_max * dt` ≈ 0.12 per step
indefinitely — the manager hands out a fresh, effectively unreachable goal every
`manager_freq` steps, so the stream never runs dry.

Discounted at `gamma = 0.99`, that is

```
V(keep farming) ≈ v_max * dt / (1 - gamma) ≈ 0.12 / 0.01 ≈ 12
V(cross the line) = 0
```

**The worker's optimal policy under its own reward is to approach the goal line
and stall in front of it.** This is not an instability, a bad seed, or a
numerical artefact: it is the correct solution to the objective the worker was
given. PPO finds it once the worker's critic becomes accurate, and the worker's
critic becomes accurate early: its `explained_variance` sits at 0.99 throughout
every collapse.

The fix is to put a terminal back into the worker's return. Two knobs, on both
`script_hppo.py`s; `0.0` on either restores the old behaviour exactly:

| knob | default | what it does | when to prefer it |
| --- | --- | --- | --- |
| `--worker-extrinsic-coef` | **0.02** | FeUdal-style `r_int + coef * r_env` | the default: the only one of the two that also makes the worker itself avoid walls |
| `--worker-success-bonus` | 0.0 | one-off bonus on a successful termination | the minimal-intervention ablation arm: "what does restoring the terminal *alone* fix?" |

Setting both was double-counting — the extrinsic mix already carries
`goal_reward` into the worker's return. **`--worker-success-bonus` was removed
on 2026-09-24**: it defaulted to 0.0 and no run used it once §5.3 had settled
the default. Its code is in §5.1, and the comment where the flag used to be in
`algorithms/hppo/hppo_train.py` keeps its sizing rule and its numbers.

---

## 2. The defect, in detail

### 2.1 Why the farm never runs dry

`max_goal_bound` is 10.0 m, while the worker can travel at most
`v_max * dt * manager_freq` = 1.2 m within one segment. Almost every goal the
manager can emit is therefore **unreachable inside the segment it was emitted
for**. The worker never arrives, `||goal||` never approaches 0, and the goal is
replaced by a fresh one of similar magnitude every `manager_freq` steps.

The practical consequence is that the worker's reward reduces to roughly
"velocity projected onto the goal direction" — a *rate*, not a budget. Its
optimal behaviour maximises `episode_length × speed`. The environment's optimal
behaviour minimises episode length. **The two objectives are directly opposed at
the finish line**, and the manager's reward (the environment's own) sits on the
opposite side of that conflict from its worker's.

### 2.2 What the collapsed policy actually does

Rolling out a collapsed checkpoint (`hppo_slalom_1_1790098565/final.pt`) from
three different starts, the trajectory is the same every time: the agent
threads **both gates cleanly, with zero wall contacts**, reaches p_x ≈ 9.3 at
t ≈ 70, then decelerates, reverses, and orbits between p_x 8.6 and 9.7 for the
remaining 130 steps. It never crosses p_x = 10.

The lateral motion is not incidental. Near the goal line the manager's goal_y
flips sign with the agent's own p_y:

```
p_y = -1.50  ->  goal_y = +5.79   (worker pushed UP)
p_y = -0.50  ->  goal_y = +2.98
p_y = +0.50  ->  goal_y = -3.03   (worker pushed DOWN)
p_y = +1.50  ->  goal_y = -5.12
```

which closes a **stable limit cycle**: the worker chases a goal that moves with
it, harvesting goal-closing reward in the one direction that carries no risk of
accidentally crossing the finish line.

### 2.3 Why there is no in-run recovery

At the state `(p_x, p_y, v_x, v_y) = (9.5, 0, 1.2, 0)`, sweeping the manager's
*entire* action box (81 goals covering `goal_x`, `goal_y` ∈ [-9, +10]), the
collapsed worker's deterministic `a_x` is:

```
best over all goals:  a_x = -1.62   (at goal = (10, -9));  u_max = +2.5
worst:                a_x = -2.14
```

**No goal the manager can emit makes this worker move forward.** The manager has
zero control authority left, which is exactly why the collapse is irreversible
within a run: the manager's policy gradient cannot fix a worker whose own
objective forbids the only action that would help.

### 2.4 The critic's own verdict

The clearest single number is the sign of `dV_worker/dp_x` across the last
approach metre, with a fixed forward goal:

| checkpoint | V_w(8.0) | V_w(9.9) | sign |
| --- | --- | --- | --- |
| still solving (best.pt, ~update 45) | 1.92 | 0.20 | **negative** |
| collapsed (final.pt) | 3.37 | 1.30 | **negative** |
| `--worker-success-bonus 20` | 18.54 | 20.15 | **positive** |
| `--worker-extrinsic-coef 0.02` | 18.90 | 20.44 | **positive** |

Note the first row: the worker is already pricing the goal line as a sink
*while the hierarchy is still scoring 100 % eval success*. The collapse is not
a discrete event, it is this gradient being slowly acted upon.

---

## 3. Why this looked like a manager instability

The previously documented signature — `manager/explained_variance` eroding from
~0.90 to ~0.30 while `manager/value_bias` grows in one direction over the
10–15 updates before the visible crash — is real, and it is **downstream**.

When the worker stops terminating episodes, the manager's returns lose the
`+goal_reward` terminal they were calibrated on. Its critic, regressed on the
old distribution, over-predicts; explained variance erodes; the bias grows. This
also explains why `--clip-vloss` helped without fixing anything: it *slows* the
manager critic's re-calibration to a return distribution that is moving under
it, so it delays the visible crash on some seeds and removes it on none.

The instrumentation added in this round (`charts/worker_ax_near_goal`) turns
negative roughly 30 updates — about 60k environment steps — **before** eval
success falls. The manager metrics move after that, not before.

`MAX_CONCENTRATION` and `--clip-vloss` are left exactly as they were: they are
genuine backstops against a genuine failure mode, and the collapsed checkpoint
does now sit pinned at the cap (`alpha = [46.7, 6.7]`, `beta = [1.00, 6.86]` —
the skewed shape the docstring describes). Neither is what was wrong.

A second contributing factor was measured and **ruled out as a cause**: the
entropy autotuner is effectively inert. Adam bounds `log_ent_coef` to about
`ent_coef_lr` per update, so at the default `3e-4` it moves `ent_coef` by 8 %
over a whole 244-update run (0.0100 → 0.0108) while the manager's entropy falls
from +1.23 to −2.20 nats against a target of +0.485. Raising it by 100×
(`--ent-coef-lr 0.05`) changes *when* the collapse happens and not *whether* it
does — see the ablation below.

---

## 4. Audit: which algorithms are affected

The repository already contained the controlled experiment; it just had not been
read as one.

| algorithm | hierarchical | worker | worker reward | affected |
| --- | --- | --- | --- | --- |
| `ppo` | no | — | environment reward (terminal `+goal_reward`) | no |
| `ppo_mpc` | yes | QP (`MPCWorker`) | none — not a learning agent | no |
| `ppo_mpc_reach` | yes | QP (`MPCWorker`) | none | no |
| `hppo` | yes | learned | intrinsic only | **yes** |

The one variant with a learned intrinsic-reward worker fails on the slalom; all
three without one solve it.

What makes this a controlled comparison rather than a list is the middle two
rows. **Hierarchy itself is not the discriminator**: `ppo_mpc` and
`ppo_mpc_reach` are hierarchical too, with the *same* `ManagerActor` network,
the same PPO manager update, the same `[-1, 1]^goal_dim` manager action space
and the same `manager_freq` cadence, and neither is affected. The single
structural difference between them and `hppo` is a worker that optimises a
reward stream of its own which termination cuts off — and that is the variable
that moves.

(`ppo_mpc`'s own 0/3 on the slalom in [`benchmark.md`](benchmark.md) is a
*separate* defect, on the goal-box side, diagnosed and fixed in
[`goal-box-saturation.md`](goal-box-saturation.md) — it never showed this
signature, and with a plant-derived box it solves 9/9.)

---

## 5. The fix

### 5.1 `--worker-success-bonus` (default 0.0 — the ablation arm; removed 2026-09-24)

Adds a one-off bonus to the worker's reward on the step the episode terminates
successfully:

```python
worker_succeeded = terminated & info["final_info"]["is_success"]
worker_reward[worker_succeeded] += args.worker_success_bonus
```

**Sizing rule.** The bonus must beat what the worker gives up by terminating,
which is its own farming value, not anything on the environment's reward scale:

```
bonus  >  v_max * dt / (1 - gamma)  ≈  1.2 * 0.1 / 0.01  ≈  12
```

The default 20.0 clears that with ~1.7× margin. **Re-derive it if `--gamma`,
`--manager-freq` or the velocity limit change** — it is tied to the intrinsic
reward scale, not to `goal_reward`.

This is the minimal intervention that makes the worker's MDP **sound**. It
changes the terminal and nothing else, leaving the worker otherwise blind to the
environment's reward — which is what keeps this a *feudal* hierarchy rather than
two agents optimising the same objective at different rates.

It was the default for one round on exactly that argument, and the evidence
overturned it: see [§5.3](#53-why-the-extrinsic-mix-is-the-default). Sound is
not the same as **aligned**.

### 5.2 `--worker-extrinsic-coef` (default 0.02, the shipped default)

FeUdal-Networks-style mixing: the worker sees `r_int + coef * r_env`.

At the scenario config defaults, `coef = 0.02` puts the environment's terminal at
`+20`, a wall contact at `-1.0` and a step at `-0.02` in the worker's units,
against an intrinsic stream of ~0.12/step. It fixes the same termination
incentive *and* makes the worker itself avoid walls and the clock, instead of
leaving every collision the manager's problem to solve at a 10-step cadence.

It costs the feudal separation. That is a real cost, and it is the reason this
was *not* the first choice — but it is the stronger performer wherever the two
have been compared directly.

### 5.3 Why the extrinsic mix is the default

**On final quality the two arms tie**, at 3 seeds on the slalom under the
benchmark protocol (uniform 500k budget, `--solved-early-stop`):

| variant | scenario | solved | % of oracle | contacts/episode |
| --- | --- | --- | --- | --- |
| `hppo` + success bonus | slalom | 3/3 | 100.5 % | 0 / 0 / 0 |
| `hppo` + extrinsic mix | slalom | 3/3 | 100.3 % | 0 / 0 / 0 |

That tie is what the first round of evidence saw, and it is why the argument
from feudal purity won: if the minimal fix gets you the same policy, take the
minimal fix. **They do not tie on sample efficiency**, and that only became
visible once the full benchmark ran under each default:

| `hppo`, slalom, 3 seeds | steps to solve | final return |
| --- | --- | --- |
| `--worker-success-bonus 20` | 266k / 296k / 399k | 1018 / 1019 / 1017 |
| `--worker-extrinsic-coef 0.02` | **71k / 81k / 81k** | 1015 / 1016 / 1016 |

A 3–5× difference in samples, at the same final policy. The mechanism is
exactly what the bonus leaves undone: it repairs the termination incentive and
nothing else, so the worker finishes, but nothing in its reward charges it for a
wall or for the clock. The worker that is told a wall costs something learns to
avoid walls directly; the worker that is not has to be steered around them by a
manager acting once every `manager_freq` steps, and that indirection is
expensive.

Restoring the terminal alone therefore makes the worker's MDP **sound**. It does
not make it **aligned**. On the slalom the manager has just enough authority to
close the gap anyway — it simply takes 3–5× the samples to do it. See
[`benchmark.md`](benchmark.md) §3 — with this default, `hPPO` is no longer
paying a measurable hierarchy tax against flat PPO at all.

Recorded here rather than quietly corrected, because "the minimal fix was right
about the mechanism and wrong about the remedy" is the actual shape of this
result.

### 5.4 What is deliberately *not* done

- **`ppo_mpc` / `ppo_mpc_reach` get no knobs.** They have no learned worker and
  no worker reward; adding dead flags would suggest otherwise.
- **`--clip-vloss` and `MAX_CONCENTRATION` are untouched.** See §3.
- **The entropy autotuner is left at `ent_coef_lr = 3e-4`.** Its ineffectiveness
  is documented but it is not the cause, and changing it would have folded an
  uncontrolled variable into every future comparison.

---

## 6. Validation

All ablations ran with early stopping **disabled**
(`--early-stop-success-rate 2.0 --solved-early-stop false`), specifically so a
variant that merely *delays* the collapse cannot be mistaken for one that
removes it. Every run used the full budget.

"Clean solve" = the final evaluation at ≥ 95 % of the oracle's mean optimal
return for that scenario. That threshold matters, because an unfixed run can end
at `success_rate = 1.00` while returning −997 — reaching the line by smashing
through both gates rather than threading them.

### 6.1 hPPO on slalom — 4 variants × 2 seeds, 500k steps

| variant | clean solves | final eval return |
| --- | --- | --- |
| baseline (both knobs 0.0) | **0/2** | −153 / −997 |
| `--ent-coef-lr 0.05` | **0/2** | −181 / 615 |
| `--worker-success-bonus 20` | **2/2** | 1018 / 1019 |
| `--worker-extrinsic-coef 0.02` | **2/2** | 1019 / 1019 |

Oracle mean optimum ≈ 1013. Two conclusions beyond the headline:

1. The pathology is the worker's reward, **not** the manager's exploration.
   Fixing the entropy autotuner moves the collapse, it does not prevent it.
2. Both fixes do more than prevent the collapse: they **close the optimality
   gap**. The unfixed hierarchy peaked at ~745 with 3–8 wall contacts per
   episode even *before* collapsing, because its worker had no reason to care
   about anything but the goal vector.

### 6.2 Cross-variant, like-for-like, on the converged checkpoints

Same 20 deterministic episodes from the same seeds for every checkpoint;
`near/ep` is steps per episode spent within 1 m of the goal line.

| variant | succ | return | len | near/ep | a_x@goal | V_w@goal | dV(8→9.9) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hPPO baseline s1 | 0.00 | −147.1 | 200.0 | **59.5** | −0.51 | 1.89 | **−2.25** |
| hPPO baseline s2 | 1.00 | −719.8 | 106.8 | 19.0 | −0.29 | 0.65 | **−1.74** |
| hPPO ent-lr s1 | 0.00 | −177.5 | 200.0 | 0.0 | n/a | n/a | +0.10 |
| hPPO succ-bonus s1 | 1.00 | 1019.1 | 75.9 | 7.8 | +1.87 | 19.53 | **+1.34** |
| hPPO succ-bonus s2 | 1.00 | 1020.5 | 74.5 | 7.7 | +2.24 | 19.98 | **+0.54** |
| hPPO extr-mix s1 | 1.00 | 1020.3 | 74.8 | 7.8 | +2.04 | 19.97 | **+1.07** |
| hPPO extr-mix s2 | 1.00 | 1020.3 | 74.7 | 7.6 | +2.13 | 19.97 | **+1.07** |
| PPO+MPC (control) | 1.00 | 956.5 | 80.9 | 8.3 | −0.02 | n/a | n/a |

Three things this table settles that the training logs alone do not:

- **`a_x@goal` alone is not the discriminator.** The PPO+MPC control sits at
  −0.02 and crosses fine — a QP decelerating onto a goal *state* legitimately
  commands negative `a_x` near the line. Read it together with `near/ep`.
- **`near/ep` is the clean cross-algorithm read.** Healthy policies, learned or
  not, spend 7.6–8.3 steps in the last metre and leave. The collapsed baseline
  spends 59.5 and never does.
- **`dV(8→9.9)` is the learned-worker-specific read**, and it separates
  perfectly: negative for every unfixed run, positive for every fixed one.
- The `ent-lr` row shows the metric's one blind spot: that policy collapsed so
  far back that it never enters the strip at all, so there is nothing to
  average. No samples is its own signature, not a clean bill of health.

Worth noting for the thesis comparison: **the fixed hPPO (1019–1020) now beats
PPO+MPC (956) on this task.**

### 6.3 hPPO on tunnel — 200k steps

**The tunnel does not discriminate.** The corridor is solved by update ~20 and
200k steps is not long enough afterwards for the pathology to bind (baseline
`worker_ax_near_goal` stays at +0.9). The bonus is neutral-to-slightly-better
(218.9 / 219.2 against 217.6, oracle ≈ 213), and `worker_value_near_goal` moves
0.6 → 19.9 exactly as designed. The fix is on by default there anyway, so that
slalom and tunnel compare *the same algorithm*.

(These runs used the tunnel's former `goal_reward` of 200. At that scale
`--worker-extrinsic-coef 0.02` gave the worker a +4 terminal, and the
coefficient's measured floor was about 0.018 — see §10. Since 2026-09-24 the
tunnel uses the slalom's 1000, so the same coefficient gives +20 on both
scenarios.)

This is the reason the slalom carries the whole argument, and it is worth being
explicit that it leaves the evidence resting on **one scenario**: see
[§10](#10-what-this-evidence-does-not-cover).

### 6.4 Non-invasiveness

The instrumentation must not perturb what it measures. Verified empirically, not
by inspection: a stripped copy of `script_hppo.py` with all three diagnostic
blocks removed, run against the instrumented file at the same seed and budget
with both knobs at 0.0, produces **bit-identical** training output across every
logged field. The same read-only pattern (index `obs`, append to a list) is what
was added to the `ppo_mpc*` scripts.

This also confirms the other half of the claim: **with both knobs at 0.0 the
patched trainers reproduce the pre-fix behaviour exactly**, so every run recorded
before this change remains directly comparable.

Full suite: 427 tests pass (`algorithms` 303, `scenarios` 124).

---

## 7. Diagnostics added, and how to read them

| metric | where | what it is |
| --- | --- | --- |
| `charts/worker_ax_near_goal` | all 6 hierarchical scripts | mean commanded `a_x` from states within 1 m of the goal line |
| `charts/worker_value_near_goal` | `hppo` | mean `V_worker` over those states |
| `charts/near_goal_samples` | all 6 | how many such states the rollout visited |

Deliberately under the **same metric names** in the PPO+MPC scripts, which have
no learned worker: that puts the control arm and the treatment arm on one plot.
What the metric measures there is different — a QP tracks whatever goal it is
handed, so near the line it reads out the *manager's* intent rather than a
worker's self-interest — and the comment at each call site says so.

Reading them:

- **Healthy:** `worker_ax_near_goal` well positive and rising,
  `worker_value_near_goal` rising as the policy approaches the line,
  `near_goal_samples / num_episodes` ≈ 8.
- **Falling in:** `worker_ax_near_goal` drifting toward 0 and then negative,
  while eval success is still 1.00. This is the leading indicator; it precedes
  the visible crash by ~30 updates.
- **Collapsed:** `near_goal_samples / num_episodes` in the tens, episode length
  at `max_steps`, `worker_ax_near_goal` clearly negative.

---

## 8. Changes, file by file

| file | change |
| --- | --- |
| `scenarios/slalom/scripts/script_hppo.py` | two knobs (`--worker-extrinsic-coef`, default **0.02**; `--worker-success-bonus`, default 0.0), the reward patch, the diagnostics, and the full in-code writeup of the diagnosis, the ablation, and why the default changed |
| `scenarios/tunnel/scripts/script_hppo.py` | same defaults, with tunnel's own (non-discriminating) evidence recorded |
| `scenarios/{slalom,tunnel}/scripts/script_ppo_mpc.py` | control-arm diagnostic only, no knobs |
| `scenarios/{slalom,tunnel}/scripts/script_ppo_mpc_reach.py` | control-arm diagnostic only, no knobs |
| `algorithms/common.py` | dated addendum to `ManagerActor`'s docstring correcting the earlier manager-side diagnosis |

Both `script_hppo.py` loops have since been merged into `algorithms/hppo/hppo_train.py`,
which carries the slalom's write-up with the tunnel's notes; the scripts are now thin
per-scenario wrappers.

---

## 9. Open items

Found while investigating, documented, **not** changed — each would fold an
uncontrolled variable into the comparisons above.

1. **`--replan-on-collision` is a train/eval mismatch.** It is applied in the
   rollout but not in the evaluation loop or in `make_policy_fn`'s solved-check,
   both of which re-plan only on the `manager_freq` boundary. It mattered while
   collision rates were 8–24 per episode; it is close to moot at 0–1.
   **Since removed** from hPPO: the manager now re-plans only on the
   `manager_freq` boundary and at episode end, in training as in evaluation.
   It also carried a small GAE bug: when a collision-forced segment was the
   last one an environment stored in a rollout, its continuation value was
   counted twice (folded into the reward, then bootstrapped again). See the
   comment at `manager_act_now` in the hPPO training loop.
2. **`max_goal_bound = 10.0` makes almost every goal unreachable within a
   segment.** The manager effectively controls a *direction* only, and the
   magnitude axis is a flat direction in its action space that lets the Beta's
   `alpha` run to the boundary at no cost. Worth an ablation against something
   near the 1.2 m a segment can actually cover.
3. **The entropy autotuner is inert at its default learning rate** (§3). Fixing
   it does not fix the collapse, but it does mean no run to date has actually
   been entropy-regularised the way its configuration claims.
   Flat PPO and hPPO have since both dropped their autotuners for the fixed
   `ent_coef = 0.01` they were effectively running at (see `PPOAgent` in
   `algorithms/ppo/ppo.py` and `HPPOAgent` in `algorithms/hppo/hppo.py`; the
   13-seed re-measurement is in `benchmark.md`).
4. **The flat baseline (`ppo`) carries no `near_goal` diagnostic.** Adding
   it would complete the plot with a reference line that is immune by
   construction; it was left out as being outside the hierarchical scope.

---

## 10. What this evidence does not cover

Stated plainly, because the shape of the argument makes it easy to over-read.

- **One affected algorithm.** `hppo` is the only agent in this tree with a
  learned, intrinsic-reward worker, so the *treatment* arm is a single
  algorithm. The three control arms (§4) are what make the comparison
  controlled — they isolate the worker's reward from hierarchy, from the
  manager network and from the goal interface, all of which they share — but
  no evidence here says how the defect behaves under a different low-level
  learner. The derivation in §1 is not PPO-specific; the measurements are.
- **One discriminating scenario.** The tunnel is solved before the pathology
  can bind (§6.3), so every number that separates the two fixes comes from the
  slalom.
- **Two to three seeds per arm.** Enough to separate 0/2 from 2/2 and 266k from
  71k; not enough for a fine ranking.
- **`--worker-extrinsic-coef` was not swept.** 0.02 is a scale argument
  (§5.2) that worked on the first value tried, not the optimum of a sweep.
  `scenarios/slalom/scripts/sweep_extrinsic_coef.py` (2026-09-24) runs that
  sweep, from 0.02 down to 0. It also probes whether the manager still steers
  the worker at each coefficient, because the extrinsic mix risks exactly that
  (§5.2). The first run was 10 slalom seeds per arm at 500k steps, every early
  stop off. **Performance is flat from 0.005 to 0.02.** Every arm solved 10/10
  seeds and was still solved at the end. First-solve medians were 67–82k
  steps, with p ≥ 0.46 against 0.02. Only 0 collapses: 2/10 seeds ever solved,
  0/10 at the end. **The coefficient trades against the hierarchy.** At 0.02
  the worker threads the slalom almost unaided: with the manager replaced by a
  fixed forward goal it averages about 1 wall contact per episode, against
  about 6 at 0.005. Its goal share of action variance is 78 %, against 91 % at
  0.005. The script's docstring derives a floor for the coefficient, below
  which the terminal no longer outweighs stalling. Its first estimate (0.011 on
  the slalom) proved conservative. Recomputed with the goal progress the
  collapsed orbits actually harvest, the floor is about 0.005 on the slalom.
  On the tunnel it was about 0.018 while that scenario's `goal_reward` was 200.
  Since 2026-09-24 both scenarios share `goal_reward` 1000, and with it the
  same floor.
