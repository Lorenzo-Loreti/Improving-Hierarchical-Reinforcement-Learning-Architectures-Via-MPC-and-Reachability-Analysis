# Speed and thrust bounded in norm, a control-effort penalty, and an exact oracle

**What this document is.** The record of one change to the plant and the
objective shared by every algorithm (2026-10-04): the speed and thrust limits
bound magnitudes instead of each axis, the disturbance becomes isotropic, the
reward charges control effort and stops paying for the overshoot past the goal
line, and the oracle and the tube MPC model all of it exactly. The first rerun
exposed a defect of the wall-contact model at the gates' faces, fixed the same
day (section 7), and with it the slalom's training budget for flat PPO and
hPPO. Why each part was made, what it changes and what it leaves alone, how it
was checked, and what the rerun of every study says. It is written so that the corresponding
sections of the thesis can be lifted from it.

**Companion documents**

- [`benchmark.md`](benchmark.md) — the evaluation protocol (the oracle, the
  strict solved-check on the 5 x 5 grid). Its numbers are on the old dynamics.
- [`progression.md`](progression.md) — the algorithms being compared.
- [`init-sampler.md`](init-sampler.md) — the last change before this one.

**Code**

- `scenarios/{slalom,tunnel}/envs/actuation.py` — the thrust and speed limits
  and the disturbance, with the rationale in the slalom copy's docstring.
- `scenarios/{slalom,tunnel}/envs/config.py` — `effort_penalty`, the cut
  progress potential, and why each value.
- `algorithms/optimal_solver.py` — the oracle.
- `algorithms/tube_mpc.py` — PPO+MPC's worker.
- Git tag `box-limits-final` — the last commit on the old dynamics. Every
  result recorded before 2026-10-04 is on them.

---

## 1. Summary

**The plant.** The vehicle is a point mass with one thruster that can point
anywhere. Its limits now bound magnitudes, `||u|| <= u_max = 2.5 m/s^2` and
`||v|| <= v_max = 1.2 m/s`, where they used to bound each axis. The
disturbance, when there is one, is uniform on two disks, `||w_p|| <= b_p` and
`||w_v|| <= b_v`, where it used to be uniform on a box.

**The objective.** The reward gains a control-effort term,
`effort_penalty * (||u|| / u_max)^2` per step on the acceleration actually
delivered, with `effort_penalty = -0.01`; and the progress potential is cut
at the goal line, so the last step earns the distance up to the line and
nothing for how far past it lands. The return of a successful episode is then
a function of its length, its effort and its contacts only.

**The models.** The oracle maximises exactly that return: minimum time
first, then minimum effort among the minimum-time trajectories (a
second-order cone program per horizon, solved by Clarabel). The tube MPC
models the disks exactly — no polygon anywhere in the solution — because the
closed loop is symmetric under rotations of the plane, which makes every
tightened set a disk.

**The gates' faces.** A vehicle entering a gate outside its opening used to
be clamped sideways through the gate's face into the opening, a jump of up to
1.5 m in one step. With the disks every flat-PPO seed learned to use it; a
gate's face is now a wall that stops the vehicle (section 7). Without the jump
the gates are an exploration problem for the learners, and the slalom studies
give flat PPO and hPPO 1 024 000 steps instead of 204 800.

**What it does.** On the box the slalom's gates cost no time at all: the
oracle arrived from every grid start on exactly the step a straight,
unobstructed run arrives on (78.0 steps on average), threading both gates
while flying at up to 1.47 m/s, 22% over v_max, on the diagonal. With the
disks it needs 79.2 steps (0 to +4 per start): steering now costs forward
speed. The tunnel's oracle runs straight and is unchanged (78.0).

**What the studies say** (section 9). The tunnel is unchanged in kind: every
seed of every algorithm solves it. The slalom becomes an exploration problem
for the learners: in 1M steps flat PPO gets through on 14 of 20 seeds (9
pass the strict check), hPPO on none, while PPO+MPC solves 18 of 20 seeds at
the old budget (median 108k), never touches a wall, and stays contact-free
under the worst disturbance in W -- at the price of time against the oracle,
and of ~8-10x the old compute per disturbed run.

## 2. Why

### 2.1 The limits

Two defects of the per-axis box, both measured on the oracle:

- **The vehicle went faster on a diagonal**: up to `sqrt(2) v_max = 1.70
  m/s`, with `sqrt(2) u_max` of thrust. Nothing about the vehicle picks out
  the corridor's axes.
- **Steering was free.** The axes were decoupled, so a lateral manoeuvre never
  took anything away from `v_x`. On the slalom's 25-start grid every min-time
  trajectory arrived on the step of a straight run along x with no walls, and
  reached 1.47 m/s on the way. The gates — the thing the slalom exists for —
  cost nothing; the task was only "go straight at full speed, and do not hit
  anything".

Measured with the same oracle code on both sets of limits (2026-10-04,
deterministic environment, 25-start grid):

| | slalom | tunnel |
|---|---|---|
| straight-run lower bound (x alone) | 78.0 steps | 78.0 |
| oracle, per-axis limits | 78.0 (= the bound from all 25 starts) | 78.0 |
| oracle, norm limits | **79.2** (0 to +4 per start) | 78.0 |
| oracle's peak speed, per-axis limits | 1.47 m/s | 1.20 |
| oracle's peak thrust, per-axis limits | 2.65 m/s^2 | 2.50 |

The largest costs are from `p_y0 = -1`, the starts furthest from the first
gate's opening (+4 steps from `(2, -1)`).

### 2.2 The disturbance

The tube MPC tightens its constraints by the error set `Z`. A disk minus a
box is the intersection of four disks centred on the box's corners — smaller
than the disk the worst case along any one axis would need. At the
disturbance levels of the disturbed studies the nominal plan's radii would
have been:

| | box W, box limits (old) | box W, disk limits | disk W, disk limits (new) |
|---|---|---|---|
| input radius, 1x (0.005 / 0.05) | 1.69 | 1.56 | **1.69** |
| speed radius, 1x | 0.99 | 0.97 | **0.99** |
| input radius, 2x (0.01 / 0.1) | 0.90 | **0.31** | **0.90** |
| speed radius, 2x | 0.78 | 0.71 | **0.78** |

At twice the disturbance a box W would have left the plan 0.31 m/s^2 of
input, and its terminal constraint (come to rest within N = 10 steps) would
have capped the planned speed near 0.3 m/s. An isotropic W keeps exactly the
radii the box gave along the axes (section 6.1 says why), and it is the
natural disturbance for this vehicle: a push of bounded strength from any
direction.

### 2.3 The objective

Two differences between what the oracle optimised and what the reward paid:

- **The oracle minimised effort and the reward did not know.** Many
  trajectories share the minimum arrival time; the oracle returned the one
  with the least `Sum ||u||^2`, a tie-break invisible to the reward. An agent
  arriving on the same step with bang-bang thrust scored the same.
- **The reward paid for the overshoot and the oracle did not.** The progress
  term telescoped to `progress_reward_coef * (p_x_final - p_x0)`, with
  `p_x_final` where the last step lands past the line. The oracle aimed at
  `L` plus 1 mm, so an agent could out-score it by up to `10 * v_max * dt =
  1.2` on the same arrival step, and still come out 0.2 ahead one step later
  (measured 2026-09-24: PPO crossed 3.6 cm past the line on average against
  the oracle's 0.1 cm, return gaps down to -0.73).

Charging the effort makes the oracle's tie-break part of the objective;
cutting the potential at `L` removes the overshoot. After both, the reward and
the oracle maximise the same quantity.

## 3. The plant

`envs/actuation.py`, used by both the single and the vector environment
(elementwise float32 code, so the two compute the same bits for the same
input).

**Thrust.** The command is saturated radially: `u <- u * min(1, u_max /
||u||)`, keeping its direction. The action space stays the box `[-u_max,
u_max]^2` (gym needs a box); its corners are saturated like anything else
outside the disk. Before, an oversized command was clipped per axis, which
turned `(5, 1)` toward the diagonal and let `(5, -5)` through at `sqrt(2)
u_max`.

**Speed.** If `v + u dt` would leave the speed disk it is projected onto it,
and `u` is replaced by what lands exactly there, `(v' - v) / dt`. The
position integrates over the velocity the vehicle really had, so a step at
top speed covers `v_max * dt` and no more, in any direction. The projection
onto a convex set that holds `v` is non-expansive, so the delivered `u` is
never longer than the saturated command: every step the environment takes is
a step the oracle's and the MPC's model can take with an admissible input.
(This is the 2026-09-24 fix of the speed limit — acting on the delivered
acceleration, not clipping the state after the update — carried over to the
disk.)

**After the step** the position is clipped to its box and the velocity
projected onto the speed disk: a guard against the disturbance and float32
rounding only.

**Disturbance.** `w_p = b_p sqrt(U_0) (cos 2 pi U_1, sin 2 pi U_1)` and
`w_v` likewise from `(U_2, U_3)`: uniform in area on each disk, independent
of each other. Nothing is drawn when both radii are 0.

## 4. The objective

### 4.1 The two changes

Per step, on top of whichever branch fired (time penalty, contact, goal):

```
reward += progress_reward_coef * (min(p_x', L) - p_x)        # cut at the line
reward += effort_penalty * (||u_delivered|| / u_max)^2       # effort, -0.01 at full thrust
```

The effort is charged on the delivered acceleration, after the thrust and
speed limits: a command the speed limit absorbs costs nothing, and the
environment's effort is exactly what the oracle's model minimises.

The return of a successful, contact-free episode that crosses on step N from
`p_x0`:

```
R = goal_reward - (N - 1) + progress_reward_coef * (L - p_x0) - 0.01 * Sum_k (||u_k|| / u_max)^2
```

### 4.2 Why `effort_penalty = -0.01`

It is chosen so that **time always comes first**, and effort only breaks
ties between equally fast trajectories.

A step at full thrust costs 0.01; one step of the episode costs 1. So two
successful episodes arriving on steps `N_1 < N_2` are always ordered by time
when `N_1 < 100`:

```
R(N_1) >= G - (N_1 - 1) + P - 0.01 N_1  >  G - (N_2 - 1) + P  >=  R(N_2)
   <=>  N_2 - N_1 > 0.01 N_1,   true for every N_2 > N_1 when N_1 < 100.
```

Every minimum-time episode from the spawn box arrives within 90 steps (88 at
most on the slalom grid), so the reward's optimum is always a minimum-time
trajectory — the task is still "as fast as possible" — and among those the one
with the least effort. The oracle's whole-episode effort on the grid is
-0.040 on average on the slalom (-0.051 at most) and -0.035 on the tunnel:
it thrusts hard for the ~5 steps it takes to reach top speed and barely at
all afterwards.

What a larger weight would do: past ~0.012 an episode's effort can outweigh a
step, the oracle may arrive later to save effort, and the task becomes a
time-energy trade-off (`test_a_heavy_effort_penalty_trades_time_for_effort`
exercises this with -20). That is a different problem, and not the one the
thesis poses.

What the weight costs the learning signal: per step the effort term is at
most 0.01 against a time penalty of 1 and a progress term of up to 1.2, so it
should change how the agents thrust (smoother, less saturated) more than
where they go. Section 9 measures it.

### 4.3 Who sees it

It is part of the environment's reward, so every algorithm optimises it,
through whatever reward it learns from:

- **Flat PPO** — directly.
- **hPPO** — the manager through the macro-step's accumulated reward, as
  before. The worker produces `u`, but sees the environment's reward only
  through `--worker-extrinsic-coef` (0.02), at which the effort would reach it
  at 0.0002 per full-thrust step, against an intrinsic goal-progress stream of
  ~0.12: not at all. So the worker takes the effort term at full weight
  instead, `--worker-effort-coef 1.0`:
  `r_worker = r_int + 0.02 * (r_env - r_effort) + 1.0 * r_effort`. At full
  thrust that is ~8% of a step's best goal progress.
- **PPO+MPC** — the manager, through the macro-step's reward, is charged for
  the effort its MPC worker spends: the environment charges the `u` the
  worker delivers. The worker itself is a fixed controller with its own
  tracking cost, `Sum z'Qz + v'Rv` with `R = 0.1 I`: a quadratic input cost of
  the same form. Its weight is a design parameter of the tube, not a copy of
  the environment's — it also sets the ancillary gain `K` and through it the
  error set `Z` — and was left as it was.

The oracle maximises the same return (section 5). What remains different
between the learners and the oracle is intrinsic to RL: the agents maximise
the discounted return (gamma = 0.99), the oracle and the solved-check the
undiscounted one.

## 5. The oracle

`algorithms/optimal_solver.py`. The objective of section 4.1, maximised
exactly, by a search over the arrival step N:

- **Per N**, minimise `Sum ||u_k||^2` subject to the dynamics, `||u_k|| <=
  u_max` and `||v_k|| <= v_max` as second-order cones, the corridor's lateral
  bound at every stage, `p_x_k <= L - 1 mm` before step N and `p_x_N >= L + 1
  mm`. The corridor bound depends on `p_x`, a decision variable, which is
  what makes the problem non-convex; it is handled as before, by sequential
  convex programming over the segment assignment, with restarts. The rest is
  convex, solved by Clarabel (OSQP, used before, solves QPs only).
- **Over N**, from the x-only lower bound upward (still valid: `||u|| <=
  u_max` implies `|u_x| <= u_max`), stopping at the first N whose return, even
  with zero effort, could not beat the best found so far. With
  `effort_penalty = -0.01` that is always the first feasible N (section 4.2).

Every number it reports is still from replaying its actions through the
environment's own `step`; the model's prediction is kept alongside
(`predicted_return`) and agrees with the replay to 3e-5 on every grid start
of both scenarios. The grid takes ~2 s on the slalom.

The oracle remains a heuristic in one respect, as before: SCP finds a fixed
point of the segment assignment, not a certified global optimum.

## 6. The tube MPC

`algorithms/tube_mpc.py`, PPO+MPC's worker.

### 6.1 Why the disks are exact

The note's sets are polytopes. With disks they stay exact because of a
symmetry. Let `R = diag(R_theta, R_theta)` rotate the plane by any angle,
acting on positions and velocities alike. Then:

- `A_K = A + B K` commutes with `R`. The plant is the per-axis double
  integrator and Q, R weigh both axes alike, so the LQR gain is one 2x2 block
  per axis, the same for both: `A_K = A1 (x) I_2`.
- `K R = R_theta K`, for the same reason.
- `R W = W`: W is a product of disks.

So `R Z = Z` for `Z = (1 - alpha)^{-1} (+)_l A_K^l W`, and every projection
of Z the tightening needs is rotation-invariant: Z's velocity part and `K Z`
are disks, of radii `r_v = h_Z(e_vx)` and `r_u = h_Z(K^T e_x)`. A disk minus a
disk is a disk, so the tightened sets are exactly `||z_v|| <= v_max - r_v` and
`||v|| <= u_max - r_u`. The walls are half-planes, tightened by Z's position
radius as before. The construction checks the commutation at three angles and
refuses a closed loop that breaks it.

The radii are the half-widths the box W gave along the axes — 0.989 m/s and
1.692 m/s^2 at the canonical disturbance, 0.782 and 0.897 at twice it —
because every support function the algorithm evaluates is along a direction
`(a_p n, a_v n)`, where the box and the disks agree. Algorithm 1's stopping
test `A_K^s W in alpha W` has a closed form for the disks too (exact here,
since every block of `A_K^s` is a multiple of the identity), and it returns
the same `s` and `alpha` as the box did.

### 6.2 How it is solved

The problem is mixed-integer (the gates' faces), and its continuous part is
convex but not polyhedral (the disks, and `x - z_0 in Z`). Two solvers split
it:

- **Gurobi** solves a relaxation as a mixed-integer QP: each disk replaced
  by its circumscribed 16-gon (one side tangent at +x, so a straight cruise is
  exact), and Z by 64 half-spaces `a^T (x - z_0) <= h_Z(a)`. Because the
  relaxation contains the exact problem, its optimal binaries come with a
  lower bound on the exact optimum.
- **Clarabel** solves the exact problem with those binaries fixed: a convex
  program with the disks, and Z through its lifted representation
  `x - z_0 = Sum_l L_l omega_l` with every `omega_l` in W, as cones (~3-5 ms).

If the exact cost is within the MIP gap (1e-4) of the lower bound, the plan
is optimal for the exact problem to that gap — the guarantee the worker always
had. Otherwise the relaxation is tightened where it was loose (tangents at the
exact plan's points on the edge of their sets, Z's supporting half-space at
its `x - z_0`, and the tangent wherever the relaxation's own plan left a set,
Z's from a separation oracle) and both are solved again. When every binary
is already fixed by the reachability pruning — most steps, and every step on
the tunnel — the problem is convex and Clarabel solves it alone.

Measured closed-loop cost per step (2026-10-04, slalom):

| | hand-written gate manager, 3 episodes | random goals (as in early training), 4 episodes |
|---|---|---|
| no disturbance | 15 ms (max 25) | 8.8 ms |
| 1x disturbance | 32 ms (max 111) | 14.3 ms |
| 2x disturbance | 34 ms (max 176) | — |

every outcome SOLVER, no contact, tube ratio <= 0.998. The old box worker
took ~10 ms.

### 6.3 What was tried first, and why it was dropped

- **Gurobi alone, the disks as quadratic constraints (MIQCP).** Its
  relaxations go through the barrier, which ran into "numerical trouble" on
  about one solve in ten and then stalled until the time limit — on instances
  Clarabel solves in milliseconds, even with every binary fixed. NumericFocus,
  BarHomogeneous, ScaleFlag, Aggregate and the outer-approximation
  MIQCPMethod each fixed some instances and not others.
- **Gurobi alone, with cutting planes only (Kelley).** Robust, but the
  violation fell only ~4x per round, so a step took ~8 MIQP solves (~60 ms)
  undisturbed; and on Z's lifted representation, whose 34 omegas the cost
  does not depend on, the solver moved them to a new vertex every round and
  often ran out of rounds.

### 6.4 Two consequences

- **No speed constraint on stage 0.** It is redundant — the measured state is
  what it is, and every later real velocity is covered by stages 1..N
  (`x_{k+1} = z_1 + e_1`, `e_1 in Z`), so theorem 4.1 does not use it — and
  harmful: without a disturbance `z_0 = x`, and a vehicle cruising at the
  limit has `||x_v||` within float32 rounding of `v_max`, sometimes above it,
  leaving the disk on a fixed point with no interior.
- **The tube keeps the environment's speed limit from ever acting.** `v + u
  dt`, the velocity before the step's disturbance, is `z_v' + (A_K e)_v`, of
  norm at most `(v_max - r_v) + (r_v - b_v) < v_max`, since `A_K Z (+) W` lies
  in Z. The plant stays linear, as the error dynamics require.

### 6.5 The one remaining approximation

The gates are enlarged by Z with their own normals (the note's proposition
2.5), an outer approximation of `O (+) (-Z)`. Under the box W, Z's position
part was a box and the enlargement exact. With a disk it is not: the exact
sum has rounded corners, and the enlarged gate walls keep square ones, up to
`(sqrt(2) - 1) r_p` beyond them — 3.4 cm at the canonical disturbance, 6.8 cm
at twice it, only at the gates' corners. It is conservative, on the safe
side. The exact rounded corner would need a non-convex constraint, `||p -
corner|| >= r_p`, which neither the relaxation nor the convex step can hold;
a finer polygonal corner would shrink the gap, not close it. Without a
disturbance `r_p = 0` and there is no gap.

## 7. The gates' faces: a contact model fixed on the way

### 7.1 What the first rerun showed

The first 20-seed flat-PPO study on the new dynamics (204 800 steps) solved
**0/20** seeds, against 20/20 on the box. Its policies reached the goal from
every grid start, within 1.3 steps of the oracle, but with ~1.1 wall contacts
per episode, and one contact (-50) alone fails the strict solved-check
(tolerance 5). Every seed showed the same pattern, stable from its second
evaluation on.

All 579 grid contacts were the same event: the vehicle entering a gate
outside its opening. The contact model then clamped `p_y` to the violated
bound and zeroed `v_y` -- right for a wall along the corridor, wrong for a
gate's face, a wall across it: it moved the vehicle sideways *through* the
face into the opening, 0.63 m on average and up to 1.50 m in one step, where
a whole step at top speed covers 0.12 m. Representative seed 10 crossed
`x = 7` at `y = 0.51`, above gate 2's opening (`y in [-1.75, -0.25]`), and
was put at `y = -0.25`. From its state just after gate 1 the oracle reaches
the goal without a contact in 44 steps against the policy's 42: the jump
saved 2 steps for 50, so it was a local optimum the policies could not leave,
not the objective's optimum.

The jump existed on the box too, but there moving sideways cost nothing, so
lining up was easy and the final policies did not use it. On the disks every
metre sideways is forward speed given up, and arriving misaligned and taking
the jump became the policies' habit. It persisted at longer budgets and under
a disturbance: in the disturbed comparison at 500 000 steps, 8 of 10 flat-PPO
seeds still took ~1.1 contacts per evaluation episode at the end, and 4 of 10
hPPO seeds never had a clean evaluation. (These runs, and the two
204 800-step studies, are kept under `scenarios/slalom/studies/gate-teleport/`,
which git ignores.)

### 7.2 The fix

A contact is a fully inelastic bounce off the wall that was hit: the velocity
component normal to it is absorbed and the position put back on its side.
The step's crossing into the new segment is located linearly between the two
samples. Outside the new segment's opening the vehicle hit the face, so
`p_x` is put back just outside it and `v_x` zeroed, its lateral motion
untouched; inside, it went through the opening, and a contact is with the
gate's side wall, as before. A corner gets both, for one contact. The oracle
and the tube MPC never touch a wall, so neither changes; nor do the tunnel's
results (its walls all run along it) or the slalom PPO+MPC study of the first
rerun, which had no contact in training, in evaluation or on the grid, so its
trajectories are the same under either rule.

### 7.3 What the fix does to learning

The clamp had also been a funnel: a misaligned learner was put into the
opening anyway, reached the goal, and collected the +1000 its learning starts
from. Without it the gates are an exploration problem. Pilots, 3 seeds each:

| | 204 800 steps | 1 024 000 steps |
|---|---|---|
| flat PPO | 0/3 solved: every seed learns to stop in front of a gate and wait out the clock | 2/3 (first solve 328k, 707k; the third stuck in front of gate 2) |
| hPPO | 0/3, all three stopped in front of gate 1 | -- |

The stopped policies are the "advance a bit, then idle" optimum the configs'
historical note records for terminal contacts (return ~-150). The slalom
studies of section 9 therefore give flat PPO and hPPO 1 024 000 steps;
PPO+MPC, whose worker never meets a face, keeps 204 800 (500 000 in the
disturbed comparison, as before), and the learning curves compare over the
common first 204 800 steps as well. The objective is unchanged: no shaping
was added to guide the learners through the gates.

## 8. Checking the implementation

All 470 tests pass. The ones written for this change:

- **Plant** (`envs/tests`, both scenarios): radial saturation keeps the
  direction; a step at top speed covers `v_max dt` for a heading off the axes
  and a thrust in any direction; steering at top speed keeps `||v|| = v_max`
  and lowers `v_x`; a step that hits the limit lands where an admissible
  input lands; over 20 000 random states and commands the delivered input is
  admissible and an in-limit command comes back bit for bit; the disturbance
  stays in its disks, reaches their edge off the axes, and is uniform in area.
- **Gate faces**: entering a gate outside its opening stops the vehicle at
  the face (the case is seed 10's), pushing on is another contact, the back
  face stops a vehicle moving backward, entering through the opening and
  grazing a side wall is a lateral contact, a corner absorbs both, and the
  vector environment matches the single one on all of them.
- **Reward**: the effort term on the delivered input (full thrust in any
  direction, half thrust, a thrust the speed limit absorbs); the progress cut
  at the line whatever the crossing speed; the vector environment reports the
  effort it charged.
- **Oracle**: return = the closed form, with the cut potential and the
  replayed effort, and = its own prediction; the plan respects the disks and
  pays for an offset gate; a heavy effort weight makes it trade time for
  effort; full thrust down the open corridor neither arrives earlier nor
  earns more.
- **Tube MPC**: the disk W's support and scaling against brute force; Z
  robustly invariant, within eps of the minimal RPI set, and
  rotation-invariant; the tightened sets are disks in every direction; an
  asymmetric closed loop is refused; the separation oracle separates; the
  candidate is feasible after a disturbance on the edge of W; the tube holds
  and the real speed and thrust stay inside their limits under worst-case
  disturbances; and the plan's cost is the same whatever relaxation found it
  (a 4-gon, a 16-gon, a 64-gon).

## 9. The studies on the new dynamics

Every study was rerun on the new dynamics (commits `0f21a0b`, `ec8caae`).
The seed studies' summaries and figures are in each study's directory under
`scenarios/{slalom,tunnel}/studies/` (ignored by git); the tables below are
also recorded in the scripts' docstrings. Effort is the episode's effort term
in reward units (`-0.01 * Sum (||u||/u_max)^2`), averaged over the 25 grid
starts; first solve is the first evaluation that passes the strict
solved-check (return within 5 of the oracle from all 25 grid starts).

### 9.1 Tunnel

Seeds 1-20 (PPO+MPC 1-10), 204 800 steps, no disturbance. The tunnel had
only a 3-seed benchmark before (2026-09-23).

| | PPO+MPC | hPPO | PPO |
|---|---|---|---|
| solved within the budget | 10/10 | 20/20 | 20/20 |
| first solve, median (range) | 72k (51k-82k) | 31k (31k-41k) | 26k (20k-31k) |
| solved at the last evaluation | 10/10 | 20/20 | 20/20 |
| grid starts on the oracle's step | 80% | 46% | 70% |
| grid mean extra steps | +0.20 | +0.58 | +0.30 |
| grid contacts per episode | 0 | 0 | 0 |
| grid effort per episode (oracle -0.035) | -0.051 | -0.038 | -0.041 |

The tunnel is a control problem, not an exploration one: every seed of every
algorithm solves it and holds it. The learners' effort is within 10-20% of
the oracle's; the tube worker spends ~45% more.

### 9.2 Slalom

PPO+MPC seeds 1-10 at 204 800 steps; flat PPO and hPPO seeds 1-20 at
1 024 000 (section 7.3).

| | PPO+MPC | hPPO | PPO |
|---|---|---|---|
| solved within the budget | 8/10 | 0/20 | 9/20 |
| first solve, median (range) | 118k (102k-195k) | -- | 748k (338k-973k) |
| solved at the last evaluation | 4/10 | 0/20 | 6/20 |
| solved-checks passed after the first solve | 50% | -- | 52% |
| training contacts per episode | 0 | 0.28 | 0.28 |
| grid starts on the oracle's step | 8% | 0% | 1% |
| grid mean extra steps | +1.36 | never arrives | +37.9 |
| grid effort per episode (oracle -0.040) | -0.073 | -0.398 | -0.052 |

Against the box (2026-09-24/29, all at 204 800 steps): PPO 20/20 solved at a
median of 51k, hPPO 20/20 at 72k, PPO+MPC 10/10 at 72k.

- **PPO+MPC** is the only algorithm that solves the slalom at the old budget,
  and its worker never touches a wall. It is less precise than on the box:
  the gates cost time now, and arriving on the oracle's step takes goals
  placed where the time-optimal path threads them. It ends 1.4 steps behind
  the oracle on average and the strict check flickers (4 of 10 seeds pass it
  at the last evaluation).
- **hPPO** never gets through: every seed ends stopped, waiting out the clock
  -- 17 between the gates in front of gate 2, 2 in front of gate 1, 1 inside
  gate 1. On the tunnel it solves 20/20, so what it cannot pass is the gates'
  faces, not a broken hierarchy.
- **Flat PPO** finds the way through on 14 of 20 seeds, late, and then reaches
  the goal from every start; 9 of them passed the strict check. The other 6
  stay stopped in front of a gate.
- **Effort**: the tube worker spends 1.8 times the oracle's effort (its
  tracking cost weighs the input lightly, r = 0.1 against 10 on position
  errors); flat PPO is within 30% of the oracle; hPPO's stopped workers keep
  thrusting to hold position.

### 9.3 Disturbed slalom

PPO+MPC at 500 000 steps, flat PPO and hPPO at 1 024 000, seeds 1-10, every
early stop off. "Clean": an evaluation with every episode at the goal and no
contact. Medians over seeds.

| `|w_p| <= 0.005, |w_v| <= 0.05` | PPO+MPC | hPPO | PPO |
|---|---|---|---|
| eval return, last 100k | 1001.8 | -146.7 | 1008.3 |
| grid gap to the undisturbed oracle, last 100k | 10.3 | 1157 | 28.7 |
| eval contacts per episode, last 100k | 0 | 0 | 0 |
| training contacts per episode, whole run | 0 | 0.31 | 0.34 |
| seeds with a clean evaluation | 10/10 | 0/10 | 6/10 |
| first clean evaluation, median | 51k | never | 532k |

| `|w_p| <= 0.01, |w_v| <= 0.1` | PPO+MPC | hPPO | PPO |
|---|---|---|---|
| eval return, last 100k | 991.3 | -147.6 | -142.7 |
| grid gap to the undisturbed oracle, last 100k | 20.9 | 1158 | 1153 |
| eval contacts per episode, last 100k | 0 | 0 | 0 |
| training contacts per episode, whole run | 0 | 0.30 | 0.16 |
| seeds with a clean evaluation | 10/10 | 0/10 | 4/10 |
| first clean evaluation, median | 61k | never | never |

- **PPO+MPC** is the only algorithm every seed of which reaches the goal
  cleanly, at both levels, from ~50-60k steps on, without a contact in any
  training episode, a candidate fallback or an emergency. It pays in time:
  10.3 and 20.9 below the undisturbed oracle, against 4.6 and 10.6 on the
  box, the gates now costing time on top of the tube's margin.
- **hPPO** never gets through, as without a disturbance.
- **Flat PPO** gets through on 6 seeds of 10 at the first level and 4 at
  twice it; where it does, it mostly ends closer to the oracle than PPO+MPC
  (gaps of 2-9 on 5 of the 6 seeds at the first level).

### 9.4 Under harder disturbances in the same W

`stress_disturbed.py` replays every final policy of 9.3 from the 25 grid
starts under three disturbances: uniform on W (as in training), at a random
extreme point of W (the disks' edges), and at the extreme point pushing
toward the nearest forbidden point (an aimed adversary). Contacts per
episode, medians over seeds:

| | uniform | edge | adversary |
|---|---|---|---|
| PPO+MPC, 1x / 2x | 0 / 0 | 0 / 0 | **0 / 0** (closest approach 4 mm / 19 mm) |
| flat PPO, 1x / 2x | 0.02 / 0 | 0.02 / 0 | **8.1 / 102** |
| hPPO, 1x / 2x | 0 / 0 | 0 / 0 | 0 / 4.5 (it never reaches the goal) |

Theorem 4.1 in practice: the tube holds against the worst disturbances in W,
at a cost of 0.6 and 3.9 in return against uniform noise. The learned
margins hold against random noise and fail against an aimed push.

### 9.5 Training starts: independent vs Sobol'

`study_init_sampler.py`, seeds 1-20 per arm, flat PPO at 1 024 000 steps and
PPO+MPC at 204 800.

| | flat PPO, independent / Sobol' | PPO+MPC, independent / Sobol' |
|---|---|---|
| solved | 9/20 / 7/20 | 18/20 / 18/20 |
| first solve, median | 748k / 553k (p = 0.8) | 108k / 108k (p = 0.65) |
| solved at the end | 6/20 / 6/20 | 11/20 / 13/20 |

Still nothing measurable. The independent arms reproduce the seed studies
seed for seed, PPO+MPC's included although its seed study was trained before
the gate-face fix and this arm after it: the fix cannot reach a controller
that never touches a wall. Over 20 seeds PPO+MPC solves the slalom 18/20 at
a median of 108k. The gradient-variance probe (`probe_start_variance.py`,
flat PPO seeds 1-3 at 102k-1024k) puts the starts at +4% (-5% to +13%) of
the policy gradient's variance, as on the old dynamics (+3%).

### 9.6 Dropped

hPPO never gets through the slalom's gates on these dynamics (0/20 at 1M,
0/10 disturbed at either level), so two studies whose question assumes
episodes that reach the goal were dropped by decision of 2026-10-04: the
`--worker-extrinsic-coef` sweep (`sweep_extrinsic_coef.py`, which measures
the coefficient below which the worker stops crossing the line) and the hPPO
arms of the init-sampler study. Their results on the old dynamics stand,
labelled as such; the commands to rerun them are in section 12.

### 9.7 What it adds up to

- On the tunnel, a control problem, nothing changes qualitatively: every
  algorithm solves every seed, flat PPO fastest, PPO+MPC most precise.
- On the slalom, with steering costing speed and the gates' faces real walls,
  the slalom becomes an exploration problem the learners mostly fail: flat
  PPO gets through on 14 of 20 seeds within 1M steps, hPPO on none. PPO+MPC,
  whose worker knows the walls, solves it at the old budget, without a
  contact, disturbed or not, and with its safety guaranteed against the worst
  disturbance in W -- paying 1.4 steps undisturbed and 9-19 disturbed against
  the oracle, and ~8-10x the old compute per disturbed run.
- The effort term leaves the task time-first by construction (section
  4.2) and is now measured per policy: the learners that reach the goal
  spend within 10-30% of the oracle's effort, the tube worker 1.5-1.8 times
  it.
- Wall clock: a disturbed PPO+MPC run took ~940 min (1x) and ~1180 min (2x)
  with 14 runs in parallel, 81-96 ms per solve under that load (~14 ms
  alone), against 121 min on the box. Calling Clarabel directly instead of
  through cvxpy is the first saving to try.

## 10. What this does not cover

- **ppo_mpc_reach** still models the box (its worker, `algorithms/mpc_worker.py`,
  and its reachable goal set). It is to be rebuilt on the tube MPC; its scripts
  refuse to run on the current environments, and the git tag
  `box-limits-final` has the ones it matches.
- **Discounting.** The learners maximise the discounted return, the oracle
  the undiscounted one (section 4.3).
- **The gates' corners** (section 6.5).
- **hPPO on the slalom.** It never gets through the gates on these dynamics,
  so whatever the hierarchy could add there is unmeasured; the
  extrinsic-coefficient sweep and its init-sampler arms were dropped
  (section 9.6). Reward shaping toward the gates' openings, potential-based
  so that the optimal policy is unchanged, is the obvious next step if the
  learners are to be compared on the slalom at all; none was added here.
- **The tube MPC's cost.** The exact disks cost ~8-10x the old compute per
  disturbed PPO+MPC run under full load (section 9.7).

## 11. Changes, file by file

| file | change |
|---|---|
| `scenarios/*/envs/actuation.py` | new: thrust and speed limits, disk projection, isotropic disturbance |
| `scenarios/*/envs/{slalom,tunnel}_env.py`, `vec_*_env.py` | use it; the cut progress term; the effort term; `info["effort_penalty"]`, `info["delivered_action"]` |
| `scenarios/*/envs/config.py` | `effort_penalty = -0.01`, with its rationale; the disk disturbance |
| `scenarios/*/envs/width_profile.py`, `*_env.py` | the gate-face contact: `boundary_between`, `outside_face` (section 7) |
| `scenarios/slalom/scripts/compare_disturbed.py`, `study_init_sampler.py` | `--budget`, per-algorithm budgets |
| `algorithms/optimal_solver.py` | SOCP per horizon (Clarabel), return-maximising search, `predicted_return` |
| `algorithms/tube_mpc.py` | `DiskProduct`, closed-form `scaling`, the symmetry check, `_ExactProblem`, `_GaugeOracle`, the relaxation-and-certificate loop |
| `algorithms/hppo/hppo_train.py` | `--worker-effort-coef` (default 1.0) |
| `algorithms/study.py`, `study_plots.py` | rollouts record the delivered acceleration and the effort; summary column; kinematics shows `||v||`, `||u||` |
| `scenarios/slalom/scripts/stress_disturbed.py` | the disturbance modes on the disks (`edge`, an aimed adversary) |
| `scenarios/*/scripts/script_ppo_mpc_reach.py`, `algorithms/mpc_worker.py` | refuse to run / documented as box-only |
| `scenarios/tunnel/scripts/study_*.py` | new: the tunnel's seed studies |
| `algorithms/study_plots.py` | comparisons of studies with different budgets drawn over every whole run, and over the shared budget |

## 12. Reproducing

```
git checkout box-limits-final        # the old dynamics, for any earlier result
git checkout main
python scenarios/slalom/scripts/study_ppo.py --seeds 1-20 --total-timesteps 1024000
python scenarios/slalom/scripts/study_hppo.py --seeds 1-20 --total-timesteps 1024000
python scenarios/slalom/scripts/study_ppo_mpc.py --seeds 1-10
python scenarios/slalom/scripts/study_compare.py
python scenarios/tunnel/scripts/study_ppo.py --seeds 1-20      # and study_hppo, study_ppo_mpc, study_compare
python scenarios/slalom/scripts/compare_disturbed.py --noise-bound-p 0.005 --noise-bound-v 0.05 --budget ppo=1024000,hppo=1024000
python scenarios/slalom/scripts/compare_disturbed.py --noise-bound-p 0.01 --noise-bound-v 0.1 --budget ppo=1024000,hppo=1024000
python scenarios/slalom/scripts/stress_disturbed.py
python scenarios/slalom/scripts/study_init_sampler.py --algos ppo,ppo_mpc --budget ppo=1024000
python scenarios/slalom/scripts/probe_start_variance.py --steps 102400,307200,614400,1024000
# dropped (section 9.6), to rerun should hPPO ever get through the slalom:
python scenarios/slalom/scripts/sweep_extrinsic_coef.py --total-timesteps 1024000
python scenarios/slalom/scripts/study_init_sampler.py --budget ppo=1024000,hppo=1024000
```
