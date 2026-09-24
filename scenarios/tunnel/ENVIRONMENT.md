# Tunnel Environment

## Purpose

Tunnel is the baseline navigation scenario in this thesis: a single agent must
cross a straight corridor of constant width, from a fixed side to the
opposite one, as quickly as possible without hitting the walls. It exists to
validate that a control policy can handle the shared dynamics and reward
machinery used across every scenario in this codebase, before that policy is
asked to also solve a genuinely harder spatial problem (see the companion
[Slalom environment](../slalom/ENVIRONMENT.md)).

## Physical Setup

The agent is a point mass moving in a two-dimensional plane. It does not
steer directly; instead, at every control step it chooses an acceleration
(in two directions), and the environment integrates that acceleration into a
velocity, and the velocity into a position. This is a *double integrator*:
position depends on velocity, velocity depends on the chosen acceleration,
and neither can change instantaneously. Concretely, this means the agent has
inertia — it cannot stop or change direction on the spot, and has to plan a
few steps ahead to avoid drifting into a wall. Speed along each axis is
capped: once the agent is at top speed in a direction, pushing harder that
way has no effect, on its velocity or on how far it travels in the step.

Time advances in fixed increments of 0.1 seconds (so ten control decisions
per simulated second). The corridor runs along one axis (call it the
"forward" direction) and has two parallel walls a fixed distance apart along
the other ("lateral") axis. The environment also supports adding small
random disturbances to position and velocity at every step, to model
imperfect actuation or measurement — but every experiment in this codebase
currently runs the deterministic, noise-free version, so in practice the
dynamics are exact.

## Objective and Episode Structure

Each episode starts with the agent placed near the entrance of the corridor,
at a randomized position inside a small box (so the agent never sees exactly
the same starting point twice), with zero initial velocity. From there, the
agent must reach the far end of the corridor.

An episode ends in one of two ways:

- **Success**: the agent's forward position reaches the far end of the
  corridor. This ends the episode immediately with a large reward.
- **Timeout**: the agent has not reached the far end after a fixed budget of
  200 control steps (20 simulated seconds). The episode is cut off without
  the success bonus.

Touching a wall does **not** end the episode. Physically, a wall contact is
treated as a fully inelastic bounce: the agent's lateral position is clamped
back to the wall, its lateral velocity is zeroed (it stops pushing into the
wall), but its forward velocity is untouched, so it keeps moving down the
corridor. The agent is simply penalized for the contact and continues.

## What Makes Tunnel Distinct

Because the corridor is a fixed, constant width along its entire length,
Tunnel does not require any lateral maneuvering strategy — the straightforward
path (down the centerline) is always valid. The actual difficulty comes
entirely from the dynamics: since the agent has inertia and a bounded
acceleration, it must brake and correct in advance rather than react at the
last moment, and any inefficient use of the acceleration budget shows up
directly as lost time. Tunnel is therefore best read as a scenario that
isolates "can the policy control the vehicle efficiently and safely" from
"can the policy plan a genuinely non-trivial path," which is the added
ingredient in Slalom.

## Reward Design

The reward given to the agent at every step is built from four pieces:

- **A small time penalty** on every step, which pushes the agent to reach
  the goal as quickly as possible rather than dawdling.
- **A wall-contact penalty**, charged on every step the agent is in contact
  with a wall. This makes hugging or grazing the wall costly without making
  a single touch catastrophic.
- **A large one-off success bonus**, paid only when the agent reaches the
  far end of the corridor.
- **A small "progress" bonus**, paid every step in proportion to how much
  forward progress was made that step (and only forward progress — lateral
  movement does not count). This gives the agent a continuous signal to
  follow even long before it is anywhere near the goal, instead of relying
  purely on the sparse, one-off success bonus.

An earlier version of this reward made wall contact *terminal* (i.e., it
ended the episode immediately) with a large penalty. That design backfired:
a training run that had already made some progress found that a full stop
followed by doing nothing for the rest of the episode was a safer bet, in
expectation, than risking the terminal penalty by attempting the trickier
parts of a route — so training consistently converged to policies that
advanced a little and then simply idled, never attempting a real crossing,
regardless of how much training budget was given. Making wall contact a
survivable, non-terminal event (as described above) removed that incentive
and is what makes an honest attempt at reaching the goal worth the risk.

## Key Parameters

| Quantity | Value | Meaning |
|---|---|---|
| Corridor length | 10 m | forward distance from start to goal |
| Corridor width | 4 m | fixed lateral distance between the two walls |
| Max speed | 1.2 m/s | per-axis velocity bound; at it, more acceleration that way has no effect |
| Max acceleration | 2.5 m/s² | the agent's control authority |
| Control step | 0.1 s | simulated time between decisions |
| Episode budget | 200 steps (20 s) | timeout if the goal isn't reached in time |
| Spawn zone | first 2 m, centered laterally | randomized starting box |

## Layout Sketch (schematic, not to scale)

```
y=+2.0 #####################
y=+1.5 SS................G
y=+1.0 ..................
y=+0.5 ..................
y= 0.0 ..................
y=-0.5 ..................
y=-1.0 ..................
y=-1.5 ..................
y=-2.0 #####################
        x=0              x=10
```

A straight, uniformly wide corridor: the agent (S, spawn zone) starts near
the left wall and must reach the goal line (G) on the right, staying between
the two constant walls the whole way.

## Role in the Thesis

Tunnel is used as the sanity-check / calibration scenario: it is where an
algorithm's basic ability to control the double-integrator dynamics and make
use of the shared reward signal is verified, and where hyperparameters can be
tuned cheaply, before moving on to the strictly harder Slalom scenario that
this thesis actually uses to compare flat and hierarchical policies.

| Aspect | Tunnel | Slalom |
|---|---|---|
| Corridor width | constant 4 m | 4 m, narrowing to 1.5 m at two points |
| Lateral maneuvering required | none (centerline is always valid) | yes — an S-shaped detour through two offset gates |
| Success bonus | 200 | 1000 (reflecting the harder task) |
| Role in this thesis | baseline / sanity check | primary testbed for flat-vs-hierarchical comparisons |
