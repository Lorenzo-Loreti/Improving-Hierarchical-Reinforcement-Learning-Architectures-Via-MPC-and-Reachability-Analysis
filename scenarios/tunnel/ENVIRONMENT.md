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
few steps ahead to avoid drifting into a wall. Its speed and its
acceleration are capped in magnitude, whatever the direction: it behaves
like a vehicle with a single thruster that can point anywhere, so moving
diagonally is no faster than moving straight, and steering at top speed
turns the velocity instead of adding to it. Once the agent is at top speed,
pushing harder in the direction it is going has no effect, on its velocity
or on how far it travels in the step. (Until 2026-10-04 both caps applied
to each axis separately, which let the agent move about 40% faster on a
diagonal.)

Time advances in fixed increments of 0.1 seconds (so ten control decisions
per simulated second). The corridor runs along one axis (call it the
"forward" direction) and has two parallel walls a fixed distance apart along
the other ("lateral") axis. The environment also supports adding small,
bounded random disturbances to position and velocity at every step
(since 2026-09-28; uniform over a disk for the position and another for the
velocity, the same in every direction, since 2026-10-04, and within a fixed
box before), to model imperfect
actuation or measurement — but the deterministic, noise-free version is the
default and what every experiment before that date ran on, so there the
dynamics are exact.

## Objective and Episode Structure

Each episode starts with the agent placed near the entrance of the corridor,
at a randomized position inside a small box (so the agent never sees exactly
the same starting point twice), with zero initial velocity. From there, the
agent must reach the far end of the corridor.

Every starting point is drawn uniformly from that box. During training the
starting points of successive episodes are, by default, drawn independently
of each other. An option (since 2026-10-03) spreads them evenly over the box
instead, with a scrambled Sobol' sequence, so that a few consecutive episodes
never all start in the same corner; each starting point is still uniform on
the box, so the task itself is unchanged. On the slalom this made no
measurable difference to learning, so it stays off by default; see
[`docs/init-sampler.md`](../../docs/init-sampler.md).

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
corridor. The agent is simply penalized for the contact and continues. (A
corridor with narrower sections, like Slalom's gates, also has walls across
it, the sections' front faces; hitting one stops the agent's forward motion
instead, see the Slalom description. The tunnel's walls are all along it.)

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

The reward given to the agent at every step is built from five pieces:

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
  purely on the sparse, one-off success bonus. It is paid only up to the
  goal line: how far past it the last step lands earns nothing (since
  2026-10-04), so a successful episode is scored only by how long it took,
  how hard the agent pushed and how often it touched a wall.
- **A small control-effort penalty**, in proportion to the squared magnitude
  of the acceleration the vehicle actually delivers (since 2026-10-04). A
  step at full thrust costs a hundredth of a time step, so over a whole
  episode the effort is worth less than a single step: the agent still goes
  as fast as it can, and among the equally fast ways of doing so it prefers
  the smoothest. It is the same quantity the reference solution (the
  minimum-time oracle) minimizes, so the agent and the oracle are scored on
  exactly the same objective.

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
| Max speed | 1.2 m/s | bound on the speed, in any direction (per axis until 2026-10-04) |
| Max acceleration | 2.5 m/s² | bound on the magnitude of the acceleration, the agent's control authority |
| Control step | 0.1 s | simulated time between decisions |
| Episode budget | 200 steps (20 s) | timeout if the goal isn't reached in time |
| Spawn zone | first 2 m, centered laterally | randomized starting box |
| Success bonus | 1000 | the same as Slalom's since 2026-09-24 (200 before) |
| Effort penalty | 0.01 per full-thrust step | proportional to the squared magnitude of the delivered acceleration (since 2026-10-04) |

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
| Success bonus | 1000 (200 until 2026-09-24) | 1000 |
| Role in this thesis | baseline / sanity check | primary testbed for flat-vs-hierarchical comparisons |
