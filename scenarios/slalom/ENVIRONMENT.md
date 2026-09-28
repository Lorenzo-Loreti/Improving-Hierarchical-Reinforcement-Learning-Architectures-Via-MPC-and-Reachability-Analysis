# Slalom Environment

## Purpose

Slalom is the harder navigation scenario in this thesis, built on the exact
same physics and reward machinery as the [Tunnel environment](../tunnel/ENVIRONMENT.md),
but with one deliberate twist: the corridor narrows into two bottleneck
"gates" positioned on opposite sides of the centerline, forcing the agent
into a genuine S-shaped detour instead of a straight run. This is the
scenario this thesis actually uses to compare a flat policy against a
hierarchical one, since it is the first case where there is a real,
multi-phase spatial decision to make, rather than just a control-efficiency
problem.

## Physical Setup

As in Tunnel, the agent is a point mass moving in a two-dimensional plane,
controlled by choosing an acceleration at every step rather than steering
directly. Its position and velocity evolve through a double-integrator
model — the agent has inertia, cannot stop or turn on the spot, and has to
plan several steps ahead. Time advances in fixed 0.1-second increments.

The difference from Tunnel is entirely in the shape of the corridor. For
most of its length the corridor is the same constant 4-meter width. But at
two fixed points along the way, the usable width shrinks to 1.5 meters, and
each of these narrow "gates" is shifted a full meter off the centerline — one
gate toward one side, the other toward the opposite side. The corridor
returns to full width, and back to being centered, in between the two gates
and again after the second one, before the goal line. Just like in Tunnel,
the environment supports adding small, bounded random disturbances to
position and velocity (uniform within a fixed box, since 2026-09-28), but
the deterministic, noise-free version is the default and what every
experiment before that date ran on.

## Objective and Episode Structure

Episodes start and end exactly as in Tunnel: the agent spawns near the
entrance at a randomized position (with zero velocity) and must reach the
far end of the corridor within a fixed budget of 200 control steps (20
simulated seconds). Reaching the goal ends the episode successfully;
otherwise it is cut off at the timeout. Wall contact — including contact
with a gate's narrower walls — is never terminal: it is a soft, inelastic
bounce (the agent's lateral push into the wall is absorbed, its forward
motion is untouched) with a penalty, and the episode continues.

## What Makes Slalom Distinct

In Tunnel, the centerline is always a valid path. In Slalom it is not: the
straight-down-the-middle line is blocked by both gates, so the agent must
commit to moving toward one side to line up with the first gate, pass
through it, then reverse that lateral commitment to line up with the second,
oppositely-offset gate, before finally recentering for the run to the goal.
Because the agent's velocity carries momentum, this cannot be done at the
last second — the lateral maneuver into each gate has to begin while the
agent is still well before it.

This is precisely what makes Slalom relevant to the flat-vs-hierarchical
question this thesis investigates. A flat policy has to encode this entire
two-phase detour implicitly, as a single continuous reaction to the
observed state. A hierarchical policy, in principle, can instead recognize
the task as a short sequence of sub-goals — "line up with gate 1," then
"line up with gate 2," then "run to the finish" — and delegate control of
each phase separately. Whether that structure is actually learned, or
whether the hierarchy instead collapses onto a single behavior, is the
central empirical question Slalom is used to probe.

## Reward Design

The reward is built from the same four ingredients as Tunnel — a small
per-step time penalty, a wall-contact penalty, a large one-off success bonus,
and a small continuous "forward progress" bonus paid every step — with the
same historical motivation: an earlier version made wall contact end the
episode outright, which taught the agent that idling after partial progress
was safer, in expectation, than ever attempting a gate, so it never did.
Making contact survivable is what makes a real attempt worth the risk.

Two details are specific to Slalom. First, the success bonus is five times
larger than in Tunnel (see the parameter table below), reflecting that this
is deliberately the harder task. Second, the "forward progress" bonus is
defined purely in terms of distance covered along the corridor's length,
never in terms of straight-line distance to the goal — which matters here
specifically because the two gates are off to the side: a shaping signal
based on straight-line distance to the goal would actively fight the agent
every time it had to move laterally away from center to line up with a gate.

## Key Parameters

| Quantity | Value | Meaning |
|---|---|---|
| Corridor length | 10 m | forward distance from start to goal |
| Corridor width (open sections) | 4 m | width everywhere except at the two gates |
| Gate opening width | 1.5 m | width of the corridor at each gate |
| Gate lateral offset | ±1 m from centerline | the two gates sit on opposite sides |
| Gate 1 location | 4 m to 5 m along the corridor | offset toward one side |
| Gate 2 location | 7 m to 8 m along the corridor | offset toward the opposite side |
| Max speed | 1.2 m/s | per-axis velocity bound; at it, more acceleration that way has no effect |
| Max acceleration | 2.5 m/s² | the agent's control authority |
| Control step | 0.1 s | simulated time between decisions |
| Episode budget | 200 steps (20 s) | timeout if the goal isn't reached in time |
| Spawn zone | first 2 m, centered laterally | randomized starting box |
| Success bonus | 1000 | vs. 200 in Tunnel |

## Layout Sketch (schematic, not to scale)

```
y=+2.0 #####################
y=+1.5 ..............##.....
y=+1.0 ..............##.....
y=+0.5 ..............##.....
y= 0.0 SS......##....##....G
y=-0.5 ........##...........
y=-1.0 ........##...........
y=-1.5 ........##...........
y=-2.0 #####################
        (x runs left to right, 0 to 10 m)
```

Reading left to right: the agent (S) spawns centered near the left wall in a
full-width entry section; the first block of `#` around one-third of the way
across is gate 1, open only in the upper band — the agent must be shifted
up to pass through it; the corridor then reopens to full width for a short
recovery stretch; the second block of `#` is gate 2, open only in the lower
band — the mirror image of gate 1; and the corridor reopens to full width
again for the final approach to the goal (G) on the right wall.

Key x-positions: 0 m start of spawn zone · 2 m end of spawn zone · 4 m gate 1
begins · 5 m gate 1 ends · 7 m gate 2 begins · 8 m gate 2 ends · 10 m goal
line.

## Role in the Thesis

Slalom is the primary scenario used to compare flat and hierarchical
policies in this thesis: its two oppositely-offset gates are what turn
"reach the far end" into a task with real, sequential sub-goals, which is
exactly the structure a hierarchical policy is meant to exploit and a flat
policy has to approximate without any explicit help.

| Aspect | Tunnel | Slalom |
|---|---|---|
| Corridor width | constant 4 m | 4 m, narrowing to 1.5 m at two points |
| Lateral maneuvering required | none (centerline is always valid) | yes — an S-shaped detour through two offset gates |
| Success bonus | 200 | 1000 (reflecting the harder task) |
| Role in this thesis | baseline / sanity check | primary testbed for flat-vs-hierarchical comparisons |
