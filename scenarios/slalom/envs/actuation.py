"""What the plant can do: its thrust limit, its speed limit, its disturbance.

The vehicle is a point mass with one thruster that can point anywhere in the
plane, so its limits bound magnitudes, not components:

    ||u|| <= u_max      the acceleration the plant delivers
    ||v|| <= v_max      the speed

and the disturbance is isotropic: w = (w_p, w_v), uniform on the two disks
||w_p|| <= noise_bound_p and ||w_v|| <= noise_bound_v. Everything that models
the plant uses the same sets: the min-time oracle (algorithms/optimal_solver.py,
as second-order cones) and the tube MPC worker (algorithms/tube_mpc.py, whose
tightened sets are again disks, exactly).

Why disks (2026-10-04). Until then every limit was per axis: |u_x|, |u_y| <=
u_max, |v_x|, |v_y| <= v_max and a uniform w in a box (git tag
box-limits-final; every study before it ran on those dynamics). Two problems:

- The vehicle went faster on a diagonal: up to sqrt(2) v_max = 1.70 m/s, with
  sqrt(2) u_max of thrust. Nothing physical about the vehicle picks out the
  corridor's axes.
- The axes were decoupled, so steering was free: a lateral manoeuvre never
  took anything away from v_x. On the slalom's 25-start solved-check grid
  (optimal_solver.spawn_grid, deterministic environment) every min-time
  oracle trajectory arrived on exactly the step a straight run along x with
  no walls arrives on, 78.0 steps on average, threading both gates at no cost
  in time while flying at up to 1.47 m/s, 22% over v_max. With the disks,
  threading the gates costs time, 79.2 steps on average (0 to +4 per start;
  the most from p_y0 = -1, the start furthest from the first gate's
  opening). The tunnel's oracle runs straight and is unchanged, 78.0 under
  both.

The box disturbance had to follow the speed and thrust limits. The tube MPC
tightens its constraints by the error set Z, and a disk minus a box is the
intersection of four disks centred on the box's corners, smaller than the
disk the worst case would need along any one axis. At twice the canonical
disturbance that cut the nominal plan's input radius from 0.90 to 0.31 m/s^2,
so its terminal "stop within N steps" constraint would have capped the
planned speed near 0.3 m/s. An isotropic w makes Z invariant under rotations,
so every tightened set is a disk again, with exactly the radii the box gave
along the axes (algorithms/tube_mpc.py). It is also the natural disturbance
for this vehicle: a push of bounded strength from any direction.

The speed limit acts on the delivered acceleration, not on the state after
the step (2026-09-24). The command u is first saturated radially, keeping its
direction; then, if the velocity it would give, v + u dt, lies outside the
speed disk, it is projected onto it and u is replaced by what reaches exactly
that velocity, (v' - v) / dt. The position then integrates over the velocity
the vehicle really had, so a step at top speed covers v_max * dt and no more.
The projection onto a convex set holding v is non-expansive, so the delivered
u is never longer than the saturated command: every step the environment
takes is a step the oracle's and the MPC's model can take, with an admissible
input. (A state outside the speed disk, reachable only by setting env.state
by hand, is pulled back onto it in one step, and only then can the delivered
u exceed u_max.)

Until 2026-09-24 the speed limit was only a clip of the state after the full
A @ s + B @ u update. The position had then already taken u's u * dt**2 / 2
term, so thrusting while at v_max covered v_max * dt + u_max * dt**2 / 2 =
0.1325 m per step instead of 0.12 (+10%), for a velocity that was then
clipped back to v_max. Trained agents found it: in the seed studies
(algorithms/study.py, 13 seeds each, 163 840 steps) every flat-PPO seed
reached the goal 5-6 steps before the min-time oracle from every one of the
25 grid starts, and every hPPO seed 3-5 steps before it, by holding a_x near
u_max at top speed. The oracle treats the speed limit as a hard constraint on
the state, so it could not do the same, and it was not an upper bound on the
environment as implemented; see docs/benchmark.md's "Is the oracle actually
optimal?".

Every function works on (..., 2) float32 arrays and is purely elementwise
(the norm is written out rather than taken with np.linalg.norm), so SlalomEnv
(one row) and SlalomVecEnv (a batch) compute the same bits.

scenarios/tunnel/envs/actuation.py is the same module for the tunnel.
"""

import numpy as np


def disk_norm(x):
    """||x|| over the last axis of a (..., 2) array."""
    return np.sqrt(x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1])


def project_disk(x, radius):
    """x scaled radially onto the disk ||x|| <= radius where it lies outside
    it, and left exactly as it is inside it. (..., 2) float32.

    The scale radius / max(||x||, radius) is exactly 1 inside the disk, so no
    branch is needed and nothing is divided by zero."""
    r = np.float32(radius)
    scale = r / np.maximum(disk_norm(x), r)
    return x * scale[..., None]


def deliver(u, v, u_max, v_max, dt):
    """The acceleration the plant delivers for the command u at velocity v:
    u saturated radially to ||u|| <= u_max, then, where v + u dt would leave
    the speed disk, replaced by the u that lands exactly on its edge (see the
    module docstring). A step that respects both limits keeps u bit for bit.
    (..., 2) float32."""
    u = project_disk(np.asarray(u, dtype=np.float32), u_max)
    dt = np.float32(dt)
    v_next = v + u * dt
    capped = disk_norm(v_next) > np.float32(v_max)
    if not np.any(capped):
        return u
    return np.where(capped[..., None], (project_disk(v_next, v_max) - v) / dt, u)


def sample_disturbance(rng, bound_p, bound_v, size):
    """`size` independent draws of w = [w_px, w_py, w_vx, w_vy], w_p uniform
    on the disk ||w_p|| <= bound_p and w_v on ||w_v|| <= bound_v, independent
    of each other. (size, 4) float32.

    A uniform point on a disk of radius b is b * sqrt(U) * (cos 2 pi U',
    sin 2 pi U'): the square root makes the density uniform in area rather
    than in radius. Four uniforms per draw, from one call to `rng`."""
    r = rng.uniform(size=(size, 4))
    rad_p = bound_p * np.sqrt(r[:, 0])
    rad_v = bound_v * np.sqrt(r[:, 2])
    ang_p = 2.0 * np.pi * r[:, 1]
    ang_v = 2.0 * np.pi * r[:, 3]
    w = np.stack([rad_p * np.cos(ang_p), rad_p * np.sin(ang_p),
                  rad_v * np.cos(ang_v), rad_v * np.sin(ang_v)], axis=1)
    return w.astype(np.float32)
