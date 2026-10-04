"""What the plant can do: its thrust limit, its speed limit, its disturbance.

||u|| <= u_max, ||v|| <= v_max, and an isotropic disturbance, uniform on the
disks ||w_p|| <= noise_bound_p and ||w_v|| <= noise_bound_v (since
2026-10-04; per-axis limits and a box disturbance before, git tag
box-limits-final). The speed limit acts on the delivered acceleration: the
command is saturated radially, then cut to what lands v + u dt on the speed
disk's edge, so a step at top speed covers v_max * dt and no more.

The same module as scenarios/slalom/envs/actuation.py, whose docstring has the
full rationale: why disks (on the box the vehicle flew sqrt(2) faster on a
diagonal and steering cost nothing), why the disturbance followed (the tube
MPC's tightened sets stay disks only under an isotropic W), why the speed
limit acts on u (the 2026-09-24 defect that let agents beat the oracle) and
the measured oracle step counts. On the tunnel the oracle runs straight, so
its step counts are the same under the box and the disks (78.0 on the
25-start grid).

Every function is purely elementwise on (..., 2) float32 arrays, so TunnelEnv
and TunnelVecEnv compute the same bits.
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
