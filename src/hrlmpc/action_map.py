"""Map from the policy's action to the input set ``U`` (decision log D15, row AM1).

The policy of flat PPO and of the hPPO Worker samples ``x`` in the square
``[-1, 1]^2``, one Beta distribution per axis. The bijection

    u = a_max (||x||_inf / ||x||_2) x        (u = 0 for x = 0)

maps the square onto the disk ``U = {u : ||u||_2 <= a_max}``. It keeps the
direction of ``x`` and sends the square of "radius" ``||x||_inf = r`` onto the
circle of radius ``a_max r``, so no two actions give the same input and the
physical corrections of the commanded action come only from the speed limiter
and the walls.

Distortion. Away from the diagonals the map is smooth, with Jacobian
determinant ``a_max^2 cos^2(theta)``, where ``theta`` is the angle of ``x``
from the nearest axis (``|theta| <= 45`` degrees). Areas, and densities,
therefore change by a factor between ``a_max^2 / 2`` and ``a_max^2``. The
log-probability of an action is taken on ``x`` (D15).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def square_to_disk(x: ArrayLike, a_max: float) -> FloatArray:
    """Map actions ``x`` of shape ``(..., 2)`` in the square onto the disk of radius ``a_max``."""
    x_arr = np.asarray(x, dtype=np.float64)
    inf_norm = np.max(np.abs(x_arr), axis=-1, keepdims=True)
    two_norm = np.hypot(x_arr[..., :1], x_arr[..., 1:])
    scale = np.divide(inf_norm, two_norm, out=np.zeros_like(inf_norm), where=two_norm > 0.0)
    result: FloatArray = a_max * scale * x_arr
    return result


def disk_to_square(u: ArrayLike, a_max: float) -> FloatArray:
    """Inverse of :func:`square_to_disk`: ``x = (||u||_2 / ||u||_inf) u / a_max``."""
    u_arr = np.asarray(u, dtype=np.float64)
    inf_norm = np.max(np.abs(u_arr), axis=-1, keepdims=True)
    two_norm = np.hypot(u_arr[..., :1], u_arr[..., 1:])
    scale = np.divide(two_norm, inf_norm, out=np.zeros_like(inf_norm), where=inf_norm > 0.0)
    result: FloatArray = scale * u_arr / a_max
    return result
