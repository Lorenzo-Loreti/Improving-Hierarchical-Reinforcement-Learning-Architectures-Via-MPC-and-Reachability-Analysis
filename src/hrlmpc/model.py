"""Linear model of the point-mass navigation problem.

The agent is a point of unit mass in the plane with isotropic viscous friction
``gamma``. Its state is ``x = (p, v)`` in R^4, with position ``p`` and
velocity ``v``; its input is the commanded acceleration ``u`` in R^2, and the
bounded disturbance ``d`` enters like the input. With the semi-implicit Euler
discretization of step ``dt`` (decision log D4),

    v+ = (1 - gamma dt) v + dt (u + d),        p+ = p + dt v+,

that is ``x+ = A x + B u + B d`` with

    A = [[I2, dt (1 - gamma dt) I2],      B = [[dt^2 I2],
         [0,       (1 - gamma dt) I2]],        [dt   I2]].

The model is exact as long as the physical correction of the environment
(speed limiter and wall reaction) is inactive. The input set U, the velocity
set V and the disturbance set D are Euclidean balls of radii ``a_max``,
``v_max`` and ``d_bar`` (decision log A5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PhysicsParams:
    """Physical parameters of the plant (decision log D9).

    Attributes:
        dt: Sampling time ``T_s`` [s].
        a_max: Radius of the input set ``U`` [m/s^2].
        v_max: Radius of the velocity set ``V`` [m/s].
        gamma: Viscous friction coefficient [1/s].
        gamma_w: Additional viscous coefficient acting on the sliding velocity
            at a wall [1/s] (decision log A3).
        d_bar: Radius of the disturbance set ``D`` [m/s^2].

    Raises:
        ValueError: If the parameters violate the specification; see
            :func:`check_physics`.
    """

    dt: float
    a_max: float
    v_max: float
    gamma: float
    gamma_w: float
    d_bar: float

    def __post_init__(self) -> None:
        check_physics(self)

    @property
    def velocity_factor(self) -> float:
        """Per-step velocity factor ``1 - gamma dt`` of the free motion."""
        return 1.0 - self.gamma * self.dt

    @property
    def invariant_speed(self) -> float:
        """Radius ``a_max / gamma`` of the speed disk that is invariant for every input in U (F1)."""
        return self.a_max / self.gamma

    @property
    def max_robust_disturbance(self) -> float:
        """Largest ``d_bar`` for which V is robustly control invariant (F2)."""
        return min(self.a_max + self.gamma * self.v_max, self.v_max / self.dt)


def check_physics(params: PhysicsParams) -> None:
    """Check that the parameters satisfy the specification.

    The conditions are: positive ``dt``, ``a_max`` and ``v_max``;
    ``0 < gamma dt < 1`` (D4); ``0 <= gamma_w dt < 1`` (A3); ``d_bar >= 0``;
    ``v_max < a_max / gamma``, without which the speed limit is redundant in
    the absence of disturbance (F1); and
    ``d_bar <= min(a_max + gamma v_max, v_max / dt)``, so that V is robustly
    control invariant (F2), which decision D9 requires at every disturbance
    level.

    Args:
        params: The parameters to check.

    Raises:
        ValueError: If any condition is violated.
    """
    if params.dt <= 0.0 or params.a_max <= 0.0 or params.v_max <= 0.0:
        raise ValueError(
            f"dt, a_max and v_max must be positive, got dt={params.dt}, "
            f"a_max={params.a_max}, v_max={params.v_max}"
        )
    if not 0.0 < params.gamma * params.dt < 1.0:
        raise ValueError(f"need 0 < gamma*dt < 1, got gamma*dt={params.gamma * params.dt}")
    if not 0.0 <= params.gamma_w * params.dt < 1.0:
        raise ValueError(f"need 0 <= gamma_w*dt < 1, got gamma_w*dt={params.gamma_w * params.dt}")
    if params.d_bar < 0.0:
        raise ValueError(f"d_bar must be non-negative, got {params.d_bar}")
    if not params.v_max < params.a_max / params.gamma:
        raise ValueError(
            f"F1: need v_max < a_max/gamma = {params.a_max / params.gamma}, got "
            f"v_max={params.v_max}; otherwise the speed limit is redundant"
        )
    bound = min(params.a_max + params.gamma * params.v_max, params.v_max / params.dt)
    if params.d_bar > bound:
        raise ValueError(
            f"F2: need d_bar <= min(a_max + gamma*v_max, v_max/dt) = {bound}, got "
            f"d_bar={params.d_bar}; otherwise V is not robustly control invariant"
        )


def system_matrices(dt: float, gamma: float) -> tuple[FloatArray, FloatArray]:
    """Return the matrices of the semi-implicit Euler discretization.

    Args:
        dt: Sampling time ``T_s`` [s].
        gamma: Viscous friction coefficient [1/s].

    Returns:
        ``(A, B)`` with shapes ``(4, 4)`` and ``(4, 2)`` for the state
        ordering ``x = (p_x, p_y, v_x, v_y)``.
    """
    a = 1.0 - gamma * dt
    i2 = np.eye(2)
    z2 = np.zeros((2, 2))
    A = np.block([[i2, dt * a * i2], [z2, a * i2]])
    B = np.vstack([dt**2 * i2, dt * i2])
    return A, B


def free_step(
    x: ArrayLike, u: ArrayLike, d: ArrayLike, dt: float, gamma: float
) -> FloatArray:
    """One step of the free motion, without the physical correction.

    This is the recursion ``v+ = (1 - gamma dt) v + dt (u + d)``,
    ``p+ = p + dt v+``, evaluated row-wise on batches.

    Args:
        x: States ``(p_x, p_y, v_x, v_y)``, shape ``(..., 4)``.
        u: Inputs, shape ``(..., 2)``.
        d: Disturbances, shape ``(..., 2)``.
        dt: Sampling time [s].
        gamma: Viscous friction coefficient [1/s].

    Returns:
        The successor states, shape ``(..., 4)``.
    """
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    v_next = (1.0 - gamma * dt) * x[..., 2:] + dt * (u + d)
    p_next = x[..., :2] + dt * v_next
    return np.concatenate([p_next, v_next], axis=-1)


def position_gains(dt: float, gamma: float, horizon: int) -> FloatArray:
    """Scalars ``m_i`` such that the position block of ``A^(H-1-i) B`` is ``m_i I2``.

    By isotropy every block of ``A^j B`` is a scalar multiple of ``I2``
    (fact F6). After ``H`` steps the position is therefore the free response
    plus ``sum_i m_i u_i``, so without state constraints the positions
    reachable with ``u_i`` in U form a disk of radius ``a_max sum_i |m_i|``.

    Args:
        dt: Sampling time [s].
        gamma: Viscous friction coefficient [1/s].
        horizon: Number of steps ``H >= 1``.

    Returns:
        The gains ``m_0, ..., m_(H-1)``, shape ``(H,)``.

    Raises:
        ValueError: If ``horizon < 1``.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    A, B = system_matrices(dt, gamma)
    gains = np.empty(horizon)
    block = B.copy()
    for i in range(horizon - 1, -1, -1):
        gains[i] = block[0, 0]
        block = A @ block
    return gains
