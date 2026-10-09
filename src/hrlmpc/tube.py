"""The tube of the MPC Worker: the error set ``Z`` for ``W = BD`` (decision log D31, D33).

The tube MPC of PPO_MPC plans a nominal trajectory ``z`` and applies
``u = v + K (x - z)`` at the real state ``x``. The error ``e = x - z`` then
evolves as

    e+ = A_K e + B d,        A_K = A + B K,        ||d||_2 <= d_bar,

with ``A`` and ``B`` the matrices of :func:`hrlmpc.model.system_matrices`. A
set ``Z`` is robustly positively invariant (RPI) if ``A_K Z (+) BD`` lies in
``Z``: an error that starts in ``Z`` stays there for every disturbance.

``W = BD`` is a disk in a plane of R^4, so the algorithm of the tube-MPC note
(which needs ``0`` inside ``W``) does not apply. D31 takes instead

    Z = (+)_{l < s} A_K^l B D  (+)  eps E,       E = {e : e' P_E e <= 1},

with ``A_K' P_E A_K <= lambda^2 P_E`` for some ``lambda < 1`` and
``eps >= d_bar ||P_E^(1/2) A_K^s B||_2 / (1 - lambda)``. Then

    A_K Z (+) BD = (+)_{l < s} A_K^l B D  (+)  A_K^s B D  (+)  eps A_K E
                 in (+)_{l < s} A_K^l B D  (+)  (mu + eps lambda) E,

where ``mu = d_bar ||P_E^(1/2) A_K^s B||_2`` bounds ``A_K^s B D`` in the norm of
``E``; since ``mu + eps lambda <= eps``, ``Z`` is RPI. Every closed RPI set
contains the minimal one, and ``Z`` lies within ``eps E`` of it.

``Z`` is kept as its support function and its lifted representation,

    e in Z  <=>  e = sum_l (A_K^l B) d_l + T eta,  ||d_l||_2 <= d_bar, ||eta||_2 <= 1,

with ``T = eps P_E^(-1/2)``, never as facets. With isotropic weights every
``A_K^l B`` and ``P_E`` are scalar multiples of ``I2`` per block, so ``Z`` is
invariant under the rotations of the plane: its projections on the positions,
the velocities and the inputs ``K e`` are disks, and the tightened speed and
thrust sets of D33 are exact disks.

This module needs NumPy and SciPy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

TAIL_TOLERANCE = 1e-3
"""Default bound on the support of the tail ``eps E`` along every coordinate axis [m or m/s]."""

MAX_TERMS = 1000
"""Largest number of terms ``s`` the construction may use before giving up."""

_ROTATION_ANGLES = (0.3, 1.1, 2.5)


@dataclass(frozen=True)
class Weights:
    """Stage weights of the MPC cost, ``Q = diag(q_pos, q_pos, q_vel, q_vel)``, ``R = r I2`` (D34).

    They also give the ancillary gain ``K`` and the terminal cost ``P`` (LQR).

    Raises:
        ValueError: If a weight is not positive and finite.
    """

    q_pos: float
    q_vel: float
    r: float

    def __post_init__(self) -> None:
        for name in ("q_pos", "q_vel", "r"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite, got {value!r}")

    @property
    def Q(self) -> FloatArray:
        """The state weight ``Q`` (4x4)."""
        return np.diag([self.q_pos, self.q_pos, self.q_vel, self.q_vel]).astype(np.float64)

    @property
    def R(self) -> FloatArray:
        """The input weight ``R`` (2x2)."""
        return self.r * np.eye(2)


def lqr(A: FloatArray, B: FloatArray, Q: FloatArray, R: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Discrete-time LQR with the convention ``u = K x``.

    Returns:
        ``(K, P)``: the gain, shape ``(2, 4)``, and the cost matrix, the
        stabilizing solution of the discrete algebraic Riccati equation.
    """
    P = sla.solve_discrete_are(A, B, Q, R)
    P = 0.5 * (P + P.T)
    K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def spectral_radius(M: FloatArray) -> float:
    """The largest modulus of the eigenvalues of ``M``."""
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def rotation(theta: float) -> FloatArray:
    """``diag(R_theta, R_theta)``: the rotation of the plane by ``theta`` acting on ``(p, v)``."""
    c, s = np.cos(theta), np.sin(theta)
    return np.kron(np.eye(2), np.array([[c, -s], [s, c]]))


def is_rotation_invariant(M: FloatArray, atol: float = 1e-10) -> bool:
    """Whether ``M`` (4x4) commutes with the rotations of the plane acting on ``(p, v)``."""
    return all(np.allclose(M @ rotation(t), rotation(t) @ M, atol=atol) for t in _ROTATION_ANGLES)


@dataclass(frozen=True, eq=False)
class Tube:
    """The error set ``Z = (+)_{l<s} A_K^l B D (+) T B_1`` of D31, with ``B_1`` the unit ball of R^4.

    Build it with :func:`design_tube`. With ``d_bar = 0`` it is ``{0}``: no
    terms and a zero tail.

    Attributes:
        gain: The ancillary gain ``K``, shape ``(2, 4)``.
        closed_loop: ``A_K = A + B K``, shape ``(4, 4)``.
        terms: ``A_K^l B`` for ``l = 0, ..., s - 1``, shape ``(s, 4, 2)``.
        d_bar: The radius of ``D`` [m/s^2].
        tail: ``T = eps P_E^(-1/2)``, shape ``(4, 4)``, symmetric; zero for ``d_bar = 0``.
        contraction: ``lambda``, with ``A_K' P_E A_K <= lambda^2 P_E``; ``0`` for ``d_bar = 0``.
    """

    gain: FloatArray
    closed_loop: FloatArray
    terms: FloatArray
    d_bar: float
    tail: FloatArray
    contraction: float

    @property
    def is_zero(self) -> bool:
        """Whether ``Z = {0}`` (no disturbance)."""
        return self.d_bar == 0.0

    @property
    def num_terms(self) -> int:
        """``s``, the number of terms of the Minkowski sum."""
        return int(self.terms.shape[0])

    def support(self, a: ArrayLike) -> FloatArray:
        """The support function ``h_Z(a) = d_bar sum_l ||(A_K^l B)' a|| + ||T a||``.

        Args:
            a: Directions, shape ``(..., 4)``.

        Returns:
            ``h_Z`` of each direction, shape ``(...)``.
        """
        a = np.asarray(a, dtype=np.float64)
        if a.shape[-1] != 4:
            raise ValueError(f"directions must have 4 components, got shape {a.shape}")
        projected = np.einsum("lij,...i->...lj", self.terms, a)  # (..., s, 2)
        result: FloatArray = self.d_bar * np.linalg.norm(projected, axis=-1).sum(axis=-1)
        return result + np.linalg.norm(a @ self.tail, axis=-1)

    def lifted(self, d: ArrayLike, eta: ArrayLike) -> FloatArray:
        """The error ``sum_l (A_K^l B) d_l + T eta`` of the lifted representation.

        Args:
            d: ``(s, 2)`` disturbances, each in ``D`` for the error to lie in ``Z``.
            eta: ``(4,)`` point of the unit ball.
        """
        d_arr = np.asarray(d, dtype=np.float64).reshape(self.num_terms, 2)
        result: FloatArray = np.einsum("lij,lj->i", self.terms, d_arr) + self.tail @ np.asarray(eta, dtype=np.float64)
        return result

    @property
    def position_radius(self) -> float:
        """Radius of the projection of ``Z`` on the positions (a disk) [m]."""
        return float(self.support(np.array([1.0, 0.0, 0.0, 0.0])))

    @property
    def speed_radius(self) -> float:
        """Radius of the projection of ``Z`` on the velocities (a disk) [m/s]."""
        return float(self.support(np.array([0.0, 0.0, 1.0, 0.0])))

    @property
    def input_radius(self) -> float:
        """Radius of ``K Z`` (a disk): the largest correction ``||K e||`` of the ancillary law [m/s^2]."""
        return float(self.support(self.gain.T @ np.array([1.0, 0.0])))

    @property
    def next_speed_radius(self) -> float:
        """Radius of the velocities of ``A_K Z``: the largest ``||(A_K e)_v||`` over ``e`` in ``Z`` [m/s] (D33)."""
        return float(self.support(self.closed_loop.T @ np.array([0.0, 0.0, 1.0, 0.0])))

    @property
    def tail_radius(self) -> float:
        """The largest support of the tail ``T B_1`` along a coordinate axis."""
        return float(np.max(np.linalg.norm(self.tail, axis=0)))


def _sqrt_and_inverse_sqrt(P: FloatArray) -> tuple[FloatArray, FloatArray]:
    values, vectors = np.linalg.eigh(0.5 * (P + P.T))
    if np.any(values <= 0.0):
        raise ValueError("P_E is not positive definite")
    root = (vectors * np.sqrt(values)) @ vectors.T
    inverse = (vectors / np.sqrt(values)) @ vectors.T
    return 0.5 * (root + root.T), 0.5 * (inverse + inverse.T)


def design_tube(
    A: FloatArray,
    B: FloatArray,
    K: FloatArray,
    d_bar: float,
    *,
    tolerance: float = TAIL_TOLERANCE,
    max_terms: int = MAX_TERMS,
) -> Tube:
    """The RPI set ``Z`` of D31 for ``e+ = A_K e + B d``, ``||d|| <= d_bar``.

    ``lambda`` is the midpoint between the spectral radius of ``A_K`` and 1, and
    ``P_E`` solves ``(A_K / lambda)' P_E (A_K / lambda) - P_E + I = 0``, so that
    ``A_K' P_E A_K = lambda^2 (P_E - I) <= lambda^2 P_E``. The number of terms
    ``s`` is the smallest whose tail ``eps E`` has a support of at most
    ``tolerance`` along every coordinate axis.

    Args:
        A: ``(4, 4)`` state matrix.
        B: ``(4, 2)`` input matrix.
        K: ``(2, 4)`` ancillary gain; ``A + B K`` must be Schur.
        d_bar: Radius of the disturbance disk ``D``, ``>= 0``.
        tolerance: Bound on the tail's support along the axes.
        max_terms: Largest ``s`` allowed.

    Raises:
        ValueError: If ``A + B K`` is not Schur, ``d_bar`` is negative or not
            finite, ``tolerance`` is not positive, or ``s`` would exceed ``max_terms``.
    """
    if not (np.isfinite(d_bar) and d_bar >= 0.0):
        raise ValueError(f"d_bar must be non-negative and finite, got {d_bar!r}")
    if not tolerance > 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance!r}")
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    AK = A + B @ K
    radius = spectral_radius(AK)
    if radius >= 1.0:
        raise ValueError(f"A + B K must be Schur, its spectral radius is {radius}")
    if d_bar == 0.0:
        return Tube(K, AK, np.zeros((0, 4, 2)), 0.0, np.zeros((4, 4)), 0.0)
    lam = 0.5 * (1.0 + radius)
    scaled = AK / lam
    P_E = sla.solve_discrete_lyapunov(scaled.T, np.eye(4))
    root, inverse_root = _sqrt_and_inverse_sqrt(P_E)
    axis_scale = np.linalg.norm(inverse_root, axis=0).max()  # support of the unit ball of E along the axes
    terms: list[FloatArray] = []
    power = B.copy()  # A_K^s B
    for _ in range(max_terms + 1):
        eps = d_bar * float(np.linalg.norm(root @ power, 2)) / (1.0 - lam)
        if eps * axis_scale <= tolerance:
            stacked = np.array(terms) if terms else np.zeros((0, 4, 2))
            return Tube(K, AK, stacked, float(d_bar), eps * inverse_root, lam)
        terms.append(power)
        power = AK @ power
    raise ValueError(f"the tail did not fall below {tolerance} within {max_terms} terms")
