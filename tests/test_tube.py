"""Tests of the tube of the MPC Worker: the RPI set Z for W = BD and the tightenings (decision log D31, D33)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hrlmpc.config import load_env_config
from hrlmpc.geometry import Layout
from hrlmpc.model import system_matrices
from hrlmpc.physics import physics_step
from hrlmpc.tube import Tube, Weights, design_tube, is_rotation_invariant, lqr, rotation

DT, GAMMA = 0.1, 0.5
WEIGHTS = Weights(10.0, 1.0, 0.1)


def _tube(d_bar: float, tolerance: float = 1e-3) -> Tube:
    A, B = system_matrices(DT, GAMMA)
    K, _ = lqr(A, B, WEIGHTS.Q, WEIGHTS.R)
    return design_tube(A, B, K, d_bar, tolerance=tolerance)


def _directions(n: int, seed: int = 0) -> np.ndarray:
    a = np.random.default_rng(seed).normal(size=(n, 4))
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _points_of(tube: Tube, n: int, seed: int = 1, extreme: bool = False) -> np.ndarray:
    """Points of Z from its lifted representation; on the boundary of each ball if ``extreme``."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d = rng.normal(size=(tube.num_terms, 2))
        d *= tube.d_bar / np.linalg.norm(d, axis=1, keepdims=True)
        if not extreme:
            d *= rng.uniform(0.0, 1.0, size=(tube.num_terms, 1))
        eta = rng.normal(size=4)
        eta /= np.linalg.norm(eta)
        out.append(tube.lifted(d, eta * (1.0 if extreme else rng.uniform())))
    return np.array(out)


def test_the_gain_is_the_lqr_gain_and_stabilizes() -> None:
    A, B = system_matrices(DT, GAMMA)
    K, P = lqr(A, B, WEIGHTS.Q, WEIGHTS.R)
    riccati = A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(WEIGHTS.R + B.T @ P @ B, B.T @ P @ A) + WEIGHTS.Q
    np.testing.assert_allclose(P, riccati, atol=1e-8)
    assert np.max(np.abs(np.linalg.eigvals(A + B @ K))) < 1.0
    np.testing.assert_allclose(K[0, [0, 2]], [-7.757, -3.784], atol=1e-3)  # the numbers of the decision log
    np.testing.assert_allclose(K[:, [0, 2]][1], 0.0, atol=1e-12)  # one block per axis


@pytest.mark.parametrize("d_bar", [0.25, 0.5, 1.0])
def test_z_is_robustly_positively_invariant(d_bar: float) -> None:
    tube = _tube(d_bar)
    _, B = system_matrices(DT, GAMMA)
    a = _directions(20000)
    image = tube.support(a @ tube.closed_loop) + d_bar * np.linalg.norm(a @ B, axis=1)  # h of A_K Z (+) BD
    assert np.all(image <= tube.support(a) + 1e-12)


@pytest.mark.parametrize("d_bar", [0.5, 1.0])
def test_z_contains_the_minimal_rpi_set_and_stays_within_the_tail_of_it(d_bar: float) -> None:
    tube = _tube(d_bar)
    _, B = system_matrices(DT, GAMMA)
    a = _directions(500, seed=3)
    minimal = np.zeros(len(a))
    power = B.copy()
    for _ in range(400):  # the minimal RPI set's support, summed far beyond s
        minimal += d_bar * np.linalg.norm(a @ power, axis=1)
        power = tube.closed_loop @ power
    support = tube.support(a)
    assert np.all(support >= minimal - 1e-12)
    assert np.all(support - minimal <= np.linalg.norm(a @ tube.tail, axis=1) + 1e-12)
    assert tube.tail_radius <= 1e-3


def test_the_tail_tolerance_sets_the_number_of_terms() -> None:
    loose, tight = _tube(0.5, tolerance=1e-2), _tube(0.5, tolerance=1e-5)
    assert loose.num_terms < tight.num_terms
    assert loose.tail_radius <= 1e-2 and tight.tail_radius <= 1e-5
    assert tight.position_radius <= loose.position_radius


def test_the_lifted_representation_attains_the_support_function() -> None:
    tube = _tube(0.5)
    for a in _directions(50, seed=4):
        d = np.array([tube.d_bar * (t.T @ a) / np.linalg.norm(t.T @ a) for t in tube.terms])
        eta = tube.tail @ a / np.linalg.norm(tube.tail @ a)
        assert a @ tube.lifted(d, eta) == pytest.approx(float(tube.support(a)), rel=1e-12)
    points = _points_of(tube, 2000)
    a = _directions(200, seed=5)
    assert np.all(points @ a.T <= tube.support(a)[None, :] + 1e-12)


def test_z_is_rotation_invariant() -> None:
    tube = _tube(1.0)
    assert is_rotation_invariant(tube.closed_loop)
    a = _directions(100, seed=6)
    for theta in (0.2, 1.3, 2.9):
        np.testing.assert_allclose(tube.support(a @ rotation(theta).T), tube.support(a), rtol=1e-10)


def test_the_radii_of_the_decision_log() -> None:
    half, one = _tube(0.5), _tube(1.0)
    assert half.position_radius == pytest.approx(0.0649, abs=2e-3)
    assert half.speed_radius == pytest.approx(0.1666, abs=2e-3)
    assert half.input_radius == pytest.approx(0.6516, abs=5e-3)
    assert 1.2 - half.next_speed_radius == pytest.approx(1.083, abs=2e-3)
    assert 2.5 - half.input_radius == pytest.approx(1.848, abs=5e-3)
    assert 1.2 - one.next_speed_radius == pytest.approx(0.967, abs=2e-3)
    assert 2.5 - one.input_radius == pytest.approx(1.197, abs=5e-3)
    # D33 gains at least T_s d_bar over the tightening by Z
    for tube in (half, one):
        assert tube.speed_radius - tube.next_speed_radius >= DT * tube.d_bar - 1e-12


def test_no_disturbance_gives_the_zero_tube() -> None:
    tube = _tube(0.0)
    assert tube.is_zero and tube.num_terms == 0
    assert np.all(tube.support(_directions(10)) == 0.0)
    assert tube.position_radius == tube.speed_radius == tube.input_radius == tube.next_speed_radius == 0.0


def test_a_tiny_disturbance_gives_a_tube_of_its_tail_alone() -> None:
    """For ``d_bar`` small enough the tail ``eps E`` is within the tolerance with ``s = 0`` terms; it is still RPI."""
    tube = _tube(1e-4)
    assert tube.num_terms == 0 and not tube.is_zero and 0.0 < tube.tail_radius <= 1e-3
    _, B = system_matrices(DT, GAMMA)
    a = _directions(5000, seed=9)
    np.testing.assert_allclose(tube.support(a), np.linalg.norm(a @ tube.tail, axis=1), rtol=1e-12)
    image = tube.support(a @ tube.closed_loop) + 1e-4 * np.linalg.norm(a @ B, axis=1)
    assert np.all(image <= tube.support(a) * (1.0 + 1e-12))
    np.testing.assert_allclose(tube.lifted(np.zeros((0, 2)), np.eye(4)[0]), tube.tail[:, 0])
    assert _tube(1e-3).num_terms == 1


def test_invalid_designs_are_refused() -> None:
    A, B = system_matrices(DT, GAMMA)
    K, _ = lqr(A, B, WEIGHTS.Q, WEIGHTS.R)
    with pytest.raises(ValueError, match="Schur"):
        design_tube(A, B, np.zeros((2, 4)), 0.5)  # A has eigenvalues 1
    with pytest.raises(ValueError, match="d_bar"):
        design_tube(A, B, K, -0.1)
    with pytest.raises(ValueError, match="tolerance"):
        design_tube(A, B, K, 0.5, tolerance=0.0)
    with pytest.raises(ValueError, match="within"):
        design_tube(A, B, K, 0.5, tolerance=1e-12, max_terms=5)
    with pytest.raises(ValueError, match="positive"):
        Weights(0.0, 1.0, 0.1)


@pytest.mark.parametrize("d_bar", [0.5, 1.0])
def test_the_tube_holds_on_the_real_plant_with_the_speed_limiter(d_bar: float) -> None:
    """D33 end to end: the nominal plan at the tightened speed, the real plant with its limiter.

    The nominal state cruises at the tightened speed limit along +x; the real
    plant is the environment's physics. Up to step 60 the disturbance follows
    the sequence that maximizes the candidate speed at step 60 (the sign of
    the velocity response ``(A_K^l B)_v`` of each lag), so the speed limiter
    acts there; afterwards it is random. By F3 the limiter's action is a
    disturbance in D, so the error must stay in Z (checked on its support
    function), and the input never needs a saturation.
    """
    cfg = load_env_config("configs/env/tunnel.yaml")
    params = dataclasses.replace(cfg.physics, d_bar=d_bar)
    layout = Layout.from_config(cfg.geometry)
    A, B = system_matrices(params.dt, params.gamma)
    K, _ = lqr(A, B, WEIGHTS.Q, WEIGHTS.R)
    tube = design_tube(A, B, K, d_bar)
    v_bar = params.v_max - tube.next_speed_radius
    u_bar = params.a_max - tube.input_radius
    rng = np.random.default_rng(7)
    response = [float((np.linalg.matrix_power(tube.closed_loop, lag) @ B)[2, 0]) for lag in range(100)]
    peak = 60
    directions = _directions(400, seed=8)
    h = tube.support(directions)
    z = np.array([-0.5, 0.0, 0.0, 0.0])
    x = z.copy()
    limited = 0
    for k in range(100):
        # nominal input: full planned thrust along +x until the tightened speed, then hold it
        v_nom = np.array([u_bar, 0.0])
        nxt = A @ z + B @ v_nom
        if np.linalg.norm(nxt[2:]) > v_bar:  # hold exactly the tightened speed
            v_nom = (v_bar - (1.0 - params.gamma * params.dt) * z[2]) / params.dt * np.array([1.0, 0.0])
            nxt = A @ z + B @ v_nom
        assert np.linalg.norm(v_nom) <= u_bar + 1e-12
        e = x - z
        u = v_nom + K @ e
        assert np.linalg.norm(u) <= params.a_max + 1e-12  # no input saturation
        if k <= peak:
            d = d_bar * np.sign(response[peak - k] or 1.0) * np.array([1.0, 0.0])
        else:
            w = rng.normal(size=2)
            d = d_bar * rng.uniform() * w / np.linalg.norm(w)
        step = physics_step(x[None, :2], x[None, 2:], u[None], d[None], params, layout)
        limited += int(np.linalg.norm(step.v_limited - step.v_candidate) > 1e-9)
        assert not step.wall_reaction[0]
        x = np.concatenate([step.p[0], step.v[0]])
        z = nxt
        assert np.all(directions @ (x - z) <= h + 1e-9), f"error left Z at step {k}"
    assert limited > 0  # the speed limiter did act, so F3 was exercised
