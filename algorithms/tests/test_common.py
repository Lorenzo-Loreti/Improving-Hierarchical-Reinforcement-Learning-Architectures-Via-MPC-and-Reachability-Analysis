"""Tests for the building blocks in algorithms/common.py.

These used to be copied into each algorithm's own suite (test_ppo.py,
test_hppo.py and both test_ppo_mpc.py files held up to four copies each),
from the time each algorithm module still had its own copy of the code
under test. With one copy of the code there is one copy of its tests; each
test below is the most complete of its former copies, docstrings merged.
Algorithm-level tests of how an agent *uses* these (e.g. a default of
clip_vloss) stay in that algorithm's suite.
"""

import math

import numpy as np
import pytest
import torch
from torch.distributions import Beta

from common import (
    ManagerActor,
    RunningMeanStd,
    ScaledBeta,
    clipped_value_loss,
    floor_normalize,
    normalize_obs,
)

LOW, HIGH = torch.tensor(-1.0), torch.tensor(1.0)
OBS_LOW = [-1.0, -2.0, -2.0, -2.0]
OBS_HIGH = [11.0, 2.0, 2.0, 2.0]
OBS_DIM, GOAL_DIM = 4, 2


# --------------------------------------------------------------------------
# ScaledBeta: the change-of-variables corrections
# --------------------------------------------------------------------------

def test_scaled_beta_density_integrates_to_one():
    """The Jacobian correction in log_prob makes it a density on [-1, 1]."""
    dist = ScaledBeta(torch.tensor([2.5]), torch.tensor([1.3]), low=LOW, high=HIGH)
    xs = torch.linspace(-1 + 1e-6, 1 - 1e-6, 200001).unsqueeze(1)
    density = dist.log_prob(xs).exp().squeeze(1)
    assert torch.trapz(density, xs.squeeze(1)).item() == pytest.approx(1.0, abs=1e-4)


def test_scaled_beta_entropy_matches_numeric_integration():
    """The +log(scale) shift in entropy() matches -E[log p] of the scaled density."""
    dist = ScaledBeta(torch.tensor([2.5]), torch.tensor([1.3]), low=LOW, high=HIGH)
    xs = torch.linspace(-1 + 1e-6, 1 - 1e-6, 200001).unsqueeze(1)
    logp = dist.log_prob(xs).squeeze(1)
    numeric = -torch.trapz(logp.exp() * logp, xs.squeeze(1))
    assert numeric.item() == pytest.approx(float(dist.entropy()), abs=1e-4)


def test_scaled_beta_corrections_are_the_log_of_the_scale():
    """Both corrections are exactly log(high - low) against the base Beta."""
    alpha, beta = torch.tensor([2.5]), torch.tensor([1.3])
    base, scaled = Beta(alpha, beta), ScaledBeta(alpha, beta, low=LOW, high=HIGH)
    action = torch.tensor([0.4])
    unscaled = (action - LOW) / (HIGH - LOW)
    assert scaled.log_prob(action).item() == pytest.approx(
        (base.log_prob(unscaled) - math.log(2.0)).item(), abs=1e-6)
    assert float(scaled.entropy()) == pytest.approx(float(base.entropy()) + math.log(2.0), abs=1e-6)


def test_scaled_beta_samples_are_inside_the_action_box():
    """Support is the action box, so the env's action clip is a no-op."""
    dist = ScaledBeta(torch.full((5000, 2), 1.7), torch.full((5000, 2), 3.1), low=LOW, high=HIGH)
    samples = dist.sample()
    assert samples.min() >= -1.0 and samples.max() <= 1.0


def test_deterministic_sample_is_the_distribution_mean():
    """Evaluation uses the mean, NOT the mode -- thesis PPO chapter, 4.5."""
    alpha, beta = torch.tensor([2.5]), torch.tensor([1.3])
    dist = ScaledBeta(alpha, beta, low=LOW, high=HIGH)
    expected_mean = 2.0 * (alpha / (alpha + beta)) - 1.0
    expected_mode = 2.0 * ((alpha - 1) / (alpha + beta - 2)) - 1.0
    assert dist.deterministic_sample().item() == pytest.approx(expected_mean.item(), abs=1e-6)
    assert dist.deterministic_sample().item() != pytest.approx(expected_mode.item(), abs=1e-3)


def test_entropy_is_bounded_above_by_the_uniform_case():
    """Unlike a Gaussian, the entropy bonus has an attainable maximum: Beta(1, 1),
    the uniform. The heads' initial Beta, roughly Beta(1 + log 2, 1 + log 2)
    (softplus(~0) + 1 from the near-zero last layer), sits just below it."""
    uniform = float(ScaledBeta(torch.tensor(1.0), torch.tensor(1.0), LOW, HIGH).entropy())
    assert uniform == pytest.approx(math.log(2.0), abs=1e-6)
    at_init = float(ScaledBeta(torch.tensor(math.log(2.0) + 1.0),
                               torch.tensor(math.log(2.0) + 1.0), LOW, HIGH).entropy())
    assert at_init < uniform


# --------------------------------------------------------------------------
# RunningMeanStd: the value-target (and advantage-floor) statistics
# --------------------------------------------------------------------------

def test_running_mean_std_tracks_batch_statistics():
    rms = RunningMeanStd()
    x = torch.randn(4096) * 30 + 400
    rms.update(x)
    assert rms.mean == pytest.approx(float(x.mean()), rel=1e-3)
    assert rms.std == pytest.approx(float(x.std(unbiased=False)), rel=1e-2)


def test_running_mean_std_count_is_capped_at_the_horizon():
    """The cap is what keeps early-training returns from poisoning late targets."""
    rms = RunningMeanStd(horizon=10)
    batch = torch.randn(2048)
    for _ in range(30):
        rms.update(batch)
    assert rms.count == 10 * 2048


def test_running_mean_std_forgets_a_stale_regime():
    """A capped estimator must migrate to a new return level; an uncapped one
    lags -- this is the whole point of `horizon` (see RunningMeanStd's
    docstring), which a count-hits-the-cap test alone does not exercise."""
    capped, uncapped = RunningMeanStd(horizon=10), RunningMeanStd(horizon=10**9)
    for _ in range(3):
        capped.update(torch.full((2048,), -600.0))
        uncapped.update(torch.full((2048,), -600.0))
    for _ in range(15):
        capped.update(torch.full((2048,), 400.0))
        uncapped.update(torch.full((2048,), 400.0))
    assert capped.mean > uncapped.mean


# --------------------------------------------------------------------------
# normalize_obs
# --------------------------------------------------------------------------

def test_normalize_obs_maps_bounds_to_plus_minus_one():
    low = np.array(OBS_LOW, dtype=np.float32)
    high = np.array(OBS_HIGH, dtype=np.float32)
    assert normalize_obs(low, low, high) == pytest.approx([-1.0] * 4)
    assert normalize_obs(high, low, high) == pytest.approx([1.0] * 4)
    assert normalize_obs((low + high) / 2, low, high) == pytest.approx([0.0] * 4)


# --------------------------------------------------------------------------
# Manager-collapse mitigations: value-loss clipping and the advantage-std
# floor (see clipped_value_loss/floor_normalize's docstrings)
# --------------------------------------------------------------------------

def test_clipped_value_loss_matches_plain_mse_when_disabled():
    """clip_vloss=False must reproduce the original, unclipped loss exactly
    -- this is the default, so every run before this option existed depends
    on it being a no-op."""
    newvalue = torch.tensor([0.0, 5.0, -3.0])
    old_value = torch.tensor([10.0, 10.0, 10.0])
    target = torch.tensor([1.0, 1.0, 1.0])
    expected = 0.5 * ((newvalue - target) ** 2).mean()
    actual = clipped_value_loss(newvalue, old_value, target, clip_coef=0.2, clip_vloss=False)
    assert torch.allclose(actual, expected)


def test_clipped_value_loss_matches_unclipped_inside_the_trust_region():
    """When newvalue is already within clip_coef of old_value, clipping has
    nothing to bind on: clipped and unclipped candidates coincide."""
    old_value = torch.tensor([10.0])
    newvalue = old_value + 0.05  # inside clip_coef=0.2
    target = torch.tensor([10.5])
    expected = 0.5 * ((newvalue - target) ** 2).mean()
    actual = clipped_value_loss(newvalue, old_value, target, clip_coef=0.2, clip_vloss=True)
    assert torch.allclose(actual, expected)


def test_clipped_value_loss_picks_the_clipped_candidate_when_it_scores_worse():
    """newvalue has moved beyond clip_coef *and* toward target (a legitimate
    improvement): the clipped candidate, pinned near the stale old_value,
    scores worse than the unclipped one and must be the one used -- this is
    the actual trust-region bite, not a no-op."""
    old_value = torch.tensor([10.0])
    target = torch.tensor([0.0])
    newvalue = torch.tensor([8.0])  # moved 2.0 toward target, past clip_coef=0.2
    unclipped_loss = 0.5 * ((newvalue - target) ** 2).mean()
    actual = clipped_value_loss(newvalue, old_value, target, clip_coef=0.2, clip_vloss=True)
    assert actual > unclipped_loss
    v_clipped = old_value - 0.2  # clamp(8-10, -0.2, 0.2) == -0.2, so old_value + (-0.2)
    expected = 0.5 * ((v_clipped - target) ** 2).mean()
    assert torch.allclose(actual, expected)


def test_clipped_value_loss_zero_gradient_beyond_the_clip_boundary():
    """The actual protection clip_vloss buys: once the clipped candidate wins
    the max() (previous test), its gradient w.r.t. newvalue is zero out there
    -- clamp() is flat beyond the boundary -- so this update stops pushing
    the critic any further in that direction."""
    old_value = torch.tensor([10.0])
    target = torch.tensor([0.0])
    newvalue = torch.tensor([8.0], requires_grad=True)
    loss = clipped_value_loss(newvalue, old_value, target, clip_coef=0.2, clip_vloss=True)
    loss.backward()
    assert newvalue.grad.item() == pytest.approx(0.0)


def test_floor_normalize_matches_previous_behaviour_when_floor_is_zero():
    """floor_frac=0.0 (the default) must reproduce the original single-batch
    normalization exactly, whether or not an adv_rms is supplied."""
    advantages = torch.tensor([1.0, -2.0, 3.0, 0.5])
    expected = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    normalized, raw_std = floor_normalize(advantages.clone(), RunningMeanStd(), 0.0)
    assert torch.allclose(normalized, expected, atol=1e-6)
    assert raw_std == pytest.approx(float(advantages.std()))


def test_floor_normalize_floors_a_degenerate_batchs_std():
    """A batch with an anomalously small std (the manager-collapse hypothesis:
    near-identical trajectories once the policy is near-converged) must be
    normalized against the floor, not its own near-zero std, or dividing by
    it would amplify whatever noise is left into an oversized update."""
    rms = RunningMeanStd()
    rms.mean, rms.var = 0.0, 100.0  # multi-update history: std == 10.0
    tiny = torch.tensor([1.0, 1.0001, 0.9999, 1.0])  # std ~= 4e-5
    normalized, raw_std = floor_normalize(tiny, rms, floor_frac=0.5)
    assert raw_std < 1e-3
    expected = (tiny - tiny.mean()) / (0.5 * 10.0 + 1e-8)
    assert torch.allclose(normalized, expected, atol=1e-4)


def test_floor_normalize_leaves_a_healthy_batch_unfloored():
    """A batch whose own std already exceeds the floor must be normalized
    against its own std, unchanged from the un-floored behaviour -- the floor
    only ever raises the denominator, never lowers it."""
    rms = RunningMeanStd()
    rms.mean, rms.var = 0.0, 1.0  # multi-update history: std == 1.0
    advantages = torch.tensor([10.0, -20.0, 30.0, 5.0])  # std >> 0.5 * 1.0
    normalized, raw_std = floor_normalize(advantages.clone(), rms, floor_frac=0.5)
    expected = (advantages - advantages.mean()) / (raw_std + 1e-8)
    assert torch.allclose(normalized, expected, atol=1e-5)


def test_floor_normalize_updates_adv_rms_but_not_in_time_for_its_own_floor():
    """Complements test_floor_normalize_floors_a_degenerate_batchs_std, which
    shows a degenerate batch cannot lift its own floor. This shows the other
    half: adv_rms is still updated with that batch, just *after* the floor
    was computed from its pre-update state -- so the next call sees it, even
    though this one did not."""
    rms = RunningMeanStd()
    rms.mean, rms.var, rms.count = 0.0, 100.0, 1000.0  # well-established std == 10.0
    tiny = torch.tensor([1.0, 1.0001, 0.9999, 1.0])  # std ~= 4e-5
    floor_normalize(tiny, rms, floor_frac=0.5)
    # The tiny batch must have pulled the running std down for whatever
    # checks it *after* this call -- proving it landed, not that it was
    # silently dropped.
    assert rms.std < 10.0


# --------------------------------------------------------------------------
# ManagerActor: the concentration cap shared by hPPO and both PPO+MPC variants
# --------------------------------------------------------------------------

def test_manager_actor_output_shapes():
    actor = ManagerActor(OBS_DIM, GOAL_DIM)
    alpha, beta = actor(torch.randn(5, OBS_DIM))
    assert alpha.shape == (5, GOAL_DIM)
    assert beta.shape == (5, GOAL_DIM)
    assert (alpha >= 1.0).all() and (beta >= 1.0).all()


def test_manager_actor_concentrations_never_exceed_cap():
    """softplus(x) + 1 alone has no ceiling (hPPO's worker head, of the same
    form, has none), so nothing stops the manager's Beta from sharpening
    arbitrarily close to a point mass -- the skewed shape hPPO's collapsed
    manager was found in on the slalom. See ManagerActor's docstring: the
    cap is a backstop, not the fix; the collapse's root cause was the
    worker's reward. Driving logits to +inf must still saturate at the cap,
    not climb past it."""
    actor = ManagerActor(OBS_DIM, GOAL_DIM)
    with torch.no_grad():
        for layer in actor.net:
            if isinstance(layer, torch.nn.Linear):
                layer.bias.fill_(50.0)  # drive softplus to ~50
    alpha, beta = actor(torch.randn(64, OBS_DIM))
    assert alpha.max().item() == pytest.approx(ManagerActor.MAX_CONCENTRATION)
    assert beta.max().item() == pytest.approx(ManagerActor.MAX_CONCENTRATION)


def test_manager_actor_concentrations_never_drop_below_one():
    """The pre-existing floor (softplus >= 0, so alpha/beta >= 1) must survive
    the clamp -- the cap should only ever bind from above."""
    actor = ManagerActor(OBS_DIM, GOAL_DIM)
    with torch.no_grad():
        for layer in actor.net:
            if isinstance(layer, torch.nn.Linear):
                layer.bias.fill_(-50.0)  # drive softplus to ~0
    alpha, beta = actor(torch.randn(64, OBS_DIM))
    assert alpha.min().item() >= 1.0 and beta.min().item() >= 1.0
