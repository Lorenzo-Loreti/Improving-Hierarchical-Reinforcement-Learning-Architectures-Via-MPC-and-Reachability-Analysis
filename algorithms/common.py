"""Pure building blocks shared by every algorithm in this tree (`ppo`, `hppo`,
`ppo_mpc`, `ppo_mpc_reach`).

These used to be copy-pasted into each algorithm's own module ("ported from
X for parity", as several of their old docstrings put it), which meant a fix
to one copy -- `clipped_value_loss` and `floor_normalize` were both added
during the hPPO manager-collapse investigation and then hand-ported into
flat PPO afterwards -- had to be re-applied by hand everywhere else, and nothing
caught the copies quietly drifting apart in the meantime (they hadn't, but
only because no one had touched them since the last port). Extracted here so
there is exactly one copy to fix.

Each algorithm's `PPOAgent`/`HPPOAgent`/`PPOMPCAgent` class, and the
`WorkerActor`/`WorkerCritic`/rollout-buffer classes, are deliberately *not*
here: those stay self-contained per algorithm file so one variant's training
loop can be modified for an ablation without any risk of quietly changing
another's. `ManagerActor`/`ManagerCritic` are the one exception: hPPO's
manager head and PPO-MPC's manager head are architecturally identical
networks (confirmed byte-identical, modulo comments, across all three
hierarchical/MPC modules), so a fix to one -- the `MAX_CONCENTRATION` clamp
below was exactly this kind of fix -- silently never reaching the other
two copies was a real, observed risk, not a hypothetical one.

Import this the same way every algorithm module and script already imports
its sibling flat modules (`ppo`, `optimal_solver`, ...): by bare name, once
`algorithms/` is on `sys.path` (every `scenarios/*/scripts/script_*.py` and
every `algorithms/*/tests/conftest.py` already puts it there).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def normalize_obs(obs, low, high):
    """Map physical [low, high] observation bounds to [-1, 1].

    The environment's observation bounds are fixed and hard-enforced by the
    env, so there is nothing to estimate: this is exact, exactly invertible,
    and needs no running statistics to keep in sync between training and
    evaluation. Physical observation dimensions can differ from each other by
    several times in raw dynamic range (e.g. a position axis spanning several
    metres against a velocity axis spanning a couple of m/s); left unmapped,
    that disparity feeds directly into the first linear layer of every actor
    and critic network below.

    Canonical definition. Each agent class stores the bounds it was built
    with and exposes this as a method, so a reloaded checkpoint carries its
    own observation map instead of depending on the caller to reproduce it.
    """
    return 2.0 * (obs - low) / (high - low) - 1.0


def normalize_goal(goal, max_goal_bound):
    """Map a physical manager goal displacement back into (approximately) the
    manager's own [-1, 1] action box.

    Only used by hPPO (`HPPOAgent` in `hppo.py`) -- flat PPO has no
    manager/goal split to normalize, and the PPO+MPC variants hand their
    worker a physical setpoint rather than a network input.

    The hierarchy's second scale disparity: the manager emits a normalized
    action in [-1, 1]^2, the rollout maps it to a physical goal displacement
    in metres (`goal_scale`/`scale_goal` -- a fixed
    multiply-by-`max_goal_bound` box), and the *worker* is then fed that
    physical vector
    concatenated onto its own [-1, 1]-mapped observation. Un-normalized, the
    goal half of the worker's input would span whatever raw physical range
    the scale-up produced -- a disparity reintroduced right after
    `normalize_obs` removed the observation's own.

    Dividing by `max_goal_bound` undoes that disparity exactly: hPPO's
    scale-up is this function's precise inverse (`normalized_goal *
    max_goal_bound`), so the pair round-trips. (`ppo_mpc_reach`'s
    state-dependent `reachable_goal` is *not* invertible this way, which is
    one reason that variant keeps its own goal handling rather than sharing
    this one.) Note that a goal decays as the agent moves toward it (each
    scenario's script_hppo.py), so a partially consumed goal sits strictly
    inside
    `[-max_goal_bound, max_goal_bound]`; only a freshly emitted one can
    approach that edge.
    """
    return goal / max_goal_bound


def clipped_value_loss(newvalue, old_value, target, clip_coef, clip_vloss):
    """PPO2/CleanRL-style optionally-clipped value loss for one minibatch.

    Added during the hPPO manager-collapse investigation (see `ManagerActor`'s
    docstring in `hppo.py`) and adopted by the three manager-based algorithms
    (hPPO, PPO+MPC, PPO+MPC-reach) for diagnostic/API parity, not because the
    underlying finding is specific to hPPO's manager head. Flat PPO carried
    it too, never enabled, and has since dropped it along with
    `floor_normalize` below. `clip_vloss=False` reproduces plain regression
    MSE exactly, so an agent built without opting in behaves exactly as
    before this function existed.

    Without clipping (`clip_vloss=False`) this is a plain regression MSE.
    With it, a *candidate* prediction is additionally clamped to within
    `clip_coef` of `old_value` before scoring, and the loss actually used is
    the worse (larger) of the clipped and unclipped candidate. That `max` is
    the point: when the unclipped prediction has already moved far enough
    from `old_value` to reduce error, the clipped candidate scores *worse*
    (it is pinned near the stale `old_value`) and wins the max, and its
    gradient is zero beyond the clip boundary -- so this update stops pushing
    the critic any further in that direction, the same trust-region
    motivation as the policy ratio clip applied to the other network. It
    does not mean the reported loss is always smaller with clipping on: when
    the unclipped prediction has overshot in the *wrong* direction, the
    unclipped term is already the larger one and wins the max regardless.

    On a 6-seed slalom ablation, enabling this for hPPO's manager head (whose
    `value_bias`/`explained_variance` were seen to drift over ~10-15 updates
    before every observed collapse) took clean solves from 0/6 to 4/6, plus
    one seed where the collapse was merely delayed by ~80k steps -- it
    mitigates the drift mechanism, it does not remove it. See each
    algorithm's own default and the investigation notes in `hppo.py` for the
    full picture; that finding does not by itself say anything about flat
    PPO or PPO-MPC, which is why it is off by default everywhere except
    where a variant's own ablation supports turning it on.
    """
    if not clip_vloss:
        return 0.5 * ((newvalue - target) ** 2).mean()
    v_loss_unclipped = (newvalue - target) ** 2
    v_clipped = old_value + torch.clamp(newvalue - old_value, -clip_coef, clip_coef)
    v_loss_clipped = (v_clipped - target) ** 2
    return 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()


def floor_normalize(advantages, adv_rms, floor_frac):
    """Zero-mean-unit-std normalize `advantages`, with the denominator
    floored at `floor_frac` times `adv_rms`'s multi-update running std
    instead of only this batch's own (possibly anomalously small) std.

    Added alongside `clipped_value_loss` during the hPPO manager-collapse
    investigation, to test the hypothesis that a small, low-variance batch's
    own std was getting driven anomalously small and amplifying noise into
    an oversized update. `floor_frac <= 0.0` (the default everywhere) or
    `adv_rms is None` reproduces the previous, un-floored normalization
    exactly.

    `adv_rms` is updated with this batch's raw advantages *after* the floor
    is computed from its pre-update state, so a single batch cannot lift its
    own floor -- the floor always reflects updates strictly before this one.

    Returns `(normalized_advantages, raw_std)`, where `raw_std` is this
    batch's own std *before* flooring -- logged as a diagnostic regardless of
    whether flooring is active. On the slalom ablation that motivated this
    function, `raw_std` climbed rather than cratered in the updates leading
    into every observed collapse, which refuted the original hypothesis
    (see `ManagerActor`'s docstring in `hppo.py`); flooring itself gave
    mixed, seed-inconsistent results. Left in as an available knob, not
    adopted as any algorithm's default.

    `raw_std` is deliberately the *unbiased* std (`advantages.std()`'s
    default, dividing by N-1) -- the pre-existing, publicly logged
    `adv_std_raw` convention -- while `adv_rms` (Chan et al.) has always
    tracked the *biased* variance (dividing by N, matching its own
    `x.var(unbiased=False)`). The two are not the same number for small
    batches, so they cannot be collapsed into one computation; what can be
    shared is `advantages.mean()`, computed once here and handed to
    `adv_rms.update()` alongside its matching biased variance instead of
    `adv_rms.update()` re-deriving both from `advantages` a second time.
    """
    raw_std = float(advantages.std())
    batch_var, batch_mean = torch.var_mean(advantages, unbiased=False)
    denom = raw_std
    if adv_rms is not None and floor_frac > 0.0:
        denom = max(raw_std, floor_frac * adv_rms.std)
    if adv_rms is not None:
        adv_rms.update(advantages, precomputed=(float(batch_mean), float(batch_var)))
    return (advantages - batch_mean) / (denom + 1e-8), raw_std


class ScaledBeta:
    """A `Beta(alpha, beta)` distribution affinely rescaled from its native
    [0, 1] support to `[low, high]`, so a bounded actor output has genuine
    bounded support (unlike a clipped/squashed Gaussian, which always has
    some density outside the action limits)."""

    def __init__(self, alpha, beta, low=-1.0, high=1.0):
        self.dist = Beta(alpha, beta)
        self.low = low
        self.scale = high - low

    def sample(self):
        return self.dist.sample() * self.scale + self.low

    # No `rsample`: every agent in this tree is a policy-gradient method
    # whose actor loss is a likelihood-ratio surrogate, so none of them ever
    # differentiates through the sampled action. `torch.distributions.Beta`
    # still supports it if a pathwise-gradient agent is ever added here.

    def deterministic_sample(self):
        # Mean of Beta distribution is alpha / (alpha + beta)
        mean = self.dist.concentration1 / (self.dist.concentration1 + self.dist.concentration0)
        return mean * self.scale + self.low

    def log_prob(self, action):
        # Unscale action back to [0, 1]
        unscaled_action = (action - self.low) / self.scale
        unscaled_action = torch.clamp(unscaled_action, 1e-5, 1.0 - 1e-5)
        # Apply log determinant of Jacobian correction
        return self.dist.log_prob(unscaled_action) - torch.log(self.scale)

    def entropy(self):
        # Apply entropy shift correction
        return self.dist.entropy() + torch.log(self.scale)


class RunningMeanStd:
    """Chan et al. parallel running mean/variance, used to normalize a
    critic's regression targets (and, as `adv_rms`, to floor the advantage-
    normalization denominator -- see `floor_normalize`).

    Regressing a freshly initialized critic (outputs ~0, moved by at most
    `lr` per Adam step) directly on raw returns takes far more gradient
    steps than a run provides whenever those returns are large -- e.g. a
    terminal reward barely discounted over a long horizon lands the critic
    correlated with the true value but persistently biased by hundreds of
    units. Regressing on standardized targets instead removes that climb
    entirely. Each call site's own comment covers *why* its particular
    target has (or doesn't have) this problem -- the raw scale varies widely
    across algorithms and, in hPPO, across the manager/worker heads of the
    same algorithm.
    """

    def __init__(self, epsilon=1e-4, horizon=10):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
        # Cap the effective sample count at `horizon` batches so the statistics
        # track the *current* return distribution. With an unbounded count the
        # early-training transient permanently inflates the variance, and late
        # targets get squeezed into a narrow band -- a milder rerun of the
        # scale problem this class exists to prevent.
        self.horizon = horizon

    def update(self, x, precomputed=None):
        """`precomputed`, when given, is this batch's own `(mean, var)` (biased
        variance, matching `x.var(unbiased=False)`) already computed by the
        caller -- e.g. `floor_normalize`, which needs the same pair for
        advantage normalization and would otherwise force a second reduction
        pass over `x` here. `x` is still required in that case, for its
        `numel()`."""
        if precomputed is None:
            batch_mean = float(x.mean())
            batch_var = float(x.var(unbiased=False))
        else:
            batch_mean, batch_var = precomputed
        batch_count = x.numel()

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / tot_count) / tot_count
        self.count = min(tot_count, self.horizon * batch_count)

    @property
    def std(self):
        return math.sqrt(self.var) + 1e-8

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


class ManagerActor(nn.Module):
    """The manager head shared by `HPPOAgent` (hppo.py) and `PPOMPCAgent`
    (ppo_mpc.py, ppo_mpc_reach.py): `obs_dim -> goal_dim`, emitting the two
    Beta concentration parameters for a `ScaledBeta` goal in
    `[-1, 1]^goal_dim`. All three agents build the manager's PPO update
    around this identical network -- only the worker/MPC side downstream of
    the goal differs between them.

    Concentration cap: alpha, beta land in [1, MAX_CONCENTRATION]. Unlike
    the "+1" floor, softplus(x) + 1 alone has no ceiling, so nothing stops
    this Beta from sharpening arbitrarily close to a point mass.

    Kept as a defensive backstop, but it is *not* what fixes the slalom
    seed-1 collapse (see each scenario's script_hppo.py's early-stopping
    instead) -- a first attempt set this to 50 on the theory that both alpha
    and beta were blowing up symmetrically into a narrow peak, and that run
    was bit-for-bit identical to the uncapped one: the cap never bound.
    Inspecting the collapsed checkpoint directly showed the real shape is a
    *skewed* collapse, not a symmetric one -- one goal dimension's beta sits
    pinned at its floor of ~1.0 for every state while its alpha alone climbs
    to ~20-40 (Beta(37, 1) already has raw entropy approx -2.6 nats, most of
    the observed collapse, on its own). Worse, the checkpoint that was still
    solving the task at 100% needed that same dimension's alpha up to ~13
    (Beta(13, 1), raw entropy approx -1.6 nats) -- so a cap tight enough to
    meaningfully bound a skewed collapse (single digits) would also have
    prevented the legitimate policy from ever forming. 50 is therefore a
    loose, last-resort ceiling only, not the active mitigation.

    Neither is early-stopping the actual fix -- it only stops training
    before the collapse is *observed*, on both seed 1 and seed 2, it never
    crossed the strict floor it needs to fire against (see
    script_hppo.py's --early-stop-optimal-frac).

    A first hypothesis for the collapse itself was that the manager's batch
    -- small (~manager_freq times smaller than the worker's) and, once the
    policy is near-converged, low-variance -- gets its advantage
    normalization denominator (this update's own batch std) driven
    anomalously small, amplifying noise into an oversized update. A 6-seed
    slalom ablation (adding the diagnostics below and testing four candidate
    mitigations) did not support it: `manager/adv_std_raw` climbs, not
    craters, in the updates leading into every observed collapse. What those
    same diagnostics *did* show, consistently across every collapsing run
    regardless of which mitigation was active: over ~10-15 updates (~20-30k
    steps) leading into the eval-visible crash, `manager/explained_variance`
    erodes steadily (e.g. 0.90 -> 0.30) while `manager/value_bias` grows
    steadily in one direction (e.g. 2 -> 100) -- a multi-update drift in the
    critic's calibration, not a single bad batch. `manager/approx_kl_max`
    and `manager/ratio_max_dev` (the worst single minibatch of an update,
    not just its mean) occasionally spike past 1.0 in the same window, i.e.
    some sample's action probability more than doubled within one update.

    Of the four candidate mitigations tested (`target_kl_manager`,
    `clip_vloss`, `adv_std_floor_frac`, `ret_rms_horizon_manager`),
    `clip_vloss` is the one adopted as hPPO's default: it directly bounds
    how far the manager's critic can move in one update, which is exactly
    the first link in the drift chain above. Across seeds 1-6, it took clean
    solves (reaching and *holding* a near-optimal policy long enough to
    trigger --early-stop-success-rate, which no unclipped run ever did) from
    0/6 to 4/6, plus one seed where the collapse was merely delayed by ~80k
    steps; one seed still collapsed, with the same drift signature, just
    slower -- this mitigates the mechanism, it does not remove it.
    `target_kl_manager` showed no benefit (a per-update KL cap does not
    catch a drift spread across many updates, each individually
    unremarkable); the other two gave single-seed-inconsistent results
    (solved one seed, collapsed the other *earlier* than an unmodified run)
    that a 1-seed-per-condition ablation cannot distinguish from this
    system's baseline sensitivity -- even a from-scratch rerun of the
    *unmodified* seed-1 configuration collapsed at a different update than
    the original run, most likely from floating-point-level differences
    alone. Take any single-seed comparison on this task with real caution.

    PPOMPCAgent's manager is architecturally identical (same network, same
    PPO update), so it carries the same risk and the same mitigations,
    ported alongside it: `target_kl_manager`, `clip_vloss` (on by default
    there too, extending this result without a dedicated ablation of its
    own), and `adv_std_floor_frac` (off by default -- this ablation gave it
    a mixed, seed-inconsistent result). Each agent's own `__init__`
    docstring notes anything specific to that variant.

    ---- ADDENDUM (2026-09-22): everything above is a symptom ----

    The "manager collapse" is not a manager problem. It is the *worker's*
    reward, and the manager's critic drift described above is downstream of
    it. Read the `--worker-success-bonus` block in
    scenarios/slalom/scripts/script_hppo.py for the full diagnosis; in
    short:

    The worker is trained on a purely intrinsic reward (the per-step
    reduction in the distance to the manager's goal) with no terminal term,
    so in the worker's own MDP the environment's goal line is an absorbing
    state of value 0, while not crossing it pays ~v_max*dt per step forever
    -- the manager hands out a fresh, effectively unreachable goal every c
    steps, so the stream never runs dry. The worker's optimal policy under
    its own reward is therefore to approach the goal line and stall, and PPO
    finds it once the worker's critic is accurate (worker
    explained_variance sits at 0.99 throughout every collapse). Episodes
    then stop terminating, the manager's own returns lose the +goal_reward
    terminal they were calibrated on, and *that* is what erodes
    `manager/explained_variance` and grows `manager/value_bias` over the
    ~10-15 updates before the eval-visible crash. `clip_vloss` helped
    because it slows the manager critic's re-calibration to that shifting
    return distribution, which is why it delays the collapse on some seeds
    and removes it on none.

    Two pieces of evidence make this hard to argue with. First, in a
    collapsed checkpoint the worker's deterministic a_x at (p_x, p_y, v) =
    (9.5, 0, 1.2, 0) is between -1.6 and -2.1 for *every goal in the
    manager's entire action box*: no manager policy can solve the task with
    that worker, which is exactly why the collapse has no in-run recovery
    path. Second, the natural experiment already in this tree -- flat PPO,
    PPO+MPC and PPO+MPC-reach all solve the same slalom cleanly, and none of
    them has a *learned* worker optimizing a reward stream that termination
    cuts off (flat PPO optimizes the environment's reward, whose terminal is
    +goal_reward; PPO+MPC's worker is a QP). hPPO, which does, fails.

    The fix lives in the training scripts, not here: putting the
    environment's outcome back into the worker's return takes clean solves
    on slalom from 0/2 to 2/2 and lands the hierarchy on the oracle optimum.
    The tunnel scenario does not discriminate, being solved long before the
    pathology can bind within its budget. The default is the FeUdal-style
    `--worker-extrinsic-coef 0.02` rather than the more surgical
    `--worker-success-bonus`: the bonus repairs the termination incentive
    alone and leaves the worker blind to walls and to the clock, so every
    collision still has to be steered out by a manager acting once every
    `manager_freq` steps. Both arms reach the same final policy on slalom,
    but the bonus needs 266k/296k/399k steps to get there against the
    extrinsic mix's 71k/81k/81k. See the flag block in
    scenarios/slalom/scripts/script_hppo.py. `MAX_CONCENTRATION` and `clip_vloss` are left as they are --
    they are real backstops against a real failure mode, and the collapsed
    checkpoint does now sit pinned at the cap (alpha=[46.7, 6.7],
    beta=[1.00, 6.86], the skewed shape described above) -- but neither is
    what was wrong.
    """
    MAX_CONCENTRATION = 50.0

    def __init__(self, obs_dim, goal_dim):
        super().__init__()
        self.goal_dim = goal_dim
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, goal_dim * 2), std=0.01)
        )

    def forward(self, obs):
        x = self.net(obs)
        alpha = torch.clamp(F.softplus(x[..., :self.goal_dim]) + 1.0, max=self.MAX_CONCENTRATION)
        beta = torch.clamp(F.softplus(x[..., self.goal_dim:]) + 1.0, max=self.MAX_CONCENTRATION)
        return alpha, beta


class ManagerCritic(nn.Module):
    """The manager head's critic, paired with `ManagerActor` -- see its
    docstring. `obs_dim -> 1`, shared across `HPPOAgent` and `PPOMPCAgent`."""

    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def forward(self, obs):
        return self.net(obs)
