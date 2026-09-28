from dataclasses import dataclass

from .width_profile import WidthProfile


@dataclass
class TunnelEnvConfig:
    """Single source of truth for TunnelEnv parameters and reward scale."""

    dt: float = 0.1
    tunnel_length: float = 10.0
    tunnel_width: float = 4.0
    v_max: float = 1.2
    u_max: float = 2.5
    # Additive process disturbance w, drawn every step independently per
    # component and uniformly from the box W = {|w_p| <= noise_bound_p on
    # each position, |w_v| <= noise_bound_v on each velocity}: the compact set
    # the tube MPC worker (algorithms/tube_mpc.py) is designed against. 0.0,
    # the default, is the deterministic environment every experiment before
    # 2026-09-28 ran on; nothing is drawn then, so those runs are
    # bit-identical.
    #
    # Until 2026-09-28 the disturbance was Gaussian (sigma_p, sigma_v, both
    # 0.0 in every run). It was replaced because a Gaussian has unbounded
    # support: a tube MPC's guarantees hold only while w stays in a compact W
    # with 0 in its interior, so under a Gaussian they are only probabilistic
    # (W taken as a confidence box, e.g. 3 sigma, which some step eventually
    # leaves), while under a bounded W they hold at every step. See the
    # tube-MPC note (Robust Tube-Based MPC for Linear Systems with
    # Non-Convex State Constraints), Assumption 2 and Remark 1.1.
    noise_bound_p: float = 0.0
    noise_bound_v: float = 0.0
    max_steps: int = 200

    step_penalty: float = -1.0
    # Historical note: a *terminal* collision reward of -500 (back when a
    # wall contact ended the episode) made a collision worse than running
    # out the clock with partial progress, so both flat PPO and hPPO
    # converged to a risk-averse "advance a bit, then idle" policy that
    # never attempted the narrow sections -- 0% success across every seed
    # and budget tested on the sibling SlalomEnvConfig. -150 kept collisions
    # clearly undesirable while making an honest attempt worth the risk in
    # expectation -- but was still a narrow, fragile window.
    #
    # The wall is no longer a terminal event: a contact clamps p_y to the
    # violated bound and zeroes v_y (a fully inelastic bounce -- v_x is
    # untouched, only the wall-normal component is absorbed), and the
    # episode continues. `contact_penalty` is the small, per-step-of-contact
    # penalty this now costs (additive to step_penalty, not a replacement),
    # flat regardless of the impact speed. -50.0 is a starting point, not a
    # tuned value -- unlike the old terminal reward this can fire on many
    # steps of the same episode (e.g. wall-hugging through a narrow
    # section), so it needs its own empirical pass rather than inheriting
    # the old terminal-reward scale.
    contact_penalty: float = -50.0
    # 1000 since 2026-09-24, the same as SlalomEnvConfig, so the two scenarios
    # share one reward scale and every reward-scaled knob means the same thing
    # on both. It had been 200 since before the slalom went from 500 to 1000;
    # the tunnel never followed. The gap mattered for hPPO. At
    # --worker-extrinsic-coef 0.02 the tunnel's worker got a +4 terminal. That
    # is below the ~5.5 that stalling in front of the line is worth to it (see
    # that flag's help in algorithms/hppo/hppo_train.py), where the slalom's
    # worker got +20. Tunnel returns measured before this change are on the old
    # scale. The oracle's mean optimal return on the solved-check grid was
    # 213.0 there, as quoted in docs/benchmark.md and
    # docs/goal-box-saturation.md; on this scale it is 1013.0, the slalom's.
    goal_reward: float = 1000.0

    # Potential-based progress shaping: adds `progress_reward_coef * (p_x' - p_x)`
    # to the reward every step, on top of whichever branch (time penalty /
    # collision / goal) fired. Depends only on p_x, never p_y.
    progress_reward_coef: float = 10.0

    # Piecewise-constant lateral width profile (see width_profile.py),
    # overriding the plain constant width `tunnel_width` implies. Left
    # `None` here rather than baked in: `dataclasses.replace(config, ...)`
    # resupplies every current field, so a pre-built profile would go stale
    # after `replace(config, tunnel_width=...)`. TunnelEnv/TunnelVecEnv/
    # MPCWorker each derive `constant_profile(self.W)` fresh whenever this
    # is `None`.
    width_profile: "WidthProfile | None" = None

    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.tunnel_length <= 0:
            raise ValueError(f"tunnel_length must be > 0, got {self.tunnel_length}")
        if self.tunnel_width <= 0:
            raise ValueError(f"tunnel_width must be > 0, got {self.tunnel_width}")
        if self.v_max <= 0:
            raise ValueError(f"v_max must be > 0, got {self.v_max}")
        if self.u_max <= 0:
            raise ValueError(f"u_max must be > 0, got {self.u_max}")
        if self.noise_bound_p < 0:
            raise ValueError(f"noise_bound_p must be >= 0, got {self.noise_bound_p}")
        if self.noise_bound_v < 0:
            raise ValueError(f"noise_bound_v must be >= 0, got {self.noise_bound_v}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {self.max_steps}")
