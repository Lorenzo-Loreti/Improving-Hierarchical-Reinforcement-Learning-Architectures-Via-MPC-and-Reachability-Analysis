from dataclasses import dataclass

from .width_profile import WidthProfile


@dataclass
class TunnelEnvConfig:
    """Single source of truth for TunnelEnv parameters and reward scale."""

    dt: float = 0.1
    tunnel_length: float = 10.0
    tunnel_width: float = 4.0
    v_max: float = 2.0
    u_max: float = 1.0
    sigma_p: float = 0.0
    sigma_v: float = 0.0
    max_steps: int = 500

    step_penalty: float = -1.0
    # -500 made a collision worse than running out the clock with partial
    # progress, so both flat PPO and hPPO converged to a risk-averse
    # "advance a bit, then idle" policy that never attempts the narrow
    # sections -- 0% success across every seed and budget tested on the
    # sibling SlalomEnvConfig (see hppo_explanation.md stall-optimum
    # discussion; kept in sync here for a consistent reward scale across
    # both environments). -150 keeps collisions clearly undesirable while
    # making an honest attempt worth the risk in expectation; combined with
    # impact_penalty_coef (unchanged) still discouraging reckless high-speed
    # impacts specifically. On Slalom, below about -100 the balance instead
    # favours a "rush and crash" policy, and above about -250 it reverts to
    # the original idle trap -- empirically a narrow window, with real
    # seed-to-seed variance even inside it.
    collision_reward: float = -150.0
    goal_reward: float = 500.0

    # Extra penalty on collision, proportional to the lateral impact speed
    # |v_y| (component normal to the wall). Kept separate from
    # `collision_reward` so the fixed outcome penalty stays comparable across
    # soft/hard impacts; 0.0 disables it (no behavior change).
    impact_penalty_coef: float = 100.0

    # Potential-based progress shaping: adds `progress_reward_coef * (p_x' - p_x)`
    # to the reward every step, on top of whichever branch (time penalty /
    # collision / goal) fired. Depends only on p_x, never p_y. See
    # TUNNEL_ENV.md Sec 8.5 for the full design rationale.
    progress_reward_coef: float = 50.0

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
        if self.sigma_p < 0:
            raise ValueError(f"sigma_p must be >= 0, got {self.sigma_p}")
        if self.sigma_v < 0:
            raise ValueError(f"sigma_v must be >= 0, got {self.sigma_v}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {self.max_steps}")


def make_env(config: "TunnelEnvConfig | None" = None, render_mode: "str | None" = None):
    """Blessed construction path for a single TunnelEnv instance."""
    from .tunnel_env import TunnelEnv

    return TunnelEnv(config=config, render_mode=render_mode)
