from dataclasses import dataclass

from .spawn_sampler import INIT_SAMPLERS
from .width_profile import WidthProfile


@dataclass
class SlalomEnvConfig:
    """Single source of truth for SlalomEnv parameters and reward scale."""

    dt: float = 0.1
    tunnel_length: float = 10.0
    tunnel_width: float = 4.0
    # The speed and thrust limits bound magnitudes, ||v|| <= v_max and
    # ||u|| <= u_max, since 2026-10-04; until then they bounded each axis
    # (git tag box-limits-final). See envs/actuation.py.
    v_max: float = 1.2
    u_max: float = 2.5
    # Additive process disturbance w = (w_p, w_v), drawn every step uniformly
    # from W = {||w_p|| <= noise_bound_p, ||w_v|| <= noise_bound_v}, two
    # disks: the compact set the tube MPC worker (algorithms/tube_mpc.py) is
    # designed against. 0.0, the default, is the deterministic environment;
    # nothing is drawn then.
    #
    # W was a box, |w| <= noise_bound per component, from 2026-09-28 until
    # 2026-10-04, when the speed and thrust limits became disks and the
    # disturbance followed them (envs/actuation.py says why: the tube MPC's
    # tightened sets stay disks only under an isotropic W). The disturbed
    # studies before 2026-10-04 ran on the box, with the same bounds.
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
    # How SlalomVecEnv draws the start of each *training* episode from the
    # spawn box (p_x0 in [0, 2], p_y0 in [-W/4, W/4], at rest): "uniform",
    # independent draws, or "sobol", a scrambled Sobol' sequence
    # (randomized quasi-Monte Carlo) that spreads successive starts evenly
    # over the box. Every start is uniform on the box under both, so the
    # objective is the same; see envs/spawn_sampler.py. Evaluation (the plain
    # env's reseeded reset) draws independently under both, from the same
    # starts. "uniform", the default, is how every run before 2026-10-03
    # trained, unchanged bit for bit. It stays the default because "sobol"
    # changed nothing measurable on the slalom, for any algorithm
    # (docs/init-sampler.md).
    init_sampler: str = "uniform"
    max_steps: int = 200

    step_penalty: float = -1.0
    # Historical note: a *terminal* collision reward of -500 (back when a
    # wall contact ended the episode) made a collision worse than running
    # out the clock with partial progress, so both flat PPO and hPPO
    # converged to a risk-averse "advance a bit, then idle" policy that
    # never attempted the gates -- 0% success across every seed and budget
    # tested. -150 kept
    # collisions clearly undesirable while making an honest attempt worth
    # the risk in expectation -- but was still a narrow, fragile window.
    #
    # The wall is no longer a terminal event: a contact is a fully inelastic
    # bounce off the wall that was hit -- only the wall-normal velocity
    # component is absorbed and the position put back on its side -- and the
    # episode continues. Along the corridor that clamps p_y and zeroes v_y;
    # against a gate's face, entered outside its opening, it puts p_x back in
    # front of the face and zeroes v_x (since 2026-10-04; before, that case
    # too clamped p_y, sideways through the face -- see the env's step). `contact_penalty` is the small, per-step-of-contact
    # penalty this now costs (additive to step_penalty, not a replacement),
    # flat regardless of the impact speed. -50.0 is a starting point, not a
    # tuned value -- unlike the old terminal reward this can fire on many
    # steps of the same episode (e.g. wall-hugging through a gate), so it
    # needs its own empirical pass rather than inheriting the old
    # terminal-reward scale.
    contact_penalty: float = -50.0
    goal_reward: float = 1000.0

    # Potential-based progress shaping: adds `progress_reward_coef *
    # (min(p_x', L) - p_x)` to the reward every step, on top of whichever
    # branch (time penalty / collision / goal) fired. Depends only on p_x,
    # never p_y -- important here since the gates are off-center, so a
    # Euclidean-distance-to-goal potential would fight them.
    #
    # The potential is cut at the goal line L since 2026-10-04, so the
    # crossing step pays for the distance up to L and not for how far past
    # it the vehicle lands. The return of a successful episode then depends
    # only on its length, its control effort and its contacts, which is what
    # the oracle maximises (algorithms/optimal_solver.py). Uncut, a policy
    # could beat the oracle's return by up to progress_reward_coef * v_max *
    # dt = 1.2 while arriving on the same step, by crossing at top speed
    # where the oracle aims for L itself.
    progress_reward_coef: float = 10.0

    # Control-effort penalty: adds `effort_penalty * (||u||/u_max)^2` to the
    # reward every step, on top of the rest (the goal step included), u the
    # acceleration the plant actually delivered (after the thrust and speed
    # limits, envs/actuation.py). A step at full thrust costs |effort_penalty|.
    # Since 2026-10-04; 0.0 before.
    #
    # Why -0.01. Small enough that time always comes first: an episode at
    # full thrust throughout costs at most 0.01 per step, and a min-time
    # episode from the spawn box lasts at most ~90 steps, so its whole effort
    # is worth less than one step of step_penalty. Every trajectory that
    # arrives earlier therefore still has the higher return, and the oracle
    # is the min-time trajectory that uses the least effort among the
    # min-time ones (before 2026-10-04 its Sum ||u||^2 objective was only a
    # tie-break the reward did not know about; now reward and oracle agree).
    # What the penalty adds is a preference among the many trajectories
    # that arrive on the same step -- smooth thrust over bang-bang chatter --
    # for every algorithm alike: it is part of the environment's reward, so
    # PPO, hPPO's manager and worker (algorithms/hppo/hppo_train.py) and
    # PPO+MPC's manager, which is charged for the effort its MPC worker
    # spends, all see it. A larger weight would make the task a time-energy
    # trade-off, where arriving later can pay.
    effort_penalty: float = -0.01

    # Piecewise-constant lateral width profile (see width_profile.py, e.g.
    # slalom_profile()), overriding the plain constant width `tunnel_width`
    # implies. Left `None` here rather than baked in: `dataclasses.replace(
    # config, ...)` resupplies every current field, so a pre-built profile
    # would go stale after `replace(config, tunnel_width=...)`. SlalomEnv/
    # SlalomVecEnv/MPCWorker each derive `constant_profile(self.W)` fresh
    # whenever this is `None`.
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
        if self.init_sampler not in INIT_SAMPLERS:
            raise ValueError(f"init_sampler must be one of {INIT_SAMPLERS}, got {self.init_sampler!r}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {self.max_steps}")
        if self.effort_penalty > 0:
            raise ValueError(f"effort_penalty must be <= 0, got {self.effort_penalty}")
