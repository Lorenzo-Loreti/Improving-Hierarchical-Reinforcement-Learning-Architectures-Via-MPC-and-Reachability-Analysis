import os
import argparse
import time
import numpy as np
import torch
import wandb
import sys
# Scenario root (this script's parent), for `envs`; then algorithms/ (for
# the flat `mpc_worker` module) and algorithms/ppo_mpc (for the flat
# `ppo_mpc` module), both imported by bare name.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithms')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithms', 'ppo_mpc')))
from envs.config import TunnelEnvConfig
from envs.tunnel_env import TunnelEnv
from envs.vec_tunnel_env import TunnelVecEnv
from ppo_mpc import PPOMPCAgent, ManagerVecRolloutBuffer
from mpc_worker import MPCWorker
from optimal_solver import spawn_grid, precompute_optimal_grid, MinTimeSolver
from solved_check import check_solved

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ppo_mpc_tunnel", help="name of this experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True)
    parser.add_argument("--cuda", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True)
    parser.add_argument("--track", action="store_true", help="track with wandb")
    parser.add_argument("--wandb-project-name", type=str, default="Tunnel-PPO-MPC-NoNoise")

    # Reward-scale override. Defaults to TunnelEnvConfig's own default, so
    # omitting this reproduces exactly the environment every other script
    # (flat PPO, hPPO) trains against -- see envs/config.py, and
    # script_hppo.py's --contact-penalty, which this mirrors.
    parser.add_argument("--env-u-max", type=float, default=None,
        help="REGIME STUDY ONLY. Override the environment's acceleration "
             "limit u_max (default None = TunnelEnvConfig's own 2.5, i.e. the "
             "canonical environment every architecture in this repo is "
             "compared on). Lowering it makes the plant sluggish, raising the "
             "agility ratio rho = v_max/(u_max*manager_freq*dt). Mirrors the "
             "identically-named flag in scenarios/slalom, and exists so the "
             "constant-width corridor can act as that sweep's control: it "
             "isolates whether a rho effect comes from plant sluggishness "
             "alone or needs the slalom's lateral precision demand. See "
             "docs/reachability-regime-study.md")
    parser.add_argument("--contact-penalty", type=float, default=TunnelEnvConfig().contact_penalty,
        help="small penalty applied every step the agent is in contact with a "
             "wall, flat regardless of impact speed -- a contact clamps the "
             "agent to the wall and absorbs v_y, but the episode continues. "
             "Defaults to TunnelEnvConfig's own default, so omitting this "
             "reproduces the shared environment every other script trains "
             "against")

    # Run params
    parser.add_argument("--total-timesteps", type=int, default=200000)

    # Manager params (PPO)
    parser.add_argument("--learning-rate-manager", type=float, default=3e-4)
    parser.add_argument("--anneal-lr", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True,
        help="if toggled, the manager's learning rate decays linearly to 0 over training")
    parser.add_argument("--lr-floor-frac", type=float, default=0.0,
        help="the linear LR anneal stops at this fraction of the base rate "
             "instead of decaying all the way to 0. At 0.0 (default) this is "
             "exactly the previous anneal-to-zero behaviour. A floor keeps "
             "the last updates from freezing (approx_kl collapsing to ~0), "
             "which otherwise burns the tail of --total-timesteps on a "
             "policy that can no longer move -- ported from script_hppo.py")
    parser.add_argument("--critic-lr-mult", type=float, default=3.0,
        help="the manager critic's learning rate as a multiple of --learning-rate-manager")
    parser.add_argument("--manager-freq", type=int, default=10, help="macro-step length c")
    parser.add_argument("--max-goal-bound", type=float, default=None,
        help="half-width of the manager's positional goal box in metres: its "
             "[-1, 1] action is scaled by this into a physical (delta_x, "
             "delta_y) displacement. Defaults to the displacement actually "
             "reachable within one macro-step, v_max * --manager-freq * dt, "
             "times --goal-bound-slack (1.80 m at the defaults). Unlike "
             "hPPO's identically-named "
             "flag, this one is not free: the MPC worker tracks the goal as "
             "a hard QP setpoint, so an unreachable one saturates it into a "
             "full-speed direction command -- see the block where this is "
             "consumed. Pass 10 to reproduce the old, inherited-from-hPPO "
             "value")
    parser.add_argument("--goal-bound-slack", type=float, default=1.5,
        help="--max-goal-bound's default, as a multiple of the "
             "segment-reachable displacement v_max * --manager-freq * dt. "
             "Ignored when --max-goal-bound is given explicitly. 1.0 pins the "
             "box at exactly the reachable radius, which costs the x axis its "
             "headroom (full speed then needs an action of exactly 1.0); much "
             "above 2.0 and the unreachable-setpoint saturation this whole "
             "block exists to avoid starts coming back. See the block where "
             "this is consumed for the sweep")
    parser.add_argument("--max-goal-vel", type=float, default=None,
        help="half-width of the manager's velocity goal box in m/s, the "
             "velocity half of --max-goal-bound's setpoint. Defaults to the "
             "plant's own v_max, past which MPCWorker's QP box makes the "
             "target unreachable by construction. Pass 2 to reproduce the "
             "previous hardcoded value (which exceeded the plant's v_max of "
             "1.2)")
    parser.add_argument("--replan-on-collision", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True,
        help="force the manager to resample a goal the instant the worker "
             "(MPC) contacts a wall, instead of waiting out --manager-freq "
             "or episode end -- a wall hit can clamp position/zero velocity "
             "in a way that makes the in-flight goal unreachable for the "
             "rest of the segment. On by default; pass "
             "--replan-on-collision false to restore the pre-fix behaviour. "
             "Ported from script_hppo.py")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel tunnel environments to collect rollouts "
             "from. Each carries its own goal, manager cadence, segment "
             "accumulator and MPC warm-start slot")
    parser.add_argument("--num-steps-worker", type=int, default=2000,
        help="PPO rollout length in env steps, summed over all --num-envs "
             "environments. Unchanged in meaning by vectorization -- it is "
             "still the step budget per update, so the update count stays "
             "where it was; what changed is that those steps now come from "
             "--num-envs parallel streams. Must be divisible by --num-envs")
    parser.add_argument("--mpc-backend", type=str, default="osqp", choices=["osqp", "cvxpy"],
        help="MPC QP backend. 'osqp' assembles the problem directly and is "
             "~81x faster per solve; 'cvxpy' is the original formulation, kept "
             "as a reference (see mpc_worker.py). The solver is essentially "
             "100%% of this algorithm's rollout cost, so this flag, not "
             "--num-envs, is what governs its wall clock")
    parser.add_argument("--update-epochs", type=int, default=10, help="PPO K epochs")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches the manager's rollout is split "
             "into. The manager writes one transition per --manager-freq "
             "worker steps, so its buffer is roughly manager_freq times "
             "smaller than --num-steps-worker (~200, not ~2000, at the "
             "defaults) -- splitting that 32 ways, as an earlier default did, "
             "left ~6 samples per gradient step. 4 matches the minibatch size "
             "HRL/hPPO's --num-minibatches-manager uses for its own, "
             "similarly-sized manager buffer")
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--target-kl-manager", type=float, default=None,
        help="if set, stop the manager's update early once approx_kl "
             "exceeds this. Off by default. Ported from HRL/hPPO's "
             "--target-kl-manager: this manager is architecturally "
             "identical to HPPOAgent's (see ManagerActor in "
             "algorithms/hppo/hppo.py), same collapse risk")
    parser.add_argument("--clip-vloss", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True,
        help="PPO2/CleanRL-style value-loss clipping: bound how far the "
             "critic's new prediction may move from its pre-update one (by "
             "--clip-coef) before scoring the value loss. On by default, "
             "extending hppo's 6-seed slalom ablation result (4/6 seeds "
             "went from never recovering after a manager collapse to a "
             "clean solve) by architectural similarity -- not yet "
             "independently re-validated on this module. Pass "
             "--clip-vloss false to restore the previous unclipped "
             "behaviour")
    parser.add_argument("--adv-std-floor-frac", type=float, default=0.0,
        help="floor the advantage-normalization denominator at this "
             "fraction of the manager's own multi-update running std, "
             "instead of dividing by the current batch's std alone. 0.0 "
             "(default) reproduces the previous behaviour exactly -- "
             "hppo's own ablation gave this a mixed, seed-inconsistent "
             "result, so it is not on by default here either")
    parser.add_argument("--ret-rms-horizon-manager", type=int, default=10,
        help="effective sample count (in multiples of one manager batch) "
             "the manager's return-normalization statistics remember. "
             "Default 10 matches the previous (hardcoded) behaviour")
    parser.add_argument("--ent-coef-manager", type=float, default=0.01,
        help="coefficient of the manager entropy. With --autotune-ent-coef, this is only the initial value")
    parser.add_argument("--autotune-ent-coef", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True,
        help="learn the manager's entropy coefficient via SAC-style dual ascent toward a target entropy. On by default; pass --autotune-ent-coef false to hold --ent-coef-manager fixed instead")
    parser.add_argument("--target-entropy-frac", type=float, default=0.35,
        help="target entropy as a fraction of the manager's max achievable entropy; only used with --autotune-ent-coef")
    parser.add_argument("--ent-coef-lr", type=float, default=3e-4,
        help="learning rate for the entropy coefficient's own optimizer; only used with --autotune-ent-coef")
    parser.add_argument("--ent-coef-min", type=float, default=1e-4,
        help="lower clamp on the autotuned entropy coefficient; only used with --autotune-ent-coef")
    parser.add_argument("--ent-coef-max", type=float, default=1.0,
        help="upper clamp on the autotuned entropy coefficient; only used with --autotune-ent-coef")
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval-freq", type=int, default=1, help="evaluate the agent every eval_freq updates")
    parser.add_argument("--eval-episodes", type=int, default=10, help="number of episodes to evaluate the agent")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
        help="directory (relative to this script) to save model checkpoints in")
    parser.add_argument("--solved-early-stop", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True,
        help="if toggled, actually stop training once the solved criterion below "
             "is met. The check, its logging, and the one-time solved.pt checkpoint "
             "still happen either way -- this only gates the early `break`, so runs "
             "meant to be plotted against each other on the same x-axis (fixed "
             "--total-timesteps for every seed) can set this false and still see "
             "where each seed crossed the threshold")
    parser.add_argument("--solved-tolerance", type=float, default=5.0,
        help="stop training once the agent's return is within this many reward "
             "units of the oracle's optimal return (see algorithms/optimal_solver.py) "
             "at every point of the fixed evaluation grid below")
    parser.add_argument("--solved-grid-nx", type=int, default=5,
        help="number of p_x0 points in the fixed grid the solved-check evaluates from")
    parser.add_argument("--solved-grid-ny", type=int, default=5,
        help="number of p_y0 points in the fixed grid the solved-check evaluates from")
    parser.add_argument("--solved-consecutive", type=int, default=2,
        help="require the solved criterion to hold for this many consecutive "
             "eval passes in a row before actually stopping training -- guards "
             "against stopping on a momentary crossing while the policy is "
             "still moving")
    parser.add_argument("--early-stop-success-rate", type=float, default=1.0,
        help="stop training once eval/success_rate has been >= this AND "
             "eval/episodic_return >= --early-stop-optimal-frac, both for "
             "--early-stop-patience consecutive evaluations. Independent of "
             "--solved-early-stop above -- either can stop training. On by "
             "default: this manager can reach 100%% eval success while "
             "still threading the tunnel with avoidable wall contacts, well "
             "short of the return --solved-early-stop's tight tolerance "
             "needs -- stopping as soon as it has demonstrably solved the "
             "task avoids burning the rest of the budget on a manager whose "
             "goal distribution keeps sharpening past that point. Set > 1.0 "
             "to disable. Ported from script_hppo.py")
    parser.add_argument("--early-stop-optimal-frac", type=float, default=0.95,
        help="the eval/episodic_return floor --early-stop-success-rate also "
             "requires, as a fraction of the solved-check oracle's mean "
             "optimal return (see the 'Solved-check: precomputed optimal "
             "returns' line printed at startup) rather than a fixed reward "
             "value -- success_rate alone is not enough, since reaching the "
             "goal line at all does not mean reaching it cleanly")
    parser.add_argument("--early-stop-patience", type=int, default=3,
        help="consecutive qualifying evaluations required before "
             "--early-stop-success-rate stops training; see its help")

    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error(f"--num-envs must be > 0, got {args.num_envs}")
    if args.num_steps_worker % args.num_envs != 0:
        parser.error(
            f"--num-steps-worker ({args.num_steps_worker}) must be divisible by "
            f"--num-envs ({args.num_envs}); otherwise the rollout is not the "
            f"step budget it names")
    args.num_steps_per_env = args.num_steps_worker // args.num_envs
    return args

def make_env(config=None):
    """A bare TunnelEnv, for evaluation.

    Rollouts come from a batched `TunnelVecEnv` instead. `RecordEpisodeStatistics`
    is gone with the single-environment rollout it wrapped: returns and lengths
    are counted in the loop, which also reaches the success and collision flags
    the wrapper does not carry.
    """
    return TunnelEnv(config=config)

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"

    # Single source of truth for this run's environment geometry -- the vec
    # rollout env, the eval env, and (below) the MPC worker's own box all
    # derive from this one object instead of independently restating the
    # same numbers, which is how those three used to silently drift apart.
    env_config = TunnelEnvConfig(
        contact_penalty=args.contact_penalty,
        **({} if args.env_u_max is None else {"u_max": args.env_u_max}),
    )

    if args.env_u_max is not None:
        _T = args.manager_freq * env_config.dt
        print(f"REGIME STUDY: env u_max overridden to {env_config.u_max} "
              f"(canonical {TunnelEnvConfig().u_max}); agility ratio rho = "
              f"v_max/(u_max*T) = {env_config.v_max / (env_config.u_max * _T):.2f}")

    
    if args.track:
        wandb.init(project=args.wandb_project_name, sync_tensorboard=False, config=vars(args), name=run_name, save_code=True)
        
    # Seeding
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Env setup: a batched TunnelVecEnv for rollout collection, a plain
    # TunnelEnv for evaluation -- the same split flat PPO and hPPO use.
    vec_env = TunnelVecEnv(num_envs=args.num_envs, config=env_config)
    eval_env = make_env(config=env_config)
    obs_dim = eval_env.observation_space.shape[0]
    obs_low = eval_env.observation_space.low
    obs_high = eval_env.observation_space.high

    # Solved-check setup: a fixed grid of initial conditions (not random
    # draws), so the oracle's optimal return for each point is solved once
    # here and reused on every later eval pass instead of being recomputed.
    # Any MinTimeSolver infeasibility raises here, at startup, rather than
    # mid-training.
    solved_grid = spawn_grid(eval_env, args.solved_grid_nx, args.solved_grid_ny)
    optimal_grid = precompute_optimal_grid(eval_env, solved_grid, solver=MinTimeSolver())
    optimal_returns = np.array([r.total_return for _, r in optimal_grid])
    print(f"Solved-check: precomputed optimal returns for {len(optimal_grid)} fixed initial "
          f"conditions (mean={optimal_returns.mean():.1f}, min={optimal_returns.min():.1f}, "
          f"max={optimal_returns.max():.1f}); solved-tolerance={args.solved_tolerance}")
    # See --early-stop-optimal-frac: derived from the oracle's mean optimal
    # return rather than a fixed value, so it stays meaningful whatever
    # contact_penalty balance this run is training against.
    early_stop_min_return = args.early_stop_optimal_frac * float(optimal_returns.mean())
    print(f"Early-stop return floor: {early_stop_min_return:.1f} "
          f"({args.early_stop_optimal_frac:.0%} of mean optimal)")
    consecutive_solved = 0
    solved_checkpoint_saved = False
    # See --early-stop-success-rate: counts consecutive qualifying
    # evaluations, reset by any evaluation that falls short.
    consecutive_high_success = 0

    goal_dim = 4 # delta x, delta y, v_x, v_y
    # The goal box. Unlike hPPO's, this one has to be sized against the
    # plant, because the two workers consume a goal in completely different
    # ways: hPPO's learned worker gets the goal *re-normalized* by the same
    # max_goal_bound before it reaches the network, so the constant cancels
    # and only the goal's direction survives -- an unreachable goal there is
    # simply a direction. This worker is a QP that tracks the goal as a hard
    # setpoint, at every stage of its horizon. Hand it a setpoint it cannot
    # reach and the quadratic tracking cost is minimized by driving at the
    # actuator/velocity limit toward it for the whole segment: the magnitude
    # stops meaning anything, and what is nominally a 4-D goal collapses to
    # "which direction, at full speed".
    #
    # That collapse is what kept this algorithm from solving the slalom (see
    # docs/benchmark.md). Inheriting hPPO's 10.0 made 100% of the trained
    # manager's delta_x goals and 75-90% of its delta_y goals unreachable
    # within a segment, so the manager could only command full-speed lateral
    # swings, re-aimed once every --manager-freq steps -- and the MPC's own
    # corridor constraint is looked up at the *current* p_x, so it does not
    # see a gate until the agent is already inside it. The resulting
    # bang-bang zig-zag crossed the gates at whatever lateral phase it
    # happened to be in: 1.3-1.8 wall contacts per episode on every seed,
    # most of them against the full-width wall between the two gates.
    #
    # Sized from the plant instead: over one segment of `manager_freq` steps
    # the displacement cannot exceed `v_max * manager_freq * dt` on either
    # axis, and the target velocity cannot exceed `v_max` (both are hard
    # box constraints inside MPCWorker's own QP, so anything past them is
    # unreachable by construction, not merely unlikely). Pass
    # --max-goal-bound 10 --max-goal-vel 2 to reproduce the previous
    # behaviour exactly.
    #
    # --goal-bound-slack, not a box pinned at exactly that radius: the two
    # axes want opposite things from it. On y the manager needs *resolution*,
    # which argues for the tightest box. On x saturation is the right answer
    # -- the optimal policy is full speed ahead -- and a box pinned at the
    # radius makes "full speed" require an action of exactly 1.0, the very
    # edge of the Beta's support. Swept on both scenarios at 3 seeds each
    # (see docs/goal-box-saturation.md section 6): a 1.0x box solved slalom
    # 8/9 but slowed the tunnel to 30-36k steps at ~210 return (oracle 213),
    # 1.5x solved slalom 3/3 and the tunnel in 24-30k at ~212.4, and 2.0x
    # started letting saturation back in (slalom 2/3, contacts returning).
    # 1.5 is the middle of that curve, not a fitted value.
    #
    # Note this is a *scale* fix and deliberately not the one
    # script_ppo_mpc_reach.py makes: that one reshapes the goal set into the
    # velocity-dependent zonotope actually reachable from the current state
    # (see `reachable_goal` in algorithms/ppo_mpc_reach/ppo_mpc_reach.py),
    # which is strictly more than getting the box's size right. The two stay
    # separate so the reach ablation still measures the reshaping.
    reach_per_segment = env_config.v_max * args.manager_freq * env_config.dt
    max_goal_bound = (args.max_goal_bound if args.max_goal_bound is not None
                      else args.goal_bound_slack * reach_per_segment)
    max_goal_vel = (args.max_goal_vel if args.max_goal_vel is not None
                    else env_config.v_max)
    goal_scale = np.array([max_goal_bound, max_goal_bound, max_goal_vel, max_goal_vel])
    _bound_src = ("explicit --max-goal-bound" if args.max_goal_bound is not None
                  else f"{args.goal_bound_slack:g}x segment-reachable")
    print(f"Goal box: position +-{max_goal_bound:.3f} m ({_bound_src}; "
          f"segment-reachable displacement = {reach_per_segment:.3f} m), "
          f"velocity +-{max_goal_vel:.3f} m/s (plant v_max = {env_config.v_max})")

    agent = PPOMPCAgent(
        obs_dim=obs_dim, goal_dim=goal_dim,
        lr_manager=args.learning_rate_manager,
        gamma=args.gamma, manager_freq=args.manager_freq,
        gae_lambda=args.gae_lambda, clip_coef=args.clip_coef,
        ent_coef_manager=args.ent_coef_manager, vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        # Carried into the checkpoint so a reloaded policy knows the
        # observation map and goal scale it was trained under.
        obs_low=obs_low, obs_high=obs_high, goal_scale=goal_scale,
        critic_lr_mult=args.critic_lr_mult,
        device=device,
        autotune_ent_coef=args.autotune_ent_coef,
        target_entropy_frac=args.target_entropy_frac,
        ent_coef_lr=args.ent_coef_lr,
        ent_coef_min=args.ent_coef_min,
        ent_coef_max=args.ent_coef_max,
        # Manager-collapse mitigations ported from hppo -- see each flag's
        # help above and HPPOAgent.ManagerActor's docstring in
        # algorithms/hppo/hppo.py.
        target_kl_manager=args.target_kl_manager,
        clip_vloss=args.clip_vloss,
        adv_std_floor_frac=args.adv_std_floor_frac,
        ret_rms_horizon_manager=args.ret_rms_horizon_manager,
    )

    # One base learning rate per parameter group (actor, then critic). The
    # anneal below scales each group against its own base; scaling every
    # group by args.learning_rate_manager would silently reset the critic to
    # the actor's rate on the first update.
    base_lrs = [group["lr"] for group in agent.manager_optimizer.param_groups]

    Q = np.diag([10.0, 10.0, 1.0, 1.0])
    R = np.diag([0.1, 0.1])
    # num_envs warm-start slots: each environment's solve resumes from its own
    # previous solution, not from whichever environment happened to solve last.
    # Evaluation runs one episode at a time and borrows slot 0.
    #
    # Every geometry arg here is read off env_config rather than restated as
    # a literal -- previously these were independently hand-typed numbers
    # that merely happened to match TunnelEnvConfig's defaults, with nothing
    # connecting the two, so changing one silently left the other behind.
    worker = MPCWorker(dt=env_config.dt, L=env_config.tunnel_length,
                       W=env_config.tunnel_width, v_max=env_config.v_max,
                       u_max=env_config.u_max, horizon=args.manager_freq,
                       Q=Q, R=R, backend=args.mpc_backend, num_envs=args.num_envs,
                       width_profile=env_config.width_profile)

    # Each column is sized for the worst case -- one segment per step, which
    # happens when every episode ends immediately.
    manager_buffer = ManagerVecRolloutBuffer(
        args.num_steps_per_env, args.num_envs, obs_dim, goal_dim, device)

    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_eval_return = -float("inf")

    # Every quantity the serial rollout held as a scalar is now a
    # per-environment array: the environments run independent episodes, so they
    # sit at different points in their manager cadence, carry different
    # partially-consumed goals, and accumulate different segment rewards.
    obs, _ = vec_env.reset(seed=args.seed)

    # Trackers
    c = args.manager_freq
    worker_step_in_c = np.zeros(args.num_envs, dtype=np.int64)
    accumulated_env_reward = np.zeros(args.num_envs, dtype=np.float32)
    accumulated_target_dist = np.zeros(args.num_envs, dtype=np.float64)
    ep_reward = np.zeros(args.num_envs, dtype=np.float64)
    ep_length = np.zeros(args.num_envs, dtype=np.int64)

    # Init Manager, for every environment at once
    manager_obs_norm = agent.normalize_obs(obs)
    with torch.no_grad():
        manager_action, manager_logprob, _, manager_value = agent.get_manager_action_and_value(
            torch.tensor(manager_obs_norm, dtype=torch.float32, device=device)
        )
    current_goal_phys = agent.scale_goal(manager_action.cpu().numpy())

    metrics_m = {}

    start_time = time.time()
    num_updates = args.total_timesteps // args.num_steps_worker
    global_step = 0
    print(f"num_envs={args.num_envs} num_steps_per_env={args.num_steps_per_env} "
          f"batch_size={args.num_steps_worker} num_updates={num_updates} "
          f"mpc_backend={args.mpc_backend}")

    for update in range(1, num_updates + 1):
        update_start_time = time.time()
        # The done flag of each environment's most recently *stored* manager
        # transition -- per environment, since their segments end at different
        # steps, and False until an environment has stored one at all.
        last_manager_done = np.zeros(args.num_envs, dtype=bool)

        completed_returns = []
        completed_lengths = []
        completed_successes = []
        completed_collision_counts = []
        completed_collision_impacts = []
        completed_target_dists = []
        collision_forced_replan_count = 0
        # The control arm of the worker termination-avoidance diagnostic that
        # script_hppo.py added (see the worker termination-avoidance block, now in
        # algorithms/hppo/hppo_train.py). There
        # the same metric detects a *learned* worker discovering that its
        # purely intrinsic reward makes crossing the goal line worth 0, and
        # stalling in front of it instead -- a worker objective that has come
        # apart from the manager's.
        #
        # What it measures here is different, and that difference is the
        # point of logging it under the same name. This worker is a QP with
        # no return of its own to protect: it tracks whatever goal it is
        # handed, so near the goal line this reads out the *manager's* intent,
        # not a worker's self-interest. It can therefore go negative early in
        # training, when the manager is still emitting bad goals -- what it
        # cannot do is go negative on a converged policy that the manager is
        # steering forward, which is exactly what the learned worker does.
        # One plot, two arms. There is no `worker_value_near_goal`
        # counterpart here because there is no worker critic to read one from.
        near_goal_ax = []

        # Linear LR decay: once the policy has converged a constant LR keeps
        # injecting noise into both heads off a shrinking advantage signal.
        if args.anneal_lr:
            frac = 1.0 - (1.0 - args.lr_floor_frac) * (update - 1.0) / num_updates
            for group, base_lr in zip(agent.manager_optimizer.param_groups, base_lrs):
                group["lr"] = frac * base_lr

        for step in range(args.num_steps_per_env):
            global_step += args.num_envs

            # Action selection (worker - MPC). One QP per environment: they
            # re-plan over different remaining horizons, so there is no single
            # batched solve to make.
            steps_left = args.manager_freq - worker_step_in_c
            worker_action_phys = worker.get_actions(obs, current_goal_phys, steps_left)

            # See near_goal_ax above. `obs` is the pre-step observation, so
            # this is "the acceleration commanded from a state within 1 m of
            # the goal line".
            near_goal_mask = obs[:, 0] >= (env_config.tunnel_length - 1.0)
            if np.any(near_goal_mask):
                near_goal_ax.append(worker_action_phys[near_goal_mask, 0])

            # Step environments (TunnelVecEnv auto-resets envs that finish)
            next_obs, env_reward, terminated, truncated, info = vec_env.step(worker_action_phys)
            done = terminated | truncated

            # The auto-reset is why this is not a mechanical substitution. For a
            # finished env `next_obs` is already the *next* episode's first
            # observation, so the goal decrement and the truncation bootstrap
            # must read the pre-reset state instead -- otherwise the goal is
            # decremented by a teleport back to the tunnel entrance.
            term_obs = next_obs.copy()
            if np.any(done):
                term_obs[done] = info["final_observation"][done]

            ep_reward += env_reward
            ep_length += 1
            accumulated_env_reward += (args.gamma ** worker_step_in_c) * env_reward

            # Update physical goals manually, against the pre-reset state
            next_goal_phys = current_goal_phys - (term_obs - obs)

            # Accumulate distance to target
            accumulated_target_dist += np.linalg.norm(current_goal_phys[:, :2], axis=-1)

            worker_step_in_c += 1

            # Which managers act now? A segment ends after c worker steps,
            # when its episode does, or (--replan-on-collision) the instant
            # the worker (MPC) contacts a wall -- a hit can clamp
            # position/zero velocity in a way that leaves the in-flight goal
            # unreachable for the rest of the segment, and the environments
            # do not agree on when any of this happens. Ported from
            # script_hppo.py.
            collision_forced = info["final_info"]["collision"] & args.replan_on_collision
            manager_act_now = (worker_step_in_c == c) | done | collision_forced
            if np.any(manager_act_now):
                # Handle bootstrapping for the manager on truncation. Its
                # segment is worker_step_in_c environment steps long, so the
                # cut-off continuation is discounted by gamma**that -- not by
                # the manager's own gamma**c, which assumes a full segment.
                # That exponent is now per environment too.
                manager_reward = accumulated_env_reward.copy()
                trunc_only = truncated & ~terminated
                manager_trunc = manager_act_now & trunc_only
                if np.any(manager_trunc):
                    with torch.no_grad():
                        true_next_manager_value = agent.get_manager_value(
                            torch.tensor(agent.normalize_obs(term_obs[manager_trunc]),
                                         dtype=torch.float32, device=device)
                        )
                    manager_reward[manager_trunc] += (
                        (args.gamma ** worker_step_in_c[manager_trunc])
                        * true_next_manager_value.cpu().numpy())

                # Collision-forced replans are a third kind of segment
                # boundary, one the fixed gamma**manager_freq recursion in
                # compute_manager_returns_and_advantage doesn't know about --
                # that recursion is only correct across a done=False boundary
                # because today the sole way to reach one is the c-step
                # counter, so every such gap really is manager_freq steps.
                # Handled like the truncation case above: fold the
                # correctly-discounted continuation into this reward and
                # close the transition with a done below (a pseudo-done, the
                # episode itself continues). Ported from script_hppo.py.
                collision_forced_continue = collision_forced & ~done
                if np.any(collision_forced_continue):
                    collision_forced_replan_count += int(np.sum(collision_forced_continue))
                    with torch.no_grad():
                        true_next_manager_value_collision = agent.get_manager_value(
                            torch.tensor(agent.normalize_obs(next_obs[collision_forced_continue]),
                                         dtype=torch.float32, device=device)
                        )
                    manager_reward[collision_forced_continue] += (
                        (args.gamma ** worker_step_in_c[collision_forced_continue])
                        * true_next_manager_value_collision.cpu().numpy())

                manager_dones_to_store = done.copy()
                manager_dones_to_store[collision_forced_continue] = True

                manager_buffer.add(
                    manager_act_now,
                    manager_obs_norm,
                    manager_action,
                    manager_logprob,
                    manager_reward,
                    manager_value,
                    manager_dones_to_store.astype(np.float32)
                )

                last_manager_done[manager_act_now] = done[manager_act_now]
                worker_step_in_c[manager_act_now] = 0
                accumulated_env_reward[manager_act_now] = 0.0

                # A fresh goal for every environment whose segment just ended.
                # The serial loop split this in two -- re-plan at the next
                # observation, or re-plan at the reset observation after an
                # episode ended -- but under an auto-resetting vector env both
                # are `next_obs`, since a done environment has already been
                # reset into the state the new segment starts from. `done`
                # implies `manager_act_now`, so no environment is left holding
                # a goal from a finished episode.
                sel = manager_act_now
                sel_t = torch.as_tensor(sel, device=device)
                next_obs_norm = agent.normalize_obs(next_obs)
                with torch.no_grad():
                    new_action, new_logprob, _, new_value = agent.get_manager_action_and_value(
                        torch.tensor(next_obs_norm[sel], dtype=torch.float32, device=device)
                    )
                manager_obs_norm[sel] = next_obs_norm[sel]
                manager_action[sel_t] = new_action
                manager_logprob[sel_t] = new_logprob
                manager_value[sel_t] = new_value
                current_goal_phys[sel] = agent.scale_goal(new_action.cpu().numpy())

            for i in np.flatnonzero(done):
                completed_returns.append(ep_reward[i])
                completed_lengths.append(ep_length[i])
                completed_successes.append(bool(info["final_info"]["is_success"][i]))
                completed_collision_counts.append(int(info["final_info"]["collision_count"][i]))
                completed_collision_impacts.extend(info["final_info"]["collision_impacts"][i])
                completed_target_dists.append(accumulated_target_dist[i])
                ep_reward[i] = 0.0
                ep_length[i] = 0
                accumulated_target_dist[i] = 0.0

            # Environments still mid-segment carry their partially consumed
            # goal forward; the rest were just given a fresh one above.
            carry = ~manager_act_now
            current_goal_phys[carry] = next_goal_phys[carry]
            obs = next_obs

        # --- Manager PPO Update (End of Rollout Phase) ---
        with torch.no_grad():
            # `manager_value` is the value of each environment's segment
            # currently in flight, whose start is exactly the state that
            # environment's last stored transition led to -- already the
            # per-column bootstrap the ragged GAE wants.
            agent.compute_manager_returns_and_advantage(
                manager_buffer, manager_value,
                next_done=torch.as_tensor(last_manager_done.astype(np.float32), device=device)
            )

        manager_minibatch_size = max(1, manager_buffer.total_steps // args.num_minibatches)
        metrics_m = agent.update_manager(manager_buffer, manager_minibatch_size, args.update_epochs)

        # Logging
        sps = int(args.num_steps_worker / (time.time() - update_start_time))
        metrics = {**metrics_m}
        # Group 0 is the actor, group 1 the critic (see PPOMPCAgent.__init__).
        metrics["charts/manager_lr"] = agent.manager_optimizer.param_groups[0]["lr"]
        metrics["charts/manager_critic_lr"] = agent.manager_optimizer.param_groups[-1]["lr"]
        metrics["charts/SPS"] = sps
        metrics["charts/num_episodes"] = len(completed_returns)
        metrics["charts/collision_forced_replans"] = collision_forced_replan_count
        if near_goal_ax:
            metrics["charts/worker_ax_near_goal"] = float(np.mean(np.concatenate(near_goal_ax)))
            metrics["charts/near_goal_samples"] = float(sum(a.size for a in near_goal_ax))
        metrics["charts/collision_forced_replan_rate"] = (
            collision_forced_replan_count / max(manager_buffer.total_steps, 1))

        log_line = (f"update={update} global_step={global_step} SPS={sps} "
                    f"m_ev={metrics['manager/explained_variance']:.3f} "
                    f"m_v_bias={metrics['manager/value_bias']:.1f} "
                    # Manager-collapse diagnostics (see ManagerActor's
                    # docstring in algorithms/hppo/hppo.py and
                    # algorithms/ppo_mpc/ppo_mpc.py): printed inline, not
                    # just logged to W&B, so the update where these spike or
                    # crater is visible without leaving the console.
                    f"m_adv_std={metrics['manager/adv_std_raw']:.4f} "
                    f"m_kl_max={metrics['manager/approx_kl_max']:.4f} "
                    f"m_ratio_max={metrics['manager/ratio_max_dev']:.3f}")
        if "charts/worker_ax_near_goal" in metrics:
            log_line += f" w_ax@goal={metrics['charts/worker_ax_near_goal']:+.2f}"
        if completed_returns:
            mean_return = float(np.mean(completed_returns))
            collision_count_mean = float(np.mean(completed_collision_counts))
            metrics["charts/episodic_return"] = mean_return
            metrics["charts/episodic_length"] = float(np.mean(completed_lengths))
            metrics["charts/success_rate"] = float(np.mean(completed_successes))
            metrics["charts/collision_count_mean"] = collision_count_mean
            metrics["charts/episodic_target_dist_sum"] = float(np.mean(completed_target_dists))
            log_line += (f" return={mean_return:.1f} "
                         f"length={np.mean(completed_lengths):.0f} "
                         f"success_rate={np.mean(completed_successes):.2f} "
                         f"collisions/ep={collision_count_mean:.2f} "
                         f"(n={len(completed_returns)})")
            if completed_collision_impacts:
                # Individual per-contact penalties (not summed), so both the
                # typical and the single worst contact this update are visible.
                metrics["charts/collision_impact_mean"] = float(np.mean(completed_collision_impacts))
                metrics["charts/collision_impact_worst"] = float(np.min(completed_collision_impacts))
        print(log_line)
        if args.track:
            wandb.log(metrics, step=global_step)

        manager_buffer.reset()
        
        # Evaluation
        if update % args.eval_freq == 0:
            agent.manager_actor.eval()
            agent.manager_critic.eval()
            eval_returns = []
            eval_lengths = []
            eval_target_dist_sums = []
            eval_successes = []
            eval_collision_counts = []
            eval_collision_impacts = []
            for i in range(args.eval_episodes):
                eval_obs, info_eval = eval_env.reset(seed=args.seed + i)
                eval_ep_reward = 0
                eval_ep_length = 0
                eval_target_dist_sum = 0.0
                done_eval = False

                # Every name in this block is eval-local on purpose. It used to
                # reuse `current_goal_phys` and `worker_step_in_c`, the same
                # names the rollout carries its in-flight segment in, so each
                # evaluation silently overwrote the training state with whatever
                # its last episode ended on and the rollout resumed against a
                # goal from a different episode -- a live bug in the
                # single-environment version, which vectorizing surfaced.
                eval_manager_obs = torch.tensor(
                    agent.normalize_obs(eval_obs), dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    eval_manager_action, _, _, _ = agent.get_manager_action_and_value(eval_manager_obs, deterministic=True)
                eval_goal_phys = agent.scale_goal(eval_manager_action.cpu().numpy()[0])
                eval_step_in_c = 0

                while not done_eval:
                    steps_left = args.manager_freq - eval_step_in_c
                    # Evaluation is one episode at a time, so it borrows the
                    # first environment's warm-start slot.
                    eval_action_phys = worker.get_action(
                        eval_obs, eval_goal_phys, steps_left=steps_left, env_index=0)

                    eval_next_obs, reward, terminated_eval, truncated_eval, info_eval = eval_env.step(eval_action_phys)
                    eval_ep_reward += reward
                    eval_ep_length += 1
                    done_eval = terminated_eval or truncated_eval

                    eval_next_goal_phys = eval_goal_phys - (eval_next_obs - eval_obs)

                    eval_target_dist_sum += np.linalg.norm(eval_goal_phys[:2])

                    eval_step_in_c += 1
                    if eval_step_in_c == args.manager_freq and not done_eval:
                        eval_manager_obs = torch.tensor(
                            agent.normalize_obs(eval_next_obs), dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        with torch.no_grad():
                            eval_manager_action, _, _, _ = agent.get_manager_action_and_value(eval_manager_obs, deterministic=True)
                        eval_goal_phys = agent.scale_goal(eval_manager_action.cpu().numpy()[0])
                        eval_step_in_c = 0
                    elif not done_eval:
                        eval_goal_phys = eval_next_goal_phys

                    eval_obs = eval_next_obs

                eval_returns.append(eval_ep_reward)
                eval_lengths.append(eval_ep_length)
                eval_target_dist_sums.append(eval_target_dist_sum)
                eval_successes.append(bool(info_eval["is_success"]))
                # collision_count/collision_impacts are cumulative over the
                # whole episode (see TunnelEnv.step), so the last step's info
                # already carries every contact the episode had.
                eval_collision_counts.append(info_eval["collision_count"])
                eval_collision_impacts.extend(info_eval["collision_impacts"])

            mean_eval_return = float(np.mean(eval_returns))
            success_rate = float(np.mean(eval_successes))
            collision_count_mean = float(np.mean(eval_collision_counts))
            print(f"Evaluation at update {update}: return={mean_eval_return:.2f}, length={np.mean(eval_lengths):.2f}, "
                  f"success_rate={success_rate:.2f}, collisions/ep={collision_count_mean:.2f}")
            eval_metrics = {
                "eval/episodic_return": mean_eval_return,
                "eval/episodic_length": np.mean(eval_lengths),
                "eval/success_rate": success_rate,
                "eval/collision_count_mean": collision_count_mean,
                "eval/episodic_target_dist_sum": np.mean(eval_target_dist_sums),
            }
            if eval_collision_impacts:
                eval_metrics["eval/collision_impact_mean"] = float(np.mean(eval_collision_impacts))
                eval_metrics["eval/collision_impact_worst"] = float(np.min(eval_collision_impacts))
            if args.track:
                wandb.log(eval_metrics, step=global_step)

            if mean_eval_return > best_eval_return:
                best_eval_return = mean_eval_return
                agent.save(os.path.join(checkpoint_dir, "best.pt"))

            # Solved-check: the same fixed grid, deterministic policy action
            # (matching the random eval above) on every pass, compared
            # point-by-point against the oracle's precomputed optimal return
            # for that exact starting position -- see algorithms/solved_check.py.
            #
            # A *factory*, not a single shared closure: the manager-replan
            # cadence (state["goal"]/state["step_in_c"]) is per-episode state,
            # and check_solved calls make_policy_fn() fresh right after each
            # grid point's own env.reset() -- exactly like this eval loop's
            # own eval_goal_phys/eval_step_in_c above, just reattributed from
            # "post-step, before the next loop iteration" to "start of the
            # next policy_fn call" (same information either way). `obs` here
            # is the raw, un-normalized state -- what worker.get_action and
            # the goal decay both operate on; only the manager forward pass
            # is normalized.
            def make_policy_fn():
                state = {}

                def policy_fn(obs):
                    obs_norm = agent.normalize_obs(obs)
                    if "goal" not in state:
                        manager_obs = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
                        with torch.no_grad():
                            manager_action, _, _, _ = agent.get_manager_action_and_value(
                                manager_obs, deterministic=True)
                        state["goal"] = agent.scale_goal(manager_action.cpu().numpy()[0])
                        state["step_in_c"] = 0
                    else:
                        decayed_goal = state["goal"] - (obs - state["prev_obs"])
                        state["step_in_c"] += 1
                        if state["step_in_c"] == args.manager_freq:
                            manager_obs = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
                            with torch.no_grad():
                                manager_action, _, _, _ = agent.get_manager_action_and_value(
                                    manager_obs, deterministic=True)
                            state["goal"] = agent.scale_goal(manager_action.cpu().numpy()[0])
                            state["step_in_c"] = 0
                        else:
                            state["goal"] = decayed_goal
                    state["prev_obs"] = obs.copy()
                    steps_left = args.manager_freq - state["step_in_c"]
                    # Evaluation is one episode at a time, so it borrows the
                    # first environment's warm-start slot -- same as the
                    # random eval above.
                    return worker.get_action(obs, state["goal"], steps_left=steps_left, env_index=0)

                return policy_fn

            solved_result = check_solved(make_policy_fn, eval_env, optimal_grid, args.solved_tolerance)
            print(f"Solved-check at update {update}: solved={solved_result.solved} "
                  f"worst_gap={solved_result.worst_gap:.2f} at "
                  f"p_x0={solved_result.worst_point[0]:.2f} p_y0={solved_result.worst_point[1]:.2f} "
                  f"(consecutive={consecutive_solved})")
            if args.track:
                wandb.log({
                    "solved/is_solved": float(solved_result.solved),
                    "solved/worst_gap": solved_result.worst_gap,
                    "solved/mean_gap": float(solved_result.gaps.mean()),
                }, step=global_step)

            consecutive_solved = consecutive_solved + 1 if solved_result.solved else 0

            agent.manager_actor.train()
            agent.manager_critic.train()

            # See --early-stop-success-rate. Checked after best.pt so the
            # qualifying evaluation is always banked before we might stop.
            if (success_rate >= args.early_stop_success_rate
                    and mean_eval_return >= early_stop_min_return):
                consecutive_high_success += 1
            else:
                consecutive_high_success = 0
            if consecutive_high_success >= args.early_stop_patience:
                print(f"Early stop at update {update}: eval/success_rate >= "
                      f"{args.early_stop_success_rate} and eval/episodic_return "
                      f">= {early_stop_min_return:.1f} ({args.early_stop_optimal_frac:.0%} "
                      f"of mean optimal) for {consecutive_high_success} consecutive evaluations")
                if args.track:
                    wandb.log({"charts/early_stopped_at_update": update}, step=global_step)
                break

            # Independent of --early-stop-success-rate above: either one can
            # stop training.
            if consecutive_solved >= args.solved_consecutive:
                # Saved once, the first time the criterion holds, regardless
                # of --solved-early-stop -- marks the moment the policy
                # became near-optimal even on a run left to run to its full
                # --total-timesteps budget for a coherent cross-seed plot.
                if not solved_checkpoint_saved:
                    print(f"Solved criterion held for {consecutive_solved} consecutive evals "
                          f"(tolerance={args.solved_tolerance}) at global_step={global_step}.")
                    agent.save(os.path.join(checkpoint_dir, "solved.pt"))
                    solved_checkpoint_saved = True
                    if args.track:
                        wandb.log({"solved/first_solved_step": global_step}, step=global_step)
                if args.solved_early_stop:
                    print("--solved-early-stop is on -- stopping training early.")
                    break

    agent.save(os.path.join(checkpoint_dir, "final.pt"))
    print(f"Saved checkpoints to {checkpoint_dir}")

    eval_env.close()
    if args.track:
        wandb.finish()
