"""The PPO+MPC training loop, shared by every scenario's script_ppo_mpc.py.

Rebuilt on 2026-09-28 from hPPO's loop of that date (algorithms/hppo/
hppo_train.py): the same manager, rollout, segment bookkeeping, GAE,
evaluation and stopping rules, with hPPO's learned worker replaced by the
robust tube MPC of algorithms/tube_mpc.py. Each scenario's script is a thin
wrapper that describes its scenario in a `Scenario` and hands it to `main`,
as hPPO's and flat PPO's do.

It replaces two ~900-line scripts (scenarios/{slalom,tunnel}/scripts/
script_ppo_mpc.py) that had fallen behind every simplification hPPO went
through, and whose worker, MPCWorker (algorithms/mpc_worker.py), saw the
corridor only at its current position. What changed against them, beyond
the worker (see tube_mpc.py for that):

- The manager is hPPO's, update for update (see PPOMPCAgent in ppo_mpc.py):
  no entropy autotuner, clip_vloss, target_kl, advantage-std floor,
  configurable ret_rms memory or vf_coef.
- The goal is hPPO's 2-D displacement instead of (delta_x, delta_y, v_x,
  v_y), with the goal box still sized to the plant (--max-goal-bound).
- The worker plans over a fixed, receding horizon (--mpc-horizon) instead of
  the rest of the manager's segment. The old worker's horizon shrank from c
  to 1 as the segment ran out, which a terminal set at rest makes
  infeasible: a plan cannot stop within one step from top speed.
- --replan-on-collision is gone, as it went from hPPO (see the
  manager_act_now comment in the rollout), and so is --lr-floor-frac. The
  learning rate is constant by default (--anneal-lr false), as hPPO's.
- hPPO's rollout and evaluation cadence: 2048 steps per update, not 2000;
  an evaluation of 20 episodes every 5 updates, not 10 every update (see
  docs/benchmark.md's protocol note for how the old cadence biased the
  old PPO+MPC's steps-to-solve).
- --early-stop-success-rate is off by default, as hPPO's since 2026-09-24.
- A bounded disturbance can be switched on (--noise-bound-p/-v), and the
  worker is robust to it. Flat PPO and hPPO take the same two flags, so the
  three can be compared on the same disturbed environment.

Imported by bare name, like `ppo_mpc`, once algorithms/ppo_mpc and
algorithms/ are on sys.path -- each script_ppo_mpc.py puts them there.
"""

import argparse
import os
import random
import time
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import torch
import wandb

from ppo_mpc import PPOMPCAgent, ManagerVecRolloutBuffer
from tube_mpc import TubeMPCWorker, CANDIDATE, EMERGENCY
from optimal_solver import spawn_grid, precompute_optimal_grid, MinTimeSolver
from solved_check import check_solved
from metrics_log import MetricsLog


# `type=bool` would be a no-op for boolean flags: argparse applies it to the
# *string*, and bool("False") is True.
def _str2bool(x):
    return x.lower() in ['true', '1', 't', 'y', 'yes']


def _optional_float(x):
    return None if x.lower() == "none" else float(x)


@dataclass(frozen=True)
class Scenario:
    """Everything the training loop needs to know about one scenario."""
    name: str                   # "slalom" / "tunnel": exp-name default and help texts
    env_cls: type               # plain env, for evaluation, the solved-check and the oracle
    vec_env_cls: type           # batched, auto-resetting env, for rollouts
    make_env_config: Callable   # **overrides -> env config
    total_timesteps: int        # default training budget
    wandb_project: str
    env_u_max_help: str         # scenario-specific: each is a different arm of the regime study
    script_dir: str             # checkpoints are written relative to the scenario's script


def parse_args(scenario):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=f"ppo_mpc_{scenario.name}",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=_str2bool, default=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=_str2bool, default=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", action="store_true",
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default=scenario.wandb_project,
        help="the wandb's project name")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
        help="directory (relative to the scenario's script) to save model checkpoints in")
    parser.add_argument("--save-eval-checkpoints", type=_str2bool, default=False,
        help="if toggled, also save the agent at every evaluation, as "
             "eval_<global_step>.pt next to best.pt and final.pt")

    # --- the environment --------------------------------------------------
    parser.add_argument("--env-u-max", type=float, default=None,
        help=scenario.env_u_max_help)
    parser.add_argument("--contact-penalty", type=float, default=scenario.make_env_config().contact_penalty,
        help="penalty for every step in contact with a wall. Defaults to the env "
             "config's own, so omitting it reproduces the environment every other "
             "script trains against -- see envs/config.py")
    parser.add_argument("--noise-bound-p", type=float, default=0.0,
        help="half-width, in metres, of the uniform disturbance added to each "
             "position every step (the env config's noise_bound_p). 0.0, the "
             "default, is the deterministic environment. The worker's tube is "
             "designed for this same box W unless a checkpoint says otherwise. "
             "0.005 with --noise-bound-v 0.05 is the level chosen for the "
             "disturbed experiments (2026-09-28): a disturbance of ~20%% of the "
             "actuator's authority, which gives a tube of +-8 cm and +-0.21 m/s "
             "and plans at <= 0.99 m/s. Flat PPO and hPPO take the same flag")
    parser.add_argument("--noise-bound-v", type=float, default=0.0,
        help="half-width, in m/s, of the uniform disturbance added to each "
             "velocity every step (the env config's noise_bound_v); see "
             "--noise-bound-p. The tube needs both bounds > 0, or both 0")

    # --- the manager ------------------------------------------------------
    parser.add_argument("--total-timesteps", type=int, default=scenario.total_timesteps,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate-manager", type=float, default=3e-4,
        help="the learning rate of the manager's actor. Its critic runs at "
             "--critic-lr-mult times this")
    parser.add_argument("--critic-lr-mult", type=float, default=3.0,
        help="the critic's learning rate as a multiple of the actor's")
    parser.add_argument("--anneal-lr", type=_str2bool, default=False,
        help="if toggled, the learning rates decay linearly to 0 over training, "
             "in both parameter groups. Off by default, as hPPO's (see its "
             "--anneal-lr for the measurements). The old PPO+MPC annealed by "
             "default, with a --lr-floor-frac that no run used")
    parser.add_argument("--manager-freq", type=int, default=10,
        help="c: the number of steps the manager's goal is held for. Also fixes "
             "the manager's discount, gamma**c")
    # --- the goal box -----------------------------------------------------
    #
    # hPPO's box is 10 m and costs nothing there: hPPO re-normalizes the goal
    # by the same constant before its worker network sees it, so a far goal
    # is simply a direction. An MPC takes the goal as a setpoint. Inheriting
    # hPPO's 10.0 made 100% of the old PPO+MPC manager's delta_x goals and
    # 75-90% of its delta_y goals unreachable within a segment; the QP then
    # drove at the actuator limit toward them, and the manager could only
    # command full-speed lateral swings, re-aimed once per segment. With a
    # worker that could not see a gate until inside it, that bang-bang
    # zig-zag crossed the gates at whatever lateral phase it happened to be
    # in: 1.3-1.8 wall contacts per episode on every seed, and the one
    # algorithm that never solved the slalom (docs/benchmark.md, section 4).
    #
    # Sized from the plant instead: over one segment the displacement cannot
    # exceed v_max * manager_freq * dt per axis. --goal-bound-slack times
    # that, not exactly that, because the two axes want opposite things: on
    # y the manager needs resolution, which argues for the tightest box; on
    # x the optimal policy is full speed ahead, and a box pinned at the reach
    # makes "full speed" an action of exactly 1.0, the edge of the Beta's
    # support. Swept for the old worker on both scenarios at 3 seeds each
    # (docs/goal-box-saturation.md, section 6): 1.0x solved the slalom 8/9
    # but slowed the tunnel, 1.5x solved both, and 2.0x let the saturation
    # back in. 1.5 is the middle of that curve, not a fitted value, and it
    # has not been re-swept for the tube MPC.
    #
    # The tube MPC is less exposed to a far goal than the old QP was: it
    # sees every gate within its horizon, and a goal past a wall is only a
    # cost. It is exposed the other way: a goal straight ahead of a gate is
    # a local minimum it cannot leave by itself (the note's remark 4.6). With
    # the goal held 1.8 m ahead of it and the manager replaced by nothing,
    # it stops in front of the slalom's second gate; with goals placed in
    # each gate's opening it finishes in 78 steps at a return of 1013.7
    # without contacts (tests/test_tube_mpc.py). Finding those goals is the
    # manager's whole job here.
    parser.add_argument("--max-goal-bound", type=float, default=None,
        help="half-width of the manager's goal box in metres: its [-1, 1]^2 "
             "action is scaled by this into a physical (delta_x, delta_y) "
             "displacement. Default: --goal-bound-slack times the displacement "
             "reachable in one segment, v_max * --manager-freq * dt (1.8 m at "
             "the defaults). See the comment above this flag")
    parser.add_argument("--goal-bound-slack", type=float, default=1.5,
        help="--max-goal-bound's default, as a multiple of the segment-reachable "
             "displacement. Ignored when --max-goal-bound is given")

    # --- the worker (algorithms/tube_mpc.py) ------------------------------
    parser.add_argument("--mpc-horizon", type=int, default=10,
        help="N, the tube MPC's prediction horizon, fixed and receding. Every "
             "plan must come to rest by its end (the terminal set), so N below "
             "the stopping time printed at startup caps the planned speed. 10 "
             "steps is 1 s and ~1 m of travel, enough to see a gate before "
             "reaching it. The MIQP has ~20 N variables, over the size-limited "
             "Gurobi licence's 200 for quadratic models once the disturbance "
             "is on; an academic licence has no limit")
    parser.add_argument("--mpc-q-pos", type=float, default=10.0,
        help="stage weight on each position error, Q = diag(q_pos, q_pos, q_vel, "
             "q_vel). With --mpc-q-vel and --mpc-r it also sets the ancillary "
             "gain K and the terminal cost P (LQR). The defaults are the old "
             "MPCWorker's")
    parser.add_argument("--mpc-q-vel", type=float, default=1.0,
        help="stage weight on each velocity (the target is at rest)")
    parser.add_argument("--mpc-r", type=float, default=0.1,
        help="stage weight on each nominal input, R = r I")
    parser.add_argument("--mpc-rho", type=float, default=1e-3,
        help="safety margin in metres around every obstacle and wall. Must exceed "
             "big-M * Gurobi's IntFeasTol (~1.3e-4 here); see tube_mpc.py")
    parser.add_argument("--mpc-rpi-eps", type=float, default=1e-2,
        help="algorithm 1's tolerance: the tube Z lies within this of the minimal "
             "RPI set, in the infinity norm")
    parser.add_argument("--mpc-mip-gap", type=float, default=1e-4,
        help="Gurobi's relative MIP gap")
    parser.add_argument("--mpc-work-limit", type=_optional_float, default=None,
        help="Gurobi's deterministic work limit per solve, or none. The shifted "
             "candidate covers a solve it cuts short (algorithm 3); a work "
             "limit, unlike a time limit, keeps the run reproducible")

    # --- rollout and update -----------------------------------------------
    parser.add_argument("--num-envs", type=int, default=8,
        help=f"the number of parallel {scenario.name} environments to collect "
             "rollouts from; each has its own goal, manager cadence, segment "
             "accumulator and worker slot")
    parser.add_argument("--num-steps-worker", type=int, default=2048,
        help="environment steps per rollout, summed over all --num-envs "
             "environments; must be divisible by --num-envs. hPPO's name and "
             "default, kept for the hierarchical family's sake (the old PPO+MPC "
             "used 2000)")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the environment-step discount factor; the manager is discounted "
             "at gamma**manager_freq")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches-manager", type=int, default=4,
        help="the number of near-equal minibatches the manager's rollout is split "
             "into, whatever its size; hPPO's flag and default")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the manager")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef-manager", type=float, default=0.01,
        help="coefficient of the manager's entropy bonus, fixed for the whole run")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")

    # --- evaluation and stopping, as hPPO's -------------------------------
    parser.add_argument("--eval-freq", type=int, default=5,
        help="evaluate the agent every eval_freq updates")
    parser.add_argument("--eval-episodes", type=int, default=20,
        help="number of episodes to evaluate the agent")
    parser.add_argument("--solved-early-stop", type=_str2bool, default=True,
        help="if toggled, stop training once the solved criterion below is met. "
             "The check, its logging and the one-time solved.pt happen either "
             "way. On a disturbed environment the criterion is measured against "
             "the undisturbed oracle, which a robust controller cannot match "
             "(it plans slower and wider), so such runs should set this false "
             "and read the gap instead")
    parser.add_argument("--solved-tolerance", type=float, default=5.0,
        help="the solved criterion: return within this many reward units of the "
             "oracle's optimal return (algorithms/optimal_solver.py) at every "
             "point of the fixed evaluation grid")
    parser.add_argument("--solved-grid-nx", type=int, default=5,
        help="number of p_x0 points in the solved-check's grid")
    parser.add_argument("--solved-grid-ny", type=int, default=5,
        help="number of p_y0 points in the solved-check's grid")
    parser.add_argument("--solved-consecutive", type=int, default=2,
        help="consecutive evaluations the solved criterion must hold for")
    parser.add_argument("--early-stop-success-rate", type=float, default=None,
        help="stop once eval/success_rate >= this and eval/episodic_return >= "
             "--early-stop-optimal-frac of the oracle, for --early-stop-patience "
             "consecutive evaluations. Off by default, as hPPO's since 2026-09-24 "
             "(the old PPO+MPC defaulted to 1.0; see hPPO's help for the history)")
    parser.add_argument("--early-stop-optimal-frac", type=float, default=0.95,
        help="the return floor of --early-stop-success-rate, as a fraction of the "
             "oracle's mean optimal return")
    parser.add_argument("--early-stop-patience", type=int, default=3,
        help="consecutive qualifying evaluations --early-stop-success-rate needs")
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error(f"--num-envs must be > 0, got {args.num_envs}")
    if args.num_steps_worker % args.num_envs != 0:
        parser.error(
            f"--num-steps-worker ({args.num_steps_worker}) must be divisible by "
            f"--num-envs ({args.num_envs})")
    args.num_steps_per_env = args.num_steps_worker // args.num_envs
    return args


def mpc_settings_from_args(args):
    """The worker's settings, in TubeMPCWorker's own names."""
    return dict(horizon=args.mpc_horizon, q_pos=args.mpc_q_pos, q_vel=args.mpc_q_vel,
                r=args.mpc_r, noise_bound_p=args.noise_bound_p, noise_bound_v=args.noise_bound_v,
                rho=args.mpc_rho, rpi_eps=args.mpc_rpi_eps, mip_gap=args.mpc_mip_gap,
                work_limit=args.mpc_work_limit)


def make_policy_fn(agent, worker, slot=0, goal_trace=None):
    """The controller evaluation scores, as a fresh closure per episode: one
    raw observation in, the worker's action out. The manager issues a
    deterministic goal at the episode's first step and every manager_freq
    steps after it, and the goal decays by the agent's displacement in
    between, so the absolute point the worker steers to stays put for the
    segment -- the rollout's cadence and goal transition exactly, without
    the sampling. hPPO's make_policy_fn with the worker swapped.

    A *factory*, because the cadence and the worker's last plan are
    per-episode state: building one resets `worker`'s `slot`, which must
    not be a slot the rollout uses (see TubeMPCWorker's docstring). Call it
    right after each env.reset(), as algorithms/solved_check.py does.

    `goal_trace`, if given, gets one `(replanned, goal)` pair per step, as
    hPPO's does.
    """
    state = {}
    worker.reset(slot)

    def policy_fn(obs):
        obs = np.asarray(obs, dtype=np.float64)
        pos = obs[:2].copy()
        if state:
            state["step_in_c"] += 1
        replanned = not state or state["step_in_c"] == agent.manager_freq
        if replanned:
            with torch.no_grad():
                manager_obs = torch.tensor(agent.normalize_obs(obs), dtype=torch.float32,
                                           device=agent.device).unsqueeze(0)
                state["goal"] = agent.scale_goal(agent.manager_act(manager_obs).cpu().numpy()[0])
            state["step_in_c"] = 0
        else:
            state["goal"] = state["goal"] - (pos - state["pos"])
        state["pos"] = pos
        if goal_trace is not None:
            goal_trace.append((replanned, np.array(state["goal"], dtype=float)))
        action, _ = worker.act(obs, pos + state["goal"], slot=slot)
        return action.astype(np.float32)

    return policy_fn


def load_agent(path, env, device="cpu"):
    """A trained controller rebuilt from the checkpoint at `path`: the manager,
    and a one-slot worker with the settings it was trained with -- including
    its disturbance model W, whatever `env`'s own disturbance is. `env` is an
    instance of the environment it was trained on (plant and corridor).
    Returns (agent, worker)."""
    agent = PPOMPCAgent(obs_dim=env.observation_space.shape[0], goal_dim=2, device=device)
    agent.load(path)
    for net in (agent.manager_actor, agent.manager_critic):
        net.eval()
    worker = TubeMPCWorker.from_env(env, num_slots=1, **agent.mpc_settings)
    return agent, worker


def main(scenario):
    train(parse_args(scenario), scenario)


def train(args, scenario):
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"

    # --env-u-max is a *regime-study* knob and nothing else. Every other
    # field of the environment is fixed so it is identical across the
    # architectures this repo compares; this one exists because "when does
    # reachability analysis actually pay?" is a question about the plant.
    # It sets the agility ratio rho = v_max / (u_max * manager_freq * dt):
    # how much of v_max the actuator can add or remove within one segment.
    # At the canonical u_max = 2.5, rho = 0.48, and the displacement reachable
    # in a segment stays roughly centred on the current position; as u_max
    # falls it narrows and re-centres on the free-drift outcome. See
    # docs/goal-box-saturation.md, section 8, and docs/reachability-regime-
    # study.md. (Moved here from the old script_ppo_mpc.py, whose comment it
    # was.)
    env_config = scenario.make_env_config(
        contact_penalty=args.contact_penalty,
        noise_bound_p=args.noise_bound_p,
        noise_bound_v=args.noise_bound_v,
        **({} if args.env_u_max is None else {"u_max": args.env_u_max}),
    )
    if args.env_u_max is not None:
        _T = args.manager_freq * env_config.dt
        print(f"REGIME STUDY: env u_max overridden to {env_config.u_max} "
              f"(canonical {scenario.make_env_config().u_max}); agility ratio rho = "
              f"v_max/(u_max*T) = {env_config.v_max / (env_config.u_max * _T):.2f}")

    if args.track:
        wandb.init(project=args.wandb_project_name, sync_tensorboard=False, config=vars(args),
                   name=run_name, monitor_gym=False, save_code=True)
        # save_code only captures the thin wrapper; log the code that trains.
        algorithms_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        wandb.run.log_code(
            root=algorithms_dir,
            include_fn=lambda path: os.path.basename(path) in ("ppo_mpc.py", "ppo_mpc_train.py", "tube_mpc.py"),
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # A batched, auto-resetting vector env for rollouts and a plain env for
    # evaluation, as hPPO's. Evaluation runs on the same, possibly disturbed,
    # environment, with fixed seeds, so every pass sees the same
    # disturbances. The oracle gets its own undisturbed copy: it replays
    # open-loop actions, which a disturbance would knock off course.
    vec_env = scenario.vec_env_cls(num_envs=args.num_envs, config=env_config)
    eval_env = scenario.env_cls(config=env_config)
    oracle_env = scenario.env_cls(config=replace(env_config, noise_bound_p=0.0, noise_bound_v=0.0))
    obs_dim = eval_env.observation_space.shape[0]
    goal_dim = 2  # delta x, delta y

    solved_grid = spawn_grid(oracle_env, args.solved_grid_nx, args.solved_grid_ny)
    optimal_grid = precompute_optimal_grid(oracle_env, solved_grid, solver=MinTimeSolver())
    optimal_returns = np.array([r.total_return for _, r in optimal_grid])
    print(f"Solved-check: precomputed optimal returns for {len(optimal_grid)} fixed initial "
          f"conditions (mean={optimal_returns.mean():.1f}, min={optimal_returns.min():.1f}, "
          f"max={optimal_returns.max():.1f}); solved-tolerance={args.solved_tolerance}")
    early_stop_min_return = args.early_stop_optimal_frac * float(optimal_returns.mean())
    if args.early_stop_success_rate is not None:
        print(f"Early-stop return floor: {early_stop_min_return:.1f} "
              f"({args.early_stop_optimal_frac:.0%} of mean optimal)")
    consecutive_solved = 0
    solved_checkpoint_saved = False

    reach_per_segment = env_config.v_max * args.manager_freq * env_config.dt
    max_goal_bound = (args.max_goal_bound if args.max_goal_bound is not None
                      else args.goal_bound_slack * reach_per_segment)
    print(f"Goal box: +-{max_goal_bound:.3f} m "
          f"({'explicit --max-goal-bound' if args.max_goal_bound is not None else f'{args.goal_bound_slack:g}x segment-reachable'}; "
          f"segment-reachable displacement = {reach_per_segment:.3f} m)")

    mpc_settings = mpc_settings_from_args(args)
    agent = PPOMPCAgent(
        obs_dim=obs_dim, goal_dim=goal_dim,
        lr_manager=args.learning_rate_manager,
        gamma=args.gamma, manager_freq=args.manager_freq,
        gae_lambda=args.gae_lambda, clip_coef=args.clip_coef,
        ent_coef_manager=args.ent_coef_manager, max_grad_norm=args.max_grad_norm,
        obs_low=eval_env.observation_space.low, obs_high=eval_env.observation_space.high,
        max_goal_bound=max_goal_bound, critic_lr_mult=args.critic_lr_mult,
        mpc_settings=mpc_settings, device=device,
    )
    base_lrs = [group["lr"] for group in agent.manager_optimizer.param_groups]

    # One worker slot per rollout environment, and a separate one-slot worker
    # for evaluation, so an evaluation never hands a rollout environment a
    # candidate plan from another episode (see TubeMPCWorker's docstring).
    worker = TubeMPCWorker.from_env(eval_env, num_slots=args.num_envs, **mpc_settings)
    eval_worker = TubeMPCWorker.from_env(eval_env, num_slots=1, **mpc_settings)
    print(worker.summary())

    # Each column is sized for the worst case, one segment per step.
    manager_buffer = ManagerVecRolloutBuffer(
        args.num_steps_per_env, args.num_envs, obs_dim, goal_dim, device)

    checkpoint_dir = os.path.join(scenario.script_dir, args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_eval_return = -float("inf")

    obs, _ = vec_env.reset(seed=args.seed)
    current_pos = obs[:, :2].copy()
    worker_step_in_c = np.zeros(args.num_envs, dtype=np.int64)
    accumulated_env_reward = np.zeros(args.num_envs, dtype=np.float32)

    manager_obs_norm = agent.normalize_obs(obs)
    with torch.no_grad():
        manager_action, manager_logprob, manager_value = agent.get_manager_action_and_value(
            torch.tensor(manager_obs_norm, dtype=torch.float32, device=device))
    current_goal = agent.scale_goal(manager_action.cpu().numpy())

    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.num_steps_worker
    actual_timesteps = num_updates * args.num_steps_worker
    print(f"num_envs={args.num_envs} num_steps_per_env={args.num_steps_per_env} "
          f"batch_size={args.num_steps_worker} num_updates={num_updates} timesteps={actual_timesteps}"
          + (f" (requested {args.total_timesteps})" if actual_timesteps != args.total_timesteps else ""))
    if args.track:
        wandb.config.update({"num_updates": num_updates, "actual_timesteps": actual_timesteps,
                             "num_steps_per_env": args.num_steps_per_env,
                             "max_goal_bound_resolved": max_goal_bound})

    metrics_log = MetricsLog(checkpoint_dir, config={
        "scenario": scenario.name, "algorithm": "ppo_mpc", "run_name": run_name,
        "num_updates": num_updates, "actual_timesteps": actual_timesteps,
        "max_goal_bound_resolved": max_goal_bound, **vars(args)})

    def log(metrics, step):
        metrics_log.log(metrics, step)
        if args.track:
            wandb.log(metrics, step=step)

    ep_reward = np.zeros(args.num_envs, dtype=np.float64)
    ep_length = np.zeros(args.num_envs, dtype=np.int64)
    consecutive_high_success = 0

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            for group, base_lr in zip(agent.manager_optimizer.param_groups, base_lrs):
                group["lr"] = frac * base_lr

        completed_returns, completed_lengths, completed_successes = [], [], []
        completed_collision_counts, completed_collision_impacts = [], []
        # The control arm of hPPO's worker termination-avoidance diagnostic
        # (see the "worker termination-avoidance" block in algorithms/hppo/
        # hppo_train.py), logged under the same name. There it catches a
        # learned worker that has found stalling in front of the goal line
        # worth more than crossing it. This worker has no return of its own to
        # protect, so near the line this reads out the *manager's* intent: it
        # can go negative early, while the manager still emits bad goals, but
        # not on a converged manager steering forward. (Carried over from the
        # old script_ppo_mpc.py.)
        near_goal_ax = []
        # The worker's own record: solve times, how often algorithm 3 fell
        # back to the shifted candidate, how often there was no plan at all,
        # how many binaries survived `_fix_unreachable`, and how far the real
        # state strayed inside the tube.
        mpc_solve_ms, mpc_outcome, mpc_free, mpc_tube = [], [], [], []

        for step in range(args.num_steps_per_env):
            global_step += args.num_envs

            # The worker steers every environment toward the absolute point its
            # goal names; one MIQP per environment.
            targets = obs[:, :2] + current_goal
            action_np, mpc_stats = worker.act_batch(obs, targets)
            action_np = action_np.astype(np.float32)
            mpc_solve_ms.append(mpc_stats["solve_ms"])
            mpc_outcome.append(mpc_stats["outcome"])
            mpc_free.append(mpc_stats["free_binaries"])
            mpc_tube.append(mpc_stats["tube_ratio"])

            near_goal_mask = obs[:, 0] >= (env_config.tunnel_length - 1.0)
            if np.any(near_goal_mask):
                near_goal_ax.append(action_np[near_goal_mask, 0])

            next_obs, env_reward, terminated, truncated, info = vec_env.step(action_np)
            done = terminated | truncated
            next_obs_norm = agent.normalize_obs(next_obs)

            # For a finished env `next_obs` is already the next episode's first
            # observation, so the goal decrement and the truncation bootstrap
            # read the pre-reset state instead.
            term_obs = next_obs.copy()
            if np.any(done):
                term_obs[done] = info["final_observation"][done]
            term_obs_norm = agent.normalize_obs(term_obs)

            ep_reward += env_reward
            ep_length += 1
            accumulated_env_reward += (agent.gamma ** worker_step_in_c) * env_reward

            next_goal = current_goal - (term_obs[:, :2] - current_pos)
            worker_step_in_c += 1

            # A finished episode's worker slot forgets its plan: the next step
            # is a fresh episode's first, where there is no candidate.
            for i in np.flatnonzero(done):
                worker.reset(i)

            # Which managers act now: those whose segment ran c steps or whose
            # episode ended -- the only two boundaries, so every done=False
            # boundary is exactly manager_freq steps long, as the manager's
            # gamma**c GAE assumes. The old PPO+MPC had a third,
            # --replan-on-collision: a re-plan the instant the worker hit a
            # wall. hPPO dropped the same option on 2026-09-23 as a train/eval
            # mismatch (evaluation never re-planned on contact) that had also
            # hidden a double-counted GAE bootstrap (see the manager_act_now
            # comment in algorithms/hppo/hppo_train.py). Here it would also be
            # moot: the tube keeps the worker off the walls.
            manager_act_now = (worker_step_in_c == args.manager_freq) | done
            if np.any(manager_act_now):
                # Truncation bootstrap: a segment cut after worker_step_in_c
                # steps continues at gamma**that, not at gamma**c.
                manager_reward = accumulated_env_reward.copy()
                manager_trunc = manager_act_now & truncated & ~terminated
                if np.any(manager_trunc):
                    with torch.no_grad():
                        true_next_manager_value = agent.get_manager_value(
                            torch.tensor(term_obs_norm[manager_trunc], dtype=torch.float32, device=device))
                    manager_reward[manager_trunc] += (
                        (agent.gamma ** worker_step_in_c[manager_trunc]) * true_next_manager_value.cpu().numpy())

                manager_buffer.add(manager_act_now, manager_obs_norm, manager_action, manager_logprob,
                                   manager_reward, manager_value, done.astype(np.float32))

                worker_step_in_c[manager_act_now] = 0
                accumulated_env_reward[manager_act_now] = 0.0

                # A fresh goal wherever a segment just ended; a done env has
                # already been reset into the state its new segment starts from.
                sel = manager_act_now
                sel_t = torch.as_tensor(sel, device=device)
                with torch.no_grad():
                    new_action, new_logprob, new_value = agent.get_manager_action_and_value(
                        torch.tensor(next_obs_norm[sel], dtype=torch.float32, device=device))
                manager_obs_norm[sel] = next_obs_norm[sel]
                manager_action[sel_t] = new_action
                manager_logprob[sel_t] = new_logprob
                manager_value[sel_t] = new_value
                current_goal[sel] = agent.scale_goal(new_action.cpu().numpy())

            for i in np.flatnonzero(done):
                completed_returns.append(ep_reward[i])
                completed_lengths.append(ep_length[i])
                completed_successes.append(bool(info["final_info"]["is_success"][i]))
                completed_collision_counts.append(int(info["final_info"]["collision_count"][i]))
                completed_collision_impacts.extend(info["final_info"]["collision_impacts"][i])
                ep_reward[i] = 0.0
                ep_length[i] = 0

            carry = ~manager_act_now
            current_goal[carry] = next_goal[carry]
            obs = next_obs
            current_pos = next_obs[:, :2].copy()

        # GAE, bootstrapped from each environment's in-flight segment (see
        # hppo_train.py for why its straddling the update is harmless).
        with torch.no_grad():
            agent.compute_manager_returns_and_advantage(manager_buffer, manager_value)
        metrics = agent.update_manager(manager_buffer, args.num_minibatches_manager, args.update_epochs)

        sps = int(global_step / (time.time() - start_time))
        metrics["charts/manager_lr"] = agent.manager_optimizer.param_groups[0]["lr"]
        metrics["charts/manager_critic_lr"] = agent.manager_optimizer.param_groups[-1]["lr"]
        metrics["charts/SPS"] = sps
        metrics["charts/num_episodes"] = len(completed_returns)
        if near_goal_ax:
            metrics["charts/worker_ax_near_goal"] = float(np.mean(np.concatenate(near_goal_ax)))
            metrics["charts/near_goal_samples"] = float(sum(a.size for a in near_goal_ax))
        solve_ms = np.concatenate(mpc_solve_ms)
        outcome = np.concatenate(mpc_outcome)
        tube = np.concatenate(mpc_tube)
        metrics["mpc/solve_ms_mean"] = float(solve_ms.mean())
        metrics["mpc/solve_ms_max"] = float(solve_ms.max())
        metrics["mpc/candidate_rate"] = float(np.mean(outcome == CANDIDATE))
        metrics["mpc/emergency_count"] = float(np.sum(outcome == EMERGENCY))
        metrics["mpc/free_binaries_mean"] = float(np.concatenate(mpc_free).mean())
        if np.any(~np.isnan(tube)):
            metrics["mpc/tube_ratio_max"] = float(np.nanmax(tube))

        log_line = (f"update={update} global_step={global_step} SPS={sps} "
                    f"m_ev={metrics['manager/explained_variance']:.3f} "
                    f"m_v_bias={metrics['manager/value_bias']:.1f} "
                    f"m_kl_max={metrics['manager/approx_kl_max']:.4f} "
                    f"mpc_ms={metrics['mpc/solve_ms_mean']:.1f}/{metrics['mpc/solve_ms_max']:.0f} "
                    f"cand={metrics['mpc/candidate_rate']:.3f}")
        if metrics["mpc/emergency_count"]:
            log_line += f" EMERGENCY={int(metrics['mpc/emergency_count'])}"
        if "charts/worker_ax_near_goal" in metrics:
            log_line += f" w_ax@goal={metrics['charts/worker_ax_near_goal']:+.2f}"
        if completed_returns:
            metrics["charts/episodic_return"] = float(np.mean(completed_returns))
            metrics["charts/episodic_length"] = float(np.mean(completed_lengths))
            metrics["charts/success_rate"] = float(np.mean(completed_successes))
            metrics["charts/collision_count_mean"] = float(np.mean(completed_collision_counts))
            log_line += (f" return={metrics['charts/episodic_return']:.1f} "
                         f"length={metrics['charts/episodic_length']:.0f} "
                         f"success_rate={metrics['charts/success_rate']:.2f} "
                         f"collisions/ep={metrics['charts/collision_count_mean']:.2f} "
                         f"(n={len(completed_returns)})")
            if completed_collision_impacts:
                metrics["charts/collision_impact_mean"] = float(np.mean(completed_collision_impacts))
                metrics["charts/collision_impact_worst"] = float(np.min(completed_collision_impacts))
        print(log_line)
        log(metrics, global_step)
        manager_buffer.reset()

        if update % args.eval_freq == 0:
            agent.manager_actor.eval()
            agent.manager_critic.eval()
            eval_returns, eval_lengths, eval_successes = [], [], []
            eval_collision_counts, eval_collision_impacts = [], []
            for i in range(args.eval_episodes):
                eval_obs, info_eval = eval_env.reset(seed=args.seed + i)
                policy_fn = make_policy_fn(agent, eval_worker)
                eval_ep_reward, eval_ep_length, done_eval = 0.0, 0, False
                while not done_eval:
                    eval_obs, reward_eval, terminated_eval, truncated_eval, info_eval = eval_env.step(policy_fn(eval_obs))
                    eval_ep_reward += reward_eval
                    eval_ep_length += 1
                    done_eval = terminated_eval or truncated_eval
                eval_returns.append(eval_ep_reward)
                eval_lengths.append(eval_ep_length)
                eval_successes.append(bool(info_eval["is_success"]))
                eval_collision_counts.append(info_eval["collision_count"])
                eval_collision_impacts.extend(info_eval["collision_impacts"])

            mean_eval_return = float(np.mean(eval_returns))
            success_rate = float(np.mean(eval_successes))
            collision_count_mean = float(np.mean(eval_collision_counts))
            print(f"Evaluation at update {update}: return={mean_eval_return:.2f}, length={np.mean(eval_lengths):.2f}, "
                  f"success_rate={success_rate:.2f}, collisions/ep={collision_count_mean:.2f}")
            eval_metrics = {
                "eval/episodic_return": mean_eval_return,
                "eval/episodic_length": float(np.mean(eval_lengths)),
                "eval/success_rate": success_rate,
                "eval/collision_count_mean": collision_count_mean,
            }
            if eval_collision_impacts:
                eval_metrics["eval/collision_impact_mean"] = float(np.mean(eval_collision_impacts))
                eval_metrics["eval/collision_impact_worst"] = float(np.min(eval_collision_impacts))
            log(eval_metrics, global_step)

            if args.save_eval_checkpoints:
                agent.save(os.path.join(checkpoint_dir, f"eval_{global_step:07d}.pt"))
            if mean_eval_return > best_eval_return:
                best_eval_return = mean_eval_return
                agent.save(os.path.join(checkpoint_dir, "best.pt"))

            solved_result = check_solved(lambda: make_policy_fn(agent, eval_worker), eval_env,
                                         optimal_grid, args.solved_tolerance, seed=args.seed)
            print(f"Solved-check at update {update}: solved={solved_result.solved} "
                  f"worst_gap={solved_result.worst_gap:.2f} at "
                  f"p_x0={solved_result.worst_point[0]:.2f} p_y0={solved_result.worst_point[1]:.2f} "
                  f"(consecutive={consecutive_solved})")
            log({
                "solved/is_solved": float(solved_result.solved),
                "solved/worst_gap": solved_result.worst_gap,
                "solved/mean_gap": float(solved_result.gaps.mean()),
            }, global_step)
            consecutive_solved = consecutive_solved + 1 if solved_result.solved else 0

            agent.manager_actor.train()
            agent.manager_critic.train()

            if (args.early_stop_success_rate is not None
                    and success_rate >= args.early_stop_success_rate
                    and mean_eval_return >= early_stop_min_return):
                consecutive_high_success += 1
            else:
                consecutive_high_success = 0
            if consecutive_high_success >= args.early_stop_patience:
                print(f"Early stop at update {update}: eval/success_rate >= "
                      f"{args.early_stop_success_rate} and eval/episodic_return "
                      f">= {early_stop_min_return:.1f} for {consecutive_high_success} consecutive evaluations")
                log({"charts/early_stopped_at_update": update}, global_step)
                break

            if consecutive_solved >= args.solved_consecutive:
                if not solved_checkpoint_saved:
                    print(f"Solved criterion held for {consecutive_solved} consecutive evals "
                          f"(tolerance={args.solved_tolerance}) at global_step={global_step}.")
                    agent.save(os.path.join(checkpoint_dir, "solved.pt"))
                    solved_checkpoint_saved = True
                    log({"solved/first_solved_step": global_step}, global_step)
                if args.solved_early_stop:
                    print("--solved-early-stop is on -- stopping training early.")
                    break

    agent.save(os.path.join(checkpoint_dir, "final.pt"))
    print(f"Saved checkpoints to {checkpoint_dir}")

    metrics_log.close()
    eval_env.close()
    worker.close()
    eval_worker.close()
    if args.track:
        wandb.finish()
