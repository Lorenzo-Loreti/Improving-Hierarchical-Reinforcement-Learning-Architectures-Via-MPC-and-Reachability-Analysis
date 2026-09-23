import os
import argparse
import time
import numpy as np
import torch
import wandb
import sys
# Scenario root (this script's parent), for `envs`; then algorithms/hppo,
# for the flat `hppo` module it imports by bare name.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithms', 'hppo')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithms')))
from envs.config import SlalomEnvConfig
from envs.slalom_env import SlalomEnv
from envs.vec_slalom_env import SlalomVecEnv
from envs.width_profile import slalom_profile
from hppo import HPPOAgent, VecRolloutBuffer, ManagerVecRolloutBuffer
from optimal_solver import spawn_grid, precompute_optimal_grid, MinTimeSolver
from solved_check import check_solved

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="hppo_slalom",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    # `type=bool` would be a no-op here: argparse applies it to the *string*,
    # and bool("False") is True, so the flag could only ever be turned on.
    parser.add_argument("--torch-deterministic", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", action="store_true",
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Slalom-hPPO-NoNoise",
        help="the wandb's project name")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
        help="directory (relative to this script) to save model checkpoints in")

    # Reward-scale overrides. Default to SlalomEnvConfig's own defaults, so
    # omitting these reproduces exactly the environment every other script
    # (flat PPO, PPO+MPC) trains against -- passing them is how a single run
    # can test a different risk/reward balance without moving that shared
    # default out from under everyone else. See envs/config.py.
    parser.add_argument("--env-u-max", type=float, default=None,
        help="REGIME STUDY ONLY. Override the environment's acceleration "
             "limit u_max (default None = SlalomEnvConfig's own 2.5, i.e. the "
             "canonical environment every architecture in this repo is "
             "compared on). Lowering it makes the plant sluggish, raising the "
             "agility ratio rho = v_max/(u_max*manager_freq*dt). Mirrors the "
             "identically-named flag in the other three scripts; hPPO is the "
             "*learned*-worker arm of that sweep, the one control that says "
             "whether robustness to plant sluggishness comes from hierarchy "
             "as such or specifically from the model-based worker. See "
             "docs/reachability-regime-study.md")
    parser.add_argument("--contact-penalty", type=float, default=SlalomEnvConfig().contact_penalty,
        help="small penalty applied every step the agent is in contact with a "
             "wall, flat regardless of impact speed -- no longer a terminal "
             "reward: a contact clamps the agent to the wall and absorbs "
             "v_y, but the episode continues. Defaults to SlalomEnvConfig's "
             "own default, so omitting this reproduces the shared environment "
             "every other script trains against -- see envs/config.py")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate-manager", type=float, default=3e-4,
        help="the learning rate of the manager's actor. Its critic runs at "
             "--critic-lr-mult times this")
    parser.add_argument("--learning-rate-worker", type=float, default=3e-4,
        help="the learning rate of the worker's actor. Its critic runs at "
             "--critic-lr-mult times this")
    parser.add_argument("--critic-lr-mult", type=float, default=3.0,
        help="each critic's learning rate as a multiple of its head's rate. "
             "Every critic gets its own optimiser parameter group; 1.0 "
             "restores the single-rate behaviour")
    parser.add_argument("--anneal-lr", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=False,
        help="if toggled, the learning rates decay linearly over training, in "
             "every parameter group of both optimisers. Off by default: with "
             "a decaying LR, every seed that solved the slalom in testing did "
             "so by escaping an intermediate local optimum (a 'rush and "
             "crash' policy that ignores the walls) at some update that "
             "varies by seed, and annealing to ~0 by the end of a fixed "
             "training budget starved that escape of gradient signal before "
             "it could happen on several seeds. See envs/config.py's "
             "contact_penalty comment for the fuller story. Annealing all "
             "the way to 0 also freezes the last updates (approx_kl "
             "collapses to ~0), burning the tail of --total-timesteps on a "
             "policy that can no longer move; a --lr-floor-frac that stopped "
             "the decay short of 0 existed for that, and was removed along "
             "with the other options no run enabled by default")
    parser.add_argument("--manager-freq", type=int, default=10,
        help="c: the number of steps the manager's goal is valid for. Also "
             "fixes the manager's discount, gamma**c")
    parser.add_argument("--max-goal-bound", type=float, default=10.0,
        help="the manager's [-1, 1] action is scaled by this to a physical "
             "goal displacement in metres, and divided by it again to form "
             "the worker's input")
    # --- worker termination-avoidance ------------------------------------
    #
    # The worker is trained on a purely *intrinsic* reward: the per-step
    # reduction in the distance to the manager's goal. That reward stream has
    # no terminal term, so in the worker's own MDP reaching the environment's
    # goal line is an absorbing state of value exactly 0, while *not* reaching
    # it keeps paying ~v_max*dt approx 0.12 per step forever (the manager hands
    # out a fresh, effectively unreachable goal every --manager-freq steps, so
    # the stream never runs dry). Discounted at --gamma that is a value of
    # O(10) for never finishing against 0 for finishing: the worker's optimal
    # policy under its own reward is to approach the goal line and stall.
    #
    # That is exactly what a 500k-step slalom run does. Around update ~50 (of
    # 244) the worker starts braking near p_x = L, episode length climbs
    # 80 -> 200, and eval success falls 1.00 -> 0.00 over ~30 updates and
    # never recovers. Probing the collapsed checkpoint at (p_x, p_y, v) =
    # (9.5, 0, 1.2, 0): the worker's deterministic a_x is between -1.6 and
    # -2.1 for *every goal in the manager's entire action box* -- no goal the
    # manager can emit makes it cross the line, which is why the hierarchy has
    # no in-run recovery path. Flat PPO and both PPO+MPC variants are immune
    # for the same reason: neither has a *learned* worker optimizing a
    # reward stream that termination cuts off (flat PPO optimizes the
    # environment's own reward, whose terminal is +goal_reward; PPO+MPC's
    # worker is a QP, not a policy).
    #
    # The two knobs below are two different ways to put the environment's
    # terminal back into the worker's return; 0.0 restores the pre-fix reward
    # exactly for either. A 4-variant x 2-seed, 500k-step ablation with
    # --early-stop-success-rate and --solved-early-stop *disabled* (so a fix
    # that merely delays the collapse cannot be mistaken for one that removes
    # it) settled which is needed:
    #
    #   variant                          clean solves   final eval return
    #   ------------------------------   ------------   ------------------
    #   baseline (both knobs 0.0)            0/2        -153 / -997
    #   --ent-coef-lr 0.05                   0/2        -181 /  615
    #   --worker-success-bonus 20            2/2        1018 / 1019
    #   --worker-extrinsic-coef 0.02         2/2        1019 / 1019
    #
    # (--ent-coef-lr no longer exists: the entropy autotuner it sped up was
    # later removed as inert at its default -- see HPPOAgent's class comment.)
    #
    # against a mean oracle optimum of ~1013 (see the "Solved-check:
    # precomputed optimal returns" line printed at startup); "clean solve"
    # is the final evaluation sitting at >= 95% of that, which separates a
    # genuine solve from the two ways an unfixed run ends -- stalling short
    # of the line at 200 steps/episode (seed 1), or reaching it by smashing
    # through both gates (seed 2: success_rate 1.00 at return -997). Two
    # things that
    # ablation settles beyond the headline: (1) the pathology is the *worker's*
    # reward, not the manager's exploration -- fixing the entropy autotuner
    # (whose dual-ascent step is lr-bounded at ~3e-4 per update, so at the
    # default it moves ent_coef by 8% over a whole run while the manager's
    # entropy falls from +1.2 to -2.2 nats) changes when the collapse happens
    # and not whether it does; (2) both fixes do not merely prevent the
    # collapse, they close the optimality gap -- the unfixed hierarchy peaked
    # at ~745 with 3-8 wall contacts per episode even *before* collapsing,
    # because the worker had no reason to care about anything but the goal
    # vector.
    #
    # --worker-extrinsic-coef is the default, and the reasoning that first
    # picked --worker-success-bonus instead is worth recording because the
    # evidence overturned it. The success bonus is the *minimal* intervention:
    # it changes the terminal only and leaves the worker otherwise blind to
    # the environment's reward, which is what keeps this a feudal hierarchy
    # rather than two agents optimizing the same objective at different
    # rates. On the 2-seed ablation above the two tie on final quality, so
    # that argument decided it.
    #
    # It does not survive the full benchmark. Re-running both arms at 3 seeds
    # under the benchmark protocol (uniform 500k budget, --solved-early-stop
    # left on and --early-stop-success-rate 2.0 to disable the looser stop)
    # separates them on samples rather than on quality:
    #
    #   hppo, slalom   success-bonus  3/3 solved, 266k / 296k / 399k steps,
    #                                 final return 1018 / 1019 / 1017
    #                  extrinsic      3/3 solved,  71k /  81k /  81k steps,
    #                                 final return 1015 / 1016 / 1016
    #
    # A 3-5x difference in samples at the same final policy, and the reason
    # is what the success bonus does *not* do. It repairs the termination
    # incentive, so the worker does finish -- success_rate 1.00 on every seed
    # -- but nothing in its reward charges it for a wall or for the clock, so
    # every collision still has to be steered out by a manager that only acts
    # every --manager-freq steps, and that indirection is expensive.
    # Restoring the terminal alone makes the worker's MDP *sound*; it does
    # not make it *aligned*.
    #
    # The size of that gap also changes the headline comparison: at
    # 71k-81k steps this hierarchy is no longer measurably slower than flat
    # PPO (51k-81k) on the slalom, where under the success bonus it read as a
    # clean 4-6x hierarchy tax. See docs/benchmark.md section 3.
    #
    # So: feudal purity was the right prior and the wrong call. 0.02 is the
    # default; --worker-success-bonus 20 is kept as the minimal-intervention
    # ablation arm, and is the one to use when the question is specifically
    # "what does the terminal alone fix?".
    parser.add_argument("--worker-success-bonus", type=float, default=0.0,
        help="one-off bonus added to the *worker's* reward on the step the "
             "episode terminates successfully. The targeted fix for the "
             "termination-avoidance pathology above: it makes crossing the "
             "goal line worth more to the worker than the goal-closing reward "
             "it forgoes by crossing. Size it against that forgone value, not "
             "against the environment's reward: the worker earns about "
             "v_max*dt per step indefinitely, so what it gives up by "
             "terminating is v_max*dt/(1-gamma) approx 12 at the defaults, "
             "and the default 20.0 clears that with ~1.7x margin. Re-derive "
             "it if --gamma, --manager-freq or the velocity limit move. "
             "Not the default: see the comparison above. 0.0 (default) with "
             "--worker-extrinsic-coef also 0.0 restores the pre-fix reward, "
             "which collapses")
    parser.add_argument("--worker-extrinsic-coef", type=float, default=0.02,
        help="FeUdal-Networks-style mixing coefficient: add this times the "
             "environment's own reward to the worker's intrinsic reward, so "
             "the worker sees r_int + coef * r_env instead of r_int alone. "
             "The broader alternative to --worker-success-bonus: it fixes the "
             "same termination-avoidance pathology (goal_reward enters the "
             "worker's return) and additionally makes the worker itself aware "
             "of wall contacts and of the clock, instead of leaving every "
             "collision the manager's problem. At the SlalomEnvConfig "
             "defaults a coefficient of 0.02 puts the terminal at +20, a "
             "contact at -1.0 and a step at -0.02 in the worker's units, "
             "against an intrinsic stream of ~0.12/step -- which is why it "
             "is the default: it is the only one of the two that makes the "
             "worker itself avoid walls, and the _reach variants need that. "
             "0.0 restores the pre-fix reward unless --worker-success-bonus "
             "is set")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel slalom environments to collect rollouts "
             "from. Rollouts come from a batched SlalomVecEnv; each "
             "environment carries its own goal, manager cadence and segment "
             "accumulator")
    parser.add_argument("--num-steps-worker", type=int, default=2048,
        help="the number of worker (= environment) steps per rollout, summed "
             "over all --num-envs environments. Unchanged in meaning by "
             "vectorization -- it is still the worker's batch size, so 2048 "
             "keeps the minibatch at 256 and the update count where they were; "
             "what changed is that the 2048 now come from --num-envs parallel "
             "streams of 2048/--num-envs steps instead of one serial stream. "
             "Must be divisible by --num-envs")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the environment-step discount factor. 0.99, not the 0.999 this "
             "script used to train at: see the thesis PPO chapter, 11.1. The "
             "manager is discounted at gamma**manager_freq")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches the *worker's* rollout is split "
             "into. 8 against 2048 holds the minibatch at 256, as flat PPO "
             "does; the previous 32 gave minibatches of 64")
    parser.add_argument("--num-minibatches-manager", type=int, default=4,
        help="the number of mini-batches the *manager's* rollout is split "
             "into. Separate from --num-minibatches because the manager's "
             "buffer is roughly manager_freq times smaller: splitting it 32 "
             "ways left ~6 samples per gradient step")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update each policy")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef-manager", type=float, default=0.01,
        help="coefficient of the manager's entropy bonus, fixed for the whole "
             "run (see HPPOAgent for why there is no autotuning)")
    parser.add_argument("--ent-coef-worker", type=float, default=0.01,
        help="coefficient of the worker's entropy bonus, fixed for the whole run")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--clip-vloss", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True,
        help="PPO2/CleanRL-style value-loss clipping: bound how far the "
             "critic's new prediction may move from its pre-update one (by "
             "--clip-coef) before scoring the value loss, so one update "
             "cannot push either critic arbitrarily far on a noisy batch. "
             "Applied to both heads -- see algorithms/hppo/hppo.py's "
             "_update_head. On by default as of a 6-seed slalom ablation "
             "(seeds 1-6): 4/6 seeds went from never recovering after the "
             "manager-collapse instability to a clean solve that triggers "
             "the existing --early-stop-success-rate; a 5th delayed the "
             "collapse from ~150k to ~230k steps; only 1/6 still collapsed, "
             "with the same value_bias/explained_variance drift signature "
             "as every unclipped run. Pass --clip-vloss false to restore "
             "the previous unclipped behaviour")
    parser.add_argument("--eval-freq", type=int, default=5,
        help="evaluate the agent every eval_freq updates. 5 rather than 1 "
             "because an evaluation of --eval-episodes episodes costs up to "
             "eval_episodes*max_steps environment steps, which at every "
             "update would dominate the wall clock over the 2048 steps of "
             "training it is measuring")
    parser.add_argument("--eval-episodes", type=int, default=20,
        help="number of episodes to evaluate the agent")
    parser.add_argument("--solved-early-stop", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True,
        help="if toggled, actually stop training once the solved criterion below "
             "is met. The check, its logging, and the one-time solved.pt checkpoint "
             "still happen either way -- this only gates the early `break`, so runs "
             "meant to be plotted against each other on the same x-axis (fixed "
             "--total-timesteps for every seed) can set this false and still see "
             "where each seed crossed the threshold. Independent of "
             "--early-stop-success-rate below -- either can stop training")
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
             "--early-stop-patience consecutive evaluations. On by default: "
             "on the slalom task, seed 1's manager reliably reaches 100%% "
             "eval success by ~1/5 of a 500k-step run and then, under "
             "continued training, its goal distribution keeps sharpening "
             "past that point -- one goal dimension's Beta collapses toward "
             "a state-insensitive skew, the other's state-dependent sign "
             "flip (needed to alternate between the slalom's two "
             "oppositely-offset gates) erodes, and eval success falls back "
             "to 0%% within a few evaluations and never recovers for the "
             "rest of the budget. This has no in-run recovery mechanism, so "
             "rather than fight it, stop as soon as the policy has "
             "demonstrably solved the task. Set > 1.0 to disable")
    parser.add_argument("--early-stop-optimal-frac", type=float, default=0.95,
        help="the eval/episodic_return floor --early-stop-success-rate also "
             "requires, as a fraction of the solved-check oracle's mean "
             "optimal return (see the 'Solved-check: precomputed optimal "
             "returns' line printed at startup) rather than a fixed reward "
             "value. success_rate alone is not enough: the manager reaches "
             "100%% success as early as ~40k steps by smashing through both "
             "gates rather than threading them -- SlalomEnv clamps a wall "
             "contact and continues rather than ending the episode (see "
             "envs/config.py's contact_penalty comment), so reaching the "
             "goal line at all does not mean reaching it cleanly. A prior "
             "version of this flag was a fixed absolute value (150.0) "
             "picked off one seed's trajectory at one goal_reward/"
             "contact_penalty balance; it went stale the moment that "
             "balance changed -- a seed-1-10 sweep at a 5x larger "
             "goal_reward found it let a policy with ~16 collisions/episode "
             "(return=215, far from the ~1013 mean optimum) trigger early "
             "stop. Deriving the floor from the oracle's own optimal return "
             "keeps it meaningful across reward-scale changes")
    parser.add_argument("--early-stop-patience", type=int, default=3,
        help="consecutive qualifying evaluations required before "
             "--early-stop-success-rate stops training; see its help")
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error(f"--num-envs must be > 0, got {args.num_envs}")
    if args.num_steps_worker % args.num_envs != 0:
        parser.error(
            f"--num-steps-worker ({args.num_steps_worker}) must be divisible by "
            f"--num-envs ({args.num_envs}); otherwise the worker's batch is not "
            f"the step budget it names")
    args.num_steps_per_env = args.num_steps_worker // args.num_envs
    return args

def make_env(config=None):
    """A bare SlalomEnv, for evaluation.

    Rollouts come from a batched `SlalomVecEnv` instead; this is the
    single-environment path, which evaluation still wants because it runs one
    seeded episode at a time and has no throughput problem to solve. Flat PPO
    splits the two the same way.

    No wrappers. `RecordEpisodeStatistics` is replaced by counting returns and
    lengths in the loop, which also gets the success and collision flags out of
    `info` -- the two metrics that distinguish a stalled run from a converged
    one, and which the wrapper does not carry.

    `NormalizeReward` is gone for a substantive reason, not tidiness. It scaled
    the reward stream by a running estimate of the return's standard deviation
    to keep the critic's regression target O(1); the agent now standardises
    that target directly (hppo.py, RunningMeanStd), which achieves the same
    thing without altering the rewards GAE sees, without a wrapper statistic
    that has to be checkpointed to reload a policy, and -- unlike the wrapper,
    which normalises the *environment* reward only -- symmetrically for the
    worker's intrinsic reward too. Runs from before this change saw a
    different reward stream and are not directly comparable.
    """
    return SlalomEnv(config=config)

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"

    env_config = SlalomEnvConfig(
        width_profile=slalom_profile(),
        contact_penalty=args.contact_penalty,
        **({} if args.env_u_max is None else {"u_max": args.env_u_max}),
    )

    if args.env_u_max is not None:
        _T = args.manager_freq * env_config.dt
        print(f"REGIME STUDY: env u_max overridden to {env_config.u_max} "
              f"(canonical {SlalomEnvConfig().u_max}); agility ratio rho = "
              f"v_max/(u_max*T) = {env_config.v_max / (env_config.u_max * _T):.2f}")

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )

    # Seeding
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Env setup: a batched SlalomVecEnv for rollout collection, a plain
    # SlalomEnv for evaluation -- the same split flat PPO uses.
    vec_env = SlalomVecEnv(num_envs=args.num_envs, config=env_config)
    eval_env = make_env(config=env_config)
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    goal_dim = 2 # delta x, delta y
    max_goal_bound = args.max_goal_bound
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
    # goal_reward/contact_penalty balance this run is training against.
    early_stop_min_return = args.early_stop_optimal_frac * float(optimal_returns.mean())
    print(f"Early-stop return floor: {early_stop_min_return:.1f} "
          f"({args.early_stop_optimal_frac:.0%} of mean optimal)")
    consecutive_solved = 0
    solved_checkpoint_saved = False

    # Agent and Buffer
    agent = HPPOAgent(
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        act_dim=act_dim,
        worker_act_limit_low=float(eval_env.action_space.low[0]),
        worker_act_limit_high=float(eval_env.action_space.high[0]),
        lr_manager=args.learning_rate_manager,
        lr_worker=args.learning_rate_worker,
        # One environment-step discount. The manager's gamma**c is derived
        # inside the agent from manager_freq, so the discount used for its GAE
        # cannot drift from the cadence the rollout actually runs at.
        gamma=args.gamma,
        manager_freq=args.manager_freq,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef_manager=args.ent_coef_manager,
        ent_coef_worker=args.ent_coef_worker,
        max_grad_norm=args.max_grad_norm,
        critic_lr_mult=args.critic_lr_mult,
        # Carried into the checkpoint so a reloaded hierarchy knows the
        # observation and goal maps it was trained under.
        obs_low=obs_low,
        obs_high=obs_high,
        max_goal_bound=max_goal_bound,
        device=device,
        # The one manager-collapse mitigation kept -- see --clip-vloss's
        # help and ManagerActor's docstring in algorithms/common.py.
        clip_vloss=args.clip_vloss,
    )

    # One base learning rate per parameter group (actor, then critic) for each
    # head. The anneal below scales each group against its own base; scaling
    # every group by the head's rate would silently reset its critic to the
    # actor's rate on the first update.
    base_lrs_manager = [group["lr"] for group in agent.manager_optimizer.param_groups]
    base_lrs_worker = [group["lr"] for group in agent.worker_optimizer.param_groups]

    # The worker acts every step in every environment, so its rollout is a
    # full (num_steps_per_env, num_envs) grid.
    worker_buffer = VecRolloutBuffer(
        args.num_steps_per_env, args.num_envs, obs_dim + goal_dim, act_dim, device)
    # The manager writes one transition per c worker steps *per environment*,
    # so each column needs only ceil(num_steps_per_env / c) slots -- plus a
    # margin, because a segment also ends early whenever an episode does. Sized
    # for the worst case (an episode ending on every step) rather than reasoned
    # about, since the buffer is a few thousand floats either way.
    manager_buffer = ManagerVecRolloutBuffer(
        args.num_steps_per_env, args.num_envs, obs_dim, goal_dim, device)

    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_eval_return = -float("inf")

    def worker_input(obs_norm, goal_phys):
        """The worker's network input: normalized observation ++ normalized goal.

        Concatenates on the last axis, so it takes either a single
        (obs_dim,)/(goal_dim,) pair -- what evaluation has -- or the
        (num_envs, obs_dim)/(num_envs, goal_dim) batch the rollout carries.
        """
        return torch.tensor(
            np.concatenate([obs_norm, agent.normalize_goal(goal_phys)], axis=-1),
            dtype=torch.float32, device=device
        )

    # Initialize environments. Every quantity the serial rollout held as a
    # scalar is now a per-environment array: the environments run independent
    # episodes, so they sit at different points in their manager cadence, carry
    # different partially-consumed goals, and accumulate different segment
    # rewards.
    obs, _ = vec_env.reset(seed=args.seed)
    obs_norm = agent.normalize_obs(obs)
    current_pos = obs[:, :2].copy()

    worker_step_in_c = np.zeros(args.num_envs, dtype=np.int64)
    accumulated_env_reward = np.zeros(args.num_envs, dtype=np.float32)

    # Initial manager action, for every environment at once.
    manager_obs_norm = obs_norm.copy()
    with torch.no_grad():
        manager_action, manager_logprob, manager_value = agent.get_manager_action_and_value(
            torch.tensor(manager_obs_norm, dtype=torch.float32, device=device)
        )
    current_goal = agent.scale_goal(manager_action.cpu().numpy())

    global_step = 0
    start_time = time.time()

    num_updates = args.total_timesteps // args.num_steps_worker
    actual_timesteps = num_updates * args.num_steps_worker
    worker_minibatch_size = max(1, args.num_steps_worker // args.num_minibatches)
    print(f"num_envs={args.num_envs} num_steps_per_env={args.num_steps_per_env} "
          f"worker_batch_size={args.num_steps_worker} "
          f"worker_minibatch_size={worker_minibatch_size} "
          f"num_updates={num_updates} timesteps={actual_timesteps}"
          + (f" (requested {args.total_timesteps}; {args.total_timesteps - actual_timesteps} "
             f"dropped by integer division)" if actual_timesteps != args.total_timesteps else ""))
    if args.track:
        wandb.config.update({"num_updates": num_updates, "actual_timesteps": actual_timesteps,
                             "worker_minibatch_size": worker_minibatch_size,
                             "num_steps_per_env": args.num_steps_per_env})

    ep_reward = np.zeros(args.num_envs, dtype=np.float64)
    ep_length = np.zeros(args.num_envs, dtype=np.int64)

    # See --early-stop-success-rate: counts consecutive qualifying
    # evaluations, reset by any evaluation that falls short.
    consecutive_high_success = 0

    for update in range(1, num_updates + 1):
        # Linear LR decay: once the policies have converged a constant LR keeps
        # injecting noise into all four heads off a shrinking advantage signal.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            for group, base_lr in zip(agent.manager_optimizer.param_groups, base_lrs_manager):
                group["lr"] = frac * base_lr
            for group, base_lr in zip(agent.worker_optimizer.param_groups, base_lrs_worker):
                group["lr"] = frac * base_lr

        completed_returns = []
        completed_lengths = []
        completed_successes = []
        completed_collision_counts = []
        completed_collision_impacts = []
        # Direct instrumentation of the termination-avoidance mechanism (see
        # --worker-success-bonus): what the worker actually does in the strip
        # of states immediately before the goal line. A healthy worker keeps
        # a_x positive there and crosses; the collapsed one brakes with
        # a_x approx -2 and orbits. Logged per update so the onset is visible
        # in the same plot as the eval success rate it precedes by ~30 updates.
        near_goal_ax = []
        near_goal_value = []

        for step in range(0, args.num_steps_per_env):
            global_step += args.num_envs

            # Worker action selection, one batched forward pass over all envs
            obs_goal_tensor = worker_input(obs_norm, current_goal)
            with torch.no_grad():
                worker_action, worker_logprob, worker_value = agent.get_worker_action_and_value(
                    obs_goal_tensor
                )

            action_np = worker_action.cpu().numpy()

            # See near_goal_ax above. `obs` is the pre-step observation, so
            # this is "the action taken from a state within 1 m of the goal
            # line", which is exactly the decision the pathology corrupts.
            near_goal_mask = obs[:, 0] >= (env_config.tunnel_length - 1.0)
            if np.any(near_goal_mask):
                near_goal_ax.append(action_np[near_goal_mask, 0])
                near_goal_value.append(worker_value.cpu().numpy()[near_goal_mask])

            # Environment step (SlalomVecEnv auto-resets envs that finish)
            next_obs, env_reward, terminated, truncated, info = vec_env.step(action_np)
            done = terminated | truncated
            next_obs_norm = agent.normalize_obs(next_obs)

            # The auto-reset is why this is not a mechanical substitution. For
            # a finished env `next_obs` is already the *next* episode's first
            # observation, so every quantity describing the transition that
            # just happened -- the goal decrement, the worker's progress
            # reward, both truncation bootstraps -- has to read the pre-reset
            # state instead. Taking it from `next_obs` would score the worker
            # on a teleport back to the slalom entrance.
            term_obs = next_obs.copy()
            if np.any(done):
                term_obs[done] = info["final_observation"][done]
            term_obs_norm = agent.normalize_obs(term_obs)

            ep_reward += env_reward
            ep_length += 1

            accumulated_env_reward += (agent.gamma ** worker_step_in_c) * env_reward

            # Update goals manually, against the pre-reset position
            term_pos = term_obs[:, :2]
            next_goal = current_goal - (term_pos - current_pos)

            worker_reward = (np.linalg.norm(current_goal, axis=-1)
                             - np.linalg.norm(next_goal, axis=-1))

            # Put the environment's own outcome back into the worker's return
            # -- see --worker-success-bonus / --worker-extrinsic-coef for the
            # termination-avoidance pathology both of these exist to remove.
            # Applied here, before the truncation bootstrap below, so the
            # bootstrapped value and the reward it is folded into are on the
            # same (mixed) scale.
            if args.worker_extrinsic_coef != 0.0:
                worker_reward = worker_reward + args.worker_extrinsic_coef * env_reward
            if args.worker_success_bonus != 0.0:
                # `terminated` is only ever a goal-line crossing in this
                # environment (a wall contact is non-terminal), but read the
                # success flag rather than relying on that, so adding another
                # terminal condition later cannot silently start paying this
                # bonus for it.
                worker_succeeded = terminated & info["final_info"]["is_success"]
                worker_reward[worker_succeeded] += args.worker_success_bonus

            # Handle truncation bootstrapping for the worker. The episode did
            # not end, the clock did: fold the value of the state it was cut at
            # into the reward, then mark the transition done so GAE stops
            # there. Discounts come from the agent, which is also what ran GAE.
            trunc_only = truncated & ~terminated
            if np.any(trunc_only):
                with torch.no_grad():
                    true_next_worker_value = agent.get_worker_value(
                        worker_input(term_obs_norm[trunc_only], next_goal[trunc_only])
                    )
                worker_reward[trunc_only] += (
                    agent.gamma_worker * true_next_worker_value.cpu().numpy())

            # `terminated`, or forced True where truncated -- and SlalomVecEnv
            # makes the two mutually exclusive, so this is exactly `done`.
            worker_done = done

            # Store worker transitions (all envs, every step)
            worker_buffer.add(
                obs_goal_tensor,
                worker_action,
                worker_logprob,
                worker_reward,
                worker_value,
                worker_done.astype(np.float32)
            )

            worker_step_in_c += 1

            # Which managers act now? An environment's segment ends after c
            # worker steps or when its episode does, and the environments do
            # not agree on when. Those are the only two boundaries, so every
            # done=False boundary is exactly manager_freq steps long -- which
            # is what the manager's fixed gamma**manager_freq GAE recursion
            # assumes.
            #
            # A wall contact used to be a third boundary (--replan-on-collision,
            # on by default): the manager re-planned the instant the worker
            # hit a wall, on the grounds that a hit can clamp position and
            # zero velocity in a way that leaves the in-flight goal
            # unreachable for the rest of the segment. It was removed because
            # (1) it was a train/eval mismatch -- evaluation and the
            # solved-check only ever re-planned on the manager_freq boundary,
            # so the policy was trained under a controller it was never
            # evaluated with; (2) it mattered while collision rates were 8-24
            # per episode and is close to moot at the 0-1 the worker-reward fix
            # brought (docs/worker-termination-avoidance.md, open item 1); and
            # (3) its segments needed a pseudo-done plus a continuation value
            # folded into the reward, and when such a segment was the last one
            # a column stored in a rollout, the GAE bootstrap (next_done taken
            # from the real `done`, not the pseudo-done) added that
            # continuation a second time.
            manager_act_now = (worker_step_in_c == args.manager_freq) | done
            if np.any(manager_act_now):
                # Handle truncation bootstrapping for the manager. Its segment
                # is worker_step_in_c environment steps long, so the cut-off
                # continuation is discounted by gamma**that -- not by the
                # manager's own gamma**c, which assumes a full segment. That
                # exponent is now per environment too.
                manager_reward = accumulated_env_reward.copy()
                manager_trunc = manager_act_now & trunc_only
                if np.any(manager_trunc):
                    with torch.no_grad():
                        true_next_manager_value = agent.get_manager_value(
                            torch.tensor(term_obs_norm[manager_trunc],
                                         dtype=torch.float32, device=device)
                        )
                    manager_reward[manager_trunc] += (
                        (agent.gamma ** worker_step_in_c[manager_trunc])
                        * true_next_manager_value.cpu().numpy())

                manager_buffer.add(
                    manager_act_now,
                    manager_obs_norm,
                    manager_action,
                    manager_logprob,
                    manager_reward,
                    manager_value,
                    done.astype(np.float32)
                )

                worker_step_in_c[manager_act_now] = 0
                accumulated_env_reward[manager_act_now] = 0.0

                # A fresh goal for every environment whose segment just ended.
                # The serial loop split this in two -- re-plan at the next
                # observation, or re-plan at the reset observation after an
                # episode ended -- but under an auto-resetting vector env both
                # are `next_obs`, since a done environment has already been
                # reset into the state the new segment starts from. Note that
                # `done` implies `manager_act_now`, so no environment is left
                # holding a goal from a finished episode.
                sel = manager_act_now
                sel_t = torch.as_tensor(sel, device=device)
                with torch.no_grad():
                    new_action, new_logprob, new_value = agent.get_manager_action_and_value(
                        torch.tensor(next_obs_norm[sel], dtype=torch.float32, device=device)
                    )
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

            # Environments still mid-segment carry their partially consumed
            # goal forward; the rest were just given a fresh one above.
            carry = ~manager_act_now
            current_goal[carry] = next_goal[carry]

            obs = next_obs
            obs_norm = next_obs_norm
            # Post-auto-reset for finished envs, pre-reset terminal for the
            # rest -- which is what the next step's goal decrement needs.
            current_pos = next_obs[:, :2].copy()

        # Bootstrap values for the states after the rollout. GAE bootstraps
        # from them only where a column's last stored transition did not end
        # its episode -- each buffer reads that from its own last done flag.
        with torch.no_grad():
            # Worker bootstrap, per environment
            next_worker_value = agent.get_worker_value(worker_input(obs_norm, current_goal))
            agent.compute_worker_returns_and_advantage(worker_buffer, next_worker_value)

            # Manager bootstrap. `manager_value` is the value of each
            # environment's segment currently in flight, whose start is exactly
            # the state that environment's last stored manager transition led
            # to -- so it is already the per-column bootstrap the ragged GAE
            # wants.
            agent.compute_manager_returns_and_advantage(manager_buffer, manager_value)

        # Optimize the policies
        manager_minibatch_size = max(1, manager_buffer.total_steps // args.num_minibatches_manager)

        worker_metrics = agent.update_worker(worker_buffer, worker_minibatch_size, args.update_epochs)
        manager_metrics = agent.update_manager(manager_buffer, manager_minibatch_size, args.update_epochs)

        # Logging
        sps = int(global_step / (time.time() - start_time))
        metrics = {**worker_metrics, **manager_metrics}
        # Group 0 is each actor, group 1 its critic (see HPPOAgent.__init__).
        metrics["charts/manager_lr"] = agent.manager_optimizer.param_groups[0]["lr"]
        metrics["charts/manager_critic_lr"] = agent.manager_optimizer.param_groups[-1]["lr"]
        metrics["charts/worker_lr"] = agent.worker_optimizer.param_groups[0]["lr"]
        metrics["charts/worker_critic_lr"] = agent.worker_optimizer.param_groups[-1]["lr"]
        metrics["charts/SPS"] = sps
        metrics["charts/num_episodes"] = len(completed_returns)
        if near_goal_ax:
            metrics["charts/worker_ax_near_goal"] = float(np.mean(np.concatenate(near_goal_ax)))
            metrics["charts/worker_value_near_goal"] = float(np.mean(np.concatenate(near_goal_value)))
            metrics["charts/near_goal_samples"] = float(sum(a.size for a in near_goal_ax))

        log_line = (f"update={update} global_step={global_step} SPS={sps} "
                    f"w_ev={metrics['worker/explained_variance']:.3f} "
                    f"m_ev={metrics['manager/explained_variance']:.3f} "
                    f"m_v_bias={metrics['manager/value_bias']:.1f} "
                    # Manager-collapse diagnostics (see ManagerActor's
                    # docstring in algorithms/hppo/hppo.py): printed inline,
                    # not just logged to W&B, so the update where these spike
                    # or crater is visible without leaving the console.
                    f"m_adv_std={metrics['manager/adv_std_raw']:.4f} "
                    f"m_kl_max={metrics['manager/approx_kl_max']:.4f} "
                    f"m_ratio_max={metrics['manager/ratio_max_dev']:.3f}")
        if "charts/worker_ax_near_goal" in metrics:
            log_line += (f" w_ax@goal={metrics['charts/worker_ax_near_goal']:+.2f} "
                         f"w_v@goal={metrics['charts/worker_value_near_goal']:.2f}")
        if completed_returns:
            mean_return = float(np.mean(completed_returns))
            mean_length = float(np.mean(completed_lengths))
            success_rate = float(np.mean(completed_successes))
            collision_count_mean = float(np.mean(completed_collision_counts))
            metrics["charts/episodic_return"] = mean_return
            metrics["charts/episodic_length"] = mean_length
            metrics["charts/success_rate"] = success_rate
            metrics["charts/collision_count_mean"] = collision_count_mean
            log_line += (f" return={mean_return:.1f} length={mean_length:.0f} "
                         f"success_rate={success_rate:.2f} collisions/ep={collision_count_mean:.2f} "
                         f"(n={len(completed_returns)})")
            if completed_collision_impacts:
                # Individual per-contact penalties (not summed), so both the
                # typical and the single worst contact this update are visible.
                metrics["charts/collision_impact_mean"] = float(np.mean(completed_collision_impacts))
                metrics["charts/collision_impact_worst"] = float(np.min(completed_collision_impacts))
        print(log_line)

        if args.track:
            wandb.log(metrics, step=global_step)

        worker_buffer.reset()
        manager_buffer.reset()

        # Evaluation
        if update % args.eval_freq == 0:
            agent.manager_actor.eval()
            agent.manager_critic.eval()
            agent.worker_actor.eval()
            agent.worker_critic.eval()
            eval_returns = []
            eval_lengths = []
            eval_successes = []
            eval_collision_counts = []
            eval_collision_impacts = []
            for i in range(args.eval_episodes):
                eval_obs, info_eval = eval_env.reset(seed=args.seed + i)
                eval_obs_norm = agent.normalize_obs(eval_obs)
                eval_ep_reward = 0
                eval_ep_length = 0
                done_eval = False

                # Manager acts at step 0.
                #
                # Every name in this block is eval-local on purpose. It used to
                # reuse `current_goal`/`current_pos`/`manager_obs_tensor`, the
                # same names the rollout carries its in-flight segment in, so
                # each evaluation silently overwrote the training state with
                # whatever its last episode ended on and the rollout resumed
                # against a goal from a different episode. Vectorizing turned
                # that into a shape error, which is how it was found; it was a
                # live bug in the single-environment version too.
                eval_manager_obs = torch.tensor(eval_obs_norm, dtype=torch.float32).to(device)
                with torch.no_grad():
                    eval_manager_action = agent.get_manager_action(eval_manager_obs.unsqueeze(0), deterministic=True)
                eval_goal = agent.scale_goal(eval_manager_action.cpu().numpy()[0])
                eval_pos = np.array([eval_obs[0], eval_obs[1]])
                eval_step_in_c = 0

                while not done_eval:
                    with torch.no_grad():
                        eval_worker_action = agent.get_worker_action(
                            worker_input(eval_obs_norm, eval_goal).unsqueeze(0), deterministic=True
                        )

                    eval_next_obs, reward_eval, terminated_eval, truncated_eval, info_eval = eval_env.step(eval_worker_action.cpu().numpy()[0])
                    eval_next_obs_norm = agent.normalize_obs(eval_next_obs)
                    eval_ep_reward += reward_eval
                    eval_ep_length += 1
                    done_eval = terminated_eval or truncated_eval

                    eval_next_pos = np.array([eval_next_obs[0], eval_next_obs[1]])
                    eval_next_goal = eval_goal - (eval_next_pos - eval_pos)

                    eval_step_in_c += 1
                    if eval_step_in_c == args.manager_freq and not done_eval:
                        eval_manager_obs = torch.tensor(eval_next_obs_norm, dtype=torch.float32).to(device)
                        with torch.no_grad():
                            eval_manager_action = agent.get_manager_action(eval_manager_obs.unsqueeze(0), deterministic=True)
                        eval_goal = agent.scale_goal(eval_manager_action.cpu().numpy()[0])
                        eval_step_in_c = 0
                    elif not done_eval:
                        eval_goal = eval_next_goal

                    eval_obs = eval_next_obs
                    eval_obs_norm = eval_next_obs_norm
                    eval_pos = eval_next_pos

                eval_returns.append(eval_ep_reward)
                eval_lengths.append(eval_ep_length)
                eval_successes.append(bool(info_eval["is_success"]))
                # collision_count/collision_impacts are cumulative over the
                # whole episode (see SlalomEnv.step), so the last step's info
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
            # own eval_goal/eval_step_in_c above, just reattributed from
            # "post-step, before the next loop iteration" to "start of the
            # next policy_fn call" (same information either way).
            def make_policy_fn():
                state = {}

                def policy_fn(obs):
                    obs_norm = agent.normalize_obs(obs)
                    pos = np.array([obs[0], obs[1]])
                    if "goal" not in state:
                        manager_obs = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
                        with torch.no_grad():
                            manager_action = agent.get_manager_action(manager_obs, deterministic=True)
                        state["goal"] = agent.scale_goal(manager_action.cpu().numpy()[0])
                        state["step_in_c"] = 0
                    else:
                        decayed_goal = state["goal"] - (pos - state["pos"])
                        state["step_in_c"] += 1
                        if state["step_in_c"] == args.manager_freq:
                            manager_obs = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
                            with torch.no_grad():
                                manager_action = agent.get_manager_action(manager_obs, deterministic=True)
                            state["goal"] = agent.scale_goal(manager_action.cpu().numpy()[0])
                            state["step_in_c"] = 0
                        else:
                            state["goal"] = decayed_goal
                    state["pos"] = pos
                    with torch.no_grad():
                        worker_action = agent.get_worker_action(
                            worker_input(obs_norm, state["goal"]).unsqueeze(0), deterministic=True)
                    return worker_action.cpu().numpy()[0]

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
            agent.worker_actor.train()
            agent.worker_critic.train()

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
            # stop training. Saved once, the first time the criterion holds,
            # regardless of --solved-early-stop -- marks the moment the
            # policy became near-optimal even on a run left to run to its
            # full --total-timesteps budget for a coherent cross-seed plot.
            if consecutive_solved >= args.solved_consecutive:
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
