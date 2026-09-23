import os
import sys
import argparse
import time
import numpy as np
import torch
import wandb
# Scenario root (this script's parent), for `envs`; then algorithms/ppo, for
# the flat `ppo` module it imports by bare name.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithms', 'ppo')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithms')))
from envs.config import SlalomEnvConfig
from envs.slalom_env import SlalomEnv
from envs.vec_slalom_env import SlalomVecEnv
from envs.width_profile import slalom_profile
from ppo import PPOAgent, RolloutBuffer
from optimal_solver import spawn_grid, precompute_optimal_grid, MinTimeSolver
from solved_check import check_solved

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ppo_slalom",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--env-u-max", type=float, default=None,
        help="REGIME STUDY ONLY. Override the environment's acceleration "
             "limit u_max (default None = the canonical 2.5). Mirrors the "
             "flag of the same name in script_ppo_mpc.py so flat PPO can be "
             "run as the control arm of the reachability regime study; see "
             "docs/reachability-regime-study.md")
    parser.add_argument("--torch-deterministic", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", action="store_true",
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Slalom-PPO-NoNoise",
        help="the wandb's project name")
    parser.add_argument("--eval-freq", type=int, default=10,
        help="evaluate the agent every eval_freq updates. 10 rather than 5 "
             "because --num-steps 128 doubled the update count: this holds the "
             "number of evaluations per run, and so the evaluation share of the "
             "wall clock, where it was")
    parser.add_argument("--eval-episodes", type=int, default=20,
        help="number of episodes to evaluate the agent")
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

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--anneal-lr", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True,
        help="if toggled, the learning rate decays linearly to 0 over training")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel slalom environments to collect rollouts from")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout. "
             "128 rather than 256 halves the batch and so doubles the number of "
             "policy updates for the same sample budget, the same minibatch size "
             "and (to within 0.5%%) the same number of gradient steps")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma. 0.99, not the 0.999 the reward scale "
             "was designed around: see the thesis PPO chapter, 11.1")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches. 4 against the halved batch keeps the "
             "minibatch size at 256, as it was with 8 against 2048")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy. With --autotune-ent-coef, this is only the initial value")
    parser.add_argument("--autotune-ent-coef", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True,
        help="learn the entropy coefficient via SAC-style dual ascent toward a target entropy. On by default; pass --autotune-ent-coef false to hold --ent-coef fixed instead")
    parser.add_argument("--target-entropy-frac", type=float, default=0.35,
        help="target entropy as a fraction of the policy's max achievable entropy (log(action range) per dim); only used with --autotune-ent-coef")
    parser.add_argument("--ent-coef-lr", type=float, default=3e-4,
        help="learning rate for the entropy coefficient's own optimizer; only used with --autotune-ent-coef")
    parser.add_argument("--ent-coef-min", type=float, default=1e-4,
        help="lower clamp on the autotuned entropy coefficient; only used with --autotune-ent-coef")
    parser.add_argument("--ent-coef-max", type=float, default=1.0,
        help="upper clamp on the autotuned entropy coefficient; only used with --autotune-ent-coef")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--critic-lr-mult", type=float, default=3.0,
        help="the critic's learning rate as a multiple of --learning-rate. The "
             "critic gets its own optimiser parameter group; 1.0 restores the "
             "single-rate behaviour")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"

    # --env-u-max: see the identically-named flag in script_ppo_mpc.py. This
    # is the flat-PPO control arm of the reachability regime study
    # (docs/reachability-regime-study.md): flat PPO has no goal space at all,
    # so it isolates how much of the degradation at low u_max is the *task*
    # getting harder rather than a hierarchy's goal interface failing.
    # Defaults to None = the canonical environment, unchanged.
    env_config = SlalomEnvConfig(
        width_profile=slalom_profile(),
        **({} if args.env_u_max is None else {"u_max": args.env_u_max}),
    )
    if args.env_u_max is not None:
        print(f"REGIME STUDY: env u_max overridden to {env_config.u_max} "
              f"(canonical {SlalomEnvConfig().u_max}); agility ratio rho = "
              f"v_max/(u_max*T) with T=1.0s is "
              f"{env_config.v_max / (env_config.u_max * 1.0):.2f}")

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
    # SlalomEnv for evaluation.
    vec_env = SlalomVecEnv(num_envs=args.num_envs, config=env_config)
    eval_env = SlalomEnv(config=env_config)
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
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
    consecutive_solved = 0
    solved_checkpoint_saved = False

    # Agent and Buffer
    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit_low=float(eval_env.action_space.low[0]),
        act_limit_high=float(eval_env.action_space.high[0]),
        lr=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        critic_lr_mult=args.critic_lr_mult,
        # Carried into the checkpoint so a reloaded policy knows the
        # observation map it was trained under.
        obs_low=obs_low,
        obs_high=obs_high,
        device=device,
        autotune_ent_coef=args.autotune_ent_coef,
        target_entropy_frac=args.target_entropy_frac,
        ent_coef_lr=args.ent_coef_lr,
        ent_coef_min=args.ent_coef_min,
        ent_coef_max=args.ent_coef_max,
    )

    # One base learning rate per parameter group (actor, then critic). The
    # anneal below scales each group against its own base; scaling every group
    # by args.learning_rate would silently reset the critic to the actor's rate
    # on the first update.
    base_lrs = [group["lr"] for group in agent.optimizer.param_groups]

    buffer = RolloutBuffer(args.num_steps, args.num_envs, obs_dim, act_dim, device)

    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_eval_return = -float("inf")

    # Initialize environment
    obs, _ = vec_env.reset(seed=args.seed)
    global_step = 0
    start_time = time.time()

    num_updates = args.total_timesteps // args.batch_size
    actual_timesteps = num_updates * args.batch_size
    print(f"batch_size={args.batch_size} minibatch_size={args.minibatch_size} "
          f"num_updates={num_updates} timesteps={actual_timesteps}"
          + (f" (requested {args.total_timesteps}; {args.total_timesteps - actual_timesteps} "
             f"dropped by integer division)" if actual_timesteps != args.total_timesteps else ""))
    if args.track:
        wandb.config.update({"num_updates": num_updates, "actual_timesteps": actual_timesteps})

    ep_reward = np.zeros(args.num_envs)
    ep_length = np.zeros(args.num_envs, dtype=np.int64)

    for update in range(1, num_updates + 1):
        # Linear LR decay: once the policy has converged a constant LR keeps
        # injecting noise into both heads off a shrinking advantage signal.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            for group, base_lr in zip(agent.optimizer.param_groups, base_lrs):
                group["lr"] = frac * base_lr

        completed_returns = []
        completed_lengths = []
        completed_successes = []
        completed_collision_counts = []
        completed_collision_impacts = []

        for step in range(0, args.num_steps):
            global_step += args.num_envs

            # Action selection
            obs_tensor = torch.tensor(agent.normalize_obs(obs), dtype=torch.float32).to(device)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_tensor)

            action_np = action.cpu().numpy()

            # Environment step (SlalomVecEnv auto-resets envs that finish)
            next_obs, reward, terminated, truncated, info = vec_env.step(action_np)

            ep_reward += reward
            ep_length += 1

            # Bootstrap the value of envs that hit max_steps without
            # terminating, since their next_obs is already a new episode.
            trunc_only = truncated & ~terminated
            if np.any(trunc_only):
                final_obs = agent.normalize_obs(info["final_observation"][trunc_only])
                with torch.no_grad():
                    final_value = agent.get_value(
                        torch.tensor(final_obs, dtype=torch.float32).to(device)
                    ).cpu().numpy()
                reward[trunc_only] += agent.gamma * final_value

            done = terminated | truncated

            buffer.add(
                obs_tensor,
                action,
                logprob,
                torch.tensor(reward, dtype=torch.float32).to(device),
                value,
                torch.tensor(done.astype(np.float32)).to(device)
            )

            obs = next_obs

            for i in np.flatnonzero(done):
                completed_returns.append(ep_reward[i])
                completed_lengths.append(ep_length[i])
                completed_successes.append(info["final_info"]["is_success"][i])
                completed_collision_counts.append(int(info["final_info"]["collision_count"][i]))
                completed_collision_impacts.extend(info["final_info"]["collision_impacts"][i])
                ep_reward[i] = 0
                ep_length[i] = 0

        # Bootstrap value if not done
        next_obs_tensor = torch.tensor(agent.normalize_obs(obs), dtype=torch.float32).to(device)
        with torch.no_grad():
            next_value = agent.get_value(next_obs_tensor)
        next_done = torch.tensor(done.astype(np.float32)).to(device)
        agent.compute_returns_and_advantage(buffer, next_value, next_done)

        # Optimize the policy and value network
        metrics = agent.update(buffer, args.minibatch_size, args.update_epochs)

        # Logging
        sps = int(global_step / (time.time() - start_time))
        # Group 0 is the actor, group 1 the critic (see PPOAgent.__init__).
        metrics["charts/learning_rate"] = agent.optimizer.param_groups[0]["lr"]
        metrics["charts/critic_learning_rate"] = agent.optimizer.param_groups[-1]["lr"]
        metrics["charts/SPS"] = sps
        metrics["charts/num_episodes"] = len(completed_returns)

        log_line = (f"update={update} global_step={global_step} SPS={sps} "
                    f"ev={metrics['loss/explained_variance']:.3f} "
                    f"v_bias={metrics['loss/value_bias']:.1f}")
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

        buffer.reset()

        # Evaluation
        if update % args.eval_freq == 0:
            agent.actor.eval()
            agent.critic.eval()
            eval_returns = []
            eval_lengths = []
            eval_successes = []
            eval_collision_counts = []
            eval_collision_impacts = []
            for i in range(args.eval_episodes):
                eval_obs, info_eval = eval_env.reset(seed=args.seed + i)
                eval_ep_reward = 0
                eval_ep_length = 0
                done_eval = False
                while not done_eval:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(agent.normalize_obs(eval_obs), dtype=torch.float32).to(device)
                        action, _, _, _ = agent.get_action_and_value(obs_tensor.unsqueeze(0), deterministic=True)
                    next_obs_eval, reward_eval, terminated_eval, truncated_eval, info_eval = eval_env.step(action.cpu().numpy()[0])
                    eval_ep_reward += reward_eval
                    eval_ep_length += 1
                    done_eval = terminated_eval or truncated_eval
                    eval_obs = next_obs_eval
                eval_returns.append(eval_ep_reward)
                eval_lengths.append(eval_ep_length)
                eval_successes.append(info_eval["is_success"])
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
            def policy_fn(obs):
                with torch.no_grad():
                    obs_tensor = torch.tensor(agent.normalize_obs(obs), dtype=torch.float32).to(device)
                    action, _, _, _ = agent.get_action_and_value(obs_tensor.unsqueeze(0), deterministic=True)
                return action.cpu().numpy()[0]

            solved_result = check_solved(lambda: policy_fn, eval_env, optimal_grid, args.solved_tolerance)
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

            agent.actor.train()
            agent.critic.train()

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
