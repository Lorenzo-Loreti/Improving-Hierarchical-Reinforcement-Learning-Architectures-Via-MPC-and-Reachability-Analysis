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
from envs.config import TunnelEnvConfig
from envs.tunnel_env import TunnelEnv
from envs.vec_tunnel_env import TunnelVecEnv
from ppo import PPOAgent, RolloutBuffer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ppo_tunnel",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", action="store_true",
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Tunnel-PPO-NoNoise",
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

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=200000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--anneal-lr", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True,
        help="if toggled, the learning rate decays linearly to 0 over training")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel tunnel environments to collect rollouts from")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout. "
             "128 rather than 256 halves the batch and so doubles the number of "
             "policy updates for the same sample budget, the same minibatch size "
             "and (to within 0.5%) the same number of gradient steps")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma. 0.99, not the 0.999 the reward scale "
             "was designed around: see PPO.md section 11.1")
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
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--critic-lr-mult", type=float, default=3.0,
        help="the critic's learning rate as a multiple of --learning-rate. The "
             "critic gets its own optimiser parameter group; 1.0 restores the "
             "single-rate behaviour")
    parser.add_argument("--stall-patience", type=int, default=10,
        help="warn after this many consecutive updates in which every finished "
             "episode was a truncation (the stall optimum, PPO.md section 17.4). "
             "0 disables the check")
    parser.add_argument("--target-kl", type=float, default=None,
        help="if set, stop an update early once approx_kl exceeds this. Off by default: clipping is then the only trust-region mechanism")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"

    env_config = TunnelEnvConfig()

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

    # Env setup: a batched TunnelVecEnv for rollout collection, a plain
    # TunnelEnv for evaluation.
    vec_env = TunnelVecEnv(num_envs=args.num_envs, config=env_config)
    eval_env = TunnelEnv(config=env_config)
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    obs_low = eval_env.observation_space.low
    obs_high = eval_env.observation_space.high

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
        target_kl=args.target_kl,
        critic_lr_mult=args.critic_lr_mult,
        # Carried into the checkpoint so a reloaded policy knows the
        # observation map it was trained under.
        obs_low=obs_low,
        obs_high=obs_high,
        device=device
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
    stalled_updates = 0

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
        completed_collisions = []

        for step in range(0, args.num_steps):
            global_step += args.num_envs

            # Action selection
            obs_tensor = torch.tensor(agent.normalize_obs(obs), dtype=torch.float32).to(device)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_tensor)

            action_np = action.cpu().numpy()

            # Environment step (TunnelVecEnv auto-resets envs that finish)
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
                completed_collisions.append(info["final_info"]["collision"][i])
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
            collision_rate = float(np.mean(completed_collisions))
            metrics["charts/episodic_return"] = mean_return
            metrics["charts/episodic_length"] = mean_length
            metrics["charts/success_rate"] = success_rate
            metrics["charts/collision_rate"] = collision_rate
            log_line += (f" return={mean_return:.1f} length={mean_length:.0f} "
                         f"success_rate={success_rate:.2f} collision_rate={collision_rate:.2f} "
                         f"(n={len(completed_returns)})")

            # Stall-trap detector. A few per cent of seeds fall into a local
            # optimum in which the policy simply stops moving: every episode
            # runs out the clock, so nothing terminates, the undiscounted
            # return is exactly `max_steps * step_penalty` for every episode,
            # and the batch's return distribution is constant. The advantages
            # then carry no signal and the run never recovers -- the instance
            # measured here sat there for 178k further steps. Detecting it is
            # free, and an undetected stalled run silently poisons a results
            # table, because its return is indistinguishable from a policy
            # that crashes immediately (TUNNEL_ENV section 8.2).
            if not any(completed_successes) and not any(completed_collisions):
                stalled_updates += 1
            else:
                stalled_updates = 0
            if args.stall_patience and stalled_updates == args.stall_patience:
                print(f"WARNING: every episode has ended in truncation for "
                      f"{args.stall_patience} consecutive updates "
                      f"(return={mean_return:.1f}). This run has almost "
                      f"certainly entered the stall optimum and will not "
                      f"recover on its own; restart with a different --seed.")
        metrics["charts/stalled_updates"] = stalled_updates
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
            eval_collisions = []
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
                eval_collisions.append(info_eval["collision"])

            mean_eval_return = float(np.mean(eval_returns))
            success_rate = float(np.mean(eval_successes))
            collision_rate = float(np.mean(eval_collisions))
            print(f"Evaluation at update {update}: return={mean_eval_return:.2f}, length={np.mean(eval_lengths):.2f}, "
                  f"success_rate={success_rate:.2f}, collision_rate={collision_rate:.2f}")
            if args.track:
                wandb.log({
                    "eval/episodic_return": mean_eval_return,
                    "eval/episodic_length": np.mean(eval_lengths),
                    "eval/success_rate": success_rate,
                    "eval/collision_rate": collision_rate,
                }, step=global_step)

            if mean_eval_return > best_eval_return:
                best_eval_return = mean_eval_return
                agent.save(os.path.join(checkpoint_dir, "best.pt"))

            agent.actor.train()
            agent.critic.train()

    agent.save(os.path.join(checkpoint_dir, "final.pt"))
    print(f"Saved checkpoints to {checkpoint_dir}")
    if args.stall_patience and stalled_updates >= args.stall_patience:
        print(f"WARNING: this run finished in the stall optimum "
              f"({stalled_updates} consecutive truncation-only updates). Its "
              f"return is not comparable with a converged run -- discard it or "
              f"rerun with a different --seed.")

    eval_env.close()
    if args.track:
        wandb.finish()
