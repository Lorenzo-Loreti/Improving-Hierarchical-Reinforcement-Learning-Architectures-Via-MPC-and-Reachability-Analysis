import os
import argparse
import time
import numpy as np
import torch
import wandb
import sys
# Repository root, for `envs`; then this script's own directory, for the
# flat `hppo` module it imports by bare name.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs.config import TunnelEnvConfig
from envs.tunnel_env import TunnelEnv
from envs.vec_tunnel_env import TunnelVecEnv
from hppo import HPPOAgent, VecRolloutBuffer, ManagerVecRolloutBuffer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="hppo_tunnel",
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
    parser.add_argument("--wandb-project-name", type=str, default="Tunnel-hPPO-NoNoise",
        help="the wandb's project name")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
        help="directory (relative to this script) to save model checkpoints in")

    # Reward-scale overrides. Default to TunnelEnvConfig's own defaults, so
    # omitting these reproduces exactly the environment every other script
    # (flat PPO, PPO+MPC) trains against -- passing them is how a single run
    # can test a different risk/reward balance without moving that shared
    # default out from under everyone else. See envs/config.py.
    parser.add_argument("--collision-reward", type=float, default=TunnelEnvConfig().collision_reward,
        help="terminal reward on collision (before --impact-penalty-coef). "
             "Defaults to TunnelEnvConfig's own default, so omitting this "
             "reproduces the shared environment every other script trains "
             "against -- see envs/config.py for why it is -150, not -500")
    parser.add_argument("--impact-penalty-coef", type=float, default=TunnelEnvConfig().impact_penalty_coef,
        help="extra collision penalty, proportional to lateral impact speed "
             "|v_y|. 0.0 disables it")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=200000,
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
             "every parameter group of both optimisers. Off by default: on "
             "the sibling SlalomEnv, every seed that solved the task in "
             "testing did so by escaping an intermediate local optimum (a "
             "'rush and crash' policy that ignores the walls) at some update "
             "that varies by seed, and annealing to ~0 by the end of a fixed "
             "training budget starved that escape of gradient signal before "
             "it could happen on several seeds. See envs/config.py's "
             "collision_reward comment for the fuller story")
    parser.add_argument("--lr-floor-frac", type=float, default=0.0,
        help="the linear LR anneal stops at this fraction of the base rate "
             "instead of decaying all the way to 0. At 0.0 (default) this is "
             "exactly the previous anneal-to-zero behaviour. A floor keeps "
             "the last updates from freezing (approx_kl collapsing to ~0), "
             "which otherwise burns the tail of --total-timesteps on a "
             "policy that can no longer move -- see hppo_explanation.md "
             "stall-optimum discussion")
    parser.add_argument("--manager-freq", type=int, default=10,
        help="c: the number of steps the manager's goal is valid for. Also "
             "fixes the manager's discount, gamma**c")
    parser.add_argument("--max-goal-bound", type=float, default=10.0,
        help="the manager's [-1, 1] action is scaled by this to a physical "
             "goal displacement in metres, and divided by it again to form "
             "the worker's input")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel tunnel environments to collect rollouts "
             "from. Rollouts come from a batched TunnelVecEnv; each "
             "environment carries its own goal, manager cadence and segment "
             "accumulator (see section 8 of hppo_explanation.md)")
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
             "script used to train at: see RL/PPO/PPO.md section 11.1. The "
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
        help="coefficient of the manager entropy")
    parser.add_argument("--ent-coef-worker", type=float, default=0.01,
        help="coefficient of the worker entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="if set, stop an update early once approx_kl exceeds this. Off "
             "by default: clipping is then the only trust-region mechanism")
    parser.add_argument("--stall-patience", type=int, default=10,
        help="warn after this many consecutive updates in which every "
             "finished episode was a truncation (the stall optimum, "
             "RL/PPO/PPO.md section 17.4). 0 disables the check")
    parser.add_argument("--eval-freq", type=int, default=5,
        help="evaluate the agent every eval_freq updates. 5 rather than 1 "
             "because an evaluation of --eval-episodes episodes costs up to "
             "eval_episodes*max_steps environment steps, which at every "
             "update would dominate the wall clock over the 2048 steps of "
             "training it is measuring")
    parser.add_argument("--eval-episodes", type=int, default=20,
        help="number of episodes to evaluate the agent")
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
    """A bare TunnelEnv, for evaluation.

    Rollouts come from a batched `TunnelVecEnv` instead; this is the
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
    return TunnelEnv(config=config)

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"

    env_config = TunnelEnvConfig(
        collision_reward=args.collision_reward,
        impact_penalty_coef=args.impact_penalty_coef,
    )

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
    # TunnelEnv for evaluation -- the same split flat PPO uses.
    vec_env = TunnelVecEnv(num_envs=args.num_envs, config=env_config)
    eval_env = make_env(config=env_config)
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    goal_dim = 2 # delta x, delta y
    max_goal_bound = args.max_goal_bound
    obs_low = eval_env.observation_space.low
    obs_high = eval_env.observation_space.high

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
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        critic_lr_mult=args.critic_lr_mult,
        # Carried into the checkpoint so a reloaded hierarchy knows the
        # observation and goal maps it was trained under.
        obs_low=obs_low,
        obs_high=obs_high,
        max_goal_bound=max_goal_bound,
        device=device
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
        manager_action, manager_logprob, _, manager_value = agent.get_manager_action_and_value(
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
    stalled_updates = 0

    for update in range(1, num_updates + 1):
        # Linear LR decay: once the policies have converged a constant LR keeps
        # injecting noise into all four heads off a shrinking advantage signal.
        if args.anneal_lr:
            frac = 1.0 - (1.0 - args.lr_floor_frac) * (update - 1.0) / num_updates
            for group, base_lr in zip(agent.manager_optimizer.param_groups, base_lrs_manager):
                group["lr"] = frac * base_lr
            for group, base_lr in zip(agent.worker_optimizer.param_groups, base_lrs_worker):
                group["lr"] = frac * base_lr

        last_worker_done = np.zeros(args.num_envs, dtype=bool)
        # The done flag of each environment's most recently *stored* manager
        # transition. Per environment, because their segments end at different
        # steps -- and False until an environment has stored one at all.
        last_manager_done = np.zeros(args.num_envs, dtype=bool)

        completed_returns = []
        completed_lengths = []
        completed_successes = []
        completed_collisions = []

        for step in range(0, args.num_steps_per_env):
            global_step += args.num_envs

            # Worker action selection, one batched forward pass over all envs
            obs_goal_tensor = worker_input(obs_norm, current_goal)
            with torch.no_grad():
                worker_action, worker_logprob, _, worker_value = agent.get_worker_action_and_value(
                    obs_goal_tensor
                )

            action_np = worker_action.cpu().numpy()

            # Environment step (TunnelVecEnv auto-resets envs that finish)
            next_obs, env_reward, terminated, truncated, info = vec_env.step(action_np)
            done = terminated | truncated
            next_obs_norm = agent.normalize_obs(next_obs)

            # The auto-reset is why this is not a mechanical substitution. For
            # a finished env `next_obs` is already the *next* episode's first
            # observation, so every quantity describing the transition that
            # just happened -- the goal decrement, the worker's progress
            # reward, both truncation bootstraps -- has to read the pre-reset
            # state instead. Taking it from `next_obs` would score the worker
            # on a teleport back to the tunnel entrance.
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

            # `terminated`, or forced True where truncated -- and TunnelVecEnv
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

            last_worker_done = worker_done

            worker_step_in_c += 1

            # Which managers act now? An environment's segment ends after c
            # worker steps or when its episode does, and the environments do
            # not agree on when that is.
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

                last_manager_done[manager_act_now] = done[manager_act_now]

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
                    new_action, new_logprob, _, new_value = agent.get_manager_action_and_value(
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
                completed_collisions.append(bool(info["final_info"]["collision"][i]))
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

        # Bootstrap value if not done
        with torch.no_grad():
            # Worker bootstrap, per environment
            next_worker_value = agent.get_worker_value(worker_input(obs_norm, current_goal))
            agent.compute_worker_returns_and_advantage(
                worker_buffer, next_worker_value,
                next_done=torch.as_tensor(last_worker_done.astype(np.float32), device=device)
            )

            # Manager bootstrap. `manager_value` is the value of each
            # environment's segment currently in flight, whose start is exactly
            # the state that environment's last stored manager transition led
            # to -- so it is already the per-column bootstrap the ragged GAE
            # wants.
            agent.compute_manager_returns_and_advantage(
                manager_buffer, manager_value,
                next_done=torch.as_tensor(last_manager_done.astype(np.float32), device=device)
            )

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

        log_line = (f"update={update} global_step={global_step} SPS={sps} "
                    f"w_ev={metrics['worker/explained_variance']:.3f} "
                    f"m_ev={metrics['manager/explained_variance']:.3f} "
                    f"m_v_bias={metrics['manager/value_bias']:.1f}")
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

            # Stall-trap detector, ported from flat PPO. A few per cent of
            # seeds fall into a local optimum in which the policy simply stops
            # moving: every episode runs out the clock, so nothing terminates,
            # the undiscounted return is exactly `max_steps * step_penalty` for
            # every episode, and the batch's return distribution is constant.
            # The advantages then carry no signal and the run never recovers.
            # Detecting it is free, and an undetected stalled run silently
            # poisons a results table, because its return is indistinguishable
            # from a policy that crashes immediately (TUNNEL_ENV section 8.2).
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
            eval_collisions = []
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
                eval_collisions.append(bool(info_eval["collision"]))

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

            agent.manager_actor.train()
            agent.manager_critic.train()
            agent.worker_actor.train()
            agent.worker_critic.train()

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
