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
from envs.config import SlalomEnvConfig
from envs.slalom_env import SlalomEnv
from envs.vec_slalom_env import SlalomVecEnv
from envs.width_profile import slalom_profile
from ppo_mpc import PPOMPCAgent, ManagerVecRolloutBuffer
from mpc_worker import MPCWorker

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ppo_mpc_slalom", help="name of this experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--cuda", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--track", action="store_true", help="track with wandb")
    parser.add_argument("--wandb-project-name", type=str, default="Slalom-PPO-MPC-NoNoise")
    
    # Run params
    parser.add_argument("--total-timesteps", type=int, default=200000)
    
    # Manager params (PPO)
    parser.add_argument("--learning-rate-manager", type=float, default=3e-4)
    parser.add_argument("--anneal-lr", type=lambda x: str(x).lower() == 'true', default=True,
        help="if toggled, the manager's learning rate decays linearly to 0 over training")
    parser.add_argument("--critic-lr-mult", type=float, default=3.0,
        help="the manager critic's learning rate as a multiple of --learning-rate-manager")
    parser.add_argument("--manager-freq", type=int, default=10, help="macro-step length c")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel slalom environments to collect rollouts "
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
    parser.add_argument("--ent-coef-manager", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval-freq", type=int, default=1, help="evaluate the agent every eval_freq updates")
    parser.add_argument("--eval-episodes", type=int, default=10, help="number of episodes to evaluate the agent")
    parser.add_argument("--stall-patience", type=int, default=10,
        help="warn after this many consecutive updates in which every "
             "finished episode was a truncation -- the stall optimum (a local "
             "optimum where the policy stops moving and every episode runs "
             "out the clock; see RL/PPO/PPO.md section 17.4 and "
             "HRL/hPPO/hppo_explanation.md section 7.2). Nothing in this "
             "script's reward is specific to the manager-worker split, so it "
             "is not immune. 0 disables the check")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
        help="directory (relative to this script) to save model checkpoints in")

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
    """A bare SlalomEnv, for evaluation.

    Rollouts come from a batched `SlalomVecEnv` instead. `RecordEpisodeStatistics`
    is gone with the single-environment rollout it wrapped: returns and lengths
    are counted in the loop, which also reaches the success and collision flags
    the wrapper does not carry.
    """
    return SlalomEnv(config=config)

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"

    # Single source of truth for this run's environment geometry -- the vec
    # rollout env, the eval env, and (below) the MPC worker's own box all
    # derive from this one object instead of independently restating the
    # same numbers, which is how those three used to silently drift apart.
    # slalom_profile()'s own calibration args (manager_freq/dt/etc.) are
    # deliberately fixed defaults, not --manager-freq -- the environment
    # must be identical bytes across every architecture this repo compares
    # and immune to any hyperparameter sweep on top of it.
    env_config = SlalomEnvConfig(width_profile=slalom_profile())
    
    if args.track:
        wandb.init(project=args.wandb_project_name, sync_tensorboard=False, config=vars(args), name=run_name, save_code=True)
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Env setup: a batched SlalomVecEnv for rollout collection, a plain
    # SlalomEnv for evaluation -- the same split flat PPO and hPPO use.
    vec_env = SlalomVecEnv(num_envs=args.num_envs, config=env_config)
    eval_env = make_env(config=env_config)
    obs_dim = eval_env.observation_space.shape[0]
    obs_low = eval_env.observation_space.low
    obs_high = eval_env.observation_space.high
    goal_dim = 4 # delta x, delta y, v_x, v_y
    max_goal_bound = 10.0
    v_max = 2.0
    goal_scale = np.array([max_goal_bound, max_goal_bound, v_max, v_max])

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
        device=device
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
    # that merely happened to match SlalomEnvConfig's defaults, with nothing
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
    stalled_updates = 0

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
        completed_collisions = []
        completed_target_dists = []

        # Linear LR decay: once the policy has converged a constant LR keeps
        # injecting noise into both heads off a shrinking advantage signal.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            for group, base_lr in zip(agent.manager_optimizer.param_groups, base_lrs):
                group["lr"] = frac * base_lr

        for step in range(args.num_steps_per_env):
            global_step += args.num_envs

            # Action selection (worker - MPC). One QP per environment: they
            # re-plan over different remaining horizons, so there is no single
            # batched solve to make.
            steps_left = args.manager_freq - worker_step_in_c
            worker_action_phys = worker.get_actions(obs, current_goal_phys, steps_left)

            # Step environments (SlalomVecEnv auto-resets envs that finish)
            next_obs, env_reward, terminated, truncated, info = vec_env.step(worker_action_phys)
            done = terminated | truncated

            # The auto-reset is why this is not a mechanical substitution. For a
            # finished env `next_obs` is already the *next* episode's first
            # observation, so the goal decrement and the truncation bootstrap
            # must read the pre-reset state instead -- otherwise the goal is
            # decremented by a teleport back to the slalom entrance.
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

            # Which managers act now? A segment ends after c worker steps or
            # when its episode does, and the environments do not agree on when.
            manager_act_now = (worker_step_in_c == c) | done
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
                completed_collisions.append(bool(info["final_info"]["collision"][i]))
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

        log_line = f"update={update} global_step={global_step} SPS={sps}"
        if completed_returns:
            mean_return = float(np.mean(completed_returns))
            metrics["charts/episodic_return"] = mean_return
            metrics["charts/episodic_length"] = float(np.mean(completed_lengths))
            metrics["charts/success_rate"] = float(np.mean(completed_successes))
            metrics["charts/collision_rate"] = float(np.mean(completed_collisions))
            metrics["charts/episodic_target_dist_sum"] = float(np.mean(completed_target_dists))
            log_line += (f" return={mean_return:.1f} "
                         f"length={np.mean(completed_lengths):.0f} "
                         f"success_rate={np.mean(completed_successes):.2f} "
                         f"collision_rate={np.mean(completed_collisions):.2f} "
                         f"(n={len(completed_returns)})")

            # Stall-trap detector, ported from flat PPO / hPPO. A policy that
            # simply stops moving makes every episode run out the clock:
            # nothing terminates, the undiscounted return is exactly
            # max_steps * step_penalty on every episode, and the batch's
            # return distribution goes constant, so the advantages carry no
            # signal and the run never recovers on its own. An undetected
            # stalled run is indistinguishable in the return column alone from
            # a policy that crashes immediately (SLALOM_ENV.md section 8.2),
            # so it silently poisons a results table if not flagged.
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

        manager_buffer.reset()
        
        # Evaluation
        if update % args.eval_freq == 0:
            agent.manager_actor.eval()
            agent.manager_critic.eval()
            eval_returns = []
            eval_lengths = []
            eval_target_dist_sums = []
            for i in range(args.eval_episodes):
                eval_obs, _ = eval_env.reset(seed=args.seed + i)
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

                    eval_next_obs, reward, terminated_eval, truncated_eval, _ = eval_env.step(eval_action_phys)
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
                
            mean_eval_return = float(np.mean(eval_returns))
            print(f"Evaluation at update {update}: return={mean_eval_return:.2f}, length={np.mean(eval_lengths):.2f}")
            if args.track:
                wandb.log({
                    "eval/episodic_return": mean_eval_return,
                    "eval/episodic_length": np.mean(eval_lengths),
                    "eval/episodic_target_dist_sum": np.mean(eval_target_dist_sums),
                }, step=global_step)

            if mean_eval_return > best_eval_return:
                best_eval_return = mean_eval_return
                agent.save(os.path.join(checkpoint_dir, "best.pt"))

            agent.manager_actor.train()
            agent.manager_critic.train()

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
