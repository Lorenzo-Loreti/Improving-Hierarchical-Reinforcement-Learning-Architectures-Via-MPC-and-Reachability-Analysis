"""The flat-PPO training loop, shared by every scenario's script_ppo.py.

scenarios/slalom/scripts/script_ppo.py and scenarios/tunnel/scripts/
script_ppo.py used to be ~430-line near-copies of this loop that differed
only in which environment classes they built, three defaults and the
--env-u-max help text, so every fix had to be made twice. The copies had
already begun to drift: the regime-study print computed the agility ratio's
T as a hardcoded 1.0 s in one and as 10 * dt in the other (the same number
today). Each script is now a thin wrapper that describes its scenario in a
`Scenario` and hands it to `main`.

The two PPO+MPC variants deliberately keep a full script per scenario:
theirs differ in substance (scenario-specific diagnostics, and the
investigations documented in their comments), not just in names. hPPO
used to as well, until its two copies turned out to be identical code and
were merged the same way, into algorithms/hppo/hppo_train.py.

Imported by bare name, like `ppo`, once algorithms/ppo and algorithms/ are
on sys.path -- each script_ppo.py puts them there.
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

from ppo import PPOAgent, RolloutBuffer
from optimal_solver import spawn_grid, precompute_optimal_grid, MinTimeSolver
from solved_check import check_solved
from metrics_log import MetricsLog
from spawn_coverage import SpawnCoverage


def _str2bool(x):
    return x.lower() in ['true', '1', 't', 'y', 'yes']


@dataclass(frozen=True)
class Scenario:
    """Everything the training loop needs to know about one scenario."""
    name: str                   # "slalom" / "tunnel": exp-name default and help texts
    env_cls: type               # plain env, for evaluation and the solved-check
    vec_env_cls: type           # batched, auto-resetting env, for rollouts
    make_env_config: Callable   # (u_max override or None, **overrides) -> env config
    total_timesteps: int        # default training budget
    wandb_project: str
    env_u_max_help: str         # scenario-specific: each is a different arm of the regime study
    script_dir: str             # checkpoints are written relative to the scenario's script


def parse_args(scenario):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=f"ppo_{scenario.name}",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--env-u-max", type=float, default=None,
        help=scenario.env_u_max_help)
    parser.add_argument("--noise-bound-p", type=float, default=0.0,
        help="radius, in metres, of the disk the position disturbance is "
             "drawn from, uniformly, every step (the env config's "
             "noise_bound_p, since 2026-09-28; a box's half-width until "
             "2026-10-04). 0.0, the default, is the deterministic environment. The same "
             "flag as PPO+MPC's (algorithms/ppo_mpc/ppo_mpc_train.py), so the "
             "algorithms can be compared on one disturbed environment; the "
             "level chosen there is 0.005 with --noise-bound-v 0.05")
    parser.add_argument("--noise-bound-v", type=float, default=0.0,
        help="radius, in m/s, of the disk the velocity disturbance is drawn "
             "from (the env config's noise_bound_v); see --noise-bound-p")
    parser.add_argument("--init-sampler", type=str, choices=["uniform", "sobol"],
        default=scenario.make_env_config(None).init_sampler,
        help="how the vector env draws the start of each training episode "
             "from the spawn box (the env config's init_sampler, since "
             "2026-10-03): 'uniform', independent draws, or 'sobol', a "
             "scrambled Sobol' sequence (randomized quasi-Monte Carlo) that "
             "spreads successive starts evenly over the box. Every start is "
             "uniform on the box under both, so the objective is the same; "
             "evaluation draws its starts independently under both. The "
             "default is the env config's. See "
             "scenarios/slalom/envs/spawn_sampler.py and docs/init-sampler.md")
    parser.add_argument("--torch-deterministic", type=_str2bool, default=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=_str2bool, default=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", action="store_true",
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default=scenario.wandb_project,
        help="the wandb's project name")
    parser.add_argument("--eval-freq", type=int, default=10,
        help="evaluate the agent every eval_freq updates. 10 rather than 5 "
             "because --num-steps 128 doubled the update count: this holds the "
             "number of evaluations per run, and so the evaluation share of the "
             "wall clock, where it was")
    parser.add_argument("--eval-episodes", type=int, default=20,
        help="number of episodes to evaluate the agent")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
        help="directory (relative to the scenario's script) to save model checkpoints in")
    parser.add_argument("--save-eval-checkpoints", type=_str2bool, default=False,
        help="if toggled, also save the agent at every evaluation, as "
             "eval_<global_step>.pt next to best.pt and final.pt, so the policy "
             "can be replayed at any point of its learning curve (the study "
             "scripts' progression figure, algorithms/study.py, uses them). Off "
             "by default: each is a few hundred kB. Saving touches no random "
             "number generator, so the run itself is unchanged")
    parser.add_argument("--solved-early-stop", type=_str2bool, default=True,
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
    parser.add_argument("--total-timesteps", type=int, default=scenario.total_timesteps,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--anneal-lr", type=_str2bool, default=True,
        help="if toggled, the learning rate decays linearly to 0 over training. "
             "On by default, unlike hPPO's: see --num-steps for the "
             "measurement that settled both")
    parser.add_argument("--num-envs", type=int, default=8,
        help=f"the number of parallel {scenario.name} environments to collect rollouts from")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout. "
             "128 rather than 256 halves the batch and so doubles the number of "
             "policy updates for the same sample budget, the same minibatch size "
             "and (to within 0.5%%) the same number of gradient steps. Both this "
             "and --anneal-lr differ from the hPPO worker's defaults (2048 "
             "steps per rollout, constant LR), and aligning the two was "
             "measured both ways on 2026-09-24 (13 slalom seeds, full 500k "
             "budget, every early stop disabled; docs/benchmark.md). Flat PPO "
             "on hPPO's settings (--anneal-lr false --num-steps 256 "
             "--num-minibatches 8 --eval-freq 5) solved no faster (median / "
             "mean 51k / 54k steps against 51k / 51k, p = 0.25) and held the "
             "strict solved criterion less well afterwards: 88%% of checks in "
             "the last 100k steps against 100%% (p = 0.02), 11/13 seeds solved "
             "at the end against 13/13, and the tunnel solved one evaluation "
             "later (31k against 20k on all 3 seeds). Each change alone was "
             "within noise (no annealing: 97%% in the last 100k, p = 0.08; the "
             "longer rollout: 100%%). hPPO on flat PPO's settings did no better "
             "either (see hppo_train.py), so each keeps its own")
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
        help="coefficient of the entropy bonus, fixed for the whole run (see "
             "PPOAgent for why there is no autotuning)")
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


def make_policy_fn(agent):
    """The controller evaluation scores: one raw observation in, the
    deterministic action out. The random-start evaluation, the solved-check
    and the study scripts (algorithms/study.py) all run exactly this, so they
    cannot drift apart. Memoryless -- hPPO's counterpart needs a fresh closure
    per episode (algorithms/hppo/hppo_train.py), this one does not.

    At module level rather than a closure inside `train` so that a policy
    reloaded from a checkpoint (`load_agent`) is driven by the same code the
    training run evaluated it with."""
    def policy_fn(obs):
        with torch.no_grad():
            obs_tensor = torch.tensor(agent.normalize_obs(obs), dtype=torch.float32).to(agent.device)
            action = agent.act(obs_tensor.unsqueeze(0))
        return action.cpu().numpy()[0]

    return policy_fn


def load_agent(path, env, device="cpu"):
    """A trained agent rebuilt from the checkpoint at `path`, for evaluation.

    `env` is an instance of the environment it was trained on: its spaces give
    the network sizes and the action limits, the same way `train` reads them.
    The observation map travels in the checkpoint itself (see PPOAgent.save)."""
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        act_limit_low=float(env.action_space.low[0]),
        act_limit_high=float(env.action_space.high[0]),
        device=device,
    )
    agent.load(path)
    agent.actor.eval()
    agent.critic.eval()
    return agent


def main(scenario):
    train(parse_args(scenario), scenario)


def train(args, scenario):
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"

    # --env-u-max: see the identically-named flag in script_ppo_mpc.py. Flat
    # PPO is the control arm of the reachability regime study
    # (docs/reachability-regime-study.md): it has no goal space at all, so it
    # isolates how much of the degradation at low u_max is the *task* getting
    # harder rather than a hierarchy's goal interface failing. Each
    # scenario's help text says what its own arm contributes. Defaults to
    # None = the canonical environment, unchanged.
    env_config = scenario.make_env_config(args.env_u_max, noise_bound_p=args.noise_bound_p,
                                          noise_bound_v=args.noise_bound_v,
                                          init_sampler=args.init_sampler)
    if args.env_u_max is not None:
        # rho is defined over the hierarchical arms' macro-step, T =
        # manager_freq * dt = 10 * dt (1.0 s); flat PPO has none of its own
        # but reports the same ratio so the arms line up.
        T = 10 * env_config.dt
        print(f"REGIME STUDY: env u_max overridden to {env_config.u_max} "
              f"(canonical {scenario.make_env_config(None).u_max}); agility ratio rho = "
              f"v_max/(u_max*T) with T=10*dt={T:.1f}s is "
              f"{env_config.v_max / (env_config.u_max * T):.2f}")

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
        # save_code only captures the entry script, which is now the thin
        # scenario wrapper; log the code that actually trains alongside it.
        wandb.run.log_code(
            root=os.path.dirname(os.path.abspath(__file__)),
            include_fn=lambda path: os.path.basename(path) in ("ppo.py", "ppo_train.py"),
        )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Env setup: a batched, auto-resetting vector env for rollout collection,
    # a plain env for evaluation.
    vec_env = scenario.vec_env_cls(num_envs=args.num_envs, config=env_config)
    eval_env = scenario.env_cls(config=env_config)
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    obs_low = eval_env.observation_space.low
    obs_high = eval_env.observation_space.high

    # Solved-check setup: a fixed grid of initial conditions (not random
    # draws), so the oracle's optimal return for each point is solved once
    # here and reused on every later eval pass instead of being recomputed.
    # Any MinTimeSolver infeasibility raises here, at startup, rather than
    # mid-training.
    # The oracle replays open-loop actions through an env, which a
    # disturbance would knock off course, so it gets an undisturbed copy of
    # the environment (since 2026-09-28; the same one as before whenever
    # --noise-bound-p/-v are 0). Under a disturbance its returns are an upper
    # reference; evaluation and the solved-check still run on the disturbed
    # eval_env, reseeded per episode so every pass sees the same disturbances.
    oracle_env = scenario.env_cls(config=replace(env_config, noise_bound_p=0.0, noise_bound_v=0.0))
    solved_grid = spawn_grid(oracle_env, args.solved_grid_nx, args.solved_grid_ny)
    optimal_grid = precompute_optimal_grid(oracle_env, solved_grid, solver=MinTimeSolver())
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
        max_grad_norm=args.max_grad_norm,
        critic_lr_mult=args.critic_lr_mult,
        # Carried into the checkpoint so a reloaded policy knows the
        # observation map it was trained under.
        obs_low=obs_low,
        obs_high=obs_high,
        device=device,
    )

    # One base learning rate per parameter group (actor, then critic). The
    # anneal below scales each group against its own base; scaling every group
    # by args.learning_rate would silently reset the critic to the actor's rate
    # on the first update.
    base_lrs = [group["lr"] for group in agent.optimizer.param_groups]

    # See make_policy_fn: the controller evaluation and the solved-check score.
    policy_fn = make_policy_fn(agent)

    buffer = RolloutBuffer(args.num_steps, args.num_envs, obs_dim, act_dim, device)

    checkpoint_dir = os.path.join(scenario.script_dir, args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_eval_return = -float("inf")

    # Initialize environment
    obs, _ = vec_env.reset(seed=args.seed)
    # How evenly the latest training starts cover the spawn box, logged every
    # update as spawn/* (see algorithms/spawn_coverage.py).
    spawn_coverage = SpawnCoverage(vec_env.spawn_low, vec_env.spawn_high)
    spawn_coverage.add(obs)
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

    # A local copy of every W&B log call, written whether or not --track is
    # on; see algorithms/metrics_log.py.
    metrics_log = MetricsLog(checkpoint_dir, config={
        "scenario": scenario.name, "algorithm": "ppo", "run_name": run_name,
        "num_updates": num_updates, "actual_timesteps": actual_timesteps, **vars(args)})

    def log(metrics, step):
        metrics_log.log(metrics, step)
        if args.track:
            wandb.log(metrics, step=step)

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
                action, logprob, value = agent.get_action_and_value(obs_tensor)

            action_np = action.cpu().numpy()

            # Environment step (the vector env auto-resets envs that finish)
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
            # A finished env's next_obs is already its next episode's start.
            spawn_coverage.add(next_obs[done])

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
                completed_successes.append(bool(info["final_info"]["is_success"][i]))
                completed_collision_counts.append(int(info["final_info"]["collision_count"][i]))
                completed_collision_impacts.extend(info["final_info"]["collision_impacts"][i])
                ep_reward[i] = 0
                ep_length[i] = 0

        # Value of the observation after the rollout; GAE bootstraps from it
        # only where the last transition did not end its episode (dones[-1]).
        next_obs_tensor = torch.tensor(agent.normalize_obs(obs), dtype=torch.float32).to(device)
        with torch.no_grad():
            next_value = agent.get_value(next_obs_tensor)
        agent.compute_returns_and_advantage(buffer, next_value)

        # Optimize the policy and value network
        metrics = agent.update(buffer, args.num_minibatches, args.update_epochs)

        # Logging
        sps = int(global_step / (time.time() - start_time))
        # Group 0 is the actor, group 1 the critic (see PPOAgent.__init__).
        metrics["charts/learning_rate"] = agent.optimizer.param_groups[0]["lr"]
        metrics["charts/critic_learning_rate"] = agent.optimizer.param_groups[-1]["lr"]
        metrics["charts/SPS"] = sps
        metrics["charts/num_episodes"] = len(completed_returns)
        metrics.update(spawn_coverage.metrics())

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

        log(metrics, global_step)

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
                    eval_obs, reward_eval, terminated_eval, truncated_eval, info_eval = eval_env.step(policy_fn(eval_obs))
                    eval_ep_reward += reward_eval
                    eval_ep_length += 1
                    done_eval = terminated_eval or truncated_eval
                eval_returns.append(eval_ep_reward)
                eval_lengths.append(eval_ep_length)
                eval_successes.append(bool(info_eval["is_success"]))
                # collision_count/collision_impacts are cumulative over the
                # whole episode (see the env's step()), so the last step's
                # info already carries every contact the episode had.
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
            log(eval_metrics, global_step)

            if args.save_eval_checkpoints:
                agent.save(os.path.join(checkpoint_dir, f"eval_{global_step:07d}.pt"))

            if mean_eval_return > best_eval_return:
                best_eval_return = mean_eval_return
                agent.save(os.path.join(checkpoint_dir, "best.pt"))

            # Solved-check: the same fixed grid and the same deterministic
            # policy_fn as the random eval above on every pass, compared
            # point-by-point against the oracle's precomputed optimal return
            # for that exact starting position -- see algorithms/solved_check.py.
            solved_result = check_solved(lambda: policy_fn, eval_env, optimal_grid, args.solved_tolerance,
                                         seed=args.seed)
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
                    log({"solved/first_solved_step": global_step}, global_step)
                if args.solved_early_stop:
                    print("--solved-early-stop is on -- stopping training early.")
                    break

    agent.save(os.path.join(checkpoint_dir, "final.pt"))
    print(f"Saved checkpoints to {checkpoint_dir}")

    metrics_log.close()
    eval_env.close()
    if args.track:
        wandb.finish()
