"""PPO+MPC's agent: hPPO's manager over a tube-MPC worker.

Rebuilt from scratch on 2026-09-28, on the hPPO of that date
(algorithms/hppo/hppo.py). The hierarchy is hPPO's with the learned worker
replaced by a model-based one: a PPO manager emits a goal every
`manager_freq` steps, and the robust tube MPC of algorithms/tube_mpc.py
drives the plant toward it, knowing every state the corridor forbids. The
worker learns nothing, so this module holds only the manager: its networks
(`ManagerActor`/`ManagerCritic`, shared with hPPO through algorithms/
common.py), its ragged rollout buffer and its PPO update. The worker's
configuration still travels in the checkpoint (`mpc_settings`), because the
trained controller is the pair.

What the previous PPOMPCAgent had and this one does not, and why:

- The manager-collapse mitigations, all removed from hPPO between 2026-09-22
  and 2026-09-24 once the collapse was traced to hPPO's worker reward (see
  HPPOAgent's class comment and ManagerActor's docstring in algorithms/
  common.py): value-loss clipping (`clip_vloss`, default on here "by
  architectural similarity", never re-validated on this module), the
  early-stopping `target_kl_manager`, the advantage-std floor
  (`adv_std_floor_frac`) and a configurable ret_rms memory
  (`ret_rms_horizon_manager`). PPO+MPC's worker is a QP, not a learner, so
  it was immune to that collapse to begin with (docs/worker-termination-
  avoidance.md): the mitigations had even less to protect against here.
- The SAC-style entropy autotuner, inert at its learning rate for the same
  reason as hPPO's (HPPOAgent's class comment), and `vf_coef`, which only
  changes anything when actor and critic share parameters. The manager's
  entropy coefficient is a fixed 0.01, as hPPO's.
- A 4-D goal (delta_x, delta_y, v_x, v_y). The goal is now hPPO's 2-D
  displacement. The tube MPC steers toward an *equilibrium* (the note's
  target, (1.8)), and every equilibrium of a double integrator is at rest,
  so a nonzero target velocity would not be one. How fast the worker moves
  is left to the MPC: toward a goal 1.8 m away it accelerates to its top
  planned speed, and it brakes as the goal comes near -- which is why the
  goal box is 1.5x, not 1x, the distance reachable in a segment (see
  --max-goal-bound in ppo_mpc_train.py).

So `update_manager` is hPPO's `_update_head` for the manager, step for step,
which is itself flat PPO's update: tests/test_ppo_mpc.py checks that this
update and HPPOAgent.update_manager end on identical weights. Keep them in
step.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common import (
    normalize_obs, ScaledBeta, RunningMeanStd, ManagerActor, ManagerCritic,
)


class ManagerVecRolloutBuffer:
    """Ragged per-environment rollout store for the manager: hPPO's
    ManagerVecRolloutBuffer (algorithms/hppo/hppo.py), copied rather than
    imported so each algorithm's buffers stay its own (see algorithms/
    common.py).

    The manager writes one transition per c-step segment, and a segment also
    ends early whenever an episode does -- so after a rollout of T steps
    across N environments, environment i holds some S_i transitions and no
    two environments need agree. The store is therefore a (T, N) grid with a
    *separate write pointer per column*, and everything downstream slices
    each column to its own S_i.

    The GAE consequence is the one that would be easy to get wrong: a
    manager transition's temporal successor is the next transition *in the
    same environment*, so the recursion runs down each column independently
    and bootstraps off that column's own in-flight segment value. Running it
    across the flattened batch instead would chain environment i's last
    segment onto environment i+1's first.
    """

    def __init__(self, max_steps, num_envs, obs_dim, goal_dim, device):
        self.max_steps = max_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.device = device

        self.states = torch.zeros((max_steps, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((max_steps, num_envs, goal_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((max_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((max_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((max_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((max_steps, num_envs), dtype=torch.float32, device=device)

        self.returns = torch.zeros((max_steps, num_envs), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((max_steps, num_envs), dtype=torch.float32, device=device)

        # One write pointer per environment, not one for the whole buffer.
        self.steps = np.zeros(num_envs, dtype=np.int64)

    def add(self, mask, state, action, logprob, reward, value, done):
        """Append one transition to each environment selected by `mask`.

        `mask` is a (num_envs,) boolean array of the environments whose
        segment ended on this step. Every other argument is a full
        (num_envs, ...) batch; only the masked rows are read.
        """
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return
        if np.any(self.steps[idx] >= self.max_steps):
            raise IndexError(
                f"manager buffer overflow: an environment tried to store more "
                f"than max_steps={self.max_steps} segments in one rollout. "
                f"Size it at the per-environment step count, which is the "
                f"worst case (one segment per step).")
        rows = torch.as_tensor(self.steps[idx], device=self.device)
        cols = torch.as_tensor(idx, device=self.device)

        def _t(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device)

        self.states[rows, cols] = _t(state)[cols]
        self.actions[rows, cols] = _t(action)[cols]
        self.logprobs[rows, cols] = _t(logprob)[cols]
        self.rewards[rows, cols] = _t(reward)[cols]
        self.values[rows, cols] = _t(value)[cols]
        self.dones[rows, cols] = _t(done)[cols]
        self.steps[idx] += 1

    def compute_returns_and_advantage(self, next_value, gamma, gae_lambda):
        """GAE(lambda) down each environment's own chain of segments.

        Vectorized across environments despite the ragged lengths: the loop
        walks t from the longest column down to 0, and at each t an
        environment is `active` only once t has fallen inside its own filled
        prefix, so each column's recursion starts from zero at its own last
        transition. `next_value` is a (num_envs,) vector, the value of each
        environment's in-flight segment; whether to bootstrap from it is that
        column's last transition's own done flag (`dones[t]` flags "the
        transition at index t ended an episode"), so there is no `next_done`.
        """
        steps = torch.as_tensor(self.steps, device=self.device)
        T = int(self.steps.max()) if self.steps.size else 0
        zero = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        lastgaelam = zero.clone()
        self.advantages.zero_()
        for t in reversed(range(T)):
            active = t < steps
            is_last = t == steps - 1
            nextnonterminal = 1.0 - self.dones[t]
            if t + 1 < T:
                nextvalues = torch.where(is_last, next_value, self.values[t + 1])
            else:
                # At the longest column's final index every active environment
                # is at its own last transition.
                nextvalues = next_value
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            lastgaelam = torch.where(active, lastgaelam, zero)
            self.advantages[t] = lastgaelam
        self.returns = self.advantages + self.values

    def reset(self):
        self.steps[:] = 0

    @property
    def total_steps(self):
        """Transitions stored across all environments -- the update's batch size."""
        return int(self.steps.sum())

    def _flat(self, tensor):
        """Concatenate each environment's filled prefix, environment-major."""
        return torch.cat([tensor[: self.steps[i], i] for i in range(self.num_envs)], dim=0)

    def get(self):
        """(states, actions, logprobs, values, returns, advantages), each
        environment's filled prefix concatenated environment-major -- the
        tuple hPPO's buffers return."""
        return (
            self._flat(self.states),
            self._flat(self.actions),
            self._flat(self.logprobs),
            self._flat(self.values),
            self._flat(self.returns),
            self._flat(self.advantages),
        )


class PPOMPCAgent:
    """The manager half of HPPOAgent (algorithms/hppo/hppo.py), plus the
    tube-MPC worker's settings for the checkpoint.

    Defaults match ppo_mpc_train.py's flags, so an agent built directly
    behaves like the trained configuration; change the two together. The one
    default that depends on the plant, `max_goal_bound`, is the value the
    training script derives for the canonical plant: 1.5 x the displacement
    reachable in one segment, 1.5 * v_max * manager_freq * dt = 1.8 m (see
    --max-goal-bound in ppo_mpc_train.py).

    `gamma` is the environment-step discount; the manager's own is derived
    as gamma**manager_freq, since one manager step is c environment steps
    (the SMDP/options treatment -- see HPPOAgent's comment for why it is
    derived rather than stored separately).
    """

    def __init__(self, obs_dim, goal_dim=2,
                 lr_manager=3e-4, gamma=0.99, manager_freq=10,
                 gae_lambda=0.95, clip_coef=0.2, ent_coef_manager=0.01,
                 max_grad_norm=0.5, obs_low=None, obs_high=None,
                 max_goal_bound=1.8, critic_lr_mult=3.0,
                 mpc_settings=None, device="cpu"):
        self.gamma = gamma
        self.manager_freq = manager_freq
        self.gamma_manager = gamma ** manager_freq
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef_manager = ent_coef_manager
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.manager_limit_low = torch.tensor(-1.0, dtype=torch.float32, device=device)
        self.manager_limit_high = torch.tensor(1.0, dtype=torch.float32, device=device)

        # Observation bounds and goal scale travel with the agent so that
        # `save()` writes a self-describing checkpoint; serialised as lists,
        # because torch.load's weights_only default rejects numpy arrays.
        self.obs_low = None if obs_low is None else np.asarray(obs_low, dtype=np.float32).ravel()
        self.obs_high = None if obs_high is None else np.asarray(obs_high, dtype=np.float32).ravel()
        self.max_goal_bound = max_goal_bound

        # The worker is not a network, but it is half of the controller: a
        # reloaded manager driven by a differently tuned MPC (another
        # horizon, weights or disturbance model W) is a different
        # controller. TubeMPCWorker.SETTINGS lists the keys.
        self.mpc_settings = dict(mpc_settings or {})

        # ManagerActor's docstring (algorithms/common.py) is the canonical
        # writeup of the manager-collapse investigation.
        self.manager_actor = ManagerActor(obs_dim, goal_dim).to(device)
        self.manager_critic = ManagerCritic(obs_dim).to(device)

        # The critic learns in standardized-return space; these statistics map
        # its output back to the raw reward scale GAE works in. The manager's
        # reward is the discounted environment reward over a segment, so its
        # returns carry the environment's raw scale (O(1000) with the goal
        # reward) against a freshly initialized critic that outputs ~0; see
        # RunningMeanStd's docstring (algorithms/common.py).
        self.manager_ret_rms = RunningMeanStd()

        # Two parameter groups, so the critic can run at a higher learning
        # rate than the actor; callers annealing the rate must scale every
        # group against its own base (see ppo_mpc_train.py).
        self.manager_optimizer = optim.Adam([
            {"params": list(self.manager_actor.parameters()), "lr": lr_manager},
            {"params": list(self.manager_critic.parameters()), "lr": lr_manager * critic_lr_mult},
        ], eps=1e-5)

    # --- input encoding ---------------------------------------------------

    def normalize_obs(self, obs):
        """Apply the observation map this agent was constructed with."""
        if self.obs_low is None or self.obs_high is None:
            raise ValueError(
                "This agent has no observation bounds, so it cannot normalize "
                "observations. Construct it with obs_low/obs_high (or load a "
                "checkpoint that carries them) -- feeding raw physical units to "
                "a policy trained on normalized ones fails silently.")
        return normalize_obs(obs, self.obs_low, self.obs_high)

    def scale_goal(self, normalized_goal):
        """The manager's [-1, 1]^2 action -> a physical goal displacement."""
        return normalized_goal * self.max_goal_bound

    # --- manager ----------------------------------------------------------

    def manager_policy_forward(self, state, action=None, deterministic=False):
        """Actor-only forward, so the update can take the critic's raw
        (standardized) output without a second critic pass."""
        alpha, beta = self.manager_actor(state)
        probs = ScaledBeta(alpha, beta, low=self.manager_limit_low, high=self.manager_limit_high)
        if action is None:
            action = probs.deterministic_sample() if deterministic else probs.sample()
        return action, probs.log_prob(action).sum(dim=-1), probs.entropy().sum(dim=-1)

    def manager_act(self, state, deterministic=True):
        """Goal only, with no critic pass: for evaluation and the solved-check."""
        return self.manager_policy_forward(state, deterministic=deterministic)[0]

    def get_manager_action_and_value(self, state):
        """Sampled goal, its log-probability and the state's value: what a
        manager transition stores."""
        action, logprob, _ = self.manager_policy_forward(state)
        return action, logprob, self.get_manager_value(state)

    def get_manager_value(self, state):
        """Value on the raw reward scale, for GAE and truncation bootstrapping."""
        return self.manager_critic(state).squeeze(-1) * self.manager_ret_rms.std + self.manager_ret_rms.mean

    def compute_manager_returns_and_advantage(self, buffer, next_value):
        """GAE over `buffer` at the segment-level discount. The agent is the
        single source of truth for the discount, which the caller's
        truncation bootstrap must also use."""
        buffer.compute_returns_and_advantage(
            next_value, gamma=self.gamma_manager, gae_lambda=self.gae_lambda)

    # --- persistence ------------------------------------------------------

    def save(self, path):
        torch.save({
            "manager_actor": self.manager_actor.state_dict(),
            "manager_critic": self.manager_critic.state_dict(),
            "manager_optimizer": self.manager_optimizer.state_dict(),
            # Without this the critic's output is meaningless on reload.
            "manager_ret_rms": self.manager_ret_rms.state_dict(),
            # Without these the actor's input and its goal are wrong on reload.
            "obs_low": None if self.obs_low is None else self.obs_low.tolist(),
            "obs_high": None if self.obs_high is None else self.obs_high.tolist(),
            "max_goal_bound": self.max_goal_bound,
            # The cadence is part of the trained controller and fixes the
            # manager's discount.
            "manager_freq": self.manager_freq,
            # And so is the worker; see __init__.
            "mpc_settings": self.mpc_settings,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.manager_actor.load_state_dict(checkpoint["manager_actor"])
        self.manager_critic.load_state_dict(checkpoint["manager_critic"])
        self.manager_optimizer.load_state_dict(checkpoint["manager_optimizer"])
        self.manager_ret_rms.load_state_dict(checkpoint["manager_ret_rms"])
        if checkpoint["obs_low"] is not None:
            self.obs_low = np.asarray(checkpoint["obs_low"], dtype=np.float32)
            self.obs_high = np.asarray(checkpoint["obs_high"], dtype=np.float32)
        self.max_goal_bound = checkpoint["max_goal_bound"]
        self.manager_freq = checkpoint["manager_freq"]
        self.gamma_manager = self.gamma ** self.manager_freq
        self.mpc_settings = dict(checkpoint["mpc_settings"])
        # Checkpoints of the PPOMPCAgent before 2026-09-28 (4-D goals,
        # `goal_scale`, an autotuned entropy coefficient) do not load: their
        # manager was trained against the old MPCWorker, a different worker.

    # --- update -----------------------------------------------------------

    def update_manager(self, buffer, num_minibatches, update_epochs):
        """One PPO update of the manager: HPPOAgent._update_head (algorithms/
        hppo/hppo.py) for the manager head, step for step, which is itself
        flat PPO's update plus what a ragged batch needs (the empty-batch
        guard, the cap on the minibatch count). The comments there explain
        each choice; tests/test_ppo_mpc.py holds the two to identical weights.
        """
        states, actions, logprobs, values, returns, advantages = buffer.get()
        batch_size = states.shape[0]
        prefix = "manager"

        if batch_size == 0:
            # No segment ended in this rollout: nothing to update, and the
            # statistics below would be NaN or raise.
            return {
                f"{prefix}/loss_policy": float("nan"),
                f"{prefix}/loss_value": float("nan"),
                f"{prefix}/entropy": float("nan"),
                f"{prefix}/approx_kl": float("nan"),
                f"{prefix}/approx_kl_max": float("nan"),
                f"{prefix}/ratio_max_dev": float("nan"),
                f"{prefix}/clipfrac": float("nan"),
                f"{prefix}/value_bias": float("nan"),
                f"{prefix}/value_target_mean": float(self.manager_ret_rms.mean),
                f"{prefix}/value_target_std": float(self.manager_ret_rms.std),
                f"{prefix}/explained_variance": float("nan"),
                f"{prefix}/adv_std_raw": float("nan"),
                f"{prefix}/batch_size": 0.0,
            }

        # Per-batch advantage normalization, with std_mean for the same
        # last-bit reason as hPPO's.
        adv_std, adv_mean = torch.std_mean(advantages)
        adv_std_raw = float(adv_std)
        advantages = (advantages - adv_mean) / (adv_std_raw + 1e-8)

        # Value-target normalization; get_manager_value undoes it.
        self.manager_ret_rms.update(returns)
        norm_returns = (returns - self.manager_ret_rms.mean) / self.manager_ret_rms.std

        clipfracs = []
        pg_losses, v_losses, entropy_losses, approx_kls = [], [], [], []
        ratio_max_dev = 0.0

        # Exactly `num_minibatches` near-equal minibatches per epoch, never
        # more than there are samples (the manager's batch is ragged).
        num_minibatches = min(num_minibatches, batch_size)

        for _ in range(update_epochs):
            b_inds = torch.randperm(batch_size, device=self.device)
            for mb_inds in torch.tensor_split(b_inds, num_minibatches):
                _, newlogprob, entropy = self.manager_policy_forward(states[mb_inds], actions[mb_inds])
                newvalue = self.manager_critic(states[mb_inds]).squeeze(-1)
                logratio = newlogprob - logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                    ratio_max_dev = max(ratio_max_dev, (ratio - 1.0).abs().max().item())

                mb_advantages = advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * ((newvalue - norm_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                # Actor and critic are separate networks, so the sum only
                # takes both steps in one backward pass.
                loss = pg_loss - self.ent_coef_manager * entropy_loss + v_loss

                self.manager_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.manager_actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.manager_critic.parameters(), self.max_grad_norm)
                self.manager_optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                approx_kls.append(approx_kl.item())

        returns_np = returns.detach().cpu().numpy()
        values_np = values.detach().cpu().numpy()
        var_y = np.var(returns_np)
        explained_var = np.nan if var_y == 0 else 1 - np.var(returns_np - values_np) / var_y

        # hPPO's metric names, so the two managers' curves compare directly.
        return {
            f"{prefix}/loss_policy": float(np.mean(pg_losses)),
            f"{prefix}/loss_value": float(np.mean(v_losses)),
            f"{prefix}/entropy": float(np.mean(entropy_losses)),
            f"{prefix}/approx_kl": float(np.mean(approx_kls)),
            f"{prefix}/approx_kl_max": float(np.max(approx_kls)),
            f"{prefix}/ratio_max_dev": float(ratio_max_dev),
            f"{prefix}/clipfrac": float(np.mean(clipfracs)),
            f"{prefix}/value_bias": float((values - returns).mean().item()),
            f"{prefix}/value_target_mean": float(self.manager_ret_rms.mean),
            f"{prefix}/value_target_std": float(self.manager_ret_rms.std),
            f"{prefix}/explained_variance": float(explained_var),
            f"{prefix}/adv_std_raw": adv_std_raw,
            f"{prefix}/batch_size": float(batch_size),
        }
