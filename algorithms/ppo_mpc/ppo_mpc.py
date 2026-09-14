import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def normalize_obs(obs, low, high):
    """Map physical [low, high] observation bounds to [-1, 1].

    Ported from RL/PPO/ppo.py and HRL/hPPO/hppo.py. The environment's
    observation bounds are fixed and hard-enforced by the env, so there is
    nothing to estimate: the map is exact, exactly invertible, and needs no
    running statistics. Without it the manager network sees raw physical units --
    p_x in [-1, 11] against v_y in [-2, 2], a >5x dynamic-range mismatch
    across input dimensions feeding the same orthogonally-initialized layer.

    Canonical definition. `PPOMPCAgent` stores the bounds it was built with
    and exposes this as a method, so a reloaded checkpoint carries its own
    observation map instead of depending on the caller to reproduce it.
    """
    return 2.0 * (obs - low) / (high - low) - 1.0


class RunningMeanStd:
    """Chan et al. parallel running mean/variance, used to normalize the
    manager critic's regression targets. Ported from RL/PPO/ppo.py.

    The manager's reward is the discounted sum of the environment's own
    reward over a c-step segment, so its returns inherit the raw +-500
    terminal scale -- O(400) targets for a freshly initialized critic that
    outputs ~0 and can move each weight by at most `lr` per step. The climb
    costs more gradient steps than a run provides, and the critic ends up
    correlated with the true value but hundreds of units biased. Regressing
    on standardized targets removes that climb entirely.
    """

    def __init__(self, epsilon=1e-4, horizon=10):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
        # Cap the effective sample count at `horizon` batches so the statistics
        # track the *current* return distribution. With an unbounded count the
        # early-training transient permanently inflates the variance, and late
        # targets get squeezed into a narrow band -- a milder rerun of the
        # scale problem this class exists to prevent.
        self.horizon = horizon

    def update(self, x):
        batch_mean = float(x.mean())
        batch_var = float(x.var(unbiased=False))
        batch_count = x.numel()

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / tot_count) / tot_count
        self.count = min(tot_count, self.horizon * batch_count)

    @property
    def std(self):
        return math.sqrt(self.var) + 1e-8

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]

# ----------------- Distributions ----------------- #

class ScaledBeta:
    def __init__(self, alpha, beta, low=-1.0, high=1.0):
        self.dist = Beta(alpha, beta)
        self.low = low
        self.scale = high - low

    def sample(self):
        return self.dist.sample() * self.scale + self.low

    def deterministic_sample(self):
        # Mean of Beta distribution is alpha / (alpha + beta)
        mean = self.dist.concentration1 / (self.dist.concentration1 + self.dist.concentration0)
        return mean * self.scale + self.low

    def log_prob(self, action):
        # Unscale action back to [0, 1]
        unscaled_action = (action - self.low) / self.scale
        unscaled_action = torch.clamp(unscaled_action, 1e-5, 1.0 - 1e-5)
        # Apply log determinant of Jacobian correction
        return self.dist.log_prob(unscaled_action) - torch.log(self.scale)

    def entropy(self):
        # Apply entropy shift correction
        return self.dist.entropy() + torch.log(self.scale)

# ----------------- Buffers ----------------- #

class ManagerRolloutBuffer:
    def __init__(self, num_steps, obs_dim, goal_dim, device):
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.device = device
        
        self.states = torch.zeros((num_steps, obs_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((num_steps, goal_dim), dtype=torch.float32).to(device)
        self.logprobs = torch.zeros((num_steps,), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((num_steps,), dtype=torch.float32).to(device)
        self.values = torch.zeros((num_steps,), dtype=torch.float32).to(device)
        self.dones = torch.zeros((num_steps,), dtype=torch.float32).to(device)
        
        self.returns = torch.zeros((num_steps,), dtype=torch.float32).to(device)
        self.advantages = torch.zeros((num_steps,), dtype=torch.float32).to(device)
        
        self.step = 0
        
    def add(self, state, action, logprob, reward, value, done):
        self.states[self.step] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.step] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.logprobs[self.step] = torch.as_tensor(logprob, dtype=torch.float32, device=self.device)
        self.rewards[self.step] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.values[self.step] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.dones[self.step] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.step += 1
        
    def compute_returns_and_advantage(self, next_value, next_done, gamma, gae_lambda):
        """GAE(lambda) over the stored (partially filled) rollout.

        `gamma` and `gae_lambda` are required rather than defaulted: they used
        to default to 0.99/0.95 while the manager is in fact discounted at
        gamma**manager_freq, so a caller who omitted them got silently
        different discounting from the trained configuration. Prefer
        `PPOMPCAgent.compute_manager_returns_and_advantage`, which sources
        both from the agent.
        """
        lastgaelam = 0
        for t in reversed(range(self.step)):
            if t == self.step - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t]
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            self.advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        self.returns = self.advantages + self.values

    def reset(self):
        self.step = 0

    def get_values(self):
        """The stored value predictions, in the same order as `get()`.

        Exists so `update_manager` can read pre-update values without knowing
        which buffer layout it was handed: this one stores a flat prefix, the
        vectorized one a ragged per-environment grid.
        """
        return self.values[:self.step]

    def get(self):
        return (
            self.states[:self.step],
            self.actions[:self.step],
            self.logprobs[:self.step],
            self.returns[:self.step],
            self.advantages[:self.step]
        )


class ManagerVecRolloutBuffer:
    """Ragged per-environment manager rollout store.

    The manager writes one transition per c-step segment, and a segment also
    ends early whenever an episode does -- so after a rollout of T steps across
    N environments, environment i holds some S_i transitions and no two
    environments need agree. The store is therefore a (T, N) grid with a
    *separate write pointer per column*, and everything downstream slices each
    column to its own S_i.

    The GAE consequence is the one that would be easy to get wrong: a manager
    transition's temporal successor is the next transition *in the same
    environment*, so the recursion runs down each column independently and
    bootstraps off that column's own in-flight segment value. Running it across
    the flattened batch instead would chain environment i's last segment onto
    environment i+1's first.

    Mirrors ManagerVecRolloutBuffer in HRL/hPPO/hppo.py, which faces the same
    ragged-cadence problem; only the goal dimension differs (4 here -- the
    manager commands a target velocity too -- against hPPO's 2).
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

        `mask` is a (num_envs,) boolean array of the environments whose segment
        ended on this step. Every other argument is a full (num_envs, ...)
        batch; only the masked rows are read, so the caller can pass the whole
        per-environment state without pre-filtering it.
        """
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return
        if np.any(self.steps[idx] >= self.max_steps):
            raise IndexError(
                f"manager buffer overflow: an environment tried to store more "
                f"than max_steps={self.max_steps} segments in one rollout. "
                f"Size it at the worker's per-environment step count, which is "
                f"the worst case (one segment per step)."
            )
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

    def compute_returns_and_advantage(self, next_value, next_done, gamma, gae_lambda):
        """GAE(lambda) down each environment's own chain of segments.

        Vectorized across environments despite the ragged lengths: the loop
        walks t from the longest column down to 0, and at each t an environment
        is `active` only once t has fallen inside its own filled prefix.
        `lastgaelam` is held at zero until then, so when a column's last
        transition is finally reached it starts the recursion from zero and
        takes the bootstrap branch -- identical to running the single-env loop
        on that column alone.

        `next_value` and `next_done` are (num_envs,) vectors describing each
        environment's in-flight segment: the state its last stored transition
        led to.
        """
        steps = torch.as_tensor(self.steps, device=self.device)
        T = int(self.steps.max()) if self.steps.size else 0
        zero = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        lastgaelam = zero.clone()
        self.advantages.zero_()
        for t in reversed(range(T)):
            active = t < steps
            is_last = t == steps - 1
            # Same indexing convention as the single-env buffer: `dones[t]`
            # flags "the transition at index t ended an episode", so the mask
            # is 1 - dones[t] -- at t, not t + 1. Only `values` is read one
            # step ahead.
            nextnonterminal = torch.where(is_last, 1.0 - next_done, 1.0 - self.dones[t])
            if t + 1 < T:
                nextvalues = torch.where(is_last, next_value, self.values[t + 1])
            else:
                # At the longest column's final index every active environment
                # is at its own last transition, so the non-bootstrap branch is
                # unreachable here and self.values[t + 1] does not exist.
                nextvalues = next_value
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            # Environments not yet inside their filled prefix contribute
            # nothing and must not carry a value into their own last step.
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

    def get_values(self):
        return self._flat(self.values)

    def get(self):
        return (
            self._flat(self.states),
            self._flat(self.actions),
            self._flat(self.logprobs),
            self._flat(self.returns),
            self._flat(self.advantages),
        )

# ----------------- Manager PPO Networks ----------------- #

class ManagerActor(nn.Module):
    def __init__(self, obs_dim, goal_dim):
        super().__init__()
        self.goal_dim = goal_dim
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, goal_dim * 2), std=0.01)
        )
        
    def forward(self, obs):
        x = self.net(obs)
        alpha = F.softplus(x[..., :self.goal_dim]) + 1.0
        beta = F.softplus(x[..., self.goal_dim:]) + 1.0
        return alpha, beta

class ManagerCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        
    def forward(self, obs):
        return self.net(obs)

# ----------------- PPOMPC Agent Wrapper ----------------- #

class PPOMPCAgent:
    # Defaults deliberately match what each scenario's script_ppo_mpc.py passes, so an
    # agent constructed directly behaves like the trained configuration
    # instead of silently differing from it. Change the two together.
    def __init__(self, obs_dim, goal_dim,
                 lr_manager=3e-4,
                 gamma=0.99, manager_freq=10,
                 gae_lambda=0.95, clip_coef=0.2,
                 ent_coef_manager=0.01, vf_coef=0.5, max_grad_norm=0.5,
                 obs_low=None, obs_high=None, goal_scale=None,
                 critic_lr_mult=3.0, device="cpu"):
        self.gamma = gamma
        self.manager_freq = manager_freq
        # The manager decides once per c-step segment, so one manager step is
        # c environment steps (the standard SMDP/options treatment). Derived
        # rather than passed separately: storing an independent gamma_manager
        # is exactly how HRL/hPPO's agent and training script previously drifted
        # apart (the agent held gamma_manager=gamma while the script ran GAE at
        # gamma**c). See HPPOAgent in HRL/hPPO/hppo.py.
        self.gamma_manager = gamma ** manager_freq
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef_manager = ent_coef_manager
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.manager_limit_low = torch.tensor(-1.0, dtype=torch.float32, device=device)
        self.manager_limit_high = torch.tensor(1.0, dtype=torch.float32, device=device)

        # Observation bounds and goal scale travel with the agent so that
        # `save()` produces a self-describing checkpoint. Without them a
        # reloaded policy silently receives un-normalized observations, or
        # scales a normalized goal action by the wrong physical bound, and
        # behaves nothing like the trained one.
        self.obs_low = None if obs_low is None else np.asarray(obs_low, dtype=np.float32).ravel()
        self.obs_high = None if obs_high is None else np.asarray(obs_high, dtype=np.float32).ravel()
        self.goal_scale = None if goal_scale is None else np.asarray(goal_scale, dtype=np.float32).ravel()

        # --- Manager (PPO) ---
        self.manager_actor = ManagerActor(obs_dim, goal_dim).to(device)
        self.manager_critic = ManagerCritic(obs_dim).to(device)

        # The critic learns in standardized-return space; these statistics
        # map its output back to the raw reward scale that GAE works in.
        self.manager_ret_rms = RunningMeanStd()

        # Two parameter groups, so the critic can run at a higher learning
        # rate than the actor -- the critic is the binding constraint early in
        # training, while the actor is the head an over-large step damages.
        # Callers must anneal *every* group; each scenario's script_ppo_mpc.py captures
        # the base learning rates once and scales each group against its own.
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
                "a policy trained on normalized ones fails silently."
            )
        return normalize_obs(obs, self.obs_low, self.obs_high)

    def scale_goal(self, normalized_goal):
        """Map the manager's [-1, 1]^goal_dim action to a physical goal."""
        if self.goal_scale is None:
            raise ValueError(
                "This agent has no goal_scale, so it cannot scale a normalized "
                "goal action. Construct it with goal_scale (or load a "
                "checkpoint that carries it)."
            )
        return normalized_goal * self.goal_scale

    # --- Manager Methods ---
    def manager_policy_forward(self, state, action=None, deterministic=False):
        """Actor-only forward, so the update loop can take the critic's raw
        (standardized) output without a second critic pass."""
        alpha, beta = self.manager_actor(state)
        probs = ScaledBeta(alpha, beta, low=self.manager_limit_low, high=self.manager_limit_high)
        if action is None:
            if deterministic:
                action = probs.deterministic_sample()
            else:
                action = probs.sample()
        return action, probs.log_prob(action).sum(dim=-1), probs.entropy().sum(dim=-1)

    def get_manager_action(self, state, deterministic=False):
        with torch.no_grad():
            action, _, _ = self.manager_policy_forward(state, deterministic=deterministic)
        return action

    def get_manager_action_and_value(self, state, action=None, deterministic=False):
        action, logprob, entropy = self.manager_policy_forward(state, action, deterministic)
        return action, logprob, entropy, self.get_manager_value(state)

    def get_manager_value(self, state):
        """Value on the raw reward scale, for GAE and truncation bootstrapping."""
        return self.manager_critic(state).squeeze(-1) * self.manager_ret_rms.std + self.manager_ret_rms.mean

    def compute_manager_returns_and_advantage(self, buffer, next_value, next_done):
        """Run GAE over `buffer` at the manager's segment-level discount.

        The agent is the single source of truth for gamma_manager/gae_lambda:
        they are also what the caller must use for truncation bootstrapping,
        and having two independent copies is how those silently drift apart.
        """
        buffer.compute_returns_and_advantage(
            next_value, next_done, gamma=self.gamma_manager, gae_lambda=self.gae_lambda
        )

    # --- persistence --------------------------------------------------------

    def save(self, path):
        torch.save({
            "manager_actor": self.manager_actor.state_dict(),
            "manager_critic": self.manager_critic.state_dict(),
            "manager_optimizer": self.manager_optimizer.state_dict(),
            # Without this the critic's output is meaningless on reload.
            "manager_ret_rms": self.manager_ret_rms.state_dict(),
            # Without these the actor's input, and the goal it emits, are
            # wrong on reload.
            "obs_low": None if self.obs_low is None else self.obs_low.tolist(),
            "obs_high": None if self.obs_high is None else self.obs_high.tolist(),
            "goal_scale": None if self.goal_scale is None else self.goal_scale.tolist(),
            # The cadence is part of the trained controller, not a run detail:
            # a reloaded manager that re-plans at a different c is a different
            # controller, and it also fixes the manager's discount.
            "manager_freq": self.manager_freq,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.manager_actor.load_state_dict(checkpoint["manager_actor"])
        self.manager_critic.load_state_dict(checkpoint["manager_critic"])
        # Guarded: checkpoints written before the critic had its own parameter
        # group hold a single-group optimiser state, which Adam refuses to
        # load into the two-group optimiser built above.
        try:
            self.manager_optimizer.load_state_dict(checkpoint["manager_optimizer"])
        except (ValueError, KeyError):
            print(f"warning: {path} predates the two-group optimiser; actor and "
                  "critic weights loaded, optimiser state discarded")
        if "manager_ret_rms" in checkpoint:
            self.manager_ret_rms.load_state_dict(checkpoint["manager_ret_rms"])
        if checkpoint.get("obs_low") is not None:
            self.obs_low = np.asarray(checkpoint["obs_low"], dtype=np.float32)
        if checkpoint.get("obs_high") is not None:
            self.obs_high = np.asarray(checkpoint["obs_high"], dtype=np.float32)
        if checkpoint.get("goal_scale") is not None:
            self.goal_scale = np.asarray(checkpoint["goal_scale"], dtype=np.float32)
        if checkpoint.get("manager_freq") is not None:
            self.manager_freq = checkpoint["manager_freq"]
            self.gamma_manager = self.gamma ** self.manager_freq

    def update_manager(self, buffer, minibatch_size, update_epochs):
        states, actions, logprobs, returns, advantages = buffer.get()
        batch_size = states.shape[0]

        # Pre-update value predictions, on the raw reward scale. These are the
        # estimates that actually produced the advantages, which is what
        # explained_variance is defined against. Scoring the critic *after*
        # its update epochs on this same batch measures training-set fit
        # instead, and is optimistically biased.
        #
        # Read through `get_values()` rather than by slicing `.values`
        # directly: the single-env buffer stores a flat prefix and the
        # vectorized one a ragged per-environment grid, and only the buffer
        # knows how to line its storage up with what `get()` returned.
        values = buffer.get_values()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Value-target normalization: refresh the running statistics on this
        # batch's returns, then regress the critic on standardized targets.
        # `get_manager_value` undoes this so GAE keeps working on the raw scale.
        self.manager_ret_rms.update(returns)
        norm_returns = (returns - self.manager_ret_rms.mean) / self.manager_ret_rms.std

        clipfracs = []

        pg_losses, v_losses, entropy_losses, approx_kls = [], [], [], []

        for epoch in range(update_epochs):
            b_inds = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy = self.manager_policy_forward(
                    states[mb_inds], actions[mb_inds]
                )
                # Raw critic output: standardized space, matching norm_returns.
                newvalue = self.manager_critic(states[mb_inds]).squeeze(-1)
                logratio = newlogprob - logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = advantages[mb_inds]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((newvalue - norm_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - self.ent_coef_manager * entropy_loss + v_loss * self.vf_coef

                self.manager_optimizer.zero_grad()
                loss.backward()
                # Clipped separately so neither head can eat the other's share
                # of a shared gradient-norm budget.
                nn.utils.clip_grad_norm_(self.manager_actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.manager_critic.parameters(), self.max_grad_norm)
                self.manager_optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                approx_kls.append(approx_kl.item())

        # Values are compared on the raw return scale, so explained_variance
        # stays comparable across runs with and without value normalization.
        returns_np = returns.cpu().numpy()
        values_np = values.cpu().numpy()
        var_y = np.var(returns_np)
        explained_var = np.nan if var_y == 0 else 1 - np.var(returns_np - values_np) / var_y

        return {
            "manager/loss_policy": float(np.mean(pg_losses)),
            "manager/loss_value": float(np.mean(v_losses)),
            "manager/entropy": float(np.mean(entropy_losses)),
            "manager/approx_kl": float(np.mean(approx_kls)),
            "manager/clipfrac": float(np.mean(clipfracs)),
            "manager/value_bias": float((values - returns).mean().item()),
            "manager/value_target_mean": float(self.manager_ret_rms.mean),
            "manager/value_target_std": float(self.manager_ret_rms.std),
            "manager/explained_variance": float(explained_var),
        }
