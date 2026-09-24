import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from common import layer_init, normalize_obs, ScaledBeta, RunningMeanStd


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.act_dim = act_dim
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim * 2), std=0.01)
        )
        
    def forward(self, obs):
        x = self.net(obs)
        alpha = F.softplus(x[..., :self.act_dim]) + 1.0
        beta = F.softplus(x[..., self.act_dim:]) + 1.0
        return alpha, beta

class CriticNetwork(nn.Module):
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

class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_dim, act_dim, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.batch_size = num_steps * num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        self.states = torch.zeros((num_steps, num_envs, obs_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((num_steps, num_envs, act_dim), dtype=torch.float32).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)

        self.returns = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)
        self.advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32).to(device)

        self.step = 0

    def add(self, state, action, logprob, reward, value, done):
        self.states[self.step] = state
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.dones[self.step] = done
        self.step += 1

    def compute_returns_and_advantage(self, next_value, gamma, gae_lambda):
        """GAE(lambda) over the stored rollout. `next_value` is the value of
        the observation that follows the last stored transition.

        `gamma` and `gae_lambda` are required rather than defaulted: they used
        to default to 0.99/0.95 while every real call site passed 0.999, so a
        caller who omitted them got silently different discounting from the
        trained configuration. Prefer `PPOAgent.compute_returns_and_advantage`,
        which sources both from the agent.

        Note the `dones` convention: `dones[t]` flags "the transition at index
        t ended an episode", so the bootstrap mask is `1 - dones[t]`. CleanRL
        stores "observation t begins a new episode" instead and masks with
        `dones[t+1]`. Both are correct; they are the same quantity indexed
        differently, and this one is the natural fit for an auto-resetting
        vector env, where the observation after a done already belongs to the
        next episode. It is also why there is no `next_done` argument: CleanRL
        needs one for the observation after the rollout, whereas here the last
        transition's own `dones[-1]` already says whether `next_value` may be
        bootstrapped from. (This method used to take a `next_done` anyway,
        which every caller filled with exactly `dones[-1]`.)
        """
        lastgaelam = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            nextvalues = next_value if t == self.num_steps - 1 else self.values[t + 1]
            nextnonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            self.advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        self.returns = self.advantages + self.values

    def reset(self):
        self.step = 0

    def get(self):
        """The rollout flattened to one batch, every field in the same order.
        `values` are the pre-update predictions stored at collection time."""
        states = self.states.reshape(self.batch_size, self.obs_dim)
        actions = self.actions.reshape(self.batch_size, self.act_dim)
        logprobs = self.logprobs.reshape(self.batch_size)
        values = self.values.reshape(self.batch_size)
        returns = self.returns.reshape(self.batch_size)
        advantages = self.advantages.reshape(self.batch_size)
        return states, actions, logprobs, values, returns, advantages

class PPOAgent:
    # Defaults deliberately match what ppo_train.py's flags default to, so an agent
    # constructed directly -- an evaluation notebook, say -- behaves like the
    # trained configuration instead of silently differing from it. Change the
    # two together.
    #
    # gamma is 0.99, not the 0.999 the reward scale was originally designed
    # around. Measured over 72 seeds, 0.999 makes the critic's regression
    # target degenerate as the policy converges: every state in a successful
    # episode earns almost the same discounted return, the spread of returns
    # inside a batch collapses, and explained_variance collapses with it. The
    # advantages that survive are then mostly critic error, which the
    # batch-level normalisation in update() rescales straight back to unit
    # variance. At 0.99 the return signal stays wide and the critic stays
    # accurate. See the thesis PPO chapter, section 11.1 (kept outside this repo).
    #
    # Deliberately absent, although HPPOAgent has them: value-loss clipping
    # (`clip_vloss`), the advantage-std floor (`adv_std_floor_frac`), a
    # configurable ret_rms horizon and an early-stopping `target_kl`. They
    # were once ported here for API parity but were never enabled in any
    # flat-PPO run, and the evidence behind them is about hPPO's manager
    # head, not about this agent -- see `clipped_value_loss` and
    # `floor_normalize` in algorithms/common.py.
    #
    # Also absent, and replaced by something simpler that does the same:
    #
    # - Entropy autotuning (SAC-style dual ascent on log(ent_coef)). Adam
    #   moves log(ent_coef) by about its learning rate per update, so at the
    #   3e-4 it always ran with, ent_coef ended at 0.0098-0.0099 on all six
    #   benchmark seeds (0.00983 after the slowest, 80-update slalom solve),
    #   and could not leave +-16% of 0.01 even over a full 500k-step run.
    #   Every flat-PPO result was in effect trained with the fixed
    #   ent_coef=0.01 used here; the same inertness was measured on hPPO
    #   (docs/worker-termination-avoidance.md, section 3).
    #
    # - vf_coef. It trades the value loss off against the policy loss only
    #   when the two share parameters. Here actor and critic are separate
    #   networks with separately clipped gradients, so the critic's gradient
    #   comes from the value loss alone and a constant factor on it is
    #   undone by Adam's normalisation -- all it could still change is the
    #   gradient-clip threshold and the weight of Adam's eps. The knob that
    #   actually sets the critic's step size is critic_lr_mult.
    def __init__(self, obs_dim, act_dim, act_limit_low=-1.0, act_limit_high=1.0, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_coef=0.2, ent_coef=0.01, max_grad_norm=0.5,
                 obs_low=None, obs_high=None,
                 critic_lr_mult=3.0, device="cpu"):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.act_limit_low = torch.tensor(act_limit_low, dtype=torch.float32, device=device)
        self.act_limit_high = torch.tensor(act_limit_high, dtype=torch.float32, device=device)

        # Observation bounds travel with the agent so that `save()` produces a
        # self-describing checkpoint. Without them a reloaded policy silently
        # receives un-normalized observations and behaves nothing like the
        # trained one. Held as arrays because normalize_obs runs every rollout
        # step; serialised as lists, because torch.load defaults to
        # weights_only=True, which rejects pickled numpy arrays.
        self.obs_low = None if obs_low is None else np.asarray(obs_low, dtype=np.float32).ravel()
        self.obs_high = None if obs_high is None else np.asarray(obs_high, dtype=np.float32).ravel()

        self.actor = ActorNetwork(obs_dim, act_dim).to(device)
        self.critic = CriticNetwork(obs_dim).to(device)

        # The critic learns in standardized-return space; these statistics map
        # its output back to the raw reward scale that GAE works in. Raw
        # returns here are large -- by the time the benchmark seeds solve the
        # task, ret_rms sits at mean ~700 / std ~170 on the slalom and mean
        # ~90 / std ~90 on the tunnel -- against a freshly initialized critic
        # that outputs ~0; see RunningMeanStd's docstring (algorithms/common.py)
        # for why that gap matters. (This comment used to quote O(400) at
        # gamma=0.999, the discount before the switch to 0.99.)
        self.ret_rms = RunningMeanStd()

        # Two parameter groups, so the critic can run at a higher learning
        # rate than the actor. The critic is the binding constraint early in
        # training -- advantages mean nothing until explained_variance has
        # climbed -- while the actor is the head an over-large step damages.
        #
        # Callers must anneal *every* group. Writing only `param_groups[0]`,
        # which was correct while there was a single group, now silently
        # leaves the critic un-annealed; ppo_train.py captures the base
        # learning rates once and scales each group against its own.
        self.optimizer = optim.Adam([
            {"params": list(self.actor.parameters()), "lr": lr},
            {"params": list(self.critic.parameters()), "lr": lr * critic_lr_mult},
        ], eps=1e-5)

    def policy_forward(self, state, action=None, deterministic=False):
        """Actor-only forward, so the update loop can take the critic's raw
        (standardized) output without a second critic pass."""
        alpha, beta = self.actor(state)
        probs = ScaledBeta(alpha, beta, low=self.act_limit_low, high=self.act_limit_high)
        if action is None:
            if deterministic:
                action = probs.deterministic_sample()
            else:
                action = probs.sample()
        # Log probability of a continuous action is the sum of the log probs of its dimensions
        return action, probs.log_prob(action).sum(dim=-1), probs.entropy().sum(dim=-1)

    def get_action_and_value(self, state):
        """Sampled action, its log-probability and the state's value: what a
        rollout step stores."""
        action, logprob, _ = self.policy_forward(state)
        return action, logprob, self.get_value(state)

    def act(self, state, deterministic=True):
        """Action only, with no critic pass -- for evaluation and the
        solved-check, which never use the value."""
        return self.policy_forward(state, deterministic=deterministic)[0]

    def get_value(self, state):
        """Value on the raw reward scale, for GAE and truncation bootstrapping."""
        return self.critic(state).squeeze(-1) * self.ret_rms.std + self.ret_rms.mean

    def compute_returns_and_advantage(self, buffer, next_value):
        """Run GAE over `buffer` using this agent's discount parameters.

        The agent is the single source of truth for gamma/gae_lambda: they are
        also what the caller must use for truncation bootstrapping, and having
        two independent copies is how those silently drift apart.
        """
        buffer.compute_returns_and_advantage(
            next_value, gamma=self.gamma, gae_lambda=self.gae_lambda
        )

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

    def save(self, path):
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            # Without these the critic's output is meaningless on reload.
            "ret_rms": self.ret_rms.state_dict(),
            # Without these the *actor's* input is wrong on reload.
            "obs_low": None if self.obs_low is None else self.obs_low.tolist(),
            "obs_high": None if self.obs_high is None else self.obs_high.tolist(),
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.ret_rms.load_state_dict(checkpoint["ret_rms"])
        # None only when the saving agent had no bounds either; this agent
        # then keeps whatever it was constructed with.
        if checkpoint["obs_low"] is not None:
            self.obs_low = np.asarray(checkpoint["obs_low"], dtype=np.float32)
            self.obs_high = np.asarray(checkpoint["obs_high"], dtype=np.float32)
        # Guards for checkpoints older than the two-group optimiser, the
        # observation bounds or ret_rms used to live here; no such flat-PPO
        # checkpoint is left (all nine under scenarios/tunnel/scripts/
        # checkpoints/bench_ppo_* carry all three), so they were dropped.
        #
        # Checkpoints from before the entropy autotuner was removed also carry
        # `log_ent_coef` and `ent_coef_optimizer`. They are ignored: ent_coef
        # only enters training, and the tuned value never left 0.01 +- 2%.

    def update(self, buffer, minibatch_size, update_epochs):
        # `values` are the pre-update value predictions, on the raw reward
        # scale. These are the estimates that actually produced the
        # advantages, which is what explained_variance is defined against.
        # Scoring the critic *after* its 10 epochs on this same batch
        # measures training-set fit instead, and is optimistically biased --
        # and not comparable with the figure other PPO implementations report.
        states, actions, logprobs, values, returns, advantages = buffer.get()
        batch_size = states.shape[0]

        # Per-batch advantage normalization. The raw std is logged: it is the
        # scale of the signal the normalization is about to hide. The mean
        # comes from `std_mean`, not `.mean()`: the two differ in the last
        # bit on about half of all batches, and `std_mean`'s -- bit-identical
        # to the `var_mean` this used to call -- is the one every reported
        # flat-PPO run was trained with; `.mean()` shifts slalom seeds 2 and
        # 3 to measurably different (if equally good) policies.
        adv_std, adv_mean = torch.std_mean(advantages)
        adv_std_raw = float(adv_std)
        advantages = (advantages - adv_mean) / (adv_std_raw + 1e-8)

        # Value-target normalization: refresh the running statistics on this
        # batch's returns, then regress the critic on standardized targets.
        # `get_value` undoes this so GAE keeps working on the raw scale.
        self.ret_rms.update(returns)
        norm_returns = (returns - self.ret_rms.mean) / self.ret_rms.std

        clipfracs = []

        pg_losses, v_losses, entropy_losses, approx_kls = [], [], [], []
        # Worst single minibatch/epoch of this update, not just the mean --
        # see `ratio_max_dev` in the metrics dict below.
        ratio_max_dev = 0.0

        for _ in range(update_epochs):
            b_inds = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy = self.policy_forward(
                    states[mb_inds], actions[mb_inds]
                )
                # Raw critic output: standardized space, matching norm_returns.
                newvalue = self.critic(states[mb_inds]).squeeze(-1)
                logratio = newlogprob - logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                    ratio_max_dev = max(ratio_max_dev, (ratio - 1.0).abs().max().item())

                mb_advantages = advantages[mb_inds]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss: plain regression on the standardized targets.
                v_loss = 0.5 * ((newvalue - norm_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss. The actor's gradient comes only from the first
                # two terms and the critic's only from the third, so summing
                # them is just a way to take both steps with one backward
                # pass -- there is no trade-off between them to weight.
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss

                self.optimizer.zero_grad()
                loss.backward()
                # Clipped separately so neither head can eat the other's share
                # of a shared gradient-norm budget.
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                approx_kls.append(approx_kl.item())

        # Final metrics, on the raw return scale GAE works in. explained_variance
        # would come out the same in the critic's standardized space (it is
        # invariant to an affine map shared by values and returns); value_bias
        # would not, and reads in reward units here.
        returns_np = returns.detach().cpu().numpy()
        values_np = values.detach().cpu().numpy()
        var_y = np.var(returns_np)
        explained_var = np.nan if var_y == 0 else 1 - np.var(returns_np - values_np) / var_y

        metrics = {
            "loss/policy_loss": float(np.mean(pg_losses)),
            "loss/value_loss": float(np.mean(v_losses)),
            "loss/entropy": float(np.mean(entropy_losses)),
            "loss/approx_kl": float(np.mean(approx_kls)),
            # Worst single minibatch/epoch of this update, not just the mean
            # -- a collapse can be one bad epoch inside an otherwise
            # unremarkable-looking update, which averaging over all of them
            # would hide. Ported from HPPOAgent for diagnostic parity.
            "loss/approx_kl_max": float(np.max(approx_kls)),
            "loss/ratio_max_dev": float(ratio_max_dev),
            "loss/clipfrac": float(np.mean(clipfracs)),
            # Mean offset between predicted and actual return, pre-update. A
            # large value here with a healthy explained_variance means the
            # critic has the shape right but not the scale.
            "loss/value_bias": float((values - returns).mean().item()),
            "loss/value_target_mean": float(self.ret_rms.mean),
            "loss/value_target_std": float(self.ret_rms.std),
            # float() not just for tidiness: np.var over a float32 batch returns
            # np.float32, which is not a Python float and serialises awkwardly.
            "loss/explained_variance": float(explained_var),
            # Pre-normalization advantage std.
            "loss/adv_std_raw": adv_std_raw,
        }
        return metrics
