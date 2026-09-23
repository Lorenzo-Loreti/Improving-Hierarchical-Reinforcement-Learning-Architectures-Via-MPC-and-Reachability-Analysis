import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math

from common import (
    layer_init, normalize_obs, clipped_value_loss, floor_normalize,
    ScaledBeta, RunningMeanStd,
)


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

    def compute_returns_and_advantage(self, next_value, next_done, gamma, gae_lambda):
        """GAE(lambda) over the stored rollout.

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
        next episode.
        """
        lastgaelam = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
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

    def get(self):
        states = self.states.reshape(self.batch_size, self.obs_dim)
        actions = self.actions.reshape(self.batch_size, self.act_dim)
        logprobs = self.logprobs.reshape(self.batch_size)
        returns = self.returns.reshape(self.batch_size)
        advantages = self.advantages.reshape(self.batch_size)
        return states, actions, logprobs, returns, advantages

class PPOAgent:
    # Defaults deliberately match what each scenario's script_ppo.py passes, so an agent
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
    def __init__(self, obs_dim, act_dim, act_limit_low=-1.0, act_limit_high=1.0, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_coef=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                 target_kl=None, obs_low=None, obs_high=None,
                 critic_lr_mult=3.0, device="cpu",
                 autotune_ent_coef=True, target_entropy_frac=0.35, ent_coef_lr=3e-4,
                 ent_coef_min=1e-4, ent_coef_max=1.0,
                 clip_vloss=False, adv_std_floor_frac=0.0, ret_rms_horizon=10):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        # PPO2/CleanRL-style value clipping -- see `clipped_value_loss`.
        # Ported from HPPOAgent (algorithms/hppo/hppo.py) for API/diagnostic
        # parity, not a finding specific to flat PPO -- see that function's
        # docstring. False reproduces the exact previous behaviour.
        self.clip_vloss = clip_vloss
        # Floor on the advantage-normalisation denominator -- see
        # `floor_normalize`. 0.0 reproduces the exact previous behaviour.
        self.adv_std_floor_frac = adv_std_floor_frac
        self.device = device

        self.act_limit_low = torch.tensor(act_limit_low, dtype=torch.float32, device=device)
        self.act_limit_high = torch.tensor(act_limit_high, dtype=torch.float32, device=device)

        # Entropy autotuning (SAC-style dual ascent on a learned log-coefficient):
        # off by default, so an agent built without these kwargs behaves exactly
        # as before -- fixed self.ent_coef, no extra tensors/optimizer/checkpoint
        # keys. See the thesis PPO chapter on why target_entropy is scaled by
        # log(scale) rather than SAC's usual -act_dim: ScaledBeta has bounded
        # support, so its max per-dimension entropy is log(scale) (Beta(1,1) =
        # Uniform), not the ~0 nats a unit-variance unbounded Gaussian gives.
        self.autotune_ent_coef = autotune_ent_coef
        self.ent_coef_min = ent_coef_min
        self.ent_coef_max = ent_coef_max
        if autotune_ent_coef:
            scale = act_limit_high - act_limit_low
            self.target_entropy = act_dim * target_entropy_frac * math.log(scale)
            self.log_ent_coef = torch.tensor(
                math.log(ent_coef), requires_grad=True, device=device
            )
            self.ent_coef_optimizer = optim.Adam([self.log_ent_coef], lr=ent_coef_lr)

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
        # returns here are O(400) (a +-500 terminal reward, barely discounted
        # at gamma=0.999) against a freshly initialized critic that outputs
        # ~0 -- see RunningMeanStd's docstring (algorithms/common.py) for why
        # that gap matters.
        self.ret_rms = RunningMeanStd(horizon=ret_rms_horizon)

        # Multi-update running std of raw (pre-normalization) advantages, used
        # only as a floor under the batch's own std (see `adv_std_floor_frac`
        # and `update`) -- not persisted in save()/load(), since it is a pure
        # training-time stabilizer with no --resume path that would need it.
        # Built unconditionally rather than only when adv_std_floor_frac > 0:
        # it is cheap to update and this avoids a None-vs-object branch in
        # update(). Ported from HPPOAgent for parity -- see clip_vloss above.
        self.adv_rms = RunningMeanStd()

        # Two parameter groups, so the critic can run at a higher learning
        # rate than the actor. The critic is the binding constraint early in
        # training -- advantages mean nothing until explained_variance has
        # climbed -- while the actor is the head an over-large step damages.
        #
        # Callers must anneal *every* group. Writing only `param_groups[0]`,
        # which was correct while there was a single group, now silently
        # leaves the critic un-annealed; each scenario's script_ppo.py captures the
        # base learning rates once and scales each group against its own.
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
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

    def get_action_and_value(self, state, action=None, deterministic=False):
        action, logprob, entropy = self.policy_forward(state, action, deterministic)
        return action, logprob, entropy, self.get_value(state)

    def get_value(self, state):
        """Value on the raw reward scale, for GAE and truncation bootstrapping."""
        return self.critic(state).squeeze(-1) * self.ret_rms.std + self.ret_rms.mean

    def compute_returns_and_advantage(self, buffer, next_value, next_done):
        """Run GAE over `buffer` using this agent's discount parameters.

        The agent is the single source of truth for gamma/gae_lambda: they are
        also what the caller must use for truncation bootstrapping, and having
        two independent copies is how those silently drift apart.
        """
        buffer.compute_returns_and_advantage(
            next_value, next_done, gamma=self.gamma, gae_lambda=self.gae_lambda
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
        if self.autotune_ent_coef:
            checkpoint["log_ent_coef"] = float(self.log_ent_coef.detach())
            checkpoint["ent_coef_optimizer"] = self.ent_coef_optimizer.state_dict()
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        # Guarded: checkpoints written before the critic had its own parameter
        # group hold a single-group optimiser state, which Adam refuses to load
        # into the two-group optimiser built above. The weights are what matter
        # for evaluation and there is no --resume path that would need the
        # moment estimates, so a stale optimiser state is dropped loudly rather
        # than made fatal.
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except (ValueError, KeyError):
            print(f"warning: {path} predates the two-group optimiser; actor and "
                  "critic weights loaded, optimiser state discarded")
        if "ret_rms" in checkpoint:
            self.ret_rms.load_state_dict(checkpoint["ret_rms"])
        # Guarded: checkpoints written before these keys existed still load.
        if checkpoint.get("obs_low") is not None:
            self.obs_low = np.asarray(checkpoint["obs_low"], dtype=np.float32)
        if checkpoint.get("obs_high") is not None:
            self.obs_high = np.asarray(checkpoint["obs_high"], dtype=np.float32)
        if checkpoint.get("log_ent_coef") is not None:
            if self.autotune_ent_coef:
                with torch.no_grad():
                    self.log_ent_coef.copy_(
                        torch.tensor(checkpoint["log_ent_coef"], device=self.device)
                    )
                try:
                    self.ent_coef_optimizer.load_state_dict(checkpoint["ent_coef_optimizer"])
                except (ValueError, KeyError):
                    print(f"warning: {path} has no usable ent_coef optimiser state; "
                          "log_ent_coef loaded, optimiser state discarded")
            else:
                # Checkpoint was trained with autotuning but this agent wasn't
                # constructed with it: fold the tuned value back into a fixed
                # scalar rather than silently ignoring it.
                self.ent_coef = math.exp(checkpoint["log_ent_coef"])
                print(f"warning: {path} was trained with autotune_ent_coef=True; "
                      f"loaded as a fixed ent_coef={self.ent_coef:.6g} instead")

    def update(self, buffer, minibatch_size, update_epochs):
        states, actions, logprobs, returns, advantages = buffer.get()
        batch_size = states.shape[0]

        # Pre-update value predictions, on the raw reward scale. These are the
        # estimates that actually produced the advantages, which is what
        # explained_variance is defined against. Scoring the critic *after* its
        # 10 epochs on this same batch measures training-set fit instead, and
        # is optimistically biased -- and not comparable with the figure other
        # PPO implementations report.
        values = buffer.values.reshape(batch_size)

        # Advantage normalization -- see floor_normalize for what
        # `adv_std_floor_frac` changes and why. `adv_std_raw` (the batch's
        # own, pre-floor std) is logged regardless.
        advantages, adv_std_raw = floor_normalize(advantages, self.adv_rms, self.adv_std_floor_frac)

        # Value-target normalization: refresh the running statistics on this
        # batch's returns, then regress the critic on standardized targets.
        # `get_value` undoes this so GAE keeps working on the raw scale.
        self.ret_rms.update(returns)
        norm_returns = (returns - self.ret_rms.mean) / self.ret_rms.std
        # Pre-update predictions, re-expressed in the same standardized space
        # as `newvalue` below, for `clip_vloss`. Unused when clip_vloss=False.
        old_values_norm = (values - self.ret_rms.mean) / self.ret_rms.std

        clipfracs = []

        pg_losses, v_losses, entropy_losses, approx_kls = [], [], [], []
        # Worst single minibatch/epoch of this update, not just the mean --
        # see `ratio_max_dev` in the metrics dict below.
        ratio_max_dev = 0.0
        epochs_ran = 0

        # Held fixed for every epoch/minibatch below, even when autotuning: PPO
        # reuses this same batch across update_epochs passes, so a coefficient
        # that moved mid-update would mean epoch 1 and epoch 10 are optimizing
        # different surrogate objectives. The learned value only moves once,
        # after the loop, from this update's mean entropy.
        ent_coef_value = (
            float(self.log_ent_coef.detach().exp()) if self.autotune_ent_coef else self.ent_coef
        )

        for epoch in range(update_epochs):
            epochs_ran += 1
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

                # Value loss -- see clipped_value_loss for what `clip_vloss`
                # changes and why.
                v_loss = clipped_value_loss(
                    newvalue, old_values_norm[mb_inds], norm_returns[mb_inds],
                    self.clip_coef, self.clip_vloss
                )

                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - ent_coef_value * entropy_loss + v_loss * self.vf_coef
                
                self.optimizer.zero_grad()
                loss.backward()
                # Clipped separately so neither head can eat the other's share
                # of a shared gradient-norm budget. Both losses are O(1) now
                # that the value target is standardized, but keeping the two
                # budgets independent means vf_coef stays the only knob that
                # trades them off.
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                approx_kls.append(approx_kl.item())

            # Optional trust-region backstop. Off by default (target_kl=None),
            # in which case all `update_epochs` always run and clipping is the
            # only mechanism keeping the update near the sampling policy --
            # so do not describe such runs as KL-constrained.
            if self.target_kl is not None and approx_kls[-1] > self.target_kl:
                break

        # One dual-ascent step per update() call (not per minibatch -- see
        # ent_coef_value above), using this update's mean entropy as the
        # measurement. If that's below target_entropy, this pushes log_ent_coef
        # up, raising ent_coef_value next update; above target, it falls.
        if self.autotune_ent_coef:
            mean_entropy = float(np.mean(entropy_losses))
            ent_coef_loss = self.log_ent_coef * (mean_entropy - self.target_entropy)
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            with torch.no_grad():
                self.log_ent_coef.clamp_(math.log(self.ent_coef_min), math.log(self.ent_coef_max))

        # Calculate some final metrics. Values are compared on the raw return
        # scale, so explained_variance stays comparable across runs with and
        # without value normalization.
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
            # Epochs actually run; below update_epochs only when target_kl fired.
            "loss/update_epochs_ran": float(epochs_ran),
            # Mean offset between predicted and actual return, pre-update. A
            # large value here with a healthy explained_variance means the
            # critic has the shape right but not the scale.
            "loss/value_bias": float((values - returns).mean().item()),
            "loss/value_target_mean": float(self.ret_rms.mean),
            "loss/value_target_std": float(self.ret_rms.std),
            # float() not just for tidiness: np.var over a float32 batch returns
            # np.float32, which is not a Python float and serialises awkwardly.
            "loss/explained_variance": float(explained_var),
            # Pre-normalization advantage std -- see floor_normalize.
            "loss/adv_std_raw": adv_std_raw,
            "loss/batch_size": float(batch_size),
        }
        if self.autotune_ent_coef:
            metrics["loss/ent_coef"] = float(self.log_ent_coef.detach().exp())
        return metrics
