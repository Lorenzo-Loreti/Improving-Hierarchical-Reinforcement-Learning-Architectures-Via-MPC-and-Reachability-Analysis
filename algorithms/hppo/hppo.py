import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from common import (
    layer_init, normalize_obs, normalize_goal, clipped_value_loss,
    ScaledBeta, RunningMeanStd, ManagerActor, ManagerCritic,
)

class WorkerActor(nn.Module):
    def __init__(self, obs_dim, goal_dim, act_dim):
        super().__init__()
        self.act_dim = act_dim
        # Worker observation is concat(obs, goal)
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + goal_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim * 2), std=0.01)
        )

    def forward(self, obs_goal):
        out = self.net(obs_goal)
        alpha = F.softplus(out[..., :self.act_dim]) + 1.0
        beta = F.softplus(out[..., self.act_dim:]) + 1.0
        return alpha, beta

class WorkerCritic(nn.Module):
    def __init__(self, obs_dim, goal_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + goal_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def forward(self, obs_goal):
        return self.net(obs_goal)


class RolloutBuffer:
    """Single-environment rollout store -- the serial reference implementation.

    The training loop does not use it: it collects from a vector env into
    `VecRolloutBuffer` (worker) and `ManagerVecRolloutBuffer` (manager).
    This one is kept because it is the simplest correct statement of the
    GAE both of those must reproduce per environment, and
    tests/test_vec_rollout.py checks them against it column by column.

    Unlike flat PPO's buffer this one is partially filled: the worker writes one
    transition per environment step, the manager one per c-step segment, so a
    manager buffer sized for the worker's step budget ends a rollout roughly
    `manager_freq` times emptier. Everything downstream slices to `self.step`
    rather than assuming a full batch.

    The buffer is agnostic about how observations are encoded -- it stores what
    the caller hands it, already in the networks' input space.
    """

    def __init__(self, num_steps, obs_dim, act_dim, device):
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        self.states = torch.zeros((num_steps, obs_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((num_steps, act_dim), dtype=torch.float32).to(device)
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

    def compute_returns_and_advantage(self, next_value, gamma, gae_lambda):
        """GAE(lambda) over the stored rollout. `next_value` is the value of
        the state that follows the last stored transition.

        `gamma` and `gae_lambda` are required rather than defaulted: they used
        to default to 0.99/0.95 while the manager was in fact discounted at
        gamma**manager_freq and the worker at the env-level gamma, so a caller
        who omitted them got silently different discounting from the trained
        configuration -- and the two heads need *different* values here, which
        makes a single shared default actively misleading. Prefer
        `HPPOAgent.compute_manager_returns_and_advantage` /
        `compute_worker_returns_and_advantage`, which source both from the agent.

        Note the `dones` convention: `dones[t]` flags "the transition at index
        t ended an episode", so the bootstrap mask is `1 - dones[t]`. CleanRL
        stores "observation t begins a new episode" instead and masks with
        `dones[t+1]`. Both are correct; they are the same quantity indexed
        differently. It is also why there is no `next_done` argument: CleanRL
        needs one for the state after the rollout, whereas here the last
        transition's own `dones[-1]` already says whether `next_value` may be
        bootstrapped from. (All three buffers used to take a `next_done`
        anyway; the training loop filled it with exactly `dones[-1]` --
        except, for the manager, after a collision-forced re-plan, which is
        where that redundancy turned into a double-counted bootstrap. See
        the manager_act_now comment in the training loop.)
        """
        lastgaelam = 0
        for t in reversed(range(self.step)):
            nextvalues = next_value if t == self.step - 1 else self.values[t + 1]
            nextnonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            self.advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        self.returns = self.advantages + self.values

    def reset(self):
        self.step = 0

    def get_values(self):
        """The stored value predictions, in the same order as `get()`.

        Exists so `_update_head` can read pre-update values without knowing
        which buffer layout it was handed: the single-env buffer stores a flat
        prefix, the vectorized ones a (T, N) grid or a ragged per-env one.
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


class VecRolloutBuffer:
    """Dense (num_steps, num_envs) rollout store for the *worker*.

    The worker acts once per environment step in every environment, so its
    rollout is a full rectangular grid -- exactly flat PPO's layout, and
    nothing like the manager's (see ManagerVecRolloutBuffer). Ported from
    flat PPO's RolloutBuffer (algorithms/ppo/ppo.py) so the two share a GAE
    convention.

    As with the single-env buffer, the caller stores already-encoded inputs:
    the training loop writes the concatenated (normalized obs, normalized
    goal) the worker networks actually take.
    """

    def __init__(self, num_steps, num_envs, obs_dim, act_dim, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.batch_size = num_steps * num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        self.states = torch.zeros((num_steps, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((num_steps, num_envs, act_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

        self.returns = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

        self.step = 0

    def add(self, state, action, logprob, reward, value, done):
        self.states[self.step] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.step] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.logprobs[self.step] = torch.as_tensor(logprob, dtype=torch.float32, device=self.device)
        self.rewards[self.step] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.values[self.step] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.dones[self.step] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.step += 1

    def compute_returns_and_advantage(self, next_value, gamma, gae_lambda):
        """GAE(lambda) run independently down each of the `num_envs` columns.

        `next_value` is a per-environment vector of shape (num_envs,). Same
        `dones[t]` convention as the single-env buffer: "the transition at
        index t ended an episode", masked with `1 - dones[t]` -- which is
        also the convention an auto-resetting vector env produces naturally,
        since the observation after a done already belongs to the next
        episode. So the last row's own dones mask the bootstrap, and there is
        no `next_done` argument (see RolloutBuffer's).
        """
        lastgaelam = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for t in reversed(range(self.step)):
            nextvalues = next_value if t == self.step - 1 else self.values[t + 1]
            nextnonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            self.advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        self.returns = self.advantages + self.values

    def reset(self):
        self.step = 0

    def get_values(self):
        return self.values[:self.step].reshape(-1)

    def get(self):
        n = self.step * self.num_envs
        return (
            self.states[:self.step].reshape(n, self.obs_dim),
            self.actions[:self.step].reshape(n, self.act_dim),
            self.logprobs[:self.step].reshape(n),
            self.returns[:self.step].reshape(n),
            self.advantages[:self.step].reshape(n),
        )


class ManagerVecRolloutBuffer:
    """Ragged per-environment rollout store for the *manager*.

    The manager is why vectorizing this hierarchy is a redesign rather than a
    substitution. It writes one transition per c-step segment, and a segment
    also ends early whenever an episode does -- so after a rollout of T steps
    across N environments, environment i holds some S_i transitions and no two
    environments need agree. The store is therefore a (T, N) grid with a
    *separate write pointer per column*, and everything downstream slices each
    column to its own S_i.

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
        (num_envs, ...) batch; only the masked rows are read, so the caller
        can pass the whole per-environment state without pre-filtering it.
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

        def _t(x, dtype=torch.float32):
            return torch.as_tensor(x, dtype=dtype, device=self.device)

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
        prefix. `lastgaelam` is held at zero until then, so when a column's
        last transition is finally reached it starts the recursion from zero
        and takes the bootstrap branch -- identical to running the single-env
        loop on that column alone.

        `next_value` is a (num_envs,) vector: the value of each environment's
        in-flight segment, whose start is the state its last stored
        transition led to. Whether to bootstrap from it is that last
        transition's own done flag (see RolloutBuffer's docstring), so there
        is no `next_done` argument.
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
            # is 1 - dones[t] -- at t, not t + 1, and for a column's last
            # transition as for any other. Only `values` is read one step
            # ahead.
            nextnonterminal = 1.0 - self.dones[t]
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


class HPPOAgent:
    # Defaults deliberately match what hppo_train.py's flags default to, so an
    # agent constructed directly -- an evaluation notebook, say -- behaves like
    # the trained configuration instead of silently differing from it. Change
    # the two together.
    #
    # gamma is 0.99, not the 0.999 this script used to train at. Measured over
    # 72 seeds on flat PPO, 0.999 makes the critic's regression target
    # degenerate as the policy converges: every state in a successful episode
    # earns almost the same discounted return, the spread of returns inside a
    # batch collapses, and explained_variance collapses with it. The advantages
    # that survive are then mostly critic error, which the batch-level
    # normalisation in the update rescales straight back to unit variance. See
    # the thesis PPO chapter, section 11.1 (kept outside this repo). That
    # measurement is flat PPO's, not this
    # hierarchy's; it is the reason for the change, not evidence about hPPO.
    #
    # `gamma` is the *environment-step* discount. The manager's own discount is
    # derived from it as gamma**manager_freq rather than passed separately: the
    # manager decides once per c-step segment, so one manager step is c
    # environment steps (the standard SMDP/options treatment). Deriving it is
    # not cosmetic -- the previous version stored gamma_manager=gamma on the
    # agent while the training script ran manager GAE at gamma**c, so the
    # agent's own attribute disagreed with the discounting actually used.
    #
    # Deliberately absent: the other candidate mitigations of the manager-
    # collapse investigation, which ManagerActor's docstring (algorithms/
    # common.py) records in full. A per-head early-stopping `target_kl` /
    # `target_kl_manager` showed no benefit (a per-update KL cap does not
    # catch a drift spread over many individually unremarkable updates); the
    # advantage-std floor (`adv_std_floor_frac`) and a longer manager
    # ret_rms memory (`ret_rms_horizon_manager`) gave seed-inconsistent
    # results. None was ever a default, and the collapse itself turned out
    # to be the worker's reward (see that docstring's addendum), so they were
    # removed rather than kept as dead options. Of that investigation only
    # `clip_vloss` survives, on by default. The removed code is in git
    # history, before the commit that dropped them.
    #
    # Also absent, as in flat PPO (see PPOAgent in ../ppo/ppo.py), and
    # replaced by something simpler that does the same:
    #
    # - Entropy autotuning (SAC-style dual ascent on each head's
    #   log(ent_coef)). Adam moves log(ent_coef) by about its learning rate
    #   per update, so at the 3e-4 it always ran with the coefficients
    #   could not leave the neighbourhood of their 0.01 start: the 16 hPPO
    #   runs on disk (tunnel) all ended with both heads at 0.0098-0.0102,
    #   13 slalom seeds under the benchmark protocol at 0.00985-0.00992,
    #   and docs/worker-termination-avoidance.md (section 3) measured
    #   0.0100 -> 0.0108 over a whole 244-update slalom run while the
    #   manager's entropy fell from +1.2 to -2.2 nats. Every hPPO result
    #   was in effect trained with the fixed
    #   ent_coef_manager = ent_coef_worker = 0.01 used here. Making it bite
    #   (--ent-coef-lr 0.05) moved the old collapse, it did not prevent it.
    #
    # - vf_coef. It trades a head's value loss off against its policy loss
    #   only when the two share parameters. Here each head's actor and
    #   critic are separate networks with separately clipped gradients, so a
    #   critic's gradient comes from its value loss alone and a constant
    #   factor on it is undone by Adam's normalisation -- all it could still
    #   change is the gradient-clip threshold and the weight of Adam's eps.
    #   The knob that actually sets a critic's step size is critic_lr_mult.
    def __init__(self, obs_dim, goal_dim, act_dim,
                 worker_act_limit_low=-1.0, worker_act_limit_high=1.0,
                 lr_manager=3e-4, lr_worker=3e-4,
                 gamma=0.99, manager_freq=10,
                 gae_lambda=0.95, clip_coef=0.2,
                 ent_coef_manager=0.01, ent_coef_worker=0.01,
                 max_grad_norm=0.5,
                 obs_low=None, obs_high=None, max_goal_bound=10.0,
                 critic_lr_mult=3.0, device="cpu",
                 clip_vloss=True):
        self.gamma = gamma
        self.manager_freq = manager_freq
        self.gamma_worker = gamma
        self.gamma_manager = gamma ** manager_freq
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef_manager = ent_coef_manager
        self.ent_coef_worker = ent_coef_worker
        self.max_grad_norm = max_grad_norm
        # PPO2/CleanRL-style value clipping: clip the critic's new prediction
        # to within `clip_coef` of its pre-update one before scoring the loss,
        # so one update cannot move the critic arbitrarily far on a noisy
        # batch. Shared across heads rather than split -- see _update_head.
        # On by default, as in both scripts: the only mitigation the 6-seed
        # manager-collapse ablation supported (see ManagerActor's docstring).
        # The default used to be False here while the scripts passed True,
        # so an agent built directly silently differed from the trained one.
        self.clip_vloss = clip_vloss
        self.device = device

        self.manager_limit_low = torch.tensor(-1.0, dtype=torch.float32, device=device)
        self.manager_limit_high = torch.tensor(1.0, dtype=torch.float32, device=device)

        self.worker_limit_low = torch.tensor(worker_act_limit_low, dtype=torch.float32, device=device)
        self.worker_limit_high = torch.tensor(worker_act_limit_high, dtype=torch.float32, device=device)

        # Observation and goal bounds travel with the agent so that `save()`
        # produces a self-describing checkpoint. Without them a reloaded
        # hierarchy silently receives un-normalized inputs and behaves nothing
        # like the trained one. Held as arrays because normalize_obs runs every
        # rollout step; serialised as lists, because torch.load defaults to
        # weights_only=True, which rejects pickled numpy arrays.
        self.obs_low = None if obs_low is None else np.asarray(obs_low, dtype=np.float32).ravel()
        self.obs_high = None if obs_high is None else np.asarray(obs_high, dtype=np.float32).ravel()
        self.max_goal_bound = max_goal_bound

        # Manager. ManagerActor's docstring (algorithms/common.py) is this
        # codebase's canonical writeup of the manager-collapse investigation
        # (MAX_CONCENTRATION, clip_vloss, adv_std_floor_frac and why each
        # was or wasn't adopted) -- referenced by name rather than repeated
        # in every algorithm that shares this network.
        self.manager_actor = ManagerActor(obs_dim, goal_dim).to(device)
        self.manager_critic = ManagerCritic(obs_dim).to(device)

        # Worker
        self.worker_actor = WorkerActor(obs_dim, goal_dim, act_dim).to(device)
        self.worker_critic = WorkerCritic(obs_dim, goal_dim).to(device)

        # Each critic learns in standardized-return space; these statistics map
        # its output back to the raw reward scale that GAE works in. Both use
        # RunningMeanStd's default 10-batch memory. (A longer one for the
        # manager, whose batch is ~manager_freq times smaller, was tried as a
        # collapse mitigation and dropped -- see the class comment.)
        #
        # The manager is the head that most needs the normalization: its
        # reward is the discounted sum of the environment's own reward over a
        # c-step segment, so its returns inherit the environment's raw scale
        # -- by the time the benchmark seeds stop, manager ret_rms sits at mean
        # ~530-630 / std ~250-310 on the slalom and mean ~75-85 / std ~85-100
        # on the tunnel -- against a freshly initialized critic that outputs
        # ~0; see RunningMeanStd's docstring (algorithms/common.py) for why
        # that gap matters. The worker's returns are smaller but not O(1)
        # either since its reward mixes in 0.02 x the environment's (see
        # --worker-extrinsic-coef): mean ~10-14 / std ~5-7 on the slalom,
        # ~1.5-2.5 / ~2 on the tunnel. Applied to both heads regardless, so
        # they never differ in the space their critic regresses in, which
        # would make their metrics incomparable. (This comment used to call
        # the worker's normalization "close to a no-op" and quote O(400)
        # manager targets: true of a purely intrinsic worker reward and of
        # gamma=0.999, neither of which is the configuration any more.)
        self.manager_ret_rms = RunningMeanStd()
        self.worker_ret_rms = RunningMeanStd()

        # Two parameter groups per head, so each critic can run at a higher
        # learning rate than its actor. The critic is the binding constraint
        # early in training -- advantages mean nothing until explained_variance
        # has climbed -- while the actor is the head an over-large step damages.
        #
        # Callers must anneal *every* group. Writing only `param_groups[0]`,
        # which was correct while there was a single group, now silently leaves
        # the critic un-annealed; hppo_train.py captures the base learning
        # rates once and scales each group against its own.
        self.manager_optimizer = optim.Adam([
            {"params": list(self.manager_actor.parameters()), "lr": lr_manager},
            {"params": list(self.manager_critic.parameters()), "lr": lr_manager * critic_lr_mult},
        ], eps=1e-5)

        self.worker_optimizer = optim.Adam([
            {"params": list(self.worker_actor.parameters()), "lr": lr_worker},
            {"params": list(self.worker_critic.parameters()), "lr": lr_worker * critic_lr_mult},
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

    def normalize_goal(self, goal):
        """Apply the goal map this agent was constructed with."""
        return normalize_goal(goal, self.max_goal_bound)

    def scale_goal(self, normalized_goal):
        """Inverse of `normalize_goal`: manager action -> physical displacement."""
        return normalized_goal * self.max_goal_bound

    # --- manager ----------------------------------------------------------

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
        # Log probability of a continuous goal is the sum over its dimensions
        return action, probs.log_prob(action).sum(dim=-1), probs.entropy().sum(dim=-1)

    def get_manager_action(self, state, deterministic=False):
        with torch.no_grad():
            action, _, _ = self.manager_policy_forward(state, deterministic=deterministic)
        return action

    def get_manager_action_and_value(self, state):
        """Sampled goal, its log-probability and the state's value: what a
        manager transition stores."""
        action, logprob, _ = self.manager_policy_forward(state)
        return action, logprob, self.get_manager_value(state)

    def get_manager_value(self, state):
        """Value on the raw reward scale, for GAE and truncation bootstrapping."""
        return self.manager_critic(state).squeeze(-1) * self.manager_ret_rms.std + self.manager_ret_rms.mean

    # --- worker -----------------------------------------------------------

    def worker_policy_forward(self, obs_goal, action=None, deterministic=False):
        """Actor-only forward over the concatenated (obs, goal) input."""
        alpha, beta = self.worker_actor(obs_goal)
        probs = ScaledBeta(alpha, beta, low=self.worker_limit_low, high=self.worker_limit_high)
        if action is None:
            if deterministic:
                action = probs.deterministic_sample()
            else:
                action = probs.sample()
        return action, probs.log_prob(action).sum(dim=-1), probs.entropy().sum(dim=-1)

    def get_worker_action(self, obs_goal, deterministic=False):
        with torch.no_grad():
            action, _, _ = self.worker_policy_forward(obs_goal, deterministic=deterministic)
        return action

    def get_worker_action_and_value(self, obs_goal):
        """Sampled action, its log-probability and the input's value: what a
        worker transition stores."""
        action, logprob, _ = self.worker_policy_forward(obs_goal)
        return action, logprob, self.get_worker_value(obs_goal)

    def get_worker_value(self, obs_goal):
        """Value on the raw reward scale, for GAE and truncation bootstrapping."""
        return self.worker_critic(obs_goal).squeeze(-1) * self.worker_ret_rms.std + self.worker_ret_rms.mean

    # --- advantage estimation ---------------------------------------------

    def compute_manager_returns_and_advantage(self, buffer, next_value):
        """Run GAE over the manager's `buffer` at the segment-level discount.

        The agent is the single source of truth for the discounts: they are
        also what the caller must use for truncation bootstrapping, and having
        two independent copies is how those silently drift apart.
        """
        buffer.compute_returns_and_advantage(
            next_value, gamma=self.gamma_manager, gae_lambda=self.gae_lambda
        )

    def compute_worker_returns_and_advantage(self, buffer, next_value):
        """Run GAE over the worker's `buffer` at the environment-step discount."""
        buffer.compute_returns_and_advantage(
            next_value, gamma=self.gamma_worker, gae_lambda=self.gae_lambda
        )

    # --- persistence ------------------------------------------------------

    def save(self, path):
        checkpoint = {
            "manager_actor": self.manager_actor.state_dict(),
            "manager_critic": self.manager_critic.state_dict(),
            "worker_actor": self.worker_actor.state_dict(),
            "worker_critic": self.worker_critic.state_dict(),
            "manager_optimizer": self.manager_optimizer.state_dict(),
            "worker_optimizer": self.worker_optimizer.state_dict(),
            # Without these each critic's output is meaningless on reload.
            "manager_ret_rms": self.manager_ret_rms.state_dict(),
            "worker_ret_rms": self.worker_ret_rms.state_dict(),
            # Without these the *actors'* inputs are wrong on reload.
            "obs_low": None if self.obs_low is None else self.obs_low.tolist(),
            "obs_high": None if self.obs_high is None else self.obs_high.tolist(),
            "max_goal_bound": self.max_goal_bound,
            # The cadence is part of the trained controller, not a run detail:
            # a reloaded hierarchy that re-plans at a different c is a
            # different controller, and it also fixes the manager's discount.
            # Saved so an evaluation script cannot get it wrong.
            "manager_freq": self.manager_freq,
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.manager_actor.load_state_dict(checkpoint["manager_actor"])
        self.manager_critic.load_state_dict(checkpoint["manager_critic"])
        self.worker_actor.load_state_dict(checkpoint["worker_actor"])
        self.worker_critic.load_state_dict(checkpoint["worker_critic"])
        self.manager_optimizer.load_state_dict(checkpoint["manager_optimizer"])
        self.worker_optimizer.load_state_dict(checkpoint["worker_optimizer"])
        self.manager_ret_rms.load_state_dict(checkpoint["manager_ret_rms"])
        self.worker_ret_rms.load_state_dict(checkpoint["worker_ret_rms"])
        # None only when the saving agent had no bounds either; this agent
        # then keeps whatever it was constructed with.
        if checkpoint["obs_low"] is not None:
            self.obs_low = np.asarray(checkpoint["obs_low"], dtype=np.float32)
            self.obs_high = np.asarray(checkpoint["obs_high"], dtype=np.float32)
        self.max_goal_bound = checkpoint["max_goal_bound"]
        self.manager_freq = checkpoint["manager_freq"]
        self.gamma_manager = self.gamma ** self.manager_freq
        # Guards for checkpoints older than the two-group optimisers, ret_rms,
        # the observation/goal bounds or the saved cadence used to live here;
        # no such hPPO checkpoint is left (all 45 under scenarios/tunnel/
        # scripts/checkpoints carry every key), so they were dropped.
        # Checkpoints from before the entropy autotuner was removed also carry
        # `{manager,worker}_log_ent_coef` and their optimiser states. They are
        # ignored: ent_coef only enters training, and the tuned values never
        # left 0.01 +- 2%.

    # --- update -----------------------------------------------------------

    def _update_head(self, buffer, minibatch_size, update_epochs, policy_forward,
                     actor, critic, optimizer, ent_coef, ret_rms, prefix):
        """One PPO update for a single head.

        The manager and the worker run the identical algorithm over their own
        buffer, networks and entropy coefficient; only those differ. They used
        to be two near-identical 60-line bodies, which is how the
        value-normalisation and pre-update explained-variance fixes below would
        have landed in one and not the other.

        Value-loss clipping follows the agent's `clip_vloss`, for both heads.
        """
        states, actions, logprobs, returns, advantages = buffer.get()
        batch_size = states.shape[0]

        if batch_size == 0:
            # This head picked up zero transitions this rollout -- possible
            # for the manager on a short rollout relative to manager_freq, or
            # any run of environments that neither hit a c-step boundary nor
            # terminated. There is no gradient to take and no entropy/KL
            # statistics to compute, so skip the update entirely: falling
            # through would mean() over empty tensors into NaN and np.max()
            # over an empty list into a ValueError below. (It also used to
            # feed that NaN into the entropy autotuner's dual-ascent step,
            # permanently corrupting log_ent_coef, before the autotuner was
            # removed.)
            metrics = {
                f"{prefix}/loss_policy": float("nan"),
                f"{prefix}/loss_value": float("nan"),
                f"{prefix}/entropy": float("nan"),
                f"{prefix}/approx_kl": float("nan"),
                f"{prefix}/approx_kl_max": float("nan"),
                f"{prefix}/ratio_max_dev": float("nan"),
                f"{prefix}/clipfrac": float("nan"),
                f"{prefix}/value_bias": float("nan"),
                f"{prefix}/value_target_mean": float(ret_rms.mean),
                f"{prefix}/value_target_std": float(ret_rms.std),
                f"{prefix}/explained_variance": float("nan"),
                f"{prefix}/adv_std_raw": float("nan"),
                f"{prefix}/batch_size": 0.0,
            }
            return metrics

        # Pre-update value predictions, on the raw reward scale. These are the
        # estimates that actually produced the advantages, which is what
        # explained_variance is defined against. Scoring the critic *after* its
        # update epochs on this same batch measures training-set fit instead,
        # and is optimistically biased -- and not comparable with the figure
        # other PPO implementations report. That was the previous behaviour
        # here, so explained_variance is not comparable across this change.
        #
        # Read through `get_values()` rather than by slicing `.values`
        # directly: the three buffer layouts (single-env prefix, dense (T, N)
        # grid, ragged per-environment) order their storage differently, and
        # only the buffer knows how to line it up with what `get()` returned.
        values = buffer.get_values()

        # Per-batch advantage normalization. `adv_std_raw`, the batch's own
        # std, is logged: the first manager-collapse hypothesis (see
        # ManagerActor's docstring) said it should crater right around a bad
        # update -- it climbed instead, which is part of how that hypothesis
        # was refuted. The mean comes from `var_mean`, not `.mean()`: the two
        # differ in the last bit on about half of all batches, and `var_mean`'s
        # is the one every reported run was trained with (it is what the
        # since-removed advantage-std floor computed).
        adv_std_raw = float(advantages.std())
        _, adv_mean = torch.var_mean(advantages)
        advantages = (advantages - adv_mean) / (adv_std_raw + 1e-8)

        # Value-target normalization: refresh the running statistics on this
        # batch's returns, then regress the critic on standardized targets.
        # `get_*_value` undoes this so GAE keeps working on the raw scale.
        ret_rms.update(returns)
        norm_returns = (returns - ret_rms.mean) / ret_rms.std
        # Pre-update predictions, re-expressed in the same standardized space
        # as `newvalue` below, for `clip_vloss`. Approximate: it re-derives
        # them from the raw-scale `values` using this update's just-refreshed
        # ret_rms rather than whatever stats were live when they were
        # collected, in exchange for not carrying a second, already-
        # standardized copy through the buffer. Unused when clip_vloss=False.
        old_values_norm = (values - ret_rms.mean) / ret_rms.std

        clipfracs = []
        pg_losses, v_losses, entropy_losses, approx_kls = [], [], [], []
        ratio_max_dev = 0.0

        for _ in range(update_epochs):
            b_inds = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy = policy_forward(states[mb_inds], actions[mb_inds])
                # Raw critic output: standardized space, matching norm_returns.
                newvalue = critic(states[mb_inds]).squeeze(-1)
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

                # Total loss. The actor's gradient comes only from the first
                # two terms and the critic's only from the third, so summing
                # them is just a way to take both steps with one backward
                # pass -- there is no trade-off between them to weight.
                loss = pg_loss - ent_coef * entropy_loss + v_loss

                optimizer.zero_grad()
                loss.backward()
                # Clipped separately so neither network can eat the other's
                # share of a shared gradient-norm budget.
                nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
                optimizer.step()

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
            f"{prefix}/loss_policy": float(np.mean(pg_losses)),
            f"{prefix}/loss_value": float(np.mean(v_losses)),
            f"{prefix}/entropy": float(np.mean(entropy_losses)),
            f"{prefix}/approx_kl": float(np.mean(approx_kls)),
            # Worst single minibatch/epoch of this update, not just the mean
            # -- a collapse can be one bad epoch inside an otherwise unremarkable
            # update, which averaging over all of them would hide.
            f"{prefix}/approx_kl_max": float(np.max(approx_kls)),
            f"{prefix}/ratio_max_dev": float(ratio_max_dev),
            f"{prefix}/clipfrac": float(np.mean(clipfracs)),
            # Mean offset between predicted and actual return, pre-update. A
            # large value here with a healthy explained_variance means the
            # critic has the shape right but not the scale.
            f"{prefix}/value_bias": float((values - returns).mean().item()),
            f"{prefix}/value_target_mean": float(ret_rms.mean),
            f"{prefix}/value_target_std": float(ret_rms.std),
            # float() not just for tidiness: np.var over a float32 batch returns
            # np.float32, which is not a Python float and serialises awkwardly.
            f"{prefix}/explained_variance": float(explained_var),
            # Pre-normalization advantage std -- see the normalization above.
            f"{prefix}/adv_std_raw": adv_std_raw,
            # The two heads see different batch sizes off the same rollout
            # (the manager's is ~c times smaller), and the manager's varies
            # with how many episodes ended early. Logged rather than inferred.
            f"{prefix}/batch_size": float(batch_size),
        }
        return metrics

    def update_manager(self, buffer, minibatch_size, update_epochs):
        return self._update_head(
            buffer, minibatch_size, update_epochs,
            policy_forward=self.manager_policy_forward,
            actor=self.manager_actor,
            critic=self.manager_critic,
            optimizer=self.manager_optimizer,
            ent_coef=self.ent_coef_manager,
            ret_rms=self.manager_ret_rms,
            prefix="manager",
        )

    def update_worker(self, buffer, minibatch_size, update_epochs):
        # buffer.states holds the CONCATENATED (normalized obs, normalized
        # goal) that the worker networks take as input; see worker_input in
        # hppo_train.py.
        return self._update_head(
            buffer, minibatch_size, update_epochs,
            policy_forward=self.worker_policy_forward,
            actor=self.worker_actor,
            critic=self.worker_critic,
            optimizer=self.worker_optimizer,
            ent_coef=self.ent_coef_worker,
            ret_rms=self.worker_ret_rms,
            prefix="worker",
        )
