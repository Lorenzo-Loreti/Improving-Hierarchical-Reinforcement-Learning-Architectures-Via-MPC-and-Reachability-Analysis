"""In-house PPO, the one implementation shared by every learner (decision log D5, D15, D19).

Flat PPO and both levels of hPPO use it (steps 7-8); the Managers of the MPC
architectures will (steps 9-10). It is a faithful port of the pre-alignment
flat PPO (``algorithms/ppo/ppo.py`` at tag ``pre-alignment``): the same
networks, initialization, losses and update schedule, so that given the same
weights, batch and seed the update ends on bit-identical weights
(``tests/test_ppo.py``, against the frozen copy in ``tests/reference/``).

Policy (D15). One Beta distribution per action axis on ``y`` in ``[0, 1]``,
with concentrations ``softplus(.) + 1 >= 1``, mapped to the action
``x = 2 y - 1`` in the square; the environment maps ``x`` onto ``U``. The
log-probability is the density of ``x``. Samples are kept in
``[LOG_PROB_EPS, 1 - LOG_PROB_EPS]``, so every log-probability is finite and
every action is admissible; the deterministic action is the mean.

Critic. A separate network regresses standardized returns. Running statistics
of the returns, averaged over ``value_norm_horizon`` updates, map its output
back to the reward scale on which GAE works.

Update. Advantages are normalized per batch. Clipped surrogate objective,
squared error of the critic and entropy bonus; Adam with one parameter group
per network, the critic's learning rate a multiple of the actor's; gradient
norms clipped per network. Actor and critic share no parameters, so summing
their losses only lets one backward pass serve both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike, NDArray
from torch import nn
from torch.distributions import Beta

from hrlmpc.ppo_config import NetworkConfig, UpdateConfig
from hrlmpc.rollout import Rollout

FloatArray = NDArray[np.float64]

LOG_PROB_EPS = 1e-5
"""Samples of the Beta are kept in ``[LOG_PROB_EPS, 1 - LOG_PROB_EPS]``, and log-probabilities are evaluated there."""


def layer_init(layer: nn.Linear, std: float = math.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal weights with gain ``std`` and a constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _mlp(input_size: int, hidden_sizes: tuple[int, ...], output_size: int, output_std: float) -> nn.Sequential:
    """Tanh MLP with orthogonal initialization; the output layer has gain ``output_std``."""
    layers: list[nn.Module] = []
    size = input_size
    for hidden in hidden_sizes:
        layers += [layer_init(nn.Linear(size, hidden)), nn.Tanh()]
        size = hidden
    layers.append(layer_init(nn.Linear(size, output_size), std=output_std))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Tanh MLP from observations to the Beta concentrations ``(alpha, beta) = softplus(.) + 1``.

    Both are at least 1 (exactly 1 in float32 for logits below about -17), so
    every density is bounded.
    """

    def __init__(self, obs_size: int, action_size: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.action_size = action_size
        self.net = _mlp(obs_size, hidden_sizes, 2 * action_size, output_std=0.01)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(obs)
        alpha = F.softplus(out[..., : self.action_size]) + 1.0
        beta = F.softplus(out[..., self.action_size :]) + 1.0
        return alpha, beta


class Critic(nn.Module):
    """Tanh MLP from observations to a standardized value."""

    def __init__(self, obs_size: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.net = _mlp(obs_size, hidden_sizes, 1, output_std=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(obs).squeeze(-1)
        return result


class RunningMeanStd:
    """Running mean and variance of the critic's regression targets (Chan et al.'s parallel update).

    The effective count is capped at ``horizon`` batches, so the statistics
    follow the current distribution of the returns instead of keeping the
    early transient forever.
    """

    def __init__(self, epsilon: float = 1e-4, horizon: int = 10) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
        self.horizon = horizon

    def update(self, x: torch.Tensor) -> None:
        """Merge the mean and the (biased) variance of the batch ``x``."""
        batch_mean = float(x.mean())
        batch_var = float(x.var(correction=0))
        batch_count = x.numel()
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / tot_count) / tot_count
        self.count = min(tot_count, self.horizon * batch_count)

    @property
    def std(self) -> float:
        """Standard deviation plus ``1e-8``, so that it is never zero."""
        return math.sqrt(self.var) + 1e-8

    def state_dict(self) -> dict[str, float]:
        """Mean, variance and effective count, as plain numbers."""
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, float]) -> None:
        """Restore the statistics written by :meth:`state_dict`."""
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


@dataclass(frozen=True, eq=False)
class Batch:
    """Flattened training data of one update, as float32 tensors.

    Attributes:
        obs: ``(N, n)`` observations.
        actions: ``(N, m)`` actions in ``[-1, 1]^m``.
        log_probs: ``(N,)`` log-probabilities under the collecting policy.
        values: ``(N,)`` value estimates at collection time, on the reward scale.
        returns: ``(N,)`` regression targets of the critic.
        advantages: ``(N,)`` advantage estimates.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

    def __len__(self) -> int:
        return int(self.obs.shape[0])


def make_batch(
    rollout: Rollout, advantages: ArrayLike, returns: ArrayLike, device: str | torch.device = "cpu"
) -> Batch:
    """Flatten a ``(T, B)`` rollout and its GAE output, time-major, into a :class:`Batch`."""
    n = rollout.num_samples

    def flat(values: ArrayLike, *shape: int) -> torch.Tensor:
        array = np.asarray(values, dtype=np.float64).reshape(n, *shape)
        return torch.tensor(array, dtype=torch.float32, device=device)

    return Batch(
        obs=flat(rollout.obs, rollout.obs.shape[-1]),
        actions=flat(rollout.actions, rollout.actions.shape[-1]),
        log_probs=flat(rollout.log_probs),
        values=flat(rollout.values),
        returns=flat(returns),
        advantages=flat(advantages),
    )


def _numpy(tensor: torch.Tensor) -> FloatArray:
    result: FloatArray = tensor.detach().cpu().numpy().astype(np.float64)
    return result


class PPOAgent:
    """Beta policy, critic and PPO update (see the module docstring).

    Args:
        obs_size: Size of an observation.
        action_size: Size of an action.
        network: Hidden sizes of the actor and of the critic.
        update: Hyperparameters of the update.
        device: PyTorch device.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        network: NetworkConfig,
        update: UpdateConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.obs_size = obs_size
        self.action_size = action_size
        self.network_config = network
        self.config = update
        self.device = torch.device(device)
        self.actor = Actor(obs_size, action_size, network.hidden_sizes).to(self.device)
        self.critic = Critic(obs_size, network.hidden_sizes).to(self.device)
        self.ret_rms = RunningMeanStd(horizon=update.value_norm_horizon)
        self.optimizer = torch.optim.Adam(
            [
                {"params": list(self.actor.parameters()), "lr": update.learning_rate},
                {"params": list(self.critic.parameters()), "lr": update.learning_rate * update.critic_lr_mult},
            ],
            eps=update.adam_eps,
        )
        self._base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        # The affine map between y in [0, 1] and x in [-1, 1], as tensors, with the
        # same arithmetic as the pre-alignment implementation.
        low = torch.tensor(-1.0, device=self.device)
        high = torch.tensor(1.0, device=self.device)
        self._low = low
        self._scale = high - low
        self._log_scale = torch.log(self._scale)

    # ------------------------------------------------------------------
    # Policy and value
    # ------------------------------------------------------------------

    def _tensor(self, values: ArrayLike) -> torch.Tensor:
        # A copy in memory allocated by PyTorch, as in the legacy code: the BLAS kernels
        # may round differently on inputs aligned differently.
        return torch.tensor(np.asarray(values, dtype=np.float32), device=self.device)

    def _distribution(self, obs: torch.Tensor) -> Beta:
        alpha, beta = self.actor(obs)
        return Beta(alpha, beta)

    def _log_prob_and_entropy(self, dist: Beta, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-density of the actions on the square and entropy, summed over the axes."""
        unit = torch.clamp((actions - self._low) / self._scale, LOG_PROB_EPS, 1.0 - LOG_PROB_EPS)
        log_prob = (dist.log_prob(unit) - self._log_scale).sum(dim=-1)
        entropy = (dist.entropy() + self._log_scale).sum(dim=-1)
        return log_prob, entropy

    @torch.no_grad()
    def act(self, obs: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Sample actions for ``(B, n)`` observations; return them and their log-probabilities."""
        dist = self._distribution(self._tensor(obs))
        unit = torch.clamp(dist.sample(), LOG_PROB_EPS, 1.0 - LOG_PROB_EPS)
        actions = unit * self._scale + self._low
        log_prob, _ = self._log_prob_and_entropy(dist, actions)
        return _numpy(actions), _numpy(log_prob)

    @torch.no_grad()
    def deterministic_action(self, obs: ArrayLike) -> FloatArray:
        """The mean of the policy, for evaluation (D20)."""
        alpha, beta = self.actor(self._tensor(obs))
        mean = alpha / (alpha + beta)
        return _numpy(mean * self._scale + self._low)

    @torch.no_grad()
    def log_prob(self, obs: ArrayLike, actions: ArrayLike) -> FloatArray:
        """Log-probabilities of ``(B, m)`` actions in the square."""
        log_prob, _ = self._log_prob_and_entropy(self._distribution(self._tensor(obs)), self._tensor(actions))
        return _numpy(log_prob)

    @torch.no_grad()
    def concentrations(self, obs: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """``(alpha, beta)`` of the policy, each ``(B, m)``."""
        alpha, beta = self.actor(self._tensor(obs))
        return _numpy(alpha), _numpy(beta)

    @torch.no_grad()
    def value(self, obs: ArrayLike) -> FloatArray:
        """Value estimates on the reward scale."""
        return _numpy(self.critic(self._tensor(obs)) * self.ret_rms.std + self.ret_rms.mean)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    @property
    def learning_rates(self) -> list[float]:
        """Current learning rates of the actor and of the critic."""
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def set_learning_rate_fraction(self, fraction: float) -> None:
        """Set every learning rate to ``fraction`` times its initial value (linear annealing).

        Raises:
            ValueError: If ``fraction`` lies outside ``[0, 1]``.
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"the learning-rate fraction must lie in [0, 1], got {fraction}")
        for group, base in zip(self.optimizer.param_groups, self._base_lrs, strict=True):
            group["lr"] = fraction * base

    def update(self, batch: Batch) -> dict[str, float]:
        """One PPO update on ``batch``: ``epochs`` passes in ``minibatches`` random minibatches.

        Minibatches are drawn with PyTorch's global generator. Returns the
        diagnostics of the update; ``explained_variance`` and ``value_bias``
        compare the returns with the value estimates that produced the
        advantages, before the update.

        Raises:
            ValueError: If the batch has fewer than two samples or fewer
                samples than minibatches (empty minibatches would make the
                losses NaN).
        """
        cfg = self.config
        n = len(batch)
        if n < max(2, cfg.minibatches):
            raise ValueError(f"a batch of {n} samples cannot be split into {cfg.minibatches} minibatches")
        adv_std, adv_mean = torch.std_mean(batch.advantages)
        adv_std_raw = float(adv_std)
        advantages = (batch.advantages - adv_mean) / (adv_std_raw + 1e-8)
        self.ret_rms.update(batch.returns)
        norm_returns = (batch.returns - self.ret_rms.mean) / self.ret_rms.std

        clipfracs: list[float] = []
        pg_losses: list[float] = []
        v_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        ratio_max_dev = 0.0
        for _ in range(cfg.epochs):
            indices = torch.randperm(n, device=self.device)
            for mb in torch.tensor_split(indices, cfg.minibatches):
                dist = self._distribution(batch.obs[mb])
                new_log_prob, entropy = self._log_prob_and_entropy(dist, batch.actions[mb])
                new_value = self.critic(batch.obs[mb])
                log_ratio = new_log_prob - batch.log_probs[mb]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    # Estimator of the KL divergence from http://joschu.net/blog/kl-approx.html.
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
                    ratio_max_dev = max(ratio_max_dev, (ratio - 1.0).abs().max().item())
                mb_advantages = advantages[mb]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * ((new_value - norm_returns[mb]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.entropy_coef * entropy_loss + v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy_loss.item())
                approx_kls.append(approx_kl.item())

        returns_np = batch.returns.detach().cpu().numpy()
        values_np = batch.values.detach().cpu().numpy()
        var_y = np.var(returns_np)
        explained_var = np.nan if var_y == 0 else 1 - np.var(returns_np - values_np) / var_y
        return {
            "train/policy_loss": float(np.mean(pg_losses)),
            "train/value_loss": float(np.mean(v_losses)),
            "train/entropy": float(np.mean(entropies)),
            "train/approx_kl": float(np.mean(approx_kls)),
            "train/approx_kl_max": float(np.max(approx_kls)),
            "train/ratio_max_dev": float(ratio_max_dev),
            "train/clipfrac": float(np.mean(clipfracs)),
            "train/value_bias": float((batch.values - batch.returns).mean().item()),
            "train/value_target_mean": float(self.ret_rms.mean),
            "train/value_target_std": float(self.ret_rms.std),
            "train/explained_variance": float(explained_var),
            "train/adv_std_raw": adv_std_raw,
        }

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Weights, optimizer state and target statistics."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ret_rms": self.ret_rms.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the state written by :meth:`state_dict`."""
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.ret_rms.load_state_dict(state["ret_rms"])

    def save(self, path: str | Path) -> None:
        """Write :meth:`state_dict` to ``path``; it loads with ``torch.load(..., weights_only=True)``."""
        torch.save(self.state_dict(), Path(path))

    def load(self, path: str | Path) -> None:
        """Restore the state written by :meth:`save`."""
        self.load_state_dict(torch.load(Path(path), map_location=self.device, weights_only=True))
