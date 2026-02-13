"""Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC).

Advanced policy gradient methods for continuous and discrete control.

PPO (Schulman et al. 2017):
    Clipped surrogate objective that constrains policy updates to a
    trust region without explicit KL penalty or conjugate gradients:

        L^CLIP(theta) = E[min(r_t(theta)*A_t, clip(r_t(theta), 1-eps, 1+eps)*A_t)]

    where r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t) is the
    importance sampling ratio and A_t is the GAE advantage estimate.

    GAE (Generalized Advantage Estimation, Schulman et al. 2016):
        A_t^{GAE(gamma,lambda)} = sum_{l=0}^{T-t} (gamma*lambda)^l * delta_{t+l}
        delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)

    The lambda parameter interpolates between TD(0) (lambda=0, low variance
    / high bias) and MC (lambda=1, high variance / low bias).

SAC (Haarnoja et al. 2018):
    Maximum entropy RL that optimizes a trade-off between expected return
    and entropy:

        pi* = argmax_pi E[sum_t r(s_t,a_t) + alpha*H(pi(.|s_t))]

    Key components:
    1. Twin Q-networks (Q1, Q2) to reduce overestimation bias
    2. Squashed Gaussian policy: a = tanh(mu + sigma*epsilon)
    3. Automatic temperature (alpha) tuning via dual gradient descent
    4. Soft target updates for Q-network stability

References:
    Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
    Schulman et al. (2016) "High-Dimensional Continuous Control Using GAE"
    Haarnoja et al. (2018) "Soft Actor-Critic Algorithms and Applications"
    Haarnoja et al. (2018) "Soft Actor-Critic: Off-Policy Maximum Entropy DRL"
"""
from __future__ import annotations

import copy
import logging
from collections import deque
from typing import Any, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical as TorchCategorical
    from torch.distributions import Normal as TorchNormal

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from quantlaxmi.models.rl.algorithms.policy_gradient import (
    PolicyGradientBase,
    _resolve_device,
    _build_mlp,
)

logger = logging.getLogger(__name__)

__all__ = ["PPO", "SAC"]


# =====================================================================
# PPO Rollout Buffer
# =====================================================================


class _RolloutBuffer:
    """N-step rollout buffer for on-policy algorithms (PPO, A2C).

    Stores complete trajectories from environment interaction, then
    provides mini-batch iteration for multiple epochs of updates.

    Each entry stores: (state, action, reward, done, log_prob, value).
    After a rollout is complete, call ``compute_gae()`` to fill in
    the advantage and return targets.
    """

    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[Any] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.advantages: np.ndarray = np.array([])
        self.returns: np.ndarray = np.array([])

    def add(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages and discounted returns.

        GAE (Schulman et al. 2016):
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}

        The return target for the value function is:
            G_t = A_t + V(s_t)

        Parameters
        ----------
        last_value : float
            V(s_T) for bootstrapping the final step.
        gamma : float
            Discount factor.
        gae_lambda : float
            GAE lambda for bias-variance trade-off.
        """
        T = len(self.rewards)
        self.advantages = np.zeros(T, dtype=np.float64)
        self.returns = np.zeros(T, dtype=np.float64)

        gae = 0.0
        next_value = last_value
        for t in reversed(range(T)):
            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]
            next_value = self.values[t]

    def get_batches(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        """Yield shuffled mini-batches for PPO's multiple-epoch updates.

        Parameters
        ----------
        batch_size : int
            Mini-batch size.
        rng : np.random.Generator
            For shuffling indices.

        Returns
        -------
        list of dict
            Each dict has keys: states, actions, log_probs, advantages, returns.
        """
        T = len(self.states)
        indices = rng.permutation(T)
        batches = []
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            idx = indices[start:end]
            batches.append({
                "states": np.array([self.states[i] for i in idx], dtype=np.float32),
                "actions": [self.actions[i] for i in idx],
                "log_probs": np.array([self.log_probs[i] for i in idx], dtype=np.float32),
                "advantages": self.advantages[idx].astype(np.float32),
                "returns": self.returns[idx].astype(np.float32),
            })
        return batches

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages = np.array([])
        self.returns = np.array([])

    def __len__(self) -> int:
        return len(self.states)


# =====================================================================
# SAC Replay Buffer
# =====================================================================


class _SACReplayBuffer:
    """Replay buffer for off-policy SAC with uniform sampling.

    Stores (state, action, reward, next_state, done) transitions in a
    circular buffer. Sampling is uniform random without replacement
    (with replacement if buffer < batch_size, which shouldn't happen
    in normal usage since we wait for warmup).
    """

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self._buffer: deque[tuple] = deque(maxlen=capacity)
        self._rng = rng

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        replace = len(self._buffer) < batch_size
        indices = self._rng.choice(len(self._buffer), size=batch_size, replace=replace)
        batch = [self._buffer[i] for i in indices]
        return (
            np.array([t[0] for t in batch], dtype=np.float32),
            np.array([t[1] for t in batch], dtype=np.float32),
            np.array([t[2] for t in batch], dtype=np.float32),
            np.array([t[3] for t in batch], dtype=np.float32),
            np.array([t[4] for t in batch], dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)


# =====================================================================
# PPO — Proximal Policy Optimization (Schulman et al. 2017)
# =====================================================================


class PPO(PolicyGradientBase):
    """Proximal Policy Optimization with clipped surrogate objective.

    PPO constrains policy updates to a trust region via clipping:

        L^CLIP(theta) = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]

    where r_t = pi_theta(a|s) / pi_theta_old(a|s).

    Combined loss (per Schulman et al. 2017, Section 5):
        L(theta) = -L^CLIP + c1 * L^VF - c2 * H[pi]

    where L^VF is the value function loss and H is the entropy bonus.

    Advantages are estimated via GAE (lambda):
        A_t = sum_{l=0}^{T-t} (gamma*lambda)^l * delta_{t+l}

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state space.
    action_dim : int
        Number of discrete actions or continuous action dimensions.
    hidden_layers : sequence of int
        Hidden layer sizes for both actor and critic networks.
    learning_rate : float
        Adam optimizer learning rate.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda for advantage estimation (0 = TD(0), 1 = MC).
    clip_eps : float
        PPO clipping parameter epsilon (typically 0.1-0.3).
    entropy_coef : float
        Entropy bonus coefficient c2 (encourages exploration).
    value_coef : float
        Value function loss coefficient c1.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    n_epochs : int
        Number of optimization epochs per rollout.
    batch_size : int
        Mini-batch size for SGD within each epoch.
    clip_value : bool
        If True, also clip value function updates (PPO-2 style).
    continuous : bool
        If True, use Gaussian policy for continuous actions.
    device : str
        "auto", "cpu", or "cuda".
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: Sequence[int] = (64, 64),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        clip_value: bool = False,
        continuous: bool = False,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        assert _HAS_TORCH, "PyTorch is required for PPO"

        self.device = _resolve_device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_value_fn = clip_value
        self.continuous = continuous
        self.state_dim = state_dim
        self.action_dim = action_dim

        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # Actor network
        if continuous:
            self.policy_network = _build_mlp(
                state_dim, action_dim * 2, hidden_layers
            ).to(self.device)
        else:
            self.policy_network = _build_mlp(
                state_dim, action_dim, hidden_layers
            ).to(self.device)

        # Critic network (separate from actor for PPO)
        self.critic_network = _build_mlp(
            state_dim, 1, hidden_layers
        ).to(self.device)

        # Single optimizer for both (common in PPO implementations)
        self.optimizer = optim.Adam(
            list(self.policy_network.parameters())
            + list(self.critic_network.parameters()),
            lr=learning_rate,
        )

        # Rollout buffer
        self._buffer = _RolloutBuffer()

    def _get_distribution(
        self, state_t: "torch.Tensor"
    ) -> "torch.distributions.Distribution":
        """Build the action distribution for a batch of states.

        Parameters
        ----------
        state_t : Tensor of shape (batch, state_dim)

        Returns
        -------
        Distribution
            Categorical for discrete, Normal for continuous.
        """
        if self.continuous:
            output = self.policy_network(state_t)
            mean = output[..., : self.action_dim]
            log_std = output[..., self.action_dim :].clamp(-20, 2)
            std = log_std.exp()
            return TorchNormal(mean, std)
        else:
            logits = self.policy_network(state_t)
            return TorchCategorical(logits=logits)

    def select_action(
        self, state: np.ndarray
    ) -> tuple[Any, float, float]:
        """Select action from the current policy.

        Parameters
        ----------
        state : np.ndarray
            Current state observation.

        Returns
        -------
        (action, log_prob, value)
            Sampled action, its log-probability, and the state value.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist = self._get_distribution(state_t)
            action_t = dist.sample()

            if self.continuous:
                log_prob = dist.log_prob(action_t).sum(dim=-1)
                action = action_t.squeeze(0).cpu().numpy()
            else:
                log_prob = dist.log_prob(action_t)
                action = action_t.item()

            value = self.critic_network(state_t).squeeze()

        return action, log_prob.item(), value.item()

    def predict(self, obs: np.ndarray) -> Any:
        """Predict the best action (deterministic) for deployment.

        Parameters
        ----------
        obs : np.ndarray
            Observation / state.

        Returns
        -------
        action
            For continuous: mean of the Gaussian policy.
            For discrete: argmax of logits.
        """
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.continuous:
                output = self.policy_network(state_t)
                mean = output[..., : self.action_dim]
                return mean.squeeze(0).cpu().numpy()
            else:
                logits = self.policy_network(state_t)
                return logits.argmax(dim=-1).item()

    def store_transition(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Store a transition in the rollout buffer."""
        self._buffer.add(state, action, reward, done, log_prob, value)

    def update(self, last_value: float = 0.0) -> dict[str, float]:
        """Run PPO update on the collected rollout buffer.

        Steps:
        1. Compute GAE advantages from the rollout.
        2. For n_epochs, shuffle and iterate mini-batches:
           a. Recompute log_probs and entropy under current policy.
           b. Compute clipped surrogate loss.
           c. Compute value function loss (optionally clipped).
           d. Combined loss = -L^CLIP + value_coef * L^VF - entropy_coef * H.
           e. Gradient step with grad norm clipping.
        3. Clear the rollout buffer.

        Parameters
        ----------
        last_value : float
            Bootstrap value V(s_T) for the final state.

        Returns
        -------
        dict with keys: policy_loss, value_loss, entropy, approx_kl
        """
        if len(self._buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        # Step 1: Compute GAE
        self._buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0

        # Step 2: Multiple epochs of mini-batch updates
        for _epoch in range(self.n_epochs):
            batches = self._buffer.get_batches(self.batch_size, self._rng)

            for batch in batches:
                states_t = torch.FloatTensor(batch["states"]).to(self.device)
                old_log_probs_t = torch.FloatTensor(batch["log_probs"]).to(self.device)
                advantages_t = torch.FloatTensor(batch["advantages"]).to(self.device)
                returns_t = torch.FloatTensor(batch["returns"]).to(self.device)

                # Normalize advantages (per mini-batch)
                if advantages_t.numel() > 1:
                    advantages_t = (advantages_t - advantages_t.mean()) / (
                        advantages_t.std() + 1e-8
                    )

                # Current policy distribution
                dist = self._get_distribution(states_t)

                # Recompute log probs for the stored actions
                if self.continuous:
                    actions_arr = np.array(batch["actions"], dtype=np.float32)
                    actions_t = torch.FloatTensor(actions_arr).to(self.device)
                    new_log_probs = dist.log_prob(actions_t).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                else:
                    actions_t = torch.LongTensor(
                        [int(a) for a in batch["actions"]]
                    ).to(self.device)
                    new_log_probs = dist.log_prob(actions_t)
                    entropy = dist.entropy().mean()

                # Importance sampling ratio
                ratio = (new_log_probs - old_log_probs_t).exp()

                # Clipped surrogate loss
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                values = self.critic_network(states_t).squeeze(-1)
                if self.clip_value_fn:
                    # Clipped value loss (PPO-2 trick)
                    old_values = returns_t - torch.FloatTensor(
                        batch["advantages"]
                    ).to(self.device)
                    value_clipped = old_values + torch.clamp(
                        values - old_values,
                        -self.clip_eps,
                        self.clip_eps,
                    )
                    vf_loss1 = (values - returns_t).pow(2)
                    vf_loss2 = (value_clipped - returns_t).pow(2)
                    value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(values, returns_t)

                # Combined loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_network.parameters())
                    + list(self.critic_network.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Diagnostics
                with torch.no_grad():
                    approx_kl = (old_log_probs_t - new_log_probs).mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl += approx_kl
                n_updates += 1

        # Step 3: Clear buffer
        self._buffer.clear()

        n = max(n_updates, 1)
        result = {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": total_kl / n,
        }
        logger.debug(
            "PPO update: policy_loss=%.4f value_loss=%.4f entropy=%.4f kl=%.4f",
            result["policy_loss"],
            result["value_loss"],
            result["entropy"],
            result["approx_kl"],
        )
        return result

    def train(
        self,
        env: Any,
        n_steps: int,
        rollout_steps: int = 128,
        log_interval: int = 1000,
    ) -> list[float]:
        """Train PPO on a gym-like environment.

        Collects rollouts of ``rollout_steps`` transitions, then performs
        PPO updates. Repeats until ``n_steps`` total environment steps.

        The environment must support:
            obs = env.reset()
            obs, reward, done, info = env.step(action)

        Parameters
        ----------
        env : gym-like environment
            Must have reset() -> obs and step(action) -> (obs, reward, done, info).
        n_steps : int
            Total number of environment steps to collect.
        rollout_steps : int
            Steps per rollout before each PPO update.
        log_interval : int
            Log episode rewards every this many steps.

        Returns
        -------
        list of float
            Episode rewards collected during training.
        """
        episode_rewards: list[float] = []
        obs = env.reset()
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)

        ep_reward = 0.0
        total_steps = 0

        while total_steps < n_steps:
            # Collect rollout
            for _ in range(rollout_steps):
                if total_steps >= n_steps:
                    break

                action, log_prob, value = self.select_action(obs)
                result = env.step(action)

                # Handle both 4-tuple and 5-tuple returns
                if len(result) == 5:
                    next_obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                elif len(result) == 4:
                    next_obs, reward, done, info = result
                else:
                    raise ValueError(f"env.step() returned {len(result)} values, expected 4 or 5")

                if not isinstance(next_obs, np.ndarray):
                    next_obs = np.array(next_obs, dtype=np.float32)

                self.store_transition(obs, action, reward, done, log_prob, value)
                obs = next_obs
                ep_reward += reward
                total_steps += 1

                if done:
                    episode_rewards.append(ep_reward)
                    if len(episode_rewards) % max(1, log_interval // rollout_steps) == 0:
                        logger.info(
                            "PPO step %d | episodes %d | mean_reward=%.2f",
                            total_steps,
                            len(episode_rewards),
                            np.mean(episode_rewards[-10:]),
                        )
                    ep_reward = 0.0
                    obs = env.reset()
                    if not isinstance(obs, np.ndarray):
                        obs = np.array(obs, dtype=np.float32)

            # Compute bootstrap value for the last state
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                last_value = self.critic_network(obs_t).squeeze().item()

            # PPO update
            self.update(last_value=last_value)

        return episode_rewards


# =====================================================================
# Squashed Gaussian Policy Network for SAC
# =====================================================================


class _SquashedGaussianPolicy(nn.Module):
    """Squashed Gaussian policy for SAC.

    Outputs a tanh-squashed Gaussian: a = tanh(mu + sigma * epsilon).

    The log-probability must account for the Jacobian of the tanh transform:
        log pi(a|s) = log N(u|mu,sigma) - sum log(1 - tanh^2(u))

    where u is the pre-squash sample.

    This ensures actions are bounded in [-1, 1], which can then be
    rescaled to the environment's action range.

    Parameters
    ----------
    state_dim : int
    action_dim : int
    hidden_layers : sequence of int
    log_std_min : float
        Lower bound for log standard deviation.
    log_std_max : float
        Upper bound for log standard deviation.
    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, action_dim)
        self.log_std_head = nn.Linear(prev, action_dim)

    def forward(
        self, state: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return (mean, log_std) of the Gaussian before squashing."""
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, state: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Sample action with reparameterization trick + tanh squashing.

        Returns
        -------
        (action, log_prob)
            action is in [-1, 1], log_prob accounts for tanh Jacobian.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = TorchNormal(mean, std)

        # Reparameterization trick: u = mu + sigma * epsilon
        u = dist.rsample()

        # Squash through tanh
        action = torch.tanh(u)

        # Log prob with Jacobian correction
        # log pi(a|s) = log N(u) - sum log(1 - tanh^2(u))
        log_prob = dist.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic(
        self, state: "torch.Tensor"
    ) -> "torch.Tensor":
        """Return the deterministic action (mean squashed)."""
        mean, _ = self.forward(state)
        return torch.tanh(mean)


# =====================================================================
# SAC — Soft Actor-Critic (Haarnoja et al. 2018)
# =====================================================================


class SAC(PolicyGradientBase):
    """Soft Actor-Critic with automatic temperature tuning.

    Maximum entropy objective:
        J(pi) = E[sum_t gamma^t (r_t + alpha * H(pi(.|s_t)))]

    The entropy bonus alpha * H(pi) encourages exploration and leads to
    more robust policies. Alpha is tuned automatically via:
        alpha* = argmin_alpha E[-alpha * (log pi(a|s) + H_target)]

    where H_target = -dim(A) (negative of action dimensionality).

    Twin Q-networks (Fujimoto et al. 2018):
        Uses min(Q1, Q2) to compute targets, reducing overestimation.

    Soft Bellman backup:
        y = r + gamma * (min(Q1', Q2')(s', a') - alpha * log pi(a'|s'))
        where a' ~ pi(.|s')

    Parameters
    ----------
    state_dim : int
    action_dim : int
    hidden_layers : sequence of int
        Hidden layer sizes for all networks.
    actor_lr : float
        Learning rate for the policy network.
    critic_lr : float
        Learning rate for the twin Q-networks.
    alpha_lr : float
        Learning rate for automatic temperature tuning.
    gamma : float
        Discount factor.
    tau : float
        Soft target update coefficient (Polyak averaging).
    alpha_init : float
        Initial temperature parameter.
    auto_alpha : bool
        If True, tune alpha automatically via dual gradient descent.
    buffer_size : int
        Replay buffer capacity.
    batch_size : int
        Mini-batch size for gradient updates.
    warmup_steps : int
        Number of random actions before training begins.
    device : str
    seed : int
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: Sequence[int] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 0.2,
        auto_alpha: bool = True,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        assert _HAS_TORCH, "PyTorch is required for SAC"

        self.device = _resolve_device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.auto_alpha = auto_alpha

        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # Policy (actor): squashed Gaussian
        self.policy_network = _SquashedGaussianPolicy(
            state_dim, action_dim, hidden_layers
        ).to(self.device)

        # Twin Q-networks (critic)
        self.q1_network = _build_mlp(
            state_dim + action_dim, 1, hidden_layers
        ).to(self.device)
        self.q2_network = _build_mlp(
            state_dim + action_dim, 1, hidden_layers
        ).to(self.device)

        # Target Q-networks
        self.q1_target = copy.deepcopy(self.q1_network)
        self.q2_target = copy.deepcopy(self.q2_network)

        # Optimizers
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=actor_lr)
        self.q1_optimizer = optim.Adam(self.q1_network.parameters(), lr=critic_lr)
        self.q2_optimizer = optim.Adam(self.q2_network.parameters(), lr=critic_lr)

        # Temperature (alpha) — entropy coefficient
        if auto_alpha:
            # Target entropy = -dim(A) (heuristic from Haarnoja et al.)
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.tensor(
                np.log(alpha_init), dtype=torch.float32, device=self.device, requires_grad=True
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = torch.tensor(
                np.log(alpha_init), dtype=torch.float32, device=self.device
            )

        # Replay buffer
        self._buffer = _SACReplayBuffer(buffer_size, self._rng)
        self._total_steps = 0

    @property
    def alpha(self) -> "torch.Tensor":
        """Current temperature parameter."""
        return self.log_alpha.exp()

    def select_action(
        self, state: np.ndarray, explore: bool = True
    ) -> np.ndarray:
        """Select action from the squashed Gaussian policy.

        During warmup (first ``warmup_steps``), returns uniform random actions.
        After warmup:
            - explore=True: sample from pi(a|s)
            - explore=False: use deterministic mean action

        Parameters
        ----------
        state : np.ndarray
        explore : bool
            If True, sample stochastically.

        Returns
        -------
        np.ndarray
            Action in [-1, 1]^action_dim.
        """
        if explore and self._total_steps < self.warmup_steps:
            return self._rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if explore:
                action, _ = self.policy_network.sample(state_t)
            else:
                action = self.policy_network.deterministic(state_t)

        return action.squeeze(0).cpu().numpy()

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict the deterministic action for deployment.

        Parameters
        ----------
        obs : np.ndarray
            Observation / state.

        Returns
        -------
        np.ndarray
            Deterministic action (tanh of mean).
        """
        return self.select_action(obs, explore=False)

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self._buffer.add(state, action, reward, next_state, done)
        self._total_steps += 1

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """Polyak averaging: target <- tau * source + (1 - tau) * target."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def update(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Perform one SAC update step.

        Steps:
        1. Sample mini-batch from replay buffer.
        2. Compute soft Bellman target:
           y = r + gamma * (min(Q1', Q2')(s', a') - alpha * log pi(a'|s'))
        3. Update Q1, Q2 to minimize MSE(Qi(s,a), y).
        4. Update policy to maximize E[min(Q1, Q2)(s, a_new) - alpha * log pi(a_new|s)].
        5. (Optional) Update alpha to target the desired entropy.
        6. Soft-update target networks.

        Returns
        -------
        dict with keys: q1_loss, q2_loss, policy_loss, alpha, entropy
        """
        if len(self._buffer) < self.batch_size:
            return {"q1_loss": 0.0, "q2_loss": 0.0, "policy_loss": 0.0,
                    "alpha": self.alpha.item(), "entropy": 0.0}

        # Step 1: Sample
        states, actions, rewards, next_states, dones = self._buffer.sample(self.batch_size)
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Step 2: Compute target Q-values
        with torch.no_grad():
            next_actions, next_log_probs = self.policy_network.sample(next_states_t)
            q1_next = self.q1_target(torch.cat([next_states_t, next_actions], dim=1))
            q2_next = self.q2_target(torch.cat([next_states_t, next_actions], dim=1))
            q_next = torch.min(q1_next, q2_next)
            # Soft Bellman target
            y = rewards_t + self.gamma * (1.0 - dones_t) * (q_next - self.alpha.detach() * next_log_probs)

        # Step 3: Update Q-networks
        q1_pred = self.q1_network(torch.cat([states_t, actions_t], dim=1))
        q2_pred = self.q2_network(torch.cat([states_t, actions_t], dim=1))

        q1_loss = F.mse_loss(q1_pred, y)
        q2_loss = F.mse_loss(q2_pred, y)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Step 4: Update policy
        new_actions, new_log_probs = self.policy_network.sample(states_t)
        q1_new = self.q1_network(torch.cat([states_t, new_actions], dim=1))
        q2_new = self.q2_network(torch.cat([states_t, new_actions], dim=1))
        q_new = torch.min(q1_new, q2_new)

        # Policy loss: minimize alpha * log_pi - Q (maximize Q - alpha * log_pi)
        policy_loss = (self.alpha.detach() * new_log_probs - q_new).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Step 5: Update alpha (automatic temperature tuning)
        alpha_loss_val = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (new_log_probs.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_val = alpha_loss.item()

        # Step 6: Soft target updates
        self._soft_update(self.q1_target, self.q1_network)
        self._soft_update(self.q2_target, self.q2_network)

        entropy = -new_log_probs.mean().item()

        result = {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": entropy,
            "alpha_loss": alpha_loss_val,
        }
        logger.debug(
            "SAC update: q1=%.4f q2=%.4f pi=%.4f alpha=%.4f ent=%.4f",
            result["q1_loss"],
            result["q2_loss"],
            result["policy_loss"],
            result["alpha"],
            result["entropy"],
        )
        return result

    def train(
        self,
        env: Any,
        n_steps: int,
        updates_per_step: int = 1,
        log_interval: int = 1000,
    ) -> list[float]:
        """Train SAC on a gym-like environment.

        Interleaves environment interaction with gradient updates.
        During warmup, actions are random; after warmup, actions come
        from the learned policy.

        Parameters
        ----------
        env : gym-like environment
            Must have reset() -> obs and step(action) -> (obs, reward, done, info).
        n_steps : int
            Total number of environment steps.
        updates_per_step : int
            Number of gradient updates per environment step (after warmup).
        log_interval : int
            Log episode rewards every this many steps.

        Returns
        -------
        list of float
            Episode rewards collected during training.
        """
        episode_rewards: list[float] = []
        obs = env.reset()
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)

        ep_reward = 0.0

        for step in range(n_steps):
            action = self.select_action(obs, explore=True)
            result = env.step(action)

            if len(result) == 5:
                next_obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            elif len(result) == 4:
                next_obs, reward, done, info = result
            else:
                raise ValueError(f"env.step() returned {len(result)} values, expected 4 or 5")

            if not isinstance(next_obs, np.ndarray):
                next_obs = np.array(next_obs, dtype=np.float32)

            self.store_transition(obs, action, float(reward), next_obs, done)
            obs = next_obs
            ep_reward += reward

            # Gradient updates (after warmup)
            if self._total_steps >= self.warmup_steps:
                for _ in range(updates_per_step):
                    self.update()

            if done:
                episode_rewards.append(ep_reward)
                if len(episode_rewards) % max(1, log_interval) == 0:
                    logger.info(
                        "SAC step %d | episodes %d | mean_reward=%.2f | alpha=%.4f",
                        step,
                        len(episode_rewards),
                        np.mean(episode_rewards[-10:]),
                        self.alpha.item(),
                    )
                ep_reward = 0.0
                obs = env.reset()
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs, dtype=np.float32)

        return episode_rewards
