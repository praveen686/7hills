"""Policy Gradient Methods for Reinforcement Learning.

Implements Chapter 14 of "Foundations of RL with Applications in Finance"
by Ashwin Rao & Tikhon Jelvis.

Policy gradient methods directly parameterize and optimize the policy
pi(a|s; theta), as opposed to value-based methods that derive the policy
from a learned value function.

Central theorem — Policy Gradient Theorem (Theorem 14.1, Sutton et al. 2000):
    nabla_theta J(theta) = E_pi [sum_t gamma^t * nabla_theta log pi(A_t|S_t; theta) * Psi_t]

where Psi_t can be:
    - G_t                                   (REINFORCE)
    - G_t - b(S_t)                          (REINFORCE with baseline)
    - Q^pi(S_t, A_t)                        (Q Actor-Critic)
    - A^pi(S_t, A_t) = Q^pi - V^pi          (Advantage Actor-Critic)
    - delta_t = R_{t+1} + gamma*V(S_{t+1}) - V(S_t)  (TD Actor-Critic)

Key algorithms:
    - REINFORCE (Section 14.4) — Monte Carlo policy gradient
    - Actor-Critic (Section 14.6) — TD-bootstrapped policy gradient
    - A2C (Advantage Actor-Critic with n-step returns)
    - Natural Policy Gradient (Section 14.7) — Fisher-aware gradient
    - Deterministic Policy Gradient / DDPG (Section 14.8)

References:
    Rao & Jelvis, Ch 14 (Policy Gradient Methods)
    Sutton et al. (2000) "Policy gradient methods for RL with function approximation"
    Williams (1992) "Simple statistical gradient-following algorithms for connectionist RL"
    Silver et al. (2014) "Deterministic Policy Gradient Algorithms"
    Lillicrap et al. (2016) "Continuous control with deep RL" (DDPG)
"""
from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Optional, Sequence

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

# ---------------------------------------------------------------------------
# Core interface imports
# ---------------------------------------------------------------------------
try:
    from models.rl.core.function_approx import FunctionApprox
    from models.rl.core.utils import set_device
except ImportError:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "PolicyGradientBase",
    "REINFORCE",
    "ActorCritic",
    "A2C",
    "NaturalPolicyGradient",
    "DeterministicPolicyGradient",
]


# =====================================================================
# Device helper
# =====================================================================


def _resolve_device(device: str) -> "torch.device":
    assert _HAS_TORCH, "PyTorch is required for policy gradient methods"
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# =====================================================================
# Network builders
# =====================================================================


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layers: Sequence[int],
    activation: type[nn.Module] = nn.ReLU,
    output_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    """Build a multi-layer perceptron."""
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


# =====================================================================
# Replay Buffer (for off-policy methods like DDPG)
# =====================================================================


class _ReplayBuffer:
    """Simple replay buffer for continuous-action off-policy methods."""

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
        indices = self._rng.choice(len(self._buffer), size=batch_size, replace=False)
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
# PolicyGradientBase (ABC)
# =====================================================================


class PolicyGradientBase(ABC):
    """Abstract base class for all policy gradient methods.

    The Policy Gradient Theorem (Ch 14.2, Theorem 14.1):
        nabla_theta J(theta) = E_pi [sum_t gamma^t * nabla_theta log pi(A_t|S_t; theta) * Psi_t]

    This theorem is remarkable because the gradient of the objective J(theta)
    does NOT require differentiating the state distribution d^pi(s), which
    depends on the policy parameters.  The gradient depends only on the
    score function nabla_theta log pi(a|s; theta) and the appropriate
    signal Psi_t.

    Subclasses differ in:
        1. What Psi_t is used (return, advantage, TD error, etc.)
        2. Whether a critic is used (and how it's trained)
        3. Whether the policy is stochastic or deterministic
    """

    policy_network: nn.Module
    optimizer: torch.optim.Optimizer
    gamma: float
    device: torch.device

    @abstractmethod
    def select_action(self, state: np.ndarray) -> Any:
        """Select an action given the current state."""

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> Any:
        """Update policy (and optionally value) parameters."""


# =====================================================================
# REINFORCE — Monte Carlo Policy Gradient (Ch 14.4)
# =====================================================================


class REINFORCE(PolicyGradientBase):
    """REINFORCE — Monte Carlo Policy Gradient (Ch 14.4, Williams 1992).

    The simplest policy gradient algorithm.  Uses the full episodic return
    G_t as the signal Psi_t in the policy gradient theorem:

        Delta_theta = alpha * gamma^t * nabla_theta log pi(A_t|S_t; theta) * G_t

    With baseline (variance reduction, Section 14.5):
        Delta_theta = alpha * gamma^t * nabla_theta log pi(A_t|S_t; theta) * (G_t - b(S_t))

    The baseline b(S_t) does NOT introduce bias (since nabla_theta is taken
    w.r.t. theta, and b does not depend on the action), but can dramatically
    reduce variance.  The optimal baseline is approximately V^pi(S_t).

    For discrete actions:
        pi(a|s; theta) = softmax of network output logits

    For continuous actions:
        pi(a|s; theta) = N(mu(s; theta), sigma^2)
        where mu is the network output and sigma is learned or fixed.

    Parameters
    ----------
    state_dim : int
        Dimensionality of state space.
    action_dim : int
        Number of discrete actions, or dimensionality of continuous action space.
    hidden_layers : sequence of int
        Hidden layer sizes for the policy network.
    learning_rate : float
        Optimizer learning rate.
    gamma : float
        Discount factor.
    baseline : optional
        Value function baseline for variance reduction.
        If None, a simple neural network baseline is created.
        If False, no baseline is used (pure REINFORCE).
    continuous : bool
        If True, outputs mean and log-std for Gaussian policy.
    device : str
        "auto", "cpu", or "cuda".
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: Sequence[int] = (64, 32),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        baseline: Any | None = None,
        continuous: bool = False,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        assert _HAS_TORCH, "PyTorch is required for REINFORCE"

        self.device = _resolve_device(device)
        self.gamma = gamma
        self.continuous = continuous
        self.state_dim = state_dim
        self.action_dim = action_dim

        torch.manual_seed(seed)

        if continuous:
            # Output: [mean_1, ..., mean_d, log_std_1, ..., log_std_d]
            self.policy_network = _build_mlp(
                state_dim, action_dim * 2, hidden_layers
            ).to(self.device)
        else:
            self.policy_network = _build_mlp(
                state_dim, action_dim, hidden_layers
            ).to(self.device)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # Baseline (value function for variance reduction)
        self._use_baseline = baseline is not False
        if self._use_baseline:
            if baseline is None:
                self._baseline_net = _build_mlp(
                    state_dim, 1, hidden_layers
                ).to(self.device)
                self._baseline_optimizer = optim.Adam(
                    self._baseline_net.parameters(), lr=learning_rate * 3
                )
            else:
                self._baseline_net = None  # use external baseline
                self._external_baseline = baseline

        # Episode storage
        self._saved_log_probs: list[torch.Tensor] = []
        self._saved_rewards: list[float] = []
        self._saved_states: list[np.ndarray] = []

    def select_action(
        self, state: np.ndarray
    ) -> tuple[int | np.ndarray, float]:
        """Select action from the current stochastic policy.

        For discrete actions: samples from Categorical(softmax(logits)).
        For continuous actions: samples from N(mu, sigma^2).

        Parameters
        ----------
        state : np.ndarray
            Current state.

        Returns
        -------
        (action, log_prob)
            The sampled action and its log-probability under the policy.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if self.continuous:
            output = self.policy_network(state_t)
            mean = output[:, : self.action_dim]
            log_std = output[:, self.action_dim :].clamp(-20, 2)
            std = log_std.exp()
            dist = TorchNormal(mean, std)
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t).sum(dim=-1)
            action = action_t.squeeze(0).cpu().detach().numpy()
        else:
            logits = self.policy_network(state_t)
            dist = TorchCategorical(logits=logits)
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t)
            action = action_t.item()

        self._saved_log_probs.append(log_prob)
        self._saved_states.append(state)

        return action, log_prob.item()

    def store_reward(self, reward: float) -> None:
        """Store reward for the current time step."""
        self._saved_rewards.append(reward)

    def update(
        self,
        episode: list[tuple[np.ndarray, int | np.ndarray, float, float]] | None = None,
    ) -> float:
        """Update policy from a complete episode.

        If ``episode`` is provided, it should be a list of
        (state, action, reward, log_prob) tuples.

        Otherwise, uses internally stored log_probs and rewards from
        ``select_action()`` and ``store_reward()``.

        REINFORCE update (Ch 14.4):
            loss = -sum_t gamma^t * log pi(a_t|s_t; theta) * (G_t - b(s_t))

        Parameters
        ----------
        episode : list or None
            If provided, each entry is (state, action, reward, log_prob).

        Returns
        -------
        float
            Policy loss value.
        """
        if episode is not None:
            states = [e[0] for e in episode]
            rewards = [e[2] for e in episode]
            T = len(rewards)
            # Recompute log probs
            log_probs = []
            for s, a, _, _ in episode:
                s_t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
                if self.continuous:
                    output = self.policy_network(s_t)
                    mean = output[:, : self.action_dim]
                    log_std = output[:, self.action_dim :].clamp(-20, 2)
                    std = log_std.exp()
                    dist = TorchNormal(mean, std)
                    a_t = torch.FloatTensor(np.array(a)).unsqueeze(0).to(self.device)
                    lp = dist.log_prob(a_t).sum(dim=-1)
                else:
                    logits = self.policy_network(s_t)
                    dist = TorchCategorical(logits=logits)
                    lp = dist.log_prob(torch.tensor(a).to(self.device))
                log_probs.append(lp)
        else:
            # Align all three lists by minimum length.
            # Each "step" should have a (state, log_prob, reward) triple.
            # select_action populates states and log_probs;
            # store_reward populates rewards.
            T = min(len(self._saved_states), len(self._saved_log_probs), len(self._saved_rewards))
            states = self._saved_states[:T]
            log_probs = self._saved_log_probs[:T]
            rewards = self._saved_rewards[:T]

        if not rewards:
            return 0.0

        # Compute discounted returns G_t
        G = np.zeros(T, dtype=np.float64)
        running = 0.0
        for t in reversed(range(T)):
            running = rewards[t] + self.gamma * running
            G[t] = running

        # Normalize returns (common practice for stability)
        G_t = torch.FloatTensor(G).to(self.device)
        if T > 1:
            G_t = (G_t - G_t.mean()) / (G_t.std() + 1e-8)

        # Baseline
        baseline_loss = torch.tensor(0.0, device=self.device)
        if self._use_baseline and hasattr(self, "_baseline_net") and self._baseline_net is not None:
            states_t = torch.FloatTensor(np.array(states)).to(self.device)
            baselines = self._baseline_net(states_t).squeeze(-1)
            advantages = G_t - baselines.detach()
            baseline_loss = F.mse_loss(baselines, G_t.detach())
        else:
            advantages = G_t

        # Policy gradient loss: -sum gamma^t * log_pi * advantage
        policy_loss = torch.tensor(0.0, device=self.device)
        discount = 1.0
        for t in range(T):
            policy_loss -= discount * log_probs[t].squeeze() * advantages[t]
            discount *= self.gamma

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Update baseline
        if self._use_baseline and hasattr(self, "_baseline_net") and self._baseline_net is not None:
            self._baseline_optimizer.zero_grad()
            # Recompute baseline loss with fresh forward pass
            states_t = torch.FloatTensor(np.array(states)).to(self.device)
            G_t_detached = torch.FloatTensor(G).to(self.device)
            baselines_fresh = self._baseline_net(states_t).squeeze(-1)
            bl_loss = F.mse_loss(baselines_fresh, G_t_detached)
            bl_loss.backward()
            self._baseline_optimizer.step()

        # Clear episode storage
        self._saved_log_probs = []
        self._saved_rewards = []
        self._saved_states = []

        return policy_loss.item()


# =====================================================================
# Actor-Critic (Ch 14.6)
# =====================================================================


class ActorCritic(PolicyGradientBase):
    """Actor-Critic — TD-bootstrapped Policy Gradient (Ch 14.6).

    Combines a policy (actor) and a value function (critic):

    Actor:  pi(a|s; theta) — the policy network
    Critic: V(s; w) — the value network

    The TD error serves as an unbiased estimate of the advantage:
        delta_t = R_{t+1} + gamma * V(S_{t+1}; w) - V(S_t; w)

    Actor update:
        theta <- theta + alpha_theta * gamma^t * delta_t * nabla_theta log pi(A_t|S_t; theta)

    Critic update:
        w <- w + alpha_w * delta_t * nabla_w V(S_t; w)

    Key insight (Ch 14.6):
        Using the TD error delta_t instead of the full return G_t gives us
        single-step updates (no need to wait for episode end) while still
        being an unbiased estimate of the advantage A^pi(s,a).

    Entropy regularization (Mnih et al. 2016):
        Adding an entropy bonus H(pi(.|s)) to the objective encourages
        exploration and prevents premature convergence to deterministic
        policies:
            J(theta) = E[sum gamma^t * (R_{t+1} + beta * H(pi(.|S_t; theta)))]

    Parameters
    ----------
    state_dim : int
    action_dim : int
    actor_hidden : sequence of int
    critic_hidden : sequence of int
    actor_lr : float
    critic_lr : float
    gamma : float
    entropy_coeff : float
        Weight of entropy bonus for exploration.
    continuous : bool
        True for continuous action spaces.
    device : str
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_hidden: Sequence[int] = (64, 32),
        critic_hidden: Sequence[int] = (64, 32),
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        continuous: bool = False,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        assert _HAS_TORCH, "PyTorch is required for Actor-Critic"

        self.device = _resolve_device(device)
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.continuous = continuous
        self.state_dim = state_dim
        self.action_dim = action_dim

        torch.manual_seed(seed)

        # Actor network
        if continuous:
            self.policy_network = _build_mlp(
                state_dim, action_dim * 2, actor_hidden
            ).to(self.device)
        else:
            self.policy_network = _build_mlp(
                state_dim, action_dim, actor_hidden
            ).to(self.device)

        # Critic network
        self.critic_network = _build_mlp(state_dim, 1, critic_hidden).to(self.device)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=critic_lr)

    def select_action(
        self, state: np.ndarray
    ) -> tuple[int | np.ndarray, float, float]:
        """Select action and return (action, log_prob, state_value).

        Parameters
        ----------
        state : np.ndarray

        Returns
        -------
        (action, log_prob, state_value)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # State value from critic
        value = self.critic_network(state_t).squeeze()

        if self.continuous:
            output = self.policy_network(state_t)
            mean = output[:, : self.action_dim]
            log_std = output[:, self.action_dim :].clamp(-20, 2)
            std = log_std.exp()
            dist = TorchNormal(mean, std)
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t).sum(dim=-1)
            action = action_t.squeeze(0).cpu().detach().numpy()
        else:
            logits = self.policy_network(state_t)
            dist = TorchCategorical(logits=logits)
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t)
            action = action_t.item()

        return action, log_prob.item(), value.item()

    def update(
        self,
        transitions: list[tuple[np.ndarray, Any, float, float, np.ndarray | None, bool]],
    ) -> tuple[float, float]:
        """Update actor and critic from a batch of transitions.

        Each transition is (state, action, reward, log_prob, next_state, done).

        Actor-Critic update (Ch 14.6):
            delta_t = R_{t+1} + gamma * V(S_{t+1}; w) - V(S_t; w)
            Actor loss:  -sum log pi(a|s; theta) * delta_t.detach() - beta * H(pi)
            Critic loss: sum delta_t^2

        Parameters
        ----------
        transitions : list of tuples
            Each: (state, action, reward, log_prob, next_state, done).

        Returns
        -------
        (actor_loss, critic_loss)
        """
        if not transitions:
            return 0.0, 0.0

        states = torch.FloatTensor(np.array([t[0] for t in transitions])).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in transitions]).to(self.device)
        dones = torch.FloatTensor([float(t[5]) for t in transitions]).to(self.device)

        # Recompute log probs and entropy
        log_probs_list = []
        entropy_list = []
        for t in transitions:
            s = t[0]
            a = t[1]
            s_t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
            if self.continuous:
                output = self.policy_network(s_t)
                mean = output[:, : self.action_dim]
                log_std = output[:, self.action_dim :].clamp(-20, 2)
                std = log_std.exp()
                dist = TorchNormal(mean, std)
                a_t = torch.FloatTensor(np.array(a)).unsqueeze(0).to(self.device)
                lp = dist.log_prob(a_t).sum(dim=-1)
                ent = dist.entropy().sum(dim=-1)
            else:
                logits = self.policy_network(s_t)
                dist = TorchCategorical(logits=logits)
                lp = dist.log_prob(torch.tensor(a).to(self.device))
                ent = dist.entropy()
            log_probs_list.append(lp)
            entropy_list.append(ent)

        log_probs = torch.stack(log_probs_list).squeeze()
        entropy = torch.stack(entropy_list).squeeze()

        # Critic values
        values = self.critic_network(states).squeeze(-1)

        # Next state values (for TD target)
        next_values = torch.zeros_like(rewards)
        for i, t in enumerate(transitions):
            if not t[5] and t[4] is not None:  # not done
                ns_t = torch.FloatTensor(t[4]).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    next_values[i] = self.critic_network(ns_t).squeeze()

        # TD error: delta = R + gamma * V(S') - V(S)
        td_targets = rewards + self.gamma * next_values * (1.0 - dones)
        advantages = (td_targets - values).detach()

        # Actor loss: -E[log pi * advantage + beta * entropy]
        actor_loss = -(log_probs * advantages + self.entropy_coeff * entropy).mean()

        # Critic loss: MSE of TD error
        critic_loss = F.mse_loss(values, td_targets.detach())

        # Update actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=5.0)
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()


# =====================================================================
# A2C — Advantage Actor-Critic with n-step returns (Ch 14.6)
# =====================================================================


class A2C(ActorCritic):
    """Advantage Actor-Critic with n-step returns (A2C).

    Extension of Actor-Critic that uses n-step returns instead of TD(0):

        G_t^{(n)} = sum_{k=0}^{n-1} gamma^k * R_{t+k+1} + gamma^n * V(S_{t+n}; w)

    The advantage estimate becomes:
        A_t = G_t^{(n)} - V(S_t; w)

    n-step returns provide a bias-variance trade-off between TD(0) (n=1,
    low variance / high bias) and MC (n=T, high variance / low bias).

    In practice, A2C also supports multiple parallel environments for
    better sample efficiency and gradient stability.

    Parameters
    ----------
    n_steps : int
        Number of steps for n-step returns.
    num_envs : int
        Number of parallel environments (for vectorized training).
    **kwargs
        Passed to ActorCritic.__init__.
    """

    def __init__(
        self,
        n_steps: int = 5,
        num_envs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_steps = n_steps
        self.num_envs = num_envs

    def compute_n_step_returns(
        self,
        rewards: list[list[float]],
        next_values: list[float],
        dones: list[list[bool]],
    ) -> list[list[float]]:
        """Compute n-step returns for each environment.

        G_t^{(n)} = sum_{k=0}^{n-1} gamma^k * R_{t+k+1} + gamma^n * V(S_{t+n})

        If the episode terminates within n steps, the return is truncated
        at the terminal state (no bootstrapping past terminal).

        Parameters
        ----------
        rewards : list of list of float
            rewards[env][step]
        next_values : list of float
            V(S_{t+n}) for each environment.
        dones : list of list of bool
            dones[env][step]

        Returns
        -------
        list of list of float
            n-step returns for each (env, step).
        """
        all_returns = []
        for env_idx in range(len(rewards)):
            env_rewards = rewards[env_idx]
            T = len(env_rewards)
            returns_list = [0.0] * T
            # Backward computation
            running = next_values[env_idx]
            for t in reversed(range(T)):
                if dones[env_idx][t]:
                    running = 0.0
                running = env_rewards[t] + self.gamma * running
                returns_list[t] = running
            all_returns.append(returns_list)
        return all_returns

    def update_n_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        n_step_returns: np.ndarray,
    ) -> tuple[float, float]:
        """Update actor and critic using n-step returns.

        Parameters
        ----------
        states : np.ndarray of shape (batch, state_dim)
        actions : np.ndarray of shape (batch,) for discrete or (batch, action_dim) for continuous
        n_step_returns : np.ndarray of shape (batch,)

        Returns
        -------
        (actor_loss, critic_loss)
        """
        states_t = torch.FloatTensor(states).to(self.device)
        returns_t = torch.FloatTensor(n_step_returns).to(self.device)

        # Critic
        values = self.critic_network(states_t).squeeze(-1)
        advantages = (returns_t - values).detach()

        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor: recompute log probs
        log_probs_list = []
        entropy_list = []
        for i in range(len(states)):
            s_t = states_t[i].unsqueeze(0)
            if self.continuous:
                output = self.policy_network(s_t)
                mean = output[:, : self.action_dim]
                log_std = output[:, self.action_dim :].clamp(-20, 2)
                std = log_std.exp()
                dist = TorchNormal(mean, std)
                a_t = torch.FloatTensor(actions[i]).unsqueeze(0).to(self.device)
                lp = dist.log_prob(a_t).sum(dim=-1)
                ent = dist.entropy().sum(dim=-1)
            else:
                logits = self.policy_network(s_t)
                dist = TorchCategorical(logits=logits)
                lp = dist.log_prob(torch.tensor(int(actions[i])).to(self.device))
                ent = dist.entropy()
            log_probs_list.append(lp)
            entropy_list.append(ent)

        log_probs = torch.stack(log_probs_list).squeeze()
        entropy = torch.stack(entropy_list).squeeze()

        actor_loss = -(log_probs * advantages + self.entropy_coeff * entropy).mean()
        critic_loss = F.mse_loss(values, returns_t.detach())

        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=5.0)
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()


# =====================================================================
# Natural Policy Gradient (Ch 14.7)
# =====================================================================


class NaturalPolicyGradient(PolicyGradientBase):
    """Natural Policy Gradient (Ch 14.7, Kakade 2002).

    Standard gradient descent moves in the direction of steepest descent
    in *parameter* space, which can be very different from the direction of
    steepest descent in *distribution* space (the space of policies).

    The Natural Policy Gradient accounts for the geometry of the policy
    distribution by pre-multiplying the gradient with the inverse Fisher
    Information Matrix (FIM):

        theta <- theta + alpha * F(theta)^{-1} * nabla_theta J(theta)

    where F(theta) is the Fisher Information Matrix:
        F(theta) = E_pi [nabla_theta log pi(a|s;theta) * nabla_theta log pi(a|s;theta)^T]

    In practice, directly computing and inverting F is prohibitively expensive.
    Instead, we solve F(theta) * x = nabla_theta J(theta) using the conjugate
    gradient (CG) method, which only requires Fisher-vector products.

    The Fisher-vector product Fv can be computed efficiently as:
        Fv = (1/N) * sum_i nabla_theta [nabla_theta log pi(a_i|s_i;theta)^T * v] * nabla_theta log pi(a_i|s_i;theta)

    This avoids explicit FIM construction (which would be O(d^2) where d is
    the number of parameters).

    Parameters
    ----------
    state_dim : int
    action_dim : int
    hidden_layers : sequence of int
    learning_rate : float
        Step size alpha.
    gamma : float
    cg_iterations : int
        Number of conjugate gradient iterations.
    cg_damping : float
        Damping coefficient for numerical stability in CG.
    device : str
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: Sequence[int] = (64, 32),
        learning_rate: float = 0.01,
        gamma: float = 0.99,
        cg_iterations: int = 10,
        cg_damping: float = 0.1,
        continuous: bool = False,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        assert _HAS_TORCH, "PyTorch is required for Natural Policy Gradient"

        self.device = _resolve_device(device)
        self.gamma = gamma
        self.lr = learning_rate
        self.cg_iterations = cg_iterations
        self.cg_damping = cg_damping
        self.continuous = continuous
        self.state_dim = state_dim
        self.action_dim = action_dim

        torch.manual_seed(seed)

        if continuous:
            self.policy_network = _build_mlp(
                state_dim, action_dim * 2, hidden_layers
            ).to(self.device)
        else:
            self.policy_network = _build_mlp(
                state_dim, action_dim, hidden_layers
            ).to(self.device)

        # Baseline critic
        self.critic_network = _build_mlp(state_dim, 1, hidden_layers).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=1e-3)

        # Dummy optimizer (parameters updated manually via natural gradient)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def select_action(
        self, state: np.ndarray
    ) -> tuple[int | np.ndarray, float, float]:
        """Select action, return (action, log_prob, value)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        value = self.critic_network(state_t).squeeze().item()

        if self.continuous:
            output = self.policy_network(state_t)
            mean = output[:, : self.action_dim]
            log_std = output[:, self.action_dim :].clamp(-20, 2)
            std = log_std.exp()
            dist = TorchNormal(mean, std)
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t).sum(dim=-1).item()
            action = action_t.squeeze(0).cpu().detach().numpy()
        else:
            logits = self.policy_network(state_t)
            dist = TorchCategorical(logits=logits)
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t).item()
            action = action_t.item()

        return action, log_prob, value

    def _fisher_vector_product(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Fisher-vector product F(theta) * v without forming F.

        Uses the identity: Fv = nabla_theta (nabla_theta KL^T * v)
        where KL is the KL divergence of the policy w.r.t. itself (its
        gradient equals the score function expectation).
        """
        # Compute KL divergence of pi_theta with itself (= 0, but gradient != 0)
        if self.continuous:
            output = self.policy_network(states)
            mean = output[:, : self.action_dim]
            log_std = output[:, self.action_dim :].clamp(-20, 2)
            std = log_std.exp()
            dist = TorchNormal(mean, std)
            dist_detached = TorchNormal(mean.detach(), std.detach())
            kl = torch.distributions.kl_divergence(dist_detached, dist).sum(dim=-1).mean()
        else:
            logits = self.policy_network(states)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            probs_detached = probs.detach()
            kl = (probs_detached * (probs_detached.log() - log_probs)).sum(dim=-1).mean()

        # First gradient: nabla_theta KL
        grads = torch.autograd.grad(kl, self.policy_network.parameters(), create_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])

        # Inner product with vector
        grad_vector_product = (flat_grad * vector).sum()

        # Second gradient: nabla_theta (nabla_theta KL^T * v)
        hvp = torch.autograd.grad(grad_vector_product, self.policy_network.parameters())
        flat_hvp = torch.cat([g.reshape(-1).detach() for g in hvp])

        return flat_hvp + self.cg_damping * vector

    def _conjugate_gradient(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Solve F(theta) * x = b using conjugate gradient.

        This avoids explicitly forming the Fisher matrix (O(d^2) memory).
        CG requires only matrix-vector products, which we compute via
        ``_fisher_vector_product``.

        Parameters
        ----------
        states, actions : tensors for computing FVP
        b : torch.Tensor
            The policy gradient vector.

        Returns
        -------
        torch.Tensor
            The natural gradient direction x = F^{-1} * b.
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = r.dot(r)

        for _ in range(self.cg_iterations):
            fvp = self._fisher_vector_product(states, actions, p)
            alpha = rdotr / (p.dot(fvp) + 1e-8)
            x += alpha * p
            r -= alpha * fvp
            new_rdotr = r.dot(r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def update(
        self,
        transitions: list[tuple[np.ndarray, Any, float, np.ndarray | None, bool]],
    ) -> tuple[float, float]:
        """Update using natural policy gradient.

        Parameters
        ----------
        transitions : list of (state, action, reward, next_state, done)

        Returns
        -------
        (policy_loss, critic_loss)
        """
        if not transitions:
            return 0.0, 0.0

        states = np.array([t[0] for t in transitions])
        actions_raw = [t[1] for t in transitions]
        rewards = [t[2] for t in transitions]

        states_t = torch.FloatTensor(states).to(self.device)

        # Compute returns
        T = len(rewards)
        G = np.zeros(T, dtype=np.float64)
        running = 0.0
        for t in reversed(range(T)):
            if transitions[t][4]:  # done
                running = 0.0
            running = rewards[t] + self.gamma * running
            G[t] = running
        returns_t = torch.FloatTensor(G).to(self.device)

        # Critic update
        values = self.critic_network(states_t).squeeze(-1)
        critic_loss = F.mse_loss(values, returns_t)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Advantages
        with torch.no_grad():
            values_detached = self.critic_network(states_t).squeeze(-1)
        advantages = returns_t - values_detached
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute vanilla policy gradient
        log_probs_list = []
        for i in range(T):
            s_t = states_t[i].unsqueeze(0)
            if self.continuous:
                output = self.policy_network(s_t)
                mean = output[:, : self.action_dim]
                log_std = output[:, self.action_dim :].clamp(-20, 2)
                std = log_std.exp()
                dist = TorchNormal(mean, std)
                a_t = torch.FloatTensor(np.array(actions_raw[i])).unsqueeze(0).to(self.device)
                lp = dist.log_prob(a_t).sum(dim=-1)
            else:
                logits = self.policy_network(s_t)
                dist = TorchCategorical(logits=logits)
                lp = dist.log_prob(torch.tensor(int(actions_raw[i])).to(self.device))
            log_probs_list.append(lp)

        log_probs = torch.stack(log_probs_list).squeeze()
        policy_loss = -(log_probs * advantages).mean()

        # Compute vanilla gradient
        self.optimizer.zero_grad()
        policy_loss.backward()
        vanilla_grad = torch.cat(
            [p.grad.reshape(-1) for p in self.policy_network.parameters()]
        ).detach()

        # Prepare actions tensor for FVP
        if self.continuous:
            actions_t = torch.FloatTensor(np.array(actions_raw)).to(self.device)
        else:
            actions_t = torch.LongTensor([int(a) for a in actions_raw]).to(self.device)

        # Compute natural gradient via CG: x = F^{-1} * g
        natural_grad = self._conjugate_gradient(states_t, actions_t, vanilla_grad)

        # Apply natural gradient step
        offset = 0
        for param in self.policy_network.parameters():
            numel = param.numel()
            param.data += self.lr * natural_grad[offset : offset + numel].reshape(param.shape)
            offset += numel

        return policy_loss.item(), critic_loss.item()


# =====================================================================
# Deterministic Policy Gradient / DDPG (Ch 14.8)
# =====================================================================


class DeterministicPolicyGradient(PolicyGradientBase):
    """Deterministic Policy Gradient / DDPG (Ch 14.8, Silver et al. 2014).

    For CONTINUOUS action spaces, the policy is a deterministic mapping:
        mu(s; theta) : S -> A

    The Deterministic Policy Gradient Theorem (Silver et al. 2014):
        nabla_theta J(theta) = E_s [nabla_theta mu(s; theta) * nabla_a Q(s, a; w)|_{a=mu(s;theta)}]

    This is much simpler than the stochastic PG theorem — no need to
    integrate over actions.

    DDPG (Lillicrap et al. 2016) combines DPG with:
    1. **Experience replay** — off-policy learning from a buffer
    2. **Target networks** — soft-updated copies for stable TD targets
    3. **Ornstein-Uhlenbeck noise** — temporally correlated exploration

    Architecture:
        Actor:  mu(s; theta) -> deterministic action
        Critic: Q(s, a; w) -> action-value

    Updates:
        Critic: minimize L = E[(y - Q(s,a;w))^2]
                where y = r + gamma * Q(s', mu(s'; theta^-); w^-)
        Actor:  maximize J = E[Q(s, mu(s; theta); w)]

    Target networks are soft-updated:
        theta^- <- tau * theta + (1 - tau) * theta^-
        w^-     <- tau * w     + (1 - tau) * w^-

    Parameters
    ----------
    state_dim : int
    action_dim : int
    action_low : np.ndarray
        Lower bounds for each action dimension.
    action_high : np.ndarray
        Upper bounds for each action dimension.
    actor_hidden : sequence of int
    critic_hidden : sequence of int
    actor_lr : float
    critic_lr : float
    gamma : float
    tau : float
        Soft target update coefficient. Small values (0.001-0.01) for stability.
    noise_std : float
        Standard deviation of Gaussian exploration noise.
    buffer_size : int
        Replay buffer capacity.
    batch_size : int
    device : str
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        actor_hidden: Sequence[int] = (256, 128),
        critic_hidden: Sequence[int] = (256, 128),
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_std: float = 0.1,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        assert _HAS_TORCH, "PyTorch is required for DDPG"

        self.device = _resolve_device(device)
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = torch.FloatTensor(action_low).to(self.device)
        self.action_high = torch.FloatTensor(action_high).to(self.device)

        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # Actor: mu(s; theta) -> action
        self.policy_network = _build_mlp(
            state_dim, action_dim, actor_hidden, output_activation=nn.Tanh
        ).to(self.device)

        # Critic: Q(s, a; w) -> scalar
        # Takes concatenated [s, a] as input
        self.critic_network = _build_mlp(
            state_dim + action_dim, 1, critic_hidden
        ).to(self.device)

        # Target networks (initialized as copies)
        self.target_actor = copy.deepcopy(self.policy_network)
        self.target_critic = copy.deepcopy(self.critic_network)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=critic_lr)

        # Replay buffer
        self.buffer = _ReplayBuffer(buffer_size, self._rng)

    def _scale_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Scale tanh output [-1, 1] to [action_low, action_high]."""
        return self.action_low + (raw_action + 1.0) * 0.5 * (self.action_high - self.action_low)

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select action using the deterministic policy + optional exploration noise.

        Parameters
        ----------
        state : np.ndarray
        explore : bool
            If True, add Gaussian noise for exploration.

        Returns
        -------
        np.ndarray
            Action vector, clipped to [action_low, action_high].
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            raw = self.policy_network(state_t)
            action = self._scale_action(raw).squeeze(0)

        action_np = action.cpu().numpy()

        if explore:
            noise = self._rng.normal(0, self.noise_std, size=action_np.shape)
            action_np = action_np + noise

        # Clip to action bounds
        action_np = np.clip(
            action_np,
            self.action_low.cpu().numpy(),
            self.action_high.cpu().numpy(),
        )
        return action_np

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.buffer.add(state, action, reward, next_state, done)

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """Polyak averaging: target <- tau * source + (1 - tau) * target.

        This is the "soft" target update from Lillicrap et al. (2016).
        Small tau (e.g. 0.005) ensures slow-moving targets for stability.
        """
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def train_step(self) -> tuple[float, float]:
        """Perform one DDPG training step.

        Critic update:
            y = r + gamma * Q(s', mu(s'; theta^-); w^-)
            L_critic = (1/N) * sum (y - Q(s, a; w))^2

        Actor update:
            J_actor = (1/N) * sum Q(s, mu(s; theta); w)
            (maximize J_actor, i.e., minimize -J_actor)

        Returns
        -------
        (actor_loss, critic_loss)
        """
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # --- Critic update ---
        with torch.no_grad():
            next_actions_raw = self.target_actor(next_states_t)
            next_actions = self._scale_action(next_actions_raw)
            target_q = self.target_critic(
                torch.cat([next_states_t, next_actions], dim=1)
            )
            y = rewards_t + self.gamma * target_q * (1.0 - dones_t)

        current_q = self.critic_network(torch.cat([states_t, actions_t], dim=1))
        critic_loss = F.mse_loss(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        # --- Actor update ---
        # Maximize Q(s, mu(s; theta); w) => minimize -Q
        actor_actions_raw = self.policy_network(states_t)
        actor_actions = self._scale_action(actor_actions_raw)
        actor_loss = -self.critic_network(
            torch.cat([states_t, actor_actions], dim=1)
        ).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # --- Soft target updates ---
        self._soft_update(self.target_actor, self.policy_network)
        self._soft_update(self.target_critic, self.critic_network)

        return actor_loss.item(), critic_loss.item()

    def update(self, *args: Any, **kwargs: Any) -> Any:
        """Alias for train_step for interface compatibility."""
        return self.train_step()
