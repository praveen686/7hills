"""Distributional RL — C51, QR-DQN, IQN, and Risk-Aware Trading.

Distributional RL learns the full return distribution Z(s,a) rather than
just the expected value Q(s,a) = E[Z(s,a)].  This enables risk-sensitive
decision-making: instead of argmax E[Q(s,a)], we can choose actions that
maximize CVaR, minimize VaR, or optimize any coherent risk measure.

Key algorithms:
- C51 / Categorical DQN (Bellemare, Dabney & Munos, 2017)
- QR-DQN / Quantile Regression DQN (Dabney et al., 2018a)
- IQN / Implicit Quantile Networks (Dabney et al., 2018b)
- RiskAwareTrader — risk-sensitive action selection via CVaR/VaR

References:
    Bellemare, Dabney, Munos (2017) "A Distributional Perspective on RL", ICML
    Dabney, Rowland, Bellemare, Munos (2018a) "Distributional RL with
        Quantile Regression", AAAI
    Dabney, Ostrovski, Silver, Munos (2018b) "Implicit Quantile Networks
        for Distributional RL", ICML
    Keramati et al. (2020) "Being Optimistic to Be Conservative in Safety"
"""
from __future__ import annotations

import copy
import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

logger = logging.getLogger(__name__)

__all__ = [
    "C51",
    "QRDQN",
    "IQN",
    "RiskAwareTrader",
]


# =====================================================================
# Replay Buffer (uniform, same interface as q_learning._ReplayBuffer)
# =====================================================================


class _ReplayBuffer:
    """Uniform random experience replay buffer."""

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self._buffer: deque[tuple] = deque(maxlen=capacity)
        self._rng = rng

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        indices = self._rng.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._buffer)


# =====================================================================
# Shared MLP builder
# =====================================================================


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layers: Sequence[int],
    activation: str = "relu",
) -> nn.Sequential:
    """Build a simple MLP with configurable activations."""
    assert _HAS_TORCH, "PyTorch is required for distributional RL"
    layers: list[nn.Module] = []
    prev = input_dim
    act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(act_fn())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


# =====================================================================
# Base class for all distributional agents
# =====================================================================


class DistributionalBase(ABC):
    """Abstract base for distributional RL agents.

    All distributional agents share:
    - Experience replay buffer
    - Target network with soft/hard updates
    - Epsilon-greedy exploration
    - .train() / .predict() / .get_distribution() interface
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_layers: Sequence[int] = (128, 64),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        tau: float = 1.0,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        assert _HAS_TORCH, "PyTorch is required for distributional RL"

        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_layers = hidden_layers
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau  # 1.0 = hard update, < 1.0 = soft (Polyak) update

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Replay buffer
        self.buffer = _ReplayBuffer(buffer_size, self._rng)

        # Counters
        self._train_steps = 0

    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        """Polyak averaging: target = tau * online + (1 - tau) * target."""
        for tp, op in zip(target.parameters(), online.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

    def sync_target(self) -> None:
        """Hard-copy online network parameters to target network."""
        self._soft_update(self.online_network, self.target_network)

    @property
    @abstractmethod
    def online_network(self) -> nn.Module:
        """Return the online network."""
        ...

    @property
    @abstractmethod
    def target_network(self) -> nn.Module:
        """Return the target network."""
        ...

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action selection based on expected Q-values."""
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.num_actions))
        q_values = self._compute_q_values(state)
        return int(np.argmax(q_values))

    def _compute_q_values(self, state: np.ndarray) -> np.ndarray:
        """Compute expected Q-values Q(s,a) = E[Z(s,a)] for all actions."""
        dist = self.get_distribution(state)
        # dist is dict: action -> distribution representation
        q = np.zeros(self.num_actions, dtype=np.float64)
        for a in range(self.num_actions):
            q[a] = self._expected_value(dist[a])
        return q

    @abstractmethod
    def _expected_value(self, dist_repr: Any) -> float:
        """Compute E[Z] from the distribution representation."""
        ...

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.add(state, action, reward, next_state, done)

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    @abstractmethod
    def train_step(self) -> float:
        """Perform one gradient step. Returns loss value."""
        ...

    @abstractmethod
    def get_distribution(self, state: np.ndarray) -> dict[int, Any]:
        """Get the full return distribution for each action.

        Returns
        -------
        dict[int, Any]
            Maps action index to its distribution representation.
            - C51: (atoms, probs) tuple
            - QR-DQN: (taus, quantile_values) tuple
            - IQN: (taus, quantile_values) tuple
        """
        ...

    def predict(self, obs: np.ndarray) -> int:
        """Greedy action selection (no exploration). Alias for select_action(greedy=True)."""
        return self.select_action(obs, greedy=True)

    def train(
        self,
        env: Any,
        n_steps: int,
        max_episode_steps: int = 1000,
    ) -> list[float]:
        """Train the agent on an environment.

        Parameters
        ----------
        env : object
            Must implement .reset() -> obs and .step(action) -> (obs, reward, done, info).
        n_steps : int
            Total number of environment steps to train for.
        max_episode_steps : int
            Maximum steps per episode.

        Returns
        -------
        list[float]
            Episode returns.
        """
        episode_returns: list[float] = []
        total_steps = 0
        while total_steps < n_steps:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]  # handle (obs, info) tuple
            ep_return = 0.0
            for _ in range(max_episode_steps):
                action = self.select_action(obs)
                result = env.step(action)
                if len(result) == 5:
                    next_obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                elif len(result) == 4:
                    next_obs, reward, done, info = result
                else:
                    raise ValueError(f"Unexpected env.step result length: {len(result)}")

                self.store_transition(obs, action, reward, next_obs, done)
                self.train_step()
                ep_return += reward
                total_steps += 1
                if done or total_steps >= n_steps:
                    break
                obs = next_obs
            self.decay_epsilon()
            episode_returns.append(ep_return)
        return episode_returns


# =====================================================================
# C51 — Categorical DQN (Bellemare, Dabney & Munos, 2017)
# =====================================================================


class C51Network(nn.Module):
    """Network that outputs categorical distributions over atoms for each action.

    Output shape: (batch, num_actions, n_atoms)
    Each [b, a, :] is a probability distribution (sums to 1 via softmax).
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        n_atoms: int,
        hidden_layers: Sequence[int],
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.n_atoms = n_atoms

        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.feature_net = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_actions * n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities: (batch, num_actions, n_atoms)."""
        features = self.feature_net(x)
        logits = self.head(features).view(-1, self.num_actions, self.n_atoms)
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs


class C51(DistributionalBase):
    """Categorical DQN (C51) — Bellemare, Dabney & Munos (2017).

    Learns the full return distribution Z(s,a) as a categorical distribution
    over a fixed set of N_atoms atoms (supports) linearly spaced in
    [V_min, V_max].

    The distributional Bellman equation:
        Z(s,a) = R + gamma * Z(s', argmax_{a'} E[Z(s',a')])

    is approximated via the PROJECTION step that maps the shifted atoms
    back onto the fixed support, distributing probability mass to neighboring
    atoms proportionally.

    Loss: Cross-entropy between projected target distribution and current
    distribution:
        L = -sum_i m_i * log p_i(s,a; theta)

    Parameters
    ----------
    state_dim : int
    num_actions : int
    n_atoms : int
        Number of atoms in the categorical support (default 51).
    v_min : float
        Minimum value of the support.
    v_max : float
        Maximum value of the support.
    hidden_layers : sequence of int
    learning_rate : float
    gamma : float
    epsilon_start, epsilon_end, epsilon_decay : float
    buffer_size, batch_size : int
    target_update_freq : int
    tau : float
        Soft update coefficient (1.0 = hard update).
    device : str
    seed : int
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        hidden_layers: Sequence[int] = (128, 64),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        tau: float = 1.0,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            tau=tau,
            device=device,
            seed=seed,
        )

        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Atoms: z_i = v_min + i * delta, i = 0, ..., n_atoms-1
        self.delta = (v_max - v_min) / (n_atoms - 1)
        self.atoms = torch.linspace(v_min, v_max, n_atoms, device=self.device)

        # Networks
        self._online = C51Network(state_dim, num_actions, n_atoms, hidden_layers).to(
            self.device
        )
        self._target = C51Network(state_dim, num_actions, n_atoms, hidden_layers).to(
            self.device
        )
        self._target.load_state_dict(self._online.state_dict())

        self.optimizer = optim.Adam(self._online.parameters(), lr=learning_rate)

    @property
    def online_network(self) -> nn.Module:
        return self._online

    @property
    def target_network(self) -> nn.Module:
        return self._target

    def _expected_value(self, dist_repr: tuple[np.ndarray, np.ndarray]) -> float:
        """E[Z] = sum_i z_i * p_i."""
        atoms, probs = dist_repr
        return float(np.dot(atoms, probs))

    def get_distribution(self, state: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Get categorical distribution for each action.

        Returns
        -------
        dict[int, tuple[np.ndarray, np.ndarray]]
            Maps action -> (atoms, probs) where atoms.shape = probs.shape = (n_atoms,).
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            log_probs = self._online(state_t)  # (1, num_actions, n_atoms)
            probs = log_probs.exp().squeeze(0).cpu().numpy()  # (num_actions, n_atoms)

        atoms_np = self.atoms.cpu().numpy()
        return {a: (atoms_np.copy(), probs[a]) for a in range(self.num_actions)}

    def train_step(self) -> float:
        """C51 distributional Bellman update with categorical projection.

        1. Compute next-state greedy actions using expected Q-values.
        2. Project the target distribution T_z = r + gamma * z onto the support.
        3. Minimize cross-entropy between projected target and current distribution.
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # --- Target distribution via projection ---
        with torch.no_grad():
            # Get target network distributions for next states
            next_log_probs = self._target(next_states_t)  # (B, A, N)
            next_probs = next_log_probs.exp()

            # Greedy actions from expected Q-values
            # Q(s',a') = sum_i z_i * p_i(s', a')
            next_q = (next_probs * self.atoms.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # (B, A)
            next_actions = next_q.argmax(dim=1)  # (B,)

            # Select distributions for greedy actions: (B, N)
            next_dist = next_probs[
                torch.arange(self.batch_size, device=self.device), next_actions
            ]

            # Projected atoms: T_z = r + gamma * z, clipped to [v_min, v_max]
            # atoms shape: (N,) -> broadcast (B, N)
            tz = (
                rewards_t.unsqueeze(1)
                + self.gamma * (1.0 - dones_t.unsqueeze(1)) * self.atoms.unsqueeze(0)
            )
            tz = tz.clamp(self.v_min, self.v_max)  # (B, N)

            # Projection onto fixed support
            b = (tz - self.v_min) / self.delta  # (B, N) in [0, N-1]
            lower = b.floor().long()
            upper = b.ceil().long()

            # Clamp to valid indices
            lower = lower.clamp(0, self.n_atoms - 1)
            upper = upper.clamp(0, self.n_atoms - 1)

            # Target distribution: distribute probability to lower and upper neighbors
            target_dist = torch.zeros(
                self.batch_size, self.n_atoms, device=self.device
            )

            # Upper portion to lower index, lower portion to upper index
            target_dist.scatter_add_(
                1, lower, next_dist * (upper.float() - b)
            )
            target_dist.scatter_add_(
                1, upper, next_dist * (b - lower.float())
            )

        # --- Current distribution ---
        log_probs = self._online(states_t)  # (B, A, N)
        # Select the log-probs for the actions that were taken
        log_probs_a = log_probs[
            torch.arange(self.batch_size, device=self.device), actions_t
        ]  # (B, N)

        # Cross-entropy loss: -sum_i m_i * log p_i
        loss = -(target_dist * log_probs_a).sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self._soft_update(self._online, self._target)

        return loss.item()


# =====================================================================
# QR-DQN — Quantile Regression DQN (Dabney et al., 2018a)
# =====================================================================


class QRDQNNetwork(nn.Module):
    """Network that outputs N quantile values for each action.

    Output shape: (batch, num_actions, n_quantiles)
    Each [b, a, i] is the quantile value Z_{tau_i}(s, a).
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        n_quantiles: int,
        hidden_layers: Sequence[int],
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.n_quantiles = n_quantiles

        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.feature_net = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_actions * n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns quantile values: (batch, num_actions, n_quantiles)."""
        features = self.feature_net(x)
        out = self.head(features).view(-1, self.num_actions, self.n_quantiles)
        return out


class QRDQN(DistributionalBase):
    """Quantile Regression DQN (Dabney et al., 2018a).

    Instead of approximating the distribution as a categorical, QR-DQN
    learns N fixed quantile levels tau_1, ..., tau_N of the return
    distribution. The quantile midpoints are:

        tau_i = (2i - 1) / (2N),  i = 1, ..., N

    The loss is the quantile Huber loss (asymmetric):

        rho^kappa_tau(u) = |tau - 1_{u<0}| * L_kappa(u)

    where L_kappa is the Huber loss with threshold kappa.

    No projection step is needed (unlike C51): the Bellman target
    quantiles are simply r + gamma * Z_{tau}(s', a*).

    Parameters
    ----------
    state_dim : int
    num_actions : int
    n_quantiles : int
        Number of quantile levels (default 32).
    kappa : float
        Huber loss threshold (default 1.0).
    hidden_layers : sequence of int
    learning_rate : float
    gamma : float
    epsilon_start, epsilon_end, epsilon_decay : float
    buffer_size, batch_size : int
    target_update_freq : int
    tau : float
        Soft update coefficient.
    device : str
    seed : int
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        n_quantiles: int = 32,
        kappa: float = 1.0,
        hidden_layers: Sequence[int] = (128, 64),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        tau: float = 1.0,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            tau=tau,
            device=device,
            seed=seed,
        )

        self.n_quantiles = n_quantiles
        self.kappa = kappa

        # Fixed quantile midpoints: tau_i = (2i - 1) / (2N), i = 1, ..., N
        self.taus = torch.tensor(
            [(2 * i - 1) / (2 * n_quantiles) for i in range(1, n_quantiles + 1)],
            dtype=torch.float32,
            device=self.device,
        )

        # Networks
        self._online = QRDQNNetwork(
            state_dim, num_actions, n_quantiles, hidden_layers
        ).to(self.device)
        self._target = QRDQNNetwork(
            state_dim, num_actions, n_quantiles, hidden_layers
        ).to(self.device)
        self._target.load_state_dict(self._online.state_dict())

        self.optimizer = optim.Adam(self._online.parameters(), lr=learning_rate)

    @property
    def online_network(self) -> nn.Module:
        return self._online

    @property
    def target_network(self) -> nn.Module:
        return self._target

    def _expected_value(self, dist_repr: tuple[np.ndarray, np.ndarray]) -> float:
        """E[Z] = (1/N) * sum_i Z_{tau_i} (equal-weight quantiles)."""
        taus, quantile_values = dist_repr
        return float(np.mean(quantile_values))

    def get_distribution(
        self, state: np.ndarray
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Get quantile distribution for each action.

        Returns
        -------
        dict[int, tuple[np.ndarray, np.ndarray]]
            Maps action -> (taus, quantile_values) where both have shape (n_quantiles,).
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            quantiles = self._online(state_t).squeeze(0).cpu().numpy()  # (A, N)

        taus_np = self.taus.cpu().numpy()
        return {a: (taus_np.copy(), quantiles[a]) for a in range(self.num_actions)}

    def _quantile_huber_loss(
        self,
        td_errors: torch.Tensor,
        taus: torch.Tensor,
        kappa: float,
    ) -> torch.Tensor:
        """Quantile Huber loss.

        rho^kappa_tau(u) = |tau - 1_{u<0}| * L_kappa(u)

        where L_kappa(u) = 0.5 * u^2 / kappa  if |u| <= kappa
                          = |u| - 0.5 * kappa  otherwise

        Parameters
        ----------
        td_errors : (B, N, N') tensor
            Pairwise TD errors between current and target quantiles.
        taus : (N,) tensor
            Current quantile levels.
        kappa : float
            Huber threshold.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        # Huber loss element: (B, N, N')
        huber = torch.where(
            td_errors.abs() <= kappa,
            0.5 * td_errors.pow(2) / kappa,
            td_errors.abs() - 0.5 * kappa,
        )

        # Asymmetric weighting: |tau - 1_{u<0}|
        # taus shape: (N,) -> (1, N, 1) for broadcasting
        indicator = (td_errors < 0).float()
        weight = (taus.unsqueeze(0).unsqueeze(2) - indicator).abs()

        loss = (weight * huber).sum(dim=2).mean(dim=1)  # mean over quantiles, then batch
        return loss.mean()

    def train_step(self) -> float:
        """QR-DQN training step with quantile Huber loss."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        B = self.batch_size
        N = self.n_quantiles

        # --- Target quantiles ---
        with torch.no_grad():
            # Get target quantiles for next states: (B, A, N)
            next_quantiles = self._target(next_states_t)

            # Greedy actions from expected Q-values (mean over quantiles)
            next_q = next_quantiles.mean(dim=2)  # (B, A)
            next_actions = next_q.argmax(dim=1)  # (B,)

            # Target quantiles for greedy actions: (B, N)
            target_quantiles = next_quantiles[
                torch.arange(B, device=self.device), next_actions
            ]

            # Bellman target: r + gamma * Z(s', a*)
            target_quantiles = (
                rewards_t.unsqueeze(1)
                + self.gamma * (1.0 - dones_t.unsqueeze(1)) * target_quantiles
            )

        # --- Current quantiles ---
        current_quantiles_all = self._online(states_t)  # (B, A, N)
        current_quantiles = current_quantiles_all[
            torch.arange(B, device=self.device), actions_t
        ]  # (B, N)

        # --- Pairwise TD errors: (B, N, N) ---
        # current_quantiles: (B, N) -> (B, N, 1)
        # target_quantiles: (B, N) -> (B, 1, N)
        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)

        loss = self._quantile_huber_loss(td_errors, self.taus, self.kappa)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self._soft_update(self._online, self._target)

        return loss.item()


# =====================================================================
# IQN — Implicit Quantile Networks (Dabney et al., 2018b)
# =====================================================================


class IQNNetwork(nn.Module):
    """Implicit Quantile Network.

    Instead of learning fixed quantile levels, IQN samples tau ~ Uniform(0,1)
    at each forward pass and uses a cosine embedding to encode tau:

        phi(tau) = ReLU(sum_{j=0}^{n-1} cos(pi * j * tau) * w_j + b)

    The quantile function is then:
        Z_tau(s, a) = f(psi(s) * phi(tau))

    where psi(s) is the state embedding and * denotes element-wise product.

    Parameters
    ----------
    state_dim : int
    num_actions : int
    n_cos_embeddings : int
        Dimension of cosine embedding (number of cosine basis functions).
    hidden_layers : sequence of int
        Hidden layer sizes for the state embedding network.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        n_cos_embeddings: int = 64,
        hidden_layers: Sequence[int] = (128, 64),
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.n_cos_embeddings = n_cos_embeddings

        # State embedding network: state -> d_embed
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.state_net = nn.Sequential(*layers)
        self.d_embed = prev  # dimension of state embedding

        # Cosine embedding: cos features -> d_embed
        self.cos_embedding = nn.Linear(n_cos_embeddings, self.d_embed)

        # Value head: d_embed -> num_actions
        self.value_head = nn.Linear(self.d_embed, num_actions)

        # Pre-compute cos basis indices: [0, 1, ..., n_cos_embeddings-1]
        self.register_buffer(
            "cos_indices",
            torch.arange(n_cos_embeddings, dtype=torch.float32).unsqueeze(0),
        )

    def forward(
        self, state: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with sampled quantile levels.

        Parameters
        ----------
        state : (B, state_dim)
        taus : (B, K) where K is number of quantile samples

        Returns
        -------
        torch.Tensor : (B, K, num_actions)
            Quantile values for each action at each sampled tau.
        """
        B, K = taus.shape

        # State embedding: (B, d_embed)
        state_embed = self.state_net(state)

        # Cosine embedding of tau: (B, K, n_cos)
        # taus: (B, K) -> (B, K, 1) * cos_indices: (1, 1, n_cos) -> (B, K, n_cos)
        cos_input = math.pi * taus.unsqueeze(2) * self.cos_indices.unsqueeze(0)
        cos_features = torch.cos(cos_input)  # (B, K, n_cos)

        # Embed cos features: (B, K, d_embed)
        tau_embed = F.relu(self.cos_embedding(cos_features))

        # Combine: element-wise product of state_embed and tau_embed
        # state_embed: (B, d_embed) -> (B, 1, d_embed)
        combined = state_embed.unsqueeze(1) * tau_embed  # (B, K, d_embed)

        # Value output: (B, K, num_actions)
        quantile_values = self.value_head(combined)

        return quantile_values


class IQN(DistributionalBase):
    """Implicit Quantile Networks (Dabney et al., 2018b).

    IQN samples tau ~ Uniform(0,1) at each forward pass and learns the
    FULL quantile function Z_tau(s,a) for any tau in [0,1], not just
    fixed quantile levels.

    Key advantages over QR-DQN:
    - Can evaluate any quantile, not just N fixed ones
    - Risk-sensitive policies via CVaR: sample tau only from [0, alpha]
    - More sample-efficient representation

    Risk-sensitive action selection:
        Instead of argmax E[Z(s,a)], sample tau ~ Uniform(0, alpha) and
        take argmax E_tau[Z_tau(s,a)].  This is CVaR_alpha action selection,
        which is pessimistic (focuses on worst-case outcomes).

    Parameters
    ----------
    state_dim : int
    num_actions : int
    n_cos_embeddings : int
        Dimension of cosine embedding (default 64).
    n_tau_samples : int
        Number of tau samples per forward pass during training (default 8).
    n_tau_prime_samples : int
        Number of tau' samples for target (default 8).
    risk_alpha : float
        CVaR alpha for risk-sensitive action selection (default 0.25).
        alpha=1.0 is risk-neutral (standard expected value).
    kappa : float
        Huber loss threshold for quantile regression (default 1.0).
    hidden_layers : sequence of int
    learning_rate : float
    gamma : float
    epsilon_start, epsilon_end, epsilon_decay : float
    buffer_size, batch_size : int
    target_update_freq : int
    tau : float
        Soft update coefficient (not quantile tau).
    device : str
    seed : int
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        n_cos_embeddings: int = 64,
        n_tau_samples: int = 8,
        n_tau_prime_samples: int = 8,
        risk_alpha: float = 0.25,
        kappa: float = 1.0,
        hidden_layers: Sequence[int] = (128, 64),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        tau: float = 1.0,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            tau=tau,
            device=device,
            seed=seed,
        )

        self.n_cos_embeddings = n_cos_embeddings
        self.n_tau_samples = n_tau_samples
        self.n_tau_prime_samples = n_tau_prime_samples
        self.risk_alpha = risk_alpha
        self.kappa = kappa

        # Networks
        self._online = IQNNetwork(
            state_dim, num_actions, n_cos_embeddings, hidden_layers
        ).to(self.device)
        self._target = IQNNetwork(
            state_dim, num_actions, n_cos_embeddings, hidden_layers
        ).to(self.device)
        self._target.load_state_dict(self._online.state_dict())

        self.optimizer = optim.Adam(self._online.parameters(), lr=learning_rate)

    @property
    def online_network(self) -> nn.Module:
        return self._online

    @property
    def target_network(self) -> nn.Module:
        return self._target

    def _expected_value(self, dist_repr: tuple[np.ndarray, np.ndarray]) -> float:
        """E[Z] = mean of sampled quantile values."""
        taus, quantile_values = dist_repr
        return float(np.mean(quantile_values))

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Risk-sensitive action selection using CVaR.

        Samples tau ~ Uniform(0, risk_alpha) and selects the action
        with the highest average quantile value over those taus.
        This implements CVaR_alpha action selection.
        """
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.num_actions))

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # Sample taus from [0, risk_alpha] for CVaR
            K = 32  # number of tau samples for action selection
            taus = torch.rand(1, K, device=self.device) * self.risk_alpha
            quantile_values = self._online(state_t, taus)  # (1, K, A)
            # Average over quantile samples to get risk-sensitive Q-values
            q_cvar = quantile_values.mean(dim=1)  # (1, A)
            return int(q_cvar.argmax(dim=1).item())

    def get_distribution(
        self, state: np.ndarray, n_samples: int = 64
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Get quantile distribution for each action by sampling many taus.

        Parameters
        ----------
        state : np.ndarray
        n_samples : int
            Number of tau samples (default 64 for a smooth approximation).

        Returns
        -------
        dict[int, tuple[np.ndarray, np.ndarray]]
            Maps action -> (taus, quantile_values), both shape (n_samples,).
            taus are sorted for interpretability.
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            taus = torch.linspace(0.01, 0.99, n_samples, device=self.device).unsqueeze(
                0
            )  # (1, n_samples)
            quantile_values = self._online(state_t, taus)  # (1, n_samples, A)
            quantile_values = quantile_values.squeeze(0).cpu().numpy()  # (n_samples, A)

        taus_np = taus.squeeze(0).cpu().numpy()
        return {
            a: (taus_np.copy(), quantile_values[:, a])
            for a in range(self.num_actions)
        }

    def train_step(self) -> float:
        """IQN training step with sampled quantiles and quantile Huber loss."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        B = self.batch_size
        K = self.n_tau_samples
        K_prime = self.n_tau_prime_samples

        # --- Sample taus for current and target ---
        taus = torch.rand(B, K, device=self.device)  # (B, K)
        taus_prime = torch.rand(B, K_prime, device=self.device)  # (B, K')

        # --- Target quantiles ---
        with torch.no_grad():
            # Get quantile values for next states with taus_prime: (B, K', A)
            next_quantiles = self._target(next_states_t, taus_prime)

            # Greedy actions from risk-neutral expected Q-values
            # Use more taus for action selection (risk-neutral: full [0,1])
            eval_taus = torch.rand(B, 32, device=self.device)
            next_q_eval = self._online(next_states_t, eval_taus).mean(dim=1)  # (B, A)
            next_actions = next_q_eval.argmax(dim=1)  # (B,)

            # Target quantiles for greedy actions: (B, K')
            target_quantiles = next_quantiles[
                torch.arange(B, device=self.device), :, next_actions
            ]  # (B, K')

            # Bellman target
            target_quantiles = (
                rewards_t.unsqueeze(1)
                + self.gamma * (1.0 - dones_t.unsqueeze(1)) * target_quantiles
            )

        # --- Current quantiles ---
        current_quantiles = self._online(states_t, taus)  # (B, K, A)
        current_quantiles = current_quantiles[
            torch.arange(B, device=self.device), :, actions_t
        ]  # (B, K)

        # --- Pairwise TD errors: (B, K, K') ---
        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)

        # Quantile Huber loss
        huber = torch.where(
            td_errors.abs() <= self.kappa,
            0.5 * td_errors.pow(2) / self.kappa,
            td_errors.abs() - 0.5 * self.kappa,
        )

        # Asymmetric weighting: |tau - 1_{u<0}|
        indicator = (td_errors < 0).float()
        weight = (taus.unsqueeze(2) - indicator).abs()

        loss = (weight * huber).sum(dim=2).mean(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self._soft_update(self._online, self._target)

        return loss.item()


# =====================================================================
# RiskAwareTrader — Risk-sensitive trading with distributional RL
# =====================================================================


class RiskAwareTrader:
    """Use distributional RL for risk-sensitive trading.

    Instead of argmax E[Q(s,a)], uses argmax CVaR_alpha[Z(s,a)]
    for downside-risk-aware action selection.

    Supports any distributional agent (C51, QR-DQN, IQN).

    Risk measures available:
    - 'mean': standard expected value (risk-neutral)
    - 'cvar': Conditional Value-at-Risk (Expected Shortfall)
    - 'var': Value-at-Risk
    - 'wang': Wang transform (distortion risk measure)

    Parameters
    ----------
    distributional_agent : DistributionalBase
        A trained distributional RL agent (C51, QRDQN, or IQN).
    risk_measure : str
        Risk measure to use for action selection ('mean', 'cvar', 'var', 'wang').
    alpha : float
        Risk level (default 0.25). For CVaR/VaR, this is the tail probability.
        Lower alpha = more conservative.
    """

    def __init__(
        self,
        distributional_agent: DistributionalBase,
        risk_measure: str = "cvar",
        alpha: float = 0.25,
    ) -> None:
        self.agent = distributional_agent
        self.risk_measure = risk_measure.lower()
        self.alpha = alpha

        if self.risk_measure not in ("mean", "cvar", "var", "wang"):
            raise ValueError(
                f"Unknown risk measure: {risk_measure}. "
                f"Choose from: mean, cvar, var, wang"
            )

    def select_action(self, state: np.ndarray) -> int:
        """Risk-sensitive action selection.

        Computes the chosen risk measure for each action and selects
        the action with the highest risk-adjusted value.
        """
        dist = self.agent.get_distribution(state)
        risk_values = np.zeros(self.agent.num_actions, dtype=np.float64)

        for a in range(self.agent.num_actions):
            if self.risk_measure == "mean":
                risk_values[a] = self._compute_mean(dist[a])
            elif self.risk_measure == "cvar":
                risk_values[a] = self.compute_cvar(state, a, self.alpha)
            elif self.risk_measure == "var":
                risk_values[a] = self.compute_var(state, a, self.alpha)
            elif self.risk_measure == "wang":
                risk_values[a] = self._compute_wang(dist[a])

        return int(np.argmax(risk_values))

    def _compute_mean(self, dist_repr: tuple[np.ndarray, np.ndarray]) -> float:
        """Expected value from the distribution."""
        return self.agent._expected_value(dist_repr)

    def compute_var(
        self, state: np.ndarray, action: int, alpha: float = 0.05
    ) -> float:
        """Value-at-Risk from the return distribution.

        VaR_alpha = inf{z : P(Z <= z) >= alpha}
        i.e., the alpha-quantile of the distribution.

        Parameters
        ----------
        state : np.ndarray
        action : int
        alpha : float
            Probability level (e.g., 0.05 for 5th percentile).

        Returns
        -------
        float
            The VaR at level alpha.
        """
        dist = self.agent.get_distribution(state)
        taus_or_atoms, values = dist[action]

        if isinstance(self.agent, C51):
            # C51: atoms and probs — compute quantile from CDF
            atoms, probs = taus_or_atoms, values
            cum_probs = np.cumsum(probs)
            idx = np.searchsorted(cum_probs, alpha)
            idx = min(idx, len(atoms) - 1)
            return float(atoms[idx])
        else:
            # QR-DQN or IQN: quantile values represent an empirical distribution.
            # Sort the values to form a proper quantile function, since the
            # network does not enforce monotonicity.
            _, quantile_values = taus_or_atoms, values
            sorted_vals = np.sort(quantile_values)
            N = len(sorted_vals)
            # Index into sorted values: alpha maps to position alpha * N
            idx = int(np.clip(np.floor(alpha * N), 0, N - 1))
            return float(sorted_vals[idx])

    def compute_cvar(
        self, state: np.ndarray, action: int, alpha: float = 0.05
    ) -> float:
        """Conditional Value-at-Risk (Expected Shortfall) from the return distribution.

        CVaR_alpha = E[Z | Z <= VaR_alpha]
        = (1/alpha) * integral_0^alpha F^{-1}(u) du

        CVaR is a coherent risk measure and is always <= E[Z] (more conservative).

        Parameters
        ----------
        state : np.ndarray
        action : int
        alpha : float
            Probability level (e.g., 0.05 for 5% CVaR).

        Returns
        -------
        float
            The CVaR at level alpha.
        """
        dist = self.agent.get_distribution(state)
        taus_or_atoms, values = dist[action]

        if isinstance(self.agent, C51):
            # C51: weighted average of atoms below VaR
            atoms, probs = taus_or_atoms, values
            cum_probs = np.cumsum(probs)
            # Average atoms weighted by probability, only in the alpha tail
            mask = cum_probs <= alpha
            if not mask.any():
                # All probability is above alpha — return the minimum atom
                return float(atoms[0])
            # Include partial weight for the boundary atom
            tail_probs = probs.copy()
            tail_probs[~mask] = 0.0
            # Adjust boundary: the last included atom may only partially belong
            boundary_idx = np.searchsorted(cum_probs, alpha)
            if boundary_idx < len(atoms):
                remaining = alpha - (cum_probs[boundary_idx - 1] if boundary_idx > 0 else 0.0)
                tail_probs[boundary_idx] = remaining
            total_weight = tail_probs.sum()
            if total_weight < 1e-12:
                return float(atoms[0])
            return float(np.dot(atoms, tail_probs) / total_weight)
        else:
            # QR-DQN or IQN: sort quantile values to form a proper empirical
            # distribution, then average the bottom alpha fraction.
            # The network does not enforce monotonicity, so we must sort.
            _, quantile_values = taus_or_atoms, values
            sorted_vals = np.sort(quantile_values)
            N = len(sorted_vals)
            # Number of quantiles in the alpha tail
            n_tail = max(1, int(np.ceil(alpha * N)))
            return float(np.mean(sorted_vals[:n_tail]))

    def _compute_wang(self, dist_repr: tuple[np.ndarray, np.ndarray]) -> float:
        """Wang transform risk measure (pessimistic distortion).

        Distorts the quantile levels by Phi^{-1}(tau) - eta, where
        eta > 0 makes the agent more conservative.
        """
        from scipy.stats import norm

        taus_or_atoms, values = dist_repr
        eta = 0.5  # conservative distortion

        if len(taus_or_atoms) == len(values):
            # Quantile-based: distort the taus
            taus = taus_or_atoms
            distorted_taus = norm.cdf(norm.ppf(taus) - eta)
            # Weight by distorted CDF differences
            weights = np.diff(np.concatenate([[0], distorted_taus]))
            weights = np.maximum(weights, 0)
            total = weights.sum()
            if total < 1e-12:
                return float(np.mean(values))
            return float(np.dot(values, weights) / total)

        return float(np.mean(values))
