"""Offline Reinforcement Learning — CQL, IQL, and TD3+BC.

Offline (batch) RL learns policies from a fixed dataset of transitions
without any online interaction.  This is critical for trading systems
where exploration in live markets is expensive and dangerous.

Key challenge — distributional shift:
    Standard off-policy methods (DQN, DDPG) fail in the offline setting
    because the learned policy visits states/actions not covered by the
    dataset, leading to catastrophic overestimation of Q-values for
    out-of-distribution (OOD) actions.

Algorithms:
    - CQL  (Conservative Q-Learning, Kumar et al. 2020)
    - IQL  (Implicit Q-Learning, Kostrikov et al. 2022)
    - TD3+BC (Fujimoto & Gu, 2021)

References:
    Kumar et al. (2020) "Conservative Q-Learning for Offline RL", NeurIPS
    Kostrikov, Nair, Levine (2022) "Offline RL with Implicit Q-Learning", ICLR
    Fujimoto & Gu (2021) "A Minimalist Approach to Offline RL", NeurIPS
"""
from __future__ import annotations

import copy
import logging
from typing import Any, List, Optional, Sequence, Tuple, Union

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
    "CQL",
    "IQL",
    "TD3BC",
    "OfflineReplayBuffer",
]


# =====================================================================
# Device helper
# =====================================================================


def _resolve_device(device: str) -> "torch.device":
    assert _HAS_TORCH, "PyTorch is required for offline RL"
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
    activation: type["nn.Module"] = None,
    output_activation: type["nn.Module"] | None = None,
) -> "nn.Sequential":
    """Build a multi-layer perceptron."""
    assert _HAS_TORCH, "PyTorch required"
    if activation is None:
        activation = nn.ReLU
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
# Offline Replay Buffer
# =====================================================================


class OfflineReplayBuffer:
    """Replay buffer for offline RL.

    Accepts a fixed dataset of (s, a, r, s', done) transitions and
    provides mini-batch sampling.  No new transitions are added after
    initialization (offline setting).
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        seed: int = 42,
    ) -> None:
        self.states = np.asarray(states, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.rewards = np.asarray(rewards, dtype=np.float32)
        self.next_states = np.asarray(next_states, dtype=np.float32)
        self.dones = np.asarray(dones, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._size = len(self.states)

    @classmethod
    def from_transitions(
        cls,
        transitions: List[Tuple[np.ndarray, Any, float, np.ndarray, bool]],
        seed: int = 42,
    ) -> "OfflineReplayBuffer":
        """Build buffer from a list of (s, a, r, s', done) tuples."""
        states = np.array([t[0] for t in transitions], dtype=np.float32)
        actions = np.array([t[1] for t in transitions], dtype=np.float32)
        rewards = np.array([t[2] for t in transitions], dtype=np.float32)
        next_states = np.array([t[3] for t in transitions], dtype=np.float32)
        dones = np.array([t[4] for t in transitions], dtype=np.float32)
        return cls(states, actions, rewards, next_states, dones, seed=seed)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a random mini-batch."""
        indices = self._rng.choice(self._size, size=min(batch_size, self._size), replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self._size


# =====================================================================
# CQL — Conservative Q-Learning (Kumar et al. 2020)
# =====================================================================


class CQL:
    """Conservative Q-Learning for offline RL.

    CQL addresses the overestimation problem in offline RL by adding a
    conservative regularizer that pushes down Q-values for actions NOT
    in the dataset while pushing up Q-values for dataset actions.

    Loss function:
        L_CQL = alpha * [E_s[ log sum_a exp(Q(s,a)) / temp ] - E_{(s,a)~D}[ Q(s,a) ]]
                + L_TD

    The first term is the conservative penalty (soft-maximum of Q over
    all actions minus the expected Q under the data distribution).
    This ensures that Q-values are a lower bound on the true Q-function.

    Alpha auto-tuning (Lagrangian formulation):
        alpha is adjusted so that the conservative penalty equals a
        target threshold tau:
            alpha* = argmin_alpha  alpha * (penalty - tau)

    For continuous actions: Q(s,a) uses a critic that takes [s, a] as input.
    Random actions are sampled uniformly to estimate the logsumexp.

    For discrete actions: Q-network outputs Q-values for all actions directly.

    Parameters
    ----------
    state_dim : int
    action_dim : int
        Continuous: dimensionality of action space.
        Discrete: number of discrete actions.
    hidden_layers : sequence of int
    learning_rate : float
    gamma : float
    cql_alpha : float
        Initial conservative penalty weight.
    cql_temp : float
        Temperature for logsumexp computation.
    min_q_weight : float
        Multiplier for the conservative loss before combining with TD loss.
    num_random_actions : int
        Number of random actions to sample for logsumexp estimation (continuous only).
    tau_target : float
        Target threshold for alpha auto-tuning.  If None, alpha is fixed.
    alpha_lr : float
        Learning rate for alpha auto-tuning.
    discrete : bool
        If True, use discrete action space (DQN-style Q-network).
    target_update_freq : int
        Steps between hard target network updates.
    action_low, action_high : float
        Bounds for continuous action space sampling.
    device : str
    seed : int
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: Sequence[int] = (256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        cql_alpha: float = 1.0,
        cql_temp: float = 1.0,
        min_q_weight: float = 5.0,
        num_random_actions: int = 10,
        tau_target: float | None = None,
        alpha_lr: float = 3e-4,
        discrete: bool = False,
        target_update_freq: int = 100,
        action_low: float = -1.0,
        action_high: float = 1.0,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        assert _HAS_TORCH, "PyTorch is required for CQL"

        self.device = _resolve_device(device)
        self.gamma = gamma
        self.cql_temp = cql_temp
        self.min_q_weight = min_q_weight
        self.num_random_actions = num_random_actions
        self.target_update_freq = target_update_freq
        self.discrete = discrete
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

        torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        if discrete:
            # Q-network outputs Q(s,a) for all actions
            self.q_network = _build_mlp(state_dim, action_dim, hidden_layers).to(self.device)
            self.target_network = _build_mlp(state_dim, action_dim, hidden_layers).to(self.device)
        else:
            # Q-network takes [s, a] and outputs scalar Q-value
            self.q_network = _build_mlp(state_dim + action_dim, 1, hidden_layers).to(self.device)
            self.target_network = _build_mlp(
                state_dim + action_dim, 1, hidden_layers
            ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Alpha auto-tuning (Lagrangian)
        self.auto_alpha = tau_target is not None
        self.tau_target = tau_target if tau_target is not None else 0.0
        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(max(cql_alpha, 1e-8)), dtype=torch.float32,
                device=self.device, requires_grad=True,
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.cql_alpha = cql_alpha

        self._train_steps = 0

    @property
    def alpha(self) -> float:
        """Current CQL alpha value."""
        if self.auto_alpha:
            return self.log_alpha.exp().item()
        return self.cql_alpha

    def _compute_cql_penalty(
        self,
        states: "torch.Tensor",
        actions: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute CQL conservative penalty.

        penalty = E_s[ log sum_a exp(Q(s,a)/temp) ] - E_{(s,a)~D}[ Q(s,a) ]

        For discrete: logsumexp is computed exactly over all actions.
        For continuous: logsumexp is approximated by sampling random actions.
        """
        batch_size = states.shape[0]

        if self.discrete:
            # Exact logsumexp over all discrete actions
            q_all = self.q_network(states)  # (batch, action_dim)
            logsumexp = torch.logsumexp(q_all / self.cql_temp, dim=1) * self.cql_temp
            # Q-values for dataset actions
            actions_long = actions.long().squeeze(-1)
            q_data = q_all.gather(1, actions_long.unsqueeze(1)).squeeze(1)
        else:
            # Sample random actions for logsumexp estimation
            # Shape: (batch, num_random, action_dim)
            random_actions = torch.FloatTensor(
                batch_size, self.num_random_actions, self.action_dim
            ).uniform_(self.action_low, self.action_high).to(self.device)

            # Expand states for each random action: (batch, num_random, state_dim)
            states_expanded = states.unsqueeze(1).expand(-1, self.num_random_actions, -1)

            # Compute Q for random actions
            sa_random = torch.cat(
                [states_expanded, random_actions], dim=-1
            )  # (batch, num_random, state_dim + action_dim)
            q_random = self.q_network(
                sa_random.reshape(-1, self.state_dim + self.action_dim)
            ).reshape(batch_size, self.num_random_actions)

            # logsumexp over random actions
            logsumexp = torch.logsumexp(q_random / self.cql_temp, dim=1) * self.cql_temp

            # Q for dataset actions
            sa_data = torch.cat([states, actions], dim=-1)
            q_data = self.q_network(sa_data).squeeze(-1)

        # Conservative penalty: push down OOD Q, push up data Q
        penalty = (logsumexp - q_data).mean()
        return penalty

    def _compute_td_loss(
        self,
        states: "torch.Tensor",
        actions: "torch.Tensor",
        rewards: "torch.Tensor",
        next_states: "torch.Tensor",
        dones: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute standard Bellman TD loss."""
        if self.discrete:
            actions_long = actions.long().squeeze(-1)
            q_values = self.q_network(states).gather(1, actions_long.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = self.target_network(next_states).max(dim=1).values
                td_target = rewards + self.gamma * next_q * (1.0 - dones)
        else:
            sa = torch.cat([states, actions], dim=-1)
            q_values = self.q_network(sa).squeeze(-1)
            with torch.no_grad():
                # For continuous: use current policy (greedy from Q) for next actions
                # In offline setting, approximate with dataset-like actions
                # Use target network with same next_states and zero action as baseline
                # Better: sample random next actions and take the max
                random_next_actions = torch.FloatTensor(
                    next_states.shape[0], self.num_random_actions, self.action_dim
                ).uniform_(self.action_low, self.action_high).to(self.device)
                next_expanded = next_states.unsqueeze(1).expand(
                    -1, self.num_random_actions, -1
                )
                sa_next = torch.cat([next_expanded, random_next_actions], dim=-1)
                q_next_all = self.target_network(
                    sa_next.reshape(-1, self.state_dim + self.action_dim)
                ).reshape(next_states.shape[0], self.num_random_actions)
                next_q = q_next_all.max(dim=1).values
                td_target = rewards + self.gamma * next_q * (1.0 - dones)

        return F.mse_loss(q_values, td_target)

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> dict[str, float]:
        """Perform one CQL training step.

        L_CQL = alpha * min_q_weight * penalty + L_TD

        Returns
        -------
        dict with keys: td_loss, cql_penalty, total_loss, alpha
        """
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        if actions_t.dim() == 1:
            actions_t = actions_t.unsqueeze(-1)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # TD loss
        td_loss = self._compute_td_loss(states_t, actions_t, rewards_t, next_states_t, dones_t)

        # CQL conservative penalty
        cql_penalty = self._compute_cql_penalty(states_t, actions_t)

        # Get current alpha
        if self.auto_alpha:
            current_alpha = self.log_alpha.exp()
        else:
            current_alpha = self.cql_alpha

        # Total loss
        total_loss = td_loss + current_alpha * self.min_q_weight * cql_penalty

        # Update Q-network
        self.q_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.q_optimizer.step()

        # Alpha auto-tuning
        if self.auto_alpha:
            alpha_loss = self.log_alpha.exp() * (cql_penalty.detach() - self.tau_target)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Update target network
        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "td_loss": td_loss.item(),
            "cql_penalty": cql_penalty.item(),
            "total_loss": total_loss.item(),
            "alpha": self.alpha,
        }

    def train(
        self,
        dataset: OfflineReplayBuffer,
        n_epochs: int = 100,
        batch_size: int = 256,
        log_interval: int = 20,
    ) -> list[dict[str, float]]:
        """Train CQL on an offline dataset.

        Parameters
        ----------
        dataset : OfflineReplayBuffer
        n_epochs : int
        batch_size : int
        log_interval : int

        Returns
        -------
        list of per-step metrics dicts
        """
        history = []
        steps_per_epoch = max(1, len(dataset) // batch_size)

        for epoch in range(n_epochs):
            epoch_metrics: dict[str, list[float]] = {
                "td_loss": [], "cql_penalty": [], "total_loss": [], "alpha": [],
            }
            for _ in range(steps_per_epoch):
                s, a, r, ns, d = dataset.sample(batch_size)
                metrics = self.train_step(s, a, r, ns, d)
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

            avg = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
            history.append(avg)

            if (epoch + 1) % log_interval == 0:
                logger.info(
                    "CQL epoch %d/%d: td_loss=%.4f cql_penalty=%.4f alpha=%.4f",
                    epoch + 1, n_epochs, avg["td_loss"], avg["cql_penalty"], avg["alpha"],
                )

        return history

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict the best action for a given observation.

        For discrete: returns argmax action index.
        For continuous: returns the action that maximizes Q among random samples.

        Parameters
        ----------
        obs : np.ndarray
            Single observation of shape (state_dim,).

        Returns
        -------
        np.ndarray
            Selected action.
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            if self.discrete:
                q_values = self.q_network(state_t)
                action_idx = q_values.argmax(dim=1).item()
                return np.array([action_idx])
            else:
                # Sample candidate actions, pick the one with highest Q
                n_candidates = max(self.num_random_actions, 100)
                candidates = torch.FloatTensor(
                    n_candidates, self.action_dim
                ).uniform_(self.action_low, self.action_high).to(self.device)
                states_expanded = state_t.expand(n_candidates, -1)
                sa = torch.cat([states_expanded, candidates], dim=-1)
                q_values = self.q_network(sa).squeeze(-1)
                best_idx = q_values.argmax().item()
                return candidates[best_idx].cpu().numpy()

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions (discrete) or for the observation (continuous).

        Parameters
        ----------
        obs : np.ndarray
            Single observation.

        Returns
        -------
        np.ndarray
            For discrete: Q-values of shape (action_dim,).
            For continuous: not directly applicable, returns empty.
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if self.discrete:
                return self.q_network(state_t).squeeze(0).cpu().numpy()
            else:
                return np.array([])


# =====================================================================
# IQL — Implicit Q-Learning (Kostrikov et al. 2022)
# =====================================================================


class IQL:
    """Implicit Q-Learning for offline RL.

    IQL avoids querying out-of-distribution actions entirely by using
    expectile regression to estimate the value function.  The key insight:

    Instead of computing max_a Q(s,a) (which requires evaluating Q at
    potentially OOD actions), IQL uses expectile regression to
    *implicitly* approximate this maximum from the dataset.

    Three networks:
        Q(s, a; phi) — Q-function (trained via standard Bellman backup to V)
        V(s; psi)    — Value function (trained via expectile regression to Q)
        pi(a|s; theta) — Policy (trained via advantage-weighted regression)

    Value function loss (expectile regression):
        L_V = E_{(s,a)~D}[ L_tau(Q(s,a) - V(s)) ]
        where L_tau(u) = |tau - 1(u < 0)| * u^2

    With tau > 0.5 (e.g., 0.7), this approximates an upper expectile of
    Q(s,a), which serves as a proxy for max_a Q(s,a).

    Q-function loss (standard Bellman to V, NOT to max Q):
        L_Q = E_{(s,a,r,s')~D}[ (r + gamma * V(s') - Q(s,a))^2 ]

    Policy extraction (advantage-weighted regression):
        L_pi = E_{(s,a)~D}[ -exp(beta * A(s,a)) * log pi(a|s) ]
        where A(s,a) = Q(s,a) - V(s)

    The brilliance: at no point do we evaluate Q at actions not in the
    dataset.  V is trained from Q at dataset (s,a), Q is trained from V
    at dataset (s'), and the policy extracts advantage weights from the
    difference Q-V, both evaluated at dataset points.

    Parameters
    ----------
    state_dim : int
    action_dim : int
    hidden_layers : sequence of int
    learning_rate : float
    gamma : float
    expectile_tau : float
        Expectile parameter (> 0.5 biases toward max).
    beta : float
        Inverse temperature for advantage-weighted regression.
    temperature : float
        Temperature scaling for advantage weights.
    polyak_tau : float
        Soft target update coefficient for Q-network.
    action_low, action_high : float
        Bounds for continuous action space.
    device : str
    seed : int
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: Sequence[int] = (256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        expectile_tau: float = 0.7,
        beta: float = 3.0,
        temperature: float = 1.0,
        polyak_tau: float = 0.005,
        action_low: float = -1.0,
        action_high: float = 1.0,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        assert _HAS_TORCH, "PyTorch is required for IQL"

        self.device = _resolve_device(device)
        self.gamma = gamma
        self.expectile_tau = expectile_tau
        self.beta = beta
        self.temperature = temperature
        self.polyak_tau = polyak_tau
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

        torch.manual_seed(seed)

        # Q-function: Q(s, a) -> scalar
        self.q_network = _build_mlp(state_dim + action_dim, 1, hidden_layers).to(self.device)
        self.q_target = _build_mlp(state_dim + action_dim, 1, hidden_layers).to(self.device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Value function: V(s) -> scalar
        self.v_network = _build_mlp(state_dim, 1, hidden_layers).to(self.device)
        self.v_optimizer = optim.Adam(self.v_network.parameters(), lr=learning_rate)

        # Policy: pi(a|s) — Gaussian for continuous actions
        # Outputs mean and log_std
        self.policy_network = _build_mlp(
            state_dim, action_dim * 2, hidden_layers
        ).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self._train_steps = 0

    def _expectile_loss(
        self, diff: "torch.Tensor", tau: float
    ) -> "torch.Tensor":
        """Asymmetric (expectile) squared loss.

        L_tau(u) = |tau - 1(u < 0)| * u^2

        When tau > 0.5, the loss penalizes underestimation more than
        overestimation, biasing V(s) toward the upper tail of Q(s,a).

        Parameters
        ----------
        diff : torch.Tensor
            Q(s,a) - V(s) residuals.
        tau : float
            Expectile parameter in (0, 1).

        Returns
        -------
        torch.Tensor
            Scalar expectile loss.
        """
        weight = torch.where(diff > 0, tau, 1.0 - tau)
        return (weight * diff.pow(2)).mean()

    def _compute_v_loss(
        self,
        states: "torch.Tensor",
        actions: "torch.Tensor",
    ) -> "torch.Tensor":
        """Value function loss via expectile regression.

        L_V = E[ L_tau(Q(s,a) - V(s)) ]
        """
        with torch.no_grad():
            sa = torch.cat([states, actions], dim=-1)
            q_values = self.q_target(sa).squeeze(-1)
        v_values = self.v_network(states).squeeze(-1)
        diff = q_values - v_values
        return self._expectile_loss(diff, self.expectile_tau)

    def _compute_q_loss(
        self,
        states: "torch.Tensor",
        actions: "torch.Tensor",
        rewards: "torch.Tensor",
        next_states: "torch.Tensor",
        dones: "torch.Tensor",
    ) -> "torch.Tensor":
        """Q-function loss via Bellman backup to V (NOT max Q).

        L_Q = E[ (r + gamma * V(s') - Q(s,a))^2 ]

        Crucially, V(s') is used instead of max_a' Q(s', a'), which
        would require evaluating Q at potentially OOD actions.
        """
        sa = torch.cat([states, actions], dim=-1)
        q_values = self.q_network(sa).squeeze(-1)
        with torch.no_grad():
            next_v = self.v_network(next_states).squeeze(-1)
            td_target = rewards + self.gamma * next_v * (1.0 - dones)
        return F.mse_loss(q_values, td_target)

    def _compute_policy_loss(
        self,
        states: "torch.Tensor",
        actions: "torch.Tensor",
    ) -> "torch.Tensor":
        """Policy loss via advantage-weighted regression.

        L_pi = -E[ exp(beta * A(s,a) / temperature) * log pi(a|s) ]
        where A(s,a) = Q(s,a) - V(s)

        Actions with higher advantages get higher regression weights.
        """
        with torch.no_grad():
            sa = torch.cat([states, actions], dim=-1)
            q_values = self.q_target(sa).squeeze(-1)
            v_values = self.v_network(states).squeeze(-1)
            advantages = q_values - v_values

            # Clamp advantages for numerical stability
            weights = torch.exp(self.beta * advantages / self.temperature)
            weights = torch.clamp(weights, max=100.0)

        # Compute log probability of dataset actions under current policy
        policy_output = self.policy_network(states)
        mean = policy_output[:, : self.action_dim]
        log_std = policy_output[:, self.action_dim :].clamp(-5, 2)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)

        # Weighted negative log-likelihood
        return -(weights * log_prob).mean()

    def _soft_update_target(self) -> None:
        """Polyak averaging for Q target network."""
        for tp, sp in zip(self.q_target.parameters(), self.q_network.parameters()):
            tp.data.copy_(self.polyak_tau * sp.data + (1.0 - self.polyak_tau) * tp.data)

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> dict[str, float]:
        """Perform one IQL training step.

        Order of updates:
        1. V-function (expectile regression from Q-target)
        2. Q-function (Bellman backup to V)
        3. Policy (advantage-weighted regression)
        4. Soft update Q-target

        Returns
        -------
        dict with keys: v_loss, q_loss, policy_loss
        """
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # 1. Update V-function
        v_loss = self._compute_v_loss(states_t, actions_t)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v_network.parameters(), max_norm=10.0)
        self.v_optimizer.step()

        # 2. Update Q-function
        q_loss = self._compute_q_loss(states_t, actions_t, rewards_t, next_states_t, dones_t)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.q_optimizer.step()

        # 3. Update policy
        policy_loss = self._compute_policy_loss(states_t, actions_t)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)
        self.policy_optimizer.step()

        # 4. Soft update target
        self._soft_update_target()
        self._train_steps += 1

        return {
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
        }

    def train(
        self,
        dataset: OfflineReplayBuffer,
        n_epochs: int = 100,
        batch_size: int = 256,
        log_interval: int = 20,
    ) -> list[dict[str, float]]:
        """Train IQL on an offline dataset.

        Parameters
        ----------
        dataset : OfflineReplayBuffer
        n_epochs : int
        batch_size : int
        log_interval : int

        Returns
        -------
        list of per-epoch average metrics dicts
        """
        history = []
        steps_per_epoch = max(1, len(dataset) // batch_size)

        for epoch in range(n_epochs):
            epoch_metrics: dict[str, list[float]] = {
                "v_loss": [], "q_loss": [], "policy_loss": [],
            }
            for _ in range(steps_per_epoch):
                s, a, r, ns, d = dataset.sample(batch_size)
                metrics = self.train_step(s, a, r, ns, d)
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

            avg = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
            history.append(avg)

            if (epoch + 1) % log_interval == 0:
                logger.info(
                    "IQL epoch %d/%d: v_loss=%.4f q_loss=%.4f policy_loss=%.4f",
                    epoch + 1, n_epochs, avg["v_loss"], avg["q_loss"], avg["policy_loss"],
                )

        return history

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict action using the learned policy.

        Parameters
        ----------
        obs : np.ndarray
            Single observation of shape (state_dim,).

        Returns
        -------
        np.ndarray
            Predicted action (mean of Gaussian policy), clipped to bounds.
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            output = self.policy_network(state_t)
            mean = output[:, : self.action_dim]
            action = mean.squeeze(0).cpu().numpy()
        return np.clip(action, self.action_low, self.action_high)

    def get_value(self, obs: np.ndarray) -> float:
        """Get V(s) for a given observation.

        Parameters
        ----------
        obs : np.ndarray

        Returns
        -------
        float
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            return self.v_network(state_t).item()

    def get_q_value(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Get Q(s, a) for a given (observation, action) pair.

        Parameters
        ----------
        obs : np.ndarray
        action : np.ndarray

        Returns
        -------
        float
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            sa = torch.cat([state_t, action_t], dim=-1)
            return self.q_network(sa).item()


# =====================================================================
# TD3+BC — TD3 with Behavioral Cloning regularizer (Fujimoto & Gu 2021)
# =====================================================================


class TD3BC:
    """TD3+BC — A minimalist approach to offline RL.

    Adds a behavioral cloning (BC) regularizer to TD3 (Twin Delayed DDPG):

        pi_loss = -lambda * Q(s, pi(s)) + (1-lambda) * ||pi(s) - a_data||^2

    where lambda is normalized by the Q-value range:
        lambda = alpha / (|Q(s, pi(s))|.mean + alpha)

    This ensures the policy stays close to the data distribution while
    still maximizing Q-values.

    Twin critics and delayed policy updates from TD3:
        - Two Q-networks, take the minimum for target (clipped double Q)
        - Policy updated every ``policy_delay`` steps
        - Target policy smoothing (noise added to target actions)

    Parameters
    ----------
    state_dim : int
    action_dim : int
    action_low : float
    action_high : float
    hidden_layers : sequence of int
    actor_lr : float
    critic_lr : float
    gamma : float
    polyak_tau : float
    bc_alpha : float
        BC regularization strength.  Higher values weight BC more.
    policy_delay : int
        Update policy every this many critic updates (TD3).
    target_noise : float
        Std of noise added to target policy actions (TD3).
    noise_clip : float
        Range to clip target noise (TD3).
    device : str
    seed : int
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: float = -1.0,
        action_high: float = 1.0,
        hidden_layers: Sequence[int] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        polyak_tau: float = 0.005,
        bc_alpha: float = 2.5,
        policy_delay: int = 2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        assert _HAS_TORCH, "PyTorch is required for TD3+BC"

        self.device = _resolve_device(device)
        self.gamma = gamma
        self.polyak_tau = polyak_tau
        self.bc_alpha = bc_alpha
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

        torch.manual_seed(seed)

        # Actor: pi(s) -> action in [-1, 1], then scaled
        self.actor = _build_mlp(
            state_dim, action_dim, hidden_layers, output_activation=nn.Tanh
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Twin critics: Q1(s, a), Q2(s, a)
        self.critic1 = _build_mlp(state_dim + action_dim, 1, hidden_layers).to(self.device)
        self.critic2 = _build_mlp(state_dim + action_dim, 1, hidden_layers).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
        )

        self._train_steps = 0

    def _scale_action(self, tanh_action: "torch.Tensor") -> "torch.Tensor":
        """Scale from [-1, 1] to [action_low, action_high]."""
        return self.action_low + (tanh_action + 1.0) * 0.5 * (self.action_high - self.action_low)

    def _soft_update(self, target: "nn.Module", source: "nn.Module") -> None:
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.polyak_tau * sp.data + (1.0 - self.polyak_tau) * tp.data)

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> dict[str, float]:
        """Perform one TD3+BC training step.

        Returns
        -------
        dict with keys: critic_loss, actor_loss (0 if not updated this step)
        """
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # --- Critic update ---
        with torch.no_grad():
            # Target policy with smoothing noise (TD3)
            noise = (torch.randn_like(actions_t) * self.target_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            raw_next_actions = self.actor_target(next_states_t)
            next_actions = (raw_next_actions + noise).clamp(-1.0, 1.0)
            scaled_next = self._scale_action(next_actions)

            # Clipped double Q
            target_q1 = self.critic1_target(torch.cat([next_states_t, scaled_next], dim=1))
            target_q2 = self.critic2_target(torch.cat([next_states_t, scaled_next], dim=1))
            target_q = torch.min(target_q1, target_q2)
            td_target = rewards_t + self.gamma * target_q * (1.0 - dones_t)

        q1 = self.critic1(torch.cat([states_t, actions_t], dim=1))
        q2 = self.critic2(torch.cat([states_t, actions_t], dim=1))
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            max_norm=10.0,
        )
        self.critic_optimizer.step()

        self._train_steps += 1
        actor_loss_val = 0.0

        # --- Delayed actor update ---
        if self._train_steps % self.policy_delay == 0:
            # Actor output
            raw_actions = self.actor(states_t)
            scaled_actions = self._scale_action(raw_actions)

            # Q-value for actor's actions
            q_actor = self.critic1(torch.cat([states_t, scaled_actions], dim=1))

            # Normalized lambda: balances Q-maximization vs BC
            lam = self.bc_alpha / (q_actor.abs().mean().detach() + self.bc_alpha)

            # BC loss: MSE between actor output (scaled) and dataset actions
            bc_loss = F.mse_loss(scaled_actions, actions_t)

            # Actor loss: lambda * (-Q) + (1 - lambda) * BC
            actor_loss = -lam * q_actor.mean() + (1.0 - lam) * bc_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.actor_optimizer.step()

            actor_loss_val = actor_loss.item()

            # Soft update targets
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss_val,
        }

    def train(
        self,
        dataset: OfflineReplayBuffer,
        n_epochs: int = 100,
        batch_size: int = 256,
        log_interval: int = 20,
    ) -> list[dict[str, float]]:
        """Train TD3+BC on an offline dataset.

        Parameters
        ----------
        dataset : OfflineReplayBuffer
        n_epochs : int
        batch_size : int
        log_interval : int

        Returns
        -------
        list of per-epoch average metrics dicts
        """
        history = []
        steps_per_epoch = max(1, len(dataset) // batch_size)

        for epoch in range(n_epochs):
            epoch_metrics: dict[str, list[float]] = {
                "critic_loss": [], "actor_loss": [],
            }
            for _ in range(steps_per_epoch):
                s, a, r, ns, d = dataset.sample(batch_size)
                metrics = self.train_step(s, a, r, ns, d)
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

            avg = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
            history.append(avg)

            if (epoch + 1) % log_interval == 0:
                logger.info(
                    "TD3+BC epoch %d/%d: critic_loss=%.4f actor_loss=%.4f",
                    epoch + 1, n_epochs, avg["critic_loss"], avg["actor_loss"],
                )

        return history

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict the best action for a given observation.

        Parameters
        ----------
        obs : np.ndarray
            Single observation of shape (state_dim,).

        Returns
        -------
        np.ndarray
            Action, clipped to [action_low, action_high].
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            raw = self.actor(state_t)
            action = self._scale_action(raw).squeeze(0).cpu().numpy()
        return np.clip(action, self.action_low, self.action_high)
