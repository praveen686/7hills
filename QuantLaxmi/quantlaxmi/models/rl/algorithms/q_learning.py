"""Q-Learning, DQN, Double DQN, and LSPI.

Implements Chapter 13 of "Foundations of RL with Applications in Finance"
by Ashwin Rao & Tikhon Jelvis.

Q-Learning is the foundational off-policy TD control algorithm.  Its deep
learning extension (DQN) demonstrated that RL agents can learn from raw
high-dimensional inputs, sparking the modern deep RL revolution.

Key algorithms:
- Tabular Q-Learning (Section 13.1)
- Deep Q-Network / DQN (Section 13.2, Mnih et al. 2015)
- Double DQN (van Hasselt et al. 2016)
- Least-Squares Policy Iteration / LSPI (Section 13.3, Lagoudakis & Parr 2003)

References:
    Rao & Jelvis, Ch 13 (Batch RL and Fitted Value Iteration)
    Mnih et al. (2015) "Human-level control through deep RL", Nature
    van Hasselt, Guez, Silver (2016) "Deep RL with Double Q-learning"
    Lagoudakis & Parr (2003) "Least-Squares Policy Iteration", JMLR
"""
from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Iterable,
    Optional,
    Sequence,
    TypeVar,
)

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ---------------------------------------------------------------------------
# Core interface imports
# ---------------------------------------------------------------------------
try:
    from quantlaxmi.models.rl.core.markov_process import (
        MarkovDecisionProcess,
        DeterministicPolicy,
        Distribution,
    )
    from quantlaxmi.models.rl.core.function_approx import (
        FunctionApprox,
        Tabular,
        LinearFunctionApprox,
    )
except ImportError:
    pass

logger = logging.getLogger(__name__)

S = TypeVar("S", bound=Hashable)
A = TypeVar("A", bound=Hashable)

__all__ = [
    "q_learning",
    "DQN",
    "DoubleDQN",
    "lspi",
]

# =====================================================================
# Minimal tabular Q-function (fallback)
# =====================================================================


class _TabularQ(Generic[S, A]):
    """Tabular Q-function with constant step-size TD updates."""

    def __init__(self, default_value: float = 0.0) -> None:
        self._values: dict[tuple[S, A], float] = defaultdict(lambda: default_value)

    def __call__(self, key: tuple[S, A]) -> float:
        return self._values[key]

    @property
    def values(self) -> dict[tuple[S, A], float]:
        return dict(self._values)


def _eps_greedy(
    Q: _TabularQ,
    state: S,
    actions: Sequence[A],
    epsilon: float,
    rng: np.random.Generator,
) -> A:
    """Epsilon-greedy action selection."""
    if rng.random() < epsilon:
        return actions[rng.integers(len(actions))]
    q_vals = np.array([Q((state, a)) for a in actions], dtype=np.float64)
    max_q = q_vals.max()
    best = [a for a, qv in zip(actions, q_vals) if np.isclose(qv, max_q)]
    return rng.choice(best)  # type: ignore[return-value]


# =====================================================================
# Tabular Q-Learning (Ch 13.1)
# =====================================================================


def q_learning(
    mdp_step: Callable[[S, A], tuple[S, float, bool]],
    start_states: Sequence[S] | Callable[[], S],
    actions: Sequence[A],
    gamma: float,
    approx: Any | None = None,
    num_episodes: int = 10_000,
    learning_rate: float = 0.01,
    epsilon: float = 0.1,
    epsilon_decay: float = 0.999,
    epsilon_min: float = 0.01,
    seed: int = 42,
    max_steps: int = 10_000,
) -> tuple[Any, Callable[[S], A]]:
    """Q-Learning — Off-policy TD Control (Ch 13.1).

    The Q-Learning update rule:
        Q(S_t, A_t) <- Q(S_t, A_t) + alpha * [R_{t+1} + gamma * max_a Q(S_{t+1}, a) - Q(S_t, A_t)]

    Key insight (Watkins, 1989):
        The target uses max_a Q(S_{t+1}, a), which is the value under the
        GREEDY policy, regardless of what action the agent actually takes
        (epsilon-greedy for exploration).  This makes Q-Learning OFF-policy:
        the behavior policy (epsilon-greedy) differs from the target policy
        (greedy).

    Theorem 13.1 (Rao & Jelvis / Watkins & Dayan 1992):
        Q-Learning converges to Q* with probability 1 provided:
        1. All (s, a) pairs are visited infinitely often.
        2. The step sizes satisfy Robbins-Monro conditions.

    Maximization bias (Ch 13.1):
        Q-Learning can overestimate Q-values because the max operator is
        applied to noisy estimates.  This is addressed by Double Q-Learning.

    Parameters
    ----------
    mdp_step : callable (state, action) -> (next_state, reward, done)
    start_states : sequence or callable
    actions : sequence of A
    gamma : float
    approx : Q-function approximation or None
    num_episodes : int
    learning_rate : float
    epsilon : float
        Initial exploration rate.
    epsilon_decay : float
        Multiplicative decay per episode.
    epsilon_min : float
        Minimum epsilon.
    seed : int
    max_steps : int

    Returns
    -------
    (q_approx, greedy_policy)
        The learned Q-function and the derived greedy (deterministic) policy.
    """
    rng = np.random.default_rng(seed)
    Q = _TabularQ() if approx is None else approx
    eps = epsilon

    for ep in range(num_episodes):
        if callable(start_states) and not isinstance(start_states, (list, tuple)):
            state = start_states()
        else:
            state = start_states[rng.integers(len(start_states))]  # type: ignore[index]

        for _ in range(max_steps):
            action = _eps_greedy(Q, state, actions, eps, rng)
            next_state, reward, done = mdp_step(state, action)

            if done:
                td_target = reward
            else:
                # Off-policy: use max over next actions
                max_q_next = max(Q((next_state, a)) for a in actions)
                td_target = reward + gamma * max_q_next

            Q._values[(state, action)] += learning_rate * (
                td_target - Q((state, action))
            )

            if done:
                break
            state = next_state

        eps = max(epsilon_min, eps * epsilon_decay)

    def greedy_policy(state: S) -> A:
        q_vals = np.array([Q((state, a)) for a in actions], dtype=np.float64)
        return actions[int(np.argmax(q_vals))]

    return Q, greedy_policy


# =====================================================================
# Replay Buffer (shared by DQN, DoubleDQN)
# =====================================================================


class _ReplayBuffer:
    """Uniform random experience replay buffer.

    Experience replay (Lin, 1992):
        Stores transitions and samples mini-batches uniformly at random,
        breaking temporal correlations and improving data efficiency.
    """

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
# DQN — Deep Q-Network (Ch 13.2, Mnih et al. 2015)
# =====================================================================


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layers: Sequence[int],
    activation: str = "relu",
) -> nn.Sequential:
    """Build a simple MLP with ReLU activations."""
    assert _HAS_TORCH, "PyTorch is required for DQN"
    layers: list[nn.Module] = []
    prev = input_dim
    act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(act_fn())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class DQN:
    """Deep Q-Network (Ch 13.2, Mnih et al. 2015).

    The three innovations that made DQN work on Atari:

    1. **Experience replay buffer**: Stores transitions and samples
       mini-batches uniformly at random, breaking temporal correlations
       and improving data efficiency.

    2. **Target network** (frozen copy, synced every C steps):
       Uses a separate network theta^- for computing TD targets,
       reducing oscillation and divergence.

    3. **Epsilon-greedy exploration with decay**:
       Balances exploration and exploitation.

    Loss function:
        L(theta) = E[(r + gamma * max_{a'} Q(s', a'; theta^-) - Q(s, a; theta))^2]

    where theta^- is the target network parameters (frozen copy of theta,
    updated every ``target_update_freq`` steps).

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state space.
    num_actions : int
        Number of discrete actions.
    hidden_layers : sequence of int
        Hidden layer sizes for the Q-network.
    learning_rate : float
        Optimizer learning rate.
    gamma : float
        Discount factor.
    epsilon_start : float
        Initial exploration rate.
    epsilon_end : float
        Final exploration rate.
    epsilon_decay : float
        Multiplicative decay per episode.
    buffer_size : int
        Replay buffer capacity.
    batch_size : int
        Mini-batch size for SGD.
    target_update_freq : int
        Number of training steps between target network syncs.
    device : str
        "auto", "cpu", or "cuda".
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
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        assert _HAS_TORCH, "PyTorch is required for DQN"

        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Online and target networks
        self.q_network = _build_mlp(state_dim, num_actions, hidden_layers).to(self.device)
        self.target_network = _build_mlp(state_dim, num_actions, hidden_layers).to(self.device)
        self.sync_target()  # initialize target = online

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.buffer = _ReplayBuffer(buffer_size, self._rng)

        # Counters
        self._train_steps = 0

    def sync_target(self) -> None:
        """Copy online network parameters to target network (hard update).

        This is the "frozen target network" technique from Mnih et al. (2015).
        By using stale parameters for the TD target, we reduce the moving-target
        problem that causes instability in naive DQN.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Select action using epsilon-greedy exploration.

        Parameters
        ----------
        state : np.ndarray
            Current state.
        greedy : bool
            If True, always select the greedy action (no exploration).

        Returns
        -------
        int
            Selected action index.
        """
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.num_actions))

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=1).item())

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

    def train_step(self) -> float:
        """Perform one gradient step on a mini-batch from the replay buffer.

        DQN loss (Mnih et al. 2015):
            L = (1/N) * sum_i (y_i - Q(s_i, a_i; theta))^2
            y_i = r_i + gamma * max_{a'} Q(s'_i, a'; theta^-)   (if not terminal)
            y_i = r_i                                             (if terminal)

        Returns
        -------
        float
            The batch loss value.
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a; theta) for the actions actually taken
        q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values using target network (frozen)
        with torch.no_grad():
            target_q = self._compute_target_q(next_states_t, dones_t, rewards_t)

        loss = F.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.sync_target()

        return loss.item()

    def _compute_target_q(
        self,
        next_states_t: torch.Tensor,
        dones_t: torch.Tensor,
        rewards_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TD target: r + gamma * max_a' Q(s', a'; theta^-).

        Overridden in DoubleDQN to decouple selection and evaluation.
        """
        next_q = self.target_network(next_states_t).max(dim=1).values
        return rewards_t + self.gamma * next_q * (1.0 - dones_t)

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_episode(
        self,
        env_step: Callable[[np.ndarray, int], tuple[np.ndarray, float, bool]],
        initial_state: np.ndarray,
        max_steps: int = 1000,
    ) -> tuple[float, int]:
        """Run one episode of training.

        Parameters
        ----------
        env_step : callable (state, action) -> (next_state, reward, done)
        initial_state : np.ndarray
        max_steps : int

        Returns
        -------
        (total_reward, num_steps)
        """
        state = initial_state
        total_reward = 0.0
        for step in range(max_steps):
            action = self.select_action(state)
            next_state, reward, done = env_step(state, action)
            self.store_transition(state, action, reward, next_state, done)
            self.train_step()
            total_reward += reward
            if done:
                break
            state = next_state
        self.decay_epsilon()
        return total_reward, step + 1


# =====================================================================
# Double DQN (van Hasselt et al. 2016)
# =====================================================================


class DoubleDQN(DQN):
    """Double DQN — addresses maximization bias in DQN.

    Standard DQN computes:
        y = r + gamma * max_{a'} Q(s', a'; theta^-)

    The max operator uses the same network (theta^-) to both SELECT and
    EVALUATE the next action, introducing an upward bias:
        E[max_a Q(s,a)] >= max_a E[Q(s,a)]

    Double DQN decouples selection and evaluation:
        a* = argmax_{a'} Q(s', a'; theta)          (online network SELECTS)
        y  = r + gamma * Q(s', a*; theta^-)         (target network EVALUATES)

    This simple change significantly reduces overestimation.

    Reference:
        van Hasselt, Guez, Silver (2016) "Deep RL with Double Q-learning"
    """

    def _compute_target_q(
        self,
        next_states_t: torch.Tensor,
        dones_t: torch.Tensor,
        rewards_t: torch.Tensor,
    ) -> torch.Tensor:
        """Double DQN target: decouple action selection from evaluation.

        a* = argmax_{a'} Q(s', a'; theta)       [online selects]
        y  = r + gamma * Q(s', a*; theta^-)      [target evaluates]
        """
        # Online network selects the best action
        with torch.no_grad():
            online_q_next = self.q_network(next_states_t)
            best_actions = online_q_next.argmax(dim=1, keepdim=True)

            # Target network evaluates the selected action
            target_q_next = self.target_network(next_states_t)
            next_q = target_q_next.gather(1, best_actions).squeeze(1)

        return rewards_t + self.gamma * next_q * (1.0 - dones_t)


# =====================================================================
# LSPI — Least-Squares Policy Iteration (Ch 13.3)
# =====================================================================


def lspi(
    transitions: Sequence[tuple[Any, Any, float, Any]],
    feature_functions: Sequence[Callable[[Any, Any], float]],
    gamma: float,
    actions: Sequence[Any],
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> np.ndarray:
    """Least-Squares Policy Iteration (Ch 13.3, Lagoudakis & Parr 2003).

    LSPI is a batch, off-policy algorithm that combines:
    - LSTD-Q: least-squares estimation of Q^pi from data
    - Policy iteration: extract greedy policy and repeat

    LSTD-Q solves the fixed-point equation for Q^pi in closed form.
    Given feature functions phi(s,a) and transitions (s, a, r, s'):

        A = sum_i phi(s_i, a_i) * [phi(s_i, a_i) - gamma * phi(s'_i, pi(s'_i))]^T
        b = sum_i phi(s_i, a_i) * r_i

    The weight vector w satisfies: A * w = b  =>  w = A^{-1} * b

    Advantages over incremental methods:
    - Sample efficient (batch processing)
    - Stable (no step-size tuning)
    - Off-policy (can use any behavior data)

    Parameters
    ----------
    transitions : sequence of (state, action, reward, next_state)
        The dataset of observed transitions.
    feature_functions : sequence of callables
        Each callable takes (state, action) and returns a float.
        The feature vector phi(s, a) = [f_1(s,a), ..., f_k(s,a)].
    gamma : float
        Discount factor.
    actions : sequence
        The finite action space.
    tolerance : float
        Convergence threshold on weight change ||w_new - w_old||.
    max_iterations : int
        Maximum number of policy iterations.

    Returns
    -------
    np.ndarray
        The learned weight vector w such that Q(s,a) = phi(s,a)^T * w.
        To get the greedy policy: pi(s) = argmax_a phi(s,a)^T * w.
    """
    k = len(feature_functions)
    w = np.zeros(k, dtype=np.float64)

    def phi(state: Any, action: Any) -> np.ndarray:
        return np.array(
            [f(state, action) for f in feature_functions], dtype=np.float64
        )

    def greedy_action(state: Any, weights: np.ndarray) -> Any:
        q_vals = [weights @ phi(state, a) for a in actions]
        return actions[int(np.argmax(q_vals))]

    for iteration in range(max_iterations):
        # LSTD-Q: build A and b matrices
        A_mat = np.eye(k, dtype=np.float64) * 1e-6  # regularization
        b_vec = np.zeros(k, dtype=np.float64)

        for s, a, r, s_prime in transitions:
            phi_sa = phi(s, a)
            a_prime = greedy_action(s_prime, w)
            phi_sa_prime = phi(s_prime, a_prime)

            A_mat += np.outer(phi_sa, phi_sa - gamma * phi_sa_prime)
            b_vec += phi_sa * r

        # Solve A * w_new = b
        try:
            w_new = np.linalg.solve(A_mat, b_vec)
        except np.linalg.LinAlgError:
            logger.warning("LSPI: singular matrix at iteration %d, using lstsq", iteration)
            w_new, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

        # Check convergence
        delta = np.linalg.norm(w_new - w)
        logger.debug("LSPI iteration %d: ||delta_w|| = %.8f", iteration, delta)
        w = w_new

        if delta < tolerance:
            logger.info("LSPI converged after %d iterations", iteration + 1)
            break
    else:
        logger.warning("LSPI did not converge within %d iterations", max_iterations)

    return w


def lspi_policy(
    w: np.ndarray,
    feature_functions: Sequence[Callable[[Any, Any], float]],
    actions: Sequence[Any],
) -> Callable[[Any], Any]:
    """Create a greedy policy function from LSPI weights.

    Parameters
    ----------
    w : np.ndarray
        Weight vector from lspi().
    feature_functions : sequence of callables
        Feature functions used in lspi().
    actions : sequence
        Action space.

    Returns
    -------
    callable
        Greedy policy: state -> action.
    """

    def phi(state: Any, action: Any) -> np.ndarray:
        return np.array(
            [f(state, action) for f in feature_functions], dtype=np.float64
        )

    def policy(state: Any) -> Any:
        q_vals = [w @ phi(state, a) for a in actions]
        return actions[int(np.argmax(q_vals))]

    return policy
