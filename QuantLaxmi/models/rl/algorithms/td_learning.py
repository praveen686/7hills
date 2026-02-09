"""Temporal-Difference Learning Methods for Reinforcement Learning.

Implements Chapter 12 of "Foundations of RL with Applications in Finance"
by Ashwin Rao & Tikhon Jelvis.

TD methods bootstrap — they update value estimates from other value estimates
rather than waiting for the full return as in Monte Carlo methods.  This gives
the key advantages of:
    1. Online, incremental learning (no need to wait for episode end).
    2. Lower variance (at the cost of some bias from bootstrapping).
    3. Ability to learn in continuing (non-episodic) environments.

Key algorithms:
- TD(0) Prediction (Section 12.1)
- TD(lambda) with eligibility traces (Section 12.2)
- SARSA — on-policy TD control (Section 12.3)
- SARSA(lambda) with eligibility traces

References:
    Rao & Jelvis, Ch 12 (Temporal-Difference Methods)
    Sutton & Barto, Ch 6-7 (TD Learning, n-step Bootstrapping)
"""
from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Core interface imports
# ---------------------------------------------------------------------------
try:
    from models.rl.core.markov_process import (
        MarkovDecisionProcess,
        Policy,
        DeterministicPolicy,
        Distribution,
        State,
    )
    from models.rl.core.function_approx import (
        FunctionApprox,
        Tabular,
        LinearFunctionApprox,
    )
    from models.rl.core.utils import epsilon_greedy_policy
except ImportError:
    pass

logger = logging.getLogger(__name__)

S = TypeVar("S", bound=Hashable)
A = TypeVar("A", bound=Hashable)

# =====================================================================
# Minimal tabular approximation (fallback when core is absent)
# =====================================================================


class _TabularTD(Generic[S]):
    """Lightweight tabular value function for TD methods.

    Supports constant step-size alpha updates:
        V(s) <- V(s) + alpha * (target - V(s))
    """

    def __init__(self, default_value: float = 0.0) -> None:
        self._values: dict[S, float] = defaultdict(lambda: default_value)

    def __call__(self, state: S) -> float:
        return self._values[state]

    def td_update(self, state: S, target: float, alpha: float) -> None:
        """In-place TD update."""
        self._values[state] += alpha * (target - self._values[state])

    def evaluate(self, states: Sequence[S]) -> np.ndarray:
        return np.array([self._values[s] for s in states], dtype=np.float64)

    @property
    def values(self) -> dict[S, float]:
        return dict(self._values)


__all__ = [
    "td_prediction",
    "td_lambda_prediction",
    "sarsa",
    "sarsa_lambda",
    "TDExperienceReplay",
]

# =====================================================================
# Helper: epsilon-greedy action selection
# =====================================================================


def _eps_greedy(
    q: _TabularTD,
    state: S,
    actions: Sequence[A],
    epsilon: float,
    rng: np.random.Generator,
) -> A:
    """Select action epsilon-greedily from Q(state, .)."""
    if rng.random() < epsilon:
        return rng.choice(actions)  # type: ignore[return-value]
    q_vals = np.array([q((state, a)) for a in actions], dtype=np.float64)
    max_q = q_vals.max()
    best = [a for a, qv in zip(actions, q_vals) if np.isclose(qv, max_q)]
    return rng.choice(best)  # type: ignore[return-value]


# =====================================================================
# TD(0) Prediction (Ch 12.1)
# =====================================================================


def td_prediction(
    mrp_episodes: Iterable[Iterable[tuple[S, float]]],
    gamma: float,
    approx: Any | None = None,
    learning_rate: float = 0.01,
) -> Any:
    """TD(0) Prediction — estimate V^pi from sampled transitions.

    For each transition (S_t, R_{t+1}, S_{t+1}):
        delta_t = R_{t+1} + gamma * V(S_{t+1}) - V(S_t)       (TD error)
        V(S_t) <- V(S_t) + alpha * delta_t

    Theorem 12.1 (Rao & Jelvis):
        TD(0) converges to V^pi under standard stochastic approximation
        conditions on the step sizes (sum alpha_t = inf, sum alpha_t^2 < inf).
        With constant step size, it converges to a neighborhood of V^pi.

    Bias-variance trade-off (Ch 12.1):
        - MC target (G_t) is unbiased but high variance.
        - TD target (R_{t+1} + gamma*V(S_{t+1})) is biased (bootstrapping)
          but lower variance.
        - In practice, TD(0) often learns faster than MC because the
          variance reduction outweighs the bias.

    Parameters
    ----------
    mrp_episodes : iterable of episodes
        Each episode is an iterable of (state, reward) pairs.
    gamma : float
        Discount factor.
    approx : value function or None
        If None, a tabular approximation is used.
    learning_rate : float
        Step size alpha.

    Returns
    -------
    Value function approximation.
    """
    if approx is None:
        approx = _TabularTD()

    for episode in mrp_episodes:
        episode_list = list(episode)
        if len(episode_list) < 2:
            continue

        for t in range(len(episode_list) - 1):
            s_t, r_tp1 = episode_list[t]
            s_tp1, _ = episode_list[t + 1]
            # TD(0) update: V(S_t) <- V(S_t) + alpha * [R + gamma*V(S') - V(S)]
            td_target = r_tp1 + gamma * approx(s_tp1)
            approx.td_update(s_t, td_target, learning_rate)

        # Handle terminal state: the last state has no successor
        # Its value is just the terminal reward (if any), already handled
        # by the loop above where s_{T-1} bootstraps from s_T.

    return approx


# =====================================================================
# TD(lambda) Prediction with Eligibility Traces (Ch 12.2)
# =====================================================================


def td_lambda_prediction(
    mrp_episodes: Iterable[Iterable[tuple[S, float]]],
    gamma: float,
    lambda_param: float,
    approx: Any | None = None,
    learning_rate: float = 0.01,
) -> Any:
    """TD(lambda) Prediction with eligibility traces (Ch 12.2).

    The forward view defines the lambda-return:
        G_t^lambda = (1-lambda) * sum_{n=1}^{T-t-1} lambda^{n-1} * G_t^{(n)}
                     + lambda^{T-t-1} * G_t

    where G_t^{(n)} is the n-step return.

    The backward view (implemented here) is equivalent and uses accumulating
    eligibility traces for online, incremental computation:

        e_t(s) = gamma * lambda * e_{t-1}(s) + 1_{S_t = s}
        delta_t = R_{t+1} + gamma * V(S_{t+1}) - V(S_t)
        V(s) <- V(s) + alpha * delta_t * e_t(s)   for all s

    Theorem 12.2 (Rao & Jelvis):
        The forward and backward views produce identical total updates
        over a complete episode (offline lambda-return equivalence).

    Special cases:
        lambda = 0: reduces to TD(0)
        lambda = 1: equivalent to every-visit MC (online)

    Parameters
    ----------
    mrp_episodes : iterable of episodes
        Each episode is an iterable of (state, reward) pairs.
    gamma : float
        Discount factor.
    lambda_param : float
        Trace decay parameter in [0, 1].
    approx : value function or None
        If None, tabular approximation is used.
    learning_rate : float
        Step size alpha.

    Returns
    -------
    Value function approximation.
    """
    V = _TabularTD() if approx is None else approx

    for episode in mrp_episodes:
        episode_list = list(episode)
        if len(episode_list) < 2:
            continue

        traces: dict[S, float] = defaultdict(float)

        for t in range(len(episode_list) - 1):
            s_t, r_tp1 = episode_list[t]
            s_tp1, _ = episode_list[t + 1]

            delta = r_tp1 + gamma * V(s_tp1) - V(s_t)

            # Decay all traces and increment current state
            for s in list(traces.keys()):
                traces[s] *= gamma * lambda_param
            traces[s_t] += 1.0

            # Apply update: V(s) += alpha * delta * e(s) for all s with nonzero trace
            dead_keys = []
            for s, e in traces.items():
                if abs(e) > 1e-12:
                    V._values[s] += learning_rate * delta * e
                else:
                    dead_keys.append(s)
            for k in dead_keys:
                del traces[k]

    return V


# =====================================================================
# SARSA — On-Policy TD Control (Ch 12.3)
# =====================================================================


def sarsa(
    mdp_step: Callable[[S, A], tuple[S, float, bool]],
    start_states: Sequence[S] | Callable[[], S],
    actions: Sequence[A],
    gamma: float,
    approx: Any | None = None,
    num_episodes: int = 10_000,
    learning_rate: float = 0.01,
    epsilon: float = 0.1,
    seed: int = 42,
) -> tuple[Any, Callable[[S], A]]:
    """SARSA — On-policy TD Control (Ch 12.3).

    The name comes from the quintuple (S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}).

    Update rule:
        Q(S_t, A_t) <- Q(S_t, A_t) + alpha * [R_{t+1} + gamma*Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]

    Key distinction from Q-learning:
        SARSA uses the *actual* next action A_{t+1} (chosen by the current
        epsilon-greedy policy), making it ON-policy.  Q-learning uses
        max_a Q(S_{t+1}, a), making it OFF-policy.

    Theorem 12.3 (Rao & Jelvis):
        SARSA converges to Q* if all state-action pairs are visited infinitely
        often and the step sizes satisfy the Robbins-Monro conditions.

    On-policy advantage:
        SARSA accounts for the exploration policy when estimating Q, which can
        be safer in stochastic environments (e.g., the cliff-walking example
        in Sutton & Barto Section 6.5).

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
    seed : int

    Returns
    -------
    (q_approx, greedy_policy)
    """
    rng = np.random.default_rng(seed)
    Q = _TabularTD() if approx is None else approx

    for ep in range(num_episodes):
        if callable(start_states) and not isinstance(start_states, (list, tuple)):
            state = start_states()
        else:
            state = start_states[rng.integers(len(start_states))]  # type: ignore[index]

        action = _eps_greedy(Q, state, actions, epsilon, rng)

        for _ in range(10_000):  # max steps per episode
            next_state, reward, done = mdp_step(state, action)

            if done:
                # Terminal update: no bootstrapping
                td_target = reward
                Q._values[(state, action)] += learning_rate * (
                    td_target - Q((state, action))
                )
                break

            next_action = _eps_greedy(Q, next_state, actions, epsilon, rng)
            td_target = reward + gamma * Q((next_state, next_action))
            Q._values[(state, action)] += learning_rate * (
                td_target - Q((state, action))
            )

            state = next_state
            action = next_action

    def greedy_policy(state: S) -> A:
        q_vals = np.array([Q((state, a)) for a in actions], dtype=np.float64)
        return actions[int(np.argmax(q_vals))]

    return Q, greedy_policy


# =====================================================================
# SARSA(lambda) with Eligibility Traces (Ch 12)
# =====================================================================


def sarsa_lambda(
    mdp_step: Callable[[S, A], tuple[S, float, bool]],
    start_states: Sequence[S] | Callable[[], S],
    actions: Sequence[A],
    gamma: float,
    lambda_param: float,
    approx: Any | None = None,
    num_episodes: int = 10_000,
    learning_rate: float = 0.01,
    epsilon: float = 0.1,
    seed: int = 42,
) -> tuple[Any, Callable[[S], A]]:
    """SARSA(lambda) with eligibility traces (Ch 12).

    Combines SARSA's on-policy TD control with eligibility traces for
    faster credit assignment.

    Update equations:
        e_t(s, a) = gamma * lambda * e_{t-1}(s, a) + 1_{(S_t, A_t) = (s, a)}
        delta_t = R_{t+1} + gamma * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
        Q(s, a) <- Q(s, a) + alpha * delta_t * e_t(s, a)   for all (s, a)

    Reduces to SARSA when lambda=0 (only the current (s,a) is updated).
    Approaches MC-like behavior when lambda=1.

    Parameters
    ----------
    mdp_step : callable
    start_states : sequence or callable
    actions : sequence
    gamma : float
    lambda_param : float in [0, 1]
    approx : Q-function approximation or None
    num_episodes : int
    learning_rate : float
    epsilon : float
    seed : int

    Returns
    -------
    (q_approx, greedy_policy)
    """
    rng = np.random.default_rng(seed)
    Q = _TabularTD() if approx is None else approx

    for ep in range(num_episodes):
        if callable(start_states) and not isinstance(start_states, (list, tuple)):
            state = start_states()
        else:
            state = start_states[rng.integers(len(start_states))]  # type: ignore[index]

        action = _eps_greedy(Q, state, actions, epsilon, rng)
        traces: dict[tuple[S, A], float] = defaultdict(float)

        for _ in range(10_000):
            next_state, reward, done = mdp_step(state, action)

            if done:
                delta = reward - Q((state, action))
                # Decay traces and increment current (s, a)
                for sa in list(traces.keys()):
                    traces[sa] *= gamma * lambda_param
                traces[(state, action)] += 1.0
                # Update all
                dead = []
                for sa, e in traces.items():
                    if abs(e) > 1e-12:
                        Q._values[sa] += learning_rate * delta * e
                    else:
                        dead.append(sa)
                for k in dead:
                    del traces[k]
                break

            next_action = _eps_greedy(Q, next_state, actions, epsilon, rng)
            delta = reward + gamma * Q((next_state, next_action)) - Q((state, action))

            # Decay all traces and increment current (s, a)
            for sa in list(traces.keys()):
                traces[sa] *= gamma * lambda_param
            traces[(state, action)] += 1.0

            # Update Q for all (s, a) with non-zero traces
            dead = []
            for sa, e in traces.items():
                if abs(e) > 1e-12:
                    Q._values[sa] += learning_rate * delta * e
                else:
                    dead.append(sa)
            for k in dead:
                del traces[k]

            state = next_state
            action = next_action

    def greedy_policy(state: S) -> A:
        q_vals = np.array([Q((state, a)) for a in actions], dtype=np.float64)
        return actions[int(np.argmax(q_vals))]

    return Q, greedy_policy


# =====================================================================
# Experience Replay Buffer (Ch 12 / supplementary)
# =====================================================================


class TDExperienceReplay:
    """Experience replay buffer for off-policy TD methods.

    Stores transitions (s, a, r, s', done) and supports both uniform
    random sampling and prioritized experience replay (PER).

    Experience replay (Lin, 1992) provides two key benefits:
        1. Data efficiency — each transition can be used in multiple updates.
        2. Decorrelation — random sampling breaks temporal correlations in
           sequential data, stabilizing learning.

    Prioritized experience replay (Schaul et al., 2016):
        Transitions with higher |TD error| are sampled more frequently,
        focusing learning on "surprising" experiences.
        Priority: p_i = |delta_i| + epsilon_prior
        Sampling probability: P(i) = p_i^alpha / sum_j p_j^alpha
        Importance sampling correction: w_i = (1/(N*P(i)))^beta

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    prioritized : bool
        If True, use proportional prioritized replay.
    alpha_priority : float
        Priority exponent (0 = uniform, 1 = full prioritization).
    beta_start : float
        Initial importance sampling exponent (annealed to 1.0).
    epsilon_prior : float
        Small constant added to priorities to prevent zero probability.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        prioritized: bool = False,
        alpha_priority: float = 0.6,
        beta_start: float = 0.4,
        epsilon_prior: float = 1e-6,
    ) -> None:
        self._capacity = capacity
        self._prioritized = prioritized
        self._alpha = alpha_priority
        self._beta = beta_start
        self._epsilon = epsilon_prior
        self._buffer: deque[tuple] = deque(maxlen=capacity)
        self._priorities: deque[float] = deque(maxlen=capacity)
        self._max_priority: float = 1.0
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, transition: tuple, priority: float | None = None) -> None:
        """Add a transition to the buffer.

        Parameters
        ----------
        transition : tuple
            Typically (state, action, reward, next_state, done).
        priority : float or None
            If None and prioritized, uses max priority seen so far.
        """
        self._buffer.append(transition)
        if self._prioritized:
            p = priority if priority is not None else self._max_priority
            self._priorities.append(p)

    def sample(
        self, batch_size: int
    ) -> list[tuple] | tuple[list[tuple], np.ndarray, np.ndarray]:
        """Sample a batch of transitions.

        For uniform replay: returns list of transitions.
        For prioritized replay: returns (transitions, weights, indices)
            where weights are importance sampling corrections.
        """
        n = len(self._buffer)
        if n == 0:
            return []

        batch_size = min(batch_size, n)

        if not self._prioritized:
            indices = self._rng.choice(n, size=batch_size, replace=False)
            return [self._buffer[i] for i in indices]

        # Prioritized sampling
        priorities = np.array(self._priorities, dtype=np.float64)
        probs = (priorities + self._epsilon) ** self._alpha
        probs /= probs.sum()

        indices = self._rng.choice(n, size=batch_size, replace=False, p=probs)

        # Importance sampling weights
        weights = (n * probs[indices]) ** (-self._beta)
        weights /= weights.max()  # normalize

        transitions = [self._buffer[i] for i in indices]
        return transitions, weights, indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors (for prioritized replay).

        Parameters
        ----------
        indices : array of int
            Indices of transitions to update.
        td_errors : array of float
            Corresponding TD errors.
        """
        for idx, td_err in zip(indices, td_errors):
            p = abs(td_err) + self._epsilon
            self._priorities[idx] = p
            self._max_priority = max(self._max_priority, p)

    def anneal_beta(self, fraction: float) -> None:
        """Anneal the importance sampling exponent toward 1.0.

        Parameters
        ----------
        fraction : float in [0, 1]
            Fraction of training complete. beta linearly increases to 1.0.
        """
        self._beta = self._beta + fraction * (1.0 - self._beta)
