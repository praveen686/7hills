"""Utility functions for the QuantLaxmi RL framework.

Provides shared helpers used across the core MDP framework:
  - Discounted return computation
  - Episode generation from MDPs
  - Policy extraction from value functions
  - Exploration policies (ε-greedy, softmax)
  - Utility theory helpers (CRRA, CARA, certainty equivalent)
  - Convergence iteration
  - Device management for GPU acceleration

References:
  - Ch 2-3: Episode generation, returns
  - Ch 7: Utility theory (CRRA, CARA, certainty equivalent)
  - Ch 11-12: Exploration (ε-greedy, softmax/Boltzmann)
"""
from __future__ import annotations

import math
import random
from typing import (
    Callable,
    Generic,
    Iterable,
    Iterator,
    Sequence,
    TypeVar,
)

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from .markov_process import (
    A,
    S,
    Categorical,
    Constant,
    DeterministicPolicy,
    Distribution,
    MarkovDecisionProcess,
    NonTerminal,
    Policy,
    SampledDistribution,
    State,
    Terminal,
)
from .dynamic_programming import ActionValueFunction

__all__ = [
    "returns",
    "returns_from_rewards",
    "episodes_from_mdp",
    "greedy_policy_from_qvf",
    "epsilon_greedy_policy",
    "softmax_policy",
    "crra_utility",
    "cara_utility",
    "certainty_equivalent",
    "iterate_converge",
    "set_device",
    "moving_average",
    "td_target",
]

X = TypeVar("X")


# ---------------------------------------------------------------------------
# Discounted returns  (Ch 2, §2.4)
# ---------------------------------------------------------------------------


def returns(
    rewards: Sequence[float],
    gamma: float,
    normalize: bool = False,
) -> list[float]:
    """Compute discounted returns from a reward sequence.

    G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ... = Σ_{k=0}^{T-t-1} γ^k · r_{t+k+1}

    Computed efficiently via backward recursion:
      G_T = 0
      G_t = r_{t+1} + γ·G_{t+1}   for t = T-1, ..., 0

    Reference: Ch 2, §2.4, Eq (2.4) — "The discounted return G_t."

    Args:
        rewards: Sequence of rewards [r_1, r_2, ..., r_T].
        gamma: Discount factor γ ∈ [0, 1].
        normalize: If True, normalise returns to zero mean, unit variance.

    Returns:
        List of returns [G_0, G_1, ..., G_{T-1}].
    """
    if not rewards:
        return []

    T = len(rewards)
    G = [0.0] * T

    # Backward recursion
    G[T - 1] = rewards[T - 1]
    for t in range(T - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]

    if normalize and T > 1:
        arr = np.array(G, dtype=np.float64)
        mean = arr.mean()
        std = arr.std(ddof=1)
        if std > 1e-8:
            G = ((arr - mean) / std).tolist()

    return G


def returns_from_rewards(
    rewards: Sequence[float],
    gamma: float,
) -> float:
    """Compute the single discounted return from a complete episode.

    G_0 = Σ_{t=0}^{T-1} γ^t · r_{t+1}

    Reference: Ch 2, §2.4.

    Args:
        rewards: Sequence of rewards [r_1, r_2, ..., r_T].
        gamma: Discount factor.

    Returns:
        The total discounted return G_0.
    """
    result = 0.0
    for t, r in enumerate(rewards):
        result += (gamma ** t) * r
    return result


# ---------------------------------------------------------------------------
# Episode generation  (Ch 3, §3.5)
# ---------------------------------------------------------------------------


def episodes_from_mdp(
    mdp: MarkovDecisionProcess[S, A],
    policy: Policy[S, A],
    start_distribution: Distribution[NonTerminal[S]],
    num_episodes: int,
    max_steps: int = 1000,
) -> list[list[tuple[NonTerminal[S], A, float]]]:
    """Generate episodes (trajectories) from an MDP following a policy.

    Each episode is a list of (state, action, reward) triples:
      [(s_0, a_0, r_1), (s_1, a_1, r_2), ..., (s_{T-1}, a_{T-1}, r_T)]

    The episode terminates when:
      1. A Terminal state is reached, or
      2. max_steps transitions have occurred.

    Reference: Ch 3, §3.5 — "Sampling episodes from an MDP."

    This is the fundamental data-generation primitive used by all
    Monte Carlo and TD learning algorithms.

    Args:
        mdp: The MDP environment.
        policy: The policy π to follow.
        start_distribution: Distribution over initial states.
        num_episodes: Number of episodes to generate.
        max_steps: Maximum steps per episode (prevents infinite loops).

    Returns:
        List of episodes, each a list of (state, action, reward) triples.
    """
    all_episodes: list[list[tuple[NonTerminal[S], A, float]]] = []

    for _ in range(num_episodes):
        episode: list[tuple[NonTerminal[S], A, float]] = []
        state: State[S] = start_distribution.sample()

        for step in range(max_steps):
            if not isinstance(state, NonTerminal):
                break

            action = policy.act(state).sample()
            next_state, reward = mdp.step(state, action).sample()
            episode.append((state, action, reward))
            state = next_state

        all_episodes.append(episode)

    return all_episodes


# ---------------------------------------------------------------------------
# Policy extraction  (Ch 4-5)
# ---------------------------------------------------------------------------


def greedy_policy_from_qvf(
    qvf: ActionValueFunction[S, A],
    mdp: MarkovDecisionProcess[S, A],
) -> DeterministicPolicy[S, A]:
    """Extract a greedy deterministic policy from a Q-value function.

    π(s) = argmax_a Q(s, a)

    Reference: Ch 4, §4.2 — "Policy improvement via the greedy policy."

    Args:
        qvf: Action-value function Q(s,a).
        mdp: The MDP (needed to enumerate available actions).

    Returns:
        Deterministic greedy policy.
    """

    def _action_for(state: NonTerminal[S]) -> A:
        actions = list(mdp.actions(state))
        if not actions:
            raise ValueError(f"No actions available in state {state}")
        return max(actions, key=lambda a: qvf[(state, a)])

    return DeterministicPolicy(_action_for)


def epsilon_greedy_policy(
    qvf: ActionValueFunction[S, A],
    mdp: MarkovDecisionProcess[S, A],
    epsilon: float,
) -> Policy[S, A]:
    """Epsilon-greedy exploration policy.

    With probability (1-ε): choose argmax_a Q(s,a)  (exploit)
    With probability ε:     choose uniformly at random (explore)

    This ensures all actions are tried with non-zero probability,
    satisfying the exploration requirement for convergence of
    MC control and Q-learning.

    Reference: Ch 11, §11.2 — "ε-greedy exploration."

    The effective action probabilities are:
      π(a|s) = ε/|A(s)|                       if a ≠ a*
      π(a*|s) = 1 - ε + ε/|A(s)|              if a = a*
    where a* = argmax_a Q(s,a).

    Args:
        qvf: Action-value function Q(s,a).
        mdp: The MDP.
        epsilon: Exploration rate ε ∈ [0, 1].

    Returns:
        Stochastic ε-greedy policy.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0,1], got {epsilon}")

    class _EpsilonGreedyPolicy(Policy[S, A]):
        def act(self, state: NonTerminal[S]) -> Distribution[A]:
            actions = list(mdp.actions(state))
            if not actions:
                raise ValueError(f"No actions in state {state}")

            n_actions = len(actions)
            best_action = max(actions, key=lambda a: qvf[(state, a)])

            probs: dict[A, float] = {}
            for a in actions:
                if a == best_action:
                    probs[a] = 1.0 - epsilon + epsilon / n_actions
                else:
                    probs[a] = epsilon / n_actions

            return Categorical(probs)

    return _EpsilonGreedyPolicy()


def softmax_policy(
    qvf: ActionValueFunction[S, A],
    mdp: MarkovDecisionProcess[S, A],
    temperature: float = 1.0,
) -> Policy[S, A]:
    """Softmax (Boltzmann) exploration policy.

    π(a|s) = exp(Q(s,a) / τ) / Σ_{a'} exp(Q(s,a') / τ)

    As τ → 0: converges to greedy policy
    As τ → ∞: converges to uniform random policy

    Reference: Ch 11, §11.3 — "Softmax exploration."

    Args:
        qvf: Action-value function Q(s,a).
        mdp: The MDP.
        temperature: Temperature parameter τ > 0.

    Returns:
        Stochastic Boltzmann policy.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    class _SoftmaxPolicy(Policy[S, A]):
        def act(self, state: NonTerminal[S]) -> Distribution[A]:
            actions = list(mdp.actions(state))
            if not actions:
                raise ValueError(f"No actions in state {state}")

            q_values = np.array(
                [qvf[(state, a)] for a in actions], dtype=np.float64
            )

            # Numerically stable softmax: subtract max
            q_shifted = q_values / temperature
            q_shifted -= q_shifted.max()
            exp_q = np.exp(q_shifted)
            probs_arr = exp_q / exp_q.sum()

            probs: dict[A, float] = {
                a: float(p) for a, p in zip(actions, probs_arr)
            }
            return Categorical(probs)

    return _SoftmaxPolicy()


# ---------------------------------------------------------------------------
# Utility theory  (Ch 7, §7.1-7.2)
# ---------------------------------------------------------------------------


def crra_utility(x: float, gamma: float) -> float:
    """Constant Relative Risk Aversion (CRRA) utility function.

    u(x) = x^{1-γ} / (1-γ)    for γ ≠ 1
    u(x) = log(x)              for γ = 1

    The parameter γ is the coefficient of relative risk aversion:
      RRA(x) = -x · u''(x) / u'(x) = γ

    Properties:
      - γ = 0: risk-neutral (linear utility)
      - γ = 1: log utility (Bernoulli, Kelly criterion)
      - γ > 1: risk-averse
      - γ < 0: risk-seeking

    Reference: Ch 7, §7.1, Eq (7.3) — "CRRA Utility."

    Args:
        x: Wealth level (must be > 0).
        gamma: Risk aversion coefficient.

    Returns:
        Utility value u(x).
    """
    if x <= 0:
        raise ValueError(f"CRRA utility requires x > 0, got {x}")
    if abs(gamma - 1.0) < 1e-10:
        return math.log(x)
    return (x ** (1.0 - gamma)) / (1.0 - gamma)


def cara_utility(x: float, alpha: float) -> float:
    """Constant Absolute Risk Aversion (CARA) utility function.

    u(x) = -exp(-α·x) / α    for α ≠ 0
    u(x) = x                  for α = 0

    The parameter α is the coefficient of absolute risk aversion:
      ARA(x) = -u''(x) / u'(x) = α

    Reference: Ch 7, §7.2, Eq (7.5) — "CARA (Exponential) Utility."

    Args:
        x: Wealth level.
        alpha: Risk aversion coefficient (α > 0 for risk-averse).

    Returns:
        Utility value u(x).
    """
    if abs(alpha) < 1e-10:
        return x
    return -math.exp(-alpha * x) / alpha


def certainty_equivalent(
    distribution: Distribution[float],
    utility_fn: Callable[[float], float],
    inverse_utility: Callable[[float], float],
    num_samples: int = 100_000,
) -> float:
    """Compute the certainty equivalent of a distribution.

    CE = u^{-1}(E[u(X)])

    The certainty equivalent is the guaranteed amount that gives
    the same utility as the uncertain prospect X:
      u(CE) = E[u(X)]

    For a risk-averse agent (concave u): CE < E[X]  (risk premium > 0).
    For a risk-neutral agent (linear u): CE = E[X].

    Reference: Ch 7, §7.3 — "Certainty Equivalents and Risk Premia."

    Args:
        distribution: Distribution over monetary outcomes.
        utility_fn: Utility function u.
        inverse_utility: Inverse u^{-1}.
        num_samples: Number of MC samples for E[u(X)].

    Returns:
        Certainty equivalent CE.
    """
    expected_utility = distribution.expectation(utility_fn)
    return inverse_utility(expected_utility)


# ---------------------------------------------------------------------------
# Convergence iteration
# ---------------------------------------------------------------------------


def iterate_converge(
    func: Callable[[X], X],
    start: X,
    done: Callable[[X, X], bool],
    max_iterations: int = 10_000,
) -> X:
    """Iterate a function until convergence.

    Repeatedly applies x_{k+1} = func(x_k) until done(x_k, x_{k+1})
    returns True or max_iterations is reached.

    This is a general-purpose fixed-point iteration used throughout DP:
    - Policy evaluation: V_{k+1} = T^π V_k
    - Value iteration: V_{k+1} = T* V_k

    Args:
        func: The iteration function f: X → X.
        start: Initial value x_0.
        done: Convergence predicate (old, new) → bool.
        max_iterations: Safety limit.

    Returns:
        Converged value x*.
    """
    current = start
    for _ in range(max_iterations):
        nxt = func(current)
        if done(current, nxt):
            return nxt
        current = nxt
    return current


def iterate_converge_numeric(
    func: Callable[[float], float],
    start: float,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
) -> float:
    """Specialised iterate_converge for float values with tolerance check.

    Args:
        func: Scalar iteration function.
        start: Initial value.
        tolerance: Stop when |x_{k+1} - x_k| < tolerance.
        max_iterations: Safety limit.

    Returns:
        Converged scalar value.
    """
    return iterate_converge(
        func=func,
        start=start,
        done=lambda old, new: abs(new - old) < tolerance,
        max_iterations=max_iterations,
    )


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------


def set_device(prefer_gpu: bool = True) -> "torch.device":
    """Auto-detect and return the best available compute device.

    Priority: CUDA GPU (T4) > CPU.

    Args:
        prefer_gpu: If True, use GPU when available.

    Returns:
        torch.device for computation.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GPU device selection")

    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
        return device
    else:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Miscellaneous helpers
# ---------------------------------------------------------------------------


def moving_average(values: Sequence[float], window: int) -> list[float]:
    """Compute the moving average of a sequence.

    Useful for smoothing learning curves and reward traces.

    Args:
        values: Input sequence.
        window: Window size.

    Returns:
        List of moving averages (length = len(values) - window + 1).
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    if window > len(values):
        return [float(np.mean(values))] if values else []

    cumsum = np.cumsum(values, dtype=np.float64)
    cumsum = np.insert(cumsum, 0, 0.0)
    return [
        float((cumsum[i + window] - cumsum[i]) / window)
        for i in range(len(values) - window + 1)
    ]


def td_target(
    reward: float,
    gamma: float,
    next_value: float,
    done: bool = False,
) -> float:
    """Compute the TD(0) target: r + γ·V(s') (or just r if terminal).

    Reference: Ch 12, §12.1, Eq (12.1) — "The TD target."

    Args:
        reward: Immediate reward r.
        gamma: Discount factor.
        next_value: V(s') or Q(s', a').
        done: Whether s' is terminal.

    Returns:
        TD target value.
    """
    if done:
        return reward
    return reward + gamma * next_value
