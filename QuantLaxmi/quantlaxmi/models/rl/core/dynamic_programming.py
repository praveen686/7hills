"""Dynamic Programming algorithms for solving Markov Decision Processes.

Implements the classical DP algorithms from:
  "Foundations of Reinforcement Learning with Applications in Finance"
  (Rao & Jelvis, Stanford CME 241), Chapters 4-5.

Chapter 4: Dynamic Programming for Prediction & Control
  - Policy Evaluation (iterative): compute V^π given policy π
  - Policy Improvement: extract greedy policy from V
  - Policy Iteration: alternate evaluation & improvement

Chapter 5: Value Iteration and Finite-Horizon Problems
  - Value Iteration: compute V* and π* directly
  - Backward Induction: finite-horizon optimal control
  - Direct solve: matrix inversion for small state spaces

All algorithms work with the Finite* classes from markov_process.py,
using explicit transition matrices for exact computation.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import (
    Callable,
    Generic,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

import numpy as np

from .markov_process import (
    A,
    S,
    Categorical,
    DeterministicPolicy,
    Distribution,
    FiniteMarkovDecisionProcess,
    FiniteMarkovRewardProcess,
    MarkovDecisionProcess,
    NonTerminal,
    Policy,
    State,
    TabularPolicy,
    Terminal,
)

__all__ = [
    "DEFAULT_TOLERANCE",
    "ValueFunction",
    "ActionValueFunction",
    "policy_evaluation",
    "policy_evaluation_direct",
    "policy_improvement",
    "policy_iteration",
    "value_iteration",
    "backward_induction",
    "bellman_optimality_operator",
    "bellman_policy_operator",
    "lstd_prediction",
]

# Module-level convergence tolerance (used across all DP algorithms)
DEFAULT_TOLERANCE: float = 1e-6

# ---------------------------------------------------------------------------
# Value functions  (Ch 3, §3.4 — state-value and action-value)
# ---------------------------------------------------------------------------


class ValueFunction(Generic[S]):
    """State-value function V: S → ℝ.

    V^π(s) = E_π[Σ_{t=0}^∞ γ^t · R_{t+1} | S_0 = s]

    Supports both tabular (dict-backed) and vector representations.

    Reference: Ch 3, §3.4, Def 3.4 — "The state-value function V^π(s)
    for policy π is the expected return when starting in s and following π."
    """

    def __init__(
        self, values: Mapping[NonTerminal[S], float] | None = None
    ) -> None:
        self._values: dict[NonTerminal[S], float] = dict(values or {})

    def __getitem__(self, state: NonTerminal[S]) -> float:
        return self._values.get(state, 0.0)

    def __setitem__(self, state: NonTerminal[S], value: float) -> None:
        self._values[state] = value

    def __contains__(self, state: NonTerminal[S]) -> bool:
        return state in self._values

    @property
    def values(self) -> dict[NonTerminal[S], float]:
        return dict(self._values)

    def as_vector(
        self, states: Sequence[NonTerminal[S]]
    ) -> np.ndarray:
        """Return values as numpy array in the given state order."""
        return np.array([self._values.get(s, 0.0) for s in states])

    @staticmethod
    def from_vector(
        states: Sequence[NonTerminal[S]],
        values: np.ndarray,
    ) -> ValueFunction[S]:
        """Create from a numpy array and state ordering."""
        vf = ValueFunction[S]()
        for s, v in zip(states, values):
            vf[s] = float(v)
        return vf

    def max_diff(self, other: ValueFunction[S]) -> float:
        """L∞ distance: max_s |V(s) - V'(s)|."""
        all_states = set(self._values.keys()) | set(other._values.keys())
        if not all_states:
            return 0.0
        return max(
            abs(self._values.get(s, 0.0) - other._values.get(s, 0.0))
            for s in all_states
        )

    def copy(self) -> ValueFunction[S]:
        return ValueFunction(dict(self._values))

    def __repr__(self) -> str:
        items = ", ".join(
            f"{s}: {v:.4f}" for s, v in sorted(
                self._values.items(), key=lambda x: repr(x[0])
            )
        )
        return f"V({{{items}}})"


class ActionValueFunction(Generic[S, A]):
    """Action-value function Q: S × A → ℝ.

    Q^π(s,a) = E_π[Σ_{t=0}^∞ γ^t · R_{t+1} | S_0=s, A_0=a]

    Reference: Ch 3, §3.4, Def 3.5 — "The action-value function Q^π(s,a)
    is the expected return starting from s, taking action a, then following π."
    """

    def __init__(
        self,
        values: Mapping[tuple[NonTerminal[S], A], float] | None = None,
    ) -> None:
        self._values: dict[tuple[NonTerminal[S], A], float] = dict(values or {})

    def __getitem__(self, key: tuple[NonTerminal[S], A]) -> float:
        return self._values.get(key, 0.0)

    def __setitem__(self, key: tuple[NonTerminal[S], A], value: float) -> None:
        self._values[key] = value

    @property
    def values(self) -> dict[tuple[NonTerminal[S], A], float]:
        return dict(self._values)

    def value_function(
        self,
        mdp: FiniteMarkovDecisionProcess[S, A],
    ) -> ValueFunction[S]:
        """Derive V(s) = max_a Q(s,a)."""
        vf = ValueFunction[S]()
        for nt in mdp.non_terminal_states:
            actions = mdp.actions(nt)
            if actions:
                vf[nt] = max(self[(nt, a)] for a in actions)
        return vf

    def copy(self) -> ActionValueFunction[S, A]:
        return ActionValueFunction(dict(self._values))

    def __repr__(self) -> str:
        items = ", ".join(
            f"({s},{a}): {v:.4f}"
            for (s, a), v in sorted(
                self._values.items(), key=lambda x: repr(x[0])
            )
        )
        return f"Q({{{items}}})"


# ---------------------------------------------------------------------------
# Helper: compute Q(s,a) from V(s) for a FiniteMarkovDecisionProcess
# ---------------------------------------------------------------------------


def _compute_q_value(
    mdp: FiniteMarkovDecisionProcess[S, A],
    state: NonTerminal[S],
    action: A,
    vf: ValueFunction[S],
    gamma: float,
) -> float:
    """Compute Q(s,a) = Σ_{s',r} p(s',r|s,a)·[r + γ·V(s')].

    Reference: Ch 3, Eq (3.5).
    """
    step_dist = mdp.step(state, action)
    probs = step_dist.probabilities()

    q = 0.0
    for (next_state, reward), prob in probs.items():
        v_next = 0.0
        if isinstance(next_state, NonTerminal):
            v_next = vf[next_state]
        q += prob * (reward + gamma * v_next)
    return q


def _compute_action_value_function(
    mdp: FiniteMarkovDecisionProcess[S, A],
    vf: ValueFunction[S],
    gamma: float,
) -> ActionValueFunction[S, A]:
    """Compute Q(s,a) for all state-action pairs from V(s).

    Reference: Ch 3, Eq (3.5).
    """
    qvf = ActionValueFunction[S, A]()
    for nt in mdp.non_terminal_states:
        for a in mdp.actions(nt):
            qvf[(nt, a)] = _compute_q_value(mdp, nt, a, vf, gamma)
    return qvf


# ---------------------------------------------------------------------------
# Policy Evaluation  (Ch 4, §4.1)
# ---------------------------------------------------------------------------


def policy_evaluation(
    mrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = 10_000,
) -> ValueFunction[S]:
    """Iterative Policy Evaluation for a finite MRP.

    Computes V^π by repeatedly applying the Bellman expectation operator:
      V_{k+1}(s) = Σ_{s',r} p^π(s',r|s)·[r + γ·V_k(s')]

    Converges when ||V_{k+1} - V_k||_∞ < tolerance.

    Reference: Ch 4, §4.1, Algorithm 4.1 — "Iterative Policy Evaluation."

    Complexity: O(|S|² · K) where K is the number of iterations.

    Args:
        mrp: The Markov Reward Process to evaluate.
        gamma: Discount factor γ ∈ [0,1].
        tolerance: Convergence threshold for L∞ norm.
        max_iterations: Maximum number of iterations.

    Returns:
        ValueFunction V^π satisfying the Bellman equation.
    """
    non_terminal = mrp.non_terminal_states
    vf = ValueFunction[S]({nt: 0.0 for nt in non_terminal})

    for iteration in range(max_iterations):
        new_vf = ValueFunction[S]()

        for nt in non_terminal:
            tr_dist = mrp.transition_reward(nt)
            probs = tr_dist.probabilities()
            val = 0.0
            for (next_state, reward), prob in probs.items():
                v_next = 0.0
                if isinstance(next_state, NonTerminal):
                    v_next = vf[next_state]
                val += prob * (reward + gamma * v_next)
            new_vf[nt] = val

        if vf.max_diff(new_vf) < tolerance:
            return new_vf
        vf = new_vf

    return vf


def policy_evaluation_direct(
    mrp: FiniteMarkovRewardProcess[S],
    gamma: float,
) -> ValueFunction[S]:
    """Direct (matrix) solution of the Bellman equation for a finite MRP.

    Solves V = (I - γP)^{-1} · R exactly using linear algebra.

    Reference: Ch 2, §2.6, Eq (2.9) — "The closed-form solution to
    the Bellman equation for MRPs."

    Complexity: O(|S|³)  — use only for small state spaces.

    Args:
        mrp: The Markov Reward Process to evaluate.
        gamma: Discount factor γ ∈ [0,1].

    Returns:
        Exact ValueFunction V^π.
    """
    values = mrp.direct_solve(gamma)
    return ValueFunction(values)


# ---------------------------------------------------------------------------
# Policy Improvement  (Ch 4, §4.2)
# ---------------------------------------------------------------------------


def policy_improvement(
    mdp: FiniteMarkovDecisionProcess[S, A],
    vf: ValueFunction[S],
) -> DeterministicPolicy[S, A]:
    """Greedy policy improvement: extract the best deterministic policy from V.

    For each state s, compute:
      π'(s) = argmax_a Σ_{s',r} p(s',r|s,a)·[r + γ·V(s')]
            = argmax_a Q^π(s,a)

    Reference: Ch 4, §4.2, Theorem 4.2 (Policy Improvement Theorem) —
    "If π'(s) = argmax_a Q^π(s,a), then V^{π'}(s) ≥ V^π(s) for all s."

    Args:
        mdp: The MDP.
        vf: Current value function V^π.

    Returns:
        Improved deterministic policy π'.
    """
    gamma = mdp.gamma
    state_action_map: dict[S, A] = {}

    for nt in mdp.non_terminal_states:
        best_action: A | None = None
        best_value: float = -float("inf")
        for a in mdp.actions(nt):
            q = _compute_q_value(mdp, nt, a, vf, gamma)
            if q > best_value:
                best_value = q
                best_action = a
        assert best_action is not None, f"No actions available in state {nt}"
        state_action_map[nt.state] = best_action

    return DeterministicPolicy.from_mapping(state_action_map)


# ---------------------------------------------------------------------------
# Policy Iteration  (Ch 4, §4.3)
# ---------------------------------------------------------------------------


def policy_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = 1_000,
    use_direct_solve: bool = False,
) -> tuple[ValueFunction[S], DeterministicPolicy[S, A]]:
    """Policy Iteration: alternate policy evaluation and improvement.

    Algorithm:
      1. Start with an arbitrary policy π_0
      2. Policy Evaluation: compute V^{π_k} (iterative or direct)
      3. Policy Improvement: π_{k+1}(s) = argmax_a Q^{π_k}(s,a)
      4. If π_{k+1} = π_k, STOP — we have found the optimal policy

    Reference: Ch 4, §4.3, Algorithm 4.3 — "Policy Iteration."

    Theorem 4.3 (Convergence): Policy iteration converges to the optimal
    policy π* and optimal value function V* in a finite number of steps
    (at most |A|^{|S|} iterations, but typically very few).

    Args:
        mdp: Finite MDP to solve.
        gamma: Discount factor. If None, uses mdp.gamma.
        tolerance: Convergence tolerance for policy evaluation.
        max_iterations: Max number of PI iterations.
        use_direct_solve: If True, use matrix inversion for evaluation
                         (faster for small state spaces).

    Returns:
        (V*, π*) — optimal value function and policy.
    """
    if gamma is None:
        gamma = mdp.gamma

    # Initialize with first available action for each state
    state_action_map: dict[S, A] = {}
    for nt in mdp.non_terminal_states:
        actions = list(mdp.actions(nt))
        if actions:
            state_action_map[nt.state] = actions[0]

    current_policy = DeterministicPolicy.from_mapping(state_action_map)

    for iteration in range(max_iterations):
        # --- Policy Evaluation ---
        mrp = mdp.apply_policy(current_policy)
        if use_direct_solve:
            vf = policy_evaluation_direct(mrp, gamma)
        else:
            vf = policy_evaluation(mrp, gamma, tolerance=tolerance)

        # --- Policy Improvement ---
        new_policy = policy_improvement(mdp, vf)

        # --- Check for convergence (policy stability) ---
        policy_stable = True
        for nt in mdp.non_terminal_states:
            old_action = current_policy.act(nt).sample()
            new_action = new_policy.act(nt).sample()
            if old_action != new_action:
                policy_stable = False
                break

        current_policy = new_policy

        if policy_stable:
            # Recompute V for the final policy
            mrp = mdp.apply_policy(current_policy)
            if use_direct_solve:
                vf = policy_evaluation_direct(mrp, gamma)
            else:
                vf = policy_evaluation(mrp, gamma, tolerance=tolerance)
            return vf, current_policy

    return vf, current_policy


# ---------------------------------------------------------------------------
# Value Iteration  (Ch 5, §5.1)
# ---------------------------------------------------------------------------


def bellman_optimality_operator(
    mdp: FiniteMarkovDecisionProcess[S, A],
    vf: ValueFunction[S],
    gamma: float,
) -> ValueFunction[S]:
    """Apply the Bellman optimality operator T* to a value function.

    (T* V)(s) = max_a Σ_{s',r} p(s',r|s,a)·[r + γ·V(s')]

    Reference: Ch 5, §5.1, Eq (5.1) — "The Bellman optimality operator
    is a contraction mapping with fixed point V*."
    """
    new_vf = ValueFunction[S]()
    for nt in mdp.non_terminal_states:
        best_value = -float("inf")
        for a in mdp.actions(nt):
            q = _compute_q_value(mdp, nt, a, vf, gamma)
            best_value = max(best_value, q)
        new_vf[nt] = best_value
    return new_vf


def bellman_policy_operator(
    mrp: FiniteMarkovRewardProcess[S],
    vf: ValueFunction[S],
    gamma: float,
) -> ValueFunction[S]:
    """Apply the Bellman policy operator T^π to a value function.

    (T^π V)(s) = Σ_{s',r} p^π(s',r|s)·[r + γ·V(s')]

    Reference: Ch 4, §4.1 — "The Bellman expectation operator."
    """
    new_vf = ValueFunction[S]()
    for nt in mrp.non_terminal_states:
        tr_dist = mrp.transition_reward(nt)
        probs = tr_dist.probabilities()
        val = 0.0
        for (next_state, reward), prob in probs.items():
            v_next = 0.0
            if isinstance(next_state, NonTerminal):
                v_next = vf[next_state]
            val += prob * (reward + gamma * v_next)
        new_vf[nt] = val
    return new_vf


def value_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = 10_000,
) -> tuple[ValueFunction[S], DeterministicPolicy[S, A]]:
    """Value Iteration: iterate the Bellman optimality operator to convergence.

    Algorithm:
      1. Initialise V_0(s) = 0 for all s
      2. V_{k+1}(s) = max_a Σ_{s',r} p(s',r|s,a)·[r + γ·V_k(s')]
      3. Stop when ||V_{k+1} - V_k||_∞ < tolerance

    Reference: Ch 5, §5.1, Algorithm 5.1 — "Value Iteration."

    Theorem 5.1 (Contraction): The Bellman optimality operator T* is a
    γ-contraction in the sup-norm. Therefore Value Iteration converges
    to V* at a geometric rate, with error bound:
      ||V_k - V*||_∞ ≤ γ^k · ||V_0 - V*||_∞ / (1-γ)

    Args:
        mdp: Finite MDP to solve.
        gamma: Discount factor. If None, uses mdp.gamma.
        tolerance: Convergence threshold for L∞ norm.
        max_iterations: Maximum iterations.

    Returns:
        (V*, π*) — optimal value function and greedy policy.
    """
    if gamma is None:
        gamma = mdp.gamma

    vf = ValueFunction[S]({nt: 0.0 for nt in mdp.non_terminal_states})

    for iteration in range(max_iterations):
        new_vf = bellman_optimality_operator(mdp, vf, gamma)
        if vf.max_diff(new_vf) < tolerance:
            vf = new_vf
            break
        vf = new_vf

    # Extract optimal policy
    optimal_policy = policy_improvement(mdp, vf)
    return vf, optimal_policy


# ---------------------------------------------------------------------------
# Backward Induction  (Ch 5, §5.3)
# ---------------------------------------------------------------------------


def backward_induction(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float | None = None,
    num_steps: int = 10,
) -> list[tuple[ValueFunction[S], DeterministicPolicy[S, A]]]:
    """Finite-horizon backward induction (Ch 5, §5.3).

    For a finite-horizon MDP with T time steps, compute the optimal
    value function and policy at each step t = T, T-1, ..., 1, 0.

    Algorithm:
      1. V_T(s) = 0 for all s  (no reward after terminal time)
      2. For t = T-1, T-2, ..., 0:
         V_t(s) = max_a Σ_{s',r} p(s',r|s,a)·[r + γ·V_{t+1}(s')]
         π_t(s) = argmax_a  (same expression)

    Reference: Ch 5, §5.3 — "Backward Induction for Finite-Horizon MDPs."

    This is particularly relevant for:
    - Optimal execution (Bertsimas-Lo): fixed horizon T = time_to_close
    - American option pricing: exercise decision at each step
    - Portfolio rebalancing: T = number of rebalance dates

    Args:
        mdp: Finite MDP (same transition structure at each step).
        gamma: Discount factor. If None, uses mdp.gamma.
        num_steps: Number of time steps T.

    Returns:
        List of (V_t, π_t) for t = 0, 1, ..., T-1 (in forward order).
    """
    if gamma is None:
        gamma = mdp.gamma

    # V_T(s) = 0 for all s
    vf_next = ValueFunction[S]({nt: 0.0 for nt in mdp.non_terminal_states})

    results: list[tuple[ValueFunction[S], DeterministicPolicy[S, A]]] = []

    for t in range(num_steps - 1, -1, -1):
        # Compute V_t using V_{t+1}
        vf_t = ValueFunction[S]()
        action_map: dict[S, A] = {}

        for nt in mdp.non_terminal_states:
            best_action: A | None = None
            best_value: float = -float("inf")

            for a in mdp.actions(nt):
                q = _compute_q_value(mdp, nt, a, vf_next, gamma)
                if q > best_value:
                    best_value = q
                    best_action = a

            assert best_action is not None
            vf_t[nt] = best_value
            action_map[nt.state] = best_action

        policy_t = DeterministicPolicy.from_mapping(action_map)
        results.append((vf_t, policy_t))
        vf_next = vf_t

    # Reverse so results[0] = (V_0, π_0), results[-1] = (V_{T-1}, π_{T-1})
    results.reverse()
    return results


# ---------------------------------------------------------------------------
# Least-Squares TD Prediction  (Ch 12 supplement)
# ---------------------------------------------------------------------------


def lstd_prediction(
    transitions: Sequence[tuple[np.ndarray, float, np.ndarray]],
    gamma: float,
    feature_dim: int,
    epsilon_reg: float = 1e-4,
) -> np.ndarray:
    """Least-Squares Temporal Difference (LSTD) prediction.

    Computes the fixed point of projected Bellman equation in a single
    batch pass — no step-size parameter needed.

    Solves:  w = A^{-1} b   where
        A = Σ_t φ(s_t) · (φ(s_t) - γ·φ(s_{t+1}))^T
        b = Σ_t φ(s_t) · r_{t+1}

    The value function approximation is V(s) = φ(s)^T · w.

    This is equivalent to performing TD(0) updates with decreasing
    step sizes in the limit (Bradtke & Barto, 1996).

    Parameters
    ----------
    transitions : sequence of (phi_s, reward, phi_s_next)
        Each transition is a tuple:
            phi_s : np.ndarray, shape (feature_dim,) — features of current state
            reward : float — reward received
            phi_s_next : np.ndarray, shape (feature_dim,) — features of next state
    gamma : float
        Discount factor.
    feature_dim : int
        Dimension of the feature vectors.
    epsilon_reg : float
        Regularization added to diagonal of A for numerical stability.

    Returns
    -------
    w : np.ndarray, shape (feature_dim,)
        Weight vector for the linear value function V(s) ≈ φ(s)^T · w.

    References
    ----------
    Bradtke & Barto (1996), "Linear Least-Squares algorithms for temporal
    difference learning."
    Rao & Jelvis, Ch 12 (LSTD supplement).
    """
    A = np.eye(feature_dim) * epsilon_reg
    b = np.zeros(feature_dim)

    for phi_s, reward, phi_s_next in transitions:
        phi_s = np.asarray(phi_s, dtype=np.float64)
        phi_s_next = np.asarray(phi_s_next, dtype=np.float64)
        A += np.outer(phi_s, phi_s - gamma * phi_s_next)
        b += phi_s * reward

    w = np.linalg.solve(A, b)
    return w
