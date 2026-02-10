"""Markov Processes, Markov Reward Processes, and Markov Decision Processes.

Implements the foundational stochastic process abstractions from:
  "Foundations of Reinforcement Learning with Applications in Finance"
  (Rao & Jelvis, Stanford CME 241), Chapters 2-3.

Chapter 2: Markov Processes and Markov Reward Processes
  - MarkovProcess: state transitions without rewards
  - MarkovRewardProcess: state transitions with associated rewards
  - Finite variants with explicit transition matrices

Chapter 3: Markov Decision Processes
  - MarkovDecisionProcess: the central abstraction of RL
  - Policy: mapping from states to action distributions
  - apply_policy: converting MDP + policy → MRP (Thm 3.2)

All classes use Distribution[A] as the core probability primitive,
supporting both discrete (Categorical) and continuous (Gaussian)
distributions for transitions, rewards, and policy outputs.
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Callable,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)

import numpy as np

__all__ = [
    # Distributions
    "Distribution",
    "Categorical",
    "Gaussian",
    "SampledDistribution",
    "Constant",
    # States
    "State",
    "Terminal",
    "NonTerminal",
    # Markov processes
    "MarkovProcess",
    "MarkovRewardProcess",
    "FiniteMarkovProcess",
    "FiniteMarkovRewardProcess",
    # MDPs
    "MarkovDecisionProcess",
    "FiniteMarkovDecisionProcess",
    # Policies
    "Policy",
    "DeterministicPolicy",
    "TabularPolicy",
    # Composition
    "apply_policy",
    "apply_finite_policy",
]

# ---------------------------------------------------------------------------
# Type variables
# ---------------------------------------------------------------------------
S = TypeVar("S")
A = TypeVar("A")

# ---------------------------------------------------------------------------
# Distribution hierarchy  (Ch 2, §2.1 — probability measures)
# ---------------------------------------------------------------------------


class Distribution(ABC, Generic[A]):
    """Abstract probability distribution over values of type A.

    This is the foundational probability primitive used throughout the MDP
    framework: transition kernels, reward distributions, and policies all
    produce Distribution instances.

    Reference: Rao & Jelvis Ch 2, §2.1 — "A probability distribution on a
    finite set S is a function p: S → [0,1] with Σ_s p(s) = 1."
    """

    @abstractmethod
    def sample(self) -> A:
        """Draw a single sample from the distribution."""

    def sample_n(self, n: int) -> list[A]:
        """Draw *n* independent samples."""
        return [self.sample() for _ in range(n)]

    def expectation(self, f: Callable[[A], float]) -> float:
        """Compute E[f(X)].

        Default implementation uses Monte-Carlo with 10 000 samples.
        Subclasses with closed-form expectations should override.
        """
        n = 10_000
        return sum(f(self.sample()) for _ in range(n)) / n

    def map(self, f: Callable[[A], "B"]) -> Distribution["B"]:  # type: ignore[type-var]
        """Push-forward distribution: if X ~ self, return the law of f(X)."""
        return SampledDistribution(sampler=lambda: f(self.sample()))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(...)"


# Auxiliary type var for map
B = TypeVar("B")


class Categorical(Distribution[A]):
    """Finite discrete distribution over a countable set.

    Stores {outcome: probability} as a mapping.  Probabilities must be
    non-negative and sum to 1 (up to floating-point tolerance).

    Reference: Ch 2, Def 2.1 — "A Markov Process is defined by … a
    transition probability function p(s'|s)."  When the state space is
    finite, p(·|s) is a Categorical distribution.
    """

    def __init__(self, probabilities: Mapping[A, float]) -> None:
        # Validate and normalise
        total = sum(probabilities.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Probabilities must sum to 1.0, got {total:.8f}"
            )
        # Store only non-zero entries
        self._probabilities: dict[A, float] = {
            k: v / total for k, v in probabilities.items() if v > 0
        }
        # Cache for efficient sampling
        self._outcomes: list[A] = list(self._probabilities.keys())
        self._weights: list[float] = list(self._probabilities.values())

    def sample(self) -> A:
        return random.choices(self._outcomes, weights=self._weights, k=1)[0]

    def probabilities(self) -> Mapping[A, float]:
        """Return the full probability mass function."""
        return dict(self._probabilities)

    def expectation(self, f: Callable[[A], float]) -> float:
        """Exact expectation: E[f(X)] = Σ_x p(x)·f(x)."""
        return sum(p * f(x) for x, p in self._probabilities.items())

    @property
    def support(self) -> frozenset[A]:
        return frozenset(self._outcomes)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}: {v:.4f}" for k, v in self._probabilities.items())
        return f"Categorical({{{items}}})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Categorical):
            return NotImplemented
        return self._probabilities == other._probabilities

    def __hash__(self) -> int:
        return hash(frozenset(self._probabilities.items()))


class Gaussian(Distribution[float]):
    """Univariate Normal distribution N(μ, σ²).

    Used for continuous state transitions (e.g. GBM price processes)
    and policy outputs in continuous-action MDPs.
    """

    def __init__(self, mu: float, sigma: float) -> None:
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self.mu = mu
        self.sigma = sigma

    def sample(self) -> float:
        return random.gauss(self.mu, self.sigma)

    def expectation(self, f: Callable[[float], float]) -> float:
        """Monte-Carlo expectation for general f.
        For f(x)=x, returns mu exactly (override for known cases).
        """
        n = 50_000
        return sum(f(random.gauss(self.mu, self.sigma)) for _ in range(n)) / n

    def __repr__(self) -> str:
        return f"Gaussian(μ={self.mu:.4f}, σ={self.sigma:.4f})"


class SampledDistribution(Distribution[A]):
    """Distribution defined only by a sampling function.

    Useful for push-forward distributions and compositions where
    an explicit density is unavailable.
    """

    def __init__(
        self,
        sampler: Callable[[], A],
        expectation_samples: int = 10_000,
    ) -> None:
        self._sampler = sampler
        self._expectation_samples = expectation_samples

    def sample(self) -> A:
        return self._sampler()

    def expectation(self, f: Callable[[A], float]) -> float:
        n = self._expectation_samples
        return sum(f(self._sampler()) for _ in range(n)) / n


class Constant(Distribution[A]):
    """Degenerate distribution that always returns the same value.

    Useful for deterministic transitions and deterministic policies.
    """

    def __init__(self, value: A) -> None:
        self.value = value

    def sample(self) -> A:
        return self.value

    def expectation(self, f: Callable[[A], float]) -> float:
        return f(self.value)

    def probabilities(self) -> Mapping[A, float]:
        """PMF with all mass on the single value."""
        return {self.value: 1.0}

    def __repr__(self) -> str:
        return f"Constant({self.value})"


# ---------------------------------------------------------------------------
# State wrappers  (Ch 2, §2.2 — terminal vs non-terminal states)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class State(Generic[S]):
    """Wrapper distinguishing terminal from non-terminal states.

    In finite-horizon or episodic MDPs, some states are *terminal*:
    once reached, no further transitions occur.  We represent this
    distinction at the type level.

    Reference: Ch 2, Def 2.2 — "A state s is *absorbing* (terminal)
    if p(s|s)=1 and R(s,s)=0."
    """

    state: S

    def on_non_terminal(
        self,
        f: Callable[[NonTerminal[S]], A],
        default: A,
    ) -> A:
        if isinstance(self, NonTerminal):
            return f(self)
        return default


@dataclass(frozen=True)
class NonTerminal(State[S]):
    """A non-terminal state from which transitions are possible."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NonTerminal):
            return self.state == other.state
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.state)

    def __repr__(self) -> str:
        return f"NonTerminal({self.state!r})"


@dataclass(frozen=True)
class Terminal(State[S]):
    """A terminal (absorbing) state — episode ends here."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Terminal):
            return self.state == other.state
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("__terminal__", self.state))

    def __repr__(self) -> str:
        return f"Terminal({self.state!r})"


# ---------------------------------------------------------------------------
# MarkovProcess  (Ch 2, §2.2)
# ---------------------------------------------------------------------------


class MarkovProcess(ABC, Generic[S]):
    """Markov Process (Markov Chain) — a memoryless stochastic process.

    A Markov Process is a tuple (S, P) where:
      - S is a (possibly infinite) state space
      - P: S → Distribution[State[S]] is the transition kernel

    The Markov property: P(S_{t+1} | S_t, S_{t-1}, ...) = P(S_{t+1} | S_t).

    Reference: Ch 2, Def 2.1 and §2.2.
    """

    @abstractmethod
    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        """Transition distribution from a non-terminal state.

        Returns a Distribution over State[S] (which may include Terminal).
        """

    def simulate(
        self, start: NonTerminal[S]
    ) -> Iterator[State[S]]:
        """Generate an infinite trajectory from *start*.

        Yields states s_0, s_1, s_2, ... where s_0 = start.
        Terminates (StopIteration) when a Terminal state is reached.

        Reference: Ch 2, §2.3 — "Simulating a Markov Process."
        """
        state: State[S] = start
        yield state
        while isinstance(state, NonTerminal):
            state = self.transition(state).sample()
            yield state

    def traces(
        self,
        start_distribution: Distribution[NonTerminal[S]],
        num_traces: int,
    ) -> list[list[State[S]]]:
        """Generate multiple independent simulation traces."""
        result: list[list[State[S]]] = []
        for _ in range(num_traces):
            start = start_distribution.sample()
            trace = list(self.simulate(start))
            result.append(trace)
        return result


# ---------------------------------------------------------------------------
# MarkovRewardProcess  (Ch 2, §2.4)
# ---------------------------------------------------------------------------


class MarkovRewardProcess(ABC, Generic[S]):
    """Markov Reward Process — a Markov Process augmented with rewards.

    An MRP is a tuple (S, P, R, γ) where:
      - S is the state space
      - P: S → Distribution[State[S]] is the transition kernel
      - R: S × S → ℝ is the expected reward on transition
      - γ ∈ [0,1] is the discount factor

    We combine P and R into a single method:
      transition_reward: S → Distribution[(State[S], float)]

    Reference: Ch 2, §2.4 — "Markov Reward Processes."

    Key equation (Bellman expectation for MRP):
      V(s) = E[R_{t+1} + γ·V(S_{t+1}) | S_t = s]
           = Σ_{s'} P(s'|s) · [R(s,s') + γ·V(s')]
    """

    @abstractmethod
    def transition_reward(
        self, state: NonTerminal[S]
    ) -> Distribution[tuple[State[S], float]]:
        """Joint distribution over (next_state, reward)."""

    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        """Marginalise out the reward to get the state transition only."""
        return self.transition_reward(state).map(lambda sr: sr[0])

    def simulate_reward(
        self, start: NonTerminal[S]
    ) -> Iterator[tuple[State[S], float]]:
        """Generate trajectory of (state, reward) pairs.

        Yields (s_0, 0), (s_1, r_1), (s_2, r_2), ...
        The first reward is 0 by convention (no transition into start).
        """
        state: State[S] = start
        reward = 0.0
        yield (state, reward)
        while isinstance(state, NonTerminal):
            next_state, reward = self.transition_reward(state).sample()
            state = next_state
            yield (state, reward)


# ---------------------------------------------------------------------------
# FiniteMarkovProcess  (Ch 2, §2.5 — tabular / matrix representation)
# ---------------------------------------------------------------------------


class FiniteMarkovProcess(MarkovProcess[S]):
    """Finite Markov Process with explicit transition matrix.

    For a finite state space {s_1, ..., s_n}, the transition kernel
    is represented as an n×n stochastic matrix P where P[i,j] = P(s_j|s_i).

    Reference: Ch 2, §2.5 — "Finite Markov Processes."
    """

    def __init__(
        self,
        transition_map: Mapping[S, Mapping[S, float] | None],
    ) -> None:
        """
        Args:
            transition_map: {state: {next_state: prob}} for non-terminal states.
                            If the value is None, that state is terminal.
        """
        # Separate terminal and non-terminal states
        non_terminal_states: list[S] = []
        terminal_states: list[S] = []
        for s, trans in transition_map.items():
            if trans is None:
                terminal_states.append(s)
            else:
                non_terminal_states.append(s)

        # Collect all states referenced as targets
        all_states_set: set[S] = set(transition_map.keys())
        for trans in transition_map.values():
            if trans is not None:
                all_states_set.update(trans.keys())

        # Also mark states that appear only as targets and not in
        # transition_map keys as terminal
        for s in all_states_set:
            if s not in transition_map:
                terminal_states.append(s)

        self._non_terminal_states: list[S] = sorted(
            non_terminal_states, key=repr
        )
        self._terminal_states: list[S] = sorted(terminal_states, key=repr)
        self._all_states: list[S] = self._non_terminal_states + self._terminal_states
        self._transition_map = transition_map

        # State → index mapping
        self._state_index: dict[S, int] = {
            s: i for i, s in enumerate(self._all_states)
        }

        # Build transition matrix
        n = len(self._all_states)
        self._transition_matrix = np.zeros((n, n), dtype=np.float64)
        for s in self._non_terminal_states:
            i = self._state_index[s]
            trans = transition_map[s]
            assert trans is not None
            for s_next, prob in trans.items():
                j = self._state_index[s_next]
                self._transition_matrix[i, j] = prob
        # Terminal states: self-loop with prob 1
        for s in self._terminal_states:
            i = self._state_index[s]
            self._transition_matrix[i, i] = 1.0

    @property
    def non_terminal_states(self) -> Sequence[NonTerminal[S]]:
        return [NonTerminal(s) for s in self._non_terminal_states]

    @property
    def all_states(self) -> Sequence[S]:
        return list(self._all_states)

    @property
    def transition_matrix(self) -> np.ndarray:
        """n×n stochastic matrix P[i,j] = P(s_j | s_i)."""
        return self._transition_matrix.copy()

    def transition(self, state: NonTerminal[S]) -> Categorical[State[S]]:
        s = state.state
        trans = self._transition_map.get(s)
        if trans is None:
            raise ValueError(f"Cannot transition from terminal state {s}")
        result: dict[State[S], float] = {}
        for s_next, prob in trans.items():
            if self._is_terminal(s_next):
                result[Terminal(s_next)] = prob
            else:
                result[NonTerminal(s_next)] = prob
        return Categorical(result)

    def _is_terminal(self, s: S) -> bool:
        trans = self._transition_map.get(s)
        return trans is None or s not in self._transition_map

    def stationary_distribution(self) -> Categorical[S]:
        """Compute the stationary distribution π such that π = π·P.

        Solves the left eigenvector problem for eigenvalue 1.
        Only valid for ergodic (irreducible, aperiodic) chains.
        """
        P = self._transition_matrix
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = pi / pi.sum()
        return Categorical({s: float(pi[i]) for i, s in enumerate(self._all_states)})


# ---------------------------------------------------------------------------
# FiniteMarkovRewardProcess  (Ch 2, §2.6)
# ---------------------------------------------------------------------------


class FiniteMarkovRewardProcess(MarkovRewardProcess[S]):
    """Finite MRP with explicit transition-reward structure.

    transition_reward_map: {state: {(next_state, reward): probability}}
    For terminal states, set value to None.

    Reference: Ch 2, §2.6 — "Finite Markov Reward Processes."

    The Bellman equation for the value function:
      V(s) = Σ_{s',r} p(s',r|s) · [r + γ·V(s')]

    For finite MRPs, this is a linear system:  V = R + γ·P·V
    whose direct solution is:  V = (I - γP)^{-1} · R
    """

    def __init__(
        self,
        transition_reward_map: Mapping[
            S, Mapping[tuple[S, float], float] | None
        ],
    ) -> None:
        """
        Args:
            transition_reward_map: {state: {(next_state, reward): prob} | None}
                None values indicate terminal states.
        """
        self._transition_reward_map = transition_reward_map

        # Identify non-terminal and terminal states
        non_terminal: list[S] = []
        terminal: list[S] = []
        all_states_set: set[S] = set(transition_reward_map.keys())

        for s, transitions in transition_reward_map.items():
            if transitions is None:
                terminal.append(s)
            else:
                non_terminal.append(s)
                for (s_next, _r), _p in transitions.items():
                    all_states_set.add(s_next)

        for s in all_states_set:
            if s not in transition_reward_map:
                terminal.append(s)

        self._non_terminal_states = sorted(non_terminal, key=repr)
        self._terminal_states = sorted(set(terminal), key=repr)
        self._all_states = self._non_terminal_states + self._terminal_states

        self._state_index: dict[S, int] = {
            s: i for i, s in enumerate(self._all_states)
        }

        # Build transition matrix and reward vector (for non-terminal states)
        n_nt = len(self._non_terminal_states)
        n_all = len(self._all_states)
        self._transition_matrix = np.zeros((n_nt, n_all), dtype=np.float64)
        self._reward_vector = np.zeros(n_nt, dtype=np.float64)

        for idx_nt, s in enumerate(self._non_terminal_states):
            transitions = transition_reward_map[s]
            assert transitions is not None
            for (s_next, reward), prob in transitions.items():
                j = self._state_index[s_next]
                self._transition_matrix[idx_nt, j] += prob
                self._reward_vector[idx_nt] += prob * reward

    @property
    def non_terminal_states(self) -> Sequence[NonTerminal[S]]:
        return [NonTerminal(s) for s in self._non_terminal_states]

    @property
    def all_states(self) -> Sequence[S]:
        return list(self._all_states)

    @property
    def transition_matrix(self) -> np.ndarray:
        """Transition matrix P of shape (n_non_terminal, n_all).

        P[i, j] = Σ_r p(s_j, r | s_i), i.e. marginalised over rewards.
        """
        return self._transition_matrix.copy()

    @property
    def reward_vector(self) -> np.ndarray:
        """Expected immediate reward for each non-terminal state.

        R[i] = Σ_{s',r} p(s',r|s_i)·r = E[R_{t+1} | S_t = s_i].
        """
        return self._reward_vector.copy()

    def transition_reward(
        self, state: NonTerminal[S]
    ) -> Categorical[tuple[State[S], float]]:
        s = state.state
        transitions = self._transition_reward_map.get(s)
        if transitions is None:
            raise ValueError(f"Cannot transition from terminal state {s}")
        result: dict[tuple[State[S], float], float] = {}
        for (s_next, reward), prob in transitions.items():
            is_term = (
                self._transition_reward_map.get(s_next) is None
                or s_next not in self._transition_reward_map
            )
            next_state: State[S] = Terminal(s_next) if is_term else NonTerminal(s_next)
            key = (next_state, reward)
            result[key] = result.get(key, 0.0) + prob
        return Categorical(result)

    def direct_solve(self, gamma: float) -> Mapping[NonTerminal[S], float]:
        """Solve the Bellman equation directly: V = (I - γP_nt)^{-1} · R.

        Where P_nt is the (n_nt × n_nt) sub-matrix of transitions
        among non-terminal states only.

        Reference: Ch 2, §2.6, Eq (2.9).

        This is O(n^3) but exact — use for small state spaces.
        """
        n_nt = len(self._non_terminal_states)
        # Extract P restricted to non-terminal → non-terminal
        P_nt = self._transition_matrix[:, :n_nt]
        R = self._reward_vector

        # V = (I - γ·P_nt)^{-1} · R
        I = np.eye(n_nt, dtype=np.float64)
        V = np.linalg.solve(I - gamma * P_nt, R)

        return {
            NonTerminal(self._non_terminal_states[i]): float(V[i])
            for i in range(n_nt)
        }


# ---------------------------------------------------------------------------
# MarkovDecisionProcess  (Ch 3, §3.1)
# ---------------------------------------------------------------------------


class MarkovDecisionProcess(ABC, Generic[S, A]):
    """Markov Decision Process — the central abstraction of RL.

    An MDP is a tuple (S, A, P, R, γ) where:
      - S is the state space
      - A is the action space (possibly state-dependent)
      - P(s'|s,a) is the transition kernel
      - R(s,a,s') is the reward function
      - γ ∈ [0,1] is the discount factor

    The agent selects actions via a policy π: S → Distribution[A],
    and the goal is to maximise the expected discounted return:
      V^π(s) = E_π[Σ_{t=0}^∞ γ^t · R_{t+1} | S_0 = s]

    Reference: Ch 3, §3.1 — "Markov Decision Processes."
    """

    @abstractmethod
    def actions(self, state: NonTerminal[S]) -> Iterable[A]:
        """Available actions in a given non-terminal state."""

    @abstractmethod
    def step(
        self, state: NonTerminal[S], action: A
    ) -> Distribution[tuple[State[S], float]]:
        """Execute action in state, returning Distribution[(next_state, reward)].

        This combines the transition kernel P(s'|s,a) and reward R(s,a,s')
        into a single joint distribution.

        Reference: Ch 3, §3.1, Eq (3.1):
          p(s',r|s,a) = P(S_{t+1}=s', R_{t+1}=r | S_t=s, A_t=a)
        """

    @property
    @abstractmethod
    def gamma(self) -> float:
        """Discount factor γ ∈ [0,1]."""

    def simulate(
        self,
        start: NonTerminal[S],
        policy: Policy[S, A],
    ) -> Iterator[tuple[State[S], A, float]]:
        """Generate trajectory: (s_0, a_0, r_1), (s_1, a_1, r_2), ...

        Terminates when a Terminal state is reached.
        """
        state: State[S] = start
        while isinstance(state, NonTerminal):
            action = policy.act(state).sample()
            next_state, reward = self.step(state, action).sample()
            yield (state, action, reward)
            state = next_state


# ---------------------------------------------------------------------------
# FiniteMarkovDecisionProcess  (Ch 3, §3.2)
# ---------------------------------------------------------------------------


class FiniteMarkovDecisionProcess(MarkovDecisionProcess[S, A]):
    """Finite MDP with explicit transition-reward tables.

    The transition structure is provided as a nested mapping:
      {state: {action: {(next_state, reward): probability}}}

    Reference: Ch 3, §3.2 — "Finite Markov Decision Processes."

    Key equations:
      Q^π(s,a) = Σ_{s',r} p(s',r|s,a)·[r + γ·V^π(s')]     (3.5)
      V^π(s) = Σ_a π(a|s)·Q^π(s,a)                          (3.6)
      V*(s) = max_a Q*(s,a)                                   (3.8)
      Q*(s,a) = Σ_{s',r} p(s',r|s,a)·[r + γ·V*(s')]         (3.9)
    """

    def __init__(
        self,
        transition_map: Mapping[
            S, Mapping[A, Mapping[tuple[S, float], float] | None]
        ],
        gamma: float,
    ) -> None:
        """
        Args:
            transition_map: {state: {action: {(next_state, reward): prob}}}
                States with empty or None action maps are terminal.
            gamma: discount factor ∈ [0, 1].
        """
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0,1], got {gamma}")

        self._transition_map = transition_map
        self._gamma = gamma

        # Collect non-terminal states (those that have at least one action)
        non_terminal: list[S] = []
        terminal_set: set[S] = set()
        all_states_set: set[S] = set(transition_map.keys())

        for s, action_map in transition_map.items():
            has_actions = False
            for a, trans in action_map.items():
                if trans is not None and len(trans) > 0:
                    has_actions = True
                    for (s_next, _r), _p in trans.items():
                        all_states_set.add(s_next)
            if has_actions:
                non_terminal.append(s)
            else:
                terminal_set.add(s)

        for s in all_states_set:
            if s not in transition_map:
                terminal_set.add(s)

        self._non_terminal_states = sorted(non_terminal, key=repr)
        self._terminal_states = sorted(terminal_set, key=repr)
        self._all_states = self._non_terminal_states + self._terminal_states
        self._terminal_set = terminal_set

        self._state_index: dict[S, int] = {
            s: i for i, s in enumerate(self._all_states)
        }

        # Precompute action lists per state
        self._actions_map: dict[S, list[A]] = {}
        for s in self._non_terminal_states:
            action_map = transition_map[s]
            self._actions_map[s] = [
                a for a, trans in action_map.items()
                if trans is not None and len(trans) > 0
            ]

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def non_terminal_states(self) -> Sequence[NonTerminal[S]]:
        return [NonTerminal(s) for s in self._non_terminal_states]

    @property
    def all_states(self) -> Sequence[S]:
        return list(self._all_states)

    def actions(self, state: NonTerminal[S]) -> Sequence[A]:
        return self._actions_map.get(state.state, [])

    def step(
        self, state: NonTerminal[S], action: A
    ) -> Categorical[tuple[State[S], float]]:
        s = state.state
        action_map = self._transition_map.get(s)
        if action_map is None:
            raise ValueError(f"State {s} not in transition map")
        transitions = action_map.get(action)
        if transitions is None:
            raise ValueError(f"Action {action} not available in state {s}")

        result: dict[tuple[State[S], float], float] = {}
        for (s_next, reward), prob in transitions.items():
            is_term = s_next in self._terminal_set
            next_state: State[S] = Terminal(s_next) if is_term else NonTerminal(s_next)
            key = (next_state, reward)
            result[key] = result.get(key, 0.0) + prob
        return Categorical(result)

    def apply_policy(self, policy: Policy[S, A]) -> FiniteMarkovRewardProcess[S]:
        """Apply a policy to this MDP to produce a FiniteMarkovRewardProcess.

        Reference: Ch 3, Thm 3.2 — "Applying a policy π to an MDP yields
        an MRP with transition kernel p^π(s'|s) = Σ_a π(a|s)·p(s'|s,a)
        and reward r^π(s,s') = Σ_a π(a|s)·r(s,a,s')."
        """
        transition_reward_map: dict[S, Mapping[tuple[S, float], float] | None] = {}

        for s in self._non_terminal_states:
            nt = NonTerminal(s)
            action_dist = policy.act(nt)
            action_probs = action_dist.probabilities()

            combined: dict[tuple[S, float], float] = defaultdict(float)
            for a, pi_a in action_probs.items():
                if pi_a <= 0:
                    continue
                action_map = self._transition_map[s]
                transitions = action_map.get(a)
                if transitions is None:
                    continue
                for (s_next, reward), prob in transitions.items():
                    combined[(s_next, reward)] += pi_a * prob

            transition_reward_map[s] = dict(combined)

        # Add terminal states
        for s in self._terminal_states:
            transition_reward_map[s] = None

        return FiniteMarkovRewardProcess(transition_reward_map)


# ---------------------------------------------------------------------------
# Policy  (Ch 3, §3.3)
# ---------------------------------------------------------------------------


class Policy(ABC, Generic[S, A]):
    """A policy π: S → Distribution[A].

    Maps each state to a probability distribution over actions.
    The agent samples from this distribution to select actions.

    Reference: Ch 3, §3.3 — "Policies."
    """

    @abstractmethod
    def act(self, state: NonTerminal[S]) -> Distribution[A]:
        """Return the action distribution for a given state."""


class DeterministicPolicy(Policy[S, A]):
    """Deterministic policy: each state maps to exactly one action.

    π(s) = a   (with probability 1)

    Reference: Ch 3, §3.3 — "A deterministic policy is a degenerate case
    where π(a|s) = 1 for exactly one action a in each state s."
    """

    def __init__(self, action_for: Callable[[NonTerminal[S]], A]) -> None:
        self._action_for = action_for

    def act(self, state: NonTerminal[S]) -> Constant[A]:
        return Constant(self._action_for(state))

    @staticmethod
    def from_mapping(mapping: Mapping[S, A]) -> DeterministicPolicy[S, A]:
        """Create from a state→action dictionary."""
        return DeterministicPolicy(lambda s, m=mapping: m[s.state])


class TabularPolicy(Policy[S, A]):
    """Tabular stochastic policy with explicit probability tables.

    Stores π(a|s) as a nested dict: {state: {action: probability}}.

    Reference: Ch 3, §3.3.
    """

    def __init__(
        self, policy_map: Mapping[S, Mapping[A, float]]
    ) -> None:
        self._policy_map = policy_map

    def act(self, state: NonTerminal[S]) -> Categorical[A]:
        action_probs = self._policy_map.get(state.state)
        if action_probs is None:
            raise ValueError(f"No policy entry for state {state.state}")
        return Categorical(action_probs)

    @property
    def policy_map(self) -> Mapping[S, Mapping[A, float]]:
        return self._policy_map


# ---------------------------------------------------------------------------
# Composition: apply_policy  (Ch 3, Thm 3.2)
# ---------------------------------------------------------------------------


def apply_policy(
    mdp: MarkovDecisionProcess[S, A],
    policy: Policy[S, A],
) -> MarkovRewardProcess[S]:
    """Apply a policy to an MDP to produce an MRP.

    For each state s, the MRP transition-reward distribution is:
      p^π(s',r|s) = Σ_a π(a|s) · p(s',r|s,a)

    Reference: Ch 3, Thm 3.2.

    For FiniteMarkovDecisionProcess, use apply_finite_policy() instead
    to get a FiniteMarkovRewardProcess.
    """

    class _InducedMRP(MarkovRewardProcess[S]):
        def transition_reward(
            self, state: NonTerminal[S]
        ) -> Distribution[tuple[State[S], float]]:
            action_dist = policy.act(state)

            def _sample() -> tuple[State[S], float]:
                a = action_dist.sample()
                return mdp.step(state, a).sample()

            return SampledDistribution(_sample)

    return _InducedMRP()


def apply_finite_policy(
    mdp: FiniteMarkovDecisionProcess[S, A],
    policy: Policy[S, A],
) -> FiniteMarkovRewardProcess[S]:
    """Apply a policy to a FiniteMarkovDecisionProcess.

    Returns a FiniteMarkovRewardProcess with exact transition-reward tables.

    Reference: Ch 3, Thm 3.2 (tabular case).
    """
    return mdp.apply_policy(policy)
