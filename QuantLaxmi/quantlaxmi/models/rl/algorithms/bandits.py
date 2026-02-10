"""Multi-Armed Bandits and Contextual Bandits.

Implements Chapter 15 of "Foundations of RL with Applications in Finance"
by Ashwin Rao & Tikhon Jelvis.

The multi-armed bandit problem is the simplest form of the exploration-
exploitation trade-off: at each step, the agent selects one of K arms
and observes a reward drawn from that arm's (unknown) distribution.
The goal is to maximize cumulative reward.

This is a single-state MDP (no state transitions), making it the ideal
setting to study exploration strategies in isolation.

Key algorithms:
    - Epsilon-Greedy (Section 15.2) — the simplest exploration strategy
    - UCB1 (Section 15.3) — optimism in the face of uncertainty
    - Gradient Bandit (Section 15.4) — policy gradient for bandits
    - Thompson Sampling (Section 15.5) — Bayesian exploration via posterior sampling
    - Contextual Bandits (Section 15.6) — when context/features are available

Regret theory:
    The Lai-Robbins lower bound (Theorem 15.1) states that any consistent
    policy must incur at least O(log T) regret:
        lim inf_{T->inf} R_T / log T >= sum_{a: mu_a < mu*} (mu* - mu_a) / KL(P_a || P*)
    UCB1 and Thompson Sampling both achieve this optimal logarithmic rate.

References:
    Rao & Jelvis, Ch 15 (Multi-Armed Bandits)
    Auer, Cesa-Bianchi, Fischer (2002) "Finite-time analysis of the MAB problem"
    Thompson (1933) "On the likelihood that one unknown probability exceeds another"
    Agrawal & Goyal (2012) "Analysis of Thompson Sampling for the MAB Problem"
    Li et al. (2010) "A Contextual-Bandit Approach to Personalized News Article Recommendation"
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
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
    "BanditArm",
    "GaussianArm",
    "BernoulliArm",
    "BanditAlgorithm",
    "EpsilonGreedy",
    "UCB1",
    "ThompsonSampling",
    "GradientBandit",
    "ContextualBandit",
    "LinUCB",
    "NeuralThompsonSampling",
]


# =====================================================================
# Bandit Arms
# =====================================================================


class BanditArm(ABC):
    """Abstract bandit arm with a stochastic reward distribution."""

    @abstractmethod
    def pull(self) -> float:
        """Draw a reward from this arm's distribution."""

    @abstractmethod
    def mean(self) -> float:
        """Return the true mean reward (for regret computation)."""


class GaussianArm(BanditArm):
    """Arm with Gaussian reward distribution N(mu, sigma^2).

    Parameters
    ----------
    mu : float
        True mean of the reward distribution.
    sigma : float
        Standard deviation.
    seed : int or None
        Random seed.
    """

    def __init__(self, mu: float, sigma: float = 1.0, seed: int | None = None) -> None:
        self._mu = mu
        self._sigma = sigma
        self._rng = np.random.default_rng(seed)

    def pull(self) -> float:
        return float(self._rng.normal(self._mu, self._sigma))

    def mean(self) -> float:
        return self._mu


class BernoulliArm(BanditArm):
    """Arm with Bernoulli reward distribution Ber(p).

    Useful for modeling click-through rates, conversion rates, etc.

    Parameters
    ----------
    p : float
        Probability of reward = 1.
    seed : int or None
    """

    def __init__(self, p: float, seed: int | None = None) -> None:
        assert 0.0 <= p <= 1.0, "p must be in [0, 1]"
        self._p = p
        self._rng = np.random.default_rng(seed)

    def pull(self) -> float:
        return float(self._rng.random() < self._p)

    def mean(self) -> float:
        return self._p


# =====================================================================
# BanditAlgorithm (ABC)
# =====================================================================


class BanditAlgorithm(ABC):
    """Abstract base class for bandit algorithms.

    Defines the interface: select_arm() and update().
    """

    @abstractmethod
    def select_arm(self) -> int:
        """Select which arm to pull (0-indexed)."""

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """Update internal estimates after observing (arm, reward)."""

    def run(
        self,
        arms: Sequence[BanditArm],
        num_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the bandit algorithm for ``num_steps`` rounds.

        Parameters
        ----------
        arms : sequence of BanditArm
        num_steps : int

        Returns
        -------
        (rewards, regrets)
            Arrays of shape (num_steps,) with per-step reward and per-step regret.
        """
        best_mean = max(a.mean() for a in arms)
        rewards = np.zeros(num_steps, dtype=np.float64)
        regrets = np.zeros(num_steps, dtype=np.float64)

        for t in range(num_steps):
            arm_idx = self.select_arm()
            reward = arms[arm_idx].pull()
            self.update(arm_idx, reward)
            rewards[t] = reward
            regrets[t] = best_mean - arms[arm_idx].mean()

        return rewards, regrets


# =====================================================================
# Epsilon-Greedy (Ch 15.2)
# =====================================================================


class EpsilonGreedy(BanditAlgorithm):
    """Epsilon-Greedy Bandit (Ch 15.2).

    The simplest exploration strategy:
        - With probability epsilon: select a random arm (explore)
        - With probability 1 - epsilon: select the arm with highest estimated mean (exploit)

    Sample-average update:
        Q_{n+1}(a) = Q_n(a) + (1/N_t(a)) * (R_t - Q_n(a))

    With decaying epsilon (e.g., epsilon_t = 1/t), convergence to the optimal
    arm is guaranteed. However, the exploration rate decays slowly, leading
    to higher regret than UCB or Thompson Sampling.

    Regret: O(epsilon * T) for constant epsilon (linear regret).
            O(log T) with epsilon_t = c * K / (d^2 * t) for appropriate c, d.

    Parameters
    ----------
    num_arms : int
    epsilon : float
        Exploration probability.
    decay : float or None
        If provided, epsilon <- epsilon * decay after each step.
    seed : int
    """

    def __init__(
        self,
        num_arms: int,
        epsilon: float = 0.1,
        decay: float | None = None,
        seed: int = 42,
    ) -> None:
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.decay = decay
        self._q_values = np.zeros(num_arms, dtype=np.float64)
        self._counts = np.zeros(num_arms, dtype=np.int64)
        self._rng = np.random.default_rng(seed)

    def select_arm(self) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.num_arms))
        # Break ties randomly
        max_q = self._q_values.max()
        best = np.where(np.isclose(self._q_values, max_q))[0]
        return int(self._rng.choice(best))

    def update(self, arm: int, reward: float) -> None:
        self._counts[arm] += 1
        n = self._counts[arm]
        self._q_values[arm] += (1.0 / n) * (reward - self._q_values[arm])
        if self.decay is not None:
            self.epsilon *= self.decay


# =====================================================================
# UCB1 — Upper Confidence Bound (Ch 15.3)
# =====================================================================


class UCB1(BanditAlgorithm):
    """Upper Confidence Bound (UCB1) Algorithm (Ch 15.3).

    Principle: "Optimism in the face of uncertainty" (OFU).

    Selection rule:
        A_t = argmax_a [Q_t(a) + c * sqrt(log(t) / N_t(a))]

    The confidence bound sqrt(log(t) / N_t(a)) derives from the Chernoff-Hoeffding
    inequality and ensures that under-explored arms have inflated estimates,
    naturally driving exploration.

    Theorem 15.2 (Auer et al. 2002):
        UCB1 achieves logarithmic regret:
        E[R_T] <= sum_{a: mu_a < mu*} [8*log(T)/(mu* - mu_a)] + (1 + pi^2/3) * sum(mu* - mu_a)

    This matches the Lai-Robbins lower bound up to constant factors.

    Parameters
    ----------
    num_arms : int
    c : float
        Exploration coefficient. c=sqrt(2) is the theoretical optimum for
        rewards in [0, 1]. Higher c encourages more exploration.
    seed : int
    """

    def __init__(
        self,
        num_arms: int,
        c: float = np.sqrt(2),
        seed: int = 42,
    ) -> None:
        self.num_arms = num_arms
        self.c = c
        self._q_values = np.zeros(num_arms, dtype=np.float64)
        self._counts = np.zeros(num_arms, dtype=np.int64)
        self._total_count = 0
        self._rng = np.random.default_rng(seed)

    def select_arm(self) -> int:
        """Select arm using UCB1.

        If any arm has zero pulls, select it first (initialization phase).
        Otherwise: A_t = argmax_a [Q_t(a) + c * sqrt(log(t) / N_t(a))].
        """
        # Ensure every arm is pulled at least once
        for a in range(self.num_arms):
            if self._counts[a] == 0:
                return a

        ucb_values = self._q_values + self.c * np.sqrt(
            np.log(self._total_count) / self._counts
        )

        max_ucb = ucb_values.max()
        best = np.where(np.isclose(ucb_values, max_ucb))[0]
        return int(self._rng.choice(best))

    def update(self, arm: int, reward: float) -> None:
        self._total_count += 1
        self._counts[arm] += 1
        n = self._counts[arm]
        self._q_values[arm] += (1.0 / n) * (reward - self._q_values[arm])


# =====================================================================
# Thompson Sampling (Ch 15.5)
# =====================================================================


class ThompsonSampling(BanditAlgorithm):
    """Thompson Sampling with Normal-Inverse-Gamma conjugate priors (Ch 15.5).

    Thompson Sampling (Thompson 1933) is a Bayesian approach: at each step,
    sample a parameter from the posterior distribution of each arm, then
    select the arm with the highest sampled value.

    For Gaussian rewards with unknown mean AND variance, the conjugate prior
    is the Normal-Inverse-Gamma (NIG) distribution:

        sigma^2 ~ InvGamma(alpha, beta)
        mu | sigma^2 ~ N(mu_0, sigma^2 / lambda)

    Posterior update after observing rewards x_1, ..., x_n from arm a:
        mu_n    = (lambda_0 * mu_0 + n * x_bar) / (lambda_0 + n)
        lambda_n = lambda_0 + n
        alpha_n  = alpha_0 + n / 2
        beta_n   = beta_0 + (n*s^2 + lambda_0*n*(x_bar - mu_0)^2 / (lambda_0 + n)) / 2

    where x_bar = sample mean, s^2 = sample variance.

    At each step:
        1. For each arm a, sample sigma^2_a ~ InvGamma(alpha_a, beta_a)
        2. Sample mu_a ~ N(mu_a, sigma^2_a / lambda_a)
        3. Select arm with highest sampled mu_a

    Theorem 15.3 (Agrawal & Goyal 2012):
        Thompson Sampling achieves the Lai-Robbins optimal logarithmic regret
        bound for Bernoulli bandits. Analogous results hold for Gaussian bandits.

    Parameters
    ----------
    num_arms : int
    prior_mu : float
        Prior mean mu_0.
    prior_lambda : float
        Prior precision scaling lambda_0 (higher = more confident in prior mean).
    prior_alpha : float
        Prior shape parameter alpha_0 for InvGamma (must be > 0).
    prior_beta : float
        Prior scale parameter beta_0 for InvGamma (must be > 0).
    seed : int
    """

    def __init__(
        self,
        num_arms: int,
        prior_mu: float = 0.0,
        prior_lambda: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.num_arms = num_arms
        self._rng = np.random.default_rng(seed)

        # NIG parameters per arm
        self._mu = np.full(num_arms, prior_mu, dtype=np.float64)
        self._lam = np.full(num_arms, prior_lambda, dtype=np.float64)
        self._alpha = np.full(num_arms, prior_alpha, dtype=np.float64)
        self._beta = np.full(num_arms, prior_beta, dtype=np.float64)

        # Sufficient statistics
        self._counts = np.zeros(num_arms, dtype=np.int64)
        self._sum = np.zeros(num_arms, dtype=np.float64)
        self._sum_sq = np.zeros(num_arms, dtype=np.float64)

    def select_arm(self) -> int:
        """Select arm via Thompson Sampling.

        1. Sample sigma^2_a ~ InvGamma(alpha_a, beta_a) for each arm.
        2. Sample mu_a ~ N(mu_a, sigma^2_a / lambda_a) for each arm.
        3. Return argmax_a mu_a.
        """
        sampled_means = np.zeros(self.num_arms, dtype=np.float64)

        for a in range(self.num_arms):
            # Sample variance from Inverse-Gamma(alpha, beta)
            # InvGamma(a, b) = 1 / Gamma(a, 1/b)
            # numpy's gamma uses shape/scale: Gamma(shape=alpha, scale=1/beta)
            precision = self._rng.gamma(self._alpha[a], 1.0 / self._beta[a])
            if precision < 1e-15:
                precision = 1e-15
            sigma_sq = 1.0 / precision

            # Sample mean from N(mu, sigma^2 / lambda)
            std = np.sqrt(sigma_sq / self._lam[a])
            sampled_means[a] = self._rng.normal(self._mu[a], std)

        return int(np.argmax(sampled_means))

    def update(self, arm: int, reward: float) -> None:
        """Update the NIG posterior for the selected arm.

        Posterior update formulas (conjugate update):
            mu_n    = (lambda_0 * mu_0 + n * x_bar) / (lambda_0 + n)
            lambda_n = lambda_0 + n
            alpha_n  = alpha_0 + n/2
            beta_n   = beta_0 + [n*s^2 + lambda_0*n*(x_bar - mu_0)^2 / (lambda_0 + n)] / 2

        For incremental single-observation update (n_new = n_old + 1):
        """
        a = arm
        n_old = self._counts[a]
        self._counts[a] += 1
        self._sum[a] += reward
        self._sum_sq[a] += reward * reward
        n = self._counts[a]

        x_bar = self._sum[a] / n

        # Recompute posterior from sufficient statistics
        # Using original prior (stored implicitly through current params)
        # Incremental update formulas:
        mu_old = self._mu[a]
        lam_old = self._lam[a]

        lam_new = lam_old + 1.0
        mu_new = (lam_old * mu_old + reward) / lam_new
        alpha_new = self._alpha[a] + 0.5
        beta_new = self._beta[a] + 0.5 * lam_old * (reward - mu_old) ** 2 / lam_new

        self._mu[a] = mu_new
        self._lam[a] = lam_new
        self._alpha[a] = alpha_new
        self._beta[a] = beta_new


# =====================================================================
# Gradient Bandit (Ch 15.4)
# =====================================================================


class GradientBandit(BanditAlgorithm):
    """Gradient Bandit Algorithm (Ch 15.4).

    Maintains a preference H_t(a) for each arm, and derives the policy
    via softmax:
        pi_t(a) = exp(H_t(a)) / sum_b exp(H_t(b))

    Update rule (stochastic gradient ascent on expected reward):
        H_{t+1}(A_t) = H_t(A_t) + alpha * (R_t - R_bar_t) * (1 - pi_t(A_t))
        H_{t+1}(a)   = H_t(a)   - alpha * (R_t - R_bar_t) * pi_t(a)    for a != A_t

    where R_bar_t is the running average reward (baseline for variance reduction).

    The update is an instance of the policy gradient theorem applied to the
    bandit setting (single-state MDP).  The baseline R_bar_t does not introduce
    bias but reduces variance.

    Theorem (Williams 1992 / Ch 15.4):
        E[Delta H_t(a)] = alpha * partial/partial H_t(a) E[R_t]
        i.e., the expected update is proportional to the gradient of expected reward.

    Parameters
    ----------
    num_arms : int
    alpha : float
        Learning rate / step size.
    use_baseline : bool
        If True, use running average reward as baseline.
    seed : int
    """

    def __init__(
        self,
        num_arms: int,
        alpha: float = 0.1,
        use_baseline: bool = True,
        seed: int = 42,
    ) -> None:
        self.num_arms = num_arms
        self.alpha = alpha
        self.use_baseline = use_baseline
        self._preferences = np.zeros(num_arms, dtype=np.float64)
        self._avg_reward = 0.0
        self._total_steps = 0
        self._rng = np.random.default_rng(seed)

    def _softmax(self) -> np.ndarray:
        """Compute softmax probabilities from preferences (numerically stable)."""
        h = self._preferences - self._preferences.max()  # shift for stability
        exp_h = np.exp(h)
        return exp_h / exp_h.sum()

    def select_arm(self) -> int:
        """Select arm by sampling from softmax policy."""
        probs = self._softmax()
        return int(self._rng.choice(self.num_arms, p=probs))

    def update(self, arm: int, reward: float) -> None:
        """Update preferences using the gradient bandit update rule.

        H_{t+1}(A_t) = H_t(A_t) + alpha * (R_t - R_bar) * (1 - pi(A_t))
        H_{t+1}(a)   = H_t(a)   - alpha * (R_t - R_bar) * pi(a)  for a != A_t
        """
        self._total_steps += 1

        baseline = self._avg_reward if self.use_baseline else 0.0
        probs = self._softmax()
        delta = reward - baseline

        # Update preferences
        for a in range(self.num_arms):
            if a == arm:
                self._preferences[a] += self.alpha * delta * (1.0 - probs[a])
            else:
                self._preferences[a] -= self.alpha * delta * probs[a]

        # Update running average reward (baseline)
        self._avg_reward += (1.0 / self._total_steps) * (reward - self._avg_reward)


# =====================================================================
# Contextual Bandits — Base (Ch 15.6)
# =====================================================================


class ContextualBandit(ABC):
    """Abstract base class for contextual bandits (Ch 15.6).

    In the contextual bandit setting, the agent observes a context vector x_t
    before selecting an arm.  The reward depends on both the context and the arm:
        R_t = f(x_t, A_t) + noise

    This is a generalization of the MAB problem that models personalized
    decision-making (e.g., recommending news articles based on user features).

    The agent must learn to map contexts to arms to maximize cumulative reward.
    """

    @abstractmethod
    def select_arm(self, context: np.ndarray) -> int:
        """Select arm given a context vector."""

    @abstractmethod
    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """Update model after observing (context, arm, reward)."""

    def run(
        self,
        contexts: np.ndarray,
        reward_func: Callable[[np.ndarray, int], float],
        num_arms: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the contextual bandit for a sequence of contexts.

        Parameters
        ----------
        contexts : np.ndarray of shape (T, d)
        reward_func : callable (context, arm) -> reward
        num_arms : int

        Returns
        -------
        (rewards, arms_chosen)
        """
        T = len(contexts)
        rewards = np.zeros(T, dtype=np.float64)
        arms_chosen = np.zeros(T, dtype=np.int64)

        for t in range(T):
            arm = self.select_arm(contexts[t])
            reward = reward_func(contexts[t], arm)
            self.update(contexts[t], arm, reward)
            rewards[t] = reward
            arms_chosen[t] = arm

        return rewards, arms_chosen


# =====================================================================
# LinUCB — Linear UCB for Contextual Bandits (Ch 15.6)
# =====================================================================


class LinUCB(ContextualBandit):
    """Linear UCB for Contextual Bandits (Li et al. 2010).

    Assumes the expected reward is linear in the context:
        E[R | x, a] = theta_a^T * x

    For each arm a, maintains:
        A_a = I_d + sum_{t: A_t=a} x_t * x_t^T    (d x d matrix)
        b_a = sum_{t: A_t=a} x_t * R_t              (d-vector)
        theta_hat_a = A_a^{-1} * b_a                 (ridge regression estimate)

    Selection rule (UCB):
        A_t = argmax_a [theta_hat_a^T * x_t + alpha * sqrt(x_t^T * A_a^{-1} * x_t)]

    The confidence width sqrt(x^T A^{-1} x) is the norm of x in the inverse
    design matrix, which naturally accounts for the uncertainty in directions
    that have been less explored.

    Theorem (Abbasi-Yadkori et al. 2011):
        LinUCB achieves regret O(d * sqrt(T * log T)) with high probability,
        where d is the context dimension.

    Parameters
    ----------
    context_dim : int
        Dimensionality of context vectors.
    num_arms : int
    alpha : float
        Exploration parameter. Higher alpha = more exploration.
        Theory suggests alpha = 1 + sqrt(log(2/delta) / 2) for confidence delta.
    """

    def __init__(
        self,
        context_dim: int,
        num_arms: int,
        alpha: float = 1.0,
    ) -> None:
        self.context_dim = context_dim
        self.num_arms = num_arms
        self.alpha = alpha

        # Per-arm design matrix and reward vector
        self._A = [np.eye(context_dim, dtype=np.float64) for _ in range(num_arms)]
        self._b = [np.zeros(context_dim, dtype=np.float64) for _ in range(num_arms)]
        self._A_inv = [np.eye(context_dim, dtype=np.float64) for _ in range(num_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        """Select arm using LinUCB.

        A_t = argmax_a [theta_hat_a^T * x + alpha * sqrt(x^T * A_a^{-1} * x)]

        Parameters
        ----------
        context : np.ndarray of shape (d,)

        Returns
        -------
        int
            Selected arm index.
        """
        x = context.astype(np.float64)
        ucb_values = np.zeros(self.num_arms, dtype=np.float64)

        for a in range(self.num_arms):
            theta_hat = self._A_inv[a] @ self._b[a]
            # Predicted reward + confidence bonus
            pred = theta_hat @ x
            confidence = self.alpha * np.sqrt(x @ self._A_inv[a] @ x)
            ucb_values[a] = pred + confidence

        return int(np.argmax(ucb_values))

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """Update the model for the selected arm.

        A_a <- A_a + x * x^T
        b_a <- b_a + x * r
        A_a^{-1} <- updated via Sherman-Morrison formula for efficiency.
        """
        x = context.astype(np.float64)
        a = arm

        self._A[a] += np.outer(x, x)
        self._b[a] += x * reward

        # Sherman-Morrison update of A^{-1}:
        # (A + xx^T)^{-1} = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
        A_inv_x = self._A_inv[a] @ x
        denom = 1.0 + x @ A_inv_x
        self._A_inv[a] -= np.outer(A_inv_x, A_inv_x) / denom


# =====================================================================
# Neural Thompson Sampling for Contextual Bandits (Ch 15.6)
# =====================================================================


class NeuralThompsonSampling(ContextualBandit):
    """Neural Thompson Sampling for Contextual Bandits.

    Uses a neural network to model the reward distribution for each arm,
    then applies Thompson Sampling with a neural posterior approximation.

    Architecture:
        - Shared feature extractor: context -> hidden representation
        - Per-arm output head: hidden -> predicted reward

    Exploration is achieved via:
        1. Posterior approximation using the last-layer weights' uncertainty
           (similar to Neural Linear models)
        2. Adding noise proportional to the prediction uncertainty

    The posterior over the last-layer weights is maintained as a Gaussian:
        w_a ~ N(mu_a, Sigma_a)
    where Sigma_a is updated using online ridge regression on the learned features.

    Parameters
    ----------
    context_dim : int
    num_arms : int
    hidden_layers : sequence of int
    learning_rate : float
    exploration_coeff : float
        Controls exploration-exploitation trade-off.
    device : str
    """

    def __init__(
        self,
        context_dim: int,
        num_arms: int,
        hidden_layers: Sequence[int] = (64, 32),
        learning_rate: float = 1e-3,
        exploration_coeff: float = 0.1,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        assert _HAS_TORCH, "PyTorch required for NeuralThompsonSampling"

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.context_dim = context_dim
        self.num_arms = num_arms
        self.exploration_coeff = exploration_coeff
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # Build feature extractor
        layers: list[nn.Module] = []
        prev = context_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.feature_extractor = nn.Sequential(*layers).to(self.device)
        self.feature_dim = prev

        # Per-arm output layers (linear)
        self.output_layers = nn.ModuleList(
            [nn.Linear(prev, 1) for _ in range(num_arms)]
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.feature_extractor.parameters())
            + list(self.output_layers.parameters()),
            lr=learning_rate,
        )

        # Bayesian last-layer parameters (for Thompson Sampling)
        # Maintain per-arm precision matrix and mean for the last layer
        self._precision = [
            np.eye(prev, dtype=np.float64) for _ in range(num_arms)
        ]
        self._precision_b = [
            np.zeros(prev, dtype=np.float64) for _ in range(num_arms)
        ]

        # Data buffer for periodic retraining
        self._data: list[tuple[np.ndarray, int, float]] = []
        self._retrain_interval = 100
        self._step = 0

    def _extract_features(self, context: np.ndarray) -> np.ndarray:
        """Extract features using the neural feature extractor."""
        with torch.no_grad():
            x_t = torch.FloatTensor(context).unsqueeze(0).to(self.device)
            features = self.feature_extractor(x_t).squeeze(0).cpu().numpy()
        return features

    def select_arm(self, context: np.ndarray) -> int:
        """Select arm via Thompson Sampling with neural features.

        1. Extract features phi = f(context; theta_shared)
        2. For each arm a:
           - Compute posterior covariance: Sigma_a = (precision_a)^{-1}
           - Sample w_a ~ N(mu_a, exploration * Sigma_a)
           - Compute predicted reward: r_a = w_a^T * phi
        3. Select arm with highest sampled reward.
        """
        phi = self._extract_features(context)
        sampled_rewards = np.zeros(self.num_arms, dtype=np.float64)

        for a in range(self.num_arms):
            try:
                cov = np.linalg.inv(self._precision[a]) * self.exploration_coeff
            except np.linalg.LinAlgError:
                cov = np.eye(self.feature_dim) * self.exploration_coeff

            # Posterior mean
            try:
                mu = np.linalg.solve(self._precision[a], self._precision_b[a])
            except np.linalg.LinAlgError:
                mu = np.zeros(self.feature_dim)

            # Sample weights from posterior
            w_sample = self._rng.multivariate_normal(mu, cov)
            sampled_rewards[a] = w_sample @ phi

        return int(np.argmax(sampled_rewards))

    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """Update the model after observing (context, arm, reward).

        Updates:
        1. Bayesian last-layer: precision and precision-weighted mean
        2. Neural network (periodic retraining on buffered data)
        """
        phi = self._extract_features(context)

        # Bayesian update for the last layer of the selected arm
        # Precision: A <- A + phi * phi^T
        # Precision-weighted mean: b <- b + phi * reward
        self._precision[arm] += np.outer(phi, phi)
        self._precision_b[arm] += phi * reward

        # Buffer data for neural network retraining
        self._data.append((context, arm, reward))
        self._step += 1

        # Periodic retraining of the feature extractor
        if self._step % self._retrain_interval == 0 and len(self._data) >= 32:
            self._retrain()

    def _retrain(self, epochs: int = 5, batch_size: int = 32) -> None:
        """Retrain the neural network on buffered data."""
        n = len(self._data)
        for _ in range(epochs):
            indices = self._rng.choice(n, size=min(batch_size, n), replace=False)
            batch = [self._data[i] for i in indices]

            contexts = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
            arms = [b[1] for b in batch]
            rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)

            features = self.feature_extractor(contexts)
            predictions = torch.zeros(len(batch), device=self.device)
            for i, a in enumerate(arms):
                predictions[i] = self.output_layers[a](features[i].unsqueeze(0)).squeeze()

            loss = F.mse_loss(predictions, rewards)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Recompute Bayesian last-layer parameters after retraining
        # (features have changed, so posterior must be refreshed)
        for a in range(self.num_arms):
            self._precision[a] = np.eye(self.feature_dim, dtype=np.float64)
            self._precision_b[a] = np.zeros(self.feature_dim, dtype=np.float64)

        for ctx, arm, reward in self._data:
            phi = self._extract_features(ctx)
            self._precision[arm] += np.outer(phi, phi)
            self._precision_b[arm] += phi * reward
