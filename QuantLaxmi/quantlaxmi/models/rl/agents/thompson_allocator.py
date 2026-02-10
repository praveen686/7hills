"""Thompson Sampling Strategy Selector.

Meta-strategy that dynamically allocates across S1-S12 + crypto
strategies using contextual bandits.

Each trading strategy (S1 VRP, S2 Microstructure, ..., S12 Vedic,
plus crypto strategies) is treated as an "arm" in a multi-armed
bandit problem.  The context includes regime, VIX, days-to-expiry,
day-of-week, and recent strategy correlations.

This module provides:
  - ThompsonStrategyAllocator: conjugate Normal-InverseGamma Thompson Sampling
  - Optional neural contextual bandit extension (requires PyTorch)

Book reference: Ch 15.5 -- Thompson Sampling with Normal-InverseGamma
conjugate priors for unknown mean AND variance.

Key formula:
    For each strategy s, maintain posterior NIG(mu_s, lambda_s, alpha_s, beta_s).
    At each decision point:
      1. Sample sigma^2_s ~ InverseGamma(alpha_s, beta_s)
      2. Sample mu_s ~ Normal(mu_s, sigma^2_s / lambda_s)
      3. Allocate proportionally to max(sampled_mu_s, 0)
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

__all__ = [
    "ThompsonStrategyAllocator",
]


# ---------------------------------------------------------------------------
# Normal-Inverse-Gamma posterior
# ---------------------------------------------------------------------------


class _NIGPosterior:
    """Normal-Inverse-Gamma conjugate posterior for a single arm.

    Prior: sigma^2 ~ IG(alpha, beta), mu | sigma^2 ~ N(mu0, sigma^2/lambda)

    Update rules after observing x:
        lambda' = lambda + 1
        mu'     = (lambda * mu + x) / (lambda + 1)
        alpha'  = alpha + 0.5
        beta'   = beta + 0.5 * lambda * (x - mu)^2 / (lambda + 1)
    """

    def __init__(
        self,
        mu: float = 0.0,
        lam: float = 1.0,
        alpha: float = 2.0,
        beta: float = 1.0,
    ) -> None:
        self.mu = mu
        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self.n_updates = 0

    def update(self, x: float) -> None:
        """Bayesian update after observing one return x."""
        lam_new = self.lam + 1.0
        mu_new = (self.lam * self.mu + x) / lam_new
        alpha_new = self.alpha + 0.5
        beta_new = self.beta + 0.5 * self.lam * (x - self.mu) ** 2 / lam_new

        self.mu = mu_new
        self.lam = lam_new
        self.alpha = alpha_new
        self.beta = beta_new
        self.n_updates += 1

    def sample(self, rng: np.random.Generator) -> float:
        """Sample from the posterior: sigma^2 ~ IG, mu ~ N."""
        # Inverse-Gamma sample via Gamma
        if self.alpha <= 0 or self.beta <= 0:
            return rng.normal(self.mu, 1.0)
        sigma2 = 1.0 / rng.gamma(self.alpha, 1.0 / self.beta)
        sigma2 = max(sigma2, 1e-12)
        mu_sample = rng.normal(self.mu, math.sqrt(sigma2 / max(self.lam, 1e-12)))
        return mu_sample

    def mean_estimate(self) -> float:
        """Posterior mean of mu."""
        return self.mu

    def variance_estimate(self) -> float:
        """Posterior mean of sigma^2."""
        if self.alpha <= 1:
            return self.beta
        return self.beta / (self.alpha - 1.0)

    def to_dict(self) -> dict:
        return {
            "mu": self.mu,
            "lambda": self.lam,
            "alpha": self.alpha,
            "beta": self.beta,
            "n_updates": self.n_updates,
            "mean": self.mean_estimate(),
            "variance": self.variance_estimate(),
        }


# ---------------------------------------------------------------------------
# Neural Contextual Bandit (optional PyTorch)
# ---------------------------------------------------------------------------


class _NeuralContextualBandit:
    """Neural network for contextual bandit reward prediction.

    Maps (context, strategy_id) -> predicted Sharpe ratio.
    Uses last-layer Thompson Sampling (Riquelme et al., 2018).
    """

    def __init__(
        self,
        context_dim: int,
        num_strategies: int,
        hidden_layers: Sequence[int] = (64, 32),
        learning_rate: float = 1e-3,
        device: str = "auto",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required for neural contextual bandit")

        self.context_dim = context_dim
        self.num_strategies = num_strategies

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build network
        layers: list[nn.Module] = []
        in_dim = context_dim + num_strategies  # one-hot strategy encoding
        for h in hidden_layers:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.1)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def predict(self, context: np.ndarray, strategy_idx: int) -> float:
        """Predict reward for (context, strategy)."""
        x = self._encode(context, strategy_idx)
        self.model.eval()
        with torch.no_grad():
            return float(self.model(x).item())

    def train_step(
        self,
        contexts: np.ndarray,
        strategy_indices: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        """One training step on a batch."""
        self.model.train()
        batch_size = len(contexts)
        x_list = []
        for i in range(batch_size):
            x_list.append(self._encode_np(contexts[i], int(strategy_indices[i])))
        x = torch.tensor(np.array(x_list), dtype=torch.float32, device=self.device)
        y = torch.tensor(returns, dtype=torch.float32, device=self.device)

        pred = self.model(x).squeeze(-1)
        loss = self.loss_fn(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def _encode(self, context: np.ndarray, strategy_idx: int) -> torch.Tensor:
        arr = self._encode_np(context, strategy_idx)
        return torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _encode_np(self, context: np.ndarray, strategy_idx: int) -> np.ndarray:
        one_hot = np.zeros(self.num_strategies, dtype=np.float32)
        one_hot[strategy_idx] = 1.0
        return np.concatenate([context.astype(np.float32), one_hot])


# ---------------------------------------------------------------------------
# ThompsonStrategyAllocator
# ---------------------------------------------------------------------------


class ThompsonStrategyAllocator:
    """Contextual Thompson Sampling for strategy selection.

    Each strategy is an "arm" in a multi-armed bandit.  Context includes:
      - Regime (trending / mean-reverting / volatile / calm)
      - VIX level
      - Days to expiry
      - Day of week
      - Recent strategy correlations

    Prior: Normal-InverseGamma for each strategy's return distribution.
    Updates: Bayesian posterior after observing daily returns.
    Selection: Sample from posterior, allocate proportionally to sampled returns.

    Optionally uses a neural contextual bandit for non-linear context
    modelling (last-layer Thompson Sampling).

    Parameters
    ----------
    strategy_names : list[str]
        Names of available strategies (e.g. ["S1_VRP", "S5_Hawkes", ...]).
    context_dim : int
        Dimensionality of context vector.
    prior_mu : float
        Prior mean of returns.
    prior_lambda : float
        Prior precision scaling.
    prior_alpha : float
        Prior IG shape.
    prior_beta : float
        Prior IG scale.
    min_allocation : float
        Minimum allocation per strategy (0 = can skip entirely).
    max_allocation : float
        Maximum allocation per strategy.
    use_neural : bool
        If True, augment with neural contextual bandit.
    device : str
        PyTorch device for neural mode.

    Book reference: Ch 15.5 (Thompson Sampling), Ch 15.7 (Contextual Bandits).
    """

    def __init__(
        self,
        strategy_names: list[str],
        context_dim: int = 10,
        prior_mu: float = 0.0,
        prior_lambda: float = 1.0,
        prior_alpha: float = 2.0,
        prior_beta: float = 1.0,
        min_allocation: float = 0.0,
        max_allocation: float = 0.5,
        use_neural: bool = False,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        self.strategy_names = list(strategy_names)
        self.num_strategies = len(strategy_names)
        self.context_dim = context_dim
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation
        self.use_neural = use_neural
        self._rng = np.random.default_rng(seed)

        # Name -> index mapping
        self._name_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(strategy_names)
        }

        # NIG posteriors per strategy
        self._posteriors: list[_NIGPosterior] = [
            _NIGPosterior(mu=prior_mu, lam=prior_lambda, alpha=prior_alpha, beta=prior_beta)
            for _ in range(self.num_strategies)
        ]

        # History tracking
        self._history: list[dict] = []
        self._cumulative_regret: float = 0.0

        # Neural contextual bandit (optional)
        self._neural: Optional[_NeuralContextualBandit] = None
        if use_neural and _TORCH_AVAILABLE:
            self._neural = _NeuralContextualBandit(
                context_dim=context_dim,
                num_strategies=self.num_strategies,
                device=device,
            )

    def select_allocation(self, context: np.ndarray) -> dict[str, float]:
        """Given context, return allocation weights {strategy: weight}.

        Algorithm:
          1. For each strategy, sample from NIG posterior.
          2. If neural mode, add neural prediction as bonus.
          3. Allocate proportionally to max(sampled_value, 0).
          4. Clip to [min_allocation, max_allocation], renormalise.

        Parameters
        ----------
        context : np.ndarray
            Context vector of shape (context_dim,).

        Returns
        -------
        dict mapping strategy name -> allocation weight (sums to 1).
        """
        sampled_values = np.zeros(self.num_strategies)

        for i, posterior in enumerate(self._posteriors):
            sampled_values[i] = posterior.sample(self._rng)

            if self._neural is not None:
                # Add neural prediction as informative prior shift
                neural_pred = self._neural.predict(context, i)
                sampled_values[i] = 0.5 * sampled_values[i] + 0.5 * neural_pred

        # Softmax-like allocation: proportional to max(sampled, 0)
        positive = np.maximum(sampled_values, 0.0)
        total = positive.sum()

        if total < 1e-12:
            # All non-positive: equal allocation
            weights = np.ones(self.num_strategies) / self.num_strategies
        else:
            weights = positive / total

        # Clip and renormalise
        weights = np.clip(weights, self.min_allocation, self.max_allocation)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(self.num_strategies) / self.num_strategies

        return {name: float(w) for name, w in zip(self.strategy_names, weights)}

    def update(self, strategy: str, daily_return: float, context: np.ndarray) -> None:
        """Update posterior for strategy after observing daily return.

        Parameters
        ----------
        strategy : str
            Strategy name.
        daily_return : float
            Observed daily return.
        context : np.ndarray
            Context at time of decision.
        """
        idx = self._name_to_idx.get(strategy)
        if idx is None:
            raise ValueError(f"Unknown strategy '{strategy}'")

        self._posteriors[idx].update(daily_return)

        # Record for history
        self._history.append({
            "strategy": strategy,
            "return": daily_return,
            "context": context.copy(),
        })

    def get_posteriors(self) -> dict[str, dict]:
        """Return posterior parameters for each strategy.

        Returns
        -------
        dict mapping strategy name -> {"mu", "lambda", "alpha", "beta",
        "n_updates", "mean", "variance"}.
        """
        return {
            name: self._posteriors[i].to_dict()
            for i, name in enumerate(self.strategy_names)
        }

    def regret(self, oracle_returns: np.ndarray) -> float:
        """Compute cumulative regret vs oracle (hindsight best arm).

        Parameters
        ----------
        oracle_returns : np.ndarray
            Shape (T, num_strategies) -- actual returns of each strategy
            at each time step.

        Returns
        -------
        Cumulative regret = sum_t (best_return_t - chosen_return_t).
        """
        T = len(self._history)
        if T == 0:
            return 0.0

        total_regret = 0.0
        for t in range(min(T, len(oracle_returns))):
            best_return = float(oracle_returns[t].max())
            chosen_return = self._history[t]["return"]
            total_regret += best_return - chosen_return

        return total_regret

    def train_neural(
        self,
        contexts: np.ndarray,
        strategies: np.ndarray,
        returns: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> dict:
        """Train neural contextual bandit on historical data.

        Parameters
        ----------
        contexts : np.ndarray
            Shape (N, context_dim).
        strategies : np.ndarray
            Shape (N,) -- integer strategy indices.
        returns : np.ndarray
            Shape (N,) -- observed returns.
        epochs : int
            Training epochs.
        batch_size : int
            Mini-batch size.

        Returns
        -------
        dict with "final_loss" and "losses" history.
        """
        if self._neural is None:
            if not _TORCH_AVAILABLE:
                raise ImportError("PyTorch required for neural training")
            self._neural = _NeuralContextualBandit(
                context_dim=self.context_dim,
                num_strategies=self.num_strategies,
            )

        N = len(contexts)
        losses = []

        for epoch in range(epochs):
            # Shuffle
            perm = self._rng.permutation(N)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                loss = self._neural.train_step(
                    contexts[idx], strategies[idx], returns[idx]
                )
                epoch_loss += loss
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

        return {"final_loss": losses[-1] if losses else 0.0, "losses": losses}

    def ranking(self) -> list[tuple[str, float]]:
        """Return strategies ranked by posterior mean (descending)."""
        ranked = [
            (name, self._posteriors[i].mean_estimate())
            for i, name in enumerate(self.strategy_names)
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
