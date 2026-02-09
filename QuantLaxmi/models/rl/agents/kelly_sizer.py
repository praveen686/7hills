"""Kelly-Merton Position Sizing Agent.

Combines the Kelly Criterion with Merton's optimal allocation for
dynamic, utility-aware position sizing.

Three analytical modes:
  1. **Kelly**: f* = (mu - r) / sigma^2
     Full Kelly maximises long-run geometric growth rate.
     Reference: Kelly (1956), Ch 8.1.

  2. **Fractional Kelly**: f = fraction * f*
     Reduces Kelly bet by a constant fraction (typically 0.5) to
     reduce drawdown risk.  Widely used in practice.

  3. **Merton-CRRA**: pi* = (mu - r) / (gamma * sigma^2)
     Optimal allocation for CRRA utility U(W) = W^{1-gamma}/(1-gamma).
     Reduces to Kelly when gamma=1 (log utility).
     Reference: Merton (1971), Ch 7-8.

One RL mode:
  4. **RL**: learns time-varying optimal fraction based on regime,
     drawdown state, and portfolio heat via Actor-Critic.

Safety mechanisms:
  - Drawdown adjustment: linearly reduce size during drawdowns
  - Portfolio heat check: sum of absolute position sizes vs capital
  - Maximum position cap per instrument

Book reference: Ch 7 (Utility Theory), Ch 8 (Merton's Problem),
               Ch 15 (Bandits for adaptive sizing).
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
    "KellySizer",
]


# ---------------------------------------------------------------------------
# RL sizing network
# ---------------------------------------------------------------------------


class _SizingNetwork(nn.Module):
    """Neural network: (features) -> optimal position fraction in [0, max_pct]."""

    def __init__(self, input_dim: int, hidden_layers: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())  # output in [0, 1], scaled externally
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# KellySizer
# ---------------------------------------------------------------------------


class KellySizer:
    """Kelly-Merton position sizing.

    Computes the optimal fraction of capital to allocate to a trade
    based on expected return, volatility, risk aversion, and current
    portfolio state.

    Modes:
      - ``"kelly"``: full Kelly f* = (mu - r) / sigma^2
      - ``"fractional_kelly"``: f = fraction * f* (default fraction=0.5)
      - ``"merton"``: CRRA optimal pi* = (mu - r) / (gamma * sigma^2)
      - ``"rl"``: learned adaptive sizing

    The returned size is always clipped to [0, max_position_pct] and
    further adjusted for:
      - Current drawdown (linear decay)
      - Portfolio heat (total risk budget)
      - Regime (optional conditioning)

    Parameters
    ----------
    mode : str
        ``"kelly"`` | ``"fractional_kelly"`` | ``"merton"`` | ``"rl"``
    fraction : float
        Kelly fraction for mode="fractional_kelly" (default 0.5).
    gamma_risk : float
        Risk aversion for Merton CRRA (gamma=1 = log utility = Kelly).
    max_position_pct : float
        Maximum allocation per position as fraction of capital.
    risk_free_rate : float
        Annualised risk-free rate (India 10Y ~ 6%).
    max_drawdown_pct : float
        Maximum drawdown at which sizing goes to zero.
    state_dim : int
        Input dimension for RL mode.
    hidden_layers : Sequence[int]
        NN layers for RL mode.
    device : str
        PyTorch device for RL mode.

    Book reference:
      Ch 7 (Utility Theory): CRRA, CARA utility functions
      Ch 8 (Merton's Problem): continuous-time optimal allocation
      Kelly (1956): "A New Interpretation of Information Rate"
      Thorp (2006): "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
    """

    def __init__(
        self,
        mode: str = "fractional_kelly",
        fraction: float = 0.5,
        gamma_risk: float = 2.0,
        max_position_pct: float = 0.25,
        risk_free_rate: float = 0.06,
        max_drawdown_pct: float = 0.20,
        # RL params
        state_dim: int = 10,
        hidden_layers: Sequence[int] = (64, 32),
        learning_rate: float = 1e-3,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        if mode not in ("kelly", "fractional_kelly", "merton", "rl"):
            raise ValueError(
                f"Unknown mode '{mode}'. "
                "Choose from: kelly, fractional_kelly, merton, rl"
            )
        self.mode = mode
        self.fraction = fraction
        self.gamma_risk = gamma_risk
        self.max_position_pct = max_position_pct
        self.risk_free_rate = risk_free_rate
        self.max_drawdown_pct = max_drawdown_pct
        self.state_dim = state_dim
        self._rng = np.random.default_rng(seed)

        # RL components
        self._model: Optional[Any] = None
        self._optimizer: Optional[Any] = None
        self._trained = False

        if mode == "rl" and _TORCH_AVAILABLE:
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            self._model = _SizingNetwork(state_dim, list(hidden_layers)).to(self.device)
            self._optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)
        else:
            self.device = None

    def optimal_size(
        self,
        expected_return: float,
        volatility: float,
        current_drawdown: float = 0.0,
        portfolio_heat: float = 0.0,
        regime: Optional[str] = None,
        features: Optional[np.ndarray] = None,
    ) -> float:
        """Compute optimal position size as fraction of capital.

        Parameters
        ----------
        expected_return : float
            Expected annualised return (mu) for the trade.
        volatility : float
            Annualised volatility (sigma) of the instrument.
        current_drawdown : float
            Current portfolio drawdown as fraction (0.0 = no DD, 0.1 = 10% DD).
        portfolio_heat : float
            Current portfolio heat (total absolute exposure / capital).
        regime : str or None
            Market regime for conditional sizing.
        features : np.ndarray or None
            Feature vector for RL mode.

        Returns
        -------
        float in [0, max_position_pct].
        """
        if volatility <= 0:
            return 0.0

        if self.mode == "kelly":
            base = self.kelly_fraction(expected_return, volatility, self.risk_free_rate)

        elif self.mode == "fractional_kelly":
            full_kelly = self.kelly_fraction(expected_return, volatility, self.risk_free_rate)
            base = self.fraction * full_kelly

        elif self.mode == "merton":
            base = self.merton_fraction(
                expected_return, volatility, self.risk_free_rate, self.gamma_risk
            )

        elif self.mode == "rl":
            base = self._rl_size(features)

        else:
            base = 0.0

        # Clip to non-negative (we don't short via sizing -- direction is separate)
        base = max(base, 0.0)

        # Drawdown adjustment
        base = self.drawdown_adjustment(base, current_drawdown)

        # Portfolio heat: reduce if already hot
        available_budget = max(0.0, 1.0 - portfolio_heat)
        base = min(base, available_budget)

        # Regime adjustment
        if regime == "volatile":
            base *= 0.7  # reduce in volatile regimes
        elif regime == "calm":
            base *= 1.1  # slightly increase in calm regimes

        # Final cap
        return float(np.clip(base, 0.0, self.max_position_pct))

    @staticmethod
    def kelly_fraction(mu: float, sigma: float, r: float = 0.0) -> float:
        """Standard Kelly fraction: f* = (mu - r) / sigma^2.

        This maximises the long-run geometric growth rate:
            G = E[log(1 + f*R)] = log(1 + f*(mu-r)) - f^2*sigma^2/2

        Taking dG/df = 0 gives f* = (mu-r)/sigma^2.

        Parameters
        ----------
        mu : float
            Expected annualised return.
        sigma : float
            Annualised volatility.
        r : float
            Risk-free rate.

        Returns
        -------
        Optimal fraction (can be > 1 for leveraged positions).

        Book reference: Ch 8.1, Kelly (1956).
        """
        if sigma <= 0:
            return 0.0
        return (mu - r) / (sigma ** 2)

    @staticmethod
    def merton_fraction(mu: float, sigma: float, r: float, gamma: float) -> float:
        """Merton-CRRA optimal allocation: pi* = (mu - r) / (gamma * sigma^2).

        For a CRRA utility function U(W) = W^{1-gamma} / (1-gamma):
          - gamma = 1 (log utility) reduces to Kelly
          - gamma = 2 gives half-Kelly
          - gamma > 1 is more conservative

        Parameters
        ----------
        mu : float
            Expected annualised return.
        sigma : float
            Annualised volatility.
        r : float
            Risk-free rate.
        gamma : float
            Relative risk aversion coefficient.

        Returns
        -------
        Optimal allocation fraction.

        Book reference: Ch 8.2, Merton (1971).
        """
        if sigma <= 0 or gamma <= 0:
            return 0.0
        return (mu - r) / (gamma * sigma ** 2)

    def drawdown_adjustment(self, base_size: float, drawdown: float) -> float:
        """Reduce position size during drawdowns.

        Linear decay: f = base_size * max(0, 1 - drawdown / max_drawdown)

        At max_drawdown, sizing goes to zero.
        At zero drawdown, sizing is unchanged.

        Parameters
        ----------
        base_size : float
            Base position size before adjustment.
        drawdown : float
            Current drawdown as fraction of peak (0.0 to 1.0).

        Returns
        -------
        Adjusted position size.
        """
        if self.max_drawdown_pct <= 0:
            return base_size
        scale = max(0.0, 1.0 - drawdown / self.max_drawdown_pct)
        return base_size * scale

    @staticmethod
    def portfolio_heat_check(positions: dict[str, float]) -> float:
        """Compute portfolio heat: sum of absolute position sizes / total capital.

        Parameters
        ----------
        positions : dict
            {instrument: position_fraction} where fraction is signed.

        Returns
        -------
        Total absolute exposure as fraction of capital.
        """
        return sum(abs(v) for v in positions.values())

    def train_rl(
        self,
        returns_history: np.ndarray,
        features_history: np.ndarray,
        num_epochs: int = 1000,
        batch_size: int = 64,
    ) -> dict:
        """Train RL-based dynamic sizing on historical data.

        The objective is to maximise the risk-adjusted geometric growth rate:
            max_f  E[log(1 + f_t * R_{t+1})] - lambda * Var[f_t * R_{t+1}]

        where f_t = pi_theta(context_t) is the learned sizing function.

        Parameters
        ----------
        returns_history : np.ndarray
            Shape (T,) -- daily returns for the instrument.
        features_history : np.ndarray
            Shape (T, state_dim) -- feature vectors at each time step.
        num_epochs : int
            Training epochs.
        batch_size : int
            Mini-batch size.

        Returns
        -------
        dict with "final_loss", "losses", "avg_sizing".
        """
        if self._model is None:
            raise RuntimeError("RL mode with PyTorch required for train_rl()")

        T = len(returns_history)
        losses = []

        for epoch in range(num_epochs):
            perm = self._rng.permutation(T - 1)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(perm), batch_size):
                idx = perm[start:start + batch_size]
                features = torch.tensor(
                    features_history[idx], dtype=torch.float32, device=self.device
                )
                next_returns = torch.tensor(
                    returns_history[idx + 1], dtype=torch.float32, device=self.device
                )

                self._model.train()
                fractions = self._model(features).squeeze(-1) * self.max_position_pct

                # Portfolio returns
                port_returns = fractions * next_returns

                # Objective: maximise E[log(1 + port_return)] - risk_penalty
                # Use first-order approx: log(1+x) ~ x - x^2/2
                growth = port_returns - 0.5 * port_returns ** 2
                risk_penalty = self.gamma_risk * port_returns.var()

                loss = -growth.mean() + risk_penalty

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                self._optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

        self._trained = True

        # Compute average sizing
        with torch.no_grad():
            all_features = torch.tensor(
                features_history, dtype=torch.float32, device=self.device
            )
            avg_size = float(self._model(all_features).mean().item()) * self.max_position_pct

        return {
            "final_loss": losses[-1] if losses else 0.0,
            "losses": losses,
            "avg_sizing": avg_size,
        }

    def _rl_size(self, features: Optional[np.ndarray]) -> float:
        """Get RL-based position size."""
        if self._model is None or features is None:
            return self.fraction * self.max_position_pct

        self._model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            frac = float(self._model(x).item())

        return frac * self.max_position_pct

    def summary(self) -> dict:
        """Return configuration summary."""
        return {
            "mode": self.mode,
            "fraction": self.fraction,
            "gamma_risk": self.gamma_risk,
            "max_position_pct": self.max_position_pct,
            "risk_free_rate": self.risk_free_rate,
            "max_drawdown_pct": self.max_drawdown_pct,
            "trained": self._trained,
        }

    def __repr__(self) -> str:
        return (
            f"KellySizer(mode={self.mode!r}, fraction={self.fraction}, "
            f"gamma={self.gamma_risk}, max_pct={self.max_position_pct})"
        )
