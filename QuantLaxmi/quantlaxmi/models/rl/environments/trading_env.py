"""Base trading MDP environment following Gymnasium-like interface.

All trading environments in models.rl inherit from TradingEnv.

This module provides:
  - TradingState: universal state representation for trading MDPs
  - TradingAction: signed trade size vector
  - StepResult: (state, reward, done, truncated, info) tuple
  - TradingEnv: abstract base class enforcing NO look-ahead bias
  - SimulatedPriceEnv: parametric price dynamics (GBM, Heston, jump-diffusion, OU)

Design principles (Ch 3 — MDP formulation):
  - State encodes only *causal* information available at time t.
  - Transition kernel is fully determined by (state, action).
  - Reward captures PnL after transaction costs.
  - Terminal condition is end-of-episode (end-of-day / end-of-horizon).

Book reference: Ch 3 (MDP formulation), Ch 10 (financial MDPs).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

__all__ = [
    "TradingState",
    "TradingAction",
    "StepResult",
    "TradingEnv",
    "SimulatedPriceEnv",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TradingState:
    """Universal trading state representation.

    All trading environments produce TradingState objects.  The ``features``
    dict carries instrument-specific extras (Greeks, order-book depth, etc.)
    while the core fields are common to every MDP.

    Attributes:
        timestamp: current discrete time step (0-indexed)
        prices: current price(s) -- shape depends on instrument count
        position: current holdings (signed; negative = short)
        cash: available cash / margin
        pnl: cumulative realised + unrealised PnL
        features: additional market features (dict of str -> float/array)
    """

    timestamp: int
    prices: np.ndarray
    position: np.ndarray
    cash: float
    pnl: float
    features: dict = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        """Flatten to numpy array suitable for NN input.

        Layout: [prices | position | cash | pnl | sorted feature values]
        """
        parts = [
            self.prices.ravel(),
            self.position.ravel(),
            np.array([self.cash, self.pnl], dtype=np.float64),
        ]
        # Deterministic ordering of feature values
        for key in sorted(self.features.keys()):
            val = self.features[key]
            if isinstance(val, np.ndarray):
                parts.append(val.ravel())
            elif isinstance(val, (int, float)):
                parts.append(np.array([float(val)], dtype=np.float64))
        return np.concatenate(parts).astype(np.float64)

    def copy(self) -> TradingState:
        return TradingState(
            timestamp=self.timestamp,
            prices=self.prices.copy(),
            position=self.position.copy(),
            cash=self.cash,
            pnl=self.pnl,
            features={k: (v.copy() if isinstance(v, np.ndarray) else v)
                      for k, v in self.features.items()},
        )


@dataclass
class TradingAction:
    """Trading action representation.

    ``trade_sizes`` is a signed array: positive = buy, negative = sell.
    For discrete-action environments the agent maps an integer action_id
    to the corresponding TradingAction externally.

    Attributes:
        trade_sizes: signed trade sizes (one per instrument / leg)
    """

    trade_sizes: np.ndarray


@dataclass
class StepResult:
    """Result of one environment step.

    Mirrors Gymnasium's ``(obs, reward, terminated, truncated, info)`` API.

    Attributes:
        state: new TradingState after the action
        reward: scalar reward (typically PnL minus costs)
        done: True if episode has naturally terminated
        truncated: True if episode was cut short (e.g. risk limit breach)
        info: extra diagnostics (costs paid, fill prices, etc.)
    """

    state: TradingState
    reward: float
    done: bool
    truncated: bool
    info: dict


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class TradingEnv(ABC):
    """Base class for all trading environments.

    Follows a Gymnasium-like API::

        state = env.reset()
        while not done:
            action = agent.act(state)
            result = env.step(action)
            state, reward, done = result.state, result.reward, result.done

    Design principles:
      - **NO look-ahead bias**: only causal data access.  ``step()`` reveals
        the *next* price only after the action is committed.
      - **Realistic costs**: transaction_cost_bps + slippage_bps deducted on
        every trade.
      - **Risk limits**: ``max_position`` enforced; breaches truncate the
        episode.
      - **Reproducibility**: seeded RNG for deterministic replay.

    Book reference: Ch 3 -- MDPs; Ch 10 -- Financial MDPs.
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        transaction_cost_bps: float = 5.0,
        slippage_bps: float = 2.0,
        max_position: float = float("inf"),
        seed: int = 42,
    ) -> None:
        self.initial_cash = initial_cash
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.max_position = max_position
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._current_state: Optional[TradingState] = None
        self._step_count: int = 0

    # ----- abstract interface -----

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> TradingState:
        """Reset environment to initial state and return it.

        If *seed* is not None, re-seed the internal RNG for reproducibility.
        """

    @abstractmethod
    def step(self, action: TradingAction) -> StepResult:
        """Execute *action* and advance the environment by one time step.

        Returns a StepResult.  Implementations must:
          1. Record the action *before* revealing the next price (causal).
          2. Compute fill price including slippage.
          3. Deduct transaction costs.
          4. Check risk limits.
        """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimensionality of the flattened state vector."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the action vector."""

    # ----- common helpers -----

    @property
    def is_done(self) -> bool:
        """True when the current episode has ended."""
        return self._current_state is None

    def render(self) -> dict:
        """Return current state as a dict for logging / visualization."""
        if self._current_state is None:
            return {"status": "not_initialised"}
        s = self._current_state
        return {
            "timestamp": s.timestamp,
            "prices": s.prices.tolist(),
            "position": s.position.tolist(),
            "cash": s.cash,
            "pnl": s.pnl,
            "features": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in s.features.items()},
        }

    def _apply_transaction_costs(self, trade_value: float) -> float:
        """Calculate transaction costs on *trade_value* (absolute).

        Total cost = (transaction_cost_bps + slippage_bps) / 10_000 * |trade_value|

        Returns the cost amount (always non-negative).
        """
        total_bps = self.transaction_cost_bps + self.slippage_bps
        return abs(trade_value) * total_bps / 10_000.0

    def _check_risk_limits(self, new_position: np.ndarray) -> bool:
        """Return True if *new_position* is within max_position limits.

        Each element of *new_position* must satisfy |pos_i| <= max_position.
        """
        return bool(np.all(np.abs(new_position) <= self.max_position))


# ---------------------------------------------------------------------------
# SimulatedPriceEnv — parametric price dynamics
# ---------------------------------------------------------------------------


class SimulatedPriceEnv(TradingEnv):
    """Trading environment with simulated price dynamics.

    Supports four widely-used stochastic models:

    **GBM** (Geometric Brownian Motion, Ch 7):
        dS = mu*S*dt + sigma*S*dW

    **Heston** (stochastic volatility):
        dS = mu*S*dt + sqrt(v)*S*dW_S
        dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_v
        corr(dW_S, dW_v) = rho

    **Jump-Diffusion** (Merton, 1976):
        dS = mu*S*dt + sigma*S*dW + J*S*dN(lambda)

    **OU** (Ornstein-Uhlenbeck / mean-reverting):
        dX = kappa*(theta - X)*dt + sigma*dW
        S = exp(X)   [log-prices are OU]

    For multi-asset, assets share a common correlation matrix and each
    follows the chosen dynamics independently (correlated Brownian motions).

    Parameters
    ----------
    dynamics : str
        One of ``"gbm"``, ``"heston"``, ``"jump_diffusion"``, ``"ou"``.
    mu : float
        Annualised drift.
    sigma : float
        Annualised volatility (or initial vol for Heston).
    num_steps : int
        Number of discrete time steps per episode.
    num_assets : int
        Number of tradeable assets.
    correlation : np.ndarray or None
        ``(num_assets, num_assets)`` correlation matrix.  None = identity.
    kappa, theta, xi, rho : float
        Heston / OU parameters.
    jump_intensity, jump_mean, jump_std : float
        Merton jump parameters.
    """

    def __init__(
        self,
        dynamics: str = "gbm",
        mu: float = 0.10,
        sigma: float = 0.20,
        num_steps: int = 252,
        num_assets: int = 1,
        correlation: Optional[np.ndarray] = None,
        # Heston parameters
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        # Jump-diffusion parameters
        jump_intensity: float = 5.0,
        jump_mean: float = 0.0,
        jump_std: float = 0.02,
        # Shared
        initial_price: float = 100.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if dynamics not in ("gbm", "heston", "jump_diffusion", "ou"):
            raise ValueError(
                f"Unknown dynamics '{dynamics}'. "
                "Choose from: gbm, heston, jump_diffusion, ou"
            )
        self.dynamics = dynamics
        self.mu = mu
        self.sigma = sigma
        self.num_steps = num_steps
        self.num_assets = num_assets
        self.initial_price = initial_price

        # Correlation structure
        if correlation is not None:
            if correlation.shape != (num_assets, num_assets):
                raise ValueError("correlation must be (num_assets, num_assets)")
            self.correlation = correlation.copy()
        else:
            self.correlation = np.eye(num_assets, dtype=np.float64)
        self._chol = np.linalg.cholesky(self.correlation)

        # Heston
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

        # Jump-diffusion
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

        # Pre-computed dt
        self.dt = 1.0 / 252.0  # daily by default

        # Simulation state
        self._prices: Optional[np.ndarray] = None
        self._variance: Optional[np.ndarray] = None  # Heston
        self._log_prices: Optional[np.ndarray] = None  # OU

    # ----- public API -----

    @property
    def state_dim(self) -> int:
        # prices + position + cash + pnl + returns_5d + vol_20d per asset
        return self.num_assets * 4 + 2

    @property
    def action_dim(self) -> int:
        return self.num_assets

    def reset(self, seed: Optional[int] = None) -> TradingState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0

        self._prices = np.full(self.num_assets, self.initial_price, dtype=np.float64)
        self._variance = np.full(self.num_assets, self.sigma ** 2, dtype=np.float64)
        self._log_prices = np.log(self._prices)
        self._price_history: list[np.ndarray] = [self._prices.copy()]

        self._current_state = TradingState(
            timestamp=0,
            prices=self._prices.copy(),
            position=np.zeros(self.num_assets, dtype=np.float64),
            cash=self.initial_cash,
            pnl=0.0,
            features={"returns_5d": 0.0, "vol_20d": self.sigma},
        )
        return self._current_state.copy()

    def step(self, action: TradingAction) -> StepResult:
        if self._current_state is None:
            raise RuntimeError("Call reset() before step()")
        if self._prices is None:
            raise RuntimeError("Internal error: prices not initialised")

        old_state = self._current_state
        old_prices = self._prices.copy()

        # 1) Advance prices (next-bar dynamics)
        self._advance_prices()
        self._step_count += 1
        self._price_history.append(self._prices.copy())

        # 2) Execute trades at *old* price + slippage (causal: decision made
        #    before new price is revealed; fill assumed at close of current bar)
        trade_sizes = np.clip(
            action.trade_sizes,
            -self.max_position - old_state.position,
            self.max_position - old_state.position,
        )
        new_position = old_state.position + trade_sizes
        trade_value = float(np.sum(np.abs(trade_sizes) * old_prices))
        cost = self._apply_transaction_costs(trade_value)
        cash_delta = -float(np.sum(trade_sizes * old_prices)) - cost

        new_cash = old_state.cash + cash_delta

        # 3) Mark-to-market PnL
        position_value = float(np.sum(new_position * self._prices))
        new_pnl = new_cash + position_value - self.initial_cash

        # 4) Reward = change in PnL
        reward = new_pnl - old_state.pnl

        # 5) Features
        features = self._compute_features()

        # 6) Terminal conditions
        done = self._step_count >= self.num_steps
        truncated = not self._check_risk_limits(new_position)
        if truncated:
            # Force-close position at market price
            close_value = float(np.sum(np.abs(new_position) * self._prices))
            close_cost = self._apply_transaction_costs(close_value)
            new_cash += float(np.sum(new_position * self._prices)) - close_cost
            new_position = np.zeros(self.num_assets, dtype=np.float64)
            new_pnl = new_cash - self.initial_cash
            reward = new_pnl - old_state.pnl

        new_state = TradingState(
            timestamp=self._step_count,
            prices=self._prices.copy(),
            position=new_position,
            cash=new_cash,
            pnl=new_pnl,
            features=features,
        )
        self._current_state = new_state if not (done or truncated) else None

        return StepResult(
            state=new_state,
            reward=reward,
            done=done,
            truncated=truncated,
            info={"cost": cost, "trade_value": trade_value},
        )

    # ----- price dynamics -----

    def _advance_prices(self) -> None:
        """Advance prices by one step using the configured dynamics."""
        assert self._prices is not None
        dt = self.dt
        z = self._rng.standard_normal(self.num_assets)
        z_corr = self._chol @ z  # correlated normals

        if self.dynamics == "gbm":
            self._prices = self._prices * np.exp(
                (self.mu - 0.5 * self.sigma ** 2) * dt
                + self.sigma * math.sqrt(dt) * z_corr
            )

        elif self.dynamics == "heston":
            assert self._variance is not None
            z_v = self._rng.standard_normal(self.num_assets)
            # Correlated with price shocks
            z_v = self.rho * z_corr + math.sqrt(1.0 - self.rho ** 2) * z_v
            var = np.maximum(self._variance, 0.0)
            vol = np.sqrt(var)
            self._prices = self._prices * np.exp(
                (self.mu - 0.5 * var) * dt + vol * math.sqrt(dt) * z_corr
            )
            self._variance = (
                var
                + self.kappa * (self.theta - var) * dt
                + self.xi * vol * math.sqrt(dt) * z_v
            )
            self._variance = np.maximum(self._variance, 0.0)

        elif self.dynamics == "jump_diffusion":
            # Merton jump-diffusion
            n_jumps = self._rng.poisson(self.jump_intensity * dt, self.num_assets)
            jump_sizes = np.zeros(self.num_assets)
            for i in range(self.num_assets):
                if n_jumps[i] > 0:
                    jump_sizes[i] = np.sum(
                        self._rng.normal(self.jump_mean, self.jump_std, int(n_jumps[i]))
                    )
            self._prices = self._prices * np.exp(
                (self.mu - 0.5 * self.sigma ** 2) * dt
                + self.sigma * math.sqrt(dt) * z_corr
                + jump_sizes
            )

        elif self.dynamics == "ou":
            assert self._log_prices is not None
            self._log_prices = (
                self._log_prices
                + self.kappa * (np.log(self.theta) - self._log_prices) * dt
                + self.sigma * math.sqrt(dt) * z_corr
            )
            self._prices = np.exp(self._log_prices)

    def _compute_features(self) -> dict:
        """Compute features from price history (causal only)."""
        history = self._price_history
        n = len(history)
        features: dict = {}

        # 5-day returns
        if n > 5:
            ret5 = (history[-1] - history[-6]) / history[-6]
            features["returns_5d"] = float(np.mean(ret5))
        else:
            features["returns_5d"] = 0.0

        # 20-day realised volatility
        if n > 20:
            log_rets = np.log(
                np.array(history[-21:][1:]) / np.array(history[-21:][:-1])
            )
            features["vol_20d"] = float(np.std(log_rets, ddof=1) * math.sqrt(252))
        else:
            features["vol_20d"] = self.sigma

        return features
