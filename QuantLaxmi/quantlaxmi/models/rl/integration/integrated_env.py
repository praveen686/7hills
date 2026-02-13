"""Integrated Multi-Asset Trading Environment.

Wraps the X-Trend backbone + per-asset env instances into a single
Gymnasium-like environment that:

1. Pre-computes backbone hidden states for the entire fold (frozen, no grad).
2. Steps each per-asset env (IndiaFnOEnv or CryptoEnv) with position targets.
3. Computes portfolio-level state: concatenated hidden states, positions,
   PnL, drawdown, heat, time features.
4. Returns reward = risk-adjusted PnL - cost penalty - drawdown penalty.

Supports two configurations:
  - India-only (4 assets): NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY
  - Unified (6 assets): 4 India + BTCUSDT, ETHUSDT

State vector layout (dim = n_assets x d_hidden + n_assets + 4):
    [backbone_hidden(asset_0, d_h), ..., backbone_hidden(asset_{n-1}, d_h),
     positions(n), norm_pnl(1), drawdown(1), heat(1), time_feats(3)]

Action: (n_assets,) continuous position targets in [-1, 1].

Calendar alignment (unified mode):
  - India assets trade 09:15-15:30 IST (252 days/year).
  - Crypto assets trade 24/7.
  - Alignment point: India close at 15:30 IST.
  - On non-trading days for India (weekends, holidays), India positions are
    held flat (forced to 0); only crypto positions can change.
  - Overnight gap signal: BTC return from India close (15:30 IST) to next
    India open (09:15 IST) is exposed via ``compute_overnight_gap()``.

Cost model:
  - India: cost = |trade| * cost_per_leg_pts / spot  (index points per leg)
  - Crypto: cost = |trade| * fee_rate  (0.1% maker/taker per side)

References:
    - Ch 3 (MDP formulation), Ch 10 (Financial MDPs)
    - IndiaFnOEnv: realistic India FnO costs (3 pts NIFTY, 5 pts BANKNIFTY)
    - CryptoEnv: Binance perpetual futures (BTC, ETH)
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from quantlaxmi.models.rl.environments.india_fno_env import (
    IndiaFnOEnv,
    COST_PER_LEG,
    LOT_SIZES,
    INITIAL_SPOTS,
    ANNUALISED_VOLS,
)
from quantlaxmi.models.rl.environments.crypto_env import CryptoEnv
from quantlaxmi.models.rl.environments.trading_env import TradingAction

# ---------------------------------------------------------------------------
# Constants for unified 6-asset mode
# ---------------------------------------------------------------------------

INDIA_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
ALL_SYMBOLS = INDIA_SYMBOLS + CRYPTO_SYMBOLS

# Crypto trading costs: 0.1% maker/taker per side (fractional)
CRYPTO_FEE_RATE = 0.001

# Approximate USD spot prices for crypto (used when no data available)
CRYPTO_INITIAL_SPOTS = {"BTCUSDT": 65_000.0, "ETHUSDT": 3_500.0}

# Annualised vols for crypto (approximate)
CRYPTO_ANNUALISED_VOLS = {"BTCUSDT": 0.55, "ETHUSDT": 0.65}


# ============================================================================
# AssetCostModel — unified cost abstraction
# ============================================================================


class AssetCostModel(ABC):
    """Abstract cost model for a single asset.

    Computes the fractional transaction cost for a position change.
    All costs are returned as a fraction of portfolio (dimensionless).
    """

    @abstractmethod
    def compute_cost(self, trade_size: float) -> float:
        """Compute fractional cost for a trade of given signed size.

        Parameters
        ----------
        trade_size : float
            Signed position change (new_pos - old_pos), in [-2, 2] range.

        Returns
        -------
        float : non-negative fractional cost.
        """

    @property
    @abstractmethod
    def asset_type(self) -> str:
        """Return ``"india"`` or ``"crypto"``."""


class IndiaCostModel(AssetCostModel):
    """India FnO cost model: index points per leg.

    cost_fraction = |trade| * cost_per_leg / spot

    Parameters
    ----------
    symbol : str
        One of NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY.
    cost_per_leg : float or None
        Override default cost per leg (index points).
    spot : float or None
        Override default spot price.
    """

    def __init__(
        self,
        symbol: str,
        cost_per_leg: Optional[float] = None,
        spot: Optional[float] = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.cost_per_leg = cost_per_leg or COST_PER_LEG.get(self.symbol, 3.0)
        self.spot = spot or INITIAL_SPOTS.get(self.symbol, 20_000.0)

    def compute_cost(self, trade_size: float) -> float:
        return abs(trade_size) * self.cost_per_leg / self.spot

    @property
    def asset_type(self) -> str:
        return "india"


class CryptoCostModel(AssetCostModel):
    """Crypto cost model: percentage-based maker/taker fee.

    cost_fraction = |trade| * fee_rate

    Parameters
    ----------
    symbol : str
        e.g. "BTCUSDT", "ETHUSDT".
    fee_rate : float
        Fee as fraction of notional per side (default 0.001 = 0.1%).
    """

    def __init__(
        self,
        symbol: str,
        fee_rate: float = CRYPTO_FEE_RATE,
    ) -> None:
        self.symbol = symbol.upper()
        self.fee_rate = fee_rate

    def compute_cost(self, trade_size: float) -> float:
        return abs(trade_size) * self.fee_rate

    @property
    def asset_type(self) -> str:
        return "crypto"


def build_cost_models(symbols: list[str]) -> list[AssetCostModel]:
    """Build a list of cost models for the given symbol list.

    India symbols (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY) get IndiaCostModel.
    Crypto symbols (BTCUSDT, ETHUSDT, or anything else) get CryptoCostModel.
    """
    india_set = {s.upper() for s in INDIA_SYMBOLS}
    models: list[AssetCostModel] = []
    for sym in symbols:
        if sym.upper() in india_set:
            models.append(IndiaCostModel(sym))
        else:
            models.append(CryptoCostModel(sym))
    return models


# ============================================================================
# Calendar alignment helpers
# ============================================================================


def is_india_trading_day(dt: pd.Timestamp) -> bool:
    """Check if a given date is an India trading day (Mon-Fri, not a holiday).

    This is a simplified check using weekday only.  For production,
    integrate with NSE holiday calendar.
    """
    # dt may be tz-aware or naive
    if hasattr(dt, 'weekday'):
        return dt.weekday() < 5  # Mon=0 .. Fri=4
    return True


def compute_overnight_gap(
    crypto_close_prices: np.ndarray,
    india_close_idx: int,
    india_open_idx: int,
) -> float:
    """Compute BTC overnight return between India close and next open.

    Parameters
    ----------
    crypto_close_prices : (n_days,) array
        Daily close prices for a crypto asset.
    india_close_idx : int
        Day index of India close (15:30 IST).
    india_open_idx : int
        Day index of India next open (09:15 IST next trading day).

    Returns
    -------
    float : log return of crypto between India close and next open.
        Returns 0.0 if indices are out of bounds or prices are invalid.
    """
    n = len(crypto_close_prices)
    if india_close_idx < 0 or india_open_idx < 0:
        return 0.0
    if india_close_idx >= n or india_open_idx >= n:
        return 0.0
    p_close = crypto_close_prices[india_close_idx]
    p_open = crypto_close_prices[india_open_idx]
    if p_close <= 0 or p_open <= 0:
        return 0.0
    return float(math.log(p_open / p_close))


# ============================================================================
# IntegratedTradingEnv
# ============================================================================


class IntegratedTradingEnv:
    """Multi-asset walk-forward environment with backbone hidden states.

    Supports both India-only (4 assets) and unified (6 assets: 4 India + 2
    crypto) configurations.  The ``include_crypto`` flag controls this.

    Parameters
    ----------
    backbone : XTrendBackbone or CrossMarketBackbone
        Frozen backbone for hidden state extraction.
    features : (n_days, n_assets, n_features)
        Normalized feature tensor.
    targets : (n_days, n_assets)
        Vol-scaled next-day returns (for reward computation).
    dates : DatetimeIndex
        Trading dates.
    symbols : list[str]
        Asset symbols (e.g. ["NIFTY", "BANKNIFTY", ..., "BTCUSDT", "ETHUSDT"]).
    reward_lambda_risk : float
        Weight for drawdown penalty in reward.
    reward_lambda_cost : float
        Weight for cost penalty in reward.
    include_crypto : bool
        If True, treat non-India symbols as crypto assets with CryptoCostModel
        and CryptoEnv instances.  Default False for backwards compatibility.
    india_trading_days : set[int] or None
        Set of day indices (into ``dates``) that are India trading days.
        On non-India-trading days, India positions are forced flat.
        If None, all days are treated as India trading days.
    """

    def __init__(
        self,
        backbone,
        features: np.ndarray,
        targets: np.ndarray,
        dates,
        symbols: list[str],
        reward_lambda_risk: float = 0.5,
        reward_lambda_cost: float = 1.0,
        include_crypto: bool = False,
        india_trading_days: Optional[set[int]] = None,
    ) -> None:
        self.backbone = backbone
        self.features = features
        self.targets = targets
        self.dates = dates
        self.symbols = symbols
        self.n_assets = len(symbols)
        self.d_hidden = backbone.d_hidden if hasattr(backbone, 'd_hidden') else 64
        self.reward_lambda_risk = reward_lambda_risk
        self.reward_lambda_cost = reward_lambda_cost
        self.include_crypto = include_crypto
        self.india_trading_days = india_trading_days

        # Build per-asset cost models
        self._cost_models = build_cost_models(symbols)

        # Classify asset indices
        india_set = {s.upper() for s in INDIA_SYMBOLS}
        self._india_indices = [
            i for i, s in enumerate(symbols) if s.upper() in india_set
        ]
        self._crypto_indices = [
            i for i, s in enumerate(symbols) if s.upper() not in india_set
        ]

        # Per-asset env instances
        self._envs: list = []
        for i, sym in enumerate(symbols):
            sym_upper = sym.upper()
            if sym_upper in india_set:
                env = IndiaFnOEnv(
                    instrument=sym_upper,
                    mode="simulated",
                    discrete_actions=False,
                    num_steps=1000,
                )
            else:
                env = CryptoEnv(
                    symbol=sym_upper,
                    mode="simulated",
                    num_steps=1000,
                )
            self._envs.append(env)

        # Portfolio tracking state
        self._positions = np.zeros(self.n_assets)
        self._cum_pnl = 0.0
        self._peak_pnl = 0.0
        self._drawdown = 0.0
        self._step_idx = 0
        self._fold_start = 0
        self._fold_end = 0
        self._hidden_states: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(42)

        # Reward shaping (Pattern 5 — attention reward)
        self._reward_shaper = None
        self._reward_bonuses: Optional[np.ndarray] = None

    @property
    def state_dim(self) -> int:
        """Total state vector dimension.

        Layout: n_assets * d_hidden (hidden states)
                + n_assets (positions)
                + 1 (norm_pnl) + 1 (drawdown) + 1 (heat)
                + 3 (time_feats: progress, day_sin, day_cos)
        """
        return self.n_assets * self.d_hidden + self.n_assets + 6

    @property
    def action_dim(self) -> int:
        """One continuous position target per asset."""
        return self.n_assets

    def reset(
        self, fold_start_idx: int, fold_end_idx: int
    ) -> np.ndarray:
        """Reset environment for a new fold.

        Pre-computes backbone hidden states for the entire fold
        (frozen, no gradient), then returns the initial state vector.

        Parameters
        ----------
        fold_start_idx : int
            Start index in the features array.
        fold_end_idx : int
            End index (exclusive).

        Returns
        -------
        state_vec : (state_dim,)
        """
        self._step_idx = 0
        self._positions = np.zeros(self.n_assets)
        self._cum_pnl = 0.0
        self._peak_pnl = 0.0
        self._drawdown = 0.0

        # Pre-compute backbone hidden states ONLY if fold boundaries changed
        # (avoids re-running 1000+ forward passes on every RL episode reset)
        if (
            self._fold_start != fold_start_idx
            or self._fold_end != fold_end_idx
            or self._hidden_states is None
        ):
            self._fold_start = fold_start_idx
            self._fold_end = fold_end_idx
            logger.info(
                "Precomputing hidden states: days [%d:%d] x %d assets...",
                fold_start_idx, fold_end_idx, self.n_assets,
            )
            self._hidden_states = self.backbone.precompute_hidden_states(
                self.features, fold_start_idx, fold_end_idx, self._rng
            )
            logger.info("Hidden states precomputed: %s", self._hidden_states.shape)
        # shape: (fold_len, n_assets, d_hidden)

        # Reset per-asset envs
        for env in self._envs:
            env.reset()

        return self._build_state_vector()

    def _is_india_trading_step(self) -> bool:
        """Check if the current step is an India trading day.

        Uses ``india_trading_days`` set if provided; otherwise checks the
        date's weekday via ``is_india_trading_day()``.
        """
        day_idx = self._fold_start + self._step_idx
        if self.india_trading_days is not None:
            return day_idx in self.india_trading_days
        # Fall back to date-based check
        if self.dates is not None and day_idx < len(self.dates):
            return is_india_trading_day(self.dates[day_idx])
        return True  # default: assume trading day

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict]:
        """Take one step in the environment.

        Parameters
        ----------
        actions : (n_assets,) -- position targets in [-1, 1]

        Returns
        -------
        state_vec : (state_dim,)
        reward : float
        done : bool -- True when fold end reached
        info : dict with per-asset details
        """
        actions = np.clip(actions, -1.0, 1.0)
        day_idx = self._fold_start + self._step_idx + 1
        fold_len = self._fold_end - self._fold_start

        if day_idx >= self._fold_end or self._step_idx + 1 >= fold_len:
            return self._build_state_vector(), 0.0, True, {"reason": "fold_end"}

        # Calendar alignment: on non-India-trading days, force India
        # positions flat; only crypto can change.
        india_active = self._is_india_trading_step()
        if not india_active and self.include_crypto:
            for idx in self._india_indices:
                actions[idx] = 0.0  # force flat

        # Per-asset PnL and costs
        total_pnl_delta = 0.0
        total_cost = 0.0
        per_asset_info = {}

        for i, sym in enumerate(self.symbols):
            # Position change
            old_pos = self._positions[i]
            new_pos = actions[i]
            trade = new_pos - old_pos

            # Unified cost model
            cost_fraction = self._cost_models[i].compute_cost(trade)

            # PnL: position x next-day return (from targets)
            if day_idx < len(self.targets):
                ret_i = self.targets[day_idx, i]
                if np.isnan(ret_i):
                    ret_i = 0.0
            else:
                ret_i = 0.0

            pnl_i = old_pos * ret_i  # position at t earns return at t+1
            total_pnl_delta += pnl_i
            total_cost += cost_fraction

            self._positions[i] = new_pos

            per_asset_info[sym] = {
                "position": new_pos,
                "return": ret_i,
                "pnl": pnl_i,
                "cost": cost_fraction,
                "asset_type": self._cost_models[i].asset_type,
            }

        # Update portfolio state
        net_pnl = total_pnl_delta - total_cost
        self._cum_pnl += net_pnl
        self._peak_pnl = max(self._peak_pnl, self._cum_pnl)

        if self._peak_pnl > 0:
            self._drawdown = (self._peak_pnl - self._cum_pnl) / self._peak_pnl
        else:
            self._drawdown = 0.0

        # Reward: risk-adjusted PnL
        dd_threshold = 0.05  # 5% drawdown threshold
        dd_penalty = self.reward_lambda_risk * max(0.0, self._drawdown - dd_threshold)
        reward = net_pnl - dd_penalty

        # Add reward shaping bonus (Pattern 5 -- attention spikes)
        if self._reward_bonuses is not None and self._step_idx < len(self._reward_bonuses):
            reward += float(self._reward_bonuses[self._step_idx])
        elif self._reward_shaper is not None:
            reward += float(self._reward_shaper.get_bonus(self._step_idx))

        self._step_idx += 1
        done = (self._step_idx + 1 >= fold_len) or (day_idx + 1 >= self._fold_end)

        state = self._build_state_vector()
        info = {
            "per_asset": per_asset_info,
            "cum_pnl": self._cum_pnl,
            "drawdown": self._drawdown,
            "total_cost": total_cost,
            "india_active": india_active,
        }
        return state, reward, done, info

    def _build_state_vector(self) -> np.ndarray:
        """Assemble the full state vector.

        Layout: [hidden(asset_0), ..., hidden(asset_n), positions(n),
                 norm_pnl(1), drawdown(1), heat(1), time_feats(3)]
        """
        parts = []

        # Backbone hidden states for current step
        if self._hidden_states is not None and self._step_idx < len(self._hidden_states):
            for a in range(self.n_assets):
                parts.append(self._hidden_states[self._step_idx, a, :])
        else:
            for a in range(self.n_assets):
                parts.append(np.zeros(self.d_hidden, dtype=np.float32))

        # Portfolio state features
        parts.append(self._positions.copy())

        # Normalized PnL (clip to [-5, 5] for stability)
        norm_pnl = np.clip(self._cum_pnl * 100.0, -5.0, 5.0)  # scale up small numbers
        parts.append(np.array([norm_pnl], dtype=np.float32))

        # Drawdown
        parts.append(np.array([self._drawdown], dtype=np.float32))

        # Portfolio heat (total absolute exposure)
        heat = float(np.sum(np.abs(self._positions)))
        parts.append(np.array([heat], dtype=np.float32))

        # Time features: fold progress, day-of-week proxy, month proxy
        fold_len = max(1, self._fold_end - self._fold_start)
        progress = self._step_idx / fold_len
        day_sin = math.sin(2.0 * math.pi * self._step_idx / 5.0)  # weekly cycle
        day_cos = math.cos(2.0 * math.pi * self._step_idx / 5.0)
        parts.append(np.array([progress, day_sin, day_cos], dtype=np.float32))

        return np.concatenate(parts).astype(np.float32)

    def set_reward_shaper(self, shaper) -> None:
        """Attach a reward shaper (e.g. AttentionRewardShaper).

        The shaper must implement ``get_bonus(step_idx: int) -> float``.
        Bonuses are added to the base reward at each step.
        """
        self._reward_shaper = shaper

    def set_reward_bonuses(self, bonuses: np.ndarray) -> None:
        """Pre-set reward bonus array for the current fold.

        Parameters
        ----------
        bonuses : (fold_len,) array of reward bonuses per step.
        """
        self._reward_bonuses = bonuses

    def get_cost_per_leg(self, symbol: str) -> float:
        """Return cost per leg in index points for a given symbol.

        For crypto symbols, returns 0.0 (crypto uses percentage fees).
        """
        return COST_PER_LEG.get(symbol.upper(), 0.0)

    def get_cost_model(self, asset_idx: int) -> AssetCostModel:
        """Return the cost model for a given asset index."""
        return self._cost_models[asset_idx]
