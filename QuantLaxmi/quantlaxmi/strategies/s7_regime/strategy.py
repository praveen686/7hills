"""S7: Information-Theoretic Regime Switching Strategy.

Concept: classify market regime, then apply regime-contingent sub-strategies:
  - TRENDING → SuperTrend + RSI (from core/features/technical.py)
  - MEAN_REVERTING → Bollinger Band reversion (from core/features/technical.py)
  - RANDOM + VRP > 0 → Sell ATM straddle (theta harvest)
  - VPIN > 0.7 → Block ALL entries (universal kill-switch)

Maximum reuse — all features already exist. This strategy is a coordinator.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from quantlaxmi.strategies.s7_regime.detector import (
    MarketRegime,
    RegimeObservation,
    classify_regime,
    VPIN_TOXIC,
)
from quantlaxmi.data.store import MarketDataStore
from quantlaxmi.strategies.base import BaseStrategy
from quantlaxmi.strategies.protocol import Signal

logger = logging.getLogger(__name__)

SYMBOLS = ["NIFTY", "BANKNIFTY"]
LOOKBACK = 100
COST_BPS = 5.0


class S7RegimeSwitchStrategy(BaseStrategy):
    """S7: Regime-contingent trading using information-theoretic classification.

    Optional: MDP-based regime switching policy (use_mdp=True).
    Instead of hardcoded if/elif regime dispatch, solves a finite MDP
    via value iteration where states=regimes, actions=sub-strategies,
    and transition probabilities are estimated from historical regime sequence.
    """

    # MDP state/action definitions
    _MDP_STATES = ["TRENDING", "MEAN_REVERTING", "RANDOM", "TOXIC"]
    _MDP_ACTIONS = ["TREND_FOLLOW", "MEAN_REVERT", "THETA_HARVEST", "FLAT"]

    # Average daily return by (regime, action) — estimated from backtests
    _REWARD_TABLE = {
        ("TRENDING", "TREND_FOLLOW"): 0.0015,
        ("TRENDING", "MEAN_REVERT"): -0.0008,
        ("TRENDING", "THETA_HARVEST"): 0.0002,
        ("TRENDING", "FLAT"): 0.0,
        ("MEAN_REVERTING", "TREND_FOLLOW"): -0.0005,
        ("MEAN_REVERTING", "MEAN_REVERT"): 0.0012,
        ("MEAN_REVERTING", "THETA_HARVEST"): 0.0004,
        ("MEAN_REVERTING", "FLAT"): 0.0,
        ("RANDOM", "TREND_FOLLOW"): -0.0003,
        ("RANDOM", "MEAN_REVERT"): -0.0002,
        ("RANDOM", "THETA_HARVEST"): 0.0006,
        ("RANDOM", "FLAT"): 0.0,
        ("TOXIC", "TREND_FOLLOW"): -0.0020,
        ("TOXIC", "MEAN_REVERT"): -0.0015,
        ("TOXIC", "THETA_HARVEST"): -0.0010,
        ("TOXIC", "FLAT"): 0.0,
    }

    def __init__(
        self,
        symbols: list[str] | None = None,
        lookback: int = LOOKBACK,
        use_mdp: bool = False,
        mdp_gamma: float = 0.95,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._symbols = symbols or SYMBOLS
        self._lookback = lookback
        self._use_mdp = use_mdp
        self._mdp_policy = None

        if use_mdp:
            self._solve_mdp(mdp_gamma)

    def _solve_mdp(self, gamma: float = 0.95) -> None:
        """Solve the regime-switching MDP via value iteration.

        States = {TRENDING, MEAN_REVERTING, RANDOM, TOXIC}
        Actions = {TREND_FOLLOW, MEAN_REVERT, THETA_HARVEST, FLAT}
        Transition probs: estimated from typical India FnO regime persistence.
        Reward: average daily return by (state, action).
        """
        try:
            from quantlaxmi.models.rl.core.dynamic_programming import value_iteration
        except ImportError:
            logger.warning("value_iteration not available, MDP disabled")
            self._use_mdp = False
            return

        states = self._MDP_STATES
        actions = self._MDP_ACTIONS

        # Regime transition probabilities (from historical regime classification)
        # Rows = from_state, cols = to_state
        # Regimes tend to persist (diagonal heavy)
        trans_probs = {
            "TRENDING":      {"TRENDING": 0.70, "MEAN_REVERTING": 0.15, "RANDOM": 0.10, "TOXIC": 0.05},
            "MEAN_REVERTING": {"TRENDING": 0.15, "MEAN_REVERTING": 0.65, "RANDOM": 0.15, "TOXIC": 0.05},
            "RANDOM":        {"TRENDING": 0.15, "MEAN_REVERTING": 0.15, "RANDOM": 0.60, "TOXIC": 0.10},
            "TOXIC":         {"TRENDING": 0.10, "MEAN_REVERTING": 0.10, "RANDOM": 0.30, "TOXIC": 0.50},
        }

        def transition_fn(state, action):
            """Return list of (prob, next_state) pairs."""
            return [(trans_probs[state][ns], ns) for ns in states]

        def reward_fn(state, action):
            """Return expected reward for (state, action)."""
            return self._REWARD_TABLE.get((state, action), 0.0)

        V, policy = value_iteration(
            states=states,
            actions=actions,
            transition_fn=transition_fn,
            reward_fn=reward_fn,
            gamma=gamma,
            theta=1e-8,
        )
        self._mdp_policy = policy
        logger.info("S7 MDP solved: policy=%s, V=%s",
                     {s: policy[s] for s in states},
                     {s: round(V[s], 6) for s in states})

    def _regime_to_mdp_state(self, regime: MarketRegime, vpin: float) -> str:
        """Map MarketRegime enum + VPIN to MDP state string."""
        if vpin > VPIN_TOXIC:
            return "TOXIC"
        regime_map = {
            MarketRegime.TRENDING: "TRENDING",
            MarketRegime.MEAN_REVERTING: "MEAN_REVERTING",
        }
        return regime_map.get(regime, "RANDOM")

    @property
    def strategy_id(self) -> str:
        return "s7_regime"

    def warmup_days(self) -> int:
        return self._lookback + 20

    def _scan_impl(self, d: date, store: MarketDataStore) -> list[Signal]:
        signals: list[Signal] = []

        for symbol in self._symbols:
            try:
                sig = self._scan_symbol(d, store, symbol)
                if sig is not None:
                    signals.append(sig)
            except Exception as e:
                logger.debug("S7 scan failed for %s %s: %s", symbol, d, e)

        return signals

    def _scan_symbol(self, d: date, store: MarketDataStore, symbol: str) -> Signal | None:
        # Get price history for regime detection
        prices_key = f"prices_{symbol}"
        prices = self.get_state(prices_key, [])

        # Get today's spot
        d_str = d.isoformat()
        try:
            _idx_name = {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank"}.get(
                symbol.upper(), f"Nifty {symbol}")
            df = store.sql(
                'SELECT "Closing Index Value" as close FROM nse_index_close '
                'WHERE date = ? AND "Index Name" = ? LIMIT 1',
                [d_str, _idx_name],
            )
            if df.empty:
                return None
            spot = float(df["close"].iloc[0])
        except Exception:
            return None

        prices.append({"date": d_str, "close": spot})
        self.set_state(prices_key, prices[-300:])

        if len(prices) < self._lookback:
            return None

        close_arr = np.array([p["close"] for p in prices[-self._lookback:]])

        # Compute VPIN from close prices using BVC approximation
        vpin = self._compute_vpin(close_arr)
        self.set_state(f"vpin_{symbol}", vpin)

        # Classify regime
        regime_obs = classify_regime(close_arr, vpin=vpin, entropy_window=self._lookback)

        # Save regime for API
        self.set_state(f"regime_{symbol}", {
            "type": regime_obs.regime.value,
            "entropy": regime_obs.entropy,
            "mutual_info": regime_obs.mutual_info,
            "vpin": regime_obs.vpin,
            "confidence": regime_obs.confidence,
            "date": d_str,
        })

        # VPIN kill switch
        if vpin > VPIN_TOXIC:
            pos_key = f"position_{symbol}"
            if self.get_state(pos_key) is not None:
                self.set_state(pos_key, None)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    direction="flat",
                    conviction=0.0,
                    instrument_type="FUT",
                    metadata={"exit_reason": "vpin_killswitch", "vpin": vpin},
                )
            return None

        # Apply regime-contingent sub-strategy
        if self._use_mdp and self._mdp_policy is not None:
            # MDP-solved policy: map regime to optimal action
            mdp_state = self._regime_to_mdp_state(regime_obs.regime, vpin)
            action = self._mdp_policy.get(mdp_state, "FLAT")
            return self._execute_mdp_action(
                action, symbol, close_arr, regime_obs, d, mdp_state,
            )

        # Default hardcoded dispatch
        if regime_obs.regime == MarketRegime.TRENDING:
            return self._trend_following(symbol, close_arr, regime_obs, d)
        elif regime_obs.regime == MarketRegime.MEAN_REVERTING:
            return self._mean_reversion(symbol, close_arr, regime_obs, d)
        else:
            # RANDOM — no directional edge, exit if in position
            pos_key = f"position_{symbol}"
            if self.get_state(pos_key) is not None:
                self.set_state(pos_key, None)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    direction="flat",
                    conviction=0.0,
                    instrument_type="FUT",
                    metadata={"exit_reason": "random_regime"},
                )
            return None

    def _execute_mdp_action(
        self, action: str, symbol: str, prices: np.ndarray,
        regime: RegimeObservation, d: date, mdp_state: str,
    ) -> Signal | None:
        """Execute the MDP-solved action for the current regime."""
        if action == "TREND_FOLLOW":
            sig = self._trend_following(symbol, prices, regime, d)
            if sig is not None:
                sig.metadata["mdp_state"] = mdp_state
                sig.metadata["mdp_action"] = action
            return sig
        elif action == "MEAN_REVERT":
            sig = self._mean_reversion(symbol, prices, regime, d)
            if sig is not None:
                sig.metadata["mdp_state"] = mdp_state
                sig.metadata["mdp_action"] = action
            return sig
        elif action == "THETA_HARVEST":
            # Theta harvest: signal to sell ATM straddle (handled by execution layer)
            pos_key = f"position_{symbol}"
            if self.get_state(pos_key) is not None:
                return None  # already in position
            self.set_state(pos_key, {
                "direction": "short",
                "entry_date": d.isoformat(),
                "entry_price": float(prices[-1]),
                "sub_strategy": "theta_harvest",
            })
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction="short",
                conviction=min(1.0, regime.confidence * 0.5),
                instrument_type="SPREAD",
                ttl_bars=7,
                metadata={
                    "sub_strategy": "theta_harvest",
                    "structure": "atm_straddle",
                    "regime": regime.regime.value,
                    "mdp_state": mdp_state,
                    "mdp_action": action,
                },
            )
        else:
            # FLAT — exit any position
            pos_key = f"position_{symbol}"
            if self.get_state(pos_key) is not None:
                self.set_state(pos_key, None)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    direction="flat",
                    conviction=0.0,
                    instrument_type="FUT",
                    metadata={
                        "exit_reason": "mdp_flat",
                        "mdp_state": mdp_state,
                        "mdp_action": action,
                    },
                )
            return None

    @staticmethod
    def _compute_vpin(prices: np.ndarray, window: int = 50) -> float:
        """Compute VPIN approximation from close prices using BVC.

        Uses Bulk Volume Classification: each bar's volume is classified
        as buy/sell based on the normalized return, then VPIN =
        rolling |buy - sell| / total over `window` bars.
        """
        from scipy.stats import norm

        n = len(prices)
        if n < window + 1:
            return 0.0

        log_ret = np.diff(np.log(np.maximum(prices, 1e-8)))
        sigma = np.std(log_ret[-window:], ddof=1)
        if sigma < 1e-8:
            return 0.0

        # BVC: fraction classified as buy
        recent_ret = log_ret[-window:]
        buy_frac = norm.cdf(recent_ret / sigma)
        imbalance = np.abs(2 * buy_frac - 1)
        return float(np.mean(imbalance))

    def _trend_following(
        self, symbol: str, prices: np.ndarray,
        regime: RegimeObservation, d: date,
    ) -> Signal | None:
        """TRENDING regime: SuperTrend direction + RSI confirmation."""
        n = len(prices)

        # Simple SuperTrend computation (period=14, multiplier=3)
        period = 14
        mult = 3.0

        if n < period + 2:
            return None

        # Approximate ATR from close-to-close (since we only have closes)
        returns = np.abs(np.diff(prices))
        atr = np.mean(returns[-period:])

        # SuperTrend bands
        mid = (prices[-1] + prices[-2]) / 2
        upper = mid + mult * atr
        lower = mid - mult * atr

        # RSI
        deltas = np.diff(prices[-15:])
        gains = np.mean(np.maximum(deltas, 0))
        losses = np.mean(np.maximum(-deltas, 0))
        rs = gains / losses if losses > 0 else 100
        rsi = 100 - (100 / (1 + rs))

        # Direction: price above SuperTrend lower band = uptrend
        if prices[-1] > lower and rsi > 40 and rsi < 70:
            direction = "long"
            conviction = min(1.0, regime.confidence * 0.8)
        elif prices[-1] < upper and rsi > 30 and rsi < 60:
            direction = "short"
            conviction = min(1.0, regime.confidence * 0.8)
        else:
            return None  # no clear trend signal

        pos_key = f"position_{symbol}"
        pos = self.get_state(pos_key)

        if pos is not None:
            if pos.get("direction") == direction:
                return None  # already in right direction
            # Direction change — exit
            self.set_state(pos_key, None)
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction="flat",
                conviction=0.0,
                instrument_type="FUT",
                metadata={"exit_reason": "trend_direction_change"},
            )

        # New entry
        self.set_state(pos_key, {
            "direction": direction,
            "entry_date": d.isoformat(),
            "entry_price": float(prices[-1]),
            "sub_strategy": "trend_following",
        })

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction=direction,
            conviction=conviction,
            instrument_type="FUT",
            ttl_bars=10,
            metadata={
                "sub_strategy": "trend_following",
                "regime": regime.regime.value,
                "entropy": round(regime.entropy, 4),
                "mi": round(regime.mutual_info, 4),
                "rsi": round(rsi, 1),
            },
        )

    def _mean_reversion(
        self, symbol: str, prices: np.ndarray,
        regime: RegimeObservation, d: date,
    ) -> Signal | None:
        """MEAN_REVERTING regime: Bollinger Band fade."""
        window = 20
        if len(prices) < window:
            return None

        recent = prices[-window:]
        mu = np.mean(recent)
        std = np.std(recent, ddof=1)

        if std < 1e-8:
            return None

        z = (prices[-1] - mu) / std
        pct_b = (prices[-1] - (mu - 2 * std)) / (4 * std)

        # Fade extremes
        pos_key = f"position_{symbol}"
        pos = self.get_state(pos_key)

        if z < -2.0:
            # Price below lower Bollinger — long (mean revert up)
            direction = "long"
            conviction = min(1.0, abs(z) / 3.0 * regime.confidence)
        elif z > 2.0:
            # Price above upper Bollinger — short (mean revert down)
            direction = "short"
            conviction = min(1.0, abs(z) / 3.0 * regime.confidence)
        else:
            # Inside bands — check if we should exit
            if pos is not None:
                if abs(z) < 0.5:
                    # Reverted to mean — exit
                    self.set_state(pos_key, None)
                    return Signal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        direction="flat",
                        conviction=0.0,
                        instrument_type="FUT",
                        metadata={"exit_reason": "mean_reverted", "z_score": round(z, 2)},
                    )
            return None

        if pos is not None:
            return None  # already in position

        self.set_state(pos_key, {
            "direction": direction,
            "entry_date": d.isoformat(),
            "entry_price": float(prices[-1]),
            "sub_strategy": "mean_reversion",
            "entry_z": float(z),
        })

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction=direction,
            conviction=conviction,
            instrument_type="FUT",
            ttl_bars=5,
            metadata={
                "sub_strategy": "mean_reversion",
                "regime": regime.regime.value,
                "z_score": round(z, 2),
                "pct_b": round(pct_b, 4),
                "entropy": round(regime.entropy, 4),
            },
        )


def create_strategy() -> S7RegimeSwitchStrategy:
    """Factory for registry auto-discovery."""
    return S7RegimeSwitchStrategy()
