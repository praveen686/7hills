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

from strategies.s7_regime.detector import (
    MarketRegime,
    RegimeObservation,
    classify_regime,
    VPIN_TOXIC,
)
from core.market.store import MarketDataStore
from core.strategy.base import BaseStrategy
from core.strategy.protocol import Signal

logger = logging.getLogger(__name__)

SYMBOLS = ["NIFTY", "BANKNIFTY"]
LOOKBACK = 100
COST_BPS = 5.0


class S7RegimeSwitchStrategy(BaseStrategy):
    """S7: Regime-contingent trading using information-theoretic classification."""

    def __init__(
        self,
        symbols: list[str] | None = None,
        lookback: int = LOOKBACK,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._symbols = symbols or SYMBOLS
        self._lookback = lookback

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
