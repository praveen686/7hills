"""Real-time feature engine — VPIN, OFI, Hawkes from live tick data.

Maintains per-symbol state objects fed by the TickCollector callbacks.
Exposes current RegimeSnapshot for each symbol, replacing the kline-based
approximations in crypto_flow.

The key improvement over kline-based features:
  - VPIN: Proper BVC with tick-level price changes, not 5m bar approximation.
    Volume buckets fill naturally as trades arrive. No aliasing.
  - OFI: True Cont-Kukanov-Stoikov from bookTicker bid/ask changes.
    Not the proxy "use taker buy ratio" hack.
  - Hawkes: Actual trade arrival timestamps, not binned kline counts.
    Self-exciting intensity is meaningful at tick scale.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from scipy.stats import norm

from apps.crypto_flow.features import (
    HawkesState,
    OFIState,
    RegimeSnapshot,
    VPINState,
    calibrate_hawkes,
)
from apps.tick_collector.storage import BookTick, TradeTick

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-symbol state
# ---------------------------------------------------------------------------

@dataclass
class SymbolFeatureState:
    """Tracks all microstructure features for one symbol in real time."""

    symbol: str

    # VPIN state — proper tick-level BVC
    vpin: VPINState = field(default_factory=lambda: VPINState(
        bucket_size=50_000.0, n_buckets=50,
    ))
    last_vpin: float = 0.0
    _prev_price: float = 0.0
    _sigma: float = 0.01  # rolling vol for BVC

    # OFI state — true CKS from bookTicker
    ofi: OFIState = field(default_factory=lambda: OFIState(ema_alpha=0.05))
    last_ofi: float = 0.0          # normalized to [-1, 1]
    _ofi_ema_abs: float = 1.0      # running EMA of |raw OFI| for normalization
    _ofi_norm_alpha: float = 0.01  # slow adaptation for normalization scale

    # Hawkes state — trade arrival intensity
    hawkes: HawkesState = field(default_factory=lambda: HawkesState(
        mu=1.0, alpha=0.5, beta=1.5,
    ))
    last_hawkes_ratio: float = 1.0
    _hawkes_calibrated: bool = False
    _recent_trade_times: list[float] = field(default_factory=list)
    _max_trade_history: int = 5000

    # Rolling volatility for BVC
    _recent_log_returns: list[float] = field(default_factory=list)
    _vol_window: int = 200

    # Tick counts
    n_trades: int = 0
    n_books: int = 0
    _last_calibrate_time: float = 0.0
    _calibrate_interval: float = 300.0  # re-calibrate Hawkes every 5 min

    def on_trade(self, tick: TradeTick) -> None:
        """Process one aggTrade tick. Updates VPIN + Hawkes."""
        self.n_trades += 1
        price = tick.price
        qty_usd = price * tick.quantity
        t_sec = tick.timestamp_ms / 1000.0

        # --- VPIN ---
        if self._prev_price > 0 and qty_usd > 0:
            # Update rolling volatility
            log_ret = math.log(price / self._prev_price) if price > 0 and self._prev_price > 0 else 0.0
            self._recent_log_returns.append(log_ret)
            if len(self._recent_log_returns) > self._vol_window:
                self._recent_log_returns = self._recent_log_returns[-self._vol_window:]

            # Compute sigma from recent returns
            if len(self._recent_log_returns) >= 20:
                mean_r = sum(self._recent_log_returns) / len(self._recent_log_returns)
                var_r = sum((r - mean_r) ** 2 for r in self._recent_log_returns) / len(self._recent_log_returns)
                self._sigma = max(math.sqrt(var_r), 1e-8)

            vpin_val = self.vpin.update(price, qty_usd, self._prev_price, self._sigma)
            if vpin_val is not None:
                self.last_vpin = vpin_val

        self._prev_price = price

        # --- Hawkes ---
        self._recent_trade_times.append(t_sec)
        if len(self._recent_trade_times) > self._max_trade_history:
            self._recent_trade_times = self._recent_trade_times[-self._max_trade_history:]

        self.hawkes.event(t_sec)
        self.last_hawkes_ratio = self.hawkes.intensity_ratio(t_sec)

        # Periodic Hawkes re-calibration
        now = time.monotonic()
        if (now - self._last_calibrate_time > self._calibrate_interval
                and len(self._recent_trade_times) >= 100):
            self._recalibrate_hawkes()
            self._last_calibrate_time = now

    def on_book(self, tick: BookTick) -> None:
        """Process one bookTicker tick. Updates OFI (normalized to [-1, 1])."""
        self.n_books += 1
        raw_ofi = self.ofi.update(
            tick.bid_price, tick.ask_price,
            tick.bid_qty, tick.ask_qty,
        )
        # Adaptive normalization: track running scale of |OFI|
        abs_ofi = abs(raw_ofi)
        if abs_ofi > 0:
            self._ofi_ema_abs = (
                (1 - self._ofi_norm_alpha) * self._ofi_ema_abs
                + self._ofi_norm_alpha * abs_ofi
            )
        # Normalize to [-1, 1] using running scale
        self.last_ofi = max(-1.0, min(1.0, raw_ofi / max(self._ofi_ema_abs, 1e-10)))

    def _recalibrate_hawkes(self) -> None:
        """Re-estimate Hawkes parameters from recent trade history."""
        import numpy as np
        times = np.array(self._recent_trade_times[-2000:])
        if len(times) < 50:
            return
        try:
            mu, alpha, beta = calibrate_hawkes(times, window_sec=300.0)
            # Smoothly blend new params (don't jump)
            blend = 0.3
            self.hawkes.mu = (1 - blend) * self.hawkes.mu + blend * mu
            self.hawkes.alpha = (1 - blend) * self.hawkes.alpha + blend * alpha
            self.hawkes.beta = (1 - blend) * self.hawkes.beta + blend * beta
            self._hawkes_calibrated = True
            logger.debug(
                "%s Hawkes recalibrated: mu=%.3f alpha=%.3f beta=%.3f n=%.2f",
                self.symbol, self.hawkes.mu, self.hawkes.alpha,
                self.hawkes.beta, self.hawkes.branching_ratio,
            )
        except Exception as e:
            logger.debug("Hawkes calibration failed for %s: %s", self.symbol, e)

    def regime_snapshot(self) -> RegimeSnapshot:
        """Return current regime assessment."""
        return RegimeSnapshot(
            symbol=self.symbol,
            vpin=self.last_vpin,
            ofi=self.last_ofi,
            hawkes_ratio=self.last_hawkes_ratio,
            kyles_lambda=0.0,  # not computed in real-time (needs batch)
            funding_residual=0.0,  # comes from PCA, not tick data
            source="tick",
        )


# ---------------------------------------------------------------------------
# LiveRegimeTracker — manages all symbols
# ---------------------------------------------------------------------------

class LiveRegimeTracker:
    """Tracks real-time microstructure features for multiple symbols.

    Register as callback with TickCollector:
        tracker = LiveRegimeTracker()
        collector = TickCollector(store, on_trade=tracker.on_trade, on_book=tracker.on_book)

    Then query:
        regime = tracker.get_regime("BTCUSDT")
        all_regimes = tracker.all_regimes()
    """

    def __init__(self, vpin_bucket_size: float = 50_000.0, vpin_n_buckets: int = 50):
        self._states: dict[str, SymbolFeatureState] = {}
        self._vpin_bucket_size = vpin_bucket_size
        self._vpin_n_buckets = vpin_n_buckets

    def _ensure_state(self, symbol: str) -> SymbolFeatureState:
        if symbol not in self._states:
            self._states[symbol] = SymbolFeatureState(
                symbol=symbol,
                vpin=VPINState(
                    bucket_size=self._vpin_bucket_size,
                    n_buckets=self._vpin_n_buckets,
                ),
            )
        return self._states[symbol]

    def on_trade(self, symbol: str, tick: TradeTick) -> None:
        """Callback for TickCollector — process trade tick."""
        state = self._ensure_state(symbol)
        state.on_trade(tick)

    def on_book(self, symbol: str, tick: BookTick) -> None:
        """Callback for TickCollector — process book tick."""
        state = self._ensure_state(symbol)
        state.on_book(tick)

    def get_regime(self, symbol: str) -> RegimeSnapshot | None:
        """Get current regime for a symbol, or None if not tracked."""
        state = self._states.get(symbol)
        if state is None:
            return None
        return state.regime_snapshot()

    def all_regimes(self) -> dict[str, RegimeSnapshot]:
        """Get regime snapshots for all tracked symbols."""
        return {sym: state.regime_snapshot() for sym, state in self._states.items()}

    def remove_symbol(self, symbol: str) -> None:
        """Stop tracking a symbol."""
        self._states.pop(symbol, None)

    def stats(self) -> dict:
        """Feature tracker statistics."""
        per_symbol = {}
        for sym, state in self._states.items():
            per_symbol[sym] = {
                "vpin": state.last_vpin,
                "ofi": state.last_ofi,
                "hawkes_ratio": state.last_hawkes_ratio,
                "hawkes_calibrated": state._hawkes_calibrated,
                "n_trades": state.n_trades,
                "n_books": state.n_books,
            }
        return {
            "n_symbols": len(self._states),
            "per_symbol": per_symbol,
        }
