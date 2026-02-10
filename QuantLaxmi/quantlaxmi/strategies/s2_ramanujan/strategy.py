"""S2: Ramanujan Cycles — Intraday phase-based trading strategy.

Uses Ramanujan sum periodograms to detect dominant intraday cycles
in 1-min futures bars, then trades phase crossings in the second half
of the day. All operations are strictly causal:
  - Trailing-only detrending (left-aligned rolling mean)
  - Expanding-window periodogram from first half of day only
  - Causal linear convolution (no circular FFT)
  - Causal phase estimation via quadrature filters (no Hilbert)

Instruments: NIFTY, BANKNIFTY index futures (1-min bars).
Holding period: intraday (entry in 2nd half, exit at EOD or phase exit).
"""

from __future__ import annotations

import logging
import math
from datetime import date

import numpy as np
import pandas as pd

from quantlaxmi.data.store import MarketDataStore
from quantlaxmi.features.ramanujan import ramanujan_periodogram, ramanujan_sum
from quantlaxmi.strategies.base import BaseStrategy
from quantlaxmi.strategies.protocol import Signal

logger = logging.getLogger(__name__)

SYMBOLS = ["NIFTY", "BANKNIFTY"]
MAX_PERIOD = 64
TOP_K = 3
PHASE_ENTRY = -1.5   # phase crosses above this -> go long
PHASE_EXIT = 1.0     # phase crosses above this -> go flat


# ---------------------------------------------------------------------------
# Causal helpers (ported from research.py — strictly causal, no look-ahead)
# ---------------------------------------------------------------------------

def _causal_detrend(close: np.ndarray, window: int) -> np.ndarray:
    """Trailing-only detrend using left-aligned rolling mean."""
    trend = pd.Series(close).rolling(window, min_periods=1).mean().values
    return close - trend


def _causal_filter(signal: np.ndarray, q: int) -> np.ndarray:
    """Causal Ramanujan filter via linear convolution.

    Uses np.convolve in 'full' mode, then takes only the causal part
    (output[j] depends only on signal[0..j]).
    """
    filt = np.array(
        [ramanujan_sum(q, n) for n in range(q)], dtype=np.float64
    ) / q
    conv = np.convolve(signal, filt, mode="full")[: len(signal)]
    return conv


def _causal_phase(signal: np.ndarray, period: int) -> np.ndarray:
    """Causal phase estimation using quadrature Ramanujan filters.

    Builds cosine-like and sine-like filter pair at the target period,
    applies both causally, then computes phase = arctan2(sin, cos).
    """
    n = len(signal)
    length = period * 2  # filter length = 2 full cycles

    cos_filt = np.array(
        [math.cos(2 * math.pi * k / period) for k in range(length)]
    )
    sin_filt = np.array(
        [math.sin(2 * math.pi * k / period) for k in range(length)]
    )

    # Hann window to reduce spectral leakage
    hann = np.hanning(length)
    cos_filt *= hann
    sin_filt *= hann

    # Causal convolution (output[j] uses only past data)
    cos_out = np.convolve(signal, cos_filt[::-1], mode="full")[:n]
    sin_out = np.convolve(signal, sin_filt[::-1], mode="full")[:n]

    return np.arctan2(sin_out, cos_out)


def _expanding_periodogram(
    detrended: np.ndarray,
    j: int,
    max_period: int,
    min_window: int = 64,
) -> np.ndarray:
    """Compute periodogram using only bars up to bar j (causal)."""
    start = max(0, j - min_window + 1)
    segment = detrended[start : j + 1]
    if len(segment) < max_period:
        return np.zeros(max_period)
    return ramanujan_periodogram(segment, max_period)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class S2RamanujanStrategy(BaseStrategy):
    """S2: Ramanujan Cycles — Intraday phase-crossing strategy.

    Parameters
    ----------
    symbols : list[str], optional
        Index names to trade. Default: ["NIFTY", "BANKNIFTY"].
    max_period : int
        Maximum cycle period to search (bars). Default: 64.
    top_k : int
        Number of top periods to consider. Default: 3.
    phase_entry : float
        Phase threshold for entry (cross above). Default: -1.5.
    phase_exit : float
        Phase threshold for exit (cross above). Default: 1.0.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        max_period: int = MAX_PERIOD,
        top_k: int = TOP_K,
        phase_entry: float = PHASE_ENTRY,
        phase_exit: float = PHASE_EXIT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._symbols = symbols or SYMBOLS
        self._max_period = max_period
        self._top_k = top_k
        self._phase_entry = phase_entry
        self._phase_exit = phase_exit

    @property
    def strategy_id(self) -> str:
        return "s2_ramanujan"

    def warmup_days(self) -> int:
        return 1  # intraday strategy — needs only today's 1-min bars

    def _load_bars(self, d: date, symbol: str, store: MarketDataStore) -> pd.DataFrame:
        """Load 1-min FUT bars for symbol on date d (nearest expiry)."""
        try:
            df = store.sql(
                "SELECT * FROM nfo_1min WHERE date = ? AND name = ? "
                "AND instrument_type = 'FUT'",
                [d.isoformat(), symbol],
            )
            if df is not None and not df.empty:
                if "expiry" in df.columns:
                    df["_exp"] = pd.to_datetime(
                        df["expiry"], format="mixed", errors="coerce"
                    )
                    min_exp = df["_exp"].min()
                    df = df[df["_exp"] == min_exp].drop(columns=["_exp"])
                if "timestamp" in df.columns:
                    df = df.sort_values("timestamp")
                return df
        except Exception:
            pass
        return pd.DataFrame()

    def _scan_impl(self, d: date, store: MarketDataStore) -> list[Signal]:
        """Scan 1-min bars for each symbol, detect cycle phase crossings."""
        signals: list[Signal] = []

        for symbol in self._symbols:
            sig = self._scan_symbol(d, symbol, store)
            if sig is not None:
                signals.append(sig)

        return signals

    def _scan_symbol(
        self, d: date, symbol: str, store: MarketDataStore
    ) -> Signal | None:
        """Detect dominant period and check for phase entry/exit on one symbol."""
        bars = self._load_bars(d, symbol, store)
        if bars.empty or len(bars) < self._max_period * 2:
            return None

        close = bars["close"].values.astype(np.float64)

        # 1. Causal detrend
        window = max(2, min(20, len(close) // 5))
        detrended = _causal_detrend(close, window)

        # 2. Detect dominant period from first half (causal)
        mid = max(len(detrended) // 2, self._max_period)
        energies = _expanding_periodogram(
            detrended, mid, self._max_period, min_window=self._max_period
        )
        energies[0] = 0  # skip DC
        dom_indices = np.argsort(energies)[::-1][: self._top_k]
        dom_periods = [
            int(idx + 1) for idx in dom_indices if energies[idx] > 0
        ]

        if not dom_periods:
            return None

        primary = dom_periods[0]

        # 3. Causal phase estimation
        phase = _causal_phase(detrended, primary)

        # 4. Trade only in second half (period detected from first half)
        trade_start = mid + primary * 2  # allow filter warmup
        if trade_start >= len(close) - 1:
            return None

        # Check the most recent phase crossing
        last_bar = len(close) - 1
        in_position = False
        entry_price = 0.0
        final_direction = None
        day_pnl = 0.0
        n_trades = 0

        for j in range(max(trade_start, 1), len(close)):
            if not in_position:
                if phase[j - 1] < self._phase_entry and phase[j] >= self._phase_entry:
                    in_position = True
                    entry_price = close[j]
            else:
                if phase[j - 1] < self._phase_exit and phase[j] >= self._phase_exit:
                    pnl = (close[j] - entry_price) / entry_price
                    day_pnl += pnl
                    n_trades += 1
                    in_position = False

        # Still in position at last bar -> emit entry signal
        if in_position:
            final_direction = "long"
            conviction = min(1.0, float(energies[primary - 1]) / max(float(energies.max()), 1e-8))
        elif n_trades > 0 and j == last_bar:
            # Last trade just exited -> flat
            final_direction = "flat"
            conviction = 0.0
        else:
            # No active position, no recent trade
            # Check if a crossing is imminent (within last 3 bars)
            if last_bar >= trade_start + 1:
                if (
                    phase[last_bar - 1] < self._phase_entry
                    and phase[last_bar] >= self._phase_entry
                ):
                    final_direction = "long"
                    conviction = min(
                        1.0,
                        float(energies[primary - 1])
                        / max(float(energies.max()), 1e-8),
                    )

        if final_direction is None:
            return None

        metadata = {
            "dominant_period": primary,
            "energy": round(float(energies[primary - 1]), 4),
            "n_bars": len(close),
            "n_trades_today": n_trades,
            "day_pnl": round(day_pnl, 6),
            "dom_periods": dom_periods[:3],
        }

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction=final_direction,
            conviction=conviction,
            instrument_type="FUT",
            ttl_bars=1,  # intraday — exit by EOD
            metadata=metadata,
        )


def create_strategy() -> S2RamanujanStrategy:
    """Factory for registry auto-discovery."""
    return S2RamanujanStrategy()
