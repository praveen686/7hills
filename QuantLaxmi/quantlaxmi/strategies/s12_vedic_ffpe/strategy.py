"""S12 Vedic Fractional Alpha — BaseStrategy wrapper.

EOD strategy: compute features from daily closing prices, emit signals
for T+1 execution.  Uses fractional diffusion α to classify regimes
and ancient Indian mathematics for signal timing.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from quantlaxmi.data.store import MarketDataStore
from quantlaxmi.strategies.base import BaseStrategy
from quantlaxmi.strategies.protocol import Signal
from quantlaxmi.strategies.s12_vedic_ffpe.signals import compute_daily_signal

logger = logging.getLogger(__name__)

SYMBOLS = ["NIFTY", "BANKNIFTY"]


class S12VedicFFPEStrategy(BaseStrategy):
    """S12: Vedic Fractional Alpha strategy.

    Combines fractional diffusion regime classification with Ramanujan
    mock theta, Madhava angular kernel, and Aryabhata phase features
    to generate contrarian/momentum signals.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        alpha_window: int = 60,
        alpha_lo: float = 0.85,
        alpha_hi: float = 1.15,
        frac_d: float = 0.226,
        min_conviction: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._symbols = symbols or SYMBOLS
        self._alpha_window = alpha_window
        self._alpha_lo = alpha_lo
        self._alpha_hi = alpha_hi
        self._frac_d = frac_d
        self._min_conviction = min_conviction

    @property
    def strategy_id(self) -> str:
        return "s12_vedic_ffpe"

    def warmup_days(self) -> int:
        return self._alpha_window + 10

    def _scan_impl(self, d: date, store: MarketDataStore) -> list[Signal]:
        signals: list[Signal] = []

        for symbol in self._symbols:
            try:
                sig = self._scan_symbol(d, store, symbol)
                if sig is not None:
                    signals.append(sig)
            except Exception as e:
                logger.debug("S12 scan failed for %s %s: %s", symbol, d, e)

        return signals

    def _scan_symbol(self, d: date, store: MarketDataStore, symbol: str) -> Signal | None:
        """Compute signal for a single symbol on date d."""
        # Load daily closing prices up to date d
        close_col = "Closing Index Value"
        name_col = "Index Name"

        # Query index close table
        df = store.sql(
            f'SELECT date, "{close_col}" as close_val '
            f'FROM nse_index_close '
            f'WHERE "{name_col}" ILIKE ? AND date <= ? '
            f'ORDER BY date',
            [f"%{symbol}%", d.isoformat()],
        )

        if df is None or df.empty or len(df) < self._alpha_window + 5:
            return None

        closes = df["close_val"].values.astype(np.float64)

        # Retrieve persistent state for this symbol
        state_key = f"vedic_state_{symbol}"
        prev_state = self.get_state(state_key, {})

        signal, new_state = compute_daily_signal(
            closes,
            alpha_window=self._alpha_window,
            alpha_lo=self._alpha_lo,
            alpha_hi=self._alpha_hi,
            frac_d=self._frac_d,
            min_conviction=self._min_conviction,
            prev_regime=prev_state.get("prev_regime", "normal"),
            bars_in_regime=prev_state.get("bars_in_regime", 0),
            regime_centroids=prev_state.get("regime_centroids"),
        )

        # Persist state (centroids are numpy arrays — convert for JSON)
        serialisable_state = {
            "prev_regime": new_state["prev_regime"],
            "bars_in_regime": new_state["bars_in_regime"],
            # centroids as lists for JSON serialisation
            "regime_centroids": {
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in new_state.get("regime_centroids", {}).items()
            },
        }
        self.set_state(state_key, serialisable_state)

        if signal.direction == "flat":
            # Check if we need to exit existing position
            pos_key = f"position_{symbol}"
            pos = self.get_state(pos_key)
            if pos is not None:
                self.set_state(pos_key, None)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    direction="flat",
                    conviction=0.0,
                    instrument_type="FUT",
                    ttl_bars=0,
                    metadata={"exit_reason": "signal_flat", "regime": signal.regime},
                )
            return None

        # Non-flat signal
        pos_key = f"position_{symbol}"
        pos = self.get_state(pos_key)

        # Already in same direction — hold
        if pos is not None and pos.get("direction") == signal.direction:
            return None

        # Direction flip — exit first
        if pos is not None:
            self.set_state(pos_key, None)
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction="flat",
                conviction=0.0,
                instrument_type="FUT",
                ttl_bars=0,
                metadata={"exit_reason": "direction_flip", "regime": signal.regime},
            )

        # New entry
        self.set_state(pos_key, {
            "entry_date": d.isoformat(),
            "direction": signal.direction,
            "regime": signal.regime,
        })

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction=signal.direction,
            conviction=signal.conviction,
            instrument_type="FUT",
            ttl_bars=5,
            metadata={
                "regime": signal.regime,
                "alpha": round(signal.alpha, 4),
                "phase": round(signal.phase, 4),
                "coherence": round(signal.coherence, 4),
                "mock_theta_div": round(signal.mock_theta_div, 4),
                "frac_d_value": round(signal.frac_d_value, 4),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in signal.metadata.items()},
            },
        )


def create_strategy() -> S12VedicFFPEStrategy:
    """Factory for registry auto-discovery."""
    return S12VedicFFPEStrategy()
