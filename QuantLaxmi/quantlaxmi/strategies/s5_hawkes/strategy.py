"""S5 Hawkes Microstructure Strategy wrapper.

Wraps the microstructure signals.py (GEX + OI + IV term + basis + PCR)
into the unified StrategyProtocol.

Edge: Dealer mechanics (GEX), institutional flow, and panic indicators.
Sharpe: 4.29
"""

from __future__ import annotations

import logging
from datetime import date

from quantlaxmi.data.store import MarketDataStore
from quantlaxmi.strategies.base import BaseStrategy
from quantlaxmi.strategies.protocol import Signal

logger = logging.getLogger(__name__)

SYMBOLS = ["NIFTY", "BANKNIFTY"]


class S5HawkesStrategy(BaseStrategy):
    """S5: Microstructure signal combining GEX, OI flow, IV term, basis, PCR.

    This strategy wraps the full microstructure analytics pipeline.
    On each scan, it computes a snapshot of the options market and
    generates a combined trade signal from weighted components.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        entry_threshold: float = 0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._symbols = symbols or SYMBOLS
        self._entry_threshold = entry_threshold

    @property
    def strategy_id(self) -> str:
        return "s5_hawkes"

    def warmup_days(self) -> int:
        return 5  # needs a few days of EMA history

    def _scan_impl(self, d: date, store: MarketDataStore) -> list[Signal]:
        signals: list[Signal] = []

        for symbol in self._symbols:
            try:
                sig = self._scan_symbol(d, store, symbol)
                if sig is not None:
                    signals.append(sig)
            except Exception as e:
                logger.debug("S5 scan failed for %s %s: %s", symbol, d, e)

        return signals

    def _scan_symbol(self, d: date, store: MarketDataStore, symbol: str) -> Signal | None:
        from quantlaxmi.strategies.s5_hawkes.analytics import compute_snapshot
        from quantlaxmi.strategies.s5_hawkes.signals import generate_signal

        # compute_snapshot needs an option chain + futures for the date
        try:
            snap = compute_snapshot(store, d, symbol)
        except Exception as e:
            logger.debug("Snapshot failed for %s %s: %s", symbol, d, e)
            return None

        if snap is None:
            return None

        # Generate combined signal
        trade_signal = generate_signal(snap, entry_threshold=self._entry_threshold)

        if trade_signal.direction == "flat":
            # Check if we need to exit an existing position
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
                    metadata={"exit_reason": "signal_flat"},
                )
            return None

        # Non-flat signal
        pos_key = f"position_{symbol}"
        pos = self.get_state(pos_key)

        # If already in position in same direction, hold
        if pos is not None and pos.get("direction") == trade_signal.direction:
            return None

        # New entry or direction flip
        if pos is not None:
            # Direction flip — emit flat first, then new signal on next scan
            self.set_state(pos_key, None)
            return Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction="flat",
                conviction=0.0,
                instrument_type="FUT",
                ttl_bars=0,
                metadata={"exit_reason": "direction_flip"},
            )

        # New entry
        self.set_state(pos_key, {
            "entry_date": d.isoformat(),
            "direction": trade_signal.direction,
            "spot": trade_signal.spot,
        })

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction=trade_signal.direction,
            conviction=trade_signal.conviction,
            instrument_type="FUT",
            ttl_bars=5,
            metadata={
                "gex_regime": trade_signal.gex_regime,
                "components": trade_signal.components,
                "reasoning": trade_signal.reasoning,
                "raw_score": round(trade_signal.raw_score, 4),
                "smoothed_score": round(trade_signal.smoothed_score, 4),
            },
        )


def create_strategy() -> S5HawkesStrategy:
    """Factory for registry auto-discovery."""
    return S5HawkesStrategy()
