"""S3: Institutional Flow — Composite flow-based stock futures strategy.

Leverages the S9 momentum scanner's composite signals (delivery spikes,
OI buildup, FII/DII flows) to generate directional trades on stock
futures. Tracks position state for hold-period management and score
reversal exits.

Instruments: FnO stock futures (universe from S9 scanner).
Holding period: 5 days (configurable) or until score reversal.
"""

from __future__ import annotations

import logging
from datetime import date

from quantlaxmi.data.store import MarketDataStore
from quantlaxmi.strategies.base import BaseStrategy
from quantlaxmi.strategies.protocol import Signal

logger = logging.getLogger(__name__)

DEFAULT_HOLD_DAYS = 5
DEFAULT_TOP_N = 10


class S3InstitutionalStrategy(BaseStrategy):
    """S3: Institutional Flow — Composite scanner strategy.

    Uses ``run_daily_scan`` from S9 momentum to obtain composite scores
    built from delivery, OI, and FII data, then maps them to directional
    signals on stock futures.

    Parameters
    ----------
    hold_days : int
        Maximum days to hold a position before forced exit. Default: 5.
    top_n : int
        Number of top signals to request from scanner. Default: 10.
    """

    def __init__(
        self,
        hold_days: int = DEFAULT_HOLD_DAYS,
        top_n: int = DEFAULT_TOP_N,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._hold_days = hold_days
        self._top_n = top_n

    @property
    def strategy_id(self) -> str:
        return "s3_institutional"

    def warmup_days(self) -> int:
        return 20  # S9 scanner needs ~20 days of delivery/OI history

    def _scan_impl(self, d: date, store: MarketDataStore) -> list[Signal]:
        """Build composite institutional flow scores and emit signals.

        For each date:
        1. Run S9 daily scan to get composite signals
        2. Convert positive scores to long, negative to short
        3. Track open positions; exit after hold_days or score reversal
        """
        # Obtain composite signals from the S9 scanner
        try:
            from quantlaxmi.strategies.s9_momentum.scanner import run_daily_scan
            composites = run_daily_scan(d, store=store, top_n=self._top_n)
        except Exception as e:
            logger.debug("S9 scanner failed for %s: %s", d, e)
            return []

        if not composites:
            # No new signals — still check for timed exits
            return self._check_exits(d)

        # Compute max absolute score for conviction normalization
        max_score = max(abs(c.composite_score) for c in composites)
        if max_score <= 0:
            max_score = 1.0

        signals: list[Signal] = []

        # Build lookup of today's scores by symbol
        score_by_symbol: dict[str, float] = {
            c.symbol: c.composite_score for c in composites
        }

        # Check existing positions for exits
        exit_signals = self._check_exits(d, score_by_symbol)
        signals.extend(exit_signals)
        exited_symbols = {s.symbol for s in exit_signals}

        # New entries from today's scan
        for comp in composites:
            symbol = comp.symbol
            score = comp.composite_score

            # Skip if we just exited this symbol
            if symbol in exited_symbols:
                continue

            # Skip if already in a position on this symbol
            pos_key = f"pos_{symbol}"
            pos = self.get_state(pos_key)
            if pos is not None:
                continue

            if score == 0:
                continue

            direction = "long" if score > 0 else "short"
            conviction = min(1.0, abs(score) / max_score)

            # Record position entry
            self.set_state(pos_key, {
                "entry_date": d.isoformat(),
                "direction": direction,
                "entry_score": round(score, 4),
            })

            metadata = {
                "composite_score": round(score, 4),
                "delivery_score": round(comp.delivery_score, 4),
                "oi_score": round(comp.oi_score, 4),
                "fii_score": round(comp.fii_score, 4),
            }

            signals.append(Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction=direction,
                conviction=conviction,
                instrument_type="FUT",
                ttl_bars=self._hold_days,
                metadata=metadata,
            ))

        return signals

    def _check_exits(
        self,
        d: date,
        score_by_symbol: dict[str, float] | None = None,
    ) -> list[Signal]:
        """Check all open positions for exit conditions.

        Exits when:
        1. Position has been held for >= hold_days
        2. Score has reversed sign (direction flip)
        """
        if score_by_symbol is None:
            score_by_symbol = {}

        exits: list[Signal] = []

        # Iterate over all state keys that represent positions
        pos_keys = [k for k in self._state if k.startswith("pos_")]
        for pos_key in pos_keys:
            pos = self._state.get(pos_key)
            if pos is None:
                continue

            symbol = pos_key[4:]  # strip "pos_" prefix
            entry_date = date.fromisoformat(pos["entry_date"])
            days_held = (d - entry_date).days
            entry_direction = pos["direction"]

            exit_reason = None

            # Condition 1: hold period expired
            if days_held >= self._hold_days:
                exit_reason = "hold_expired"

            # Condition 2: score reversal
            if symbol in score_by_symbol:
                new_score = score_by_symbol[symbol]
                if entry_direction == "long" and new_score < 0:
                    exit_reason = "score_reversal"
                elif entry_direction == "short" and new_score > 0:
                    exit_reason = "score_reversal"

            if exit_reason is not None:
                self.set_state(pos_key, None)
                exits.append(Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    direction="flat",
                    conviction=0.0,
                    instrument_type="FUT",
                    ttl_bars=0,
                    metadata={
                        "exit_reason": exit_reason,
                        "days_held": days_held,
                    },
                ))

        return exits


def create_strategy() -> S3InstitutionalStrategy:
    """Factory for registry auto-discovery."""
    return S3InstitutionalStrategy()
