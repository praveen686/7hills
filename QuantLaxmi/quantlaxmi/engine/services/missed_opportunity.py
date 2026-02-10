"""Missed Opportunity Service — blocked signals + hypothetical P&L.

For each risk-gate-blocked signal, computes:
  - What the hypothetical P&L would have been
  - Entry price = spot on signal date
  - Exit price = close at signal_date + ttl_bars (strict bar-close rule)
  - Hypothetical MFM over the holding window
  - The gate that blocked it and why

No optimization — exit is deterministic at close of bar N.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import date as date_type, timedelta
from pathlib import Path

from quantlaxmi.core.events.types import EventType
from quantlaxmi.engine.replay.reader import WalReader

logger = logging.getLogger(__name__)

# Strategies that operate on intraday (1-min) bars
_INTRADAY_STRATEGIES = frozenset({"s5_hawkes"})

# Index name mapping for spot queries
_INDEX_NAME_MAP = {
    "NIFTY": "Nifty 50",
    "NIFTY50": "Nifty 50",
    "BANKNIFTY": "Nifty Bank",
    "FINNIFTY": "Nifty Financial Services",
    "MIDCPNIFTY": "NIFTY MidSmall Financial Services",
}


@dataclass
class MissedOpportunity:
    """A signal that was blocked by a risk gate."""

    signal_seq: int
    ts: str
    strategy_id: str
    symbol: str
    direction: str
    conviction: float
    instrument_type: str
    ttl_bars: int
    regime: str
    block_reason: str
    gate: str
    risk_metrics: dict
    entry_price: float = 0.0
    hypothetical_exit_price: float = 0.0
    hypothetical_pnl_pct: float = 0.0
    hypothetical_mfm: float = 0.0
    price_data_available: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class MissedOpportunityService:
    """Detect blocked signals and compute hypothetical P&L."""

    def __init__(self, base_dir: Path | str, store=None):
        self._base_dir = Path(base_dir)
        self._store = store  # MarketDataStore, optional

    def _reader(self) -> WalReader:
        return WalReader(base_dir=self._base_dir)

    # ------------------------------------------------------------------
    # Core: get blocked signals from WAL
    # ------------------------------------------------------------------

    def get_blocked_signals(self, day: str) -> list[dict]:
        """Return raw blocked signal+gate pairs for a day.

        For each SIGNAL event, find the subsequent GATE_DECISION with
        same strategy_id + symbol. If approved=False, it's blocked.
        """
        reader = self._reader()
        events = reader.read_date(day)
        if not events:
            return []

        blocked = []
        signals = []
        gates = []

        for e in events:
            if e.event_type == EventType.SIGNAL.value:
                signals.append(e)
            elif e.event_type == EventType.GATE_DECISION.value:
                gates.append(e)

        # Match each signal to its gate decision
        for sig in signals:
            # Find gate decisions after this signal for same strategy+symbol
            matched_gate = None
            for g in gates:
                if (g.seq > sig.seq
                        and g.strategy_id == sig.strategy_id
                        and g.symbol == sig.symbol):
                    matched_gate = g
                    break

            if matched_gate and not matched_gate.payload.get("approved", True):
                sp = sig.payload
                gp = matched_gate.payload
                blocked.append({
                    "signal_seq": sig.seq,
                    "ts": sig.ts,
                    "strategy_id": sig.strategy_id,
                    "symbol": sig.symbol,
                    "direction": sp.get("direction", ""),
                    "conviction": sp.get("conviction", 0.0),
                    "instrument_type": sp.get("instrument_type", "FUT"),
                    "ttl_bars": sp.get("ttl_bars", 5),
                    "regime": sp.get("regime", ""),
                    "block_reason": gp.get("reason", ""),
                    "gate": gp.get("gate", ""),
                    "risk_metrics": {
                        "vpin": gp.get("vpin", 0.0),
                        "portfolio_dd": gp.get("portfolio_dd", 0.0),
                        "strategy_dd": gp.get("strategy_dd", 0.0),
                        "total_exposure": gp.get("total_exposure", 0.0),
                    },
                })

        return blocked

    # ------------------------------------------------------------------
    # Analyze with hypothetical P&L
    # ------------------------------------------------------------------

    def analyze_missed(self, day: str) -> list[MissedOpportunity]:
        """Get blocked signals for a day and enrich with hypothetical P&L."""
        blocked = self.get_blocked_signals(day)
        results = []
        for b in blocked:
            opp = MissedOpportunity(**b)
            if self._store is not None:
                self._enrich_hypothetical(opp, day)
            results.append(opp)
        return results

    def analyze_range(self, start_date: str, end_date: str) -> list[MissedOpportunity]:
        """Analyze blocked signals across a date range."""
        d = date_type.fromisoformat(start_date)
        end_d = date_type.fromisoformat(end_date)
        results = []
        while d <= end_d:
            day_str = d.isoformat()
            results.extend(self.analyze_missed(day_str))
            d += timedelta(days=1)
        return results

    def summary_by_strategy(self, opportunities: list[MissedOpportunity]) -> dict[str, dict]:
        """Aggregate missed opportunities by strategy."""
        buckets: dict[str, list[MissedOpportunity]] = {}
        for opp in opportunities:
            buckets.setdefault(opp.strategy_id, []).append(opp)

        result = {}
        for sid, items in buckets.items():
            with_price = [o for o in items if o.price_data_available]
            profitable = [o for o in with_price if o.hypothetical_pnl_pct > 0]
            result[sid] = {
                "n_blocked": len(items),
                "n_with_price_data": len(with_price),
                "n_profitable": len(profitable),
                "avg_hypothetical_pnl_pct": _safe_mean(
                    [o.hypothetical_pnl_pct for o in with_price]
                ),
                "total_hypothetical_pnl_pct": sum(
                    o.hypothetical_pnl_pct for o in with_price
                ),
                "avg_conviction": _safe_mean([o.conviction for o in items]),
                "top_block_reasons": _top_counts(
                    [o.block_reason for o in items], n=5,
                ),
            }
        return result

    # ------------------------------------------------------------------
    # Hypothetical P&L enrichment
    # ------------------------------------------------------------------

    def _enrich_hypothetical(self, opp: MissedOpportunity, signal_day: str) -> None:
        """Add hypothetical P&L to a MissedOpportunity.

        Entry price = spot close on signal date.
        Exit price = close at signal_date + ttl_bars trading days.
        """
        symbol = opp.symbol
        ttl_bars = opp.ttl_bars if opp.ttl_bars > 0 else 5

        # For intraday strategies, ttl_bars are 1-min bars → use same day
        if opp.strategy_id in _INTRADAY_STRATEGIES:
            exit_day = signal_day
        else:
            # Find the Nth trading day by scanning calendar days for data
            exit_day = self._find_nth_trading_day(symbol, signal_day, ttl_bars)
            if exit_day is None:
                return

        # Get entry price (close on signal day)
        entry_price = self._get_spot_close(symbol, signal_day)
        if entry_price is None or entry_price <= 0:
            return

        opp.entry_price = entry_price

        # Get exit price (close on exit day)
        exit_price = self._get_spot_close(symbol, exit_day)

        if exit_price is None:
            return

        opp.hypothetical_exit_price = exit_price
        opp.price_data_available = True

        # Hypothetical P&L
        if opp.direction == "long":
            opp.hypothetical_pnl_pct = (exit_price - entry_price) / entry_price
        else:
            opp.hypothetical_pnl_pct = (entry_price - exit_price) / entry_price

        # Hypothetical MFM over the window
        highs, lows = self._get_hl_range(symbol, signal_day, exit_day)
        if highs and lows:
            max_high = max(highs)
            min_low = min(lows)
            if opp.direction == "long":
                opp.hypothetical_mfm = max(0.0, (max_high - entry_price) / entry_price)
            else:
                opp.hypothetical_mfm = max(0.0, (entry_price - min_low) / entry_price)

    def _find_nth_trading_day(
        self, symbol: str, signal_day: str, n: int,
    ) -> str | None:
        """Find the Nth trading day after signal_day using actual market data.

        Scans calendar days looking for dates with closing prices.
        Falls back to calendar-day offset if no store is available.
        """
        if self._store is None or n <= 0:
            # Fallback: calendar-day approximation
            try:
                d = date_type.fromisoformat(signal_day)
                return (d + timedelta(days=max(1, n))).isoformat()
            except (ValueError, TypeError):
                return None

        try:
            d = date_type.fromisoformat(signal_day)
        except (ValueError, TypeError):
            return None

        trading_days_found = 0
        # Scan up to n*3 calendar days to find n trading days
        for offset in range(1, n * 3 + 5):
            candidate = (d + timedelta(days=offset)).isoformat()
            close = self._get_spot_close(symbol, candidate)
            if close is not None:
                trading_days_found += 1
                if trading_days_found >= n:
                    return candidate

        # Fallback: couldn't find enough trading days
        return (d + timedelta(days=n)).isoformat()

    def _get_spot_close(self, symbol: str, day: str) -> float | None:
        """Get closing price for a symbol on a date."""
        if self._store is None:
            return None

        index_name = _INDEX_NAME_MAP.get(symbol.upper(), symbol)
        try:
            df = self._store.sql(
                'SELECT "Closing Index Value" as close '
                "FROM nse_index_close "
                'WHERE "Index Name" = $1 AND date = $2 '
                "LIMIT 1",
                [index_name, day],
            )
            if not df.empty:
                return float(df["close"].iloc[0])
        except Exception as e:
            logger.debug("Index close lookup failed for %s on %s: %s", symbol, day, e)

        # Fallback: try nfo_1min FUT last close
        try:
            df = self._store.sql(
                "SELECT close FROM nfo_1min "
                "WHERE name = $1 AND instrument_type = 'FUT' AND date = $2 "
                "ORDER BY date DESC LIMIT 1",
                [symbol, day],
            )
            if not df.empty:
                return float(df["close"].iloc[0])
        except Exception as e:
            logger.debug("FUT close lookup failed for %s on %s: %s", symbol, day, e)

        return None

    def _get_hl_range(
        self, symbol: str, start: str, end: str,
    ) -> tuple[list[float], list[float]]:
        """Get high/low arrays over a date range."""
        if self._store is None:
            return [], []

        index_name = _INDEX_NAME_MAP.get(symbol.upper(), symbol)
        try:
            df = self._store.sql(
                'SELECT "High Index Value" as high, "Low Index Value" as low '
                "FROM nse_index_close "
                'WHERE "Index Name" = $1 AND date >= $2 AND date <= $3 '
                "ORDER BY date",
                [index_name, start, end],
            )
            if not df.empty:
                return (
                    [float(x) for x in df["high"]],
                    [float(x) for x in df["low"]],
                )
        except Exception as e:
            logger.debug("HL range lookup failed for %s (%s to %s): %s", symbol, start, end, e)
        return [], []


def _safe_mean(values: list) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _top_counts(items: list[str], n: int = 5) -> list[dict]:
    """Top N most common items with counts."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    return [{"reason": k, "count": v} for k, v in sorted_items[:n]]
