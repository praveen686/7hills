"""Trade Analytics Service — MFM, MDA, efficiency, exit quality.

For every closed trade, computes:
  - MFM (Maximum Favourable Move): best-case unrealised gain during hold
  - MDA (Maximum Drawdown Adverse): worst-case unrealised loss during hold
  - efficiency: pnl_pct / mfm  (how much of the available edge was captured)
  - exit_quality: 1 - |optimal_exit - actual_exit| / (mfm * entry_price)
  - duration_days

Direction-aware: long MFM uses max(high), short MFM uses min(low).
Options/spreads use underlying spot OHLC as proxy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict

from quantlaxmi.data.store import MarketDataStore

logger = logging.getLogger(__name__)

# Index name mapping: trading symbol → DuckDB "Index Name"
_INDEX_NAME_MAP = {
    "NIFTY": "Nifty 50",
    "NIFTY50": "Nifty 50",
    "BANKNIFTY": "Nifty Bank",
    "FINNIFTY": "Nifty Financial Services",
    "MIDCPNIFTY": "NIFTY MidSmall Financial Services",
}


@dataclass
class TradeAnalytics:
    """Analytics result for a single closed trade."""

    trade_id: str
    strategy_id: str
    symbol: str
    direction: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    mfm: float = 0.0
    mda: float = 0.0
    efficiency: float = 0.0
    exit_quality: float = 0.0
    duration_days: int = 0
    optimal_exit_price: float = 0.0
    worst_price: float = 0.0
    mfm_source: str = ""  # "index_ohlc" | "futures_ohlc" | "spot_proxy"
    price_path_available: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        # Sanitize NaN/Inf for JSON safety
        for k, v in d.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                d[k] = None
        return d


class TradeAnalyticsService:
    """Compute MFM/MDA/efficiency for closed trades using DuckDB price data."""

    def __init__(self, store: MarketDataStore):
        self._store = store

    def analyze_trade(self, trade) -> TradeAnalytics:
        """Analyze a single ClosedTrade and return TradeAnalytics."""
        from datetime import date as date_type

        # Base analytics
        entry_date = trade.entry_date
        exit_date = trade.exit_date
        entry_price = trade.entry_price
        direction = trade.direction
        symbol = trade.symbol
        instrument_type = getattr(trade, "instrument_type", "FUT")

        # Duration
        try:
            d_entry = date_type.fromisoformat(entry_date)
            d_exit = date_type.fromisoformat(exit_date)
            duration_days = (d_exit - d_entry).days
        except (ValueError, TypeError):
            duration_days = 0

        result = TradeAnalytics(
            trade_id=getattr(trade, "trade_id", "") or "",
            strategy_id=trade.strategy_id,
            symbol=symbol,
            direction=direction,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=trade.exit_price,
            pnl_pct=trade.pnl_pct,
            duration_days=duration_days,
        )

        # Skip if entry_price is 0 or negative
        if entry_price <= 0:
            return result

        # Get price path
        highs, lows, closes, mfm_source = self._get_price_path(
            symbol, entry_date, exit_date, instrument_type,
        )

        if not highs:
            return result

        result.price_path_available = True
        result.mfm_source = mfm_source

        # Compute MFM/MDA (direction-aware)
        mfm, mda, optimal_exit, worst_price = self._compute_mfm_mda(
            direction, entry_price, highs, lows,
        )
        result.mfm = mfm
        result.mda = mda
        result.optimal_exit_price = optimal_exit
        result.worst_price = worst_price

        # Efficiency: pnl_pct / mfm
        if mfm > 0:
            result.efficiency = trade.pnl_pct / mfm
        else:
            result.efficiency = 0.0

        # Exit quality: 1 - |optimal_exit - actual_exit| / (mfm * entry_price)
        if mfm > 0:
            denom = mfm * entry_price
            if denom > 0:
                raw = 1.0 - abs(optimal_exit - trade.exit_price) / denom
                result.exit_quality = max(0.0, min(1.0, raw))
        else:
            result.exit_quality = 0.0

        return result

    def analyze_all(self, trades: list) -> list[TradeAnalytics]:
        """Analyze a list of ClosedTrades."""
        return [self.analyze_trade(t) for t in trades]

    def summary_by_strategy(self, analytics: list[TradeAnalytics]) -> dict[str, dict]:
        """Aggregate analytics by strategy_id."""
        buckets: dict[str, list[TradeAnalytics]] = {}
        for a in analytics:
            buckets.setdefault(a.strategy_id, []).append(a)

        result = {}
        for sid, items in buckets.items():
            n = len(items)
            with_path = [a for a in items if a.price_path_available]
            result[sid] = {
                "n_trades": n,
                "n_with_price_data": len(with_path),
                "avg_mfm": _safe_mean([a.mfm for a in with_path]),
                "avg_mda": _safe_mean([a.mda for a in with_path]),
                "avg_efficiency": _safe_mean([a.efficiency for a in with_path]),
                "avg_exit_quality": _safe_mean([a.exit_quality for a in with_path]),
                "avg_duration_days": _safe_mean([a.duration_days for a in items]),
                "avg_pnl_pct": _safe_mean([a.pnl_pct for a in items]),
            }
        return result

    # ------------------------------------------------------------------
    # Price path retrieval
    # ------------------------------------------------------------------

    def _get_price_path(
        self,
        symbol: str,
        start: str,
        end: str,
        instrument_type: str,
    ) -> tuple[list[float], list[float], list[float], str]:
        """Fetch OHLC price path for a trade's holding period.

        Returns (highs, lows, closes, mfm_source).
        For options/spreads, falls back to spot proxy.
        """
        # Options/spreads → use underlying spot as proxy
        if instrument_type in ("CE", "PE", "SPREAD"):
            return self._get_index_ohlc(symbol, start, end, source="spot_proxy")

        # Try futures 1-min aggregated daily first
        highs, lows, closes = self._query_futures_ohlc(symbol, start, end)
        if highs:
            return highs, lows, closes, "futures_ohlc"

        # Fallback to index daily OHLC
        h, l, c, src = self._get_index_ohlc(symbol, start, end, source="index_ohlc")
        return h, l, c, src

    def _query_futures_ohlc(
        self, symbol: str, start: str, end: str,
    ) -> tuple[list[float], list[float], list[float]]:
        """Query nfo_1min for FUT OHLC aggregated per day."""
        try:
            df = self._store.sql(
                "SELECT date, MAX(high) as high, MIN(low) as low, "
                "ARG_MAX(close, close) as close "
                "FROM nfo_1min "
                "WHERE name = $1 AND instrument_type = 'FUT' "
                "AND date >= $2 AND date <= $3 "
                "GROUP BY date ORDER BY date",
                [symbol, start, end],
            )
            if df.empty:
                return [], [], []
            return (
                df["high"].tolist(),
                df["low"].tolist(),
                df["close"].tolist(),
            )
        except Exception as e:
            logger.debug("Futures OHLC query failed for %s: %s", symbol, e)
            return [], [], []

    def _get_index_ohlc(
        self, symbol: str, start: str, end: str, source: str = "index_ohlc",
    ) -> tuple[list[float], list[float], list[float], str]:
        """Query nse_index_close for daily OHLC."""
        index_name = _INDEX_NAME_MAP.get(symbol.upper(), symbol)
        try:
            df = self._store.sql(
                'SELECT date, "High Index Value" as high, '
                '"Low Index Value" as low, '
                '"Closing Index Value" as close '
                "FROM nse_index_close "
                'WHERE "Index Name" = $1 AND date >= $2 AND date <= $3 '
                "ORDER BY date",
                [index_name, start, end],
            )
            if df.empty:
                return [], [], [], source
            return (
                [float(x) for x in df["high"]],
                [float(x) for x in df["low"]],
                [float(x) for x in df["close"]],
                source,
            )
        except Exception as e:
            logger.debug("Index OHLC query failed for %s: %s", symbol, e)
            return [], [], [], source

    # ------------------------------------------------------------------
    # MFM/MDA computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_mfm_mda(
        direction: str,
        entry_price: float,
        highs: list[float],
        lows: list[float],
    ) -> tuple[float, float, float, float]:
        """Compute direction-aware MFM, MDA, optimal exit, worst price.

        Returns (mfm, mda, optimal_exit_price, worst_price).
        """
        max_high = max(highs)
        min_low = min(lows)

        if direction == "long":
            mfm = max(0.0, (max_high - entry_price) / entry_price)
            mda = max(0.0, (entry_price - min_low) / entry_price)
            optimal_exit = max_high
            worst_price = min_low
        else:  # short
            mfm = max(0.0, (entry_price - min_low) / entry_price)
            mda = max(0.0, (max_high - entry_price) / entry_price)
            optimal_exit = min_low
            worst_price = max_high

        return mfm, mda, optimal_exit, worst_price


def _safe_mean(values: list) -> float:
    """Mean that returns 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)
