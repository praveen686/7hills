"""Signal computation for the India institutional footprint scanner.

Three signals:
  1. Delivery Spike — high delivery % relative to 20-day average
  2. OI Buildup — 4-quadrant classification from stock futures
  3. FII Flow Regime — 3-day cumulative FII net buy/sell

Each returns frozen dataclasses. The composite score is a weighted sum.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeliverySignal:
    """Delivery spike detection for a single stock on a single day."""

    symbol: str
    delivery_pct: float         # today's delivery %
    avg_delivery_pct: float     # 20-day average delivery %
    delivery_ratio: float       # today / avg
    volume: float               # today's traded qty
    avg_volume: float           # 20-day average volume
    volume_ratio: float         # today / avg
    close: float
    open_: float
    score: float                # +1 accumulation, -1 distribution, 0 neutral


@dataclass(frozen=True)
class OISignal:
    """Open interest buildup classification for a single stock."""

    symbol: str
    oi: float                   # current OI
    oi_change: float            # change in OI
    price_change_pct: float     # % change in futures price
    classification: str         # LONG_BUILDUP, SHORT_BUILDUP, SHORT_COVERING, LONG_UNWINDING
    score: float                # +1, -1, +0.5, -0.5


@dataclass(frozen=True)
class FIIFlowSignal:
    """FII flow regime signal (market-wide, not per-stock)."""

    cumulative_net_inr_cr: float   # 3-day cumulative FII net (in crores)
    regime: str                     # "bullish" or "bearish"
    score: float                    # +0.5 or -0.5


@dataclass(frozen=True)
class CompositeSignal:
    """Weighted composite signal for a single stock."""

    symbol: str
    delivery_score: float
    oi_score: float
    fii_score: float
    composite_score: float
    delivery_signal: DeliverySignal | None
    oi_signal: OISignal | None


# ---------------------------------------------------------------------------
# Signal 1: Delivery Spike
# ---------------------------------------------------------------------------

# Thresholds
DELIVERY_RATIO_THRESHOLD = 1.5
VOLUME_RATIO_THRESHOLD = 1.5
LOOKBACK_DAYS = 20


def compute_delivery_signals(
    equity_history: dict[date, pd.DataFrame],
    delivery_history: dict[date, pd.DataFrame],
    target_date: date,
    symbols: set[str] | None = None,
) -> dict[str, DeliverySignal]:
    """Compute delivery spike signals for all stocks on target_date.

    equity_history: {date: equity_bhav_df} for at least 20+1 trading days
    delivery_history: {date: delivery_df} for the same period
    symbols: if provided, only compute for these symbols

    Returns {symbol: DeliverySignal}.
    """
    if target_date not in equity_history or target_date not in delivery_history:
        logger.warning("Missing data for target date %s", target_date)
        return {}

    today_eq = equity_history[target_date]
    today_del = delivery_history[target_date]

    # Build symbol-level delivery % and volume history
    sorted_dates = sorted(d for d in equity_history if d <= target_date)
    if len(sorted_dates) < 2:
        return {}

    # History dates (excluding target)
    hist_dates = sorted_dates[:-1][-LOOKBACK_DAYS:]

    # Build historical averages
    delivery_avgs: dict[str, list[float]] = {}
    volume_avgs: dict[str, list[float]] = {}

    for d in hist_dates:
        eq_df = equity_history.get(d)
        del_df = delivery_history.get(d)
        if eq_df is None or del_df is None:
            continue

        # Merge equity and delivery on symbol
        if "SYMBOL" in del_df.columns and "DELIVERY_PCT" in del_df.columns:
            for _, row in del_df.iterrows():
                sym = str(row.get("SYMBOL", "")).strip()
                if symbols and sym not in symbols:
                    continue
                dpct = _safe_float(row.get("DELIVERY_PCT", 0))
                if dpct > 0:
                    delivery_avgs.setdefault(sym, []).append(dpct)

        if "SYMBOL" in eq_df.columns and "TOTTRDQTY" in eq_df.columns:
            for _, row in eq_df.iterrows():
                sym = str(row.get("SYMBOL", "")).strip()
                if symbols and sym not in symbols:
                    continue
                vol = _safe_float(row.get("TOTTRDQTY", 0))
                if vol > 0:
                    volume_avgs.setdefault(sym, []).append(vol)

    # Today's data
    signals: dict[str, DeliverySignal] = {}

    # Merge today's equity + delivery
    if "SYMBOL" not in today_del.columns or "DELIVERY_PCT" not in today_del.columns:
        logger.warning("Delivery data missing expected columns")
        return {}

    today_merged = today_del.copy()
    if "SYMBOL" in today_eq.columns:
        eq_cols = ["SYMBOL"]
        for c in ["OPEN", "CLOSE", "TOTTRDQTY"]:
            if c in today_eq.columns:
                eq_cols.append(c)
        today_merged = today_del.merge(
            today_eq[eq_cols], on="SYMBOL", how="left", suffixes=("", "_eq"),
        )

    for _, row in today_merged.iterrows():
        sym = str(row.get("SYMBOL", "")).strip()
        if symbols and sym not in symbols:
            continue

        dpct = _safe_float(row.get("DELIVERY_PCT", 0))
        vol = _safe_float(row.get("TOTTRDQTY", 0))
        open_ = _safe_float(row.get("OPEN", 0))
        close = _safe_float(row.get("CLOSE", 0))

        # Use TRADED_QTY from delivery data if equity merge failed
        if vol == 0:
            vol = _safe_float(row.get("TRADED_QTY", 0))

        hist_del = delivery_avgs.get(sym, [])
        hist_vol = volume_avgs.get(sym, [])

        if not hist_del or not hist_vol:
            continue

        avg_del = sum(hist_del) / len(hist_del)
        avg_vol = sum(hist_vol) / len(hist_vol)

        if avg_del <= 0 or avg_vol <= 0:
            continue

        del_ratio = dpct / avg_del
        vol_ratio = vol / avg_vol

        # Score: both ratios must exceed threshold
        score = 0.0
        if del_ratio >= DELIVERY_RATIO_THRESHOLD and vol_ratio >= VOLUME_RATIO_THRESHOLD:
            if close > open_ and open_ > 0:
                score = 1.0   # Accumulation
            elif close < open_ and open_ > 0:
                score = -1.0  # Distribution

        signals[sym] = DeliverySignal(
            symbol=sym,
            delivery_pct=dpct,
            avg_delivery_pct=avg_del,
            delivery_ratio=del_ratio,
            volume=vol,
            avg_volume=avg_vol,
            volume_ratio=vol_ratio,
            close=close,
            open_=open_,
            score=score,
        )

    return signals


# ---------------------------------------------------------------------------
# Signal 2: OI Buildup
# ---------------------------------------------------------------------------


def compute_oi_signals(
    futures_oi_today: pd.DataFrame,
    futures_oi_prev: pd.DataFrame,
    equity_today: pd.DataFrame,
    equity_prev: pd.DataFrame,
    symbols: set[str] | None = None,
) -> dict[str, OISignal]:
    """Compute OI buildup signals from stock futures data.

    4-quadrant classification:
      OI↑ + Price↑ → LONG_BUILDUP (+1)
      OI↑ + Price↓ → SHORT_BUILDUP (-1)
      OI↓ + Price↑ → SHORT_COVERING (+0.5)
      OI↓ + Price↓ → LONG_UNWINDING (-0.5)
    """
    signals: dict[str, OISignal] = {}

    if futures_oi_today.empty or futures_oi_prev.empty:
        return signals

    # Build lookup: symbol → OI for previous day
    prev_oi = {}
    for _, row in futures_oi_prev.iterrows():
        sym = str(row.get("SYMBOL", "")).strip()
        prev_oi[sym] = _safe_float(row.get("OPEN_INT", 0))

    # Build lookup: symbol → close for previous day
    prev_close = {}
    if not equity_prev.empty and "SYMBOL" in equity_prev.columns:
        for _, row in equity_prev.iterrows():
            sym = str(row.get("SYMBOL", "")).strip()
            prev_close[sym] = _safe_float(row.get("CLOSE", 0))

    # Today's close
    today_close = {}
    if not equity_today.empty and "SYMBOL" in equity_today.columns:
        for _, row in equity_today.iterrows():
            sym = str(row.get("SYMBOL", "")).strip()
            today_close[sym] = _safe_float(row.get("CLOSE", 0))

    for _, row in futures_oi_today.iterrows():
        sym = str(row.get("SYMBOL", "")).strip()
        if symbols and sym not in symbols:
            continue

        oi_today = _safe_float(row.get("OPEN_INT", 0))
        oi_prev = prev_oi.get(sym, 0)

        if oi_prev <= 0:
            continue

        oi_change = oi_today - oi_prev

        close_today = today_close.get(sym, 0)
        close_prev = prev_close.get(sym, 0)

        if close_prev <= 0:
            # Try using futures close if equity close unavailable
            close_today = _safe_float(row.get("CLOSE", 0))
            continue

        price_change_pct = (close_today - close_prev) / close_prev * 100

        # 4-quadrant classification
        if oi_change > 0 and price_change_pct > 0:
            classification = "LONG_BUILDUP"
            score = 1.0
        elif oi_change > 0 and price_change_pct <= 0:
            classification = "SHORT_BUILDUP"
            score = -1.0
        elif oi_change <= 0 and price_change_pct > 0:
            classification = "SHORT_COVERING"
            score = 0.5
        else:
            classification = "LONG_UNWINDING"
            score = -0.5

        signals[sym] = OISignal(
            symbol=sym,
            oi=oi_today,
            oi_change=oi_change,
            price_change_pct=price_change_pct,
            classification=classification,
            score=score,
        )

    return signals


# ---------------------------------------------------------------------------
# Signal 3: FII Flow Regime
# ---------------------------------------------------------------------------


def compute_fii_flow_signal(
    fii_dii_history: dict[date, pd.DataFrame],
    target_date: date,
    lookback: int = 3,
) -> FIIFlowSignal | None:
    """Compute FII flow regime from cumulative 3-day FII net.

    FII/DII data has columns like: category, buyValue, sellValue, netValue
    where category includes "FII/FPI" and "DII".
    """
    sorted_dates = sorted(d for d in fii_dii_history if d <= target_date)
    recent = sorted_dates[-lookback:]

    if not recent:
        return None

    cumulative_net = 0.0
    for d in recent:
        df = fii_dii_history[d]
        if df.empty:
            continue

        # Look for FII/FPI total row (prefer TOTAL_FUTURES, fallback to first FII row)
        best_net = None
        for _, row in df.iterrows():
            cat = str(row.get("category", "")).strip().upper()
            sub = str(row.get("sub_category", "")).strip().upper()
            if "FII" in cat or "FPI" in cat:
                net = _safe_float(row.get("netValue", 0))
                if sub == "TOTAL_FUTURES":
                    best_net = net
                    break
                if best_net is None:
                    best_net = net
        if best_net is not None:
            cumulative_net += best_net

    regime = "bullish" if cumulative_net > 0 else "bearish"
    score = 0.5 if cumulative_net > 0 else -0.5

    return FIIFlowSignal(
        cumulative_net_inr_cr=cumulative_net,
        regime=regime,
        score=score,
    )


# ---------------------------------------------------------------------------
# Composite Score
# ---------------------------------------------------------------------------

# Default weights
W_DELIVERY = 1.0
W_OI = 0.8
W_FII = 0.5


def compute_composite_scores(
    delivery_signals: dict[str, DeliverySignal],
    oi_signals: dict[str, OISignal],
    fii_signal: FIIFlowSignal | None,
    w_delivery: float = W_DELIVERY,
    w_oi: float = W_OI,
    w_fii: float = W_FII,
) -> list[CompositeSignal]:
    """Compute weighted composite score for all symbols.

    FII flow is a market-wide overlay:
      - Adds to longs if bullish, adds to shorts if bearish.

    Returns list sorted by |composite_score| descending.
    """
    all_symbols = set(delivery_signals.keys()) | set(oi_signals.keys())
    fii_score = fii_signal.score if fii_signal else 0.0

    composites = []
    for sym in all_symbols:
        del_sig = delivery_signals.get(sym)
        oi_sig = oi_signals.get(sym)

        d_score = del_sig.score if del_sig else 0.0
        o_score = oi_sig.score if oi_sig else 0.0

        # FII overlay: only adds to the direction of the stock signal
        base_score = w_delivery * d_score + w_oi * o_score
        if base_score > 0:
            fii_contribution = w_fii * abs(fii_score) * (1 if fii_score > 0 else -1)
        elif base_score < 0:
            fii_contribution = w_fii * abs(fii_score) * (-1 if fii_score > 0 else 1)
        else:
            fii_contribution = 0.0

        composite = base_score + fii_contribution

        composites.append(CompositeSignal(
            symbol=sym,
            delivery_score=d_score,
            oi_score=o_score,
            fii_score=fii_score,
            composite_score=composite,
            delivery_signal=del_sig,
            oi_signal=oi_sig,
        ))

    # Sort by absolute composite score
    composites.sort(key=lambda x: abs(x.composite_score), reverse=True)
    return composites


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert to float, returning default on failure."""
    if pd.isna(val):
        return default
    try:
        s = str(val).replace(",", "").strip()
        if s == "" or s == "-":
            return default
        return float(s)
    except (ValueError, TypeError):
        return default
