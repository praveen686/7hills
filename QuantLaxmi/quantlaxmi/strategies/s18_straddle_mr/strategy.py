"""Intraday ATM Straddle Mean-Reversion Strategy for NIFTY / BANKNIFTY.

Core thesis:
    ATM straddle prices exhibit intraday mean-reversion driven by market-maker
    inventory management.  When the *normalized* straddle (straddle/spot) becomes
    abnormally expensive or cheap relative to its rolling distribution, trade the
    reversion.  Normalizing by spot removes directional drift and isolates the
    pure volatility component.

Signal chain:
    1. At each 1-min bar (10:00-14:30), find the ATM strike closest to the
       front-month futures close and construct synthetic straddle price.
    2. Normalize: straddle_pct = (ATM_CE + ATM_PE) / FUT_close (implied vol proxy).
    3. Compute 120-bar rolling z-score of straddle_pct.
    4. SELL straddle when z > 2.5 (straddle expensive, expect mean reversion).
    5. BUY straddle when z < -2.5 (straddle cheap, expect mean reversion).
    6. Exit: z crosses 0 (mean reverted), |z| > 4.0 (stop), 60-min hold, 15:00 cutoff.
    7. Maximum 1 round-trip per day (only the best signal).

Filters:
    - Only trade 10:00 to 14:30 (avoid open/close volatility clusters).
    - Skip days where intraday range > 2% (trending day).
    - Skip days where VIX change > 5% from previous close (event day).

Cost model:
    3 index points per leg x 2 legs x 2 (entry + exit) = 12 pts round-trip.

Data:
    Uses ``MarketDataStore`` (DuckDB + hive-partitioned Parquet).
    Table: ``nfo_1min`` -- columns: timestamp, date, open, high, low, close,
    volume, oi, symbol, name, expiry, strike, instrument_type.

Author : AlphaForge
Created: 2026-02-08
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# Trading window: avoid open/close vol clusters
ENTRY_OPEN = time(10, 0)
ENTRY_CLOSE = time(14, 30)

# Force-close all positions at this time
FORCE_CLOSE_TIME = time(15, 0)

# Z-score thresholds
Z_ENTRY_SELL = 2.5       # sell straddle when z > this (more selective)
Z_ENTRY_BUY = -2.5       # buy straddle when z < this
Z_EXIT_MEAN = 0.0        # exit when z reverts to 0
Z_STOP = 4.0             # stop: extreme trend, not mean-reverting

# Rolling window for z-score
ZSCORE_WINDOW = 120      # 2-hour rolling window for stability

# Position limits
MAX_HOLD_BARS = 60       # max 60 minutes
MAX_ROUNDTRIPS_PER_DAY = 1  # only the best signal per day

# Cost per leg in index points (NIFTY)
COST_PER_LEG: dict[str, float] = {
    "NIFTY": 3.0,
    "BANKNIFTY": 5.0,
    "FINNIFTY": 3.0,
    "MIDCPNIFTY": 3.0,
}

# Filter thresholds
MAX_INTRADAY_RANGE_PCT = 0.02   # 2%
MAX_VIX_CHANGE_PCT = 0.05       # 5%

# Minimum straddle volume filter -- bars with zero volume in either leg
# are unreliable price quotes
MIN_LEG_VOLUME = 100


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StraddleTrade:
    """Single intraday straddle trade record."""
    trade_date: date
    entry_time: datetime
    exit_time: datetime | None = None
    direction: int = 0               # +1 = long straddle, -1 = short straddle
    entry_straddle_price: float = 0.0
    exit_straddle_price: float = 0.0
    entry_straddle_pct: float = 0.0
    exit_straddle_pct: float = 0.0
    entry_ce_price: float = 0.0
    exit_ce_price: float = 0.0
    entry_pe_price: float = 0.0
    exit_pe_price: float = 0.0
    entry_z: float = 0.0
    exit_z: float = 0.0
    atm_strike: float = 0.0
    expiry: str = ""
    pnl_points: float = 0.0          # P&L in index points (after costs)
    pnl_ce_points: float = 0.0       # CE leg contribution
    pnl_pe_points: float = 0.0       # PE leg contribution
    exit_reason: str = ""
    bars_held: int = 0
    cost_points: float = 0.0


@dataclass
class DayResult:
    """Per-day backtest output."""
    trade_date: date
    trades: list[StraddleTrade] = field(default_factory=list)
    daily_pnl: float = 0.0
    n_roundtrips: int = 0
    skipped: bool = False
    skip_reason: str = ""
    fut_close_start: float = 0.0
    fut_close_end: float = 0.0
    intraday_range_pct: float = 0.0
    n_signals: int = 0               # z-score crossings (before filtering)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_day_data(
    store: Any,
    symbol: str,
    d_str: str,
) -> pd.DataFrame | None:
    """Load ALL 1-min data for symbol on a date in a single query.

    Returns a DataFrame with columns:
        timestamp, close, volume, oi, strike, instrument_type, expiry

    This is much more efficient than per-strike queries.
    """
    try:
        df = store.sql(
            "SELECT timestamp, close, volume, oi, strike, instrument_type, expiry "
            "FROM nfo_1min "
            "WHERE name = ? AND date = ? "
            "AND instrument_type IN ('FUT', 'CE', 'PE') "
            "ORDER BY timestamp",
            [symbol, d_str],
        )
        return df if not df.empty else None
    except Exception as e:
        logger.debug("Error loading data for %s: %s", d_str, e)
        return None


# ---------------------------------------------------------------------------
# VIX helpers
# ---------------------------------------------------------------------------

def _load_vix_series(store: Any) -> dict[str, float]:
    """Load India VIX closing values keyed by ISO date string."""
    try:
        df = store.sql(
            'SELECT "Closing Index Value", date '
            'FROM nse_index_close '
            "WHERE \"Index Name\" = 'India VIX' "
            'ORDER BY date'
        )
        if df.empty:
            return {}
        vix_map: dict[str, float] = {}
        for _, row in df.iterrows():
            try:
                vix_map[str(row["date"])] = float(row["Closing Index Value"])
            except (ValueError, TypeError):
                continue
        return vix_map
    except Exception as e:
        logger.debug("VIX data unavailable: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Rolling z-score computation (vectorized)
# ---------------------------------------------------------------------------

def _rolling_zscore(
    series: np.ndarray,
    window: int = ZSCORE_WINDOW,
) -> np.ndarray:
    """Compute causal rolling z-score using pandas for speed.

    z(t) = (x(t) - mean(x[t-window+1:t+1])) / std(x[t-window+1:t+1], ddof=1)

    Returns NaN for the warm-up period (first `window-1` bars).
    """
    s = pd.Series(series)
    rolling = s.rolling(window=window, min_periods=window)
    mu = rolling.mean()
    sigma = rolling.std(ddof=1)
    z = (s - mu) / sigma
    # Replace inf/nan from zero-sigma periods
    z = z.replace([np.inf, -np.inf], 0.0)
    return z.values


# ---------------------------------------------------------------------------
# Single-day backtest
# ---------------------------------------------------------------------------

def _run_single_day(
    all_data: pd.DataFrame,
    symbol: str,
    d: date,
    cost_rt: float,
    prev_vix: float | None,
    vix_today: float | None,
) -> DayResult:
    """Run the straddle mean-reversion strategy for a single trading day.

    Parameters
    ----------
    all_data : pd.DataFrame
        All 1-min bars for this symbol on this date (FUT + CE + PE).
    symbol : str
        Underlying name (e.g. "NIFTY").
    d : date
        Trading date.
    cost_rt : float
        Round-trip cost in index points (for 2 legs, entry + exit).
    prev_vix, vix_today : float or None
        Previous and current VIX close (for event-day filter).

    Returns
    -------
    DayResult with trades and daily P&L.
    """
    result = DayResult(trade_date=d)

    # ---- Extract front-month FUT bars ----
    fut_data = all_data[all_data["instrument_type"] == "FUT"].copy()
    if fut_data.empty:
        result.skipped = True
        result.skip_reason = "no_fut_data"
        return result

    # Pick front-month (nearest expiry)
    front_exp = fut_data["expiry"].min()
    fut_bars = fut_data[fut_data["expiry"] == front_exp].sort_values("timestamp").reset_index(drop=True)

    if len(fut_bars) < ZSCORE_WINDOW + 10:
        result.skipped = True
        result.skip_reason = "insufficient_fut_bars"
        return result

    fut_timestamps = pd.to_datetime(fut_bars["timestamp"].values)
    fut_close = fut_bars["close"].values.astype(np.float64)

    result.fut_close_start = float(fut_close[0])
    result.fut_close_end = float(fut_close[-1])

    # ---- Filter: intraday range > 2% ----
    fut_high = float(np.max(fut_close))
    fut_low = float(np.min(fut_close))
    fut_mid = (fut_high + fut_low) / 2.0
    intraday_range = (fut_high - fut_low) / fut_mid if fut_mid > 0 else 0
    result.intraday_range_pct = intraday_range

    if intraday_range > MAX_INTRADAY_RANGE_PCT:
        result.skipped = True
        result.skip_reason = f"trending_day_{intraday_range:.3f}"
        return result

    # ---- Filter: VIX change > 5% ----
    if prev_vix is not None and vix_today is not None and prev_vix > 0:
        vix_change = abs(vix_today - prev_vix) / prev_vix
        if vix_change > MAX_VIX_CHANGE_PCT:
            result.skipped = True
            result.skip_reason = f"vix_event_{vix_change:.3f}"
            return result

    # ---- Find nearest expiry for options ----
    opt_data = all_data[all_data["instrument_type"].isin(["CE", "PE"])].copy()
    if opt_data.empty:
        result.skipped = True
        result.skip_reason = "no_option_data"
        return result

    nearest_expiry = opt_data["expiry"].min()
    opt_nearest = opt_data[opt_data["expiry"] == nearest_expiry].copy()

    # ---- Build timestamp-indexed FUT lookup for spot at each bar ----
    fut_lookup = dict(zip(fut_timestamps.values, fut_close))

    # ---- Find ATM strike at ~10:00 ----
    bar_10am_idx = None
    for idx, ts in enumerate(fut_timestamps):
        if ts.time() >= ENTRY_OPEN:
            bar_10am_idx = idx
            break

    if bar_10am_idx is None or bar_10am_idx >= len(fut_close):
        result.skipped = True
        result.skip_reason = "no_10am_bar"
        return result

    spot_at_10am = fut_close[bar_10am_idx]

    # Get available strikes
    ce_strikes = sorted(opt_nearest[opt_nearest["instrument_type"] == "CE"]["strike"].unique())
    pe_strikes = sorted(opt_nearest[opt_nearest["instrument_type"] == "PE"]["strike"].unique())
    common_strikes = sorted(set(ce_strikes) & set(pe_strikes))

    if not common_strikes:
        result.skipped = True
        result.skip_reason = "no_common_strikes"
        return result

    common_strikes_arr = np.array(common_strikes, dtype=np.float64)
    atm_strike = float(common_strikes_arr[np.argmin(np.abs(common_strikes_arr - spot_at_10am))])

    # ---- Load ATM CE + PE bars and align ----
    ce_bars = opt_nearest[
        (opt_nearest["instrument_type"] == "CE") &
        (opt_nearest["strike"] == atm_strike)
    ][["timestamp", "close", "volume"]].copy()

    pe_bars = opt_nearest[
        (opt_nearest["instrument_type"] == "PE") &
        (opt_nearest["strike"] == atm_strike)
    ][["timestamp", "close", "volume"]].copy()

    if ce_bars.empty or pe_bars.empty:
        result.skipped = True
        result.skip_reason = "missing_atm_options"
        return result

    # Align on timestamp (inner join)
    ce_indexed = ce_bars.set_index("timestamp").rename(
        columns={"close": "ce_close", "volume": "ce_vol"}
    )
    pe_indexed = pe_bars.set_index("timestamp").rename(
        columns={"close": "pe_close", "volume": "pe_vol"}
    )
    combined = ce_indexed.join(pe_indexed, how="inner").sort_index()

    if len(combined) < ZSCORE_WINDOW + 10:
        result.skipped = True
        result.skip_reason = f"insufficient_aligned_bars_{len(combined)}"
        return result

    timestamps = combined.index.to_pydatetime()
    ce_close = combined["ce_close"].values.astype(np.float64)
    pe_close = combined["pe_close"].values.astype(np.float64)
    ce_vol = combined["ce_vol"].values.astype(np.float64)
    pe_vol = combined["pe_vol"].values.astype(np.float64)

    # ---- Construct spot-normalized straddle (implied vol proxy) ----
    straddle_raw = ce_close + pe_close

    # Get corresponding FUT close for each option timestamp
    spot_series = np.array([
        fut_lookup.get(ts, np.nan) for ts in combined.index.values
    ], dtype=np.float64)

    # Forward-fill any missing spot values
    for i in range(1, len(spot_series)):
        if np.isnan(spot_series[i]):
            spot_series[i] = spot_series[i - 1]

    # Normalize: straddle as percentage of spot
    straddle_pct = np.where(
        spot_series > 0,
        straddle_raw / spot_series,
        np.nan,
    )

    # Fill any remaining NaNs at the start
    valid_mask = ~np.isnan(straddle_pct)
    if not valid_mask.any():
        result.skipped = True
        result.skip_reason = "all_nan_straddle"
        return result

    first_valid = np.argmax(valid_mask)
    for i in range(first_valid):
        straddle_pct[i] = straddle_pct[first_valid]

    # ---- Compute rolling z-score of normalized straddle ----
    z_scores = _rolling_zscore(straddle_pct, window=ZSCORE_WINDOW)

    # ---- Signal generation + position management ----
    trades: list[StraddleTrade] = []
    position = 0             # +1 long straddle, -1 short straddle
    entry_price = 0.0
    entry_ce = 0.0
    entry_pe = 0.0
    entry_straddle_pct_val = 0.0
    entry_z = 0.0
    entry_bar = 0
    entry_time_dt: datetime | None = None
    n_roundtrips = 0
    n_signals = 0
    cumulative_pnl = 0.0

    n = len(timestamps)
    for i in range(n):
        ts = timestamps[i]
        t = ts.time() if hasattr(ts, "time") else datetime.fromtimestamp(0).time()
        z = z_scores[i]
        px_straddle = straddle_raw[i]
        px_ce = ce_close[i]
        px_pe = pe_close[i]
        str_pct = straddle_pct[i]

        # Skip if z-score is not yet available (warm-up)
        if np.isnan(z):
            continue

        exit_reason = ""

        # ---- Exit logic (check before entry) ----
        if position != 0 and entry_time_dt is not None:
            bars_held = i - entry_bar

            # (a) Mean reversion complete: z crosses 0
            if position == -1 and z <= Z_EXIT_MEAN:
                exit_reason = "mean_revert"
            elif position == 1 and z >= Z_EXIT_MEAN:
                exit_reason = "mean_revert"

            # (b) Stop: extreme z (trending away, not reverting)
            elif position == -1 and z > Z_STOP:
                exit_reason = "stop_z_trend"
            elif position == 1 and z < -Z_STOP:
                exit_reason = "stop_z_trend"

            # (c) Max hold time (60 bars = 60 minutes)
            elif bars_held >= MAX_HOLD_BARS:
                exit_reason = "max_hold"

            # (d) Force close at 15:00
            elif t >= FORCE_CLOSE_TIME:
                exit_reason = "eod_force"

            if exit_reason:
                # Compute P&L per leg
                pnl_ce = position * (px_ce - entry_ce)
                pnl_pe = position * (px_pe - entry_pe)
                pnl_gross = pnl_ce + pnl_pe
                pnl_net = pnl_gross - cost_rt

                trade = StraddleTrade(
                    trade_date=d,
                    entry_time=entry_time_dt,
                    exit_time=ts if isinstance(ts, datetime) else datetime.combine(d, t),
                    direction=position,
                    entry_straddle_price=entry_price,
                    exit_straddle_price=px_straddle,
                    entry_straddle_pct=entry_straddle_pct_val,
                    exit_straddle_pct=str_pct,
                    entry_ce_price=entry_ce,
                    exit_ce_price=px_ce,
                    entry_pe_price=entry_pe,
                    exit_pe_price=px_pe,
                    entry_z=entry_z,
                    exit_z=z,
                    atm_strike=atm_strike,
                    expiry=str(nearest_expiry),
                    pnl_points=pnl_net,
                    pnl_ce_points=pnl_ce,
                    pnl_pe_points=pnl_pe,
                    exit_reason=exit_reason,
                    bars_held=bars_held,
                    cost_points=cost_rt,
                )
                trades.append(trade)
                cumulative_pnl += pnl_net
                n_roundtrips += 1

                position = 0
                entry_price = 0.0
                entry_ce = 0.0
                entry_pe = 0.0
                entry_straddle_pct_val = 0.0
                entry_z = 0.0
                entry_time_dt = None
                entry_bar = 0

        # ---- Entry logic ----
        in_trading_window = ENTRY_OPEN <= t < ENTRY_CLOSE
        # Volume filter: both legs must have meaningful volume
        has_volume = ce_vol[i] >= MIN_LEG_VOLUME and pe_vol[i] >= MIN_LEG_VOLUME
        can_trade = (
            position == 0
            and in_trading_window
            and has_volume
            and n_roundtrips < MAX_ROUNDTRIPS_PER_DAY
        )

        if can_trade:
            if z > Z_ENTRY_SELL:
                n_signals += 1
                # Sell straddle (straddle is expensive, expect it to fall)
                position = -1
                entry_price = px_straddle
                entry_ce = px_ce
                entry_pe = px_pe
                entry_straddle_pct_val = str_pct
                entry_z = z
                entry_bar = i
                entry_time_dt = ts if isinstance(ts, datetime) else datetime.combine(d, t)

            elif z < Z_ENTRY_BUY:
                n_signals += 1
                # Buy straddle (straddle is cheap, expect it to rise)
                position = 1
                entry_price = px_straddle
                entry_ce = px_ce
                entry_pe = px_pe
                entry_straddle_pct_val = str_pct
                entry_z = z
                entry_bar = i
                entry_time_dt = ts if isinstance(ts, datetime) else datetime.combine(d, t)

    # ---- Force-close any remaining position at last bar ----
    if position != 0 and entry_time_dt is not None:
        px_ce = ce_close[-1]
        px_pe = pe_close[-1]
        px_straddle = straddle_raw[-1]

        pnl_ce = position * (px_ce - entry_ce)
        pnl_pe = position * (px_pe - entry_pe)
        pnl_gross = pnl_ce + pnl_pe
        pnl_net = pnl_gross - cost_rt

        trade = StraddleTrade(
            trade_date=d,
            entry_time=entry_time_dt,
            exit_time=timestamps[-1] if isinstance(timestamps[-1], datetime) else datetime.combine(d, time(15, 29)),
            direction=position,
            entry_straddle_price=entry_price,
            exit_straddle_price=px_straddle,
            entry_straddle_pct=entry_straddle_pct_val,
            exit_straddle_pct=straddle_pct[-1],
            entry_ce_price=entry_ce,
            exit_ce_price=px_ce,
            entry_pe_price=entry_pe,
            exit_pe_price=px_pe,
            entry_z=entry_z,
            exit_z=z_scores[-1] if not np.isnan(z_scores[-1]) else 0.0,
            atm_strike=atm_strike,
            expiry=str(nearest_expiry),
            pnl_points=pnl_net,
            pnl_ce_points=pnl_ce,
            pnl_pe_points=pnl_pe,
            exit_reason="eod_force",
            bars_held=n - 1 - entry_bar,
            cost_points=cost_rt,
        )
        trades.append(trade)
        cumulative_pnl += pnl_net
        n_roundtrips += 1

    result.trades = trades
    result.daily_pnl = cumulative_pnl
    result.n_roundtrips = n_roundtrips
    result.n_signals = n_signals
    return result


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

def run_intraday_straddle_backtest(
    symbol: str = "NIFTY",
    store: Any = None,
    n_days: int | None = None,
    date_range: list[date] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the intraday straddle mean-reversion backtest.

    Parameters
    ----------
    symbol : str
        Underlying name ("NIFTY" or "BANKNIFTY").
    store : MarketDataStore, optional
        If None, creates one automatically.
    n_days : int, optional
        Number of most recent trading days to run. If None, runs all.
    date_range : list[date], optional
        Specific dates to run.

    Returns
    -------
    (trade_df, daily_df, bar_df)
        - trade_df: per-trade details
        - daily_df: per-day summary
        - bar_df: placeholder (empty for now)
    """
    own_store = store is None
    if store is None:
        project_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(project_root))
        from quantlaxmi.data.store import MarketDataStore
        store = MarketDataStore()

    try:
        return _run_backtest_impl(symbol, store, n_days, date_range)
    finally:
        if own_store:
            store.close()


def _run_backtest_impl(
    symbol: str,
    store: Any,
    n_days: int | None,
    date_range: list[date] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Core backtest implementation."""
    # Cost
    cost_per_leg = COST_PER_LEG.get(symbol.upper(), 3.0)
    # 2 legs x entry + 2 legs x exit = 4 legs
    cost_rt = cost_per_leg * 4

    # Resolve dates
    if date_range is None:
        all_dates = store.available_dates("nfo_1min")
        if not all_dates:
            logger.error("No nfo_1min dates available")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if n_days is not None and len(all_dates) > n_days:
            date_range = all_dates[-n_days:]
        else:
            date_range = all_dates

    sorted_dates = sorted(date_range)

    # Load VIX series for event-day filtering
    vix_map = _load_vix_series(store)

    # Results accumulators
    all_trades: list[StraddleTrade] = []
    daily_summaries: list[dict[str, Any]] = []

    prev_vix: float | None = None
    n_skipped = 0
    n_traded = 0

    for d in sorted_dates:
        d_str = d.isoformat()
        vix_today = vix_map.get(d_str, None)

        # Load all data for this day in a single efficient query
        all_data = _load_day_data(store, symbol, d_str)
        if all_data is None:
            n_skipped += 1
            daily_summaries.append({
                "date": d, "daily_pnl": 0.0, "n_trades": 0,
                "n_roundtrips": 0, "skipped": True,
                "skip_reason": "load_error", "fut_open": 0.0,
                "fut_close": 0.0, "intraday_range_pct": 0.0,
                "n_signals": 0,
            })
            if vix_today is not None:
                prev_vix = vix_today
            continue

        day_result = _run_single_day(
            all_data=all_data,
            symbol=symbol,
            d=d,
            cost_rt=cost_rt,
            prev_vix=prev_vix,
            vix_today=vix_today,
        )

        if day_result.skipped:
            n_skipped += 1
            logger.debug("%s SKIP: %s", d, day_result.skip_reason)
        else:
            n_traded += 1
            if day_result.trades:
                all_trades.extend(day_result.trades)

        daily_summaries.append({
            "date": d,
            "daily_pnl": day_result.daily_pnl,
            "n_trades": len(day_result.trades),
            "n_roundtrips": day_result.n_roundtrips,
            "skipped": day_result.skipped,
            "skip_reason": day_result.skip_reason,
            "fut_open": day_result.fut_close_start,
            "fut_close": day_result.fut_close_end,
            "intraday_range_pct": day_result.intraday_range_pct,
            "n_signals": day_result.n_signals,
        })

        # Update prev_vix for next day's filter
        if vix_today is not None:
            prev_vix = vix_today

        if (n_traded + n_skipped) % 50 == 0:
            logger.info(
                "Progress: %d/%d days processed (%d skipped, %d trades so far)",
                n_traded + n_skipped, len(sorted_dates), n_skipped,
                len(all_trades),
            )

    # Build output DataFrames
    trade_df = pd.DataFrame([
        {
            "trade_date": t.trade_date,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction,
            "entry_straddle": t.entry_straddle_price,
            "exit_straddle": t.exit_straddle_price,
            "entry_straddle_pct": t.entry_straddle_pct,
            "exit_straddle_pct": t.exit_straddle_pct,
            "entry_ce": t.entry_ce_price,
            "exit_ce": t.exit_ce_price,
            "entry_pe": t.entry_pe_price,
            "exit_pe": t.exit_pe_price,
            "entry_z": t.entry_z,
            "exit_z": t.exit_z,
            "atm_strike": t.atm_strike,
            "expiry": t.expiry,
            "pnl_points": t.pnl_points,
            "pnl_ce": t.pnl_ce_points,
            "pnl_pe": t.pnl_pe_points,
            "cost_points": t.cost_points,
            "exit_reason": t.exit_reason,
            "bars_held": t.bars_held,
        }
        for t in all_trades
    ])

    daily_df = pd.DataFrame(daily_summaries)

    # Print statistics
    _print_statistics(daily_df, trade_df, symbol, cost_rt, n_skipped)

    return trade_df, daily_df, pd.DataFrame()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _print_statistics(
    daily_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    symbol: str,
    cost_rt: float,
    n_skipped: int,
) -> None:
    """Print backtest statistics to stdout."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Intraday Straddle Mean-Reversion — {symbol}")
    print(sep)

    if daily_df.empty:
        print("  No trading days in backtest.")
        return

    n_total_days = len(daily_df)
    active_df = daily_df[~daily_df["skipped"]]
    n_active = len(active_df)
    n_trades = len(trade_df) if not trade_df.empty else 0

    # Daily P&L includes ALL days (skipped = 0 P&L) per protocol
    daily_pnl = daily_df["daily_pnl"].values
    mean_daily = np.mean(daily_pnl)
    std_daily = np.std(daily_pnl, ddof=1) if n_total_days > 1 else 1.0
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 1e-10 else 0.0

    total_pnl = np.sum(daily_pnl)

    # Cumulative equity curve stats
    cum_pnl = np.cumsum(daily_pnl)
    peak = np.maximum.accumulate(cum_pnl)
    eq_dd = peak - cum_pnl
    max_eq_dd = float(np.max(eq_dd)) if len(eq_dd) > 0 else 0.0

    # Trade stats
    trades_per_day = n_trades / max(n_active, 1)

    if n_trades > 0:
        wins = trade_df[trade_df["pnl_points"] > 0]
        losses = trade_df[trade_df["pnl_points"] <= 0]
        win_rate = len(wins) / n_trades
        avg_win = float(wins["pnl_points"].mean()) if len(wins) > 0 else 0.0
        avg_loss = float(abs(losses["pnl_points"].mean())) if len(losses) > 0 else 1.0
        win_loss_ratio = avg_win / max(avg_loss, 1e-10)
        avg_bars_held = float(trade_df["bars_held"].mean())
        avg_pnl_per_trade = float(trade_df["pnl_points"].mean())
        median_pnl = float(trade_df["pnl_points"].median())
        total_gross_pnl = float(trade_df["pnl_points"].sum() + trade_df["cost_points"].sum())
        avg_gross_pnl = total_gross_pnl / n_trades

        # Direction breakdown
        short_trades = trade_df[trade_df["direction"] == -1]
        long_trades = trade_df[trade_df["direction"] == 1]

        # Exit reason breakdown
        exit_counts = trade_df["exit_reason"].value_counts().to_dict()
    else:
        win_rate = avg_win = avg_loss = win_loss_ratio = 0.0
        avg_bars_held = avg_pnl_per_trade = median_pnl = 0.0
        total_gross_pnl = avg_gross_pnl = 0.0
        short_trades = long_trades = pd.DataFrame()
        exit_counts = {}

    print(f"  Period         : {daily_df['date'].iloc[0]} to {daily_df['date'].iloc[-1]}")
    print(f"  Total days     : {n_total_days}")
    print(f"  Active days    : {n_active} (skipped: {n_skipped})")
    print(f"  Cost per RT    : {cost_rt:.1f} index pts (4 legs x {cost_rt / 4:.1f})")
    print()
    print(f"  Total P&L      : {total_pnl:>+10.1f} pts (net)")
    print(f"  Total P&L gross: {total_gross_pnl:>+10.1f} pts (before costs)")
    print(f"  Sharpe (ann.)  : {sharpe:>10.2f}")
    print(f"  Max equity DD  : {max_eq_dd:>10.1f} pts")
    print()
    print(f"  Total trades   : {n_trades:>10d}")
    print(f"  Trades/active  : {trades_per_day:>10.2f}")
    print(f"  Win rate       : {win_rate:>10.1%}")
    print(f"  Avg win/loss   : {win_loss_ratio:>10.2f}")
    print(f"  Avg win        : {avg_win:>+10.1f} pts")
    print(f"  Avg loss       : {-avg_loss:>+10.1f} pts")
    print(f"  Avg P&L/trade  : {avg_pnl_per_trade:>+10.1f} pts (net)")
    print(f"  Avg P&L gross  : {avg_gross_pnl:>+10.1f} pts")
    print(f"  Median P&L     : {median_pnl:>+10.1f} pts")
    print(f"  Avg hold       : {avg_bars_held:>10.1f} bars (minutes)")

    if len(short_trades) > 0 or len(long_trades) > 0:
        print()
        print(f"  Direction breakdown:")
        if len(short_trades) > 0:
            short_pnl = float(short_trades["pnl_points"].sum())
            short_wr = len(short_trades[short_trades["pnl_points"] > 0]) / len(short_trades)
            short_avg = float(short_trades["pnl_points"].mean())
            print(f"    SHORT straddle : {len(short_trades):4d} trades | "
                  f"P&L {short_pnl:>+8.1f} pts | WR {short_wr:.1%} | "
                  f"Avg {short_avg:>+.1f}")
        if len(long_trades) > 0:
            long_pnl = float(long_trades["pnl_points"].sum())
            long_wr = len(long_trades[long_trades["pnl_points"] > 0]) / len(long_trades)
            long_avg = float(long_trades["pnl_points"].mean())
            print(f"    LONG  straddle : {len(long_trades):4d} trades | "
                  f"P&L {long_pnl:>+8.1f} pts | WR {long_wr:.1%} | "
                  f"Avg {long_avg:>+.1f}")

    if exit_counts:
        print()
        print(f"  Exit reasons  :")
        for reason, cnt in sorted(exit_counts.items(), key=lambda x: -x[1]):
            pnl_for_reason = float(trade_df[trade_df["exit_reason"] == reason]["pnl_points"].sum())
            avg_for_reason = pnl_for_reason / cnt
            print(f"    {reason:18s} {cnt:5d}  ({cnt / n_trades:.0%})  "
                  f"P&L {pnl_for_reason:>+8.1f}  Avg {avg_for_reason:>+.1f}")

    # Skip reason breakdown
    if n_skipped > 0:
        print()
        skip_reasons = daily_df[daily_df["skipped"]]["skip_reason"].value_counts().to_dict()
        print(f"  Skip reasons  :")
        for reason, cnt in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:30s} {cnt:5d}")

    print(sep)
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Ensure project root is on path
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from quantlaxmi.data.store import MarketDataStore

    store = MarketDataStore()

    symbol = sys.argv[1] if len(sys.argv) > 1 else "NIFTY"
    n_days_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"Running Intraday Straddle Mean-Reversion — {symbol}")
    if n_days_arg:
        print(f"  Last {n_days_arg} trading days")
    else:
        print(f"  All available dates")
    print()

    trade_df, daily_df, _ = run_intraday_straddle_backtest(
        symbol=symbol,
        store=store,
        n_days=n_days_arg,
    )

    # Save results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not trade_df.empty:
        trade_path = results_dir / f"intraday_straddle_mr_{symbol}_{ts_str}_trades.csv"
        trade_df.to_csv(trade_path, index=False)
        print(f"Trade results saved to {trade_path}")

    if not daily_df.empty:
        daily_path = results_dir / f"intraday_straddle_mr_{symbol}_{ts_str}_daily.csv"
        daily_df.to_csv(daily_path, index=False)
        print(f"Daily results saved to {daily_path}")

    store.close()
