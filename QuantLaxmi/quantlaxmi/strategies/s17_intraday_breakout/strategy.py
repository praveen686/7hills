"""Intraday 1-Minute Bar Breakout + Momentum Strategy for NIFTY / BANKNIFTY.

Combines two active sub-strategies on 1-min futures bars:

1. **Opening Range Breakout (ORB)** -- 15-min opening range (09:15-09:30),
   volume-confirmed breakout with 2-bar close confirmation.
   Target: 1.0x OR width, Stop: 0.75x OR width. Max hold: 60 bars.
   Only fires between 09:30-11:30. OR width filter: 30-150 pts.
   *Primary alpha source* -- +223 pts / 81 trades over 316 days.

2. **Momentum Ignition** -- detect 3 consecutive same-direction 1-min bars
   with bar3 volume > 2x bar1 volume and total move > 0.3%.
   Target: repeat of 3-bar move, Stop: retrace to bar1 open. Max hold: 30 bars.

(VWAP Mean Reversion was tested at multiple sigma thresholds but loses
consistently on NIFTY 1-min data. Disabled by default.)

Signals are scored (-1, 0, +1) and weighted (ORB 0.5, Momentum 0.5).
Any single sub-strategy can trigger independently.

Risk Management:
- No new entries after 14:30
- All positions closed by 15:15
- Max 3 trades per day
- No trades in first bar (09:15)
- Daily stop: -0.5% of notional -> shut down for day
- Cost: 3 index pts round-trip NIFTY, 5 pts BANKNIFTY

Backtest Results (316 days, 2024-10-29 to 2026-02-06):
- Total PnL: +58.9 pts (+0.24% return) after 3 pts/trade cost
- Sharpe: 0.08, Profit Factor: 1.02, Win Rate: 49.2%
- 130 trades (0.41/day), Max Drawdown: 456.9 pts

Data: hive-partitioned Parquet at ``nfo_1min/date=YYYY-MM-DD/data.parquet``.

Author : AlphaForge
Created: 2026-02-08
"""

from __future__ import annotations

import logging
import os
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
OR_END = time(9, 30)          # opening range computed over 09:15-09:29 (15 bars)
NO_ENTRY_AFTER = time(14, 30)
FORCE_CLOSE_BY = time(15, 15)

VWAP_MR_START = time(10, 0)
VWAP_MR_END = time(14, 30)

# Round-trip cost in index points
COST_PER_RT: dict[str, float] = {
    "NIFTY": 3.0,
    "BANKNIFTY": 5.0,
    "FINNIFTY": 3.0,
    "MIDCPNIFTY": 3.0,
}

# Sub-strategy weights
WEIGHT_ORB = 0.5
WEIGHT_VWAP = 0.0       # VWAP disabled: MR loses across all sigma thresholds
WEIGHT_MOMENTUM = 0.5
COMBINED_THRESHOLD = 0.7

# VWAP confirmation filter for ORB
ORB_REQUIRE_VWAP_CONFIRM = False  # disabled: VWAP confirm degrades ORB performance

# ORB parameters
ORB_TARGET_MULT = 1.0       # target = 1.0 * OR width
ORB_STOP_MULT = 0.75        # stop = 0.75 * OR width
ORB_VOLUME_MULT = 1.5       # breakout bar volume > 1.5x average OR volume
ORB_CONFIRM_BARS = 2        # require 2 consecutive bars past OR boundary
ORB_MIN_WIDTH = 30.0        # minimum OR width in points (skip chop days)
ORB_MAX_WIDTH = 150.0       # skip ORB on very volatile opens
ORB_ENTRY_DEADLINE = time(11, 30)  # no new ORB entries after 11:30
ENABLE_TRAILING_STOP = False # trailing stop disabled (hurts performance)

# VWAP MR parameters
VWAP_ENTRY_SIGMA = 4.0      # enter only at extreme 4.0 sigma deviations
VWAP_STOP_SIGMA = 5.5       # stop at 5.5 sigma from VWAP
VWAP_TARGET_SIGMA = 2.0     # target: partial reversion to 2 sigma from VWAP
VWAP_COOLDOWN_BARS = 60     # min bars between VWAP MR trades
VWAP_MAX_OR_WIDTH = 100.0   # skip VWAP MR on wide-OR (trending) days

# Momentum ignition parameters
MOMENTUM_MIN_BARS = 3        # 3 consecutive same-direction bars
MOMENTUM_VOL_MULT = 2.0      # bar3 volume > 2x bar1 volume
MOMENTUM_MIN_MOVE_PCT = 0.003  # minimum 0.3% move

# Risk management
MAX_TRADES_PER_DAY = 3
DAILY_STOP_PCT = 0.005       # -0.5% of notional -> shut down

# Default data root
from quantlaxmi.data._paths import DATA_ROOT as _DEFAULT_DATA_ROOT, MARKET_DIR
_DEFAULT_NFO_DIR = MARKET_DIR / "nfo_1min"


# ---------------------------------------------------------------------------
# Data loading -- reads Parquet directly, no DuckDB required
# ---------------------------------------------------------------------------

def load_all_dates(nfo_dir: Path | None = None) -> list[date]:
    """Discover all available trading dates from hive partitions."""
    nfo_dir = nfo_dir or _DEFAULT_NFO_DIR
    dates = []
    if not nfo_dir.exists():
        logger.error("NFO data directory not found: %s", nfo_dir)
        return dates
    for d_dir in nfo_dir.iterdir():
        if d_dir.is_dir() and d_dir.name.startswith("date="):
            try:
                dates.append(date.fromisoformat(d_dir.name[5:]))
            except ValueError:
                continue
    return sorted(dates)


def load_day_bars(
    d: date,
    symbol_name: str,
    nfo_dir: Path | None = None,
) -> pd.DataFrame:
    """Load 1-min bars for the front-month FUT contract on a given date.

    Uses the expiry with the highest volume (handles rollover days).
    Returns DataFrame sorted by timestamp with columns:
    timestamp, open, high, low, close, volume, oi, symbol, expiry.
    """
    nfo_dir = nfo_dir or _DEFAULT_NFO_DIR
    parquet_path = nfo_dir / f"date={d.isoformat()}" / "data.parquet"
    if not parquet_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(parquet_path)

    # Filter for the requested underlying futures
    mask = (df["name"] == symbol_name) & (df["instrument_type"] == "FUT")
    fut = df.loc[mask].copy()
    if fut.empty:
        return pd.DataFrame()

    # Ensure a 'timestamp' column exists.
    # Some partitions only have 'date' (which contains full timestamps).
    if "timestamp" not in fut.columns:
        if "date" in fut.columns:
            fut["timestamp"] = pd.to_datetime(fut["date"])
        else:
            return pd.DataFrame()
    else:
        fut["timestamp"] = pd.to_datetime(fut["timestamp"])

    # On rollover days, pick the expiry with the most volume
    if fut["expiry"].nunique() > 1:
        vol_by_exp = fut.groupby("expiry")["volume"].sum()
        best_expiry = vol_by_exp.idxmax()
        fut = fut[fut["expiry"] == best_expiry].copy()

    # Cast numeric columns to float64 for precision
    for col in ("open", "high", "low", "close"):
        fut[col] = fut[col].astype(np.float64)

    return (
        fut[["timestamp", "open", "high", "low", "close", "volume", "oi", "symbol", "expiry"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Trade / result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Single intraday trade record."""
    entry_time: datetime
    exit_time: datetime | None = None
    direction: int = 0           # +1 long, -1 short
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl_points: float = 0.0     # after costs
    exit_reason: str = ""
    sub_strategy: str = ""       # which sub-strategy triggered entry
    combined_score: float = 0.0
    bars_held: int = 0


@dataclass
class DayResult:
    """Per-day backtest output."""
    trade_date: date
    trades: list[Trade] = field(default_factory=list)
    daily_pnl: float = 0.0
    n_trades: int = 0
    max_intraday_dd: float = 0.0
    notional: float = 0.0  # approximate notional (first bar close)


# ---------------------------------------------------------------------------
# Sub-strategy 1: Opening Range Breakout (ORB)
# ---------------------------------------------------------------------------

def compute_opening_range(bars: pd.DataFrame) -> dict[str, float] | None:
    """Compute the 15-minute opening range (09:15-09:29).

    Returns dict with or_high, or_low, or_width, avg_or_volume, or None if
    insufficient data.
    """
    times = bars["timestamp"].dt.time
    or_mask = (times >= MARKET_OPEN) & (times < OR_END)
    or_bars = bars.loc[or_mask]

    if len(or_bars) < 5:  # need at least 5 bars of 15 to be meaningful
        return None

    or_high = or_bars["high"].max()
    or_low = or_bars["low"].min()
    or_width = or_high - or_low

    if or_width < 1.0:  # trivially narrow range
        return None

    avg_or_volume = or_bars["volume"].mean()

    return {
        "or_high": or_high,
        "or_low": or_low,
        "or_width": or_width,
        "avg_or_volume": avg_or_volume,
    }


def orb_raw_signal(
    bar_close: float,
    bar_volume: float,
    or_info: dict[str, float] | None,
) -> int:
    """Return raw ORB direction: +1 (above OR_high), -1 (below OR_low), 0.

    Does NOT check confirmation bars -- the caller tracks multi-bar confirmation.
    Breakout requires:
    - Close above OR_high (long) or below OR_low (short)
    - Volume > 1.5x average OR volume
    - OR width >= ORB_MIN_WIDTH (skip chop days)
    """
    if or_info is None:
        return 0
    if or_info["or_width"] < ORB_MIN_WIDTH or or_info["or_width"] > ORB_MAX_WIDTH:
        return 0

    vol_ok = bar_volume > ORB_VOLUME_MULT * or_info["avg_or_volume"]
    if not vol_ok:
        return 0

    if bar_close > or_info["or_high"]:
        return 1
    elif bar_close < or_info["or_low"]:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Sub-strategy 2: VWAP Mean Reversion
# ---------------------------------------------------------------------------

def compute_running_vwap(bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute running VWAP and running standard deviation from 09:15.

    Returns (vwap_array, sigma_array), both aligned with bars index.

    VWAP = cumsum(close * volume) / cumsum(volume)
    Sigma = running std of (close - vwap) using expanding window.
    """
    close = bars["close"].values
    volume = bars["volume"].values.astype(np.float64)

    n = len(close)
    vwap = np.full(n, np.nan)
    sigma = np.full(n, np.nan)

    cum_pv = 0.0
    cum_vol = 0.0
    deviations = []

    for i in range(n):
        cum_pv += close[i] * volume[i]
        cum_vol += volume[i]

        if cum_vol > 0:
            vwap[i] = cum_pv / cum_vol
            dev = close[i] - vwap[i]
            deviations.append(dev)

            if len(deviations) >= 20:
                sigma[i] = np.std(deviations, ddof=1)
            elif len(deviations) >= 5:
                sigma[i] = np.std(deviations, ddof=1)

    return vwap, sigma


def vwap_mr_signal(
    bar_close: float,
    vwap_val: float,
    sigma_val: float,
    bar_time: time,
) -> int:
    """Return VWAP MR signal: +1 (long/fade short), -1 (short/fade long), 0.

    Only active between 10:00 and 14:30.
    Fires at extreme deviations (VWAP_ENTRY_SIGMA from VWAP).
    Entry: close > VWAP + N*sigma -> SHORT, close < VWAP - N*sigma -> LONG.
    """
    if not (VWAP_MR_START <= bar_time <= VWAP_MR_END):
        return 0

    if np.isnan(vwap_val) or np.isnan(sigma_val) or sigma_val < 1e-6:
        return 0

    deviation = bar_close - vwap_val

    if deviation > VWAP_ENTRY_SIGMA * sigma_val:
        return -1  # fade the up-move
    elif deviation < -VWAP_ENTRY_SIGMA * sigma_val:
        return 1   # fade the down-move
    return 0


# ---------------------------------------------------------------------------
# Sub-strategy 3: Momentum Ignition Detection
# ---------------------------------------------------------------------------

def momentum_ignition_signal(
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    open_arr: np.ndarray,
    idx: int,
) -> int:
    """Detect momentum ignition at bar index `idx`.

    Conditions:
    - 3 consecutive 1-min bars in the same direction (close > open or close < open)
    - Volume increasing: vol[i] > vol[i-1] > vol[i-2]  (not strictly required,
      but bar3 volume > 2x bar1 volume is required)
    - Total 3-bar move > 0.2%
    - Enter in direction of momentum.

    Returns +1 (long momentum), -1 (short momentum), 0 (no signal).
    """
    if idx < MOMENTUM_MIN_BARS:
        return 0

    # Check 3 consecutive bars (idx-2, idx-1, idx)
    directions = []
    for j in range(idx - 2, idx + 1):
        if close_arr[j] > open_arr[j]:
            directions.append(1)
        elif close_arr[j] < open_arr[j]:
            directions.append(-1)
        else:
            directions.append(0)

    # All 3 must be in the same direction (and non-zero)
    if directions[0] == 0 or not all(d == directions[0] for d in directions):
        return 0

    # Volume on bar 3 > 2x volume on bar 1
    vol_bar1 = volume_arr[idx - 2]
    vol_bar3 = volume_arr[idx]
    if vol_bar1 <= 0 or vol_bar3 < MOMENTUM_VOL_MULT * vol_bar1:
        return 0

    # Total move > 0.2%
    move_pct = abs(close_arr[idx] - open_arr[idx - 2]) / max(open_arr[idx - 2], 1.0)
    if move_pct < MOMENTUM_MIN_MOVE_PCT:
        return 0

    return directions[0]


# ---------------------------------------------------------------------------
# Position management for active trades
# ---------------------------------------------------------------------------

@dataclass
class ActivePosition:
    """Tracks an open intraday position."""
    direction: int              # +1 or -1
    entry_price: float
    entry_time: datetime
    entry_bar_idx: int
    sub_strategy: str
    combined_score: float

    # Targets/stops (set per sub-strategy)
    target_price: float | None = None
    stop_price: float | None = None
    initial_stop: float | None = None  # original stop (before trailing)

    # VWAP-specific
    vwap_at_entry: float | None = None

    # Trailing stop tracking
    peak_favorable: float = 0.0  # best price seen in our direction


def _compute_orb_stops(
    direction: int,
    entry_price: float,
    or_info: dict[str, float],
) -> tuple[float, float]:
    """Compute target and stop for ORB trade."""
    width = or_info["or_width"]
    if direction == 1:
        target = entry_price + ORB_TARGET_MULT * width
        stop = entry_price - ORB_STOP_MULT * width
    else:
        target = entry_price - ORB_TARGET_MULT * width
        stop = entry_price + ORB_STOP_MULT * width
    return target, stop


def _compute_vwap_stops(
    direction: int,
    entry_price: float,
    vwap_val: float,
    sigma_val: float,
) -> tuple[float, float]:
    """Compute target and stop for VWAP MR trade.

    Target: partial reversion to VWAP_TARGET_SIGMA from VWAP.
    Stop: VWAP_STOP_SIGMA from VWAP.
    """
    if direction == 1:
        # Long (fading a down-move): target above entry, stop below
        target = vwap_val - VWAP_TARGET_SIGMA * sigma_val
        stop = vwap_val - VWAP_STOP_SIGMA * sigma_val
    else:
        # Short (fading an up-move): target below entry, stop above
        target = vwap_val + VWAP_TARGET_SIGMA * sigma_val
        stop = vwap_val + VWAP_STOP_SIGMA * sigma_val
    return target, stop


def _compute_momentum_stops(
    direction: int,
    entry_price: float,
    close_arr: np.ndarray,
    open_arr: np.ndarray,
    idx: int,
) -> tuple[float, float]:
    """Compute target and stop for momentum ignition trade.

    Target: same magnitude as the 3-bar move from bar 1 open.
    Stop: retrace to bar 1 open.
    """
    bar1_open = open_arr[idx - 2]
    move_size = abs(close_arr[idx] - bar1_open)
    stop = bar1_open

    if direction == 1:
        target = entry_price + move_size
    else:
        target = entry_price - move_size

    return target, stop


# ---------------------------------------------------------------------------
# Single-day simulation
# ---------------------------------------------------------------------------

def _run_single_day(
    bars: pd.DataFrame,
    cost_rt: float,
    notional_ref: float,
) -> DayResult:
    """Run all three sub-strategies for a single trading day.

    Each sub-strategy generates signals independently.  When any signal fires
    and no position is open, a trade is taken in that sub-strategy's direction
    with its own target/stop.  If multiple signals fire on the same bar, the
    combined weighted score determines direction and the dominant sub-strategy
    manages the exit.

    Parameters
    ----------
    bars : DataFrame
        1-min bars for a single FUT contract, sorted by timestamp.
    cost_rt : float
        Round-trip cost in index points.
    notional_ref : float
        Reference notional for daily stop calculation.

    Returns
    -------
    DayResult
    """
    trade_date_val = bars["timestamp"].iloc[0].date()
    n = len(bars)

    if n < 20:
        return DayResult(trade_date=trade_date_val)

    timestamps = bars["timestamp"].values
    time_arr = np.array([pd.Timestamp(t).time() for t in timestamps])

    close = bars["close"].values.astype(np.float64)
    open_arr = bars["open"].values.astype(np.float64)
    high_arr = bars["high"].values.astype(np.float64)
    low_arr = bars["low"].values.astype(np.float64)
    volume = bars["volume"].values.astype(np.float64)

    # -- Compute features once for the whole day --
    or_info = compute_opening_range(bars)
    vwap_vals, sigma_vals = compute_running_vwap(bars)

    # Track ORB state: once a breakout fires and the trade is taken (or
    # rejected), don't fire ORB again in the same direction.  This prevents
    # repeated entries when price oscillates around OR boundary.
    orb_long_fired = False
    orb_short_fired = False

    # ORB confirmation: track consecutive bars past OR boundary
    orb_consec_long = 0
    orb_consec_short = 0

    # VWAP cooldown: bar index of last VWAP MR entry
    last_vwap_entry_bar = -999

    # -- State tracking --
    trades: list[Trade] = []
    active_pos: ActivePosition | None = None
    daily_pnl = 0.0
    peak_pnl = 0.0
    max_dd = 0.0
    n_trades_today = 0
    daily_shutdown = False

    for i in range(n):
        t = time_arr[i]
        ts = pd.Timestamp(timestamps[i])
        px = close[i]

        # -- Check daily stop --
        if daily_shutdown:
            if active_pos is not None:
                pnl = active_pos.direction * (px - active_pos.entry_price) - cost_rt
                trades.append(Trade(
                    entry_time=active_pos.entry_time,
                    exit_time=ts.to_pydatetime(),
                    direction=active_pos.direction,
                    entry_price=active_pos.entry_price,
                    exit_price=px,
                    pnl_points=pnl,
                    exit_reason="daily_stop",
                    sub_strategy=active_pos.sub_strategy,
                    combined_score=active_pos.combined_score,
                    bars_held=i - active_pos.entry_bar_idx,
                ))
                daily_pnl += pnl
                active_pos = None
            continue

        # -- Exit logic (check before entry) --
        if active_pos is not None:
            bars_held = i - active_pos.entry_bar_idx
            exit_reason = ""

            # Update peak favorable price for trailing stop
            if active_pos.direction == 1:
                active_pos.peak_favorable = max(active_pos.peak_favorable, high_arr[i])
            else:
                if active_pos.peak_favorable == 0:
                    active_pos.peak_favorable = low_arr[i]
                else:
                    active_pos.peak_favorable = min(active_pos.peak_favorable, low_arr[i])

            # Trailing stop: move stop to breakeven once we have 65% of
            # target distance in profit, then trail at 60% of peak once
            # we have 85% of target distance.
            if (
                ENABLE_TRAILING_STOP
                and active_pos.target_price is not None
                and active_pos.initial_stop is not None
                and active_pos.stop_price is not None
            ):
                target_dist = abs(active_pos.target_price - active_pos.entry_price)
                if target_dist > 1.0:
                    if active_pos.direction == 1:
                        peak_profit = active_pos.peak_favorable - active_pos.entry_price
                    else:
                        peak_profit = active_pos.entry_price - active_pos.peak_favorable

                    if peak_profit > 0.85 * target_dist:
                        # Trail at 60% of peak profit
                        trail_stop = (
                            active_pos.entry_price + active_pos.direction * 0.6 * peak_profit
                        )
                        if active_pos.direction == 1:
                            active_pos.stop_price = max(active_pos.stop_price, trail_stop)
                        else:
                            active_pos.stop_price = min(active_pos.stop_price, trail_stop)
                    elif peak_profit > 0.65 * target_dist:
                        # Move stop to breakeven (entry + cost buffer)
                        be_stop = active_pos.entry_price + active_pos.direction * 2.0
                        if active_pos.direction == 1:
                            active_pos.stop_price = max(active_pos.stop_price, be_stop)
                        else:
                            active_pos.stop_price = min(active_pos.stop_price, be_stop)

            # Force close by 15:15
            if t >= FORCE_CLOSE_BY:
                exit_reason = "force_close_1515"

            # Target hit (use high/low for intrabar check)
            elif active_pos.target_price is not None:
                if active_pos.direction == 1 and high_arr[i] >= active_pos.target_price:
                    exit_reason = "target"
                    px = active_pos.target_price
                elif active_pos.direction == -1 and low_arr[i] <= active_pos.target_price:
                    exit_reason = "target"
                    px = active_pos.target_price

            # Stop hit (including trailing stop)
            if not exit_reason and active_pos.stop_price is not None:
                if active_pos.direction == 1 and low_arr[i] <= active_pos.stop_price:
                    # Check if this is a trailing stop (stop moved above initial)
                    is_trailing = (
                        active_pos.initial_stop is not None
                        and active_pos.stop_price > active_pos.initial_stop
                    )
                    exit_reason = "trailing_stop" if is_trailing else "stop_loss"
                    px = active_pos.stop_price
                elif active_pos.direction == -1 and high_arr[i] >= active_pos.stop_price:
                    is_trailing = (
                        active_pos.initial_stop is not None
                        and active_pos.stop_price < active_pos.initial_stop
                    )
                    exit_reason = "trailing_stop" if is_trailing else "stop_loss"
                    px = active_pos.stop_price

            # VWAP MR: exit if price has reverted past VWAP
            if not exit_reason and active_pos.sub_strategy == "vwap_mr":
                if not np.isnan(vwap_vals[i]):
                    if active_pos.direction == 1 and close[i] >= vwap_vals[i]:
                        exit_reason = "vwap_reversion"
                    elif active_pos.direction == -1 and close[i] <= vwap_vals[i]:
                        exit_reason = "vwap_reversion"

            # Max hold: 60 bars for ORB, 30 for momentum, 45 for VWAP MR
            if not exit_reason:
                max_hold_map = {"orb": 60, "momentum": 30, "vwap_mr": 45}
                max_hold = max_hold_map.get(active_pos.sub_strategy, 45)
                if bars_held >= max_hold:
                    exit_reason = "max_hold"
                    px = close[i]

            if exit_reason:
                pnl = active_pos.direction * (px - active_pos.entry_price) - cost_rt
                trades.append(Trade(
                    entry_time=active_pos.entry_time,
                    exit_time=ts.to_pydatetime(),
                    direction=active_pos.direction,
                    entry_price=active_pos.entry_price,
                    exit_price=px,
                    pnl_points=pnl,
                    exit_reason=exit_reason,
                    sub_strategy=active_pos.sub_strategy,
                    combined_score=active_pos.combined_score,
                    bars_held=bars_held,
                ))
                daily_pnl += pnl
                n_trades_today += 1
                active_pos = None
                px = close[i]

                # Check daily stop after closing trade
                if notional_ref > 0 and daily_pnl < -DAILY_STOP_PCT * notional_ref:
                    daily_shutdown = True
                    continue

        # -- Entry logic --
        # Skip first bar, skip after 14:30, skip if already in position,
        # skip if max trades reached
        if (
            active_pos is not None
            or i == 0
            or t <= MARKET_OPEN
            or t > NO_ENTRY_AFTER
            or n_trades_today >= MAX_TRADES_PER_DAY
            or daily_shutdown
        ):
            peak_pnl = max(peak_pnl, daily_pnl)
            max_dd = max(max_dd, peak_pnl - daily_pnl)
            continue

        # Compute sub-strategy signals independently
        sig_orb = 0
        sig_vwap = 0
        sig_momentum = 0

        # ORB: only after opening range is complete and before deadline
        if t >= OR_END and t <= ORB_ENTRY_DEADLINE and or_info is not None:
            raw_orb = orb_raw_signal(close[i], volume[i], or_info)
            # Track consecutive bars past OR boundary for confirmation
            if raw_orb == 1:
                orb_consec_long += 1
                orb_consec_short = 0
            elif raw_orb == -1:
                orb_consec_short += 1
                orb_consec_long = 0
            else:
                orb_consec_long = 0
                orb_consec_short = 0

            # Only signal after N consecutive confirmation bars
            if orb_consec_long >= ORB_CONFIRM_BARS and not orb_long_fired:
                sig_orb = 1
            elif orb_consec_short >= ORB_CONFIRM_BARS and not orb_short_fired:
                sig_orb = -1

            # VWAP confirmation: ORB long requires price above VWAP
            if ORB_REQUIRE_VWAP_CONFIRM and sig_orb != 0 and not np.isnan(vwap_vals[i]):
                if sig_orb == 1 and close[i] < vwap_vals[i]:
                    sig_orb = 0  # long breakout rejected: price below VWAP
                elif sig_orb == -1 and close[i] > vwap_vals[i]:
                    sig_orb = 0  # short breakdown rejected: price above VWAP

        # VWAP MR: between 10:00 and 14:30, with cooldown and OR width filter
        vwap_ok = (
            (i - last_vwap_entry_bar) >= VWAP_COOLDOWN_BARS
            and (or_info is None or or_info["or_width"] <= VWAP_MAX_OR_WIDTH)
        )
        if vwap_ok:
            sig_vwap = vwap_mr_signal(close[i], vwap_vals[i], sigma_vals[i], t)

        # Momentum ignition
        sig_momentum = momentum_ignition_signal(close, volume, open_arr, i)

        # Collect active signals
        active_signals: list[tuple[str, int, float]] = []
        if sig_orb != 0:
            active_signals.append(("orb", sig_orb, WEIGHT_ORB))
        if sig_vwap != 0:
            active_signals.append(("vwap_mr", sig_vwap, WEIGHT_VWAP))
        if sig_momentum != 0:
            active_signals.append(("momentum", sig_momentum, WEIGHT_MOMENTUM))

        if not active_signals:
            peak_pnl = max(peak_pnl, daily_pnl)
            max_dd = max(max_dd, peak_pnl - daily_pnl)
            continue

        # Compute combined score from all active signals
        combined = sum(sig * w for _, sig, w in active_signals)

        # Determine entry direction:
        # - If combined score is strong enough (>= threshold), use combined
        # - If only one signal fires, take it directly (its weight alone
        #   may be below threshold, but that is intentional for diversity)
        # We use a relaxed threshold: any single sub-strategy can trigger
        # on its own, but direction is determined by the combined score.
        if abs(combined) < 0.01:
            # Signals cancel out
            peak_pnl = max(peak_pnl, daily_pnl)
            max_dd = max(max_dd, peak_pnl - daily_pnl)
            continue

        direction = 1 if combined > 0 else -1

        # Pick the dominant sub-strategy (largest weighted contribution
        # aligned with the chosen direction)
        best_sub = None
        best_contrib = -1.0
        for sub_name, sig, w in active_signals:
            contrib = sig * direction * w  # positive if aligned
            if contrib > best_contrib:
                best_contrib = contrib
                best_sub = sub_name

        entry_px = close[i]

        # Compute stops/targets based on dominant strategy
        target_px = None
        stop_px = None
        vwap_at_entry = None

        if best_sub == "orb" and or_info is not None:
            target_px, stop_px = _compute_orb_stops(direction, entry_px, or_info)
            if direction == 1:
                orb_long_fired = True
            else:
                orb_short_fired = True
        elif best_sub == "vwap_mr" and not np.isnan(vwap_vals[i]) and not np.isnan(sigma_vals[i]):
            target_px, stop_px = _compute_vwap_stops(
                direction, entry_px, vwap_vals[i], sigma_vals[i],
            )
            vwap_at_entry = vwap_vals[i]
            last_vwap_entry_bar = i
        elif best_sub == "momentum" and i >= MOMENTUM_MIN_BARS:
            target_px, stop_px = _compute_momentum_stops(
                direction, entry_px, close, open_arr, i,
            )
        else:
            # Fallback: 0.3% target/stop
            target_px = entry_px + direction * entry_px * 0.003
            stop_px = entry_px - direction * entry_px * 0.003

        active_pos = ActivePosition(
            direction=direction,
            entry_price=entry_px,
            entry_time=ts.to_pydatetime(),
            entry_bar_idx=i,
            sub_strategy=best_sub or "combined",
            combined_score=combined,
            target_price=target_px,
            stop_price=stop_px,
            initial_stop=stop_px,
            vwap_at_entry=vwap_at_entry,
            peak_favorable=entry_px,
        )

        # Track drawdown
        peak_pnl = max(peak_pnl, daily_pnl)
        max_dd = max(max_dd, peak_pnl - daily_pnl)

    # Force-close any remaining position at last bar
    if active_pos is not None:
        px = close[-1]
        pnl = active_pos.direction * (px - active_pos.entry_price) - cost_rt
        trades.append(Trade(
            entry_time=active_pos.entry_time,
            exit_time=pd.Timestamp(timestamps[-1]).to_pydatetime(),
            direction=active_pos.direction,
            entry_price=active_pos.entry_price,
            exit_price=px,
            pnl_points=pnl,
            exit_reason="eod_force",
            sub_strategy=active_pos.sub_strategy,
            combined_score=active_pos.combined_score,
            bars_held=n - 1 - active_pos.entry_bar_idx,
        ))
        daily_pnl += pnl
        n_trades_today += 1

    peak_pnl = max(peak_pnl, daily_pnl)
    max_dd = max(max_dd, peak_pnl - daily_pnl)

    return DayResult(
        trade_date=trade_date_val,
        trades=trades,
        daily_pnl=daily_pnl,
        n_trades=n_trades_today,
        max_intraday_dd=max_dd,
        notional=notional_ref,
    )


# ---------------------------------------------------------------------------
# Full backtest
# ---------------------------------------------------------------------------

def backtest(
    symbol: str = "NIFTY",
    nfo_dir: Path | str | None = None,
    n_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full intraday breakout backtest.

    Parameters
    ----------
    symbol : str
        Underlying name ("NIFTY" or "BANKNIFTY").
    nfo_dir : Path, optional
        Override for nfo_1min parquet directory.
    n_days : int, optional
        If set, only backtest last N days.

    Returns
    -------
    (daily_df, trade_df) : tuple of DataFrames
        daily_df: per-day summary (date, pnl, n_trades, max_dd)
        trade_df: per-trade detail
    """
    nfo_path = Path(nfo_dir) if nfo_dir else _DEFAULT_NFO_DIR
    cost_rt = COST_PER_RT.get(symbol.upper(), 3.0)

    all_dates = load_all_dates(nfo_path)
    if not all_dates:
        logger.error("No data found in %s", nfo_path)
        return pd.DataFrame(), pd.DataFrame()

    if n_days is not None and n_days < len(all_dates):
        all_dates = all_dates[-n_days:]

    logger.info("Backtesting %s over %d days (%s to %s)",
                symbol, len(all_dates), all_dates[0], all_dates[-1])

    daily_summaries: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []

    for idx, d in enumerate(all_dates):
        bars = load_day_bars(d, symbol, nfo_path)
        if bars.empty or len(bars) < 20:
            logger.debug("Skipping %s — insufficient bars", d)
            continue

        # Reference notional = first bar close
        notional_ref = bars["close"].iloc[0]

        day_result = _run_single_day(bars, cost_rt, notional_ref)

        daily_summaries.append({
            "date": day_result.trade_date,
            "daily_pnl": day_result.daily_pnl,
            "n_trades": day_result.n_trades,
            "max_intraday_dd": day_result.max_intraday_dd,
            "notional": day_result.notional,
        })

        for tr in day_result.trades:
            all_trades.append({
                "date": day_result.trade_date,
                "entry_time": tr.entry_time,
                "exit_time": tr.exit_time,
                "direction": tr.direction,
                "entry_price": tr.entry_price,
                "exit_price": tr.exit_price,
                "pnl_points": tr.pnl_points,
                "exit_reason": tr.exit_reason,
                "sub_strategy": tr.sub_strategy,
                "combined_score": tr.combined_score,
                "bars_held": tr.bars_held,
            })

        if (idx + 1) % 50 == 0:
            logger.info("  ... processed %d / %d days", idx + 1, len(all_dates))

    daily_df = pd.DataFrame(daily_summaries)
    trade_df = pd.DataFrame(all_trades)

    return daily_df, trade_df


# ---------------------------------------------------------------------------
# Statistics printing
# ---------------------------------------------------------------------------

def print_statistics(
    daily_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    symbol: str,
    cost_rt: float,
) -> dict[str, float]:
    """Print and return backtest statistics."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Intraday Breakout Strategy (ORB + VWAP MR + Momentum) — {symbol}")
    print(sep)

    if daily_df.empty:
        print("  No trading days in backtest.")
        return {}

    n_days = len(daily_df)
    daily_pnl = daily_df["daily_pnl"].values

    # -- Sharpe (annualized, all-day, ddof=1) --
    mean_daily = np.mean(daily_pnl)
    std_daily = np.std(daily_pnl, ddof=1) if n_days > 1 else 1.0
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 1e-10 else 0.0

    # -- Total P&L --
    total_pnl = np.sum(daily_pnl)

    # -- Daily return as % of notional --
    avg_notional = daily_df["notional"].mean()
    total_return_pct = (total_pnl / avg_notional * 100) if avg_notional > 0 else 0.0

    # -- Trade stats --
    n_trades = len(trade_df) if not trade_df.empty else 0
    trades_per_day = n_trades / max(n_days, 1)

    win_rate = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    win_loss_ratio = 0.0
    avg_bars_held = 0.0
    exit_counts: dict[str, int] = {}
    sub_strat_counts: dict[str, int] = {}
    sub_strat_pnl: dict[str, float] = {}

    if n_trades > 0:
        wins = trade_df[trade_df["pnl_points"] > 0]
        losses = trade_df[trade_df["pnl_points"] <= 0]
        win_rate = len(wins) / n_trades
        avg_win = wins["pnl_points"].mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses["pnl_points"].mean()) if len(losses) > 0 else 1.0
        win_loss_ratio = avg_win / max(avg_loss, 1e-10)
        avg_bars_held = trade_df["bars_held"].mean()
        exit_counts = trade_df["exit_reason"].value_counts().to_dict()
        sub_strat_counts = trade_df["sub_strategy"].value_counts().to_dict()
        for ss, group in trade_df.groupby("sub_strategy"):
            sub_strat_pnl[ss] = group["pnl_points"].sum()

    # -- Cumulative equity drawdown --
    cum_pnl = np.cumsum(daily_pnl)
    peak = np.maximum.accumulate(cum_pnl)
    eq_dd = peak - cum_pnl
    max_eq_dd = np.max(eq_dd) if len(eq_dd) > 0 else 0.0

    # -- Max intraday drawdown --
    max_intra_dd = daily_df["max_intraday_dd"].max()

    # -- Profit factor --
    if n_trades > 0:
        gross_profit = trade_df.loc[trade_df["pnl_points"] > 0, "pnl_points"].sum()
        gross_loss = abs(trade_df.loc[trade_df["pnl_points"] <= 0, "pnl_points"].sum())
        profit_factor = gross_profit / max(gross_loss, 1e-10)
    else:
        profit_factor = 0.0

    # -- Calmar ratio --
    annualized_pnl = mean_daily * 252
    calmar = annualized_pnl / max(max_eq_dd, 1e-10)

    print(f"  Period          : {daily_df['date'].iloc[0]} to {daily_df['date'].iloc[-1]}")
    print(f"  Trading days    : {n_days}")
    print(f"  Cost per RT     : {cost_rt:.1f} index pts")
    print(f"  Avg notional    : {avg_notional:,.0f}")
    print()
    print(f"  Total P&L       : {total_pnl:>+12.1f} pts")
    print(f"  Total return    : {total_return_pct:>+12.2f} %")
    print(f"  Sharpe (ann.)   : {sharpe:>12.2f}")
    print(f"  Calmar ratio    : {calmar:>12.2f}")
    print(f"  Profit factor   : {profit_factor:>12.2f}")
    print()
    print(f"  Total trades    : {n_trades:>12d}")
    print(f"  Trades/day      : {trades_per_day:>12.2f}")
    print(f"  Win rate        : {win_rate:>12.1%}")
    print(f"  Avg win         : {avg_win:>+12.1f} pts")
    print(f"  Avg loss        : {-avg_loss:>+12.1f} pts")
    print(f"  Win/loss ratio  : {win_loss_ratio:>12.2f}")
    print(f"  Avg hold        : {avg_bars_held:>12.1f} bars")
    print()
    print(f"  Max intraday DD : {max_intra_dd:>12.1f} pts")
    print(f"  Max equity DD   : {max_eq_dd:>12.1f} pts")

    if sub_strat_counts:
        print()
        print(f"  Sub-strategy breakdown:")
        for ss in sorted(sub_strat_counts.keys()):
            cnt = sub_strat_counts[ss]
            pnl = sub_strat_pnl.get(ss, 0.0)
            print(f"    {ss:15s}  trades={cnt:5d}  PnL={pnl:>+10.1f} pts")

    if exit_counts:
        print()
        print(f"  Exit reasons:")
        for reason, cnt in sorted(exit_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason:20s} {cnt:5d}  ({cnt / n_trades:.0%})")

    print(sep)
    print()

    return {
        "sharpe": sharpe,
        "total_pnl": total_pnl,
        "total_return_pct": total_return_pct,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "max_eq_dd": max_eq_dd,
        "profit_factor": profit_factor,
        "calmar": calmar,
    }


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

    symbol = "NIFTY"
    print(f"Running Intraday Breakout Backtest — {symbol} (all available dates)")
    print()

    daily_df, trade_df = backtest(symbol=symbol)

    cost_rt = COST_PER_RT.get(symbol, 3.0)
    stats = print_statistics(daily_df, trade_df, symbol, cost_rt)

    # Save results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    daily_path = results_dir / f"intraday_breakout_{symbol}_{ts_str}_daily.csv"
    trade_path = results_dir / f"intraday_breakout_{symbol}_{ts_str}_trades.csv"

    if not daily_df.empty:
        daily_df.to_csv(daily_path, index=False)
        print(f"Daily results saved to {daily_path}")

    if not trade_df.empty:
        trade_df.to_csv(trade_path, index=False)
        print(f"Trade results saved to {trade_path}")


# ---------------------------------------------------------------------------
# BaseStrategy wrapper for registry integration
# ---------------------------------------------------------------------------

from quantlaxmi.strategies.base import BaseStrategy
from quantlaxmi.strategies.protocol import Signal


class S17IntradayBreakoutStrategy(BaseStrategy):
    """Intraday breakout strategy — BaseStrategy wrapper for registry."""

    @property
    def strategy_id(self) -> str:
        return "s17_intraday_breakout"

    def warmup_days(self) -> int:
        return 0

    def _scan_impl(self, d, store) -> list[Signal]:
        """Research-only strategy — no live signals yet."""
        return []


def create_strategy() -> S17IntradayBreakoutStrategy:
    """Factory for registry auto-discovery."""
    return S17IntradayBreakoutStrategy()
