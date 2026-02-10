"""Order Flow Imbalance (OFI) Intraday Trading Strategy for India FnO.

Constructs a composite pressure score from four microstructure signals derived
from 1-minute bars and tick data, then trades short-term directional moves in
NIFTY / BANKNIFTY futures.

Features
--------
1. Multi-Level OFI (MLOFI) -- adapted for bar data using OI-weighted volume
   imbalance across the current-month and next-month contracts as a depth proxy.
2. Volume Imbalance via Bulk Volume Classification (BVC).
3. Trade Intensity -- intraday trades-per-minute normalized by 20-day average.
4. VPIN -- bar-level volume-synchronized probability of informed trading.

Signal
------
Weighted pressure score = 0.4*OFI + 0.3*VI + 0.2*TI + 0.1*dVPIN.
Long  when score > +threshold for 2+ consecutive minutes.
Short when score < -threshold for 2+ consecutive minutes.
No new entries in first/last 15 minutes of the session.

Exit: (a) 5 min after entry, (b) pressure reverses sign, (c) -0.3% stop-loss.

Data
----
Uses ``MarketDataStore`` (DuckDB + hive-partitioned Parquet).
Table: ``nfo_1min`` -- columns: timestamp, date, open, high, low, close,
volume, oi, symbol, name, expiry, instrument_type.

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
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
NO_ENTRY_OPEN = time(9, 30)    # no new entries before 9:30
NO_ENTRY_CLOSE = time(15, 15)  # no new entries after 15:15

# Round-trip cost in *index points* (per trade)
COST_PER_RT: dict[str, float] = {
    "NIFTY": 3.0,
    "BANKNIFTY": 5.0,
    "FINNIFTY": 3.0,
    "MIDCPNIFTY": 3.0,
}

PRESSURE_WEIGHTS = {
    "ofi": 0.4,
    "volume_imbalance": 0.3,
    "trade_intensity": 0.2,
    "dvpin": 0.1,
}

DEFAULT_THRESHOLD_SIGMA = 1.5
CONFIRMATION_BARS = 2          # consecutive bars above threshold
MAX_HOLD_BARS = 5              # exit after 5 minutes
STOP_LOSS_PCT = 0.003          # 0.3 %
VPIN_WINDOW = 50               # bars for rolling VPIN
SIGMA_WINDOW = 20              # bars for rolling sigma of returns


# ---------------------------------------------------------------------------
# Feature computation helpers (fully causal, no look-ahead)
# ---------------------------------------------------------------------------

def _log_returns(close: np.ndarray) -> np.ndarray:
    """Element-wise log return; first element is 0."""
    ret = np.zeros(len(close), dtype=np.float64)
    c = np.maximum(close.astype(np.float64), 1e-8)
    ret[1:] = np.log(c[1:] / c[:-1])
    return ret


def _rolling_sigma(log_ret: np.ndarray, window: int = SIGMA_WINDOW) -> np.ndarray:
    """Causal rolling std of log returns (ddof=1)."""
    n = len(log_ret)
    sigma = np.full(n, 0.01)
    for i in range(window, n):
        s = np.std(log_ret[i - window: i], ddof=1)
        if s > 1e-8:
            sigma[i] = s
    return sigma


def compute_bvc_volume_imbalance(
    close: np.ndarray,
    volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bulk Volume Classification volume imbalance.

    Returns (buy_volume, sell_volume, imbalance) where
    imbalance = (buy - sell) / total  in [-1, +1].
    """
    log_ret = _log_returns(close)
    sigma = _rolling_sigma(log_ret)

    buy_frac = norm.cdf(log_ret / sigma)
    vol = volume.astype(np.float64)
    buy_vol = vol * buy_frac
    sell_vol = vol * (1.0 - buy_frac)
    total = np.maximum(vol, 1.0)
    imbalance = (buy_vol - sell_vol) / total
    return buy_vol, sell_vol, imbalance


def compute_bar_vpin(
    close: np.ndarray,
    volume: np.ndarray,
    window: int = VPIN_WINDOW,
) -> np.ndarray:
    """Bar-level VPIN (ratio of absolute BVC imbalance to total volume)."""
    buy_vol, sell_vol, _ = compute_bvc_volume_imbalance(close, volume)
    abs_imb = np.abs(buy_vol - sell_vol)
    total_vol = np.maximum(volume.astype(np.float64), 1.0)

    n = len(close)
    vpin = np.full(n, np.nan)
    for i in range(window, n):
        vpin[i] = np.sum(abs_imb[i - window: i]) / np.sum(total_vol[i - window: i])
    return vpin


def compute_ofi_from_bars(
    close: np.ndarray,
    volume: np.ndarray,
    oi: np.ndarray,
) -> np.ndarray:
    """Multi-Level OFI adapted for bar data without full order book.

    Since we lack L2/L3 depth, we construct a proxy OFI:
    - Level 1: BVC-signed volume change  (weight 1.0)
    - Level 2: OI-signed volume change   (weight 0.5)
    - Level 3: Price-signed OI change     (weight 0.333)

    OFI_t = w1*(V_t * sign(dp)) + w2*(dOI_t * sign(dp)) + w3*(dV_t * sign(dOI))

    This captures both the directional flow (via BVC) and the positioning
    pressure (via OI changes), mimicking multi-level depth imbalance.
    """
    n = len(close)
    ofi = np.zeros(n, dtype=np.float64)

    log_ret = _log_returns(close)
    sign_ret = np.sign(log_ret)

    vol = volume.astype(np.float64)
    oi_f = oi.astype(np.float64)

    # Volume deltas
    dvol = np.zeros(n)
    dvol[1:] = vol[1:] - vol[:-1]

    # OI deltas
    doi = np.zeros(n)
    doi[1:] = oi_f[1:] - oi_f[:-1]

    # Level 1: BVC-signed volume (weight 1.0)
    level1 = vol * sign_ret

    # Level 2: OI-change signed by price direction (weight 0.5)
    level2 = doi * sign_ret

    # Level 3: Volume-change signed by OI direction (weight 1/3)
    sign_doi = np.sign(doi)
    level3 = dvol * sign_doi

    ofi = 1.0 * level1 + 0.5 * level2 + (1.0 / 3.0) * level3
    return ofi


def compute_trade_intensity(
    volume: np.ndarray,
    avg_volume_per_bar: float,
) -> np.ndarray:
    """Trade intensity: volume per bar normalized by historical average.

    Returns z-score-like values: (vol - mean) / mean.
    Since we don't have exact trade-count, volume is the best proxy from
    1-min bars.  Values > 0 indicate above-average activity.
    """
    if avg_volume_per_bar <= 0:
        return np.zeros(len(volume))
    return (volume.astype(np.float64) - avg_volume_per_bar) / max(avg_volume_per_bar, 1.0)


# ---------------------------------------------------------------------------
# Pressure score
# ---------------------------------------------------------------------------

def compute_pressure_score(
    ofi: np.ndarray,
    vol_imb: np.ndarray,
    trade_int: np.ndarray,
    vpin: np.ndarray,
) -> np.ndarray:
    """Weighted combination of the four signals, z-scored causally.

    Each component is standardized using its own expanding (causal) mean/std
    before combination.  The composite score is then a weighted sum.
    """
    n = len(ofi)
    components = {
        "ofi": ofi.copy(),
        "volume_imbalance": vol_imb.copy(),
        "trade_intensity": trade_int.copy(),
        "dvpin": np.zeros(n),
    }

    # dVPIN = first-difference of VPIN (momentum of informed trading)
    vpin_clean = np.where(np.isnan(vpin), 0.0, vpin)
    components["dvpin"][1:] = vpin_clean[1:] - vpin_clean[:-1]

    # Causal z-score each component using expanding window (min 20 bars)
    z_components: dict[str, np.ndarray] = {}
    for key, arr in components.items():
        z = np.zeros(n)
        for i in range(SIGMA_WINDOW, n):
            window = arr[:i + 1]
            mu = np.mean(window)
            sd = np.std(window, ddof=1)
            if sd > 1e-10:
                z[i] = (arr[i] - mu) / sd
        z_components[key] = z

    # Weighted sum
    pressure = np.zeros(n)
    for key, weight in PRESSURE_WEIGHTS.items():
        pressure += weight * z_components[key]

    return pressure


# ---------------------------------------------------------------------------
# Trading logic
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Single intraday trade record."""
    entry_time: datetime
    exit_time: datetime | None = None
    direction: int = 0          # +1 long, -1 short
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl_points: float = 0.0    # P&L in index points (after costs)
    exit_reason: str = ""
    bars_held: int = 0


@dataclass
class DayResult:
    """Per-day backtest output."""
    date: date
    trades: list[Trade] = field(default_factory=list)
    daily_pnl: float = 0.0
    max_intraday_dd: float = 0.0
    bar_log: list[dict[str, Any]] = field(default_factory=list)


def _run_single_day(
    bars: pd.DataFrame,
    threshold: float,
    cost_rt: float,
    avg_volume_per_bar: float,
) -> DayResult:
    """Run OFI strategy for a single trading day.

    Parameters
    ----------
    bars : DataFrame
        1-min bars for a single FUT contract, sorted by timestamp.
        Must contain: timestamp, open, high, low, close, volume, oi.
    threshold : float
        Absolute pressure-score threshold (e.g. 1.5).
    cost_rt : float
        Round-trip cost in index points.
    avg_volume_per_bar : float
        20-day average volume per 1-min bar (for trade intensity normalization).

    Returns
    -------
    DayResult with trades, daily P&L, and per-bar log.
    """
    if len(bars) < SIGMA_WINDOW + 5:
        return DayResult(date=bars["date"].iloc[0] if len(bars) > 0 else date.today())

    trade_date = pd.Timestamp(bars["timestamp"].iloc[0]).date()

    close = bars["close"].values.astype(np.float64)
    volume = bars["volume"].values.astype(np.float64)
    oi_arr = bars["oi"].values.astype(np.float64)
    timestamps = pd.to_datetime(bars["timestamp"].values)

    n = len(close)

    # -- Compute features --
    ofi = compute_ofi_from_bars(close, volume, oi_arr)
    _, _, vol_imb = compute_bvc_volume_imbalance(close, volume)
    trade_int = compute_trade_intensity(volume, avg_volume_per_bar)
    vpin = compute_bar_vpin(close, volume, window=min(VPIN_WINDOW, n - 1))

    pressure = compute_pressure_score(ofi, vol_imb, trade_int, vpin)

    # -- Signal generation + position management --
    trades: list[Trade] = []
    bar_log: list[dict[str, Any]] = []

    position = 0          # +1, -1, or 0
    entry_price = 0.0
    entry_bar = 0
    entry_time: datetime | None = None
    consec_above = 0      # consecutive bars above +threshold
    consec_below = 0      # consecutive bars below -threshold
    cumulative_pnl = 0.0
    peak_pnl = 0.0
    max_dd = 0.0

    half_cost = cost_rt / 2.0  # half-turn cost applied at entry and exit

    for i in range(n):
        ts = timestamps[i]
        t = ts.time()
        p = pressure[i]
        px = close[i]

        # Track consecutive threshold crossings
        if p > threshold:
            consec_above += 1
            consec_below = 0
        elif p < -threshold:
            consec_below += 1
            consec_above = 0
        else:
            consec_above = 0
            consec_below = 0

        exit_reason = ""

        # -- Exit logic (check before entry) --
        if position != 0 and entry_time is not None:
            bars_held = i - entry_bar
            unrealized = position * (px - entry_price)

            # (a) Max hold time
            if bars_held >= MAX_HOLD_BARS:
                exit_reason = "max_hold"
            # (b) Pressure reversal
            elif position == 1 and p < 0:
                exit_reason = "pressure_reversal"
            elif position == -1 and p > 0:
                exit_reason = "pressure_reversal"
            # (c) Stop-loss
            elif unrealized < -abs(entry_price) * STOP_LOSS_PCT:
                exit_reason = "stop_loss"
            # (d) Force close at market close
            elif t >= MARKET_CLOSE:
                exit_reason = "eod_close"

            if exit_reason:
                pnl = position * (px - entry_price) - cost_rt
                trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=ts.to_pydatetime(),
                    direction=position,
                    entry_price=entry_price,
                    exit_price=px,
                    pnl_points=pnl,
                    exit_reason=exit_reason,
                    bars_held=bars_held,
                ))
                cumulative_pnl += pnl
                position = 0
                entry_price = 0.0
                entry_time = None
                entry_bar = 0

        # -- Entry logic --
        in_trading_window = NO_ENTRY_OPEN <= t < NO_ENTRY_CLOSE
        if position == 0 and in_trading_window and i >= SIGMA_WINDOW:
            if consec_above >= CONFIRMATION_BARS:
                position = 1
                entry_price = px
                entry_bar = i
                entry_time = ts.to_pydatetime()
            elif consec_below >= CONFIRMATION_BARS:
                position = -1
                entry_price = px
                entry_bar = i
                entry_time = ts.to_pydatetime()

        # Track intraday drawdown
        peak_pnl = max(peak_pnl, cumulative_pnl)
        dd = peak_pnl - cumulative_pnl
        max_dd = max(max_dd, dd)

        bar_log.append({
            "date": trade_date,
            "time": t,
            "timestamp": ts,
            "close": px,
            "position": position,
            "signal": (1 if consec_above >= CONFIRMATION_BARS
                       else (-1 if consec_below >= CONFIRMATION_BARS else 0)),
            "pressure": p,
            "ofi": ofi[i],
            "vol_imb": vol_imb[i],
            "trade_int": trade_int[i],
            "vpin": vpin[i] if not np.isnan(vpin[i]) else 0.0,
            "daily_pnl": cumulative_pnl,
        })

    # Force-close any remaining position at last bar
    if position != 0 and entry_time is not None:
        px = close[-1]
        pnl = position * (px - entry_price) - cost_rt
        trades.append(Trade(
            entry_time=entry_time,
            exit_time=timestamps[-1].to_pydatetime(),
            direction=position,
            entry_price=entry_price,
            exit_price=px,
            pnl_points=pnl,
            exit_reason="eod_force",
            bars_held=n - 1 - entry_bar,
        ))
        cumulative_pnl += pnl

    return DayResult(
        date=trade_date,
        trades=trades,
        daily_pnl=cumulative_pnl,
        max_intraday_dd=max_dd,
        bar_log=bar_log,
    )


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

def _get_front_month_bars(store: Any, symbol: str, d: date) -> pd.DataFrame:
    """Load 1-min bars for the front-month (nearest expiry) FUT contract.

    Picks the expiry with the highest volume when multiple are present
    (rollover days have both near and next month active).

    Handles schema variations (some parquet files use 'date' instead of
    'timestamp') by attempting both column orderings.
    """
    d_str = d.isoformat()
    try:
        df = store.sql(
            "SELECT * FROM nfo_1min "
            "WHERE name = ? AND date = ? AND instrument_type = 'FUT' "
            "ORDER BY timestamp",
            [symbol, d_str],
        )
    except Exception:
        # Some partitions lack a 'timestamp' column; fall back to 'date' sort
        try:
            df = store.sql(
                "SELECT * FROM nfo_1min "
                "WHERE name = ? AND date = ? AND instrument_type = 'FUT'",
                [symbol, d_str],
            )
        except Exception:
            logger.warning("Could not load bars for %s on %s — skipping", symbol, d)
            return pd.DataFrame()

    if df.empty:
        return df

    # Ensure a 'timestamp' column exists (some files only have 'date')
    if "timestamp" not in df.columns:
        if "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"])
        else:
            logger.warning("No timestamp or date column for %s on %s", symbol, d)
            return pd.DataFrame()

    # On rollover days, pick the expiry with the most volume
    if df["expiry"].nunique() > 1:
        vol_by_exp = df.groupby("expiry")["volume"].sum()
        best_expiry = vol_by_exp.idxmax()
        df = df[df["expiry"] == best_expiry].copy()

    return df.sort_values("timestamp").reset_index(drop=True)


def _compute_historical_avg_volume(
    store: Any,
    symbol: str,
    dates: list[date],
    current_idx: int,
    lookback: int = 20,
) -> float:
    """Compute average volume per 1-min bar from the previous `lookback` days.

    Fully causal -- only uses dates *before* current_idx.
    """
    start = max(0, current_idx - lookback)
    history_dates = dates[start:current_idx]
    if not history_dates:
        return 1.0  # fallback

    total_volume = 0.0
    total_bars = 0
    for hd in history_dates:
        hd_str = hd.isoformat()
        vol_df = store.sql(
            "SELECT SUM(volume) as tv, COUNT(*) as nb FROM nfo_1min "
            "WHERE name = ? AND date = ? AND instrument_type = 'FUT'",
            [symbol, hd_str],
        )
        if not vol_df.empty and vol_df["tv"].iloc[0] is not None:
            total_volume += float(vol_df["tv"].iloc[0])
            total_bars += int(vol_df["nb"].iloc[0])

    return total_volume / max(total_bars, 1)


def run_intraday_backtest(
    symbol: str,
    date_range: list[date] | None = None,
    store: Any = None,
    threshold_sigma: float = DEFAULT_THRESHOLD_SIGMA,
    n_days: int = 60,
) -> pd.DataFrame:
    """Run the OFI intraday backtest over a date range.

    Parameters
    ----------
    symbol : str
        Underlying name, e.g. "NIFTY" or "BANKNIFTY".
    date_range : list[date], optional
        Specific dates to run. If None, uses last ``n_days`` available dates.
    store : MarketDataStore, optional
        If None, creates one automatically.
    threshold_sigma : float
        Pressure score threshold in sigma units (default 1.5).
    n_days : int
        Number of most recent trading days to run if date_range is None.

    Returns
    -------
    pd.DataFrame
        Per-bar log with columns: date, time, position, signal, pressure,
        ofi, vol_imb, trade_int, vpin, daily_pnl.
    """
    # Lazy import to avoid circular dependency at module level
    if store is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from quantlaxmi.data.store import MarketDataStore
        store = MarketDataStore()

    cost_rt = COST_PER_RT.get(symbol.upper(), 3.0)

    # Resolve dates
    if date_range is None:
        all_dates = store.available_dates("nfo_1min")
        if len(all_dates) > n_days:
            date_range = all_dates[-n_days:]
        else:
            date_range = all_dates

    if not date_range:
        logger.error("No dates available for %s", symbol)
        return pd.DataFrame()

    # Pre-compute sorted date list for historical volume lookups
    all_dates_sorted = sorted(date_range)

    # Results
    all_bar_logs: list[dict[str, Any]] = []
    all_trades: list[Trade] = []
    daily_summaries: list[dict[str, Any]] = []

    for idx, d in enumerate(all_dates_sorted):
        bars = _get_front_month_bars(store, symbol, d)
        if bars.empty or len(bars) < SIGMA_WINDOW + 5:
            logger.debug("Skipping %s — insufficient bars (%d)", d, len(bars))
            continue

        # Causal historical avg volume (from previous days only)
        avg_vol = _compute_historical_avg_volume(
            store, symbol, all_dates_sorted, idx, lookback=20,
        )

        day_result = _run_single_day(
            bars=bars,
            threshold=threshold_sigma,
            cost_rt=cost_rt,
            avg_volume_per_bar=avg_vol,
        )

        all_bar_logs.extend(day_result.bar_log)
        all_trades.extend(day_result.trades)
        daily_summaries.append({
            "date": d,
            "daily_pnl": day_result.daily_pnl,
            "n_trades": len(day_result.trades),
            "max_intraday_dd": day_result.max_intraday_dd,
        })

        logger.info(
            "%s | PnL: %+.1f pts | Trades: %d | MaxDD: %.1f",
            d, day_result.daily_pnl, len(day_result.trades),
            day_result.max_intraday_dd,
        )

    # Build output DataFrame
    bar_df = pd.DataFrame(all_bar_logs)
    daily_df = pd.DataFrame(daily_summaries)
    trade_df = pd.DataFrame([
        {
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl_points": t.pnl_points,
            "exit_reason": t.exit_reason,
            "bars_held": t.bars_held,
        }
        for t in all_trades
    ])

    # -- Print statistics --
    _print_statistics(daily_df, trade_df, symbol, cost_rt)

    return bar_df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _print_statistics(
    daily_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    symbol: str,
    cost_rt: float,
) -> None:
    """Print backtest statistics to stdout."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  OFI Intraday Strategy — {symbol}")
    print(sep)

    if daily_df.empty:
        print("  No trading days in backtest.")
        return

    n_days = len(daily_df)
    daily_pnl = daily_df["daily_pnl"].values

    # -- Sharpe (annualized, all-day, ddof=1) --
    # Include all days (flat days have PnL=0) per protocol
    mean_daily = np.mean(daily_pnl)
    std_daily = np.std(daily_pnl, ddof=1) if n_days > 1 else 1.0
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 1e-10 else 0.0

    # -- Total P&L --
    total_pnl = np.sum(daily_pnl)

    # -- Trade stats --
    n_trades = len(trade_df) if not trade_df.empty else 0
    trades_per_day = n_trades / max(n_days, 1)

    if n_trades > 0:
        wins = trade_df[trade_df["pnl_points"] > 0]
        losses = trade_df[trade_df["pnl_points"] <= 0]
        win_rate = len(wins) / n_trades
        avg_win = wins["pnl_points"].mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses["pnl_points"].mean()) if len(losses) > 0 else 1.0
        win_loss_ratio = avg_win / max(avg_loss, 1e-10)
        avg_bars_held = trade_df["bars_held"].mean()

        # Exit reason breakdown
        exit_counts = trade_df["exit_reason"].value_counts().to_dict()
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        win_loss_ratio = 0.0
        avg_bars_held = 0.0
        exit_counts = {}

    # -- Max intraday drawdown --
    max_dd = daily_df["max_intraday_dd"].max()

    # -- Cumulative equity drawdown --
    cum_pnl = np.cumsum(daily_pnl)
    peak = np.maximum.accumulate(cum_pnl)
    eq_dd = peak - cum_pnl
    max_eq_dd = np.max(eq_dd) if len(eq_dd) > 0 else 0.0

    print(f"  Period        : {daily_df['date'].iloc[0]} to {daily_df['date'].iloc[-1]}")
    print(f"  Trading days  : {n_days}")
    print(f"  Cost per RT   : {cost_rt:.1f} index pts")
    print(f"  {'':14s}{'':>10s}")
    print(f"  Total P&L     : {total_pnl:>+10.1f} pts")
    print(f"  Sharpe (ann.) : {sharpe:>10.2f}")
    print(f"  Win rate      : {win_rate:>10.1%}")
    print(f"  Avg win/loss  : {win_loss_ratio:>10.2f}")
    print(f"  Avg win       : {avg_win:>+10.1f} pts")
    print(f"  Avg loss      : {-avg_loss:>+10.1f} pts")
    print(f"  Total trades  : {n_trades:>10d}")
    print(f"  Trades/day    : {trades_per_day:>10.1f}")
    print(f"  Avg hold      : {avg_bars_held:>10.1f} bars")
    print(f"  Max intra DD  : {max_dd:>10.1f} pts")
    print(f"  Max equity DD : {max_eq_dd:>10.1f} pts")
    if exit_counts:
        print(f"  Exit reasons  :")
        for reason, cnt in sorted(exit_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason:20s} {cnt:5d}  ({cnt / n_trades:.0%})")
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

    print("Running OFI Intraday Backtest — NIFTY (last 60 trading days)")
    print()

    result_df = run_intraday_backtest(
        symbol="NIFTY",
        store=store,
        threshold_sigma=DEFAULT_THRESHOLD_SIGMA,
        n_days=60,
    )

    # Save results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"ofi_intraday_NIFTY_{ts_str}.parquet"
    result_df.to_parquet(out_path, index=False)
    print(f"Bar-level results saved to {out_path}")
