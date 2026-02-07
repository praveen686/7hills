"""Risk-Neutral Density Regime Strategy (RNDR) -- Options Variant.

When the RNDR density signal fires (fear overpriced), **sell OTM put credit
spreads** to directly harvest the skew premium with defined risk per trade.

Trade structure (bull put spread)
---------------------------------
- SELL ~3 % OTM put  (short leg — higher strike, higher premium)
- BUY  ~6 % OTM put  (long  leg — lower  strike, lower  premium)
- Net credit  = short_premium − long_premium
- Max risk    = strike_width  − net_credit

Exit: hold_days reached, signal decay, spot ≤ long_strike (max loss),
or end-of-data.

See ``density_strategy.py`` for the futures variant.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from strategies.s1_vrp.density import (
    DensityDayObs,
    build_density_series,
    compute_composite_signal,
    _rolling_percentile,
    DEFAULT_LOOKBACK,
    DEFAULT_ENTRY_PCTILE,
    DEFAULT_EXIT_PCTILE,
    DEFAULT_HOLD_DAYS,
    DEFAULT_PHYS_WINDOW,
)
from strategies.s9_momentum.data import is_trading_day, get_fno
from core.market.store import MarketDataStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHORT_OFFSET = 0.03        # short leg: 3 % OTM
LONG_OFFSET = 0.06         # long  leg: 6 % OTM
MIN_DTE_BUFFER = 2         # expiry >= hold_days + buffer
OPTION_COST_BPS = 20.0     # wider spread than futures (slippage on 4 legs)
MIN_OI = 100               # minimum open interest per leg
MIN_PREMIUM = 0.50         # Rs — reject illiquid options


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpreadLeg:
    """One leg of a put spread."""

    strike: float
    premium: float
    expiry: date
    dte: int
    open_interest: int


@dataclass(frozen=True)
class PutSpreadEntry:
    """Snapshot at the time a bull-put spread is opened."""

    symbol: str
    entry_date: date
    spot: float
    short_leg: SpreadLeg
    long_leg: SpreadLeg
    net_credit: float      # short_premium − long_premium
    max_risk: float        # strike_width − net_credit
    strike_width: float    # short_strike − long_strike
    entry_signal: float


@dataclass(frozen=True)
class SpreadTrade:
    """One completed round-trip spread trade."""

    symbol: str
    entry_date: date
    exit_date: date
    entry_spot: float
    exit_spot: float
    short_strike: float
    long_strike: float
    expiry: date
    net_credit: float
    exit_cost: float
    pnl_points: float       # net_credit − exit_cost − slippage
    pnl_on_risk: float      # pnl_points / max_risk
    hold_days: int
    exit_reason: str
    entry_signal: float


@dataclass
class SpreadBacktestResult:
    """Backtest output for one index (options variant)."""

    symbol: str
    daily: list[DensityDayObs]
    signals: list[float]
    trades: list[SpreadTrade]
    total_return_on_risk_pct: float = 0.0
    sharpe: float = 0.0
    max_dd_pct: float = 0.0
    win_rate: float = 0.0
    n_signals: int = 0
    avg_credit_pts: float = 0.0


# ---------------------------------------------------------------------------
# Option-chain helpers
# ---------------------------------------------------------------------------

def _get_put_chain(
    store,
    d: date,
    symbol: str,
) -> pd.DataFrame | None:
    """Return PE options for *symbol* on date *d*.

    Uses raw NSE columns: TckrSymb, FinInstrmTp, OptnTp, XpryDt,
    StrkPric, ClsPric, OpnIntrst.  Adds computed DTE column.
    Returns None when data is unavailable or empty.
    """
    try:
        fno = get_fno(store, d)
        if fno.empty:
            return None
    except Exception:
        return None

    opts = fno[
        (fno["TckrSymb"] == symbol)
        & (fno["FinInstrmTp"] == "IDO")
        & (fno["OptnTp"].str.strip() == "PE")
    ].copy()

    if opts.empty:
        return None

    # Parse expiry dates
    opts["_expiry"] = pd.to_datetime(
        opts["XpryDt"].astype(str).str.strip(),
        format="mixed",
        dayfirst=True,
    )

    # DTE = calendar days to expiry
    today_ts = pd.Timestamp(d)
    opts["DTE"] = (opts["_expiry"] - today_ts).dt.days

    # Drop expired / zero-priced
    opts = opts[opts["DTE"] >= 0]
    opts = opts[opts["ClsPric"] > 0]

    if opts.empty:
        return None

    return opts


def _select_spread_strikes(
    puts_df: pd.DataFrame,
    spot: float,
    entry_date: date,
    symbol: str,
    signal: float,
    short_offset: float = SHORT_OFFSET,
    long_offset: float = LONG_OFFSET,
    min_dte: int = 7,
    min_oi: int = MIN_OI,
    min_premium: float = MIN_PREMIUM,
) -> PutSpreadEntry | None:
    """Select short and long put strikes for a bull-put credit spread.

    Returns None when suitable strikes cannot be found.
    """
    # Pick nearest expiry with sufficient DTE
    valid_expiries = puts_df[puts_df["DTE"] >= min_dte]["_expiry"].unique()
    if len(valid_expiries) == 0:
        return None

    nearest_expiry = min(valid_expiries)
    chain = puts_df[puts_df["_expiry"] == nearest_expiry].copy()

    # Target strikes
    short_target = spot * (1.0 - short_offset)
    long_target = spot * (1.0 - long_offset)

    # Nearest available strike to each target
    chain["_short_dist"] = (chain["StrkPric"] - short_target).abs()
    chain["_long_dist"] = (chain["StrkPric"] - long_target).abs()

    short_row = chain.loc[chain["_short_dist"].idxmin()]
    long_row = chain.loc[chain["_long_dist"].idxmin()]

    short_strike = float(short_row["StrkPric"])
    long_strike = float(long_row["StrkPric"])

    # Validate: short strike must be above long strike
    if short_strike <= long_strike:
        return None

    short_premium = float(short_row["ClsPric"])
    long_premium = float(long_row["ClsPric"])
    short_oi = int(short_row["OpnIntrst"])
    long_oi = int(long_row["OpnIntrst"])

    # Liquidity filters
    if short_oi < min_oi or long_oi < min_oi:
        return None
    if short_premium < min_premium or long_premium < min_premium:
        return None

    net_credit = short_premium - long_premium
    if net_credit <= 0:
        return None

    strike_width = short_strike - long_strike
    max_risk = strike_width - net_credit

    expiry_date = pd.Timestamp(nearest_expiry).date()
    dte = int(short_row["DTE"])

    return PutSpreadEntry(
        symbol=symbol,
        entry_date=entry_date,
        spot=spot,
        short_leg=SpreadLeg(
            strike=short_strike,
            premium=short_premium,
            expiry=expiry_date,
            dte=dte,
            open_interest=short_oi,
        ),
        long_leg=SpreadLeg(
            strike=long_strike,
            premium=long_premium,
            expiry=expiry_date,
            dte=dte,
            open_interest=long_oi,
        ),
        net_credit=net_credit,
        max_risk=max_risk,
        strike_width=strike_width,
        entry_signal=signal,
    )


def _mark_to_market_spread(
    store,
    exit_date: date,
    symbol: str,
    short_strike: float,
    long_strike: float,
    expiry: date,
    exit_spot: float,
) -> tuple[float, float] | None:
    """Get closing prices for both legs on *exit_date*.

    If the option has expired (exit_date >= expiry), use intrinsic value
    (European cash-settled: max(K − S, 0)).

    Returns (short_exit_premium, long_exit_premium) or None if prices
    cannot be determined.
    """
    # Expired → intrinsic
    if exit_date >= expiry:
        short_val = max(short_strike - exit_spot, 0.0)
        long_val = max(long_strike - exit_spot, 0.0)
        return short_val, long_val

    # Try to look up market prices for the SAME expiry
    try:
        fno = get_fno(store, exit_date)
        if fno.empty:
            short_val = max(short_strike - exit_spot, 0.0)
            long_val = max(long_strike - exit_spot, 0.0)
            return short_val, long_val
    except Exception:
        # Fallback to intrinsic
        short_val = max(short_strike - exit_spot, 0.0)
        long_val = max(long_strike - exit_spot, 0.0)
        return short_val, long_val

    opts = fno[
        (fno["TckrSymb"] == symbol)
        & (fno["FinInstrmTp"] == "IDO")
        & (fno["OptnTp"].str.strip() == "PE")
    ].copy()

    if opts.empty:
        short_val = max(short_strike - exit_spot, 0.0)
        long_val = max(long_strike - exit_spot, 0.0)
        return short_val, long_val

    # Filter to the SAME expiry we entered on — critical to avoid
    # matching strikes from a different expiry with wrong premiums
    opts["_expiry"] = pd.to_datetime(
        opts["XpryDt"].astype(str).str.strip(),
        format="mixed", dayfirst=True,
    )
    expiry_ts = pd.Timestamp(expiry)
    same_exp = opts[opts["_expiry"] == expiry_ts]
    if same_exp.empty:
        # Expiry no longer listed (may have already settled)
        short_val = max(short_strike - exit_spot, 0.0)
        long_val = max(long_strike - exit_spot, 0.0)
        return short_val, long_val

    short_rows = same_exp[
        (same_exp["StrkPric"] == short_strike)
        & (same_exp["ClsPric"] > 0)
    ]
    long_rows = same_exp[
        (same_exp["StrkPric"] == long_strike)
        & (same_exp["ClsPric"] > 0)
    ]

    if short_rows.empty or long_rows.empty:
        # One or both strikes missing → intrinsic fallback
        short_val = max(short_strike - exit_spot, 0.0)
        long_val = max(long_strike - exit_spot, 0.0)
        return short_val, long_val

    return float(short_rows.iloc[0]["ClsPric"]), float(long_rows.iloc[0]["ClsPric"])


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run_density_options_backtest(
    series: list[DensityDayObs],
    store,
    lookback: int = DEFAULT_LOOKBACK,
    entry_pctile: float = DEFAULT_ENTRY_PCTILE,
    exit_pctile: float = DEFAULT_EXIT_PCTILE,
    hold_days: int = DEFAULT_HOLD_DAYS,
    cost_bps: float = OPTION_COST_BPS,
    short_offset: float = SHORT_OFFSET,
    long_offset: float = LONG_OFFSET,
    symbol: str = "BANKNIFTY",
) -> SpreadBacktestResult:
    """Run the density-regime options backtest (bull put spreads).

    Entry: composite signal percentile >= entry_pctile AND signal > 0.
    Exit : max hold_days, signal decay, spot <= long_strike, or end-of-data.
    """
    signals = compute_composite_signal(series, lookback)
    n = len(series)
    cost_frac = cost_bps / 10_000
    trades: list[SpreadTrade] = []
    n_signals = 0

    # Active position state
    active: PutSpreadEntry | None = None
    entry_idx: int = 0
    pending_signal_idx: int = -1  # T+1 execution: signal at close, enter next day

    for i in range(lookback, n):
        sig_pctile = _rolling_percentile(signals, i, lookback)
        obs = series[i]

        # Execute pending entry from yesterday's signal
        if pending_signal_idx >= 0 and active is None:
            puts = _get_put_chain(store, obs.date, symbol)
            if puts is not None:
                min_dte = hold_days + MIN_DTE_BUFFER
                entry = _select_spread_strikes(
                    puts, obs.spot, obs.date, symbol,
                    signals[pending_signal_idx],
                    short_offset=short_offset,
                    long_offset=long_offset,
                    min_dte=min_dte,
                )
                if entry is not None:
                    active = entry
                    entry_idx = i
            pending_signal_idx = -1

        if active is None:
            # --- Look for entry (signal fires, execution deferred to T+1) ---
            if sig_pctile >= entry_pctile and signals[i] > 0:
                n_signals += 1
                pending_signal_idx = i  # will try to enter next bar
        else:
            # --- Check exit conditions ---
            days_held = i - entry_idx
            exit_reason: str | None = None

            if days_held >= hold_days:
                exit_reason = "max_hold"
            elif sig_pctile < exit_pctile:
                exit_reason = "signal_decay"
            elif obs.spot <= active.long_leg.strike:
                exit_reason = "max_loss"

            if exit_reason is not None:
                trade = _close_trade(
                    active, obs.date, obs.spot, store, days_held,
                    exit_reason, cost_frac,
                )
                trades.append(trade)
                active = None

    # Close open position at end-of-data
    if active is not None:
        obs = series[-1]
        days_held = len(series) - 1 - entry_idx
        trade = _close_trade(
            active, obs.date, obs.spot, store, days_held,
            "end_of_data", cost_frac,
        )
        trades.append(trade)

    # --- Metrics ---
    n_total_days = n - lookback  # total tradeable days
    result = _compute_metrics(symbol, series, signals, trades, n_signals, n_total_days)
    return result


def _close_trade(
    entry: PutSpreadEntry,
    exit_date: date,
    exit_spot: float,
    store,
    days_held: int,
    exit_reason: str,
    cost_frac: float,
) -> SpreadTrade:
    """Close an open spread and compute P&L."""
    mtm = _mark_to_market_spread(
        store, exit_date, entry.symbol,
        entry.short_leg.strike, entry.long_leg.strike,
        entry.short_leg.expiry, exit_spot,
    )

    if mtm is not None:
        short_exit, long_exit = mtm
        # To close: buy back short, sell back long
        exit_cost = short_exit - long_exit  # debit to close
    else:
        short_exit = 0.0
        long_exit = 0.0
        exit_cost = 0.0

    # Cap exit cost at strike_width — defined-risk spread cannot lose
    # more than the width minus credit received
    exit_cost = min(max(exit_cost, 0.0), entry.strike_width)

    # Slippage on total premium exchanged (entry + exit)
    total_premium = (
        entry.short_leg.premium + entry.long_leg.premium
        + abs(short_exit) + abs(long_exit)
    )
    slippage = total_premium * cost_frac

    pnl_points = entry.net_credit - exit_cost - slippage
    max_risk = entry.max_risk if entry.max_risk > 0 else entry.strike_width
    pnl_on_risk = pnl_points / max_risk if max_risk > 0 else 0.0

    return SpreadTrade(
        symbol=entry.symbol,
        entry_date=entry.entry_date,
        exit_date=exit_date,
        entry_spot=entry.spot,
        exit_spot=exit_spot,
        short_strike=entry.short_leg.strike,
        long_strike=entry.long_leg.strike,
        expiry=entry.short_leg.expiry,
        net_credit=entry.net_credit,
        exit_cost=exit_cost,
        pnl_points=pnl_points,
        pnl_on_risk=pnl_on_risk,
        hold_days=days_held,
        exit_reason=exit_reason,
        entry_signal=entry.entry_signal,
    )


def _compute_metrics(
    symbol: str,
    series: list[DensityDayObs],
    signals: list[float],
    trades: list[SpreadTrade],
    n_signals: int,
    n_total_days: int = 0,
) -> SpreadBacktestResult:
    """Compute backtest summary metrics from a list of trades."""
    if trades:
        # Equity curve (compound return on risk)
        equity = 1.0
        eq_curve = [1.0]
        for t in trades:
            equity *= (1 + t.pnl_on_risk)
            eq_curve.append(equity)

        total_ror = (equity - 1) * 100

        # Sharpe from ALL daily returns (including flat days, ddof=1)
        # Allocate each trade's P&L to its exit day; flat days = 0
        n_days = max(n_total_days, len(series) - 1)
        n_trade_days = sum(max(t.hold_days, 1) for t in trades)
        n_flat_days = max(n_days - n_trade_days, 0)
        # Build daily P&L: trade P&L spread across hold days + zeros for flat days
        daily_pnl: list[float] = []
        for t in trades:
            hold = max(t.hold_days, 1)
            daily_ret = t.pnl_on_risk / hold
            daily_pnl.extend([daily_ret] * hold)
        daily_pnl.extend([0.0] * n_flat_days)

        sharpe = 0.0
        if len(daily_pnl) > 1:
            pnl_arr = np.array(daily_pnl)
            std = np.std(pnl_arr, ddof=1)
            if std > 0:
                sharpe = float(np.mean(pnl_arr) / std * math.sqrt(252))

        # Max drawdown
        peak = 1.0
        max_dd = 0.0
        for eq in eq_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        win_rate = sum(1 for t in trades if t.pnl_on_risk > 0) / len(trades)
        avg_credit_pts = float(np.mean([t.net_credit for t in trades]))
    else:
        total_ror = sharpe = max_dd = win_rate = avg_credit_pts = 0.0

    return SpreadBacktestResult(
        symbol=symbol,
        daily=series,
        signals=signals,
        trades=trades,
        total_return_on_risk_pct=total_ror,
        sharpe=sharpe,
        max_dd_pct=max_dd * 100,
        win_rate=win_rate,
        n_signals=n_signals,
        avg_credit_pts=avg_credit_pts,
    )


# ---------------------------------------------------------------------------
# Multi-index wrapper
# ---------------------------------------------------------------------------

def run_multi_index_options_backtest(
    store,
    start: date,
    end: date,
    symbols: list[str] | None = None,
    lookback: int = DEFAULT_LOOKBACK,
    entry_pctile: float = DEFAULT_ENTRY_PCTILE,
    exit_pctile: float = DEFAULT_EXIT_PCTILE,
    hold_days: int = DEFAULT_HOLD_DAYS,
    cost_bps: float = OPTION_COST_BPS,
    phys_window: int = DEFAULT_PHYS_WINDOW,
    short_offset: float = SHORT_OFFSET,
    long_offset: float = LONG_OFFSET,
) -> dict[str, SpreadBacktestResult]:
    """Run density-options backtest for multiple indices."""
    if symbols is None:
        symbols = ["BANKNIFTY", "MIDCPNIFTY", "FINNIFTY"]

    results: dict[str, SpreadBacktestResult] = {}

    for sym in symbols:
        print(f"\n  {sym}: building density series...", end="", flush=True)
        series = build_density_series(store, start, end, sym, phys_window)
        print(f" {len(series)} days", end="", flush=True)

        if len(series) < lookback + 10:
            print(" (too few days, skipping)")
            continue

        result = run_density_options_backtest(
            series,
            store,
            lookback=lookback,
            entry_pctile=entry_pctile,
            exit_pctile=exit_pctile,
            hold_days=hold_days,
            cost_bps=cost_bps,
            short_offset=short_offset,
            long_offset=long_offset,
            symbol=sym,
        )
        results[sym] = result
        print(
            f" → {len(result.trades)} trades, "
            f"RoR {result.total_return_on_risk_pct:+.2f}%, "
            f"Sharpe {result.sharpe:.2f}, "
            f"WR {result.win_rate:.0%}"
        )

    return results
