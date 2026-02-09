"""Volatility Skew Mean-Reversion Strategy.

Trade the 25-delta put-call implied volatility skew using risk reversals.

Core thesis:
    Institutional hedging demand creates inelastic put-buying that pushes OTM put
    IV disproportionately above OTM call IV.  This "fear premium" overshoots and
    mean-reverts as the hedging wave ebbs.

Signal chain:
    1. Compute 25-delta skew = IV_25d_put - IV_25d_call  (via GPU IV engine)
    2. Z-score the skew over a 30-day rolling window
    3. Sell skew when z > 2.0 (risk reversal: sell OTM put, buy OTM call)
    4. Buy skew when z < -1.0 (reverse risk reversal)
    5. Exit when z crosses 0, after 5 days, or stop-loss at -3% of notional
    6. No trade when India VIX > 25 (regime filter: elevated fear persists)

P&L model:
    Mark-to-market using actual DuckDB nse_fo_bhavcopy closing prices.
    Cost: 3 index points per leg for NIFTY (6 pts round-trip per side).

Author: AlphaForge
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from data.store import MarketDataStore
from core.pricing.iv_engine import compute_chain_iv, OptionChainIV
from strategies.s9_momentum.data import is_trading_day, get_fno

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SKEW_LOOKBACK: int = 30           # rolling window for z-score (trading days)
ENTRY_Z_SELL: float = 2.0         # sell skew when z > this
ENTRY_Z_BUY: float = -1.0         # buy skew when z < this
EXIT_Z_CROSS: float = 0.0         # exit when z reverts past this
MAX_HOLD_DAYS: int = 5            # maximum holding period (trading days)
STOP_LOSS_PCT: float = 0.03       # stop-loss: 3% of notional
VIX_CEILING: float = 25.0         # no new trades when India VIX > this
MIN_DTE: int = 3                  # minimum days-to-expiry for strike selection
COST_PER_LEG_PTS: float = 3.0     # transaction cost per leg in index points (NIFTY)
MIN_OI: int = 500                 # minimum open interest per leg
TARGET_DELTA: float = 0.25        # 25-delta strikes
OTM_PCT_FALLBACK: float = 0.035   # ~3.5% OTM if delta unavailable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SkewObservation:
    """Single-day skew measurement."""
    date: date
    spot: float
    atm_iv: float                  # annualized
    skew_25d: float                # IV_25d_put - IV_25d_call
    z_score: float                 # standardized skew
    vix: float                     # India VIX close
    term_slope: float              # (next_month_ATM - curr_month_ATM) / curr_month_ATM
    nearest_expiry: str


@dataclass
class RiskReversalLeg:
    """One leg of a risk reversal."""
    strike: float
    premium: float                 # closing price in points
    option_type: str               # "CE" or "PE"
    expiry: str
    dte: int
    open_interest: int


@dataclass
class SkewTrade:
    """One completed risk-reversal trade."""
    entry_date: date
    exit_date: date
    direction: Literal["sell_skew", "buy_skew"]
    entry_spot: float
    exit_spot: float
    put_leg: RiskReversalLeg       # the OTM put leg
    call_leg: RiskReversalLeg      # the OTM call leg
    entry_net_premium: float       # net premium received (+) or paid (-)
    exit_net_premium: float        # cost to unwind
    pnl_points: float              # after costs
    pnl_pct: float                 # as fraction of notional (spot)
    hold_days: int
    exit_reason: str
    entry_z: float
    exit_z: float


@dataclass
class SkewBacktestResult:
    """Full backtest output."""
    symbol: str
    observations: list[SkewObservation]
    trades: list[SkewTrade]
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe: float = 0.0
    max_dd_pct: float = 0.0
    win_rate: float = 0.0
    avg_hold_days: float = 0.0
    n_trades: int = 0
    avg_skew: float = 0.0
    avg_vix: float = 0.0


# ---------------------------------------------------------------------------
# VIX loader
# ---------------------------------------------------------------------------

def _load_vix_series(store: MarketDataStore) -> dict[str, float]:
    """Load India VIX closing values keyed by ISO date string.

    Returns {date_str: vix_close} for all available dates.
    """
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
        logger.warning("Failed to load India VIX: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Skew computation
# ---------------------------------------------------------------------------

def _compute_daily_skew(
    store: MarketDataStore,
    d: date,
    symbol: str = "NIFTY",
) -> tuple[float, float, float, float, str] | None:
    """Compute 25-delta skew, ATM IV, term slope, and spot for a single day.

    Returns (skew_25d, atm_iv, term_slope, spot, nearest_expiry) or None.

    Uses the GPU IV engine via compute_chain_iv for accurate delta-based
    strike selection.
    """
    try:
        fno = get_fno(store, d)
        if fno.empty:
            return None
    except Exception:
        return None

    # Filter to index options for the target symbol
    opts = fno[
        (fno["TckrSymb"] == symbol)
        & (fno["FinInstrmTp"] == "IDO")
        & (fno["OptnTp"].isin(["CE", "PE"]))
    ].copy()

    if opts.empty or "UndrlygPric" not in opts.columns:
        return None

    spot = float(opts["UndrlygPric"].iloc[0])
    if spot <= 0:
        return None

    # Compute IV + delta for the full chain via GPU engine
    chain_iv = compute_chain_iv(opts)
    if math.isnan(chain_iv.atm_iv):
        return None

    # 25-delta skew from the IV engine (already computed)
    skew_25d = chain_iv.iv_skew_25d

    # If IV engine skew is NaN, try ATM-OTM proxy
    if math.isnan(skew_25d):
        skew_25d = _fallback_skew(chain_iv, spot)
        if math.isnan(skew_25d):
            return None

    # Term structure slope: (next_month_ATM - curr_month_ATM) / curr_month_ATM
    term_slope = _compute_term_slope(chain_iv, spot, d)

    return (skew_25d, chain_iv.atm_iv, term_slope, spot, chain_iv.nearest_expiry)


def _fallback_skew(chain_iv: OptionChainIV, spot: float) -> float:
    """Compute a skew proxy when 25-delta is unavailable.

    Uses ATM IV minus ~3.5% OTM put IV as a simpler skew measure.
    """
    df = chain_iv.df.dropna(subset=["IV"])
    if df.empty:
        return float("nan")

    # Nearest expiry only
    if chain_iv.nearest_expiry:
        nearest = df[df["XpryDt"].astype(str).str.strip() == chain_iv.nearest_expiry]
        if not nearest.empty:
            df = nearest

    calls = df[df["OptnTp"].str.strip() == "CE"].copy()
    puts = df[df["OptnTp"].str.strip() == "PE"].copy()

    if calls.empty or puts.empty:
        return float("nan")

    # ATM: nearest to spot
    calls["_dist"] = abs(calls["StrkPric"] - spot)
    puts["_dist"] = abs(puts["StrkPric"] - spot)

    atm_call_iv = float(calls.loc[calls["_dist"].idxmin(), "IV"])

    # OTM put: ~3.5% below spot
    target_otm = spot * (1.0 - OTM_PCT_FALLBACK)
    puts["_otm_dist"] = abs(puts["StrkPric"] - target_otm)
    otm_put_iv = float(puts.loc[puts["_otm_dist"].idxmin(), "IV"])

    if math.isnan(atm_call_iv) or math.isnan(otm_put_iv):
        return float("nan")

    return otm_put_iv - atm_call_iv


def _compute_term_slope(
    chain_iv: OptionChainIV,
    spot: float,
    d: date,
) -> float:
    """Compute term structure slope: (next_month - curr_month) / curr_month ATM IV."""
    df = chain_iv.df.dropna(subset=["IV"])
    if df.empty:
        return 0.0

    # Parse expiries and find the two nearest monthly expiries
    df = df.copy()
    df["_expiry_dt"] = pd.to_datetime(df["XpryDt"].astype(str).str.strip(), format="mixed")
    expiries = sorted(df["_expiry_dt"].unique())

    if len(expiries) < 2:
        return 0.0

    # Get ATM IV for the two nearest expiries
    atm_ivs = []
    for exp in expiries[:2]:
        exp_df = df[df["_expiry_dt"] == exp]
        exp_df = exp_df.copy()
        exp_df["_dist"] = abs(exp_df["StrkPric"] - spot)
        nearest_strike_idx = exp_df["_dist"].idxmin()
        atm_strike = exp_df.loc[nearest_strike_idx, "StrkPric"]
        atm_rows = exp_df[exp_df["StrkPric"] == atm_strike]
        if not atm_rows.empty:
            mean_iv = float(atm_rows["IV"].mean())
            if not math.isnan(mean_iv):
                atm_ivs.append(mean_iv)

    if len(atm_ivs) < 2 or atm_ivs[0] <= 0:
        return 0.0

    return (atm_ivs[1] - atm_ivs[0]) / atm_ivs[0]


# ---------------------------------------------------------------------------
# Build observation series
# ---------------------------------------------------------------------------

def build_skew_series(
    store: MarketDataStore,
    start: date,
    end: date,
    symbol: str = "NIFTY",
    lookback: int = SKEW_LOOKBACK,
) -> list[SkewObservation]:
    """Build the daily skew observation series with z-scores.

    Iterates from (start - lookback buffer) to end so that z-scores are
    available from `start` onwards.
    """
    vix_map = _load_vix_series(store)

    # Start collecting skew data with enough buffer for z-score warm-up
    buffer_start = start - timedelta(days=int(lookback * 2.0))
    raw_skews: list[dict] = []

    d = buffer_start
    while d <= end:
        if not is_trading_day(d):
            d += timedelta(days=1)
            continue

        result = _compute_daily_skew(store, d, symbol)
        if result is not None:
            skew_25d, atm_iv, term_slope, spot, nearest_expiry = result
            vix = vix_map.get(d.isoformat(), float("nan"))
            raw_skews.append({
                "date": d,
                "spot": spot,
                "atm_iv": atm_iv,
                "skew_25d": skew_25d,
                "vix": vix,
                "term_slope": term_slope,
                "nearest_expiry": nearest_expiry,
            })

            if len(raw_skews) % 10 == 0:
                logger.info(
                    "Skew computed for %d days (latest: %s, skew=%.4f, IV=%.1f%%)",
                    len(raw_skews), d, skew_25d, atm_iv * 100,
                )

        d += timedelta(days=1)

    if len(raw_skews) < lookback + 1:
        logger.warning(
            "Insufficient skew data: %d days (need %d+1)",
            len(raw_skews), lookback,
        )
        return []

    # Compute rolling z-scores
    skew_values = np.array([r["skew_25d"] for r in raw_skews])
    observations: list[SkewObservation] = []

    for i in range(lookback, len(raw_skews)):
        window = skew_values[i - lookback : i]
        mu = float(np.mean(window))
        sigma = float(np.std(window, ddof=1))
        z = (skew_values[i] - mu) / sigma if sigma > 1e-8 else 0.0

        r = raw_skews[i]
        if r["date"] >= start:
            observations.append(SkewObservation(
                date=r["date"],
                spot=r["spot"],
                atm_iv=r["atm_iv"],
                skew_25d=r["skew_25d"],
                z_score=z,
                vix=r["vix"],
                term_slope=r["term_slope"],
                nearest_expiry=r["nearest_expiry"],
            ))

    logger.info("Built %d skew observations from %s to %s", len(observations), start, end)
    return observations


# ---------------------------------------------------------------------------
# Strike selection for risk reversals
# ---------------------------------------------------------------------------

def _select_risk_reversal_legs(
    store: MarketDataStore,
    d: date,
    symbol: str,
    spot: float,
    direction: Literal["sell_skew", "buy_skew"],
    min_dte: int = MIN_DTE,
    min_oi: int = MIN_OI,
) -> tuple[RiskReversalLeg, RiskReversalLeg] | None:
    """Select put and call legs for a risk reversal on date `d`.

    For "sell_skew": sell OTM put, buy OTM call
    For "buy_skew": buy OTM put, sell OTM call

    Returns (put_leg, call_leg) or None if suitable strikes not found.
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
        & (fno["OptnTp"].isin(["CE", "PE"]))
    ].copy()

    if opts.empty:
        return None

    # Parse expiry, compute DTE
    opts["_expiry_dt"] = pd.to_datetime(
        opts["XpryDt"].astype(str).str.strip(), format="mixed"
    )
    opts["_dte"] = (opts["_expiry_dt"] - pd.Timestamp(d)).dt.days
    opts = opts[opts["_dte"] >= min_dte]
    opts = opts[opts["ClsPric"] > 0]

    if opts.empty:
        return None

    # Pick nearest weekly expiry with >= min_dte
    nearest_expiry = opts["_expiry_dt"].min()
    chain = opts[opts["_expiry_dt"] == nearest_expiry].copy()

    calls = chain[chain["OptnTp"].str.strip() == "CE"].copy()
    puts = chain[chain["OptnTp"].str.strip() == "PE"].copy()

    if calls.empty or puts.empty:
        return None

    # Try delta-based selection first (if IV engine already computed delta)
    # Otherwise fall back to %-OTM heuristic
    put_leg = _select_by_delta_or_pct(puts, spot, "PE", nearest_expiry, min_oi)
    call_leg = _select_by_delta_or_pct(calls, spot, "CE", nearest_expiry, min_oi)

    if put_leg is None or call_leg is None:
        return None

    return (put_leg, call_leg)


def _select_by_delta_or_pct(
    df: pd.DataFrame,
    spot: float,
    opt_type: str,
    expiry_dt: pd.Timestamp,
    min_oi: int,
) -> RiskReversalLeg | None:
    """Select a ~25-delta strike using delta if available, else %-OTM fallback.

    For CE: 25-delta call is ~3.5% OTM above spot
    For PE: 25-delta put is ~3.5% OTM below spot
    """
    # Compute IV+delta on-the-fly for this subset if not already present
    # We'll use the IV engine's chain computation to get delta
    has_delta = "DELTA" in df.columns and not df["DELTA"].isna().all()

    if has_delta:
        df = df.copy()
        if opt_type == "CE":
            # 25-delta call: delta closest to 0.25
            df["_delta_dist"] = abs(df["DELTA"] - TARGET_DELTA)
        else:
            # 25-delta put: delta closest to -0.25
            df["_delta_dist"] = abs(df["DELTA"] + TARGET_DELTA)

        valid = df[df["_delta_dist"].notna() & (df["OpnIntrst"] >= min_oi)]
        if not valid.empty:
            best_idx = valid["_delta_dist"].idxmin()
            row = valid.loc[best_idx]
            return RiskReversalLeg(
                strike=float(row["StrkPric"]),
                premium=float(row["ClsPric"]),
                option_type=opt_type,
                expiry=str(expiry_dt.date()),
                dte=int(row["_dte"]),
                open_interest=int(row["OpnIntrst"]),
            )

    # Fallback: %-OTM heuristic
    if opt_type == "CE":
        target = spot * (1.0 + OTM_PCT_FALLBACK)
    else:
        target = spot * (1.0 - OTM_PCT_FALLBACK)

    df = df.copy()
    df["_strike_dist"] = abs(df["StrkPric"] - target)
    valid = df[df["OpnIntrst"] >= min_oi]
    if valid.empty:
        # Relax OI filter
        valid = df[df["OpnIntrst"] > 0]
    if valid.empty:
        return None

    best_idx = valid["_strike_dist"].idxmin()
    row = valid.loc[best_idx]
    return RiskReversalLeg(
        strike=float(row["StrkPric"]),
        premium=float(row["ClsPric"]),
        option_type=opt_type,
        expiry=str(expiry_dt.date()),
        dte=int(row["_dte"]),
        open_interest=int(row["OpnIntrst"]),
    )


# ---------------------------------------------------------------------------
# Mark-to-market at exit
# ---------------------------------------------------------------------------

def _mark_to_market_leg(
    store: MarketDataStore,
    exit_date: date,
    symbol: str,
    strike: float,
    opt_type: str,
    expiry_str: str,
    exit_spot: float,
) -> float:
    """Get the closing price for a specific option leg on exit_date.

    Falls back to intrinsic value if market price is unavailable.
    """
    expiry_date = date.fromisoformat(expiry_str)

    # If expired, use intrinsic (cash-settled, European)
    if exit_date >= expiry_date:
        if opt_type == "CE":
            return max(exit_spot - strike, 0.0)
        else:
            return max(strike - exit_spot, 0.0)

    # Look up market price
    try:
        fno = get_fno(store, exit_date)
        if fno.empty:
            return _intrinsic(strike, opt_type, exit_spot)
    except Exception:
        return _intrinsic(strike, opt_type, exit_spot)

    opts = fno[
        (fno["TckrSymb"] == symbol)
        & (fno["FinInstrmTp"] == "IDO")
        & (fno["OptnTp"].str.strip() == opt_type)
    ].copy()

    if opts.empty:
        return _intrinsic(strike, opt_type, exit_spot)

    # Filter to same expiry
    opts["_expiry_dt"] = pd.to_datetime(
        opts["XpryDt"].astype(str).str.strip(), format="mixed"
    )
    same_exp = opts[opts["_expiry_dt"] == pd.Timestamp(expiry_date)]

    if same_exp.empty:
        return _intrinsic(strike, opt_type, exit_spot)

    strike_match = same_exp[
        (same_exp["StrkPric"] == strike) & (same_exp["ClsPric"] > 0)
    ]

    if strike_match.empty:
        return _intrinsic(strike, opt_type, exit_spot)

    return float(strike_match.iloc[0]["ClsPric"])


def _intrinsic(strike: float, opt_type: str, spot: float) -> float:
    """Intrinsic value fallback."""
    if opt_type == "CE":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    symbol: str = "NIFTY",
    start_date: date | None = None,
    end_date: date | None = None,
    store: MarketDataStore | None = None,
    lookback: int = SKEW_LOOKBACK,
    entry_z_sell: float = ENTRY_Z_SELL,
    entry_z_buy: float = ENTRY_Z_BUY,
    exit_z_cross: float = EXIT_Z_CROSS,
    max_hold: int = MAX_HOLD_DAYS,
    stop_loss_pct: float = STOP_LOSS_PCT,
    vix_ceiling: float = VIX_CEILING,
    cost_per_leg: float = COST_PER_LEG_PTS,
) -> SkewBacktestResult:
    """Run the volatility skew mean-reversion backtest.

    Parameters
    ----------
    symbol : str
        Index name ("NIFTY", "BANKNIFTY", etc.)
    start_date, end_date : date
        Backtest date range.  Defaults to full available data range.
    store : MarketDataStore
        DuckDB market data store.  Creates a new one if None.
    lookback : int
        Rolling window for z-score (trading days).
    entry_z_sell : float
        Enter sell-skew (risk reversal) when z > this.
    entry_z_buy : float
        Enter buy-skew (reverse RR) when z < this.
    exit_z_cross : float
        Exit when z reverts past this level.
    max_hold : int
        Maximum holding period (trading days).
    stop_loss_pct : float
        Stop loss as fraction of notional.
    vix_ceiling : float
        No new trades when VIX > this.
    cost_per_leg : float
        Transaction cost per leg in index points.

    Returns
    -------
    SkewBacktestResult with trades, statistics, and observation series.
    """
    own_store = store is None
    if store is None:
        store = MarketDataStore()

    try:
        return _run_backtest_impl(
            symbol, start_date, end_date, store, lookback,
            entry_z_sell, entry_z_buy, exit_z_cross, max_hold,
            stop_loss_pct, vix_ceiling, cost_per_leg,
        )
    finally:
        if own_store:
            store.close()


def _run_backtest_impl(
    symbol: str,
    start_date: date | None,
    end_date: date | None,
    store: MarketDataStore,
    lookback: int,
    entry_z_sell: float,
    entry_z_buy: float,
    exit_z_cross: float,
    max_hold: int,
    stop_loss_pct: float,
    vix_ceiling: float,
    cost_per_leg: float,
) -> SkewBacktestResult:
    """Core backtest implementation."""
    # Determine date range from available data if not specified
    avail = store.available_dates("nse_fo_bhavcopy")
    if not avail:
        logger.error("No nse_fo_bhavcopy data available")
        return _empty_result(symbol)

    if start_date is None:
        start_date = avail[0]
    if end_date is None:
        end_date = avail[-1]

    logger.info(
        "Building skew series for %s from %s to %s (lookback=%d)...",
        symbol, start_date, end_date, lookback,
    )

    # Build observation series (skew z-scores, VIX, term structure)
    observations = build_skew_series(store, start_date, end_date, symbol, lookback)
    if len(observations) < 5:
        logger.warning("Too few observations (%d), aborting", len(observations))
        return _empty_result(symbol)

    # ---- Event-driven trade simulation ----
    # Causal: signal at close of day i, execute at close of day i+1 (T+1 lag)
    trades: list[SkewTrade] = []
    daily_pnl: list[float] = []

    # Active position state
    active_entry: dict | None = None
    pending_signal: dict | None = None  # signal fires at close, execute next day

    for i, obs in enumerate(observations):
        # -- Execute pending entry from yesterday --
        if pending_signal is not None and active_entry is None:
            direction = pending_signal["direction"]
            legs = _select_risk_reversal_legs(
                store, obs.date, symbol, obs.spot, direction,
            )
            if legs is not None:
                put_leg, call_leg = legs
                # Net premium for sell-skew: +put premium - call premium
                # (sell put, buy call)
                if direction == "sell_skew":
                    net_prem = put_leg.premium - call_leg.premium
                else:
                    # buy_skew: buy put, sell call
                    net_prem = call_leg.premium - put_leg.premium

                active_entry = {
                    "entry_date": obs.date,
                    "entry_spot": obs.spot,
                    "direction": direction,
                    "put_leg": put_leg,
                    "call_leg": call_leg,
                    "entry_net_premium": net_prem,
                    "entry_z": pending_signal["z"],
                    "entry_idx": i,
                }
            pending_signal = None

        # -- Check exit conditions for active trade --
        if active_entry is not None:
            bars_held = i - active_entry["entry_idx"]
            should_exit = False
            exit_reason = ""

            # 1. Z-score crosses zero (mean-reversion complete)
            if active_entry["direction"] == "sell_skew" and obs.z_score <= exit_z_cross:
                should_exit = True
                exit_reason = "z_revert"
            elif active_entry["direction"] == "buy_skew" and obs.z_score >= exit_z_cross:
                should_exit = True
                exit_reason = "z_revert"

            # 2. Max holding period
            if bars_held >= max_hold:
                should_exit = True
                exit_reason = "max_hold"

            # 3. Stop-loss on notional
            # Approximate P&L from spot move (simplified for stop check)
            if bars_held > 0:
                put_exit = _mark_to_market_leg(
                    store, obs.date, symbol,
                    active_entry["put_leg"].strike,
                    "PE", active_entry["put_leg"].expiry, obs.spot,
                )
                call_exit = _mark_to_market_leg(
                    store, obs.date, symbol,
                    active_entry["call_leg"].strike,
                    "CE", active_entry["call_leg"].expiry, obs.spot,
                )

                if active_entry["direction"] == "sell_skew":
                    # We are short put, long call
                    # MTM = entry_net_prem - (put_exit_val - call_exit_val)
                    mtm = active_entry["entry_net_premium"] - (put_exit - call_exit)
                else:
                    # buy_skew: long put, short call
                    mtm = active_entry["entry_net_premium"] - (call_exit - put_exit)

                unrealized_pnl_pct = mtm / active_entry["entry_spot"]
                if unrealized_pnl_pct < -stop_loss_pct:
                    should_exit = True
                    exit_reason = "stop_loss"

            # 4. End of backtest
            if i == len(observations) - 1 and not should_exit:
                should_exit = True
                exit_reason = "end_of_data"

            if should_exit and bars_held > 0:
                trade = _close_position(
                    active_entry, obs, store, symbol,
                    bars_held, exit_reason, cost_per_leg,
                )
                trades.append(trade)
                daily_pnl.append(trade.pnl_pct)
                active_entry = None
            elif should_exit and bars_held == 0:
                # Same-day exit: just cancel the entry
                active_entry = None
                daily_pnl.append(0.0)
            else:
                daily_pnl.append(0.0)
            continue

        # -- Check entry conditions (signal fires at close, execute T+1) --
        vix_ok = math.isnan(obs.vix) or obs.vix <= vix_ceiling

        if vix_ok and obs.z_score > entry_z_sell:
            pending_signal = {"direction": "sell_skew", "z": obs.z_score}
            daily_pnl.append(0.0)
        elif vix_ok and obs.z_score < entry_z_buy:
            pending_signal = {"direction": "buy_skew", "z": obs.z_score}
            daily_pnl.append(0.0)
        else:
            daily_pnl.append(0.0)

    # ---- Compute performance metrics ----
    return _compute_metrics(symbol, observations, trades, daily_pnl)


def _close_position(
    entry: dict,
    exit_obs: SkewObservation,
    store: MarketDataStore,
    symbol: str,
    bars_held: int,
    exit_reason: str,
    cost_per_leg: float,
) -> SkewTrade:
    """Close an active risk-reversal and compute P&L."""
    put_leg: RiskReversalLeg = entry["put_leg"]
    call_leg: RiskReversalLeg = entry["call_leg"]

    # Get exit prices for each leg
    put_exit_price = _mark_to_market_leg(
        store, exit_obs.date, symbol,
        put_leg.strike, "PE", put_leg.expiry, exit_obs.spot,
    )
    call_exit_price = _mark_to_market_leg(
        store, exit_obs.date, symbol,
        call_leg.strike, "CE", call_leg.expiry, exit_obs.spot,
    )

    direction = entry["direction"]

    if direction == "sell_skew":
        # Entry: sold put at put_leg.premium, bought call at call_leg.premium
        # Entry net = put_prem - call_prem
        # Exit: buy back put at put_exit, sell call at call_exit
        # Exit net = call_exit - put_exit (received from closing)
        # P&L = entry_net + exit_net = (put_prem - call_prem) + (call_exit - put_exit)
        #      = (put_prem - put_exit) - (call_prem - call_exit)
        exit_net = call_exit_price - put_exit_price
    else:
        # buy_skew: bought put, sold call
        # Entry net = call_prem - put_prem
        # Exit: sell put, buy back call
        # Exit net = put_exit - call_exit
        exit_net = put_exit_price - call_exit_price

    # Total costs: 2 legs x entry + 2 legs x exit = 4 legs
    total_cost = cost_per_leg * 4

    pnl_points = entry["entry_net_premium"] + exit_net - total_cost
    pnl_pct = pnl_points / entry["entry_spot"]

    return SkewTrade(
        entry_date=entry["entry_date"],
        exit_date=exit_obs.date,
        direction=direction,
        entry_spot=entry["entry_spot"],
        exit_spot=exit_obs.spot,
        put_leg=put_leg,
        call_leg=call_leg,
        entry_net_premium=entry["entry_net_premium"],
        exit_net_premium=exit_net,
        pnl_points=pnl_points,
        pnl_pct=pnl_pct,
        hold_days=bars_held,
        exit_reason=exit_reason,
        entry_z=entry["entry_z"],
        exit_z=exit_obs.z_score,
    )


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _compute_metrics(
    symbol: str,
    observations: list[SkewObservation],
    trades: list[SkewTrade],
    daily_pnl: list[float],
) -> SkewBacktestResult:
    """Compute backtest statistics: Sharpe, drawdown, win rate, etc."""
    n_trades = len(trades)

    if n_trades == 0:
        return SkewBacktestResult(
            symbol=symbol,
            observations=observations,
            trades=[],
            avg_skew=float(np.mean([o.skew_25d for o in observations])) if observations else 0.0,
            avg_vix=float(np.nanmean([o.vix for o in observations])) if observations else 0.0,
        )

    pnl_arr = np.array(daily_pnl)
    cumulative = np.cumsum(pnl_arr)

    total_return = float(cumulative[-1]) if len(cumulative) > 0 else 0.0
    years = max(len(daily_pnl) / 252.0, 1.0 / 252.0)
    annual_return = total_return / years

    # Sharpe: ALL daily returns (including flat days), ddof=1, sqrt(252)
    if len(pnl_arr) > 1 and np.std(pnl_arr, ddof=1) > 0:
        sharpe = float(np.mean(pnl_arr) / np.std(pnl_arr, ddof=1) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

    # Win rate
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    win_rate = wins / n_trades

    # Average holding period
    avg_hold = float(np.mean([t.hold_days for t in trades]))

    # Averages
    avg_skew = float(np.mean([o.skew_25d for o in observations]))
    avg_vix = float(np.nanmean([o.vix for o in observations]))

    return SkewBacktestResult(
        symbol=symbol,
        observations=observations,
        trades=trades,
        total_return_pct=total_return * 100,
        annual_return_pct=annual_return * 100,
        sharpe=sharpe,
        max_dd_pct=max_dd * 100,
        win_rate=win_rate,
        avg_hold_days=avg_hold,
        n_trades=n_trades,
        avg_skew=avg_skew,
        avg_vix=avg_vix,
    )


def _empty_result(symbol: str) -> SkewBacktestResult:
    """Return an empty result when data is insufficient."""
    return SkewBacktestResult(symbol=symbol, observations=[], trades=[])


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_results(result: SkewBacktestResult) -> str:
    """Format backtest results for display."""
    lines = [
        f"Volatility Skew Mean-Reversion — {result.symbol}",
        "=" * 65,
        f"  Avg 25d Skew:     {result.avg_skew:.4f} (IV_put25d - IV_call25d)",
        f"  Avg India VIX:    {result.avg_vix:.1f}",
        "",
        f"  Total return:     {result.total_return_pct:+.2f}%",
        f"  Annualized:       {result.annual_return_pct:+.2f}%",
        f"  Sharpe ratio:     {result.sharpe:.2f}",
        f"  Max drawdown:     {result.max_dd_pct:.2f}%",
        f"  Win rate:         {result.win_rate:.1%} ({result.n_trades} trades)",
        f"  Avg hold period:  {result.avg_hold_days:.1f} days",
        "",
    ]

    if result.trades:
        lines.append(
            f"  {'Entry':>12} {'Exit':>12} {'Dir':>10} {'Days':>5} "
            f"{'EntryZ':>7} {'ExitZ':>7} "
            f"{'P&L pts':>8} {'P&L%':>8} {'Reason'}"
        )
        lines.append("  " + "-" * 95)
        for t in result.trades:
            lines.append(
                f"  {t.entry_date.isoformat():>12} {t.exit_date.isoformat():>12} "
                f"{t.direction:>10} {t.hold_days:5d} "
                f"{t.entry_z:+6.2f}  {t.exit_z:+6.2f}  "
                f"{t.pnl_points:+7.1f}  {t.pnl_pct * 100:+7.2f}%  {t.exit_reason}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Allow: python skew_mr.py [symbol] [start] [end]
    symbol = sys.argv[1] if len(sys.argv) > 1 else "NIFTY"
    start_str = sys.argv[2] if len(sys.argv) > 2 else None
    end_str = sys.argv[3] if len(sys.argv) > 3 else None

    start = date.fromisoformat(start_str) if start_str else None
    end = date.fromisoformat(end_str) if end_str else None

    result = run_backtest(symbol=symbol, start_date=start, end_date=end)
    print(format_results(result))
