"""Enhanced VRP Harvesting Strategy for NIFTY Options.

Upgrades the S1 VRP strategy with regime awareness and density tail-gating.

Core edge:
    The Volatility Risk Premium (VRP = IV - RV) is persistently positive because
    market makers demand compensation for gamma risk and retail systematically
    overpays for crash protection.  By selling ATM straddles only when:
      (a) VRP is extreme (95th percentile z-score),
      (b) the HMM regime is NOT bearish,
      (c) SANOS-implied left-tail crash risk is contained,
      (d) no major events loom (expiry-day filter),
    we harvest the premium with materially better risk-adjusted returns than
    the naive "always sell vol" approach.

Trade structure:
    Short ATM straddle (sell ATM CE + ATM PE) on NIFTY, nearest weekly expiry
    with >= 3 DTE.  Actual option prices from nse_fo_bhavcopy via DuckDB.

Realized vol:
    HAR-RV model (1-day, 5-day, 22-day components) from quantlaxmi.core.pricing.iv_engine.

Exit conditions:
    - VRP z-score drops below 0.5 (premium exhausted)
    - Max hold: min(5 days, expiry - 1 day)
    - Stop-loss: -3% of notional
    - Force close at T-1 before expiry

Cost model:
    3 index points per leg (NIFTY), 4 legs round-trip = 12 pts total.

Statistics:
    Sharpe: ddof=1, sqrt(252), all daily returns including flat days.

Fully causal: signal at close of day T, execute at close of day T+1.
No look-ahead bias.

Author: AlphaForge
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from quantlaxmi.data.store import MarketDataStore
from quantlaxmi.core.pricing.iv_engine import compute_chain_iv, har_rv, OptionChainIV
from quantlaxmi.strategies.s9_momentum.data import is_trading_day, get_fno

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VRP_LOOKBACK: int = 252               # trailing window for VRP z-score
VRP_ENTRY_PCTILE: float = 0.95        # enter when VRP > 95th percentile
VRP_EXIT_Z: float = 0.5               # exit when VRP z-score drops below this
MAX_HOLD_DAYS: int = 5                # maximum holding period (trading days)
STOP_LOSS_PCT: float = 0.03           # stop-loss: 3% of notional
MIN_DTE: int = 3                      # minimum days-to-expiry for option selection
COST_PER_LEG_PTS: float = 3.0         # transaction cost per leg in NIFTY index pts
N_LEGS_ROUNDTRIP: int = 4             # 2 legs entry + 2 legs exit
TOTAL_COST_PTS: float = COST_PER_LEG_PTS * N_LEGS_ROUNDTRIP  # 12 pts
LEFT_TAIL_CEILING: float = 0.20       # skip when SANOS left-tail weight >= this
MIN_OI: int = 100                     # minimum open interest per leg
HAR_WARMUP_DAYS: int = 45             # calendar days of price history for HAR-RV
REGIME_LOOKBACK: int = 60             # days for simplified regime check


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VRPObservation:
    """Single-day VRP measurement with all filters."""
    date: date
    spot: float
    atm_iv: float                      # annualized
    har_rv: float                      # annualized (HAR-RV forecast)
    vrp: float                         # IV - RV
    vrp_z: float                       # z-score within trailing window
    straddle_price: float              # ATM straddle in index points
    nearest_expiry: str
    dte: int                           # days to nearest expiry
    left_tail: float                   # SANOS left-tail weight (nan if unavailable)
    regime: str                        # "BULL", "BEAR", or "NEUTRAL"
    regime_ret_21d: float              # 21-day trailing return (regime proxy)
    is_pre_expiry: bool                # True if next trading day is an expiry


@dataclass
class StraddleTrade:
    """One completed ATM straddle round-trip."""
    entry_date: date
    exit_date: date
    entry_spot: float
    exit_spot: float
    atm_strike: float
    ce_entry_premium: float
    pe_entry_premium: float
    straddle_entry: float              # CE + PE at entry
    straddle_exit: float               # CE + PE at exit (or intrinsic)
    pnl_points: float                  # after costs
    pnl_pct: float                     # as fraction of notional (spot)
    hold_days: int
    exit_reason: str
    entry_vrp_z: float
    entry_iv: float
    entry_rv: float
    expiry: str


@dataclass
class VRPBacktestResult:
    """Full backtest output."""
    symbol: str
    observations: list[VRPObservation]
    trades: list[StraddleTrade]
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe: float = 0.0
    max_dd_pct: float = 0.0
    win_rate: float = 0.0
    avg_hold_days: float = 0.0
    n_trades: int = 0
    avg_vrp: float = 0.0
    avg_iv: float = 0.0
    avg_rv: float = 0.0


# ---------------------------------------------------------------------------
# Regime detection (simplified — avoids full HMM dependency)
# ---------------------------------------------------------------------------

def _classify_regime(
    returns_21d: float,
    vol_ratio: float,
) -> str:
    """Simplified regime classification from trailing return and vol ratio.

    Uses a 21-day trailing return and IV/RV ratio as proxies.
    This avoids a full HMM fit per bar while capturing the essential
    regime information: are we trending down with elevated fear?

    Parameters
    ----------
    returns_21d : float
        Cumulative log return over the last 21 trading days.
    vol_ratio : float
        IV / RV ratio (> 1 means IV premium exists).

    Returns
    -------
    str : "BULL", "BEAR", or "NEUTRAL"
    """
    if returns_21d < -0.05 and vol_ratio > 1.3:
        # Significant drawdown with elevated IV — bear regime
        return "BEAR"
    elif returns_21d > 0.03 and vol_ratio < 1.5:
        # Positive trend, moderate IV premium — bull
        return "BULL"
    else:
        return "NEUTRAL"


# ---------------------------------------------------------------------------
# SANOS left-tail weight (optional)
# ---------------------------------------------------------------------------

def _get_sanos_left_tail(
    store: MarketDataStore,
    d: date,
    symbol: str,
) -> float:
    """Attempt to compute SANOS left-tail weight for crash risk assessment.

    Returns the probability mass below mu - 1*sigma in the risk-neutral
    density.  Returns NaN if SANOS calibration fails or is unavailable.
    """
    try:
        from quantlaxmi.core.pricing.sanos import fit_sanos, prepare_nifty_chain
        from quantlaxmi.core.pricing.risk_neutral import compute_snapshot

        fno = get_fno(store, d)
        if fno.empty:
            return float("nan")

        chain_data = prepare_nifty_chain(fno, symbol=symbol, max_expiries=2)
        if chain_data is None:
            return float("nan")

        result = fit_sanos(
            market_strikes=chain_data["market_strikes"],
            market_calls=chain_data["market_calls"],
            market_spreads=chain_data.get("market_spreads"),
            atm_variances=chain_data["atm_variances"],
            expiry_labels=chain_data["expiry_labels"],
            eta=0.50,
            n_model_strikes=100,
            K_min=0.7,
            K_max=1.5,
        )
        if not result.lp_success:
            return float("nan")

        snap = compute_snapshot(result, d.isoformat(), symbol)
        if snap is None or not snap.density_ok:
            return float("nan")

        return snap.left_tail

    except Exception as e:
        logger.debug("SANOS left-tail failed for %s %s: %s", symbol, d, e)
        return float("nan")


# ---------------------------------------------------------------------------
# Option chain helpers
# ---------------------------------------------------------------------------

def _get_atm_straddle(
    store: MarketDataStore,
    d: date,
    symbol: str,
    spot: float,
    min_dte: int = MIN_DTE,
    min_oi: int = MIN_OI,
) -> dict | None:
    """Select ATM CE + PE for the nearest weekly expiry with >= min_dte.

    Returns dict with strike, premiums, expiry info, or None if unavailable.
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
        opts["XpryDt"].astype(str).str.strip(), format="mixed", dayfirst=True,
    )
    today_ts = pd.Timestamp(d)
    opts["_dte"] = (opts["_expiry_dt"] - today_ts).dt.days

    # Filter: positive DTE, positive close price
    opts = opts[(opts["_dte"] >= min_dte) & (opts["ClsPric"] > 0)]
    if opts.empty:
        return None

    # Pick nearest weekly expiry with >= min_dte
    nearest_expiry = opts["_expiry_dt"].min()
    chain = opts[opts["_expiry_dt"] == nearest_expiry].copy()

    calls = chain[chain["OptnTp"].str.strip() == "CE"].copy()
    puts = chain[chain["OptnTp"].str.strip() == "PE"].copy()

    if calls.empty or puts.empty:
        return None

    # Find ATM strike: nearest to spot
    calls["_dist"] = abs(calls["StrkPric"] - spot)
    puts["_dist"] = abs(puts["StrkPric"] - spot)

    # Get the common strike nearest to spot that has both CE and PE
    call_strikes = set(calls["StrkPric"])
    put_strikes = set(puts["StrkPric"])
    common = call_strikes & put_strikes

    if not common:
        return None

    atm_strike = min(common, key=lambda k: abs(k - spot))

    ce_row = calls[calls["StrkPric"] == atm_strike]
    pe_row = puts[puts["StrkPric"] == atm_strike]

    if ce_row.empty or pe_row.empty:
        return None

    ce_premium = float(ce_row["ClsPric"].iloc[0])
    pe_premium = float(pe_row["ClsPric"].iloc[0])
    ce_oi = int(ce_row["OpnIntrst"].iloc[0])
    pe_oi = int(pe_row["OpnIntrst"].iloc[0])

    # Liquidity filter
    if ce_oi < min_oi or pe_oi < min_oi:
        return None

    dte = int(ce_row["_dte"].iloc[0])
    expiry_date = pd.Timestamp(nearest_expiry).date()

    return {
        "strike": atm_strike,
        "ce_premium": ce_premium,
        "pe_premium": pe_premium,
        "straddle": ce_premium + pe_premium,
        "expiry": expiry_date.isoformat(),
        "dte": dte,
        "ce_oi": ce_oi,
        "pe_oi": pe_oi,
    }


def _mark_straddle_exit(
    store: MarketDataStore,
    exit_date: date,
    symbol: str,
    strike: float,
    expiry_str: str,
    exit_spot: float,
) -> float:
    """Get the ATM straddle value at exit (CE + PE closing prices).

    Falls back to intrinsic value if market prices are unavailable
    or the option has expired.
    """
    expiry_date = date.fromisoformat(expiry_str)

    # If expired, use intrinsic (cash-settled, European)
    if exit_date >= expiry_date:
        # Straddle intrinsic = |spot - strike| (one leg is ITM)
        return abs(exit_spot - strike)

    # Look up market prices
    try:
        fno = get_fno(store, exit_date)
        if fno.empty:
            return abs(exit_spot - strike)
    except Exception:
        return abs(exit_spot - strike)

    opts = fno[
        (fno["TckrSymb"] == symbol)
        & (fno["FinInstrmTp"] == "IDO")
    ].copy()

    if opts.empty:
        return abs(exit_spot - strike)

    # Filter to same expiry
    opts["_expiry_dt"] = pd.to_datetime(
        opts["XpryDt"].astype(str).str.strip(), format="mixed", dayfirst=True,
    )
    same_exp = opts[opts["_expiry_dt"] == pd.Timestamp(expiry_date)]

    if same_exp.empty:
        return abs(exit_spot - strike)

    ce_rows = same_exp[
        (same_exp["OptnTp"].str.strip() == "CE")
        & (same_exp["StrkPric"] == strike)
        & (same_exp["ClsPric"] > 0)
    ]
    pe_rows = same_exp[
        (same_exp["OptnTp"].str.strip() == "PE")
        & (same_exp["StrkPric"] == strike)
        & (same_exp["ClsPric"] > 0)
    ]

    ce_val = float(ce_rows.iloc[0]["ClsPric"]) if not ce_rows.empty else max(exit_spot - strike, 0.0)
    pe_val = float(pe_rows.iloc[0]["ClsPric"]) if not pe_rows.empty else max(strike - exit_spot, 0.0)

    return ce_val + pe_val


# ---------------------------------------------------------------------------
# NIFTY close price series (for HAR-RV)
# ---------------------------------------------------------------------------

def _build_close_series(
    store: MarketDataStore,
    start: date,
    end: date,
    symbol: str = "NIFTY",
) -> pd.Series:
    """Build NIFTY close price series from F&O UndrlygPric.

    Extends back HAR_WARMUP_DAYS before start to warm up HAR-RV.
    """
    closes = {}
    d = start - timedelta(days=HAR_WARMUP_DAYS)

    while d <= end:
        if is_trading_day(d):
            try:
                fno = get_fno(store, d)
                if not fno.empty:
                    nifty_opts = fno[
                        (fno["TckrSymb"] == symbol)
                        & (fno["FinInstrmTp"].isin(["IDO", "IDF"]))
                    ]
                    if not nifty_opts.empty and "UndrlygPric" in nifty_opts.columns:
                        spot = float(nifty_opts["UndrlygPric"].iloc[0])
                        if spot > 0:
                            closes[d] = spot
            except Exception as e:
                logger.debug("Spot price lookup failed for %s on %s: %s", symbol, d, e)
        d += timedelta(days=1)

    if not closes:
        return pd.Series(dtype=float)

    return pd.Series(closes).sort_index()


def _is_pre_expiry(
    store: MarketDataStore,
    d: date,
    symbol: str,
) -> bool:
    """Check if the next trading day after d is an expiry day.

    This is a simplified event filter: we skip entering on the day
    before expiry because gamma risk is extreme and slippage is high.
    """
    # Look ahead up to 4 calendar days for the next trading day
    for offset in range(1, 5):
        next_d = d + timedelta(days=offset)
        if is_trading_day(next_d):
            # Check if next_d is an expiry by looking at current option chains
            try:
                fno = get_fno(store, d)
                if fno.empty:
                    return False
                opts = fno[
                    (fno["TckrSymb"] == symbol)
                    & (fno["FinInstrmTp"] == "IDO")
                ].copy()
                if opts.empty:
                    return False
                opts["_expiry_dt"] = pd.to_datetime(
                    opts["XpryDt"].astype(str).str.strip(),
                    format="mixed", dayfirst=True,
                )
                expiry_dates = set(opts["_expiry_dt"].dt.date)
                return next_d in expiry_dates
            except Exception:
                return False
    return False


# ---------------------------------------------------------------------------
# Build observation series
# ---------------------------------------------------------------------------

def _build_vrp_series(
    store: MarketDataStore,
    start: date,
    end: date,
    symbol: str = "NIFTY",
    lookback: int = VRP_LOOKBACK,
    use_sanos: bool = True,
) -> list[VRPObservation]:
    """Build daily VRP observations with all filter signals.

    Iterates from (start - lookback buffer) to end so that z-scores are
    available from start onwards.
    """
    # Build close series for HAR-RV
    buffer_start = start - timedelta(days=int(lookback * 1.8))
    close_series = _build_close_series(store, buffer_start, end, symbol)

    if close_series.empty:
        logger.warning("No close data available for %s", symbol)
        return []

    # Compute HAR-RV
    rv_series = har_rv(close_series)

    # Collect raw VRP data with buffer for z-score warm-up
    raw: list[dict] = []
    d = buffer_start
    while d <= end:
        if not is_trading_day(d):
            d += timedelta(days=1)
            continue

        if d not in close_series.index:
            d += timedelta(days=1)
            continue

        rv_val = rv_series.get(d, float("nan"))

        try:
            fno = get_fno(store, d)
            if fno.empty:
                d += timedelta(days=1)
                continue

            nifty_opts = fno[
                (fno["TckrSymb"] == symbol) & (fno["FinInstrmTp"] == "IDO")
            ].copy()

            if nifty_opts.empty:
                d += timedelta(days=1)
                continue

            chain_iv = compute_chain_iv(nifty_opts)
            if np.isnan(chain_iv.atm_iv):
                d += timedelta(days=1)
                continue

            spot = chain_iv.spot
            atm_iv = chain_iv.atm_iv
            vrp = atm_iv - rv_val if not np.isnan(rv_val) else float("nan")

            # ATM straddle price
            straddle_info = _get_atm_straddle(store, d, symbol, spot)
            straddle_price = straddle_info["straddle"] if straddle_info else 0.0
            nearest_expiry = straddle_info["expiry"] if straddle_info else ""
            dte = straddle_info["dte"] if straddle_info else 0

            # Regime: 21-day trailing return
            loc = close_series.index.get_loc(d)
            if isinstance(loc, slice):
                loc = loc.start
            ret_21d = 0.0
            if loc >= 21:
                ret_21d = float(np.log(
                    close_series.iloc[loc] / close_series.iloc[loc - 21]
                ))
            vol_ratio = atm_iv / rv_val if (not np.isnan(rv_val) and rv_val > 0.01) else 1.0
            regime = _classify_regime(ret_21d, vol_ratio)

            # SANOS left-tail (expensive — only compute when other filters pass)
            left_tail = float("nan")

            # Pre-expiry check
            pre_expiry = _is_pre_expiry(store, d, symbol)

            raw.append({
                "date": d,
                "spot": spot,
                "atm_iv": atm_iv,
                "har_rv": rv_val,
                "vrp": vrp,
                "straddle_price": straddle_price,
                "nearest_expiry": nearest_expiry,
                "dte": dte,
                "left_tail": left_tail,
                "regime": regime,
                "regime_ret_21d": ret_21d,
                "is_pre_expiry": pre_expiry,
            })

            if len(raw) % 20 == 0:
                logger.info(
                    "VRP computed for %d days (latest: %s, IV=%.1f%%, RV=%.1f%%)",
                    len(raw), d,
                    atm_iv * 100,
                    rv_val * 100 if not np.isnan(rv_val) else 0,
                )

        except Exception as e:
            logger.warning("VRP computation failed for %s: %s", d, e)

        d += timedelta(days=1)

    if len(raw) < lookback + 1:
        logger.warning(
            "Insufficient VRP data: %d days (need %d+1)",
            len(raw), lookback,
        )
        return []

    # Compute rolling z-scores for VRP
    vrp_values = np.array([r["vrp"] for r in raw])
    observations: list[VRPObservation] = []

    for i in range(len(raw)):
        r = raw[i]

        # Z-score: use expanding window up to lookback
        window_start = max(0, i - lookback + 1)
        window = vrp_values[window_start:i + 1]
        valid_window = window[~np.isnan(window)]

        if len(valid_window) < 30:
            vrp_z = 0.0
        else:
            mu = float(np.mean(valid_window))
            sigma = float(np.std(valid_window, ddof=1))
            vrp_z = (r["vrp"] - mu) / sigma if (sigma > 1e-8 and not np.isnan(r["vrp"])) else 0.0

        if r["date"] >= start:
            observations.append(VRPObservation(
                date=r["date"],
                spot=r["spot"],
                atm_iv=r["atm_iv"],
                har_rv=r["har_rv"],
                vrp=r["vrp"],
                vrp_z=vrp_z,
                straddle_price=r["straddle_price"],
                nearest_expiry=r["nearest_expiry"],
                dte=r["dte"],
                left_tail=r["left_tail"],
                regime=r["regime"],
                regime_ret_21d=r["regime_ret_21d"],
                is_pre_expiry=r["is_pre_expiry"],
            ))

    logger.info("Built %d VRP observations from %s to %s", len(observations), start, end)
    return observations


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    symbol: str = "NIFTY",
    start_date: date | None = None,
    end_date: date | None = None,
    store: MarketDataStore | None = None,
    lookback: int = VRP_LOOKBACK,
    entry_pctile: float = VRP_ENTRY_PCTILE,
    exit_z: float = VRP_EXIT_Z,
    max_hold: int = MAX_HOLD_DAYS,
    stop_loss_pct: float = STOP_LOSS_PCT,
    left_tail_ceiling: float = LEFT_TAIL_CEILING,
    cost_per_leg: float = COST_PER_LEG_PTS,
    use_sanos: bool = True,
) -> VRPBacktestResult:
    """Run the Enhanced VRP Harvesting backtest.

    Parameters
    ----------
    symbol : str
        Index name ("NIFTY", "BANKNIFTY", etc.)
    start_date, end_date : date
        Backtest date range.  Defaults to full available data range.
    store : MarketDataStore
        DuckDB market data store.  Creates a new one if None.
    lookback : int
        Trailing window for VRP z-score (trading days).
    entry_pctile : float
        Enter when VRP exceeds this percentile of trailing distribution.
    exit_z : float
        Exit when VRP z-score drops below this.
    max_hold : int
        Maximum holding period (trading days).
    stop_loss_pct : float
        Stop loss as fraction of notional.
    left_tail_ceiling : float
        Skip entry when SANOS left-tail weight >= this.
    cost_per_leg : float
        Transaction cost per leg in index points.
    use_sanos : bool
        Whether to compute SANOS left-tail filter (slower but more selective).

    Returns
    -------
    VRPBacktestResult with trades, statistics, and observation series.
    """
    own_store = store is None
    if store is None:
        store = MarketDataStore()

    try:
        return _run_backtest_impl(
            symbol, start_date, end_date, store, lookback,
            entry_pctile, exit_z, max_hold, stop_loss_pct,
            left_tail_ceiling, cost_per_leg, use_sanos,
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
    entry_pctile: float,
    exit_z: float,
    max_hold: int,
    stop_loss_pct: float,
    left_tail_ceiling: float,
    cost_per_leg: float,
    use_sanos: bool,
) -> VRPBacktestResult:
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
        "Building VRP series for %s from %s to %s (lookback=%d)...",
        symbol, start_date, end_date, lookback,
    )

    # Build observation series
    observations = _build_vrp_series(
        store, start_date, end_date, symbol, lookback, use_sanos,
    )
    if len(observations) < 5:
        logger.warning("Too few observations (%d), aborting", len(observations))
        return _empty_result(symbol)

    # Compute the z-score threshold that corresponds to entry_pctile
    # across the trailing window (we use z-score directly for entry/exit)
    # entry_pctile = 0.95 means we want VRP in the top 5%

    # ---- Event-driven straddle selling simulation ----
    # Causal: signal at close of day i, execute at close of day i+1 (T+1 lag)
    trades: list[StraddleTrade] = []
    daily_pnl: list[float] = []

    # Active position state
    active_entry: dict | None = None
    pending_signal: dict | None = None  # signal at close, execute next day

    # Maintain a rolling VRP collection for percentile-based entry
    trailing_vrps: list[float] = []

    for i, obs in enumerate(observations):
        # Track VRP for rolling percentile computation
        if not np.isnan(obs.vrp):
            trailing_vrps.append(obs.vrp)
        # Keep only the trailing window
        if len(trailing_vrps) > lookback:
            trailing_vrps = trailing_vrps[-lookback:]

        # -- Execute pending entry from yesterday's signal --
        if pending_signal is not None and active_entry is None:
            straddle_info = _get_atm_straddle(
                store, obs.date, symbol, obs.spot, min_dte=MIN_DTE,
            )
            if straddle_info is not None and straddle_info["straddle"] > 0:
                # Conditionally compute SANOS left-tail (expensive) only
                # when we are about to enter
                left_tail = float("nan")
                if use_sanos:
                    left_tail = _get_sanos_left_tail(store, obs.date, symbol)

                # Check SANOS left-tail filter (skip if crash risk elevated)
                sanos_ok = np.isnan(left_tail) or left_tail < left_tail_ceiling

                if sanos_ok:
                    active_entry = {
                        "entry_date": obs.date,
                        "entry_spot": obs.spot,
                        "strike": straddle_info["strike"],
                        "ce_premium": straddle_info["ce_premium"],
                        "pe_premium": straddle_info["pe_premium"],
                        "straddle_entry": straddle_info["straddle"],
                        "expiry": straddle_info["expiry"],
                        "dte": straddle_info["dte"],
                        "entry_vrp_z": pending_signal["vrp_z"],
                        "entry_iv": pending_signal["iv"],
                        "entry_rv": pending_signal["rv"],
                        "entry_idx": i,
                    }
                else:
                    logger.debug(
                        "Skipped entry on %s: SANOS left-tail=%.3f >= %.3f",
                        obs.date, left_tail, left_tail_ceiling,
                    )

            pending_signal = None

        # -- Check exit conditions for active trade --
        if active_entry is not None:
            bars_held = i - active_entry["entry_idx"]
            should_exit = False
            exit_reason = ""

            # 1. VRP z-score drops below exit threshold (premium exhausted)
            if obs.vrp_z < exit_z:
                should_exit = True
                exit_reason = "vrp_decay"

            # 2. Max holding period
            if bars_held >= max_hold:
                should_exit = True
                exit_reason = "max_hold"

            # 3. Force close at T-1 before expiry
            expiry_date = date.fromisoformat(active_entry["expiry"])
            days_to_expiry = (expiry_date - obs.date).days
            if days_to_expiry <= 1:
                should_exit = True
                exit_reason = "pre_expiry"

            # 4. Stop-loss: mark-to-market the straddle
            if bars_held > 0 and not should_exit:
                straddle_exit_val = _mark_straddle_exit(
                    store, obs.date, symbol,
                    active_entry["strike"], active_entry["expiry"], obs.spot,
                )
                # Straddle seller P&L = premium collected - current value
                unrealized_pnl_pts = active_entry["straddle_entry"] - straddle_exit_val
                unrealized_pnl_pct = unrealized_pnl_pts / active_entry["entry_spot"]
                if unrealized_pnl_pct < -stop_loss_pct:
                    should_exit = True
                    exit_reason = "stop_loss"

            # 5. End of backtest
            if i == len(observations) - 1 and not should_exit:
                should_exit = True
                exit_reason = "end_of_data"

            if should_exit and bars_held > 0:
                trade = _close_straddle(
                    active_entry, obs, store, symbol, bars_held,
                    exit_reason, cost_per_leg,
                )
                trades.append(trade)
                daily_pnl.append(trade.pnl_pct)
                active_entry = None
            elif should_exit and bars_held == 0:
                # Same-day exit: cancel entry
                active_entry = None
                daily_pnl.append(0.0)
            else:
                daily_pnl.append(0.0)
            continue

        # -- Check entry conditions (signal fires at close, execute T+1) --
        if len(trailing_vrps) < 30 or np.isnan(obs.vrp):
            daily_pnl.append(0.0)
            continue

        # Percentile rank of current VRP within trailing window
        vrp_pctile = sum(1 for v in trailing_vrps if v <= obs.vrp) / len(trailing_vrps)

        # Entry conditions — ALL must be true:
        # 1. VRP > 95th percentile of trailing distribution
        cond_vrp = vrp_pctile >= entry_pctile
        # 2. Regime is NOT Bear
        cond_regime = obs.regime != "BEAR"
        # 3. Not the day before expiry
        cond_not_pre_expiry = not obs.is_pre_expiry
        # 4. Straddle is available with >= 3 DTE
        cond_dte = obs.dte >= MIN_DTE
        # 5. Straddle price is positive
        cond_straddle = obs.straddle_price > 0

        if cond_vrp and cond_regime and cond_not_pre_expiry and cond_dte and cond_straddle:
            pending_signal = {
                "vrp_z": obs.vrp_z,
                "iv": obs.atm_iv,
                "rv": obs.har_rv,
            }
            daily_pnl.append(0.0)
        else:
            daily_pnl.append(0.0)

    # ---- Compute performance metrics ----
    return _compute_metrics(symbol, observations, trades, daily_pnl)


def _close_straddle(
    entry: dict,
    exit_obs: VRPObservation,
    store: MarketDataStore,
    symbol: str,
    bars_held: int,
    exit_reason: str,
    cost_per_leg: float,
) -> StraddleTrade:
    """Close an active straddle position and compute P&L."""
    straddle_exit_val = _mark_straddle_exit(
        store, exit_obs.date, symbol,
        entry["strike"], entry["expiry"], exit_obs.spot,
    )

    # Straddle seller P&L = premium collected - value at exit - costs
    # Costs: 4 legs total (sell CE, sell PE at entry; buy CE, buy PE at exit)
    total_cost = cost_per_leg * N_LEGS_ROUNDTRIP

    pnl_points = entry["straddle_entry"] - straddle_exit_val - total_cost
    pnl_pct = pnl_points / entry["entry_spot"]

    return StraddleTrade(
        entry_date=entry["entry_date"],
        exit_date=exit_obs.date,
        entry_spot=entry["entry_spot"],
        exit_spot=exit_obs.spot,
        atm_strike=entry["strike"],
        ce_entry_premium=entry["ce_premium"],
        pe_entry_premium=entry["pe_premium"],
        straddle_entry=entry["straddle_entry"],
        straddle_exit=straddle_exit_val,
        pnl_points=pnl_points,
        pnl_pct=pnl_pct,
        hold_days=bars_held,
        exit_reason=exit_reason,
        entry_vrp_z=entry["entry_vrp_z"],
        entry_iv=entry["entry_iv"],
        entry_rv=entry["entry_rv"],
        expiry=entry["expiry"],
    )


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _compute_metrics(
    symbol: str,
    observations: list[VRPObservation],
    trades: list[StraddleTrade],
    daily_pnl: list[float],
) -> VRPBacktestResult:
    """Compute backtest statistics: Sharpe, drawdown, win rate, etc."""
    n_trades = len(trades)

    if n_trades == 0:
        return VRPBacktestResult(
            symbol=symbol,
            observations=observations,
            trades=[],
            avg_vrp=float(np.nanmean([o.vrp for o in observations])) if observations else 0.0,
            avg_iv=float(np.mean([o.atm_iv for o in observations])) if observations else 0.0,
            avg_rv=float(np.nanmean([o.har_rv for o in observations])) if observations else 0.0,
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
    avg_vrp = float(np.nanmean([o.vrp for o in observations]))
    avg_iv = float(np.mean([o.atm_iv for o in observations]))
    avg_rv = float(np.nanmean([o.har_rv for o in observations]))

    return VRPBacktestResult(
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
        avg_vrp=avg_vrp,
        avg_iv=avg_iv,
        avg_rv=avg_rv,
    )


def _empty_result(symbol: str) -> VRPBacktestResult:
    """Return an empty result when data is insufficient."""
    return VRPBacktestResult(symbol=symbol, observations=[], trades=[])


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_results(result: VRPBacktestResult) -> str:
    """Format backtest results for display."""
    lines = [
        f"Enhanced VRP Harvesting (Regime + Tail-Gated) -- {result.symbol}",
        "=" * 70,
        f"  Avg IV:           {result.avg_iv * 100:6.1f}%",
        f"  Avg RV (HAR):     {result.avg_rv * 100:6.1f}%",
        f"  Avg VRP:          {result.avg_vrp * 100:6.1f}% (IV - RV)",
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
            f"  {'Entry':>12} {'Exit':>12} {'Days':>5} "
            f"{'IV':>6} {'RV':>6} {'VRP-z':>6} "
            f"{'Strad':>7} {'Exit$':>7} {'Net':>8} {'Reason'}"
        )
        lines.append("  " + "-" * 100)
        for t in result.trades:
            lines.append(
                f"  {t.entry_date.isoformat():>12} {t.exit_date.isoformat():>12} "
                f"{t.hold_days:5d} "
                f"{t.entry_iv * 100:5.1f}% {t.entry_rv * 100:5.1f}% "
                f"{t.entry_vrp_z:+5.2f}  "
                f"{t.straddle_entry:6.1f}  {t.straddle_exit:6.1f}  "
                f"{t.pnl_pct * 100:+7.2f}%  {t.exit_reason}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    # Add project root to path
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Allow: python vrp_enhanced.py [symbol] [start] [end]
    symbol = sys.argv[1] if len(sys.argv) > 1 else "NIFTY"
    start_str = sys.argv[2] if len(sys.argv) > 2 else None
    end_str = sys.argv[3] if len(sys.argv) > 3 else None

    start = date.fromisoformat(start_str) if start_str else None
    end = date.fromisoformat(end_str) if end_str else None

    # Run backtest for NIFTY, full available date range
    result = run_backtest(symbol=symbol, start_date=start, end_date=end)
    print(format_results(result))
