"""Regime-VRP Combo Strategy: HMM Regime x Enhanced VRP for NIFTY Options.

Combines the two best-performing AlphaForge strategies:
  - HMM Regime (Sharpe 1.02): 3-state HMM detects Bull/Bear/Neutral regimes
  - Enhanced VRP (Sharpe 2.03): sells ATM straddles when VRP is extreme

Core idea:
    The Volatility Risk Premium (VRP = IV - RV) is persistent, but naive
    short-vol gets killed by regime shifts.  By using the HMM's posterior
    probabilities as a filter AND a position sizer, we:
      1. NEVER sell vol in BEAR regime (avoids blow-ups)
      2. Sell vol in BULL/NEUTRAL only when VRP z-score > 0.5
      3. Size positions via Kelly fraction scaled by regime confidence

Trade structure:
    Short ATM straddle (sell ATM CE + ATM PE) on NIFTY, nearest weekly expiry
    with >= 3 DTE.  Actual option prices from nse_fo_bhavcopy via DuckDB.

Data sources:
    - Kite API: 2-year daily OHLCV (for HMM fitting on longer history)
    - DuckDB nse_fo_bhavcopy: actual option prices (2025-08-06 to 2026-02-06)
    - DuckDB nse_index_close: India VIX for feature enrichment
    - core.pricing.iv_engine: compute_chain_iv for ATM IV, har_rv for realized vol

Cost model:
    3 pts/leg NIFTY x 4 legs round-trip = 12 pts total.

Execution:
    Fully causal: signal at close of day T, execute at close of day T+1.
    No look-ahead bias.

Statistics:
    Sharpe: ddof=1, sqrt(252), all daily returns including flat days.
"""

from __future__ import annotations

import logging
import math
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HMM
MIN_HMM_HISTORY: int = 120       # minimum days before first HMM fit
REFIT_INTERVAL: int = 20         # refit every N trading days
CONFIDENCE_THRESHOLD: float = 0.55  # minimum regime confidence to trade

# VRP entry/exit
VRP_ENTRY_Z: float = 0.5         # enter when VRP z-score > this (relaxed: HMM regime is primary filter)
VRP_EXIT_Z: float = -1.0         # exit when VRP z-score drops below this (deeply negative = RV > IV)
VRP_LOOKBACK: int = 60           # trailing window for VRP z-score (shorter for faster reaction)

# Position sizing
FULL_KELLY: float = 0.25         # max Kelly fraction (quarter-Kelly for safety)
MIN_KELLY: float = 0.05          # minimum position fraction

# Trade management
MAX_HOLD_DAYS: int = 5           # maximum holding period (trading days)
STOP_LOSS_PCT: float = 0.025     # stop-loss: 2.5% of notional
MIN_DTE: int = 3                 # minimum days-to-expiry for option selection
MIN_OI: int = 100                # minimum open interest per leg

# Costs
COST_PER_LEG_PTS: float = 3.0    # per leg in NIFTY index pts
N_LEGS_ROUNDTRIP: int = 4        # 2 legs entry + 2 legs exit
TOTAL_COST_PTS: float = COST_PER_LEG_PTS * N_LEGS_ROUNDTRIP  # 12 pts

# HAR-RV
HAR_WARMUP_DAYS: int = 45        # calendar days of price history for HAR-RV warmup

# Kite
NIFTY_INSTRUMENT_TOKEN: int = 256265


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RegimeVRPObservation:
    """Single-day observation combining HMM regime with VRP data."""
    date: date
    spot: float
    atm_iv: float
    har_rv: float
    vrp: float
    vrp_z: float
    straddle_price: float
    nearest_expiry: str
    dte: int
    regime: str              # "BULL", "BEAR", or "NEUTRAL"
    regime_confidence: float # HMM posterior probability
    kelly_fraction: float    # position size = confidence * kelly
    is_pre_expiry: bool


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
    straddle_entry: float
    straddle_exit: float
    pnl_points: float
    pnl_pct: float           # as fraction of notional (spot), scaled by kelly
    hold_days: int
    exit_reason: str
    entry_vrp_z: float
    entry_iv: float
    entry_rv: float
    entry_regime: str
    entry_confidence: float
    entry_kelly: float
    expiry: str


@dataclass
class ComboBacktestResult:
    """Full backtest output."""
    symbol: str
    observations: list[RegimeVRPObservation]
    trades: list[StraddleTrade]
    daily_returns: list[float] = field(default_factory=list)
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe: float = 0.0
    max_dd_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_hold_days: float = 0.0
    n_trades: int = 0
    avg_vrp: float = 0.0
    avg_iv: float = 0.0
    avg_rv: float = 0.0
    regime_distribution: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HMM helpers (reuse from hmm_regime strategy)
# ---------------------------------------------------------------------------

def _fit_hmm_on_kite_data(
    kite_df: pd.DataFrame,
    vix_series: Optional[pd.Series],
    up_to_idx: int,
) -> tuple:
    """Fit the 3-state HMM on Kite OHLCV data up to (exclusive) up_to_idx.

    Returns (hmm, state_map, fit_mean, fit_std) or (None, None, None, None)
    on failure.
    """
    from strategies.s13_hmm_regime.strategy import (
        extract_features, GaussianHMM, N_STATES, N_EM_ITER, EM_TOL,
        _standardize_features, _label_states,
    )

    # Build feature DataFrame from Kite data
    feat_df = pd.DataFrame({
        "date": kite_df["date"].values[:up_to_idx],
        "open": kite_df["open"].values[:up_to_idx],
        "high": kite_df["high"].values[:up_to_idx],
        "low": kite_df["low"].values[:up_to_idx],
        "close": kite_df["close"].values[:up_to_idx],
    })
    if "volume" in kite_df.columns:
        feat_df["volume"] = kite_df["volume"].values[:up_to_idx]

    features = extract_features(feat_df, vix_series)

    feat_cols = ["ret_1d", "ret_5d", "ret_21d", "vol_5d", "vol_21d", "rsi_14"]
    valid_mask = features[feat_cols].notna().all(axis=1)
    X_raw = features.loc[valid_mask, feat_cols].values.astype(np.float64)

    if len(X_raw) < MIN_HMM_HISTORY:
        return None, None, None, None

    # Standardize
    fit_mean = X_raw.mean(axis=0)
    fit_std = X_raw.std(axis=0, ddof=1)
    X_std = _standardize_features(X_raw, fit_mean, fit_std)

    hmm = GaussianHMM(
        n_states=N_STATES,
        n_iter=N_EM_ITER,
        tol=EM_TOL,
        covariance_type="diag",
        random_state=42,
    )

    try:
        hmm.fit(X_std)
        result = hmm.decode(X_std)
        raw_means = result.means * fit_std + fit_mean
        state_map = _label_states(raw_means)
        return hmm, state_map, fit_mean, fit_std
    except Exception as e:
        logger.warning("HMM fit failed: %s", e)
        return None, None, None, None


def _decode_regime_at(
    hmm,
    state_map: dict,
    fit_mean: np.ndarray,
    fit_std: np.ndarray,
    kite_df: pd.DataFrame,
    vix_series: Optional[pd.Series],
    up_to_idx: int,
) -> tuple[str, float]:
    """Decode the regime at the most recent observation.

    Returns (regime_name, confidence) using data up to (exclusive) up_to_idx.
    """
    from strategies.s13_hmm_regime.strategy import (
        extract_features, _standardize_features, Regime,
    )

    feat_df = pd.DataFrame({
        "date": kite_df["date"].values[:up_to_idx],
        "open": kite_df["open"].values[:up_to_idx],
        "high": kite_df["high"].values[:up_to_idx],
        "low": kite_df["low"].values[:up_to_idx],
        "close": kite_df["close"].values[:up_to_idx],
    })
    if "volume" in kite_df.columns:
        feat_df["volume"] = kite_df["volume"].values[:up_to_idx]

    features = extract_features(feat_df, vix_series)

    feat_cols = ["ret_1d", "ret_5d", "ret_21d", "vol_5d", "vol_21d", "rsi_14"]
    valid_mask = features[feat_cols].notna().all(axis=1)
    X_raw = features.loc[valid_mask, feat_cols].values.astype(np.float64)

    if len(X_raw) < 10:
        return "NEUTRAL", 0.0

    X_std = _standardize_features(X_raw, fit_mean, fit_std)

    try:
        posteriors = hmm.predict_proba(X_std)
        last_post = posteriors[-1]
        hmm_state = int(np.argmax(last_post))
        confidence = float(last_post[hmm_state])
        regime = state_map.get(hmm_state, Regime.NEUTRAL)
        return regime.name, confidence
    except Exception:
        return "NEUTRAL", 0.0


# ---------------------------------------------------------------------------
# Kelly fraction from VRP z-score
# ---------------------------------------------------------------------------

def _kelly_from_vrp_z(vrp_z: float, confidence: float) -> float:
    """Compute position size as Kelly fraction scaled by regime confidence.

    Kelly fraction increases with VRP z-score (higher z = more edge).
    Then scaled by HMM regime confidence (0.55 to 1.0).

    Formula:
        base_kelly = clip(vrp_z / 6.0, MIN_KELLY, FULL_KELLY)
        kelly = base_kelly * confidence
    """
    # VRP z of 1.5 -> 0.25 * 1.5/6 = ~0.063
    # VRP z of 3.0 -> 0.25 * 3.0/6 = 0.125
    # VRP z of 6.0 -> 0.25 (max)
    base = np.clip(FULL_KELLY * vrp_z / 6.0, MIN_KELLY, FULL_KELLY)
    return float(base * confidence)


# ---------------------------------------------------------------------------
# Option chain helpers (reuse from vrp_enhanced)
# ---------------------------------------------------------------------------

def _get_atm_straddle(
    store,
    d: date,
    symbol: str,
    spot: float,
    min_dte: int = MIN_DTE,
    min_oi: int = MIN_OI,
) -> dict | None:
    """Select ATM CE + PE for the nearest weekly expiry with >= min_dte.

    Returns dict with strike, premiums, expiry info, or None if unavailable.
    """
    from strategies.s9_momentum.data import get_fno

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

    # Find ATM strike: nearest to spot among common strikes
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
    store,
    exit_date: date,
    symbol: str,
    strike: float,
    expiry_str: str,
    exit_spot: float,
) -> float:
    """Get the ATM straddle value at exit (CE + PE closing prices).

    Falls back to intrinsic value if market prices are unavailable.
    """
    from strategies.s9_momentum.data import get_fno

    expiry_date_val = date.fromisoformat(expiry_str)

    # If expired, use intrinsic (cash-settled, European)
    if exit_date >= expiry_date_val:
        return abs(exit_spot - strike)

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

    opts["_expiry_dt"] = pd.to_datetime(
        opts["XpryDt"].astype(str).str.strip(), format="mixed", dayfirst=True,
    )
    same_exp = opts[opts["_expiry_dt"] == pd.Timestamp(expiry_date_val)]

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


def _is_pre_expiry(store, d: date, symbol: str) -> bool:
    """Check if the next trading day after d is an expiry day."""
    from strategies.s9_momentum.data import is_trading_day, get_fno

    for offset in range(1, 5):
        next_d = d + timedelta(days=offset)
        if is_trading_day(next_d):
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
# Data loading
# ---------------------------------------------------------------------------

def _load_kite_ohlcv() -> pd.DataFrame:
    """Load ~2 years of daily NIFTY OHLCV from Kite API.

    Returns DataFrame with columns: date, open, high, low, close, volume.
    Falls back to DuckDB nse_index_close if Kite is unavailable.
    """
    try:
        from data.collectors.auth import headless_login
        from data.zerodha import fetch_historical_chunked

        kite = headless_login()
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=730)  # ~2 years

        ohlcv = fetch_historical_chunked(
            kite,
            instrument_token=NIFTY_INSTRUMENT_TOKEN,
            interval="1d",
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
        )

        df = ohlcv.df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.date

        logger.info(
            "Loaded %d days of NIFTY OHLCV from Kite (%s to %s)",
            len(df), df["date"].iloc[0], df["date"].iloc[-1],
        )
        return df

    except Exception as e:
        logger.warning("Kite data load failed (%s), falling back to DuckDB", e)
        return _load_nifty_from_duckdb()


def _load_nifty_from_duckdb() -> pd.DataFrame:
    """Fallback: load NIFTY close from DuckDB nse_index_close.

    Only has close price (no OHLV), but sufficient for HMM on close-based features.
    """
    from data.store import MarketDataStore

    with MarketDataStore() as store:
        df = store.sql(
            'SELECT date, "Closing Index Value" as close '
            'FROM nse_index_close '
            'WHERE "Index Name" = ? '
            'ORDER BY date',
            ["Nifty 50"],
        )

    if df.empty:
        raise ValueError("No NIFTY data in DuckDB nse_index_close")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    logger.info(
        "Loaded %d days of NIFTY close from DuckDB (%s to %s)",
        len(df), df["date"].iloc[0], df["date"].iloc[-1],
    )
    return df


def _load_vix_series() -> Optional[pd.Series]:
    """Load India VIX series from DuckDB nse_index_close."""
    try:
        from data.store import MarketDataStore

        with MarketDataStore() as store:
            df = store.sql(
                'SELECT date, "Closing Index Value" as vix '
                'FROM nse_index_close '
                'WHERE "Index Name" = ? '
                'ORDER BY date',
                ["India VIX"],
            )

        if df.empty:
            return None

        df["vix"] = pd.to_numeric(df["vix"], errors="coerce") / 100.0  # percent to decimal
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.dropna(subset=["vix"])

        return pd.Series(df["vix"].values, index=df["date"].values, name="vix")

    except Exception as e:
        logger.warning("VIX load failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Close price series for HAR-RV (from FnO underlying price)
# ---------------------------------------------------------------------------

def _build_close_series_from_fno(
    store,
    start: date,
    end: date,
    symbol: str = "NIFTY",
) -> pd.Series:
    """Build NIFTY close price series from F&O UndrlygPric.

    Extends back HAR_WARMUP_DAYS before start to warm up HAR-RV.
    """
    from strategies.s9_momentum.data import is_trading_day, get_fno

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
            except Exception:
                pass
        d += timedelta(days=1)

    if not closes:
        return pd.Series(dtype=float)

    return pd.Series(closes).sort_index()


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def run_backtest(
    symbol: str = "NIFTY",
    start_date: date = date(2025, 8, 6),
    end_date: date = date(2026, 2, 6),
) -> ComboBacktestResult:
    """Run the Regime-VRP Combo backtest.

    Architecture:
        1. Load ~2 years of daily NIFTY from Kite (for HMM fitting)
        2. Fit 3-state HMM, refit every REFIT_INTERVAL days
        3. For the FnO data range (start_date to end_date):
           - Compute daily VRP = ATM IV - HAR RV
           - Apply regime filter: only enter when BULL or NEUTRAL
           - Size positions via Kelly scaled by regime confidence
        4. Fully causal: signal at close T, execute at close T+1

    Returns ComboBacktestResult with trades, daily returns, and statistics.
    """
    from data.store import MarketDataStore
    from core.pricing.iv_engine import compute_chain_iv, har_rv
    from strategies.s9_momentum.data import is_trading_day, get_fno

    # --- Step 1: Load Kite OHLCV for HMM ---
    logger.info("Loading NIFTY OHLCV for HMM fitting...")
    kite_df = _load_kite_ohlcv()

    # Load VIX for feature enrichment
    vix_series = _load_vix_series()

    # --- Step 2: Load FnO data for VRP computation ---
    logger.info("Opening MarketDataStore for FnO data...")
    store = MarketDataStore()

    try:
        # Build close series from FnO for HAR-RV
        close_series = _build_close_series_from_fno(store, start_date, end_date, symbol)
        if close_series.empty:
            logger.error("No close price data from FnO bhavcopy")
            return _empty_result(symbol)

        # Compute HAR-RV
        rv_series = har_rv(close_series)

        # --- Step 3: Build day-by-day observation + trade simulation ---
        # Map Kite dates to indices for HMM fitting
        kite_dates = list(kite_df["date"].values)
        kite_date_to_idx = {}
        for idx_val, d_val in enumerate(kite_dates):
            kite_date_to_idx[d_val] = idx_val

        # HMM state
        hmm = None
        state_map = None
        fit_mean = None
        fit_std = None
        last_fit_date = None

        # VRP tracking
        trailing_vrps: list[float] = []

        # Trade state
        observations: list[RegimeVRPObservation] = []
        trades: list[StraddleTrade] = []
        daily_returns: list[float] = []
        active_entry: dict | None = None
        pending_signal: dict | None = None

        # Get all FnO dates
        avail_dates = store.available_dates("nse_fo_bhavcopy")
        bt_dates = [d for d in avail_dates if start_date <= d <= end_date]
        bt_dates.sort()

        if not bt_dates:
            logger.error("No FnO dates in range %s to %s", start_date, end_date)
            return _empty_result(symbol)

        logger.info(
            "Running combo backtest: %d trading days (%s to %s)",
            len(bt_dates), bt_dates[0], bt_dates[-1],
        )

        for day_idx, d in enumerate(bt_dates):
            # --- HMM regime detection (causal: use data up to d-1) ---
            # Find the Kite index for yesterday (last available before d)
            kite_idx = None
            search_d = d - timedelta(days=1)
            for back in range(10):
                candidate = search_d - timedelta(days=back)
                if candidate in kite_date_to_idx:
                    kite_idx = kite_date_to_idx[candidate] + 1  # exclusive upper bound
                    break

            if kite_idx is None:
                # Try using d itself if we have kite data for it
                if d in kite_date_to_idx:
                    kite_idx = kite_date_to_idx[d]  # use up to but NOT including d
                else:
                    # No Kite data available for this date
                    daily_returns.append(0.0)
                    continue

            # Refit HMM if needed
            days_since_fit = (
                (d - last_fit_date).days if last_fit_date else REFIT_INTERVAL + 1
            )
            if hmm is None or days_since_fit >= REFIT_INTERVAL:
                hmm, state_map, fit_mean, fit_std = _fit_hmm_on_kite_data(
                    kite_df, vix_series, kite_idx,
                )
                if hmm is not None:
                    last_fit_date = d
                    logger.debug("HMM refit at %s (using %d Kite bars)", d, kite_idx)

            # Decode regime
            if hmm is not None and state_map is not None:
                regime, confidence = _decode_regime_at(
                    hmm, state_map, fit_mean, fit_std,
                    kite_df, vix_series, kite_idx,
                )
            else:
                regime, confidence = "NEUTRAL", 0.0

            # --- VRP computation ---
            # Get spot and ATM IV from option chain
            try:
                fno = get_fno(store, d)
                if fno.empty:
                    daily_returns.append(0.0)
                    continue

                nifty_opts = fno[
                    (fno["TckrSymb"] == symbol) & (fno["FinInstrmTp"] == "IDO")
                ].copy()

                if nifty_opts.empty:
                    daily_returns.append(0.0)
                    continue

                chain_iv = compute_chain_iv(nifty_opts)
                if np.isnan(chain_iv.atm_iv):
                    daily_returns.append(0.0)
                    continue

                spot = chain_iv.spot
                atm_iv = chain_iv.atm_iv

            except Exception as e:
                logger.debug("IV computation failed for %s: %s", d, e)
                daily_returns.append(0.0)
                continue

            # HAR-RV for this date
            rv_val = rv_series.get(d, float("nan"))
            vrp = atm_iv - rv_val if not np.isnan(rv_val) else float("nan")

            # Track VRP for z-score
            if not np.isnan(vrp):
                trailing_vrps.append(vrp)
            if len(trailing_vrps) > VRP_LOOKBACK:
                trailing_vrps = trailing_vrps[-VRP_LOOKBACK:]

            # Compute VRP z-score
            if len(trailing_vrps) >= 15 and not np.isnan(vrp):
                valid_vrps = np.array(trailing_vrps)
                mu = float(np.mean(valid_vrps))
                sigma = float(np.std(valid_vrps, ddof=1))
                vrp_z = (vrp - mu) / sigma if sigma > 1e-8 else 0.0
            else:
                vrp_z = 0.0

            # ATM straddle info
            straddle_info = _get_atm_straddle(store, d, symbol, spot)
            straddle_price = straddle_info["straddle"] if straddle_info else 0.0
            nearest_expiry = straddle_info["expiry"] if straddle_info else ""
            dte = straddle_info["dte"] if straddle_info else 0

            # Pre-expiry check
            pre_expiry = _is_pre_expiry(store, d, symbol)

            # Kelly fraction
            kelly = _kelly_from_vrp_z(vrp_z, confidence) if vrp_z > VRP_ENTRY_Z else 0.0

            obs = RegimeVRPObservation(
                date=d,
                spot=spot,
                atm_iv=atm_iv,
                har_rv=rv_val,
                vrp=vrp,
                vrp_z=vrp_z,
                straddle_price=straddle_price,
                nearest_expiry=nearest_expiry,
                dte=dte,
                regime=regime,
                regime_confidence=confidence,
                kelly_fraction=kelly,
                is_pre_expiry=pre_expiry,
            )
            observations.append(obs)

            # --- Execute pending entry from yesterday's signal ---
            if pending_signal is not None and active_entry is None:
                entry_straddle = _get_atm_straddle(store, d, symbol, spot, min_dte=MIN_DTE)
                if entry_straddle is not None and entry_straddle["straddle"] > 0:
                    active_entry = {
                        "entry_date": d,
                        "entry_spot": spot,
                        "strike": entry_straddle["strike"],
                        "ce_premium": entry_straddle["ce_premium"],
                        "pe_premium": entry_straddle["pe_premium"],
                        "straddle_entry": entry_straddle["straddle"],
                        "expiry": entry_straddle["expiry"],
                        "dte": entry_straddle["dte"],
                        "entry_vrp_z": pending_signal["vrp_z"],
                        "entry_iv": pending_signal["iv"],
                        "entry_rv": pending_signal["rv"],
                        "entry_regime": pending_signal["regime"],
                        "entry_confidence": pending_signal["confidence"],
                        "entry_kelly": pending_signal["kelly"],
                        "entry_day_idx": day_idx,
                    }
                pending_signal = None

            # --- Check exit conditions ---
            if active_entry is not None:
                bars_held = day_idx - active_entry["entry_day_idx"]
                should_exit = False
                exit_reason = ""

                # 1. VRP z-score drops below exit threshold
                if vrp_z < VRP_EXIT_Z:
                    should_exit = True
                    exit_reason = "vrp_decay"

                # 2. Regime flips to BEAR
                if regime == "BEAR" and confidence >= CONFIDENCE_THRESHOLD:
                    should_exit = True
                    exit_reason = "bear_regime"

                # 3. Max holding period
                if bars_held >= MAX_HOLD_DAYS:
                    should_exit = True
                    exit_reason = "max_hold"

                # 4. Force close at T-1 before expiry
                expiry_date = date.fromisoformat(active_entry["expiry"])
                days_to_expiry = (expiry_date - d).days
                if days_to_expiry <= 1:
                    should_exit = True
                    exit_reason = "pre_expiry"

                # 5. Stop-loss (mark-to-market)
                if bars_held > 0 and not should_exit:
                    straddle_exit_val = _mark_straddle_exit(
                        store, d, symbol,
                        active_entry["strike"], active_entry["expiry"], spot,
                    )
                    unrealized_pnl_pts = active_entry["straddle_entry"] - straddle_exit_val
                    unrealized_pnl_pct = unrealized_pnl_pts / active_entry["entry_spot"]
                    # Scale by Kelly for actual position P&L
                    if unrealized_pnl_pct * active_entry["entry_kelly"] < -STOP_LOSS_PCT:
                        should_exit = True
                        exit_reason = "stop_loss"

                # 6. End of backtest
                if day_idx == len(bt_dates) - 1 and not should_exit:
                    should_exit = True
                    exit_reason = "end_of_data"

                if should_exit and bars_held > 0:
                    trade = _close_straddle_trade(
                        active_entry, d, spot, store, symbol,
                        bars_held, exit_reason,
                    )
                    trades.append(trade)
                    daily_returns.append(trade.pnl_pct)
                    active_entry = None
                elif should_exit and bars_held == 0:
                    active_entry = None
                    daily_returns.append(0.0)
                else:
                    daily_returns.append(0.0)
                continue

            # --- Check entry conditions (signal at close, execute T+1) ---
            if np.isnan(vrp) or len(trailing_vrps) < 15:
                daily_returns.append(0.0)
                continue

            # Entry conditions -- ALL must be true:
            # Note: pre-expiry filter removed because NIFTY has weekly expiries
            # (every Thursday), so ~20% of trading days are pre-expiry.
            # The HMM regime filter + VRP z-score provide sufficient protection.
            # We still force-close at T-1 before OUR position's expiry (exit rule).
            cond_vrp = vrp_z > VRP_ENTRY_Z                              # VRP z-score > 1.0
            cond_regime = regime != "BEAR"                               # Not in Bear regime
            cond_confidence = confidence >= CONFIDENCE_THRESHOLD         # Regime confidence > 0.55
            cond_dte = dte >= MIN_DTE                                    # >= 3 DTE
            cond_straddle = straddle_price > 0                           # Straddle available

            if all([cond_vrp, cond_regime, cond_confidence,
                    cond_dte, cond_straddle]):
                pending_signal = {
                    "vrp_z": vrp_z,
                    "iv": atm_iv,
                    "rv": rv_val,
                    "regime": regime,
                    "confidence": confidence,
                    "kelly": kelly,
                }
                daily_returns.append(0.0)
            else:
                daily_returns.append(0.0)

        # --- Compute performance metrics ---
        result = _compute_metrics(symbol, observations, trades, daily_returns)
        return result

    finally:
        store.close()


def _close_straddle_trade(
    entry: dict,
    exit_d: date,
    exit_spot: float,
    store,
    symbol: str,
    bars_held: int,
    exit_reason: str,
) -> StraddleTrade:
    """Close an active straddle position and compute Kelly-scaled P&L."""
    straddle_exit_val = _mark_straddle_exit(
        store, exit_d, symbol,
        entry["strike"], entry["expiry"], exit_spot,
    )

    # Straddle seller P&L = premium collected - value at exit - costs
    pnl_points = entry["straddle_entry"] - straddle_exit_val - TOTAL_COST_PTS

    # Scale by Kelly fraction for actual portfolio return
    pnl_pct = (pnl_points / entry["entry_spot"]) * entry["entry_kelly"]

    return StraddleTrade(
        entry_date=entry["entry_date"],
        exit_date=exit_d,
        entry_spot=entry["entry_spot"],
        exit_spot=exit_spot,
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
        entry_regime=entry["entry_regime"],
        entry_confidence=entry["entry_confidence"],
        entry_kelly=entry["entry_kelly"],
        expiry=entry["expiry"],
    )


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _compute_metrics(
    symbol: str,
    observations: list[RegimeVRPObservation],
    trades: list[StraddleTrade],
    daily_returns: list[float],
) -> ComboBacktestResult:
    """Compute backtest statistics: Sharpe, drawdown, win rate, profit factor."""
    n_trades = len(trades)

    # Regime distribution
    regime_counts = {}
    for obs in observations:
        regime_counts[obs.regime] = regime_counts.get(obs.regime, 0) + 1

    if n_trades == 0:
        return ComboBacktestResult(
            symbol=symbol,
            observations=observations,
            trades=[],
            daily_returns=daily_returns,
            regime_distribution=regime_counts,
            avg_vrp=float(np.nanmean([o.vrp for o in observations])) if observations else 0.0,
            avg_iv=float(np.mean([o.atm_iv for o in observations])) if observations else 0.0,
            avg_rv=float(np.nanmean([o.har_rv for o in observations])) if observations else 0.0,
        )

    pnl_arr = np.array(daily_returns)

    # Total return (compounding)
    cum = np.cumprod(1.0 + pnl_arr)
    total_return = float(cum[-1] - 1.0) if len(cum) > 0 else 0.0

    years = max(len(daily_returns) / 252.0, 1.0 / 252.0)
    annual_return = total_return / years

    # Sharpe: ALL daily returns (including flat days), ddof=1, sqrt(252)
    if len(pnl_arr) > 1 and np.std(pnl_arr, ddof=1) > 0:
        sharpe = float(np.mean(pnl_arr) / np.std(pnl_arr, ddof=1) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Max drawdown (on cumulative return curve)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

    # Win rate
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    win_rate = wins / n_trades

    # Profit factor = gross profits / gross losses
    gross_profit = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    gross_loss = abs(sum(t.pnl_pct for t in trades if t.pnl_pct < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Average holding period
    avg_hold = float(np.mean([t.hold_days for t in trades]))

    # Averages
    avg_vrp = float(np.nanmean([o.vrp for o in observations]))
    avg_iv = float(np.mean([o.atm_iv for o in observations]))
    avg_rv = float(np.nanmean([o.har_rv for o in observations]))

    return ComboBacktestResult(
        symbol=symbol,
        observations=observations,
        trades=trades,
        daily_returns=daily_returns,
        total_return_pct=total_return * 100,
        annual_return_pct=annual_return * 100,
        sharpe=sharpe,
        max_dd_pct=max_dd * 100,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_hold_days=avg_hold,
        n_trades=n_trades,
        avg_vrp=avg_vrp,
        avg_iv=avg_iv,
        avg_rv=avg_rv,
        regime_distribution=regime_counts,
    )


def _empty_result(symbol: str) -> ComboBacktestResult:
    """Return an empty result when data is insufficient."""
    return ComboBacktestResult(symbol=symbol, observations=[], trades=[])


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_results(result: ComboBacktestResult) -> str:
    """Format backtest results for display."""
    lines = [
        "",
        "=" * 72,
        f"  Regime-VRP Combo Strategy -- {result.symbol}",
        "  (HMM Regime Filter x Enhanced VRP Harvesting)",
        "=" * 72,
        "",
        "  -- Market Summary --",
        f"  Avg IV:           {result.avg_iv * 100:6.1f}%",
        f"  Avg RV (HAR):     {result.avg_rv * 100:6.1f}%",
        f"  Avg VRP:          {result.avg_vrp * 100:6.1f}% (IV - RV)",
        "",
        "  -- Regime Distribution --",
    ]
    for regime, count in sorted(result.regime_distribution.items()):
        pct = count / sum(result.regime_distribution.values()) * 100
        lines.append(f"    {regime:>8s}: {count:4d} days ({pct:.0f}%)")

    lines += [
        "",
        "  -- Performance --",
        f"  Total return:     {result.total_return_pct:+.2f}%",
        f"  Annualized:       {result.annual_return_pct:+.2f}%",
        f"  Sharpe ratio:     {result.sharpe:.3f}  (ddof=1, sqrt(252), all days)",
        f"  Max drawdown:     {result.max_dd_pct:.2f}%",
        f"  Win rate:         {result.win_rate:.1%} ({result.n_trades} trades)",
        f"  Profit factor:    {result.profit_factor:.2f}",
        f"  Avg hold period:  {result.avg_hold_days:.1f} days",
        "",
    ]

    if result.trades:
        lines.append(
            f"  {'Entry':>12} {'Exit':>12} {'Days':>5} {'Regime':>8} "
            f"{'Conf':>5} {'Kelly':>6} "
            f"{'IV':>6} {'RV':>6} {'VRP-z':>6} "
            f"{'Strad':>7} {'Exit$':>7} {'Net':>8} {'Reason'}"
        )
        lines.append("  " + "-" * 120)
        for t in result.trades:
            lines.append(
                f"  {t.entry_date.isoformat():>12} {t.exit_date.isoformat():>12} "
                f"{t.hold_days:5d} {t.entry_regime:>8} "
                f"{t.entry_confidence:5.2f} {t.entry_kelly:5.3f}  "
                f"{t.entry_iv * 100:5.1f}% {t.entry_rv * 100:5.1f}% "
                f"{t.entry_vrp_z:+5.2f}  "
                f"{t.straddle_entry:6.1f}  {t.straddle_exit:6.1f}  "
                f"{t.pnl_pct * 100:+7.3f}%  {t.exit_reason}"
            )

    lines.append("=" * 72)
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
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("\n[Regime-VRP Combo] Starting backtest...")
    print(f"  Project root: {project_root}")
    print(f"  Strategy: HMM Regime Filter x VRP Harvesting")
    print(f"  Symbol: NIFTY")
    print(f"  Period: 2025-08-06 to 2026-02-06")
    print()

    result = run_backtest(
        symbol="NIFTY",
        start_date=date(2025, 8, 6),
        end_date=date(2026, 2, 6),
    )

    print(format_results(result))

    # Save results
    if result.trades:
        out_dir = Path(__file__).resolve().parent.parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save trades
        trades_df = pd.DataFrame([{
            "entry_date": t.entry_date.isoformat(),
            "exit_date": t.exit_date.isoformat(),
            "entry_spot": t.entry_spot,
            "exit_spot": t.exit_spot,
            "atm_strike": t.atm_strike,
            "straddle_entry": t.straddle_entry,
            "straddle_exit": t.straddle_exit,
            "pnl_points": t.pnl_points,
            "pnl_pct": t.pnl_pct,
            "hold_days": t.hold_days,
            "exit_reason": t.exit_reason,
            "entry_vrp_z": t.entry_vrp_z,
            "entry_iv": t.entry_iv,
            "entry_rv": t.entry_rv,
            "entry_regime": t.entry_regime,
            "entry_confidence": t.entry_confidence,
            "entry_kelly": t.entry_kelly,
            "expiry": t.expiry,
        } for t in result.trades])
        trades_file = out_dir / "regime_vrp_combo_trades.csv"
        trades_df.to_csv(trades_file, index=False)

        # Save daily returns
        rets_df = pd.DataFrame({"daily_return": result.daily_returns})
        rets_file = out_dir / "regime_vrp_combo_daily_returns.csv"
        rets_df.to_csv(rets_file, index=False)

        print(f"\n  Trades saved to:  {trades_file}")
        print(f"  Returns saved to: {rets_file}")
