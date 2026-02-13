"""TFT + Deep Hedging Pipeline (Pattern 3) + Standalone Deep Hedger (NEW-2).

Pattern 3 — TFT Deep Hedging Pipeline
--------------------------------------
TFT backbone provides directional signal (hidden state embedding) which
conditions a DeepHedgingAgent that selects option structures and manages
Greeks dynamically.  Walk-forward: backbone pretrained on each fold, frozen,
then deep hedger trained on historical spot+IV paths from DuckDB.

NEW-2 — Standalone Deep Hedger
-------------------------------
For S1 VRP straddles and S10 gamma scalp strategies.  Trains on real
historical spot and IV paths from nse_fo_bhavcopy (via Newton-Raphson IV
extraction), then compares OOS PnL variance against Black-Scholes delta
hedging baseline.

Architecture:
    load_historical_iv_paths() → Newton-Raphson per date from DuckDB
    _build_iv_augmented_paths() → GBM intraday bootstrap from daily endpoints
    DeepHedgingAgent.train_on_paths() with CVaR objective
    Walk-forward: TFT hidden → directional signal, DH manages Greeks

References:
    - Ch 9.2: Deep Hedging (Buehler et al. 2019)
    - Ch 10.2: Optimal Execution (Bertsimas-Lo)
    - MEMORY.md: option costs 3 pts NIFTY, 5 pts BANKNIFTY per leg
    - Sharpe: ddof=1, sqrt(252), all daily returns including flat days
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    _HAS_TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _HAS_TORCH = False
    _DEVICE = None

from quantlaxmi.models.rl.agents.deep_hedger import DeepHedgingAgent
from quantlaxmi.models.rl.environments.options_env import GammaScalpEnv, OptionsEnv
from quantlaxmi.models.rl.environments.india_fno_env import (
    COST_PER_LEG,
    INITIAL_SPOTS,
    ANNUALISED_VOLS,
)

__all__ = [
    "TFTDeepHedgingPipeline",
    "StandaloneDeepHedger",
    "load_historical_iv_paths",
]


# ---------------------------------------------------------------------------
# Black-Scholes helpers for Newton-Raphson IV extraction
# ---------------------------------------------------------------------------

_SQRT2PI = math.sqrt(2.0 * math.pi)


def _bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """BS d1 parameter."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """European Black-Scholes option price."""
    if T <= 1e-10:
        return max(0.0, (S - K) if is_call else (K - S))
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    # Use math.erf for CDF to avoid scipy dependency in hot path
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
    if is_call:
        return S * nd1 - K * math.exp(-r * T) * nd2
    else:
        return K * math.exp(-r * T) * (1.0 - nd2) - S * (1.0 - nd1)


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """BS vega = S * sqrt(T) * phi(d1)."""
    if T <= 1e-10 or sigma <= 1e-8 or S <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    return S * math.sqrt(T) * math.exp(-0.5 * d1 ** 2) / _SQRT2PI


def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """BS delta."""
    if T <= 1e-10:
        if is_call:
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    if sigma <= 1e-8:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    if is_call:
        return nd1
    else:
        return nd1 - 1.0


def _newton_iv(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> float:
    """Implied vol via Newton-Raphson on Black-Scholes model.

    Returns NaN if convergence fails or price is below intrinsic.
    """
    intrinsic = max(0.0, (S - K) if is_call else (K - S))
    if market_price < intrinsic + 1e-4:
        return float("nan")
    if T <= 1e-10 or S <= 0 or K <= 0:
        return float("nan")

    # Brenner-Subrahmanyam initial guess
    sigma = math.sqrt(2.0 * math.pi / T) * (market_price / S)
    sigma = max(0.01, min(sigma, 5.0))

    for _ in range(max_iter):
        price = _bs_price(S, K, T, r, sigma, is_call)
        vega = _bs_vega(S, K, T, r, sigma)
        if vega < 1e-12:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
        sigma = max(0.001, min(sigma, 10.0))

    # Final convergence check with relaxed tolerance
    if abs(_bs_price(S, K, T, r, sigma, is_call) - market_price) < tol * 10:
        return sigma
    return float("nan")


# ---------------------------------------------------------------------------
# nse_fo_bhavcopy column mapping (matches OptionsFeatureBuilder)
# ---------------------------------------------------------------------------

_FO_COLS = {
    "symbol": "TckrSymb",
    "instr_type": "FinInstrmTp",
    "strike": "StrkPric",
    "option_type": "OptnTp",
    "close": "ClsPric",
    "settle": "SttlmPric",
    "volume": "TtlTradgVol",
    "oi": "OpnIntrst",
    "expiry": "XpryDt",
    "lot_size": "NewBrdLotQty",
}

# Index name mapping for nse_index_close
_INDEX_NAME_MAP = {
    "NIFTY": "Nifty 50",
    "BANKNIFTY": "Nifty Bank",
    "FINNIFTY": "Nifty Financial Services",
    "MIDCPNIFTY": "NIFTY MidCap Select",
}


# ---------------------------------------------------------------------------
# Historical data loading from DuckDB
# ---------------------------------------------------------------------------


def _load_spot_series(
    store, ticker: str, start_date: str, end_date: str, rv_buffer_days: int = 60
) -> pd.Series:
    """Load daily spot prices from nse_index_close.

    Parameters
    ----------
    store : MarketDataStore
        DuckDB store.
    ticker : str
        Underlying (e.g. "NIFTY", "BANKNIFTY").
    start_date, end_date : str
        Date range "YYYY-MM-DD".
    rv_buffer_days : int
        Extra leading days for realized vol warm-up.

    Returns
    -------
    pd.Series indexed by date string "YYYY-MM-DD", values = close price.
    """
    idx_name = _INDEX_NAME_MAP.get(ticker.upper(), f"Nifty {ticker}")
    extended_start = (
        pd.Timestamp(start_date) - pd.Timedelta(days=rv_buffer_days)
    ).strftime("%Y-%m-%d")

    df = store.sql(
        'SELECT date, CAST("Closing Index Value" AS DOUBLE) as close '
        "FROM nse_index_close "
        'WHERE LOWER("Index Name") = LOWER(?) '
        "AND date >= ? AND date <= ? "
        "ORDER BY date",
        [idx_name, extended_start, end_date],
    )
    if df.empty:
        logger.warning("No spot data for %s from nse_index_close", ticker)
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.set_index("date")["close"]


def _load_fo_chain(
    store, ticker: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Load FnO bhavcopy chain data from DuckDB.

    Loads CE + PE for the ticker across the date range, nearest
    expiry per date (IDO for indices, STO for stocks).

    Returns
    -------
    pd.DataFrame with columns: date, symbol, strike, option_type, close, volume, oi, expiry.
    """
    query = (
        "SELECT date, "
        f'"{_FO_COLS["symbol"]}" AS symbol, '
        f'CAST("{_FO_COLS["strike"]}" AS DOUBLE) AS strike, '
        f'"{_FO_COLS["option_type"]}" AS option_type, '
        f'CAST("{_FO_COLS["close"]}" AS DOUBLE) AS close, '
        f'CAST("{_FO_COLS["volume"]}" AS DOUBLE) AS volume, '
        f'CAST("{_FO_COLS["oi"]}" AS DOUBLE) AS oi, '
        f'"{_FO_COLS["expiry"]}" AS expiry '
        "FROM nse_fo_bhavcopy "
        f'WHERE "{_FO_COLS["symbol"]}" = ? '
        f'AND "{_FO_COLS["instr_type"]}" IN (\'IDO\', \'STO\') '
        f'AND "{_FO_COLS["option_type"]}" IN (\'CE\', \'PE\') '
        "AND date >= ? AND date <= ? "
        "ORDER BY date, expiry, strike, option_type"
    )
    df = store.sql(query, [ticker.upper(), start_date, end_date])
    return df


def _extract_atm_iv_for_date(
    chain_day: pd.DataFrame,
    spot: float,
    dt_str: str,
    risk_free_rate: float = 0.065,
) -> float:
    """Extract ATM implied vol from a single day's option chain via Newton-Raphson.

    Finds nearest-to-spot strike, computes IV for both call and put, returns average.
    """
    if chain_day.empty or spot <= 0:
        return float("nan")

    calls = chain_day[chain_day["option_type"] == "CE"]
    puts = chain_day[chain_day["option_type"] == "PE"]
    if calls.empty and puts.empty:
        return float("nan")

    # Nearest expiry only
    expiries = sorted(chain_day["expiry"].unique())
    if not expiries:
        return float("nan")
    nearest_exp = expiries[0]
    chain_exp = chain_day[chain_day["expiry"] == nearest_exp]
    calls = chain_exp[chain_exp["option_type"] == "CE"]
    puts = chain_exp[chain_exp["option_type"] == "PE"]

    # Time to expiry in years
    try:
        exp_date = pd.Timestamp(nearest_exp)
        dt_date = pd.Timestamp(dt_str)
        dte = max(1, (exp_date - dt_date).days)
        T = dte / 365.0
    except Exception:
        T = 7.0 / 365.0

    # Find ATM strike
    all_strikes = pd.concat([calls["strike"], puts["strike"]]).unique()
    if len(all_strikes) == 0:
        return float("nan")
    atm_strike = float(all_strikes[np.argmin(np.abs(all_strikes - spot))])

    r = risk_free_rate
    ivs: list[float] = []

    atm_call = calls[calls["strike"] == atm_strike]
    if not atm_call.empty:
        price_c = float(atm_call.iloc[0]["close"])
        if price_c > 0:
            iv_c = _newton_iv(price_c, spot, atm_strike, T, r, True)
            if not np.isnan(iv_c) and 0.01 < iv_c < 5.0:
                ivs.append(iv_c)

    atm_put = puts[puts["strike"] == atm_strike]
    if not atm_put.empty:
        price_p = float(atm_put.iloc[0]["close"])
        if price_p > 0:
            iv_p = _newton_iv(price_p, spot, atm_strike, T, r, False)
            if not np.isnan(iv_p) and 0.01 < iv_p < 5.0:
                ivs.append(iv_p)

    if ivs:
        return float(np.mean(ivs))

    # Fallback: Brenner-Subrahmanyam straddle approximation
    straddle = 0.0
    if not atm_call.empty:
        straddle += max(0.0, float(atm_call.iloc[0]["close"]))
    if not atm_put.empty:
        straddle += max(0.0, float(atm_put.iloc[0]["close"]))
    if straddle > 0 and T > 0:
        approx_iv = math.sqrt(2.0 * math.pi / T) * (straddle / spot)
        if 0.01 < approx_iv < 5.0:
            return approx_iv

    return float("nan")


def load_historical_iv_paths(
    store,
    ticker: str,
    start_date: str,
    end_date: str,
    risk_free_rate: float = 0.065,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load real daily spot and ATM IV from DuckDB nse_fo_bhavcopy.

    For each trading date in [start_date, end_date]:
    1. Query nse_index_close for spot close.
    2. Query nse_fo_bhavcopy for nearest-expiry ATM option chain.
    3. Newton-Raphson to extract ATM IV.

    Parameters
    ----------
    store : MarketDataStore
        DuckDB-backed store with nse_index_close and nse_fo_bhavcopy views.
    ticker : str
        Underlying symbol (e.g. "NIFTY", "BANKNIFTY").
    start_date, end_date : str
        Date range "YYYY-MM-DD".
    risk_free_rate : float
        Annualized risk-free rate.

    Returns
    -------
    spot_daily : np.ndarray of shape (n_days,)
        Daily spot closes.
    iv_daily : np.ndarray of shape (n_days,)
        Daily ATM implied vols (NaN where extraction failed, forward-filled).
    dates : pd.DatetimeIndex
        Trading dates.
    """
    spot_series = _load_spot_series(store, ticker, start_date, end_date, rv_buffer_days=0)
    if spot_series.empty:
        logger.warning("load_historical_iv_paths: no spot data for %s", ticker)
        return np.array([]), np.array([]), pd.DatetimeIndex([])

    fo_chain = _load_fo_chain(store, ticker, start_date, end_date)

    # Filter spot to requested range
    mask = (spot_series.index >= start_date) & (spot_series.index <= end_date)
    spot_filtered = spot_series[mask]
    if spot_filtered.empty:
        return np.array([]), np.array([]), pd.DatetimeIndex([])

    dates_list = sorted(spot_filtered.index)
    n_days = len(dates_list)
    spot_daily = np.zeros(n_days, dtype=np.float64)
    iv_daily = np.full(n_days, np.nan, dtype=np.float64)

    for i, dt_str in enumerate(dates_list):
        spot_val = float(spot_filtered[dt_str])
        spot_daily[i] = spot_val

        if not fo_chain.empty:
            day_chain = fo_chain[fo_chain["date"].astype(str).str[:10] == dt_str]
            if not day_chain.empty:
                iv = _extract_atm_iv_for_date(day_chain, spot_val, dt_str, risk_free_rate)
                iv_daily[i] = iv

    # Forward-fill NaN IVs (causal: only past values)
    for i in range(1, n_days):
        if np.isnan(iv_daily[i]) and not np.isnan(iv_daily[i - 1]):
            iv_daily[i] = iv_daily[i - 1]

    # Backfill any leading NaN with first valid value
    first_valid = -1
    for i in range(n_days):
        if not np.isnan(iv_daily[i]):
            first_valid = i
            break
    if first_valid > 0:
        for i in range(first_valid):
            iv_daily[i] = iv_daily[first_valid]

    # If still all NaN, use annualized vol as fallback
    if np.all(np.isnan(iv_daily)):
        fallback_vol = ANNUALISED_VOLS.get(ticker.upper(), 0.15)
        iv_daily[:] = fallback_vol
        logger.warning(
            "load_historical_iv_paths: all IV NaN for %s, using fallback %.2f",
            ticker, fallback_vol,
        )

    dates_idx = pd.DatetimeIndex(dates_list)
    logger.info(
        "load_historical_iv_paths: %s, %d days, spot [%.0f, %.0f], IV [%.3f, %.3f]",
        ticker, n_days,
        np.min(spot_daily), np.max(spot_daily),
        float(np.nanmin(iv_daily)), float(np.nanmax(iv_daily)),
    )
    return spot_daily, iv_daily, dates_idx


# ---------------------------------------------------------------------------
# GBM intraday path bootstrap
# ---------------------------------------------------------------------------


def _build_iv_augmented_paths(
    spot_daily: np.ndarray,
    iv_daily: np.ndarray,
    steps_per_day: int = 78,
    risk_free_rate: float = 0.065,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap intraday paths from daily endpoints using GBM micro-steps.

    For each consecutive pair of daily observations (S_d, S_{d+1}), generates
    `steps_per_day` intraday steps conditioned on matching the daily endpoint
    via a Brownian bridge approach:

        S_{t+dt} = S_t * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)

    where mu is calibrated so the path ends at S_{d+1} (endpoint conditioning).
    IV is interpolated linearly within each day.

    Parameters
    ----------
    spot_daily : (n_days,) array of daily spot closes.
    iv_daily : (n_days,) array of daily ATM IVs.
    steps_per_day : int
        Intraday steps per trading day (78 for 5-min intervals).
    risk_free_rate : float
        Annualized risk-free rate.
    rng : numpy Generator or None.

    Returns
    -------
    spot_paths : ((n_days-1), steps_per_day+1) intraday spot paths per day-pair.
    iv_paths : ((n_days-1), steps_per_day+1) interpolated IV paths.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_days = len(spot_daily)
    if n_days < 2:
        return np.array([]).reshape(0, 0), np.array([]).reshape(0, 0)

    n_paths = n_days - 1
    steps = steps_per_day
    dt = 1.0 / (252.0 * steps_per_day)

    spot_paths = np.zeros((n_paths, steps + 1), dtype=np.float64)
    iv_paths = np.zeros((n_paths, steps + 1), dtype=np.float64)

    for d in range(n_paths):
        S_start = spot_daily[d]
        S_end = spot_daily[d + 1]
        iv_start = iv_daily[d]
        iv_end = iv_daily[d + 1]

        if S_start <= 0 or S_end <= 0 or np.isnan(S_start) or np.isnan(S_end):
            spot_paths[d, :] = S_start if S_start > 0 else INITIAL_SPOTS.get("NIFTY", 24000.0)
            iv_paths[d, :] = iv_start if not np.isnan(iv_start) else 0.15
            continue

        sigma = iv_start if not np.isnan(iv_start) else 0.15

        # Calibrate drift so GBM path matches endpoint on average
        # log(S_end/S_start) = mu_total over the day
        log_return_daily = math.log(S_end / S_start)

        # Generate Brownian bridge: random increments conditioned on total displacement
        # Step 1: Generate free increments
        Z = rng.standard_normal(steps)

        # Step 2: Build unconditioned log-increments
        uncond_increments = (risk_free_rate - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * Z
        uncond_total = float(np.sum(uncond_increments))

        # Step 3: Adjust all increments uniformly to match actual daily return
        # This is the Brownian bridge conditioning
        adjustment = (log_return_daily - uncond_total) / steps
        conditioned_increments = uncond_increments + adjustment

        # Step 4: Build path from conditioned increments
        spot_paths[d, 0] = S_start
        log_spot = math.log(S_start)
        for t in range(steps):
            log_spot += conditioned_increments[t]
            spot_paths[d, t + 1] = math.exp(log_spot)

        # Linear interpolation of IV within the day
        for t in range(steps + 1):
            frac = t / steps
            iv_val = iv_start * (1.0 - frac) + iv_end * frac
            iv_paths[d, t] = max(iv_val, 0.01)

    return spot_paths, iv_paths


def _build_multi_day_paths(
    spot_daily: np.ndarray,
    iv_daily: np.ndarray,
    expiry_days: int = 30,
    steps_per_day: int = 78,
    risk_free_rate: float = 0.065,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build multi-day continuous paths by stitching together daily intraday segments.

    Generates paths of length `expiry_days * steps_per_day` steps, each
    corresponding to one option lifetime (e.g. 30 days).  Uses a rolling
    window over the daily data.

    Parameters
    ----------
    spot_daily : (n_days,) daily spots.
    iv_daily : (n_days,) daily ATM IVs.
    expiry_days : int
        Length of each path in trading days.
    steps_per_day : int
        Intraday granularity.
    risk_free_rate : float
    rng : numpy Generator or None.

    Returns
    -------
    spot_paths : (n_paths, total_steps) where total_steps = expiry_days * steps_per_day.
    iv_paths : (n_paths, total_steps).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_days = len(spot_daily)
    if n_days < expiry_days + 1:
        # Not enough data; build from what we have
        per_day_spot, per_day_iv = _build_iv_augmented_paths(
            spot_daily, iv_daily, steps_per_day, risk_free_rate, rng
        )
        if per_day_spot.size == 0:
            return np.array([]).reshape(0, 0), np.array([]).reshape(0, 0)
        # Flatten into a single long path
        flat_spot = per_day_spot[:, :-1].flatten()
        flat_iv = per_day_iv[:, :-1].flatten()
        # Append final step
        flat_spot = np.append(flat_spot, per_day_spot[-1, -1])
        flat_iv = np.append(flat_iv, per_day_iv[-1, -1])
        return flat_spot.reshape(1, -1), flat_iv.reshape(1, -1)

    total_steps = expiry_days * steps_per_day
    n_paths = n_days - expiry_days
    spot_paths = np.zeros((n_paths, total_steps), dtype=np.float64)
    iv_paths_out = np.zeros((n_paths, total_steps), dtype=np.float64)

    for p in range(n_paths):
        window_spot = spot_daily[p: p + expiry_days + 1]
        window_iv = iv_daily[p: p + expiry_days + 1]

        per_day_spot, per_day_iv = _build_iv_augmented_paths(
            window_spot, window_iv, steps_per_day, risk_free_rate, rng
        )
        # per_day_spot: (expiry_days, steps_per_day+1)
        # Stitch: take all steps except the last of each day (which equals first of next)
        idx = 0
        for d in range(min(expiry_days, per_day_spot.shape[0])):
            day_steps = min(steps_per_day, total_steps - idx)
            spot_paths[p, idx: idx + day_steps] = per_day_spot[d, :day_steps]
            iv_paths_out[p, idx: idx + day_steps] = per_day_iv[d, :day_steps]
            idx += day_steps
        # Fill any remaining steps (shouldn't happen normally)
        if idx < total_steps:
            spot_paths[p, idx:] = spot_paths[p, idx - 1]
            iv_paths_out[p, idx:] = iv_paths_out[p, idx - 1]

    return spot_paths, iv_paths_out


# ---------------------------------------------------------------------------
# Realized vol computation
# ---------------------------------------------------------------------------

def _compute_realized_vol_series(spot_daily: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute rolling annualized realized vol (close-to-close, ddof=1).

    Parameters
    ----------
    spot_daily : (n_days,) array.
    window : int — lookback window.

    Returns
    -------
    rv : (n_days,) array, NaN for insufficient history.
    """
    n = len(spot_daily)
    rv = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return rv

    log_rets = np.diff(np.log(np.maximum(spot_daily, 1e-10)))

    for i in range(window, n):
        window_rets = log_rets[i - window: i]
        if len(window_rets) >= window:
            vol = float(np.std(window_rets, ddof=1)) * math.sqrt(252.0)
            rv[i] = max(vol, 0.001)

    return rv


# ===========================================================================
# TFTDeepHedgingPipeline (Pattern 3)
# ===========================================================================


@dataclass
class DHPipelineConfig:
    """Configuration for TFT + Deep Hedging pipeline."""

    # Hedging parameters
    instrument: str = "NIFTY"
    strategy: str = "straddle"
    hedging_interval: str = "5min"
    expiry_days: int = 30
    risk_free_rate: float = 0.065

    # Deep hedging training
    dh_hidden_layers: Tuple[int, ...] = (128, 64, 32)
    dh_learning_rate: float = 1e-4
    dh_risk_aversion: float = 1.0
    dh_train_epochs: int = 200
    dh_batch_size: int = 128

    # Walk-forward
    train_window_days: int = 252
    test_window_days: int = 63
    step_size_days: int = 21

    # Backbone pretrain
    pretrain_epochs: int = 50


class TFTDeepHedgingPipeline:
    """TFT backbone provides directional signal, DeepHedgingAgent manages Greeks.

    Walk-forward protocol:
    1. For each fold, pretrain backbone on supervised MLE+Sharpe loss.
    2. Freeze backbone; extract hidden states as directional signal.
    3. Load real historical spot+IV paths from DuckDB.
    4. Bootstrap intraday paths from daily via GBM bridge.
    5. Train DeepHedgingAgent on paths with CVaR objective.
    6. Evaluate OOS: backbone hidden → direction, DH → hedge actions.

    Parameters
    ----------
    cfg : DHPipelineConfig
        Pipeline configuration.
    backbone_cfg : XTrendConfig or None
        X-Trend backbone hyperparameters (imported lazily).
    """

    def __init__(
        self,
        cfg: Optional[DHPipelineConfig] = None,
        backbone_cfg: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg or DHPipelineConfig()
        self._backbone_cfg = backbone_cfg
        self._results: Dict[str, Any] = {}

    def run(
        self,
        start: str,
        end: str,
        store=None,
        features: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        targets: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run the full TFT + Deep Hedging pipeline.

        Parameters
        ----------
        start, end : str
            Date range "YYYY-MM-DD".
        store : MarketDataStore, optional
            DuckDB store. Created if not provided.
        features : optional — pre-built (n_days, n_assets, n_features).
        feature_names : optional — feature column names.
        dates : optional — trading dates.
        targets : optional — (n_days, n_assets) vol-scaled next-day returns.

        Returns
        -------
        dict with keys:
            fold_metrics : list[dict] per walk-forward fold
            oos_deep_pnls : list[np.ndarray] — OOS hedged PnL per fold
            oos_bs_pnls : list[np.ndarray] — OOS BS baseline PnL per fold
            total_improvement_pct : float — aggregate DH vs BS improvement
            directional_accuracy : float — backbone directional signal accuracy
        """
        cfg = self.cfg

        # Load store if needed
        if store is None:
            from quantlaxmi.data.store import MarketDataStore
            store = MarketDataStore()

        # Load historical spot + IV
        spot_daily, iv_daily, data_dates = load_historical_iv_paths(
            store, cfg.instrument, start, end, cfg.risk_free_rate
        )
        if len(spot_daily) < cfg.train_window_days + cfg.test_window_days:
            logger.warning(
                "Insufficient data: %d days < %d train + %d test",
                len(spot_daily), cfg.train_window_days, cfg.test_window_days,
            )
            return {"fold_metrics": [], "error": "insufficient_data"}

        # Build features for backbone if not provided
        if features is None and _HAS_TORCH:
            from .backbone import MegaFeatureAdapter
            symbols = [cfg.instrument]
            adapter = MegaFeatureAdapter(symbols)
            try:
                features, feature_names, dates = adapter.build_multi_asset(start, end)
            except Exception as e:
                logger.warning("MegaFeatureAdapter failed: %s. Using spot-only features.", e)
                features = spot_daily.reshape(-1, 1, 1)
                feature_names = ["spot_close"]
                dates = data_dates

        # Compute realized vol for directional signal evaluation
        rv_daily = _compute_realized_vol_series(spot_daily, window=20)

        # Walk-forward
        n_days = len(spot_daily)
        fold_metrics: List[Dict[str, Any]] = []
        all_deep_pnls: List[np.ndarray] = []
        all_bs_pnls: List[np.ndarray] = []
        directional_hits = 0
        directional_total = 0

        fold_idx = 0
        fold_start = 0

        while fold_start + cfg.train_window_days + cfg.test_window_days <= n_days:
            train_end = fold_start + cfg.train_window_days
            test_end = min(train_end + cfg.test_window_days, n_days)
            test_len = test_end - train_end

            logger.info(
                "TFTDeepHedging fold %d: train=[%d:%d], test=[%d:%d]",
                fold_idx, fold_start, train_end, train_end, test_end,
            )

            # --- Phase 1: Pre-train backbone (if features available) ---
            backbone_hidden_direction = np.zeros(test_len, dtype=np.float64)
            if _HAS_TORCH and features is not None and features.shape[0] >= train_end:
                try:
                    backbone_hidden_direction = self._extract_directional_signal(
                        features, feature_names, dates, targets,
                        fold_start, train_end, test_end,
                    )
                except Exception as e:
                    logger.warning("Backbone failed in fold %d: %s", fold_idx, e)

            # --- Phase 2: Train deep hedger on historical paths ---
            train_spot = spot_daily[fold_start:train_end]
            train_iv = iv_daily[fold_start:train_end]

            rng = np.random.default_rng(42 + fold_idx)
            train_paths_spot, train_paths_iv = _build_multi_day_paths(
                train_spot, train_iv,
                expiry_days=cfg.expiry_days,
                steps_per_day=DeepHedgingAgent.STEPS_PER_DAY.get(cfg.hedging_interval, 78),
                risk_free_rate=cfg.risk_free_rate,
                rng=rng,
            )

            if train_paths_spot.size == 0:
                logger.warning("No training paths in fold %d, skipping", fold_idx)
                fold_start += cfg.step_size_days
                fold_idx += 1
                continue

            # Average ATM strike from training data
            strike = float(np.mean(train_spot))

            agent = DeepHedgingAgent(
                instrument=cfg.instrument,
                strategy=cfg.strategy,
                hedging_interval=cfg.hedging_interval,
                hidden_layers=cfg.dh_hidden_layers,
                learning_rate=cfg.dh_learning_rate,
                risk_aversion=cfg.dh_risk_aversion,
            )

            train_result = agent.train_on_paths(
                train_paths_spot,
                iv_paths=train_paths_iv,
                num_epochs=cfg.dh_train_epochs,
                batch_size=cfg.dh_batch_size,
                strike=strike,
                sigma=float(np.nanmean(train_iv)),
                risk_free_rate=cfg.risk_free_rate,
            )
            logger.info(
                "Fold %d DH training: final_loss=%.4f",
                fold_idx, train_result.get("final_loss", float("nan")),
            )

            # --- Phase 3: Evaluate OOS ---
            test_spot = spot_daily[train_end:test_end]
            test_iv = iv_daily[train_end:test_end]

            test_paths_spot, test_paths_iv = _build_multi_day_paths(
                test_spot, test_iv,
                expiry_days=min(cfg.expiry_days, test_len - 1),
                steps_per_day=DeepHedgingAgent.STEPS_PER_DAY.get(cfg.hedging_interval, 78),
                risk_free_rate=cfg.risk_free_rate,
                rng=rng,
            )

            if test_paths_spot.size == 0:
                fold_start += cfg.step_size_days
                fold_idx += 1
                continue

            comparison = agent.compare_vs_bs(
                test_paths_spot, strike=strike,
                sigma=float(np.nanmean(test_iv)),
                risk_free_rate=cfg.risk_free_rate,
            )

            # --- Phase 4: Evaluate directional signal ---
            # Check if backbone direction prediction agrees with actual spot move
            actual_returns = np.diff(np.log(np.maximum(test_spot, 1e-10)))
            for t in range(min(len(actual_returns), len(backbone_hidden_direction))):
                if abs(backbone_hidden_direction[t]) > 0.01:
                    directional_total += 1
                    if np.sign(backbone_hidden_direction[t]) == np.sign(actual_returns[t]):
                        directional_hits += 1

            # Collect results
            fold_result = {
                "fold": fold_idx,
                "train_range": (fold_start, train_end),
                "test_range": (train_end, test_end),
                "dh_train_loss": train_result.get("final_loss", float("nan")),
                "dh_vs_bs": comparison,
                "n_train_paths": train_paths_spot.shape[0],
                "n_test_paths": test_paths_spot.shape[0],
            }
            fold_metrics.append(fold_result)
            all_deep_pnls.append(np.array([comparison["deep_hedge_pnl_mean"]]))
            all_bs_pnls.append(np.array([comparison["bs_hedge_pnl_mean"]]))

            logger.info(
                "Fold %d OOS: DH std=%.4f, BS std=%.4f, improvement=%.1f%%",
                fold_idx,
                comparison["deep_hedge_pnl_std"],
                comparison["bs_hedge_pnl_std"],
                comparison["improvement_pct"],
            )

            fold_start += cfg.step_size_days
            fold_idx += 1

            # Cleanup
            del agent
            if _HAS_TORCH:
                torch.cuda.empty_cache()

        # Aggregate
        directional_accuracy = (
            directional_hits / directional_total if directional_total > 0 else 0.0
        )

        # Total improvement across folds
        total_dh_std = 0.0
        total_bs_std = 0.0
        n_folds_with_data = 0
        for fm in fold_metrics:
            comp = fm["dh_vs_bs"]
            total_dh_std += comp["deep_hedge_pnl_std"]
            total_bs_std += comp["bs_hedge_pnl_std"]
            n_folds_with_data += 1

        avg_dh_std = total_dh_std / max(n_folds_with_data, 1)
        avg_bs_std = total_bs_std / max(n_folds_with_data, 1)
        total_improvement = (
            (avg_bs_std - avg_dh_std) / max(avg_bs_std, 1e-8) * 100.0
        )

        self._results = {
            "fold_metrics": fold_metrics,
            "oos_deep_pnls": all_deep_pnls,
            "oos_bs_pnls": all_bs_pnls,
            "total_improvement_pct": total_improvement,
            "directional_accuracy": directional_accuracy,
            "avg_dh_std": avg_dh_std,
            "avg_bs_std": avg_bs_std,
            "n_folds": n_folds_with_data,
        }
        return self._results

    def _extract_directional_signal(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]],
        dates: Optional[pd.DatetimeIndex],
        targets: Optional[np.ndarray],
        fold_start: int,
        train_end: int,
        test_end: int,
    ) -> np.ndarray:
        """Extract backbone directional signal for the test fold.

        Pre-trains backbone on the train fold, freezes it, then extracts
        hidden states for each test day.  Returns the first hidden dimension
        as a directional signal proxy.

        Returns
        -------
        direction : (test_len,) — positive = bullish, negative = bearish.
        """
        from quantlaxmi.models.ml.tft.x_trend import XTrendConfig
        from .backbone import XTrendBackbone

        test_len = test_end - train_end
        direction = np.zeros(test_len, dtype=np.float64)

        n_features = features.shape[2]
        n_assets = features.shape[1]

        bcfg = self._backbone_cfg
        if bcfg is None:
            bcfg = XTrendConfig(
                n_features=n_features,
                n_assets=n_assets,
                d_hidden=64,
                seq_len=min(42, fold_start + 10),
                ctx_len=min(42, fold_start + 10),
                n_context=min(16, n_assets * 4),
                loss_mode="joint_mle",
                mle_weight=0.1,
            )
        else:
            bcfg.n_features = n_features
            bcfg.n_assets = n_assets

        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(n_features)]

        backbone = XTrendBackbone(bcfg, feature_names)
        dev = _DEVICE
        backbone.to(dev)

        # Pre-train on the fold
        if targets is not None:
            backbone.pretrain(
                features, targets, dates,
                train_start=fold_start,
                train_end=train_end,
                epochs=self.cfg.pretrain_epochs,
                lr=bcfg.lr if hasattr(bcfg, "lr") else 1e-3,
            )

        # Freeze
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False

        # Extract hidden states for test fold
        rng = np.random.default_rng(42)
        hidden_states = backbone.precompute_hidden_states(
            features, train_end, test_end, rng
        )
        # hidden_states: (test_len, n_assets, d_hidden)
        # Use mean across assets, first dimension as directional proxy
        for t in range(min(test_len, hidden_states.shape[0])):
            mean_hidden = np.mean(hidden_states[t, :, :], axis=0)
            direction[t] = float(mean_hidden[0]) if len(mean_hidden) > 0 else 0.0

        del backbone
        return direction

    def report(self, results: Optional[Dict[str, Any]] = None) -> None:
        """Print human-readable pipeline results."""
        if results is None:
            results = self._results
        if not results:
            print("No results to report.")
            return

        print(f"\n{'=' * 70}")
        print(f"{'TFT + DEEP HEDGING PIPELINE (Pattern 3)':^70}")
        print(f"{'=' * 70}")
        print(f"\nInstrument: {self.cfg.instrument}")
        print(f"Strategy: {self.cfg.strategy}")
        print(f"Hedging interval: {self.cfg.hedging_interval}")
        print(f"\nTotal folds: {results.get('n_folds', 0)}")
        print(f"DH avg PnL std: {results.get('avg_dh_std', 0):.4f}")
        print(f"BS avg PnL std: {results.get('avg_bs_std', 0):.4f}")
        print(f"Total improvement: {results.get('total_improvement_pct', 0):.1f}%")
        print(f"Directional accuracy: {results.get('directional_accuracy', 0):.1%}")

        for fm in results.get("fold_metrics", []):
            comp = fm["dh_vs_bs"]
            print(
                f"\n  Fold {fm['fold']}: "
                f"test=[{fm['test_range'][0]}:{fm['test_range'][1]}] "
                f"DH_std={comp['deep_hedge_pnl_std']:.4f} "
                f"BS_std={comp['bs_hedge_pnl_std']:.4f} "
                f"improv={comp['improvement_pct']:.1f}%"
            )

        print(f"\n{'=' * 70}\n")


# ===========================================================================
# StandaloneDeepHedger (NEW-2)
# ===========================================================================


class StandaloneDeepHedger:
    """Standalone Deep Hedger for S1 VRP straddles and S10 gamma scalp.

    Trains on real historical spot+IV paths from DuckDB, evaluates OOS,
    and compares against Black-Scholes delta hedging baseline.

    Supports two modes:
    - "straddle": S1 VRP — short straddle, hedge dynamically.
    - "gamma_scalp": S10 — long straddle, scalp gamma via delta hedging.

    Parameters
    ----------
    instrument : str
        Trading instrument (e.g. "NIFTY", "BANKNIFTY").
    mode : str
        "straddle" or "gamma_scalp".
    hedging_interval : str
        "1min", "5min", "hourly", "daily".
    expiry_days : int
        Option lifetime in trading days.
    risk_aversion : float
        CVaR risk aversion parameter.
    """

    def __init__(
        self,
        instrument: str = "NIFTY",
        mode: str = "straddle",
        hedging_interval: str = "5min",
        expiry_days: int = 30,
        risk_aversion: float = 1.0,
    ) -> None:
        self.instrument = instrument.upper()
        self.mode = mode
        self.hedging_interval = hedging_interval
        self.expiry_days = expiry_days
        self.risk_aversion = risk_aversion

        self._agent: Optional[DeepHedgingAgent] = None
        self._train_metrics: Dict[str, Any] = {}
        self._spot_daily: Optional[np.ndarray] = None
        self._iv_daily: Optional[np.ndarray] = None

    def train_on_historical(
        self,
        store,
        ticker: str,
        start_date: str,
        end_date: str,
        train_epochs: int = 200,
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        """Train the deep hedger on real historical spot+IV paths from DuckDB.

        Parameters
        ----------
        store : MarketDataStore
            DuckDB store with nse_index_close and nse_fo_bhavcopy.
        ticker : str
            Underlying (e.g. "NIFTY").
        start_date, end_date : str
            Date range "YYYY-MM-DD".
        train_epochs : int
            Training epochs for the deep hedger.
        batch_size : int
            Mini-batch size.

        Returns
        -------
        dict with training metrics: final_loss, n_paths, avg_iv, avg_spot.
        """
        # Load historical data
        spot_daily, iv_daily, dates = load_historical_iv_paths(
            store, ticker, start_date, end_date
        )
        if len(spot_daily) < self.expiry_days + 1:
            raise ValueError(
                f"Insufficient data: {len(spot_daily)} days < {self.expiry_days + 1} needed"
            )

        self._spot_daily = spot_daily
        self._iv_daily = iv_daily

        # Build multi-day paths
        steps_per_day = DeepHedgingAgent.STEPS_PER_DAY.get(self.hedging_interval, 78)
        rng = np.random.default_rng(42)
        spot_paths, iv_paths = _build_multi_day_paths(
            spot_daily, iv_daily,
            expiry_days=self.expiry_days,
            steps_per_day=steps_per_day,
            risk_free_rate=0.065,
            rng=rng,
        )

        if spot_paths.size == 0:
            raise ValueError("Could not build training paths from historical data")

        # ATM strike = average spot over training period
        strike = float(np.mean(spot_daily))
        avg_iv = float(np.nanmean(iv_daily))

        logger.info(
            "StandaloneDeepHedger: %d paths of %d steps, strike=%.0f, avg_iv=%.3f",
            spot_paths.shape[0], spot_paths.shape[1], strike, avg_iv,
        )

        # Create and train agent
        strategy_type = "straddle" if self.mode in ("straddle", "gamma_scalp") else "call"
        self._agent = DeepHedgingAgent(
            instrument=self.instrument,
            strategy=strategy_type,
            hedging_interval=self.hedging_interval,
            hidden_layers=(128, 64, 32),
            learning_rate=1e-4,
            risk_aversion=self.risk_aversion,
        )

        train_result = self._agent.train_on_paths(
            spot_paths,
            iv_paths=iv_paths,
            num_epochs=train_epochs,
            batch_size=batch_size,
            strike=strike,
            sigma=avg_iv,
            risk_free_rate=0.065,
        )

        self._train_metrics = {
            "final_loss": train_result.get("final_loss", float("nan")),
            "n_paths": spot_paths.shape[0],
            "n_steps": spot_paths.shape[1],
            "strike": strike,
            "avg_iv": avg_iv,
            "avg_spot": float(np.mean(spot_daily)),
            "n_days": len(spot_daily),
            "mode": self.mode,
        }
        return self._train_metrics

    def compare_vs_bs(
        self,
        test_paths: Optional[np.ndarray] = None,
        store=None,
        ticker: Optional[str] = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare deep hedging vs Black-Scholes delta hedging on OOS data.

        Can provide test_paths directly OR load from DuckDB via store.

        Parameters
        ----------
        test_paths : np.ndarray, optional
            Shape (n_paths, n_steps). If None, loads from store.
        store : MarketDataStore, optional
            Required if test_paths is None.
        ticker : str, optional
            Required if test_paths is None.
        test_start, test_end : str, optional
            Required if test_paths is None.

        Returns
        -------
        dict with:
            deep_hedge_pnl_mean, deep_hedge_pnl_std
            bs_hedge_pnl_mean, bs_hedge_pnl_std
            improvement_pct (reduction in std)
            sharpe_dh, sharpe_bs (Sharpe of hedged PnL, ddof=1, sqrt(252))
            mode : str
        """
        if self._agent is None:
            raise RuntimeError("Must call train_on_historical() before compare_vs_bs()")

        strike = self._train_metrics.get("strike", 24000.0)
        avg_iv = self._train_metrics.get("avg_iv", 0.15)

        if test_paths is None:
            if store is None or ticker is None or test_start is None or test_end is None:
                raise ValueError(
                    "Either provide test_paths or (store, ticker, test_start, test_end)"
                )
            spot_daily, iv_daily, _ = load_historical_iv_paths(
                store, ticker, test_start, test_end
            )
            steps_per_day = DeepHedgingAgent.STEPS_PER_DAY.get(self.hedging_interval, 78)
            rng = np.random.default_rng(99)
            test_paths, test_iv_paths = _build_multi_day_paths(
                spot_daily, iv_daily,
                expiry_days=min(self.expiry_days, len(spot_daily) - 1),
                steps_per_day=steps_per_day,
                risk_free_rate=0.065,
                rng=rng,
            )
            if test_paths.size == 0:
                return {"error": "no_test_paths"}

        # Run comparison
        comparison = self._agent.compare_vs_bs(
            test_paths, strike=strike, sigma=avg_iv, risk_free_rate=0.065
        )

        # Compute Sharpe ratios on hedged PnL (daily granularity)
        # Approximate: treat each path as an independent "day" of hedging outcome
        n_paths = comparison["num_paths"]
        dh_sharpe = self._compute_pnl_sharpe(
            comparison["deep_hedge_pnl_mean"],
            comparison["deep_hedge_pnl_std"],
            n_paths,
        )
        bs_sharpe = self._compute_pnl_sharpe(
            comparison["bs_hedge_pnl_mean"],
            comparison["bs_hedge_pnl_std"],
            n_paths,
        )

        result = {
            **comparison,
            "sharpe_dh": dh_sharpe,
            "sharpe_bs": bs_sharpe,
            "mode": self.mode,
            "instrument": self.instrument,
        }

        logger.info(
            "StandaloneDeepHedger OOS: mode=%s, DH_std=%.4f, BS_std=%.4f, "
            "improvement=%.1f%%, Sharpe DH=%.3f, Sharpe BS=%.3f",
            self.mode,
            comparison["deep_hedge_pnl_std"],
            comparison["bs_hedge_pnl_std"],
            comparison["improvement_pct"],
            dh_sharpe, bs_sharpe,
        )
        return result

    def run_gamma_scalp_env(
        self,
        n_episodes: int = 100,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run the deep hedger in the GammaScalpEnv for S10-like evaluation.

        Creates a GammaScalpEnv (long straddle, hedge delta) and runs the
        deep hedger's get_hedge_action() at each step.  Compares against
        a baseline Black-Scholes delta hedging policy.

        Parameters
        ----------
        n_episodes : int
            Number of episodes to simulate.
        seed : int
            Random seed.

        Returns
        -------
        dict with:
            dh_avg_pnl, dh_std_pnl, dh_sharpe
            bs_avg_pnl, bs_std_pnl, bs_sharpe
            n_episodes, avg_steps_per_episode
        """
        if self._agent is None:
            raise RuntimeError("Must call train_on_historical() first")

        from quantlaxmi.models.rl.environments.trading_env import TradingAction

        spot_init = INITIAL_SPOTS.get(self.instrument, 24000.0)
        vol = ANNUALISED_VOLS.get(self.instrument, 0.15)

        env = GammaScalpEnv(
            spot_init=spot_init,
            sigma=vol,
            expiry_days=self.expiry_days,
            cost_per_trade=COST_PER_LEG.get(self.instrument, 3.0),
        )

        dh_pnls: list[float] = []
        bs_pnls: list[float] = []
        total_steps = 0

        for ep in range(n_episodes):
            ep_seed = seed + ep
            state_dh = env.reset(seed=ep_seed)
            state_bs = env.reset(seed=ep_seed)  # same seed for fair comparison

            # DH episode
            dh_cumulative = 0.0
            dh_prev_hedge = 0.0
            state = env.reset(seed=ep_seed)
            step_count = 0
            for _ in range(env.num_steps):
                spot = float(state.prices[0])
                tau = state.features.get("time_to_expiry", 0.01)
                iv_val = state.features.get("iv", vol)
                delta = state.features.get("delta", 0.0)
                gamma = state.features.get("gamma", 0.0)

                # Deep hedge action
                hedge_result = self._agent.get_hedge_action(
                    spot=spot,
                    positions={"strike": spot_init, "tau": tau, "underlying": dh_prev_hedge},
                    greeks={"delta": delta, "gamma": gamma, "iv": iv_val, "rv": vol},
                    regime=None,
                )
                hedge_trade = hedge_result.get("underlying", 0.0)
                new_hedge = dh_prev_hedge + hedge_trade

                action = TradingAction(trade_sizes=np.array([hedge_trade]))
                result = env.step(action)
                state = result.state
                dh_cumulative += result.reward
                dh_prev_hedge = new_hedge
                step_count += 1

                if result.done:
                    break

            dh_pnls.append(dh_cumulative)
            total_steps += step_count

            # BS baseline episode (same random seed)
            state = env.reset(seed=ep_seed)
            bs_cumulative = 0.0
            bs_prev_hedge = 0.0
            for _ in range(env.num_steps):
                spot = float(state.prices[0])
                tau = state.features.get("time_to_expiry", 0.01)
                delta = state.features.get("delta", 0.0)

                # BS delta hedge
                bs_target = -delta  # Hedge against portfolio delta
                hedge_trade = bs_target - bs_prev_hedge

                action = TradingAction(trade_sizes=np.array([hedge_trade]))
                result = env.step(action)
                state = result.state
                bs_cumulative += result.reward
                bs_prev_hedge = bs_target

                if result.done:
                    break

            bs_pnls.append(bs_cumulative)

        dh_arr = np.array(dh_pnls)
        bs_arr = np.array(bs_pnls)

        dh_mean = float(np.mean(dh_arr))
        dh_std = float(np.std(dh_arr, ddof=1)) if len(dh_arr) > 1 else 0.0
        bs_mean = float(np.mean(bs_arr))
        bs_std = float(np.std(bs_arr, ddof=1)) if len(bs_arr) > 1 else 0.0

        dh_sharpe = (dh_mean / dh_std * math.sqrt(252)) if dh_std > 1e-10 else 0.0
        bs_sharpe = (bs_mean / bs_std * math.sqrt(252)) if bs_std > 1e-10 else 0.0

        return {
            "dh_avg_pnl": dh_mean,
            "dh_std_pnl": dh_std,
            "dh_sharpe": dh_sharpe,
            "bs_avg_pnl": bs_mean,
            "bs_std_pnl": bs_std,
            "bs_sharpe": bs_sharpe,
            "n_episodes": n_episodes,
            "avg_steps_per_episode": total_steps / max(n_episodes, 1),
            "mode": self.mode,
            "instrument": self.instrument,
        }

    @staticmethod
    def _compute_pnl_sharpe(mean_pnl: float, std_pnl: float, n_obs: int) -> float:
        """Compute annualized Sharpe from PnL mean and std (ddof=1, sqrt(252))."""
        if std_pnl < 1e-10 or n_obs < 2:
            return 0.0
        return (mean_pnl / std_pnl) * math.sqrt(252)

    def report(self, comparison: Dict[str, Any]) -> None:
        """Print human-readable comparison results."""
        print(f"\n{'=' * 70}")
        title = "STANDALONE DEEP HEDGER (NEW-2)"
        if self.mode == "gamma_scalp":
            title += " — S10 Gamma Scalp"
        else:
            title += " — S1 VRP Straddle"
        print(f"{title:^70}")
        print(f"{'=' * 70}")
        print(f"\nInstrument: {self.instrument}")
        print(f"Mode: {self.mode}")
        print(f"Hedging interval: {self.hedging_interval}")
        print(f"Expiry days: {self.expiry_days}")

        if "error" in comparison:
            print(f"\nError: {comparison['error']}")
            return

        if "dh_avg_pnl" in comparison:
            # GammaScalpEnv results
            print(f"\n{'Deep Hedger':>20} {'BS Delta Hedge':>20}")
            print(f"  Avg PnL: {comparison['dh_avg_pnl']:>13.4f} {comparison['bs_avg_pnl']:>13.4f}")
            print(f"  Std PnL: {comparison['dh_std_pnl']:>13.4f} {comparison['bs_std_pnl']:>13.4f}")
            print(f"  Sharpe:  {comparison['dh_sharpe']:>13.4f} {comparison['bs_sharpe']:>13.4f}")
        else:
            # Path comparison results
            print(f"\n{'Deep Hedger':>20} {'BS Delta Hedge':>20}")
            print(
                f"  Mean PnL: {comparison['deep_hedge_pnl_mean']:>12.4f} "
                f"{comparison['bs_hedge_pnl_mean']:>12.4f}"
            )
            print(
                f"  Std PnL:  {comparison['deep_hedge_pnl_std']:>12.4f} "
                f"{comparison['bs_hedge_pnl_std']:>12.4f}"
            )
            print(f"  Improvement: {comparison['improvement_pct']:.1f}%")
            if "sharpe_dh" in comparison:
                print(
                    f"  Sharpe:   {comparison['sharpe_dh']:>12.4f} "
                    f"{comparison['sharpe_bs']:>12.4f}"
                )

        print(f"\n{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Convenience runners for CLI
# ---------------------------------------------------------------------------


def run_tft_deep_hedging(
    start: str = "2024-01-01",
    end: str = "2026-02-06",
    instrument: str = "NIFTY",
    strategy: str = "straddle",
    hedging_interval: str = "5min",
) -> Dict[str, Any]:
    """Run the TFT + Deep Hedging pipeline (Pattern 3).

    Parameters
    ----------
    start, end : str
        Date range.
    instrument : str
        Trading instrument.
    strategy : str
        "straddle", "call", etc.
    hedging_interval : str
        "1min", "5min", "hourly", "daily".

    Returns
    -------
    dict of pipeline results.
    """
    from quantlaxmi.data.store import MarketDataStore

    cfg = DHPipelineConfig(
        instrument=instrument,
        strategy=strategy,
        hedging_interval=hedging_interval,
    )
    pipeline = TFTDeepHedgingPipeline(cfg=cfg)

    with MarketDataStore() as store:
        results = pipeline.run(start, end, store=store)

    pipeline.report(results)
    return results


def run_standalone_deep_hedger(
    start_train: str = "2024-01-01",
    end_train: str = "2025-06-30",
    start_test: str = "2025-07-01",
    end_test: str = "2026-02-06",
    instrument: str = "NIFTY",
    mode: str = "straddle",
) -> Dict[str, Any]:
    """Run the standalone deep hedger (NEW-2).

    Trains on [start_train, end_train], tests on [start_test, end_test].

    Parameters
    ----------
    start_train, end_train : str
        Training date range.
    start_test, end_test : str
        Testing date range.
    instrument : str
        Trading instrument.
    mode : str
        "straddle" (S1 VRP) or "gamma_scalp" (S10).

    Returns
    -------
    dict with training and comparison metrics.
    """
    from quantlaxmi.data.store import MarketDataStore

    hedger = StandaloneDeepHedger(instrument=instrument, mode=mode)

    with MarketDataStore() as store:
        train_metrics = hedger.train_on_historical(
            store, instrument, start_train, end_train
        )
        comparison = hedger.compare_vs_bs(
            store=store, ticker=instrument,
            test_start=start_test, test_end=end_test,
        )

    hedger.report(comparison)

    return {
        "train_metrics": train_metrics,
        "comparison": comparison,
    }
