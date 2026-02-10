"""OptionsFeatureBuilder — 16 causal options features from DuckDB nse_fo_bhavcopy.

Features
--------
optx_atm_iv             ATM implied vol via Newton-Raphson BS inversion
optx_iv_skew_25d        25-delta put IV / 25-delta call IV
optx_pcr_vol            Put / Call volume ratio
optx_pcr_oi             Put / Call OI ratio
optx_term_slope         2nd expiry ATM IV / 1st expiry ATM IV
optx_vrp                Implied vol − Realized vol (20d)
optx_net_gamma          Net dealer gamma exposure (assumes short options)
optx_theta_rate         Portfolio theta / notional (annualised)
optx_oi_pcr_zscore_21d  PCR OI z-score vs 21d rolling
optx_iv_rv_ratio        IV / RV ratio
optx_skew_zscore_21d    Skew z-score vs 21d rolling
optx_gamma_zscore_21d   Gamma exposure z-score vs 21d rolling
optx_put_wall           Strike with max put OI (distance from spot as %)
optx_call_wall          Strike with max call OI (distance from spot as %)
optx_max_pain_dist      Max pain strike distance from spot (as %)
optx_iv_term_contango   Binary: term structure in contango (1) or backwardation (0)

All features are **causal** — only use option chain data available on or before that date.
BS Greeks assume European exercise (adequate for index options).

Data flow
---------
nse_fo_bhavcopy (SQL via DuckDB) → per-date chain → BS Greeks → aggregate features → DataFrame
"""
from __future__ import annotations

import logging
import math
import warnings
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

_SQRT2PI = math.sqrt(2.0 * math.pi)


def _bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """BS d1 parameter."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """European BS option price."""
    if T <= 1e-10:
        return max(0.0, (S - K) if is_call else (K - S))
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * _norm.cdf(d1) - K * math.exp(-r * T) * _norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm.cdf(-d2) - S * _norm.cdf(-d1)


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """BS vega = S * sqrt(T) * phi(d1)."""
    if T <= 1e-10 or sigma <= 1e-8 or S <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    return S * math.sqrt(T) * math.exp(-0.5 * d1 ** 2) / _SQRT2PI


def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """BS gamma = phi(d1) / (S * sigma * sqrt(T))."""
    if T <= 1e-10 or sigma <= 1e-8 or S <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    return math.exp(-0.5 * d1 ** 2) / (_SQRT2PI * S * sigma * math.sqrt(T))


def _bs_theta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """BS theta (per calendar day)."""
    if T <= 1e-10 or sigma <= 1e-8 or S <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    phi_d1 = math.exp(-0.5 * d1 ** 2) / _SQRT2PI
    term1 = -S * phi_d1 * sigma / (2.0 * math.sqrt(T))
    if is_call:
        term2 = -r * K * math.exp(-r * T) * _norm.cdf(d2)
    else:
        term2 = r * K * math.exp(-r * T) * _norm.cdf(-d2)
    return (term1 + term2) / 365.0  # per calendar day


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
    if is_call:
        return _norm.cdf(d1)
    else:
        return _norm.cdf(d1) - 1.0


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
    """Implied vol via Newton-Raphson on BS model.

    Returns NaN if convergence fails or price < intrinsic.
    """
    intrinsic = max(0.0, (S - K) if is_call else (K - S))
    if market_price < intrinsic + 1e-4:
        return float("nan")
    if T <= 1e-10 or S <= 0 or K <= 0:
        return float("nan")

    # Initial guess: Brenner-Subrahmanyam approximation
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

    # Check convergence
    if abs(_bs_price(S, K, T, r, sigma, is_call) - market_price) < tol * 10:
        return sigma
    return float("nan")


def _rolling_zscore_arr(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score of array values."""
    s = pd.Series(arr)
    m = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=1)
    return ((s - m) / sd.replace(0, np.nan)).values


# ============================================================================
# OptionsFeatureBuilder
# ============================================================================


class OptionsFeatureBuilder:
    """Computes 16 causal options features from DuckDB nse_fo_bhavcopy.

    Parameters
    ----------
    market_dir : str or Path, optional
        Root of hive-partitioned parquet data.
    risk_free_rate : float
        Annual risk-free rate for BS model (default 0.065 = RBI repo rate).
    rv_window : int
        Window for realised vol computation (default 20 trading days).
    zscore_window : int
        Window for z-score rolling features (default 21 trading days).
    """

    # Column names in nse_fo_bhavcopy
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

    def __init__(
        self,
        market_dir=None,
        risk_free_rate: float = 0.065,
        rv_window: int = 20,
        zscore_window: int = 21,
    ):
        self._risk_free_rate = risk_free_rate
        self._rv_window = rv_window
        self._zw = zscore_window
        self._market_dir = market_dir

    def build(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        spot_series: Optional[pd.Series] = None,
        store=None,
    ) -> pd.DataFrame:
        """Build 16 options features for a ticker over a date range.

        Parameters
        ----------
        ticker : str
            Underlying symbol (e.g. "NIFTY", "BANKNIFTY").
        start_date, end_date : str
            Date range "YYYY-MM-DD" inclusive.
        spot_series : pd.Series, optional
            Daily spot prices indexed by date. If None, uses close from
            nse_index_close via the store.
        store : MarketDataStore, optional
            DuckDB store. Created from market_dir if not provided.

        Returns
        -------
        pd.DataFrame
            16 columns, indexed by date.
        """
        if store is None:
            from quantlaxmi.data.store import MarketDataStore
            store = MarketDataStore(self._market_dir) if self._market_dir else MarketDataStore()

        # Load spot series if not provided
        if spot_series is None:
            spot_series = self._load_spot_series(store, ticker, start_date, end_date)

        # Load all FnO chain data for the date range
        chains = self._load_chains(store, ticker, start_date, end_date)

        if chains.empty:
            logger.warning("No FnO chain data for %s %s–%s", ticker, start_date, end_date)
            return pd.DataFrame()

        # Compute realized vol
        rv_series = self._compute_realized_vol(spot_series, self._rv_window)

        # Compute per-date features
        records = []
        dates_with_data = sorted(chains["date"].unique())

        for dt in dates_with_data:
            dt_str = str(dt)[:10]
            spot = spot_series.get(dt_str)
            if spot is None or np.isnan(spot) or spot <= 0:
                continue

            chain = chains[chains["date"] == dt]
            rv = rv_series.get(dt_str, float("nan"))

            rec = self._compute_day_features(chain, spot, rv, dt_str)
            if rec is not None:
                rec["date"] = pd.Timestamp(dt_str)
                records.append(rec)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("date").sort_index()

        # Post-process: rolling z-scores
        df = self._add_zscore_features(df)

        # Filter to requested range
        df = df.loc[start_date:end_date]

        return df

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_spot_series(
        self, store, ticker: str, start_date: str, end_date: str,
    ) -> pd.Series:
        """Load daily spot prices from nse_index_close."""
        idx_name = {
            "NIFTY": "Nifty 50",
            "BANKNIFTY": "Nifty Bank",
            "FINNIFTY": "Nifty Financial Services",
            "MIDCPNIFTY": "NIFTY MidCap Select",
        }.get(ticker.upper(), f"Nifty {ticker}")

        # Extend start to get enough for RV window
        extended_start = (
            pd.Timestamp(start_date) - pd.Timedelta(days=self._rv_window * 2)
        ).strftime("%Y-%m-%d")

        try:
            df = store.sql(
                'SELECT date, CAST("Closing Index Value" AS DOUBLE) as close '
                'FROM nse_index_close '
                'WHERE LOWER("Index Name") = LOWER(?) '
                'AND date >= ? AND date <= ? '
                'ORDER BY date',
                [idx_name, extended_start, end_date],
            )
            if df.empty:
                return pd.Series(dtype=float)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            return df.set_index("date")["close"]
        except Exception as e:
            logger.warning("Failed to load spot for %s: %s", ticker, e)
            return pd.Series(dtype=float)

    def _load_chains(
        self, store, ticker: str, start_date: str, end_date: str,
    ) -> pd.DataFrame:
        """Load FnO bhavcopy chain from DuckDB.

        Loads CE + PE for the ticker, nearest 2 expiries on each date.
        """
        # Extend start for z-score warmup
        extended_start = (
            pd.Timestamp(start_date) - pd.Timedelta(days=self._zw * 2)
        ).strftime("%Y-%m-%d")

        try:
            query = (
                'SELECT date, '
                f'"{self._FO_COLS["symbol"]}" AS symbol, '
                f'"{self._FO_COLS["instr_type"]}" AS instr_type, '
                f'CAST("{self._FO_COLS["strike"]}" AS DOUBLE) AS strike, '
                f'"{self._FO_COLS["option_type"]}" AS option_type, '
                f'CAST("{self._FO_COLS["close"]}" AS DOUBLE) AS close, '
                f'CAST("{self._FO_COLS["settle"]}" AS DOUBLE) AS settle, '
                f'CAST("{self._FO_COLS["volume"]}" AS DOUBLE) AS volume, '
                f'CAST("{self._FO_COLS["oi"]}" AS DOUBLE) AS oi, '
                f'"{self._FO_COLS["expiry"]}" AS expiry, '
                f'CAST("{self._FO_COLS["lot_size"]}" AS DOUBLE) AS lot_size '
                'FROM nse_fo_bhavcopy '
                f'WHERE "{self._FO_COLS["symbol"]}" = ? '
                f'AND "{self._FO_COLS["instr_type"]}" IN (\'IDO\', \'STO\') '
                f'AND "{self._FO_COLS["option_type"]}" IN (\'CE\', \'PE\') '
                'AND date >= ? AND date <= ? '
                'ORDER BY date, expiry, strike, option_type'
            )
            df = store.sql(query, [ticker.upper(), extended_start, end_date])
            return df
        except Exception as e:
            logger.warning("Failed to load FnO chain for %s: %s", ticker, e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Realised vol
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_realized_vol(spot_series: pd.Series, window: int) -> pd.Series:
        """Close-to-close realised vol, annualised, ddof=1."""
        if len(spot_series) < 2:
            return pd.Series(dtype=float)
        log_rets = np.log(spot_series / spot_series.shift(1))
        rv = log_rets.rolling(window, min_periods=window).std(ddof=1) * math.sqrt(252)
        return rv

    # ------------------------------------------------------------------
    # Per-date feature computation
    # ------------------------------------------------------------------

    def _compute_day_features(
        self, chain: pd.DataFrame, spot: float, rv: float, dt_str: str,
    ) -> Optional[dict]:
        """Compute all 16 (raw) features for a single date.

        Some features (z-scores) are added in _add_zscore_features() after all dates.
        """
        if chain.empty or spot <= 0:
            return None

        r = self._risk_free_rate

        # Separate calls and puts
        calls = chain[chain["option_type"] == "CE"].copy()
        puts = chain[chain["option_type"] == "PE"].copy()
        if calls.empty or puts.empty:
            return None

        # Get nearest 2 expiries
        expiries = sorted(chain["expiry"].unique())
        if len(expiries) < 1:
            return None

        # First expiry chain
        exp1 = expiries[0]
        chain1 = chain[chain["expiry"] == exp1]
        calls1 = chain1[chain1["option_type"] == "CE"]
        puts1 = chain1[chain1["option_type"] == "PE"]

        # Time to expiry in years
        try:
            exp1_date = pd.Timestamp(exp1)
            dt_date = pd.Timestamp(dt_str)
            dte1 = max(1, (exp1_date - dt_date).days)
            T1 = dte1 / 365.0
        except Exception:
            T1 = 7.0 / 365.0
            dte1 = 7

        # ATM IV (nearest to spot)
        atm_iv = self._compute_atm_iv(calls1, puts1, spot, T1, r)
        if np.isnan(atm_iv):
            atm_iv = self._straddle_approx_iv(calls1, puts1, spot, T1)

        # IV Skew (25-delta put IV / 25-delta call IV)
        iv_skew = self._compute_25d_skew(calls1, puts1, spot, T1, r)

        # PCR Volume
        total_call_vol = max(calls["volume"].sum(), 1.0)
        total_put_vol = puts["volume"].sum()
        pcr_vol = total_put_vol / total_call_vol

        # PCR OI
        total_call_oi = max(calls["oi"].sum(), 1.0)
        total_put_oi = puts["oi"].sum()
        pcr_oi = total_put_oi / total_call_oi

        # Term slope (2nd expiry ATM IV / 1st)
        term_slope = float("nan")
        if len(expiries) >= 2:
            exp2 = expiries[1]
            chain2 = chain[chain["expiry"] == exp2]
            calls2 = chain2[chain2["option_type"] == "CE"]
            puts2 = chain2[chain2["option_type"] == "PE"]
            try:
                exp2_date = pd.Timestamp(exp2)
                T2 = max(1, (exp2_date - dt_date).days) / 365.0
            except Exception:
                T2 = 30.0 / 365.0
            atm_iv2 = self._compute_atm_iv(calls2, puts2, spot, T2, r)
            if not np.isnan(atm_iv2) and not np.isnan(atm_iv) and atm_iv > 1e-6:
                term_slope = atm_iv2 / atm_iv

        # VRP: implied − realised
        vrp = atm_iv - rv if not np.isnan(atm_iv) and not np.isnan(rv) else float("nan")

        # Net gamma exposure
        net_gamma = self._compute_net_gamma_exposure(chain1, spot, T1, r)

        # Theta rate (portfolio theta / spot)
        theta_rate = self._compute_theta_rate(chain1, spot, T1, r)

        # Put wall / Call wall (distance from spot as %)
        put_wall = self._compute_oi_wall(puts1, spot, is_put=True)
        call_wall = self._compute_oi_wall(calls1, spot, is_put=False)

        # Max pain distance
        max_pain_dist = self._compute_max_pain_distance(chain1, spot)

        # IV / RV ratio
        iv_rv_ratio = atm_iv / rv if not np.isnan(rv) and rv > 1e-6 else float("nan")

        # Term contango (binary)
        iv_term_contango = 1.0 if not np.isnan(term_slope) and term_slope > 1.0 else 0.0

        return {
            "optx_atm_iv": atm_iv,
            "optx_iv_skew_25d": iv_skew,
            "optx_pcr_vol": pcr_vol,
            "optx_pcr_oi": pcr_oi,
            "optx_term_slope": term_slope,
            "optx_vrp": vrp,
            "optx_net_gamma": net_gamma,
            "optx_theta_rate": theta_rate,
            "optx_put_wall": put_wall,
            "optx_call_wall": call_wall,
            "optx_max_pain_dist": max_pain_dist,
            "optx_iv_rv_ratio": iv_rv_ratio,
            "optx_iv_term_contango": iv_term_contango,
            # Raw values for z-score computation (added in post-processing)
            "_raw_pcr_oi": pcr_oi,
            "_raw_skew": iv_skew,
            "_raw_gamma": net_gamma,
        }

    # ------------------------------------------------------------------
    # ATM IV computation
    # ------------------------------------------------------------------

    def _compute_atm_iv(
        self, calls: pd.DataFrame, puts: pd.DataFrame,
        spot: float, T: float, r: float,
    ) -> float:
        """ATM IV from the nearest-to-spot straddle via Newton-Raphson."""
        # Find ATM strike (nearest to spot)
        all_strikes = pd.concat([calls["strike"], puts["strike"]]).unique()
        if len(all_strikes) == 0:
            return float("nan")
        atm_strike = all_strikes[np.argmin(np.abs(all_strikes - spot))]

        # Get ATM call and put prices
        atm_call = calls[calls["strike"] == atm_strike]
        atm_put = puts[puts["strike"] == atm_strike]

        ivs = []
        if not atm_call.empty:
            price_c = float(atm_call.iloc[0]["close"])
            if price_c > 0:
                iv_c = _newton_iv(price_c, spot, atm_strike, T, r, True)
                if not np.isnan(iv_c) and 0.01 < iv_c < 5.0:
                    ivs.append(iv_c)

        if not atm_put.empty:
            price_p = float(atm_put.iloc[0]["close"])
            if price_p > 0:
                iv_p = _newton_iv(price_p, spot, atm_strike, T, r, False)
                if not np.isnan(iv_p) and 0.01 < iv_p < 5.0:
                    ivs.append(iv_p)

        if ivs:
            return float(np.mean(ivs))
        return float("nan")

    @staticmethod
    def _straddle_approx_iv(
        calls: pd.DataFrame, puts: pd.DataFrame,
        spot: float, T: float,
    ) -> float:
        """Brenner-Subrahmanyam straddle IV approximation: σ ≈ √(2π/T) × (straddle/S)."""
        all_strikes = pd.concat([calls["strike"], puts["strike"]]).unique()
        if len(all_strikes) == 0:
            return float("nan")
        atm_strike = all_strikes[np.argmin(np.abs(all_strikes - spot))]

        atm_c = calls[calls["strike"] == atm_strike]
        atm_p = puts[puts["strike"] == atm_strike]

        straddle = 0.0
        if not atm_c.empty:
            straddle += float(atm_c.iloc[0]["close"])
        if not atm_p.empty:
            straddle += float(atm_p.iloc[0]["close"])

        if straddle <= 0 or T <= 0:
            return float("nan")

        iv = math.sqrt(2.0 * math.pi / T) * (straddle / spot)
        return iv if 0.01 < iv < 5.0 else float("nan")

    # ------------------------------------------------------------------
    # Skew
    # ------------------------------------------------------------------

    def _compute_25d_skew(
        self, calls: pd.DataFrame, puts: pd.DataFrame,
        spot: float, T: float, r: float,
    ) -> float:
        """25-delta skew: IV of 25-delta put / IV of 25-delta call.

        Finds the strike whose BS delta is closest to ±0.25 and extracts IV.
        """
        # Estimate initial sigma from ATM for delta calculation
        atm_iv = self._compute_atm_iv(calls, puts, spot, T, r)
        if np.isnan(atm_iv):
            atm_iv = 0.20  # fallback

        # Find 25-delta call strike
        iv_25d_call = float("nan")
        if not calls.empty:
            best_dist = 999.0
            for _, row in calls.iterrows():
                K = row["strike"]
                price = row["close"]
                if price <= 0:
                    continue
                delta = abs(_bs_delta(spot, K, T, r, atm_iv, True))
                dist = abs(delta - 0.25)
                if dist < best_dist:
                    best_dist = dist
                    iv_c = _newton_iv(price, spot, K, T, r, True)
                    if not np.isnan(iv_c):
                        iv_25d_call = iv_c

        # Find 25-delta put strike
        iv_25d_put = float("nan")
        if not puts.empty:
            best_dist = 999.0
            for _, row in puts.iterrows():
                K = row["strike"]
                price = row["close"]
                if price <= 0:
                    continue
                delta = abs(_bs_delta(spot, K, T, r, atm_iv, False))
                dist = abs(delta - 0.25)
                if dist < best_dist:
                    best_dist = dist
                    iv_p = _newton_iv(price, spot, K, T, r, False)
                    if not np.isnan(iv_p):
                        iv_25d_put = iv_p

        if not np.isnan(iv_25d_put) and not np.isnan(iv_25d_call) and iv_25d_call > 1e-6:
            return iv_25d_put / iv_25d_call
        return float("nan")

    # ------------------------------------------------------------------
    # Gamma exposure
    # ------------------------------------------------------------------

    def _compute_net_gamma_exposure(
        self, chain: pd.DataFrame, spot: float, T: float, r: float,
    ) -> float:
        """Net dealer gamma exposure.

        Assumes dealers are net short options (standard model).
        net_gamma = sum(OI_call * gamma_call - OI_put * gamma_put) × lot_size

        Normalised by spot² for cross-asset comparability.
        """
        total_gamma = 0.0
        for _, row in chain.iterrows():
            K = row["strike"]
            oi = row["oi"]
            lot = row.get("lot_size", 1.0)
            if pd.isna(lot) or lot <= 0:
                lot = 1.0
            is_call = row["option_type"] == "CE"
            gamma = _bs_gamma(spot, K, T, r, 0.20)  # use 20% vol estimate
            if is_call:
                total_gamma += oi * gamma * lot
            else:
                total_gamma -= oi * gamma * lot

        # Normalise by spot²
        if spot > 0:
            return total_gamma / (spot * spot) * 1e6  # scale for readability
        return 0.0

    # ------------------------------------------------------------------
    # Theta rate
    # ------------------------------------------------------------------

    def _compute_theta_rate(
        self, chain: pd.DataFrame, spot: float, T: float, r: float,
    ) -> float:
        """Aggregate theta / spot (annualised).

        Aggregate theta = sum(OI * theta * lot_size) — sum over nearest expiry.
        """
        total_theta = 0.0
        for _, row in chain.iterrows():
            K = row["strike"]
            oi = row["oi"]
            lot = row.get("lot_size", 1.0)
            if pd.isna(lot) or lot <= 0:
                lot = 1.0
            is_call = row["option_type"] == "CE"
            theta = _bs_theta(spot, K, T, r, 0.20, is_call)
            total_theta += oi * theta * lot

        if spot > 0:
            return total_theta / spot * 252  # annualised rate
        return 0.0

    # ------------------------------------------------------------------
    # OI Walls
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_oi_wall(options: pd.DataFrame, spot: float, is_put: bool) -> float:
        """Distance from spot to the strike with max OI (as % of spot).

        Positive = below spot for puts, above spot for calls.
        """
        if options.empty or spot <= 0:
            return float("nan")

        max_oi_idx = options["oi"].idxmax()
        max_oi_strike = float(options.loc[max_oi_idx, "strike"])
        return (max_oi_strike - spot) / spot * 100.0

    # ------------------------------------------------------------------
    # Max Pain
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_max_pain_distance(chain: pd.DataFrame, spot: float) -> float:
        """Max pain: strike minimising total option buyer PnL. Distance from spot as %.

        At each candidate strike, compute the total intrinsic value
        that option buyers would receive if the underlying settled there.
        The max pain strike minimises this total.
        """
        if chain.empty or spot <= 0:
            return float("nan")

        strikes = sorted(chain["strike"].unique())
        if len(strikes) < 2:
            return float("nan")

        calls = chain[chain["option_type"] == "CE"]
        puts = chain[chain["option_type"] == "PE"]

        min_pain = float("inf")
        max_pain_strike = spot

        for settle_price in strikes:
            # Pain for calls: sum(OI * max(settle - K, 0))
            call_pain = 0.0
            for _, row in calls.iterrows():
                call_pain += row["oi"] * max(0.0, settle_price - row["strike"])

            # Pain for puts: sum(OI * max(K - settle, 0))
            put_pain = 0.0
            for _, row in puts.iterrows():
                put_pain += row["oi"] * max(0.0, row["strike"] - settle_price)

            total_pain = call_pain + put_pain
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = settle_price

        return (max_pain_strike - spot) / spot * 100.0

    # ------------------------------------------------------------------
    # Z-score post-processing
    # ------------------------------------------------------------------

    def _add_zscore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling z-score features from raw values, then drop raw columns."""
        w = self._zw

        if "_raw_pcr_oi" in df.columns:
            df["optx_oi_pcr_zscore_21d"] = _rolling_zscore_arr(
                df["_raw_pcr_oi"].values, w,
            )

        if "_raw_skew" in df.columns:
            df["optx_skew_zscore_21d"] = _rolling_zscore_arr(
                df["_raw_skew"].values, w,
            )

        if "_raw_gamma" in df.columns:
            df["optx_gamma_zscore_21d"] = _rolling_zscore_arr(
                df["_raw_gamma"].values, w,
            )

        # Drop raw columns
        raw_cols = [c for c in df.columns if c.startswith("_raw_")]
        df = df.drop(columns=raw_cols, errors="ignore")

        return df
