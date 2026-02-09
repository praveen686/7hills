"""MegaFeatureBuilder -- aggregate ALL available data sources into a daily feature matrix.

Extracts ~150-200 features from every available data source, aligns them to
daily frequency, and returns a single DataFrame suitable for the TFT model.

Feature groups
--------------
1. Price/Technical   (nse_index_close)          ~40 features
2. Options/Volatility (nse_fo_bhavcopy)         ~20 features
3. Institutional Flow (nse_participant_oi)       ~20 features
4. Market Breadth     (nse_cm_bhavcopy, delivery)~15 features
5. India VIX          (nse_index_close VIX row)  ~10 features
6. Microstructure     (tick data)                ~12 features
7. Intraday Patterns  (kite_1min data)           ~15 features
8. Crypto Signals     (Binance BTC/ETH)          ~15 features
9. Futures Premium    (nse_settlement_prices)    ~10 features
10. FII/DII Activity  (nse_fii_stats)            ~10 features

Design principles
-----------------
- Every feature is **causal** -- only uses data available at or before that date.
- Missing data produces NaN (outer-joined across groups).
- Features are roughly normalized (z-scored, ratio-bounded, or [0,1]).
- Feature names carry group prefix for interpretability.

Usage
-----
    from features.mega import MegaFeatureBuilder

    builder = MegaFeatureBuilder()
    features, names = builder.build("NIFTY 50", "2025-08-06", "2026-02-06")
    print(features.shape, len(names))
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path defaults
# ---------------------------------------------------------------------------

from data._paths import DATA_ROOT as _DATA_ROOT, TICK_DIR as _TICK_DIR, \
    KITE_1MIN_DIR as _KITE_1MIN_DIR, BINANCE_DIR as _BINANCE_DIR


# ---------------------------------------------------------------------------
# Small TA helpers (pure numpy, no external TA library needed)
# ---------------------------------------------------------------------------

def _sma(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=w).mean().values


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, min_periods=span).mean().values


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=w).std(ddof=1).values


def _rolling_zscore(x: np.ndarray, w: int) -> np.ndarray:
    """Rolling z-score: (x - mean) / std."""
    s = pd.Series(x)
    m = s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std(ddof=1)
    return ((s - m) / sd.replace(0, np.nan)).values


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder-smoothed RSI."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full_like(close, np.nan, dtype=np.float64)
    avg_loss = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) < period + 1:
        return np.full_like(close, 50.0)
    avg_gain[period] = np.mean(gain[1: period + 1])
    avg_loss[period] = np.mean(loss[1: period + 1])
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(close: np.ndarray):
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bb_pctb(close: np.ndarray, w: int = 20, n: float = 2.0) -> np.ndarray:
    sma = _sma(close, w)
    std = _rolling_std(close, w)
    upper = sma + n * std
    lower = sma - n * std
    bw = upper - lower
    return np.where(bw > 0, (close - lower) / bw, 0.5)


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a/b with NaN where b==0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(b) > 1e-12, a / b, np.nan)
    return r


# ============================================================================
# MegaFeatureBuilder
# ============================================================================


class MegaFeatureBuilder:
    """Extracts features from ALL data sources and aligns to daily frequency.

    Feature groups:
    1. Price/Technical (from nse_index_close): returns, vol, RSI, MACD, BB, etc.
    2. Options/Volatility (from nse_fo_bhavcopy): ATM IV proxy, skew, PCR, term structure
    3. Institutional Flow (from nse_participant_oi): FII/DII net OI z-scores
    4. Market Breadth (from nse_cm_bhavcopy): advance/decline, delivery %, sector rotation
    5. India VIX (from nse_index_close): level, change, term structure
    6. Microstructure (from tick data): daily VPIN, Kyle's lambda, Amihud, OFI aggregates
    7. Intraday Patterns (from kite_1min data): ORB width, VWAP dev, volume profile
    8. Crypto Signals (from Binance): BTC/ETH overnight return, momentum, correlation
    9. Futures Premium (from nse_fo_bhavcopy IDF + nse_settlement_prices): basis, roll yield
    10. FII/DII Activity (from nse_fii_stats): buy/sell flows, net position
    11. NFO 1-min aggregates (from nfo_1min): futures vol/OI, options PCR
    12. NSE Volatility (from nse_volatility): cross-sectional vol, regime
    13. Participant Volume (from nse_participant_vol): FII/Client volume flows
    14. Market Activity (from nse_market_activity): F&O contract/value ratios
    15. MWPL/Ban Stress (from nse_combined_oi, nse_security_ban): ban proximity, stress
    16. Settlement Basis (from nse_settlement_prices): futures settlement basis
    """

    def __init__(
        self,
        market_dir: str | Path | None = None,
        kite_1min_dir: str | Path | None = None,
        binance_dir: str | Path | None = None,
        tick_dir: str | Path | None = None,
    ):
        self._market_dir = Path(market_dir) if market_dir else _DATA_ROOT / "market"
        self._kite_dir = Path(kite_1min_dir) if kite_1min_dir else _KITE_1MIN_DIR
        self._binance_dir = Path(binance_dir) if binance_dir else _BINANCE_DIR
        self._tick_dir = Path(tick_dir) if tick_dir else _TICK_DIR

        # Lazy-loaded store
        self._store = None

    @property
    def store(self):
        if self._store is None:
            from data.store import MarketDataStore
            self._store = MarketDataStore(market_dir=self._market_dir)
        return self._store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Build the complete feature matrix.

        Parameters
        ----------
        symbol : str
            Index symbol for nse_index_close (e.g. "NIFTY 50", "Nifty Bank").
        start_date, end_date : str
            Date range "YYYY-MM-DD" (inclusive).

        Returns
        -------
        features_df : pd.DataFrame
            Indexed by date (datetime), one row per trading day.
        feature_names : list[str]
            Column names of the features.
        """
        logger.info(
            "MegaFeatureBuilder.build(symbol=%r, %s to %s)",
            symbol, start_date, end_date,
        )

        # Derive the underlying FnO ticker from the index name
        ticker = self._symbol_to_ticker(symbol)

        # Build each feature group independently.
        # Each returns a DataFrame indexed by date (datetime).
        groups: list[pd.DataFrame] = []
        group_names: list[str] = []

        builders = [
            ("price_tech", self._build_price_features, (symbol, start_date, end_date)),
            ("options", self._build_options_features, (ticker, start_date, end_date)),
            ("institutional", self._build_institutional_features, (start_date, end_date)),
            ("breadth", self._build_breadth_features, (ticker, start_date, end_date)),
            ("vix", self._build_vix_features, (start_date, end_date)),
            ("micro", self._build_microstructure_features, (ticker, start_date, end_date)),
            ("intraday", self._build_intraday_features, (ticker, start_date, end_date)),
            ("crypto", self._build_crypto_features, (start_date, end_date)),
            ("futures", self._build_futures_features, (ticker, start_date, end_date)),
            ("fii", self._build_fii_features, (ticker, start_date, end_date)),
            ("nfo1m", self._build_nfo_1min_features, (ticker, start_date, end_date)),
            ("nsevol", self._build_nse_volatility_features, (ticker, start_date, end_date)),
            ("partvol", self._build_participant_vol_features, (start_date, end_date)),
            ("mktact", self._build_market_activity_features, (start_date, end_date)),
            ("mwpl", self._build_mwpl_stress_features, (ticker, start_date, end_date)),
            ("settle", self._build_settlement_features, (ticker, start_date, end_date)),
        ]

        for name, fn, args in builders:
            try:
                df = fn(*args)
                if df is not None and not df.empty:
                    groups.append(df)
                    group_names.append(name)
                    logger.info(
                        "  [%s] %d features, %d rows",
                        name, len(df.columns), len(df),
                    )
                else:
                    logger.warning("  [%s] returned empty", name)
            except Exception:
                logger.exception("  [%s] FAILED", name)

        if not groups:
            raise ValueError("No feature groups produced any data")

        # Outer-join all groups on date
        combined = groups[0]
        for df in groups[1:]:
            combined = combined.join(df, how="outer")

        # Sort by date
        combined = combined.sort_index()

        # Filter to requested range
        combined = combined.loc[start_date:end_date]

        feature_names = list(combined.columns)
        logger.info(
            "MegaFeatureBuilder complete: %d features, %d rows, date range %s to %s",
            len(feature_names),
            len(combined),
            combined.index.min(),
            combined.index.max(),
        )

        return combined, feature_names

    def feature_importance(
        self,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return feature coverage and basic stats.

        Parameters
        ----------
        features_df : pd.DataFrame
            Output of ``build()``.

        Returns
        -------
        pd.DataFrame with columns: feature, non_null_pct, mean, std, group
        """
        records = []
        for col in features_df.columns:
            s = features_df[col]
            group = col.split("_")[0] if "_" in col else "unknown"
            records.append({
                "feature": col,
                "non_null_pct": s.notna().mean(),
                "mean": s.mean(),
                "std": s.std(ddof=1),
                "group": group,
            })
        df = pd.DataFrame(records)
        return df.sort_values("non_null_pct", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal: symbol mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _symbol_to_ticker(symbol: str) -> str:
        """Map index display name to FnO ticker symbol.

        'NIFTY 50' -> 'NIFTY', 'Nifty Bank' -> 'BANKNIFTY', etc.
        """
        s = symbol.upper().strip()
        mapping = {
            "NIFTY 50": "NIFTY",
            "NIFTY50": "NIFTY",
            "NIFTY BANK": "BANKNIFTY",
            "NIFTY FINANCIAL SERVICES": "FINNIFTY",
            "NIFTY MIDCAP SELECT": "MIDCPNIFTY",
        }
        return mapping.get(s, s.replace(" ", ""))

    # ==================================================================
    # GROUP 1: Price / Technical features  (~40 features)
    # ==================================================================

    def _build_price_features(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Price and technical indicator features from Kite spot + nse_index_close.

        Uses Kite 1-min spot data (488 days) as primary OHLCV source.
        Augments with P/E, P/B, Div Yield from nse_index_close where available.
        """
        import pyarrow.parquet as pq

        lookback_start = (
            pd.Timestamp(start_date) - pd.Timedelta(days=400)
        ).strftime("%Y-%m-%d")

        # --- Primary: Kite spot 1-min → daily OHLCV ---
        ticker = self._symbol_to_ticker(symbol)
        kite_spot_map = {
            "NIFTY": "NIFTY_SPOT", "BANKNIFTY": "BANKNIFTY_SPOT",
            "FINNIFTY": "FINNIFTY_SPOT", "MIDCPNIFTY": "MIDCPNIFTY_SPOT",
        }
        spot_name = kite_spot_map.get(ticker)
        kite_df = pd.DataFrame()

        if spot_name:
            spot_dir = self._kite_dir / spot_name
            if spot_dir.exists():
                records = []
                for d_dir in sorted(spot_dir.iterdir()):
                    if not d_dir.is_dir() or not d_dir.name.startswith("date="):
                        continue
                    d_str = d_dir.name[5:]
                    if d_str < lookback_start or d_str > end_date:
                        continue
                    pfiles = list(d_dir.glob("*.parquet"))
                    if not pfiles:
                        continue
                    try:
                        bars = pq.read_table(pfiles[0]).to_pandas()
                        if bars.empty:
                            continue
                        for c in ["open", "high", "low", "close", "volume"]:
                            bars[c] = pd.to_numeric(bars[c], errors="coerce")
                        records.append({
                            "date": d_str,
                            "open": float(bars["open"].iloc[0]),
                            "high": float(bars["high"].max()),
                            "low": float(bars["low"].min()),
                            "close": float(bars["close"].iloc[-1]),
                            "volume": float(bars["volume"].sum()),
                        })
                    except Exception:
                        continue
                if records:
                    kite_df = pd.DataFrame(records)
                    kite_df["date"] = pd.to_datetime(kite_df["date"])
                    kite_df = kite_df.set_index("date").sort_index()
                    logger.info(
                        "Kite spot OHLCV: %d days (%s to %s)",
                        len(kite_df), kite_df.index[0].date(), kite_df.index[-1].date(),
                    )

        # --- Secondary: nse_index_close for PE/PB/DivYield + any missing OHLCV ---
        nse_df = pd.DataFrame()
        try:
            raw = self.store.sql(
                'SELECT date, '
                '"Closing Index Value" as close, '
                '"Open Index Value" as open, '
                '"High Index Value" as high, '
                '"Low Index Value" as low, '
                '"Volume" as volume, '
                '"Turnover (Rs. Cr.)" as turnover, '
                '"P/E" as pe, '
                '"P/B" as pb, '
                '"Div Yield" as div_yield '
                'FROM nse_index_close '
                'WHERE LOWER("Index Name") = LOWER(?) '
                'AND date BETWEEN ? AND ? '
                'ORDER BY date',
                [symbol, lookback_start, end_date],
            )
            if not raw.empty:
                raw["date"] = pd.to_datetime(raw["date"])
                for col in ["close", "open", "high", "low", "volume", "turnover",
                             "pe", "pb", "div_yield"]:
                    raw[col] = pd.to_numeric(raw[col], errors="coerce")
                nse_df = raw.dropna(subset=["close"]).set_index("date").sort_index()
                nse_df = nse_df[~nse_df.index.duplicated(keep="last")]
        except Exception:
            pass

        # --- Merge: Kite OHLCV as base, augment with NSE valuations ---
        if not kite_df.empty:
            df = kite_df.copy()
            # Add turnover, PE, PB, div_yield from NSE where available
            if not nse_df.empty:
                for col in ["turnover", "pe", "pb", "div_yield"]:
                    if col in nse_df.columns:
                        df[col] = nse_df[col].reindex(df.index)
            else:
                for col in ["turnover", "pe", "pb", "div_yield"]:
                    df[col] = np.nan
        elif not nse_df.empty:
            df = nse_df
        else:
            return pd.DataFrame()

        df = df[~df.index.duplicated(keep="last")]

        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        volume = df["volume"].values.astype(np.float64)
        n = len(close)

        feats: dict[str, np.ndarray] = {}
        log_c = np.log(close)

        # -- Returns --
        for lag in [1, 2, 3, 5, 10, 21, 63]:
            ret = np.full(n, np.nan)
            if n > lag:
                ret[lag:] = log_c[lag:] - log_c[:-lag]
            feats[f"px_ret_{lag}d"] = ret

        # -- Realized volatility --
        daily_ret = np.full(n, np.nan)
        daily_ret[1:] = np.diff(log_c)
        for w in [5, 10, 21, 63]:
            feats[f"px_rvol_{w}d"] = _rolling_std(daily_ret, w)

        # -- RSI --
        for p in [7, 14, 21]:
            feats[f"px_rsi_{p}"] = (_rsi(close, p) - 50.0) / 50.0

        # -- MACD --
        macd_l, macd_s, macd_h = _macd(close)
        cstd = _rolling_std(close, 21)
        safe_cstd = np.where(cstd > 0, cstd, 1.0)
        feats["px_macd_line"] = macd_l / safe_cstd
        feats["px_macd_signal"] = macd_s / safe_cstd
        feats["px_macd_hist"] = macd_h / safe_cstd

        # -- Bollinger Band %B --
        feats["px_bb_pctb_20"] = _bb_pctb(close, 20, 2.0)

        # -- Moving average crossovers --
        for fast, slow in [(5, 20), (10, 50), (20, 100)]:
            sf = _sma(close, fast)
            ss = _sma(close, slow)
            safe_ss = np.where(ss > 0, ss, 1.0)
            feats[f"px_ma_cross_{fast}_{slow}"] = (sf - ss) / safe_ss

        # -- Rate of change --
        for w in [5, 10, 21]:
            roc = np.full(n, np.nan)
            if n > w:
                roc[w:] = (close[w:] - close[:-w]) / close[:-w]
            feats[f"px_roc_{w}d"] = roc

        # -- Momentum oscillator: (close - SMA) / std --
        for w in [10, 21]:
            sw = _sma(close, w)
            stw = _rolling_std(close, w)
            safe_stw = np.where(stw > 0, stw, 1.0)
            feats[f"px_mom_osc_{w}d"] = (close - sw) / safe_stw

        # -- Vol ratio short/long --
        rv5 = feats["px_rvol_5d"]
        rv21 = feats["px_rvol_21d"]
        safe_rv21 = np.where((rv21 > 0) & ~np.isnan(rv21), rv21, 1.0)
        feats["px_vol_ratio_5_21"] = np.where(
            ~np.isnan(rv5), rv5 / safe_rv21, np.nan
        )

        # -- 52-week high/low distance --
        if n >= 252:
            rh = pd.Series(close).rolling(252, min_periods=252).max().values
            rl = pd.Series(close).rolling(252, min_periods=252).min().values
            rng = rh - rl
            safe_rng = np.where(rng > 0, rng, 1.0)
            feats["px_dist_52w_high"] = (close - rh) / safe_rng
            feats["px_dist_52w_low"] = (close - rl) / safe_rng
        else:
            feats["px_dist_52w_high"] = np.full(n, np.nan)
            feats["px_dist_52w_low"] = np.full(n, np.nan)

        # -- True Range / ATR normalized --
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1)),
            ),
        )
        tr[0] = high[0] - low[0]
        atr_14 = _ema(tr, 14)
        feats["px_atr_14_norm"] = _safe_div(atr_14, close)

        # -- Volume ratio (20-day) --
        if np.nansum(volume) > 0:
            vsma = _sma(volume, 20)
            safe_vsma = np.where(vsma > 0, vsma, 1.0)
            feats["px_volume_ratio_20d"] = volume / safe_vsma
        else:
            feats["px_volume_ratio_20d"] = np.zeros(n)

        # -- Valuation features --
        pe_arr = df["pe"].values.astype(np.float64)
        pb_arr = df["pb"].values.astype(np.float64)
        dy_arr = df["div_yield"].values.astype(np.float64)
        feats["px_pe_zscore_63d"] = _rolling_zscore(pe_arr, 63)
        feats["px_pb_zscore_63d"] = _rolling_zscore(pb_arr, 63)
        feats["px_div_yield_zscore_63d"] = _rolling_zscore(dy_arr, 63)

        # -- Intraday range --
        feats["px_range_norm"] = _safe_div(high - low, close)

        # -- Gap (open - prev close) --
        gap = np.full(n, np.nan)
        gap[1:] = (df["open"].values[1:] - close[:-1]) / close[:-1]
        feats["px_gap"] = gap

        # Assemble result
        out = pd.DataFrame(feats, index=df.index)
        # Trim to requested date range
        out = out.loc[start_date:end_date]
        return out

    # ==================================================================
    # GROUP 2: Options / Volatility features  (~20 features)
    # ==================================================================

    def _build_options_features(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Options-derived features from nse_fo_bhavcopy (IDO)."""
        try:
            raw = self.store.sql(
                """
                SELECT date,
                       OptnTp,
                       CAST(StrkPric AS DOUBLE) as strike,
                       CAST(ClsPric AS DOUBLE)  as opt_close,
                       CAST(UndrlygPric AS DOUBLE) as underlying,
                       CAST(OpnIntrst AS DOUBLE) as oi,
                       CAST(TtlTradgVol AS DOUBLE) as vol,
                       XpryDt as expiry
                FROM nse_fo_bhavcopy
                WHERE FinInstrmTp = 'IDO'
                  AND TckrSymb = ?
                  AND date BETWEEN ? AND ?
                ORDER BY date, strike
                """,
                [ticker, start_date, end_date],
            )
        except Exception:
            logger.warning("Options query failed for %s", ticker)
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        raw["date"] = pd.to_datetime(raw["date"])
        raw["expiry"] = pd.to_datetime(raw["expiry"])

        # Process per-date
        records: list[dict] = []
        for dt, grp in raw.groupby("date"):
            underlying = grp["underlying"].iloc[0]
            if pd.isna(underlying) or underlying <= 0:
                continue

            # Find nearest (weekly) expiry
            future_expiries = grp.loc[grp["expiry"] > dt, "expiry"]
            if future_expiries.empty:
                continue
            nearest_expiry = future_expiries.min()
            near = grp[grp["expiry"] == nearest_expiry].copy()

            # Next expiry (second nearest)
            second_expiries = future_expiries[future_expiries > nearest_expiry]
            has_second = not second_expiries.empty
            if has_second:
                second_expiry = second_expiries.min()
                far = grp[grp["expiry"] == second_expiry].copy()

            # ATM strike = closest to underlying
            all_strikes = near["strike"].unique()
            if len(all_strikes) == 0:
                continue
            atm_strike = all_strikes[np.argmin(np.abs(all_strikes - underlying))]

            # PCR (put-call ratio) by volume and OI -- nearest expiry
            calls_near = near[near["OptnTp"] == "CE"]
            puts_near = near[near["OptnTp"] == "PE"]
            total_call_vol = calls_near["vol"].sum()
            total_put_vol = puts_near["vol"].sum()
            total_call_oi = calls_near["oi"].sum()
            total_put_oi = puts_near["oi"].sum()

            pcr_vol = (
                total_put_vol / total_call_vol
                if total_call_vol > 0 else np.nan
            )
            pcr_oi = (
                total_put_oi / total_call_oi
                if total_call_oi > 0 else np.nan
            )

            # ATM call and put prices
            atm_calls = calls_near[calls_near["strike"] == atm_strike]
            atm_puts = puts_near[puts_near["strike"] == atm_strike]
            atm_call_price = atm_calls["opt_close"].iloc[0] if not atm_calls.empty else np.nan
            atm_put_price = atm_puts["opt_close"].iloc[0] if not atm_puts.empty else np.nan

            # ATM straddle as fraction of underlying -- proxy for implied vol
            atm_straddle_pct = np.nan
            if not np.isnan(atm_call_price) and not np.isnan(atm_put_price):
                atm_straddle_pct = (atm_call_price + atm_put_price) / underlying

            # OTM skew: compare 5% OTM put IV proxy vs 5% OTM call IV proxy
            # Use option price / underlying as IV proxy
            otm_put_strike = underlying * 0.95
            otm_call_strike = underlying * 1.05
            otm_put_k = all_strikes[np.argmin(np.abs(all_strikes - otm_put_strike))]
            otm_call_k = all_strikes[np.argmin(np.abs(all_strikes - otm_call_strike))]

            otm_puts = puts_near[puts_near["strike"] == otm_put_k]
            otm_calls = calls_near[calls_near["strike"] == otm_call_k]
            otm_put_price = otm_puts["opt_close"].iloc[0] if not otm_puts.empty else np.nan
            otm_call_price = otm_calls["opt_close"].iloc[0] if not otm_calls.empty else np.nan

            # Skew = OTM put price / ATM call price (higher = more skew)
            skew_ratio = np.nan
            if not np.isnan(otm_put_price) and not np.isnan(atm_call_price) and atm_call_price > 0:
                skew_ratio = otm_put_price / atm_call_price

            # Term structure: far ATM straddle / near ATM straddle
            term_structure = np.nan
            if has_second:
                far_calls = far[far["OptnTp"] == "CE"]
                far_puts = far[far["OptnTp"] == "PE"]
                far_strikes = far["strike"].unique()
                if len(far_strikes) > 0:
                    far_atm_k = far_strikes[np.argmin(np.abs(far_strikes - underlying))]
                    far_atm_c = far_calls[far_calls["strike"] == far_atm_k]
                    far_atm_p = far_puts[far_puts["strike"] == far_atm_k]
                    if not far_atm_c.empty and not far_atm_p.empty:
                        far_straddle = far_atm_c["opt_close"].iloc[0] + far_atm_p["opt_close"].iloc[0]
                        near_straddle = (atm_call_price or 0) + (atm_put_price or 0)
                        if near_straddle > 0:
                            term_structure = far_straddle / near_straddle

            # Max pain (strike where total option OI is highest, weighted)
            oi_by_strike = near.groupby("strike")["oi"].sum()
            max_pain_strike = oi_by_strike.idxmax() if not oi_by_strike.empty else np.nan
            max_pain_dist = np.nan
            if not np.isnan(max_pain_strike):
                max_pain_dist = (underlying - max_pain_strike) / underlying

            # OI concentration: top-3 strikes % of total
            top3_oi = oi_by_strike.nlargest(3).sum()
            total_oi_all = oi_by_strike.sum()
            oi_concentration = top3_oi / total_oi_all if total_oi_all > 0 else np.nan

            # Volume-weighted average strike
            vol_by_strike = near.groupby("strike")["vol"].sum()
            vwas = np.nan
            if vol_by_strike.sum() > 0:
                vwas = (vol_by_strike.index.to_numpy(dtype=np.float64) * vol_by_strike.values).sum() / vol_by_strike.sum()
                vwas = (vwas - underlying) / underlying

            # Days to expiry
            dte = (nearest_expiry - dt).days if hasattr(nearest_expiry - dt, "days") else np.nan

            records.append({
                "date": dt,
                "opt_pcr_vol": pcr_vol,
                "opt_pcr_oi": pcr_oi,
                "opt_atm_straddle_pct": atm_straddle_pct,
                "opt_skew_ratio": skew_ratio,
                "opt_term_structure": term_structure,
                "opt_max_pain_dist": max_pain_dist,
                "opt_oi_concentration": oi_concentration,
                "opt_vwas_dist": vwas,
                "opt_dte": dte,
                "opt_total_call_oi": total_call_oi,
                "opt_total_put_oi": total_put_oi,
                "opt_total_vol": total_call_vol + total_put_vol,
            })

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records).set_index("date").sort_index()

        # Add rolling features
        for col in ["opt_pcr_vol", "opt_pcr_oi", "opt_atm_straddle_pct"]:
            if col in result.columns:
                result[f"{col}_zscore_21d"] = _rolling_zscore(result[col].values, 21)
                result[f"{col}_ma5"] = _sma(result[col].values, 5)

        # OI change rate
        for col in ["opt_total_call_oi", "opt_total_put_oi"]:
            if col in result.columns:
                result[f"{col}_chg"] = result[col].pct_change()

        return result

    # ==================================================================
    # GROUP 3: Institutional Flow (nse_participant_oi)  (~20 features)
    # ==================================================================

    def _build_institutional_features(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """FII/DII/Client/Pro OI features from nse_participant_oi."""
        try:
            raw = self.store.sql(
                """
                SELECT * FROM nse_participant_oi
                WHERE date BETWEEN ? AND ?
                ORDER BY date
                """,
                [start_date, end_date],
            )
        except Exception:
            logger.warning("participant_oi query failed")
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        raw["date"] = pd.to_datetime(raw["date"])

        # Numeric conversion
        numeric_cols = [c for c in raw.columns if c not in ("Client Type", "date")]
        for col in numeric_cols:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

        records: list[dict] = []
        for dt, grp in raw.groupby("date"):
            row: dict = {"date": dt}
            for _, r in grp.iterrows():
                ct = str(r["Client Type"]).strip().lower().replace(" ", "_")
                if ct == "total":
                    continue
                prefix = f"inst_{ct}"

                # Net futures position (index + stock)
                fi_long = r.get("Future Index Long", 0) or 0
                fi_short = r.get("Future Index Short", 0) or 0
                fs_long = r.get("Future Stock Long", 0) or 0
                fs_short = r.get("Future Stock Short", 0) or 0
                row[f"{prefix}_fut_idx_net"] = fi_long - fi_short
                row[f"{prefix}_fut_stk_net"] = fs_long - fs_short

                # Option index: call-put OI imbalance
                oic_long = r.get("Option Index Call Long", 0) or 0
                oip_long = r.get("Option Index Put Long", 0) or 0
                oic_short = r.get("Option Index Call Short", 0) or 0
                oip_short = r.get("Option Index Put Short", 0) or 0
                row[f"{prefix}_opt_idx_net_call"] = oic_long - oic_short
                row[f"{prefix}_opt_idx_net_put"] = oip_long - oip_short

                # Total long vs short
                tl = r.get("Total Long Contracts", 0) or 0
                ts = r.get("Total Short Contracts", 0) or 0
                row[f"{prefix}_total_net"] = tl - ts

            records.append(row)

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records).set_index("date").sort_index()

        # Z-score key columns over 21 days
        key_cols = [c for c in result.columns if "fii" in c or "dii" in c]
        for col in key_cols:
            vals = result[col].values.astype(np.float64)
            result[f"{col}_z21"] = _rolling_zscore(vals, 21)

        return result

    # ==================================================================
    # GROUP 4: Market Breadth (nse_cm_bhavcopy + nse_delivery)  (~15 features)
    # ==================================================================

    # Banking sector stocks for BANKNIFTY-specific breadth
    _BANK_STOCKS = {
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK",
        "INDUSINDBK", "BANKBARODA", "PNB", "IDFCFIRSTB", "FEDERALBNK",
        "BANDHANBNK", "AUBANK", "CANBK", "UNIONBANK", "MAHABANK",
        "IOB", "INDIANB", "CENTRALBK", "BANKINDIA", "UCOBANK",
    }

    def _build_breadth_features(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Market breadth from cash market bhavcopy.

        For BANKNIFTY, filters to banking sector stocks for sector-specific
        breadth signals instead of broad-market breadth.
        """
        try:
            raw = self.store.sql(
                """
                SELECT date,
                       TckrSymb as symbol,
                       CAST(ClsPric AS DOUBLE) as close,
                       CAST(PrvsClsgPric AS DOUBLE) as prev_close,
                       CAST(TtlTradgVol AS DOUBLE) as volume,
                       CAST(TtlTrfVal AS DOUBLE) as turnover
                FROM nse_cm_bhavcopy
                WHERE SctySrs IN ('EQ', 'BE')
                  AND date BETWEEN ? AND ?
                ORDER BY date
                """,
                [start_date, end_date],
            )
        except Exception:
            logger.warning("cm_bhavcopy query failed")
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        raw["date"] = pd.to_datetime(raw["date"])

        # Filter to sector-specific stocks for non-NIFTY indices
        if ticker == "BANKNIFTY":
            raw = raw[raw["symbol"].isin(self._BANK_STOCKS)]
            if raw.empty:
                logger.warning("No bank stocks found in cm_bhavcopy for breadth")
                return pd.DataFrame()
            logger.info("Breadth: filtered to %d bank stocks", raw["symbol"].nunique())

        # Delivery data
        try:
            deliv = self.store.sql(
                """
                SELECT date, SYMBOL as symbol,
                       CAST(DELIV_PER AS DOUBLE) as deliv_pct,
                       CAST(TTL_TRD_QNTY AS DOUBLE) as total_qty,
                       CAST(DELIV_QTY AS DOUBLE) as deliv_qty
                FROM nse_delivery
                WHERE SERIES = 'EQ'
                  AND date BETWEEN ? AND ?
                """,
                [start_date, end_date],
            )
            deliv["date"] = pd.to_datetime(deliv["date"])
        except Exception:
            deliv = pd.DataFrame()

        # Also try MTO for broader delivery data
        try:
            mto = self.store.sql(
                """
                SELECT date, symbol,
                       CAST(deliv_pct AS DOUBLE) as mto_deliv_pct
                FROM nse_mto
                WHERE series = 'EQ'
                  AND date BETWEEN ? AND ?
                """,
                [start_date, end_date],
            )
            mto["date"] = pd.to_datetime(mto["date"])
        except Exception:
            mto = pd.DataFrame()

        records: list[dict] = []
        for dt, grp in raw.groupby("date"):
            n_stocks = len(grp)
            if n_stocks == 0:
                continue

            # Advance/Decline
            advancers = (grp["close"] > grp["prev_close"]).sum()
            decliners = (grp["close"] < grp["prev_close"]).sum()
            unchanged = n_stocks - advancers - decliners

            ad_ratio = advancers / max(decliners, 1)
            ad_line = advancers - decliners

            # Returns for breadth
            ret = (grp["close"] - grp["prev_close"]) / grp["prev_close"].replace(0, np.nan)
            mean_ret = ret.mean()
            median_ret = ret.median()
            ret_dispersion = ret.std(ddof=1)

            # Volume breadth
            total_volume = grp["volume"].sum()
            total_turnover = grp["turnover"].sum()

            # Percent of stocks above thresholds
            pct_up_1pct = (ret > 0.01).mean()
            pct_down_1pct = (ret < -0.01).mean()
            pct_up_3pct = (ret > 0.03).mean()
            pct_down_3pct = (ret < -0.03).mean()

            row = {
                "date": dt,
                "brd_ad_ratio": ad_ratio,
                "brd_ad_line": ad_line,
                "brd_pct_advancers": advancers / max(n_stocks, 1),
                "brd_n_stocks": n_stocks,
                "brd_mean_ret": mean_ret,
                "brd_median_ret": median_ret,
                "brd_ret_dispersion": ret_dispersion,
                "brd_pct_up_1pct": pct_up_1pct,
                "brd_pct_down_1pct": pct_down_1pct,
                "brd_pct_up_3pct": pct_up_3pct,
                "brd_pct_down_3pct": pct_down_3pct,
                "brd_total_turnover": total_turnover,
            }

            # Delivery features
            if not deliv.empty:
                day_deliv = deliv[deliv["date"] == dt]
                if not day_deliv.empty:
                    row["brd_mean_deliv_pct"] = day_deliv["deliv_pct"].mean()
                    row["brd_median_deliv_pct"] = day_deliv["deliv_pct"].median()
                    # High delivery = informed buying/selling
                    row["brd_pct_high_deliv"] = (day_deliv["deliv_pct"] > 70).mean()

            records.append(row)

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records).set_index("date").sort_index()

        # Rolling features
        for col in ["brd_ad_ratio", "brd_ad_line", "brd_pct_advancers"]:
            if col in result.columns:
                vals = result[col].values.astype(np.float64)
                result[f"{col}_ma5"] = _sma(vals, 5)

        # Cumulative AD line
        if "brd_ad_line" in result.columns:
            result["brd_cum_ad_line"] = result["brd_ad_line"].cumsum()
            # Normalize cumulative AD by z-scoring
            result["brd_cum_ad_z21"] = _rolling_zscore(
                result["brd_cum_ad_line"].values.astype(np.float64), 21
            )

        # Turnover z-score
        if "brd_total_turnover" in result.columns:
            result["brd_turnover_z21"] = _rolling_zscore(
                result["brd_total_turnover"].values.astype(np.float64), 21
            )

        return result

    # ==================================================================
    # GROUP 5: India VIX  (~10 features)
    # ==================================================================

    def _build_vix_features(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """India VIX features from nse_index_close."""
        lookback_start = (
            pd.Timestamp(start_date) - pd.Timedelta(days=120)
        ).strftime("%Y-%m-%d")

        try:
            df = self.store.sql(
                'SELECT date, '
                '"Closing Index Value" as vix_close, '
                '"Open Index Value" as vix_open, '
                '"High Index Value" as vix_high, '
                '"Low Index Value" as vix_low '
                'FROM nse_index_close '
                'WHERE "Index Name" = \'India VIX\' '
                'AND date BETWEEN ? AND ? '
                'ORDER BY date',
                [lookback_start, end_date],
            )
        except Exception:
            logger.warning("VIX query failed")
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        for col in ["vix_close", "vix_open", "vix_high", "vix_low"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["vix_close"]).set_index("date").sort_index()

        vix = df["vix_close"].values.astype(np.float64)
        n = len(vix)

        feats: dict[str, np.ndarray] = {}

        # VIX level (already bounded ~8-80)
        feats["vix_level"] = vix

        # VIX log and z-score
        feats["vix_log"] = np.log(np.maximum(vix, 0.01))
        feats["vix_zscore_21d"] = _rolling_zscore(vix, 21)
        feats["vix_zscore_63d"] = _rolling_zscore(vix, 63)

        # VIX change
        feats["vix_ret_1d"] = np.concatenate([[np.nan], np.diff(np.log(np.maximum(vix, 0.01)))])
        for lag in [5, 10, 21]:
            ret = np.full(n, np.nan)
            lv = np.log(np.maximum(vix, 0.01))
            if n > lag:
                ret[lag:] = lv[lag:] - lv[:-lag]
            feats[f"vix_ret_{lag}d"] = ret

        # VIX intraday range (normalized by level)
        vix_high = df["vix_high"].values.astype(np.float64)
        vix_low = df["vix_low"].values.astype(np.float64)
        feats["vix_range_norm"] = _safe_div(vix_high - vix_low, vix)

        # VIX mean reversion signal (deviation from 21d SMA)
        vix_sma21 = _sma(vix, 21)
        safe_sma = np.where(vix_sma21 > 0, vix_sma21, 1.0)
        feats["vix_mr_signal"] = (vix - vix_sma21) / safe_sma

        out = pd.DataFrame(feats, index=df.index)
        out = out.loc[start_date:end_date]
        return out

    # ==================================================================
    # GROUP 6: Microstructure (tick data)  (~12 features)
    # ==================================================================

    def _build_microstructure_features(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Daily aggregated microstructure features from tick data.

        Uses near-month FUTURES ticks (which have volume) for volume-based
        features (VPIN, Kyle's lambda, Amihud, OFI), and the INDEX ticks
        for price-only statistics (tick vol, range).
        """
        if not self._tick_dir.exists():
            logger.info("Tick data directory not found; skipping microstructure")
            return pd.DataFrame()

        # Index token for price-only stats
        index_token_map = {
            "NIFTY": 256265,
            "BANKNIFTY": 260105,
            "FINNIFTY": 257801,
            "MIDCPNIFTY": 288009,
        }
        index_token = index_token_map.get(ticker)
        if index_token is None:
            logger.info("No tick token mapping for %s", ticker)
            return pd.DataFrame()

        # Find available tick dates in range
        avail = sorted(
            d.name[5:]
            for d in self._tick_dir.iterdir()
            if d.is_dir()
            and d.name.startswith("date=")
            and start_date <= d.name[5:] <= end_date
        )
        if not avail:
            logger.info("No tick data in date range")
            return pd.DataFrame()

        from data.tick_loader import TickLoader

        try:
            loader = TickLoader(tick_dir=self._tick_dir)
        except FileNotFoundError:
            return pd.DataFrame()

        records: list[dict] = []
        for d_str in avail:
            try:
                # Load index ticks for price-based microstructure
                ticks_idx = loader.load(instrument_token=index_token, date=d_str)
                if ticks_idx.empty or len(ticks_idx) < 100:
                    continue

                dt = pd.Timestamp(d_str)

                # Basic tick stats from index ticks
                n_ticks = len(ticks_idx)
                price_range = ticks_idx["ltp"].max() - ticks_idx["ltp"].min()
                mid_price = ticks_idx["ltp"].median()
                tick_returns = ticks_idx["ltp"].pct_change().dropna()
                tick_vol = tick_returns.std() if len(tick_returns) > 10 else np.nan
                realized_tick_vol = (
                    tick_vol * np.sqrt(len(tick_returns))
                    if not np.isnan(tick_vol) else np.nan
                )

                # Autocorrelation of tick returns (mean-reversion vs momentum at tick level)
                tick_autocorr = (
                    tick_returns.autocorr(lag=1)
                    if len(tick_returns) > 30 else np.nan
                )

                # Tick return skewness and kurtosis
                tick_skew = tick_returns.skew() if len(tick_returns) > 30 else np.nan
                tick_kurt = tick_returns.kurtosis() if len(tick_returns) > 30 else np.nan

                # Roll spread from tick autocovariance (works without volume)
                dp = ticks_idx["ltp"].diff().fillna(0)
                dp_lag = dp.shift(1)
                autocov = dp.rolling(500, min_periods=50).cov(dp_lag)
                last_autocov = autocov.dropna().iloc[-1] if autocov.notna().any() else np.nan
                roll_spread = 2.0 * np.sqrt(max(-last_autocov, 0)) if not np.isnan(last_autocov) else np.nan
                roll_spread_norm = roll_spread / max(mid_price, 1) if not np.isnan(roll_spread) else np.nan

                # Price jump detection: count of 5-sigma moves in tick returns
                if tick_vol > 0 and not np.isnan(tick_vol):
                    n_jumps = (tick_returns.abs() > 5 * tick_vol).sum()
                else:
                    n_jumps = 0

                row: dict = {
                    "date": dt,
                    "micro_n_ticks": n_ticks,
                    "micro_tick_vol": tick_vol,
                    "micro_realized_vol": realized_tick_vol,
                    "micro_range_norm": price_range / max(mid_price, 1),
                    "micro_tick_autocorr": tick_autocorr,
                    "micro_tick_skew": tick_skew,
                    "micro_tick_kurt": tick_kurt,
                    "micro_roll_spread_norm": roll_spread_norm,
                    "micro_n_jumps": n_jumps,
                }

                records.append(row)

            except Exception:
                logger.debug("Tick processing failed for %s", d_str, exc_info=True)
                continue

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records).set_index("date").sort_index()

        # Rolling features
        for col in ["micro_tick_vol", "micro_roll_spread_norm", "micro_n_ticks"]:
            if col in result.columns:
                result[f"{col}_z21"] = _rolling_zscore(
                    result[col].values.astype(np.float64), 21
                )

        return result

    # ==================================================================
    # GROUP 7: Intraday Patterns (kite_1min data)  (~15 features)
    # ==================================================================

    def _build_intraday_features(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Intraday pattern features from Kite 1-minute spot + futures data."""
        # Map ticker to Kite directory name
        kite_map = {
            "NIFTY": "NIFTY_SPOT",
            "BANKNIFTY": "BANKNIFTY_SPOT",
            "FINNIFTY": "FINNIFTY_SPOT",
            "MIDCPNIFTY": "MIDCPNIFTY_SPOT",
        }
        # Futures directory for volume data (spot has volume=0 for indices)
        kite_fut_map = {
            "NIFTY": "NIFTY_FUT_DAILY",
            "BANKNIFTY": "BANKNIFTY_FUT_DAILY",
            "FINNIFTY": "FINNIFTY_FUT_DAILY",
            "MIDCPNIFTY": "MIDCPNIFTY_FUT_DAILY",
        }
        spot_name = kite_map.get(ticker)
        fut_name = kite_fut_map.get(ticker)
        if spot_name is None:
            logger.info("No Kite spot mapping for %s", ticker)
            return pd.DataFrame()

        spot_dir = self._kite_dir / spot_name
        fut_dir = self._kite_dir / fut_name if fut_name else None
        if not spot_dir.exists():
            logger.info("Kite spot dir not found: %s", spot_dir)
            return pd.DataFrame()

        import pyarrow.parquet as pq

        records: list[dict] = []
        for d_dir in sorted(spot_dir.iterdir()):
            if not d_dir.is_dir() or not d_dir.name.startswith("date="):
                continue
            d_str = d_dir.name[5:]
            if d_str < start_date or d_str > end_date:
                continue

            parquet_files = list(d_dir.glob("*.parquet"))
            if not parquet_files:
                continue

            try:
                df = pq.read_table(parquet_files[0]).to_pandas()
                if df.empty or len(df) < 10:
                    continue

                # Ensure numeric
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                dt = pd.Timestamp(d_str)
                o = df["open"].values.astype(np.float64)
                h = df["high"].values.astype(np.float64)
                lo = df["low"].values.astype(np.float64)
                c = df["close"].values.astype(np.float64)
                vol = df["volume"].values.astype(np.float64)

                day_open = o[0]
                day_close = c[-1]
                day_high = np.nanmax(h)
                day_low = np.nanmin(lo)

                if day_open <= 0:
                    continue

                # ORB (Opening Range Breakout): first 15 min range
                orb_bars = min(15, len(df))
                orb_high = np.nanmax(h[:orb_bars])
                orb_low = np.nanmin(lo[:orb_bars])
                orb_width = (orb_high - orb_low) / day_open

                # Was ORB broken up or down during the day?
                rest_high = np.nanmax(h[orb_bars:]) if len(h) > orb_bars else orb_high
                rest_low = np.nanmin(lo[orb_bars:]) if len(lo) > orb_bars else orb_low
                orb_break_up = 1.0 if rest_high > orb_high else 0.0
                orb_break_down = 1.0 if rest_low < orb_low else 0.0

                # VWAP deviation at close
                # Use spot volume if available, else fall back to price-only VWAP
                has_vol = np.nansum(vol) > 0
                if has_vol:
                    typical = (h + lo + c) / 3.0
                    vwap = np.nansum(typical * vol) / np.nansum(vol)
                    vwap_dev = (day_close - vwap) / day_open
                else:
                    # Price-only TWAP as VWAP proxy
                    typical = (h + lo + c) / 3.0
                    vwap = np.nanmean(typical)
                    vwap_dev = (day_close - vwap) / day_open

                # Close position within day range
                day_range = day_high - day_low
                close_position = (day_close - day_low) / max(day_range, 0.01)

                # Intraday momentum: (close - open) / open
                intraday_ret = (day_close - day_open) / day_open

                # First vs second half return
                mid_idx = len(c) // 2
                first_half_ret = (c[mid_idx] - day_open) / day_open if mid_idx > 0 else 0
                second_half_ret = (day_close - c[mid_idx]) / c[mid_idx] if mid_idx > 0 and c[mid_idx] > 0 else 0

                # Volume profile from nfo_1min futures data (via DuckDB)
                vol_first_30_frac = np.nan
                vol_last_30_frac = np.nan
                try:
                    fut_1min = self.store.sql(
                        """
                        SELECT date, volume FROM nfo_1min
                        WHERE name = ? AND instrument_type = 'FUT'
                          AND date = ?
                        ORDER BY date
                        """,
                        [ticker, d_str],
                    )
                    if not fut_1min.empty and len(fut_1min) >= 30:
                        fvol = pd.to_numeric(fut_1min["volume"], errors="coerce").values.astype(np.float64)
                        total_fvol = np.nansum(fvol)
                        if total_fvol > 0:
                            vol_first_30_frac = np.nansum(fvol[:30]) / total_fvol
                            vol_last_30_frac = np.nansum(fvol[-30:]) / total_fvol
                except Exception:
                    pass

                # Intraday realized vol (from 1-min returns)
                c_series = pd.Series(c)
                min_returns = c_series.pct_change().dropna()
                intraday_rvol = min_returns.std() * np.sqrt(375) if len(min_returns) > 10 else np.nan

                # Max drawdown intraday
                cum_max = np.maximum.accumulate(c)
                dd = (cum_max - c) / cum_max
                max_intraday_dd = np.nanmax(dd)

                # Mean reversion within day: how much of the gap was filled
                gap = day_open - c[-1] if len(c) > 1 else 0
                # Not well-defined without prev close; use close position instead

                records.append({
                    "date": dt,
                    "intra_orb_width": orb_width,
                    "intra_orb_break_up": orb_break_up,
                    "intra_orb_break_down": orb_break_down,
                    "intra_vwap_dev": vwap_dev,
                    "intra_close_position": close_position,
                    "intra_ret": intraday_ret,
                    "intra_first_half_ret": first_half_ret,
                    "intra_second_half_ret": second_half_ret,
                    "intra_vol_first_30_frac": vol_first_30_frac,
                    "intra_vol_last_30_frac": vol_last_30_frac,
                    "intra_rvol_1min": intraday_rvol,
                    "intra_max_dd": max_intraday_dd,
                    "intra_range_norm": day_range / day_open,
                })

            except Exception:
                logger.debug("Intraday processing failed for %s", d_str, exc_info=True)
                continue

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records).set_index("date").sort_index()

        # Rolling features
        for col in ["intra_orb_width", "intra_vwap_dev", "intra_rvol_1min"]:
            if col in result.columns:
                result[f"{col}_ma5"] = _sma(result[col].values.astype(np.float64), 5)

        return result

    # ==================================================================
    # GROUP 8: Crypto Signals (Binance BTC/ETH)  (~15 features)
    # ==================================================================

    def _build_crypto_features(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Crypto sentiment/momentum features from Binance daily data."""
        if not self._binance_dir.exists():
            logger.info("Binance data directory not found")
            return pd.DataFrame()

        import pyarrow.parquet as pq

        all_records: dict[str, pd.DataFrame] = {}

        for coin in ["BTCUSDT", "ETHUSDT"]:
            coin_dir = self._binance_dir / coin / "1d"
            if not coin_dir.exists():
                continue

            frames: list[pd.DataFrame] = []
            for d_dir in sorted(coin_dir.iterdir()):
                if not d_dir.is_dir() or not d_dir.name.startswith("date="):
                    continue
                d_str = d_dir.name[5:]
                if d_str < start_date or d_str > end_date:
                    continue

                parquet_files = list(d_dir.glob("*.parquet"))
                if not parquet_files:
                    continue
                try:
                    df = pq.read_table(parquet_files[0]).to_pandas()
                    df["_date_str"] = d_str
                    frames.append(df)
                except Exception:
                    continue

            if not frames:
                continue

            combined = pd.concat(frames, ignore_index=True)
            for col in ["open", "high", "low", "close", "volume", "trades",
                         "taker_buy_volume"]:
                if col in combined.columns:
                    combined[col] = pd.to_numeric(combined[col], errors="coerce")

            combined["date"] = pd.to_datetime(combined["_date_str"])
            combined = combined.sort_values("date").set_index("date")
            combined = combined[~combined.index.duplicated(keep="last")]
            all_records[coin] = combined

        if not all_records:
            return pd.DataFrame()

        feats_frames: list[pd.DataFrame] = []

        for coin, df in all_records.items():
            prefix = "crypto_btc" if "BTC" in coin else "crypto_eth"
            c = df["close"].values.astype(np.float64)
            n = len(c)

            feats: dict[str, np.ndarray] = {}
            log_c = np.log(np.maximum(c, 0.01))

            # Returns
            for lag in [1, 5, 21]:
                ret = np.full(n, np.nan)
                if n > lag:
                    ret[lag:] = log_c[lag:] - log_c[:-lag]
                feats[f"{prefix}_ret_{lag}d"] = ret

            # Volatility
            daily_ret = np.full(n, np.nan)
            daily_ret[1:] = np.diff(log_c)
            feats[f"{prefix}_rvol_21d"] = _rolling_std(daily_ret, 21)

            # RSI 14
            feats[f"{prefix}_rsi_14"] = (_rsi(c, 14) - 50) / 50

            # Volume change
            vol = df["volume"].values.astype(np.float64)
            vol_sma = _sma(vol, 20)
            safe_vol_sma = np.where(vol_sma > 0, vol_sma, 1.0)
            feats[f"{prefix}_vol_ratio"] = np.clip(vol / safe_vol_sma, 0, 10)

            # Taker buy ratio (proxy for aggression)
            if "taker_buy_volume" in df.columns:
                tbv = df["taker_buy_volume"].values.astype(np.float64)
                feats[f"{prefix}_taker_buy_pct"] = _safe_div(tbv, vol)

            feats_frames.append(
                pd.DataFrame(feats, index=df.index)
            )

        if not feats_frames:
            return pd.DataFrame()

        result = feats_frames[0]
        for f in feats_frames[1:]:
            result = result.join(f, how="outer")

        result = result.loc[start_date:end_date]
        return result

    # ==================================================================
    # GROUP 9: Futures Premium (nse_fo_bhavcopy IDF)  (~10 features)
    # ==================================================================

    def _build_futures_features(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Futures basis, roll yield, and OI features."""
        try:
            raw = self.store.sql(
                """
                SELECT date,
                       XpryDt as expiry,
                       CAST(ClsPric AS DOUBLE)     as fut_close,
                       CAST(UndrlygPric AS DOUBLE) as underlying,
                       CAST(OpnIntrst AS DOUBLE)   as oi,
                       CAST(TtlTradgVol AS DOUBLE) as vol
                FROM nse_fo_bhavcopy
                WHERE FinInstrmTp = 'IDF'
                  AND TckrSymb = ?
                  AND date BETWEEN ? AND ?
                ORDER BY date, XpryDt
                """,
                [ticker, start_date, end_date],
            )
        except Exception:
            logger.warning("Futures query failed for %s", ticker)
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        raw["date"] = pd.to_datetime(raw["date"])
        raw["expiry"] = pd.to_datetime(raw["expiry"])

        records: list[dict] = []
        for dt, grp in raw.groupby("date"):
            underlying = grp["underlying"].iloc[0]
            if pd.isna(underlying) or underlying <= 0:
                continue

            # Sort by expiry
            grp = grp.sort_values("expiry")
            future_contracts = grp[grp["expiry"] > dt]

            if future_contracts.empty:
                continue

            # Near-month (M1) and next-month (M2)
            m1 = future_contracts.iloc[0]
            m1_basis = (m1["fut_close"] - underlying) / underlying
            m1_dte = (m1["expiry"] - dt).days
            m1_annualized_basis = np.nan
            if m1_dte > 0:
                m1_annualized_basis = m1_basis * (365 / m1_dte)

            row = {
                "date": dt,
                "fut_m1_basis": m1_basis,
                "fut_m1_basis_ann": m1_annualized_basis,
                "fut_m1_dte": m1_dte,
                "fut_m1_oi": m1["oi"],
                "fut_m1_vol": m1["vol"],
            }

            if len(future_contracts) >= 2:
                m2 = future_contracts.iloc[1]
                m2_basis = (m2["fut_close"] - underlying) / underlying
                m2_dte = (m2["expiry"] - dt).days

                # Roll yield = M2 basis - M1 basis
                roll_yield = m2_basis - m1_basis

                # Calendar spread
                calendar_spread = (m2["fut_close"] - m1["fut_close"]) / underlying

                row["fut_m2_basis"] = m2_basis
                row["fut_roll_yield"] = roll_yield
                row["fut_calendar_spread"] = calendar_spread
                row["fut_m2_oi"] = m2["oi"]

            # Total futures OI
            row["fut_total_oi"] = grp["oi"].sum()
            row["fut_total_vol"] = grp["vol"].sum()

            records.append(row)

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records).set_index("date").sort_index()

        # Rolling features
        for col in ["fut_m1_basis", "fut_m1_basis_ann", "fut_roll_yield"]:
            if col in result.columns:
                vals = result[col].values.astype(np.float64)
                result[f"{col}_z21"] = _rolling_zscore(vals, 21)
                result[f"{col}_ma5"] = _sma(vals, 5)

        # OI change
        if "fut_total_oi" in result.columns:
            result["fut_total_oi_chg"] = result["fut_total_oi"].pct_change()

        return result

    # ==================================================================
    # GROUP 10: FII/DII Activity (nse_fii_stats)  (~10 features)
    # ==================================================================

    def _build_fii_features(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """FII/DII buying/selling from nse_fii_stats.

        Filters to self-referential categories only: e.g. for BANKNIFTY,
        uses BANKNIFTY FUTURES + INDEX aggregates, but NOT NIFTY-specific.
        """
        try:
            raw = self.store.sql(
                """
                SELECT date, category,
                       CAST(buy_contracts AS DOUBLE) as buy,
                       CAST(sell_contracts AS DOUBLE) as sell,
                       CAST(buy_amt_cr AS DOUBLE) as buy_amt,
                       CAST(sell_amt_cr AS DOUBLE) as sell_amt,
                       CAST(oi_contracts AS DOUBLE) as oi,
                       CAST(oi_amt_cr AS DOUBLE) as oi_amt
                FROM nse_fii_stats
                WHERE date BETWEEN ? AND ?
                ORDER BY date
                """,
                [start_date, end_date],
            )
        except Exception:
            logger.warning("fii_stats query failed")
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        raw["date"] = pd.to_datetime(raw["date"])

        records: list[dict] = []
        for dt, grp in raw.groupby("date"):
            row: dict = {"date": dt}

            for _, r in grp.iterrows():
                cat = str(r["category"]).strip().upper()

                if cat == "INDEX FUTURES":
                    prefix = "fii_idx_fut"
                elif cat == "INDEX OPTIONS":
                    prefix = "fii_idx_opt"
                elif cat == "NIFTY FUTURES":
                    # Skip NIFTY-specific when building for BANKNIFTY
                    if ticker == "BANKNIFTY":
                        continue
                    prefix = "fii_nifty_fut"
                elif cat == "NIFTY OPTIONS":
                    if ticker == "BANKNIFTY":
                        continue
                    prefix = "fii_nifty_opt"
                elif cat == "BANKNIFTY FUTURES":
                    # Skip BANKNIFTY-specific when building for NIFTY
                    if ticker == "NIFTY":
                        continue
                    prefix = "fii_bnf_fut"
                else:
                    continue

                buy = r["buy"] or 0
                sell = r["sell"] or 0
                net = buy - sell
                total = buy + sell
                buy_ratio = buy / max(total, 1)

                row[f"{prefix}_net"] = net
                row[f"{prefix}_buy_ratio"] = buy_ratio
                row[f"{prefix}_oi"] = r["oi"] or 0

            records.append(row)

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records).set_index("date").sort_index()

        # Z-score and rolling
        for col in result.columns:
            if "_net" in col:
                vals = result[col].values.astype(np.float64)
                result[f"{col}_z21"] = _rolling_zscore(vals, 21)
                result[f"{col}_cum5"] = pd.Series(vals).rolling(5, min_periods=1).sum().values

        return result

    # ==================================================================
    # GROUP 11: NFO 1-min aggregate features (316 days)
    # ==================================================================

    def _build_nfo_1min_features(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Daily aggregates from nfo_1min (316 days of FUT+options 1-min bars).

        This extends coverage beyond the 125-day nse_fo_bhavcopy with
        futures volume/OI, options PCR, and intraday volatility from
        the 1-min NFO data.
        """
        try:
            # Futures daily aggregates
            fut_raw = self.store.sql(
                """
                SELECT date,
                       SUM(CAST(volume AS DOUBLE)) as fut_vol,
                       MAX(CAST(oi AS DOUBLE)) as fut_oi,
                       MAX(CAST(close AS DOUBLE)) as fut_close,
                       MIN(CAST(low AS DOUBLE)) as fut_low,
                       MAX(CAST(high AS DOUBLE)) as fut_high
                FROM nfo_1min
                WHERE name = ? AND instrument_type = 'FUT'
                  AND date BETWEEN ? AND ?
                GROUP BY date
                ORDER BY date
                """,
                [ticker, start_date, end_date],
            )
        except Exception:
            logger.warning("nfo_1min FUT query failed for %s", ticker)
            return pd.DataFrame()

        if fut_raw.empty:
            return pd.DataFrame()

        fut_raw["date"] = pd.to_datetime(fut_raw["date"])
        for col in ["fut_vol", "fut_oi", "fut_close", "fut_low", "fut_high"]:
            fut_raw[col] = pd.to_numeric(fut_raw[col], errors="coerce")
        fut_raw = fut_raw.set_index("date").sort_index()

        feats: dict[str, np.ndarray] = {}
        n = len(fut_raw)
        fc = fut_raw["fut_close"].values.astype(np.float64)
        fv = fut_raw["fut_vol"].values.astype(np.float64)
        foi = fut_raw["fut_oi"].values.astype(np.float64)

        # Futures vol z-score
        vol_sma = _sma(fv, 20)
        safe_vol_sma = np.where(vol_sma > 0, vol_sma, 1.0)
        feats["nfo_fut_vol_ratio"] = fv / safe_vol_sma

        # OI change
        feats["nfo_fut_oi_chg"] = np.concatenate([[np.nan], np.diff(foi) / np.maximum(foi[:-1], 1)])
        feats["nfo_fut_oi_z21"] = _rolling_zscore(foi, 21)

        # Futures intraday range
        fh = fut_raw["fut_high"].values.astype(np.float64)
        fl = fut_raw["fut_low"].values.astype(np.float64)
        feats["nfo_fut_range_norm"] = _safe_div(fh - fl, fc)

        # Futures realized vol from closes
        log_fc = np.log(np.maximum(fc, 1.0))
        daily_ret = np.full(n, np.nan)
        daily_ret[1:] = np.diff(log_fc)
        feats["nfo_fut_rvol_5d"] = _rolling_std(daily_ret, 5)
        feats["nfo_fut_rvol_21d"] = _rolling_std(daily_ret, 21)

        # --- Options PCR from nfo_1min (316 days vs 125 from bhavcopy) ---
        try:
            opt_pcr = self.store.sql(
                """
                SELECT date, instrument_type,
                       SUM(CAST(volume AS DOUBLE)) as total_vol,
                       SUM(CAST(oi AS DOUBLE)) as total_oi
                FROM nfo_1min
                WHERE name = ? AND instrument_type IN ('CE', 'PE')
                  AND date BETWEEN ? AND ?
                GROUP BY date, instrument_type
                ORDER BY date
                """,
                [ticker, start_date, end_date],
            )
            if not opt_pcr.empty:
                opt_pcr["date"] = pd.to_datetime(opt_pcr["date"])
                opt_pcr["total_vol"] = pd.to_numeric(opt_pcr["total_vol"], errors="coerce")
                opt_pcr["total_oi"] = pd.to_numeric(opt_pcr["total_oi"], errors="coerce")

                calls = opt_pcr[opt_pcr["instrument_type"] == "CE"].set_index("date")
                puts = opt_pcr[opt_pcr["instrument_type"] == "PE"].set_index("date")

                # PCR by volume and OI
                pcr_vol = puts["total_vol"].reindex(fut_raw.index) / calls["total_vol"].reindex(fut_raw.index).replace(0, np.nan)
                pcr_oi = puts["total_oi"].reindex(fut_raw.index) / calls["total_oi"].reindex(fut_raw.index).replace(0, np.nan)

                feats["nfo_pcr_vol"] = pcr_vol.values
                feats["nfo_pcr_oi"] = pcr_oi.values
                feats["nfo_pcr_vol_z21"] = _rolling_zscore(pcr_vol.values.astype(np.float64), 21)
                feats["nfo_pcr_oi_z21"] = _rolling_zscore(pcr_oi.values.astype(np.float64), 21)

                # Total options volume ratio
                total_opt_vol = (calls["total_vol"].reindex(fut_raw.index).fillna(0) +
                                 puts["total_vol"].reindex(fut_raw.index).fillna(0))
                opt_vol_sma = _sma(total_opt_vol.values.astype(np.float64), 20)
                safe_opt_sma = np.where(opt_vol_sma > 0, opt_vol_sma, 1.0)
                feats["nfo_opt_vol_ratio"] = total_opt_vol.values / safe_opt_sma
        except Exception:
            logger.debug("nfo_1min options PCR query failed", exc_info=True)

        result = pd.DataFrame(feats, index=fut_raw.index)
        return result

    # ==================================================================
    # GROUP 12: NSE Volatility (nse_volatility)  (~12 features)
    # ==================================================================

    def _build_nse_volatility_features(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Cross-sectional and index-specific volatility from nse_volatility."""
        try:
            # Index-specific vol (use underlying symbol mapping)
            idx_sym_map = {
                "NIFTY": "NIFTY",
                "BANKNIFTY": "BANKNIFTY",
                "FINNIFTY": "FINNIFTY",
                "MIDCPNIFTY": "MIDCPNIFTY",
            }
            idx_sym = idx_sym_map.get(ticker, ticker)

            raw = self.store.sql(
                """
                SELECT date,
                       AVG(CAST(applicable_ann_vol AS DOUBLE)) as mkt_avg_vol,
                       STDDEV(CAST(applicable_ann_vol AS DOUBLE)) as mkt_vol_disp,
                       MEDIAN(CAST(applicable_ann_vol AS DOUBLE)) as mkt_med_vol,
                       PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY CAST(applicable_ann_vol AS DOUBLE))
                           as mkt_vol_p90,
                       COUNT(*) as n_symbols
                FROM nse_volatility
                WHERE date BETWEEN ? AND ?
                  AND CAST(applicable_ann_vol AS DOUBLE) > 0
                GROUP BY date
                ORDER BY date
                """,
                [start_date, end_date],
            )
        except Exception:
            logger.warning("nse_volatility cross-sectional query failed")
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.set_index("date").sort_index()

        feats: dict[str, np.ndarray] = {}
        avg_vol = raw["mkt_avg_vol"].values.astype(np.float64)
        med_vol = raw["mkt_med_vol"].values.astype(np.float64)
        disp = raw["mkt_vol_disp"].values.astype(np.float64)
        p90 = raw["mkt_vol_p90"].values.astype(np.float64)

        feats["nsevol_mkt_avg"] = avg_vol
        feats["nsevol_mkt_median"] = med_vol
        feats["nsevol_mkt_dispersion"] = disp
        feats["nsevol_mkt_p90"] = p90
        feats["nsevol_mkt_avg_z21"] = _rolling_zscore(avg_vol, 21)
        feats["nsevol_mkt_disp_z21"] = _rolling_zscore(disp, 21)
        # Vol regime: ratio of short-term vs long-term avg vol
        feats["nsevol_regime_5_21"] = _safe_div(_sma(avg_vol, 5), _sma(avg_vol, 21))
        # Tail risk: P90 / median
        feats["nsevol_tail_ratio"] = _safe_div(p90, np.maximum(med_vol, 0.01))

        # Index-specific vol
        try:
            idx_vol = self.store.sql(
                """
                SELECT date,
                       CAST(applicable_ann_vol AS DOUBLE) as idx_ann_vol,
                       CAST(underlying_log_return AS DOUBLE) as idx_log_ret
                FROM nse_volatility
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date
                """,
                [idx_sym, start_date, end_date],
            )
            if not idx_vol.empty:
                idx_vol["date"] = pd.to_datetime(idx_vol["date"])
                idx_vol = idx_vol.set_index("date").sort_index().reindex(raw.index)
                iv = idx_vol["idx_ann_vol"].values.astype(np.float64)
                feats["nsevol_idx_ann"] = iv
                feats["nsevol_idx_z21"] = _rolling_zscore(iv, 21)
                feats["nsevol_idx_vs_mkt"] = _safe_div(iv, avg_vol)
                feats["nsevol_idx_ret"] = idx_vol["idx_log_ret"].values.astype(np.float64)
        except Exception:
            logger.debug("nse_volatility index-specific query failed for %s", idx_sym)

        return pd.DataFrame(feats, index=raw.index)

    # ==================================================================
    # GROUP 13: Participant Volume (nse_participant_vol)  (~15 features)
    # ==================================================================

    def _build_participant_vol_features(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """FII/Client/DII/Pro trading volume features from nse_participant_vol."""
        try:
            raw = self.store.sql(
                """
                SELECT * FROM nse_participant_vol
                WHERE date BETWEEN ? AND ?
                ORDER BY date
                """,
                [start_date, end_date],
            )
        except Exception:
            logger.warning("nse_participant_vol query failed")
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        raw["date"] = pd.to_datetime(raw["date"])

        records: list[dict] = []
        for dt, grp in raw.groupby("date"):
            row: dict = {"date": dt}
            for _, r in grp.iterrows():
                ct = str(r["Client Type"]).strip().upper()
                if ct == "FII":
                    prefix = "pvol_fii"
                elif ct == "DII":
                    prefix = "pvol_dii"
                elif ct == "CLIENT":
                    prefix = "pvol_client"
                elif ct == "PRO":
                    prefix = "pvol_pro"
                else:
                    continue

                fut_idx_net = (r.get("Future Index Long", 0) or 0) - (r.get("Future Index Short", 0) or 0)
                opt_idx_call = (r.get("Option Index Call Long", 0) or 0) - (r.get("Option Index Call Short", 0) or 0)
                opt_idx_put = (r.get("Option Index Put Long", 0) or 0) - (r.get("Option Index Put Short", 0) or 0)
                total_long = r.get("Total Long Contracts", 0) or 0
                total_short = r.get("Total Short Contracts", 0) or 0

                row[f"{prefix}_fut_idx_net"] = fut_idx_net
                row[f"{prefix}_opt_call_net"] = opt_idx_call
                row[f"{prefix}_opt_put_net"] = opt_idx_put
                row[f"{prefix}_total_net"] = total_long - total_short
                if total_long + total_short > 0:
                    row[f"{prefix}_long_ratio"] = total_long / (total_long + total_short)

            records.append(row)

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records).set_index("date").sort_index()

        # Z-scores for key columns
        for col in result.columns:
            if "_net" in col:
                vals = result[col].values.astype(np.float64)
                result[f"{col}_z21"] = _rolling_zscore(vals, 21)

        return result

    # ==================================================================
    # GROUP 14: Market Activity (nse_market_activity)  (~10 features)
    # ==================================================================

    def _build_market_activity_features(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Market-wide F&O activity features from nse_market_activity."""
        try:
            raw = self.store.sql(
                """
                SELECT date, "Product" as product,
                       CAST("No of Contracts" AS DOUBLE) as contracts,
                       CAST("Traded Value (Rs. Crs.)" AS DOUBLE) as value_cr
                FROM nse_market_activity
                WHERE date BETWEEN ? AND ?
                ORDER BY date
                """,
                [start_date, end_date],
            )
        except Exception:
            logger.warning("nse_market_activity query failed")
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        raw["date"] = pd.to_datetime(raw["date"])

        records: list[dict] = []
        for dt, grp in raw.groupby("date"):
            row: dict = {"date": dt}
            total_contracts = 0.0
            total_value = 0.0
            for _, r in grp.iterrows():
                prod = str(r["product"]).strip()
                contracts = r["contracts"] or 0
                value = r["value_cr"] or 0
                total_contracts += contracts
                total_value += value

                if prod == "Index Futures":
                    row["mktact_idx_fut_contracts"] = contracts
                    row["mktact_idx_fut_value"] = value
                elif prod == "Stock Futures":
                    row["mktact_stk_fut_contracts"] = contracts
                elif prod == "Index Options":
                    row["mktact_idx_opt_contracts"] = contracts
                    row["mktact_idx_opt_value"] = value
                elif prod == "Stock Options":
                    row["mktact_stk_opt_contracts"] = contracts

            row["mktact_total_contracts"] = total_contracts
            row["mktact_total_value"] = total_value
            # Options vs futures ratio
            idx_opt = row.get("mktact_idx_opt_contracts", 0)
            idx_fut = row.get("mktact_idx_fut_contracts", 0)
            row["mktact_opt_fut_ratio"] = idx_opt / max(idx_fut, 1)
            records.append(row)

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records).set_index("date").sort_index()

        # Z-scores for volume metrics
        for col in ["mktact_total_contracts", "mktact_total_value", "mktact_opt_fut_ratio"]:
            if col in result.columns:
                vals = result[col].values.astype(np.float64)
                result[f"{col}_z21"] = _rolling_zscore(vals, 21)
                result[f"{col}_ratio5"] = _safe_div(
                    _sma(vals, 5), np.maximum(_sma(vals, 21), 1e-12)
                )

        return result

    # ==================================================================
    # GROUP 15: MWPL / Ban Stress (nse_combined_oi + nse_security_ban)  (~8 features)
    # ==================================================================

    def _build_mwpl_stress_features(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """MWPL utilization and security ban stress from nse_combined_oi + nse_security_ban."""
        feats_list: list[pd.DataFrame] = []

        # --- MWPL utilization across all FnO stocks ---
        try:
            raw = self.store.sql(
                """
                SELECT "Date" as date,
                       AVG(CAST("Open Interest" AS DOUBLE) / NULLIF(CAST("MWPL" AS DOUBLE), 0))
                           as avg_mwpl_util,
                       MAX(CAST("Open Interest" AS DOUBLE) / NULLIF(CAST("MWPL" AS DOUBLE), 0))
                           as max_mwpl_util,
                       SUM(CASE WHEN "Limit for Next Day" = 'No Fresh Positions' THEN 1 ELSE 0 END)
                           as n_banned,
                       COUNT(*) as n_total
                FROM nse_combined_oi
                WHERE "Date" BETWEEN ? AND ?
                GROUP BY "Date"
                ORDER BY "Date"
                """,
                [start_date, end_date],
            )
            if not raw.empty:
                raw["date"] = pd.to_datetime(raw["date"])
                raw = raw.set_index("date").sort_index()
                result = pd.DataFrame(index=raw.index)
                result["mwpl_avg_util"] = raw["avg_mwpl_util"].values.astype(np.float64)
                result["mwpl_max_util"] = raw["max_mwpl_util"].values.astype(np.float64)
                result["mwpl_n_banned"] = raw["n_banned"].values.astype(np.float64)
                result["mwpl_ban_pct"] = _safe_div(
                    raw["n_banned"].values.astype(np.float64),
                    raw["n_total"].values.astype(np.float64),
                )
                result["mwpl_avg_util_z21"] = _rolling_zscore(
                    result["mwpl_avg_util"].values, 21
                )
                result["mwpl_n_banned_z21"] = _rolling_zscore(
                    result["mwpl_n_banned"].values, 21
                )
                feats_list.append(result)
        except Exception:
            logger.warning("nse_combined_oi MWPL query failed")

        # --- Security ban count from nse_security_ban ---
        try:
            ban = self.store.sql(
                """
                SELECT date, COUNT(*) as ban_count
                FROM nse_security_ban
                WHERE date BETWEEN ? AND ?
                GROUP BY date
                ORDER BY date
                """,
                [start_date, end_date],
            )
            if not ban.empty:
                ban["date"] = pd.to_datetime(ban["date"])
                ban = ban.set_index("date").sort_index()
                ban_df = pd.DataFrame(index=ban.index)
                ban_df["ban_count"] = ban["ban_count"].values.astype(np.float64)
                ban_df["ban_count_z21"] = _rolling_zscore(ban_df["ban_count"].values, 21)
                feats_list.append(ban_df)
        except Exception:
            logger.debug("nse_security_ban query failed")

        if not feats_list:
            return pd.DataFrame()

        combined = feats_list[0]
        for df in feats_list[1:]:
            combined = combined.join(df, how="outer")
        return combined

    # ==================================================================
    # GROUP 16: Settlement Basis (nse_settlement_prices)  (~8 features)
    # ==================================================================

    def _build_settlement_features(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Futures basis and term structure from nse_settlement_prices."""
        try:
            raw = self.store.sql(
                """
                SELECT "DATE" as date, "INSTRUMENT" as instrument,
                       "UNDERLYING" as underlying,
                       "EXPIRY DATE" as expiry,
                       CAST("MTM SETTLEMENT PRICE" AS DOUBLE) as settle_price
                FROM nse_settlement_prices
                WHERE "UNDERLYING" = ?
                  AND "INSTRUMENT" = 'FUTIDX'
                  AND "DATE" BETWEEN ? AND ?
                ORDER BY "DATE", "EXPIRY DATE"
                """,
                [ticker, start_date, end_date],
            )
        except Exception:
            logger.warning("nse_settlement_prices query failed for %s", ticker)
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        raw["date"] = pd.to_datetime(raw["date"])
        raw["settle_price"] = pd.to_numeric(raw["settle_price"], errors="coerce")
        raw = raw.dropna(subset=["settle_price"])

        records: list[dict] = []
        for dt, grp in raw.groupby("date"):
            grp_sorted = grp.sort_values("expiry")
            row: dict = {"date": dt}
            prices = grp_sorted["settle_price"].values

            if len(prices) >= 1:
                row["settle_near"] = prices[0]
            if len(prices) >= 2:
                row["settle_far"] = prices[1]
                # Calendar spread (far - near)
                row["settle_cal_spread"] = prices[1] - prices[0]
                # Normalized spread
                if prices[0] > 0:
                    row["settle_cal_spread_pct"] = (prices[1] - prices[0]) / prices[0]
            if len(prices) >= 3:
                row["settle_3rd"] = prices[2]
                # Term structure curvature
                row["settle_curvature"] = prices[2] - 2 * prices[1] + prices[0]

            records.append(row)

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records).set_index("date").sort_index()

        # Rolling z-scores
        if "settle_cal_spread_pct" in result.columns:
            vals = result["settle_cal_spread_pct"].values.astype(np.float64)
            result["settle_spread_z21"] = _rolling_zscore(vals, 21)
            result["settle_spread_ma5"] = _sma(vals, 5)

        return result
