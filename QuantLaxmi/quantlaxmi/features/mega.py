"""MegaFeatureBuilder -- aggregate ALL available data sources into a daily feature matrix.

Extracts ~160-250 features from every available data source, aligns them to
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
11. Divergence Flow   (nse_participant_oi DFF)   ~12 features
12. News Sentiment    (FinBERT on headline archive)~11 features
21. Contract Delta    (nse_contract_delta)       ~8 features
22. Delta-Eq OI       (nse_combined_oi_deleq)    ~6 features
23. Pre-Open OFI      (nse_preopen)              ~8 features
24. OI Spurts         (nse_oi_spurts)            ~6 features
25. Crypto Expanded   (Binance FAPI)             ~30 features
26. Cross-Asset       (nse_index_close multi)     ~6 features
27. Macroeconomic     (RBI rate, INR/USD, crude, US10Y) ~8 features

Design principles
-----------------
- Every feature is **causal** -- only uses data available at or before that date.
- Missing data produces NaN (outer-joined across groups).
- Features are roughly normalized (z-scored, ratio-bounded, or [0,1]).
- Feature names carry group prefix for interpretability.

Usage
-----
    from quantlaxmi.features.mega import MegaFeatureBuilder

    builder = MegaFeatureBuilder()
    features, names = builder.build("NIFTY 50", "2025-08-06", "2026-02-06")
    print(features.shape, len(names))
"""

from __future__ import annotations

import hashlib
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

from quantlaxmi.data._paths import DATA_ROOT as _DATA_ROOT, TICK_DIR as _TICK_DIR, \
    KITE_1MIN_DIR as _KITE_1MIN_DIR, BINANCE_DIR as _BINANCE_DIR

from quantlaxmi.features.divergence_flow import DivergenceFlowBuilder, DFFConfig


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
    17. Extended Options (from nse_fo_bhavcopy): ATM IV, gamma exposure, theta, max pain
    18. Divergence Flow (from nse_participant_oi DFF): Helmholtz decomposition signals
    """

    def __init__(
        self,
        market_dir: str | Path | None = None,
        kite_1min_dir: str | Path | None = None,
        binance_dir: str | Path | None = None,
        tick_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        clip_sigma: Optional[float] = 5.0,
    ):
        self._market_dir = Path(market_dir) if market_dir else _DATA_ROOT / "market"
        self._kite_dir = Path(kite_1min_dir) if kite_1min_dir else _KITE_1MIN_DIR
        self._binance_dir = Path(binance_dir) if binance_dir else _BINANCE_DIR
        self._tick_dir = Path(tick_dir) if tick_dir else _TICK_DIR
        self.clip_sigma = clip_sigma

        # Feature cache settings
        self._cache_dir = Path(cache_dir) if cache_dir else _DATA_ROOT / "feature_cache"
        self._use_cache = use_cache

        # Lazy-loaded store
        self._store = None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _version_hash() -> str:
        """Short hash of mega.py source for cache invalidation on code change."""
        src = Path(__file__).read_bytes()
        return hashlib.md5(src).hexdigest()[:8]

    def _cache_key(self, symbol: str, start_date: str, end_date: str) -> str:
        """Build the cache key string."""
        return f"{symbol}_{start_date}_{end_date}_{self._version_hash()}"

    def _cache_path(self, symbol: str, start_date: str, end_date: str) -> Path:
        """Full path to the cached Parquet file."""
        key = self._cache_key(symbol, start_date, end_date)
        return self._cache_dir / f"{key}.parquet"

    def _load_from_cache(
        self, symbol: str, start_date: str, end_date: str,
    ) -> tuple[pd.DataFrame, list[str]] | None:
        """Try to load features from cache. Returns None on miss."""
        if not self._use_cache:
            return None
        path = self._cache_path(symbol, start_date, end_date)
        if not path.exists():
            logger.info("Cache MISS: %s", path.name)
            return None
        try:
            df = pd.read_parquet(path)
            feature_names = list(df.columns)
            logger.info(
                "Cache HIT: %s (%d features, %d rows)",
                path.name, len(feature_names), len(df),
            )
            return df, feature_names
        except Exception:
            logger.exception("Cache read failed for %s, recomputing", path.name)
            return None

    def _save_to_cache(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        df: pd.DataFrame,
    ) -> None:
        """Save features to cache."""
        if not self._use_cache:
            return
        path = self._cache_path(symbol, start_date, end_date)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
            logger.info("Cache SAVE: %s (%d features, %d rows)",
                        path.name, len(df.columns), len(df))
        except Exception:
            logger.exception("Cache save failed for %s", path.name)

    @property
    def store(self):
        if self._store is None:
            from quantlaxmi.data.store import MarketDataStore
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

        # Check cache first
        cached = self._load_from_cache(symbol, start_date, end_date)
        if cached is not None:
            return cached

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
            ("ext_options", self._build_extended_options_features, (ticker, start_date, end_date)),
            ("divergence_flow", self._build_divergence_flow_features, (start_date, end_date)),
            ("crypto_alpha", self._build_crypto_alpha_features, (start_date, end_date)),
            ("news_sentiment", self._build_news_sentiment_features, (start_date, end_date)),
            ("contract_delta", self._build_contract_delta_features, (ticker, start_date, end_date)),
            ("deltaeq_oi", self._build_deltaeq_oi_features, (start_date, end_date)),
            ("preopen_ofi", self._build_preopen_ofi_features, (start_date, end_date)),
            ("oi_spurts", self._build_oi_spurts_features, (start_date, end_date)),
            ("crypto_expanded", self._build_crypto_expanded_features, (start_date, end_date)),
            ("cross_asset", self._build_cross_asset_features, (symbol, start_date, end_date)),
            ("macro", self._build_macro_features, (start_date, end_date)),
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

        # Save to cache for future calls
        self._save_to_cache(symbol, start_date, end_date, combined)

        # Outlier clipping: clip extreme values to [-clip_sigma, clip_sigma]
        # to prevent +10 sigma values from destabilizing attention weights.
        if self.clip_sigma is not None:
            numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                before = combined[numeric_cols]
                n_clipped = (before.abs() >= self.clip_sigma).sum().sum()
                combined[numeric_cols] = before.clip(
                    -self.clip_sigma, self.clip_sigma,
                )
                if n_clipped > 0:
                    logger.info(
                        "Outlier clipping: %d values clipped to [%.1f, %.1f]",
                        n_clipped, -self.clip_sigma, self.clip_sigma,
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
                        bars = bars.sort_index()
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
                    except Exception as e:
                        logger.debug("Bar processing failed for %s: %s", d_str, e)
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
        except Exception as e:
            logger.debug("NSE index query failed: %s", e)

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

        from quantlaxmi.data.tick_loader import TickLoader

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
                except Exception as e:
                    logger.debug("Futures volume query failed for %s: %s", d_str, e)

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
                except Exception as e:
                    logger.debug("Parquet read failed for %s: %s", parquet_files[0], e)
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

    def _build_extended_options_features(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """16 extended options features from DuckDB via OptionsFeatureBuilder.

        Features: ATM IV, skew, PCR, VRP, gamma exposure, theta rate,
        max pain, OI walls, z-scores, term structure — all causal.
        """
        try:
            from quantlaxmi.features.options_features import OptionsFeatureBuilder
        except ImportError:
            logger.warning("OptionsFeatureBuilder not available")
            return pd.DataFrame()

        try:
            # Get spot series from already-loaded price data if available
            builder = OptionsFeatureBuilder(market_dir=self._market_dir)
            return builder.build(ticker, start_date, end_date, store=self.store)
        except Exception:
            logger.exception("Extended options features failed for %s", ticker)
            return pd.DataFrame()

    # ==================================================================
    # GROUP 18: Divergence Flow Field (nse_participant_oi DFF)  (~12 features)
    # ==================================================================

    def _build_divergence_flow_features(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """12 DFF features from Helmholtz decomposition of participant OI flows.

        Features: dff_d_hat, dff_r_hat, dff_z_d, dff_z_r, dff_interaction,
        dff_energy, dff_composite, dff_d_hat_5d, dff_r_hat_5d, dff_energy_z,
        dff_regime, dff_momentum — all causal.
        """
        try:
            builder = DivergenceFlowBuilder(config=DFFConfig())
            return builder.build(start_date, end_date, store=self.store)
        except Exception:
            logger.exception("Divergence flow features failed")
            return pd.DataFrame()

    def _build_crypto_alpha_features(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """7 crypto-specific alpha features from BTC/ETH kline data.

        Features: ca_vol_regime, ca_vol_zscore, ca_mr_z_24, ca_mr_z_72,
        ca_mr_z_168, ca_vwap_dev, ca_vol_profile, ca_ret_skew, ca_ret_kurt,
        ca_mtf_momentum, ca_range_pos — all causal.
        """
        try:
            from quantlaxmi.features.crypto_alpha import (
                VolatilityRegime,
                MeanReversionZ,
                VWAPDeviation,
                VolumeProfile,
                ReturnDistribution,
                MultiTimeframeMomentum,
                RangePosition,
            )

            # Load BTC/ETH kline data from store
            btc = self.store.load("binance_kline", start_date=start_date, end_date=end_date)
            if btc is None or btc.empty:
                return pd.DataFrame()

            # Ensure required columns exist
            needed = {"close", "high", "low", "volume"}
            if not needed.issubset(set(btc.columns)):
                return pd.DataFrame()

            features = []
            prefix = "ca_"

            vol_regime = VolatilityRegime(name=f"{prefix}vol_regime")
            features.append(vol_regime.transform(btc))

            mr_z = MeanReversionZ(name=f"{prefix}mr_z")
            features.append(mr_z.transform(btc))

            vwap = VWAPDeviation(name=f"{prefix}vwap_dev")
            features.append(vwap.transform(btc))

            vol_prof = VolumeProfile(name=f"{prefix}vol_profile")
            features.append(vol_prof.transform(btc))

            ret_dist = ReturnDistribution(name=f"{prefix}ret_dist")
            features.append(ret_dist.transform(btc))

            mtf = MultiTimeframeMomentum(name=f"{prefix}mtf_momentum")
            features.append(mtf.transform(btc))

            rng = RangePosition(name=f"{prefix}range_pos")
            features.append(rng.transform(btc))

            # Concatenate all feature DataFrames
            valid = [f for f in features if f is not None and not f.empty]
            if not valid:
                return pd.DataFrame()

            result = pd.concat(valid, axis=1)
            logger.info("Crypto alpha features: %d columns", len(result.columns))
            return result

        except Exception:
            logger.exception("Crypto alpha features failed")
            return pd.DataFrame()

    def _build_news_sentiment_features(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """11 news sentiment features from FinBERT-scored headline archive.

        Features: ns_sent_mean, ns_sent_std, ns_sent_max, ns_sent_min,
        ns_pos_ratio, ns_neg_ratio, ns_confidence_mean, ns_news_count,
        ns_sent_5d_ma, ns_news_count_5d, ns_sent_momentum — all causal (T+1 lag).
        """
        try:
            from quantlaxmi.features.news_sentiment import NewsSentimentBuilder

            builder = NewsSentimentBuilder()
            return builder.build(start_date, end_date)
        except Exception:
            logger.exception("News sentiment features failed")
            return pd.DataFrame()

    # ==================================================================
    # GROUP 21: Contract Delta (nse_contract_delta)  (~8 features)
    # ==================================================================

    def _build_contract_delta_features(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Contract delta factor features from nse_contract_delta.

        Schema: Date, Symbol, Expiry day, Strike Price, Option Type, Delta Factor.
        Features: cd_mean_delta, cd_mean_delta_chg, cd_put_call_delta_ratio,
        cd_atm_delta_near, cd_delta_skew, cd_n_contracts,
        cd_delta_dispersion, cd_delta_z21 — all causal.
        """
        try:
            raw = self.store.sql(
                """
                SELECT date,
                       "Symbol" as symbol,
                       CAST("Delta Factor" AS DOUBLE) as delta_factor,
                       "Option Type" as option_type,
                       CAST("Strike Price" AS DOUBLE) as strike_price
                FROM nse_contract_delta
                WHERE date BETWEEN ? AND ?
                ORDER BY date
                """,
                [start_date, end_date],
            )
        except Exception:
            logger.warning("nse_contract_delta query failed")
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        raw["date"] = pd.to_datetime(raw["date"])

        # Filter to index contracts matching ticker
        idx_contracts = raw[raw["symbol"].str.upper() == ticker.upper()].copy()
        if idx_contracts.empty:
            logger.debug("No contract delta data for %s, using market-wide", ticker)
            idx_contracts = raw.copy()

        records: list[dict] = []
        for dt, grp in idx_contracts.groupby("date"):
            row: dict = {"date": dt}
            d = grp["delta_factor"].values.astype(np.float64)
            opt_type = grp["option_type"].fillna("").str.upper()

            # Mean delta factor across all contracts
            row["cd_mean_delta"] = float(np.nanmean(d))

            # Number of contracts (market activity proxy)
            row["cd_n_contracts"] = float(len(d))

            # Put/Call delta ratio: mean |put delta| / mean |call delta|
            call_mask = opt_type.str.contains("CE|CALL", na=False)
            put_mask = opt_type.str.contains("PE|PUT", na=False)
            avg_call = float(np.nanmean(np.abs(d[call_mask.values]))) if call_mask.any() else np.nan
            avg_put = float(np.nanmean(np.abs(d[put_mask.values]))) if put_mask.any() else np.nan
            row["cd_put_call_delta_ratio"] = (
                avg_put / avg_call if not np.isnan(avg_call) and avg_call > 0 else np.nan
            )

            # ATM delta: mean delta factor near 0.5 (|delta| between 0.3 and 0.7)
            abs_d = np.abs(d)
            atm_mask = (abs_d > 0.3) & (abs_d < 0.7)
            row["cd_atm_delta_near"] = float(np.nanmean(abs_d[atm_mask])) if atm_mask.any() else np.nan

            # Delta skew: avg call delta factor - avg |put delta factor|
            if not np.isnan(avg_call) and not np.isnan(avg_put):
                row["cd_delta_skew"] = avg_call - avg_put
            else:
                row["cd_delta_skew"] = np.nan

            # Delta dispersion: std of delta factors
            row["cd_delta_dispersion"] = float(np.nanstd(d, ddof=1)) if len(d) > 1 else np.nan

            records.append(row)

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records).set_index("date").sort_index()

        # Daily change in mean delta
        result["cd_mean_delta_chg"] = result["cd_mean_delta"].diff()

        # 21-day z-score of mean delta
        result["cd_delta_z21"] = _rolling_zscore(
            result["cd_mean_delta"].values, 21
        )

        return result

    # ==================================================================
    # GROUP 22: Delta-Eq OI (nse_combined_oi_deleq)  (~6 features)
    # ==================================================================

    def _build_deltaeq_oi_features(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Delta-equivalent OI features from nse_combined_oi_deleq (215 symbols).

        Schema: Date, ISIN, Scrip Name, Symbol, Open Interest,
        Delta Equivalent Open Interest Contract wise,
        Delta Equivalent Open Interest Portfolio wise.

        Features: deleq_total_oi_z21, deleq_total_deltaeq_z21,
        deleq_concentration, deleq_chg_pct, deleq_breadth,
        deleq_oi_vs_deltaeq — all causal.
        """
        try:
            raw = self.store.sql(
                """
                SELECT date,
                       "Symbol" as symbol,
                       CAST("Open Interest" AS DOUBLE) as oi,
                       CAST("Delta Equivalent Open Interest Contract wise" AS DOUBLE) as deltaeq_cw
                FROM nse_combined_oi_deleq
                WHERE date BETWEEN ? AND ?
                ORDER BY date
                """,
                [start_date, end_date],
            )
        except Exception:
            logger.warning("nse_combined_oi_deleq query failed")
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        raw["date"] = pd.to_datetime(raw["date"])

        records: list[dict] = []
        for dt, grp in raw.groupby("date"):
            row: dict = {"date": dt}
            oi = grp["oi"].values.astype(np.float64)
            deltaeq = grp["deltaeq_cw"].values.astype(np.float64)

            total_oi = np.nansum(oi)
            total_deltaeq = np.nansum(np.abs(deltaeq))

            row["deleq_total_oi"] = total_oi
            row["deleq_total_deltaeq"] = total_deltaeq

            # Concentration: HHI-like — top-5 share of total delta-eq OI
            abs_deleq = np.abs(deltaeq)
            abs_deleq_sorted = np.sort(abs_deleq[~np.isnan(abs_deleq)])
            if len(abs_deleq_sorted) >= 5 and total_deltaeq > 0:
                top5 = abs_deleq_sorted[-5:]
                row["deleq_concentration"] = float(np.sum(top5)) / total_deltaeq
            else:
                row["deleq_concentration"] = np.nan

            # Breadth: fraction of symbols with positive delta-eq
            valid_deleq = deltaeq[~np.isnan(deltaeq)]
            if len(valid_deleq) > 0:
                row["deleq_breadth"] = float(np.sum(valid_deleq > 0)) / len(valid_deleq)
            else:
                row["deleq_breadth"] = np.nan

            # OI vs delta-eq ratio (leverage proxy)
            row["deleq_oi_vs_deltaeq"] = (
                total_oi / total_deltaeq if total_deltaeq > 0 else np.nan
            )

            records.append(row)

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records).set_index("date").sort_index()

        # Day-over-day change (using .diff() for safety)
        result["deleq_chg_pct"] = result["deleq_total_deltaeq"].pct_change()

        # Rolling z-scores
        result["deleq_total_oi_z21"] = _rolling_zscore(
            result["deleq_total_oi"].values, 21
        )
        result["deleq_total_deltaeq_z21"] = _rolling_zscore(
            result["deleq_total_deltaeq"].values, 21
        )

        # Drop raw totals (keep z-scored versions)
        result.drop(columns=["deleq_total_oi", "deleq_total_deltaeq"], inplace=True)

        return result

    # ==================================================================
    # GROUP 23: Pre-Open OFI (nse_preopen)  (~8 features)
    # ==================================================================

    def _build_preopen_ofi_features(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Pre-open order flow features from nse_preopen (207 FnO stocks).

        Features: preopen_ofi_mean, preopen_ofi_skew, preopen_spread_mean,
        preopen_spread_std, preopen_iep_gap, preopen_depth_imb,
        preopen_vol_surprise, preopen_participation — all causal (same day).
        """
        try:
            raw = self.store.sql(
                """
                SELECT date, symbol, iep, final_quantity, total_turnover,
                       ofi, bid1_price, ask1_price,
                       bid1_qty, bid2_qty, bid3_qty, bid4_qty, bid5_qty,
                       ask1_qty, ask2_qty, ask3_qty, ask4_qty, ask5_qty
                FROM nse_preopen
                WHERE date BETWEEN ? AND ?
                ORDER BY date
                """,
                [start_date, end_date],
            )
        except Exception:
            logger.warning("nse_preopen query failed")
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        raw["date"] = pd.to_datetime(raw["date"])

        records: list[dict] = []
        for dt, grp in raw.groupby("date"):
            row: dict = {"date": dt}
            ofi_vals = grp["ofi"].values.astype(np.float64)

            # Mean OFI across all FnO stocks
            row["preopen_ofi_mean"] = float(np.nanmean(ofi_vals))

            # OFI skewness
            if len(ofi_vals) > 2:
                m = np.nanmean(ofi_vals)
                s = np.nanstd(ofi_vals, ddof=1)
                if s > 0:
                    row["preopen_ofi_skew"] = float(
                        np.nanmean(((ofi_vals - m) / s) ** 3)
                    )
                else:
                    row["preopen_ofi_skew"] = 0.0
            else:
                row["preopen_ofi_skew"] = np.nan

            # Bid-ask spread
            bid1 = grp["bid1_price"].values.astype(np.float64)
            ask1 = grp["ask1_price"].values.astype(np.float64)
            mid = (bid1 + ask1) / 2.0
            spread = np.where(mid > 0, (ask1 - bid1) / mid, np.nan)
            row["preopen_spread_mean"] = float(np.nanmean(spread))
            row["preopen_spread_std"] = float(np.nanstd(spread, ddof=1)) if len(spread) > 1 else np.nan

            # IEP gap (would need prev close — approximate with bid1 as proxy)
            iep = grp["iep"].values.astype(np.float64)
            if np.nansum(bid1 > 0) > 0:
                gaps = np.abs(iep - bid1) / np.maximum(bid1, 1.0)
                row["preopen_iep_gap"] = float(np.nanmean(gaps))
            else:
                row["preopen_iep_gap"] = np.nan

            # Depth imbalance
            bid_cols = ["bid1_qty", "bid2_qty", "bid3_qty", "bid4_qty", "bid5_qty"]
            ask_cols = ["ask1_qty", "ask2_qty", "ask3_qty", "ask4_qty", "ask5_qty"]
            total_bid_qty = sum(
                grp[c].fillna(0).astype(np.float64).values for c in bid_cols
            )
            total_ask_qty = sum(
                grp[c].fillna(0).astype(np.float64).values for c in ask_cols
            )
            denom = total_bid_qty + total_ask_qty
            imb = np.where(denom > 0, (total_bid_qty - total_ask_qty) / denom, np.nan)
            row["preopen_depth_imb"] = float(np.nanmean(imb))

            # Turnover
            turnover = grp["total_turnover"].values.astype(np.float64)
            row["_preopen_turnover"] = float(np.nansum(turnover))

            # Participation: count of stocks with positive volume
            qty = grp["final_quantity"].values.astype(np.float64)
            row["preopen_participation"] = float(np.nansum(qty > 0))

            records.append(row)

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records).set_index("date").sort_index()

        # Vol surprise: turnover z-scored vs 21d rolling
        if "_preopen_turnover" in result.columns:
            result["preopen_vol_surprise"] = _rolling_zscore(
                result["_preopen_turnover"].values, 21
            )
            result.drop(columns=["_preopen_turnover"], inplace=True)

        return result

    # ==================================================================
    # GROUP 24: OI Spurts (nse_oi_spurts)  (~6 features)
    # ==================================================================

    def _build_oi_spurts_features(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """OI spurt features from nse_oi_spurts (4 build-up/unwinding categories).

        Features: oisp_build_count, oisp_unwind_count, oisp_build_unwind_ratio,
        oisp_net_oi_change, oisp_call_put_build, oisp_max_oi_change_pct — all causal.
        """
        try:
            raw = self.store.sql(
                """
                SELECT date, category, change_in_oi, pchange, latest_oi, prev_oi
                FROM nse_oi_spurts
                WHERE date BETWEEN ? AND ?
                ORDER BY date
                """,
                [start_date, end_date],
            )
        except Exception:
            logger.warning("nse_oi_spurts query failed")
            return pd.DataFrame()

        if raw.empty:
            return pd.DataFrame()

        raw["date"] = pd.to_datetime(raw["date"])

        records: list[dict] = []
        for dt, grp in raw.groupby("date"):
            row: dict = {"date": dt}
            cats = grp["category"].fillna("").str.lower()

            build_mask = cats.str.contains("build", na=False)
            unwind_mask = cats.str.contains("unwind", na=False)

            build_count = int(build_mask.sum())
            unwind_count = int(unwind_mask.sum())

            row["oisp_build_count"] = float(build_count)
            row["oisp_unwind_count"] = float(unwind_count)

            total = build_count + unwind_count
            row["oisp_build_unwind_ratio"] = (
                build_count / total if total > 0 else 0.5
            )

            # Net OI change across all spurt contracts
            oi_chg = grp["change_in_oi"].values.astype(np.float64)
            row["oisp_net_oi_change"] = float(np.nansum(oi_chg))

            # Call vs Put builds
            call_build = cats.str.contains("call", na=False) & build_mask
            put_build = cats.str.contains("put", na=False) & build_mask
            n_call_build = int(call_build.sum())
            n_put_build = int(put_build.sum())
            row["oisp_call_put_build"] = (
                n_call_build / n_put_build if n_put_build > 0 else np.nan
            )

            # Max absolute % change in OI
            pchg = grp["pchange"].values.astype(np.float64)
            row["oisp_max_oi_change_pct"] = (
                float(np.nanmax(np.abs(pchg))) if len(pchg) > 0 else np.nan
            )

            records.append(row)

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records).set_index("date").sort_index()

    # ==================================================================
    # GROUP 25: Crypto Expanded (Binance FAPI)  (~30 features)
    # ==================================================================

    def _build_crypto_expanded_features(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """~30 expanded crypto features: funding rates, OI, L/S positioning,
        altcoin breadth, liquidation proxy.

        Features: fr_mean_8h, fr_std_8h, fr_skew, fr_max_abs, fr_z_score,
        fr_momentum_3d, fr_extreme_count, oi_btc_usd_m, oi_eth_usd_m,
        oi_total_z21, oi_momentum_5d, oi_expanding, oi_concentration,
        ls_ratio_global, ls_ratio_z_5d, ls_top_ratio, ls_divergence,
        ls_taker_buy_pct, ls_taker_z_5d, ls_taker_flip,
        ab_eth_btc_ratio, ab_eth_btc_z20, ab_sol_momentum, ab_altcoin_spread,
        ab_correlation_btc_eth, ab_breadth_2of3,
        liq_extreme_move_z, liq_volume_spike, liq_cascade_ratio,
        liq_bounce_strength — all causal.
        """
        try:
            from quantlaxmi.features.crypto_expanded import build_crypto_expanded_features

            binance_dir = self._binance_dir if self._binance_dir.exists() else None
            return build_crypto_expanded_features(
                start_date=start_date,
                end_date=end_date,
                binance_dir=binance_dir,
                connector=None,  # Live API connector not used in batch mode
            )
        except Exception:
            logger.exception("Crypto expanded features failed")
            return pd.DataFrame()

    # ==================================================================
    # GROUP 26: Cross-Asset Features  (~6 features)
    # ==================================================================

    # All 4 tradeable indices
    _ALL_INDEX_NAMES = [
        "NIFTY 50", "NIFTY BANK", "NIFTY FINANCIAL SERVICES", "NIFTY MIDCAP SELECT",
    ]

    def _build_cross_asset_features(
        self, primary_symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Cross-asset features comparing primary index with other indices.

        Features:
        - ca_nifty_bnf_corr_21d:   21-day rolling correlation of NIFTY vs BANKNIFTY returns
        - ca_nifty_bnf_spread:     (BANKNIFTY_return - NIFTY_return) daily spread
        - ca_nifty_bnf_spread_z21: z-score of spread over 21 days
        - ca_relative_strength:    primary_return / avg(all_index_returns)
        - ca_breadth_corr:         correlation of primary with equal-weight index basket
        - ca_lead_lag_1d:          yesterday's return of the "other" major index

        All features are causal (only use data available at or before each date).
        """
        import pyarrow.parquet as pq

        lookback_start = (
            pd.Timestamp(start_date) - pd.Timedelta(days=100)
        ).strftime("%Y-%m-%d")

        # ---------------------------------------------------------
        # 1. Load close prices for all 4 indices from DuckDB or Kite
        # ---------------------------------------------------------
        index_closes: dict[str, pd.Series] = {}

        for idx_name in self._ALL_INDEX_NAMES:
            try:
                raw = self.store.sql(
                    'SELECT date, "Closing Index Value" as close '
                    'FROM nse_index_close '
                    'WHERE LOWER("Index Name") = LOWER(?) '
                    'AND date BETWEEN ? AND ? '
                    'ORDER BY date',
                    [idx_name, lookback_start, end_date],
                )
                if raw.empty:
                    continue
                raw["date"] = pd.to_datetime(raw["date"])
                raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
                raw = raw.dropna(subset=["close"]).set_index("date").sort_index()
                raw = raw[~raw.index.duplicated(keep="last")]
                index_closes[idx_name] = raw["close"]
            except Exception as e:
                logger.debug("Cross-asset: failed to load %s: %s", idx_name, e)
                continue

        if len(index_closes) < 2:
            logger.warning("Cross-asset: need at least 2 indices, got %d", len(index_closes))
            return pd.DataFrame()

        # ---------------------------------------------------------
        # 2. Build aligned returns DataFrame
        # ---------------------------------------------------------
        prices = pd.DataFrame(index_closes)
        prices = prices.sort_index().ffill()
        returns = np.log(prices / prices.shift(1))

        feats: dict[str, pd.Series] = {}

        # ---------------------------------------------------------
        # 3. NIFTY-BANKNIFTY correlation (21d rolling)
        # ---------------------------------------------------------
        nifty_col = "NIFTY 50"
        bnf_col = "NIFTY BANK"

        if nifty_col in returns.columns and bnf_col in returns.columns:
            feats["ca_nifty_bnf_corr_21d"] = (
                returns[nifty_col]
                .rolling(21, min_periods=21)
                .corr(returns[bnf_col])
            )

            # ---------------------------------------------------------
            # 4. Spread = BANKNIFTY return - NIFTY return
            # ---------------------------------------------------------
            spread = returns[bnf_col] - returns[nifty_col]
            feats["ca_nifty_bnf_spread"] = spread

            # Z-score of spread over 21 days
            spread_mean = spread.rolling(21, min_periods=21).mean()
            spread_std = spread.rolling(21, min_periods=21).std(ddof=1)
            feats["ca_nifty_bnf_spread_z21"] = (
                (spread - spread_mean) / spread_std.replace(0, np.nan)
            )

        # ---------------------------------------------------------
        # 5. Relative strength: primary return / avg(all returns)
        # ---------------------------------------------------------
        primary_upper = primary_symbol.upper().strip()
        if primary_upper in returns.columns:
            avg_return = returns.mean(axis=1)
            # Avoid division by zero
            safe_avg = avg_return.replace(0, np.nan)
            feats["ca_relative_strength"] = returns[primary_upper] / safe_avg
        else:
            feats["ca_relative_strength"] = pd.Series(np.nan, index=returns.index)

        # ---------------------------------------------------------
        # 6. Breadth correlation: correlation of primary with equal-weight basket
        # ---------------------------------------------------------
        if primary_upper in returns.columns:
            other_cols = [c for c in returns.columns if c != primary_upper]
            if other_cols:
                basket_return = returns[other_cols].mean(axis=1)
                feats["ca_breadth_corr"] = (
                    returns[primary_upper]
                    .rolling(21, min_periods=21)
                    .corr(basket_return)
                )
            else:
                feats["ca_breadth_corr"] = pd.Series(np.nan, index=returns.index)
        else:
            feats["ca_breadth_corr"] = pd.Series(np.nan, index=returns.index)

        # ---------------------------------------------------------
        # 7. Lead-lag: yesterday's return of the "other" major index
        # ---------------------------------------------------------
        # NIFTY <-> BANKNIFTY are the two major indices
        if primary_upper == nifty_col and bnf_col in returns.columns:
            feats["ca_lead_lag_1d"] = returns[bnf_col].shift(1)
        elif primary_upper == bnf_col and nifty_col in returns.columns:
            feats["ca_lead_lag_1d"] = returns[nifty_col].shift(1)
        elif nifty_col in returns.columns:
            # For FINNIFTY/MIDCPNIFTY, use NIFTY as the "other" major
            feats["ca_lead_lag_1d"] = returns[nifty_col].shift(1)
        else:
            feats["ca_lead_lag_1d"] = pd.Series(np.nan, index=returns.index)

        # ---------------------------------------------------------
        # 8. Assemble and trim
        # ---------------------------------------------------------
        result = pd.DataFrame(feats, index=returns.index)
        result = result.loc[start_date:end_date]
        return result

    # ==================================================================
    # GROUP 27: Macroeconomic (RBI rate, INR/USD, Crude, US10Y)  (~8 features)
    # ==================================================================

    def _build_macro_features(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """~8 macroeconomic regime features from static CSV files.

        Features: macro_rbi_rate, macro_rbi_rate_chg, macro_inr_usd_return_5d,
        macro_inr_usd_vol_21d, macro_crude_return_5d, macro_crude_z21,
        macro_us10y_level, macro_us10y_chg_5d — all causal.

        Data is loaded from CSV files in ``common/data/macro/``.
        If a file is missing, the corresponding features are NaN (graceful
        degradation).
        """
        try:
            from quantlaxmi.data._paths import DATA_ROOT
            macro_dir = DATA_ROOT / "macro"

            feats: dict[str, pd.Series] = {}

            # We need a date range to build the feature index
            # Use a generous pre-buffer for rolling computations
            buffer_start = str(
                pd.Timestamp(start_date) - pd.Timedelta(days=60)
            )

            # ---- RBI Repo Rate (step function) ----
            rbi_path = macro_dir / "rbi_rates.csv"
            if rbi_path.exists():
                rbi = pd.read_csv(rbi_path, parse_dates=["date"])
                rbi = rbi.set_index("date").sort_index()

                # Create a daily step function by forward-filling
                daily_idx = pd.date_range(buffer_start, end_date, freq="D")
                rbi_daily = rbi["rate"].reindex(daily_idx).ffill()
                # Back-fill the first value for dates before the first decision
                rbi_daily = rbi_daily.bfill()

                feats["macro_rbi_rate"] = rbi_daily

                # Change since the previous rate decision (in bps)
                # Compute the rate at the prior decision date
                rate_at_decisions = rbi["rate"]
                prev_rate = rate_at_decisions.shift(1)
                # Map each day to the change since the last decision
                rbi_chg = (rate_at_decisions - prev_rate) * 100  # bps
                rbi_chg_daily = rbi_chg.reindex(daily_idx).ffill().fillna(0.0)
                feats["macro_rbi_rate_chg"] = rbi_chg_daily
            else:
                logger.warning("Macro: rbi_rates.csv not found at %s", rbi_path)

            # ---- INR/USD ----
            inr_path = macro_dir / "inr_usd.csv"
            if inr_path.exists():
                inr = pd.read_csv(inr_path, parse_dates=["date"])
                inr = inr.set_index("date").sort_index()
                inr_close = inr["close"]

                # 5-day return (depreciation = positive, i.e. INR weakening)
                inr_ret_5d = inr_close.pct_change(5)
                feats["macro_inr_usd_return_5d"] = inr_ret_5d

                # 21-day realized volatility (annualized)
                inr_ret_1d = inr_close.pct_change(1)
                inr_vol_21d = inr_ret_1d.rolling(21, min_periods=21).std(ddof=1) * np.sqrt(252)
                feats["macro_inr_usd_vol_21d"] = inr_vol_21d
            else:
                logger.warning("Macro: inr_usd.csv not found at %s", inr_path)

            # ---- Crude Oil (Brent) ----
            crude_path = macro_dir / "crude_brent.csv"
            if crude_path.exists():
                crude = pd.read_csv(crude_path, parse_dates=["date"])
                crude = crude.set_index("date").sort_index()
                crude_close = crude["close"]

                # 5-day return
                feats["macro_crude_return_5d"] = crude_close.pct_change(5)

                # 21-day z-score of price level
                crude_mean = crude_close.rolling(21, min_periods=21).mean()
                crude_std = crude_close.rolling(21, min_periods=21).std(ddof=1)
                feats["macro_crude_z21"] = (
                    (crude_close - crude_mean)
                    / crude_std.replace(0, np.nan)
                )
            else:
                logger.warning("Macro: crude_brent.csv not found at %s", crude_path)

            # ---- US 10-Year Treasury Yield ----
            us10y_path = macro_dir / "us10y_yield.csv"
            if us10y_path.exists():
                us10y = pd.read_csv(us10y_path, parse_dates=["date"])
                us10y = us10y.set_index("date").sort_index()
                us10y_level = us10y["yield_pct"]

                # Level (raw)
                feats["macro_us10y_level"] = us10y_level

                # 5-day change in bps
                feats["macro_us10y_chg_5d"] = us10y_level.diff(5) * 100  # bps
            else:
                logger.warning("Macro: us10y_yield.csv not found at %s", us10y_path)

            if not feats:
                return pd.DataFrame()

            # Combine all features, align to common index
            result = pd.DataFrame(feats)
            result.index.name = "date"
            result = result.sort_index()

            # Filter to requested date range
            result = result.loc[start_date:end_date]

            logger.info("Macro features: %d columns, %d rows", len(result.columns), len(result))
            return result

        except Exception:
            logger.exception("Macro features failed")
            return pd.DataFrame()
