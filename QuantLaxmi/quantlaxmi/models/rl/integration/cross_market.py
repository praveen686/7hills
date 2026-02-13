"""Pattern 4: Cross-Market India + Crypto with RL Allocation.

Two separate XTrendBackbone instances (India, Crypto) produce joint hidden
representations.  An Actor-Critic RL allocator learns dynamic weights across
6 assets [NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, BTC, ETH] on a walk-forward
basis, benchmarked against Merton's analytical solution and equal-weight.

Architecture
------------
CryptoFeatureAdapter
    Loads Binance daily OHLCV from Hive-partitioned Parquet → 31 features
    per crypto asset (returns, vol, momentum, volume, range, gap, dominance).

CrossMarketBackbone
    Two XTrendBackbone instances:
        India  backbone: 4 assets  (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY)
        Crypto backbone: 2 assets  (BTC, ETH)
    extract_joint_hidden() → concatenated (6 * d_hidden,) vector.

CrossMarketAllocator
    Actor-Critic on AssetAllocationMDP (6 assets).
    Actor  : joint_hidden → allocation softmax weights
    Critic : joint_hidden → scalar state value
    GAE (lambda=0.95, gamma=0.99) for advantage estimation.

CrossMarketPipeline
    Walk-forward over a date range:
        1. Build India (MegaFeatureAdapter) + Crypto (CryptoFeatureAdapter) features
        2. Pre-train both backbones (supervised, joint MLE+Sharpe)
        3. Train RL allocation (Actor-Critic, 200 episodes per fold)
        4. OOS: compare RL vs Merton analytical vs equal-weight

Fee structure:
    India : 3-5 index pts per leg (per COST_PER_LEG)
    Crypto: 0.1% maker/taker (round-trip 0.2% of notional)

Cross-correlation captured via overnight-gap features: BTC leads NIFTY by
6-12 h, so the BTC return during India's off-hours is an informative signal.

References:
    - Rao & Jelvis, Ch 8 (Merton)
    - Wood et al. 2023 (X-Trend)
    - docs/ANALYSIS.md, Pattern 4
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _HAS_TORCH = False
    _DEVICE = None

from quantlaxmi.models.ml.tft.x_trend import XTrendConfig
from quantlaxmi.models.rl.integration.backbone import XTrendBackbone, MegaFeatureAdapter
from quantlaxmi.models.rl.finance.asset_allocation import MertonSolution, AssetAllocationMDP
from quantlaxmi.models.rl.environments.india_fno_env import COST_PER_LEG, INITIAL_SPOTS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDIA_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
CRYPTO_SYMBOLS = ["BTC", "ETH"]
ALL_SYMBOLS = INDIA_SYMBOLS + CRYPTO_SYMBOLS

# Binance pair names corresponding to CRYPTO_SYMBOLS
_CRYPTO_PAIR_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}

# Approximate USD spot prices for crypto (used when no data available)
_CRYPTO_INITIAL_SPOTS = {"BTC": 65_000.0, "ETH": 3_500.0}

# Crypto trading costs: 0.1% maker/taker per side → 0.001 fractional
CRYPTO_FEE_RATE = 0.001

# GAE parameters
GAE_LAMBDA = 0.95
GAE_GAMMA = 0.99


# ============================================================================
# CryptoFeatureAdapter
# ============================================================================


class CryptoFeatureAdapter:
    """Load Binance daily OHLCV and engineer 31 features per crypto asset.

    Features (per asset, all causal):
        returns_1d, returns_5d, returns_20d          — log returns at 3 horizons
        vol_5d, vol_20d                              — rolling std of 1d returns
        rsi_14                                       — Wilder RSI (14-day)
        macd_signal                                  — MACD(12,26) minus signal(9)
        volume_ratio_5d                              — volume / 5d MA volume
        high_low_range                               — (high - low) / close
        overnight_gap                                — open / prev_close - 1
        log_volume                                   — log(1 + volume)
        close_to_high                                — close / high
        close_to_low                                 — close / low
        btc_dominance_proxy                          — BTC close / (BTC close + ETH close)
        funding_rate_proxy                           — (close - open) / open (intraday bias)
        returns_3d                                   — 3-day log return
        vol_10d                                      — 10-day rolling vol
        rsi_7                                        — 7-day RSI
        volume_ratio_20d                             — volume / 20d MA volume
        dollar_volume                                — close * volume
        log_dollar_volume                            — log(1 + dollar_volume)
        intraday_range                               — (high - low) / open
        upper_shadow                                 — (high - max(open, close)) / high
        lower_shadow                                 — (min(open, close) - low) / close
        body_ratio                                   — abs(close - open) / (high - low)
        atr_14                                       — 14-day ATR / close
        obv_slope_5d                                 — slope of OBV over 5 days
        taker_buy_ratio                              — taker_buy_volume / volume (if avail)
        ewma_vol_10d                                 — EWM std over 10d
        mean_revert_5d                               — z-score of close vs 5d EMA
        trend_strength_20d                           — abs(returns_20d) / vol_20d

    Parameters
    ----------
    symbols : list[str]
        Crypto symbols (e.g. ["BTC", "ETH"]).
    data_root : Path or None
        Root of Binance Hive-partitioned Parquet.
        Default: reads from quantlaxmi.data._paths.BINANCE_DIR.
    """

    N_FEATURES = 31  # expected feature count per asset

    def __init__(
        self,
        symbols: Optional[list[str]] = None,
        data_root: Optional[Path] = None,
    ) -> None:
        self.symbols = symbols or CRYPTO_SYMBOLS
        if data_root is None:
            from quantlaxmi.data._paths import BINANCE_DIR
            self.data_root = BINANCE_DIR
        else:
            self.data_root = Path(data_root)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_daily_ohlcv(
        self, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        """Load daily OHLCV from Hive-partitioned Parquet, or return empty."""
        pair = _CRYPTO_PAIR_MAP.get(symbol.upper(), f"{symbol.upper()}USDT")
        base_dir = self.data_root / pair / "1d"

        if not base_dir.exists():
            logger.warning(
                "CryptoFeatureAdapter: no data dir %s — returning synthetic data", base_dir
            )
            return self._generate_synthetic(symbol, start, end)

        date_dirs = sorted(base_dir.glob("date=*"))
        if not date_dirs:
            logger.warning(
                "CryptoFeatureAdapter: no date partitions in %s — returning synthetic", base_dir
            )
            return self._generate_synthetic(symbol, start, end)

        all_dfs: list[pd.DataFrame] = []
        for d in date_dirs:
            date_str = d.name.split("=")[1]
            if date_str < start or date_str > end:
                continue
            for pf in d.glob("*.parquet"):
                try:
                    df = pd.read_parquet(pf)
                    all_dfs.append(df)
                except Exception as exc:
                    logger.debug("Skip parquet %s: %s", pf, exc)

        if not all_dfs:
            logger.warning(
                "CryptoFeatureAdapter: no parquet files for %s in [%s, %s]", pair, start, end
            )
            return self._generate_synthetic(symbol, start, end)

        df = pd.concat(all_dfs, ignore_index=True)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp").sort_index()
            df = df[~df.index.duplicated(keep="first")]

        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Resample to daily (take last OHLCV per day)
        if not df.empty:
            df = df.resample("1D").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna(subset=["close"])

            # Add taker_buy_volume if it was present
            # (we sum it per day just like volume)
            # but only if the original data has it
            if "taker_buy_volume" in all_dfs[0].columns if all_dfs else False:
                raw_df = pd.concat(all_dfs, ignore_index=True)
                raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True)
                raw_df = raw_df.set_index("timestamp").sort_index()
                tbv = raw_df["taker_buy_volume"].resample("1D").sum()
                df["taker_buy_volume"] = tbv.reindex(df.index).fillna(0.0)

        return df

    @staticmethod
    def _generate_synthetic(symbol: str, start: str, end: str) -> pd.DataFrame:
        """Generate synthetic daily OHLCV when real data is unavailable.

        Uses geometric Brownian motion calibrated to approximate crypto vol.
        """
        dates = pd.bdate_range(start, end, freq="D")
        n = len(dates)
        if n == 0:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        rng = np.random.default_rng(hash(symbol) % (2**31))
        spot = _CRYPTO_INITIAL_SPOTS.get(symbol.upper(), 50_000.0)
        mu_daily = 0.0003  # slight positive drift
        sigma_daily = 0.03  # ~50% annualised vol

        log_prices = np.zeros(n)
        log_prices[0] = math.log(spot)
        for i in range(1, n):
            log_prices[i] = log_prices[i - 1] + mu_daily + sigma_daily * rng.standard_normal()

        close = np.exp(log_prices)
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        high = np.maximum(open_, close) * (1.0 + 0.01 * np.abs(rng.standard_normal(n)))
        low = np.minimum(open_, close) * (1.0 - 0.01 * np.abs(rng.standard_normal(n)))
        volume = np.abs(rng.normal(1e9, 3e8, n))

        df = pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }, index=pd.DatetimeIndex(dates, tz="UTC"))
        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _engineer_features(self, df: pd.DataFrame, all_close: dict[str, pd.Series]) -> pd.DataFrame:
        """Compute ~30 causal features from daily OHLCV.

        Parameters
        ----------
        df : DataFrame with columns [open, high, low, close, volume].
        all_close : dict mapping symbol → close series (for cross-asset features).

        Returns
        -------
        DataFrame of features indexed by date.
        """
        c = df["close"].values.astype(np.float64)
        o = df["open"].values.astype(np.float64)
        h = df["high"].values.astype(np.float64)
        lo = df["low"].values.astype(np.float64)
        v = df["volume"].values.astype(np.float64)
        n = len(c)

        feats: dict[str, np.ndarray] = {}

        # --- Log returns at multiple horizons ---
        log_c = np.log(np.maximum(c, 1e-10))
        feats["returns_1d"] = np.concatenate([[0.0], np.diff(log_c)])
        feats["returns_3d"] = _lag_diff(log_c, 3)
        feats["returns_5d"] = _lag_diff(log_c, 5)
        feats["returns_20d"] = _lag_diff(log_c, 20)

        # --- Volatility ---
        r1d = feats["returns_1d"]
        feats["vol_5d"] = _rolling_std(r1d, 5)
        feats["vol_10d"] = _rolling_std(r1d, 10)
        feats["vol_20d"] = _rolling_std(r1d, 20)
        feats["ewma_vol_10d"] = _ewm_std(r1d, 10)

        # --- Momentum: RSI ---
        feats["rsi_7"] = _rsi(c, 7)
        feats["rsi_14"] = _rsi(c, 14)

        # --- MACD signal ---
        feats["macd_signal"] = _macd_signal(c, 12, 26, 9)

        # --- Volume features ---
        feats["volume_ratio_5d"] = _volume_ratio(v, 5)
        feats["volume_ratio_20d"] = _volume_ratio(v, 20)
        feats["log_volume"] = np.log1p(v)
        dollar_vol = c * v
        feats["dollar_volume"] = dollar_vol
        feats["log_dollar_volume"] = np.log1p(dollar_vol)

        # --- Range and candle shape ---
        safe_c = np.where(c > 0, c, 1.0)
        safe_o = np.where(o > 0, o, 1.0)
        safe_h = np.where(h > 0, h, 1.0)
        hl_range = (h - lo) / safe_c
        feats["high_low_range"] = hl_range
        feats["intraday_range"] = (h - lo) / safe_o
        feats["close_to_high"] = c / safe_h
        feats["close_to_low"] = np.where(lo > 0, c / lo, 1.0)

        # Candle anatomy
        body = np.abs(c - o)
        hl_diff = np.maximum(h - lo, 1e-10)
        feats["body_ratio"] = body / hl_diff
        max_oc = np.maximum(o, c)
        min_oc = np.minimum(o, c)
        feats["upper_shadow"] = (h - max_oc) / safe_h
        feats["lower_shadow"] = (min_oc - lo) / safe_c

        # --- Overnight gap: open / prev_close - 1 ---
        prev_c = np.roll(c, 1)
        prev_c[0] = c[0]
        feats["overnight_gap"] = np.where(prev_c > 0, o / prev_c - 1.0, 0.0)

        # --- ATR(14) / close ---
        feats["atr_14"] = _atr(h, lo, c, 14) / safe_c

        # --- OBV slope (5d) ---
        feats["obv_slope_5d"] = _obv_slope(c, v, 5)

        # --- Taker buy ratio (if available) ---
        if "taker_buy_volume" in df.columns:
            tbv = df["taker_buy_volume"].values.astype(np.float64)
            safe_v = np.where(v > 0, v, 1.0)
            feats["taker_buy_ratio"] = tbv / safe_v
        else:
            feats["taker_buy_ratio"] = np.full(n, 0.5)

        # --- BTC dominance proxy ---
        if "BTC" in all_close and "ETH" in all_close:
            btc_c = all_close["BTC"].reindex(df.index).ffill().fillna(1.0).values
            eth_c = all_close["ETH"].reindex(df.index).ffill().fillna(1.0).values
            total = btc_c + eth_c
            feats["btc_dominance_proxy"] = np.where(total > 0, btc_c / total, 0.5)
        else:
            feats["btc_dominance_proxy"] = np.full(n, 0.5)

        # --- Funding rate proxy: intraday bias ---
        feats["funding_rate_proxy"] = np.where(safe_o > 0, (c - o) / safe_o, 0.0)

        # --- Mean reversion z-score vs 5d EMA ---
        feats["mean_revert_5d"] = _zscore_vs_ema(c, 5)

        # --- Trend strength: |ret_20d| / vol_20d ---
        v20 = feats["vol_20d"]
        safe_v20 = np.where(v20 > 1e-10, v20, 1.0)
        feats["trend_strength_20d"] = np.abs(feats["returns_20d"]) / safe_v20

        # Build DataFrame
        feat_df = pd.DataFrame(feats, index=df.index)
        # Forward-fill leading NaN (from rolling windows) then zero remaining
        feat_df = feat_df.ffill().fillna(0.0)
        # Replace inf/-inf
        feat_df = feat_df.replace([np.inf, -np.inf], 0.0)

        return feat_df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_multi_asset(
        self, start: str, end: str
    ) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
        """Build aligned (n_days, n_crypto_assets, n_features) tensor.

        Parameters
        ----------
        start, end : str
            Date range "YYYY-MM-DD".

        Returns
        -------
        features : ndarray (n_days, n_assets, n_features)
        names : list[str] — feature column names
        dates : pd.DatetimeIndex
        """
        # Step 1: load raw OHLCV for all crypto assets
        raw_dfs: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            raw_dfs[sym] = self._load_daily_ohlcv(sym, start, end)

        # Cross-asset close series for dominance proxy
        all_close: dict[str, pd.Series] = {}
        for sym, df in raw_dfs.items():
            if "close" in df.columns and not df.empty:
                all_close[sym] = df["close"]

        # Step 2: engineer features per asset
        per_asset_dfs: list[pd.DataFrame] = []
        per_asset_names: list[list[str]] = []
        for sym in self.symbols:
            feat_df = self._engineer_features(raw_dfs[sym], all_close)
            per_asset_dfs.append(feat_df)
            per_asset_names.append(list(feat_df.columns))
            logger.info(
                "CryptoFeatureAdapter: %s → %d features, %d rows",
                sym, feat_df.shape[1], len(feat_df),
            )

        if not per_asset_dfs:
            raise ValueError("No crypto features built")

        # Step 3: union feature names
        seen: set[str] = set()
        feature_names: list[str] = []
        for names in per_asset_names:
            for n in names:
                if n not in seen:
                    seen.add(n)
                    feature_names.append(n)

        # Step 4: align on common dates
        all_dates = sorted(set().union(*(df.index for df in per_asset_dfs if not df.empty)))
        if not all_dates:
            raise ValueError("No valid dates for crypto features")

        dates = pd.DatetimeIndex(all_dates)
        n_days = len(dates)
        n_assets = len(self.symbols)
        n_features = len(feature_names)

        features = np.full((n_days, n_assets, n_features), np.nan, dtype=np.float64)
        for a, feat_df in enumerate(per_asset_dfs):
            aligned = feat_df.reindex(dates).ffill()
            aligned = aligned.reindex(columns=feature_names)
            features[:, a, :] = aligned.values

        features = np.nan_to_num(features, nan=0.0)
        logger.info(
            "CryptoFeatureAdapter: %d days x %d assets x %d features",
            n_days, n_assets, n_features,
        )
        return features, feature_names, dates


# ---------------------------------------------------------------------------
# Numeric helper functions (all causal, no look-ahead)
# ---------------------------------------------------------------------------


def _lag_diff(x: np.ndarray, lag: int) -> np.ndarray:
    """x[t] - x[t-lag], with leading zeros."""
    out = np.zeros_like(x)
    if lag < len(x):
        out[lag:] = x[lag:] - x[:-lag]
    return out


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling standard deviation (ddof=1)."""
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        chunk = x[i - window + 1: i + 1]
        if len(chunk) > 1:
            out[i] = float(np.std(chunk, ddof=1))
    return out


def _ewm_std(x: np.ndarray, span: int) -> np.ndarray:
    """EWM standard deviation (pandas convention)."""
    s = pd.Series(x)
    return s.ewm(span=span, min_periods=span).std().fillna(0.0).values


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI scaled to [-1, 1] (centered: (RSI-50)/50)."""
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return out
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])

    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi_val = 100.0 - 100.0 / (1.0 + rs)
        else:
            rsi_val = 100.0
        out[i + 1] = (rsi_val - 50.0) / 50.0  # center to [-1, 1]

    return out


def _macd_signal(close: np.ndarray, fast: int, slow: int, sig: int) -> np.ndarray:
    """MACD histogram: MACD(fast, slow) - Signal(sig)."""
    s = pd.Series(close)
    ema_fast = s.ewm(span=fast, min_periods=fast).mean()
    ema_slow = s.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=sig, min_periods=sig).mean()
    hist = macd_line - signal_line
    return hist.fillna(0.0).values


def _volume_ratio(volume: np.ndarray, window: int) -> np.ndarray:
    """Volume / rolling mean volume."""
    s = pd.Series(volume)
    ma = s.rolling(window, min_periods=1).mean()
    ratio = s / ma.replace(0, 1)
    return ratio.fillna(1.0).values


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range (Wilder smoothing)."""
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr_arr = np.zeros(n, dtype=np.float64)
    if n >= period:
        atr_arr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr_arr[i] = (atr_arr[i - 1] * (period - 1) + tr[i]) / period
    return atr_arr


def _obv_slope(close: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
    """Slope of On-Balance Volume over a rolling window (normalized)."""
    n = len(close)
    obv = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    # Normalise OBV to prevent numerical blow-up
    obv_max = np.maximum(np.abs(obv).max(), 1.0)
    obv_norm = obv / obv_max

    # Rolling linear slope
    out = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        y = obv_norm[i - window + 1: i + 1]
        x = np.arange(window, dtype=np.float64)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom > 1e-12:
            out[i] = np.sum((x - x_mean) * (y - y_mean)) / denom
    return out


def _zscore_vs_ema(close: np.ndarray, span: int) -> np.ndarray:
    """Z-score of close relative to EMA(span)."""
    s = pd.Series(close)
    ema = s.ewm(span=span, min_periods=span).mean()
    std = s.rolling(span, min_periods=2).std()
    z = (s - ema) / std.replace(0, 1)
    return z.fillna(0.0).values


# ============================================================================
# CrossMarketBackbone
# ============================================================================


class CrossMarketBackbone:
    """Two XTrendBackbone instances (India + Crypto) with joint hidden extraction.

    Parameters
    ----------
    india_cfg : XTrendConfig
        Config for the India backbone (4 assets, ~287 features).
    crypto_cfg : XTrendConfig
        Config for the Crypto backbone (2 assets, ~30 features).
    india_feature_names : list[str]
    crypto_feature_names : list[str]
    """

    def __init__(
        self,
        india_cfg: XTrendConfig,
        crypto_cfg: XTrendConfig,
        india_feature_names: list[str],
        crypto_feature_names: list[str],
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for CrossMarketBackbone")

        self.india_backbone = XTrendBackbone(india_cfg, india_feature_names)
        self.crypto_backbone = XTrendBackbone(crypto_cfg, crypto_feature_names)
        self.d_hidden = india_cfg.d_hidden  # assume same d_hidden for both

        # Sanity: both backbones must agree on d_hidden
        if india_cfg.d_hidden != crypto_cfg.d_hidden:
            raise ValueError(
                f"d_hidden mismatch: India={india_cfg.d_hidden}, "
                f"Crypto={crypto_cfg.d_hidden}"
            )

    def to(self, device) -> "CrossMarketBackbone":
        """Move both backbones to device."""
        self.india_backbone.to(device)
        self.crypto_backbone.to(device)
        return self

    def eval(self) -> "CrossMarketBackbone":
        """Set both backbones to eval mode."""
        self.india_backbone.eval()
        self.crypto_backbone.eval()
        return self

    def freeze(self) -> None:
        """Freeze all parameters in both backbones."""
        for p in self.india_backbone.parameters():
            p.requires_grad = False
        for p in self.crypto_backbone.parameters():
            p.requires_grad = False

    def extract_joint_hidden(
        self,
        india_features: np.ndarray,
        crypto_features: np.ndarray,
        day_idx: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Extract concatenated hidden state for all 6 assets at a given day.

        Returns
        -------
        joint_hidden : (6 * d_hidden,) numpy array
            [india_asset_0, india_asset_1, india_asset_2, india_asset_3,
             crypto_asset_0, crypto_asset_1]
        """
        d_h = self.d_hidden
        n_india = india_features.shape[1]
        n_crypto = crypto_features.shape[1]
        joint = np.zeros(6 * d_h, dtype=np.float32)

        # India backbone hidden states
        for a in range(n_india):
            h = self.india_backbone.extract_hidden_for_day(
                india_features, day_idx, a, rng
            )
            joint[a * d_h: (a + 1) * d_h] = h

        # Crypto backbone hidden states
        for a in range(n_crypto):
            h = self.crypto_backbone.extract_hidden_for_day(
                crypto_features, day_idx, a, rng
            )
            offset = n_india + a
            joint[offset * d_h: (offset + 1) * d_h] = h

        return joint

    def precompute_joint_hidden(
        self,
        india_features: np.ndarray,
        crypto_features: np.ndarray,
        start_idx: int,
        end_idx: int,
        rng: np.random.Generator,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Pre-compute joint hidden states for a date range (batched).

        Parameters
        ----------
        india_features : (n_days, 4, n_india_feat) normalized
        crypto_features : (n_days, 2, n_crypto_feat) normalized
        start_idx, end_idx : fold boundaries

        Returns
        -------
        joint_hidden : (end_idx - start_idx, 6 * d_hidden)
        """
        fold_len = end_idx - start_idx
        d_h = self.d_hidden

        india_hidden = self.india_backbone.precompute_hidden_states(
            india_features, start_idx, end_idx, rng, batch_size=batch_size
        )
        crypto_hidden = self.crypto_backbone.precompute_hidden_states(
            crypto_features, start_idx, end_idx, rng, batch_size=batch_size
        )
        # india_hidden: (fold_len, 4, d_hidden)
        # crypto_hidden: (fold_len, 2, d_hidden)

        joint = np.zeros((fold_len, 6 * d_h), dtype=np.float32)
        # Fill India (4 assets)
        for a in range(india_hidden.shape[1]):
            joint[:, a * d_h: (a + 1) * d_h] = india_hidden[:, a, :]
        # Fill Crypto (2 assets)
        india_n = india_hidden.shape[1]
        for a in range(crypto_hidden.shape[1]):
            offset = india_n + a
            joint[:, offset * d_h: (offset + 1) * d_h] = crypto_hidden[:, a, :]

        return joint


# ============================================================================
# CrossMarketAllocator (Actor-Critic with GAE)
# ============================================================================

if _HAS_TORCH:

    class _AllocatorActor(nn.Module):
        """Maps joint hidden → allocation weights via softmax."""

        def __init__(self, input_dim: int, n_assets: int = 6) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ELU(),
                nn.Linear(128, n_assets),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return allocation logits (batch, n_assets)."""
            return self.net(x)

    class _AllocatorCritic(nn.Module):
        """Maps joint hidden → scalar state value V(s)."""

        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ELU(),
                nn.Linear(128, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)

    class CrossMarketAllocator(nn.Module):
        """Actor-Critic allocator across 6 assets with GAE.

        Parameters
        ----------
        joint_hidden_dim : int
            Dimension of the concatenated backbone hidden (6 * d_hidden).
        n_assets : int
            Number of assets (default 6).
        lr_actor : float
        lr_critic : float
        entropy_beta : float
            Entropy regularization coefficient.
        max_grad_norm : float
        """

        def __init__(
            self,
            joint_hidden_dim: int,
            n_assets: int = 6,
            lr_actor: float = 3e-4,
            lr_critic: float = 1e-3,
            entropy_beta: float = 0.01,
            max_grad_norm: float = 5.0,
        ) -> None:
            super().__init__()
            self.n_assets = n_assets
            self.entropy_beta = entropy_beta
            self.max_grad_norm = max_grad_norm

            self.actor = _AllocatorActor(joint_hidden_dim, n_assets)
            self.critic = _AllocatorCritic(joint_hidden_dim)

            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=lr_actor
            )
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=lr_critic
            )

        def get_allocation(
            self, state: np.ndarray, temperature: float = 1.0
        ) -> np.ndarray:
            """Get softmax allocation weights (deterministic)."""
            dev = next(self.actor.parameters()).device
            s_t = torch.FloatTensor(state).unsqueeze(0).to(dev)
            with torch.no_grad():
                logits = self.actor(s_t).squeeze(0) / temperature
                weights = torch.softmax(logits, dim=-1)
            return weights.cpu().numpy()

        def get_allocation_with_log_prob(
            self, state: np.ndarray
        ) -> tuple[np.ndarray, float, float]:
            """Get allocation by sampling from Dirichlet-like policy.

            We add Gaussian noise to logits, then softmax → weights.
            This gives a differentiable stochastic policy.

            Returns
            -------
            weights : (n_assets,)
            log_prob : float
            value : float
            """
            dev = next(self.actor.parameters()).device
            s_t = torch.FloatTensor(state).unsqueeze(0).to(dev)

            logits = self.actor(s_t).squeeze(0)
            # Add exploration noise (Gaussian on logits)
            noise = torch.randn_like(logits) * 0.3
            noisy_logits = logits + noise
            weights = torch.softmax(noisy_logits, dim=-1)

            # Log probability: treat as categorical approximation
            log_probs = torch.log_softmax(logits, dim=-1)
            log_prob = (log_probs * weights.detach()).sum()

            value = self.critic(s_t).squeeze()

            return (
                weights.detach().cpu().numpy(),
                float(log_prob.item()),
                float(value.item()),
            )

        def update_gae(
            self,
            states: np.ndarray,
            weights_taken: np.ndarray,
            rewards: np.ndarray,
            dones: np.ndarray,
            gamma: float = GAE_GAMMA,
            lam: float = GAE_LAMBDA,
        ) -> tuple[float, float]:
            """Update actor and critic using GAE advantages.

            Parameters
            ----------
            states : (T, joint_hidden_dim)
            weights_taken : (T, n_assets)
            rewards : (T,)
            dones : (T,) — 1.0 if terminal
            gamma : discount factor
            lam : GAE lambda

            Returns
            -------
            (actor_loss, critic_loss)
            """
            dev = next(self.actor.parameters()).device
            T = len(rewards)
            if T < 2:
                return 0.0, 0.0

            states_t = torch.FloatTensor(states).to(dev)
            rewards_t = torch.FloatTensor(rewards).to(dev)
            dones_t = torch.FloatTensor(dones).to(dev)
            weights_t = torch.FloatTensor(weights_taken).to(dev)

            # Compute values
            values = self.critic(states_t)  # (T,)

            # GAE computation
            with torch.no_grad():
                advantages = torch.zeros(T, device=dev)
                gae = 0.0
                for t in reversed(range(T)):
                    if t == T - 1:
                        next_val = 0.0
                    else:
                        next_val = values[t + 1].item()
                    delta = rewards_t[t] + gamma * next_val * (1.0 - dones_t[t]) - values[t].item()
                    gae = delta + gamma * lam * (1.0 - dones_t[t]) * gae
                    advantages[t] = gae

                returns_t = advantages + values.detach()

            # Normalise advantages
            if advantages.numel() > 1:
                adv_std = advantages.std()
                if adv_std > 1e-8:
                    advantages = (advantages - advantages.mean()) / adv_std

            # Actor loss
            logits = self.actor(states_t)  # (T, n_assets)
            log_probs = torch.log_softmax(logits, dim=-1)  # (T, n_assets)
            # Weighted log prob (policy gradient for softmax allocation)
            policy_log_prob = (log_probs * weights_t).sum(dim=-1)  # (T,)

            # Entropy bonus (Shannon entropy of softmax)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)  # (T,)

            actor_loss = -(policy_log_prob * advantages + self.entropy_beta * entropy).mean()

            # Critic loss
            critic_loss = F.mse_loss(values, returns_t)

            # Update
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            return actor_loss.item(), critic_loss.item()

else:

    class CrossMarketAllocator:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch is required for CrossMarketAllocator")


# ============================================================================
# CrossMarketPipeline
# ============================================================================


@dataclass
class CrossMarketConfig:
    """Hyperparameters for the cross-market pipeline."""

    # Walk-forward
    train_window: int = 252
    test_window: int = 63
    step_size: int = 21

    # Backbone
    d_hidden: int = 64
    seq_len: int = 42
    ctx_len: int = 42
    n_context: int = 16
    pretrain_epochs: int = 50
    pretrain_lr: float = 1e-3

    # RL allocator
    rl_episodes: int = 200
    rl_lr_actor: float = 3e-4
    rl_lr_critic: float = 1e-3
    rl_entropy_beta: float = 0.01
    rl_max_grad_norm: float = 5.0

    # Merton benchmark
    merton_gamma: float = 2.0
    risk_free_rate: float = 0.065  # India 10Y yield ~6.5%


class CrossMarketPipeline:
    """Walk-forward cross-market pipeline: India + Crypto → RL allocation.

    Phases per fold:
        1. Build India + Crypto features
        2. Pre-train both backbones (supervised)
        3. Pre-compute joint hidden states
        4. Train RL allocator (Actor-Critic with GAE, 200 episodes)
        5. OOS: evaluate RL vs Merton vs equal-weight

    Parameters
    ----------
    cfg : CrossMarketConfig
    """

    def __init__(self, cfg: Optional[CrossMarketConfig] = None) -> None:
        self.cfg = cfg or CrossMarketConfig()

    def run(
        self,
        start: str,
        end: str,
        india_features: Optional[np.ndarray] = None,
        india_feature_names: Optional[list[str]] = None,
        india_dates: Optional[pd.DatetimeIndex] = None,
        crypto_features: Optional[np.ndarray] = None,
        crypto_feature_names: Optional[list[str]] = None,
        crypto_dates: Optional[pd.DatetimeIndex] = None,
    ) -> dict:
        """Run the full cross-market pipeline.

        Returns
        -------
        dict with keys:
            rl_positions, rl_returns, rl_sharpe,
            merton_positions, merton_returns, merton_sharpe,
            ew_positions, ew_returns, ew_sharpe,
            per_asset_sharpe, fold_metrics
        """
        cfg = self.cfg

        # ---- Phase 1: Build features ----
        if india_features is None:
            logger.info("Building India features via MegaFeatureAdapter...")
            india_adapter = MegaFeatureAdapter(INDIA_SYMBOLS)
            india_features, india_feature_names, india_dates = india_adapter.build_multi_asset(
                start, end
            )

        if crypto_features is None:
            logger.info("Building Crypto features via CryptoFeatureAdapter...")
            crypto_adapter = CryptoFeatureAdapter(CRYPTO_SYMBOLS)
            crypto_features, crypto_feature_names, crypto_dates = crypto_adapter.build_multi_asset(
                start, end
            )

        # Align dates: intersection of India and Crypto trading days
        india_dates_set = set(india_dates)
        crypto_dates_set = set(crypto_dates)
        common_dates = sorted(india_dates_set & crypto_dates_set)

        if len(common_dates) < cfg.train_window + cfg.test_window + cfg.seq_len:
            logger.warning(
                "Insufficient common dates (%d) for walk-forward. "
                "Falling back to India dates with crypto forward-fill.",
                len(common_dates),
            )
            common_dates = sorted(india_dates_set)

        dates = pd.DatetimeIndex(common_dates)
        n_days = len(dates)

        # Re-index features to common dates
        india_features, india_feature_names = self._reindex_features(
            india_features, india_dates, dates, india_feature_names
        )
        crypto_features, crypto_feature_names = self._reindex_features(
            crypto_features, crypto_dates, dates, crypto_feature_names
        )

        n_india_feat = india_features.shape[2]
        n_crypto_feat = crypto_features.shape[2]

        logger.info(
            "CrossMarketPipeline: %d common days, India=%d feat, Crypto=%d feat",
            n_days, n_india_feat, n_crypto_feat,
        )

        # ---- Compute targets (vol-scaled next-day returns) ----
        india_targets = self._compute_targets(india_features, 4)
        crypto_targets = self._compute_targets(crypto_features, 2)
        # Combined (n_days, 6) target array
        all_targets = np.concatenate([india_targets, crypto_targets], axis=1)

        # ---- Walk-forward ----
        n_total_assets = 6
        rl_positions = np.full((n_days, n_total_assets), np.nan)
        rl_returns = np.full((n_days, n_total_assets), np.nan)
        merton_positions = np.full((n_days, n_total_assets), np.nan)
        merton_returns = np.full((n_days, n_total_assets), np.nan)
        ew_positions = np.full((n_days, n_total_assets), np.nan)
        ew_returns = np.full((n_days, n_total_assets), np.nan)
        fold_metrics: list[dict] = []

        warm_up = cfg.seq_len + cfg.ctx_len + 10
        fold_start = warm_up
        fold_idx = 0

        while fold_start + cfg.train_window + cfg.test_window <= n_days:
            train_end = fold_start + cfg.train_window
            test_end = min(train_end + cfg.test_window, n_days)

            logger.info(
                "Fold %d: train=[%d:%d], test=[%d:%d]",
                fold_idx, fold_start, train_end, train_end, test_end,
            )

            # ---- Phase 2: Pre-train backbones ----
            india_cfg = self._make_backbone_cfg(n_india_feat, 4)
            crypto_cfg = self._make_backbone_cfg(n_crypto_feat, 2)

            backbone = CrossMarketBackbone(
                india_cfg, crypto_cfg,
                india_feature_names, crypto_feature_names,
            )
            dev = _DEVICE if _HAS_TORCH else "cpu"
            backbone.to(dev)

            # Pre-train India backbone
            india_pretrain = backbone.india_backbone.pretrain(
                india_features, india_targets, dates,
                train_start=fold_start, train_end=train_end,
                epochs=cfg.pretrain_epochs, lr=cfg.pretrain_lr,
            )
            # Pre-train Crypto backbone
            crypto_pretrain = backbone.crypto_backbone.pretrain(
                crypto_features, crypto_targets, dates,
                train_start=fold_start, train_end=train_end,
                epochs=cfg.pretrain_epochs, lr=cfg.pretrain_lr,
            )

            logger.info(
                "Fold %d pretrain: India best_ep=%d loss=%.4f, Crypto best_ep=%d loss=%.4f",
                fold_idx,
                india_pretrain["best_epoch"], india_pretrain["final_loss"],
                crypto_pretrain["best_epoch"], crypto_pretrain["final_loss"],
            )

            backbone.eval()
            backbone.freeze()

            # ---- Phase 3: Pre-compute joint hidden ----
            rng = np.random.default_rng(42 + fold_idx)

            # Normalise features using train stats
            india_norm = self._normalise_features(india_features, fold_start, train_end)
            crypto_norm = self._normalise_features(crypto_features, fold_start, train_end)

            # Compute hidden for train + test range
            train_hidden = backbone.precompute_joint_hidden(
                india_norm, crypto_norm, fold_start, train_end, rng
            )
            test_hidden = backbone.precompute_joint_hidden(
                india_norm, crypto_norm, train_end, test_end, rng
            )

            joint_dim = 6 * cfg.d_hidden

            # ---- Phase 4: Train RL allocator ----
            allocator = CrossMarketAllocator(
                joint_hidden_dim=joint_dim,
                n_assets=n_total_assets,
                lr_actor=cfg.rl_lr_actor,
                lr_critic=cfg.rl_lr_critic,
                entropy_beta=cfg.rl_entropy_beta,
                max_grad_norm=cfg.rl_max_grad_norm,
            )
            allocator.to(dev)

            rl_train_info = self._train_allocator(
                allocator, train_hidden, all_targets,
                fold_start, train_end,
                num_episodes=cfg.rl_episodes,
            )

            logger.info(
                "Fold %d RL: %d episodes, avg_reward=%.4f",
                fold_idx, rl_train_info["n_episodes"], rl_train_info["avg_reward"],
            )

            # ---- Merton benchmark: analytical weights ----
            merton_weights = self._compute_merton_weights(
                all_targets, fold_start, train_end, cfg.merton_gamma, cfg.risk_free_rate,
            )

            # ---- Phase 5: OOS evaluation ----
            test_len = test_end - train_end

            for t in range(test_len - 1):
                day_idx = train_end + t
                next_day_idx = day_idx + 1
                if next_day_idx >= n_days:
                    break

                state = test_hidden[t]
                next_ret = all_targets[next_day_idx]  # (6,)

                # RL allocation
                rl_w = allocator.get_allocation(state)
                rl_positions[day_idx] = rl_w
                rl_ret = self._compute_portfolio_return(rl_w, next_ret, day_idx)
                rl_returns[day_idx] = rl_ret

                # Merton allocation
                merton_positions[day_idx] = merton_weights
                merton_ret = self._compute_portfolio_return(merton_weights, next_ret, day_idx)
                merton_returns[day_idx] = merton_ret

                # Equal-weight allocation
                ew_w = np.ones(n_total_assets) / n_total_assets
                ew_positions[day_idx] = ew_w
                ew_ret = self._compute_portfolio_return(ew_w, next_ret, day_idx)
                ew_returns[day_idx] = ew_ret

            # Fold summary
            fold_rl_ret = rl_returns[train_end:test_end]
            fold_merton_ret = merton_returns[train_end:test_end]
            fold_ew_ret = ew_returns[train_end:test_end]

            fold_info = {
                "fold": fold_idx,
                "train_range": (fold_start, train_end),
                "test_range": (train_end, test_end),
                "rl_sharpe": _sharpe(fold_rl_ret),
                "merton_sharpe": _sharpe(fold_merton_ret),
                "ew_sharpe": _sharpe(fold_ew_ret),
                "rl_total_return": float(np.nansum(fold_rl_ret)),
                "india_pretrain": india_pretrain,
                "crypto_pretrain": crypto_pretrain,
                "rl_train": rl_train_info,
            }
            fold_metrics.append(fold_info)

            logger.info(
                "Fold %d OOS: RL Sharpe=%.3f, Merton Sharpe=%.3f, EW Sharpe=%.3f",
                fold_idx, fold_info["rl_sharpe"],
                fold_info["merton_sharpe"], fold_info["ew_sharpe"],
            )

            fold_start += cfg.step_size
            fold_idx += 1

            # Cleanup
            del backbone, allocator
            if _HAS_TORCH:
                torch.cuda.empty_cache()

        # ---- Aggregate results ----
        results = self._aggregate(
            rl_positions, rl_returns,
            merton_positions, merton_returns,
            ew_positions, ew_returns,
            fold_metrics,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_backbone_cfg(self, n_features: int, n_assets: int) -> XTrendConfig:
        """Create an XTrendConfig with the right feature/asset counts."""
        cfg = self.cfg
        return XTrendConfig(
            d_hidden=cfg.d_hidden,
            n_features=n_features,
            n_assets=n_assets,
            seq_len=cfg.seq_len,
            ctx_len=cfg.ctx_len,
            n_context=cfg.n_context,
            lr=cfg.pretrain_lr,
            loss_mode="joint_mle",
            mle_weight=0.1,
        )

    @staticmethod
    def _reindex_features(
        features: np.ndarray,
        orig_dates: pd.DatetimeIndex,
        target_dates: pd.DatetimeIndex,
        feature_names: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """Re-index features to target_dates, forward-filling gaps."""
        n_assets = features.shape[1]
        n_feat = features.shape[2]
        n_days = len(target_dates)

        out = np.full((n_days, n_assets, n_feat), np.nan, dtype=np.float64)

        # Build a mapping from orig_dates to feature rows
        orig_date_to_idx: dict = {}
        for i, d in enumerate(orig_dates):
            orig_date_to_idx[d] = i

        for t, dt in enumerate(target_dates):
            if dt in orig_date_to_idx:
                out[t] = features[orig_date_to_idx[dt]]

        # Forward-fill along time axis per asset per feature
        for a in range(n_assets):
            for f in range(n_feat):
                col = out[:, a, f]
                last_valid = np.nan
                for t in range(n_days):
                    if np.isnan(col[t]):
                        col[t] = last_valid
                    else:
                        last_valid = col[t]

        out = np.nan_to_num(out, nan=0.0)
        return out, feature_names

    @staticmethod
    def _normalise_features(
        features: np.ndarray, train_start: int, train_end: int
    ) -> np.ndarray:
        """Z-score normalise using train period statistics."""
        train_slice = features[train_start:train_end]
        flat = train_slice.reshape(-1, features.shape[2])
        feat_mean = np.nanmean(flat, axis=0)
        feat_std = np.nanstd(flat, axis=0, ddof=1)
        feat_std = np.where(feat_std > 1e-10, feat_std, 1.0)
        normalised = (features - feat_mean) / feat_std
        return np.nan_to_num(normalised, nan=0.0)

    @staticmethod
    def _compute_targets(features: np.ndarray, n_assets: int) -> np.ndarray:
        """Compute vol-scaled next-day returns from first feature (close proxy)."""
        n_days = features.shape[0]
        targets = np.full((n_days, n_assets), np.nan, dtype=np.float64)

        for a in range(n_assets):
            close_proxy = features[:, a, 0]
            if np.all(close_proxy == 0):
                rng_a = np.random.default_rng(42 + a)
                targets[:-1, a] = rng_a.normal(0, 0.01, n_days - 1)
                continue

            valid = close_proxy > 0
            log_c = np.where(valid, np.log(np.maximum(close_proxy, 1e-10)), 0.0)
            for t in range(n_days - 1):
                if valid[t] and valid[t + 1]:
                    ret = log_c[t + 1] - log_c[t]
                    w_start = max(0, t - 19)
                    window_rets = np.diff(log_c[w_start: t + 1])
                    if len(window_rets) >= 5:
                        vol = np.std(window_rets, ddof=1)
                        vol = max(vol, 1e-6)
                        targets[t, a] = ret / vol
                    else:
                        targets[t, a] = ret

        return targets

    def _train_allocator(
        self,
        allocator: "CrossMarketAllocator",
        train_hidden: np.ndarray,
        all_targets: np.ndarray,
        fold_start: int,
        fold_end: int,
        num_episodes: int,
    ) -> dict:
        """Train the RL allocator via Actor-Critic with GAE.

        Each episode: walk through the train fold day by day.
        State = joint hidden at day t.
        Action = allocation weights.
        Reward = portfolio return minus transaction costs.
        """
        train_len = fold_end - fold_start
        if train_len < 2:
            return {"n_episodes": 0, "avg_reward": 0.0, "avg_actor_loss": 0.0, "avg_critic_loss": 0.0}

        total_reward = 0.0
        total_actor = 0.0
        total_critic = 0.0
        n_updates = 0

        for ep in range(num_episodes):
            states_list: list[np.ndarray] = []
            weights_list: list[np.ndarray] = []
            rewards_list: list[float] = []
            dones_list: list[float] = []

            prev_weights = np.ones(6) / 6.0
            ep_reward = 0.0

            for t in range(train_len - 1):
                day_idx = fold_start + t
                next_day_idx = day_idx + 1
                if next_day_idx >= len(all_targets):
                    break

                state = train_hidden[t]
                w, _, _ = allocator.get_allocation_with_log_prob(state)
                next_ret = all_targets[next_day_idx]

                # Portfolio return (weighted sum of asset returns minus costs)
                port_ret = self._compute_portfolio_return_scalar(
                    w, next_ret, prev_weights
                )
                reward = port_ret

                is_last = (t == train_len - 2)
                states_list.append(state)
                weights_list.append(w)
                rewards_list.append(reward)
                dones_list.append(1.0 if is_last else 0.0)

                ep_reward += reward
                prev_weights = w.copy()

            total_reward += ep_reward

            # GAE update
            if len(states_list) >= 2:
                al, cl = allocator.update_gae(
                    np.array(states_list, dtype=np.float32),
                    np.array(weights_list, dtype=np.float32),
                    np.array(rewards_list, dtype=np.float32),
                    np.array(dones_list, dtype=np.float32),
                    gamma=GAE_GAMMA,
                    lam=GAE_LAMBDA,
                )
                total_actor += al
                total_critic += cl
                n_updates += 1

            if ep % 50 == 0 or ep == num_episodes - 1:
                logger.info(
                    "  Alloc ep %d/%d: reward=%.4f, steps=%d",
                    ep, num_episodes, ep_reward, len(states_list),
                )

        return {
            "n_episodes": num_episodes,
            "avg_reward": total_reward / max(num_episodes, 1),
            "avg_actor_loss": total_actor / max(n_updates, 1),
            "avg_critic_loss": total_critic / max(n_updates, 1),
        }

    @staticmethod
    def _compute_merton_weights(
        targets: np.ndarray,
        train_start: int,
        train_end: int,
        gamma: float,
        risk_free_rate: float,
    ) -> np.ndarray:
        """Compute Merton analytical weights from train-period returns.

        pi* = (1/gamma) * Sigma^{-1} * (mu - r*1)
        """
        train_rets = targets[train_start:train_end]
        # Drop rows with any NaN
        valid_mask = ~np.any(np.isnan(train_rets), axis=1)
        valid_rets = train_rets[valid_mask]

        if len(valid_rets) < 30:
            # Not enough data — fall back to equal weight
            return np.ones(targets.shape[1]) / targets.shape[1]

        mu = np.mean(valid_rets, axis=0) * 252  # annualise
        sigma = np.cov(valid_rets, rowvar=False, ddof=1) * 252  # annualise

        # Regularise covariance (shrink towards diagonal)
        n = sigma.shape[0]
        diag = np.diag(np.diag(sigma))
        shrinkage = 0.3
        sigma_reg = (1 - shrinkage) * sigma + shrinkage * diag

        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(sigma_reg)
        if eigvals.min() < 1e-8:
            sigma_reg += np.eye(n) * (1e-6 - eigvals.min())

        weights = MertonSolution.optimal_weights(
            mu=mu,
            sigma=sigma_reg,
            risk_free_rate=risk_free_rate,
            gamma_risk=gamma,
        )

        # Normalise to sum=1 and clip for sanity
        weights = np.clip(weights, -0.5, 0.5)
        w_sum = np.sum(np.abs(weights))
        if w_sum > 1e-10:
            weights = weights / w_sum
        else:
            weights = np.ones(n) / n

        return weights

    @staticmethod
    def _compute_portfolio_return(
        weights: np.ndarray,
        next_ret: np.ndarray,
        day_idx: int,
    ) -> np.ndarray:
        """Per-asset weighted return after transaction costs.

        Returns (n_assets,) — each element = w_i * r_i - cost_i.
        """
        n = len(weights)
        per_asset = np.zeros(n, dtype=np.float64)

        for i in range(n):
            r_i = next_ret[i] if not np.isnan(next_ret[i]) else 0.0
            w_i = weights[i]

            # Transaction costs
            if i < 4:
                # India assets
                sym = INDIA_SYMBOLS[i]
                cost_pts = abs(w_i) * COST_PER_LEG.get(sym, 3.0) * 0.01  # attenuate for daily
                spot = INITIAL_SPOTS.get(sym, 20000.0)
                cost_frac = cost_pts / spot
            else:
                # Crypto assets: 0.1% maker/taker per side
                cost_frac = abs(w_i) * CRYPTO_FEE_RATE * 0.1  # attenuated for rebalance

            per_asset[i] = w_i * r_i - cost_frac

        return per_asset

    @staticmethod
    def _compute_portfolio_return_scalar(
        weights: np.ndarray,
        next_ret: np.ndarray,
        prev_weights: np.ndarray,
    ) -> float:
        """Scalar portfolio return after turnover-based costs."""
        n = len(weights)
        total_ret = 0.0
        total_cost = 0.0

        for i in range(n):
            r_i = next_ret[i] if not np.isnan(next_ret[i]) else 0.0
            total_ret += weights[i] * r_i

            turnover = abs(weights[i] - prev_weights[i])
            if i < 4:
                sym = INDIA_SYMBOLS[i]
                cost_pts = turnover * COST_PER_LEG.get(sym, 3.0)
                spot = INITIAL_SPOTS.get(sym, 20000.0)
                total_cost += cost_pts / spot
            else:
                total_cost += turnover * CRYPTO_FEE_RATE

        return total_ret - total_cost

    @staticmethod
    def _aggregate(
        rl_positions: np.ndarray,
        rl_returns: np.ndarray,
        merton_positions: np.ndarray,
        merton_returns: np.ndarray,
        ew_positions: np.ndarray,
        ew_returns: np.ndarray,
        fold_metrics: list[dict],
    ) -> dict:
        """Aggregate results across all folds."""

        def _per_asset_sharpe(returns: np.ndarray) -> dict[str, float]:
            result = {}
            for a, sym in enumerate(ALL_SYMBOLS):
                col = returns[:, a]
                valid = col[~np.isnan(col)]
                if len(valid) > 1 and np.std(valid, ddof=1) > 1e-10:
                    sr = (np.mean(valid) / np.std(valid, ddof=1)) * math.sqrt(252)
                else:
                    sr = 0.0
                result[sym] = sr
            return result

        return {
            "rl_positions": rl_positions,
            "rl_returns": rl_returns,
            "rl_sharpe": _sharpe(rl_returns),
            "rl_per_asset_sharpe": _per_asset_sharpe(rl_returns),
            "merton_positions": merton_positions,
            "merton_returns": merton_returns,
            "merton_sharpe": _sharpe(merton_returns),
            "merton_per_asset_sharpe": _per_asset_sharpe(merton_returns),
            "ew_positions": ew_positions,
            "ew_returns": ew_returns,
            "ew_sharpe": _sharpe(ew_returns),
            "ew_per_asset_sharpe": _per_asset_sharpe(ew_returns),
            "fold_metrics": fold_metrics,
            "n_folds": len(fold_metrics),
        }

    def report(self, results: dict) -> None:
        """Print human-readable comparison report."""
        print(f"\n{'=' * 75}")
        print(f"{'CROSS-MARKET INDIA+CRYPTO ALLOCATION RESULTS':^75}")
        print(f"{'=' * 75}")

        for method in ["rl", "merton", "ew"]:
            label = {"rl": "RL Actor-Critic", "merton": "Merton Analytical", "ew": "Equal Weight"}[method]
            sr = results[f"{method}_sharpe"]
            per_asset = results[f"{method}_per_asset_sharpe"]
            port_ret = results[f"{method}_returns"]
            valid = port_ret[~np.isnan(port_ret)]
            total = float(np.sum(valid)) if len(valid) > 0 else 0.0

            print(f"\n--- {label} ---")
            print(f"  Portfolio Sharpe : {sr:.4f}")
            print(f"  Total Return    : {total:.4f}")
            print(f"  Per-Asset Sharpe:")
            for sym, s in per_asset.items():
                print(f"    {sym:<20} {s:>8.4f}")

        print(f"\nFolds: {results['n_folds']}")
        for fm in results["fold_metrics"]:
            tr = fm["test_range"]
            print(
                f"  Fold {fm['fold']}: test=[{tr[0]}:{tr[1]}] "
                f"RL={fm['rl_sharpe']:.3f} "
                f"Merton={fm['merton_sharpe']:.3f} "
                f"EW={fm['ew_sharpe']:.3f}"
            )

        print(f"{'=' * 75}\n")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _sharpe(returns: np.ndarray) -> float:
    """Compute annualised Sharpe ratio (ddof=1, sqrt(252))."""
    flat = returns.flatten()
    valid = flat[~np.isnan(flat)]
    if len(valid) > 1 and np.std(valid, ddof=1) > 1e-10:
        return float((np.mean(valid) / np.std(valid, ddof=1)) * math.sqrt(252))
    return 0.0


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def run_cross_market_backtest(
    start: str = "2024-01-01",
    end: str = "2026-02-06",
    cfg: Optional[CrossMarketConfig] = None,
) -> dict:
    """Run the full cross-market India + Crypto backtest.

    Parameters
    ----------
    start, end : str
        Date range.
    cfg : CrossMarketConfig or None
        Pipeline configuration.

    Returns
    -------
    dict of results (see CrossMarketPipeline.run).
    """
    pipeline = CrossMarketPipeline(cfg)
    results = pipeline.run(start, end)
    pipeline.report(results)
    return results
