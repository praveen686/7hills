"""Feature builder for X-Trend model — paper features + rich intraday signals.

Core (paper) features — 8 per asset per day:
  5 normalized returns: r̂(t', t) = r(t-t', t) / (σ_t * √t')
  3 MACD signals (standardized)

Intraday features — 16 per asset per day (from 1-min OHLCV bars):
  Realized vol, Garman-Klass vol, Parkinson vol, overnight gap,
  open-to-close return, first/second half momentum, VWAP deviation,
  close position, ORB width, ORB breakout, intraday max drawdown,
  range, volume concentration, 1-min return skew/kurtosis

Total: 24 features per asset per day.
All features are causal (use data up to day t only).

References:
  Wood et al. (2023), arXiv:2310.10500 — X-Trend architecture
  Garman & Klass (1980) — Efficient volatility estimator from OHLC
  Parkinson (1980) — High-low volatility estimator
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper feature constants
# ---------------------------------------------------------------------------
RETURN_WINDOWS = [1, 21, 63, 126, 252]
MACD_PAIRS = [(8, 24), (16, 48), (32, 96)]
VOL_SPAN = 60
MACD_NORM_WINDOW = 63
MACD_STD_WINDOW = 126

# ---------------------------------------------------------------------------
# Feature name registry
# ---------------------------------------------------------------------------
PAPER_FEATURE_NAMES = [
    "norm_ret_1d",
    "norm_ret_21d",
    "norm_ret_63d",
    "norm_ret_126d",
    "norm_ret_252d",
    "macd_8_24",
    "macd_16_48",
    "macd_32_96",
]

INTRADAY_FEATURE_NAMES = [
    "intra_rvol_1min",       # Realized vol from 1-min returns (annualized)
    "intra_gk_vol",          # Garman-Klass volatility (from OHLC, annualized)
    "intra_parkinson_vol",   # Parkinson vol (from high-low, annualized)
    "intra_overnight_gap",   # (today open - yesterday close) / yesterday close
    "intra_oc_return",       # (close - open) / open
    "intra_first_half_ret",  # 1st half return
    "intra_second_half_ret", # 2nd half return
    "intra_vwap_dev",        # (close - TWAP) / open
    "intra_close_position",  # (close - low) / (high - low)
    "intra_orb_width",       # Opening range (15min) width / open
    "intra_orb_break_up",    # 1 if rest-of-day high > ORB high
    "intra_orb_break_down",  # 1 if rest-of-day low < ORB low
    "intra_max_dd",          # Max intraday drawdown
    "intra_range_norm",      # (high - low) / open
    "intra_1min_skew",       # Skewness of 1-min returns
    "intra_1min_kurt",       # Kurtosis of 1-min returns
]

FEATURE_NAMES = PAPER_FEATURE_NAMES + INTRADAY_FEATURE_NAMES
N_PAPER_FEATURES = len(PAPER_FEATURE_NAMES)  # 8
N_INTRADAY_FEATURES = len(INTRADAY_FEATURE_NAMES)  # 16
N_FEATURES = len(FEATURE_NAMES)  # 24

# ---------------------------------------------------------------------------
# Symbol → Kite spot directory mapping
# ---------------------------------------------------------------------------
_SYMBOL_TO_SPOT = {
    "NIFTY 50": "NIFTY_SPOT",
    "NIFTY BANK": "BANKNIFTY_SPOT",
    "NIFTY FINANCIAL SERVICES": "FINNIFTY_SPOT",
    "NIFTY MIDCAP SELECT": "MIDCPNIFTY_SPOT",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average via pandas."""
    return pd.Series(x).ewm(span=span, min_periods=span).mean().values


def _ewm_std(x: np.ndarray, span: int) -> np.ndarray:
    """Exponential weighted moving standard deviation."""
    return pd.Series(x).ewm(span=span, min_periods=span).std().values


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation (ddof=1)."""
    return pd.Series(x).rolling(window, min_periods=window).std(ddof=1).values


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window, min_periods=window).mean().values


# ---------------------------------------------------------------------------
# Intraday feature extraction from 1-min bars
# ---------------------------------------------------------------------------

def compute_intraday_features_single_day(
    bars: pd.DataFrame,
    prev_close: Optional[float] = None,
) -> dict[str, float]:
    """Compute all 16 intraday features from one day of 1-min OHLCV bars.

    Parameters
    ----------
    bars : pd.DataFrame
        1-min bars with columns: open, high, low, close, volume.
        Must have at least 10 bars.
    prev_close : float, optional
        Previous day's close (for overnight gap). NaN if unavailable.

    Returns
    -------
    dict mapping feature name → float value
    """
    o = bars["open"].values.astype(np.float64)
    h = bars["high"].values.astype(np.float64)
    lo = bars["low"].values.astype(np.float64)
    c = bars["close"].values.astype(np.float64)

    n_bars = len(bars)
    day_open = o[0]
    day_close = c[-1]
    day_high = np.nanmax(h)
    day_low = np.nanmin(lo)
    day_range = day_high - day_low

    feats: dict[str, float] = {}

    # --- Realized vol from 1-min returns (annualized) ---
    min_returns = np.diff(np.log(np.maximum(c, 1e-10)))
    if len(min_returns) > 10:
        feats["intra_rvol_1min"] = float(np.std(min_returns, ddof=1) * np.sqrt(375))
    else:
        feats["intra_rvol_1min"] = np.nan

    # --- Garman-Klass volatility (from per-bar OHLC) ---
    # GK = 0.5 * (log H/L)^2 - (2ln2 - 1) * (log C/O)^2
    log_hl = np.log(np.maximum(h, 1e-10)) - np.log(np.maximum(lo, 1e-10))
    log_co = np.log(np.maximum(c, 1e-10)) - np.log(np.maximum(o, 1e-10))
    gk_var = np.nanmean(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)
    feats["intra_gk_vol"] = float(np.sqrt(max(gk_var, 0)) * np.sqrt(375))

    # --- Parkinson volatility (from high-low) ---
    parkinson_var = np.nanmean(log_hl**2) / (4 * np.log(2))
    feats["intra_parkinson_vol"] = float(np.sqrt(max(parkinson_var, 0)) * np.sqrt(375))

    # --- Overnight gap ---
    if prev_close is not None and prev_close > 0 and not np.isnan(prev_close):
        feats["intra_overnight_gap"] = float((day_open - prev_close) / prev_close)
    else:
        feats["intra_overnight_gap"] = np.nan

    # --- Open-to-close return ---
    feats["intra_oc_return"] = float((day_close - day_open) / max(day_open, 1e-10))

    # --- First/second half momentum ---
    mid_idx = n_bars // 2
    if mid_idx > 0:
        feats["intra_first_half_ret"] = float((c[mid_idx] - day_open) / max(day_open, 1e-10))
        feats["intra_second_half_ret"] = float(
            (day_close - c[mid_idx]) / max(c[mid_idx], 1e-10)
        )
    else:
        feats["intra_first_half_ret"] = np.nan
        feats["intra_second_half_ret"] = np.nan

    # --- TWAP deviation (VWAP proxy for indices with 0 volume) ---
    typical = (h + lo + c) / 3.0
    twap = np.nanmean(typical)
    feats["intra_vwap_dev"] = float((day_close - twap) / max(day_open, 1e-10))

    # --- Close position in day range ---
    feats["intra_close_position"] = float(
        (day_close - day_low) / max(day_range, 0.01)
    )

    # --- Opening Range Breakout (first 15 min) ---
    orb_bars = min(15, n_bars)
    orb_high = np.nanmax(h[:orb_bars])
    orb_low = np.nanmin(lo[:orb_bars])
    feats["intra_orb_width"] = float((orb_high - orb_low) / max(day_open, 1e-10))

    if n_bars > orb_bars:
        rest_high = np.nanmax(h[orb_bars:])
        rest_low = np.nanmin(lo[orb_bars:])
        feats["intra_orb_break_up"] = 1.0 if rest_high > orb_high else 0.0
        feats["intra_orb_break_down"] = 1.0 if rest_low < orb_low else 0.0
    else:
        feats["intra_orb_break_up"] = 0.0
        feats["intra_orb_break_down"] = 0.0

    # --- Max intraday drawdown ---
    cum_max = np.maximum.accumulate(c)
    dd = np.where(cum_max > 0, (cum_max - c) / cum_max, 0.0)
    feats["intra_max_dd"] = float(np.nanmax(dd))

    # --- Normalized range ---
    feats["intra_range_norm"] = float(day_range / max(day_open, 1e-10))

    # --- 1-min return skew & kurtosis ---
    if len(min_returns) > 30:
        mr = pd.Series(min_returns)
        feats["intra_1min_skew"] = float(mr.skew())
        feats["intra_1min_kurt"] = float(mr.kurtosis())
    else:
        feats["intra_1min_skew"] = np.nan
        feats["intra_1min_kurt"] = np.nan

    return feats


def load_intraday_features_for_asset(
    spot_dir: Path,
    dates: Sequence[str],
) -> pd.DataFrame:
    """Load 1-min bars and compute daily intraday features for one asset.

    Parameters
    ----------
    spot_dir : Path
        Kite spot directory (e.g., .../kite_1min/NIFTY_SPOT/)
    dates : sequence of str
        Date strings "YYYY-MM-DD" to process.

    Returns
    -------
    DataFrame indexed by date string with 16 intraday feature columns.
    """
    import pyarrow.parquet as pq

    records: list[dict] = []
    prev_close: Optional[float] = None

    for d_str in dates:
        d_dir = spot_dir / f"date={d_str}"
        if not d_dir.exists():
            prev_close = None
            continue

        pfiles = list(d_dir.glob("*.parquet"))
        if not pfiles:
            prev_close = None
            continue

        try:
            bars = pq.read_table(pfiles[0]).to_pandas()
            if bars.empty or len(bars) < 10:
                prev_close = None
                continue

            for col in ["open", "high", "low", "close", "volume"]:
                if col in bars.columns:
                    bars[col] = pd.to_numeric(bars[col], errors="coerce")

            feats = compute_intraday_features_single_day(bars, prev_close)
            feats["date"] = d_str
            records.append(feats)

            prev_close = float(bars["close"].iloc[-1])
        except Exception:
            logger.debug("Intraday feature extraction failed for %s", d_str)
            prev_close = None
            continue

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).set_index("date").sort_index()


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class TrendFeatureBuilder:
    """Builds paper features (8) + intraday features (16) = 24 total.

    If 1-min bar data is unavailable, falls back to 8 paper-only features
    with the intraday columns filled as NaN.

    Usage::

        builder = TrendFeatureBuilder()
        features, targets, vol = builder.build(prices_df, symbols)

        # Or with intraday data:
        features, targets, vol = builder.build(
            prices_df, symbols, kite_1min_dir=Path("common/data/kite_1min")
        )
    """

    def __init__(self, vol_span: int = VOL_SPAN) -> None:
        self.vol_span = vol_span

    def build(
        self,
        prices_df: pd.DataFrame,
        symbols: Sequence[str],
        kite_1min_dir: Optional[Path] = None,
        symbol_to_spot: Optional[dict[str, str]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build features for multiple assets.

        Parameters
        ----------
        prices_df : pd.DataFrame
            Must have a ``date`` column and one column per symbol with close prices.
        symbols : sequence of str
            Column names for each asset.
        kite_1min_dir : Path, optional
            Root of Kite 1-min data. If provided, computes intraday features.
        symbol_to_spot : dict, optional
            Map from symbol name → Kite spot directory name.
            Defaults to _SYMBOL_TO_SPOT.

        Returns
        -------
        features : np.ndarray, shape (n_days, n_assets, N_FEATURES)
        targets : np.ndarray, shape (n_days, n_assets)
        vol : np.ndarray, shape (n_days, n_assets)
        """
        if symbol_to_spot is None:
            symbol_to_spot = _SYMBOL_TO_SPOT

        n_days = len(prices_df)
        n_assets = len(symbols)
        n_feat = N_FEATURES if kite_1min_dir else N_PAPER_FEATURES

        features = np.full((n_days, n_assets, n_feat), np.nan)
        targets = np.full((n_days, n_assets), np.nan)
        vol_out = np.full((n_days, n_assets), np.nan)

        # Date strings for intraday lookup
        dates_str = [str(d)[:10] for d in prices_df["date"].values]

        for j, sym in enumerate(symbols):
            close = prices_df[sym].values.astype(np.float64)

            # Paper features (8)
            paper_feat, tgt_j, vol_j = self._build_paper_features(close)
            features[:, j, :N_PAPER_FEATURES] = paper_feat
            targets[:, j] = tgt_j
            vol_out[:, j] = vol_j

            # Intraday features (16) from 1-min bars
            if kite_1min_dir is not None:
                spot_name = symbol_to_spot.get(sym)
                if spot_name is None:
                    # Try partial match
                    for k, v in symbol_to_spot.items():
                        if k.lower() in sym.lower() or sym.lower() in k.lower():
                            spot_name = v
                            break

                if spot_name:
                    spot_dir = kite_1min_dir / spot_name
                    if spot_dir.exists():
                        intra_df = load_intraday_features_for_asset(spot_dir, dates_str)

                        if not intra_df.empty:
                            # Align to our date index
                            for fi, fname in enumerate(INTRADAY_FEATURE_NAMES):
                                if fname in intra_df.columns:
                                    for di, d_str in enumerate(dates_str):
                                        if d_str in intra_df.index:
                                            val = intra_df.loc[d_str, fname]
                                            if not np.isnan(val):
                                                features[di, j, N_PAPER_FEATURES + fi] = val

                            logger.info(
                                "%s: loaded %d days of intraday features from %s",
                                sym, len(intra_df), spot_name,
                            )
                        else:
                            logger.warning("%s: no intraday data from %s", sym, spot_name)
                    else:
                        logger.warning("%s: spot dir not found: %s", sym, spot_dir)

        return features, targets, vol_out

    def build_single(self, close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build paper-only features for a single asset (backward compat).

        Returns
        -------
        features : (n, 8)
        targets : (n,)
        vol : (n,)
        """
        return self._build_paper_features(close)

    def _build_paper_features(
        self, close: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the 8 paper-specified features for a single asset.

        Returns
        -------
        features : (n, 8)
        targets : (n,) — vol-scaled next-day return
        vol : (n,)
        """
        n = len(close)
        log_close = np.log(close)

        daily_ret = np.full(n, np.nan)
        daily_ret[1:] = np.diff(log_close)

        sigma = _ewm_std(daily_ret, self.vol_span)
        safe_sigma = np.where((sigma > 0) & ~np.isnan(sigma), sigma, np.nan)

        features = np.full((n, N_PAPER_FEATURES), np.nan)

        # 5 normalized returns
        for i, window in enumerate(RETURN_WINDOWS):
            ret = np.full(n, np.nan)
            if n > window:
                ret[window:] = log_close[window:] - log_close[:-window]
            features[:, i] = ret / (safe_sigma * np.sqrt(window))

        # 3 MACD signals
        for i, (short, long) in enumerate(MACD_PAIRS):
            ema_short = _ema(close, short)
            ema_long = _ema(close, long)
            raw_macd = ema_short - ema_long

            norm_std = _rolling_std(close, MACD_NORM_WINDOW)
            safe_norm = np.where((norm_std > 0) & ~np.isnan(norm_std), norm_std, np.nan)
            q = raw_macd / safe_norm

            q_std = _rolling_std(q, MACD_STD_WINDOW)
            safe_q_std = np.where((q_std > 0) & ~np.isnan(q_std), q_std, np.nan)
            features[:, 5 + i] = q / safe_q_std

        # Target: vol-scaled next-day return
        targets = np.full(n, np.nan)
        if n > 1:
            next_ret = np.full(n, np.nan)
            next_ret[:-1] = log_close[1:] - log_close[:-1]
            targets = next_ret / safe_sigma

        return features, targets, sigma
