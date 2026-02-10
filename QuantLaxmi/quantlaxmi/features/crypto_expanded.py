"""Expanded crypto feature sources -- funding, OI, positioning, breadth, liquidation.

Adds ~30 daily features from Binance Futures endpoints and derived analytics
beyond the basic BTC/ETH OHLCV features in crypto_alpha.py.

Feature classes
---------------
1. FundingRateRegime   (prefix ``fr_``)  -- 7 features from 8h funding rates
2. OpenInterestDynamics (prefix ``oi_``) -- 6 features from OI snapshots
3. LongShortPositioning (prefix ``ls_``) -- 7 features from L/S ratio APIs
4. AltcoinBreadth      (prefix ``ab_``)  -- 6 features from BTC/ETH/SOL klines
5. LiquidationProxy    (prefix ``liq_``) -- 4 features from price/volume extremes

Design principles
-----------------
- Every feature is **causal** -- only uses data available at or before that date.
- Missing API data degrades gracefully to NaN (never crashes).
- All computations are vectorized numpy/pandas -- no loops over rows.
- Rolling windows only; no expanding windows (bounded lookback).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - mean) / std, causal."""
    m = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=1)
    return (s - m) / sd.replace(0, np.nan)


def _safe_div(a, b):
    """Element-wise a/b returning NaN where b is zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) > 1e-12, a / b, np.nan)
    return result


# ============================================================================
# 1. FundingRateRegime
# ============================================================================


@dataclass
class FundingRateRegime:
    """Features from Binance perpetual funding rates (8-hourly, aggregated daily).

    Funding rates reveal market positioning: positive = longs pay shorts,
    negative = shorts pay longs. Extreme funding precedes reversals.
    """

    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])

    def compute(self, funding_df: pd.DataFrame) -> pd.DataFrame:
        """Compute funding rate features from a DataFrame of funding records.

        Parameters
        ----------
        funding_df : pd.DataFrame
            Must have columns: ``fundingRate``, ``symbol``.
            Index: UTC DatetimeIndex (8-hourly).

        Returns
        -------
        pd.DataFrame
            Daily features indexed by date, prefixed ``fr_``.
        """
        if funding_df is None or funding_df.empty:
            return pd.DataFrame()

        if "fundingRate" not in funding_df.columns:
            return pd.DataFrame()

        # Assign date column for daily aggregation
        df = funding_df.copy()
        df["_date"] = df.index.normalize()

        # Average across symbols per 8h period, then aggregate daily
        daily = df.groupby("_date").agg(
            rate_mean=("fundingRate", "mean"),
            rate_std=("fundingRate", "std"),
            rate_max=("fundingRate", "max"),
            rate_min=("fundingRate", "min"),
            rate_count=("fundingRate", "count"),
        )
        daily.index.name = "date"

        out = pd.DataFrame(index=daily.index)

        # Mean 8h rate across today's 3 settlements
        out["fr_mean_8h"] = daily["rate_mean"]

        # Intra-day volatility of funding
        out["fr_std_8h"] = daily["rate_std"].fillna(0.0)

        # Skewness of funding: (mean - median) proxy via (max+min)/2 vs mean
        midpoint = (daily["rate_max"] + daily["rate_min"]) / 2.0
        skew_raw = _safe_div(
            (daily["rate_mean"] - midpoint).values,
            daily["rate_std"].replace(0, np.nan).values,
        )
        out["fr_skew"] = pd.Series(np.asarray(skew_raw).ravel(), index=daily.index)

        # Maximum absolute rate (extreme funding detection)
        out["fr_max_abs"] = np.maximum(
            daily["rate_max"].abs().values,
            daily["rate_min"].abs().values,
        )

        # Z-score of daily mean funding over 21-day window
        out["fr_z_score"] = _rolling_zscore(out["fr_mean_8h"], 21)

        # 3-day momentum of funding (change over 3 days)
        out["fr_momentum_3d"] = out["fr_mean_8h"].diff(3)

        # Count of "extreme" funding events (|rate| > 0.001 = 0.1%) in last 7 days
        extreme = (daily["rate_mean"].abs() > 0.001).astype(float)
        out["fr_extreme_count"] = extreme.rolling(7, min_periods=1).sum()

        return out

    @staticmethod
    def from_connector(connector, start_date: str, end_date: str,
                       symbols: list[str] | None = None) -> pd.DataFrame:
        """Convenience: fetch funding history from a BinanceConnector and compute.

        Parameters
        ----------
        connector : BinanceConnector
            Must have ``fetch_funding_rate_history`` method.
        start_date, end_date : str
            Date range "YYYY-MM-DD".
        symbols : list[str], optional
            Defaults to BTCUSDT + ETHUSDT.

        Returns
        -------
        pd.DataFrame
            Daily funding rate features.
        """
        if symbols is None:
            symbols = ["BTCUSDT", "ETHUSDT"]

        frames = []
        for sym in symbols:
            try:
                df = connector.fetch_funding_rate_history(
                    symbol=sym,
                    start_time=start_date,
                    end_time=end_date,
                    limit=1000,
                )
                if df is not None and not df.empty:
                    df["symbol"] = sym
                    frames.append(df)
            except Exception as exc:
                logger.warning("Failed to fetch funding for %s: %s", sym, exc)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames).sort_index()
        return FundingRateRegime(symbols=symbols).compute(combined)


# ============================================================================
# 2. OpenInterestDynamics
# ============================================================================


@dataclass
class OpenInterestDynamics:
    """Features from daily open interest snapshots.

    OI dynamics reveal market conviction: rising OI with rising price = strong
    trend; rising OI with falling price = aggressive shorting.
    """

    def compute(self, oi_df: pd.DataFrame) -> pd.DataFrame:
        """Compute OI dynamics features.

        Parameters
        ----------
        oi_df : pd.DataFrame
            Must have columns: ``oi_btc`` (BTC OI in USD), ``oi_eth`` (ETH OI in USD).
            Index: daily DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Daily features prefixed ``oi_``.
        """
        if oi_df is None or oi_df.empty:
            return pd.DataFrame()

        out = pd.DataFrame(index=oi_df.index)

        # Raw OI in millions USD
        if "oi_btc" in oi_df.columns:
            btc_oi = oi_df["oi_btc"].astype(float)
            out["oi_btc_usd_m"] = btc_oi / 1e6
        else:
            out["oi_btc_usd_m"] = np.nan

        if "oi_eth" in oi_df.columns:
            eth_oi = oi_df["oi_eth"].astype(float)
            out["oi_eth_usd_m"] = eth_oi / 1e6
        else:
            out["oi_eth_usd_m"] = np.nan

        # Total OI z-score over 21-day window
        total_oi = out["oi_btc_usd_m"].fillna(0) + out["oi_eth_usd_m"].fillna(0)
        out["oi_total_z21"] = _rolling_zscore(total_oi, 21)

        # 5-day OI momentum (pct change)
        out["oi_momentum_5d"] = total_oi.pct_change(5)

        # OI expanding: 1 if today's OI > 20-day SMA
        oi_sma = total_oi.rolling(20, min_periods=20).mean()
        out["oi_expanding"] = (total_oi > oi_sma).astype(np.int8)

        # BTC concentration: BTC OI / total OI
        total_nonzero = total_oi.replace(0, np.nan)
        out["oi_concentration"] = out["oi_btc_usd_m"] / total_nonzero

        return out


# ============================================================================
# 3. LongShortPositioning
# ============================================================================


@dataclass
class LongShortPositioning:
    """Features from Binance global + top-trader long/short ratios.

    When retail is overwhelmingly long, smart money often fades.
    Divergence between global and top-trader ratios signals institutional moves.
    """

    def compute(
        self,
        global_ls: pd.DataFrame | None = None,
        top_ls: pd.DataFrame | None = None,
        taker_ls: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute positioning features.

        Parameters
        ----------
        global_ls : pd.DataFrame, optional
            From ``fetch_long_short_ratio``.
            Columns: longShortRatio, longAccount, shortAccount.
        top_ls : pd.DataFrame, optional
            From ``fetch_top_long_short_ratio``.
        taker_ls : pd.DataFrame, optional
            From ``fetch_taker_long_short_ratio``.
            Columns: buySellRatio, buyVol, sellVol.

        Returns
        -------
        pd.DataFrame
            Daily features prefixed ``ls_``.
        """
        out_frames = []

        # --- Global L/S ratio ---
        if global_ls is not None and not global_ls.empty and "longShortRatio" in global_ls.columns:
            gdf = global_ls.copy()
            gdf["_date"] = gdf.index.normalize()
            daily_g = gdf.groupby("_date").agg(
                ratio_last=("longShortRatio", "last"),
            )
            daily_g.index.name = "date"

            g_out = pd.DataFrame(index=daily_g.index)
            g_out["ls_ratio_global"] = daily_g["ratio_last"]
            g_out["ls_ratio_z_5d"] = _rolling_zscore(g_out["ls_ratio_global"], 5)
            out_frames.append(g_out)

        # --- Top-trader L/S ratio ---
        if top_ls is not None and not top_ls.empty and "longShortRatio" in top_ls.columns:
            tdf = top_ls.copy()
            tdf["_date"] = tdf.index.normalize()
            daily_t = tdf.groupby("_date").agg(
                ratio_last=("longShortRatio", "last"),
            )
            daily_t.index.name = "date"

            t_out = pd.DataFrame(index=daily_t.index)
            t_out["ls_top_ratio"] = daily_t["ratio_last"]
            out_frames.append(t_out)

        # --- Divergence between global and top ---
        if len(out_frames) >= 2:
            merged = out_frames[0].join(out_frames[1], how="outer")
            div_out = pd.DataFrame(index=merged.index)
            div_out["ls_divergence"] = (
                merged.get("ls_ratio_global", pd.Series(dtype=float))
                - merged.get("ls_top_ratio", pd.Series(dtype=float))
            )
            out_frames.append(div_out)

        # --- Taker buy/sell ratio ---
        if taker_ls is not None and not taker_ls.empty and "buySellRatio" in taker_ls.columns:
            tkdf = taker_ls.copy()
            tkdf["_date"] = tkdf.index.normalize()
            daily_tk = tkdf.groupby("_date").agg(
                bsr_last=("buySellRatio", "last"),
                buy_vol=("buyVol", "sum"),
                sell_vol=("sellVol", "sum"),
            )
            daily_tk.index.name = "date"

            tk_out = pd.DataFrame(index=daily_tk.index)
            total_vol = daily_tk["buy_vol"] + daily_tk["sell_vol"]
            tk_out["ls_taker_buy_pct"] = daily_tk["buy_vol"] / total_vol.replace(0, np.nan)
            tk_out["ls_taker_z_5d"] = _rolling_zscore(tk_out["ls_taker_buy_pct"], 5)

            # Taker flip: 1 if buy pct crosses 0.5 from below, -1 from above, 0 else
            above = (tk_out["ls_taker_buy_pct"] > 0.5).astype(int)
            tk_out["ls_taker_flip"] = above.diff().fillna(0).astype(np.int8)

            out_frames.append(tk_out)

        if not out_frames:
            return pd.DataFrame()

        # Join all sub-frames on date
        result = out_frames[0]
        for f in out_frames[1:]:
            result = result.join(f, how="outer")

        return result


# ============================================================================
# 4. AltcoinBreadth
# ============================================================================


@dataclass
class AltcoinBreadth:
    """Features from BTC + ETH + SOL daily klines -- cross-asset dynamics.

    ETH/BTC ratio, SOL momentum, and cross-crypto correlations reveal
    risk appetite and rotation patterns across the crypto ecosystem.
    """

    correlation_window: int = 20

    def compute(
        self,
        btc_df: pd.DataFrame,
        eth_df: pd.DataFrame,
        sol_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute altcoin breadth features from daily OHLCV DataFrames.

        Parameters
        ----------
        btc_df, eth_df : pd.DataFrame
            Must have ``close`` and ``volume`` columns.
            Index: daily DatetimeIndex.
        sol_df : pd.DataFrame, optional
            Same schema. If None, SOL features are NaN.

        Returns
        -------
        pd.DataFrame
            Daily features prefixed ``ab_``.
        """
        if btc_df is None or btc_df.empty or eth_df is None or eth_df.empty:
            return pd.DataFrame()

        # Align on dates
        btc_close = btc_df["close"].rename("btc")
        eth_close = eth_df["close"].rename("eth")
        aligned = pd.concat([btc_close, eth_close], axis=1, join="inner")

        if aligned.empty:
            return pd.DataFrame()

        out = pd.DataFrame(index=aligned.index)

        # ETH/BTC ratio
        btc_safe = aligned["btc"].replace(0, np.nan)
        out["ab_eth_btc_ratio"] = aligned["eth"] / btc_safe

        # ETH/BTC ratio z-score over 20 days
        out["ab_eth_btc_z20"] = _rolling_zscore(out["ab_eth_btc_ratio"], self.correlation_window)

        # SOL momentum (5-day log return)
        if sol_df is not None and not sol_df.empty and "close" in sol_df.columns:
            sol_close = sol_df["close"].reindex(aligned.index, method="ffill")
            log_sol = np.log(sol_close.clip(lower=0.01))
            out["ab_sol_momentum"] = log_sol.diff(5)
        else:
            out["ab_sol_momentum"] = np.nan

        # Altcoin spread: ETH return - BTC return (5d)
        btc_ret_5d = np.log(aligned["btc"].clip(lower=0.01)).diff(5)
        eth_ret_5d = np.log(aligned["eth"].clip(lower=0.01)).diff(5)
        out["ab_altcoin_spread"] = eth_ret_5d - btc_ret_5d

        # Rolling BTC-ETH return correlation
        btc_ret = aligned["btc"].pct_change()
        eth_ret = aligned["eth"].pct_change()
        out["ab_correlation_btc_eth"] = btc_ret.rolling(
            self.correlation_window, min_periods=self.correlation_window
        ).corr(eth_ret)

        # Breadth 2-of-3: how many of {BTC, ETH, SOL} have positive 5d return
        btc_pos = (btc_ret_5d > 0).astype(int)
        eth_pos = (eth_ret_5d > 0).astype(int)

        if sol_df is not None and not sol_df.empty and "close" in sol_df.columns:
            sol_close = sol_df["close"].reindex(aligned.index, method="ffill")
            sol_ret_5d = np.log(sol_close.clip(lower=0.01)).diff(5)
            sol_pos = (sol_ret_5d > 0).astype(int)
        else:
            sol_pos = pd.Series(0, index=aligned.index)

        out["ab_breadth_2of3"] = (btc_pos + eth_pos + sol_pos).astype(np.int8)

        return out


# ============================================================================
# 5. LiquidationProxy
# ============================================================================


@dataclass
class LiquidationProxy:
    """Estimate liquidation events from price and volume extremes.

    Actual liquidation data requires authenticated Binance endpoints.
    This proxy uses extreme moves + volume spikes to detect cascade events.
    """

    z_window: int = 21
    vol_window: int = 21

    def compute(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Compute liquidation proxy features from daily OHLCV.

        Parameters
        ----------
        ohlcv_df : pd.DataFrame
            Must have ``close``, ``high``, ``low``, ``volume`` columns.
            Index: daily DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Daily features prefixed ``liq_``.
        """
        if ohlcv_df is None or ohlcv_df.empty:
            return pd.DataFrame()

        needed = {"close", "high", "low", "volume"}
        if not needed.issubset(set(ohlcv_df.columns)):
            return pd.DataFrame()

        df = ohlcv_df.copy()
        out = pd.DataFrame(index=df.index)

        # Extreme move z-score: daily range relative to rolling average range
        daily_range = (df["high"] - df["low"]).clip(lower=0)
        range_mean = daily_range.rolling(self.z_window, min_periods=self.z_window).mean()
        range_std = daily_range.rolling(self.z_window, min_periods=self.z_window).std(ddof=1)
        out["liq_extreme_move_z"] = (daily_range - range_mean) / range_std.replace(0, np.nan)

        # Volume spike: volume / 21-day SMA
        vol = df["volume"].astype(float)
        vol_sma = vol.rolling(self.vol_window, min_periods=self.vol_window).mean()
        out["liq_volume_spike"] = vol / vol_sma.replace(0, np.nan)

        # Cascade ratio: fraction of daily range that is the close-to-low move
        # (high cascade = price closed near lows = sell cascade)
        hl_range = (df["high"] - df["low"]).replace(0, np.nan)
        close_to_low = df["close"] - df["low"]
        out["liq_cascade_ratio"] = 1.0 - (close_to_low / hl_range)

        # Bounce strength: next day, we cannot use -- so use same-day recovery
        # Recovery from low: (close - low) / (high - low), high = bounced from low
        out["liq_bounce_strength"] = close_to_low / hl_range

        return out


# ============================================================================
# Aggregate builder for MegaFeatureBuilder integration
# ============================================================================


def build_crypto_expanded_features(
    start_date: str,
    end_date: str,
    binance_dir=None,
    connector=None,
) -> pd.DataFrame:
    """Build all crypto expanded features and merge into a single DataFrame.

    This function is called by MegaFeatureBuilder._build_crypto_expanded_features.

    It can work in two modes:
    1. **From stored data** (binance_dir): reads Parquet klines from disk
    2. **From live API** (connector): calls BinanceConnector methods

    If neither is available, returns empty DataFrame (graceful degradation).

    Parameters
    ----------
    start_date, end_date : str
        Date range "YYYY-MM-DD".
    binance_dir : Path, optional
        Path to Binance kline Parquet data.
    connector : BinanceConnector, optional
        Live connector for funding/OI/L-S data.

    Returns
    -------
    pd.DataFrame
        ~30 daily features with prefixes: fr_, oi_, ls_, ab_, liq_.
    """
    frames: list[pd.DataFrame] = []

    # --- Load daily klines from disk ---
    kline_data: dict[str, pd.DataFrame] = {}
    if binance_dir is not None:
        from pathlib import Path
        import pyarrow.parquet as pq

        bdir = Path(binance_dir)
        for coin in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            coin_dir = bdir / coin / "1d"
            if not coin_dir.exists():
                continue

            coin_frames: list[pd.DataFrame] = []
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
                    pf = pq.read_table(parquet_files[0]).to_pandas()
                    pf["_date_str"] = d_str
                    coin_frames.append(pf)
                except Exception:
                    continue

            if coin_frames:
                combined = pd.concat(coin_frames, ignore_index=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in combined.columns:
                        combined[col] = pd.to_numeric(combined[col], errors="coerce")
                combined["date"] = pd.to_datetime(combined["_date_str"])
                combined = combined.sort_values("date").set_index("date")
                combined = combined[~combined.index.duplicated(keep="last")]
                kline_data[coin] = combined

    # --- 1. Funding Rate Regime ---
    try:
        if connector is not None:
            fr_df = FundingRateRegime.from_connector(
                connector, start_date, end_date,
            )
            if fr_df is not None and not fr_df.empty:
                frames.append(fr_df)
    except Exception as exc:
        logger.warning("FundingRateRegime failed: %s", exc)

    # --- 2. Open Interest Dynamics ---
    # OI requires live snapshots; from stored data we can approximate from klines
    # For now, OI features only available with connector or pre-stored OI data
    # (We leave a hook for future stored OI support)

    # --- 3. Long/Short Positioning ---
    try:
        if connector is not None:
            global_ls_frames = []
            top_ls_frames = []
            taker_ls_frames = []

            for sym in ["BTCUSDT", "ETHUSDT"]:
                try:
                    g = connector.fetch_long_short_ratio(sym, period="1d", limit=30)
                    if g is not None and not g.empty:
                        global_ls_frames.append(g)
                except Exception:
                    pass

                try:
                    t = connector.fetch_top_long_short_ratio(sym, period="1d", limit=30)
                    if t is not None and not t.empty:
                        top_ls_frames.append(t)
                except Exception:
                    pass

                try:
                    tk = connector.fetch_taker_long_short_ratio(sym, period="1d", limit=30)
                    if tk is not None and not tk.empty:
                        taker_ls_frames.append(tk)
                except Exception:
                    pass

            global_ls = pd.concat(global_ls_frames).sort_index() if global_ls_frames else None
            top_ls = pd.concat(top_ls_frames).sort_index() if top_ls_frames else None
            taker_ls = pd.concat(taker_ls_frames).sort_index() if taker_ls_frames else None

            ls_df = LongShortPositioning().compute(global_ls, top_ls, taker_ls)
            if ls_df is not None and not ls_df.empty:
                frames.append(ls_df)
    except Exception as exc:
        logger.warning("LongShortPositioning failed: %s", exc)

    # --- 4. Altcoin Breadth ---
    try:
        btc_klines = kline_data.get("BTCUSDT")
        eth_klines = kline_data.get("ETHUSDT")
        sol_klines = kline_data.get("SOLUSDT")

        if btc_klines is not None and eth_klines is not None:
            ab_df = AltcoinBreadth().compute(btc_klines, eth_klines, sol_klines)
            if ab_df is not None and not ab_df.empty:
                frames.append(ab_df)
    except Exception as exc:
        logger.warning("AltcoinBreadth failed: %s", exc)

    # --- 5. Liquidation Proxy ---
    try:
        btc_klines = kline_data.get("BTCUSDT")
        if btc_klines is not None:
            liq_df = LiquidationProxy().compute(btc_klines)
            if liq_df is not None and not liq_df.empty:
                frames.append(liq_df)
    except Exception as exc:
        logger.warning("LiquidationProxy failed: %s", exc)

    if not frames:
        return pd.DataFrame()

    # Outer-join all frames on date
    result = frames[0]
    for f in frames[1:]:
        result = result.join(f, how="outer")

    # Filter to requested range
    result = result.loc[start_date:end_date]
    logger.info("crypto_expanded features: %d columns, %d rows", len(result.columns), len(result))

    return result
