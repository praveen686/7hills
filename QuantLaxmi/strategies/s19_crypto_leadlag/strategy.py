"""Crypto Overnight Lead-Lag Strategy for NIFTY.

Global risk assets (BTC, ETH) trade 24/7. Indian markets open at 09:15 IST
and close at 15:30 IST. Crypto price action during the overnight window
(15:30 IST previous day to 09:15 IST current day) contains predictive
information about the direction of the Indian market open.

Architecture:
    1. For each Indian trading day T, compute overnight crypto features:
       - BTC/ETH overnight return: close@15:30 IST (T-1) to price@09:15 IST (T)
       - BTC overnight volatility: std of 5-min returns in overnight window
       - BTC 24h / 72h momentum
       - Rolling 20-day crypto-equity correlation
    2. Signal construction:
       - Base = 0.5*btc_overnight + 0.3*eth_overnight + 0.2*btc_24h_momentum
       - Scaled by rolling crypto-equity correlation
       - Z-scored over 20-day window
       - LONG when z > 1.0, SHORT when z < -1.0
    3. Execution:
       - Enter at 09:16 IST (first reliable 1-min bar)
       - Exit at 15:00 IST (before close, same day) — pure intraday
       - OR exit if NIFTY moves > 1% against position (stop loss)
    4. Position sizing:
       - Proportional to |z|/3, capped at 25% of capital

Cost: 3 index points round-trip (NIFTY futures).

Fully causal: all crypto data is from BEFORE the Indian market open.
No look-ahead bias.

Sharpe protocol: ddof=1, sqrt(252), all daily returns including flat days.

Author: AlphaForge
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timezone constants
# ---------------------------------------------------------------------------

IST_OFFSET = timedelta(hours=5, minutes=30)
IST = timezone(IST_OFFSET)
UTC = timezone.utc

# Indian market hours (IST)
MARKET_OPEN_H, MARKET_OPEN_M = 9, 15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30
EXIT_H, EXIT_M = 15, 0  # Exit before close

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------

ZSCORE_WINDOW: int = 20         # Rolling z-score lookback
ZSCORE_ENTRY: float = 1.0       # Entry threshold
CORR_LOOKBACK: int = 20         # Rolling crypto-equity correlation window
MIN_WARMUP: int = 25            # Minimum trading days before first signal

# Signal blending weights
W_BTC_OVERNIGHT: float = 0.50
W_ETH_OVERNIGHT: float = 0.30
W_BTC_MOMENTUM: float = 0.20

# Position sizing
MAX_POSITION_FRAC: float = 0.25  # Max 25% of capital
Z_SCALING_DIVISOR: float = 3.0   # position_frac = |z| / 3, capped

# Stop loss: exit if NIFTY moves > 1% against position
INTRADAY_STOP_PCT: float = 0.01

# Cost: 3 points round-trip for NIFTY futures
COST_POINTS_RT: float = 3.0
NIFTY_APPROX_LEVEL: float = 23000.0
COST_FRACTION_RT: float = COST_POINTS_RT / NIFTY_APPROX_LEVEL

# Binance fetch parameters
CRYPTO_INTERVAL: str = "5m"     # 5-minute candles for overnight vol
CRYPTO_1D_INTERVAL: str = "1d"  # Daily candles for momentum


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OvernightFeatures:
    """Crypto overnight features for a single Indian trading day."""
    trade_date: date
    btc_overnight_ret: float
    eth_overnight_ret: float
    btc_overnight_vol: float
    btc_24h_momentum: float
    btc_72h_momentum: float
    raw_signal: float
    crypto_equity_corr: float
    scaled_signal: float
    z_score: float


@dataclass
class IntradayTrade:
    """A single intraday trade record."""
    trade_date: date
    direction: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    position_frac: float
    pnl_pct: float
    z_score: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Full backtest output."""
    daily_df: pd.DataFrame
    trades: list[IntradayTrade]
    features_df: pd.DataFrame
    stats: dict


# ---------------------------------------------------------------------------
# Crypto data fetching with caching
# ---------------------------------------------------------------------------

class CryptoDataCache:
    """Fetches and caches Binance crypto data for the backtest.

    Fetches BTC/ETH 5-minute and 1-day klines in bulk, then provides
    lookup methods for specific time windows. This minimizes API calls
    since Binance rate limits are generous for public endpoints.
    """

    def __init__(self, start_date: date, end_date: date):
        self.start_date = start_date
        self.end_date = end_date
        self._btc_5m: pd.DataFrame | None = None
        self._eth_5m: pd.DataFrame | None = None
        self._btc_1d: pd.DataFrame | None = None
        self._eth_1d: pd.DataFrame | None = None

    def load(self) -> None:
        """Bulk-fetch all required crypto data."""
        # Lazy import to avoid circular dependencies
        sys_path_root = Path(__file__).resolve().parents[2]
        if str(sys_path_root) not in sys.path:
            sys.path.insert(0, str(sys_path_root))

        from data.connectors.binance_connector import BinanceConnector

        bc = BinanceConnector()

        # We need data from a few days before the start (for 72h momentum)
        # and the overnight window starts the evening before start_date
        fetch_start = datetime(
            self.start_date.year, self.start_date.month, self.start_date.day,
            tzinfo=UTC,
        ) - timedelta(days=5)
        fetch_end = datetime(
            self.end_date.year, self.end_date.month, self.end_date.day,
            hour=23, minute=59, tzinfo=UTC,
        )

        logger.info(
            "Fetching BTC/ETH data from %s to %s ...",
            fetch_start.strftime("%Y-%m-%d"),
            fetch_end.strftime("%Y-%m-%d"),
        )

        # 5-minute candles (chunked for large ranges)
        self._btc_5m = bc.fetch_klines_chunked(
            "BTCUSDT", "5m",
            start_date=fetch_start, end_date=fetch_end,
            rate_limit_sleep=0.05,
        )
        logger.info("BTC 5m: %d candles", len(self._btc_5m))

        time.sleep(0.2)

        self._eth_5m = bc.fetch_klines_chunked(
            "ETHUSDT", "5m",
            start_date=fetch_start, end_date=fetch_end,
            rate_limit_sleep=0.05,
        )
        logger.info("ETH 5m: %d candles", len(self._eth_5m))

        time.sleep(0.2)

        # Daily candles for longer-term momentum
        daily_start = fetch_start - timedelta(days=5)
        self._btc_1d = bc.fetch_klines_chunked(
            "BTCUSDT", "1d",
            start_date=daily_start, end_date=fetch_end,
        )
        logger.info("BTC 1d: %d candles", len(self._btc_1d))

        time.sleep(0.1)

        self._eth_1d = bc.fetch_klines_chunked(
            "ETHUSDT", "1d",
            start_date=daily_start, end_date=fetch_end,
        )
        logger.info("ETH 1d: %d candles", len(self._eth_1d))

    def get_5m_slice(
        self, symbol: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp
    ) -> pd.DataFrame:
        """Return 5-min candles for a symbol in [start_utc, end_utc]."""
        df = self._btc_5m if symbol == "BTCUSDT" else self._eth_5m
        if df is None or df.empty:
            return pd.DataFrame()
        mask = (df.index >= start_utc) & (df.index <= end_utc)
        return df.loc[mask]

    def get_daily(self, symbol: str) -> pd.DataFrame:
        """Return daily candles for a symbol."""
        if symbol == "BTCUSDT":
            return self._btc_1d if self._btc_1d is not None else pd.DataFrame()
        return self._eth_1d if self._eth_1d is not None else pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature computation (all causal)
# ---------------------------------------------------------------------------

def _ist_to_utc(year: int, month: int, day: int, hour: int, minute: int) -> pd.Timestamp:
    """Create a UTC timestamp from IST date/time components."""
    ist_dt = datetime(year, month, day, hour, minute, tzinfo=IST)
    return pd.Timestamp(ist_dt).tz_convert(UTC)


def _compute_overnight_return(
    cache: CryptoDataCache,
    symbol: str,
    prev_close_utc: pd.Timestamp,
    curr_open_utc: pd.Timestamp,
) -> float:
    """Compute the overnight crypto return.

    Return from the 5-min candle nearest to prev market close
    to the 5-min candle nearest to current market open.
    """
    # Get the close price nearest to prev_close_utc
    # Look in a window around that time
    window_start = prev_close_utc - pd.Timedelta(minutes=10)
    window_end = prev_close_utc + pd.Timedelta(minutes=10)
    df_close = cache.get_5m_slice(symbol, window_start, window_end)

    # Get the price nearest to curr_open_utc
    window_start_open = curr_open_utc - pd.Timedelta(minutes=10)
    window_end_open = curr_open_utc + pd.Timedelta(minutes=10)
    df_open = cache.get_5m_slice(symbol, window_start_open, window_end_open)

    if df_close.empty or df_open.empty:
        return np.nan

    # Use the candle closest to the target time
    close_idx = (df_close.index - prev_close_utc).map(abs).argmin()
    open_idx = (df_open.index - curr_open_utc).map(abs).argmin()

    price_at_prev_close = float(df_close["close"].iloc[close_idx])
    price_at_curr_open = float(df_open["close"].iloc[open_idx])

    if price_at_prev_close <= 0:
        return np.nan

    return np.log(price_at_curr_open / price_at_prev_close)


def _compute_overnight_volatility(
    cache: CryptoDataCache,
    prev_close_utc: pd.Timestamp,
    curr_open_utc: pd.Timestamp,
) -> float:
    """Compute BTC overnight volatility as std of 5-min log returns."""
    df = cache.get_5m_slice("BTCUSDT", prev_close_utc, curr_open_utc)
    if df.empty or len(df) < 5:
        return np.nan

    log_rets = np.log(df["close"].values[1:] / df["close"].values[:-1])
    valid = log_rets[np.isfinite(log_rets)]
    if len(valid) < 3:
        return np.nan

    return float(np.std(valid, ddof=1))


def _compute_btc_momentum(
    cache: CryptoDataCache,
    curr_open_utc: pd.Timestamp,
    hours: int,
) -> float:
    """Compute BTC return over the last N hours ending at curr_open_utc."""
    start = curr_open_utc - pd.Timedelta(hours=hours)
    df = cache.get_5m_slice("BTCUSDT", start, curr_open_utc)
    if df.empty or len(df) < 2:
        return np.nan

    p_start = float(df["close"].iloc[0])
    p_end = float(df["close"].iloc[-1])
    if p_start <= 0:
        return np.nan
    return np.log(p_end / p_start)


def compute_daily_crypto_return(cache: CryptoDataCache, symbol: str) -> pd.Series:
    """Compute daily log returns for a crypto symbol from daily candles.

    Returns a Series indexed by date with daily log returns.
    """
    df = cache.get_daily(symbol)
    if df.empty:
        return pd.Series(dtype=float)

    # Convert UTC index to IST dates for alignment with Indian trading days
    df_copy = df.copy()
    df_copy["ist_date"] = df_copy.index.tz_convert(IST).date
    df_copy = df_copy.drop_duplicates(subset=["ist_date"], keep="last")

    rets = np.log(df_copy["close"].values[1:] / df_copy["close"].values[:-1])
    dates = df_copy["ist_date"].values[1:]

    return pd.Series(rets, index=dates)


def compute_all_overnight_features(
    cache: CryptoDataCache,
    nifty_trading_dates: list[date],
    nifty_daily_returns: pd.Series,
) -> pd.DataFrame:
    """Compute overnight crypto features for each trading day.

    Parameters
    ----------
    cache : CryptoDataCache
        Pre-loaded crypto data.
    nifty_trading_dates : list[date]
        Sorted list of Indian trading dates.
    nifty_daily_returns : pd.Series
        NIFTY daily log returns indexed by date.

    Returns
    -------
    pd.DataFrame with one row per trading day and all crypto features.
    """
    btc_daily_rets = compute_daily_crypto_return(cache, "BTCUSDT")

    records = []

    for i, trade_date in enumerate(nifty_trading_dates):
        if i == 0:
            # Need previous trading day
            records.append(_empty_feature_row(trade_date))
            continue

        prev_date = nifty_trading_dates[i - 1]

        # IST timestamps for the overnight window
        # Previous day market close: 15:30 IST
        prev_close_utc = _ist_to_utc(
            prev_date.year, prev_date.month, prev_date.day,
            MARKET_CLOSE_H, MARKET_CLOSE_M,
        )
        # Current day market open: 09:15 IST
        curr_open_utc = _ist_to_utc(
            trade_date.year, trade_date.month, trade_date.day,
            MARKET_OPEN_H, MARKET_OPEN_M,
        )

        # 1. Overnight returns
        btc_overnight = _compute_overnight_return(
            cache, "BTCUSDT", prev_close_utc, curr_open_utc
        )
        eth_overnight = _compute_overnight_return(
            cache, "ETHUSDT", prev_close_utc, curr_open_utc
        )

        # 2. Overnight BTC volatility
        btc_overnight_vol = _compute_overnight_volatility(
            cache, prev_close_utc, curr_open_utc
        )

        # 3. BTC 24h and 72h momentum (ending at India open)
        btc_24h_mom = _compute_btc_momentum(cache, curr_open_utc, hours=24)
        btc_72h_mom = _compute_btc_momentum(cache, curr_open_utc, hours=72)

        # 4. Raw base signal
        if all(np.isfinite([btc_overnight, eth_overnight, btc_24h_mom])):
            raw_signal = (
                W_BTC_OVERNIGHT * btc_overnight
                + W_ETH_OVERNIGHT * eth_overnight
                + W_BTC_MOMENTUM * btc_24h_mom
            )
        else:
            raw_signal = np.nan

        # 5. Rolling 20-day crypto-equity correlation
        #    corr(BTC daily return, NIFTY daily return) over last 20 trading days
        crypto_equity_corr = np.nan
        if i >= CORR_LOOKBACK:
            lookback_dates = nifty_trading_dates[i - CORR_LOOKBACK: i]
            nifty_rets_window = []
            btc_rets_window = []
            for d in lookback_dates:
                n_r = nifty_daily_returns.get(d, np.nan)
                b_r = btc_daily_rets.get(d, np.nan)
                if np.isfinite(n_r) and np.isfinite(b_r):
                    nifty_rets_window.append(n_r)
                    btc_rets_window.append(b_r)

            if len(nifty_rets_window) >= 10:
                crypto_equity_corr = float(np.corrcoef(
                    nifty_rets_window, btc_rets_window
                )[0, 1])

        # 6. Scaled signal: base * |corr| (stronger when corr is high)
        if np.isfinite(raw_signal) and np.isfinite(crypto_equity_corr):
            # Scale factor: use abs(corr) so both positive and negative
            # correlations amplify the signal magnitude
            corr_scale = max(abs(crypto_equity_corr), 0.1)  # floor at 0.1
            scaled_signal = raw_signal * corr_scale
        elif np.isfinite(raw_signal):
            scaled_signal = raw_signal * 0.3  # default low scaling
        else:
            scaled_signal = np.nan

        records.append({
            "trade_date": trade_date,
            "btc_overnight_ret": btc_overnight,
            "eth_overnight_ret": eth_overnight,
            "btc_overnight_vol": btc_overnight_vol,
            "btc_24h_momentum": btc_24h_mom,
            "btc_72h_momentum": btc_72h_mom,
            "raw_signal": raw_signal,
            "crypto_equity_corr": crypto_equity_corr,
            "scaled_signal": scaled_signal,
            "z_score": np.nan,  # computed below
        })

    df = pd.DataFrame(records)

    # 7. Z-score the scaled signal over a rolling 20-day window
    df["z_score"] = _rolling_zscore(df["scaled_signal"].values, ZSCORE_WINDOW)

    return df


def _empty_feature_row(trade_date: date) -> dict:
    """Return a row of NaN features for a given date."""
    return {
        "trade_date": trade_date,
        "btc_overnight_ret": np.nan,
        "eth_overnight_ret": np.nan,
        "btc_overnight_vol": np.nan,
        "btc_24h_momentum": np.nan,
        "btc_72h_momentum": np.nan,
        "raw_signal": np.nan,
        "crypto_equity_corr": np.nan,
        "scaled_signal": np.nan,
        "z_score": np.nan,
    }


def _rolling_zscore(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling z-score with expanding start.

    Causal: at index i, uses values[max(0, i-window+1):i+1].
    Returns NaN until at least `window` observations.
    """
    n = len(values)
    result = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - window + 1)
        chunk = values[start: i + 1]
        valid = chunk[np.isfinite(chunk)]
        if len(valid) < window:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma < 1e-12 or not np.isfinite(values[i]):
            continue
        result[i] = (values[i] - mu) / sigma

    return result


# ---------------------------------------------------------------------------
# Intraday execution simulation using 1-min NIFTY futures data
# ---------------------------------------------------------------------------

def _simulate_intraday(
    store,
    trade_date: date,
    direction: str,
    position_frac: float,
    z_score: float,
) -> IntradayTrade | None:
    """Simulate intraday entry/exit using 1-min NIFTY futures bars.

    Entry: 09:16 IST (first reliable bar after open)
    Exit: 15:00 IST or stop loss (whichever comes first)
    Stop loss: 1% adverse move from entry

    Returns None if no 1-min data available for the day.
    """
    # Read the specific date's parquet file directly to avoid schema
    # mismatch across dates (some files lack the timestamp column).
    import duckdb as _ddb

    date_str = trade_date.isoformat()
    parquet_path = (
        Path("/home/ubuntu/Desktop/7hills/QuantLaxmi/data/market/nfo_1min")
        / f"date={date_str}" / "data.parquet"
    )

    if not parquet_path.exists():
        return None

    try:
        con = _ddb.connect()
        df = con.execute(
            f"""SELECT * FROM read_parquet('{parquet_path}')
                WHERE name = 'NIFTY'
                  AND instrument_type = 'FUT'
                ORDER BY expiry ASC""",
        ).fetchdf()
        con.close()
    except Exception as exc:
        logger.debug("Failed to read 1-min data for %s: %s", date_str, exc)
        return None

    if df.empty:
        return None

    # Handle cases where timestamp column may be missing
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "date" in df.columns:
        # Reconstruct timestamp from row index (1-min bars starting 09:15)
        n_rows = len(df)
        base = pd.Timestamp(trade_date.year, trade_date.month, trade_date.day, 9, 15)
        df["timestamp"] = [base + pd.Timedelta(minutes=j) for j in range(n_rows)]
    else:
        return None

    # Use nearest expiry only
    # The data is sorted by expiry ASC, so first rows are nearest
    # We need to identify the nearest expiry from the data
    # Get bar timestamps as naive (they're IST in the store)
    entry_target = pd.Timestamp(
        trade_date.year, trade_date.month, trade_date.day, 9, 16
    )
    exit_target = pd.Timestamp(
        trade_date.year, trade_date.month, trade_date.day, EXIT_H, EXIT_M
    )

    # Filter to trading window
    df = df[(df["timestamp"] >= entry_target) & (df["timestamp"] <= exit_target)]
    if df.empty:
        return None

    entry_price = float(df["open"].iloc[0])
    entry_time = str(df["timestamp"].iloc[0])

    if entry_price <= 0:
        return None

    sign = 1.0 if direction == "long" else -1.0
    stop_price = entry_price * (1.0 - sign * INTRADAY_STOP_PCT)

    exit_price = entry_price
    exit_time = entry_time
    exit_reason = "eod"

    for _, row in df.iterrows():
        bar_high = float(row["high"])
        bar_low = float(row["low"])
        bar_close = float(row["close"])

        # Check stop loss
        if direction == "long" and bar_low <= stop_price:
            exit_price = stop_price
            exit_time = str(row["timestamp"])
            exit_reason = "stop_loss"
            break
        elif direction == "short" and bar_high >= stop_price:
            exit_price = stop_price
            exit_time = str(row["timestamp"])
            exit_reason = "stop_loss"
            break

        # Update exit price and time (will be final if loop ends normally)
        exit_price = bar_close
        exit_time = str(row["timestamp"])

    # PnL computation
    gross_ret = sign * (exit_price / entry_price - 1.0)
    # Cost in fraction (3 points / entry_price)
    cost_frac = COST_POINTS_RT / entry_price
    net_ret = gross_ret - cost_frac

    # Scale by position fraction
    pnl_pct = net_ret * position_frac

    return IntradayTrade(
        trade_date=trade_date,
        direction=direction,
        entry_time=entry_time,
        entry_price=round(entry_price, 2),
        exit_time=exit_time,
        exit_price=round(exit_price, 2),
        position_frac=round(position_frac, 4),
        pnl_pct=round(pnl_pct, 6),
        z_score=round(z_score, 4),
        exit_reason=exit_reason,
    )


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    store,
    start_date: date = date(2025, 8, 6),
    end_date: date = date(2026, 2, 6),
    z_entry: float = ZSCORE_ENTRY,
    max_pos_frac: float = MAX_POSITION_FRAC,
    stop_pct: float = INTRADAY_STOP_PCT,
    verbose: bool = True,
) -> BacktestResult:
    """Run the Crypto Overnight Lead-Lag backtest.

    For each Indian trading day:
    1. Compute overnight crypto features (all causal — data before 09:15 IST)
    2. Generate signal and z-score
    3. If |z| > threshold, enter intraday NIFTY futures trade
    4. Simulate using 1-min bars with stop loss
    """

    # ------------------------------------------------------------------
    # 1. Load NIFTY daily data for trading dates and returns
    # ------------------------------------------------------------------
    nifty_daily = store.sql(
        """SELECT date, CAST("Closing Index Value" AS DOUBLE) as close
           FROM nse_index_close
           WHERE "Index Name" = 'Nifty 50'
             AND date >= ? AND date <= ?
           ORDER BY date""",
        [start_date.isoformat(), end_date.isoformat()],
    )

    if nifty_daily.empty:
        raise ValueError(f"No NIFTY daily data between {start_date} and {end_date}")

    nifty_daily["date"] = pd.to_datetime(nifty_daily["date"]).dt.date
    nifty_daily = nifty_daily.sort_values("date").reset_index(drop=True)

    trading_dates = nifty_daily["date"].tolist()
    n_days = len(trading_dates)

    # Daily log returns
    close_arr = nifty_daily["close"].values.astype(np.float64)
    daily_rets = np.full(n_days, np.nan)
    daily_rets[1:] = np.log(close_arr[1:] / close_arr[:-1])
    nifty_daily_returns = pd.Series(daily_rets, index=trading_dates)

    logger.info(
        "NIFTY daily: %d trading days (%s to %s)",
        n_days, trading_dates[0], trading_dates[-1],
    )

    # ------------------------------------------------------------------
    # 2. Fetch and cache crypto data
    # ------------------------------------------------------------------
    cache = CryptoDataCache(start_date, end_date)
    cache.load()

    # ------------------------------------------------------------------
    # 3. Compute overnight features for each trading day
    # ------------------------------------------------------------------
    if verbose:
        print("\nComputing overnight crypto features ...")

    features_df = compute_all_overnight_features(
        cache, trading_dates, nifty_daily_returns
    )

    if verbose:
        valid_signals = features_df["z_score"].notna().sum()
        print(f"  Computed features for {len(features_df)} trading days")
        print(f"  Valid z-scores: {valid_signals}")
        print(f"  Z-score range: [{features_df['z_score'].min():.2f}, "
              f"{features_df['z_score'].max():.2f}]")

    # ------------------------------------------------------------------
    # 4. Simulate day by day
    # ------------------------------------------------------------------
    records = []
    all_trades: list[IntradayTrade] = []
    cum_ret = 1.0

    for i in range(n_days):
        trade_date = trading_dates[i]
        feat_row = features_df.iloc[i]
        z = feat_row["z_score"]
        nifty_close = float(close_arr[i])

        daily_pnl = 0.0
        signal_name = "flat"
        position = 0.0
        trade_info = None

        if np.isfinite(z):
            # Determine direction
            if z > z_entry:
                direction = "long"
                position_frac = min(abs(z) / Z_SCALING_DIVISOR, max_pos_frac)
                signal_name = "long"
            elif z < -z_entry:
                direction = "short"
                position_frac = min(abs(z) / Z_SCALING_DIVISOR, max_pos_frac)
                signal_name = "short"
            else:
                direction = None
                position_frac = 0.0

            if direction is not None:
                # Simulate intraday using 1-min data
                trade = _simulate_intraday(
                    store, trade_date, direction, position_frac, z,
                )
                if trade is not None:
                    all_trades.append(trade)
                    daily_pnl = trade.pnl_pct
                    position = position_frac if direction == "long" else -position_frac
                    trade_info = trade
                else:
                    signal_name = "signal_no_1min_data"

        cum_ret *= (1.0 + daily_pnl)

        records.append({
            "date": trade_date,
            "nifty_close": round(nifty_close, 2),
            "btc_overnight_ret": _safe_round(feat_row["btc_overnight_ret"], 6),
            "eth_overnight_ret": _safe_round(feat_row["eth_overnight_ret"], 6),
            "btc_overnight_vol": _safe_round(feat_row["btc_overnight_vol"], 6),
            "btc_24h_momentum": _safe_round(feat_row["btc_24h_momentum"], 6),
            "crypto_equity_corr": _safe_round(feat_row["crypto_equity_corr"], 4),
            "raw_signal": _safe_round(feat_row["raw_signal"], 6),
            "scaled_signal": _safe_round(feat_row["scaled_signal"], 6),
            "z_score": _safe_round(z, 4),
            "signal": signal_name,
            "position": round(position, 4),
            "daily_return": round(daily_pnl, 6),
            "cumulative_return": round(cum_ret, 6),
            "entry_price": trade_info.entry_price if trade_info else np.nan,
            "exit_price": trade_info.exit_price if trade_info else np.nan,
            "exit_reason": trade_info.exit_reason if trade_info else "",
        })

    # ------------------------------------------------------------------
    # 5. Compute statistics
    # ------------------------------------------------------------------
    result_df = pd.DataFrame(records)
    stats = _compute_statistics(result_df, all_trades, verbose=verbose)

    return BacktestResult(
        daily_df=result_df,
        trades=all_trades,
        features_df=features_df,
        stats=stats,
    )


def _safe_round(val: float, decimals: int) -> float:
    """Round a value, returning NaN if not finite."""
    if np.isfinite(val):
        return round(val, decimals)
    return np.nan


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_statistics(
    df: pd.DataFrame,
    trades: list[IntradayTrade],
    verbose: bool = True,
) -> dict:
    """Compute and print backtest statistics.

    Sharpe: ddof=1, sqrt(252), all daily returns including flat days.
    """
    rets = df["daily_return"].values

    # Sharpe
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1)
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    # Max drawdown
    cum = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    # Total return
    total_ret = float(cum[-1] - 1.0)

    # Annualized
    n_years = len(rets) / 252.0
    annual_ret = (1.0 + total_ret) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    # Trade statistics
    n_trades = len(trades)
    if n_trades > 0:
        trade_pnls = [t.pnl_pct for t in trades]
        win_rate = sum(1 for p in trade_pnls if p > 0) / n_trades
        avg_trade_pnl = np.mean(trade_pnls)
        best_trade = max(trade_pnls)
        worst_trade = min(trade_pnls)
        gross_profit = sum(p for p in trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in trade_pnls if p <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        long_trades = [t for t in trades if t.direction == "long"]
        short_trades = [t for t in trades if t.direction == "short"]
        stopped_trades = [t for t in trades if t.exit_reason == "stop_loss"]
    else:
        win_rate = avg_trade_pnl = best_trade = worst_trade = 0.0
        profit_factor = 0.0
        long_trades = short_trades = stopped_trades = []

    active_days = int((df["position"] != 0).sum())
    exposure_pct = active_days / len(df) * 100 if len(df) > 0 else 0.0

    calmar = abs(annual_ret / max_dd) if max_dd != 0 else 0.0

    # Correlation between signal and next-day NIFTY return
    z_scores = df["z_score"].values
    nifty_rets = df["daily_return"].values
    valid_mask = np.isfinite(z_scores) & (df["position"].values != 0)
    if valid_mask.sum() > 5:
        signal_hit_rate = float(np.mean(
            np.sign(z_scores[valid_mask]) == np.sign(nifty_rets[valid_mask])
        ))
    else:
        signal_hit_rate = np.nan

    stats = {
        "strategy": "Crypto Overnight Lead-Lag",
        "target": "NIFTY Futures (intraday)",
        "start_date": str(df["date"].iloc[0]),
        "end_date": str(df["date"].iloc[-1]),
        "total_days": len(df),
        "active_days": active_days,
        "exposure_pct": round(exposure_pct, 1),
        "total_return_pct": round(total_ret * 100, 4),
        "annual_return_pct": round(annual_ret * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "calmar_ratio": round(calmar, 3),
        "n_trades": n_trades,
        "n_long": len(long_trades),
        "n_short": len(short_trades),
        "n_stopped": len(stopped_trades),
        "win_rate_pct": round(win_rate * 100, 1),
        "avg_trade_pnl_pct": round(avg_trade_pnl * 100, 4) if n_trades > 0 else 0.0,
        "best_trade_pct": round(best_trade * 100, 4) if n_trades > 0 else 0.0,
        "worst_trade_pct": round(worst_trade * 100, 4) if n_trades > 0 else 0.0,
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "signal_hit_rate": round(signal_hit_rate, 3) if np.isfinite(signal_hit_rate) else "N/A",
        "cost_per_trade_pts": COST_POINTS_RT,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("  Crypto Overnight Lead-Lag Strategy: NIFTY Futures (Intraday)")
        print("=" * 70)
        for k, v in stats.items():
            print(f"  {k:>25s}: {v}")
        print("=" * 70)

        if n_trades > 0:
            print(f"\n  Trade log ({n_trades} trades):")
            print(f"  {'Date':<12s} {'Dir':<6s} {'Entry':>10s} {'Exit':>10s} "
                  f"{'PnL%':>8s} {'|z|':>6s} {'Reason':<10s}")
            print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*10} "
                  f"{'-'*8} {'-'*6} {'-'*10}")
            for tr in trades:
                print(
                    f"  {str(tr.trade_date):<12s} {tr.direction:<6s} "
                    f"{tr.entry_price:>10.2f} {tr.exit_price:>10.2f} "
                    f"{tr.pnl_pct * 100:>+8.4f} {abs(tr.z_score):>6.2f} "
                    f"{tr.exit_reason:<10s}"
                )
        print()

    return stats


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def run_sensitivity(
    store,
    param_name: str = "z_entry",
    param_range: list | None = None,
    **base_kwargs,
) -> pd.DataFrame:
    """Run sensitivity over a single parameter."""
    if param_range is None:
        if param_name == "z_entry":
            param_range = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        elif param_name == "max_pos_frac":
            param_range = [0.10, 0.15, 0.20, 0.25, 0.33, 0.50]
        elif param_name == "stop_pct":
            param_range = [0.005, 0.0075, 0.01, 0.015, 0.02]
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

    results = []
    for val in param_range:
        kwargs = {**base_kwargs, param_name: val, "verbose": False}
        try:
            bt = run_backtest(store=store, **kwargs)
            row = {"param": param_name, "value": val}
            row.update(bt.stats)
            results.append(row)
        except Exception as e:
            logger.warning("Sensitivity run failed for %s=%s: %s", param_name, val, e)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Crypto Overnight Lead-Lag strategy backtest for NIFTY."""
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from data.store import MarketDataStore

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("=" * 70)
    print("  Crypto Overnight Lead-Lag Strategy")
    print("  Target: NIFTY Futures (Intraday)")
    print("  Signal: BTC/ETH overnight returns + momentum + corr scaling")
    print("  Execution: 09:16 entry, 15:00 exit, 1% stop loss")
    print("=" * 70)

    with MarketDataStore() as store:
        # ----------------------------------------------------------
        # 1. Main backtest
        # ----------------------------------------------------------
        result = run_backtest(
            store=store,
            start_date=date(2025, 8, 6),
            end_date=date(2026, 2, 6),
        )

        if result.daily_df.empty:
            logger.error("Backtest returned empty results")
            return

        # ----------------------------------------------------------
        # 2. Save results
        # ----------------------------------------------------------
        out_dir = Path(__file__).resolve().parent.parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        today = date.today().isoformat()

        daily_file = out_dir / f"crypto_leadlag_daily_{today}.csv"
        result.daily_df.to_csv(daily_file, index=False)
        logger.info("Daily results saved to %s", daily_file)

        features_file = out_dir / f"crypto_leadlag_features_{today}.csv"
        result.features_df.to_csv(features_file, index=False)
        logger.info("Features saved to %s", features_file)

        if result.trades:
            trades_data = [
                {
                    "trade_date": t.trade_date,
                    "direction": t.direction,
                    "entry_time": t.entry_time,
                    "entry_price": t.entry_price,
                    "exit_time": t.exit_time,
                    "exit_price": t.exit_price,
                    "position_frac": t.position_frac,
                    "pnl_pct": t.pnl_pct,
                    "z_score": t.z_score,
                    "exit_reason": t.exit_reason,
                }
                for t in result.trades
            ]
            trades_file = out_dir / f"crypto_leadlag_trades_{today}.csv"
            pd.DataFrame(trades_data).to_csv(trades_file, index=False)
            logger.info("Trade log saved to %s", trades_file)

        # ----------------------------------------------------------
        # 3. Sensitivity: z_entry threshold
        # ----------------------------------------------------------
        print("\n" + "=" * 70)
        print("  Sensitivity Analysis: z_entry threshold")
        print("=" * 70)

        sens = run_sensitivity(
            store=store,
            param_name="z_entry",
            param_range=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            start_date=date(2025, 8, 6),
            end_date=date(2026, 2, 6),
        )
        if not sens.empty:
            display_cols = [
                "value", "n_trades", "total_return_pct",
                "sharpe_ratio", "max_drawdown_pct", "win_rate_pct",
                "profit_factor", "signal_hit_rate",
            ]
            avail_cols = [c for c in display_cols if c in sens.columns]
            print(sens[avail_cols].to_string(index=False))

            sens_file = out_dir / f"crypto_leadlag_sensitivity_{today}.csv"
            sens.to_csv(sens_file, index=False)
            logger.info("Sensitivity saved to %s", sens_file)

        # ----------------------------------------------------------
        # 4. Summary
        # ----------------------------------------------------------
        print(f"\nFinal cumulative return: {result.daily_df['cumulative_return'].iloc[-1]:.6f}")
        print(f"Total trades: {len(result.trades)}")

        # Feature summary stats
        print("\nOvernight Feature Statistics:")
        feat_cols = [
            "btc_overnight_ret", "eth_overnight_ret",
            "btc_overnight_vol", "btc_24h_momentum",
            "crypto_equity_corr", "z_score",
        ]
        for col in feat_cols:
            vals = result.features_df[col].dropna()
            if len(vals) > 0:
                print(f"  {col:>25s}: mean={vals.mean():>+8.5f}  "
                      f"std={vals.std():>8.5f}  "
                      f"min={vals.min():>+8.5f}  "
                      f"max={vals.max():>+8.5f}")


if __name__ == "__main__":
    main()
