"""Binance public market data connector — REST klines, trades, and order book.

Provides:

1. **Klines** — :meth:`BinanceConnector.fetch_klines` downloads OHLCV candles
2. **Chunked Klines** — :meth:`BinanceConnector.fetch_klines_chunked` handles pagination
3. **Trades** — :meth:`BinanceConnector.fetch_recent_trades` returns tick-level trades
4. **Order Book** — :meth:`BinanceConnector.fetch_order_book` returns L2 depth

All public endpoints — no API keys required for market data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

import pandas as pd
from binance.client import Client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Map short interval strings to Binance Client constants
INTERVALS: dict[str, str] = {
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "3m": Client.KLINE_INTERVAL_3MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "2h": Client.KLINE_INTERVAL_2HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "6h": Client.KLINE_INTERVAL_6HOUR,
    "8h": Client.KLINE_INTERVAL_8HOUR,
    "12h": Client.KLINE_INTERVAL_12HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY,
    "3d": Client.KLINE_INTERVAL_3DAY,
    "1w": Client.KLINE_INTERVAL_1WEEK,
    "1M": Client.KLINE_INTERVAL_1MONTH,
}

# Default symbols for crypto FnO
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Binance kline API returns max 1000 candles per request
MAX_KLINES_PER_REQUEST = 1000

# IST offset from UTC
IST_OFFSET = timedelta(hours=5, minutes=30)
IST = timezone(IST_OFFSET)

# Interval durations in minutes (for chunking calculations)
INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
    "1M": 43200,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms_to_utc(ms: int) -> pd.Timestamp:
    """Convert millisecond epoch to UTC Timestamp."""
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def utc_to_ist(ts: pd.Timestamp | datetime) -> pd.Timestamp:
    """Convert a UTC timestamp to IST (Asia/Kolkata, UTC+05:30)."""
    if isinstance(ts, datetime) and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return pd.Timestamp(ts).tz_convert(IST)


def ist_to_utc(ts: pd.Timestamp | datetime) -> pd.Timestamp:
    """Convert an IST timestamp to UTC."""
    if isinstance(ts, datetime) and ts.tzinfo is None:
        ts = ts.replace(tzinfo=IST)
    return pd.Timestamp(ts).tz_convert(timezone.utc)


def _parse_kline_row(row: list) -> dict:
    """Parse a single raw Binance kline row into a dict.

    Binance kline format (12 elements):
    [open_time, open, high, low, close, volume, close_time,
     quote_asset_volume, number_of_trades, taker_buy_base,
     taker_buy_quote, ignore]
    """
    return {
        "timestamp": _ms_to_utc(int(row[0])),
        "open": float(row[1]),
        "high": float(row[2]),
        "low": float(row[3]),
        "close": float(row[4]),
        "volume": float(row[5]),
        "close_time": _ms_to_utc(int(row[6])),
        "quote_volume": float(row[7]),
        "trades": int(row[8]),
        "taker_buy_volume": float(row[9]),
        "taker_buy_quote_volume": float(row[10]),
    }


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class BinanceConnector:
    """Binance public market data connector (REST, no auth required).

    Fetches klines (OHLCV candles), recent trades, and order book
    depth via the Binance REST API.

    Usage::

        bc = BinanceConnector()
        df = bc.fetch_klines("BTCUSDT", "1h", days=7)
        print(df.head())

    Parameters
    ----------
    api_key : str, optional
        Binance API key. Not required for public endpoints.
    api_secret : str, optional
        Binance API secret. Not required for public endpoints.
    requests_params : dict, optional
        Extra params passed to the underlying requests session
        (e.g. proxies, timeout).
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        requests_params: dict | None = None,
    ):
        self._client = Client(
            api_key=api_key,
            api_secret=api_secret,
            requests_params=requests_params or {},
        )
        logger.info("BinanceConnector initialized (public endpoints)")

    # ----- Klines (OHLCV) -----

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
        days: int | None = None,
        limit: int = MAX_KLINES_PER_REQUEST,
    ) -> pd.DataFrame:
        """Fetch klines (OHLCV candles) for a symbol.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTCUSDT"``.
        interval : str
            Candle interval: ``1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h,
            6h, 8h, 12h, 1d, 3d, 1w, 1M``.
        start_date : str or datetime, optional
            Start of range (inclusive). Parsed via ``pd.Timestamp``.
        end_date : str or datetime, optional
            End of range (inclusive). Defaults to now.
        days : int, optional
            Convenience shorthand — fetch the last *N* days.
            Overrides ``start_date`` if both provided.
        limit : int
            Max candles per request (Binance caps at 1000).

        Returns
        -------
        pd.DataFrame
            Columns: open, high, low, close, volume, quote_volume,
            trades, taker_buy_volume, taker_buy_quote_volume.
            Index: ``timestamp`` (UTC DatetimeIndex).
        """
        binance_interval = INTERVALS.get(interval)
        if binance_interval is None:
            raise ValueError(
                f"Unsupported interval '{interval}'. "
                f"Use one of: {list(INTERVALS)}"
            )

        now = datetime.now(timezone.utc)

        if days is not None:
            start_dt = now - timedelta(days=days)
            end_dt = now
        else:
            end_dt = pd.Timestamp(end_date).to_pydatetime() if end_date else now
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            if start_date:
                start_dt = pd.Timestamp(start_date).to_pydatetime()
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
            else:
                start_dt = end_dt - timedelta(days=1)

        start_ms = str(int(start_dt.timestamp() * 1000))
        end_ms = str(int(end_dt.timestamp() * 1000))

        raw = self._client.get_klines(
            symbol=symbol,
            interval=binance_interval,
            startTime=start_ms,
            endTime=end_ms,
            limit=limit,
        )

        if not raw:
            raise ValueError(
                f"No kline data for {symbol} {interval} "
                f"({start_dt:%Y-%m-%d %H:%M} to {end_dt:%Y-%m-%d %H:%M})"
            )

        records = [_parse_kline_row(row) for row in raw]
        df = pd.DataFrame(records)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # Drop close_time from output (internal use only)
        df = df.drop(columns=["close_time"], errors="ignore")

        # Ensure float64 for OHLCV columns
        float_cols = [
            "open", "high", "low", "close", "volume",
            "quote_volume", "taker_buy_volume", "taker_buy_quote_volume",
        ]
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].astype("float64")
        df["trades"] = df["trades"].astype("int64")

        logger.info(
            "Fetched %d klines for %s %s (%s to %s)",
            len(df), symbol, interval,
            df.index[0].strftime("%Y-%m-%d %H:%M"),
            df.index[-1].strftime("%Y-%m-%d %H:%M"),
        )

        return df

    def fetch_klines_chunked(
        self,
        symbol: str,
        interval: str = "1m",
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
        days: int | None = None,
        rate_limit_sleep: float = 0.1,
    ) -> pd.DataFrame:
        """Fetch klines with automatic pagination for large date ranges.

        Binance limits each request to 1000 candles.  This method
        splits the range into chunks, fetches each with a small
        rate-limit pause, and concatenates the results.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTCUSDT"``.
        interval : str
            Candle interval (see :meth:`fetch_klines`).
        start_date : str or datetime, optional
            Start of range.
        end_date : str or datetime, optional
            End of range.  Defaults to now.
        days : int, optional
            Convenience — fetch the last *N* days.
        rate_limit_sleep : float
            Seconds to sleep between paginated requests.

        Returns
        -------
        pd.DataFrame
            Same schema as :meth:`fetch_klines`.
        """
        binance_interval = INTERVALS.get(interval)
        if binance_interval is None:
            raise ValueError(
                f"Unsupported interval '{interval}'. "
                f"Use one of: {list(INTERVALS)}"
            )

        now = datetime.now(timezone.utc)

        if days is not None:
            start_dt = now - timedelta(days=days)
            end_dt = now
        else:
            end_dt = pd.Timestamp(end_date).to_pydatetime() if end_date else now
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            if start_date:
                start_dt = pd.Timestamp(start_date).to_pydatetime()
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
            else:
                start_dt = end_dt - timedelta(days=1)

        interval_mins = INTERVAL_MINUTES.get(interval, 60)
        chunk_duration = timedelta(minutes=interval_mins * MAX_KLINES_PER_REQUEST)

        all_dfs: list[pd.DataFrame] = []
        cursor = start_dt
        chunk_num = 0

        total_chunks = max(
            1, int((end_dt - start_dt) / chunk_duration) + 1
        )
        logger.info(
            "Fetching %s %s chunked: %s to %s (~%d chunks)",
            symbol, interval,
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
            total_chunks,
        )

        while cursor < end_dt:
            chunk_end = min(cursor + chunk_duration, end_dt)
            try:
                df_chunk = self.fetch_klines(
                    symbol=symbol,
                    interval=interval,
                    start_date=cursor,
                    end_date=chunk_end,
                    limit=MAX_KLINES_PER_REQUEST,
                )
                all_dfs.append(df_chunk)
                chunk_num += 1
                if chunk_num % 10 == 0:
                    logger.info(
                        "  Progress: %d/%d chunks (%d rows so far)",
                        chunk_num, total_chunks,
                        sum(len(d) for d in all_dfs),
                    )
            except ValueError:
                # No data for this chunk (e.g. symbol didn't exist yet)
                logger.debug(
                    "No data for %s %s chunk %s to %s",
                    symbol, interval,
                    cursor.strftime("%Y-%m-%d %H:%M"),
                    chunk_end.strftime("%Y-%m-%d %H:%M"),
                )
            except Exception as exc:
                logger.warning(
                    "Chunk %s to %s failed: %s",
                    cursor.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                    exc,
                )

            cursor = chunk_end
            if rate_limit_sleep > 0:
                time.sleep(rate_limit_sleep)

        if not all_dfs:
            raise ValueError(
                f"No kline data for {symbol} {interval} "
                f"({start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d})"
            )

        df = pd.concat(all_dfs)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        logger.info(
            "Fetched %d total klines for %s %s",
            len(df), symbol, interval,
        )

        return df

    # ----- Recent Trades (tick-level) -----

    def fetch_recent_trades(
        self,
        symbol: str,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch recent trades (tick-level) for a symbol.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTCUSDT"``.
        limit : int
            Number of trades to fetch (max 1000).

        Returns
        -------
        pd.DataFrame
            Columns: trade_id, price, qty, quote_qty, timestamp,
            is_buyer_maker, is_best_match.
            Index: integer.
        """
        raw = self._client.get_recent_trades(symbol=symbol, limit=limit)

        if not raw:
            raise ValueError(f"No recent trades for {symbol}")

        records = []
        for t in raw:
            records.append({
                "trade_id": int(t["id"]),
                "price": float(t["price"]),
                "qty": float(t["qty"]),
                "quote_qty": float(t["quoteQty"]),
                "timestamp": _ms_to_utc(int(t["time"])),
                "is_buyer_maker": bool(t["isBuyerMaker"]),
                "is_best_match": bool(t["isBestMatch"]),
            })

        df = pd.DataFrame(records)
        df["price"] = df["price"].astype("float64")
        df["qty"] = df["qty"].astype("float64")
        df["quote_qty"] = df["quote_qty"].astype("float64")

        logger.info("Fetched %d recent trades for %s", len(df), symbol)
        return df

    # ----- Order Book (L2 Depth) -----

    def fetch_order_book(
        self,
        symbol: str,
        limit: int = 20,
    ) -> dict:
        """Fetch current L2 order book depth.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTCUSDT"``.
        limit : int
            Number of price levels per side.  Valid values:
            5, 10, 20, 50, 100, 500, 1000, 5000.

        Returns
        -------
        dict
            Keys: ``bids`` (pd.DataFrame), ``asks`` (pd.DataFrame),
            ``last_update_id`` (int).
            Each side has columns: ``price``, ``qty``.
        """
        raw = self._client.get_order_book(symbol=symbol, limit=limit)

        def _parse_side(side_data: list[list]) -> pd.DataFrame:
            df = pd.DataFrame(side_data, columns=["price", "qty"])
            df["price"] = df["price"].astype("float64")
            df["qty"] = df["qty"].astype("float64")
            return df

        result = {
            "bids": _parse_side(raw.get("bids", [])),
            "asks": _parse_side(raw.get("asks", [])),
            "last_update_id": raw.get("lastUpdateId", 0),
        }

        logger.info(
            "Fetched order book for %s: %d bids, %d asks (update_id=%d)",
            symbol,
            len(result["bids"]),
            len(result["asks"]),
            result["last_update_id"],
        )
        return result

    # ----- Futures API (FAPI) -----

    FAPI_BASE = "https://fapi.binance.com"

    def fetch_funding_rate_history(
        self,
        symbol: str,
        start_time: datetime | str | None = None,
        end_time: datetime | str | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch historical funding rates from Binance Futures.

        Parameters
        ----------
        symbol : str
            Perpetual pair, e.g. ``"BTCUSDT"``.
        start_time : datetime or str, optional
            Start of range.
        end_time : datetime or str, optional
            End of range.
        limit : int
            Max records (Binance caps at 1000).

        Returns
        -------
        pd.DataFrame
            Columns: symbol, fundingRate, fundingTime, markPrice.
            Index: ``fundingTime`` (UTC DatetimeIndex).
        """
        import requests

        params: dict = {"symbol": symbol, "limit": limit}
        if start_time is not None:
            ts = pd.Timestamp(start_time)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            params["startTime"] = int(ts.timestamp() * 1000)
        if end_time is not None:
            ts = pd.Timestamp(end_time)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            params["endTime"] = int(ts.timestamp() * 1000)

        resp = requests.get(
            f"{self.FAPI_BASE}/fapi/v1/fundingRate",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return pd.DataFrame()

        records = []
        for d in data:
            records.append({
                "symbol": d.get("symbol", symbol),
                "fundingRate": float(d["fundingRate"]),
                "fundingTime": _ms_to_utc(int(d["fundingTime"])),
                "markPrice": float(d.get("markPrice", 0)),
            })

        df = pd.DataFrame(records)
        df = df.set_index("fundingTime").sort_index()
        logger.info("Fetched %d funding rate records for %s", len(df), symbol)
        return df

    def fetch_open_interest(self, symbol: str) -> dict:
        """Fetch current open interest for a futures symbol.

        Parameters
        ----------
        symbol : str
            Perpetual pair, e.g. ``"BTCUSDT"``.

        Returns
        -------
        dict
            Keys: ``symbol``, ``openInterest`` (float), ``time`` (pd.Timestamp).
        """
        import requests

        resp = requests.get(
            f"{self.FAPI_BASE}/fapi/v1/openInterest",
            params={"symbol": symbol},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        return {
            "symbol": data.get("symbol", symbol),
            "openInterest": float(data["openInterest"]),
            "time": _ms_to_utc(int(data["time"])),
        }

    def fetch_long_short_ratio(
        self,
        symbol: str,
        period: str = "1d",
        limit: int = 30,
    ) -> pd.DataFrame:
        """Fetch global long/short account ratio.

        Parameters
        ----------
        symbol : str
            Perpetual pair, e.g. ``"BTCUSDT"``.
        period : str
            Period: ``"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"``.
        limit : int
            Number of records (max 500).

        Returns
        -------
        pd.DataFrame
            Columns: longShortRatio, longAccount, shortAccount.
            Index: ``timestamp`` (UTC DatetimeIndex).
        """
        import requests

        resp = requests.get(
            f"{self.FAPI_BASE}/futures/data/globalLongShortAccountRatio",
            params={"symbol": symbol, "period": period, "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return pd.DataFrame()

        records = []
        for d in data:
            records.append({
                "timestamp": _ms_to_utc(int(d["timestamp"])),
                "longShortRatio": float(d["longShortRatio"]),
                "longAccount": float(d["longAccount"]),
                "shortAccount": float(d["shortAccount"]),
            })

        df = pd.DataFrame(records).set_index("timestamp").sort_index()
        logger.info("Fetched %d L/S ratio records for %s", len(df), symbol)
        return df

    def fetch_top_long_short_ratio(
        self,
        symbol: str,
        period: str = "1d",
        limit: int = 30,
    ) -> pd.DataFrame:
        """Fetch top-trader long/short position ratio.

        Parameters
        ----------
        symbol : str
            Perpetual pair, e.g. ``"BTCUSDT"``.
        period : str
            Period: ``"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"``.
        limit : int
            Number of records (max 500).

        Returns
        -------
        pd.DataFrame
            Columns: longShortRatio, longAccount, shortAccount.
            Index: ``timestamp`` (UTC DatetimeIndex).
        """
        import requests

        resp = requests.get(
            f"{self.FAPI_BASE}/futures/data/topLongShortPositionRatio",
            params={"symbol": symbol, "period": period, "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return pd.DataFrame()

        records = []
        for d in data:
            records.append({
                "timestamp": _ms_to_utc(int(d["timestamp"])),
                "longShortRatio": float(d["longShortRatio"]),
                "longAccount": float(d["longAccount"]),
                "shortAccount": float(d["shortAccount"]),
            })

        df = pd.DataFrame(records).set_index("timestamp").sort_index()
        logger.info("Fetched %d top L/S ratio records for %s", len(df), symbol)
        return df

    def fetch_taker_long_short_ratio(
        self,
        symbol: str,
        period: str = "1d",
        limit: int = 30,
    ) -> pd.DataFrame:
        """Fetch taker buy/sell volume ratio.

        Parameters
        ----------
        symbol : str
            Perpetual pair, e.g. ``"BTCUSDT"``.
        period : str
            Period: ``"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"``.
        limit : int
            Number of records (max 500).

        Returns
        -------
        pd.DataFrame
            Columns: buySellRatio, buyVol, sellVol.
            Index: ``timestamp`` (UTC DatetimeIndex).
        """
        import requests

        resp = requests.get(
            f"{self.FAPI_BASE}/futures/data/takerlongshortRatio",
            params={"symbol": symbol, "period": period, "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return pd.DataFrame()

        records = []
        for d in data:
            records.append({
                "timestamp": _ms_to_utc(int(d["timestamp"])),
                "buySellRatio": float(d["buySellRatio"]),
                "buyVol": float(d["buyVol"]),
                "sellVol": float(d["sellVol"]),
            })

        df = pd.DataFrame(records).set_index("timestamp").sort_index()
        logger.info("Fetched %d taker L/S ratio records for %s", len(df), symbol)
        return df

    # ----- Utility -----

    def get_server_time(self) -> pd.Timestamp:
        """Get Binance server time as a UTC Timestamp."""
        server_time = self._client.get_server_time()
        return _ms_to_utc(server_time["serverTime"])

    def get_symbol_info(self, symbol: str) -> dict | None:
        """Get exchange info for a single symbol."""
        return self._client.get_symbol_info(symbol)
