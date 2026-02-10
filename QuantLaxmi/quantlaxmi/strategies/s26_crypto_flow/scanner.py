"""Data scanner for CLRS -- collects funding, volumes, and klines.

Uses Binance public REST endpoints (no API key needed).
Maintains a rolling funding matrix for PCA decomposition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

FAPI_BASE = "https://fapi.binance.com"
SAPI_BASE = "https://api.binance.com"
SETTLEMENTS_PER_YEAR = 3 * 365  # 1095 (8h funding)

EXCLUDE = frozenset({
    "XAGUSDT", "XAUUSDT", "PAXGUSDT", "USDCUSDT", "BUSDUSDT",
    "TUSDUSDT", "FDUSDUSDT", "EURUSDT", "GBPUSDT",
})


def annualize_funding(rate: float) -> float:
    """Convert raw 8h funding rate to annualized percentage."""
    return rate * SETTLEMENTS_PER_YEAR * 100


@dataclass(frozen=True)
class SymbolSnapshot:
    """Live snapshot for one perpetual symbol."""

    symbol: str
    funding_rate: float          # raw 8h rate
    ann_funding_pct: float       # annualized %
    mark_price: float
    index_price: float
    next_funding_time_ms: int
    time_to_funding_min: float
    volume_24h_usd: float = 0.0
    open_interest_usd: float = 0.0


def _fetch_premium_index() -> list[dict]:
    resp = requests.get(f"{FAPI_BASE}/fapi/v1/premiumIndex", timeout=15)
    resp.raise_for_status()
    return resp.json()


def _fetch_volumes() -> dict[str, float]:
    try:
        resp = requests.get(f"{FAPI_BASE}/fapi/v1/ticker/24hr", timeout=15)
        resp.raise_for_status()
        return {t["symbol"]: float(t["quoteVolume"]) for t in resp.json()}
    except Exception as e:
        logger.warning("Failed to fetch volumes: %s", e)
        return {}


def _fetch_open_interest() -> dict[str, float]:
    """Fetch open interest for all symbols."""
    try:
        resp = requests.get(f"{FAPI_BASE}/fapi/v1/openInterest", timeout=15)
        resp.raise_for_status()
        # This endpoint is per-symbol; use ticker for all at once
        return {}
    except Exception:
        return {}


def _fetch_oi_batch() -> dict[str, float]:
    """Fetch open interest via ticker endpoint (all symbols)."""
    try:
        resp = requests.get(f"{FAPI_BASE}/fapi/v1/ticker/24hr", timeout=15)
        resp.raise_for_status()
        result = {}
        for t in resp.json():
            sym = t.get("symbol", "")
            # quoteVolume as proxy; actual OI needs per-symbol call
            if "openInterest" in t:
                result[sym] = float(t["openInterest"]) * float(t.get("lastPrice", 0))
        return result
    except Exception:
        return {}


def scan_all_symbols(
    min_volume_usd: float = 10_000_000,
) -> list[SymbolSnapshot]:
    """Scan all USDT perpetual symbols with funding data.

    Returns list sorted by annualized funding descending.
    """
    data = _fetch_premium_index()
    volumes = _fetch_volumes()
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    snapshots = []
    for item in data:
        sym = item["symbol"]
        if not sym.endswith("USDT") or sym in EXCLUDE:
            continue

        try:
            rate = float(item.get("lastFundingRate", 0))
            mark = float(item.get("markPrice", 0))
            index_p = float(item.get("indexPrice", 0))
            next_ts = int(item.get("nextFundingTime", 0))
            time_to = max(0, (next_ts - now_ms) / 60_000)
            vol = volumes.get(sym, 0.0)

            if vol < min_volume_usd:
                continue

            snapshots.append(SymbolSnapshot(
                symbol=sym,
                funding_rate=rate,
                ann_funding_pct=annualize_funding(rate),
                mark_price=mark,
                index_price=index_p,
                next_funding_time_ms=next_ts,
                time_to_funding_min=time_to,
                volume_24h_usd=vol,
            ))
        except (ValueError, KeyError, TypeError) as e:
            logger.debug("Skip %s: %s", sym, e)

    snapshots.sort(key=lambda s: s.ann_funding_pct, reverse=True)
    return snapshots


def fetch_recent_klines(
    symbol: str,
    interval: str = "5m",
    limit: int = 200,
) -> pd.DataFrame:
    """Fetch recent klines for VPIN / Kyle's Lambda computation.

    Returns DataFrame with columns: Open, High, Low, Close, Volume, QuoteVolume.
    """
    url = f"{FAPI_BASE}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    rows = resp.json()

    df = pd.DataFrame(rows, columns=[
        "Open_time", "Open", "High", "Low", "Close", "Volume",
        "Close_time", "QuoteVolume", "Trade_count",
        "Taker_buy_base", "Taker_buy_quote", "Ignore",
    ])
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms", utc=True)
    for col in ("Open", "High", "Low", "Close", "Volume", "QuoteVolume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("Open_time").sort_index()

    return df[["Open", "High", "Low", "Close", "Volume", "QuoteVolume"]]


def fetch_recent_trades(
    symbol: str,
    limit: int = 1000,
) -> np.ndarray:
    """Fetch recent aggregate trades for Hawkes calibration.

    Returns array of timestamps in seconds (float64).
    """
    url = f"{FAPI_BASE}/fapi/v1/aggTrades"
    params = {"symbol": symbol, "limit": limit}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()

    timestamps = [float(t["T"]) / 1000.0 for t in resp.json()]
    return np.array(timestamps, dtype=np.float64)


# ---------------------------------------------------------------------------
# Funding matrix builder (for PCA)
# ---------------------------------------------------------------------------

@dataclass
class FundingMatrixBuilder:
    """Accumulates funding rate observations into a matrix for PCA.

    Each row = one observation time, each column = one symbol.
    Maintains a rolling window of the last N observations.
    """

    max_rows: int = 90  # ~30 days of 3x/day funding

    _data: dict[str, list[float]] = field(default_factory=dict)
    _timestamps: list[str] = field(default_factory=list)

    def add_observation(self, snapshots: list[SymbolSnapshot]) -> None:
        """Record one funding snapshot across all symbols."""
        now = datetime.now(timezone.utc).isoformat()
        self._timestamps.append(now)

        symbols_seen = set()
        for snap in snapshots:
            if snap.symbol not in self._data:
                self._data[snap.symbol] = [np.nan] * (len(self._timestamps) - 1)
            self._data[snap.symbol].append(snap.funding_rate)
            symbols_seen.add(snap.symbol)

        # Pad missing symbols with NaN
        for sym in self._data:
            if sym not in symbols_seen:
                self._data[sym].append(np.nan)

        # Trim to window
        if len(self._timestamps) > self.max_rows:
            excess = len(self._timestamps) - self.max_rows
            self._timestamps = self._timestamps[excess:]
            for sym in self._data:
                self._data[sym] = self._data[sym][excess:]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for PCA computation."""
        if not self._timestamps:
            return pd.DataFrame()

        df = pd.DataFrame(self._data, index=self._timestamps)
        # Drop symbols with too many NaNs (>50% missing)
        threshold = len(df) * 0.5
        df = df.dropna(axis=1, thresh=int(threshold))
        return df

    def to_dict(self) -> dict:
        """Serialize for JSON state persistence."""
        return {
            "timestamps": self._timestamps,
            "data": self._data,
        }

    @classmethod
    def from_dict(cls, d: dict, max_rows: int = 90) -> FundingMatrixBuilder:
        """Deserialize from JSON state."""
        builder = cls(max_rows=max_rows)
        builder._timestamps = d.get("timestamps", [])
        builder._data = d.get("data", {})
        return builder
