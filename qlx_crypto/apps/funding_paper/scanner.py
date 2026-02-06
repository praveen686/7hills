"""Funding rate scanner — polls Binance for current funding data.

Uses the public GET /fapi/v1/premiumIndex endpoint (no API key needed).
Returns annualized funding rates for all USDT perpetual symbols.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FAPI_BASE = "https://fapi.binance.com"

# Binance funds 3x/day (every 8h). Annualize: rate * SETTLEMENTS_PER_YEAR * 100 for %.
SETTLEMENTS_PER_YEAR = 3 * 365  # 1095


def annualize_funding(rate: float) -> float:
    """Convert raw 8h funding rate to annualized percentage."""
    return rate * SETTLEMENTS_PER_YEAR * 100

EXCLUDE = frozenset({
    "XAGUSDT", "XAUUSDT", "PAXGUSDT", "USDCUSDT", "BUSDUSDT",
    "TUSDUSDT", "FDUSDUSDT", "EURUSDT", "GBPUSDT",
})


@dataclass(frozen=True)
class FundingSnapshot:
    """Funding rate data for one symbol at one point in time."""

    symbol: str
    funding_rate: float          # raw 8h rate (e.g. 0.0001)
    ann_funding_pct: float       # annualized as percentage
    mark_price: float
    index_price: float
    next_funding_time_ms: int
    time_to_funding_min: float   # minutes until next settlement
    volume_24h_usd: float = 0.0  # 24h quote volume in USD


def _fetch_volumes() -> dict[str, float]:
    """Fetch 24h quote volumes for all futures symbols."""
    try:
        url = f"{FAPI_BASE}/fapi/v1/ticker/24hr"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return {t["symbol"]: float(t["quoteVolume"]) for t in resp.json()}
    except Exception as e:
        logger.warning("Failed to fetch volumes: %s", e)
        return {}


def scan_funding() -> list[FundingSnapshot]:
    """Fetch current funding rates for all USDT perpetual symbols.

    Returns list of FundingSnapshot sorted by annualized funding descending.
    """
    url = f"{FAPI_BASE}/fapi/v1/premiumIndex"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

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
            index = float(item.get("indexPrice", 0))
            next_ts = int(item.get("nextFundingTime", 0))
            time_to = max(0, (next_ts - now_ms) / 60_000)  # minutes

            ann_pct = annualize_funding(rate)

            snapshots.append(FundingSnapshot(
                symbol=sym,
                funding_rate=rate,
                ann_funding_pct=ann_pct,
                mark_price=mark,
                index_price=index,
                next_funding_time_ms=next_ts,
                time_to_funding_min=time_to,
                volume_24h_usd=volumes.get(sym, 0.0),
            ))
        except (ValueError, KeyError, TypeError) as e:
            logger.debug("Skip %s: %s", sym, e)

    snapshots.sort(key=lambda s: s.ann_funding_pct, reverse=True)
    return snapshots


def scan_top_opportunities(
    min_ann_pct: float = 15.0,
    min_volume_usd: float = 50_000_000,
) -> list[FundingSnapshot]:
    """Scan for actionable funding opportunities.

    Filters to symbols with annualized funding > threshold and sufficient volume.
    """
    return [
        s for s in scan_funding()
        if s.ann_funding_pct >= min_ann_pct and s.volume_24h_usd >= min_volume_usd
    ]
