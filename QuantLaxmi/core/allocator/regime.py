"""VIX regime detection from NSE India VIX data.

Classifies the market into four volatility regimes:
  LOW       — VIX < 13   (complacent)
  NORMAL    — 13 ≤ VIX < 20
  ELEVATED  — 20 ≤ VIX < 30 (stressed)
  EXTREME   — VIX ≥ 30  (crisis — kill switch)

Data source: ``nse_volatility`` table in MarketDataStore, which contains
the India VIX published daily by NSE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum

from core.market.store import MarketDataStore

logger = logging.getLogger(__name__)


class VIXRegimeType(Enum):
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    EXTREME = "extreme"


# Thresholds for India VIX
VIX_LOW = 13.0
VIX_ELEVATED = 20.0
VIX_EXTREME = 30.0


@dataclass(frozen=True)
class VIXRegime:
    """Current VIX regime observation."""

    vix: float
    regime: VIXRegimeType
    date: date

    @staticmethod
    def classify(vix: float) -> VIXRegimeType:
        if vix < VIX_LOW:
            return VIXRegimeType.LOW
        elif vix < VIX_ELEVATED:
            return VIXRegimeType.NORMAL
        elif vix < VIX_EXTREME:
            return VIXRegimeType.ELEVATED
        else:
            return VIXRegimeType.EXTREME


def get_vix(store: MarketDataStore, d: date) -> float | None:
    """Fetch India VIX for a given date from nse_volatility.

    The nse_volatility table has a column for India VIX.  We look for
    the NIFTY row and extract applicable_ann_vol (annualized volatility).
    Falls back to the ``nse_index_close`` table for INDIA VIX index.
    """
    d_str = d.isoformat()

    # Try nse_index_close first (has India VIX as a direct index)
    try:
        df = store.sql(
            'SELECT "Closing Index Value" as close FROM nse_index_close '
            'WHERE date = ? AND "Index Name" = \'India VIX\' '
            "LIMIT 1",
            [d_str],
        )
        if not df.empty:
            return float(df["close"].iloc[0])
    except Exception:
        pass

    # Fallback: nse_volatility (NIFTY's annualized vol ≈ India VIX)
    try:
        df = store.sql(
            "SELECT applicable_ann_vol FROM nse_volatility "
            "WHERE date = ? AND UPPER(symbol) = 'NIFTY' "
            "LIMIT 1",
            [d_str],
        )
        if not df.empty:
            val = float(df["applicable_ann_vol"].iloc[0])
            # Convert from decimal (0.12) to percentage (12) if needed
            return val * 100 if val < 1.0 else val
    except Exception:
        pass

    return None


def detect_regime(store: MarketDataStore, d: date) -> VIXRegime:
    """Detect current VIX regime for a trading date.

    Returns VIXRegime with regime classification.
    Falls back to NORMAL if VIX data unavailable.
    """
    vix = get_vix(store, d)
    if vix is None:
        logger.warning("No VIX data for %s, defaulting to NORMAL", d)
        vix = 15.0  # safe default

    regime = VIXRegime.classify(vix)
    return VIXRegime(vix=vix, regime=regime, date=d)
