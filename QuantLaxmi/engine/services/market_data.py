"""Market data service — wraps MarketDataStore for the API layer.

Provides async-compatible methods that the FastAPI routes call.  All
heavy DuckDB queries are delegated to ``MarketDataStore``; this service
adds caching, error handling, and response shaping.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from data.store import MarketDataStore

logger = logging.getLogger(__name__)


class MarketDataService:
    """API-layer wrapper around MarketDataStore."""

    def __init__(self, store: MarketDataStore) -> None:
        self._store = store

    @property
    def store(self) -> MarketDataStore:
        return self._store

    # ------------------------------------------------------------------
    # Option chain
    # ------------------------------------------------------------------

    def get_option_chain(
        self,
        symbol: str,
        d: date | None = None,
        expiry: str | None = None,
    ) -> dict[str, Any]:
        """Return the option chain for *symbol* as a JSON-serialisable dict.

        Parameters
        ----------
        symbol : str
            Underlying name (e.g. "NIFTY", "BANKNIFTY").
        d : date, optional
            Trading date.  Defaults to the latest available date.
        expiry : str, optional
            Expiry date as "YYYY-MM-DD".  If None, uses nearest expiry.
        """
        if d is None:
            dates = self._store.available_dates("nfo_1min")
            if not dates:
                return {"symbol": symbol, "date": None, "expiry": None, "chain": []}
            d = dates[-1]

        df = self._store.get_option_chain(symbol, d, expiry=expiry)
        if df.empty:
            return {"symbol": symbol, "date": d.isoformat(), "expiry": expiry, "chain": []}

        # Determine actual expiry from the data
        actual_expiry = expiry
        if "expiry" in df.columns and not df.empty:
            actual_expiry = str(df["expiry"].iloc[0])

        records = df.to_dict(orient="records")
        # Ensure all values are JSON-serialisable
        for rec in records:
            for k, v in rec.items():
                if hasattr(v, "isoformat"):
                    rec[k] = v.isoformat()
                elif hasattr(v, "item"):
                    rec[k] = v.item()

        return {
            "symbol": symbol,
            "date": d.isoformat(),
            "expiry": actual_expiry,
            "n_strikes": len(records) // 2 if records else 0,
            "chain": records,
        }

    # ------------------------------------------------------------------
    # Volatility surface (strikes x expiries)
    # ------------------------------------------------------------------

    def get_vol_surface(
        self,
        symbol: str,
        d: date | None = None,
    ) -> dict[str, Any]:
        """Return a volatility surface grid for *symbol*.

        The surface is a simplified representation built from the last
        1-minute bar IV for each (strike, expiry, instrument_type) tuple
        available on *d*.

        Because the MarketDataStore may not carry pre-computed IV, this
        method returns the close prices for all options which the frontend
        can use to compute implied vol via Black-Scholes client-side, or
        a future enhancement can add server-side IV computation.
        """
        if d is None:
            dates = self._store.available_dates("nfo_1min")
            if not dates:
                return {"symbol": symbol, "date": None, "surface": []}
            d = dates[-1]

        bfo_names = {"SENSEX", "BANKEX", "SENSEX50"}
        table = "bfo_1min" if symbol.upper() in bfo_names else "nfo_1min"
        d_str = d.isoformat()

        try:
            df = self._store.sql(
                f"SELECT DISTINCT expiry, strike, instrument_type, "
                f"LAST(close) AS last_close, LAST(oi) AS last_oi "
                f"FROM {table} "
                f"WHERE date={d_str!r} AND name={symbol!r} "
                f"AND instrument_type IN ('CE', 'PE') "
                f"GROUP BY expiry, strike, instrument_type "
                f"ORDER BY expiry, strike, instrument_type"
            )
        except Exception as exc:
            logger.warning("Vol surface query failed for %s on %s: %s", symbol, d, exc)
            return {"symbol": symbol, "date": d.isoformat(), "surface": []}

        if df.empty:
            return {"symbol": symbol, "date": d.isoformat(), "surface": []}

        records = df.to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if hasattr(v, "isoformat"):
                    rec[k] = v.isoformat()
                elif hasattr(v, "item"):
                    rec[k] = v.item()

        # Extract unique expiries and strikes for metadata
        expiries = sorted(df["expiry"].unique().tolist())
        strikes = sorted(df["strike"].unique().tolist())

        return {
            "symbol": symbol,
            "date": d.isoformat(),
            "expiries": [str(e) for e in expiries],
            "strikes": [float(s) for s in strikes],
            "n_points": len(records),
            "surface": records,
        }

    # ------------------------------------------------------------------
    # Store summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return data store summary."""
        return self._store.summary()

    def close(self) -> None:
        """Close underlying DuckDB connection."""
        self._store.close()
