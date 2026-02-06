"""NSE data access via MarketDataStore (DuckDB).

Query helpers for scanner signals, backtest, and strategy modules.
FNO data uses raw NSE column names (TckrSymb, FinInstrmTp, ClsPric, etc.).
Delivery data has a thin rename layer for convenience.

All data comes from the nse_daily collector pipeline:
    nse_daily collect → raw files → auto-ingest → DuckDB parquet views
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)


# NSE holidays (Saturday/Sunday handled separately) — major ones.
# Not exhaustive; missing data on holidays is handled gracefully.
_KNOWN_HOLIDAYS_2025_2026 = {
    date(2025, 1, 26), date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31),
    date(2025, 4, 10), date(2025, 4, 14), date(2025, 4, 18), date(2025, 5, 1),
    date(2025, 8, 15), date(2025, 8, 27), date(2025, 10, 2), date(2025, 10, 20),
    date(2025, 10, 21), date(2025, 10, 22), date(2025, 11, 5), date(2025, 11, 26),
    date(2025, 12, 25),
    date(2026, 1, 26), date(2026, 3, 17), date(2026, 3, 30), date(2026, 4, 3),
    date(2026, 4, 14), date(2026, 5, 1), date(2026, 8, 15), date(2026, 10, 2),
    date(2026, 10, 20), date(2026, 11, 25), date(2026, 12, 25),
}


def is_trading_day(d: date) -> bool:
    """Check if a date is likely an NSE trading day."""
    if d.weekday() >= 5:
        return False
    if d in _KNOWN_HOLIDAYS_2025_2026:
        return False
    return True


_DELIVERY_RENAME = {
    "OPEN_PRICE": "OPEN",
    "HIGH_PRICE": "HIGH",
    "LOW_PRICE": "LOW",
    "CLOSE_PRICE": "CLOSE",
    "TTL_TRD_QNTY": "TOTTRDQTY",
    "TURNOVER_LACS": "TOTTRDVAL",
    "DELIV_PER": "DELIVERY_PCT",
    "DELIV_QTY": "DELIVERY_QTY",
}

# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_delivery(store, d: date) -> pd.DataFrame:
    """Get delivery data (equity OHLCV + delivery %) for a date."""
    df = store.sql("SELECT * FROM nse_delivery WHERE date = ?", [d.isoformat()])
    if df is None or df.empty:
        return pd.DataFrame()
    return df.rename(columns=_DELIVERY_RENAME)


def get_equity(store, d: date) -> pd.DataFrame:
    """Get equity data (alias for get_delivery — same source file)."""
    return get_delivery(store, d)


def get_fno(store, d: date) -> pd.DataFrame:
    """Get F&O bhavcopy data with raw NSE column names.

    Returns columns as-is from NSE fo_bhavcopy: TckrSymb, FinInstrmTp,
    XpryDt, StrkPric, OptnTp, ClsPric, UndrlygPric, OpnIntrst, etc.
    FinInstrmTp values: STO (stock option), STF (stock future),
    IDO (index option), IDF (index future).
    """
    df = store.sql("SELECT * FROM nse_fo_bhavcopy WHERE date = ?", [d.isoformat()])
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def get_fii_dii(store, d: date) -> pd.DataFrame:
    """Get FII stats, transformed to the format signals.py expects.

    The DuckDB view has per-instrument rows (INDEX FUTURES, STOCK FUTURES, etc.)
    with buy_amt_cr / sell_amt_cr.  We aggregate into the FII/FPI format with
    netValue that compute_fii_flow_signal expects.
    """
    df = store.sql("SELECT * FROM nse_fii_stats WHERE date = ?", [d.isoformat()])
    if df is None or df.empty:
        return pd.DataFrame([{"category": "FII/FPI", "netValue": 0.0}])

    fut_mask = df["category"].str.contains("FUTURES", case=False, na=False)
    fut = df[fut_mask]
    if fut.empty:
        return pd.DataFrame([{"category": "FII/FPI", "netValue": 0.0}])

    rows = []
    for _, row in fut.iterrows():
        buy = float(row.get("buy_amt_cr", 0) or 0)
        sell = float(row.get("sell_amt_cr", 0) or 0)
        rows.append({
            "category": "FII/FPI",
            "sub_category": str(row["category"]).strip().upper(),
            "buyValue": buy,
            "sellValue": sell,
            "netValue": buy - sell,
        })

    total_buy = sum(r["buyValue"] for r in rows)
    total_sell = sum(r["sellValue"] for r in rows)
    rows.append({
        "category": "FII/FPI",
        "sub_category": "TOTAL_FUTURES",
        "buyValue": total_buy,
        "sellValue": total_sell,
        "netValue": total_buy - total_sell,
    })
    return pd.DataFrame(rows)


def extract_nearest_futures_oi(fno_df: pd.DataFrame, d: date) -> pd.DataFrame:
    """Extract stock futures OI from the nearest expiry.

    Reads raw NSE columns (FinInstrmTp=="STF", TckrSymb, XpryDt, etc.)
    and returns scanner-compatible output: SYMBOL, CLOSE, OPEN_INT, CHG_IN_OI.
    """
    out_cols = ["SYMBOL", "CLOSE", "OPEN_INT", "CHG_IN_OI"]

    if fno_df.empty or "FinInstrmTp" not in fno_df.columns:
        return pd.DataFrame(columns=out_cols)

    stk = fno_df[fno_df["FinInstrmTp"].str.strip() == "STF"].copy()
    if stk.empty:
        return pd.DataFrame(columns=out_cols)

    stk["_expiry"] = pd.to_datetime(
        stk["XpryDt"].astype(str).str.strip(), format="mixed", dayfirst=True,
    )
    stk = stk[stk["_expiry"] >= pd.Timestamp(d)]
    if stk.empty:
        return pd.DataFrame(columns=out_cols)

    stk = stk.sort_values("_expiry")
    nearest = stk.groupby("TckrSymb").first().reset_index()
    # Rename raw NSE columns to scanner-compatible names
    rename = {"TckrSymb": "SYMBOL", "ClsPric": "CLOSE",
              "OpnIntrst": "OPEN_INT", "ChngInOpnIntrst": "CHG_IN_OI"}
    nearest = nearest.rename(columns=rename)
    available = [c for c in out_cols if c in nearest.columns]
    return nearest[available].copy()


def get_futures_oi(store, d: date) -> pd.DataFrame:
    """Get nearest-expiry stock futures OI for a date."""
    return extract_nearest_futures_oi(get_fno(store, d), d)


def available_dates(store, category: str = "nse_delivery") -> list[date]:
    """List dates with data for a DuckDB category."""
    return store.available_dates(category)
