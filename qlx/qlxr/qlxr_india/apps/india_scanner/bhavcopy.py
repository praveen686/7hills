"""NSE bhavcopy data download, parsing, and parquet caching.

Data sources (all public, no API key required):
  - Delivery data (sec_bhavdata_full): OHLCV + delivery qty/% — primary equity source
  - F&O bhavcopy (new format): OI, contracts, OHLCV for derivatives
  - FII stats (XLS): FII buy/sell in crores for derivatives

All data is cached as parquet files under data/india/{category}/YYYY-MM-DD.parquet.

URL patterns (verified Feb 2026):
  - Delivery:  archives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv
  - F&O:       archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip
  - FII stats: archives.nseindia.com/content/fo/fii_stats_DD-Mon-YYYY.xls
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# NSE blocks requests without browser-like headers
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

_MONTH_ABBR = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# NSE holidays (Saturday/Sunday handled separately) — major ones
# Not exhaustive; download failures on holidays are handled gracefully.
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
    if d.weekday() >= 5:  # Saturday or Sunday
        return False
    if d in _KNOWN_HOLIDAYS_2025_2026:
        return False
    return True


class NSESession:
    """HTTP session with rate limiting for NSE archive downloads.

    Archives don't require cookies. The main nseindia.com API does,
    but we avoid it (uses Akamai bot protection).
    """

    ARCHIVE_URL = "https://archives.nseindia.com"

    def __init__(self, min_delay: float = 0.5):
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        self.min_delay = min_delay
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self._last_request_time = time.monotonic()

    def get(self, url: str, **kwargs) -> requests.Response:
        """Rate-limited GET."""
        self._rate_limit()
        kwargs.setdefault("timeout", 30)
        resp = self.session.get(url, **kwargs)
        resp.raise_for_status()
        return resp

    def get_bytes(self, url: str, **kwargs) -> bytes:
        """GET and return raw bytes."""
        return self.get(url, **kwargs).content

    def get_text(self, url: str, **kwargs) -> str:
        """GET and return text."""
        return self.get(url, **kwargs).text


def _parse_csv_from_zip(zip_bytes: bytes, filename_hint: str = "") -> pd.DataFrame:
    """Extract and parse CSV from a zip archive."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        csv_name = names[0]  # typically only one file
        if filename_hint:
            for n in names:
                if filename_hint.lower() in n.lower():
                    csv_name = n
                    break
        with zf.open(csv_name) as f:
            return pd.read_csv(f)


# ---------------------------------------------------------------------------
# Delivery data (primary equity + delivery source)
# ---------------------------------------------------------------------------


def download_delivery_data(session: NSESession, d: date) -> pd.DataFrame:
    """Download sec_bhavdata_full for a date.

    This is the primary data source: contains OHLCV + delivery qty/%.
    Columns: SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE,
    LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TRADED_QTY,
    TURNOVER_LACS, NO_OF_TRADES, DELIVERY_QTY, DELIV_PER

    We normalize to standard names for downstream consumers.
    """
    dd = f"{d.day:02d}"
    mm = f"{d.month:02d}"
    yyyy = str(d.year)

    url = (
        f"{session.ARCHIVE_URL}/products/content/"
        f"sec_bhavdata_full_{dd}{mm}{yyyy}.csv"
    )
    logger.debug("Downloading delivery data: %s", url)
    text = session.get_text(url)
    df = pd.read_csv(io.StringIO(text))

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Normalize column names for consistency
    rename = {
        "OPEN_PRICE": "OPEN",
        "HIGH_PRICE": "HIGH",
        "LOW_PRICE": "LOW",
        "CLOSE_PRICE": "CLOSE",
        "TRADED_QTY": "TOTTRDQTY",
        "TURNOVER_LACS": "TOTTRDVAL",
        "DELIV_PER": "DELIVERY_PCT",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    return df


# ---------------------------------------------------------------------------
# F&O bhavcopy (new NSE format as of ~2024)
# ---------------------------------------------------------------------------


def download_fno_bhav(session: NSESession, d: date) -> pd.DataFrame:
    """Download F&O bhavcopy in the new NSE format.

    URL: archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip

    New column names → normalized to old-style names:
      TckrSymb → SYMBOL, FinInstrmTp → INSTRUMENT,
      XpryDt → EXPIRY_DT, ClsPric → CLOSE,
      OpnIntrst → OPEN_INT, ChngInOpnIntrst → CHG_IN_OI
    """
    yyyymmdd = d.strftime("%Y%m%d")
    url = (
        f"{session.ARCHIVE_URL}/content/fo/"
        f"BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
    )
    logger.debug("Downloading F&O bhav: %s", url)
    data = session.get_bytes(url)
    df = _parse_csv_from_zip(data)
    df.columns = df.columns.str.strip()

    # Map new column names to the old ones used by signals.py
    rename = {
        "TckrSymb": "SYMBOL",
        "FinInstrmTp": "INSTRUMENT",
        "XpryDt": "EXPIRY_DT",
        "OpnPric": "OPEN",
        "HghPric": "HIGH",
        "LwPric": "LOW",
        "ClsPric": "CLOSE",
        "OpnIntrst": "OPEN_INT",
        "ChngInOpnIntrst": "CHG_IN_OI",
        "TtlTradgVol": "CONTRACTS",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Map instrument types: STF → FUTSTK, STO → OPTSTK, etc.
    instr_map = {"STF": "FUTSTK", "STO": "OPTSTK", "IDF": "FUTIDX", "IDO": "OPTIDX"}
    if "INSTRUMENT" in df.columns:
        df["INSTRUMENT"] = df["INSTRUMENT"].map(instr_map).fillna(df["INSTRUMENT"])

    return df


# ---------------------------------------------------------------------------
# FII stats (XLS from archives — historical OK)
# ---------------------------------------------------------------------------


def download_fii_stats(session: NSESession, d: date) -> pd.DataFrame:
    """Download FII derivative statistics for a date.

    URL: archives.nseindia.com/content/fo/fii_stats_DD-Mon-YYYY.xls
    This is a binary XLS file. We extract:
      - Index Futures buy/sell
      - Stock Futures buy/sell
    and compute FII net = (buy - sell) for stock+index futures in crores.

    Returns DataFrame with columns: category, buyValue, sellValue, netValue.
    """
    dd = f"{d.day:02d}"
    mon = _MONTH_ABBR[d.month]
    yyyy = str(d.year)

    url = f"{session.ARCHIVE_URL}/content/fo/fii_stats_{dd}-{mon}-{yyyy}.xls"
    logger.debug("Downloading FII stats: %s", url)
    data = session.get_bytes(url)

    df = pd.read_excel(io.BytesIO(data), engine="xlrd", header=None)

    # Parse the structured XLS:
    # Row 2: INDEX FUTURES with buy_crores in col 2, sell_crores in col 4
    # Row 16: STOCK FUTURES with buy_crores in col 2, sell_crores in col 4
    rows = []
    for idx, row in df.iterrows():
        label = str(row.iloc[0]).strip().upper() if pd.notna(row.iloc[0]) else ""
        if label in ("INDEX FUTURES", "STOCK FUTURES"):
            try:
                buy = float(row.iloc[2])
                sell = float(row.iloc[4])
                rows.append({
                    "category": "FII/FPI",
                    "sub_category": label,
                    "buyValue": buy,
                    "sellValue": sell,
                    "netValue": buy - sell,
                })
            except (ValueError, TypeError, IndexError):
                continue

    if not rows:
        # Return aggregate net = 0 if parsing failed
        return pd.DataFrame([{
            "category": "FII/FPI",
            "buyValue": 0.0, "sellValue": 0.0, "netValue": 0.0,
        }])

    result = pd.DataFrame(rows)

    # Also add a combined row for total FII futures net
    total_buy = result["buyValue"].sum()
    total_sell = result["sellValue"].sum()
    combined = pd.DataFrame([{
        "category": "FII/FPI",
        "sub_category": "TOTAL_FUTURES",
        "buyValue": total_buy,
        "sellValue": total_sell,
        "netValue": total_buy - total_sell,
    }])
    return pd.concat([result, combined], ignore_index=True)


# ---------------------------------------------------------------------------
# extract_nearest_futures_oi
# ---------------------------------------------------------------------------


def extract_nearest_futures_oi(fno_df: pd.DataFrame, d: date) -> pd.DataFrame:
    """Extract stock futures OI from the nearest expiry.

    From the F&O bhavcopy, filters to FUTSTK (stock futures),
    picks the nearest expiry for each symbol, and returns
    SYMBOL, CLOSE, OPEN_INT, CHG_IN_OI.
    """
    if fno_df.empty:
        return pd.DataFrame(columns=["SYMBOL", "CLOSE", "OPEN_INT", "CHG_IN_OI"])

    if "INSTRUMENT" not in fno_df.columns:
        return pd.DataFrame(columns=["SYMBOL", "CLOSE", "OPEN_INT", "CHG_IN_OI"])

    # Filter to stock futures
    mask = fno_df["INSTRUMENT"].str.strip() == "FUTSTK"
    stk = fno_df[mask].copy()

    if stk.empty:
        return pd.DataFrame(columns=["SYMBOL", "CLOSE", "OPEN_INT", "CHG_IN_OI"])

    # Parse expiry dates
    stk["EXPIRY_DT"] = pd.to_datetime(
        stk["EXPIRY_DT"].astype(str).str.strip(), format="mixed", dayfirst=True,
    )

    # Keep only future expiries (>= current date)
    stk = stk[stk["EXPIRY_DT"] >= pd.Timestamp(d)]

    if stk.empty:
        return pd.DataFrame(columns=["SYMBOL", "CLOSE", "OPEN_INT", "CHG_IN_OI"])

    # Nearest expiry per symbol
    stk = stk.sort_values("EXPIRY_DT")
    nearest = stk.groupby("SYMBOL").first().reset_index()

    cols = ["SYMBOL", "CLOSE", "OPEN_INT", "CHG_IN_OI"]
    available = [c for c in cols if c in nearest.columns]
    return nearest[available].copy()


# ---------------------------------------------------------------------------
# Parquet cache
# ---------------------------------------------------------------------------


class BhavcopyCache:
    """Parquet-backed cache for NSE daily data.

    Directory layout:
      data/india/delivery/YYYY-MM-DD.parquet   (equity OHLCV + delivery %)
      data/india/fno/YYYY-MM-DD.parquet        (F&O contracts + OI)
      data/india/fii_dii/YYYY-MM-DD.parquet    (FII buy/sell/net)
    """

    def __init__(self, base_dir: str | Path = "data/india"):
        self.base_dir = Path(base_dir)
        self._session: NSESession | None = None

    @property
    def session(self) -> NSESession:
        if self._session is None:
            self._session = NSESession()
        return self._session

    def _path(self, category: str, d: date) -> Path:
        return self.base_dir / category / f"{d.isoformat()}.parquet"

    def _has(self, category: str, d: date) -> bool:
        return self._path(category, d).exists()

    def _save(self, category: str, d: date, df: pd.DataFrame) -> None:
        path = self._path(category, d)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.debug("Cached %s/%s (%d rows)", category, d, len(df))

    def _load(self, category: str, d: date) -> pd.DataFrame:
        return pd.read_parquet(self._path(category, d))

    @staticmethod
    def _normalize_delivery_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure delivery DataFrame has consistent column names.

        Handles both old-format cached files (OPEN_PRICE, DELIV_PER, etc.)
        and new-format files (OPEN, DELIVERY_PCT, etc.).
        """
        rename = {
            "OPEN_PRICE": "OPEN",
            "HIGH_PRICE": "HIGH",
            "LOW_PRICE": "LOW",
            "CLOSE_PRICE": "CLOSE",
            "TRADED_QTY": "TOTTRDQTY",
            "TURNOVER_LACS": "TOTTRDVAL",
            "DELIV_PER": "DELIVERY_PCT",
        }
        return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    def get_equity(self, d: date) -> pd.DataFrame:
        """Get equity data (from delivery file which contains OHLCV).

        Returns the delivery DataFrame filtered to EQ series with
        normalized columns: SYMBOL, OPEN, HIGH, LOW, CLOSE, TOTTRDQTY, etc.
        """
        return self.get_delivery(d)

    def get_delivery(self, d: date) -> pd.DataFrame:
        """Get delivery data (includes equity OHLCV + delivery %)."""
        if self._has("delivery", d):
            return self._normalize_delivery_cols(self._load("delivery", d))
        df = download_delivery_data(self.session, d)
        self._save("delivery", d, df)
        return df

    def get_fno(self, d: date) -> pd.DataFrame:
        """Get F&O bhavcopy, downloading if not cached."""
        if self._has("fno", d):
            return self._load("fno", d)
        df = download_fno_bhav(self.session, d)
        self._save("fno", d, df)
        return df

    def get_fii_dii(self, d: date) -> pd.DataFrame:
        """Get FII/DII data, downloading if not cached."""
        if self._has("fii_dii", d):
            return self._load("fii_dii", d)
        df = download_fii_stats(self.session, d)
        self._save("fii_dii", d, df)
        return df

    def get_futures_oi(self, d: date) -> pd.DataFrame:
        """Get nearest-expiry stock futures OI for a date."""
        fno_df = self.get_fno(d)
        return extract_nearest_futures_oi(fno_df, d)

    def get_all(self, d: date) -> dict[str, pd.DataFrame]:
        """Download all data types for a date. Returns dict of DataFrames."""
        result = {}
        errors = []

        for name, getter in [
            ("delivery", self.get_delivery),
            ("fno", self.get_fno),
            ("fii_dii", self.get_fii_dii),
        ]:
            try:
                result[name] = getter(d)
            except Exception as e:
                logger.warning("Failed to get %s for %s: %s", name, d, e)
                errors.append(name)

        if errors:
            logger.warning("Missing data for %s: %s", d, errors)

        return result

    def backfill(
        self,
        start: date,
        end: date | None = None,
        categories: list[str] | None = None,
    ) -> dict[str, int]:
        """Bulk download historical data.

        Returns {category: count_of_days_downloaded}.
        """
        if end is None:
            end = date.today() - timedelta(days=1)

        if categories is None:
            categories = ["delivery", "fno", "fii_dii"]

        getter_map = {
            "delivery": self.get_delivery,
            "fno": self.get_fno,
            "fii_dii": self.get_fii_dii,
        }

        counts: dict[str, int] = {c: 0 for c in categories}
        d = start

        while d <= end:
            if not is_trading_day(d):
                d += timedelta(days=1)
                continue

            for cat in categories:
                if cat not in getter_map:
                    continue
                if self._has(cat, d):
                    counts[cat] += 1
                    continue
                try:
                    getter_map[cat](d)
                    counts[cat] += 1
                except Exception as e:
                    logger.warning("Backfill %s/%s failed: %s", cat, d, e)

            d += timedelta(days=1)

        return counts

    def available_dates(self, category: str = "delivery") -> list[date]:
        """List dates that have cached data for a category."""
        cat_dir = self.base_dir / category
        if not cat_dir.exists():
            return []
        dates = []
        for p in cat_dir.glob("*.parquet"):
            try:
                dates.append(date.fromisoformat(p.stem))
            except ValueError:
                continue
        return sorted(dates)
