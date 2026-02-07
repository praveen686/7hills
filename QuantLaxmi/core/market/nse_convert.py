"""Convert NSE daily files to hive-partitioned Parquet.

Source layout (from collectors.nse_daily):
  data/nse/daily/YYYY-MM-DD/<23 file types>

Output layout (in MARKET_DIR):
  nse_fo_bhavcopy/date=YYYY-MM-DD/data.parquet
  nse_cm_bhavcopy/date=YYYY-MM-DD/data.parquet
  ... (23 categories total)

Usage:
  python -m core.market.nse_convert                        # convert all dates
  python -m core.market.nse_convert --dates 2026-02-05     # specific date(s)
  python -m core.market.nse_convert --dry-run              # show what would be done
  python -m core.market.nse_convert --force                # re-convert existing
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_ROOT = Path(
    os.environ.get(
        "QLX_DATA_ROOT",
        "/home/ubuntu/Desktop/7hills/QuantLaxmi/data",
    )
)

# NSE daily source files (from collectors.nse_daily)
NSE_SOURCE_DIR = Path(
    os.environ.get(
        "QLX_NSE_DAILY_DIR",
        "/home/ubuntu/Desktop/7hills/QuantLaxmi/data/nse/daily",
    )
)

MARKET_DIR = _DATA_ROOT / "market"

MIN_FILE_SIZE = 10  # skip files smaller than this (empty/corrupt)

# ---------------------------------------------------------------------------
# Registry — maps source filename → parquet category + parser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NseFileSpec:
    source_name: str        # filename in date dir
    category: str           # parquet category (nse_*)
    parser: str             # parser function name suffix
    optional: bool = False  # True if file may not exist on all dates


REGISTRY: list[NseFileSpec] = [
    # Tier 1 — Critical
    NseFileSpec("fo_bhavcopy.csv.zip",    "nse_fo_bhavcopy",       "zip_csv"),
    NseFileSpec("cm_bhavcopy.csv.zip",    "nse_cm_bhavcopy",       "zip_csv"),
    NseFileSpec("participant_oi.csv",     "nse_participant_oi",    "participant"),
    NseFileSpec("participant_vol.csv",    "nse_participant_vol",   "participant"),
    NseFileSpec("settlement_prices.csv",  "nse_settlement_prices", "csv"),
    NseFileSpec("volatility.csv",         "nse_volatility",        "volatility"),
    NseFileSpec("contract_delta.csv",     "nse_contract_delta",    "csv", optional=True),
    NseFileSpec("index_close.csv",        "nse_index_close",       "csv"),
    NseFileSpec("delivery_bhavcopy.csv",  "nse_delivery",          "csv"),
    # Tier 2 — High
    NseFileSpec("fii_stats.xls",          "nse_fii_stats",         "fii_stats"),
    NseFileSpec("market_activity.zip",    "nse_market_activity",   "market_activity"),
    NseFileSpec("nse_oi.zip",             "nse_oi",                "zip_csv"),
    NseFileSpec("combined_oi.zip",        "nse_combined_oi",       "zip_csv"),
    NseFileSpec("combined_oi_deleq.csv",  "nse_combined_oi_deleq", "csv"),
    NseFileSpec("security_ban.csv",       "nse_security_ban",      "security_ban", optional=True),
    NseFileSpec("fo_contract.csv.gz",     "nse_fo_contract",       "gzip_csv"),
    NseFileSpec("bulk_deals.csv",         "nse_bulk_deals",        "deals", optional=True),
    NseFileSpec("block_deals.csv",        "nse_block_deals",       "deals", optional=True),
    NseFileSpec("mto.dat",               "nse_mto",               "mto"),
    NseFileSpec("margin_data.dat",       "nse_margin_data",       "margin_data"),
    NseFileSpec("fo_mktlots.csv",        "nse_fo_mktlots",        "fo_mktlots", optional=True),
    NseFileSpec("52wk_highlow.csv",      "nse_52wk_highlow",      "52wk", optional=True),
    NseFileSpec("top_gainers.json",     "nse_top_gainers",       "gainers_json", optional=True),
]

REGISTRY_MAP: dict[str, NseFileSpec] = {s.category: s for s in REGISTRY}

# Volatility column rename mapping (16 long names → short)
_VOL_COLS = {
    "Date": "date",
    "Symbol": "symbol",
    "Underlying Close Price (A)": "underlying_close",
    "Underlying Previous Day Close Price (B)": "underlying_prev_close",
    "Underlying Log Returns (C) = LN(A/B)": "underlying_log_return",
    "Previous Day Underlying Volatility (D)": "prev_underlying_vol",
    "Current Day Underlying Daily Volatility (E) = Sqrt (0.995*D*D + 0.005* C*C)": "current_underlying_vol",
    "Underlying Annualised Volatility (F) = E*sqrt(365)": "underlying_ann_vol",
    "Futures Close Price (G)": "futures_close",
    "Futures Previous Day Close Price (H)": "futures_prev_close",
    "Futures Log Returns (I) = LN(G/H)": "futures_log_return",
    "Previous Day Futures Volatility (J)": "prev_futures_vol",
    "Current Day Futures Daily Volatility (K) = Sqrt (0.995*J*J + 0.005* I*I)": "current_futures_vol",
    "Futures Annualised Volatility (L) = K*sqrt(365)": "futures_ann_vol",
    "Applicable Daily Volatility (M) = Max (E or K)": "applicable_daily_vol",
    "Applicable Annualised Volatility (N) = Max (F or L)": "applicable_ann_vol",
}


# ---------------------------------------------------------------------------
# Parsers — each returns pd.DataFrame | None
# ---------------------------------------------------------------------------


def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and string values."""
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    return df


def parse_csv(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    return _strip_cols(df)


def parse_zip_csv(path: Path) -> pd.DataFrame | None:
    with zipfile.ZipFile(path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            logger.warning("No CSV found in %s", path.name)
            return None
        df = pd.read_csv(zf.open(csv_names[0]))
    return _strip_cols(df)


def parse_gzip_csv(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path, compression="gzip")
    return _strip_cols(df)


def parse_participant(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path, skiprows=1)
    return _strip_cols(df)


def parse_volatility(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    df = _strip_cols(df)
    df.rename(columns={k.strip(): v for k, v in _VOL_COLS.items()}, inplace=True)
    return df


def parse_fii_stats(path: Path) -> pd.DataFrame | None:
    try:
        import xlrd
    except ImportError:
        logger.error("xlrd not installed — cannot parse %s", path.name)
        return None

    wb = xlrd.open_workbook(str(path))
    ws = wb.sheet_by_index(0)

    rows = []
    for i in range(3, ws.nrows):
        vals = ws.row_values(i)
        label = str(vals[0]).strip()
        if not label or label.upper() in ("", "TOTAL"):
            continue
        try:
            rows.append({
                "category": label,
                "buy_contracts": int(vals[1]) if vals[1] != "" else 0,
                "buy_amt_cr": float(vals[2]) if vals[2] != "" else 0.0,
                "sell_contracts": int(vals[3]) if vals[3] != "" else 0,
                "sell_amt_cr": float(vals[4]) if vals[4] != "" else 0.0,
                "oi_contracts": int(vals[5]) if vals[5] != "" else 0,
                "oi_amt_cr": float(vals[6]) if vals[6] != "" else 0.0,
            })
        except (ValueError, IndexError):
            continue

    if not rows:
        return None
    return pd.DataFrame(rows)


def parse_market_activity(path: Path) -> pd.DataFrame | None:
    with zipfile.ZipFile(path, "r") as zf:
        # Find the fo_DDMMYYYY.csv summary file (with underscore)
        # May be at root or inside a subdirectory
        import posixpath
        summary = [
            n for n in zf.namelist()
            if posixpath.basename(n).startswith("fo_") and n.endswith(".csv")
        ]
        if not summary:
            logger.warning("No fo_*.csv summary in %s", path.name)
            return None
        df = pd.read_csv(zf.open(summary[0]), skiprows=1)
    return _strip_cols(df)


def parse_security_ban(path: Path) -> pd.DataFrame | None:
    text = path.read_text().strip()
    lines = text.splitlines()
    if len(lines) <= 1:
        # Only header, no bans
        return pd.DataFrame(columns=["rank", "symbol"])

    rows = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 1)
        if len(parts) == 2:
            rows.append({"rank": int(parts[0].strip()), "symbol": parts[1].strip()})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["rank", "symbol"])


def parse_deals(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    df = _strip_cols(df)
    # Check for "NO RECORDS" sentinel
    if len(df) > 0:
        first_val = str(df.iloc[0, 0]).strip().upper()
        if first_val == "NO RECORDS":
            return pd.DataFrame(columns=df.columns)
    return df


def parse_mto(path: Path) -> pd.DataFrame | None:
    # MTO format: header line(s) with rec_type 10, then descriptive lines,
    # then column header line, then data lines with rec_type 20.
    # We read all lines, skip non-data, parse rec_type==20.
    lines = path.read_text().splitlines()
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("20,"):
            data_lines.append(line)

    if not data_lines:
        return None

    names = ["rec_type", "sr_no", "symbol", "series",
             "qty_traded", "deliv_qty", "deliv_pct"]
    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        header=None,
        names=names,
    )
    df = _strip_cols(df)
    df.drop(columns=["rec_type"], inplace=True)
    return df


def parse_margin_data(path: Path) -> pd.DataFrame | None:
    lines = path.read_text().splitlines()
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("20,"):
            data_lines.append(line)

    if not data_lines:
        return None

    names = ["rec_type", "symbol", "series", "isin",
             "var_margin", "elm", "adhoc_margin",
             "applicable_var", "applicable_elm", "applicable_adhoc"]
    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        header=None,
        names=names,
    )
    df = _strip_cols(df)
    df.drop(columns=["rec_type"], inplace=True)
    return df


def parse_fo_mktlots(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path, skipinitialspace=True)
    df = _strip_cols(df)

    # Remove divider rows (e.g. "Derivatives on Individual Securities")
    if "UNDERLYING" in df.columns:
        df = df[~df["UNDERLYING"].str.contains("Derivatives", case=False, na=False)]

    # Identify the fixed columns and month columns
    fixed = ["UNDERLYING", "SYMBOL"]
    month_cols = [c for c in df.columns if c not in fixed]

    # Melt wide → long
    df_long = df.melt(
        id_vars=fixed,
        value_vars=month_cols,
        var_name="expiry_month",
        value_name="lot_size",
    )

    # Clean up: remove empty lot sizes, convert to int
    df_long["lot_size"] = pd.to_numeric(df_long["lot_size"], errors="coerce")
    df_long = df_long.dropna(subset=["lot_size"])
    df_long["lot_size"] = df_long["lot_size"].astype(int)

    # Lowercase column names for consistency
    df_long.columns = [c.lower() for c in df_long.columns]

    return df_long if len(df_long) > 0 else None


def parse_52wk(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(
        path,
        skiprows=2,
        quotechar='"',
        skipinitialspace=True,
    )
    df = _strip_cols(df)
    # Strip quotes from string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip('"').str.strip()
    # Normalize column names (some files have spaces, others underscores)
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def parse_gainers_json(path: Path) -> pd.DataFrame | None:
    data = json.loads(path.read_text())
    rows = []
    if isinstance(data, dict):
        # Each key is a category (e.g. "NIFTY", "BANKNIFTY", etc.)
        # Each value is a dict with "data" list of dicts
        for category, category_data in data.items():
            if not isinstance(category_data, dict):
                continue
            items = category_data.get("data", [])
            if not isinstance(items, list):
                continue
            for item in items:
                rows.append({
                    "index_category": category,
                    "symbol": item.get("symbol", ""),
                    "series": item.get("series", ""),
                    "open_price": item.get("open_price", item.get("openPrice")),
                    "high_price": item.get("high_price", item.get("highPrice")),
                    "low_price": item.get("low_price", item.get("lowPrice")),
                    "ltp": item.get("ltp"),
                    "prev_price": item.get("prev_price", item.get("previousPrice")),
                    "perChange": item.get("perChange", item.get("pChange")),
                    "trade_quantity": item.get("trade_quantity", item.get("tradedQuantity")),
                    "turnover": item.get("turnover"),
                })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # Coerce numeric columns
    for col in ("open_price", "high_price", "low_price", "ltp", "prev_price",
                "perChange", "trade_quantity", "turnover"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# Parser dispatch
_PARSERS = {
    "csv": parse_csv,
    "zip_csv": parse_zip_csv,
    "gzip_csv": parse_gzip_csv,
    "participant": parse_participant,
    "volatility": parse_volatility,
    "fii_stats": parse_fii_stats,
    "market_activity": parse_market_activity,
    "security_ban": parse_security_ban,
    "deals": parse_deals,
    "mto": parse_mto,
    "margin_data": parse_margin_data,
    "fo_mktlots": parse_fo_mktlots,
    "52wk": parse_52wk,
    "gainers_json": parse_gainers_json,
}


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------


def _parquet_out(category: str, d: date) -> Path:
    return MARKET_DIR / category / f"date={d.isoformat()}" / "data.parquet"


def convert_nse_file(spec: NseFileSpec, d: date, force: bool = False) -> int:
    """Convert a single NSE file for a date. Returns row count (0 if skipped)."""
    dst = _parquet_out(spec.category, d)

    # Skip if already converted (unless force)
    if dst.exists() and not force:
        return -1  # sentinel: already done

    src = NSE_SOURCE_DIR / d.isoformat() / spec.source_name
    if not src.exists():
        if not spec.optional:
            logger.debug("Missing (non-optional): %s for %s", spec.source_name, d)
        return 0

    if src.stat().st_size < MIN_FILE_SIZE:
        logger.debug("Skipping tiny file: %s (%d bytes)", src, src.stat().st_size)
        return 0

    parser = _PARSERS.get(spec.parser)
    if parser is None:
        logger.error("No parser '%s' for %s", spec.parser, spec.category)
        return 0

    try:
        df = parser(src)
    except Exception as e:
        logger.error("Parse error %s/%s: %s", d, spec.source_name, e)
        return 0

    if df is None or df.empty:
        # For deals files, still write empty parquet to avoid re-parsing
        if spec.parser == "deals" and df is not None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst, engine="pyarrow", compression="zstd", index=False)
            return 0
        return 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, engine="pyarrow", compression="zstd", index=False)
    return len(df)


def convert_nse_day(d: date, force: bool = False) -> dict[str, int]:
    """Convert all NSE files for a single date. Returns {category: row_count}."""
    day_dir = NSE_SOURCE_DIR / d.isoformat()
    if not day_dir.exists():
        logger.debug("No NSE source dir for %s", d)
        return {}

    results: dict[str, int] = {}
    for spec in REGISTRY:
        n = convert_nse_file(spec, d, force=force)
        if n > 0:
            results[spec.category] = n
        elif n == -1:
            pass  # already done, don't log
    return results


def discover_nse_dates() -> list[date]:
    """Find all date directories in NSE source."""
    if not NSE_SOURCE_DIR.exists():
        return []
    dates = []
    for d_dir in NSE_SOURCE_DIR.iterdir():
        if d_dir.is_dir():
            try:
                dates.append(date.fromisoformat(d_dir.name))
            except ValueError:
                continue
    return sorted(dates)


def convert_all_nse(
    dates: list[date] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, dict[str, int]]:
    """Batch-convert all NSE daily data."""
    all_dates = discover_nse_dates()

    if dates:
        requested = set(dates)
        all_dates = [d for d in all_dates if d in requested]

    if not all_dates:
        logger.info("No NSE dates to convert")
        return {}

    # Count how many need work (have at least one unconverted file)
    to_convert = []
    for d in all_dates:
        if force:
            to_convert.append(d)
            continue
        for spec in REGISTRY:
            dst = _parquet_out(spec.category, d)
            src = NSE_SOURCE_DIR / d.isoformat() / spec.source_name
            if src.exists() and src.stat().st_size >= MIN_FILE_SIZE and not dst.exists():
                to_convert.append(d)
                break

    logger.info(
        "NSE daily: %d total dates, %d need conversion",
        len(all_dates), len(to_convert),
    )

    if dry_run:
        for d in to_convert:
            day_dir = NSE_SOURCE_DIR / d.isoformat()
            files = [f.name for f in day_dir.iterdir() if f.is_file()]
            print(f"  {d.isoformat()}: {len(files)} source files")
        return {}

    results: dict[str, dict[str, int]] = {}
    t0 = time.time()

    for i, d in enumerate(to_convert, 1):
        t1 = time.time()
        try:
            day_results = convert_nse_day(d, force=force)
            results[d.isoformat()] = day_results
            elapsed = time.time() - t1
            summary = ", ".join(f"{k.replace('nse_', '')}={v:,}" for k, v in day_results.items())
            logger.info(
                "[%d/%d] %s: %s (%.1fs)",
                i, len(to_convert), d.isoformat(), summary or "nothing new", elapsed,
            )
        except Exception as e:
            logger.error("[%d/%d] %s: FAILED: %s", i, len(to_convert), d.isoformat(), e)
            results[d.isoformat()] = {"error": str(e)}

    total_time = time.time() - t0
    total_rows = sum(
        sum(v for v in day.values() if isinstance(v, int))
        for day in results.values()
    )
    logger.info(
        "NSE conversion complete: %d dates, %d total rows, %.1fs",
        len(results), total_rows, total_time,
    )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NSE daily files to hive-partitioned Parquet"
    )
    parser.add_argument(
        "--dates", nargs="+", type=date.fromisoformat,
        help="Specific dates (YYYY-MM-DD)",
    )
    parser.add_argument("--force", action="store_true", help="Re-convert existing")
    parser.add_argument("--dry-run", action="store_true", help="Show plan only")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    results = convert_all_nse(
        dates=args.dates,
        force=args.force,
        dry_run=args.dry_run,
    )

    if not args.dry_run and results:
        # Aggregate stats per category
        cat_stats: dict[str, tuple[int, int]] = {}  # {cat: (days, rows)}
        for day in results.values():
            for cat, n in day.items():
                if isinstance(n, int):
                    days, rows = cat_stats.get(cat, (0, 0))
                    cat_stats[cat] = (days + 1, rows + n)

        print("\nSummary:")
        for cat in sorted(cat_stats):
            days, rows = cat_stats[cat]
            print(f"  {cat}: {days} days, {rows:,} rows")

        # Total parquet size for nse_* dirs
        total_bytes = 0
        for cat_dir in MARKET_DIR.iterdir():
            if cat_dir.is_dir() and cat_dir.name.startswith("nse_"):
                total_bytes += sum(f.stat().st_size for f in cat_dir.rglob("*.parquet"))
        if total_bytes:
            print(f"\n  Total NSE parquet size: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
