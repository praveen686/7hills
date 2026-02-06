"""NSE daily file definitions — URL patterns, date formats, tiers.

Each NSEFile defines one downloadable archive from nsearchives.nseindia.com.
URL templates use {date} as placeholder, formatted per the file's date_format.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# Base URL for all NSE archive downloads
BASE_URL = "https://nsearchives.nseindia.com"


def format_date(d: date, fmt: str) -> str:
    """Format a date according to NSE URL conventions.

    Supported formats:
        YYYYMMDD    -> 20260205
        DDMMYYYY    -> 05022026
        DD-Mon-YYYY -> 05-Feb-2026
    """
    if fmt == "YYYYMMDD":
        return d.strftime("%Y%m%d")
    elif fmt == "DDMMYYYY":
        return d.strftime("%d%m%Y")
    elif fmt == "DD-Mon-YYYY":
        return d.strftime("%d-%b-%Y")
    else:
        raise ValueError(f"Unknown date format: {fmt}")


@dataclass(frozen=True)
class NSEFile:
    """Definition of a single NSE archive file."""

    name: str           # Local filename (stable, human-readable)
    url_template: str   # Path on nsearchives with {date} placeholder
    date_format: str    # One of: YYYYMMDD, DDMMYYYY, DD-Mon-YYYY
    tier: int           # 1 = critical, 2 = high
    optional: bool = False  # True if file may not exist on all dates
    base_url: str = BASE_URL  # Override for files on a different host

    def url_for_date(self, d: date) -> str:
        """Build the full download URL for a given date."""
        date_str = format_date(d, self.date_format)
        path = self.url_template.replace("{date}", date_str)
        return f"{self.base_url}/{path}"


# ---------------------------------------------------------------------------
# Tier 1 — CRITICAL (9 files)
# ---------------------------------------------------------------------------

_TIER1 = [
    NSEFile(
        name="fo_bhavcopy.csv.zip",
        url_template="content/fo/BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip",
        date_format="YYYYMMDD",
        tier=1,
    ),
    NSEFile(
        name="cm_bhavcopy.csv.zip",
        url_template="content/cm/BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip",
        date_format="YYYYMMDD",
        tier=1,
    ),
    NSEFile(
        name="participant_oi.csv",
        url_template="content/nsccl/fao_participant_oi_{date}.csv",
        date_format="DDMMYYYY",
        tier=1,
    ),
    NSEFile(
        name="participant_vol.csv",
        url_template="content/nsccl/fao_participant_vol_{date}.csv",
        date_format="DDMMYYYY",
        tier=1,
    ),
    NSEFile(
        name="settlement_prices.csv",
        url_template="archives/nsccl/sett/FOSett_prce_{date}.csv",
        date_format="DDMMYYYY",
        tier=1,
    ),
    NSEFile(
        name="volatility.csv",
        url_template="archives/nsccl/volt/FOVOLT_{date}.csv",
        date_format="DDMMYYYY",
        tier=1,
    ),
    NSEFile(
        name="contract_delta.csv",
        url_template="content/nsccl/Contract_Delta_{date}.csv",
        date_format="DDMMYYYY",
        tier=1,
    ),
    NSEFile(
        name="index_close.csv",
        url_template="content/indices/ind_close_all_{date}.csv",
        date_format="DDMMYYYY",
        tier=1,
    ),
    NSEFile(
        name="delivery_bhavcopy.csv",
        url_template="products/content/sec_bhavdata_full_{date}.csv",
        date_format="DDMMYYYY",
        tier=1,
    ),
]

# ---------------------------------------------------------------------------
# Tier 2 — HIGH (14 files)
# ---------------------------------------------------------------------------

_TIER2 = [
    NSEFile(
        name="fii_stats.xls",
        url_template="content/fo/fii_stats_{date}.xls",
        date_format="DD-Mon-YYYY",
        tier=2,
    ),
    NSEFile(
        name="market_activity.zip",
        url_template="archives/fo/mkt/fo{date}.zip",
        date_format="DDMMYYYY",
        tier=2,
    ),
    NSEFile(
        name="nse_oi.zip",
        url_template="archives/nsccl/mwpl/ncloi_{date}.zip",
        date_format="DDMMYYYY",
        tier=2,
    ),
    NSEFile(
        name="combined_oi.zip",
        url_template="archives/nsccl/mwpl/combineoi_{date}.zip",
        date_format="DDMMYYYY",
        tier=2,
    ),
    NSEFile(
        name="combined_oi_deleq.csv",
        url_template="archives/nsccl/mwpl/combineoi_deleq_{date}.csv",
        date_format="DDMMYYYY",
        tier=2,
    ),
    NSEFile(
        name="security_ban.csv",
        url_template="archives/fo/sec_ban/fo_secban_{date}.csv",
        date_format="DDMMYYYY",
        tier=2,
        optional=True,
    ),
    NSEFile(
        name="fo_contract.csv.gz",
        url_template="content/fo/NSE_FO_contract_{date}.csv.gz",
        date_format="DDMMYYYY",
        tier=2,
    ),
    # --- New files ---
    NSEFile(
        name="bulk_deals.csv",
        url_template="content/equities/bulk.csv",  # today-only, no date archive
        date_format="DDMMYYYY",
        tier=2,
        optional=True,
    ),
    NSEFile(
        name="block_deals.csv",
        url_template="content/equities/block.csv",  # today-only, no date archive
        date_format="DDMMYYYY",
        tier=2,
        optional=True,
    ),
    NSEFile(
        name="mto.dat",
        url_template="archives/equities/mto/MTO_{date}.DAT",
        date_format="DDMMYYYY",
        tier=2,
    ),
    NSEFile(
        name="margin_data.dat",
        url_template="archives/nsccl/var/C_VAR1_{date}_1.DAT",
        date_format="DDMMYYYY",
        tier=2,
    ),
    NSEFile(
        name="fo_mktlots.csv",
        url_template="content/fo/fo_mktlots.csv",  # snapshot, no date in URL
        date_format="DDMMYYYY",
        tier=2,
    ),
    NSEFile(
        name="52wk_highlow.csv",
        url_template="content/CM_52_wk_High_low_{date}.csv",
        date_format="DDMMYYYY",
        tier=2,
    ),
    NSEFile(
        name="top_gainers.json",
        url_template="api/live-analysis-variations?index=gainers",
        date_format="DDMMYYYY",
        tier=2,
        optional=True,
        base_url="https://www.nseindia.com",
    ),
]

# All 23 files
ALL_FILES: list[NSEFile] = _TIER1 + _TIER2
