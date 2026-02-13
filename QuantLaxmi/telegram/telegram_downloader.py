"""Download NFO data from Telegram nfo_data channel.

Modes:
  Recent (default):  Scan last 50 messages, download new files, extract, ingest.
  Full (--full):     Scan entire channel history (50K messages).
  GDrive (--gdrive): Upload downloaded files to Google Drive via rclone after sync.

Usage::

    # Daily sync (called by LiveEngine at 15:35 IST)
    python telegram/telegram_downloader.py

    # Full channel backfill + GDrive upload
    python telegram/telegram_downloader.py --full --gdrive

    # Custom limit
    python telegram/telegram_downloader.py --limit 500 --gdrive
"""

import argparse
import os
import re
import asyncio
import shutil
import subprocess
import traceback
from datetime import date
from pathlib import Path
from telethon import TelegramClient
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# Resolve project root (telegram/ is one level below QuantLaxmi/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load credentials from .env
load_dotenv(_PROJECT_ROOT / ".env")

API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH")
TELEGRAM_PHONE = os.getenv("TELEGRAM_PHONE") or os.getenv("ph_number_telegram")
CHANNEL_NAME = "nfo_data"
DOWNLOAD_DIR = _PROJECT_ROOT / "data" / "telegram_source_files" / "india_tick_data"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOWNLOAD_TIMEOUT = 600  # 10 minutes per file
MAX_ZIP_DEPTH = 5  # Prevent infinite recursion in nested zips
DEFAULT_RECENT_LIMIT = 50  # Enough to catch a day's uploads (~4 files/day)
FULL_HISTORY_LIMIT = 50000  # For backfill scans
UPLOAD_BATCH = 50  # Upload to GDrive every N new downloads

# Session file lives alongside this script
SESSION_PATH = Path(__file__).parent / "brahmastra_session"

# ---------------------------------------------------------------------------
# Google Drive upload via rclone
# ---------------------------------------------------------------------------

_RCLONE_PATHS = [
    Path.home() / "bin" / "rclone",
    Path("/usr/local/bin/rclone"),
    Path("/usr/bin/rclone"),
]

GDRIVE_DEST = "gdrive:QuantLaxmi_Telegram"


def _find_rclone() -> str | None:
    """Find rclone binary on disk."""
    for p in _RCLONE_PATHS:
        if p.exists():
            return str(p)
    return shutil.which("rclone")


def rclone_upload(local_dir: Path | None = None, dest: str | None = None) -> bool:
    """Upload local files directory to Google Drive via rclone.

    Returns True on success, False on failure.
    """
    rclone = _find_rclone()
    if not rclone:
        print("Warning: rclone not found, skipping GDrive upload")
        return False

    local = str(local_dir or DOWNLOAD_DIR)
    remote = dest or GDRIVE_DEST

    cmd = [rclone, "copy", local, remote, "--progress", "--transfers", "8"]
    print(f"\nUploading {local} → {remote} ...")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("Upload to Google Drive complete!")
        return True
    else:
        print(f"rclone upload exited with code {result.returncode}")
        return False


# ---------------------------------------------------------------------------
# Main download function
# ---------------------------------------------------------------------------

async def download_nfo_data(
    message_limit: int = DEFAULT_RECENT_LIMIT,
    gdrive: bool = False,
):
    """Download data files from the nfo_data Telegram channel.

    Args:
        message_limit: Number of recent messages to scan.
        gdrive: If True, upload to Google Drive after download.
    """
    if not API_ID or not API_HASH:
        print("Error: TELEGRAM_API_ID or TELEGRAM_API_HASH not found in .env")
        return

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Small sleep to ensure any previous session closure is processed
    await asyncio.sleep(1)

    client = TelegramClient(str(SESSION_PATH), API_ID, API_HASH)

    try:
        await client.start(phone=TELEGRAM_PHONE)
        print(f"Connected to Telegram. Scanning last {message_limit} messages in '{CHANNEL_NAME}'")
        print(f"  Download dir: {DOWNLOAD_DIR}")
        if gdrive:
            print(f"  GDrive dest:  {GDRIVE_DEST}")

        existing = set(os.listdir(DOWNLOAD_DIR))
        count = 0
        skipped = 0
        errors = 0
        total = 0
        total_bytes = 0

        async for message in client.iter_messages(CHANNEL_NAME, limit=message_limit):
            if not message.file or not message.file.name:
                continue

            fname = message.file.name
            total += 1
            target_path = DOWNLOAD_DIR / fname

            if fname in existing:
                skipped += 1
                # Still extract zips if not yet extracted
                if fname.lower().endswith('.zip'):
                    await extract_nested_zips(target_path, DOWNLOAD_DIR)
                continue

            count += 1
            size_mb = (message.file.size or 0) / 1024 / 1024
            print(f"[{count}/{total}] Downloading {fname} ({size_mb:.1f} MB)...")

            try:
                await asyncio.wait_for(
                    message.download_media(file=str(target_path)),
                    timeout=DOWNLOAD_TIMEOUT,
                )
                total_bytes += message.file.size or 0
                # Extract after download
                if fname.lower().endswith('.zip'):
                    await extract_nested_zips(target_path, DOWNLOAD_DIR)
            except asyncio.TimeoutError:
                print(f"  -> TIMEOUT: Download of {fname} exceeded {DOWNLOAD_TIMEOUT}s, skipping")
                errors += 1
                if target_path.exists():
                    target_path.unlink()
            except Exception as e:
                print(f"  -> ERROR downloading {fname}: {e}")
                errors += 1
                if target_path.exists():
                    target_path.unlink()

            # Periodic GDrive upload during large backfills
            if gdrive and count > 0 and count % UPLOAD_BATCH == 0:
                print(f"\n--- Batch upload ({count} files so far) ---")
                rclone_upload()

        total_gb = total_bytes / 1024**3
        print(
            f"\nSync complete. Downloaded {count} new files ({total_gb:.2f} GB), "
            f"skipped {skipped} existing, {errors} errors, {total} total scanned."
        )

        # Always check for unconverted data
        await _ingest_new_dates(DOWNLOAD_DIR)

        # Final GDrive upload
        if gdrive and count > 0:
            rclone_upload()
        elif gdrive and count == 0:
            print("No new files to upload.")

    except Exception as e:
        print(f"Error during Telegram sync: {e}")
        traceback.print_exc()
    finally:
        await client.disconnect()


# ---------------------------------------------------------------------------
# Zip/Zstd extraction
# ---------------------------------------------------------------------------

def _sync_extract_zip(zip_path: Path, extract_to: Path) -> list:
    """Synchronous zip extraction — runs in thread pool."""
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        return zip_ref.namelist()


def _sync_decompress_zstd(zip_path: Path, target_path: Path) -> None:
    """Synchronous zstd decompression — runs in thread pool."""
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    with open(zip_path, 'rb') as ifh, open(target_path, 'wb') as ofh:
        dctx.copy_stream(ifh, ofh)


async def extract_nested_zips(zip_path: Path, extract_to: Path, depth: int = 0):
    """Extract a zip file, recursively handling inner zips and zstd fallback.

    Args:
        zip_path: Path to the zip file to extract.
        extract_to: Directory to extract files into.
        depth: Current recursion depth (zip bomb protection).
    """
    import zipfile

    if depth > MAX_ZIP_DEPTH:
        print(f"  -> WARNING: Max extraction depth ({MAX_ZIP_DEPTH}) reached for {zip_path.name}, skipping")
        return

    print(f"{'  ' * depth}Checking {zip_path.name}...")

    # Idempotency: check if target already exists
    target_name = zip_path.stem
    if "tick_data" in target_name:
        target_name += ".pkl"
    elif "instrument_df" in target_name:
        target_name += ".pkl"

    if (extract_to / target_name).exists():
        print(f"{'  ' * depth}  -> Already extracted: {target_name}")
        return

    # 1. Try standard Zip
    if zipfile.is_zipfile(zip_path):
        print(f"{'  ' * depth}  -> Detected as ZIP. Extracting...")
        try:
            extracted_files = await asyncio.to_thread(_sync_extract_zip, zip_path, extract_to)
            for extracted_file in extracted_files:
                inner_path = extract_to / extracted_file
                if inner_path.suffix.lower() == '.zip':
                    await extract_nested_zips(inner_path, extract_to, depth + 1)
            return
        except Exception as e:
            print(f"{'  ' * depth}  -> ZIP Extraction failed: {e}")
            traceback.print_exc()

    # 2. Fallback: Zstandard decompression
    try:
        print(f"{'  ' * depth}  -> Attempting Zstandard decompression...")
        target_name = zip_path.stem if zip_path.suffix.lower() == '.zip' else zip_path.name + ".decompressed"
        if "tick_data" in target_name:
            target_name += ".pkl"
        elif "instrument_df" in target_name:
            target_name += ".pkl"

        target_path = extract_to / target_name
        await asyncio.to_thread(_sync_decompress_zstd, zip_path, target_path)
        print(f"{'  ' * depth}  -> Success! Decompressed to {target_name}")
    except Exception as e:
        print(f"{'  ' * depth}  -> Zstandard Extraction failed: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Auto-ingestion to parquet
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


async def _ingest_new_dates(download_dir: Path) -> None:
    """Convert newly downloaded files to hive-partitioned parquet.

    Runs conversion in a thread pool so it doesn't block the event loop.
    Only converts dates that haven't been converted yet (idempotent).
    """
    try:
        from quantlaxmi.data.convert import convert_all, discover_sources, discover_converted

        sources = discover_sources()
        converted = discover_converted()

        all_source_dates = sorted(
            sources["nfo_feather"]
            | sources["bfo_feather"]
            | sources["tick_zip"]
            | sources["tick_pkl"]
            | sources["instrument_pkl"]
        )

        # Find dates needing conversion
        all_converted = (
            converted.get("nfo_1min", set())
            & converted.get("bfo_1min", set())
        )
        new_dates = [d for d in all_source_dates if d not in all_converted]

        if new_dates:
            print(f"Ingesting {len(new_dates)} dates into parquet store...")
            results = await asyncio.to_thread(convert_all, new_dates)
            total_rows = sum(
                sum(v for v in day.values() if isinstance(v, int))
                for day in results.values()
            )
            print(f"Ingestion complete: {len(results)} dates, {total_rows:,} rows")
        else:
            print("All dates already ingested into parquet store.")
    except Exception as e:
        print(f"Warning: Parquet ingestion failed (data still available as raw files): {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NFO data from Telegram")
    parser.add_argument(
        "--full", action="store_true",
        help=f"Scan full channel history ({FULL_HISTORY_LIMIT} messages) instead of recent {DEFAULT_RECENT_LIMIT}",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Custom message scan limit",
    )
    parser.add_argument(
        "--gdrive", action="store_true",
        help="Upload to Google Drive via rclone after download",
    )
    args = parser.parse_args()

    limit = FULL_HISTORY_LIMIT if args.full else (args.limit or DEFAULT_RECENT_LIMIT)
    asyncio.run(download_nfo_data(message_limit=limit, gdrive=args.gdrive))
