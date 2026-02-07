
import argparse
import os
import re
import asyncio
import traceback
from datetime import date
from pathlib import Path
from telethon import TelegramClient, events
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv(Path(__file__).parent.parent / ".env")

API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH")
TELEGRAM_PHONE = os.getenv("TELEGRAM_PHONE") or os.getenv("ph_number_telegram")
CHANNEL_NAME = "nfo_data"
DOWNLOAD_DIR = Path("/home/ubuntu/Desktop/7hills/QuantLaxmi/data/telegram_source_files/india_tick_data")

# Configuration constants
DOWNLOAD_TIMEOUT = 600  # 10 minutes per file
MAX_ZIP_DEPTH = 5  # Prevent infinite recursion in nested zips
DEFAULT_RECENT_LIMIT = 50  # Enough to catch a day's uploads (~4 files/day)
FULL_HISTORY_LIMIT = 50000  # For backfill scans

# Session file lives alongside this script
SESSION_PATH = Path(__file__).parent / "brahmastra_session"

async def download_nfo_data(message_limit: int = DEFAULT_RECENT_LIMIT):
    """
    Scrapes the specified Telegram channel for data files.

    Args:
        message_limit: Number of recent messages to scan (default: 50).
    """
    if not API_ID or not API_HASH:
        print("Error: TELEGRAM_API_ID or TELEGRAM_API_HASH not found in .env")
        return

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Adding a small sleep to ensure any previous session closure is processed
    await asyncio.sleep(1)

    client = TelegramClient(str(SESSION_PATH), API_ID, API_HASH)

    try:
        await client.start(phone=TELEGRAM_PHONE)
        print(f"Connected to Telegram. Scanning last {message_limit} messages in '{CHANNEL_NAME}'")

        existing = set(os.listdir(DOWNLOAD_DIR))
        count = 0
        skipped = 0
        total = 0
        async for message in client.iter_messages(CHANNEL_NAME, limit=message_limit):
            if message.file and message.file.name:
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
                size_mb = message.file.size / 1024 / 1024
                print(f"[{count}/{total}] Downloading {fname} ({size_mb:.1f} MB)...")
                try:
                    await asyncio.wait_for(
                        message.download_media(file=str(target_path)),
                        timeout=DOWNLOAD_TIMEOUT
                    )
                    # Extract after download
                    if fname.lower().endswith('.zip'):
                        await extract_nested_zips(target_path, DOWNLOAD_DIR)
                except asyncio.TimeoutError:
                    print(f"  -> TIMEOUT: Download of {fname} exceeded {DOWNLOAD_TIMEOUT}s, skipping")
                    if target_path.exists():
                        target_path.unlink()
                except Exception as e:
                    print(f"  -> ERROR downloading {fname}: {e}")
                    if target_path.exists():
                        target_path.unlink()
        print(f"Sync complete. Downloaded {count} new files, skipped {skipped} existing, {total} total scanned.")

        # Always check for unconverted data (covers both new downloads
        # and previously downloaded files that weren't ingested yet)
        await _ingest_new_dates(DOWNLOAD_DIR)
    except Exception as e:
        print(f"Error during Telegram sync: {e}")
        traceback.print_exc()
    finally:
        await client.disconnect()

def _sync_extract_zip(zip_path: Path, extract_to: Path) -> list:
    """Synchronous zip extraction - runs in thread pool to avoid blocking event loop."""
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        return zip_ref.namelist()


def _sync_decompress_zstd(zip_path: Path, target_path: Path) -> None:
    """Synchronous zstd decompression - runs in thread pool to avoid blocking event loop."""
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    with open(zip_path, 'rb') as ifh, open(target_path, 'wb') as ofh:
        dctx.copy_stream(ifh, ofh)


async def extract_nested_zips(zip_path: Path, extract_to: Path, depth: int = 0):
    """
    Extracts a zip file and recursively checks for inner zip files to extract them too.
    Handles fallback to Zstandard decompression if zipfile fails.

    Args:
        zip_path: Path to the zip file to extract
        extract_to: Directory to extract files into
        depth: Current recursion depth (for protection against zip bombs)
    """
    import zipfile

    # Prevent infinite recursion / zip bomb attacks
    if depth > MAX_ZIP_DEPTH:
        print(f"  -> WARNING: Max extraction depth ({MAX_ZIP_DEPTH}) reached for {zip_path.name}, skipping")
        return

    print(f"{'  ' * depth}Checking {zip_path.name}...")

    # Idempotency check: If it's a tick_data zip, check if pkl already exists
    target_name = zip_path.stem
    if "tick_data" in target_name:
        target_name += ".pkl"
    elif "instrument_df" in target_name:
        target_name += ".pkl"

    if (extract_to / target_name).exists():
        print(f"{'  ' * depth}  -> Already extracted: {target_name}")
        return

    # 1. Try standard Zip (run in thread to avoid blocking event loop)
    if zipfile.is_zipfile(zip_path):
        print(f"{'  ' * depth}  -> Detected as ZIP. Extracting...")
        try:
            # Run blocking I/O in thread pool
            extracted_files = await asyncio.to_thread(_sync_extract_zip, zip_path, extract_to)
            # Check for inner zips/zst
            for extracted_file in extracted_files:
                inner_path = extract_to / extracted_file
                if inner_path.suffix.lower() == '.zip':
                    await extract_nested_zips(inner_path, extract_to, depth + 1)
            return
        except Exception as e:
            print(f"{'  ' * depth}  -> ZIP Extraction failed: {e}")
            traceback.print_exc()

    # 2. Try Zstandard (run in thread to avoid blocking event loop)
    try:
        print(f"{'  ' * depth}  -> Attempting Zstandard decompression...")
        # If the file is named .zip but is zstd, we strip .zip
        target_name = zip_path.stem if zip_path.suffix.lower() == '.zip' else zip_path.name + ".decompressed"
        # Special case: if it's tick_data_*.zip, it's likely a pickle dump
        if "tick_data" in target_name:
            target_name += ".pkl"
        elif "instrument_df" in target_name:
            target_name += ".pkl"

        target_path = extract_to / target_name

        # Run blocking I/O in thread pool - these files are 130+ MB!
        await asyncio.to_thread(_sync_decompress_zstd, zip_path, target_path)
        print(f"{'  ' * depth}  -> Success! Decompressed to {target_name}")
    except Exception as e:
        print(f"{'  ' * depth}  -> Zstandard Extraction failed: {e}")
        traceback.print_exc()

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


async def _ingest_new_dates(download_dir: Path) -> None:
    """Convert newly downloaded files to hive-partitioned parquet.

    Runs conversion in a thread pool so it doesn't block the event loop.
    Only converts dates that haven't been converted yet (idempotent).
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "QuantLaxmi"))
        from core.market.convert import convert_all, discover_sources, discover_converted

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
    args = parser.parse_args()

    limit = FULL_HISTORY_LIMIT if args.full else (args.limit or DEFAULT_RECENT_LIMIT)
    asyncio.run(download_nfo_data(message_limit=limit))
