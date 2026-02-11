"""
Fetch additional market data from Telegram channels beyond the primary nfo_data channel.

Targets:
  - NFO_DAILY_DATA (ID: -1001751394648) — NFO historical data
  - Data for Algo Traders (ID: -1001433446148) — structured market data
  - Nifty BankNifty Options Data (ID: -1001795149313) — options data
  - optionsdata (user ID: 6099030903) — option chain data

Downloads .feather, .parquet, .csv, .zip files to data/telegram_extra/.
Idempotent: skips files that already exist locally.
"""

import os
import sys
import asyncio
import traceback
from pathlib import Path
from datetime import datetime
from collections import Counter

from dotenv import load_dotenv
from telethon import TelegramClient

# ── Config ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = _PROJECT_ROOT / ".env"
SESSION_PATH = _PROJECT_ROOT / "telegram" / "brahmastra_session"
DOWNLOAD_DIR = _PROJECT_ROOT / "data" / "telegram_extra"
DOWNLOAD_TIMEOUT = 600  # seconds per file
MAX_FILE_SIZE_MB = 500
SCAN_LIMIT = int(os.getenv("TELEGRAM_SCAN_LIMIT", "200"))  # messages per channel

# Extensions we care about
DATA_EXTENSIONS = {".feather", ".parquet", ".csv", ".zip", ".pkl", ".gz", ".zst", ".xls", ".xlsx"}

# Channels to scan — use IDs for reliability (entity resolution by name can fail)
CHANNELS = [
    {"id": -1001751394648, "name": "NFO_DAILY_DATA"},
    {"id": -1001433446148, "name": "Data for Algo Traders"},
    {"id": -1001795149313, "name": "Nifty BankNifty Options Data"},
    {"id": 6099030903,     "name": "optionsdata"},
]

# ── Helpers ─────────────────────────────────────────────────────────────────
load_dotenv(ENV_PATH)

API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH", "")
TELEGRAM_PHONE = os.getenv("TELEGRAM_PHONE") or os.getenv("ph_number_telegram")


def human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def is_data_file(filename: str) -> bool:
    """Check if filename has a data-related extension."""
    if not filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in DATA_EXTENSIONS


# ── Phase 1: Scan and catalog ──────────────────────────────────────────────
async def scan_channel(client: TelegramClient, channel_info: dict, limit: int = SCAN_LIMIT):
    """
    Scan a channel and return a catalog of files found.

    Returns list of dicts: {msg_id, filename, size, date, ext, message}
    """
    channel_id = channel_info["id"]
    channel_name = channel_info["name"]
    catalog = []

    print(f"\n{'='*60}")
    print(f"SCANNING: {channel_name} (ID: {channel_id})")
    print(f"{'='*60}")

    try:
        entity = await client.get_entity(channel_id)
    except Exception as e:
        print(f"  [SKIP] Cannot resolve entity: {e}")
        return catalog

    msg_count = 0
    file_count = 0
    ext_counter = Counter()

    try:
        async for message in client.iter_messages(entity, limit=limit):
            msg_count += 1
            if message.file and message.file.name:
                fname = message.file.name
                fsize = message.file.size or 0
                ext = os.path.splitext(fname)[1].lower()
                ext_counter[ext] += 1
                file_count += 1
                catalog.append({
                    "msg_id": message.id,
                    "filename": fname,
                    "size": fsize,
                    "date": message.date,
                    "ext": ext,
                    "message": message,
                })
            elif message.file:
                # File without a name (e.g. photos, stickers)
                ext_counter["<unnamed>"] += 1
    except Exception as e:
        print(f"  [ERROR] Failed to iterate messages: {e}")
        traceback.print_exc()
        return catalog

    # Print summary
    print(f"  Messages scanned: {msg_count}")
    print(f"  Named files found: {file_count}")
    if ext_counter:
        print(f"  File types:")
        for ext, count in ext_counter.most_common(15):
            print(f"    {ext or '<no ext>'}: {count}")

    # Print file listing (data files only)
    data_files = [f for f in catalog if is_data_file(f["filename"])]
    print(f"\n  Data files ({len(data_files)} of {file_count} total):")
    for f in data_files[:50]:  # show first 50
        size_str = human_size(f["size"])
        date_str = f["date"].strftime("%Y-%m-%d")
        print(f"    [{date_str}] {f['filename']} ({size_str})")
    if len(data_files) > 50:
        print(f"    ... and {len(data_files) - 50} more")

    return catalog


# ── Phase 2: Download ──────────────────────────────────────────────────────
async def download_files(client: TelegramClient, catalog: list, channel_name: str):
    """
    Download data files from the catalog.
    Skips files that already exist or exceed size limit.
    """
    # Create per-channel subdirectory
    channel_dir = DOWNLOAD_DIR / channel_name.replace(" ", "_")
    channel_dir.mkdir(parents=True, exist_ok=True)

    existing = set(os.listdir(channel_dir))
    data_files = [f for f in catalog if is_data_file(f["filename"])]

    downloaded = 0
    skipped_exists = 0
    skipped_size = 0
    failed = 0
    total_bytes = 0

    for i, finfo in enumerate(data_files, 1):
        fname = finfo["filename"]
        fsize = finfo["size"]
        size_mb = fsize / (1024 * 1024)

        # Skip if already exists
        if fname in existing:
            skipped_exists += 1
            continue

        # Skip if too large
        if size_mb > MAX_FILE_SIZE_MB:
            print(f"  [SKIP] {fname} too large ({size_mb:.0f} MB > {MAX_FILE_SIZE_MB} MB)")
            skipped_size += 1
            continue

        target_path = channel_dir / fname
        print(f"  [{i}/{len(data_files)}] Downloading {fname} ({human_size(fsize)})...")

        try:
            await asyncio.wait_for(
                finfo["message"].download_media(file=str(target_path)),
                timeout=DOWNLOAD_TIMEOUT,
            )
            downloaded += 1
            total_bytes += fsize
            print(f"    -> OK")
        except asyncio.TimeoutError:
            print(f"    -> TIMEOUT ({DOWNLOAD_TIMEOUT}s)")
            if target_path.exists():
                target_path.unlink()
            failed += 1
        except Exception as e:
            print(f"    -> ERROR: {e}")
            if target_path.exists():
                target_path.unlink()
            failed += 1

    print(f"\n  Download summary for {channel_name}:")
    print(f"    Downloaded: {downloaded} ({human_size(total_bytes)})")
    print(f"    Skipped (exists): {skipped_exists}")
    print(f"    Skipped (too large): {skipped_size}")
    print(f"    Failed: {failed}")

    return downloaded, total_bytes


# ── Main ────────────────────────────────────────────────────────────────────
async def main():
    if not API_ID or not API_HASH:
        print("ERROR: TELEGRAM_API_ID or TELEGRAM_API_HASH not found in .env")
        sys.exit(1)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Telegram Data Fetcher")
    print(f"Session: {SESSION_PATH}")
    print(f"Download dir: {DOWNLOAD_DIR}")
    print(f"Scan limit: {SCAN_LIMIT} messages per channel")
    print(f"Max file size: {MAX_FILE_SIZE_MB} MB")
    print(f"Channels to scan: {len(CHANNELS)}")

    client = TelegramClient(str(SESSION_PATH), API_ID, API_HASH)

    total_downloaded = 0
    total_bytes = 0

    try:
        await client.start(phone=TELEGRAM_PHONE)
        print("Connected to Telegram.\n")

        for ch in CHANNELS:
            # Phase 1: scan and catalog
            catalog = await scan_channel(client, ch, limit=SCAN_LIMIT)

            # Phase 2: download data files
            if catalog:
                dl_count, dl_bytes = await download_files(client, catalog, ch["name"])
                total_downloaded += dl_count
                total_bytes += dl_bytes

        print(f"\n{'='*60}")
        print(f"GRAND TOTAL: Downloaded {total_downloaded} files ({human_size(total_bytes)})")
        print(f"{'='*60}")

    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
