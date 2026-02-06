
import os
import asyncio
import traceback
from pathlib import Path
from telethon import TelegramClient, events
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv(Path(__file__).parent.parent / "qlxr_vault" / ".env")

API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH")
TELEGRAM_PHONE = os.getenv("ph_number_telegram")
CHANNEL_NAME = "nfo_data"
DOWNLOAD_DIR = Path("/home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_common/qlxr_data/telegram_source_files/india_tick_data")

# Configuration constants
DOWNLOAD_TIMEOUT = 600  # 10 minutes per file
MAX_ZIP_DEPTH = 5  # Prevent infinite recursion in nested zips
MESSAGE_LIMIT = 50000 # Full history scan

# Session file lives alongside this script
SESSION_PATH = Path(__file__).parent / "brahmastra_session"

async def download_nfo_data():
    """
    Scrapes the specified Telegram channel for CSV data files.
    """
    if not API_ID or not API_HASH:
        print("Error: TELEGRAM_API_ID or TELEGRAM_API_HASH not found in .env")
        return

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use a specific session file name to avoid conflicts if multiple scripts run
    # Adding a small sleep to ensure any previous session closure is processed
    await asyncio.sleep(1)
    
    client = TelegramClient(str(SESSION_PATH), API_ID, API_HASH)
    
    try:
        await client.start(phone=TELEGRAM_PHONE)
        print(f"Connected to Telegram. Checking channel: {CHANNEL_NAME}")

        count = 0
        processed = 0
        async for message in client.iter_messages(CHANNEL_NAME, limit=MESSAGE_LIMIT):
            if message.file and message.file.name:
                fname = message.file.name
                if fname.lower().endswith(('.csv', '.zip', '.feather')):
                    # Filter for relevant files
                    is_relevant = any(x in fname.upper() for x in ["NIFTY", "BANKNIFTY", "INDEX-NFO", "TICK_DATA"])
                    if is_relevant:
                        processed += 1
                        target_path = DOWNLOAD_DIR / fname
                        if not target_path.exists():
                            print(f"[{processed}] Downloading {fname}...")
                            try:
                                await asyncio.wait_for(
                                    message.download_media(file=str(target_path)),
                                    timeout=DOWNLOAD_TIMEOUT
                                )
                                count += 1

                                # Extract after download
                                if fname.lower().endswith('.zip'):
                                    await extract_nested_zips(target_path, DOWNLOAD_DIR)
                            except asyncio.TimeoutError:
                                print(f"  -> TIMEOUT: Download of {fname} exceeded {DOWNLOAD_TIMEOUT}s, skipping")
                                # Clean up partial download
                                if target_path.exists():
                                    target_path.unlink()
                            except Exception as e:
                                print(f"  -> ERROR downloading {fname}: {e}")
                                if target_path.exists():
                                    target_path.unlink()
                        else:
                            # Only extract if not already extracted (idempotency handled in extract_nested_zips)
                            if fname.lower().endswith('.zip'):
                                await extract_nested_zips(target_path, DOWNLOAD_DIR)
        print(f"Sync complete. Downloaded {count} new files, processed {processed} total.")
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

if __name__ == "__main__":
    asyncio.run(download_nfo_data())
