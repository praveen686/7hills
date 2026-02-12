"""Download entire nfo_data Telegram channel → Google Drive (QuantLaxmi_Telegram/)."""

import os
import asyncio
import subprocess
import traceback
from pathlib import Path
from telethon import TelegramClient
from dotenv import load_dotenv

# Load credentials from QuantLaxmi/.env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent / "QuantLaxmi"
load_dotenv(_PROJECT_ROOT / ".env")

API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH")
TELEGRAM_PHONE = os.getenv("TELEGRAM_PHONE") or os.getenv("ph_number_telegram")
CHANNEL_NAME = "nfo_data"

# Temp local dir on EC2, then upload to Google Drive
LOCAL_DIR = Path(__file__).resolve().parent / "files"
GDRIVE_DEST = "gdrive:QuantLaxmi_Telegram"
RCLONE_BIN = Path.home() / "bin" / "rclone"

# Reuse existing session
SESSION_PATH = _PROJECT_ROOT / "telegram" / "brahmastra_session"

DOWNLOAD_TIMEOUT = 600  # 10 min per file
SCAN_LIMIT = 100000  # effectively unlimited
UPLOAD_BATCH = 10  # upload to gdrive every N files


def rclone_upload():
    """Upload local files dir to Google Drive."""
    cmd = [
        str(RCLONE_BIN), "copy",
        str(LOCAL_DIR), GDRIVE_DEST,
        "--progress", "--transfers", "8",
    ]
    print(f"\nUploading {LOCAL_DIR} → {GDRIVE_DEST} ...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        print("Upload to Google Drive complete!")
    else:
        print(f"rclone upload exited with code {result.returncode}")


async def download_all():
    if not API_ID or not API_HASH:
        print("Error: TELEGRAM_API_ID or TELEGRAM_API_HASH not found in .env")
        return

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    client = TelegramClient(str(SESSION_PATH), API_ID, API_HASH)

    try:
        await client.connect()
        if not await client.is_user_authorized():
            print("Error: Telegram session not authorized. Run auth manually first.")
            return
        print(f"Connected. Downloading ALL files from '{CHANNEL_NAME}'")
        print(f"  Local staging: {LOCAL_DIR}")
        print(f"  Google Drive:  {GDRIVE_DEST}")

        existing = set(os.listdir(LOCAL_DIR))
        downloaded = 0
        skipped = 0
        errors = 0
        total_bytes = 0

        async for message in client.iter_messages(CHANNEL_NAME, limit=SCAN_LIMIT):
            if not message.file or not message.file.name:
                continue

            fname = message.file.name
            target_path = LOCAL_DIR / fname

            if fname in existing:
                skipped += 1
                continue

            size_mb = (message.file.size or 0) / 1024 / 1024
            downloaded += 1
            print(f"[{downloaded}] Downloading {fname} ({size_mb:.1f} MB)...")

            try:
                await asyncio.wait_for(
                    message.download_media(file=str(target_path)),
                    timeout=DOWNLOAD_TIMEOUT,
                )
                total_bytes += message.file.size or 0
            except asyncio.TimeoutError:
                print(f"  TIMEOUT: {fname}")
                errors += 1
                if target_path.exists():
                    target_path.unlink()
            except Exception as e:
                print(f"  ERROR: {fname}: {e}")
                errors += 1
                if target_path.exists():
                    target_path.unlink()

            # Periodic upload to Google Drive
            if downloaded > 0 and downloaded % UPLOAD_BATCH == 0:
                print(f"\n--- Batch upload ({downloaded} files so far) ---")
                rclone_upload()

        total_gb = total_bytes / 1024**3
        print(f"\nTelegram download done! {downloaded} files ({total_gb:.2f} GB), "
              f"skipped {skipped} existing, {errors} errors.")

        # Final upload
        if downloaded > 0:
            rclone_upload()
        else:
            print("No new files to upload.")

    except Exception as e:
        print(f"Fatal: {e}")
        traceback.print_exc()
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(download_all())
