"""Fetch trading/quant books from Telegram channels.

Downloads PDFs, EPUBs, and other document files from
"Technical Library" and "Algorithmic Trading" channels
into telegram_books/ folder.

Usage:
    python -m research.fetch_telegram_books
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH", "")
SESSION_PATH = PROJECT_ROOT / "telegram" / "brahmastra_session"

# Target channels
CHANNELS = {
    "Technical Library": -1001329468681,
    "Algorithmic Trading": -1001174214654,
}

# Output directory
BOOKS_DIR = PROJECT_ROOT / "telegram_books"

# File extensions to download
BOOK_EXTENSIONS = re.compile(
    r"\.(pdf|epub|mobi|djvu|azw3|doc|docx|pptx?|xlsx?|zip|rar|7z)$",
    re.IGNORECASE,
)

# Skip files smaller than this (likely thumbnails/stickers)
MIN_FILE_SIZE = 50_000  # 50 KB

# Download timeout per file
DOWNLOAD_TIMEOUT = 600  # 10 minutes


async def download_books_from_channel(
    client: TelegramClient,
    channel_name: str,
    channel_id: int,
    output_dir: Path,
    limit: int = 50000,
) -> int:
    """Download all book files from a channel."""
    channel_dir = output_dir / channel_name.replace(" ", "_")
    channel_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    errors = 0

    logger.info("Scanning '%s' (ID: %d) for books...", channel_name, channel_id)

    try:
        async for message in client.iter_messages(channel_id, limit=limit):
            if not message.file or not message.file.name:
                continue

            fname = message.file.name
            fsize = message.file.size or 0

            # Check if it's a book/document file
            if not BOOK_EXTENSIONS.search(fname):
                continue

            # Skip tiny files
            if fsize < MIN_FILE_SIZE:
                continue

            target = channel_dir / fname

            # Skip if already downloaded
            if target.exists() and target.stat().st_size > 0:
                skipped += 1
                continue

            size_mb = fsize / (1024 * 1024)
            logger.info(
                "  Downloading: %s (%.1f MB)", fname, size_mb
            )

            try:
                await asyncio.wait_for(
                    message.download_media(file=str(target)),
                    timeout=DOWNLOAD_TIMEOUT,
                )
                downloaded += 1
            except asyncio.TimeoutError:
                logger.warning("  Timeout downloading: %s", fname)
                if target.exists():
                    target.unlink()
                errors += 1
            except Exception as e:
                logger.warning("  Error downloading %s: %s", fname, e)
                errors += 1

    except Exception as e:
        logger.error("Error scanning channel '%s': %s", channel_name, e)

    logger.info(
        "'%s': downloaded=%d, skipped=%d (already exist), errors=%d",
        channel_name, downloaded, skipped, errors,
    )
    return downloaded


async def main():
    if not API_ID or not API_HASH:
        logger.error("TELEGRAM_API_ID or TELEGRAM_API_HASH not found in .env")
        return

    BOOKS_DIR.mkdir(parents=True, exist_ok=True)

    client = TelegramClient(str(SESSION_PATH), API_ID, API_HASH)

    try:
        await client.start()
        logger.info("Connected to Telegram")

        total = 0
        for name, cid in CHANNELS.items():
            count = await download_books_from_channel(
                client, name, cid, BOOKS_DIR
            )
            total += count

        logger.info("Total books downloaded: %d", total)

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"  Telegram Books Download Summary")
        print(f"{'=' * 60}")
        for name in CHANNELS:
            channel_dir = BOOKS_DIR / name.replace(" ", "_")
            if channel_dir.exists():
                files = list(channel_dir.iterdir())
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                print(f"  {name}: {len(files)} files, {total_size / 1024 / 1024:.1f} MB")
        print(f"  Output: {BOOKS_DIR}")
        print(f"{'=' * 60}\n")

    except Exception as e:
        logger.error("Error: %s", e)
    finally:
        await client.disconnect()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    asyncio.run(main())
