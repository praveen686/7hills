"""Audit nfo_data channel for date range coverage and file types."""

import os
import re
import asyncio
import datetime
from collections import defaultdict
from pathlib import Path
from telethon import TelegramClient
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH")
CHANNEL_NAME = "nfo_data"
SESSION_PATH = Path(__file__).parent / "brahmastra_session"


async def audit_dates():
    if not API_ID or not API_HASH:
        print("Error: TELEGRAM_API_ID or TELEGRAM_API_HASH not found in .env")
        return

    client = TelegramClient(str(SESSION_PATH), API_ID, API_HASH)

    try:
        await client.start()
        print(f"Auditing channel: {CHANNEL_NAME}...")

        stats = defaultdict(lambda: {"start": None, "end": None, "count": 0})

        async for message in client.iter_messages(CHANNEL_NAME, limit=5000):
            if message.file and message.file.name:
                fname = message.file.name
                ext = "Unknown"
                if fname.endswith('.feather'):
                    ext = "1-Min Feather"
                elif "tick_data" in fname and fname.endswith('.zip'):
                    ext = "Tick Data (Zip)"
                elif "1 HOUR" in fname:
                    ext = "1-Hour Legacy"

                if ext != "Unknown":
                    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
                    if date_match:
                        dt = datetime.datetime.strptime(date_match.group(1), "%Y-%m-%d").date()
                        stats[ext]["count"] += 1
                        if not stats[ext]["start"] or dt < stats[ext]["start"]:
                            stats[ext]["start"] = dt
                        if not stats[ext]["end"] or dt > stats[ext]["end"]:
                            stats[ext]["end"] = dt

        print("\nData Coverage Audit:")
        print(f"{'Data Type':<20} | {'Start Date':<12} | {'End Date':<12} | {'File Count':<10}")
        print("-" * 65)
        for ext, s in stats.items():
            print(f"{ext:<20} | {str(s['start']):<12} | {str(s['end']):<12} | {s['count']:<10}")
    except Exception as e:
        print(f"Error during audit: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(audit_dates())
