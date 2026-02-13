"""Sample earliest messages in nfo_data channel (date-sorted)."""

import os
import asyncio
from pathlib import Path
from telethon import TelegramClient
from telethon.tl.functions.channels import GetFullChannelRequest
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH")
CHANNEL_NAME = "nfo_data"
SESSION_PATH = Path(__file__).parent / "brahmastra_session"


async def search_telegram():
    if not API_ID or not API_HASH:
        print("Error: TELEGRAM_API_ID or TELEGRAM_API_HASH not found in .env")
        return

    client = TelegramClient(str(SESSION_PATH), API_ID, API_HASH)

    try:
        await client.start()
        print(f"Connected to Telegram. Searching channel: {CHANNEL_NAME}")

        # Get channel entity
        entity = await client.get_input_entity(CHANNEL_NAME)

        # Get full channel info for total message count
        full_channel = await client(GetFullChannelRequest(channel=entity))
        total_msg = full_channel.full_chat.read_outbox_max_id
        print(f"Total Message ID (Approx): {total_msg}")

        print("\nSAMPLING THE BEGINNING (REVERSE=TRUE) - FIRST 500 MESSAGES:")
        found_samples = []
        async for message in client.iter_messages(CHANNEL_NAME, reverse=True, limit=500):
            if message.file and message.file.name:
                found_samples.append({
                    "name": message.file.name,
                    "date": message.date,
                    "size": message.file.size,
                })

        if not found_samples:
            print("No files found in the first 500 messages.")
        else:
            found_samples.sort(key=lambda x: x["date"])
            for f in found_samples[:100]:
                print(f"- {f['name']} ({f['date'].strftime('%Y-%m-%d %H:%M')}, {f['size']/1024/1024:.2f} MB)")

    except Exception as e:
        print(f"Error searching Telegram: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(search_telegram())
