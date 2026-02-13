"""Search for dialogs containing 'NFO' in the name."""

import os
import asyncio
from pathlib import Path
from telethon import TelegramClient
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH")
SESSION_PATH = Path(__file__).parent / "brahmastra_session"


async def search_nfo():
    if not API_ID or not API_HASH:
        print("Error: TELEGRAM_API_ID or TELEGRAM_API_HASH not found in .env")
        return

    client = TelegramClient(str(SESSION_PATH), API_ID, API_HASH)

    try:
        await client.start()
        print("Searching for 'NFO' in dialog titles...")
        print("-" * 50)
        found = False
        async for dialog in client.iter_dialogs():
            if "NFO" in dialog.name.upper():
                username = getattr(dialog.entity, 'username', 'N/A')
                print(f"Title: {dialog.name} | Username: {username} | ID: {dialog.id}")
                found = True

        if not found:
            print("No dialogs found containing 'NFO'.")
    except Exception as e:
        print(f"Error searching dialogs: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(search_nfo())
