"""List all accessible Telegram dialogs with names and IDs."""

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


async def list_dialogs():
    if not API_ID or not API_HASH:
        print("Error: TELEGRAM_API_ID or TELEGRAM_API_HASH not found in .env")
        return

    client = TelegramClient(str(SESSION_PATH), API_ID, API_HASH)

    try:
        await client.start()
        print(f"{'Title':<30} | {'Username':<20} | {'ID':<20}")
        print("-" * 75)
        async for dialog in client.iter_dialogs():
            username = getattr(dialog.entity, 'username', 'N/A')
            print(f"{dialog.name[:30]:<30} | {str(username):<20} | {dialog.id:<20}")
    except Exception as e:
        print(f"Error listing dialogs: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(list_dialogs())
