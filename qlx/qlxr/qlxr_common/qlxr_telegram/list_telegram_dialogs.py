
import os
import asyncio
from pathlib import Path
from telethon import TelegramClient
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "qlxr_vault" / ".env")

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
        print("Connected to Telegram. Listing dialogs...")

        async for dialog in client.iter_dialogs():
            # Check for channels or groups that might contain data
            if dialog.is_channel or dialog.is_group:
                print(f"ID: {dialog.id} | TYPE: {'Channel' if dialog.is_channel else 'Group'} | NAME: {dialog.name}")

    except Exception as e:
        print(f"Error listing dialogs: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(list_dialogs())
