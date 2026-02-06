
import os
import asyncio
from telethon import TelegramClient
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")

async def list_dialogs():
    async with TelegramClient('brahmastra_session', API_ID, API_HASH) as client:
        print(f"{'Title':<30} | {'Username':<20} | {'ID':<20}")
        print("-" * 75)
        async for dialog in client.iter_dialogs():
            username = getattr(dialog.entity, 'username', 'N/A')
            print(f"{dialog.name[:30]:<30} | {str(username):<20} | {dialog.id:<20}")

if __name__ == "__main__":
    if not API_ID or not API_HASH:
        print("Error: credentials missing in .env")
    else:
        asyncio.run(list_dialogs())
