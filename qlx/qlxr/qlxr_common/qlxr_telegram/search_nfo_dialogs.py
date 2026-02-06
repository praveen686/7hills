
import os
import asyncio
from telethon import TelegramClient
from dotenv import load_dotenv

load_dotenv()

API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")

async def search_nfo():
    async with TelegramClient('brahmastra_session', API_ID, API_HASH) as client:
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

if __name__ == "__main__":
    if not API_ID or not API_HASH:
        print("Error: credentials missing in .env")
    else:
        asyncio.run(search_nfo())
