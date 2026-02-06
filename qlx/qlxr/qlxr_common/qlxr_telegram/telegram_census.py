
import os
import asyncio
from pathlib import Path
from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from dotenv import load_dotenv
from collections import Counter

load_dotenv(Path(__file__).parent.parent / "qlxr_vault" / ".env")

API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH")
CHANNEL_NAME = "nfo_data"
SESSION_PATH = Path(__file__).parent / "brahmastra_session"

async def channel_census():
    if not API_ID or not API_HASH:
        print("Error: TELEGRAM_API_ID or TELEGRAM_API_HASH not found in .env")
        return

    client = TelegramClient(str(SESSION_PATH), API_ID, API_HASH)
    
    try:
        await client.start()
        print(f"Connected to Telegram. Performing Census for channel: {CHANNEL_NAME}...")
        
        counts = Counter()
        file_types = Counter()
        total_files = 0
        
        async for message in client.iter_messages(CHANNEL_NAME):
            if message.file:
                total_files += 1
                year = message.date.year
                counts[year] += 1
                
                ext = "None"
                if message.file.name:
                    ext = os.path.splitext(message.file.name)[1].lower()
                file_types[ext] += 1

        print("\n" + "="*40)
        print(f"TELEGRAM CHANNEL CENSUS: {CHANNEL_NAME}")
        print("="*40)
        print(f"Total Messages with Files: {total_files}")
        print("\nFiles per Year:")
        for year in sorted(counts.keys()):
            print(f"  {year}: {counts[year]}")
            
        print("\nFile Types:")
        for ext, count in file_types.most_common(10):
            print(f"  {ext if ext else 'No Ext'}: {count}")
        print("="*40)

    except Exception as e:
        print(f"Error during census: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(channel_census())
