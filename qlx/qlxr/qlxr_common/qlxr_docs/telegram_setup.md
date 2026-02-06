# Telegram Connection Guide

## Overview
We use Telegram to access NFO tick data from the `NFO_DAILY_DATA` channel (554 days of historical tick data).

## Credentials

### Location
`/home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_common/qlxr_vault/.env`

### Values
```
TELEGRAM_API_ID=22957880
TELEGRAM_API_HASH=5345db7a31962869f1203981dc2b82c4
TELEGRAM_PHONE=+919676501414
```

## Session File

### Location
`/home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_common/qlxr_telegram/brahmastra_session.session`

### What it is
- SQLite database storing Telegram auth key
- Created after first successful login
- Must not be shared (contains encryption keys)

## Quick Connection Test

```bash
source /home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_common/qlxr_env/bin/activate

python3 -c "
from telethon import TelegramClient
import asyncio

api_id = 22957880
api_hash = '5345db7a31962869f1203981dc2b82c4'
session_path = '/home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_common/qlxr_telegram/brahmastra_session'

async def test():
    client = TelegramClient(session_path, api_id, api_hash)
    await client.connect()
    me = await client.get_me()
    print(f'Connected: {me.first_name} ({me.phone})')
    await client.disconnect()

asyncio.run(test())
"
```

Expected output:
```
Connected: Praveen (919676501414)
```

## Telegram Scripts

### Location
`/home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_common/qlxr_telegram/`

### Available Scripts
| Script | Purpose |
|--------|---------|
| `telegram_downloader.py` | Main data downloader |
| `list_telegram_channels.py` | List accessible channels |
| `list_telegram_dialogs.py` | List all dialogs |
| `search_nfo_dialogs.py` | Search for NFO data channels |
| `search_telegram_archive.py` | Search message archives |
| `telegram_census.py` | Count messages per channel |
| `audit_telegram_dates.py` | Check available date ranges |
| `test_telegram_load.py` | Test data loading |

## NFO Data Channel

### Channel Name
`NFO_DAILY_DATA`

### Data Available
- **Period**: Nov 2023 - Feb 2026 (554 days)
- **Size**: ~11.8 GB total
- **Format**: Zstd-compressed pickle files

### File Types
| Pattern | Content |
|---------|---------|
| `*-index-nfo-data.feather` | 1-min OHLCV bars |
| `*_tick_data.zip` | Tick-by-tick data |

### Tick Data Format
```python
# Each tick: [timestamp, price, volume, oi]
# Stored as: {instrument_token: [[ts, px, vol, oi], ...]}
```

## Re-authentication (if session expires)

If the session file becomes invalid, re-authenticate:

```python
from telethon import TelegramClient
import asyncio

api_id = 22957880
api_hash = '5345db7a31962869f1203981dc2b82c4'
phone = '+919676501414'
session_path = '/home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_common/qlxr_telegram/brahmastra_session'

async def login():
    client = TelegramClient(session_path, api_id, api_hash)
    await client.start(phone=phone)
    me = await client.get_me()
    print(f'Logged in as: {me.first_name}')
    await client.disconnect()

asyncio.run(login())
```

You will receive an OTP on Telegram. Enter it when prompted.

## Troubleshooting

### "NOT AUTHORIZED"
- Session expired or corrupted
- Re-run authentication script above

### "No module named 'telethon'"
- Activate venv first: `source /home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_common/qlxr_env/bin/activate`

### Session file missing
- Check path: `/home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_common/qlxr_telegram/`
- Re-authenticate if needed

## Dependencies
- Python package: `telethon 1.42.0`
- Installed in: `qlxr_env`

## Updated
2026-02-06
