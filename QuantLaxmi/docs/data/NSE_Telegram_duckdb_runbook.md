# NSE & Telegram Data Pipeline Runbook

Daily market data flows from two sources into a unified DuckDB + Parquet store.

```
Telegram (nfo_data channel)          NSE Archives (nsearchives.nseindia.com)
        |                                       |
  telegram_downloader.py              apps.nse_daily collect
        |                                       |
  Raw files (pkl/zip/feather)         Raw files (csv/zip/xls/dat)
        |                                       |
  qlx.data.convert                    qlx.data.nse_convert
        |                                       |
        +----------> market/ <-----------------+
                  (hive-partitioned parquet, zstd)
                         |
                    MarketDataStore
                   (DuckDB SQL views)
```

---

## 1. Daily Operations

### Telegram Downloader (tick data, 1-min bars, instruments)

```bash
cd qlxr/qlxr_common/qlxr_telegram
python telegram_downloader.py
```

- Scans the last 50 messages in the `nfo_data` Telegram channel
- Downloads new files (zip/feather) to `qlxr_data/telegram_source_files/india_tick_data/`
- Extracts nested zips (zstd-compressed pkl inside)
- **Auto-ingests into parquet** — converts any unconverted dates to hive-partitioned parquet

**Output categories** (4):

| Category | Content | ~Rows/day |
|---|---|---|
| `nfo_1min` | NSE F&O 1-minute OHLCV bars | 560K |
| `bfo_1min` | BSE F&O 1-minute OHLCV bars | 214K |
| `ticks` | Tick-level LTP, volume, OI | 3.3M |
| `instruments` | Zerodha instrument master | 195K |

**Flags:**
```
--full          Scan full channel history (50,000 messages) — for backfill
--limit N       Custom scan depth
```

### NSE Daily Collector (23 archive file types)

```bash
cd qlxr/QuantLaxmi
python -m apps.nse_daily collect
```

- Downloads 23 file types from NSE for today (IST)
- Saves to `data/nse/daily/YYYY-MM-DD/`
- **Auto-ingests into parquet** after download

**Flags:**
```
--date YYYY-MM-DD     Specific date (default: today IST)
--tier 1              Download only Tier 1 (critical) files
--no-ingest           Skip parquet conversion
```

**Backfill a date range:**
```bash
python -m apps.nse_daily backfill --from 2026-01-01 --to 2026-02-06
```

**Check status:**
```bash
python -m apps.nse_daily status                     # overview of all dates
python -m apps.nse_daily status --date 2026-02-06   # detailed per-file status
```

---

## 2. NSE Daily File Types (23)

### Tier 1 — Critical (9 files)

| File | Parquet Category | Description |
|---|---|---|
| `fo_bhavcopy.csv.zip` | `nse_fo_bhavcopy` | F&O derivatives OHLCV, OI, settlement |
| `cm_bhavcopy.csv.zip` | `nse_cm_bhavcopy` | Cash market equities bhavcopy |
| `participant_oi.csv` | `nse_participant_oi` | Client/DII/FII/Pro open interest |
| `participant_vol.csv` | `nse_participant_vol` | Client/DII/FII/Pro trading volume |
| `settlement_prices.csv` | `nse_settlement_prices` | MTM settlement prices |
| `volatility.csv` | `nse_volatility` | Daily + annualized volatility (216 symbols) |
| `contract_delta.csv` | `nse_contract_delta` | Delta factors for all contracts (from 2025-09-26) |
| `index_close.csv` | `nse_index_close` | Closing values for 146 indices |
| `delivery_bhavcopy.csv` | `nse_delivery` | Delivery quantity and percentage |

### Tier 2 — Supplementary (14 files)

| File | Parquet Category | Description |
|---|---|---|
| `fii_stats.xls` | `nse_fii_stats` | FII buy/sell/OI by index breakdown |
| `market_activity.zip` | `nse_market_activity` | F&O volume summary (6 rows) |
| `nse_oi.zip` | `nse_oi` | NSE Clearing open interest |
| `combined_oi.zip` | `nse_combined_oi` | Cross-exchange combined OI |
| `combined_oi_deleq.csv` | `nse_combined_oi_deleq` | Delta-equivalent combined OI |
| `security_ban.csv` | `nse_security_ban` | F&O ban list |
| `fo_contract.csv.gz` | `nse_fo_contract` | Full F&O contract master (100K+ rows, 120+ cols) |
| `bulk_deals.csv` | `nse_bulk_deals` | Bulk deals for the day |
| `block_deals.csv` | `nse_block_deals` | Block deals (often empty) |
| `mto.dat` | `nse_mto` | Security-wise delivery position |
| `margin_data.dat` | `nse_margin_data` | VaR margin data |
| `fo_mktlots.csv` | `nse_fo_mktlots` | Market lot sizes by expiry |
| `52wk_highlow.csv` | `nse_52wk_highlow` | 52-week high/low (adjusted) |
| `top_gainers.json` | `nse_top_gainers` | Top 20 gainers across 7 index categories (today-only snapshot from www.nseindia.com API; losers can be derived from `nse_cm_bhavcopy`) |

---

## 3. Parquet Store Layout

```
qlxr_common/qlxr_data/market/
  nfo_1min/date=2024-10-29/data.parquet      # Telegram: F&O 1-min bars
  bfo_1min/date=2024-10-29/data.parquet      # Telegram: BSE 1-min bars
  ticks/date=2024-10-29/data.parquet         # Telegram: tick data
  instruments/date=2024-10-29/data.parquet   # Telegram: instrument master
  nse_fo_bhavcopy/date=2025-08-06/data.parquet  # NSE daily: F&O bhavcopy
  nse_index_close/date=2025-08-06/data.parquet   # NSE daily: index close
  ...                                        # 23 nse_* categories total
```

- Format: hive-partitioned, zstd compressed
- DuckDB reads parquet files directly (no data duplication)
- Predicate pushdown on `date` partition for fast filtered queries

### Current Coverage

| Source | Categories | Date Range | Days | Disk Size |
|---|---|---|---|---|
| Telegram | 4 | 2024-10-29 to 2026-02-06 | 311-317 | 10.6 GB |
| NSE daily | 23 | 2025-08-06 to 2026-02-06 | 89-133 | 0.6 GB |
| **Total** | **27** | | | **11.2 GB** |

---

## 4. Querying with MarketDataStore

```python
from qlx.data import MarketDataStore

store = MarketDataStore()

# Summary of all categories
store.summary()

# Raw SQL on any category
store.sql("SELECT count(*) FROM nse_fo_bhavcopy WHERE date='2026-02-06'")
store.sql("SELECT * FROM nse_participant_oi WHERE date='2026-02-06'")
store.sql("SELECT symbol, underlying_ann_vol FROM nse_volatility WHERE date='2026-02-06' AND symbol='NIFTY'")
store.sql("SELECT symbol, lot_size FROM nse_fo_mktlots WHERE date='2026-02-06' AND symbol='NIFTY'")

# Telegram data queries
chain = store.get_option_chain("NIFTY", date(2026, 2, 6), "2026-02-06")
bars  = store.get_symbol_bars("NIFTY26FEB23000CE", date(2026, 2, 6))
ticks = store.get_ticks(256265, date(2026, 2, 6))

# Ingestion hooks (called automatically by downloaders, but available manually)
store.ingest_day(date(2026, 2, 6))       # Telegram data
store.ingest_nse_day(date(2026, 2, 6))   # NSE daily data
```

---

## 5. Manual / Bulk Conversion

If you need to convert data without running the downloaders:

```bash
cd qlxr/QuantLaxmi

# Telegram data
python -m qlx.data.convert                          # convert all unconverted dates
python -m qlx.data.convert --dates 2026-02-06       # specific date
python -m qlx.data.convert --force                  # re-convert everything
python -m qlx.data.convert --dry-run                # show what would be done

# NSE daily data
python -m qlx.data.nse_convert                      # convert all unconverted dates
python -m qlx.data.nse_convert --dates 2026-02-06   # specific date
python -m qlx.data.nse_convert --force              # re-convert everything
python -m qlx.data.nse_convert --dry-run            # show what would be done
```

---

## 6. Troubleshooting

### Telegram downloader hangs or is slow
The default scan is 50 messages (seconds). If you accidentally ran `--full`, it scans 50,000 messages and takes minutes. Ctrl+C and rerun without `--full`.

### NSE files return 403/404
NSE archives use Akamai CDN. A 403 means the session needs refreshing (the collector does this automatically with retries). A 404 means the file isn't available yet — NSE publishes files between 6 PM and 8 PM IST.

### Missing parquet for a date
Run the converter manually:
```bash
python -m qlx.data.nse_convert --dates 2026-02-06
python -m qlx.data.convert --dates 2026-02-06
```

### DuckDB schema mismatch errors
NSE occasionally changes column names across dates. The store uses `union_by_name=true` to handle this. If a new column appears, existing dates will have NULL for that column.

### Holiday dates
Holidays have only 3-5 files (snapshot files like bulk_deals, fo_mktlots). Trading-day files are silently skipped. This is normal.

### contract_delta.csv missing
Only available from 2025-09-26 onwards. Marked as optional — the converter skips it for earlier dates.

---

## 7. File Locations

| Component | Path |
|---|---|
| Telegram downloader | `qlxr/qlxr_common/qlxr_telegram/telegram_downloader.py` |
| Telegram session | `qlxr/qlxr_common/qlxr_telegram/brahmastra_session.session` |
| Telegram credentials | `qlxr/qlxr_common/qlxr_vault/.env` |
| Telegram raw downloads | `qlxr/qlxr_common/qlxr_data/telegram_source_files/india_tick_data/` |
| NSE daily collector | `qlxr/QuantLaxmi/apps/nse_daily/` |
| NSE daily raw downloads | `qlxr/QuantLaxmi/data/nse/daily/YYYY-MM-DD/` |
| NSE file definitions | `qlxr/QuantLaxmi/apps/nse_daily/files.py` |
| Telegram converter | `qlxr/QuantLaxmi/qlx/data/convert.py` |
| NSE converter | `qlxr/QuantLaxmi/qlx/data/nse_convert.py` |
| MarketDataStore | `qlxr/QuantLaxmi/qlx/data/store.py` |
| Parquet store | `qlxr/qlxr_common/qlxr_data/market/` |
