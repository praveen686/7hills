# NSE Daily Data Collector

Daily automated download of ~15 archive files from `nsearchives.nseindia.com` into date-partitioned local storage. Total ~12 MB/day.

## Location

```
qlxr_india/apps/nse_daily/
    __init__.py              # Package docstring
    __main__.py              # CLI: collect, backfill, status
    collector.py             # NSEDailyCollector — download logic + session management
    files.py                 # File definitions: URL patterns, date formats, tiers

qlxr_india/scripts/nse_daily.sh   # cron launcher (18:00 IST daily)
qlxr_india/tests/test_nse_daily.py
```

## Output Structure

```
data/nse/daily/{YYYY-MM-DD}/
    fo_bhavcopy.csv.zip          # UDiFF FO Bhavcopy (45K rows)
    cm_bhavcopy.csv.zip          # UDiFF CM Bhavcopy
    participant_oi.csv           # Client/DII/FII/Pro OI
    participant_vol.csv          # Client/DII/FII/Pro volume
    settlement_prices.csv        # MTM settlement prices
    volatility.csv               # Daily + annualized vol (217 symbols)
    contract_delta.csv           # Delta factors (154K rows)
    index_close.csv              # 146 indices + VIX
    fii_stats.xls                # FII buy/sell breakdown
    market_activity.zip          # 11 files: summary + OHLCV by instrument
    nse_oi.zip                   # NCL OI
    combined_oi.zip              # Cross-exchange OI
    combined_oi_deleq.csv        # Delta-equivalent OI
    security_ban.csv             # Banned securities
    fo_contract.csv.gz           # Contract master (100K+ contracts)
```

## Files Collected (15 files)

### Tier 1 — CRITICAL (8 files)

| Local Name | Archive URL Pattern | Date Fmt | Size |
|---|---|---|---|
| `fo_bhavcopy.csv.zip` | `content/fo/BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip` | YYYYMMDD | ~1.3 MB |
| `cm_bhavcopy.csv.zip` | `content/cm/BhavCopy_NSE_CM_0_0_0_{YYYYMMDD}_F_0000.csv.zip` | YYYYMMDD | ~182 KB |
| `participant_oi.csv` | `content/nsccl/fao_participant_oi_{DDMMYYYY}.csv` | DDMMYYYY | ~1 KB |
| `participant_vol.csv` | `content/nsccl/fao_participant_vol_{DDMMYYYY}.csv` | DDMMYYYY | ~1 KB |
| `settlement_prices.csv` | `archives/nsccl/sett/FOSett_prce_{DDMMYYYY}.csv` | DDMMYYYY | ~66 KB |
| `volatility.csv` | `archives/nsccl/volt/FOVOLT_{DDMMYYYY}.csv` | DDMMYYYY | ~38 KB |
| `contract_delta.csv` | `content/nsccl/Contract_Delta_{DDMMYYYY}.csv` | DDMMYYYY | ~7.5 MB |
| `index_close.csv` | `content/indices/ind_close_all_{DDMMYYYY}.csv` | DDMMYYYY | ~7 KB |

### Tier 2 — HIGH (7 files)

| Local Name | Archive URL Pattern | Date Fmt | Size |
|---|---|---|---|
| `fii_stats.xls` | `content/fo/fii_stats_{DD-Mon-YYYY}.xls` | DD-Mon-YYYY | ~9 KB |
| `market_activity.zip` | `archives/fo/mkt/fo{DDMMYYYY}.zip` | DDMMYYYY | ~576 KB |
| `nse_oi.zip` | `archives/nsccl/mwpl/ncloi_{DDMMYYYY}.zip` | DDMMYYYY | ~18 KB |
| `combined_oi.zip` | `archives/nsccl/mwpl/combineoi_{DDMMYYYY}.zip` | DDMMYYYY | ~20 KB |
| `combined_oi_deleq.csv` | `archives/nsccl/mwpl/combineoi_deleq_{DDMMYYYY}.csv` | DDMMYYYY | ~17 KB |
| `security_ban.csv` | `archives/fo/sec_ban/fo_secban_{DDMMYYYY}.csv` | DDMMYYYY | <1 KB |
| `fo_contract.csv.gz` | `content/fo/NSE_FO_contract_{DDMMYYYY}.csv.gz` | DDMMYYYY | ~2 MB |

## Usage

```bash
cd qlxr_india

# Collect today's files
python -m apps.nse_daily collect

# Collect a specific date
python -m apps.nse_daily collect --date 2026-02-05

# Tier 1 only
python -m apps.nse_daily collect --tier 1

# Backfill a range
python -m apps.nse_daily backfill --from 2026-01-01 --to 2026-02-05

# Show status
python -m apps.nse_daily status
python -m apps.nse_daily status --date 2026-02-05
```

## Cron Setup

```bash
# Add to crontab (runs Mon-Fri at 18:00 IST = 12:30 UTC)
crontab -e
30 12 * * 1-5 /home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_india/scripts/nse_daily.sh
```

## Session Management

NSE requires browser-like sessions with cookies. The collector:
1. GETs `https://www.nseindia.com/` to obtain session cookies
2. Uses proper User-Agent, Accept, Referer headers
3. Auto-refreshes session on 403/timeout (sessions expire ~5 min)
4. Retries 3x with exponential backoff (2s, 4s, 8s)

## Design Decisions

- **Idempotent**: skips files already downloaded (exists + size > 0)
- **Weekend-aware**: skips Sat/Sun automatically
- **Rate-limited backfill**: 1s pause between dates
- **Tier filtering**: can download only critical files (tier 1) via `--tier 1`
- **No dependencies beyond requests**: uses stdlib + requests only
