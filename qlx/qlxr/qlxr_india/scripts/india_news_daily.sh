#!/bin/bash
# Daily India news sentiment paper trader scan.
# Runs after NSE market close (3:30 PM IST = 10:00 UTC).
# Prices fetched via Zerodha Kite API (last traded price).
#
# Add to crontab:
#   30 10 * * 1-5 /home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_india/scripts/india_news_daily.sh
#
# This fetches recent India business RSS headlines, scores them with FinBERT,
# and manages paper trading positions in F&O stock futures.

set -euo pipefail

PROJ="/home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_india"
LOG="$PROJ/data/india_news_daily.log"

cd "$PROJ"

echo "$(date -u): Starting India news paper daily scan" >> "$LOG"

# Run paper trading scan (uses Kite API for prices)
python3 -m apps.india_news paper --once >> "$LOG" 2>&1

echo "$(date -u): India news paper daily scan complete" >> "$LOG"
echo "---" >> "$LOG"
