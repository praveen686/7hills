#!/bin/bash
# Daily multi-index IV mean-reversion paper trader scan.
# Trades: NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY
# Runs after NSE market close (3:30 PM IST = 10:00 UTC).
#
# Add to crontab:
#   15 10 * * 1-5 /home/ubuntu/Desktop/7hills/qlx_india/scripts/iv_paper_daily.sh
#
# This fetches today's F&O bhavcopy, calibrates SANOS for each index,
# computes ATM IV, checks entry/exit signals, and manages paper positions.

set -euo pipefail

PROJ="/home/ubuntu/Desktop/7hills/qlx_india"
LOG="$PROJ/data/iv_paper_daily.log"

cd "$PROJ"

echo "$(date -u): Starting IV paper daily scan" >> "$LOG"

# First, ensure today's bhavcopy is downloaded
python3 -m apps.india_scanner backfill \
    --start "$(date +%Y-%m-%d)" \
    --category fno \
    >> "$LOG" 2>&1 || true

# Run paper trading scan for today
python3 -m apps.india_fno paper --once >> "$LOG" 2>&1

echo "$(date -u): IV paper daily scan complete" >> "$LOG"
echo "---" >> "$LOG"
