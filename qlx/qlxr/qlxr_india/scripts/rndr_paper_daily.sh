#!/bin/bash
# Daily RNDR (Risk-Neutral Density Regime) paper trader scan.
# Trades: BANKNIFTY, MIDCPNIFTY, FINNIFTY
# Runs after NSE market close (3:30 PM IST = 10:00 UTC).
#
# Add to crontab:
#   20 10 * * 1-5 /home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_india/scripts/rndr_paper_daily.sh
#
# This fetches today's F&O bhavcopy, calibrates SANOS for each index,
# extracts risk-neutral density features, computes composite signal,
# and manages paper positions.

set -euo pipefail

PROJ="/home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_india"
LOG="$PROJ/data/rndr_paper_daily.log"

cd "$PROJ"

echo "$(date -u): Starting RNDR paper daily scan" >> "$LOG"

# First, ensure today's bhavcopy is downloaded
python3 -m apps.india_scanner backfill \
    --start "$(date +%Y-%m-%d)" \
    --category fno \
    >> "$LOG" 2>&1 || true

# Run RNDR paper trading scan for today
python3 -m apps.india_fno.rndr paper --once >> "$LOG" 2>&1

echo "$(date -u): RNDR paper daily scan complete" >> "$LOG"
echo "---" >> "$LOG"
