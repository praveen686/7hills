#!/bin/bash
# Download NSE daily archive files after market close.
# Usage: ./scripts/nse_daily.sh
#
# Crontab (Mon-Fri at 18:00 IST = 12:30 UTC):
#   30 12 * * 1-5 /home/ubuntu/Desktop/7hills/QuantLaxmi/scripts/nse_daily.sh

set -euo pipefail
cd /home/ubuntu/Desktop/7hills/QuantLaxmi

VENV="/home/ubuntu/Desktop/7hills/QuantLaxmi/venv/bin/python3"
LOG="logs/nse_daily.log"
mkdir -p logs

echo "$(date -u '+%Y-%m-%d %H:%M:%S'): Starting NSE daily collection" >> "$LOG"

"$VENV" -m collectors.nse_daily collect >> "$LOG" 2>&1

echo "$(date -u '+%Y-%m-%d %H:%M:%S'): NSE daily collection finished" >> "$LOG"
