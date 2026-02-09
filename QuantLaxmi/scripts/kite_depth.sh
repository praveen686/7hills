#!/bin/bash
# Launch Kite 5-level depth collector.
# Usage: ./scripts/kite_depth.sh
#
# Runs in background with nohup. Logs to logs/kite_depth.log.
# Use scripts/watchdog_kite_depth.sh via cron for auto-restart.

set -euo pipefail
cd /home/ubuntu/Desktop/7hills/QuantLaxmi

VENV="/home/ubuntu/Desktop/7hills/QuantLaxmi/venv/bin/python3"
LOG="logs/kite_depth.log"
mkdir -p logs

nohup "$VENV" -m data.data.collectors.kite_depth collect \
    >> "$LOG" 2>&1 &

echo "$(date -u): Kite depth started with PID $!" >> logs/watchdog_kite_depth.log
echo "Kite depth collector started (PID $!)"
echo "Logs: $LOG"
