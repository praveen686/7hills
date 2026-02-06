#!/bin/bash
# Launch Kite 5-level depth collector.
# Usage: ./scripts/kite_depth.sh
#
# Runs in background with nohup. Logs to data/kite_depth.log.
# Use scripts/watchdog_kite_depth.sh via cron for auto-restart.

set -euo pipefail
cd /home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_india

VENV="/home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_common/qlxr_env/bin/python3"
LOG="data/kite_depth.log"
mkdir -p data

nohup "$VENV" -m apps.kite_depth collect \
    >> "$LOG" 2>&1 &

echo "$(date -u): Kite depth started with PID $!" >> data/watchdog_kite_depth.log
echo "Kite depth collector started (PID $!)"
echo "Logs: $LOG"
