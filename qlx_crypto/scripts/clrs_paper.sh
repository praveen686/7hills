#!/bin/bash
# Launch CLRS paper trader with tick collection.
# Usage: ./scripts/clrs_paper.sh [--reset]
#
# Runs in background with nohup. Logs to data/clrs_output.log.
# Use scripts/watchdog_clrs.sh via cron for auto-restart.

set -euo pipefail
cd /home/ubuntu/Desktop/7hills/qlx_crypto

VENV="/home/ubuntu/Desktop/7hills/qlx_india/.venv/bin/python3"
LOG="data/clrs_output.log"
mkdir -p data

EXTRA_ARGS=""
if [[ "${1:-}" == "--reset" ]]; then
    EXTRA_ARGS="--reset"
    echo "$(date -u): Starting CLRS with --reset" >> data/watchdog_clrs.log
fi

nohup "$VENV" -m apps.crypto_flow \
    --with-ticks \
    --scan-interval 300 \
    --carry-entry 20 \
    --carry-exit 3 \
    --max-carry 10 \
    --cost-bps 8 \
    --min-volume 10 \
    $EXTRA_ARGS \
    >> "$LOG" 2>&1 &

echo "$(date -u): CLRS started with PID $!" >> data/watchdog_clrs.log
echo "CLRS paper trader started (PID $!)"
echo "Logs: $LOG"
