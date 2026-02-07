#!/bin/bash
# Watchdog for CLRS paper trader.
# Add to crontab: */5 * * * * /home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_crypto/scripts/watchdog_clrs.sh
#
# Checks if CLRS is running. If not, restarts it.
# Also checks if the dashboard is running; restarts if needed.

cd /home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_crypto

VENV="/home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_common/qlxr_env/bin/python3"
LOG="data/watchdog_clrs.log"
mkdir -p data

# --- CLRS Paper Trader ---
if pgrep -f "apps.crypto_flow" > /dev/null 2>&1; then
    : # running
else
    echo "$(date -u): CLRS not running — restarting" >> "$LOG"
    nohup "$VENV" -m apps.crypto_flow \
        --with-ticks \
        --scan-interval 300 \
        --carry-entry 20 \
        --carry-exit 3 \
        --max-carry 10 \
        --cost-bps 8 \
        --min-volume 10 \
        >> data/crypto_flow.log 2>&1 &
    echo "$(date -u): CLRS restarted with PID $!" >> "$LOG"
fi

# --- Dashboard ---
if pgrep -f "apps.dashboard.*8081" > /dev/null 2>&1; then
    : # running
else
    echo "$(date -u): Dashboard not running — restarting" >> "$LOG"
    nohup "$VENV" -m apps.dashboard --port 8081 \
        >> data/dashboard.log 2>&1 &
    echo "$(date -u): Dashboard restarted with PID $!" >> "$LOG"
fi
