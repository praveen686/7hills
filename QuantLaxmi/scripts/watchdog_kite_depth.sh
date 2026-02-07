#!/bin/bash
# Watchdog for Kite depth collector.
# Add to crontab: */5 3-10 * * 1-5 /home/ubuntu/Desktop/7hills/QuantLaxmi/scripts/watchdog_kite_depth.sh
#
# Runs Mon-Fri, 9:00-16:00 IST (3:30-10:30 UTC).
# Checks if depth collector is running. If not, restarts it.

cd /home/ubuntu/Desktop/7hills/QuantLaxmi

VENV="/home/ubuntu/Desktop/7hills/QuantLaxmi/common/qlxr_env/bin/python3"
LOG="data/watchdog_kite_depth.log"
mkdir -p data

if pgrep -f "apps.kite_depth collect" > /dev/null 2>&1; then
    : # running
else
    echo "$(date -u): Kite depth not running — restarting" >> "$LOG"
    nohup "$VENV" -m apps.kite_depth collect \
        >> data/kite_depth.log 2>&1 &
    echo "$(date -u): Kite depth restarted with PID $!" >> "$LOG"
fi
