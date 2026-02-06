#!/bin/bash
# Watchdog for funding paper trader.
# Add to crontab: */10 * * * * /home/ubuntu/Desktop/7hills/qlx_python/scripts/watchdog_paper.sh
#
# Checks if the paper trader is running. If not, restarts it.

cd /home/ubuntu/Desktop/7hills/qlx_python

if pgrep -f "apps.funding_paper" > /dev/null 2>&1; then
    exit 0
fi

echo "$(date -u): Paper trader not running — restarting" >> data/watchdog.log
nohup python3 -m apps.funding_paper --scan-interval 300 >> data/funding_paper_output.log 2>&1 &
echo "$(date -u): Restarted with PID $!" >> data/watchdog.log
