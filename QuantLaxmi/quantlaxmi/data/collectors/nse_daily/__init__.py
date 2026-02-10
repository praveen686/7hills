"""NSE Daily Data Collector.

Downloads ~15 archive files daily from nsearchives.nseindia.com into
date-partitioned local storage at data/nse/daily/{YYYY-MM-DD}/.

Files include FO/CM bhavcopy, participant OI/volume, settlement prices,
volatility, contract delta, index close, FII stats, market activity,
OI data, security ban list, and contract master.
"""
