"""Kite 5-level depth collector for NSE futures.

Streams NIFTY + BANKNIFTY futures depth (5 bid/ask levels) from
Zerodha KiteTicker in MODE_FULL at ~1 tick/sec per instrument.

Stores date-partitioned parquet files at:
  data/zerodha/5level/{YYYY-MM-DD}/{SYMBOL}_FUT.parquet
"""
