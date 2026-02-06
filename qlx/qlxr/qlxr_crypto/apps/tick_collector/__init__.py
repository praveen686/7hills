"""Tick-level data collection and storage for Binance Futures.

Collects aggTrade + bookTicker streams via JSON WebSocket (fstream.binance.com)
and stores them in date-partitioned Parquet files.

Architecture:
  - TickStore: Buffered Parquet writer with per-symbol, per-day files.
    Uses pyarrow.ParquetWriter with multiple row groups for true append.
  - TickCollector: WebSocket manager. Connects via ResilientWs, parses
    aggTrade + bookTicker JSON, feeds TickStore and LiveRegimeTracker.
  - LiveRegimeTracker: Real-time VPIN/OFI/Hawkes from actual tick data.
    Exposes RegimeSnapshot per symbol for the paper trader.

Storage layout:
  data/ticks/{YYYY-MM-DD}/{SYMBOL}_trades.parquet
  data/ticks/{YYYY-MM-DD}/{SYMBOL}_book.parquet

Usage:
  python -m apps.tick_collector collect          # run collector
  python -m apps.tick_collector status           # show collection stats
  python -m apps.tick_collector read BTCUSDT     # read stored ticks
"""
