"""Option chain snapshot collector for India FnO indices.

Fetches full option chain from Zerodha Kite every N minutes and stores
as timestamped parquet files.  Captures NIFTY/BANKNIFTY/MIDCPNIFTY/FINNIFTY
spot, futures, and option quotes for GEX, dealer-flow, and IV surface analysis.

Storage layout:
  data/india/chain_snapshots/{YYYY-MM-DD}/{SYMBOL}_{HHMMSS}.parquet

Usage:
  python -m quantlaxmi.data.collectors.option_chain collect
  python -m quantlaxmi.data.collectors.option_chain status
"""
