"""Canonical data path defaults for QuantLaxmi.

All data modules import from here to avoid scattering hardcoded paths.

Environment variables (override via .env or shell export):
    QUANTLAXMI_DATA_ROOT      Root of hive-partitioned market data.
                              Default: <project>/common/data
    QUANTLAXMI_NSE_DAILY_DIR  Raw NSE daily ZIP/CSV downloads.
                              Default: <QUANTLAXMI_DATA_ROOT>/nse/daily
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root: QuantLaxmi/  (quantlaxmi/data/_paths.py → parents[2])
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(
    os.environ.get("QUANTLAXMI_DATA_ROOT", str(_PROJECT_ROOT / "common" / "data"))
)

MARKET_DIR = DATA_ROOT / "market"

NSE_DAILY_DIR = Path(
    os.environ.get("QUANTLAXMI_NSE_DAILY_DIR", str(DATA_ROOT / "nse" / "daily"))
)

KITE_1MIN_DIR = DATA_ROOT / "kite_1min"
BINANCE_DIR = DATA_ROOT / "binance"
TICK_DIR = MARKET_DIR / "ticks"
MACRO_DIR = DATA_ROOT / "macro"

# Strategy and portfolio state
STRATEGY_STATE = _PROJECT_ROOT / "data" / "strategy_state"
PORTFOLIO_STATE = _PROJECT_ROOT / "data" / "state" / "portfolio.json"

# Collector output directories
CHAIN_SNAPSHOTS = _PROJECT_ROOT / "data" / "india" / "chain_snapshots"
CRYPTO_TICK_DATA = _PROJECT_ROOT / "data" / "ticks"

# Engine directories
EVENTS_DIR = _PROJECT_ROOT / "data" / "events"
SESSIONS_DIR = _PROJECT_ROOT / "data" / "sessions"
BACKTEST_RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "backtest_results"

# Zerodha / India collector paths
ZERODHA_SESSION_CACHE = _PROJECT_ROOT / "data" / "state" / "zerodha_session.json"
KITE_DEPTH_DIR = _PROJECT_ROOT / "data" / "zerodha" / "5level"
HEADLINE_ARCHIVE_DIR = _PROJECT_ROOT / "data" / "india" / "headlines"
NEWS_STATE_FILE = _PROJECT_ROOT / "data" / "state" / "india_news.json"
