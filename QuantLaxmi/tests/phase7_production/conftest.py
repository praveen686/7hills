"""Shared fixtures for Phase 7 production hardening tests."""

from __future__ import annotations

import tempfile
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.state import BrahmastraState, Position, ClosedTrade
from engine.live.event_log import EventLogWriter
from core.events.envelope import EventEnvelope


# -----------------------------------------------------------------------
# Option chain fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def mock_chain_df() -> pd.DataFrame:
    """22-strike option chain with healthy OI (passes all DQ checks)."""
    np.random.seed(42)
    strikes = np.arange(21000, 22100, 50)  # 22 strikes
    rows = []
    for s in strikes:
        for otype in ("CE", "PE"):
            rows.append({
                "strike": float(s),
                "option_type": otype,
                "expiry": "2025-12-25",
                "ltp": float(np.random.uniform(50, 500)),
                "oi": int(np.random.randint(200, 5000)),
                "volume": int(np.random.randint(100, 2000)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sparse_chain_df() -> pd.DataFrame:
    """3-strike chain — below minimum threshold."""
    rows = []
    for s in [21500, 21550, 21600]:
        for otype in ("CE", "PE"):
            rows.append({
                "strike": float(s),
                "option_type": otype,
                "expiry": "2025-12-25",
                "ltp": 150.0,
                "oi": 500,
                "volume": 200,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def zero_oi_chain_df() -> pd.DataFrame:
    """10-strike chain with zero OI everywhere."""
    rows = []
    for s in np.arange(21000, 21500, 50):
        for otype in ("CE", "PE"):
            rows.append({
                "strike": float(s),
                "option_type": otype,
                "expiry": "2025-12-25",
                "ltp": 100.0,
                "oi": 0,
                "volume": 0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def low_oi_chain_df() -> pd.DataFrame:
    """10-strike chain with all OI below threshold (< 100)."""
    rows = []
    for s in np.arange(21000, 21500, 50):
        for otype in ("CE", "PE"):
            rows.append({
                "strike": float(s),
                "option_type": otype,
                "expiry": "2025-12-25",
                "ltp": 100.0,
                "oi": int(np.random.randint(10, 90)),
                "volume": 50,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def high_oi_chain_df() -> pd.DataFrame:
    """10-strike chain with high OI everywhere."""
    rows = []
    for s in np.arange(21000, 21500, 50):
        for otype in ("CE", "PE"):
            rows.append({
                "strike": float(s),
                "option_type": otype,
                "expiry": "2025-12-25",
                "ltp": 100.0,
                "oi": 5000,
                "volume": 2000,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def mixed_oi_chain_df() -> pd.DataFrame:
    """10-strike chain with mixed OI (some above, some below threshold)."""
    rows = []
    for i, s in enumerate(np.arange(21000, 21500, 50)):
        for otype in ("CE", "PE"):
            # First 3 strikes below threshold, rest above
            oi = 50 if i < 3 else 500
            rows.append({
                "strike": float(s),
                "option_type": otype,
                "expiry": "2025-12-25",
                "ltp": 100.0,
                "oi": oi,
                "volume": 200,
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Event log fixture
# -----------------------------------------------------------------------

@pytest.fixture
def event_log(tmp_path):
    """Fresh EventLogWriter in a temp directory."""
    log = EventLogWriter(
        base_dir=tmp_path / "events",
        run_id="test-phase7",
        fsync_policy="none",
    )
    yield log
    log.close()


# -----------------------------------------------------------------------
# Price fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def seeded_prices_500() -> np.ndarray:
    """500 bars of GBM price data with seed=42."""
    np.random.seed(42)
    log_ret = np.random.normal(0.0001, 0.01, 500)
    return 21500 * np.exp(np.cumsum(log_ret))


# -----------------------------------------------------------------------
# State fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def fresh_state() -> BrahmastraState:
    """Fresh BrahmastraState with default equity."""
    return BrahmastraState()


@pytest.fixture
def state_with_positions() -> BrahmastraState:
    """State with 3 active positions of varying age."""
    state = BrahmastraState()
    # Fresh position (today)
    state.open_position(Position(
        strategy_id="s5_hawkes", symbol="NIFTY", direction="long",
        weight=0.10, instrument_type="FUT",
        entry_date="2025-12-01", entry_price=21500.0,
    ))
    # Old position (60 days ago)
    state.open_position(Position(
        strategy_id="s4_iv_mr", symbol="BANKNIFTY", direction="short",
        weight=0.08, instrument_type="FUT",
        entry_date="2025-10-01", entry_price=48000.0,
    ))
    # Medium position (25 days ago)
    state.open_position(Position(
        strategy_id="s1_vrp_options", symbol="NIFTY", direction="long",
        weight=0.05, instrument_type="CE",
        entry_date="2025-11-05", entry_price=200.0,
    ))
    return state
