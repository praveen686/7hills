"""Shared fixtures for QLX tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlx.core.types import OHLCV


@pytest.fixture
def sample_ohlcv() -> OHLCV:
    """Generate 1000 bars of synthetic OHLCV data with realistic properties."""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")

    # Geometric brownian motion for price
    log_returns = np.random.normal(0.0001, 0.01, n)
    close = 30000 * np.exp(np.cumsum(log_returns))

    # Generate OHLC with reasonable relationships
    noise = np.abs(np.random.normal(0, 0.003, n))
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close * (1 + np.random.normal(0, 0.002, n))
    volume = np.random.lognormal(10, 1, n)

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    return OHLCV(df)


@pytest.fixture
def small_ohlcv() -> OHLCV:
    """Tiny 50-bar dataset for fast unit tests."""
    np.random.seed(123)
    n = 50
    dates = pd.date_range("2023-06-01", periods=n, freq="h", tz="UTC")
    close = 30000 + np.cumsum(np.random.normal(0, 50, n))
    high = close + np.abs(np.random.normal(0, 20, n))
    low = close - np.abs(np.random.normal(0, 20, n))

    df = pd.DataFrame(
        {
            "Open": close + np.random.normal(0, 10, n),
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.random.lognormal(8, 0.5, n),
        },
        index=dates,
    )
    return OHLCV(df)
