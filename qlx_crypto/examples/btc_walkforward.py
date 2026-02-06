#!/usr/bin/env python3
"""BTC Walk-Forward Research — complete example.

This script demonstrates the full QLX pipeline on synthetic data
(replace with real BTC data for production research).

What this does vs sigma:
  - TimeGuard prevents lookahead bias at every stage
  - Features are composable, immutable, and conflict-free
  - CV gap is enforced >= target horizon (non-negotiable)
  - Transaction costs are mandatory in every backtest
  - One canonical pipeline — no three-way copy-paste divergence
  - Cyclical time encoding instead of raw year/hour integers

Usage:
    python examples/btc_walkforward.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlx.core.types import OHLCV
from qlx.features import (
    RSI,
    ATR,
    BollingerBands,
    CyclicalTime,
    HistoricalReturns,
    Momentum,
    Stochastic,
    SuperTrend,
)
from qlx.pipeline.config import PipelineConfig
from qlx.pipeline.engine import ResearchEngine
from qlx.targets import FutureReturn


def generate_synthetic_btc(n_bars: int = 5000) -> OHLCV:
    """Generate realistic-ish BTC price data for demonstration."""
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=n_bars, freq="h", tz="UTC")

    # GBM with mean-reverting volatility
    log_returns = np.random.normal(0.00005, 0.008, n_bars)
    close = 30000 * np.exp(np.cumsum(log_returns))

    noise = np.abs(np.random.normal(0, 0.003, n_bars))
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close * (1 + np.random.normal(0, 0.002, n_bars))
    volume = np.random.lognormal(10, 1, n_bars)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    return OHLCV(df)


def main():
    print("=" * 60)
    print("QLX Research Engine — BTC Walk-Forward Example")
    print("=" * 60)

    # --- 1. Load data ---
    ohlcv = generate_synthetic_btc(n_bars=5000)
    print(f"\nData: {len(ohlcv)} bars, {ohlcv.df.index[0]} to {ohlcv.df.index[-1]}")

    # --- 2. Define features (composable, zero-lookforward enforced) ---
    features = [
        RSI(window=14),
        RSI(window=28),
        BollingerBands(window=20),
        SuperTrend(period=14, multiplier=3.0),
        Stochastic(k_window=14, d_window=3),
        ATR(window=14),
        HistoricalReturns(periods=(1, 5, 10, 20, 60, 100)),
        Momentum(fast=20, slow=100, run_window=14),
        CyclicalTime(),
    ]

    # --- 3. Define target ---
    horizon = 24  # predict 24 bars (hours) ahead
    target = FutureReturn(horizon=horizon)

    # --- 4. Configure pipeline ---
    config = PipelineConfig.from_dict({
        "target": {"horizon": horizon},
        "model": {
            "name": "xgboost",
            "task": "regression",
            "params": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
        },
        "cv": {
            "method": "walk_forward",
            "window": 1000,
            "train_frac": 0.8,
            "gap": horizon,  # gap == horizon: minimum safe value
        },
        "costs": {
            "commission_bps": 4,    # Binance taker fee
            "slippage_bps": 2,      # conservative slippage estimate
        },
        "long_entry_threshold": 0.005,
        "long_exit_threshold": 0.0,
        "short_entry_threshold": -0.005,
        "short_exit_threshold": 0.0,
    })

    # --- 5. Run ---
    engine = ResearchEngine(
        ohlcv=ohlcv,
        features=features,
        target_transform=target,
        config=config,
    )

    result = engine.run(verbose=True)

    # --- 6. Report ---
    print("\n" + result.summary())

    if result.backtest and result.backtest.n_trades > 0:
        print(f"\nFirst 5 trades:")
        print(result.backtest.trades.head().to_string(index=False))

    print(f"\nFeature count: {len(result.feature_names)}")
    print(f"Top features by name: {result.feature_names[:10]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
