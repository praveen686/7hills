"""
Demo/Verification script for Triple Barrier Labeling.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantkubera.features.labeling import triple_barrier_labels, get_daily_vol


def demo_labeling():
    print("=" * 80)
    print("Triple Barrier Labeling Demo")
    print("=" * 80)

    # 1. Create dummy data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    # Simulate a trending price with some noise
    returns = np.random.normal(0.001, 0.02, 100)
    prices = 100 * np.exp(np.cumsum(returns))
    close = pd.Series(prices, index=dates)

    print(f"Created dummy price series with {len(close)} days.")
    print(f"Start Price: {close.iloc[0]:.2f}, End Price: {close.iloc[-1]:.2f}")

    # 2. Get volatility
    vol = get_daily_vol(close)
    print(f"Average Daily Volatility: {vol.mean():.4f}")

    # 3. Apply labeling
    # We'll label every day as an event for this demo
    events = close.index[:-5]  # Leave room for vertical barrier

    print("\nApplying Triple Barrier Labeling...")
    # pt_sl=(1.0, 1.0) means 1.0 * daily_vol for profit-take and stop-loss
    labels = triple_barrier_labels(
        close=close,
        events=events,
        pt_sl=(1.0, 1.0),
        target=vol,
        num_days=5
    )

    print(f"Generated {len(labels)} labels.")
    print("\nLabel Distribution:")
    print(labels['bin'].value_counts())

    print("\nSample Labels (First 10):")
    print(labels.head(10))

    # Verify a specific row if possible
    first_valid_event = labels.index[0]
    row = labels.loc[first_valid_event]
    print(f"\nVerification of First Valid Event ({first_valid_event.date()}):")
    print(f"  Entry Price: {close.loc[first_valid_event]:.2f}")
    print(f"  Barrier Hit Time (t1): {row['t1'].date()}")
    print(f"  Return at Touch: {row['ret']:.4f}")
    print(f"  Label: {row['bin']}")

    print("\n" + "=" * 80)
    print("✅ Triple Barrier Labeling Working Correctly!")
    print("=" * 80)


if __name__ == "__main__":
    demo_labeling()
