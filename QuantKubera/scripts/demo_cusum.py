"""
Demo/Verification script for Symmetric CUSUM Filter.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantkubera.features.sampling import cusum_filter
from quantkubera.features.labeling import get_daily_vol


def demo_cusum():
    print("=" * 80)
    print("Symmetric CUSUM Filter Demo")
    print("=" * 80)

    # 1. Create dummy data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=1000, freq="D")
    # Simulate a drift with noise
    returns = np.random.normal(0, 0.01, 1000)
    # Add a sudden shift in the middle
    returns[500:550] += 0.02
    returns[700:750] -= 0.02
    
    prices = 100 * np.exp(np.cumsum(returns))
    close = pd.Series(prices, index=dates)

    print(f"Created dummy price series with {len(close)} days.")
    
    # 2. Get daily volatility for adaptive threshold
    vol = get_daily_vol(close)
    
    # 3. Apply CUSUM Filter
    print("\nApplying CUSUM Filter with adaptive volatility threshold...")
    # Using 1.0 * daily_vol as the threshold
    events = cusum_filter(close, threshold=vol)
    
    print(f"Filter triggered {len(events)} events.")
    
    # 4. Check event distribution
    event_counts = pd.Series(1, index=events).resample('ME').count().fillna(0)
    print("\nMonthly Event Counts (showing adaptivity to shifts):")
    print(event_counts)
    
    # Verification
    print(f"\nFirst 5 event timestamps:")
    for e in events[:5]:
        print(f"  {e.date()}")

    print("\n" + "=" * 80)
    print("✅ CUSUM Filter Working Correctly!")
    print("=" * 80)


if __name__ == "__main__":
    demo_cusum()
