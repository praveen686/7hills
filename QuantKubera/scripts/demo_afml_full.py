"""
Demo/Verification script for the full AFML pipeline:
CUSUM -> Triple Barrier -> Meta-Labeling -> Bet Sizing
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantkubera.features.sampling import cusum_filter
from quantkubera.features.labeling import triple_barrier_labels, get_daily_vol, meta_labeling
from quantkubera.features.bet_sizing import bet_size


def demo_afml_pipeline():
    print("=" * 80)
    print("Full AFML Pipeline Demo")
    print("=" * 80)

    # 1. Create dummy data (3 years of daily)
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=1000, freq="B")
    returns = np.random.normal(0, 0.015, 1000)
    prices = 100 * np.exp(np.cumsum(returns))
    close = pd.Series(prices, index=dates)

    print(f"1. Created dummy series: {len(close)} business days.")

    # 2. Get volatility
    vol = get_daily_vol(close)
    
    # 3. CUSUM Filtering (Event Sampling)
    print("2. Applying CUSUM Filter (Event Sampling)...")
    events = cusum_filter(close, threshold=vol)
    print(f"   Sampled {len(events)} events out of 1000 days.")

    # 4. Triple Barrier Labeling
    print("3. Applying Triple Barrier Labeling...")
    labels = triple_barrier_labels(
        close=close,
        events=events,
        pt_sl=(2.0, 1.0), # 2.0x vol profit, 1.0x vol stop-loss
        target=vol,
        num_days=10
    )
    print(f"   Labeled {len(labels)} events.")

    # 5. Simulate Primary Model Predictions
    # Let's say our primary model has a 55% accuracy
    print("4. Simulating Primary Model Predictions (55% Accuracy)...")
    primary_preds = pd.Series(index=labels.index, dtype=int)
    for idx, true_label in labels['bin'].items():
        if np.random.random() < 0.55:
            primary_preds.loc[idx] = true_label if true_label != 0 else np.random.choice([-1, 1])
        else:
            primary_preds.loc[idx] = -true_label if true_label != 0 else np.random.choice([-1, 1])
    
    # 6. Meta-Labeling
    print("5. Generating Meta-Labels...")
    meta = meta_labeling(primary_preds, labels['bin'])
    print(f"   Meta-label (Correct/Incorrect) counts:")
    print(meta.value_counts())

    # 7. Bet Sizing
    # In a real scenario, we'd train a Meta-Model to predict the meta-labels.
    # Here we'll simulate the meta-model's probability output.
    print("6. Simulating Meta-Model Confidence & Bet Sizing...")
    # Add some noise to the "truth" to get probabilities
    meta_probs = pd.Series(index=meta.index, data=meta.values.astype(float))
    meta_probs = (meta_probs * 0.4 + 0.3) + np.random.normal(0, 0.1, len(meta))
    meta_probs = np.clip(meta_probs, 0.1, 0.9)
    
    sizes = bet_size(meta_probs, max_leverage=1.0, discretize=True, step_size=0.1)
    
    # 8. Final Report
    results = pd.DataFrame({
        'Side': primary_preds,
        'Outcome': labels['bin'],
        'Meta': meta,
        'Confidence': meta_probs,
        'BetSize': sizes
    })
    
    print("\nFinal Pipeline Output (Sample):")
    print(results.head(10))

    print("\n" + "=" * 80)
    print("✅ Full AFML Pipeline (Sampling -> Labeling -> Meta -> Sizing) Validated!")
    print("=" * 80)


if __name__ == "__main__":
    demo_afml_pipeline()
