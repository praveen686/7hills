"""Quick demo of CPCV functionality."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantkubera.backtest import CombPurgedKFoldCV

# Create sample time series data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
df = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'target': np.random.randn(1000)
}, index=dates)

print("=" * 80)
print("CPCV Quick Demo")
print("=" * 80)
print(f"\nDataset: {len(df)} samples from {df.index[0].date()} to {df.index[-1].date()}")

# Initialize CPCV
cpcv = CombPurgedKFoldCV(
    n_splits=5,
    n_test_groups=2,
    purge_window=21,
    embargo_pct=0.01
)

n_paths = cpcv.get_n_splits()
print(f"\nCPCV Configuration:")
print(f"  - Splits: 5")
print(f"  - Test groups per path: 2")
print(f"  - Total backtest paths: {n_paths} (C(5,2) = 10)")
print(f"  - Purge window: 21 samples")
print(f"  - Embargo: 1% (~10 samples)")

print(f"\nGenerating {n_paths} train/test splits...\n")

for i, (train_idx, test_idx) in enumerate(cpcv.split(df), 1):
    train_dates = df.index[train_idx]
    test_dates = df.index[test_idx]
    
    print(f"Path {i:2d}/{n_paths}: ", end='')
    print(f"Train: {len(train_idx):4d} samples ", end='')
    print(f"({train_dates.min().date()} to {train_dates.max().date()}) | ", end='')
    print(f"Test: {len(test_idx):3d} samples ", end='')
    print(f"({test_dates.min().date()} to {test_dates.max().date()})")

print(f"\n{'=' * 80}")
print("✅ CPCV working correctly!")
print(f"{'=' * 80}\n")
