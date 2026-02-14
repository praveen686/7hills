"""
Backtesting engine with CPCV support for QuantKubera.

This script demonstrates robust backtesting using Combinatorial Purged K-Fold CV
to validate the TMT model's performance across multiple temporal splits.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantkubera.backtest.engine import BacktestEngine
from config.train_config import TRAIN_CONFIG, DATA_CONFIG, MODEL_CONFIG


def run_cpcv_backtest(
    features_df: pd.DataFrame,
    n_splits: int = 5,
    n_test_groups: int = 2,
    purge_window: int = 21,
    embargo_pct: float = 0.01,
    verbose: bool = True
):
    """Run CPCV backtest on features using BacktestEngine."""
    engine = BacktestEngine(
        model_config=MODEL_CONFIG,
        data_config=DATA_CONFIG,
        train_config=TRAIN_CONFIG
    )
    
    results_df = engine.run_cpcv(
        features_df,
        n_splits=n_splits,
        n_test_groups=n_test_groups,
        purge_window=purge_window,
        embargo_pct=embargo_pct,
        verbose=verbose
    )
    
    if verbose:
        print(f"\n{'='*80}")
        print("CPCV Backtest Results Summary")
        print(f"{'='*80}")
        print(f"\nSharpe Ratio Distribution:")
        print(f"  Mean:   {results_df['sharpe_ratio'].mean():.3f}")
        print(f"  Median: {results_df['sharpe_ratio'].median():.3f}")
        print(f"  Std:    {results_df['sharpe_ratio'].std():.3f}")
        print(f"  Min:    {results_df['sharpe_ratio'].min():.3f}")
        print(f"  Max:    {results_df['sharpe_ratio'].max():.3f}")
        print(f"\nTotal Result Summary:")
        print(results_df[['fold_id', 'sharpe_ratio', 'total_return', 'test_samples']])
        print(f"{'='*80}\n")
    
    return {
        'results_df': results_df,
        'mean_sharpe': results_df['sharpe_ratio'].mean(),
        'std_sharpe': results_df['sharpe_ratio'].std()
    }


if __name__ == '__main__':
    print("="*80)
    print("QuantKubera CPCV Backtest")
    print("="*80)
    
    # Load and prepare data
    print("\n[1/3] Loading data...")
    df = load_processed_data('data/processed/NIFTY_processed.csv')
    
    # Build features
    print("[2/3] Building features...")
    engineer = FeatureEngineer()
    features_df = engineer.process_ticker(df)
    features_df = engineer.add_volatility(features_df, window=20)
    features_df = engineer.add_volatility(features_df, window=60)
    
    # Try to add CPD features
    try:
        features_df = engineer.add_cpd_features(features_df, ticker='NIFTY', lookback=21)
    except Exception as e:
        print(f"  Warning: Could not load CPD features: {e}")
    
    features_df = features_df.dropna()
    print(f"  Features shape: {features_df.shape}")
    
    # Run CPCV backtest
    print("\n[3/3] Running CPCV backtest...")
    results = run_cpcv_backtest(
        features_df=features_df,
        n_splits=5,
        n_test_groups=2,
        purge_window=21,  # CPD lookback
        embargo_pct=0.01,
        verbose=True
    )
    
    # Save results
    results['results_df'].to_csv('backtest_results_cpcv.csv', index=False)
    print(f"\n✅ Results saved to: backtest_results_cpcv.csv")
