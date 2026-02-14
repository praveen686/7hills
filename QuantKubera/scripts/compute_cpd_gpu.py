import os
import sys
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import glob
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantkubera.features.cpd_gpu import batched_optimal_partition_nig, extract_features_from_prev, NIGPrior

def main():
    parser = argparse.ArgumentParser(description='Mass CPD computation via GPU-DP')
    parser.add_argument('--lookback', type=int, default=1024, help='Max regime duration')
    parser.add_argument('--window', type=int, default=21, help='TMT context window for Norm Loc')
    parser.add_argument('--beta', type=float, default=20.0, help='CP penalty')
    parser.add_argument('--output-dir', default='data/cpd', help='Output directory')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Discover all tickers
    raw_files = glob.glob('data/raw/**/*.csv', recursive=True)
    if not raw_files:
        print("No data found in data/raw. Please run data acquisition first.")
        return
        
    print(f"Discovered {len(raw_files)} tickers.")
    
    # 2. Load all data into a batched tensor [B, T]
    # We need to align them on the same date index. 
    # For daily futures, most will align, but some might have gaps.
    # We will padding with 0 returns for gaps or pick a common range.
    
    all_dfs = {}
    all_dates = set()
    
    print("Aligning data...")
    for f in raw_files:
        ticker = Path(f).stem
        df = pd.read_csv(f, parse_dates=['date'])
        df.set_index('date', inplace=True)
        # Use returns for CPD, drop any pre-existing NaNs
        df = df.dropna(subset=['close'])
        df['ret'] = df['close'].pct_change().fillna(0)
        all_dfs[ticker] = df
        all_dates.update(df.index)
        
    sorted_dates = sorted(list(all_dates))
    T = len(sorted_dates)
    B = len(all_dfs)
    
    print(f"Universe: B={B}, T={T}")
    
    # Create the matrix
    x_np = np.zeros((B, T), dtype=np.float32)
    tickers = list(all_dfs.keys())
    
    for i, ticker in enumerate(tickers):
        df = all_dfs[ticker]
        # Reindex to common sorted_dates
        reindexed = df['ret'].reindex(sorted_dates, fill_value=0.0)
        x_np[i, :] = reindexed.values
        
    # Standardize per ticker
    means = x_np.mean(axis=1, keepdims=True)
    stds = x_np.std(axis=1, keepdims=True)
    x_np = (x_np - means) / (stds + 1e-8)
    
    # 3. Run GPU CPD
    print(f"\nInitiating GPU-DP CPD for {B} symbols...")
    start_time = time.time()
    
    F, prev = batched_optimal_partition_nig(
        x=tf.convert_to_tensor(x_np),
        beta=args.beta,
        lookback=args.lookback,
        min_seg=10
    )
    
    # Pull to numpy
    F_np = F.numpy()
    prev_np = prev.numpy()
    
    gpu_time = time.time() - start_time
    print(f"GPU Computation finished in {gpu_time:.2f} seconds.")
    
    # 4. Extract features
    print("Extracting TMT features...")
    cp_rl, cp_score = extract_features_from_prev(prev_np, lookback_window=args.window, F_np=F_np)
    
    # 5. Save results
    print(f"Saving results to {args.output_dir}...")
    for i, ticker in enumerate(tickers):
        out_df = pd.DataFrame({
            'date': sorted_dates,
            'cp_location_norm': cp_rl[i],
            'cp_score': cp_score[i]
        })
        # Note: we should only save rows that were in the original data for that ticker
        original_index = all_dfs[ticker].index
        out_df = out_df[out_df['date'].isin(original_index)]
        
        output_path = os.path.join(args.output_dir, f'{ticker}_cpd_{args.window}.csv')
        out_df.to_csv(output_path, index=False)
        if i % 50 == 0:
            print(f"   Stored {i}/{B}...")

    total_time = time.time() - start_time
    print(f"\n✅ Mass CPD Complete!")
    print(f"Processed {B} tickers in {total_time:.2f} seconds.")
    print(f"Speedup: Compared to GP-CPD (hours), this took seconds.")

if __name__ == '__main__':
    main()
