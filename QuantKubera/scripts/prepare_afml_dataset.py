#!/usr/bin/env python
"""
AFML Dataset Preparation Script.
Processes the 212-symbol universe to create an event-driven dataset using:
1. CUSUM Filtering (Event Sampling)
2. Triple Barrier Labeling (Risk-Aware Targets)
3. CPD Feature Integration
"""
import os
import sys
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import gc

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantkubera.features.build_features import FeatureEngineer
from quantkubera.features.sampling import cusum_filter
from quantkubera.features.labeling import triple_barrier_labels, get_daily_vol

def main():
    raw_dir = 'data/raw'
    output_dir = 'data/afml_events'
    os.makedirs(output_dir, exist_ok=True)
    
    engineer = FeatureEngineer()
    raw_files = glob.glob(f'{raw_dir}/**/*.csv', recursive=True)
    
    if not raw_files:
        print(f"Error: No raw data found in {raw_dir}")
        return

    print(f"Preparing AFML dataset for {len(raw_files)} tickers...")
    
    processed_count = 0
    total_events = 0
    
    for f in raw_files:
        ticker = Path(f).stem
        try:
            # 1. Load data
            df = pd.read_csv(f, parse_dates=['date'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 2. Add Base Features (Returns, MACD, etc.)
            df = engineer.process_ticker(df)
            df = engineer.add_volatility(df, window=20)
            df = engineer.add_volatility(df, window=60)
            
            # 3. Add CPD Features
            df = engineer.add_cpd_features(df, ticker=ticker, lookback=21)
            
            # Drop rows with NaNs in features
            feature_cols = [c for c in df.columns if c != 'target_returns']
            df.dropna(subset=feature_cols, inplace=True)
            
            if len(df) < 100:
                continue
                
            # 4. CUSUM Filter (Event Detection)
            # Use 50-day EWM Vol as threshold
            daily_vol = get_daily_vol(df['close'])
            # Threshold: usually 1.0 or 2.0 * daily_vol
            # Setting threshold to 1.0 * daily_vol for sufficient event density
            sampled_timestamps = cusum_filter(df['close'], threshold=daily_vol)
            
            if len(sampled_timestamps) == 0:
                print(f"   ⚠️ No CUSUM events found for {ticker}")
                continue
                
            # 5. Triple Barrier Labeling
            # pt_sl=(1.0, 1.0) -> Profit Take and Stop Loss at 1x Vol
            # num_days=5 -> 5-day vertical barrier
            labels = triple_barrier_labels(
                close=df['close'],
                events=sampled_timestamps,
                pt_sl=(2.0, 1.0), # Aggressive PT, tight SL
                target=daily_vol,
                num_days=10 # 10-day holding max
            )
            
            if labels.empty:
                print(f"   ⚠️ No labels generated for {ticker}")
                continue
                
            # 6. Merge Features and Labels
            # The labels DF is indexed by the event timestamps
            event_df = df.loc[labels.index].copy()
            event_df['label_ret'] = labels['ret']
            event_df['label_bin'] = labels['bin']
            event_df['label_t1'] = labels['t1']
            
            # 7. Save to disk
            output_path = os.path.join(output_dir, f"{ticker}_afml.csv")
            event_df.to_csv(output_path)
            
            processed_count += 1
            total_events += len(event_df)
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(raw_files)} tickers... Total events: {total_events}")
                gc.collect()
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    print("\n" + "="*50)
    print("AFML DATASET PREPARATION COMPLETE")
    print("="*50)
    print(f"Tickers Processed: {processed_count}")
    print(f"Total CUSUM Events: {total_events}")
    print(f"Average Events per Ticker: {total_events/processed_count:.2f}")
    print(f"Dataset saved to: {output_dir}")

if __name__ == '__main__':
    main()
