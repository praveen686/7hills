"""
Batch fetcher for all futures contracts available in Kite.
Supports NSE, BSE, MCX, CDS, etc.
Implements rate limiting and organized storage.
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantkubera.data.kite_fetcher import KiteFetcher

def fetch_universe(
    exchanges=['NFO', 'BFO', 'MCX', 'CDS', 'NCO'],
    start_date='2011-01-01',
    end_date='2026-02-13',
    batch_delay=1.0  # Increased for stability with chunked requests
):
    fetcher = KiteFetcher()
    print(f"Fetching all instruments from Kite...")
    instruments = fetcher.kite.instruments()
    df = pd.DataFrame(instruments)
    
    # Filter for futures
    futures = df[df['instrument_type'] == 'FUT']
    futures = futures[futures['exchange'].isin(exchanges)]
    
    unique_names = futures['name'].unique()
    print(f"Universe identified: {len(unique_names)} unique tickers across {len(exchanges)} exchanges.")
    print(f"Total specific contracts: {len(futures)}")
    
    base_raw_path = Path('data/raw')
    os.makedirs(base_raw_path, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for i, ticker in enumerate(unique_names):
        # Find which exchange this belongs to (priority to NFO)
        ticker_info = futures[futures['name'] == ticker].iloc[0]
        exchange = ticker_info['exchange']
        
        target_dir = base_raw_path / exchange
        os.makedirs(target_dir, exist_ok=True)
        
        target_file = target_dir / f"{ticker}.csv"
        
        if target_file.exists():
            print(f"[{i+1}/{len(unique_names)}] Skipping {ticker} (already exists)")
            continue
            
        print(f"[{i+1}/{len(unique_names)}] Fetching {ticker} ({exchange})...", end='', flush=True)
        
        try:
            # Use continuous futures if possible for better backtesting
            # KiteFetcher has fetch_continuous_futures which handles roll-overs
            data = fetcher.fetch_continuous_futures(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval='day'
            )
            
            if not data.empty:
                data.to_csv(target_file)
                print(f" SUCCESS ({len(data)} rows)")
                success_count += 1
            else:
                print(" EMPTY")
                fail_count += 1
                
        except Exception as e:
            print(f" FAILED: {str(e)[:50]}...")
            fail_count += 1
            
        # Throttling to respect Kite API limits (3 req/sec)
        time.sleep(batch_delay)
        
    print("\n" + "="*50)
    print("BATCH FETCH COMPLETE")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Total:   {len(unique_names)}")
    print("="*50)

if __name__ == '__main__':
    # Usage: python scripts/fetch_all_futures.py
    # Defaulting to NSE only for first pass if desired, or all
    fetch_universe()
