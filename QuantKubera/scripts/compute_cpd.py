import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantkubera.features.cpd import run_module


def compute_cpd_for_ticker(
    ticker: str,
    file_path: str = None,
    start_date: str = '2011-01-01',
    end_date: str = '2026-02-13',
    lookback_window: int = 21,
    output_dir: str = 'data/cpd'
):
    """Compute CPD features for a single ticker."""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{ticker}_cpd_{lookback_window}.csv')
        
        # Check if already exists
        if os.path.exists(output_path):
            return f"SKIP: {ticker} (Exists)"

        # Load data
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['date'])
            df.set_index('date', inplace=True)
            # Ensure index is naive before filtering (prevents tz-aware vs naive error)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        else:
            return f"FAIL: {ticker} (No data source)"

        # Filter dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df.sort_index()
        df = df[start_dt:end_dt]
        
        if len(df) < lookback_window * 2:
            return f"FAIL: {ticker} (Insufficient data: {len(df)} rows)"

        # Calculate daily returns (required for CPD)
        if 'daily_returns' not in df.columns:
            df['daily_returns'] = df['close'].pct_change()
        df = df.dropna(subset=['daily_returns'])
        
        # Convert timezone-aware index to naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Run CPD module
        run_module(
            time_series_data=df,
            lookback_window_length=lookback_window,
            output_csv_file_path=output_path,
            start_date=start_dt.to_pydatetime(),
            end_date=end_dt.to_pydatetime(),
            use_kM_hyp_to_initialise_kC=True
        )
        
        return f"SUCCESS: {ticker} ({len(df)} rows)"
    except Exception as e:
        return f"ERROR: {ticker} ({str(e)})"


def main():
    parser = argparse.ArgumentParser(description='Compute CPD features in parallel')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to compute')
    parser.add_argument('--all', action='store_true', help='Process all files in data/raw')
    parser.add_argument('--lookback', type=int, default=21, help='Lookback window length')
    parser.add_argument('--start-date', default='2011-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2026-02-13', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default='data/cpd', help='Output directory')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Number of parallel workers')
    
    args = parser.parse_args()
    
    tasks = []
    if args.all:
        raw_files = glob.glob('data/raw/**/*.csv', recursive=True)
        for f in raw_files:
            ticker = Path(f).stem
            tasks.append((ticker, f))
    elif args.tickers:
        for ticker in args.tickers:
            # Try to find file
            files = glob.glob(f'data/raw/**/{ticker}.csv', recursive=True)
            if files:
                tasks.append((ticker, files[0]))
            else:
                print(f"Warning: No data found for {ticker}")

    print(f"Found {len(tasks)} symbols to process using {args.workers} workers.")
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_ticker = {
            executor.submit(
                compute_cpd_for_ticker, 
                ticker, path, args.start_date, args.end_date, args.lookback, args.output_dir
            ): ticker for ticker, path in tasks
        }
        
        completed = 0
        for future in as_completed(future_to_ticker):
            result = future.result()
            completed += 1
            print(f"[{completed}/{len(tasks)}] {result}")

    print("\nCPD Process Finished.")


if __name__ == '__main__':
    main()
