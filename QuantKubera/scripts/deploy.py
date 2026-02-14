#!/usr/bin/env python
"""
Main Deployment Script for QuantKubera.
Orchestrates live data fetching, model inference, and execution.
"""
import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantkubera.trading.executor import TradingExecutor
from quantkubera.monitoring.logger import TradeLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', default='NIFTY,BANKNIFTY,RELIANCE')
    parser.add_argument('--interval', type=int, default=60, help='Execution interval in seconds')
    parser.add_argument('--dry-run', action='store_true', help='Log signals but do not execute orders')
    args = parser.parse_args()
    
    ticker_list = args.tickers.split(',')
    
    # Initialize components
    logger = TradeLogger()
    executor = TradingExecutor(logger=logger)
    
    print(f"🚀 QuantKubera Live Deployment Started at {datetime.now()}")
    print(f"Tickers: {ticker_list}")
    print(f"Dry Run: {args.dry_run}")
    
    try:
        while True:
            start_time = time.time()
            
            for ticker in ticker_list:
                try:
                    # Execute trading logic for this ticker
                    # This includes:
                    # 1. Fetching live candles
                    # 2. Building features
                    # 3. Primary & Meta model inference
                    # 4. Bet sizing
                    # 5. Kite order placement (if not dry run)
                    
                    status = executor.execute_ticker_step(ticker, dry_run=args.dry_run)
                    
                except Exception as e:
                    print(f"Error executing {ticker}: {e}")
            
            # Sleep until next interval
            elapsed = time.time() - start_time
            sleep_time = max(0, args.interval - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n🛑 Deployment stopped by user.")
    except Exception as e:
        print(f"❌ Fatal Error: {e}")

if __name__ == '__main__':
    main()
