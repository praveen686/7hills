"""Count available futures instruments in Kite."""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantkubera.data.kite_fetcher import KiteFetcher

def count_instruments():
    fetcher = KiteFetcher()
    instruments = fetcher.kite.instruments()
    
    df = pd.DataFrame(instruments)
    
    # Filter for futures
    futures = df[df['instrument_type'] == 'FUT']
    
    print(f"\nTotal Instruments: {len(df)}")
    print(f"Total Futures: {len(futures)}")
    
    # Breakdown by exchange
    print("\nBreakdown by Exchange:")
    print(futures.groupby('exchange').size())
    
    # Breakdown by segment
    print("\nBreakdown by Segment:")
    print(futures.groupby('segment').size())
    
    # Identify Index vs Stock futures on NSE
    nse_fut = futures[futures['exchange'] == 'NFO']
    indices = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
    index_fut = nse_fut[nse_fut['name'].isin(indices)]
    stock_fut = nse_fut[~nse_fut['name'].isin(indices)]
    
    print(f"\nNSE Futures (NFO) Breakdown:")
    print(f"  Index Futures: {len(index_fut)}")
    print(f"  Stock Futures: {len(stock_fut)}")
    print(f"  Unique Stocks: {stock_fut['name'].nunique()}")

if __name__ == '__main__':
    import pandas as pd
    try:
        count_instruments()
    except Exception as e:
        print(f"Error: {e}")
