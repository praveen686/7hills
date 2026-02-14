import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from quantkubera.data.kite_auth import KiteAuth
from quantkubera.data.kite_fetcher import KiteFetcher

logging.basicConfig(level=logging.INFO)

def check_availability():
    auth = KiteAuth()
    fetcher = KiteFetcher()
    
    # Target Symbols: Major Indices Continuous Futures (Current Month: Feb 2026)
    indices_to_check = [
        "NIFTY26FEBFUT",
        "BANKNIFTY26FEBFUT",
        "FINNIFTY26FEBFUT",
        "MIDCPNIFTY26FEBFUT"
    ]
    
    results = {}

    for symbol in indices_to_check:
        print(f"\n--- Checking {symbol} ---")
        token = fetcher.get_instrument_token(symbol, exchange="NFO")
        
        if not token:
            print(f"Could not resolve token for {symbol}")
            results[symbol] = "Token Not Found"
            continue

        print(f"Resolved {symbol} to token {token}. Probing historical depth...")

        # Probe years backwards
        years_to_check = range(2025, 2005, -1)
        earliest_found = None
        
        for year in years_to_check:
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 1, 10) # Just check first 10 days
            
            try:
                # We need a small delay to avoid rate limits if we loop fast, though 3/sec is allowed.
                df = fetcher.fetch_historical(token, start_date, end_date, interval="day", continuous=True, oi=True)
                if not df.empty:
                    print(f"[OK] Data found for Jan {year}")
                    earliest_found = year
                else:
                    print(f"[FAIL] No data for Jan {year}")
                    break 
            except Exception as e:
                print(f"[ERROR] Fetch failed for {year}: {e}")
                break
                
        if earliest_found:
            results[symbol] = earliest_found
            print(f"Earliest available data for {symbol}: ~{earliest_found}")
        else:
            results[symbol] = "None"
            print(f"No data found for {symbol} in probed range.")

    print("\n\n=== Data Availability Summary ===")
    for sym, year in results.items():
        print(f"{sym}: Start Year ~{year}")

if __name__ == "__main__":
    check_availability()
