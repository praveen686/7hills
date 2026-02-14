import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from quantkubera.data.kite_auth import KiteAuth
from quantkubera.data.kite_fetcher import KiteFetcher
from quantkubera.features.build_features import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_pipeline():
    print("1. Verifying Authentication...")
    try:
        auth = KiteAuth()
        # Mocking check: if credentials aren't set, this will fail or log warning.
        # For verification without live creds, we might need a mock mode.
        if not auth.api_key:
            print("   [WARN] No API key found. Skipping live fetch test.")
            return
        
        kite = auth.get_session()
        print("   [OK] Session object created.")
    except Exception as e:
        print(f"   [FAIL] Auth failed: {e}")
        return

    print("\n2. Verifying Data Fetching (Mock/Live)...")
    fetcher = KiteFetcher()
    
    # Try fetching a small chunk of data (e.g., NIFTY 50 -> 256265)
    # We use a recent date range.
    start_date = datetime.now() - timedelta(days=5)
    end_date = datetime.now()
    
    try:
        # Search for the current month's NIFTY future.
        # Date is 2026-02-13, so target NIFTY26FEBFUT.
        symbol = "NIFTY26FEBFUT"
        print(f"   [INFO] Looking up token for {symbol}...")
        token = fetcher.get_instrument_token(symbol, exchange="NFO")
        
        if not token:
            print(f"   [WARN] Could not find token for {symbol}. Trying generic search...")
            # Fallback/Debug: print some NIFTY symbols
            # instruments = fetcher.kite.instruments("NFO")
            # nifty = [i for i in instruments if i['name'] == 'NIFTY' and i['segment'] == 'NFO-FUT']
            # if nifty:
            #     print(f"   [INFO] Found {nifty[0]['tradingsymbol']} -> {nifty[0]['instrument_token']}")
            #     token = nifty[0]['instrument_token']
            #     symbol = nifty[0]['tradingsymbol']
            raise Exception(f"Token not found for {symbol}")

        print(f"   [INFO] Found token {token} for {symbol}. Fetching continuous data (continuous=True)...")
        df = fetcher.fetch_historical(token, start_date, end_date, interval="day", continuous=True, oi=True)
        
        if df.empty:
             print("   [WARN] Fetch returned empty dataframe.")
             raise Exception("Empty dataframe")
        else:
             print(f"   [OK] Connection Verified. Fetched {len(df)} rows of continuous futures data.")
             
             # For feature engineering, we need A LOT of data (annual returns = 252 lag)
             if len(df) < 300:
                print("   [INFO] Data insufficient for annual returns. Switching to synthetic data for Feature Engineering verification.")
                raise Exception("Trigger synthetic data generation")
             else:
                # Use real data if we have enough
                pass
             
    except Exception as e:
        if "Trigger synthetic" not in str(e):
            print(f"   [FAIL] Fetch failed: {e}")
            
        # Create dummy data for next steps
        # Need at least 252 + epsilon rows for annual returns
        dates = pd.date_range(end=datetime.now(), periods=400) 
        df = pd.DataFrame({
            'open': 100,
            'high': 105,
            'low': 95,
            'close': 100 + np.cumsum(np.random.randn(400)),
            'volume': 1000
        }, index=dates)
        print(f"   [INFO] Generated synthetic data ({len(df)} rows) for feature testing.")

    print("\n3. Verifying Feature Engineering...")
    try:
        engineer = FeatureEngineer()
        processed_df = engineer.process_ticker(df)
        
        expected_cols = ['norm_daily_return', 'norm_annual_return', 'macd_8_24', 'target_returns']
        missing = [c for c in expected_cols if c not in processed_df.columns]
        
        if missing:
            print(f"   [FAIL] Missing columns: {missing}")
        else:
            print("   [OK] Features calculated successfully.")
            print(f"   Shape: {processed_df.shape}")
            print(f"   Columns: {processed_df.columns.tolist()[:5]}...")

        # Test scaling
        final_df = engineer.format_for_model(processed_df)
        print("   [OK] Scaling applied.")
        
    except Exception as e:
        print(f"   [FAIL] Feature engineering failed: {e}")

if __name__ == "__main__":
    verify_pipeline()
