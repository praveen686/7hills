import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from kiteconnect import KiteConnect
from quantkubera.data.kite_auth import KiteAuth

logger = logging.getLogger(__name__)

class KiteFetcher:
    def __init__(self):
        self.auth = KiteAuth()
        self.kite = self.auth.get_session()

    def fetch_historical(
        self,
        instrument_token: int,
        start_date: datetime,
        end_date: datetime,
        interval: str = "day",
        continuous: bool = True,
        oi: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data for a given instrument token.
        
        Args:
            instrument_token: Kite instrument token.
            start_date: Start datetime.
            end_date: End datetime.
            interval: One of 'minute', 'day', '3minute', '5minute', ...
            continuous: Whether to fetch continuous data (for futures).
            oi: Whether to include open interest.
        """
        logger.info(f"Fetching data for token {instrument_token} from {start_date} to {end_date}...")
        
        # Kite API has limits on data duration per request, especially for minute data.
        # Ideally, we should chunk requests here if the range is large.
        # For daily data, limits are generous (2000 days).
        
        records = self.kite.historical_data(
            instrument_token=instrument_token,
            from_date=start_date.strftime("%Y-%m-%d"),
            to_date=end_date.strftime("%Y-%m-%d"),
            interval=interval,
            continuous=continuous,
            oi=oi
        )
        
        if not records:
            logger.warning(f"No data returned for token {instrument_token}")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def get_instrument_token(self, trading_symbol: str, exchange: str = "NFO") -> Optional[int]:
        """Resolves a trading symbol to an instrument token."""
        # This is expensive as it fetches the full instrument dump.
        # In production, cache this or use a lookup file.
        instruments = self.kite.instruments(exchange)
        for instr in instruments:
            if instr['tradingsymbol'] == trading_symbol:
                return instr['instrument_token']
        return None
    
    def fetch_continuous_futures(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "day"
    ) -> pd.DataFrame:
        """Fetch continuous futures data for a ticker (e.g., NIFTY, BANKNIFTY).
        
        Args:
            ticker: Ticker symbol (e.g., 'NIFTY', 'BANKNIFTY')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval (default='day')
            
        Returns:
            DataFrame with OHLCV data
        """
        # For continuous futures, we use the index instrument
        # Map ticker names to NSE index symbols
        index_map = {
            'NIFTY': 'NIFTY 50',
            'BANKNIFTY': 'NIFTY BANK',
            'FINNIFTY': 'NIFTY FIN SERVICE',
            'MIDCPNIFTY': 'NIFTY MID SELECT'
        }
        
        index_symbol = index_map.get(ticker, ticker)
        
        # Get instrument token for the index
        instruments = self.kite.instruments('NSE')
        token = None
        for instr in instruments:
            if instr['tradingsymbol'] == index_symbol:
                token = instr['instrument_token']
                break
        
        if not token:
            raise ValueError(f"Could not find instrument token for {ticker} ({index_symbol})")
        
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Chunk requests to handle Kite's 2000-day limit for daily data
        chunk_size_days = 1800  # Safe margin below 2000
        all_data = []
        
        current_start = start_dt
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_size_days), end_dt)
            
            logger.info(f"Fetching {ticker} from {current_start.date()} to {current_end.date()}")
            
            chunk_df = self.fetch_historical(
                instrument_token=token,
                start_date=current_start,
                end_date=current_end,
                interval=interval,
                continuous=False,  # NSE indices don't support continuous flag
                oi=False  # Indices don't have OI
            )
            
            if not chunk_df.empty:
                all_data.append(chunk_df)
            
            current_start = current_end + timedelta(days=1)
            import time
            time.sleep(0.5)  # Respect Kite API limits (3 req/sec)
        
        # Combine all chunks
        if not all_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data)
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]  # Remove duplicates at chunk boundaries
        
        return combined_df

if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    fetcher = KiteFetcher()
    # E.g., NIFTY 50 equivalent or a future. Using a known token or looking one up would be better.
    # For now, let's just print that it initialized.
    print("KiteFetcher initialized successfully.")
