import pandas as pd
import os
import glob
from typing import List, Dict

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_ticker(self, ticker: str) -> pd.DataFrame:
        """Loads data for a single ticker."""
        file_path = os.path.join(self.data_dir, f"{ticker}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data for {ticker} not found at {file_path}")
        
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        # Ensure standard columns (lower case)
        df.rename(columns=str.lower, inplace=True)
        return df

    def load_all_tickers(self) -> Dict[str, pd.DataFrame]:
        """Loads all CSV files in the data directory."""
        all_data = {}
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        for f in csv_files:
            ticker = os.path.splitext(os.path.basename(f))[0]
            try:
                all_data[ticker] = self.load_ticker(ticker)
            except Exception as e:
                print(f"Failed to load {ticker}: {e}")
        return all_data
