import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    """
    Handles feature extraction and normalization for the Trading Momentum Transformer.
    """
    def __init__(self):
        self.scalers = {}

    def calculate_returns(self, df: pd.DataFrame):
        """Adds return features matching TMT expectations."""
        # TMT expects: norm_daily, norm_monthly, norm_quarterly, norm_biannual, norm_annual
        # Mapping approx trading days:
        # daily: 1
        # monthly: 21
        # quarterly: 63
        # biannual: 126
        # annual: 252
        
        df['norm_daily_return'] = df['close'].pct_change(1)
        df['norm_monthly_return'] = df['close'].pct_change(21)
        df['norm_quarterly_return'] = df['close'].pct_change(63)
        df['norm_biannual_return'] = df['close'].pct_change(126)
        df['norm_annual_return'] = df['close'].pct_change(252)
        
        # Target: typically next day's return
        df['target_returns'] = df['norm_daily_return'].shift(-1)
        
        return df

    def calculate_macd(self, df: pd.DataFrame, fast=12, slow=26, signal=9):
        """Calculates MACD indicators."""
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        # signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        # TMT uses just the MACD line in its definition list, unnamed 'macd_X_Y'
        df[f'macd_{fast}_{slow}'] = macd
        return df

    def add_volatility(self, df: pd.DataFrame, window=21):
        """Adds rolling volatility."""
        df[f'volatility_{window}d'] = df['close'].pct_change().rolling(window=window).std()
        return df

    def process_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all feature transformations."""
        df = df.copy()
        df = self.calculate_returns(df)
        df = self.calculate_macd(df, 8, 24)
        df = self.calculate_macd(df, 16, 48)
        df = self.calculate_macd(df, 32, 96)
        # df = self.add_volatility(df) # Not strictly in TMT inputs, commenting out for strict parity
        
        # Drop NaN rows created by rolling windows/shifting
        df.dropna(inplace=True)
        return df

    def format_for_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final formatting: Scaling and selecting columns.
        """
        # Select numeric columns for scaling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        self.scalers['latest'] = scaler # Store for inverse transform if needed
        return df
    
    def add_cpd_features(self, df: pd.DataFrame, ticker: str, lookback: int = 21) -> pd.DataFrame:
        """Load and merge CPD (Changepoint Detection) features."""
        cpd_path = f'data/cpd/{ticker}_cpd_{lookback}.csv'
        
        if not os.path.exists(cpd_path):
            return df
        
        # Load CPD results
        cpd_df = pd.read_csv(cpd_path, parse_dates=['date']).set_index('date')
        
    def add_cpd_features(self, df: pd.DataFrame, ticker: str, lookback: int = 21) -> pd.DataFrame:
        """Load and merge CPD (Changepoint Detection) features."""
        cpd_path = f'data/cpd/{ticker}_cpd_{lookback}.csv'
        
        if not os.path.exists(cpd_path):
            return df
        
        # Load CPD results
        cpd_df = pd.read_csv(cpd_path, parse_dates=['date']).set_index('date')
        
        # Robust normalization (matches train_afml.py)
        MARKET_TZ = "Asia/Kolkata"
        def normalize_dt_index(idx, source_tz=MARKET_TZ, target_tz=MARKET_TZ):
            di = pd.to_datetime(idx, errors="raise")
            di = pd.DatetimeIndex(di)
            if di.tz is None:
                di = di.tz_localize(source_tz)
            else:
                di = di.tz_convert(target_tz)
            di = di.tz_convert(target_tz).tz_localize(None)
            return di

        df.index = normalize_dt_index(df.index)
        cpd_df.index = normalize_dt_index(cpd_df.index)
        
        # Direct assignment to avoid merge TZ sensitivity
        cpd_aligned = cpd_df.reindex(df.index)
        
        df[f'cp_rl_{lookback}'] = cpd_aligned['cp_location_norm']
        df[f'cp_score_{lookback}'] = cpd_aligned['cp_score']
        
        return df
        
        df[f'cp_rl_{lookback}'] = cpd_aligned['cp_location_norm']
        df[f'cp_score_{lookback}'] = cpd_aligned['cp_score']
        
        print(f"   ✅ Added CPD features: cp_rl_{lookback}, cp_score_{lookback}")
        return df
