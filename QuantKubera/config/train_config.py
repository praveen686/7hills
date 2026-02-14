"""Training configuration for QuantKubera Momentum Transformer."""

# ============================================================================
# Model Architecture Configuration
# ============================================================================
MODEL_CONFIG = {
    'hidden_size': 128,  # Hidden dimension for GRN, LSTM, Attention
    'num_heads': 4,  # Multi-head attention heads
    'dropout_rate': 0.1,  # Dropout for regularization
}

# ============================================================================
# Data Configuration
# ============================================================================
DATA_CONFIG = {
    # Window size for time series (21 ~ 1 trading month)
    'window_size': 21,
    
    # Tickers to train on
    'tickers': ['NIFTY', 'BANKNIFTY'],
    
    # Date splits (YYYY-MM-DD format)
    'train_end': '2020-12-31',  # End of training period
    'val_end': '2022-12-31',    # End of validation period
    # Test period: after val_end
    
    # Feature columns (must match build_features.py output)
    'feature_cols': [
        'norm_daily_return',
        'norm_monthly_return', 
        'norm_quarterly_return',
        'norm_biannual_return',
        'norm_annual_return',
        'macd_8_24',
        'macd_16_48',
        'macd_32_96',
        'volatility_20d',  # Note: FeatureEngineer adds 'd' suffix
        'volatility_60d',
        'cp_rl_21',        # CPD: Changepoint relative location
        'cp_score_21',     # CPD: Changepoint severity score
    ],
    
    'target_col': 'target_returns',
}

# ============================================================================
# Training Configuration
# ============================================================================
TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 1e-3,
    'early_stop_patience': 10,
    'model_dir': 'models/trained',
}

# ============================================================================
# Smoke Test Configuration (for quick validation)
# ============================================================================
SMOKE_TEST_CONFIG = {
    'train_end': '2023-12-31',
    'val_end': '2024-06-30',
    'tickers': ['NIFTY'],  # Single ticker
    'model_config': {
        'hidden_size': 32,  # Smaller model
        'num_heads': 2,
        'dropout_rate': 0.1,
    },
    'train_config': {
        'batch_size': 16,
        'epochs': 20,
        'learning_rate': 1e-3,
        'early_stop_patience': 5,
        'model_dir': 'models/smoke_test',
    }
}
