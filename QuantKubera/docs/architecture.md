# QuantKubera Architecture Overview

## Goal
Transform `QuantKubera` into a scalable, production-ready quantitative trading platform with a focus on ML workflows.

## Directory Structure

```text
QuantKubera/
├── config/                 # Configuration files (model params, trading limits)
│   ├── base.yaml
│   └── hydra/
├── data/                   # Data storage (gitignored)
│   ├── raw/                # Immutable raw data
│   ├── processed/          # Cleaned/Featured data
│   └── external/           # Third-party data sources
├── docs/                   # Project documentation
│   ├── architecture.md     # This file
│   └── data_availability.md
├── external/               # Third-party repositories and dependencies
│   └── trading-momentum-transformer/
├── notebooks/              # Exploration and prototyping
│   ├── 01_exploration.ipynb
│   └── verify_pipeline.py  # Pipeline verification script
├── references/             # Papers, manuals, data dictionaries
├── reports/                # Generated reports and figures
├── src/                    # Main source code
│   └── quantkubera/
│       ├── __init__.py
│       ├── data/           # Data loaders and ETL (Kite integration)
│       ├── features/       # Feature engineering (TMT logic)
│       ├── models/         # ML model definitions (TFT, LSTM)
│       ├── strategies/     # Trading logic (Signal generation)
│       ├── backtest/       # Backtesting engine
│       ├── execution/      # Broker connectors
│       └── utils/          # Helper functions
├── tests/                  # Unit and integration tests
├── .env                    # Environment variables (API keys)
├── .gitignore              # Standard Python/Data gitignore
├── Makefile                # Automation commands (clean, test, run)
├── pyproject.toml          # Project metadata and dependencies
└── README.md               # Entry point
```

## Key Decisions
1.  **Src-Layout:** Using `src/quantkubera` prevents import errors and enforces packaging best practices.
2.  **External Isolation:** Moving `trading-momentum-transformer` to `external/` prevents codebase pollution while keeping it accessible for reference or integration.
3.  **Config Management:** Using `config/` establishes a pattern for reproducible experiments (crucial for ML).
4.  **Data Separation:** strict separation of `raw` and `processed` data ensures reproducibility.
5.  **Data Source:** Integration with **Zerodha Kite** for continuous futures data (replacing Quandl).

## Core Components
*   **Data Pipeline:** `src/quantkubera/data/` handles fetching from Kite and loading into Pandas DataFrames.
*   **Feature Engineering:** `src/quantkubera/features/` transforms raw OHLCV data into model features (Returns, MACD, Volatility).
*   **Model Architecture:** `src/quantkubera/models/` hosts the Temporal Fusion Transformer (TFT) and LSTM implementations adapted from the TMT paper.
