# Getting Started

A day-1 guide to setting up QuantLaxmi and running your first backtest.

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.11+ | 3.12 |
| RAM | 16 GB | 64 GB+ |
| GPU | None (CPU works) | NVIDIA T4+ with CUDA 12 (for TFT/RL training) |
| Disk | 20 GB free | 50 GB free |
| OS | Ubuntu 22.04+ / macOS 13+ | Ubuntu 24.04 |

## 1. Clone and Install

```bash
git clone <repo-url>
cd QuantLaxmi

# Create and activate the Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the package with all optional dependencies
make install
# This runs: pip install -e ".[dev,ml,zerodha,telegram]"
```

If you only need the core research tools (no broker connectivity or ML):

```bash
pip install -e ".[dev]"
```

## 2. Verify the Installation

```bash
make test
```

You should see approximately 1,290 tests passing. There are 4 pre-existing failures in `test_kite_depth.py` related to live Kite API connectivity -- these are expected when running offline.

To run only unit tests (skipping integration tests that need live connections):

```bash
make test-unit
```

## 3. Run Your First Backtest

Research scripts live in `research/`. A good starting point:

```bash
# Activate the venv if not already active
source venv/bin/activate

# Run an S25 Divergence Flow Field backtest
python research/s25_backtest.py
```

Backtest scripts produce:
- Sharpe ratio, total return, max drawdown
- Trade log with entry/exit dates
- Equity curve data

All backtests enforce the cost model and Sharpe protocol described in [ARCHITECTURE.md](ARCHITECTURE.md).

## 4. Start the API Server

```bash
make run-api
# Starts FastAPI at http://localhost:8000
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI with all 19 API routes.

Key endpoints:
- `GET /api/strategies` -- List all registered strategies
- `GET /api/backtest/{strategy_id}` -- Run a backtest
- `GET /api/features/{asset}` -- Get computed features
- `GET /api/positions` -- Current positions (paper/live)

## 5. Start the Frontend

In a separate terminal:

```bash
make run-ui
# Starts Next.js at http://localhost:3000
```

The dashboard provides:
- Strategy performance overview
- Equity curves and drawdown charts
- Feature importance visualizations
- Real-time position monitoring

## 6. Where to Look First

### Strategy Code

All 25 strategies are in `quantlaxmi/strategies/`:

```
quantlaxmi/strategies/
  base.py               BaseStrategy -- interface all strategies implement
  registry.py           Strategy registry (maps IDs to classes)
  s1_vrp/strategy.py    S1: Variance Risk Premium (options)
  s4_iv_mr/strategy.py  S4: IV Mean-Reversion
  s5_hawkes/strategy.py S5: Hawkes Intensity
  s25_divergence_flow/  S25: Divergence Flow Field
  ...
```

Each strategy module contains:
- `strategy.py` -- The strategy class (extends `BaseStrategy`)
- `research.py` -- Research notebook / backtest runner (where applicable)
- `__init__.py` -- Module exports

### Feature Engineering

Feature builders are in `quantlaxmi/features/`:

```
quantlaxmi/features/
  mega.py               MegaFeatureBuilder -- 200+ features from all sources
  base.py               BaseFeatureBuilder interface
  technical.py          RSI, MACD, Bollinger, ADX, etc.
  volatility.py         Realized vol, Parkinson, Garman-Klass
  divergence_flow.py    DFF features (Helmholtz decomposition)
```

To compute features for an asset:

```python
from quantlaxmi.features.mega import MegaFeatureBuilder

builder = MegaFeatureBuilder()
features = builder.build(asset="NIFTY", end_date="2026-02-01")
print(features.columns.tolist())  # 200+ feature columns
```

### Data Access

```python
from quantlaxmi.data.store import DataStore

store = DataStore()
df = store.load(category="nse_index_close", start="2025-01-01", end="2026-01-01")
```

### Tests

Tests mirror the source structure:

```
tests/
  test_features.py
  test_strategies.py
  test_s25_divergence_flow.py
  test_data_store.py
  ...
```

Run a single test file:

```bash
pytest tests/test_s25_divergence_flow.py -v
```

## 7. Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design details
- Read [CONTRIBUTING.md](CONTRIBUTING.md) before making changes
- Browse `docs/strategies/` for deep dives on individual strategies
- Browse `docs/data/` for data pipeline and collector documentation
- Check `research/results/` for the latest scorecard and backtest results
