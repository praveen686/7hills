# QuantLaxmi

Institutional-grade trading system for India FnO (Futures & Options).

## Overview

QuantLaxmi is a full-stack quantitative trading platform covering the complete pipeline from data ingestion through live execution:

- **Python** -- Research, backtesting, feature engineering, ML/RL models
- **Rust** -- Live execution engine, real-time connectors, risk gates
- **Next.js** -- Web dashboard for monitoring, strategy analytics, and controls

## Quick Start

```bash
git clone <repo-url> && cd QuantLaxmi
make install          # Install Python package + dependencies
make test             # Run test suite (expect 1,290 tests passing)
make run-api          # Start FastAPI backend at localhost:8000
make run-ui           # Start Next.js frontend at localhost:3000
```

## Directory Structure

```
quantlaxmi/          Python package
  data/              Data access -- store, loaders, connectors, collectors
  features/          Feature engineering -- MegaFeatureBuilder (200+ features)
  models/            ML (TFT), RL (MDP/DQN/DDPG), AFML (meta-labeling, HRP)
  strategies/        25 strategies (S1-S25) + base, protocol, registry
  core/              Quant primitives -- pricing, risk, execution, events, backtest
  engine/            Trading runtime -- FastAPI, paper trading, live engine
rust/                Rust workspace (execution, connectors, risk gates)
ui/                  Next.js 14 frontend (React 18, TanStack Query, Recharts)
tests/               Python test suite (1,290 tests)
research/            Research scripts and results
docs/                Documentation
```

## Key Features

- **25 Strategies** (S1-S25): VRP options, IV mean-reversion, Hawkes intensity, HMM regime, TFT momentum, Divergence Flow Field, and more
- **MegaFeatureBuilder**: 200+ features from 10 groups (returns, technical, volatility, microstructure, information, fractional, Ramanujan, FTI, RMT, DFF)
- **Temporal Fusion Transformer**: Walk-forward validated with Sharpe ~2.0 on NIFTY/BANKNIFTY
- **RL Integration**: MDP, DQN, DDPG, Actor-Critic, Thompson Sampling -- Merton allocation, Deep Hedging, Optimal Execution
- **Real-Time Data**: Zerodha Kite (tick + 1-min + daily), Binance (crypto), NSE (index, FII/DII)
- **Walk-Forward Validation**: 4-fold OOS testing with realistic transaction costs
- **TimeGuard**: Systematic prevention of look-ahead bias across all features and strategies
- **Event Sourcing**: WAL + hash chain for full audit trail

## Performance Summary

| Strategy | Sharpe (OOS) | Return | Max Drawdown |
|---|---|---|---|
| S25 Divergence Flow Field (NIFTY) | 1.87 | +6.32% | -1.8% |
| S25 Divergence Flow Field (BANKNIFTY) | 2.16 | +8.06% | -2.1% |
| S4 IV Mean-Reversion | 1.85 | +4.8% | -1.5% |
| S5 Hawkes Intensity | 4.29 | +3.2% | -0.8% |
| Ensemble (S25+S4+S5) | 1.90 | +5.04% | -1.20% |

All results include realistic per-leg transaction costs (3 pts NIFTY, 5 pts BANKNIFTY) and use ddof=1 Sharpe with sqrt(252) annualization.

## Tech Stack

| Layer | Technology |
|---|---|
| Research & Backtesting | Python 3.12, pandas, NumPy, SciPy |
| Data Storage | DuckDB 1.4, PyArrow 23, Hive-partitioned Parquet |
| ML / Deep Learning | PyTorch 2.10, Transformers 5.1, XGBoost 3.1, scikit-learn 1.8 |
| API | FastAPI, Uvicorn |
| Execution Engine | Rust (Tokio, async) |
| Frontend | Next.js 14, React 18, TypeScript 5.7, Tailwind CSS, Recharts |
| Broker | Zerodha Kite (REST + WebSocket) |
| NLP | FinBERT on CUDA (T4 GPU) |

## Documentation

- [Architecture](docs/ARCHITECTURE.md) -- System design, data flow, key decisions
- [Getting Started](docs/GETTING_STARTED.md) -- Day 1 setup guide
- [Contributing](docs/CONTRIBUTING.md) -- Code style, testing, PR process
- [Strategy Docs](docs/strategies/) -- Per-strategy deep dives
- [Data Pipeline Docs](docs/data/) -- NSE, Telegram, Kite collector guides

## License

Proprietary. All rights reserved.
