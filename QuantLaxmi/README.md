# QuantLaxmi - Institutional-Grade Rust Trading Platform

A high-frequency, multi-venue trading system built in Rust for NSE F&O and crypto markets.

## Features

- **Multi-Venue Support**: Binance and Zerodha connectors with unified `MarketConnector` trait
- **HYDRA Strategy**: Multi-expert ensemble with 5 specialized strategies and meta-allocation
- **Real-Time Greeks**: Black-Scholes pricing with delta hedging execution
- **Circuit Breakers**: Rate limiting, latency monitoring, drawdown protection
- **Event Bus Architecture**: Low-latency (~230ns) async message passing
- **Institutional Observability**: Prometheus metrics, Grafana dashboards, OpenTelemetry tracing

## Project Structure

```
QuantLaxmi/
├── apps/                          # Application binaries
│   ├── quantlaxmi-india/          # India (NSE F&O) trading binary
│   └── quantlaxmi-crypto/         # Crypto (Binance) trading binary
├── crates/                        # Rust workspace
│   ├── quantlaxmi-core/           # Event bus, strategies, risk management
│   ├── quantlaxmi-models/         # Market data types, depth events
│   ├── quantlaxmi-data/           # L2 book snapshots, VPIN calculator
│   ├── quantlaxmi-executor/       # Order execution (simulated + live)
│   ├── quantlaxmi-options/        # Black-Scholes, Greeks, KiteSim, NSE specs
│   ├── quantlaxmi-risk/           # Position limits, order validation
│   ├── quantlaxmi-sbe/            # SBE binary protocol for Binance depth
│   ├── quantlaxmi-connectors-zerodha/   # Zerodha WebSocket connector
│   ├── quantlaxmi-connectors-binance/   # Binance WebSocket connector
│   ├── quantlaxmi-runner-common/  # Shared runner utilities (TUI, artifacts)
│   ├── quantlaxmi-runner-india/   # India-specific capture and replay
│   └── quantlaxmi-runner-crypto/  # Crypto-specific capture and replay
├── configs/                       # Configuration files
│   ├── backtest.toml
│   ├── paper.toml
│   └── live.toml
├── infra/                         # Infrastructure
│   └── observability/             # Grafana, Prometheus configs
├── python/                        # Python utilities (Zerodha auth, data)
├── scripts/                       # Build and validation scripts
├── tests/                         # Integration test fixtures
└── docs/                          # Documentation
```

## Quick Start

```bash
# Build all crates
cargo build --release

# Run India paper trading
cargo run -p quantlaxmi-india --release

# Run Crypto paper trading
cargo run -p quantlaxmi-crypto --release

# Run workspace tests
cargo test --workspace

# Run isolation checks
bash scripts/check_isolation.sh
```

## Binary Commands

### quantlaxmi-india
```bash
# Discover NSE F&O universe
quantlaxmi-india discover-zerodha --underlying BANKNIFTY --strikes 5

# Capture session (RECOMMENDED - audit-grade data with integrity tracking)
quantlaxmi-india capture-session \
    --underlying NIFTY,BANKNIFTY \
    --strike-band 20 \
    --expiry-policy t1t2t3 \
    --out-dir data/sessions/session_20260124 \
    --duration-secs 7200

# Run KiteSim backtest
quantlaxmi-india backtest-kitesim --replay data/quotes.jsonl --orders orders.json
```

> **Note**: The legacy `capture-zerodha` command is deprecated. Use `capture-session` for
> research-grade data with mantissa pricing and integrity_tier tracking (L2Present vs L1Only).
> Synthetic quotes are rejected by default in scoring to prevent illusory fills.

### quantlaxmi-crypto
```bash
# Capture Binance book ticker
quantlaxmi-crypto capture-binance --symbols BTCUSDT,ETHUSDT --duration-secs 300

# Capture SBE depth stream
quantlaxmi-crypto capture-sbe-depth --symbols BTCUSDT --duration-secs 60

# Get exchange info
quantlaxmi-crypto exchange-info
```

## Configuration

All configuration files are in `configs/`:

- `configs/paper.toml` - Paper trading configuration
- `configs/backtest.toml` - Backtesting configuration
- `configs/live.toml` - Live trading configuration (use with caution)

## Environment Variables

Required for Zerodha:
```bash
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_api_secret
ZERODHA_USER_ID=your_user_id
ZERODHA_PASSWORD=your_password
ZERODHA_TOTP_SECRET=your_totp_secret
```

Required for Binance:
```bash
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

## Architecture

The platform enforces strict dependency isolation:

- **India binary** (`quantlaxmi-india`): No Binance/SBE dependencies
- **Crypto binary** (`quantlaxmi-crypto`): No Zerodha dependencies

This isolation is enforced by CI and the `check_isolation.sh` script.

## License

PROPRIETARY - All rights reserved. See [LICENSE](./LICENSE).
