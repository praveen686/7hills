# QuantLaxmi Monitoring Stack

This directory contains the configuration for Prometheus and Grafana to monitor the QuantLaxmi trading system.

## 🛠️ Components
- **Prometheus**: Scrapes metrics from the QuantLaxmi application (default: `localhost:9000`).
- **Grafana**: Provides a visual dashboard for monitoring ingestion latency, strategy performance, risk checks, and system health.

## 🚀 How to Run

### 1. Start the Monitoring Stack
Ensure you have Docker and Docker Compose installed, then run:
```bash
cd monitoring
docker-compose up -d
```

### 2. Access the Dashboards
- **Grafana**: Visit `http://localhost:3000` (User: `admin`, Password: `admin`)
- **Prometheus**: Visit `http://localhost:9090`

### 3. Run QuantLaxmi
When running the `quantlaxmi-india`, ensure the `METRICS_PORT` environment variable matches the Prometheus configuration (default is `9000`).

```bash
cargo run --release --bin quantlaxmi-india -- --mode paper
```

## 📊 Available Metrics
The following metrics are pre-configured in the Grafana dashboard:
- `quantlaxmi_ingestion_latency_ms`: p99 latency for market data ingestion.
- `quantlaxmi_strategy_decision_latency_ms`: p99 latency for strategy loops.
- `quantlaxmi_risk_checks_total`: Count of passed and failed risk checks.
- `quantlaxmi_ingestion_messages_total`: Throughput of the data connectors.
