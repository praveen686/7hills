# Build stage
FROM rust:1.75-slim-bookworm as builder

WORKDIR /usr/src/quantlaxmi
COPY . .

# Install dependencies for build
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Build optimized binaries
RUN cargo build --release -p quantlaxmi-runner-india -p quantlaxmi-runner-crypto

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/src/quantlaxmi/target/release/quantlaxmi-india /app/quantlaxmi-india
COPY --from=builder /usr/src/quantlaxmi/target/release/quantlaxmi-crypto /app/quantlaxmi-crypto
COPY --from=builder /usr/src/quantlaxmi/crates/quantlaxmi-connectors-zerodha/scripts/zerodha_auth.py /app/zerodha_auth.py

# Install python dependencies for zerodha sidecar
RUN pip3 install kiteconnect pyotp --break-system-packages

ENV RUST_LOG=info

# Default entrypoint (india runner)
ENTRYPOINT ["/app/quantlaxmi-india"]
