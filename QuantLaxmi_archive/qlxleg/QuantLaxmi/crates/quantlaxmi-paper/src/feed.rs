use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};

/// Canonical market events consumed by the paper engine.
///
/// Runners can either stream raw ticks and build snapshots internally,
/// or stream already-aggregated snapshots as `MarketEvent::Snapshot`.
#[derive(Debug, Clone)]
pub enum MarketEvent<TSnapshot> {
    /// A time-stamped snapshot (best bid/ask, greeks, etc.) defined by the venue adapter.
    Snapshot {
        ts: DateTime<Utc>,
        snapshot: TSnapshot,
    },

    /// A heartbeat event for liveness / UI updates.
    Heartbeat { ts: DateTime<Utc> },
}

/// Venue adapter that yields market events.
///
/// Implementations:
/// - India: Zerodha WS → option-chain snapshot builder
/// - Crypto: Binance WS/SBE → book/funding snapshot builder
#[async_trait]
pub trait MarketFeed<TSnapshot>: Send + Sync {
    async fn next(&mut self) -> Result<MarketEvent<TSnapshot>>;
}
