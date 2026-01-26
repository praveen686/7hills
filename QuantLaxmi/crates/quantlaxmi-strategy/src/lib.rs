//! Strategy SDK for QuantLaxmi
//!
//! ## Phase 2: Strategy as First-Class Citizen
//!
//! Strategies become deterministic, versioned producers of DecisionEvents.
//! The strategy authors both decisions AND intents together (not inferred by engine).
//!
//! ## Key Principles
//!
//! 1. **Strategy authors decisions**: `on_event()` returns `DecisionOutput` containing
//!    the `DecisionEvent` AND the `OrderIntent`s together.
//!
//! 2. **Engine records trace**: The engine records decisions to the trace after
//!    receiving outputs. Strategy never touches the trace builder.
//!
//! 3. **Deterministic config**: All config fields use fixed-point (mantissa + exponent)
//!    or integer units. No f64 in configs.
//!
//! 4. **Canonical hashing**: Config hash computed from `canonical_bytes()`, not JSON.
//!
//! ## Example
//!
//! ```ignore
//! use quantlaxmi_strategy::{Strategy, DecisionOutput, StrategyContext};
//!
//! struct MyStrategy { /* config fields */ }
//!
//! impl Strategy for MyStrategy {
//!     fn name(&self) -> &str { "my_strategy" }
//!     fn version(&self) -> &str { "1.0.0" }
//!     fn config_hash(&self) -> String { /* canonical bytes hash */ }
//!     fn strategy_id(&self) -> String {
//!         format!("{}:{}:{}", self.name(), self.version(), self.config_hash())
//!     }
//!
//!     fn on_event(&mut self, event: &ReplayEvent, ctx: &StrategyContext) -> Vec<DecisionOutput> {
//!         // Create DecisionEvent and OrderIntent together
//!         // Return DecisionOutput { decision, intents }
//!     }
//! }
//! ```

pub mod canonical;
pub mod context;
pub mod output;
pub mod registry;
pub mod strategies;

// Re-exports
pub use canonical::{CONFIG_ENCODING_VERSION, CanonicalBytes, canonical_hash};
pub use context::{FillNotification, StrategyContext};
pub use output::{DecisionOutput, OrderIntent, Side};
pub use registry::{StrategyFactory, StrategyRegistry};

// Re-export from quantlaxmi-models for convenience
pub use quantlaxmi_models::events::{DecisionEvent, MarketSnapshot};

/// Replay event kind (mirrors backtest EventKind).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    SpotQuote,
    PerpQuote,
    PerpDepth,
    Funding,
    Trade,
    Unknown,
}

/// Replay event (simplified view for strategy consumption).
#[derive(Debug, Clone)]
pub struct ReplayEvent {
    pub ts: chrono::DateTime<chrono::Utc>,
    pub symbol: String,
    pub kind: EventKind,
    pub payload: serde_json::Value,
}

/// Core Strategy trait for Phase 2 SDK.
///
/// Strategies author decisions AND intents together.
/// The engine records the trace after receiving outputs.
pub trait Strategy: Send {
    /// Unique strategy name (e.g., "funding_bias", "basis_capture").
    fn name(&self) -> &str;

    /// Strategy version for tracking changes (e.g., "1.0.0", "2.0.0").
    fn version(&self) -> &str;

    /// SHA-256 of canonical_bytes() - NOT JSON text.
    ///
    /// This hash is used for:
    /// - Reproducibility verification (same config → same hash)
    /// - Manifest binding
    /// - Strategy identity
    fn config_hash(&self) -> String;

    /// Stable strategy_id: "{name}:{version}:{config_hash}"
    ///
    /// The full config hash is used (not truncated) for uniqueness.
    /// Use `short_id()` for display purposes.
    fn strategy_id(&self) -> String {
        format!("{}:{}:{}", self.name(), self.version(), self.config_hash())
    }

    /// Short ID for display: "{name}:{version}:{hash[0:8]}"
    fn short_id(&self) -> String {
        let hash = self.config_hash();
        let short_hash = &hash[..8.min(hash.len())];
        format!("{}:{}:{}", self.name(), self.version(), short_hash)
    }

    /// Process event and return authored decisions with intents.
    ///
    /// The strategy MUST:
    /// 1. Create `DecisionEvent` for any trading decision
    /// 2. Create `OrderIntent`(s) for execution
    /// 3. Return `DecisionOutput { decision, intents }`
    ///
    /// The engine will:
    /// 1. Record the decision to the trace
    /// 2. Execute the intents through risk/execution pipeline
    /// 3. Send fills back via `on_fill()`
    fn on_event(&mut self, event: &ReplayEvent, ctx: &StrategyContext) -> Vec<DecisionOutput>;

    /// Receive fill notification for position tracking.
    ///
    /// Called by engine after successful execution.
    /// Use this to update internal position state.
    fn on_fill(&mut self, _fill: &FillNotification, _ctx: &StrategyContext) {
        // Default: no-op
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_kind_equality() {
        assert_eq!(EventKind::Funding, EventKind::Funding);
        assert_ne!(EventKind::Funding, EventKind::PerpQuote);
    }
}
