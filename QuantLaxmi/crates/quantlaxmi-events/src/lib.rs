//! # QuantLaxmi Events Crate
//!
//! Canonical event types with decision trace hashing for replay parity.
//!
//! This crate provides:
//! - Re-exports of canonical events from `quantlaxmi-models`
//! - `DecisionTrace` for tracking strategy decision sequences
//! - `DecisionTraceBuilder` with SHA-256 hashing for replay parity verification
//! - `verify_replay_parity()` for comparing original and replayed decision traces
//!
//! ## Replay Parity
//!
//! The core invariant for deterministic replay:
//! ```text
//! DecisionTraceHash(original) == DecisionTraceHash(replay)
//! ```
//!
//! This is achieved through strict canonical binary encoding of `DecisionEvent`,
//! independent of JSON/serde serialization order.
//!
//! ## Usage
//!
//! ```ignore
//! use quantlaxmi_events::{DecisionTraceBuilder, verify_replay_parity};
//! use quantlaxmi_events::DecisionEvent;
//!
//! // Build a trace during backtest
//! let mut builder = DecisionTraceBuilder::new();
//! builder.record(&decision1);
//! builder.record(&decision2);
//! let original_trace = builder.finalize();
//!
//! // Build a trace during replay
//! let mut replay_builder = DecisionTraceBuilder::new();
//! replay_builder.record(&replay_decision1);
//! replay_builder.record(&replay_decision2);
//! let replay_trace = replay_builder.finalize();
//!
//! // Verify parity
//! let result = verify_replay_parity(&original_trace, &replay_trace);
//! assert!(matches!(result, ReplayParityResult::Match));
//! ```

pub mod trace;

// Re-export canonical events from quantlaxmi-models
pub use quantlaxmi_models::{
    CanonicalQuoteEvent as QuoteEvent,
    // Canonical events with fixed-point representation
    CorrelationContext,
    DecisionEvent,
    // Depth events for L2 replay
    DepthEvent,
    DepthLevel,
    // Fill events
    FillEvent,

    IntegrityTier,

    MarketSnapshot,
    // Order lifecycle events
    OrderEvent,
    OrderPayload,
    OrderStatus,

    OrderType,
    ParseMantissaError,

    // Risk events
    RiskEvent,
    RiskEventType,

    // Common types
    Side,
    parse_to_mantissa_pure,
};

// Re-export trace types at crate root
pub use trace::{DecisionTrace, DecisionTraceBuilder, ReplayParityResult, verify_replay_parity};
