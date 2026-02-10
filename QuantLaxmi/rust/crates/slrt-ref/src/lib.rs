//! SLRT-GPU v1.1 Phase 1: CPU Reference Implementation
//!
//! This is the golden reference implementation. Speed is irrelevant. Correctness is everything.
//!
//! # Hard Constraints (from spec)
//! - No GPU
//! - No ML models
//! - No performance optimization
//! - No formula changes
//! - No inventing defaults
//! - Treat the spec as law
//!
//! # Module Structure
//! - `data`: Core data structures (LOB, Trade, MarketSnapshot)
//! - `features`: Snapshot and trade-flow feature computation
//! - `fti`: Follow-Through Indicator (port of Timothy Masters' FTI.CPP)
//! - `fragility`: Liquidity fragility score computation
//! - `normalization`: Rolling median/MAD normalization
//! - `subspace`: EW covariance + eigensolve for subspace tracking
//! - `regime`: Regime classification (R0-R3)
//! - `wal`: WAL-style structured logging and digests

pub mod alignment;
pub mod data;
pub mod features;
pub mod fragility;
pub mod fti;
pub mod normalization;
pub mod regime;
pub mod subspace;
pub mod wal;

/// Sealed constants from SLRT-GPU v1.1 specification.
/// These values are locked and must not be changed.
pub mod sealed {
    /// Epsilon for division safety (spec: implicit)
    pub const EPSILON: f64 = 1e-12;

    /// Volume epsilon for elasticity computation.
    /// When abs(signed_volume) < VOLUME_EPS, elasticity is undefined.
    /// Separate from EPSILON to allow independent tuning.
    pub const VOLUME_EPS: f64 = 1e-6;

    /// Maximum elasticity value for state vector capping.
    /// Prevents 1e12 values from poisoning normalization/covariance.
    /// Uses same bound as fragility clipping.
    pub const ELASTICITY_MAX: f64 = 10.0;

    /// Default LOB depth (spec: Section 3.1)
    pub const LOB_DEPTH: usize = 20;

    /// Normalization warmup: 5 minutes (spec: Section 6.2)
    pub const WARMUP_DURATION_SECS: u64 = 300;

    /// Normalization warmup: 30,000 ticks minimum (spec: Section 6.2)
    pub const WARMUP_MIN_TICKS: usize = 30_000;

    /// Normalization window: 30 minutes (spec: Section 6.2)
    pub const NORM_WINDOW_SECS: u64 = 1800;

    /// MAD zero threshold for confidence penalty (spec: Section 6.2)
    pub const MAD_ZERO_REFUSE_SECS: f64 = 2.0;

    /// Subspace rank r (spec: Section 6.3)
    pub const SUBSPACE_RANK: usize = 4;

    /// EW covariance decay lambda (spec: Section 6.3)
    pub const EW_DECAY_LAMBDA: f64 = 0.995;

    /// Eigensolve cadence in milliseconds (spec: Section 6.3)
    pub const EIGENSOLVE_CADENCE_MS: u64 = 250;

    /// Eigensolve cadence in batches (spec: Section 6.3)
    pub const EIGENSOLVE_CADENCE_BATCHES: usize = 500;

    /// Diagonal jitter for covariance (spec: Section 6.3)
    pub const DIAGONAL_JITTER: f64 = 1e-6;

    /// State vector dimension (spec: Section 6.1)
    pub const STATE_DIM: usize = 7;

    /// Confidence penalties (spec: Section 6.5)
    pub mod confidence {
        pub const MISSING_OPTIONAL_FIELD: f64 = 0.95;
        pub const ZERO_MAD: f64 = 0.80;
        pub const SPREAD_SOFT_BREACH: f64 = 0.70;
        pub const GPU_FALLBACK: f64 = 0.90;
        pub const NORMALIZATION_INSTABILITY: f64 = 0.85;
        /// Penalty for undefined elasticity (volume too low)
        pub const UNDEFINED_ELASTICITY: f64 = 0.85;
        /// Penalty for NoTradesWindow (toxicity not computable)
        pub const NO_TRADES_WINDOW: f64 = 0.95;
        /// Penalty for InsufficientHistory (FTI not computable)
        pub const INSUFFICIENT_HISTORY: f64 = 0.95;
        /// Penalty for non-certified trades source (public WS vs SBE)
        /// Smaller penalty than missing trades - data exists but lower integrity
        pub const NON_CERTIFIED_TRADES: f64 = 0.97;
    }
}
