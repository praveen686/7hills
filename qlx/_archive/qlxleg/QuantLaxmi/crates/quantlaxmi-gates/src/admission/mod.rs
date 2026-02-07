//! Signal Admission Control (Phase 18)
//!
//! Prevents invalid or dishonest signal computation by refusing to compute
//! signals when required inputs are missing or uncertain.
//!
//! ## Core Question
//! "Are we allowed to compute this signal without lying to ourselves?"
//!
//! ## Hard Laws (Frozen)
//! - L1: No Fabrication — Missing inputs (None) MUST NOT become values
//! - L2: Deterministic — Same inputs → identical decision + digest
//! - L3: Explicit Refusal — Missing required inputs → Refuse with reasons
//! - L4: Separation — Does not inspect risk, PnL, or session state
//! - L5: Zero Is Valid — Some(0) is vendor-asserted and MUST be admitted
//! - L6: Observability — Every admission produces an auditable artifact
//!
//! ## Pipeline Position
//! ```text
//! Vendor / Internal Snapshots
//!         ↓
//! SignalAdmissionController   ← Phase 18 (this module)
//!         ↓
//! Signal Compute / Features
//!         ↓
//! RiskEvaluator               (15.1)
//! ```

mod controller;

pub use controller::{
    AdmissionContext, InternalSnapshot, SignalAdmissionController, VendorSnapshot,
};
