//! Signal Admission Control types.
//!
//! Phase 18: Prevents invalid or dishonest signal computation by refusing
//! to compute signals when required inputs are missing or uncertain.
//!
//! ## Core Question
//! "Are we allowed to compute this signal without lying to ourselves?"
//!
//! ## Hard Laws (Frozen)
//! - L1: No Fabrication — Missing inputs (None) MUST NOT become values
//! - L2: Deterministic — Same inputs → identical decision + digest
//! - L3: Explicit Refusal — Missing required inputs → Refuse with reasons
//! - L4: Separation — Admission does not inspect risk, PnL, or session state
//! - L5: Zero Is Valid — Some(0) is vendor-asserted and MUST be admitted
//! - L6: Observability — Every admission produces an auditable artifact
//!
//! See: Phase 18 spec for full details.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

// =============================================================================
// Schema Version
// =============================================================================

/// Schema version for admission decision serialization.
pub const ADMISSION_SCHEMA_VERSION: &str = "1.0.0";

// =============================================================================
// VendorField — What vendor data does a signal require?
// =============================================================================

/// Vendor-provided market data fields that signals may depend on.
///
/// These represent data that comes from external sources (exchanges, brokers)
/// where `None` means "vendor omitted" and `Some(0)` means "vendor asserted zero".
///
/// Lives in quantlaxmi-models (venue-agnostic). Venue-specific extensions use
/// the `VenueSpecific` variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VendorField {
    /// Best bid quantity (depth at top of book)
    BuyQuantity,
    /// Best ask quantity (depth at top of book)
    SellQuantity,
    /// Trading volume (period-dependent)
    Volume,
    /// Last traded quantity
    LastQuantity,
    /// Last traded price
    LastPrice,
    /// OHLC candle data
    Ohlc,
    /// Best bid price
    BidPrice,
    /// Best ask price
    AskPrice,
    /// Venue-specific field (for extensions without enum churn)
    VenueSpecific { venue: String, field: String },
}

impl fmt::Display for VendorField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VendorField::BuyQuantity => write!(f, "buy_quantity"),
            VendorField::SellQuantity => write!(f, "sell_quantity"),
            VendorField::Volume => write!(f, "volume"),
            VendorField::LastQuantity => write!(f, "last_quantity"),
            VendorField::LastPrice => write!(f, "last_price"),
            VendorField::Ohlc => write!(f, "ohlc"),
            VendorField::BidPrice => write!(f, "bid_price"),
            VendorField::AskPrice => write!(f, "ask_price"),
            VendorField::VenueSpecific { venue, field } => {
                write!(f, "{}:{}", venue, field)
            }
        }
    }
}

// =============================================================================
// InternalField — What internal data does a signal require?
// =============================================================================

/// Internal data fields that signals may depend on.
///
/// These represent data computed or aggregated internally, not session state.
/// Phase 18 checks data availability only — session gating remains Phase 16.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InternalField {
    /// Last computed mid price
    LastMidPrice,
    /// Level 1 book snapshot (best bid/ask)
    L1Book,
    /// Level 2 book snapshot (full depth)
    L2Book,
    /// Trade tape / recent trades
    TradeTape,
    /// Named feature dependency (explicit)
    Feature(String),
}

impl fmt::Display for InternalField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InternalField::LastMidPrice => write!(f, "last_mid_price"),
            InternalField::L1Book => write!(f, "l1_book"),
            InternalField::L2Book => write!(f, "l2_book"),
            InternalField::TradeTape => write!(f, "trade_tape"),
            InternalField::Feature(name) => write!(f, "feature:{}", name),
        }
    }
}

// =============================================================================
// SignalRequirements — Declaration of what a signal needs
// =============================================================================

/// Declaration of input requirements for a signal.
///
/// Each signal must declare what it needs before computation.
/// Silence is not allowed — requirements must be explicit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalRequirements {
    /// Unique identifier for this signal
    pub signal_id: String,

    /// Vendor fields that MUST be present (None → Refuse)
    pub required_vendor_fields: Vec<VendorField>,

    /// Internal data that MUST be present (None → Refuse)
    pub required_internal_fields: Vec<InternalField>,

    /// Vendor fields that may be missing without refusal (analytics-only)
    pub optional_vendor_fields: Vec<VendorField>,
}

impl SignalRequirements {
    /// Create requirements for a signal with only required vendor fields.
    pub fn new(signal_id: impl Into<String>, required_vendor: Vec<VendorField>) -> Self {
        Self {
            signal_id: signal_id.into(),
            required_vendor_fields: required_vendor,
            required_internal_fields: Vec::new(),
            optional_vendor_fields: Vec::new(),
        }
    }

    /// Add required internal fields.
    pub fn with_internal(mut self, fields: Vec<InternalField>) -> Self {
        self.required_internal_fields = fields;
        self
    }

    /// Add optional vendor fields.
    pub fn with_optional(mut self, fields: Vec<VendorField>) -> Self {
        self.optional_vendor_fields = fields;
        self
    }
}

// =============================================================================
// AdmissionOutcome — The binary decision
// =============================================================================

/// The outcome of an admission decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdmissionOutcome {
    /// Signal may compute — all required inputs present
    Admit,
    /// Signal MUST NOT compute — required inputs missing
    Refuse,
}

impl fmt::Display for AdmissionOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdmissionOutcome::Admit => write!(f, "Admit"),
            AdmissionOutcome::Refuse => write!(f, "Refuse"),
        }
    }
}

// =============================================================================
// AdmissionDecision — The audit artifact
// =============================================================================

/// Audit artifact for a signal admission decision.
///
/// Every admission attempt produces this artifact, whether admitted or refused.
/// Written to WAL before signal computation.
///
/// ## Null vs Absent Distinction (Doctrine: No Silent Poisoning)
/// - `missing_vendor_fields`: Vendor did not send this field (Absent)
/// - `null_vendor_fields`: Vendor explicitly sent null (Null)
///
/// Both trigger refusal, but the diagnostic reason differs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionDecision {
    /// Schema version for forward compatibility
    pub schema_version: String,

    /// Timestamp in nanoseconds since epoch
    pub ts_ns: i64,

    /// Session identifier
    pub session_id: String,

    /// Signal being evaluated
    pub signal_id: String,

    /// The admission outcome
    pub outcome: AdmissionOutcome,

    /// Vendor fields that were required but absent (vendor omitted)
    pub missing_vendor_fields: Vec<VendorField>,

    /// Vendor fields that were required but vendor sent explicit null
    pub null_vendor_fields: Vec<VendorField>,

    /// Internal fields that were required but missing (empty if Admit)
    pub missing_internal_fields: Vec<InternalField>,

    /// Optional linkage to upstream decision/intent context
    pub correlation_id: Option<String>,

    /// SHA-256 digest of canonical representation
    pub digest: String,
}

impl AdmissionDecision {
    /// Check if this decision allows signal computation.
    pub fn is_admitted(&self) -> bool {
        self.outcome == AdmissionOutcome::Admit
    }

    /// Check if this decision refuses signal computation.
    pub fn is_refused(&self) -> bool {
        self.outcome == AdmissionOutcome::Refuse
    }
}

// =============================================================================
// CanonicalBytes — Deterministic serialization for digest
// =============================================================================

/// Trait for canonical byte serialization (same as Phase 15).
/// Used to produce deterministic digests.
pub trait AdmissionCanonicalBytes {
    fn canonical_bytes(&self) -> Vec<u8>;
}

impl AdmissionCanonicalBytes for VendorField {
    fn canonical_bytes(&self) -> Vec<u8> {
        // Tag byte + payload
        match self {
            VendorField::BuyQuantity => vec![0x01],
            VendorField::SellQuantity => vec![0x02],
            VendorField::Volume => vec![0x03],
            VendorField::LastQuantity => vec![0x04],
            VendorField::LastPrice => vec![0x05],
            VendorField::Ohlc => vec![0x06],
            VendorField::BidPrice => vec![0x07],
            VendorField::AskPrice => vec![0x08],
            VendorField::VenueSpecific { venue, field } => {
                let mut bytes = vec![0xFF]; // VenueSpecific tag
                // Length-prefixed UTF-8 strings
                let venue_bytes = venue.as_bytes();
                bytes.extend_from_slice(&(venue_bytes.len() as u32).to_le_bytes());
                bytes.extend_from_slice(venue_bytes);
                let field_bytes = field.as_bytes();
                bytes.extend_from_slice(&(field_bytes.len() as u32).to_le_bytes());
                bytes.extend_from_slice(field_bytes);
                bytes
            }
        }
    }
}

impl AdmissionCanonicalBytes for InternalField {
    fn canonical_bytes(&self) -> Vec<u8> {
        match self {
            InternalField::LastMidPrice => vec![0x01],
            InternalField::L1Book => vec![0x02],
            InternalField::L2Book => vec![0x03],
            InternalField::TradeTape => vec![0x04],
            InternalField::Feature(name) => {
                let mut bytes = vec![0xFF]; // Feature tag
                let name_bytes = name.as_bytes();
                bytes.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
                bytes.extend_from_slice(name_bytes);
                bytes
            }
        }
    }
}

impl AdmissionCanonicalBytes for AdmissionOutcome {
    fn canonical_bytes(&self) -> Vec<u8> {
        match self {
            AdmissionOutcome::Admit => vec![0x01],
            AdmissionOutcome::Refuse => vec![0x02],
        }
    }
}

/// Canonical bytes for AdmissionDecision (excluding digest field).
///
/// Field order (frozen):
/// 1. schema_version (len-prefixed UTF-8)
/// 2. ts_ns (i64 LE)
/// 3. session_id (len-prefixed UTF-8)
/// 4. signal_id (len-prefixed UTF-8)
/// 5. outcome (tag byte)
/// 6. missing_vendor_fields (count + each field's canonical bytes)
/// 7. null_vendor_fields (count + each field's canonical bytes)
/// 8. missing_internal_fields (count + each field's canonical bytes)
/// 9. correlation_id (0x00 for None, 0x01 + len-prefixed UTF-8 for Some)
impl AdmissionCanonicalBytes for AdmissionDecision {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // 1. schema_version
        let sv_bytes = self.schema_version.as_bytes();
        bytes.extend_from_slice(&(sv_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(sv_bytes);

        // 2. ts_ns
        bytes.extend_from_slice(&self.ts_ns.to_le_bytes());

        // 3. session_id
        let sid_bytes = self.session_id.as_bytes();
        bytes.extend_from_slice(&(sid_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(sid_bytes);

        // 4. signal_id
        let sig_bytes = self.signal_id.as_bytes();
        bytes.extend_from_slice(&(sig_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(sig_bytes);

        // 5. outcome
        bytes.extend_from_slice(&self.outcome.canonical_bytes());

        // 6. missing_vendor_fields
        bytes.extend_from_slice(&(self.missing_vendor_fields.len() as u32).to_le_bytes());
        for field in &self.missing_vendor_fields {
            let fb = field.canonical_bytes();
            bytes.extend_from_slice(&(fb.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&fb);
        }

        // 7. null_vendor_fields
        bytes.extend_from_slice(&(self.null_vendor_fields.len() as u32).to_le_bytes());
        for field in &self.null_vendor_fields {
            let fb = field.canonical_bytes();
            bytes.extend_from_slice(&(fb.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&fb);
        }

        // 8. missing_internal_fields
        bytes.extend_from_slice(&(self.missing_internal_fields.len() as u32).to_le_bytes());
        for field in &self.missing_internal_fields {
            let fb = field.canonical_bytes();
            bytes.extend_from_slice(&(fb.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&fb);
        }

        // 9. correlation_id
        match &self.correlation_id {
            None => bytes.push(0x00),
            Some(cid) => {
                bytes.push(0x01);
                let cid_bytes = cid.as_bytes();
                bytes.extend_from_slice(&(cid_bytes.len() as u32).to_le_bytes());
                bytes.extend_from_slice(cid_bytes);
            }
        }

        bytes
    }
}

/// Compute SHA-256 digest of canonical bytes (matches Phase 15 pattern).
pub fn compute_digest(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    hex::encode(result)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vendor_field_display() {
        assert_eq!(VendorField::BuyQuantity.to_string(), "buy_quantity");
        assert_eq!(VendorField::SellQuantity.to_string(), "sell_quantity");
        assert_eq!(
            VendorField::VenueSpecific {
                venue: "zerodha".to_string(),
                field: "oi".to_string()
            }
            .to_string(),
            "zerodha:oi"
        );
    }

    #[test]
    fn test_internal_field_display() {
        assert_eq!(InternalField::LastMidPrice.to_string(), "last_mid_price");
        assert_eq!(
            InternalField::Feature("vwap".to_string()).to_string(),
            "feature:vwap"
        );
    }

    #[test]
    fn test_signal_requirements_builder() {
        let req = SignalRequirements::new(
            "book_imbalance",
            vec![VendorField::BuyQuantity, VendorField::SellQuantity],
        )
        .with_internal(vec![InternalField::L1Book])
        .with_optional(vec![VendorField::Volume]);

        assert_eq!(req.signal_id, "book_imbalance");
        assert_eq!(req.required_vendor_fields.len(), 2);
        assert_eq!(req.required_internal_fields.len(), 1);
        assert_eq!(req.optional_vendor_fields.len(), 1);
    }

    #[test]
    fn test_admission_outcome_display() {
        assert_eq!(AdmissionOutcome::Admit.to_string(), "Admit");
        assert_eq!(AdmissionOutcome::Refuse.to_string(), "Refuse");
    }

    #[test]
    fn test_admission_decision_helpers() {
        let admit = AdmissionDecision {
            schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
            ts_ns: 1234567890,
            session_id: "sess_001".to_string(),
            signal_id: "test_signal".to_string(),
            outcome: AdmissionOutcome::Admit,
            missing_vendor_fields: vec![],
            null_vendor_fields: vec![],
            missing_internal_fields: vec![],
            correlation_id: None,
            digest: "test".to_string(),
        };
        assert!(admit.is_admitted());
        assert!(!admit.is_refused());

        let refuse = AdmissionDecision {
            outcome: AdmissionOutcome::Refuse,
            missing_vendor_fields: vec![VendorField::BuyQuantity],
            ..admit.clone()
        };
        assert!(!refuse.is_admitted());
        assert!(refuse.is_refused());
    }

    #[test]
    fn test_canonical_bytes_determinism() {
        let decision = AdmissionDecision {
            schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
            ts_ns: 1706400000000000000,
            session_id: "sess_test".to_string(),
            signal_id: "book_imbalance".to_string(),
            outcome: AdmissionOutcome::Refuse,
            missing_vendor_fields: vec![VendorField::BuyQuantity],
            null_vendor_fields: vec![],
            missing_internal_fields: vec![],
            correlation_id: Some("corr_123".to_string()),
            digest: String::new(), // Not included in canonical bytes
        };

        let bytes1 = decision.canonical_bytes();
        let bytes2 = decision.canonical_bytes();

        assert_eq!(bytes1, bytes2, "Canonical bytes must be deterministic");

        let digest1 = compute_digest(&bytes1);
        let digest2 = compute_digest(&bytes2);

        assert_eq!(digest1, digest2, "Digest must be deterministic");
    }

    #[test]
    fn test_vendor_field_canonical_bytes_unique() {
        // Each variant must produce unique bytes
        let fields = [
            VendorField::BuyQuantity,
            VendorField::SellQuantity,
            VendorField::Volume,
            VendorField::LastQuantity,
            VendorField::LastPrice,
            VendorField::Ohlc,
            VendorField::BidPrice,
            VendorField::AskPrice,
        ];

        let bytes: Vec<Vec<u8>> = fields.iter().map(|f| f.canonical_bytes()).collect();

        for i in 0..bytes.len() {
            for j in (i + 1)..bytes.len() {
                assert_ne!(
                    bytes[i], bytes[j],
                    "Fields {:?} and {:?} have same canonical bytes",
                    fields[i], fields[j]
                );
            }
        }
    }

    #[test]
    fn test_venue_specific_canonical_bytes() {
        let f1 = VendorField::VenueSpecific {
            venue: "zerodha".to_string(),
            field: "oi".to_string(),
        };
        let f2 = VendorField::VenueSpecific {
            venue: "binance".to_string(),
            field: "oi".to_string(),
        };
        let f3 = VendorField::VenueSpecific {
            venue: "zerodha".to_string(),
            field: "ltp".to_string(),
        };

        assert_ne!(f1.canonical_bytes(), f2.canonical_bytes());
        assert_ne!(f1.canonical_bytes(), f3.canonical_bytes());
        assert_ne!(f2.canonical_bytes(), f3.canonical_bytes());
    }
}
