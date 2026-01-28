//! Signal Admission Controller implementation.
//!
//! Evaluates whether a signal may be computed based on data availability.
//! Does not inspect risk, session state, or execution context.

use quantlaxmi_models::{
    ADMISSION_SCHEMA_VERSION, AdmissionCanonicalBytes, AdmissionDecision, AdmissionOutcome,
    InternalField, SignalRequirements, VendorField, compute_digest,
};

// =============================================================================
// Snapshot Types — What data is available?
// =============================================================================

/// Snapshot of vendor data availability.
///
/// Each field indicates whether the vendor provided the value.
/// `Some(_)` = present (including `Some(0)`), `None` = vendor omitted.
#[derive(Debug, Clone, Default)]
pub struct VendorSnapshot {
    pub buy_quantity: Option<u64>,
    pub sell_quantity: Option<u64>,
    pub volume: Option<u64>,
    pub last_quantity: Option<u64>,
    pub last_price: Option<i64>,
    pub bid_price: Option<i64>,
    pub ask_price: Option<i64>,
    pub has_ohlc: bool,
    /// Venue-specific fields (key = "venue:field")
    pub venue_specific: std::collections::HashMap<String, bool>,
}

impl VendorSnapshot {
    /// Create an empty snapshot (all fields absent).
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if a vendor field is present.
    pub fn has_field(&self, field: &VendorField) -> bool {
        match field {
            VendorField::BuyQuantity => self.buy_quantity.is_some(),
            VendorField::SellQuantity => self.sell_quantity.is_some(),
            VendorField::Volume => self.volume.is_some(),
            VendorField::LastQuantity => self.last_quantity.is_some(),
            VendorField::LastPrice => self.last_price.is_some(),
            VendorField::BidPrice => self.bid_price.is_some(),
            VendorField::AskPrice => self.ask_price.is_some(),
            VendorField::Ohlc => self.has_ohlc,
            VendorField::VenueSpecific { venue, field } => {
                let key = format!("{}:{}", venue, field);
                self.venue_specific.get(&key).copied().unwrap_or(false)
            }
        }
    }
}

/// Snapshot of internal data availability.
///
/// Indicates whether internally-computed data is available.
/// Does NOT include session state (that's Phase 16).
#[derive(Debug, Clone, Default)]
pub struct InternalSnapshot {
    pub has_last_mid_price: bool,
    pub has_l1_book: bool,
    pub has_l2_book: bool,
    pub has_trade_tape: bool,
    /// Named features that are available
    pub available_features: std::collections::HashSet<String>,
}

impl InternalSnapshot {
    /// Create an empty snapshot (no internal data available).
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if an internal field is present.
    pub fn has_field(&self, field: &InternalField) -> bool {
        match field {
            InternalField::LastMidPrice => self.has_last_mid_price,
            InternalField::L1Book => self.has_l1_book,
            InternalField::L2Book => self.has_l2_book,
            InternalField::TradeTape => self.has_trade_tape,
            InternalField::Feature(name) => self.available_features.contains(name),
        }
    }
}

// =============================================================================
// Admission Context — Metadata for the decision
// =============================================================================

/// Context for an admission evaluation.
#[derive(Debug, Clone)]
pub struct AdmissionContext {
    /// Timestamp in nanoseconds
    pub ts_ns: i64,
    /// Session identifier
    pub session_id: String,
    /// Optional correlation to upstream decision/intent
    pub correlation_id: Option<String>,
}

impl AdmissionContext {
    pub fn new(ts_ns: i64, session_id: impl Into<String>) -> Self {
        Self {
            ts_ns,
            session_id: session_id.into(),
            correlation_id: None,
        }
    }

    pub fn with_correlation(mut self, id: impl Into<String>) -> Self {
        self.correlation_id = Some(id.into());
        self
    }
}

// =============================================================================
// Signal Admission Controller
// =============================================================================

/// Deterministic controller for signal admission decisions.
///
/// Evaluates whether a signal may be computed based on:
/// - Required vendor fields (must all be present)
/// - Required internal fields (must all be present)
///
/// Does NOT:
/// - Inspect risk, PnL, or exposure
/// - Check session state or kill-switches
/// - Fabricate or default missing values
pub struct SignalAdmissionController;

impl SignalAdmissionController {
    /// Evaluate admission for a signal.
    ///
    /// Returns an `AdmissionDecision` with:
    /// - `Admit` if all required fields are present
    /// - `Refuse` if any required field is missing
    ///
    /// The decision includes a deterministic digest for audit/replay.
    pub fn evaluate(
        requirements: &SignalRequirements,
        vendor_snapshot: &VendorSnapshot,
        internal_snapshot: &InternalSnapshot,
        ctx: AdmissionContext,
    ) -> AdmissionDecision {
        // Collect missing vendor fields
        let missing_vendor: Vec<VendorField> = requirements
            .required_vendor_fields
            .iter()
            .filter(|f| !vendor_snapshot.has_field(f))
            .cloned()
            .collect();

        // Collect missing internal fields
        let missing_internal: Vec<InternalField> = requirements
            .required_internal_fields
            .iter()
            .filter(|f| !internal_snapshot.has_field(f))
            .cloned()
            .collect();

        // Determine outcome
        let outcome = if missing_vendor.is_empty() && missing_internal.is_empty() {
            AdmissionOutcome::Admit
        } else {
            AdmissionOutcome::Refuse
        };

        // Build decision (without digest initially)
        let mut decision = AdmissionDecision {
            schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
            ts_ns: ctx.ts_ns,
            session_id: ctx.session_id,
            signal_id: requirements.signal_id.clone(),
            outcome,
            missing_vendor_fields: missing_vendor,
            null_vendor_fields: Vec::new(), // TODO: populate from VendorSnapshot with FieldState
            missing_internal_fields: missing_internal,
            correlation_id: ctx.correlation_id,
            digest: String::new(),
        };

        // Compute digest from canonical bytes
        let canonical = decision.canonical_bytes();
        decision.digest = compute_digest(&canonical);

        decision
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx() -> AdmissionContext {
        AdmissionContext::new(1706400000000000000, "test_session")
    }

    // -------------------------------------------------------------------------
    // Test: Missing vendor field → Refuse
    // -------------------------------------------------------------------------
    #[test]
    fn test_missing_vendor_field_refuse() {
        let requirements = SignalRequirements::new(
            "book_imbalance",
            vec![VendorField::BuyQuantity, VendorField::SellQuantity],
        );

        // Only buy_quantity present
        let vendor = VendorSnapshot {
            buy_quantity: Some(100),
            sell_quantity: None, // Missing!
            ..VendorSnapshot::empty()
        };

        let decision = SignalAdmissionController::evaluate(
            &requirements,
            &vendor,
            &InternalSnapshot::empty(),
            make_ctx(),
        );

        assert!(decision.is_refused());
        assert_eq!(
            decision.missing_vendor_fields,
            vec![VendorField::SellQuantity]
        );
        assert!(decision.missing_internal_fields.is_empty());
    }

    // -------------------------------------------------------------------------
    // Test: Missing internal field → Refuse
    // -------------------------------------------------------------------------
    #[test]
    fn test_missing_internal_field_refuse() {
        let requirements = SignalRequirements::new("mid_price_signal", vec![])
            .with_internal(vec![InternalField::LastMidPrice]);

        let internal = InternalSnapshot {
            has_last_mid_price: false, // Missing!
            ..InternalSnapshot::empty()
        };

        let decision = SignalAdmissionController::evaluate(
            &requirements,
            &VendorSnapshot::empty(),
            &internal,
            make_ctx(),
        );

        assert!(decision.is_refused());
        assert!(decision.missing_vendor_fields.is_empty());
        assert_eq!(
            decision.missing_internal_fields,
            vec![InternalField::LastMidPrice]
        );
    }

    // -------------------------------------------------------------------------
    // Test: Optional vendor missing → Admit
    // -------------------------------------------------------------------------
    #[test]
    fn test_optional_vendor_missing_admit() {
        let requirements = SignalRequirements::new(
            "book_imbalance",
            vec![VendorField::BuyQuantity, VendorField::SellQuantity],
        )
        .with_optional(vec![VendorField::Volume]); // Optional

        let vendor = VendorSnapshot {
            buy_quantity: Some(100),
            sell_quantity: Some(50),
            volume: None, // Missing but optional
            ..VendorSnapshot::empty()
        };

        let decision = SignalAdmissionController::evaluate(
            &requirements,
            &vendor,
            &InternalSnapshot::empty(),
            make_ctx(),
        );

        assert!(decision.is_admitted());
        assert!(decision.missing_vendor_fields.is_empty());
    }

    // -------------------------------------------------------------------------
    // Test: Some(0) vendor field → Admit (zero is valid)
    // -------------------------------------------------------------------------
    #[test]
    fn test_some_zero_is_admitted() {
        let requirements = SignalRequirements::new(
            "book_imbalance",
            vec![VendorField::BuyQuantity, VendorField::SellQuantity],
        );

        // Both present, but one is zero (vendor-asserted, valid!)
        let vendor = VendorSnapshot {
            buy_quantity: Some(0), // Vendor said "no buyers"
            sell_quantity: Some(100),
            ..VendorSnapshot::empty()
        };

        let decision = SignalAdmissionController::evaluate(
            &requirements,
            &vendor,
            &InternalSnapshot::empty(),
            make_ctx(),
        );

        assert!(
            decision.is_admitted(),
            "Some(0) must be admitted (L5: Zero Is Valid)"
        );
        assert!(decision.missing_vendor_fields.is_empty());
    }

    // -------------------------------------------------------------------------
    // Test: Deterministic digest
    // -------------------------------------------------------------------------
    #[test]
    fn test_deterministic_digest() {
        let requirements = SignalRequirements::new(
            "book_imbalance",
            vec![VendorField::BuyQuantity, VendorField::SellQuantity],
        );

        let vendor = VendorSnapshot {
            buy_quantity: Some(100),
            sell_quantity: None,
            ..VendorSnapshot::empty()
        };

        let ctx = make_ctx();

        let decision1 = SignalAdmissionController::evaluate(
            &requirements,
            &vendor,
            &InternalSnapshot::empty(),
            ctx.clone(),
        );
        let decision2 = SignalAdmissionController::evaluate(
            &requirements,
            &vendor,
            &InternalSnapshot::empty(),
            ctx,
        );

        assert_eq!(
            decision1.digest, decision2.digest,
            "Same inputs must produce same digest (L2)"
        );
    }

    // -------------------------------------------------------------------------
    // Test: No silent fabrication (doctrine compliance)
    // -------------------------------------------------------------------------
    #[test]
    fn test_no_silent_fabrication() {
        // This test verifies the doctrine: None never becomes a value.
        // The controller only checks presence; it does not transform data.

        let requirements = SignalRequirements::new(
            "book_imbalance",
            vec![VendorField::BuyQuantity, VendorField::SellQuantity],
        );

        let vendor = VendorSnapshot {
            buy_quantity: None,  // Missing
            sell_quantity: None, // Missing
            ..VendorSnapshot::empty()
        };

        let decision = SignalAdmissionController::evaluate(
            &requirements,
            &vendor,
            &InternalSnapshot::empty(),
            make_ctx(),
        );

        // Must refuse, not silently fabricate
        assert!(decision.is_refused());
        assert_eq!(decision.missing_vendor_fields.len(), 2);

        // Verify the missing fields are correctly reported
        assert!(
            decision
                .missing_vendor_fields
                .contains(&VendorField::BuyQuantity)
        );
        assert!(
            decision
                .missing_vendor_fields
                .contains(&VendorField::SellQuantity)
        );
    }

    // -------------------------------------------------------------------------
    // Test: All required present → Admit
    // -------------------------------------------------------------------------
    #[test]
    fn test_all_required_present_admit() {
        let requirements = SignalRequirements::new(
            "book_imbalance",
            vec![VendorField::BuyQuantity, VendorField::SellQuantity],
        )
        .with_internal(vec![InternalField::L1Book]);

        let vendor = VendorSnapshot {
            buy_quantity: Some(100),
            sell_quantity: Some(50),
            ..VendorSnapshot::empty()
        };

        let internal = InternalSnapshot {
            has_l1_book: true,
            ..InternalSnapshot::empty()
        };

        let decision =
            SignalAdmissionController::evaluate(&requirements, &vendor, &internal, make_ctx());

        assert!(decision.is_admitted());
        assert!(decision.missing_vendor_fields.is_empty());
        assert!(decision.missing_internal_fields.is_empty());
    }

    // -------------------------------------------------------------------------
    // Test: WAL write occurs for both Admit and Refuse (decision always produced)
    // -------------------------------------------------------------------------
    #[test]
    fn test_decision_always_produced() {
        let requirements = SignalRequirements::new("test_signal", vec![VendorField::BuyQuantity]);

        // Case 1: Admit
        let vendor_present = VendorSnapshot {
            buy_quantity: Some(100),
            ..VendorSnapshot::empty()
        };
        let admit = SignalAdmissionController::evaluate(
            &requirements,
            &vendor_present,
            &InternalSnapshot::empty(),
            make_ctx(),
        );
        assert!(admit.is_admitted());
        assert!(!admit.digest.is_empty(), "Admit decision must have digest");

        // Case 2: Refuse
        let vendor_missing = VendorSnapshot::empty();
        let refuse = SignalAdmissionController::evaluate(
            &requirements,
            &vendor_missing,
            &InternalSnapshot::empty(),
            make_ctx(),
        );
        assert!(refuse.is_refused());
        assert!(
            !refuse.digest.is_empty(),
            "Refuse decision must have digest"
        );

        // Both have schema version
        assert_eq!(admit.schema_version, ADMISSION_SCHEMA_VERSION);
        assert_eq!(refuse.schema_version, ADMISSION_SCHEMA_VERSION);
    }

    // -------------------------------------------------------------------------
    // Test: Venue-specific field handling
    // -------------------------------------------------------------------------
    #[test]
    fn test_venue_specific_field() {
        let requirements = SignalRequirements::new(
            "zerodha_oi_signal",
            vec![VendorField::VenueSpecific {
                venue: "zerodha".to_string(),
                field: "open_interest".to_string(),
            }],
        );

        // Missing venue-specific field
        let vendor_missing = VendorSnapshot::empty();
        let refuse = SignalAdmissionController::evaluate(
            &requirements,
            &vendor_missing,
            &InternalSnapshot::empty(),
            make_ctx(),
        );
        assert!(refuse.is_refused());

        // Present venue-specific field
        let mut vendor_present = VendorSnapshot::empty();
        vendor_present
            .venue_specific
            .insert("zerodha:open_interest".to_string(), true);
        let admit = SignalAdmissionController::evaluate(
            &requirements,
            &vendor_present,
            &InternalSnapshot::empty(),
            make_ctx(),
        );
        assert!(admit.is_admitted());
    }

    // -------------------------------------------------------------------------
    // Test: Feature dependency
    // -------------------------------------------------------------------------
    #[test]
    fn test_feature_dependency() {
        let requirements = SignalRequirements::new("vwap_signal", vec![])
            .with_internal(vec![InternalField::Feature("vwap".to_string())]);

        // Missing feature
        let internal_missing = InternalSnapshot::empty();
        let refuse = SignalAdmissionController::evaluate(
            &requirements,
            &VendorSnapshot::empty(),
            &internal_missing,
            make_ctx(),
        );
        assert!(refuse.is_refused());
        assert!(
            refuse
                .missing_internal_fields
                .contains(&InternalField::Feature("vwap".to_string()))
        );

        // Present feature
        let mut internal_present = InternalSnapshot::empty();
        internal_present
            .available_features
            .insert("vwap".to_string());
        let admit = SignalAdmissionController::evaluate(
            &requirements,
            &VendorSnapshot::empty(),
            &internal_present,
            make_ctx(),
        );
        assert!(admit.is_admitted());
    }
}
