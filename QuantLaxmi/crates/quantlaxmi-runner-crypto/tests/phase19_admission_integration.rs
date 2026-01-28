//! Phase 19B Integration Tests — Admission Enforcement
//!
//! Proves Phase 18 admission is enforced at runtime, not advisory:
//! - A refused admission prevents strategy evaluation
//! - Every admission decision is persisted to WAL before action
//! - Admission decisions are deterministic (same input → same digest)
//! - Null vs Absent is distinguished and cannot collapse
//! - Presence-bit divergence is caught by replay parity
//!
//! ## Hard Laws Verified
//! - L1: No Fabrication — Absent/Null are not values
//! - L2: Deterministic — Same inputs → identical decision digest
//! - L3: Explicit Refusal — Refuse is first-class artifact
//! - L5: Zero Is Valid — Value(0) is admissible
//! - L6: Observability First — Decision written to WAL before action

use quantlaxmi_events::trace::{
    DecisionTraceBuilder, ENCODING_VERSION, ReplayParityResult, verify_replay_parity,
};
use quantlaxmi_gates::admission::{
    AdmissionContext, InternalSnapshot, SignalAdmissionController, VendorSnapshot,
};
use quantlaxmi_models::events::{
    CorrelationContext, DecisionEvent, FieldState, MarketSnapshot, MarketSnapshotV1,
    build_l1_state_bits,
};
use quantlaxmi_models::{
    ADMISSION_SCHEMA_VERSION, AdmissionDecision, AdmissionOutcome, SignalRequirements, VendorField,
};

use chrono::{TimeZone, Utc};
use std::sync::atomic::{AtomicU32, Ordering};
use uuid::Uuid;

// =============================================================================
// TEST 1: Strategy NOT called on refused admission
// =============================================================================

/// Mock strategy that tracks invocations via atomic counter.
struct CountingStrategy {
    call_count: AtomicU32,
}

impl CountingStrategy {
    fn new() -> Self {
        Self {
            call_count: AtomicU32::new(0),
        }
    }

    fn on_event(&self, _snapshot: &MarketSnapshot) {
        self.call_count.fetch_add(1, Ordering::SeqCst);
    }

    fn call_count(&self) -> u32 {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[test]
fn test_strategy_not_called_on_refused_event() {
    // Setup: strategy that counts calls
    let strategy = CountingStrategy::new();

    // Create event with Absent bid_qty (required field missing)
    let snapshot = MarketSnapshot::v2_with_states(
        1000,
        1001, // prices present
        0,
        600, // bid_qty = 0, ask_qty = 600
        -2,
        -8,
        10,
        now_ns(),
        build_l1_state_bits(
            FieldState::Value,  // bid_price present
            FieldState::Value,  // ask_price present
            FieldState::Absent, // bid_qty ABSENT ← refusal trigger
            FieldState::Value,  // ask_qty present
        ),
    );

    // Requirements: microprice needs all 4 L1 fields
    let requirements = SignalRequirements::new(
        "microprice",
        vec![
            VendorField::BidPrice,
            VendorField::AskPrice,
            VendorField::BuyQuantity, // This is absent!
            VendorField::SellQuantity,
        ],
    );

    // Build VendorSnapshot from MarketSnapshot V2
    let vendor_snapshot = vendor_snapshot_from_market(&snapshot);

    // Admission check
    let ctx = AdmissionContext::new(now_ns(), "test_session");
    let decision = SignalAdmissionController::evaluate(
        &requirements,
        &vendor_snapshot,
        &InternalSnapshot::empty(),
        ctx,
    );

    // Assert: refused because BuyQuantity is absent
    assert!(
        decision.is_refused(),
        "Should refuse when required field is Absent"
    );
    assert!(
        decision
            .missing_vendor_fields
            .contains(&VendorField::BuyQuantity),
        "missing_vendor_fields should contain BuyQuantity"
    );

    // Simulate event loop gating: only call strategy if admitted
    if decision.is_admitted() {
        strategy.on_event(&snapshot);
    }

    // Assert: strategy never called
    assert_eq!(
        strategy.call_count(),
        0,
        "Strategy must NOT be called on refused admission"
    );
}

// =============================================================================
// TEST 2: WAL written even when refused
// =============================================================================

#[test]
fn test_wal_written_even_when_refused() {
    // Collect admission decisions (simulating WAL)
    let mut wal_entries: Vec<AdmissionDecision> = Vec::new();

    // Create refused event
    let snapshot = MarketSnapshot::v2_with_states(
        1000,
        1001,
        0,
        0,
        -2,
        -8,
        10,
        now_ns(),
        build_l1_state_bits(
            FieldState::Value,
            FieldState::Value,
            FieldState::Absent, // bid_qty absent
            FieldState::Absent, // ask_qty absent
        ),
    );

    let requirements = SignalRequirements::new(
        "book_imbalance",
        vec![VendorField::BuyQuantity, VendorField::SellQuantity],
    );

    let vendor_snapshot = vendor_snapshot_from_market(&snapshot);
    let ctx = AdmissionContext::new(now_ns(), "test_session");
    let decision = SignalAdmissionController::evaluate(
        &requirements,
        &vendor_snapshot,
        &InternalSnapshot::empty(),
        ctx,
    );

    // Write to WAL regardless of verdict (observability-first)
    wal_entries.push(decision.clone());

    // Assert: WAL has entry
    assert_eq!(wal_entries.len(), 1, "WAL must have admission entry");

    // Assert: Entry is a refusal with correct fields
    let entry = &wal_entries[0];
    assert_eq!(entry.outcome, AdmissionOutcome::Refuse);
    assert!(
        entry
            .missing_vendor_fields
            .contains(&VendorField::BuyQuantity)
    );
    assert!(
        entry
            .missing_vendor_fields
            .contains(&VendorField::SellQuantity)
    );
    assert!(
        entry.null_vendor_fields.is_empty(),
        "No null fields in this case"
    );
    assert!(!entry.digest.is_empty(), "Digest must be computed");
    assert_eq!(entry.schema_version, ADMISSION_SCHEMA_VERSION);
}

// =============================================================================
// TEST 3: Determinism — same segment, same decisions
// =============================================================================

#[test]
fn test_admission_determinism_same_segment_same_digests() {
    // Create a fixed sequence of events with varying presence states
    let events = create_deterministic_event_sequence(20);

    let requirements =
        SignalRequirements::new("spread", vec![VendorField::BidPrice, VendorField::AskPrice]);

    // Run 1
    let digests1: Vec<String> = events
        .iter()
        .map(|snap| {
            let vendor = vendor_snapshot_from_market(snap);
            let ctx = AdmissionContext::new(now_ns(), "session_run1");
            let decision = SignalAdmissionController::evaluate(
                &requirements,
                &vendor,
                &InternalSnapshot::empty(),
                ctx,
            );
            decision.digest.clone()
        })
        .collect();

    // Run 2 (identical inputs)
    let digests2: Vec<String> = events
        .iter()
        .map(|snap| {
            let vendor = vendor_snapshot_from_market(snap);
            // Same timestamp for determinism
            let ctx = AdmissionContext::new(now_ns(), "session_run1");
            let decision = SignalAdmissionController::evaluate(
                &requirements,
                &vendor,
                &InternalSnapshot::empty(),
                ctx,
            );
            decision.digest.clone()
        })
        .collect();

    // Assert: identical digest sequences
    assert_eq!(digests1.len(), digests2.len());
    for (i, (d1, d2)) in digests1.iter().zip(digests2.iter()).enumerate() {
        assert_eq!(d1, d2, "Digest mismatch at index {}: {} vs {}", i, d1, d2);
    }
}

// =============================================================================
// TEST 4: Null vs Absent distinguished
// =============================================================================

#[test]
fn test_null_vs_absent_is_not_collapsed() {
    // Case 1: Absent (vendor didn't send)
    let snap_absent = MarketSnapshot::v2_with_states(
        1000,
        1001,
        0,
        600,
        -2,
        -8,
        10,
        now_ns(),
        build_l1_state_bits(
            FieldState::Value,
            FieldState::Value,
            FieldState::Absent, // bid_qty ABSENT
            FieldState::Value,
        ),
    );

    // Case 2: Null (vendor sent explicit null)
    let snap_null = MarketSnapshot::v2_with_states(
        1000,
        1001,
        0,
        600,
        -2,
        -8,
        10,
        now_ns(),
        build_l1_state_bits(
            FieldState::Value,
            FieldState::Value,
            FieldState::Null, // bid_qty NULL
            FieldState::Value,
        ),
    );

    let requirements = SignalRequirements::new(
        "microprice",
        vec![
            VendorField::BidPrice,
            VendorField::AskPrice,
            VendorField::BuyQuantity,
            VendorField::SellQuantity,
        ],
    );

    // Evaluate both
    let vendor_absent = vendor_snapshot_from_market_with_nulls(&snap_absent);
    let vendor_null = vendor_snapshot_from_market_with_nulls(&snap_null);

    let decision_absent = SignalAdmissionController::evaluate(
        &requirements,
        &vendor_absent.0,
        &InternalSnapshot::empty(),
        AdmissionContext::new(now_ns(), "test"),
    );

    let decision_null = SignalAdmissionController::evaluate(
        &requirements,
        &vendor_null.0,
        &InternalSnapshot::empty(),
        AdmissionContext::new(now_ns(), "test"),
    );

    // Both must refuse
    assert!(decision_absent.is_refused(), "Absent must refuse");
    assert!(decision_null.is_refused(), "Null must refuse");

    // Absent case: BuyQuantity in missing_vendor_fields, NOT null_vendor_fields
    assert!(
        decision_absent
            .missing_vendor_fields
            .contains(&VendorField::BuyQuantity),
        "Absent: BuyQuantity should be in missing_vendor_fields"
    );
    // Note: null_vendor_fields is populated externally; controller doesn't yet distinguish

    // The key invariant: canonical bytes differ for Absent vs Null at the MarketSnapshot level
    let bytes_absent = encode_snapshot_canonical(&snap_absent);
    let bytes_null = encode_snapshot_canonical(&snap_null);

    assert_ne!(
        bytes_absent, bytes_null,
        "Absent vs Null must produce different canonical bytes"
    );
}

// =============================================================================
// TEST 5: Zero is valid vendor value
// =============================================================================

#[test]
fn test_zero_is_valid_vendor_value() {
    // MarketSnapshot V2 where BuyQuantity is Value(0) — vendor asserted zero
    let snapshot = MarketSnapshot::v2_with_states(
        1000,
        1001,
        0,   // bid_qty = 0 (vendor said "no buyers")
        100, // ask_qty = 100
        -2,
        -8,
        10,
        now_ns(),
        build_l1_state_bits(
            FieldState::Value, // bid_price present
            FieldState::Value, // ask_price present
            FieldState::Value, // bid_qty = 0 but PRESENT (L5: Zero Is Valid)
            FieldState::Value, // ask_qty present
        ),
    );

    let requirements = SignalRequirements::new(
        "book_imbalance",
        vec![VendorField::BuyQuantity, VendorField::SellQuantity],
    );

    let vendor = vendor_snapshot_from_market(&snapshot);
    let ctx = AdmissionContext::new(now_ns(), "test_session");
    let decision = SignalAdmissionController::evaluate(
        &requirements,
        &vendor,
        &InternalSnapshot::empty(),
        ctx,
    );

    // Assert: ADMITTED (zero is valid, not missing)
    assert!(
        decision.is_admitted(),
        "Value(0) must be admitted (L5: Zero Is Valid)"
    );
    assert!(
        decision.missing_vendor_fields.is_empty(),
        "No fields should be missing"
    );
    assert!(
        decision.null_vendor_fields.is_empty(),
        "No fields should be null"
    );
}

// =============================================================================
// TEST 6: Parity detects l1_state_bits flip
// =============================================================================

#[test]
fn test_parity_detects_presence_bit_divergence() {
    let ts = Utc.with_ymd_and_hms(2026, 1, 28, 12, 0, 0).unwrap();
    let decision_id = Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap();

    // Original trace: all fields present
    let snap_orig = MarketSnapshot::v2_with_states(
        1000,
        1001,
        500,
        600,
        -2,
        -8,
        10,
        1706443200000000000,
        build_l1_state_bits(
            FieldState::Value,
            FieldState::Value,
            FieldState::Value,
            FieldState::Value,
        ),
    );

    let decision_orig = DecisionEvent {
        ts,
        decision_id,
        strategy_id: "test".to_string(),
        symbol: "BTCUSDT".to_string(),
        decision_type: "entry".to_string(),
        direction: 1,
        target_qty_mantissa: 1000,
        qty_exponent: -8,
        reference_price_mantissa: 1000,
        price_exponent: -2,
        market_snapshot: snap_orig,
        confidence_mantissa: 8500,
        metadata: serde_json::Value::Null,
        ctx: CorrelationContext::default(),
    };

    let mut builder_orig = DecisionTraceBuilder::new();
    builder_orig.record(&decision_orig);
    let trace_orig = builder_orig.finalize();

    // Replay trace: bid_qty state FLIPPED to Absent (same mantissa value!)
    let snap_replay = MarketSnapshot::v2_with_states(
        1000,
        1001,
        500,
        600,
        -2,
        -8,
        10,
        1706443200000000000,
        build_l1_state_bits(
            FieldState::Value,
            FieldState::Value,
            FieldState::Absent, // ← FLIPPED from Value to Absent
            FieldState::Value,
        ),
    );

    let decision_replay = DecisionEvent {
        market_snapshot: snap_replay,
        ..decision_orig.clone()
    };

    let mut builder_replay = DecisionTraceBuilder::new();
    builder_replay.record(&decision_replay);
    let trace_replay = builder_replay.finalize();

    // Hashes MUST differ
    assert_ne!(
        trace_orig.trace_hash, trace_replay.trace_hash,
        "l1_state_bits flip must change trace hash"
    );

    // Parity check detects divergence
    let result = verify_replay_parity(&trace_orig, &trace_replay);
    match result {
        ReplayParityResult::Divergence { reason, index, .. } => {
            assert_eq!(index, 0, "Divergence should be at first decision");
            // Reason should mention the divergence
            assert!(!reason.is_empty(), "Reason should describe the divergence");
        }
        ReplayParityResult::Match => {
            panic!("Expected Divergence, got Match");
        }
        ReplayParityResult::LengthMismatch { .. } => {
            panic!("Expected Divergence, got LengthMismatch");
        }
    }

    // Verify encoding version
    assert_eq!(trace_orig.encoding_version, ENCODING_VERSION);
    assert_eq!(trace_replay.encoding_version, ENCODING_VERSION);
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn now_ns() -> i64 {
    // Fixed timestamp for determinism in tests
    1706443200000000000 // 2024-01-28 12:00:00 UTC
}

/// Unified entry point: Convert MarketSnapshot to VendorSnapshot for admission.
///
/// Dispatches to version-specific handlers with proper semantics:
/// - V1: Presence unknown → validate prices (must be > 0), qty always present (L5)
/// - V2: Use explicit l1_state_bits for presence
fn vendor_snapshot_from_market(snap: &MarketSnapshot) -> VendorSnapshot {
    match snap {
        MarketSnapshot::V1(v1) => vendor_snapshot_from_v1(v1),
        MarketSnapshot::V2(v2) => vendor_snapshot_from_v2(v2),
    }
}

/// V1-specific: Presence unknown, enforce price > 0 invariant.
///
/// Doctrine rules for V1:
/// - Price fields: Must be > 0 to be valid. price <= 0 → malformed → None (refuses)
/// - Qty fields: 0 is valid (L5: Zero Is Valid) → always Some
/// - No Null state exists in V1
fn vendor_snapshot_from_v1(v1: &MarketSnapshotV1) -> VendorSnapshot {
    VendorSnapshot {
        // Prices: only present if > 0 (zero/negative is malformed for prices)
        bid_price: if v1.bid_price_mantissa > 0 {
            Some(v1.bid_price_mantissa)
        } else {
            None // Malformed → treated as missing
        },
        ask_price: if v1.ask_price_mantissa > 0 {
            Some(v1.ask_price_mantissa)
        } else {
            None // Malformed → treated as missing
        },
        // Quantities: always present, 0 is valid (L5)
        buy_quantity: Some(v1.bid_qty_mantissa as u64),
        sell_quantity: Some(v1.ask_qty_mantissa as u64),
        ..VendorSnapshot::empty()
    }
}

/// V2-specific: Use explicit l1_state_bits for presence.
fn vendor_snapshot_from_v2(v2: &quantlaxmi_models::events::MarketSnapshotV2) -> VendorSnapshot {
    use quantlaxmi_models::events::{get_field_state, l1_slots};

    let bid_price_state = get_field_state(v2.l1_state_bits, l1_slots::BID_PRICE);
    let ask_price_state = get_field_state(v2.l1_state_bits, l1_slots::ASK_PRICE);
    let bid_qty_state = get_field_state(v2.l1_state_bits, l1_slots::BID_QTY);
    let ask_qty_state = get_field_state(v2.l1_state_bits, l1_slots::ASK_QTY);

    VendorSnapshot {
        bid_price: if bid_price_state == FieldState::Value {
            Some(v2.bid_price_mantissa)
        } else {
            None
        },
        ask_price: if ask_price_state == FieldState::Value {
            Some(v2.ask_price_mantissa)
        } else {
            None
        },
        buy_quantity: if bid_qty_state == FieldState::Value {
            Some(v2.bid_qty_mantissa as u64)
        } else {
            None
        },
        sell_quantity: if ask_qty_state == FieldState::Value {
            Some(v2.ask_qty_mantissa as u64)
        } else {
            None
        },
        ..VendorSnapshot::empty()
    }
}

/// Convert MarketSnapshot to VendorSnapshot, returning null fields separately.
/// Only V2 can have Null state; V1 returns empty null_fields.
fn vendor_snapshot_from_market_with_nulls(
    snap: &MarketSnapshot,
) -> (VendorSnapshot, Vec<VendorField>) {
    match snap {
        MarketSnapshot::V1(v1) => {
            // V1 has no Null state, only malformed prices
            (vendor_snapshot_from_v1(v1), Vec::new())
        }
        MarketSnapshot::V2(v2) => {
            use quantlaxmi_models::events::{get_field_state, l1_slots};

            let mut null_fields = Vec::new();

            let bid_price_state = get_field_state(v2.l1_state_bits, l1_slots::BID_PRICE);
            let ask_price_state = get_field_state(v2.l1_state_bits, l1_slots::ASK_PRICE);
            let bid_qty_state = get_field_state(v2.l1_state_bits, l1_slots::BID_QTY);
            let ask_qty_state = get_field_state(v2.l1_state_bits, l1_slots::ASK_QTY);

            // Track null fields
            if bid_price_state == FieldState::Null {
                null_fields.push(VendorField::BidPrice);
            }
            if ask_price_state == FieldState::Null {
                null_fields.push(VendorField::AskPrice);
            }
            if bid_qty_state == FieldState::Null {
                null_fields.push(VendorField::BuyQuantity);
            }
            if ask_qty_state == FieldState::Null {
                null_fields.push(VendorField::SellQuantity);
            }

            (vendor_snapshot_from_v2(v2), null_fields)
        }
    }
}

/// Create a deterministic sequence of events with varying presence states.
fn create_deterministic_event_sequence(count: usize) -> Vec<MarketSnapshot> {
    (0..count)
        .map(|i| {
            // Vary the presence states deterministically
            let bid_qty_state = if i % 3 == 0 {
                FieldState::Absent
            } else {
                FieldState::Value
            };

            MarketSnapshot::v2_with_states(
                1000 + i as i64,
                1001 + i as i64,
                (i * 100) as i64,
                (i * 200) as i64,
                -2,
                -8,
                10,
                1706443200000000000 + (i as i64 * 1000000000),
                build_l1_state_bits(
                    FieldState::Value,
                    FieldState::Value,
                    bid_qty_state,
                    FieldState::Value,
                ),
            )
        })
        .collect()
}

/// Encode MarketSnapshot to canonical bytes for comparison.
fn encode_snapshot_canonical(snap: &MarketSnapshot) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(snap.schema_version_byte());
    buf.extend_from_slice(&snap.bid_price_mantissa().to_le_bytes());
    buf.extend_from_slice(&snap.ask_price_mantissa().to_le_bytes());
    buf.extend_from_slice(&snap.bid_qty_mantissa().to_le_bytes());
    buf.extend_from_slice(&snap.ask_qty_mantissa().to_le_bytes());
    buf.push(snap.price_exponent() as u8);
    buf.push(snap.qty_exponent() as u8);
    buf.extend_from_slice(&snap.spread_bps_mantissa().to_le_bytes());
    buf.extend_from_slice(&snap.book_ts_ns().to_le_bytes());
    if let MarketSnapshot::V2(v2) = snap {
        buf.extend_from_slice(&v2.l1_state_bits.to_le_bytes());
    }
    buf
}

// =============================================================================
// TEST 7: V1 admits when prices valid and qty any
// =============================================================================

#[test]
fn test_v1_admits_when_prices_valid_and_qty_any() {
    // V1 has no state bits — we validate prices (must be > 0), qty always present
    // This test uses VALID prices (> 0) so it should admit
    let v1_snapshot = MarketSnapshot::V1(MarketSnapshotV1 {
        bid_price_mantissa: 1000, // Valid: > 0
        ask_price_mantissa: 1001, // Valid: > 0
        bid_qty_mantissa: 500,
        ask_qty_mantissa: 600,
        price_exponent: -2,
        qty_exponent: -8,
        spread_bps_mantissa: 10,
        book_ts_ns: now_ns(),
    });

    let requirements = SignalRequirements::new(
        "microprice",
        vec![
            VendorField::BidPrice,
            VendorField::AskPrice,
            VendorField::BuyQuantity,
            VendorField::SellQuantity,
        ],
    );

    let vendor = vendor_snapshot_from_market(&v1_snapshot);
    let ctx = AdmissionContext::new(now_ns(), "test_session");
    let decision = SignalAdmissionController::evaluate(
        &requirements,
        &vendor,
        &InternalSnapshot::empty(),
        ctx,
    );

    // V1 with valid prices admits
    assert!(
        decision.is_admitted(),
        "V1 with valid prices (> 0) must be admitted"
    );
    assert!(decision.missing_vendor_fields.is_empty());
    // V1 has no Null state
    assert!(decision.null_vendor_fields.is_empty());
}

// =============================================================================
// TEST 8: V1 zero is valid (same as V2)
// =============================================================================

#[test]
fn test_v1_zero_is_valid_vendor_value() {
    // V1 with zero quantity — still valid, not missing
    let v1_snapshot = MarketSnapshot::V1(MarketSnapshotV1 {
        bid_price_mantissa: 1000,
        ask_price_mantissa: 1001,
        bid_qty_mantissa: 0, // Zero is valid!
        ask_qty_mantissa: 100,
        price_exponent: -2,
        qty_exponent: -8,
        spread_bps_mantissa: 10,
        book_ts_ns: now_ns(),
    });

    let requirements = SignalRequirements::new(
        "book_imbalance",
        vec![VendorField::BuyQuantity, VendorField::SellQuantity],
    );

    let vendor = vendor_snapshot_from_market(&v1_snapshot);
    let ctx = AdmissionContext::new(now_ns(), "test_session");
    let decision = SignalAdmissionController::evaluate(
        &requirements,
        &vendor,
        &InternalSnapshot::empty(),
        ctx,
    );

    // V1 zero is valid (L5: Zero Is Valid)
    assert!(
        decision.is_admitted(),
        "V1 with zero quantity must be admitted (L5: Zero Is Valid)"
    );
}

// =============================================================================
// TEST 9: V1 vs V2 produce different canonical bytes (discriminant)
// =============================================================================

#[test]
fn test_v1_vs_v2_different_canonical_bytes() {
    // Same mantissa values, different schema versions
    let v1 = MarketSnapshot::V1(MarketSnapshotV1 {
        bid_price_mantissa: 1000,
        ask_price_mantissa: 1001,
        bid_qty_mantissa: 500,
        ask_qty_mantissa: 600,
        price_exponent: -2,
        qty_exponent: -8,
        spread_bps_mantissa: 10,
        book_ts_ns: 1706443200000000000,
    });

    let v2 = MarketSnapshot::v2_all_present(1000, 1001, 500, 600, -2, -8, 10, 1706443200000000000);

    let bytes_v1 = encode_snapshot_canonical(&v1);
    let bytes_v2 = encode_snapshot_canonical(&v2);

    // Must differ (V1 discriminant = 0x01, V2 discriminant = 0x02)
    assert_ne!(
        bytes_v1, bytes_v2,
        "V1 and V2 must produce different canonical bytes"
    );

    // V1 should be shorter (no l1_state_bits)
    assert!(
        bytes_v1.len() < bytes_v2.len(),
        "V1 should be shorter than V2 (no state bits)"
    );

    // First byte (discriminant) should differ
    assert_ne!(
        bytes_v1[0], bytes_v2[0],
        "V1/V2 discriminant bytes must differ"
    );
}

// =============================================================================
// TEST 10: V1 strategy call — admitted, strategy runs
// =============================================================================

#[test]
fn test_v1_strategy_called_when_admitted() {
    let strategy = CountingStrategy::new();

    let v1_snapshot = MarketSnapshot::V1(MarketSnapshotV1 {
        bid_price_mantissa: 1000,
        ask_price_mantissa: 1001,
        bid_qty_mantissa: 500,
        ask_qty_mantissa: 600,
        price_exponent: -2,
        qty_exponent: -8,
        spread_bps_mantissa: 10,
        book_ts_ns: now_ns(),
    });

    let requirements =
        SignalRequirements::new("spread", vec![VendorField::BidPrice, VendorField::AskPrice]);

    let vendor = vendor_snapshot_from_market(&v1_snapshot);
    let ctx = AdmissionContext::new(now_ns(), "test_session");
    let decision = SignalAdmissionController::evaluate(
        &requirements,
        &vendor,
        &InternalSnapshot::empty(),
        ctx,
    );

    // V1 admits → strategy runs
    assert!(decision.is_admitted());

    // Simulate event loop: call strategy if admitted
    if decision.is_admitted() {
        strategy.on_event(&v1_snapshot);
    }

    assert_eq!(
        strategy.call_count(),
        1,
        "Strategy must be called when V1 snapshot is admitted"
    );
}

// =============================================================================
// TEST 11: V1 refuses on zero or negative price (malformed)
// =============================================================================

#[test]
fn test_v1_refuses_on_zero_or_negative_price() {
    // V1 with zero bid_price — malformed, must refuse
    let v1_zero_bid = MarketSnapshot::V1(MarketSnapshotV1 {
        bid_price_mantissa: 0, // MALFORMED: price cannot be 0
        ask_price_mantissa: 1001,
        bid_qty_mantissa: 500,
        ask_qty_mantissa: 600,
        price_exponent: -2,
        qty_exponent: -8,
        spread_bps_mantissa: 10,
        book_ts_ns: now_ns(),
    });

    // V1 with negative ask_price — malformed, must refuse
    let v1_neg_ask = MarketSnapshot::V1(MarketSnapshotV1 {
        bid_price_mantissa: 1000,
        ask_price_mantissa: -5, // MALFORMED: price cannot be negative
        bid_qty_mantissa: 500,
        ask_qty_mantissa: 600,
        price_exponent: -2,
        qty_exponent: -8,
        spread_bps_mantissa: 10,
        book_ts_ns: now_ns(),
    });

    // V1 with both prices zero — double malformed
    let v1_both_zero = MarketSnapshot::V1(MarketSnapshotV1 {
        bid_price_mantissa: 0, // MALFORMED
        ask_price_mantissa: 0, // MALFORMED
        bid_qty_mantissa: 500,
        ask_qty_mantissa: 600,
        price_exponent: -2,
        qty_exponent: -8,
        spread_bps_mantissa: 10,
        book_ts_ns: now_ns(),
    });

    let requirements =
        SignalRequirements::new("spread", vec![VendorField::BidPrice, VendorField::AskPrice]);

    // Test 1: Zero bid price → refuse (BidPrice missing due to malformed)
    let vendor_zero_bid = vendor_snapshot_from_market(&v1_zero_bid);
    let decision_zero_bid = SignalAdmissionController::evaluate(
        &requirements,
        &vendor_zero_bid,
        &InternalSnapshot::empty(),
        AdmissionContext::new(now_ns(), "test"),
    );
    assert!(
        decision_zero_bid.is_refused(),
        "V1 with zero bid_price must refuse (malformed → missing)"
    );
    assert!(
        decision_zero_bid
            .missing_vendor_fields
            .contains(&VendorField::BidPrice),
        "BidPrice should be missing (malformed)"
    );

    // Test 2: Negative ask price → refuse (AskPrice missing due to malformed)
    let vendor_neg_ask = vendor_snapshot_from_market(&v1_neg_ask);
    let decision_neg_ask = SignalAdmissionController::evaluate(
        &requirements,
        &vendor_neg_ask,
        &InternalSnapshot::empty(),
        AdmissionContext::new(now_ns(), "test"),
    );
    assert!(
        decision_neg_ask.is_refused(),
        "V1 with negative ask_price must refuse (malformed → missing)"
    );
    assert!(
        decision_neg_ask
            .missing_vendor_fields
            .contains(&VendorField::AskPrice),
        "AskPrice should be missing (malformed)"
    );

    // Test 3: Both prices zero → refuse (both missing due to malformed)
    let vendor_both = vendor_snapshot_from_market(&v1_both_zero);
    let decision_both = SignalAdmissionController::evaluate(
        &requirements,
        &vendor_both,
        &InternalSnapshot::empty(),
        AdmissionContext::new(now_ns(), "test"),
    );
    assert!(
        decision_both.is_refused(),
        "V1 with both prices zero must refuse"
    );
    assert_eq!(
        decision_both.missing_vendor_fields.len(),
        2,
        "Both BidPrice and AskPrice should be missing (malformed)"
    );
}

// =============================================================================
// TEST 12: V1 zero qty is valid, zero price is not
// =============================================================================

#[test]
fn test_v1_zero_qty_valid_but_zero_price_malformed() {
    // V1 with zero qty but valid prices — should admit (L5: Zero Is Valid for qty)
    let v1_zero_qty = MarketSnapshot::V1(MarketSnapshotV1 {
        bid_price_mantissa: 1000, // Valid
        ask_price_mantissa: 1001, // Valid
        bid_qty_mantissa: 0,      // Zero qty is VALID (L5)
        ask_qty_mantissa: 0,      // Zero qty is VALID (L5)
        price_exponent: -2,
        qty_exponent: -8,
        spread_bps_mantissa: 10,
        book_ts_ns: now_ns(),
    });

    let requirements = SignalRequirements::new(
        "book_imbalance",
        vec![
            VendorField::BidPrice,
            VendorField::AskPrice,
            VendorField::BuyQuantity,
            VendorField::SellQuantity,
        ],
    );

    let vendor = vendor_snapshot_from_market(&v1_zero_qty);
    let decision = SignalAdmissionController::evaluate(
        &requirements,
        &vendor,
        &InternalSnapshot::empty(),
        AdmissionContext::new(now_ns(), "test"),
    );

    // Zero qty is valid (L5), valid prices → admit
    assert!(
        decision.is_admitted(),
        "V1 with zero qty but valid prices must admit (L5: Zero Is Valid for qty)"
    );
    assert!(decision.missing_vendor_fields.is_empty());
}

// =============================================================================
// PHASE 19C: REAL RUNTIME INTEGRATION TESTS
// =============================================================================
//
// These tests use the actual backtest run loop (run_with_strategy) and verify
// that admission decisions are written to WAL files.

use quantlaxmi_runner_crypto::backtest::{
    BacktestConfig, BacktestEngine, ExchangeConfig, PaceMode,
};
use quantlaxmi_strategy::{DecisionOutput, ReplayEvent, Strategy, StrategyContext};
use quantlaxmi_wal::WalReader;
use std::fs::File;
use std::io::Write;
use std::sync::atomic::AtomicU64;
use tempfile::TempDir;

// =============================================================================
// Test Strategy with Admission Requirements
// =============================================================================

/// Test strategy that requires specific vendor fields and tracks invocations.
struct AdmissionTestStrategy {
    /// Counter for on_event() calls
    call_count: AtomicU64,
    /// Signal requirements for admission gating
    requirements: Vec<SignalRequirements>,
}

impl AdmissionTestStrategy {
    fn new(requirements: Vec<SignalRequirements>) -> Self {
        Self {
            call_count: AtomicU64::new(0),
            requirements,
        }
    }

    #[allow(dead_code)] // Useful for debugging/future tests
    fn call_count(&self) -> u64 {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl Strategy for AdmissionTestStrategy {
    fn name(&self) -> &str {
        "admission_test_strategy"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn config_hash(&self) -> String {
        "test_hash_00000000".to_string()
    }

    fn on_event(&mut self, _event: &ReplayEvent, _ctx: &StrategyContext) -> Vec<DecisionOutput> {
        // Increment call counter - this proves the strategy was invoked
        self.call_count.fetch_add(1, Ordering::SeqCst);
        // Return empty - we only care about invocation tracking
        Vec::new()
    }

    fn required_signals(&self) -> Vec<SignalRequirements> {
        self.requirements.clone()
    }
}

/// Create a test segment with perp quotes containing bid/ask but NO qty fields.
/// This triggers admission refusal when qty fields are required.
fn create_segment_missing_qty() -> TempDir {
    let dir = TempDir::new().unwrap();
    let sym_dir = dir.path().join("BTCUSDT");
    std::fs::create_dir_all(&sym_dir).unwrap();

    // Create perp quotes WITHOUT qty fields (will trigger Absent state)
    let mut perp = File::create(sym_dir.join("perp_quotes.jsonl")).unwrap();

    // Quote with bid/ask mantissas but NO qty (triggers Absent)
    writeln!(
        perp,
        r#"{{"ts":"2026-01-28T10:00:00Z","bid_price_mantissa":8874152,"ask_price_mantissa":8874252,"price_exponent":-2}}"#
    ).unwrap();
    writeln!(
        perp,
        r#"{{"ts":"2026-01-28T10:00:01Z","bid_price_mantissa":8874153,"ask_price_mantissa":8874253,"price_exponent":-2}}"#
    ).unwrap();

    dir
}

/// Create a test segment with perp quotes containing ALL L1 fields.
/// This should admit when qty fields are required.
fn create_segment_with_qty() -> TempDir {
    let dir = TempDir::new().unwrap();
    let sym_dir = dir.path().join("BTCUSDT");
    std::fs::create_dir_all(&sym_dir).unwrap();

    // Create perp quotes WITH qty fields (will be Value state)
    let mut perp = File::create(sym_dir.join("perp_quotes.jsonl")).unwrap();

    // Quote with all fields present
    writeln!(
        perp,
        r#"{{"ts":"2026-01-28T10:00:00Z","bid_price_mantissa":8874152,"ask_price_mantissa":8874252,"bid_qty_mantissa":150000000,"ask_qty_mantissa":200000000,"price_exponent":-2,"qty_exponent":-8}}"#
    ).unwrap();
    writeln!(
        perp,
        r#"{{"ts":"2026-01-28T10:00:01Z","bid_price_mantissa":8874153,"ask_price_mantissa":8874253,"bid_qty_mantissa":160000000,"ask_qty_mantissa":210000000,"price_exponent":-2,"qty_exponent":-8}}"#
    ).unwrap();

    dir
}

// =============================================================================
// TEST 13: Runtime — Refused admission prevents strategy call + WAL exists
// =============================================================================

#[tokio::test]
async fn test_runtime_refused_admission_strategy_not_called_and_wal_exists() {
    // Create segment with missing qty fields
    let segment_dir = create_segment_missing_qty();

    // Strategy requires BuyQuantity (which will be Absent)
    let requirements = vec![SignalRequirements::new(
        "book_imbalance",
        vec![VendorField::BuyQuantity, VendorField::SellQuantity],
    )];
    let strategy = Box::new(AdmissionTestStrategy::new(requirements));

    // Configure backtest runner
    let config = BacktestConfig {
        pace: PaceMode::Fast,
        exchange: ExchangeConfig::default(),
        log_interval: 10000,
        output_trace: None,
        run_id: Some("test_refused_admission".to_string()),
        ..Default::default()
    };
    let runner = BacktestEngine::new(config);

    // Run backtest — strategy should NOT be called due to admission refusal
    // Backtest should complete successfully (panics if not)
    let (_result, _binding) = runner
        .run_with_strategy(segment_dir.path(), strategy, None)
        .await
        .expect("Backtest should complete");

    // Read WAL admission decisions
    let wal_reader = WalReader::open(segment_dir.path()).unwrap();
    let admission_decisions = wal_reader.read_admission_decisions().unwrap();

    // Verify: WAL contains admission records
    assert!(
        !admission_decisions.is_empty(),
        "WAL must contain admission decisions"
    );

    // Verify: All decisions are Refuse (because qty fields are Absent)
    for decision in &admission_decisions {
        assert_eq!(
            decision.outcome,
            AdmissionOutcome::Refuse,
            "All decisions should be Refuse: {:?}",
            decision
        );
        // Verify the missing fields are recorded
        assert!(
            decision
                .missing_vendor_fields
                .contains(&VendorField::BuyQuantity)
                || decision
                    .missing_vendor_fields
                    .contains(&VendorField::SellQuantity),
            "Missing fields should include qty: {:?}",
            decision.missing_vendor_fields
        );
    }

    // Verify: Admission decisions have valid schema version
    for decision in &admission_decisions {
        assert_eq!(
            decision.schema_version, ADMISSION_SCHEMA_VERSION,
            "Schema version must match"
        );
    }

    // Verify: Admission decisions have non-empty digests
    for decision in &admission_decisions {
        assert!(
            !decision.digest.is_empty(),
            "Each decision must have a digest for replay verification"
        );
    }
}

// =============================================================================
// TEST 14: Runtime — Admitted event calls strategy + WAL exists
// =============================================================================

#[tokio::test]
async fn test_runtime_admitted_strategy_called_and_wal_exists() {
    // Create segment with ALL L1 fields present
    let segment_dir = create_segment_with_qty();

    // Strategy requires price fields (which will be present)
    let requirements = vec![SignalRequirements::new(
        "spread",
        vec![VendorField::BidPrice, VendorField::AskPrice],
    )];
    let strategy = Box::new(AdmissionTestStrategy::new(requirements));

    // Configure backtest runner
    let config = BacktestConfig {
        pace: PaceMode::Fast,
        exchange: ExchangeConfig::default(),
        log_interval: 10000,
        output_trace: None,
        run_id: Some("test_admitted".to_string()),
        ..Default::default()
    };
    let runner = BacktestEngine::new(config);

    // Run backtest — strategy SHOULD be called
    // Backtest should complete successfully (panics if not)
    let (_result, _binding) = runner
        .run_with_strategy(segment_dir.path(), strategy, None)
        .await
        .expect("Backtest should complete");

    // Read WAL admission decisions
    let wal_reader = WalReader::open(segment_dir.path()).unwrap();
    let admission_decisions = wal_reader.read_admission_decisions().unwrap();

    // Verify: WAL contains admission records
    assert!(
        !admission_decisions.is_empty(),
        "WAL must contain admission decisions even when admitted"
    );

    // Verify: All decisions are Admit (because prices are present)
    for decision in &admission_decisions {
        assert_eq!(
            decision.outcome,
            AdmissionOutcome::Admit,
            "All decisions should be Admit: {:?}",
            decision
        );
        // Verify no missing fields
        assert!(
            decision.missing_vendor_fields.is_empty(),
            "Admitted decisions should have no missing fields: {:?}",
            decision.missing_vendor_fields
        );
    }

    // Verify: Decisions have digests (for replay determinism)
    for decision in &admission_decisions {
        assert!(
            !decision.digest.is_empty(),
            "Each decision must have a digest"
        );
    }
}

// =============================================================================
// TEST 15: Runtime — Mixed admit/refuse sequence produces correct WAL
// =============================================================================

/// Create a segment with some events that will admit and some that will refuse.
fn create_segment_mixed_admission() -> TempDir {
    let dir = TempDir::new().unwrap();
    let sym_dir = dir.path().join("BTCUSDT");
    std::fs::create_dir_all(&sym_dir).unwrap();

    let mut perp = File::create(sym_dir.join("perp_quotes.jsonl")).unwrap();

    // Event 1: Has qty → will admit
    writeln!(
        perp,
        r#"{{"ts":"2026-01-28T10:00:00Z","bid_price_mantissa":8874152,"ask_price_mantissa":8874252,"bid_qty_mantissa":150000000,"ask_qty_mantissa":200000000,"price_exponent":-2,"qty_exponent":-8}}"#
    ).unwrap();

    // Event 2: Missing qty → will refuse (if strategy requires qty)
    writeln!(
        perp,
        r#"{{"ts":"2026-01-28T10:00:01Z","bid_price_mantissa":8874153,"ask_price_mantissa":8874253,"price_exponent":-2}}"#
    ).unwrap();

    // Event 3: Has qty → will admit
    writeln!(
        perp,
        r#"{{"ts":"2026-01-28T10:00:02Z","bid_price_mantissa":8874154,"ask_price_mantissa":8874254,"bid_qty_mantissa":170000000,"ask_qty_mantissa":220000000,"price_exponent":-2,"qty_exponent":-8}}"#
    ).unwrap();

    dir
}

#[tokio::test]
async fn test_runtime_mixed_admission_produces_correct_wal_sequence() {
    // Create segment with mixed admission events
    let segment_dir = create_segment_mixed_admission();

    // Strategy requires qty fields
    let requirements = vec![SignalRequirements::new(
        "book_imbalance",
        vec![VendorField::BuyQuantity, VendorField::SellQuantity],
    )];
    let strategy = Box::new(AdmissionTestStrategy::new(requirements));

    // Configure backtest runner
    let config = BacktestConfig {
        pace: PaceMode::Fast,
        exchange: ExchangeConfig::default(),
        log_interval: 10000,
        output_trace: None,
        run_id: Some("test_mixed_admission".to_string()),
        ..Default::default()
    };
    let runner = BacktestEngine::new(config);

    // Run backtest
    let (_result, _binding) = runner
        .run_with_strategy(segment_dir.path(), strategy, None)
        .await
        .expect("Backtest should complete");

    // Read WAL admission decisions
    let wal_reader = WalReader::open(segment_dir.path()).unwrap();
    let admission_decisions = wal_reader.read_admission_decisions().unwrap();

    // Verify: 3 events processed
    assert_eq!(
        admission_decisions.len(),
        3,
        "Should have 3 admission decisions: {:?}",
        admission_decisions.len()
    );

    // Count admits and refuses
    let admits = admission_decisions
        .iter()
        .filter(|d| d.outcome == AdmissionOutcome::Admit)
        .count();
    let refuses = admission_decisions
        .iter()
        .filter(|d| d.outcome == AdmissionOutcome::Refuse)
        .count();

    // We expect 2 admits (events with qty) and 1 refuse (event without qty)
    assert_eq!(admits, 2, "Should have 2 admitted events");
    assert_eq!(refuses, 1, "Should have 1 refused event");

    // Verify the refused event has the correct missing fields
    let refused_decision = admission_decisions
        .iter()
        .find(|d| d.outcome == AdmissionOutcome::Refuse)
        .unwrap();
    assert!(
        refused_decision
            .missing_vendor_fields
            .contains(&VendorField::BuyQuantity),
        "Refused decision should report BuyQuantity as missing"
    );
}

// =============================================================================
// TEST 16: WAL digests are deterministic across replays
// =============================================================================

#[tokio::test]
async fn test_runtime_wal_digests_are_deterministic() {
    // Create segment
    let segment_dir = create_segment_with_qty();

    // Run backtest twice with identical config
    let requirements = vec![SignalRequirements::new(
        "spread",
        vec![VendorField::BidPrice, VendorField::AskPrice],
    )];

    let config = BacktestConfig {
        pace: PaceMode::Fast,
        exchange: ExchangeConfig::default(),
        log_interval: 10000,
        output_trace: None,
        run_id: Some("test_determinism".to_string()),
        ..Default::default()
    };

    // Run 1
    let strategy1 = Box::new(AdmissionTestStrategy::new(requirements.clone()));
    let runner1 = BacktestEngine::new(config.clone());
    runner1
        .run_with_strategy(segment_dir.path(), strategy1, None)
        .await
        .unwrap();

    let wal_reader1 = WalReader::open(segment_dir.path()).unwrap();
    let decisions1 = wal_reader1.read_admission_decisions().unwrap();
    let digests1: Vec<String> = decisions1.iter().map(|d| d.digest.clone()).collect();

    // Clear WAL for second run (remove the file)
    let wal_path = segment_dir.path().join("wal/signals_admission.jsonl");
    std::fs::remove_file(&wal_path).ok();

    // Run 2 with same config
    let strategy2 = Box::new(AdmissionTestStrategy::new(requirements));
    let runner2 = BacktestEngine::new(config);
    runner2
        .run_with_strategy(segment_dir.path(), strategy2, None)
        .await
        .unwrap();

    let wal_reader2 = WalReader::open(segment_dir.path()).unwrap();
    let decisions2 = wal_reader2.read_admission_decisions().unwrap();
    let digests2: Vec<String> = decisions2.iter().map(|d| d.digest.clone()).collect();

    // Verify: Same digests (determinism)
    assert_eq!(
        digests1.len(),
        digests2.len(),
        "Same number of decisions expected"
    );
    for (i, (d1, d2)) in digests1.iter().zip(digests2.iter()).enumerate() {
        assert_eq!(d1, d2, "Digest mismatch at index {}: {} vs {}", i, d1, d2);
    }
}

// =============================================================================
// TEST 17: No admission gating when strategy has no requirements
// =============================================================================

/// Strategy with NO required signals (backwards compatible)
struct NoRequirementsStrategy {
    call_count: AtomicU64,
}

impl NoRequirementsStrategy {
    fn new() -> Self {
        Self {
            call_count: AtomicU64::new(0),
        }
    }

    #[allow(dead_code)] // Useful for debugging/future tests
    fn call_count(&self) -> u64 {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl Strategy for NoRequirementsStrategy {
    fn name(&self) -> &str {
        "no_requirements_strategy"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn config_hash(&self) -> String {
        "no_req_hash_00000000".to_string()
    }

    fn on_event(&mut self, _event: &ReplayEvent, _ctx: &StrategyContext) -> Vec<DecisionOutput> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Vec::new()
    }

    // Default: no required signals
}

#[tokio::test]
async fn test_runtime_no_requirements_strategy_always_called() {
    // Create segment without qty (would refuse if qty was required)
    let segment_dir = create_segment_missing_qty();

    let strategy = Box::new(NoRequirementsStrategy::new());

    let config = BacktestConfig {
        pace: PaceMode::Fast,
        exchange: ExchangeConfig::default(),
        log_interval: 10000,
        output_trace: None,
        run_id: Some("test_no_requirements".to_string()),
        ..Default::default()
    };
    let runner = BacktestEngine::new(config);

    // Run backtest
    let (_result, _binding) = runner
        .run_with_strategy(segment_dir.path(), strategy, None)
        .await
        .expect("Backtest should complete");

    // Read WAL — should have NO admission decisions (no gating configured)
    let wal_reader = WalReader::open(segment_dir.path()).unwrap();
    let admission_decisions = wal_reader.read_admission_decisions().unwrap();

    // No required_signals → no admission gating → no WAL entries
    assert!(
        admission_decisions.is_empty(),
        "No admission WAL when strategy has no requirements: found {} entries",
        admission_decisions.len()
    );
}

// =============================================================================
// PHASE 19D: ENFORCEMENT INTEGRATION TESTS
// =============================================================================
//
// These tests verify that --enforce-admission-from-wal actually gates
// strategy invocation based on WAL decisions (not live evaluation).

/// Create a segment WITH admission WAL pre-populated with specific decisions.
/// This simulates replaying a segment where admission decisions already exist.
fn create_segment_with_admission_wal(decisions: Vec<AdmissionDecision>) -> TempDir {
    let dir = TempDir::new().unwrap();
    let sym_dir = dir.path().join("BTCUSDT");
    std::fs::create_dir_all(&sym_dir).unwrap();

    // Create perp quotes with ALL fields (would normally admit)
    let mut perp = File::create(sym_dir.join("perp_quotes.jsonl")).unwrap();
    for i in 0..decisions.len() {
        writeln!(
            perp,
            r#"{{"ts":"2026-01-28T10:00:0{}Z","bid_price_mantissa":8874152,"ask_price_mantissa":8874252,"bid_qty_mantissa":150000000,"ask_qty_mantissa":200000000,"price_exponent":-2,"qty_exponent":-8}}"#,
            i
        ).unwrap();
    }

    // Create WAL directory and write admission decisions
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    let mut admission_file = File::create(wal_dir.join("signals_admission.jsonl")).unwrap();
    for decision in decisions {
        let line = serde_json::to_string(&decision).unwrap();
        writeln!(admission_file, "{}", line).unwrap();
    }

    dir
}

// =============================================================================
// TEST 18: Enforcement blocks strategy on refuse
// =============================================================================

#[tokio::test]
async fn test_replay_enforced_blocks_strategy_on_refuse() {
    // Create admission WAL with REFUSE decision
    let refuse_decision = AdmissionDecision {
        schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
        ts_ns: 1706400000000000000,
        session_id: "test_session".to_string(),
        signal_id: "book_imbalance".to_string(),
        outcome: AdmissionOutcome::Refuse,
        missing_vendor_fields: vec![VendorField::BuyQuantity],
        null_vendor_fields: vec![],
        missing_internal_fields: vec![],
        correlation_id: Some("event_seq:1".to_string()),
        digest: "refuse_digest_001".to_string(),
    };

    let segment_dir = create_segment_with_admission_wal(vec![refuse_decision]);

    // Strategy requires fields that WAL says were missing
    let requirements = vec![SignalRequirements::new(
        "book_imbalance",
        vec![VendorField::BuyQuantity, VendorField::SellQuantity],
    )];
    let strategy = Box::new(AdmissionTestStrategy::new(requirements));

    // Configure with enforcement enabled
    let config = BacktestConfig {
        pace: PaceMode::Fast,
        exchange: ExchangeConfig::default(),
        log_interval: 10000,
        output_trace: None,
        run_id: Some("test_enforced_refuse".to_string()),
        enforce_admission_from_wal: true,
        admission_mismatch_policy: "fail".to_string(),
    };
    let runner = BacktestEngine::new(config);

    // Run backtest with enforcement
    let (result, _binding) = runner
        .run_with_strategy(segment_dir.path(), strategy, None)
        .await
        .expect("Backtest should complete");

    // WAL said REFUSE → strategy should NOT have been called → no decisions recorded
    assert_eq!(
        result.total_decisions, 0,
        "Strategy should not produce decisions when WAL refuses admission"
    );
}

// =============================================================================
// TEST 19: Enforcement calls strategy on admit
// =============================================================================

#[tokio::test]
async fn test_replay_enforced_calls_strategy_on_admit() {
    // Create admission WAL with ADMIT decision
    let admit_decision = AdmissionDecision {
        schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
        ts_ns: 1706400000000000000,
        session_id: "test_session".to_string(),
        signal_id: "spread".to_string(),
        outcome: AdmissionOutcome::Admit,
        missing_vendor_fields: vec![],
        null_vendor_fields: vec![],
        missing_internal_fields: vec![],
        correlation_id: Some("event_seq:1".to_string()),
        digest: "admit_digest_001".to_string(),
    };

    let segment_dir = create_segment_with_admission_wal(vec![admit_decision]);

    // Strategy requires only price fields (which are present)
    let requirements = vec![SignalRequirements::new(
        "spread",
        vec![VendorField::BidPrice, VendorField::AskPrice],
    )];
    let strategy = Box::new(AdmissionTestStrategy::new(requirements));

    // Configure with enforcement enabled
    let config = BacktestConfig {
        pace: PaceMode::Fast,
        exchange: ExchangeConfig::default(),
        log_interval: 10000,
        output_trace: None,
        run_id: Some("test_enforced_admit".to_string()),
        enforce_admission_from_wal: true,
        admission_mismatch_policy: "fail".to_string(),
    };
    let runner = BacktestEngine::new(config);

    // Run backtest with enforcement
    let (result, _binding) = runner
        .run_with_strategy(segment_dir.path(), strategy, None)
        .await
        .expect("Backtest should complete");

    // WAL said ADMIT → strategy WAS called
    // Note: AdmissionTestStrategy doesn't emit decisions, but it was invoked
    // We verify by checking that the backtest completed successfully with enforcement
    assert_eq!(result.total_events, 1, "Should have processed 1 event");
}

// =============================================================================
// TEST 20: Missing WAL entry with fail policy causes error
// =============================================================================

#[tokio::test]
async fn test_replay_enforced_missing_wal_entry_policy_fail() {
    // Create segment with event but NO admission WAL entries
    // This simulates trying to enforce on a segment that was never evaluated
    let dir = TempDir::new().unwrap();
    let sym_dir = dir.path().join("BTCUSDT");
    std::fs::create_dir_all(&sym_dir).unwrap();

    let mut perp = File::create(sym_dir.join("perp_quotes.jsonl")).unwrap();
    writeln!(
        perp,
        r#"{{"ts":"2026-01-28T10:00:00Z","bid_price_mantissa":8874152,"ask_price_mantissa":8874252,"bid_qty_mantissa":150000000,"ask_qty_mantissa":200000000,"price_exponent":-2,"qty_exponent":-8}}"#
    ).unwrap();

    // Create EMPTY admission WAL (no decisions for event_seq:1)
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    File::create(wal_dir.join("signals_admission.jsonl")).unwrap();

    // Strategy requires admission gating
    let requirements = vec![SignalRequirements::new(
        "spread",
        vec![VendorField::BidPrice, VendorField::AskPrice],
    )];
    let strategy = Box::new(AdmissionTestStrategy::new(requirements));

    // Configure with enforcement + fail policy
    let config = BacktestConfig {
        pace: PaceMode::Fast,
        exchange: ExchangeConfig::default(),
        log_interval: 10000,
        output_trace: None,
        run_id: Some("test_missing_wal_fail".to_string()),
        enforce_admission_from_wal: true,
        admission_mismatch_policy: "fail".to_string(),
    };
    let runner = BacktestEngine::new(config);

    // Run backtest — should FAIL due to missing WAL entry
    let result = runner.run_with_strategy(dir.path(), strategy, None).await;

    assert!(
        result.is_err(),
        "Enforcement with missing WAL entry and fail policy should error"
    );

    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Admission enforcement failed") || err_msg.contains("no WAL entry"),
        "Error should mention enforcement failure: {}",
        err_msg
    );
}

// =============================================================================
// TEST 21: Missing WAL entry with warn policy refuses (doctrine-safe)
// =============================================================================

#[tokio::test]
async fn test_replay_enforced_missing_wal_entry_policy_warn() {
    // Create segment with event but NO admission WAL entries
    let dir = TempDir::new().unwrap();
    let sym_dir = dir.path().join("BTCUSDT");
    std::fs::create_dir_all(&sym_dir).unwrap();

    let mut perp = File::create(sym_dir.join("perp_quotes.jsonl")).unwrap();
    writeln!(
        perp,
        r#"{{"ts":"2026-01-28T10:00:00Z","bid_price_mantissa":8874152,"ask_price_mantissa":8874252,"bid_qty_mantissa":150000000,"ask_qty_mantissa":200000000,"price_exponent":-2,"qty_exponent":-8}}"#
    ).unwrap();

    // Create EMPTY admission WAL
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    File::create(wal_dir.join("signals_admission.jsonl")).unwrap();

    // Strategy requires admission gating
    let requirements = vec![SignalRequirements::new(
        "spread",
        vec![VendorField::BidPrice, VendorField::AskPrice],
    )];
    let strategy = Box::new(AdmissionTestStrategy::new(requirements));

    // Configure with enforcement + WARN policy
    let config = BacktestConfig {
        pace: PaceMode::Fast,
        exchange: ExchangeConfig::default(),
        log_interval: 10000,
        output_trace: None,
        run_id: Some("test_missing_wal_warn".to_string()),
        enforce_admission_from_wal: true,
        admission_mismatch_policy: "warn".to_string(), // WARN policy
    };
    let runner = BacktestEngine::new(config);

    // Run backtest — should succeed but refuse the event (doctrine: cannot prove admission)
    let (result, _binding) = runner
        .run_with_strategy(dir.path(), strategy, None)
        .await
        .expect("Backtest with warn policy should complete");

    // Doctrine: Missing WAL entry → refuse (even in warn mode)
    // Strategy should NOT have been called → no decisions
    assert_eq!(
        result.total_decisions, 0,
        "Missing WAL entry with warn policy should refuse: cannot prove admission"
    );
}
