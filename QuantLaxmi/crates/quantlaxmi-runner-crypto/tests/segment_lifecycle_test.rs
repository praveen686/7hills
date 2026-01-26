//! Phase 2B.2 CI Gate: Golden Segment Fixtures
//!
//! Validates segment manifest lifecycle invariants:
//! - Bootstrap manifests exist with correct schema
//! - Finalized segments have digests
//! - Retro-finalization produces correct state transitions

use quantlaxmi_runner_crypto::segment_manifest::{
    EventCounts, SEGMENT_MANIFEST_SCHEMA_VERSION, SegmentManifest, SegmentState,
    compute_segment_digests,
};
use std::path::Path;

const FIXTURES_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/segment_lifecycle"
);

/// Test that finalized segment fixture has correct structure
#[test]
fn test_finalized_segment_fixture() {
    let segment_dir = Path::new(FIXTURES_DIR).join("finalized_segment");
    assert!(
        segment_dir.exists(),
        "Finalized segment fixture missing: {:?}",
        segment_dir
    );

    let manifest = SegmentManifest::load(&segment_dir).expect("Failed to load finalized manifest");

    // Schema version must match
    assert_eq!(
        manifest.schema_version, SEGMENT_MANIFEST_SCHEMA_VERSION,
        "Schema version mismatch"
    );

    // Quote schema must be canonical_v1
    assert_eq!(
        manifest.quote_schema, "canonical_v1",
        "Quote schema mismatch"
    );

    // State must be FINALIZED
    assert_eq!(
        manifest.state,
        SegmentState::Finalized,
        "State should be FINALIZED"
    );

    // Must have digests
    assert!(
        manifest.digests.is_some(),
        "Finalized segment must have digests"
    );
    let digests = manifest.digests.as_ref().unwrap();

    // Perp digest should exist
    assert!(
        digests.perp.is_some(),
        "Finalized segment should have perp digest"
    );

    // Binary hash must exist
    assert!(
        !manifest.binary_hash.is_empty(),
        "Binary hash must not be empty"
    );

    // Config must exist
    assert!(manifest.config.price_exponent != 0 || manifest.config.qty_exponent != 0);

    // Duration should be set
    assert!(
        manifest.duration_secs.is_some(),
        "Finalized segment should have duration"
    );

    // End timestamp should be set
    assert!(
        manifest.end_ts.is_some(),
        "Finalized segment should have end_ts"
    );
}

/// Test that bootstrap segment fixture has correct structure
#[test]
fn test_bootstrap_segment_fixture() {
    let segment_dir = Path::new(FIXTURES_DIR).join("bootstrap_segment");
    assert!(
        segment_dir.exists(),
        "Bootstrap segment fixture missing: {:?}",
        segment_dir
    );

    let manifest = SegmentManifest::load(&segment_dir).expect("Failed to load bootstrap manifest");

    // Schema version must match
    assert_eq!(
        manifest.schema_version, SEGMENT_MANIFEST_SCHEMA_VERSION,
        "Schema version mismatch"
    );

    // Quote schema must be canonical_v1
    assert_eq!(
        manifest.quote_schema, "canonical_v1",
        "Quote schema mismatch"
    );

    // State must be BOOTSTRAP
    assert_eq!(
        manifest.state,
        SegmentState::Bootstrap,
        "State should be BOOTSTRAP"
    );

    // Should NOT have digests (bootstrap only)
    assert!(
        manifest.digests.is_none(),
        "Bootstrap segment should not have digests"
    );

    // Binary hash must exist
    assert!(
        !manifest.binary_hash.is_empty(),
        "Binary hash must not be empty"
    );

    // Stop reason should be RUNNING
    assert_eq!(
        manifest.stop_reason,
        quantlaxmi_runner_crypto::segment_manifest::StopReason::Running,
        "Bootstrap stop_reason should be RUNNING"
    );
}

/// Test retro-finalization of bootstrap segment
#[test]
fn test_retro_finalize_bootstrap_segment() {
    let segment_dir = Path::new(FIXTURES_DIR).join("bootstrap_segment");

    // Load the bootstrap manifest
    let mut manifest =
        SegmentManifest::load(&segment_dir).expect("Failed to load bootstrap manifest");

    assert_eq!(manifest.state, SegmentState::Bootstrap);

    // Compute digests from the fixture files
    let digests = compute_segment_digests(&segment_dir).expect("Failed to compute digests");

    // Count events from digests
    let events = EventCounts {
        spot_quotes: digests.spot.as_ref().map_or(0, |d| d.event_count),
        perp_quotes: digests.perp.as_ref().map_or(0, |d| d.event_count),
        funding: digests.funding.as_ref().map_or(0, |d| d.event_count),
        depth: digests.depth.as_ref().map_or(0, |d| d.event_count),
    };

    // Perform retro-finalization (in memory only - don't modify fixture)
    manifest.retro_finalize(events.clone(), digests);

    // Verify state transition
    assert_eq!(
        manifest.state,
        SegmentState::FinalizedRetro,
        "State should be FINALIZED_RETRO after retro_finalize"
    );

    // Verify digests are present
    assert!(
        manifest.digests.is_some(),
        "Digests should be present after retro_finalize"
    );

    // Verify stop_reason changed from RUNNING
    assert_eq!(
        manifest.stop_reason,
        quantlaxmi_runner_crypto::segment_manifest::StopReason::Unknown,
        "Stop reason should be UNKNOWN after retro_finalize"
    );

    // Verify end_ts is set
    assert!(
        manifest.end_ts.is_some(),
        "end_ts should be set after retro_finalize"
    );

    // Verify duration is computed
    assert!(
        manifest.duration_secs.is_some(),
        "duration_secs should be computed after retro_finalize"
    );

    // Verify event counts
    assert!(
        events.perp_quotes > 0,
        "Should have perp events from fixture"
    );
    assert!(
        events.spot_quotes > 0,
        "Should have spot events from fixture"
    );
}

/// Test digest computation produces valid SHA256
#[test]
fn test_digest_computation_validity() {
    let segment_dir = Path::new(FIXTURES_DIR).join("bootstrap_segment");

    let digests = compute_segment_digests(&segment_dir).expect("Failed to compute digests");

    // Perp digest should exist
    let perp = digests.perp.expect("Perp digest should exist");
    assert_eq!(perp.sha256.len(), 64, "SHA256 should be 64 hex chars");
    assert!(
        perp.sha256.chars().all(|c| c.is_ascii_hexdigit()),
        "SHA256 should be valid hex"
    );
    assert_eq!(perp.event_count, 3, "Should have 3 perp events");

    // Spot digest should exist
    let spot = digests.spot.expect("Spot digest should exist");
    assert_eq!(spot.sha256.len(), 64, "SHA256 should be 64 hex chars");
    assert_eq!(spot.event_count, 2, "Should have 2 spot events");

    // Funding should be None (no funding.jsonl in fixture)
    assert!(digests.funding.is_none(), "No funding file in fixture");
}

/// Test that schema version constant is correct
#[test]
fn test_schema_version_constant() {
    assert_eq!(
        SEGMENT_MANIFEST_SCHEMA_VERSION, 3,
        "Schema version should be 3 for Phase 2B"
    );
}

/// Test is_finalized helper
#[test]
fn test_is_finalized_helper() {
    let finalized_dir = Path::new(FIXTURES_DIR).join("finalized_segment");
    let bootstrap_dir = Path::new(FIXTURES_DIR).join("bootstrap_segment");

    let finalized = SegmentManifest::load(&finalized_dir).unwrap();
    let bootstrap = SegmentManifest::load(&bootstrap_dir).unwrap();

    assert!(finalized.is_finalized(), "FINALIZED should return true");
    assert!(!bootstrap.is_finalized(), "BOOTSTRAP should return false");
}
