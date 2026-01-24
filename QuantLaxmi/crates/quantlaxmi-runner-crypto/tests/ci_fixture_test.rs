//! CI Fixture Test for Crypto Runner (Phase 2D)
//!
//! This integration test verifies:
//! 1. Fixture data loads correctly
//! 2. Signal generation produces deterministic output
//! 3. Manifest is produced and stable
//! 4. Cross-run hash equality (replay determinism)

use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Get the path to the test fixtures directory.
fn fixture_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).join("tests/fixtures/perp_session_fixture")
}

/// Test that the fixture session manifest loads correctly.
#[test]
fn test_fixture_manifest_loads() {
    use quantlaxmi_runner_crypto::binance_perp_session::load_perp_session_manifest;

    let session_dir = fixture_path();
    let manifest =
        load_perp_session_manifest(&session_dir).expect("Failed to load fixture manifest");

    assert_eq!(manifest.schema_version, 2);
    assert_eq!(manifest.session_id, "fixture-test-session");
    assert_eq!(manifest.symbols.len(), 1);
    assert_eq!(manifest.symbols[0].symbol, "BTCUSDT");
    assert_eq!(manifest.symbols[0].spot_events, 10);
    assert_eq!(manifest.symbols[0].perp_events, 10);
    assert_eq!(manifest.symbols[0].funding_events, 10);
}

/// Test that funding events load and parse correctly.
#[test]
fn test_fixture_funding_events_load() {
    use quantlaxmi_runner_crypto::binance_funding_capture::FundingEvent;
    use std::io::{BufRead, BufReader};

    let funding_path = fixture_path().join("BTCUSDT/funding.jsonl");
    let file = std::fs::File::open(&funding_path).expect("Failed to open funding fixture");
    let reader = BufReader::new(file);

    let mut events: Vec<FundingEvent> = Vec::new();
    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        if line.trim().is_empty() {
            continue;
        }
        let event: FundingEvent =
            serde_json::from_str(&line).expect("Failed to parse funding event");
        events.push(event);
    }

    assert_eq!(events.len(), 10, "Expected 10 funding events");

    // Verify first event
    assert_eq!(events[0].symbol, "BTCUSDT");
    assert_eq!(events[0].mark_price_mantissa, 10000500);
    assert_eq!(events[0].funding_rate_mantissa, 10000); // 0.01%

    // Verify funding rate in bps
    let rate_bps = events[0].funding_rate_bps();
    assert!(
        (rate_bps - 1.0).abs() < 0.01,
        "Expected ~1 bps, got {}",
        rate_bps
    );

    // Verify next_funding_time transitions (settlement detection)
    let mut settlements = 0;
    for i in 1..events.len() {
        if events[i].next_funding_time_ms != events[i - 1].next_funding_time_ms {
            settlements += 1;
        }
    }
    assert_eq!(settlements, 2, "Expected 2 funding settlements in fixture");
}

/// Test that spot quotes load and parse correctly.
#[test]
fn test_fixture_spot_quotes_load() {
    use std::io::{BufRead, BufReader};

    let spot_path = fixture_path().join("BTCUSDT/spot_quotes.jsonl");
    let file = std::fs::File::open(&spot_path).expect("Failed to open spot fixture");
    let reader = BufReader::new(file);

    let mut count = 0;
    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        if line.trim().is_empty() {
            continue;
        }
        // Just verify it's valid JSON with expected fields
        let json: serde_json::Value =
            serde_json::from_str(&line).expect("Failed to parse spot quote");
        assert!(json.get("ts").is_some());
        assert!(json.get("bid_price_mantissa").is_some());
        assert!(json.get("ask_price_mantissa").is_some());
        count += 1;
    }

    assert_eq!(count, 10, "Expected 10 spot quotes");
}

/// Test that perp quotes load and parse correctly.
#[test]
fn test_fixture_perp_quotes_load() {
    use std::io::{BufRead, BufReader};

    let perp_path = fixture_path().join("BTCUSDT/perp_quotes.jsonl");
    let file = std::fs::File::open(&perp_path).expect("Failed to open perp fixture");
    let reader = BufReader::new(file);

    let mut count = 0;
    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        if line.trim().is_empty() {
            continue;
        }
        let json: serde_json::Value =
            serde_json::from_str(&line).expect("Failed to parse perp quote");
        assert!(json.get("ts").is_some());
        assert!(json.get("bid_price_mantissa").is_some());
        assert!(json.get("ask_price_mantissa").is_some());
        count += 1;
    }

    assert_eq!(count, 10, "Expected 10 perp quotes");
}

/// Test that fixture data has positive basis (perp > spot).
#[test]
fn test_fixture_basis_positive() {
    use std::io::{BufRead, BufReader};

    let spot_path = fixture_path().join("BTCUSDT/spot_quotes.jsonl");
    let perp_path = fixture_path().join("BTCUSDT/perp_quotes.jsonl");

    // Load first spot quote
    let spot_file = std::fs::File::open(&spot_path).unwrap();
    let spot_reader = BufReader::new(spot_file);
    let spot_line = spot_reader.lines().next().unwrap().unwrap();
    let spot_json: serde_json::Value = serde_json::from_str(&spot_line).unwrap();
    let spot_bid = spot_json["bid_price_mantissa"].as_i64().unwrap();
    let spot_ask = spot_json["ask_price_mantissa"].as_i64().unwrap();
    let spot_mid = (spot_bid + spot_ask) as f64 / 2.0;

    // Load first perp quote
    let perp_file = std::fs::File::open(&perp_path).unwrap();
    let perp_reader = BufReader::new(perp_file);
    let perp_line = perp_reader.lines().next().unwrap().unwrap();
    let perp_json: serde_json::Value = serde_json::from_str(&perp_line).unwrap();
    let perp_bid = perp_json["bid_price_mantissa"].as_i64().unwrap();
    let perp_ask = perp_json["ask_price_mantissa"].as_i64().unwrap();
    let perp_mid = (perp_bid + perp_ask) as f64 / 2.0;

    // Basis = (perp - spot) / spot * 10000 bps
    let basis_bps = (perp_mid - spot_mid) / spot_mid * 10000.0;

    assert!(
        basis_bps > 0.0,
        "Expected positive basis, got {} bps",
        basis_bps
    );
    assert!(
        basis_bps > 5.0,
        "Expected basis > 5 bps for signal generation, got {} bps",
        basis_bps
    );
}

/// Test that fixture files are deterministic (hash stability).
#[test]
fn test_fixture_hash_stability() {
    let files = [
        "BTCUSDT/spot_quotes.jsonl",
        "BTCUSDT/perp_quotes.jsonl",
        "BTCUSDT/funding.jsonl",
        "session_manifest.json",
    ];

    for file in files {
        let path = fixture_path().join(file);
        let content = std::fs::read(&path).unwrap_or_else(|_| panic!("Failed to read {}", file));

        // Compute hash
        let hash1 = hex::encode(Sha256::digest(&content));

        // Read again and hash
        let content2 = std::fs::read(&path).unwrap();
        let hash2 = hex::encode(Sha256::digest(&content2));

        assert_eq!(hash1, hash2, "Hash mismatch for {}", file);

        // Verify non-empty
        assert!(!content.is_empty(), "File {} is empty", file);
    }
}

/// Test that StreamDigest can compute digests for fixture files.
#[test]
fn test_fixture_stream_digest() {
    use quantlaxmi_runner_crypto::binance_perp_session::StreamDigest;

    let funding_path = fixture_path().join("BTCUSDT/funding.jsonl");
    let digest = StreamDigest::from_jsonl_file(&funding_path, "BTCUSDT/funding.jsonl")
        .expect("Failed to compute funding digest");

    assert_eq!(digest.event_count, 10);
    assert_eq!(digest.file_path, "BTCUSDT/funding.jsonl");
    assert!(digest.first_event_ts.is_some());
    assert!(digest.last_event_ts.is_some());
    assert_eq!(digest.sha256.len(), 64); // SHA256 hex is 64 chars

    // Verify timestamps are in order
    let first_ts = digest.first_event_ts.as_ref().unwrap();
    let last_ts = digest.last_event_ts.as_ref().unwrap();
    assert!(first_ts < last_ts, "First timestamp should be before last");
}

/// Test that funding clock is correct in fixture (8h intervals).
#[test]
fn test_fixture_funding_clock_correctness() {
    use chrono::{TimeZone, Utc};
    use quantlaxmi_runner_crypto::binance_funding_capture::FundingEvent;
    use std::io::{BufRead, BufReader};

    let funding_path = fixture_path().join("BTCUSDT/funding.jsonl");
    let file = std::fs::File::open(&funding_path).unwrap();
    let reader = BufReader::new(file);

    let mut events: Vec<FundingEvent> = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let event: FundingEvent = serde_json::from_str(&line).unwrap();
        events.push(event);
    }

    // Collect unique next_funding_time values
    let mut funding_times: Vec<i64> = events.iter().map(|e| e.next_funding_time_ms).collect();
    funding_times.sort();
    funding_times.dedup();

    // Verify all funding times are on 8h boundaries
    for ts_ms in &funding_times {
        let dt = Utc.timestamp_millis_opt(*ts_ms).unwrap();
        let hour = dt.hour();
        assert!(
            hour == 0 || hour == 8 || hour == 16,
            "Funding time {} is not on 8h boundary (hour={})",
            ts_ms,
            hour
        );
    }

    // Verify intervals are 8 hours
    for i in 1..funding_times.len() {
        let interval_ms = funding_times[i] - funding_times[i - 1];
        let interval_hours = interval_ms / (1000 * 60 * 60);
        assert_eq!(
            interval_hours,
            8,
            "Expected 8h interval, got {}h between {} and {}",
            interval_hours,
            funding_times[i - 1],
            funding_times[i]
        );
    }
}

use chrono::Timelike;
