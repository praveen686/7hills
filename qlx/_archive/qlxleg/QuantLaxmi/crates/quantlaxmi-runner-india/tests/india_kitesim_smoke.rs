//! India KiteSim Smoke Tests (Phase 27-IND)
//!
//! This integration test verifies:
//! 1. KiteSim backtest runs on fixture data
//! 2. Output is deterministic (identical across runs)
//! 3. Expected fields match (legs_filled, orders, pnl)
//!
//! Fixtures location: tests/fixtures/india_kitesim/phase27ind_smoke/

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::process::Command;

/// Get the workspace root directory.
fn workspace_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Get the path to the India KiteSim smoke fixtures.
fn fixture_path() -> PathBuf {
    workspace_root().join("tests/fixtures/india_kitesim/phase27ind_smoke")
}

/// Get the path to the quantlaxmi-india binary.
fn india_binary() -> PathBuf {
    workspace_root().join("target/release/quantlaxmi-india")
}

#[derive(Debug, Deserialize)]
struct PnlReport {
    orders: u64,
    legs_filled: u64,
    gross_mtm_inr: f64,
}

#[derive(Debug, Deserialize)]
struct BacktestReport {
    engine: String,
    venue: String,
    fill: FillMetrics,
    #[allow(dead_code)]
    notes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct FillMetrics {
    orders_total: u64,
    legs_total: u64,
    legs_filled: u64,
    #[allow(dead_code)]
    legs_rejected: u64,
    rollbacks: u64,
    timeouts: u64,
}

/// Run backtest-kitesim and return (report.json content, pnl.json content)
fn run_backtest(replay: &str, orders: &str, out_dir: &str) -> (String, String) {
    let binary = india_binary();
    if !binary.exists() {
        panic!(
            "Binary not found: {:?}. Run `cargo build --release` first.",
            binary
        );
    }

    let fixtures = fixture_path();
    let replay_path = fixtures.join(replay);
    let orders_path = fixtures.join(orders);

    // Clean output dir
    let out_path = PathBuf::from(out_dir);
    if out_path.exists() {
        std::fs::remove_dir_all(&out_path).expect("Failed to clean output dir");
    }

    let output = Command::new(&binary)
        .arg("backtest-kitesim")
        .arg("--replay")
        .arg(&replay_path)
        .arg("--orders")
        .arg(&orders_path)
        .arg("--out")
        .arg(out_dir)
        .arg("--latency-ms")
        .arg("150")
        .output()
        .expect("Failed to execute backtest-kitesim");

    if !output.status.success() {
        panic!(
            "backtest-kitesim failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let report =
        std::fs::read_to_string(out_path.join("report.json")).expect("Failed to read report.json");
    let pnl = std::fs::read_to_string(out_path.join("pnl.json")).expect("Failed to read pnl.json");

    (report, pnl)
}

/// Compute SHA256 hash of a string
fn sha256_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Test 1: ATM CE sanity test (single-leg)
#[test]
fn test_atm_ce_sanity() {
    let (report_json, pnl_json) = run_backtest(
        "quotes_atm_ce.jsonl",
        "orders_sanity_atm_ce.json",
        "/tmp/india_smoke_sanity",
    );

    let pnl: PnlReport = serde_json::from_str(&pnl_json).expect("Failed to parse pnl.json");
    let report: BacktestReport =
        serde_json::from_str(&report_json).expect("Failed to parse report.json");

    // Assert expected values
    assert_eq!(pnl.orders, 2, "Expected 2 orders");
    assert_eq!(pnl.legs_filled, 2, "Expected 2 legs filled");
    assert_eq!(report.fill.orders_total, 2);
    assert_eq!(report.fill.legs_total, 2);
    assert_eq!(report.fill.legs_filled, 2);
    assert_eq!(report.fill.rollbacks, 0, "Expected no rollbacks");
    assert_eq!(report.fill.timeouts, 0, "Expected no timeouts");

    // PnL should be negative (spread loss) and within expected range
    // Expected: -148.50 (tolerance: +/- 1.0 for rounding)
    assert!(
        pnl.gross_mtm_inr < 0.0,
        "Expected negative PnL for buy-then-sell"
    );
    assert!(
        (pnl.gross_mtm_inr - (-148.50)).abs() < 1.0,
        "PnL {} not within tolerance of -148.50",
        pnl.gross_mtm_inr
    );

    assert_eq!(report.engine, "KiteSim");
    assert_eq!(report.venue, "NSE-Zerodha-Sim");
}

/// Test 2: ATM straddle test (multi-leg)
#[test]
fn test_atm_straddle() {
    let (report_json, pnl_json) = run_backtest(
        "quotes_straddle.jsonl",
        "orders_straddle_test.json",
        "/tmp/india_smoke_straddle",
    );

    let pnl: PnlReport = serde_json::from_str(&pnl_json).expect("Failed to parse pnl.json");
    let report: BacktestReport =
        serde_json::from_str(&report_json).expect("Failed to parse report.json");

    // Assert expected values
    assert_eq!(pnl.orders, 2, "Expected 2 orders (open + close)");
    assert_eq!(pnl.legs_filled, 4, "Expected 4 legs filled (2 per order)");
    assert_eq!(report.fill.orders_total, 2);
    assert_eq!(report.fill.legs_total, 4);
    assert_eq!(report.fill.legs_filled, 4);
    assert_eq!(report.fill.rollbacks, 0, "Expected no rollbacks");
    assert_eq!(report.fill.timeouts, 0, "Expected no timeouts");

    // PnL should be negative and within expected range
    // Expected: -174.00 (tolerance: +/- 1.0 for rounding)
    assert!(
        pnl.gross_mtm_inr < 0.0,
        "Expected negative PnL for straddle round-trip"
    );
    assert!(
        (pnl.gross_mtm_inr - (-174.0)).abs() < 1.0,
        "PnL {} not within tolerance of -174.0",
        pnl.gross_mtm_inr
    );
}

/// Test 3: No lookahead bias - equity MTM must use live quotes, not future quotes
/// This test verifies that:
/// 1. Equity curve at time t uses only quotes with timestamp <= t
/// 2. Early equity point (before price jump) has MTM based on bid=100
/// 3. Final gross_mtm matches terminal quote mark (bid=200)
#[test]
fn test_no_lookahead_bias() {
    let binary = india_binary();
    if !binary.exists() {
        panic!(
            "Binary not found: {:?}. Run `cargo build --release` first.",
            binary
        );
    }

    let fixtures = workspace_root().join("tests/fixtures/india_kitesim/no_lookahead_test");
    let replay_path = fixtures.join("quotes_lookahead.jsonl");
    let orders_path = fixtures.join("orders_lookahead.json");
    let intents_path = fixtures.join("intents_lookahead.json");
    let out_dir = "/tmp/india_smoke_lookahead";

    // Clean output dir
    let out_path = PathBuf::from(out_dir);
    if out_path.exists() {
        std::fs::remove_dir_all(&out_path).expect("Failed to clean output dir");
    }

    // Use intent mode for proper equity curve with quote updates
    let output = Command::new(&binary)
        .arg("backtest-kitesim")
        .arg("--replay")
        .arg(&replay_path)
        .arg("--orders")
        .arg(&orders_path)
        .arg("--intents")
        .arg(&intents_path)
        .arg("--out")
        .arg(out_dir)
        .arg("--latency-ms")
        .arg("0")
        .output()
        .expect("Failed to execute backtest-kitesim");

    if !output.status.success() {
        panic!(
            "backtest-kitesim failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Read pnl.json
    let pnl_json =
        std::fs::read_to_string(out_path.join("pnl.json")).expect("Failed to read pnl.json");
    let pnl: serde_json::Value = serde_json::from_str(&pnl_json).expect("Failed to parse pnl.json");

    // Read equity_curve.jsonl and parse each line
    let equity_curve = std::fs::read_to_string(out_path.join("equity_curve.jsonl"))
        .expect("Failed to read equity_curve.jsonl");
    let equity_points: Vec<serde_json::Value> = equity_curve
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("Failed to parse equity point"))
        .collect();

    // Verify: terminal mark in pnl.json uses final quote (bid=200 for long position)
    let eod_marks = &pnl["eod_marks"]["NIFTY26FEB22000CE"];
    let terminal_mark = eod_marks["mark"].as_f64().expect("Missing terminal mark");

    // Terminal mark should be bid=200 (long position marked at bid)
    assert!(
        (terminal_mark - 200.0).abs() < 0.01,
        "Terminal mark {} should be 200 (terminal bid for long position)",
        terminal_mark
    );

    // Verify gross_mtm uses terminal mark: -6532.5 (cashflow) + 13000 (65*200) = 6467.5
    let gross_mtm = pnl["gross_mtm_inr"]
        .as_f64()
        .expect("Missing gross_mtm_inr");
    assert!(
        (gross_mtm - 6467.5).abs() < 1.0,
        "gross_mtm {} should be ~6467.5 (cashflow + terminal mark)",
        gross_mtm
    );

    // CRITICAL ASSERTION: NO lookahead in equity curve
    // The equity point after the fill should have mtm_inr based on bid=100
    // (the live quote at fill time), NOT bid=200 (the future quote).
    //
    // If lookahead bug exists: mtm_inr would be 65 * 200 = 13000
    // Correct behavior: mtm_inr = 65 * 100 = 6500
    let fill_equity_point = equity_points.iter().find(|pt| {
        let mtm = pt["mtm_inr"].as_f64().unwrap_or(0.0);
        // MTM should be around 6500 (using bid=100), not 13000 (using bid=200)
        mtm > 6000.0 && mtm < 7000.0
    });

    assert!(
        fill_equity_point.is_some(),
        "Must have equity point with MTM ~6500 (bid=100 at fill time), not 13000 (bid=200). \
         This proves NO LOOKAHEAD. Points: {:?}",
        equity_points
    );

    // Verify the fill equity point is using the correct mark
    let fill_pt = fill_equity_point.unwrap();
    let fill_mtm = fill_pt["mtm_inr"].as_f64().unwrap();
    assert!(
        (fill_mtm - 6500.0).abs() < 10.0,
        "Fill-time MTM {} should be ~6500 (65 * bid=100)",
        fill_mtm
    );

    // Note: equity_last != gross_mtm because equity curve only updates on order fills,
    // not on quote updates. This is expected behavior for intent mode.
    // The gross_mtm in pnl.json uses terminal marks (bid=200), while equity_last
    // uses the mark at last fill time (bid=100).
}

/// Test 4: Determinism - same input produces identical output (including equity curve)
/// Hashes equity_curve.jsonl and run_summary.json to verify byte-identical output
#[test]
fn test_deterministic_replay() {
    // Run sanity test twice
    let (_report1, pnl1) = run_backtest(
        "quotes_atm_ce.jsonl",
        "orders_sanity_atm_ce.json",
        "/tmp/india_smoke_det_run1",
    );

    let (_report2, pnl2) = run_backtest(
        "quotes_atm_ce.jsonl",
        "orders_sanity_atm_ce.json",
        "/tmp/india_smoke_det_run2",
    );

    // PnL must be byte-identical (determinism)
    assert_eq!(
        sha256_hash(&pnl1),
        sha256_hash(&pnl2),
        "PnL outputs differ between runs - replay is non-deterministic!"
    );

    // Parse and compare numerically as well
    let p1: PnlReport = serde_json::from_str(&pnl1).unwrap();
    let p2: PnlReport = serde_json::from_str(&pnl2).unwrap();

    assert_eq!(p1.orders, p2.orders);
    assert_eq!(p1.legs_filled, p2.legs_filled);
    assert!(
        (p1.gross_mtm_inr - p2.gross_mtm_inr).abs() < f64::EPSILON,
        "PnL values differ: {} vs {}",
        p1.gross_mtm_inr,
        p2.gross_mtm_inr
    );

    // Hash equity_curve.jsonl - must be identical across runs
    let eq1 = std::fs::read_to_string("/tmp/india_smoke_det_run1/equity_curve.jsonl")
        .expect("Failed to read equity_curve run1");
    let eq2 = std::fs::read_to_string("/tmp/india_smoke_det_run2/equity_curve.jsonl")
        .expect("Failed to read equity_curve run2");
    assert_eq!(
        sha256_hash(&eq1),
        sha256_hash(&eq2),
        "equity_curve.jsonl differs between runs - non-deterministic!"
    );

    // Hash run_summary.json (excluding git_commit which may vary)
    // Parse, remove volatile fields, re-serialize for comparison
    let sum1: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("/tmp/india_smoke_det_run1/run_summary.json")
            .expect("Failed to read run_summary run1"),
    )
    .expect("Parse run_summary1");
    let sum2: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("/tmp/india_smoke_det_run2/run_summary.json")
            .expect("Failed to read run_summary run2"),
    )
    .expect("Parse run_summary2");

    // Compare non-volatile fields
    assert_eq!(
        sum1["equity_first_inr"], sum2["equity_first_inr"],
        "equity_first_inr differs"
    );
    assert_eq!(
        sum1["equity_last_inr"], sum2["equity_last_inr"],
        "equity_last_inr differs"
    );
    assert_eq!(
        sum1["gross_mtm_inr"], sum2["gross_mtm_inr"],
        "gross_mtm_inr differs"
    );
    assert_eq!(
        sum1["equity_bars"], sum2["equity_bars"],
        "equity_bars differs"
    );
    assert_eq!(
        sum1["verification"], sum2["verification"],
        "verification differs"
    );
}
