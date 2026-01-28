//! SANOS I/O Helpers (Commit D: Manifest-Driven Mode)
//!
//! Provides shared infrastructure for SANOS binaries to operate in
//! manifest-driven mode when `session_manifest.json` exists.
//!
//! ## Modes
//! - **Manifest-driven**: Uses `session_manifest.json` + `universe_manifest.json`
//!   for deterministic inventory. No directory scanning or symbol parsing.
//! - **Legacy**: Falls back to directory scanning + symbol parsing when no manifest.

use anyhow::{Context, Result};
use chrono::NaiveDate;
use quantlaxmi_connectors_zerodha::UniverseManifest;
use quantlaxmi_runner_common::{
    TickOutputEntry, UnderlyingEntry, load_session_manifest, load_universe_manifest_bytes,
    session_manifest_exists,
};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tracing::info;

// =============================================================================
// MANIFEST INVENTORY TYPES
// =============================================================================

/// Complete manifest-driven inventory for a SANOS session.
#[derive(Debug, Clone)]
pub struct SanosManifestInventory {
    /// Root session directory
    pub session_dir: PathBuf,
    /// Session ID from manifest
    pub session_id: String,
    /// Per-underlying inventories
    pub underlyings: Vec<SanosUnderlyingInventory>,
}

/// Manifest-driven inventory for a single underlying.
#[derive(Debug, Clone)]
pub struct SanosUnderlyingInventory {
    /// Underlying symbol (e.g., "NIFTY", "BANKNIFTY")
    pub underlying: String,
    /// Subdirectory relative to session root (e.g., "nifty/")
    pub underlying_subdir: String,
    /// SHA-256 hash of universe manifest (audit trail)
    pub universe_sha256: String,
    /// T1 expiry date (ISO format string)
    pub expiry_t1: String,
    /// T2 expiry date if available
    pub expiry_t2: Option<String>,
    /// T3 expiry date if available
    pub expiry_t3: Option<String>,
    /// Strike step for this underlying
    pub strike_step: f64,
    /// Tick output files for this underlying: (symbol, rel_path, ticks_written, has_depth)
    pub tick_outputs: Vec<TickOutputInfo>,
    /// Instruments grouped by expiry date (from UniverseManifest)
    pub instruments_by_expiry: HashMap<NaiveDate, Vec<InstrumentInfo>>,
}

/// Tick output file info.
#[derive(Debug, Clone)]
pub struct TickOutputInfo {
    pub symbol: String,
    pub path: String,
    pub ticks_written: usize,
    pub has_depth: bool,
}

/// Instrument info from UniverseManifest.
#[derive(Debug, Clone)]
pub struct InstrumentInfo {
    pub tradingsymbol: String,
    pub instrument_token: u32,
    pub expiry: NaiveDate,
    pub strike: f64,
    pub instrument_type: String, // "CE" / "PE"
}

// =============================================================================
// MANIFEST LOADING
// =============================================================================

/// Try to load SANOS inventory from session manifest.
///
/// Returns `Ok(Some(inventory))` if `session_manifest.json` exists and is valid.
/// Returns `Ok(None)` if manifest does not exist (legacy mode should be used).
/// Returns `Err` if manifest exists but cannot be parsed.
pub fn try_load_sanos_inventory(session_dir: &Path) -> Result<Option<SanosManifestInventory>> {
    if !session_manifest_exists(session_dir) {
        return Ok(None);
    }

    let sm = load_session_manifest(session_dir)?;
    let mut underlyings = Vec::new();

    for ue in &sm.underlyings {
        let underlying_inv = load_underlying_inventory(session_dir, ue, &sm.tick_outputs)?;
        underlyings.push(underlying_inv);
    }

    Ok(Some(SanosManifestInventory {
        session_dir: session_dir.to_path_buf(),
        session_id: sm.session_id.clone(),
        underlyings,
    }))
}

/// Load inventory for a single underlying from its universe manifest.
fn load_underlying_inventory(
    session_dir: &Path,
    ue: &UnderlyingEntry,
    all_tick_outputs: &[TickOutputEntry],
) -> Result<SanosUnderlyingInventory> {
    // Load universe manifest
    let um_bytes = load_universe_manifest_bytes(session_dir, &ue.universe_manifest_path)?;
    let um: UniverseManifest = serde_json::from_slice(&um_bytes).with_context(|| {
        format!(
            "Failed to parse UniverseManifest: {}",
            ue.universe_manifest_path
        )
    })?;

    // Build symbol set for this underlying (normalized to uppercase)
    let symbol_set: HashSet<String> = um
        .instruments
        .iter()
        .map(|i| i.tradingsymbol.to_uppercase())
        .collect();

    // Filter tick outputs to those belonging to this underlying (normalize keys)
    let mut tick_outputs: Vec<TickOutputInfo> = all_tick_outputs
        .iter()
        .filter(|te| symbol_set.contains(&te.symbol.to_uppercase()))
        .map(|te| TickOutputInfo {
            symbol: te.symbol.to_uppercase(), // Normalize to uppercase
            path: te.path.clone(),
            ticks_written: te.ticks_written,
            has_depth: te.has_depth,
        })
        .collect();

    // Sort tick outputs for deterministic ordering
    tick_outputs.sort_by(|a, b| a.symbol.cmp(&b.symbol));

    // Group instruments by expiry
    let mut instruments_by_expiry: HashMap<NaiveDate, Vec<InstrumentInfo>> = HashMap::new();
    for instr in &um.instruments {
        let info = InstrumentInfo {
            tradingsymbol: instr.tradingsymbol.to_uppercase(), // Normalize
            instrument_token: instr.instrument_token,
            expiry: instr.expiry,
            strike: instr.strike,
            instrument_type: instr.instrument_type.to_uppercase(), // Normalize
        };
        instruments_by_expiry
            .entry(instr.expiry)
            .or_default()
            .push(info);
    }

    // Sort instruments within each expiry for deterministic ordering
    // Order by: (strike, instrument_type, tradingsymbol)
    for instruments in instruments_by_expiry.values_mut() {
        instruments.sort_by(|a, b| {
            a.strike
                .partial_cmp(&b.strike)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.instrument_type.cmp(&b.instrument_type))
                .then_with(|| a.tradingsymbol.cmp(&b.tradingsymbol))
        });
    }

    Ok(SanosUnderlyingInventory {
        underlying: ue.underlying.clone(),
        underlying_subdir: ue.subdir.clone(),
        universe_sha256: ue.universe_manifest_sha256.clone(),
        expiry_t1: ue.t1_expiry.clone(),
        expiry_t2: ue.t2_expiry.clone(),
        expiry_t3: ue.t3_expiry.clone(),
        strike_step: ue.strike_step,
        tick_outputs,
        instruments_by_expiry,
    })
}

// =============================================================================
// EXPIRY HELPERS
// =============================================================================

impl SanosUnderlyingInventory {
    /// Get all expiry dates sorted chronologically.
    pub fn get_sorted_expiries(&self) -> Vec<NaiveDate> {
        let mut expiries: Vec<NaiveDate> = self.instruments_by_expiry.keys().copied().collect();
        expiries.sort();
        expiries
    }

    /// Get instruments for a specific expiry.
    pub fn get_instruments_for_expiry(&self, expiry: NaiveDate) -> Vec<&InstrumentInfo> {
        self.instruments_by_expiry
            .get(&expiry)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Get symbols for a specific expiry.
    pub fn get_symbols_for_expiry(&self, expiry: NaiveDate) -> Vec<String> {
        self.instruments_by_expiry
            .get(&expiry)
            .map(|v| v.iter().map(|i| i.tradingsymbol.clone()).collect())
            .unwrap_or_default()
    }

    /// Get tick output path for a symbol (case-insensitive lookup).
    pub fn get_tick_path(&self, symbol: &str) -> Option<&str> {
        let normalized = symbol.to_uppercase();
        self.tick_outputs
            .iter()
            .find(|t| t.symbol == normalized)
            .map(|t| t.path.as_str())
    }

    /// Parse T1 expiry as NaiveDate.
    pub fn parse_t1_expiry(&self) -> Result<NaiveDate> {
        NaiveDate::parse_from_str(&self.expiry_t1, "%Y-%m-%d")
            .with_context(|| format!("Failed to parse T1 expiry: {}", self.expiry_t1))
    }

    /// Parse T2 expiry as NaiveDate (if present).
    pub fn parse_t2_expiry(&self) -> Result<Option<NaiveDate>> {
        match &self.expiry_t2 {
            Some(s) => Ok(Some(
                NaiveDate::parse_from_str(s, "%Y-%m-%d")
                    .with_context(|| format!("Failed to parse T2 expiry: {}", s))?,
            )),
            None => Ok(None),
        }
    }

    /// Parse T3 expiry as NaiveDate (if present).
    pub fn parse_t3_expiry(&self) -> Result<Option<NaiveDate>> {
        match &self.expiry_t3 {
            Some(s) => Ok(Some(
                NaiveDate::parse_from_str(s, "%Y-%m-%d")
                    .with_context(|| format!("Failed to parse T3 expiry: {}", s))?,
            )),
            None => Ok(None),
        }
    }
}

// =============================================================================
// LOGGING (Audit-grade)
// =============================================================================

/// Log manifest-driven mode activation for audit trail.
pub fn log_manifest_mode(inv: &SanosManifestInventory) {
    info!(
        session_id = %inv.session_id,
        session_dir = %inv.session_dir.display(),
        underlying_count = inv.underlyings.len(),
        "SANOS manifest-driven mode activated"
    );

    for u in &inv.underlyings {
        info!(
            underlying = %u.underlying,
            universe_manifest_sha256 = %u.universe_sha256,
            instrument_count = u.instruments_by_expiry.values().map(|v| v.len()).sum::<usize>(),
            tick_file_count = u.tick_outputs.len(),
            t1 = %u.expiry_t1,
            t2 = ?u.expiry_t2,
            t3 = ?u.expiry_t3,
            strike_step = u.strike_step,
            "Underlying inventory loaded from manifest"
        );
    }
}

/// Log legacy mode activation (fallback when no manifest).
pub fn log_legacy_mode(session_dir: &Path) {
    info!(
        session_dir = %session_dir.display(),
        "SANOS legacy mode (no session_manifest.json found)"
    );
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_no_manifest_returns_none() {
        // Create a temp directory without session_manifest.json
        let tmp = std::env::temp_dir().join("sanos_io_test_no_manifest");
        let _ = fs::remove_dir_all(&tmp); // Clean up from previous runs
        fs::create_dir_all(&tmp).unwrap();

        let result = try_load_sanos_inventory(&tmp).unwrap();
        assert!(result.is_none());

        let _ = fs::remove_dir_all(&tmp);
    }

    /// Guardrail Test B: Golden inventory determinism snapshot
    /// Verifies that manifest loading produces stable, sorted output
    #[test]
    fn test_inventory_determinism() {
        let tmp = std::env::temp_dir().join("sanos_io_test_determinism");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::create_dir_all(tmp.join("nifty")).unwrap();

        // Create universe manifest with multiple expiries (intentionally unsorted)
        let universe = serde_json::json!({
            "underlying": "NIFTY",
            "today": "2026-01-24",
            "spot": 24750.0,
            "atm": 24750.0,
            "strike_step": 50.0,
            "strike_band": 10,
            "expiry_selection": {
                "t1": "2026-01-30",
                "t2": "2026-02-26",
                "t3": null,
                "selected": ["2026-01-30", "2026-02-26"]
            },
            "target_strikes": [24500.0, 25000.0],
            "instruments": [
                {"tradingsymbol": "NIFTY26FEB25000CE", "instrument_token": 1001, "underlying": "NIFTY", "expiry": "2026-02-26", "strike": 25000.0, "instrument_type": "CE", "lot_size": 50},
                {"tradingsymbol": "NIFTY26JAN24500PE", "instrument_token": 1002, "underlying": "NIFTY", "expiry": "2026-01-30", "strike": 24500.0, "instrument_type": "PE", "lot_size": 50},
                {"tradingsymbol": "NIFTY26JAN25000CE", "instrument_token": 1003, "underlying": "NIFTY", "expiry": "2026-01-30", "strike": 25000.0, "instrument_type": "CE", "lot_size": 50},
                {"tradingsymbol": "NIFTY26FEB24500PE", "instrument_token": 1004, "underlying": "NIFTY", "expiry": "2026-02-26", "strike": 24500.0, "instrument_type": "PE", "lot_size": 50}
            ],
            "missing": {}
        });
        let universe_path = tmp.join("nifty/universe_manifest.json");
        fs::write(
            &universe_path,
            serde_json::to_string_pretty(&universe).unwrap(),
        )
        .unwrap();

        // Compute SHA256 of universe manifest
        let universe_bytes = fs::read(&universe_path).unwrap();
        let universe_sha = quantlaxmi_runner_common::manifest_io::sha256_hex(&universe_bytes);

        // Create session manifest (tick_outputs intentionally unsorted)
        let session = serde_json::json!({
            "schema_version": 1,
            "created_at_utc": "2026-01-24T10:00:00Z",
            "session_id": "test-determinism-001",
            "capture_mode": "india_capture",
            "out_dir": tmp.to_string_lossy(),
            "duration_secs": 300,
            "price_exponent": -2,
            "underlyings": [{
                "underlying": "NIFTY",
                "subdir": "nifty/",
                "universe_manifest_path": "nifty/universe_manifest.json",
                "universe_manifest_sha256": universe_sha,
                "instrument_count": 4,
                "t1_expiry": "2026-01-30",
                "t2_expiry": "2026-02-26",
                "t3_expiry": null,
                "strike_step": 50.0
            }],
            "tick_outputs": [
                {"symbol": "NIFTY26FEB24500PE", "path": "nifty/NIFTY26FEB24500PE/ticks.jsonl", "ticks_written": 10, "has_depth": false},
                {"symbol": "NIFTY26JAN25000CE", "path": "nifty/NIFTY26JAN25000CE/ticks.jsonl", "ticks_written": 20, "has_depth": false},
                {"symbol": "NIFTY26JAN24500PE", "path": "nifty/NIFTY26JAN24500PE/ticks.jsonl", "ticks_written": 15, "has_depth": false},
                {"symbol": "NIFTY26FEB25000CE", "path": "nifty/NIFTY26FEB25000CE/ticks.jsonl", "ticks_written": 5, "has_depth": false}
            ],
            "integrity": {
                "out_of_universe_ticks_dropped": 0,
                "subscribe_mode": "manifest_tokens",
                "notes": []
            }
        });
        fs::write(
            tmp.join("session_manifest.json"),
            serde_json::to_string_pretty(&session).unwrap(),
        )
        .unwrap();

        // Load inventory
        let inventory = try_load_sanos_inventory(&tmp)
            .unwrap()
            .expect("Should load manifest");

        // Verify session ID
        assert_eq!(inventory.session_id, "test-determinism-001");

        // Get underlying inventory
        assert_eq!(inventory.underlyings.len(), 1);
        let u_inv = &inventory.underlyings[0];
        assert_eq!(u_inv.underlying, "NIFTY");
        assert_eq!(u_inv.universe_sha256, universe_sha);

        // Verify expiries are sorted chronologically
        let expiries = u_inv.get_sorted_expiries();
        assert_eq!(expiries.len(), 2);
        assert_eq!(expiries[0].to_string(), "2026-01-30"); // T1 comes first
        assert_eq!(expiries[1].to_string(), "2026-02-26"); // T2 comes second

        // Verify instruments per expiry
        let t1_instruments = u_inv.get_instruments_for_expiry(expiries[0]);
        assert_eq!(t1_instruments.len(), 2); // 2 instruments for Jan expiry

        let t2_instruments = u_inv.get_instruments_for_expiry(expiries[1]);
        assert_eq!(t2_instruments.len(), 2); // 2 instruments for Feb expiry

        // Verify tick outputs are present
        assert_eq!(u_inv.tick_outputs.len(), 4);

        // Verify tick path lookup works
        assert!(u_inv.get_tick_path("NIFTY26JAN25000CE").is_some());
        assert!(u_inv.get_tick_path("NIFTY26FEB24500PE").is_some());
        assert!(u_inv.get_tick_path("NONEXISTENT").is_none());

        // Verify expiry parsing
        let t1 = u_inv.parse_t1_expiry().unwrap();
        assert_eq!(t1.to_string(), "2026-01-30");

        let t2 = u_inv.parse_t2_expiry().unwrap().unwrap();
        assert_eq!(t2.to_string(), "2026-02-26");

        let t3 = u_inv.parse_t3_expiry().unwrap();
        assert!(t3.is_none());

        // Cleanup
        let _ = fs::remove_dir_all(&tmp);
    }

    /// Guardrail Test: Verify expiry ordering is stable across multiple loads
    #[test]
    fn test_expiry_ordering_stability() {
        let tmp = std::env::temp_dir().join("sanos_io_test_ordering");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::create_dir_all(tmp.join("nifty")).unwrap();

        // Create universe with 3 expiries in random order
        let universe = serde_json::json!({
            "underlying": "NIFTY",
            "today": "2026-01-24",
            "spot": 24750.0,
            "atm": 24750.0,
            "strike_step": 50.0,
            "strike_band": 10,
            "expiry_selection": {
                "t1": "2026-01-30",
                "t2": "2026-02-26",
                "t3": "2026-03-26",
                "selected": ["2026-01-30", "2026-02-26", "2026-03-26"]
            },
            "target_strikes": [25000.0],
            "instruments": [
                {"tradingsymbol": "NIFTY26MAR25000CE", "instrument_token": 3001, "underlying": "NIFTY", "expiry": "2026-03-26", "strike": 25000.0, "instrument_type": "CE", "lot_size": 50},
                {"tradingsymbol": "NIFTY26JAN25000CE", "instrument_token": 1001, "underlying": "NIFTY", "expiry": "2026-01-30", "strike": 25000.0, "instrument_type": "CE", "lot_size": 50},
                {"tradingsymbol": "NIFTY26FEB25000CE", "instrument_token": 2001, "underlying": "NIFTY", "expiry": "2026-02-26", "strike": 25000.0, "instrument_type": "CE", "lot_size": 50}
            ],
            "missing": {}
        });
        let universe_path = tmp.join("nifty/universe_manifest.json");
        fs::write(
            &universe_path,
            serde_json::to_string_pretty(&universe).unwrap(),
        )
        .unwrap();

        let universe_bytes = fs::read(&universe_path).unwrap();
        let universe_sha = quantlaxmi_runner_common::manifest_io::sha256_hex(&universe_bytes);

        let session = serde_json::json!({
            "schema_version": 1,
            "created_at_utc": "2026-01-24T10:00:00Z",
            "session_id": "test-ordering-001",
            "capture_mode": "india_capture",
            "out_dir": tmp.to_string_lossy(),
            "duration_secs": 300,
            "price_exponent": -2,
            "underlyings": [{
                "underlying": "NIFTY",
                "subdir": "nifty/",
                "universe_manifest_path": "nifty/universe_manifest.json",
                "universe_manifest_sha256": universe_sha,
                "instrument_count": 3,
                "t1_expiry": "2026-01-30",
                "t2_expiry": "2026-02-26",
                "t3_expiry": "2026-03-26",
                "strike_step": 50.0
            }],
            "tick_outputs": [],
            "integrity": {
                "out_of_universe_ticks_dropped": 0,
                "subscribe_mode": "manifest_tokens",
                "notes": []
            }
        });
        fs::write(
            tmp.join("session_manifest.json"),
            serde_json::to_string_pretty(&session).unwrap(),
        )
        .unwrap();

        // Load multiple times and verify ordering is stable
        for _ in 0..3 {
            let inventory = try_load_sanos_inventory(&tmp)
                .unwrap()
                .expect("Should load");
            let u_inv = &inventory.underlyings[0];
            let expiries = u_inv.get_sorted_expiries();

            // Must always be in chronological order
            assert_eq!(expiries.len(), 3);
            assert_eq!(expiries[0].to_string(), "2026-01-30"); // Jan
            assert_eq!(expiries[1].to_string(), "2026-02-26"); // Feb
            assert_eq!(expiries[2].to_string(), "2026-03-26"); // Mar
        }

        let _ = fs::remove_dir_all(&tmp);
    }

    /// Guardrail Test A: Manifest mode structural isolation
    /// Verifies that manifest-driven inventory provides all required data without
    /// relying on directory scanning or symbol parsing.
    ///
    /// This test ensures:
    /// 1. Inventory contains complete instrument info (tradingsymbol, expiry, strike, type)
    /// 2. Tick paths are explicit (no directory scanning needed)
    /// 3. Expiry dates come from manifest (not parsed from symbols)
    /// 4. Universe SHA256 is captured for audit trail
    #[test]
    fn test_manifest_mode_structural_isolation() {
        let tmp = std::env::temp_dir().join("sanos_io_test_isolation");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::create_dir_all(tmp.join("nifty")).unwrap();

        // Create universe manifest with full instrument info
        let universe = serde_json::json!({
            "underlying": "NIFTY",
            "today": "2026-01-24",
            "spot": 24750.0,
            "atm": 24750.0,
            "strike_step": 50.0,
            "strike_band": 10,
            "expiry_selection": {
                "t1": "2026-01-30",
                "t2": null,
                "t3": null,
                "selected": ["2026-01-30"]
            },
            "target_strikes": [24750.0],
            "instruments": [
                {
                    "tradingsymbol": "NIFTY26JAN24750CE",
                    "instrument_token": 1001,
                    "underlying": "NIFTY",
                    "expiry": "2026-01-30",
                    "strike": 24750.0,
                    "instrument_type": "CE",
                    "lot_size": 50
                },
                {
                    "tradingsymbol": "NIFTY26JAN24750PE",
                    "instrument_token": 1002,
                    "underlying": "NIFTY",
                    "expiry": "2026-01-30",
                    "strike": 24750.0,
                    "instrument_type": "PE",
                    "lot_size": 50
                }
            ],
            "missing": {}
        });
        let universe_path = tmp.join("nifty/universe_manifest.json");
        fs::write(
            &universe_path,
            serde_json::to_string_pretty(&universe).unwrap(),
        )
        .unwrap();

        let universe_bytes = fs::read(&universe_path).unwrap();
        let universe_sha = quantlaxmi_runner_common::manifest_io::sha256_hex(&universe_bytes);

        // Create session manifest with explicit tick paths
        let session = serde_json::json!({
            "schema_version": 1,
            "created_at_utc": "2026-01-24T10:00:00Z",
            "session_id": "test-isolation-001",
            "capture_mode": "india_capture",
            "out_dir": tmp.to_string_lossy(),
            "duration_secs": 300,
            "price_exponent": -2,
            "underlyings": [{
                "underlying": "NIFTY",
                "subdir": "nifty/",
                "universe_manifest_path": "nifty/universe_manifest.json",
                "universe_manifest_sha256": universe_sha,
                "instrument_count": 2,
                "t1_expiry": "2026-01-30",
                "t2_expiry": null,
                "t3_expiry": null,
                "strike_step": 50.0
            }],
            "tick_outputs": [
                {"symbol": "NIFTY26JAN24750CE", "path": "nifty/NIFTY26JAN24750CE/ticks.jsonl", "ticks_written": 100, "has_depth": true},
                {"symbol": "NIFTY26JAN24750PE", "path": "nifty/NIFTY26JAN24750PE/ticks.jsonl", "ticks_written": 95, "has_depth": true}
            ],
            "integrity": {
                "out_of_universe_ticks_dropped": 0,
                "subscribe_mode": "manifest_tokens",
                "notes": []
            }
        });
        fs::write(
            tmp.join("session_manifest.json"),
            serde_json::to_string_pretty(&session).unwrap(),
        )
        .unwrap();

        // Load inventory
        let inventory = try_load_sanos_inventory(&tmp)
            .unwrap()
            .expect("Should load manifest");

        // === STRUCTURAL ISOLATION CHECKS ===

        // 1. Verify session info comes from manifest
        assert_eq!(inventory.session_id, "test-isolation-001");
        assert_eq!(inventory.underlyings.len(), 1);

        let u_inv = &inventory.underlyings[0];

        // 2. Verify universe SHA256 is captured (audit trail)
        assert!(!u_inv.universe_sha256.is_empty());
        assert_eq!(u_inv.universe_sha256.len(), 64); // SHA256 hex length

        // 3. Verify expiries come from manifest (not parsed from symbols)
        let expiries = u_inv.get_sorted_expiries();
        assert_eq!(expiries.len(), 1);
        assert_eq!(expiries[0].to_string(), "2026-01-30");

        // 4. Verify instruments have complete info (no symbol parsing needed)
        let instruments = u_inv.get_instruments_for_expiry(expiries[0]);
        assert_eq!(instruments.len(), 2);

        // Find CE and PE
        let ce = instruments
            .iter()
            .find(|i| i.instrument_type == "CE")
            .expect("Should have CE");
        let pe = instruments
            .iter()
            .find(|i| i.instrument_type == "PE")
            .expect("Should have PE");

        // Verify instrument info is complete
        assert_eq!(ce.tradingsymbol, "NIFTY26JAN24750CE");
        assert_eq!(ce.strike, 24750.0);
        assert_eq!(ce.expiry.to_string(), "2026-01-30");
        assert_eq!(ce.instrument_token, 1001);

        assert_eq!(pe.tradingsymbol, "NIFTY26JAN24750PE");
        assert_eq!(pe.strike, 24750.0);
        assert_eq!(pe.expiry.to_string(), "2026-01-30");
        assert_eq!(pe.instrument_token, 1002);

        // 5. Verify tick paths are explicit (no directory scanning needed)
        assert_eq!(u_inv.tick_outputs.len(), 2);

        let ce_tick = u_inv
            .tick_outputs
            .iter()
            .find(|t| t.symbol == "NIFTY26JAN24750CE");
        let pe_tick = u_inv
            .tick_outputs
            .iter()
            .find(|t| t.symbol == "NIFTY26JAN24750PE");

        assert!(ce_tick.is_some());
        assert!(pe_tick.is_some());

        // Verify tick path lookup works
        assert_eq!(
            u_inv.get_tick_path("NIFTY26JAN24750CE"),
            Some("nifty/NIFTY26JAN24750CE/ticks.jsonl")
        );
        assert_eq!(
            u_inv.get_tick_path("NIFTY26JAN24750PE"),
            Some("nifty/NIFTY26JAN24750PE/ticks.jsonl")
        );

        // 6. Verify symbols for expiry lookup works (used by manifest-mode straddle builders)
        let symbols = u_inv.get_symbols_for_expiry(expiries[0]);
        assert_eq!(symbols.len(), 2);
        assert!(symbols.contains(&"NIFTY26JAN24750CE".to_string()));
        assert!(symbols.contains(&"NIFTY26JAN24750PE".to_string()));

        // 7. Verify strike step is captured
        assert_eq!(u_inv.strike_step, 50.0);

        // Cleanup
        let _ = fs::remove_dir_all(&tmp);
    }
}
