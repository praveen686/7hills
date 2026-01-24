//! Binance Perp Session Capture - Spot + Perp + Funding orchestrator.
//!
//! This module captures all data needed for funding rate arbitrage:
//! - Spot bookTicker (reference price)
//! - Perp bookTicker (trading price)
//! - Funding rate stream (funding signals)
//!
//! The combined capture enables basis calculation:
//! ```text
//! Basis = (Perp_Price - Spot_Price) / Spot_Price
//! ```
//!
//! Directory structure:
//! ```text
//! data/perp_sessions/{tag}/
//! ├── session_manifest.json
//! ├── BTCUSDT/
//! │   ├── spot_quotes.jsonl      # Spot bookTicker
//! │   ├── perp_quotes.jsonl      # Perp bookTicker
//! │   ├── funding.jsonl          # Funding rate stream
//! │   └── manifest.json          # Per-symbol manifest
//! └── ETHUSDT/
//!     └── ...
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::binance_capture;
use crate::binance_funding_capture;
use crate::binance_perp_capture;

/// Configuration for perp session capture.
#[derive(Debug, Clone)]
pub struct PerpSessionConfig {
    /// Symbols to capture (e.g., BTCUSDT, ETHUSDT)
    pub symbols: Vec<String>,
    /// Output directory for session data
    pub out_dir: PathBuf,
    /// Capture duration in seconds
    pub duration_secs: u64,
    /// Include spot capture (for basis calculation)
    pub include_spot: bool,
    /// Include perp depth (L2 order book)
    pub include_depth: bool,
    /// Price exponent for depth capture
    pub price_exponent: i8,
    /// Quantity exponent for depth capture
    pub qty_exponent: i8,
}

impl Default for PerpSessionConfig {
    fn default() -> Self {
        Self {
            symbols: vec!["BTCUSDT".to_string()],
            out_dir: PathBuf::from("data/perp_sessions"),
            duration_secs: 3600, // 1 hour default
            include_spot: true,
            include_depth: false, // bookTicker is sufficient for MVP
            price_exponent: -2,
            qty_exponent: -8,
        }
    }
}

/// Per-symbol capture statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolCaptureStats {
    pub symbol: String,
    pub spot_events: usize,
    pub perp_events: usize,
    pub funding_events: usize,
    pub last_spot_bid: f64,
    pub last_spot_ask: f64,
    pub last_perp_bid: f64,
    pub last_perp_ask: f64,
    pub last_funding_rate: f64,
    pub funding_settlements: usize,
    pub basis_bps: f64,
}

impl SymbolCaptureStats {
    /// Calculate basis in basis points.
    pub fn calculate_basis_bps(&mut self) {
        if self.last_spot_ask > 0.0 && self.last_perp_bid > 0.0 {
            let spot_mid = (self.last_spot_bid + self.last_spot_ask) / 2.0;
            let perp_mid = (self.last_perp_bid + self.last_perp_ask) / 2.0;
            if spot_mid > 0.0 {
                self.basis_bps = ((perp_mid - spot_mid) / spot_mid) * 10000.0;
            }
        }
    }
}

/// Session-level statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerpSessionStats {
    pub session_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration_secs: f64,
    pub symbols: Vec<SymbolCaptureStats>,
    pub total_spot_events: usize,
    pub total_perp_events: usize,
    pub total_funding_events: usize,
    pub all_symbols_complete: bool,
}

/// Session manifest for perp capture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerpSessionManifest {
    pub schema_version: u32,
    pub created_at_utc: String,
    pub session_id: String,
    pub capture_mode: String,
    pub duration_secs: f64,
    pub symbols: Vec<SymbolManifestEntry>,
    pub config: PerpSessionManifestConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolManifestEntry {
    pub symbol: String,
    pub spot_file: Option<String>,
    pub perp_file: String,
    pub funding_file: String,
    pub spot_events: usize,
    pub perp_events: usize,
    pub funding_events: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerpSessionManifestConfig {
    pub include_spot: bool,
    pub include_depth: bool,
    pub price_exponent: i8,
    pub qty_exponent: i8,
}

/// Capture a perp session with Spot + Perp + Funding data.
pub async fn capture_perp_session(config: PerpSessionConfig) -> Result<PerpSessionStats> {
    let session_id = uuid::Uuid::new_v4().to_string();
    let start_time = Utc::now();

    // Create session directory
    let session_tag = format!("perp_{}", start_time.format("%Y%m%d_%H%M%S"));
    let session_dir = config.out_dir.join(&session_tag);
    std::fs::create_dir_all(&session_dir)
        .with_context(|| format!("create session dir: {:?}", session_dir))?;

    tracing::info!("=== Perp Session Capture ===");
    tracing::info!("Session ID: {}", &session_id[..8]);
    tracing::info!("Symbols: {:?}", config.symbols);
    tracing::info!("Duration: {}s", config.duration_secs);
    tracing::info!("Output: {:?}", session_dir);

    // Spawn capture tasks for each symbol
    let mut handles = Vec::new();
    for symbol in &config.symbols {
        let sym = symbol.clone();
        let sym_dir = session_dir.join(symbol.to_uppercase());
        std::fs::create_dir_all(&sym_dir)?;

        let duration = config.duration_secs;
        let include_spot = config.include_spot;
        let include_depth = config.include_depth;
        let price_exp = config.price_exponent;
        let qty_exp = config.qty_exponent;

        let handle = tokio::spawn(async move {
            capture_symbol(
                &sym,
                &sym_dir,
                duration,
                include_spot,
                include_depth,
                price_exp,
                qty_exp,
            )
            .await
        });
        handles.push((symbol.clone(), handle));
    }

    // Wait for all captures to complete
    let mut symbol_stats = Vec::new();
    let mut total_spot = 0;
    let mut total_perp = 0;
    let mut total_funding = 0;
    let mut all_complete = true;

    for (symbol, handle) in handles {
        match handle.await {
            Ok(Ok(stats)) => {
                total_spot += stats.spot_events;
                total_perp += stats.perp_events;
                total_funding += stats.funding_events;
                symbol_stats.push(stats);
            }
            Ok(Err(e)) => {
                tracing::error!("Capture failed for {}: {}", symbol, e);
                all_complete = false;
            }
            Err(e) => {
                tracing::error!("Task panicked for {}: {}", symbol, e);
                all_complete = false;
            }
        }
    }

    let end_time = Utc::now();
    let duration_secs = (end_time - start_time).num_milliseconds() as f64 / 1000.0;

    // Write session manifest
    let manifest = PerpSessionManifest {
        schema_version: 1,
        created_at_utc: start_time.to_rfc3339(),
        session_id: session_id.clone(),
        capture_mode: "perp_session".to_string(),
        duration_secs,
        symbols: symbol_stats
            .iter()
            .map(|s| SymbolManifestEntry {
                symbol: s.symbol.clone(),
                spot_file: if config.include_spot {
                    Some(format!("{}/spot_quotes.jsonl", s.symbol.to_uppercase()))
                } else {
                    None
                },
                perp_file: format!("{}/perp_quotes.jsonl", s.symbol.to_uppercase()),
                funding_file: format!("{}/funding.jsonl", s.symbol.to_uppercase()),
                spot_events: s.spot_events,
                perp_events: s.perp_events,
                funding_events: s.funding_events,
            })
            .collect(),
        config: PerpSessionManifestConfig {
            include_spot: config.include_spot,
            include_depth: config.include_depth,
            price_exponent: config.price_exponent,
            qty_exponent: config.qty_exponent,
        },
    };

    let manifest_path = session_dir.join("session_manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, manifest_json)?;
    tracing::info!("Session manifest written: {:?}", manifest_path);

    let stats = PerpSessionStats {
        session_id,
        start_time,
        end_time,
        duration_secs,
        symbols: symbol_stats,
        total_spot_events: total_spot,
        total_perp_events: total_perp,
        total_funding_events: total_funding,
        all_symbols_complete: all_complete,
    };

    // Print summary
    tracing::info!("=== Session Summary ===");
    tracing::info!("Duration: {:.1}s", stats.duration_secs);
    tracing::info!("Spot events: {}", stats.total_spot_events);
    tracing::info!("Perp events: {}", stats.total_perp_events);
    tracing::info!("Funding events: {}", stats.total_funding_events);
    for sym_stat in &stats.symbols {
        tracing::info!(
            "  {}: basis={:.2}bps, funding={:.4}%",
            sym_stat.symbol,
            sym_stat.basis_bps,
            sym_stat.last_funding_rate * 100.0
        );
    }

    Ok(stats)
}

/// Capture result enum for unified task handling.
enum CaptureResult {
    Spot {
        events: usize,
    },
    Perp {
        events: usize,
        bid: f64,
        ask: f64,
    },
    Funding {
        events: usize,
        rate: f64,
        settlements: usize,
    },
}

/// Capture all streams for a single symbol.
async fn capture_symbol(
    symbol: &str,
    out_dir: &Path,
    duration_secs: u64,
    include_spot: bool,
    include_depth: bool,
    price_exponent: i8,
    qty_exponent: i8,
) -> Result<SymbolCaptureStats> {
    let mut stats = SymbolCaptureStats {
        symbol: symbol.to_string(),
        spot_events: 0,
        perp_events: 0,
        funding_events: 0,
        last_spot_bid: 0.0,
        last_spot_ask: 0.0,
        last_perp_bid: 0.0,
        last_perp_ask: 0.0,
        last_funding_rate: 0.0,
        funding_settlements: 0,
        basis_bps: 0.0,
    };

    // Spawn parallel capture tasks with unified return type
    let mut handles: Vec<tokio::task::JoinHandle<Result<CaptureResult>>> = Vec::new();

    // 1. Spot bookTicker (optional)
    if include_spot {
        let sym = symbol.to_string();
        let path = out_dir.join("spot_quotes.jsonl");
        handles.push(tokio::spawn(async move {
            let result =
                binance_capture::capture_book_ticker_jsonl(&sym, &path, duration_secs).await?;
            Ok(CaptureResult::Spot {
                events: result.events_written,
            })
        }));
    }

    // 2. Perp bookTicker (or depth)
    let sym = symbol.to_string();
    let path = if include_depth {
        out_dir.join("perp_depth.jsonl")
    } else {
        out_dir.join("perp_quotes.jsonl")
    };
    let dur = duration_secs;
    if include_depth {
        handles.push(tokio::spawn(async move {
            let result = binance_perp_capture::capture_perp_depth_jsonl(
                &sym,
                &path,
                dur,
                price_exponent,
                qty_exponent,
            )
            .await?;
            Ok(CaptureResult::Perp {
                events: result.events_written,
                bid: result.last_bid,
                ask: result.last_ask,
            })
        }));
    } else {
        handles.push(tokio::spawn(async move {
            let result =
                binance_perp_capture::capture_perp_bookticker_jsonl(&sym, &path, dur).await?;
            Ok(CaptureResult::Perp {
                events: result.events_written,
                bid: result.last_bid,
                ask: result.last_ask,
            })
        }));
    }

    // 3. Funding rate stream
    let sym = symbol.to_string();
    let path = out_dir.join("funding.jsonl");
    let dur = duration_secs;
    handles.push(tokio::spawn(async move {
        let result = binance_funding_capture::capture_funding_jsonl(&sym, &path, dur).await?;
        Ok(CaptureResult::Funding {
            events: result.events_written,
            rate: result.last_funding_rate,
            settlements: result.funding_settlements,
        })
    }));

    // Collect results
    for handle in handles {
        match handle.await {
            Ok(Ok(CaptureResult::Spot { events })) => {
                stats.spot_events = events;
            }
            Ok(Ok(CaptureResult::Perp { events, bid, ask })) => {
                stats.perp_events = events;
                stats.last_perp_bid = bid;
                stats.last_perp_ask = ask;
            }
            Ok(Ok(CaptureResult::Funding {
                events,
                rate,
                settlements,
            })) => {
                stats.funding_events = events;
                stats.last_funding_rate = rate;
                stats.funding_settlements = settlements;
            }
            Ok(Err(e)) => {
                tracing::warn!("Capture for {} failed: {}", symbol, e);
            }
            Err(e) => {
                tracing::warn!("Task panicked for {}: {}", symbol, e);
            }
        }
    }

    // Calculate final basis
    stats.calculate_basis_bps();

    Ok(stats)
}

/// Load a perp session manifest from disk.
pub fn load_perp_session_manifest(session_dir: &Path) -> Result<PerpSessionManifest> {
    let manifest_path = session_dir.join("session_manifest.json");
    let content = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("read manifest: {:?}", manifest_path))?;
    let manifest: PerpSessionManifest = serde_json::from_str(&content)
        .with_context(|| format!("parse manifest: {:?}", manifest_path))?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_calculation() {
        let mut stats = SymbolCaptureStats {
            symbol: "BTCUSDT".to_string(),
            spot_events: 100,
            perp_events: 100,
            funding_events: 10,
            last_spot_bid: 99990.0,
            last_spot_ask: 100010.0,
            last_perp_bid: 100040.0,
            last_perp_ask: 100060.0,
            last_funding_rate: 0.0001,
            funding_settlements: 0,
            basis_bps: 0.0,
        };

        stats.calculate_basis_bps();

        // Spot mid = 100000, Perp mid = 100050
        // Basis = (100050 - 100000) / 100000 = 0.0005 = 5 bps
        assert!((stats.basis_bps - 5.0).abs() < 0.1);
    }
}
