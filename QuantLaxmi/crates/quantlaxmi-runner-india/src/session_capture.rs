//! Multi-Instrument Session Capture for India/Zerodha.
//!
//! Captures quotes for multiple instruments in parallel, producing
//! a unified session with consistent timestamps and per-symbol manifests.
//!
//! ## Output Structure
//! ```text
//! data/sessions/{tag}/
//! ├── session_manifest.json    # Meta-manifest for entire session
//! ├── BANKNIFTY26JAN48000CE/
//! │   ├── ticks.jsonl
//! │   └── manifest.json
//! ├── BANKNIFTY26JAN48000PE/
//! │   ├── ticks.jsonl
//! │   └── manifest.json
//! └── ...
//! ```

use anyhow::{Context, Result, bail};
use byteorder::{BigEndian, ByteOrder};
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::io::AsyncWriteExt;
use tokio_tungstenite::tungstenite::Message;

use quantlaxmi_runner_common::artifact::{ArtifactFamily, FileHash, RunManifest, RunProfile};

const KITE_API_URL: &str = "https://api.kite.trade";

/// Tick event with mantissa-based pricing for deterministic replay.
///
/// Prices are stored as scaled integers (price_mantissa / 10^price_exponent).
/// For Kite, prices come as integers * 100, so price_exponent = -2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickEvent {
    pub ts: DateTime<Utc>,
    pub tradingsymbol: String,
    pub instrument_token: u32,
    /// Best bid price as mantissa (divide by 10^|price_exponent| to get actual price)
    pub bid_price: i64,
    /// Best ask price as mantissa
    pub ask_price: i64,
    /// Best bid quantity
    pub bid_qty: u32,
    /// Best ask quantity
    pub ask_qty: u32,
    /// Last traded price as mantissa
    pub ltp: i64,
    /// Last traded quantity
    pub ltq: u32,
    /// Total volume
    pub volume: u64,
    /// Price exponent (typically -2 for Kite, meaning divide by 100)
    pub price_exponent: i8,
    /// Integrity tier for this tick
    pub integrity_tier: IntegrityTier,
}

/// Integrity tier for captured data.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum IntegrityTier {
    /// Full L2 depth data available
    Certified,
    /// Synthetic spread applied (depth unavailable)
    Research,
}

/// Session manifest for multi-instrument capture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionManifest {
    pub session_id: String,
    pub watermark: String,
    pub family: String,
    pub profile: String,
    pub instruments: Vec<String>,
    pub captures: Vec<InstrumentCapture>,
    pub determinism: SessionDeterminism,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub duration_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstrumentCapture {
    pub tradingsymbol: String,
    pub instrument_token: u32,
    pub manifest_path: String,
    pub ticks_file: String,
    pub ticks_hash: Option<String>,
    pub ticks_written: usize,
    pub has_depth: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDeterminism {
    pub certified: bool,
    pub all_instruments_clean: bool,
    pub instrument_hashes: HashMap<String, String>,
}

/// Per-instrument capture statistics.
#[derive(Debug, Clone, Default)]
pub struct InstrumentCaptureStats {
    pub tradingsymbol: String,
    pub instrument_token: u32,
    pub ticks_written: usize,
    pub has_depth: bool,
    pub certified_ticks: usize,
    pub research_ticks: usize,
}

/// Session-level capture statistics.
#[derive(Debug, Clone)]
pub struct SessionCaptureStats {
    pub session_id: String,
    pub instruments: Vec<String>,
    pub instrument_stats: HashMap<String, InstrumentCaptureStats>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration_secs: f64,
    pub total_ticks: usize,
    pub all_certified: bool,
}

/// Configuration for session capture.
#[derive(Debug, Clone)]
pub struct SessionCaptureConfig {
    pub instruments: Vec<String>,
    pub out_dir: PathBuf,
    pub duration_secs: u64,
    pub price_exponent: i8,
}

/// Authentication response from Python sidecar.
#[derive(Deserialize)]
struct AuthOutput {
    access_token: String,
    api_key: String,
}

/// Authenticate with Zerodha via Python sidecar.
fn authenticate() -> Result<(String, String)> {
    println!("Authenticating with Zerodha...");
    let output = Command::new("python3")
        .arg("crates/kubera-connectors/scripts/zerodha_auth.py")
        .output()
        .context("Failed to run zerodha_auth.py")?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        bail!("Zerodha authentication failed: {}", err);
    }

    let auth: AuthOutput =
        serde_json::from_slice(&output.stdout).context("Failed to parse auth response")?;

    println!("✅ Authenticated with Zerodha");
    Ok((auth.api_key, auth.access_token))
}

/// Fetch instrument tokens from NFO instruments master.
async fn fetch_instrument_tokens(
    api_key: &str,
    access_token: &str,
    symbols: &[String],
) -> Result<Vec<(String, u32)>> {
    let client = reqwest::Client::new();

    // Fetch NFO instruments master
    let url = format!("{}/instruments/NFO", KITE_API_URL);
    let response = client
        .get(&url)
        .header("X-Kite-Version", "3")
        .header(
            "Authorization",
            format!("token {}:{}", api_key, access_token),
        )
        .send()
        .await?
        .text()
        .await?;

    let mut tokens = Vec::new();
    let symbols_upper: Vec<String> = symbols.iter().map(|s| s.to_uppercase()).collect();

    // Parse CSV: instrument_token,exchange_token,tradingsymbol,...
    for line in response.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 3 {
            let token_str = parts[0];
            let tradingsymbol = parts[2].to_uppercase();

            if symbols_upper.contains(&tradingsymbol)
                && let Ok(token) = token_str.parse::<u32>()
            {
                tokens.push((tradingsymbol, token));
            }
        }
    }

    Ok(tokens)
}

/// Parse binary tick data from Kite WebSocket.
/// Returns parsed tick data if valid.
///
/// Kite Binary Format (Full mode - 184 bytes):
/// - Offset 0-4: Token (uint32)
/// - Offset 4-8: LTP (int32, divide by 100)
/// - Offset 8-12: Last traded quantity
/// - Offset 44-104: Buy depth (5 levels x 12 bytes each)
/// - Offset 104-164: Sell depth (5 levels x 12 bytes each)
fn parse_kite_tick(data: &[u8], _price_exponent: i8) -> Option<Vec<ParsedTick>> {
    if data.len() < 4 {
        return None;
    }

    let num_packets = BigEndian::read_i16(&data[0..2]) as usize;
    let mut ticks = Vec::new();
    let mut offset = 2;

    for _ in 0..num_packets {
        if offset + 2 > data.len() {
            break;
        }

        let packet_len = BigEndian::read_i16(&data[offset..offset + 2]) as usize;
        offset += 2;

        if offset + packet_len > data.len() || packet_len < 8 {
            break;
        }

        let packet = &data[offset..offset + packet_len];
        let token = BigEndian::read_u32(&packet[0..4]);
        let ltp_raw = BigEndian::read_i32(&packet[4..8]);
        let ltq = if packet_len >= 12 {
            BigEndian::read_u32(&packet[8..12])
        } else {
            0
        };
        let volume = if packet_len >= 20 {
            BigEndian::read_u32(&packet[16..20]) as u64
        } else {
            0
        };

        // Full mode (184 bytes) has depth data
        let (bid_price, ask_price, bid_qty, ask_qty, has_depth) = if packet_len >= 184 {
            let depth_start = 44;

            // Find best bid (highest price with qty > 0 from buy side)
            let mut best_bid_price = 0i64;
            let mut best_bid_qty = 0u32;
            for i in 0..5 {
                let level_offset = depth_start + (i * 12);
                let qty = BigEndian::read_i32(&packet[level_offset..level_offset + 4]);
                let price = BigEndian::read_i32(&packet[level_offset + 4..level_offset + 8]) as i64;
                if qty > 0 && price > 0 && price > best_bid_price {
                    best_bid_price = price;
                    best_bid_qty = qty as u32;
                }
            }

            // Find best ask (lowest price with qty > 0 from sell side)
            let mut best_ask_price = i64::MAX;
            let mut best_ask_qty = 0u32;
            for i in 0..5 {
                let level_offset = depth_start + 60 + (i * 12);
                let qty = BigEndian::read_i32(&packet[level_offset..level_offset + 4]);
                let price = BigEndian::read_i32(&packet[level_offset + 4..level_offset + 8]) as i64;
                if qty > 0 && price > 0 && price < best_ask_price {
                    best_ask_price = price;
                    best_ask_qty = qty as u32;
                }
            }

            // Validate depth
            let valid_depth = best_bid_price > 0
                && best_ask_price < i64::MAX
                && best_bid_price < best_ask_price;

            if valid_depth {
                (best_bid_price, best_ask_price, best_bid_qty, best_ask_qty, true)
            } else {
                // Fallback: use LTP with synthetic spread
                let spread = (ltp_raw as i64) / 1000; // 0.1% spread
                let spread = spread.max(1); // At least 1 unit
                (
                    ltp_raw as i64 - spread,
                    ltp_raw as i64 + spread,
                    150, // Synthetic qty
                    150,
                    false,
                )
            }
        } else {
            // Quote/LTP mode - use synthetic spread
            let spread = (ltp_raw as i64) / 1000;
            let spread = spread.max(1);
            (
                ltp_raw as i64 - spread,
                ltp_raw as i64 + spread,
                150,
                150,
                false,
            )
        };

        ticks.push(ParsedTick {
            token,
            ltp: ltp_raw as i64,
            ltq,
            volume,
            bid_price,
            ask_price,
            bid_qty,
            ask_qty,
            has_depth,
        });

        offset += packet_len;
    }

    if ticks.is_empty() { None } else { Some(ticks) }
}

#[derive(Debug)]
struct ParsedTick {
    token: u32,
    ltp: i64,
    ltq: u32,
    volume: u64,
    bid_price: i64,
    ask_price: i64,
    bid_qty: u32,
    ask_qty: u32,
    has_depth: bool,
}

/// Capture a multi-instrument session.
pub async fn capture_session(config: SessionCaptureConfig) -> Result<SessionCaptureStats> {
    let session_id = uuid::Uuid::new_v4().to_string();
    let start_time = Utc::now();

    println!(
        "Starting India session capture: {} instruments, {} seconds",
        config.instruments.len(),
        config.duration_secs
    );
    println!("Session ID: {}", session_id);
    println!("Output directory: {:?}", config.out_dir);

    // Create session directory
    tokio::fs::create_dir_all(&config.out_dir).await?;

    // Authenticate
    let (api_key, access_token) = authenticate()?;

    // Fetch instrument tokens
    println!("Fetching instrument tokens for {} symbols...", config.instruments.len());
    let tokens = fetch_instrument_tokens(&api_key, &access_token, &config.instruments).await?;

    if tokens.is_empty() {
        bail!("No valid instrument tokens found for symbols: {:?}", config.instruments);
    }

    println!("Found {} instrument tokens:", tokens.len());
    for (sym, tok) in &tokens {
        println!("  {} -> {}", sym, tok);
    }

    // Build token -> symbol map
    let token_to_symbol: HashMap<u32, String> =
        tokens.iter().map(|(s, t)| (*t, s.clone())).collect();
    let symbol_to_token: HashMap<String, u32> =
        tokens.iter().map(|(s, t)| (s.clone(), *t)).collect();

    // Create per-instrument directories and files
    let mut files: HashMap<String, tokio::fs::File> = HashMap::new();
    for (sym, _) in &tokens {
        let sym_dir = config.out_dir.join(sym);
        tokio::fs::create_dir_all(&sym_dir).await?;
        let tick_path = sym_dir.join("ticks.jsonl");
        let file = tokio::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tick_path)
            .await?;
        files.insert(sym.clone(), file);
    }

    // Connect to WebSocket
    let ws_url = format!(
        "wss://ws.kite.trade/?api_key={}&access_token={}",
        api_key, access_token
    );
    println!("Connecting to Kite WebSocket...");

    let (ws_stream, _) = tokio_tungstenite::connect_async(&ws_url)
        .await
        .context("Failed to connect to Kite WebSocket")?;

    println!("✅ Connected to Kite WebSocket");
    let (mut write, mut read) = ws_stream.split();

    // Subscribe to instruments in Full mode
    let token_list: Vec<u32> = tokens.iter().map(|(_, t)| *t).collect();
    let subscribe_msg = serde_json::json!({
        "a": "subscribe",
        "v": token_list
    });
    write.send(Message::Text(subscribe_msg.to_string())).await?;

    // Set mode to Full (for depth data)
    let mode_msg = serde_json::json!({
        "a": "mode",
        "v": ["full", token_list]
    });
    write.send(Message::Text(mode_msg.to_string())).await?;
    println!("Subscribed to {} instruments in Full mode", tokens.len());

    // Initialize stats
    let mut instrument_stats: HashMap<String, InstrumentCaptureStats> = HashMap::new();
    for (sym, tok) in &tokens {
        instrument_stats.insert(
            sym.clone(),
            InstrumentCaptureStats {
                tradingsymbol: sym.clone(),
                instrument_token: *tok,
                ..Default::default()
            },
        );
    }

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(config.duration_secs);
    let mut total_ticks = 0usize;

    // Capture loop
    while tokio::time::Instant::now() < deadline {
        let msg = tokio::time::timeout(std::time::Duration::from_secs(5), read.next()).await;

        let item = match msg {
            Ok(Some(Ok(m))) => m,
            Ok(Some(Err(e))) => {
                eprintln!("WebSocket error: {}", e);
                continue;
            }
            Ok(None) => {
                println!("WebSocket closed");
                break;
            }
            Err(_) => continue, // Timeout, retry
        };

        match item {
            Message::Binary(data) => {
                if let Some(parsed_ticks) = parse_kite_tick(&data, config.price_exponent) {
                    for tick in parsed_ticks {
                        if let Some(symbol) = token_to_symbol.get(&tick.token) {
                            let event = TickEvent {
                                ts: Utc::now(),
                                tradingsymbol: symbol.clone(),
                                instrument_token: tick.token,
                                bid_price: tick.bid_price,
                                ask_price: tick.ask_price,
                                bid_qty: tick.bid_qty,
                                ask_qty: tick.ask_qty,
                                ltp: tick.ltp,
                                ltq: tick.ltq,
                                volume: tick.volume,
                                price_exponent: config.price_exponent,
                                integrity_tier: if tick.has_depth {
                                    IntegrityTier::Certified
                                } else {
                                    IntegrityTier::Research
                                },
                            };

                            if let Some(file) = files.get_mut(symbol) {
                                let line = serde_json::to_string(&event)?;
                                file.write_all(line.as_bytes()).await?;
                                file.write_all(b"\n").await?;

                                if let Some(stats) = instrument_stats.get_mut(symbol) {
                                    stats.ticks_written += 1;
                                    if tick.has_depth {
                                        stats.has_depth = true;
                                        stats.certified_ticks += 1;
                                    } else {
                                        stats.research_ticks += 1;
                                    }
                                }
                                total_ticks += 1;
                            }
                        }
                    }
                }

                // Progress indicator
                if total_ticks % 500 == 0 && total_ticks > 0 {
                    print!("\rCaptured {} ticks...", total_ticks);
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                }
            }
            Message::Text(text) => {
                if text.contains("error") {
                    eprintln!("Server message: {}", text);
                }
            }
            Message::Ping(p) => {
                let _ = write.send(Message::Pong(p)).await;
            }
            _ => {}
        }
    }

    // Flush all files
    for (_, mut file) in files {
        file.flush().await?;
    }

    let end_time = Utc::now();
    let duration_secs = (end_time - start_time).num_milliseconds() as f64 / 1000.0;

    // Determine if all certified
    let all_certified = instrument_stats.values().all(|s| s.has_depth && s.research_ticks == 0);

    // Generate session manifest
    let session_manifest = generate_session_manifest(
        &config,
        &session_id,
        &instrument_stats,
        &symbol_to_token,
        start_time,
        end_time,
        all_certified,
    ).await?;

    let manifest_path = config.out_dir.join("session_manifest.json");
    let json = serde_json::to_string_pretty(&session_manifest)?;
    tokio::fs::write(&manifest_path, json).await?;
    println!("\nSession manifest written: {:?}", manifest_path);

    // Generate per-instrument manifests
    for (sym, stats) in &instrument_stats {
        let sym_dir = config.out_dir.join(sym);
        generate_instrument_manifest(&sym_dir, sym, stats, &config).await?;
    }

    // Print summary
    println!("\n=== Session Capture Complete ===");
    println!("  Duration: {:.1}s", duration_secs);
    println!("  Total ticks: {}", total_ticks);
    for (sym, stats) in &instrument_stats {
        println!(
            "  {}: {} ticks (certified={}, research={})",
            sym, stats.ticks_written, stats.certified_ticks, stats.research_ticks
        );
    }
    println!(
        "  Status: {}",
        if all_certified { "CERTIFIED" } else { "RESEARCH" }
    );

    Ok(SessionCaptureStats {
        session_id,
        instruments: config.instruments.clone(),
        instrument_stats,
        start_time,
        end_time,
        duration_secs,
        total_ticks,
        all_certified,
    })
}

async fn generate_session_manifest(
    config: &SessionCaptureConfig,
    session_id: &str,
    instrument_stats: &HashMap<String, InstrumentCaptureStats>,
    symbol_to_token: &HashMap<String, u32>,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    all_certified: bool,
) -> Result<SessionManifest> {
    let mut captures = Vec::new();
    let mut instrument_hashes = HashMap::new();

    for sym in &config.instruments {
        let sym_upper = sym.to_uppercase();
        let sym_dir = config.out_dir.join(&sym_upper);
        let ticks_path = sym_dir.join("ticks.jsonl");

        let ticks_hash = if ticks_path.exists() {
            FileHash::from_file(&ticks_path).ok().map(|h| h.sha256)
        } else {
            None
        };

        let stats = instrument_stats.get(&sym_upper);
        let token = symbol_to_token.get(&sym_upper).copied().unwrap_or(0);

        captures.push(InstrumentCapture {
            tradingsymbol: sym_upper.clone(),
            instrument_token: token,
            manifest_path: format!("{}/manifest.json", sym_upper),
            ticks_file: format!("{}/ticks.jsonl", sym_upper),
            ticks_hash: ticks_hash.clone(),
            ticks_written: stats.map(|s| s.ticks_written).unwrap_or(0),
            has_depth: stats.map(|s| s.has_depth).unwrap_or(false),
        });

        if let Some(hash) = ticks_hash {
            instrument_hashes.insert(sym_upper, hash);
        }
    }

    let duration_secs = (end_time - start_time).num_milliseconds() as f64 / 1000.0;
    let watermark = format!(
        "QuantLaxmi-india-session-{}-{}",
        start_time.format("%Y%m%d-%H%M%S"),
        &session_id[..8]
    );

    Ok(SessionManifest {
        session_id: session_id.to_string(),
        watermark,
        family: "india".to_string(),
        profile: if all_certified { "certified".to_string() } else { "research".to_string() },
        instruments: config.instruments.iter().map(|s| s.to_uppercase()).collect(),
        captures,
        determinism: SessionDeterminism {
            certified: all_certified,
            all_instruments_clean: all_certified,
            instrument_hashes,
        },
        started_at: start_time,
        finished_at: end_time,
        duration_secs,
    })
}

async fn generate_instrument_manifest(
    sym_dir: &Path,
    symbol: &str,
    stats: &InstrumentCaptureStats,
    config: &SessionCaptureConfig,
) -> Result<()> {
    let profile = if stats.has_depth && stats.research_ticks == 0 {
        RunProfile::Certified
    } else {
        RunProfile::Research
    };

    let mut manifest = RunManifest::new(ArtifactFamily::India, profile);

    // Record ticks file hash
    let ticks_path = sym_dir.join("ticks.jsonl");
    if ticks_path.exists() {
        manifest.inputs.tick_events = Some(FileHash::from_file(&ticks_path)?);
    }

    manifest.determinism.certified = stats.has_depth && stats.research_ticks == 0;
    manifest.compute_input_hash();

    manifest.diagnostics.regime_transitions.push(
        quantlaxmi_runner_common::artifact::RegimeTransition {
            timestamp: Utc::now(),
            previous_regime: "capture_start".to_string(),
            new_regime: "capture_complete".to_string(),
            confidence: 1.0,
            features: serde_json::json!({
                "tradingsymbol": symbol.to_uppercase(),
                "instrument_token": stats.instrument_token,
                "ticks_written": stats.ticks_written,
                "certified_ticks": stats.certified_ticks,
                "research_ticks": stats.research_ticks,
                "has_depth": stats.has_depth,
                "price_exponent": config.price_exponent,
                "source": "zerodha_session_capture"
            }),
        },
    );

    manifest.finish();

    let manifest_path = sym_dir.join("manifest.json");
    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, json)?;

    println!("  {} manifest written: {:?}", symbol.to_uppercase(), manifest_path);

    Ok(())
}

/// Count lines in a file (for stats recovery on error).
fn count_file_lines(path: &Path) -> std::io::Result<usize> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    Ok(reader.lines().count())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_event_serialization() {
        let event = TickEvent {
            ts: Utc::now(),
            tradingsymbol: "BANKNIFTY26JAN48000CE".to_string(),
            instrument_token: 12345678,
            bid_price: 15050,
            ask_price: 15100,
            bid_qty: 150,
            ask_qty: 200,
            ltp: 15075,
            ltq: 50,
            volume: 10000,
            price_exponent: -2,
            integrity_tier: IntegrityTier::Certified,
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("BANKNIFTY26JAN48000CE"));
        assert!(json.contains("15050"));
    }

    #[test]
    fn test_session_manifest_serialization() {
        let manifest = SessionManifest {
            session_id: "test-123".to_string(),
            watermark: "QuantLaxmi-india-session-20260122-120000-test1234".to_string(),
            family: "india".to_string(),
            profile: "certified".to_string(),
            instruments: vec!["BANKNIFTY26JAN48000CE".to_string()],
            captures: vec![],
            determinism: SessionDeterminism {
                certified: true,
                all_instruments_clean: true,
                instrument_hashes: HashMap::new(),
            },
            started_at: Utc::now(),
            finished_at: Utc::now(),
            duration_secs: 300.0,
        };

        let json = serde_json::to_string_pretty(&manifest).unwrap();
        assert!(json.contains("session_id"));
        assert!(json.contains("india"));
    }
}
