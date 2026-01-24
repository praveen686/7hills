//! Multi-Instrument Session Capture for India/Zerodha.
//!
//! Captures quotes for multiple instruments in parallel, producing
//! a unified session with consistent timestamps and per-symbol manifests.
//!
//! ## Output Structure
//! ```text
//! data/sessions/{tag}/
//! ├── session_manifest.json    # Canonical manifest (written by lib.rs, not here)
//! ├── capture_debug.json       # Debug manifest (written by this module)
//! ├── BANKNIFTY26JAN48000CE/
//! │   ├── ticks.jsonl
//! │   └── manifest.json
//! ├── BANKNIFTY26JAN48000PE/
//! │   ├── ticks.jsonl
//! │   └── manifest.json
//! └── ...
//! ```
//!
//! NOTE: The canonical `session_manifest.json` is produced by the top-level
//! runner (lib.rs), not by this capture module. This module writes only
//! `capture_debug.json` for diagnostic purposes.

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

/// Integrity tier for captured tick data.
///
/// Note: "Certified" is reserved for artifacts that persist FULL L2 depth.
/// Since TickEvent only stores L1 (best bid/ask), we use L2Present/L1Only
/// to indicate whether the source tick had depth data available.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum IntegrityTier {
    /// Source tick had L2 depth data (but only L1 is persisted in TickEvent)
    L2Present,
    /// Source tick had no depth data; synthetic spread applied
    L1Only,
}

/// Legacy capture-level manifest (debug/diagnostic only).
///
/// NOTE: This is NOT the canonical session manifest. The canonical manifest
/// is `SessionManifest` from `quantlaxmi_runner_common::session_manifest`.
/// This struct is retained only for per-capture debug diagnostics and is
/// written to `capture_debug.json` (not `session_manifest.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureDebugManifest {
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
    /// Subscription mode: "manifest_tokens" (Commit B path) or "api_lookup" (legacy)
    pub subscribe_mode: String,
    /// Number of out-of-universe ticks dropped during capture (Commit B validation)
    pub out_of_universe_ticks_dropped: usize,
    /// Per-symbol tick output info: (tradingsymbol, relative_path, ticks_written, has_depth)
    pub tick_outputs: Vec<(String, String, usize, bool)>,
}

/// Configuration for session capture.
#[derive(Debug, Clone)]
pub struct SessionCaptureConfig {
    /// Instrument symbols (used for display and file naming)
    pub instruments: Vec<String>,
    /// Output directory for session data
    pub out_dir: PathBuf,
    /// Capture duration in seconds
    pub duration_secs: u64,
    /// Price exponent for mantissa conversion (typically -2 for Kite)
    pub price_exponent: i8,
    /// Manifest-provided tokens for subscription (Commit B: Phase 5.0)
    /// If provided, these tokens are used directly instead of fetching from API.
    /// Format: Vec<(tradingsymbol, instrument_token)>
    pub manifest_tokens: Option<Vec<(String, u32)>>,
}

/// Authentication response from Python sidecar.
#[derive(Deserialize)]
struct AuthOutput {
    access_token: String,
    api_key: String,
}

/// Authenticate with Zerodha via Python sidecar.
///
/// The sidecar script is located at `scripts/zerodha_auth.py` in the QuantLaxmi repo root.
fn authenticate() -> Result<(String, String)> {
    println!("Authenticating with Zerodha...");
    let output = Command::new("python3")
        .arg("scripts/zerodha_auth.py")
        .output()
        .context("Failed to run scripts/zerodha_auth.py")?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        bail!("Zerodha authentication failed: {}", err);
    }

    let auth: AuthOutput =
        serde_json::from_slice(&output.stdout).context("Failed to parse auth response")?;

    println!("✅ Authenticated with Zerodha");
    Ok((auth.api_key, auth.access_token))
}

/// Kite exchange segment for instrument master download.
#[derive(Debug, Clone, Copy, Default)]
pub enum KiteSegment {
    /// NSE Futures & Options (default for options capture)
    #[default]
    NFO,
    /// NSE Cash/Equities (includes indices like NIFTY 50, NIFTY BANK)
    NSE,
    /// BSE Cash/Equities
    BSE,
    /// BSE Futures & Options
    BFO,
}

impl KiteSegment {
    fn as_str(&self) -> &'static str {
        match self {
            KiteSegment::NFO => "NFO",
            KiteSegment::NSE => "NSE",
            KiteSegment::BSE => "BSE",
            KiteSegment::BFO => "BFO",
        }
    }
}

/// Fetch instrument tokens from Kite instruments master.
///
/// # Parameters
/// * `segment` - Exchange segment (NFO for options, NSE for indices/equities)
///
/// For indices (NIFTY 50, NIFTY BANK), use `KiteSegment::NSE`.
/// For F&O instruments, use `KiteSegment::NFO` (default).
async fn fetch_instrument_tokens(
    api_key: &str,
    access_token: &str,
    symbols: &[String],
    segment: KiteSegment,
) -> Result<Vec<(String, u32)>> {
    let client = reqwest::Client::new();

    // Fetch instruments master for the specified segment
    let url = format!("{}/instruments/{}", KITE_API_URL, segment.as_str());
    println!("Fetching instruments from: {}", url);

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

/// Fetch instrument tokens from multiple segments.
///
/// Automatically determines segment based on symbol patterns:
/// - Symbols containing expiry codes (e.g., "26JAN") → NFO
/// - Symbols like "NIFTY 50", "NIFTY BANK" → NSE
async fn fetch_instrument_tokens_auto(
    api_key: &str,
    access_token: &str,
    symbols: &[String],
) -> Result<Vec<(String, u32)>> {
    let mut nfo_symbols = Vec::new();
    let mut nse_symbols = Vec::new();

    for sym in symbols {
        let upper = sym.to_uppercase();
        // F&O symbols contain expiry patterns like "26JAN", "26FEB", etc.
        if upper.contains("JAN")
            || upper.contains("FEB")
            || upper.contains("MAR")
            || upper.contains("APR")
            || upper.contains("MAY")
            || upper.contains("JUN")
            || upper.contains("JUL")
            || upper.contains("AUG")
            || upper.contains("SEP")
            || upper.contains("OCT")
            || upper.contains("NOV")
            || upper.contains("DEC")
            || upper.ends_with("CE")
            || upper.ends_with("PE")
            || upper.ends_with("FUT")
        {
            nfo_symbols.push(upper);
        } else {
            // Likely an index or equity
            nse_symbols.push(upper);
        }
    }

    let mut all_tokens = Vec::new();

    // Fetch from NFO if any F&O symbols
    if !nfo_symbols.is_empty() {
        let tokens =
            fetch_instrument_tokens(api_key, access_token, &nfo_symbols, KiteSegment::NFO).await?;
        all_tokens.extend(tokens);
    }

    // Fetch from NSE if any index/equity symbols
    if !nse_symbols.is_empty() {
        let tokens =
            fetch_instrument_tokens(api_key, access_token, &nse_symbols, KiteSegment::NSE).await?;
        all_tokens.extend(tokens);
    }

    Ok(all_tokens)
}

/// Parse binary tick data from Kite WebSocket.
/// Returns parsed tick data if valid.
///
/// Kite Binary Format (Full mode - 184 bytes):
/// - Offset 0-4: Token (uint32)
/// - Offset 4-8: LTP (int32, divide by 100)
/// - Offset 8-12: Last traded quantity
/// - Offset 12-16: Average traded price
/// - Offset 16-20: Volume traded
/// - Offset 20-24: Total buy quantity
/// - Offset 24-28: Total sell quantity
/// - Offset 28-32: Open
/// - Offset 32-36: High
/// - Offset 36-40: Low
/// - Offset 40-44: Close
/// - Offset 44-48: Last trade time (Unix timestamp)
/// - Offset 48-52: OI (open interest)
/// - Offset 52-56: OI day high
/// - Offset 56-60: OI day low
/// - Offset 60-64: Exchange timestamp (Unix timestamp)
/// - Offset 64-124: Buy depth (5 levels x 12 bytes each)
/// - Offset 124-184: Sell depth (5 levels x 12 bytes each)
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
            // Depth starts at offset 64, after the 64-byte header that includes:
            // OHLC, timestamps, OI, and exchange_timestamp
            let depth_start = 64;

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
            let valid_depth =
                best_bid_price > 0 && best_ask_price < i64::MAX && best_bid_price < best_ask_price;

            // Debug: dump full packet to find where depth actually is
            static DEBUG_COUNT: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let count = DEBUG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count < 1 {
                eprintln!(
                    "[DEBUG] token={} ltp={} (₹{:.2})",
                    token,
                    ltp_raw,
                    ltp_raw as f64 / 100.0
                );
                eprintln!("[DEBUG] Full packet ({} bytes):", packet_len);
                // Dump in 20-byte chunks with offset labels
                for chunk_start in (0..packet_len).step_by(20) {
                    let chunk_end = (chunk_start + 20).min(packet_len);
                    let chunk = &packet[chunk_start..chunk_end];
                    // Also show as i32 values
                    let mut i32_vals = String::new();
                    for i in (0..chunk.len()).step_by(4) {
                        if i + 4 <= chunk.len() {
                            let val = BigEndian::read_i32(&chunk[i..i + 4]);
                            i32_vals.push_str(&format!("{:>10} ", val));
                        }
                    }
                    eprintln!("[{:3}] {:02x?} | {}", chunk_start, chunk, i32_vals);
                }
            }

            if valid_depth {
                (
                    best_bid_price,
                    best_ask_price,
                    best_bid_qty,
                    best_ask_qty,
                    true,
                )
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
    use std::collections::HashSet;
    use tracing::{debug, warn};

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

    // Determine tokens: use manifest tokens if provided (Commit B), otherwise fetch from API
    let (tokens, subscribe_mode): (Vec<(String, u32)>, String) =
        if let Some(manifest_tokens) = &config.manifest_tokens {
            println!(
                "Using {} manifest-provided tokens (subscribe_mode=manifest_tokens)",
                manifest_tokens.len()
            );
            debug!(
                token_count = manifest_tokens.len(),
                sample_tokens = ?manifest_tokens.iter().take(5).map(|(_, t)| t).collect::<Vec<_>>(),
                subscribe_mode = "manifest_tokens",
                "Subscription tokens from UniverseManifest"
            );
            (manifest_tokens.clone(), "manifest_tokens".to_string())
        } else {
            // Legacy path: fetch instrument tokens from API (auto-detects segment)
            println!(
                "Fetching instrument tokens for {} symbols (subscribe_mode=api_lookup)...",
                config.instruments.len()
            );
            let fetched =
                fetch_instrument_tokens_auto(&api_key, &access_token, &config.instruments).await?;
            debug!(
                token_count = fetched.len(),
                subscribe_mode = "api_lookup",
                "Subscription tokens from API lookup"
            );
            (fetched, "api_lookup".to_string())
        };

    if tokens.is_empty() {
        bail!(
            "No valid instrument tokens found for symbols: {:?}",
            config.instruments
        );
    }

    println!("Subscribing to {} instrument tokens:", tokens.len());
    for (sym, tok) in tokens.iter().take(10) {
        println!("  {} -> {}", sym, tok);
    }
    if tokens.len() > 10 {
        println!("  ... and {} more", tokens.len() - 10);
    }

    // Build token validation set for out-of-universe detection (Commit B)
    let valid_token_set: HashSet<u32> = tokens.iter().map(|(_, t)| *t).collect();

    // Build token -> symbol map
    let token_to_symbol: HashMap<u32, String> =
        tokens.iter().map(|(s, t)| (*t, s.clone())).collect();
    let symbol_to_token: HashMap<String, u32> =
        tokens.iter().map(|(s, t)| (s.clone(), *t)).collect();

    // Create per-instrument directories and files
    // Track relative paths for session manifest (Commit C audit fix 1.2)
    let mut files: HashMap<String, tokio::fs::File> = HashMap::new();
    let mut tick_paths: HashMap<String, String> = HashMap::new();
    for (sym, _) in &tokens {
        let sym_dir = config.out_dir.join(sym);
        tokio::fs::create_dir_all(&sym_dir).await?;
        let tick_path = sym_dir.join("ticks.jsonl");
        // Record relative path from out_dir (decoupled from format string)
        let relative_path = format!("{}/ticks.jsonl", sym);
        tick_paths.insert(sym.clone(), relative_path);
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

    let deadline =
        tokio::time::Instant::now() + std::time::Duration::from_secs(config.duration_secs);
    let mut total_ticks = 0usize;
    let mut out_of_universe_ticks = 0usize;

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
                        // Commit B: Validate tick token is in manifest universe
                        if !valid_token_set.contains(&tick.token) {
                            out_of_universe_ticks += 1;
                            if out_of_universe_ticks <= 5 {
                                warn!(
                                    token = tick.token,
                                    "Out-of-universe tick dropped (token not in manifest)"
                                );
                            }
                            continue; // Drop tick - preserves determinism
                        }

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
                                    IntegrityTier::L2Present
                                } else {
                                    IntegrityTier::L1Only
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
                if total_ticks > 0 && total_ticks.is_multiple_of(500) {
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
    let all_certified = instrument_stats
        .values()
        .all(|s| s.has_depth && s.research_ticks == 0);

    // Generate session manifest
    let session_manifest = generate_session_manifest(
        &config,
        &session_id,
        &instrument_stats,
        &symbol_to_token,
        start_time,
        end_time,
        all_certified,
    )
    .await?;

    // Write capture debug manifest (diagnostic only, NOT the canonical session manifest)
    let debug_manifest_path = config.out_dir.join("capture_debug.json");
    let json = serde_json::to_string_pretty(&session_manifest)?;
    tokio::fs::write(&debug_manifest_path, json).await?;
    println!(
        "\nCapture debug manifest written: {:?}",
        debug_manifest_path
    );

    // Generate per-instrument manifests
    for (sym, stats) in &instrument_stats {
        let sym_dir = config.out_dir.join(sym);
        generate_instrument_manifest(&sym_dir, sym, stats, &config).await?;
    }

    // Print summary
    println!("\n=== Session Capture Complete ===");
    println!("  Duration: {:.1}s", duration_secs);
    println!("  Total ticks: {}", total_ticks);
    if out_of_universe_ticks > 0 {
        println!("  Out-of-universe ticks dropped: {}", out_of_universe_ticks);
        warn!(
            out_of_universe_ticks,
            "Session had out-of-universe ticks (dropped)"
        );
    }
    for (sym, stats) in &instrument_stats {
        println!(
            "  {}: {} ticks (certified={}, research={})",
            sym, stats.ticks_written, stats.certified_ticks, stats.research_ticks
        );
    }
    println!(
        "  Status: {}",
        if all_certified {
            "CERTIFIED"
        } else {
            "RESEARCH"
        }
    );

    // Build tick outputs list for session manifest (Commit C)
    // Use tracked tick_paths for decoupled path derivation (audit fix 1.2)
    let tick_outputs: Vec<(String, String, usize, bool)> = instrument_stats
        .iter()
        .map(|(sym, stats)| {
            let relative_path = tick_paths
                .get(sym)
                .cloned()
                .unwrap_or_else(|| format!("{}/ticks.jsonl", sym));
            (
                sym.clone(),
                relative_path,
                stats.ticks_written,
                stats.has_depth,
            )
        })
        .collect();

    Ok(SessionCaptureStats {
        session_id,
        instruments: config.instruments.clone(),
        instrument_stats,
        start_time,
        end_time,
        duration_secs,
        total_ticks,
        all_certified,
        subscribe_mode,
        out_of_universe_ticks_dropped: out_of_universe_ticks,
        tick_outputs,
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
) -> Result<CaptureDebugManifest> {
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

    Ok(CaptureDebugManifest {
        session_id: session_id.to_string(),
        watermark,
        family: "india".to_string(),
        profile: if all_certified {
            "certified".to_string()
        } else {
            "research".to_string()
        },
        instruments: config
            .instruments
            .iter()
            .map(|s| s.to_uppercase())
            .collect(),
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

    println!(
        "  {} manifest written: {:?}",
        symbol.to_uppercase(),
        manifest_path
    );

    Ok(())
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
            integrity_tier: IntegrityTier::L2Present,
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("BANKNIFTY26JAN48000CE"));
        assert!(json.contains("15050"));
    }

    #[test]
    fn test_capture_debug_manifest_serialization() {
        let manifest = CaptureDebugManifest {
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
