//! Convert Binance perp depth and trade data to SLRT MarketEvent format.
//!
//! Usage: convert_depth <depth.jsonl> [trades.jsonl] <output.jsonl>

use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::io::{BufRead, BufReader, Write};

/// Binance depth level format.
#[derive(Debug, Deserialize)]
struct BinanceLevel {
    price: i64,
    qty: i64,
}

/// Binance depth snapshot format.
#[derive(Debug, Deserialize)]
struct BinanceDepth {
    ts: String,
    tradingsymbol: String,
    #[serde(default)]
    #[allow(dead_code)]
    market: Option<String>,
    #[allow(dead_code)]
    first_update_id: u64,
    #[allow(dead_code)]
    last_update_id: u64,
    price_exponent: i8,
    qty_exponent: i8,
    bids: Vec<BinanceLevel>,
    asks: Vec<BinanceLevel>,
}

/// Binance trade format.
#[derive(Debug, Deserialize)]
struct BinanceTrade {
    ts: String,
    tradingsymbol: String,
    price: i64,
    qty: i64,
    price_exponent: i8,
    qty_exponent: i8,
    is_buyer_maker: bool,
}

/// SLRT PriceLevel format.
#[derive(Debug, Serialize)]
struct PriceLevel {
    price_mantissa: i64,
    price_exponent: i8,
    qty_mantissa: i64,
    qty_exponent: i8,
}

/// SLRT OrderBook format.
#[derive(Debug, Serialize)]
struct OrderBook {
    ts_ns: i64,
    symbol: String,
    bids: Vec<PriceLevel>,
    asks: Vec<PriceLevel>,
}

/// SLRT TradeSide format (matching data.rs).
#[derive(Debug, Serialize)]
enum TradeSide {
    Buy,
    Sell,
}

/// SLRT TriState format (matching data.rs).
#[derive(Debug, Serialize)]
enum TriState {
    Present(TradeSide),
}

/// SLRT Trade format (matching data.rs).
#[derive(Debug, Serialize)]
struct Trade {
    ts_ns: i64,
    symbol: String,
    price_mantissa: i64,
    price_exponent: i8,
    qty_mantissa: i64,
    qty_exponent: i8,
    side: TriState,
}

/// SLRT MarketEvent wrapper.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum MarketEvent {
    Book(OrderBook),
    Trade(Trade),
}

/// Timestamped event for sorting.
#[derive(Debug)]
struct TimestampedEvent {
    ts_ns: i64,
    event: MarketEvent,
}

impl PartialEq for TimestampedEvent {
    fn eq(&self, other: &Self) -> bool {
        self.ts_ns == other.ts_ns
    }
}

impl Eq for TimestampedEvent {}

impl PartialOrd for TimestampedEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimestampedEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order for min-heap behavior (we want earliest first)
        other.ts_ns.cmp(&self.ts_ns)
    }
}

fn parse_timestamp(ts: &str) -> i64 {
    // Parse ISO8601 timestamp to nanoseconds
    // Format: 2026-01-25T13:34:27.979374170Z
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(ts) {
        dt.timestamp_nanos_opt().unwrap_or(0)
    } else {
        0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: convert_depth <depth.jsonl> [trades.jsonl] <output.jsonl>");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  convert_depth depth.jsonl output.jsonl");
        eprintln!("  convert_depth depth.jsonl trades.jsonl output.jsonl");
        std::process::exit(1);
    }

    let (depth_path, trades_path, output_path) = if args.len() == 3 {
        (&args[1], None, &args[2])
    } else {
        (&args[1], Some(&args[2]), &args[3])
    };

    let mut events: BinaryHeap<TimestampedEvent> = BinaryHeap::new();
    let mut depth_count = 0;
    let mut trade_count = 0;

    // Read depth data
    eprintln!("Reading depth data from {}...", depth_path);
    let depth_file = std::fs::File::open(depth_path)?;
    let reader = BufReader::new(depth_file);

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        if let Ok(depth) = serde_json::from_str::<BinanceDepth>(&line) {
            let ts_ns = parse_timestamp(&depth.ts);

            let bids: Vec<PriceLevel> = depth
                .bids
                .iter()
                .take(20)
                .map(|l| PriceLevel {
                    price_mantissa: l.price,
                    price_exponent: depth.price_exponent,
                    qty_mantissa: l.qty,
                    qty_exponent: depth.qty_exponent,
                })
                .collect();

            let asks: Vec<PriceLevel> = depth
                .asks
                .iter()
                .take(20)
                .map(|l| PriceLevel {
                    price_mantissa: l.price,
                    price_exponent: depth.price_exponent,
                    qty_mantissa: l.qty,
                    qty_exponent: depth.qty_exponent,
                })
                .collect();

            let book = OrderBook {
                ts_ns,
                symbol: depth.tradingsymbol,
                bids,
                asks,
            };

            events.push(TimestampedEvent {
                ts_ns,
                event: MarketEvent::Book(book),
            });
            depth_count += 1;
        }
    }
    eprintln!("Loaded {} depth snapshots", depth_count);

    // Read trade data if provided
    if let Some(trades_path) = trades_path {
        eprintln!("Reading trade data from {}...", trades_path);
        let trades_file = std::fs::File::open(trades_path)?;
        let reader = BufReader::new(trades_file);

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            if let Ok(trade) = serde_json::from_str::<BinanceTrade>(&line) {
                let ts_ns = parse_timestamp(&trade.ts);

                // is_buyer_maker: true means buyer was maker, so trade was sell-initiated
                let side = if trade.is_buyer_maker {
                    TriState::Present(TradeSide::Sell)
                } else {
                    TriState::Present(TradeSide::Buy)
                };

                let slrt_trade = Trade {
                    ts_ns,
                    symbol: trade.tradingsymbol,
                    price_mantissa: trade.price,
                    price_exponent: trade.price_exponent,
                    qty_mantissa: trade.qty,
                    qty_exponent: trade.qty_exponent,
                    side,
                };

                events.push(TimestampedEvent {
                    ts_ns,
                    event: MarketEvent::Trade(slrt_trade),
                });
                trade_count += 1;
            }
        }
        eprintln!("Loaded {} trades", trade_count);
    }

    // Write sorted events
    eprintln!("Writing {} events to {}...", events.len(), output_path);
    let mut output = std::fs::File::create(output_path)?;

    // Pop from heap (min-heap gives us earliest first)
    let mut sorted_events: Vec<_> = events.into_sorted_vec();
    sorted_events.reverse(); // sorted_vec gives max-first, we want min-first

    for event in sorted_events {
        writeln!(output, "{}", serde_json::to_string(&event.event)?)?;
    }

    eprintln!(
        "Converted {} depth snapshots + {} trades = {} total events",
        depth_count,
        trade_count,
        depth_count + trade_count
    );
    Ok(())
}
