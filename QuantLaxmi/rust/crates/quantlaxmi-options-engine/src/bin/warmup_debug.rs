//! Debug warmup - no TUI, just console output

use anyhow::Result;
use chrono::Utc;
use quantlaxmi_connectors_zerodha::ZerodhaAutoDiscovery;
use quantlaxmi_options_engine::{
    aggregate_warmup_data,
    pcr::{OptionData, OptionDataType},
    EngineConfig, OptionsEngine,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging to console
    tracing_subscriber::fmt()
        .with_env_filter("info,quantlaxmi_connectors_zerodha=debug")
        .init();

    println!("=== QuantLaxmi Warmup Debug ===\n");

    // Connect to Zerodha
    println!("Connecting to Zerodha via sidecar...");
    let discovery = ZerodhaAutoDiscovery::from_sidecar()?;
    println!("Connected!\n");

    // NIFTY 50 index token
    let nifty_index_token = 256265u32;
    let lookback_minutes = 60i64;

    println!("Fetching historical data:");
    println!("  Symbol: NIFTY 50");
    println!("  Token: {}", nifty_index_token);
    println!("  Lookback: {} minutes", lookback_minutes);
    println!(
        "  From: {}",
        Utc::now() - chrono::Duration::minutes(lookback_minutes)
    );
    println!("  To: {}", Utc::now());
    println!();

    // Try fetching
    println!("Calling fetch_warmup_data...");
    let data = discovery
        .fetch_warmup_data(
            &[("NIFTY 50".to_string(), nifty_index_token)],
            lookback_minutes,
        )
        .await?;

    println!("\nSUCCESS! Got data for {} symbols", data.len());

    for (symbol, candles) in &data {
        println!("\n  {}: {} candles", symbol, candles.len());

        if candles.is_empty() {
            println!("    (no candles returned)");
            continue;
        }

        if let Some(c) = candles.first() {
            println!(
                "    First: {} O:{:.2} H:{:.2} L:{:.2} C:{:.2}",
                c.timestamp, c.open, c.high, c.low, c.close
            );
        }
        if let Some(c) = candles.last() {
            println!(
                "    Last:  {} O:{:.2} H:{:.2} L:{:.2} C:{:.2}",
                c.timestamp, c.open, c.high, c.low, c.close
            );
        }
    }

    // Convert to feature vectors
    println!("\nConverting to feature vectors...");
    let vectors = aggregate_warmup_data(&data);
    println!("Got {} feature vectors", vectors.len());

    // Create engine and process warmup
    println!("\n=== Processing through Options Engine ===\n");

    let config = EngineConfig {
        symbol: "NIFTY".into(),
        lot_size: 50,
        risk_free_rate: 0.065,
        dividend_yield: 0.012,
        max_positions: 3,
        max_loss_per_position: 5000.0,
        max_portfolio_delta: 500.0,
        min_iv_percentile_sell: 60.0,
        max_iv_percentile_buy: 40.0,
        min_strategy_score: 60.0,
        ramanujan_enabled: true,
        block_on_hft: true,
        pcr_enabled: true,
        pcr_lookback: 100,
    };

    let mut engine = OptionsEngine::new(config);

    println!("Processing {} warmup ticks...", vectors.len());
    println!(
        "(Note: Regime engine needs {} ticks for first subspace)\n",
        32
    );

    for (i, (ts, spot, features)) in vectors.iter().enumerate() {
        engine.on_tick(*ts, *spot, features);

        // Print debug info at key points
        if (i + 1) == 20 || (i + 1) == 32 || (i + 1) == 33 || (i + 1) == 40 || (i + 1) == 60 {
            let status = engine.status();
            let (count, mean_abs, variance, spread) = engine.regime_debug_stats();
            let heuristic = engine.heuristic_regime();
            let proto_count = engine.prototype_count();
            println!("  Tick {}: spot={:.2}", i + 1, spot);
            println!(
                "    Stats: n={} mean_abs={} var={} spread={}",
                count, mean_abs, variance, spread
            );
            println!(
                "    Heuristic: {:?}, Status: {:?}, Prototypes: {}",
                heuristic, status.regime, proto_count
            );
        }
    }

    // Final status
    let status = engine.status();
    println!("\n=== Final Engine Status ===");
    println!("  Regime: {:?}", status.regime);
    println!("  HFT Detected: {}", status.hft_detected);
    println!("  Spot: {:.2}", status.spot);
    println!("  ATM IV: {:.1}%", status.atm_iv * 100.0);
    println!("  IV Percentile: {:.0}", status.iv_percentile);
    println!("  Vol Regime: {:?}", status.vol_regime);
    println!("  PCR: {:.2}", status.pcr);
    println!("  PCR Signal: {:?}", status.pcr_signal);

    // Test strategy decision BEFORE option chain update
    println!("\n=== Strategy Decision (Before Option Chain Update) ===");
    let decision = engine.decide(Utc::now());
    println!("  Action: {:?}", decision.action);
    if let Some(rec) = &decision.strategy {
        println!("  Strategy: {:?}, Score: {:.1}", rec.strategy, rec.score);
        println!(
            "    Regime: {:.1}, Vol: {:.1}, PCR: {:.1}, Risk: {:.1}, Edge: {:.1}",
            rec.component_scores.regime_score,
            rec.component_scores.vol_score,
            rec.component_scores.pcr_score,
            rec.component_scores.risk_score,
            rec.component_scores.edge_score
        );
    }

    // Simulate option chain with different IV levels
    println!("\n=== Simulating Option Chain Updates ===");
    let spot = status.spot;

    // Create mock option chain (ATM strikes near spot)
    let atm_strike = (spot / 50.0).round() * 50.0; // Round to nearest 50
    let options = vec![
        OptionData {
            strike: atm_strike - 100.0,
            expiry_dte: 7,
            option_type: OptionDataType::Call,
            volume: 10000,
            open_interest: 50000,
            last_price: 180.0, // ITM call
            delta: 0.0,
        },
        OptionData {
            strike: atm_strike,
            expiry_dte: 7,
            option_type: OptionDataType::Call,
            volume: 15000,
            open_interest: 80000,
            last_price: 120.0, // ATM call
            delta: 0.0,
        },
        OptionData {
            strike: atm_strike + 100.0,
            expiry_dte: 7,
            option_type: OptionDataType::Call,
            volume: 8000,
            open_interest: 40000,
            last_price: 70.0, // OTM call
            delta: 0.0,
        },
        OptionData {
            strike: atm_strike - 100.0,
            expiry_dte: 7,
            option_type: OptionDataType::Put,
            volume: 8000,
            open_interest: 35000,
            last_price: 60.0, // OTM put
            delta: 0.0,
        },
        OptionData {
            strike: atm_strike,
            expiry_dte: 7,
            option_type: OptionDataType::Put,
            volume: 12000,
            open_interest: 70000,
            last_price: 115.0, // ATM put
            delta: 0.0,
        },
        OptionData {
            strike: atm_strike + 100.0,
            expiry_dte: 7,
            option_type: OptionDataType::Put,
            volume: 10000,
            open_interest: 45000,
            last_price: 175.0, // ITM put
            delta: 0.0,
        },
    ];

    println!(
        "  Updating engine with {} options (spot={:.2}, ATM strike={:.0})",
        options.len(),
        spot,
        atm_strike
    );
    engine.on_chain_update(Utc::now(), &options, spot);

    // Check status after option chain update
    let status = engine.status();
    println!("\n=== Status After Option Chain Update ===");
    println!("  ATM IV: {:.1}%", status.atm_iv * 100.0);
    println!("  IV Percentile: {:.0}", status.iv_percentile);
    println!("  Vol Regime: {:?}", status.vol_regime);
    println!("  PCR: {:.2}", status.pcr);
    println!("  PCR Signal: {:?}", status.pcr_signal);

    // Test strategy decision AFTER option chain update
    println!("\n=== Strategy Decision (After Option Chain Update) ===");
    let decision = engine.decide(Utc::now());
    println!("  Action: {:?}", decision.action);
    println!("  Confidence: {:.2}", decision.confidence);
    if let Some(rec) = &decision.strategy {
        println!("  Strategy: {:?}, Score: {:.1}", rec.strategy, rec.score);
        println!(
            "    Regime: {:.1}, Vol: {:.1}, PCR: {:.1}, Risk: {:.1}, Edge: {:.1}",
            rec.component_scores.regime_score,
            rec.component_scores.vol_score,
            rec.component_scores.pcr_score,
            rec.component_scores.risk_score,
            rec.component_scores.edge_score
        );
    } else {
        println!("  Strategy: None");
    }

    println!("\n=== Debug Complete ===");
    Ok(())
}
