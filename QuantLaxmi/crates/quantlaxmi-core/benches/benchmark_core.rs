use chrono::Utc;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use quantlaxmi_core::{EventBus, MarketPayload, WalMarketRecord};
use tokio::runtime::Runtime;

fn bench_event_bus(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let bus = EventBus::new(1000);

    c.bench_function("event_bus_publish_market", |b| {
        b.iter(|| {
            rt.block_on(async {
                let event = WalMarketRecord {
                    ts: Utc::now(),
                    symbol: black_box("BTCUSDT".to_string()),
                    payload: MarketPayload::Quote {
                        bid_price_mantissa: 5000000, // 50000.00
                        ask_price_mantissa: 5000100, // 50001.00
                        bid_qty_mantissa: 100,
                        ask_qty_mantissa: 100,
                        price_exponent: -2,
                        qty_exponent: -2,
                    },
                    ctx: Default::default(),
                };
                bus.publish_market(event).await.unwrap();
            })
        });
    });
}

fn bench_sbe_decoding(c: &mut Criterion) {
    use quantlaxmi_sbe::{BinanceSbeDecoder, SbeHeader};

    // Example Binance SBE Trade message (Template ID 10000)
    // Header (8 bytes) + Body
    let mut trade_bin = [0u8; 64];
    trade_bin[2..4].copy_from_slice(&10000u16.to_le_bytes()); // Template ID
    // Minimal data for decoder to not panic

    c.bench_function("binance_sbe_trade_decode", |b| {
        b.iter(|| {
            let header = SbeHeader::decode(&trade_bin[0..8]).unwrap();
            let _ = black_box(BinanceSbeDecoder::decode_trade(&header, &trade_bin[8..]));
        });
    });
}

fn bench_e2e_tick_to_decision(c: &mut Criterion) {
    use quantlaxmi_core::MomentumStrategy;
    use quantlaxmi_core::Strategy;
    use std::sync::Arc;

    let rt = Runtime::new().unwrap();
    let bus = Arc::new(EventBus::new(1000));

    c.bench_function("e2e_tick_to_strategy_decision", |b| {
        b.iter(|| {
            rt.block_on(async {
                // Simulate quote arrival (canonical mantissa-based)
                let record = WalMarketRecord {
                    ts: Utc::now(),
                    symbol: black_box("BTCUSDT".to_string()),
                    payload: MarketPayload::Quote {
                        bid_price_mantissa: 5000000, // 50000.00
                        ask_price_mantissa: 5000100, // 50001.00
                        bid_qty_mantissa: 100,
                        ask_qty_mantissa: 100,
                        price_exponent: -2,
                        qty_exponent: -2,
                    },
                    ctx: Default::default(),
                };

                // Publish to bus
                bus.publish_market(record.clone()).await.unwrap();

                // Strategy processes market event
                let mut strategy = MomentumStrategy::new(5);
                strategy.on_market(&record);
            })
        });
    });
}

criterion_group!(
    benches,
    bench_event_bus,
    bench_sbe_decoding,
    bench_e2e_tick_to_decision
);
criterion_main!(benches);
