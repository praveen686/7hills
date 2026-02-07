//! Phase 1 MVP: Print ranked funding rate table.
//!
//! Zero config, no API key needed. Fetches public Binance data.
//!
//! Usage: cargo run -p funding-harvester --bin funding-scanner

use std::time::Duration;

use funding_harvester::scanner;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let top_n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    let refresh_secs: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    println!("Funding Harvester — Scanner (top {} USDT perps, refresh {}s)", top_n, refresh_secs);
    println!("Press Ctrl+C to stop.\n");

    loop {
        match scanner::fetch_premium_index().await {
            Ok(entries) => {
                let opps = scanner::rank_opportunities(&entries);

                // Clear screen
                print!("\x1B[2J\x1B[H");

                let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
                println!("Funding Harvester Scanner  |  {}  |  {} USDT perps\n", now, entries.len());
                println!("{}", scanner::format_header());
                println!("{}", "-".repeat(78));

                for (i, opp) in opps.iter().take(top_n).enumerate() {
                    println!("{}", scanner::format_row(i, opp));
                }

                // Summary
                let positive = opps.iter().filter(|o| o.funding_rate > 0.0).count();
                let negative = opps.iter().filter(|o| o.funding_rate < 0.0).count();
                let top_ann = opps.first().map(|o| o.annualized_pct).unwrap_or(0.0);

                println!("\n{}", "-".repeat(78));
                println!(
                    "Summary: {} positive, {} negative | Top annualized: {:.1}%",
                    positive, negative, top_ann
                );
                println!("Next refresh in {}s...", refresh_secs);
            }
            Err(e) => {
                eprintln!("Error fetching premium index: {}", e);
            }
        }

        tokio::time::sleep(Duration::from_secs(refresh_secs)).await;
    }
}
