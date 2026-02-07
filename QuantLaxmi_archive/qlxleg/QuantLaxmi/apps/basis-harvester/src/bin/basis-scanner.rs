//! Print basis ranking table.
//!
//! Fetches Binance premium index + 24h volume, filters by liquidity,
//! and ranks by |basis_bps| for mean-reversion.
//!
//! Usage: cargo run -p basis-harvester --bin basis-scanner [TOP_N] [REFRESH_SECS]

use std::collections::HashMap;
use std::time::Duration;

use basis_harvester::scanner;

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

    println!(
        "Basis Harvester — Scanner (top {} liquid by |basis|, refresh {}s)",
        top_n, refresh_secs
    );
    println!("Filters: vol >= $2M/day, |basis| <= 200 bps");
    println!("Press Ctrl+C to stop.\n");

    // Cache volumes — refresh less frequently
    let mut volumes: HashMap<String, f64> = scanner::fetch_24h_volumes().await.unwrap_or_default();
    let mut vol_refresh = std::time::Instant::now();

    loop {
        // Refresh volumes every 60s
        if vol_refresh.elapsed() >= Duration::from_secs(60) {
            if let Ok(v) = scanner::fetch_24h_volumes().await {
                volumes = v;
            }
            vol_refresh = std::time::Instant::now();
        }

        match scanner::fetch_premium_index().await {
            Ok(entries) => {
                let opps = scanner::rank_basis_opportunities(&entries, &volumes);

                print!("\x1B[2J\x1B[H");

                let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
                println!(
                    "Basis Harvester Scanner  |  {}  |  {} liquid pairs (vol>$5M, |basis|<=50bp)\n",
                    now,
                    opps.len()
                );
                println!("{}", scanner::format_header());
                println!("{}", "-".repeat(72));

                for (i, opp) in opps.iter().take(top_n).enumerate() {
                    println!("{}", scanner::format_row(i, opp));
                }

                let positive = opps.iter().filter(|o| o.basis_bps > 0.0).count();
                let negative = opps.iter().filter(|o| o.basis_bps < 0.0).count();
                let top_basis = opps.first().map(|o| o.abs_basis_bps).unwrap_or(0.0);

                println!("\n{}", "-".repeat(72));
                println!(
                    "Summary: {} positive basis, {} negative | Top |basis|: {:.1} bps",
                    positive, negative, top_basis
                );
                println!("Next refresh in {}s...", refresh_secs);
            }
            Err(e) => {
                eprintln!("Error fetching data: {}", e);
            }
        }

        tokio::time::sleep(Duration::from_secs(refresh_secs)).await;
    }
}
