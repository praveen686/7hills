//! # TUI Utilities
//!
//! Minimal TUI helpers for QuantLaxmi runners.
//! Full TUI implementations live in their respective binaries (e.g., live_paper_tui).

/// Print headless mode banner to stdout.
///
/// Called when running without TUI (--headless flag).
pub fn print_headless_banner(service_name: &str, initial_capital: f64) {
    println!("═══════════════════════════════════════════════════════════════");
    println!("        {} - Headless Mode", service_name);
    println!("═══════════════════════════════════════════════════════════════");
    println!("[HEADLESS] Initial Capital: ₹{:.2}", initial_capital);
    println!("[HEADLESS] Press Ctrl+C to stop");
    println!("═══════════════════════════════════════════════════════════════");
}
