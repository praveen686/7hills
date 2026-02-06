//! Paper trading mode for crypto.
//!
//! ## Architecture
//! ```text
//! Binance SBE Depth → WAL Writer → PaperEngine → Fills
//!                         ↓              ↓
//!                   depth_wal.jsonl  fills.jsonl
//!                                        ↑
//!                    HTTP /order ────────┘
//! ```

mod artifacts;
mod http;
mod run;

pub use run::{PaperModeConfig, run_paper_mode};
