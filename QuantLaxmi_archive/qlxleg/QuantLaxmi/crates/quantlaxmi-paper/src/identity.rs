//! Instrument Identity Trait
//!
//! Provides a market-agnostic way for the engine to identify instruments
//! from trade intents. This keeps the engine generic while allowing
//! proper position tracking.
//!
//! ## Usage
//!
//! India:
//! ```ignore
//! impl InstrumentIdentity for TradeIntent {
//!     type Key = u32; // Zerodha instrument_token
//!     fn instrument_key(&self) -> u32 { self.instrument_token }
//! }
//! ```
//!
//! Crypto (future):
//! ```ignore
//! impl InstrumentIdentity for CryptoIntent {
//!     type Key = u32; // Internal symbol ID
//!     fn instrument_key(&self) -> u32 { self.symbol_id }
//! }
//! ```

use std::fmt::Debug;
use std::hash::Hash;

/// Trait for extracting instrument identity from trade intents.
///
/// The engine uses this to:
/// - Track positions per instrument
/// - Log fills and rejections with proper identity
/// - Maintain venue-agnostic position accounting
pub trait InstrumentIdentity {
    /// The key type used to identify instruments.
    ///
    /// Must be:
    /// - `Copy` for efficient passing
    /// - `Eq + Hash` for use in HashMaps
    /// - `Debug` for logging
    type Key: Copy + Eq + Hash + Debug;

    /// Extract the instrument key from this intent.
    fn instrument_key(&self) -> Self::Key;
}
