//! Fill sink trait and implementations.

use crate::sim::Fill;

/// Trait for consuming fills (logging, persistence, etc.)
pub trait FillSink: Send + Sync {
    /// Called when a fill is generated.
    fn on_fill(&mut self, fill: &Fill) -> anyhow::Result<()>;

    /// Flush any buffered data.
    fn flush(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// No-op fill sink for testing.
pub struct NoopFillSink;

impl FillSink for NoopFillSink {
    fn on_fill(&mut self, _fill: &Fill) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Fill sink that logs fills.
pub struct LoggingFillSink;

impl FillSink for LoggingFillSink {
    fn on_fill(&mut self, fill: &Fill) -> anyhow::Result<()> {
        tracing::info!(
            "FILL: {} {} {:.4} @ {:.2} (fee={:.4}, {:?})",
            fill.side,
            fill.symbol,
            fill.qty,
            fill.price,
            fill.fee,
            fill.fill_type
        );
        Ok(())
    }
}

/// Fill sink that collects fills in memory.
pub struct VecFillSink {
    fills: Vec<Fill>,
}

impl VecFillSink {
    pub fn new() -> Self {
        Self { fills: Vec::new() }
    }

    pub fn fills(&self) -> &[Fill] {
        &self.fills
    }

    pub fn into_fills(self) -> Vec<Fill> {
        self.fills
    }
}

impl Default for VecFillSink {
    fn default() -> Self {
        Self::new()
    }
}

impl FillSink for VecFillSink {
    fn on_fill(&mut self, fill: &Fill) -> anyhow::Result<()> {
        self.fills.push(fill.clone());
        Ok(())
    }
}
