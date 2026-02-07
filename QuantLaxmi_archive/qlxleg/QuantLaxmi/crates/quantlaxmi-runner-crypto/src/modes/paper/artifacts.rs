//! Artifact writers for paper trading sessions.

use std::path::Path;
use std::sync::Mutex;

use quantlaxmi_executor::{Fill, FillSink};
use quantlaxmi_wal::JsonlWriter;

/// Fill sink that writes to a JSONL file.
///
/// Uses internal synchronization to handle the sync FillSink trait
/// with async file I/O.
pub struct JsonlFillSink {
    w: Mutex<Option<JsonlWriter<Fill>>>,
    rt: tokio::runtime::Handle,
}

impl JsonlFillSink {
    /// Open a JSONL fill sink at the given path.
    pub async fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let w = JsonlWriter::open_append(path).await?;
        Ok(Self {
            w: Mutex::new(Some(w)),
            rt: tokio::runtime::Handle::current(),
        })
    }
}

impl FillSink for JsonlFillSink {
    fn on_fill(&mut self, fill: &Fill) -> anyhow::Result<()> {
        let fill = fill.clone();
        let mut guard = self.w.lock().unwrap();
        if let Some(ref mut w) = *guard {
            // Use block_in_place to run async code from sync context
            // (block_on panics if called from within async, block_in_place doesn't)
            tokio::task::block_in_place(|| {
                self.rt.block_on(async {
                    w.write(&fill).await?;
                    w.flush().await // Flush after each write to ensure durability
                })
            })?;
        }
        Ok(())
    }

    fn flush(&mut self) -> anyhow::Result<()> {
        let mut guard = self.w.lock().unwrap();
        if let Some(ref mut w) = *guard {
            tokio::task::block_in_place(|| self.rt.block_on(async { w.flush().await }))?;
        }
        Ok(())
    }
}

// Implement Send + Sync for JsonlFillSink
unsafe impl Send for JsonlFillSink {}
unsafe impl Sync for JsonlFillSink {}
