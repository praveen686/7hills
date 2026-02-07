use crate::paper::state::UiSnapshot;
use tokio::sync::watch;

/// Shared telemetry bus for the TUI.
/// Runner publishes snapshots; TUI subscribes via watch::Receiver.
#[derive(Clone)]
pub struct TelemetryBus {
    tx: watch::Sender<UiSnapshot>,
}

impl TelemetryBus {
    pub fn new() -> (Self, watch::Receiver<UiSnapshot>) {
        let (tx, rx) = watch::channel(UiSnapshot::default());
        (Self { tx }, rx)
    }

    pub fn publish(&self, snap: UiSnapshot) {
        // Ignore errors if no receivers.
        let _ = self.tx.send(snap);
    }

    pub fn sender(&self) -> watch::Sender<UiSnapshot> {
        self.tx.clone()
    }
}

impl Default for TelemetryBus {
    fn default() -> Self {
        Self::new().0
    }
}
