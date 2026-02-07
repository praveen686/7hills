//! Resilient WebSocket connection with auto-reconnect and liveness monitoring.
//!
//! This module provides a self-healing WebSocket wrapper that:
//! 1. Automatically reconnects on disconnect with exponential backoff
//! 2. Monitors liveness and forces reconnect if no data for N seconds
//! 3. Records connection gaps for audit/manifest purposes
//!
//! # Usage
//! ```ignore
//! let ws = ResilientWs::connect("wss://...", ResilientWsConfig::default()).await?;
//! while let Some(msg) = ws.next_message().await? {
//!     // process msg
//! }
//! let gaps = ws.connection_gaps();
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::Instant;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

/// Configuration for resilient WebSocket connection.
#[derive(Debug, Clone)]
pub struct ResilientWsConfig {
    /// Initial backoff duration after disconnect (default: 1s)
    pub initial_backoff: Duration,
    /// Maximum backoff duration (default: 30s)
    pub max_backoff: Duration,
    /// Backoff multiplier (default: 2.0)
    pub backoff_multiplier: f64,
    /// Liveness timeout - force reconnect if no message for this long (default: 30s)
    pub liveness_timeout: Duration,
    /// Read timeout per message attempt (default: 5s)
    pub read_timeout: Duration,
    /// Maximum consecutive reconnect attempts before giving up (default: 100)
    pub max_reconnect_attempts: u32,
    /// Whether to send ping frames to keep connection alive (default: true)
    pub enable_ping: bool,
    /// Ping interval if enabled (default: 30s)
    pub ping_interval: Duration,
}

impl Default for ResilientWsConfig {
    fn default() -> Self {
        Self {
            initial_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            liveness_timeout: Duration::from_secs(30),
            read_timeout: Duration::from_secs(5),
            max_reconnect_attempts: 100,
            enable_ping: true,
            ping_interval: Duration::from_secs(30),
        }
    }
}

/// A recorded gap in the connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionGap {
    /// When the disconnect was detected
    pub disconnect_ts: DateTime<Utc>,
    /// When reconnection succeeded
    pub reconnect_ts: DateTime<Utc>,
    /// Duration of the gap in milliseconds
    pub gap_ms: u64,
    /// Reason for disconnect (if known)
    pub reason: String,
    /// Number of reconnect attempts before success
    pub attempts: u32,
}

/// Resilient WebSocket connection state.
enum WsState {
    Connected {
        read: futures_util::stream::SplitStream<
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
        >,
        write: futures_util::stream::SplitSink<
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
            Message,
        >,
        last_message: Instant,
        last_ping: Instant,
    },
    Disconnected {
        since: DateTime<Utc>,
        reason: String,
        attempts: u32,
        current_backoff: Duration,
    },
}

/// A resilient WebSocket connection that auto-reconnects.
pub struct ResilientWs {
    url: String,
    config: ResilientWsConfig,
    state: WsState,
    gaps: Vec<ConnectionGap>,
    total_reconnects: u32,
}

impl ResilientWs {
    /// Connect to a WebSocket URL with resilient handling.
    pub async fn connect(url: &str, config: ResilientWsConfig) -> Result<Self> {
        info!(url = %url, "Connecting to WebSocket (resilient mode)");

        let (ws_stream, _) = connect_async(url)
            .await
            .with_context(|| format!("Initial connection to {}", url))?;

        let (write, read) = ws_stream.split();
        let now = Instant::now();

        Ok(Self {
            url: url.to_string(),
            config,
            state: WsState::Connected {
                read,
                write,
                last_message: now,
                last_ping: now,
            },
            gaps: Vec::new(),
            total_reconnects: 0,
        })
    }

    /// Get the next message, handling reconnection automatically.
    /// Returns None only when max reconnect attempts exhausted.
    pub async fn next_message(&mut self) -> Result<Option<Message>> {
        loop {
            match &mut self.state {
                WsState::Connected {
                    read,
                    write,
                    last_message,
                    last_ping,
                } => {
                    // Check liveness timeout
                    if last_message.elapsed() > self.config.liveness_timeout {
                        warn!(
                            elapsed_secs = last_message.elapsed().as_secs(),
                            timeout_secs = self.config.liveness_timeout.as_secs(),
                            "Liveness timeout exceeded, forcing reconnect"
                        );
                        self.transition_to_disconnected("liveness timeout".to_string());
                        continue;
                    }

                    // Send ping if needed
                    if self.config.enable_ping && last_ping.elapsed() > self.config.ping_interval {
                        if let Err(e) = write.send(Message::Ping(vec![])).await {
                            warn!(error = %e, "Failed to send ping, triggering reconnect");
                            self.transition_to_disconnected(format!("ping failed: {}", e));
                            continue;
                        }
                        *last_ping = Instant::now();
                    }

                    // Try to read next message with timeout
                    let result = tokio::time::timeout(self.config.read_timeout, read.next()).await;

                    match result {
                        Ok(Some(Ok(msg))) => {
                            *last_message = Instant::now();

                            // Handle control frames
                            match &msg {
                                Message::Ping(data) => {
                                    let _ = write.send(Message::Pong(data.clone())).await;
                                    continue;
                                }
                                Message::Pong(_) => {
                                    continue;
                                }
                                Message::Close(frame) => {
                                    let reason = frame
                                        .as_ref()
                                        .map(|f| f.reason.to_string())
                                        .unwrap_or_else(|| "close frame received".to_string());
                                    warn!(reason = %reason, "Received close frame");
                                    self.transition_to_disconnected(reason);
                                    continue;
                                }
                                _ => {}
                            }

                            return Ok(Some(msg));
                        }
                        Ok(Some(Err(e))) => {
                            warn!(error = %e, "WebSocket error");
                            self.transition_to_disconnected(format!("ws error: {}", e));
                            continue;
                        }
                        Ok(None) => {
                            warn!("WebSocket stream ended (None)");
                            self.transition_to_disconnected("stream ended".to_string());
                            continue;
                        }
                        Err(_) => {
                            // Read timeout - normal, just continue
                            debug!("Read timeout, checking liveness");
                            continue;
                        }
                    }
                }
                WsState::Disconnected {
                    since,
                    reason,
                    attempts,
                    current_backoff,
                } => {
                    if *attempts >= self.config.max_reconnect_attempts {
                        error!(
                            attempts = *attempts,
                            max = self.config.max_reconnect_attempts,
                            "Max reconnect attempts exhausted"
                        );
                        return Ok(None);
                    }

                    info!(
                        attempt = *attempts + 1,
                        backoff_ms = current_backoff.as_millis(),
                        reason = %reason,
                        "Attempting reconnection after backoff"
                    );

                    // Wait for backoff
                    tokio::time::sleep(*current_backoff).await;

                    // Try to reconnect
                    match connect_async(&self.url).await {
                        Ok((ws_stream, _)) => {
                            let (write, read) = ws_stream.split();
                            let now_ts = Utc::now();
                            let now_instant = Instant::now();

                            // Record the gap
                            let gap = ConnectionGap {
                                disconnect_ts: *since,
                                reconnect_ts: now_ts,
                                gap_ms: (now_ts - *since).num_milliseconds().max(0) as u64,
                                reason: reason.clone(),
                                attempts: *attempts + 1,
                            };
                            self.gaps.push(gap.clone());
                            self.total_reconnects += 1;

                            info!(
                                gap_ms = gap.gap_ms,
                                attempts = gap.attempts,
                                total_reconnects = self.total_reconnects,
                                "Reconnected successfully"
                            );

                            self.state = WsState::Connected {
                                read,
                                write,
                                last_message: now_instant,
                                last_ping: now_instant,
                            };
                            continue;
                        }
                        Err(e) => {
                            warn!(
                                error = %e,
                                attempt = *attempts + 1,
                                "Reconnection failed, will retry"
                            );
                            *attempts += 1;
                            *current_backoff = Duration::from_secs_f64(
                                (current_backoff.as_secs_f64() * self.config.backoff_multiplier)
                                    .min(self.config.max_backoff.as_secs_f64()),
                            );
                            continue;
                        }
                    }
                }
            }
        }
    }

    /// Transition from Connected to Disconnected state.
    fn transition_to_disconnected(&mut self, reason: String) {
        self.state = WsState::Disconnected {
            since: Utc::now(),
            reason,
            attempts: 0,
            current_backoff: self.config.initial_backoff,
        };
    }

    /// Get all recorded connection gaps.
    pub fn connection_gaps(&self) -> &[ConnectionGap] {
        &self.gaps
    }

    /// Get total number of reconnections.
    pub fn total_reconnects(&self) -> u32 {
        self.total_reconnects
    }

    /// Check if currently connected.
    pub fn is_connected(&self) -> bool {
        matches!(self.state, WsState::Connected { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ResilientWsConfig::default();
        assert_eq!(config.initial_backoff, Duration::from_secs(1));
        assert_eq!(config.max_backoff, Duration::from_secs(30));
        assert_eq!(config.liveness_timeout, Duration::from_secs(30));
    }
}
