# WebSocket Auto-Reconnect Integration Guide

## Problem

The perp capture functions break on WebSocket disconnect with no recovery:
```rust
// binance_perp_capture.rs:200
Ok(None) => break,  // Stream ended, just give up
```

This caused the 17-hour gap in the `perp_20260129_053328` session.

## Solution

New module `ws_resilient.rs` provides `ResilientWs` with:
- Auto-reconnect with exponential backoff (1s → 30s)
- Liveness watchdog (force reconnect if no data for 30s)
- Gap recording for manifest/audit

## Integration Steps

### Step 1: Add module to lib.rs

```rust
// crates/quantlaxmi-runner-crypto/src/lib.rs
pub mod ws_resilient;
```

### Step 2: Refactor capture_perp_bookticker_jsonl

**Before:**
```rust
pub async fn capture_perp_bookticker_jsonl(
    symbol: &str,
    out_path: &Path,
    duration_secs: u64,
) -> Result<PerpCaptureStats> {
    let url = format!("wss://fstream.binance.com/ws/{}@bookTicker", symbol.to_lowercase());
    let (ws_stream, _) = connect_async(&url).await?;
    let (_write, mut read) = ws_stream.split();

    // ...

    while tokio::time::Instant::now() < deadline {
        let msg = tokio::time::timeout(Duration::from_secs(5), read.next()).await;
        match msg {
            Ok(Some(v)) => { /* process */ }
            Ok(None) => break,      // <-- BUG: gives up on disconnect
            Err(_) => continue,
        }
    }
}
```

**After:**
```rust
use crate::ws_resilient::{ResilientWs, ResilientWsConfig, ConnectionGap};

pub async fn capture_perp_bookticker_jsonl(
    symbol: &str,
    out_path: &Path,
    duration_secs: u64,
) -> Result<PerpCaptureResult> {
    let url = format!("wss://fstream.binance.com/ws/{}@bookTicker", symbol.to_lowercase());

    let config = ResilientWsConfig {
        liveness_timeout: Duration::from_secs(30),
        read_timeout: Duration::from_secs(5),
        ..Default::default()
    };

    let mut ws = ResilientWs::connect(&url, config).await?;

    let deadline = Instant::now() + Duration::from_secs(duration_secs);
    let mut stats = PerpCaptureStats::default();

    while Instant::now() < deadline {
        let msg = match ws.next_message().await? {
            Some(m) => m,
            None => {
                // Max reconnects exhausted - this is a hard failure
                error!("WebSocket reconnection exhausted");
                break;
            }
        };

        if !msg.is_text() { continue; }
        // ... rest of processing unchanged ...
    }

    // Return gaps for manifest
    Ok(PerpCaptureResult {
        stats,
        gaps: ws.connection_gaps().to_vec(),
        total_reconnects: ws.total_reconnects(),
    })
}
```

### Step 3: Update PerpCaptureStats to include gap info

```rust
#[derive(Debug, Default)]
pub struct PerpCaptureResult {
    pub stats: PerpCaptureStats,
    pub gaps: Vec<ConnectionGap>,
    pub total_reconnects: u32,
}
```

### Step 4: Record gaps in session manifest

```rust
// binance_perp_session.rs - in manifest generation
pub struct SymbolManifestEntry {
    // existing fields...
    pub connection_gaps: Vec<ConnectionGap>,  // NEW
    pub total_reconnects: u32,                // NEW
}
```

## Functions to Update

| File | Function | Priority |
|------|----------|----------|
| `binance_perp_capture.rs` | `capture_perp_bookticker_jsonl` | HIGH |
| `binance_perp_capture.rs` | `capture_perp_depth_jsonl` | HIGH |
| `binance_perp_capture.rs` | `capture_book_ticker_jsonl` (spot) | HIGH |
| `binance_funding_capture.rs` | `capture_funding_jsonl` | MEDIUM |

## Testing

```bash
# Simulate disconnect by killing network briefly
sudo iptables -A OUTPUT -p tcp --dport 443 -j DROP
sleep 5
sudo iptables -D OUTPUT -p tcp --dport 443 -j DROP

# Observe logs for reconnection
tail -f /tmp/capture_phase27c.log | grep -i reconnect
```

## Expected Behavior After Fix

1. **On disconnect**: Log warning, start backoff timer
2. **After backoff**: Attempt reconnect
3. **On reconnect**: Log success, record gap, continue capturing
4. **On max retries**: Hard error, capture fails gracefully
5. **In manifest**: Gaps recorded with timestamps and durations

## Monitoring

Add to capture log output:
```
[INFO] Reconnected successfully (gap_ms=2340, attempts=2, total_reconnects=1)
```

Check capture health:
```bash
# Should see recent activity on all feeds
ls -la data/perp_sessions/perp_*/BTCUSDT/*.jsonl

# Check for gaps in manifest
jq '.symbols[0].connection_gaps' data/perp_sessions/perp_*/session_manifest.json
```
