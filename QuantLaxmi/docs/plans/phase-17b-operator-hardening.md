# Phase 17B: Operator Control Hardening (baseline_v1)

## Objective
Make operator control safe-by-default in real deployments:
- Explicit confirmation flows
- Authentication / authorization (minimal viable)
- Rate-limit + cooldown enforcement at transport edge
- Deterministic correlation + replayable operator command audit
- UI command ergonomics (TUI/CLI "danger actions" guardrails)

## Frozen Laws
1. **No new execution semantics** - Phase 16 only
2. **WAL-first** - Command must produce WAL event(s) before effect
3. **Least privilege** - Read-only clients can't mutate
4. **Explicit intent** - All destructive actions require confirmation & reason
5. **No backward compatibility** - Greenfield project, all fields required
6. **Canonical bytes for signing** - Never use JSON for signature payloads
7. **Secrets never logged** - Redact secret_hex, signature_hex at all log levels
8. **Auth bound to session** - Include session_id in signed payload for replay protection

---

## Deliverables

### 17B.1: Command Authentication (Minimal Viable)

**File:** `crates/quantlaxmi-runner-common/src/operator_auth.rs`

```rust
//! Operator authentication and authorization.
//!
//! Minimal viable auth using HMAC-SHA256 over canonical bytes.
//! No OAuth, no user DB - just deterministic signing.

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

/// Schema version for operator auth.
pub const OPERATOR_AUTH_SCHEMA_VERSION: &str = "operator_auth_v1.0";

/// Operator authentication payload.
///
/// Attached to OperatorRequest for server verification.
///
/// SECURITY: Implements custom Debug to redact signature.
/// Logging raw signatures aids attackers in replay analysis.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperatorAuth {
    /// Operator identifier (matches key in operator_keys config).
    pub operator_id: String,

    /// Unique nonce to prevent replay (monotonic counter or UUID).
    pub nonce: String,

    /// Timestamp of signing (nanoseconds).
    pub ts_ns: i64,

    /// HMAC-SHA256 signature over canonical payload.
    /// Hex-encoded, 64 characters.
    /// NEVER LOG THIS VALUE - use custom Debug impl.
    pub signature: String,
}

impl std::fmt::Debug for OperatorAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperatorAuth")
            .field("operator_id", &self.operator_id)
            .field("nonce", &self.nonce)
            .field("ts_ns", &self.ts_ns)
            .field("signature", &"[REDACTED]")  // NEVER log signatures
            .finish()
    }
}

impl OperatorAuth {
    /// Create new auth with signature.
    pub fn new(operator_id: String, nonce: String, ts_ns: i64, secret: &[u8], payload: &[u8]) -> Self {
        let signature = Self::compute_signature(secret, &operator_id, &nonce, ts_ns, payload);
        Self {
            operator_id,
            nonce,
            ts_ns,
            signature,
        }
    }

    /// Compute HMAC-SHA256 signature over canonical payload.
    ///
    /// Canonical format (fixed order, deterministic):
    ///   len(operator_id) as u32 LE || operator_id bytes
    ///   || len(nonce) as u32 LE || nonce bytes
    ///   || ts_ns as i64 LE
    ///   || len(payload) as u32 LE || payload bytes
    ///
    /// The payload is AuthenticatedRequest::signing_payload(), which includes:
    /// - correlation_id (length-prefixed)
    /// - session_id (length-prefixed) - CRITICAL for replay protection
    /// - request canonical bytes (length-prefixed)
    ///
    /// This encoding is deterministic: same inputs → same signature, always.
    pub fn compute_signature(
        secret: &[u8],
        operator_id: &str,
        nonce: &str,
        ts_ns: i64,
        payload: &[u8],
    ) -> String {
        let mut mac = Hmac::<Sha256>::new_from_slice(secret)
            .expect("HMAC can take key of any size");

        // Canonical order with length prefixes for determinism
        // operator_id: len(u32 LE) + bytes
        mac.update(&(operator_id.len() as u32).to_le_bytes());
        mac.update(operator_id.as_bytes());

        // nonce: len(u32 LE) + bytes
        mac.update(&(nonce.len() as u32).to_le_bytes());
        mac.update(nonce.as_bytes());

        // ts_ns: i64 LE
        mac.update(&ts_ns.to_le_bytes());

        // payload: len(u32 LE) + bytes
        mac.update(&(payload.len() as u32).to_le_bytes());
        mac.update(payload);

        let result = mac.finalize();
        hex::encode(result.into_bytes())
    }

    /// Verify signature against secret and payload.
    pub fn verify(&self, secret: &[u8], payload: &[u8]) -> bool {
        let expected = Self::compute_signature(
            secret,
            &self.operator_id,
            &self.nonce,
            self.ts_ns,
            payload,
        );
        // Constant-time comparison
        constant_time_eq(self.signature.as_bytes(), expected.as_bytes())
    }

    /// Check if auth is expired (default: 5 minute window).
    pub fn is_expired(&self, now_ns: i64, max_age_ns: i64) -> bool {
        let age = now_ns.saturating_sub(self.ts_ns);
        age > max_age_ns || age < 0 // Also reject future timestamps
    }
}

/// Constant-time comparison to prevent timing attacks.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

// =============================================================================
// Operator Role & RBAC
// =============================================================================

/// Operator role for RBAC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum OperatorRole {
    /// Read-only: can connect WS, call /status, receive broadcasts.
    Observer = 0,

    /// Can send standard OperatorRequest (ForceHalt, ForceReduceOnly, etc.)
    Operator = 1,

    /// Can perform dangerous actions (EmergencyFlatten, CancelAllOrders, AdjustLimit).
    Admin = 2,
}

impl OperatorRole {
    /// Check if role can perform the given action.
    pub fn can_perform(&self, action: &OperatorAction) -> bool {
        match action {
            OperatorAction::Read => true, // All roles can read
            OperatorAction::Control => *self >= OperatorRole::Operator,
            OperatorAction::Dangerous => *self >= OperatorRole::Admin,
        }
    }
}

impl Default for OperatorRole {
    fn default() -> Self {
        OperatorRole::Observer
    }
}

/// Action classification for RBAC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorAction {
    /// Read-only operations (status, subscribe).
    Read,
    /// Standard control operations (halt, reduce-only, clear, restore).
    Control,
    /// Dangerous operations requiring Admin role.
    Dangerous,
}

// =============================================================================
// Operator Key Store
// =============================================================================

/// Operator key entry in configuration.
///
/// SECURITY: Implements custom Debug to redact secret_hex.
/// Never log secrets at any level.
#[derive(Clone, Serialize, Deserialize)]
pub struct OperatorKeyEntry {
    /// Operator identifier.
    pub operator_id: String,

    /// Role for this operator.
    pub role: OperatorRole,

    /// HMAC secret (hex-encoded, 32+ bytes recommended).
    /// NEVER LOG THIS VALUE - use custom Debug impl.
    pub secret_hex: String,

    /// Whether this operator is enabled.
    pub enabled: bool,
}

impl std::fmt::Debug for OperatorKeyEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperatorKeyEntry")
            .field("operator_id", &self.operator_id)
            .field("role", &self.role)
            .field("secret_hex", &"[REDACTED]")  // NEVER log secrets
            .field("enabled", &self.enabled)
            .finish()
    }
}

/// Operator key store loaded from config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorKeyStore {
    /// Schema version.
    pub schema_version: String,

    /// Operator keys indexed by operator_id.
    pub operators: Vec<OperatorKeyEntry>,

    /// Max auth age in nanoseconds.
    pub max_auth_age_ns: i64,
}

impl OperatorKeyStore {
    /// Load from JSON file.
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, OperatorAuthError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| OperatorAuthError::ConfigLoadFailed { source: e.to_string() })?;
        serde_json::from_str(&content)
            .map_err(|e| OperatorAuthError::ConfigParseFailed { source: e.to_string() })
    }

    /// Load from environment variable (JSON string).
    pub fn load_from_env(var_name: &str) -> Result<Self, OperatorAuthError> {
        let content = std::env::var(var_name)
            .map_err(|_| OperatorAuthError::ConfigLoadFailed {
                source: format!("Environment variable {} not set", var_name),
            })?;
        serde_json::from_str(&content)
            .map_err(|e| OperatorAuthError::ConfigParseFailed { source: e.to_string() })
    }

    /// Get operator entry by ID.
    pub fn get(&self, operator_id: &str) -> Option<&OperatorKeyEntry> {
        self.operators.iter().find(|e| e.operator_id == operator_id && e.enabled)
    }

    /// Verify auth and return role if valid.
    pub fn verify_auth(
        &self,
        auth: &OperatorAuth,
        payload: &[u8],
        now_ns: i64,
    ) -> Result<OperatorRole, OperatorAuthError> {
        // Check expiry
        if auth.is_expired(now_ns, self.max_auth_age_ns) {
            return Err(OperatorAuthError::AuthExpired {
                age_ns: now_ns.saturating_sub(auth.ts_ns),
            });
        }

        // Find operator
        let entry = self.get(&auth.operator_id).ok_or_else(|| {
            OperatorAuthError::UnknownOperator {
                operator_id: auth.operator_id.clone(),
            }
        })?;

        // Decode secret
        let secret = hex::decode(&entry.secret_hex)
            .map_err(|_| OperatorAuthError::InvalidSecretFormat)?;

        // Verify signature
        if !auth.verify(&secret, payload) {
            return Err(OperatorAuthError::InvalidSignature);
        }

        Ok(entry.role)
    }
}

// =============================================================================
// Errors
// =============================================================================

/// Operator authentication errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum OperatorAuthError {
    #[error("Config load failed: {source}")]
    ConfigLoadFailed { source: String },

    #[error("Config parse failed: {source}")]
    ConfigParseFailed { source: String },

    #[error("Unknown operator: {operator_id}")]
    UnknownOperator { operator_id: String },

    #[error("Invalid secret format")]
    InvalidSecretFormat,

    #[error("Invalid signature")]
    InvalidSignature,

    #[error("Auth expired (age: {age_ns}ns)")]
    AuthExpired { age_ns: i64 },

    #[error("Insufficient permissions: {required:?} required, have {actual:?}")]
    InsufficientPermissions {
        required: OperatorRole,
        actual: OperatorRole,
    },
}
```

**Modify:** `crates/quantlaxmi-runner-common/src/control_view.rs`

Add `OperatorAuth` to `OperatorRequest`:

```rust
/// Operator request types (maps exactly to Phase 16 operations).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "action", content = "params")]
pub enum OperatorRequest {
    // ... existing variants unchanged ...
}

impl OperatorRequest {
    /// Canonical bytes encoding for deterministic signing.
    ///
    /// Format:
    /// - Byte 0: variant discriminant (u8)
    /// - Bytes 1+: variant-specific payload (fixed order, length-prefixed strings)
    ///
    /// Example for ForceHalt { operator_id, reason, source }:
    ///   0x01 || len(operator_id) || operator_id || len(reason) || reason || source_byte
    ///
    /// This encoding is deterministic: same input → same bytes, always.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        match self {
            OperatorRequest::ForceHalt { operator_id, reason, source } => {
                buf.push(0x01); // discriminant
                // operator_id: len(u32 LE) + UTF-8
                buf.extend_from_slice(&(operator_id.len() as u32).to_le_bytes());
                buf.extend_from_slice(operator_id.as_bytes());
                // reason: len(u32 LE) + UTF-8
                buf.extend_from_slice(&(reason.len() as u32).to_le_bytes());
                buf.extend_from_slice(reason.as_bytes());
                // source: enum discriminant (u8)
                buf.push(source.canonical_discriminant());
            }
            OperatorRequest::ForceReduceOnly { operator_id, reason, source } => {
                buf.push(0x02);
                buf.extend_from_slice(&(operator_id.len() as u32).to_le_bytes());
                buf.extend_from_slice(operator_id.as_bytes());
                buf.extend_from_slice(&(reason.len() as u32).to_le_bytes());
                buf.extend_from_slice(reason.as_bytes());
                buf.push(source.canonical_discriminant());
            }
            // ... similar patterns for all variants ...
            // Each variant has a unique discriminant and fixed-order field encoding
        }

        buf
    }
}

/// Authenticated operator request wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticatedRequest {
    /// Correlation ID for idempotency and audit linkage (required).
    pub correlation_id: String,

    /// Session ID for replay protection (required).
    /// Binds this command to a specific session - prevents cross-session replay.
    pub session_id: String,

    /// Authentication (required).
    pub auth: OperatorAuth,

    /// The actual request.
    pub request: OperatorRequest,
}

impl AuthenticatedRequest {
    /// Get canonical bytes for signing.
    ///
    /// NEVER uses JSON - uses deterministic binary encoding:
    /// - Strings: length (u32 LE) + UTF-8 bytes
    /// - Integers: LE bytes
    /// - Enums: discriminant (u8) + variant payload
    /// - Options: presence byte (0/1) + payload if present
    ///
    /// Field order is FIXED and must match this exact sequence.
    pub fn signing_payload(&self) -> Vec<u8> {
        let mut payload = Vec::new();

        // 1. correlation_id: len(u32 LE) + UTF-8 bytes
        payload.extend_from_slice(&(self.correlation_id.len() as u32).to_le_bytes());
        payload.extend_from_slice(self.correlation_id.as_bytes());

        // 2. session_id: len(u32 LE) + UTF-8 bytes
        payload.extend_from_slice(&(self.session_id.len() as u32).to_le_bytes());
        payload.extend_from_slice(self.session_id.as_bytes());

        // 3. request: canonical_bytes (uses OperatorRequest::canonical_bytes)
        let request_bytes = self.request.canonical_bytes();
        payload.extend_from_slice(&(request_bytes.len() as u32).to_le_bytes());
        payload.extend_from_slice(&request_bytes);

        payload
    }

    /// Classify action for RBAC.
    pub fn action_class(&self) -> OperatorAction {
        use crate::control_view::OperatorRequest::*;
        match &self.request {
            ForceHalt { .. } | ForceReduceOnly { .. } | ClearHalt { .. } | RestoreFull { .. } => {
                OperatorAction::Control
            }
            ActivateKillSwitch { scope, .. } => {
                // Global kill-switch is dangerous
                if matches!(scope, KillSwitchScope::Global) {
                    OperatorAction::Dangerous
                } else {
                    OperatorAction::Control
                }
            }
            DeactivateKillSwitch { .. } => OperatorAction::Control,
            EmergencyFlatten { .. } | CancelAllOrders { .. } | AdjustLimit { .. } => {
                OperatorAction::Dangerous
            }
        }
    }
}
```

---

### 17B.2: Transport-level RBAC

**File:** `crates/quantlaxmi-runner-common/src/command_gate.rs`

```rust
//! Command gate: rate limiting, idempotency, and RBAC enforcement.
//!
//! Sits at the transport edge, before commands reach the controller.

use crate::control_view::{AuthenticatedRequest, OperatorResponse, OperatorOutcome};
use crate::operator_auth::{OperatorAuthError, OperatorKeyStore, OperatorRole};
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

/// Command gate configuration.
#[derive(Debug, Clone)]
pub struct CommandGateConfig {
    /// Max commands per operator per minute.
    pub max_commands_per_operator_per_minute: u32,

    /// Max scope toggles per minute (kill-switch activations/deactivations).
    pub max_scope_toggles_per_minute: u32,

    /// Idempotency window: how many correlation IDs to track.
    pub idempotency_window_size: usize,

    /// Minimum reason length (duplicates Phase 16 check at edge).
    pub min_reason_length: usize,
}

impl Default for CommandGateConfig {
    fn default() -> Self {
        Self {
            max_commands_per_operator_per_minute: 10,
            max_scope_toggles_per_minute: 5,
            idempotency_window_size: 1000,
            min_reason_length: 10,
        }
    }
}

/// Command gate state.
pub struct CommandGate {
    config: CommandGateConfig,
    key_store: Arc<RwLock<Option<OperatorKeyStore>>>,

    /// Per-operator command timestamps (for rate limiting).
    operator_commands: RwLock<HashMap<String, VecDeque<Instant>>>,

    /// Scope toggle timestamps (for rate limiting).
    scope_toggles: RwLock<VecDeque<Instant>>,

    /// Processed correlation IDs (for idempotency).
    processed_ids: RwLock<VecDeque<String>>,

    /// Cached responses for idempotent retries.
    cached_responses: RwLock<HashMap<String, OperatorResponse>>,
}

impl CommandGate {
    /// Create new command gate.
    pub fn new(config: CommandGateConfig, key_store: Option<OperatorKeyStore>) -> Self {
        Self {
            config,
            key_store: Arc::new(RwLock::new(key_store)),
            operator_commands: RwLock::new(HashMap::new()),
            scope_toggles: RwLock::new(VecDeque::new()),
            processed_ids: RwLock::new(VecDeque::new()),
            cached_responses: RwLock::new(HashMap::new()),
        }
    }

    /// Reload key store (for hot-reload support).
    pub fn reload_key_store(&self, key_store: OperatorKeyStore) {
        *self.key_store.write() = Some(key_store);
    }

    /// Validate request at transport edge.
    ///
    /// Returns Ok(role) if request passes all gates, Err otherwise.
    pub fn validate(&self, request: &AuthenticatedRequest, now_ns: i64) -> Result<OperatorRole, CommandGateError> {
        // 1. Check idempotency (correlation_id already processed?)
        if let Some(cached) = self.check_idempotency(&request.correlation_id) {
            return Err(CommandGateError::AlreadyProcessed { cached_response: cached });
        }

        // 2. Authenticate and get role
        let role = self.authenticate(request, now_ns)?;

        // 3. Check RBAC
        let action = request.action_class();
        if !role.can_perform(&action) {
            return Err(CommandGateError::InsufficientPermissions {
                required: action,
                actual: role,
            });
        }

        // 4. Validate reason length (early rejection)
        if let Some(reason) = request.request.reason() {
            if reason.len() < self.config.min_reason_length {
                return Err(CommandGateError::ReasonTooShort {
                    length: reason.len(),
                    min: self.config.min_reason_length,
                });
            }
        }

        // 5. Check rate limits
        let operator_id = request.request.operator_id();
        self.check_rate_limit(operator_id, &request.request)?;

        Ok(role)
    }

    /// Record successful processing (for idempotency).
    pub fn record_processed(&self, correlation_id: &str, response: OperatorResponse) {
        let mut processed = self.processed_ids.write();
        let mut cached = self.cached_responses.write();

        // Add to processed set
        processed.push_back(correlation_id.to_string());

        // Cache response
        cached.insert(correlation_id.to_string(), response);

        // Trim if over window size
        while processed.len() > self.config.idempotency_window_size {
            if let Some(old_id) = processed.pop_front() {
                cached.remove(&old_id);
            }
        }
    }

    /// Check if correlation ID was already processed.
    fn check_idempotency(&self, correlation_id: &str) -> Option<OperatorResponse> {
        self.cached_responses.read().get(correlation_id).cloned()
    }

    /// Authenticate request and return role.
    ///
    /// SECURITY INVARIANT: If no keystore is configured:
    /// - Mutating commands are REJECTED (Observer-only mode)
    /// - UNLESS env var QLX_ALLOW_UNAUTH_OPERATOR=1 is set (dev mode)
    /// - A loud warning is logged in either case
    ///
    /// This prevents "oops = full control" footguns in production.
    fn authenticate(&self, request: &AuthenticatedRequest, now_ns: i64) -> Result<OperatorRole, CommandGateError> {
        let key_store = self.key_store.read();

        match &*key_store {
            // No key store configured
            None => {
                // Check for explicit dev override
                let allow_unauth = std::env::var("QLX_ALLOW_UNAUTH_OPERATOR")
                    .map(|v| v == "1")
                    .unwrap_or(false);

                if allow_unauth {
                    // Dev mode: allow Admin but log LOUD warning
                    tracing::warn!(
                        target: "security",
                        correlation_id = %request.correlation_id,
                        "UNAUTH OPERATOR MODE: No keystore configured, QLX_ALLOW_UNAUTH_OPERATOR=1 set. \
                         Granting Admin role WITHOUT authentication. DO NOT USE IN PRODUCTION."
                    );
                    Ok(OperatorRole::Admin)
                } else {
                    // Production-safe default: Observer only (no mutations)
                    tracing::warn!(
                        target: "security",
                        correlation_id = %request.correlation_id,
                        "No keystore configured. Operator channel restricted to Observer role. \
                         Set QLX_ALLOW_UNAUTH_OPERATOR=1 for dev mode (NOT FOR PRODUCTION)."
                    );
                    Ok(OperatorRole::Observer)
                }
            }

            // Key store configured, verify auth
            Some(ks) => {
                let payload = request.signing_payload();
                ks.verify_auth(&request.auth, &payload, now_ns)
                    .map_err(CommandGateError::AuthError)
            }
        }
    }

    /// Check rate limits.
    fn check_rate_limit(&self, operator_id: &str, request: &crate::control_view::OperatorRequest) -> Result<(), CommandGateError> {
        let now = Instant::now();
        let one_minute_ago = now - std::time::Duration::from_secs(60);

        // Per-operator rate limit
        {
            let mut commands = self.operator_commands.write();
            let entry = commands.entry(operator_id.to_string()).or_insert_with(VecDeque::new);

            // Remove old entries
            while entry.front().map(|t| *t < one_minute_ago).unwrap_or(false) {
                entry.pop_front();
            }

            if entry.len() >= self.config.max_commands_per_operator_per_minute as usize {
                return Err(CommandGateError::RateLimitExceeded {
                    limit_type: "operator".to_string(),
                    limit: self.config.max_commands_per_operator_per_minute,
                });
            }

            entry.push_back(now);
        }

        // Scope toggle rate limit (for kill-switch operations)
        if matches!(
            request,
            crate::control_view::OperatorRequest::ActivateKillSwitch { .. }
                | crate::control_view::OperatorRequest::DeactivateKillSwitch { .. }
        ) {
            let mut toggles = self.scope_toggles.write();

            // Remove old entries
            while toggles.front().map(|t| *t < one_minute_ago).unwrap_or(false) {
                toggles.pop_front();
            }

            if toggles.len() >= self.config.max_scope_toggles_per_minute as usize {
                return Err(CommandGateError::RateLimitExceeded {
                    limit_type: "scope_toggle".to_string(),
                    limit: self.config.max_scope_toggles_per_minute,
                });
            }

            toggles.push_back(now);
        }

        Ok(())
    }
}

/// Command gate errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CommandGateError {
    #[error("Already processed (idempotent)")]
    AlreadyProcessed { cached_response: OperatorResponse },

    #[error("Authentication error: {0}")]
    AuthError(#[from] OperatorAuthError),

    #[error("Insufficient permissions: {required:?} required, have {actual:?}")]
    InsufficientPermissions {
        required: crate::operator_auth::OperatorAction,
        actual: OperatorRole,
    },

    #[error("Reason too short: {length} chars, minimum {min}")]
    ReasonTooShort { length: usize, min: usize },

    #[error("Rate limit exceeded: {limit_type} limit is {limit}/minute")]
    RateLimitExceeded { limit_type: String, limit: u32 },
}

impl CommandGateError {
    /// Convert to OperatorResponse for sending back to client.
    pub fn to_response(&self, correlation_id: Option<String>, ts_ns: i64) -> OperatorResponse {
        let outcome = match self {
            CommandGateError::AlreadyProcessed { cached_response } => {
                return cached_response.clone();
            }
            CommandGateError::AuthError(e) => OperatorOutcome::rejected("AUTH_ERROR", e.to_string()),
            CommandGateError::InsufficientPermissions { .. } => {
                OperatorOutcome::rejected("PERMISSION_DENIED", self.to_string())
            }
            CommandGateError::ReasonTooShort { .. } => {
                OperatorOutcome::rejected("REASON_TOO_SHORT", self.to_string())
            }
            CommandGateError::RateLimitExceeded { .. } => {
                OperatorOutcome::rejected("RATE_LIMITED", self.to_string())
            }
        };

        OperatorResponse::new(correlation_id, outcome, ts_ns)
    }
}
```

---

### 17B.3: Correlation + Idempotency

Already incorporated in 17B.2 via `CommandGate::check_idempotency()` and `record_processed()`.

**Modify:** `crates/quantlaxmi-runner-common/src/web_server.rs`

Update WebMessage to use `AuthenticatedRequest`:

```rust
/// WebSocket message types for client communication
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", content = "payload")]
pub enum WebMessage {
    // ... existing variants ...

    /// Operator request (inbound from client) - NOW USES AuthenticatedRequest.
    OperatorRequest {
        request: AuthenticatedRequest,
    },

    // ... rest unchanged ...
}
```

Update `handle_socket` to route through `CommandGate`:

```rust
async fn handle_socket(socket: WebSocket, state: Arc<ServerState>) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.tx.subscribe();

    info!("WebSocket client connected");

    // Send initial control view snapshot on connect
    if let Some(view) = state.control_view.read().await.clone() {
        let msg = WebMessage::ExecutionControlView { view };
        let json = serde_json::to_string(&msg).unwrap_or_default();
        let _ = sender.send(Message::Text(json)).await;
    }

    // ... send task unchanged ...

    // Handle incoming messages with command gate
    let tx_for_commands = state.tx.clone();
    let command_gate = state.command_gate.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(Message::Text(text))) = receiver.next().await {
            match serde_json::from_str::<WebMessage>(&text) {
                Ok(WebMessage::OperatorRequest { request }) => {
                    let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);

                    // Validate through command gate
                    match command_gate.validate(&request, now_ns) {
                        Ok(role) => {
                            info!(
                                "Operator request validated: correlation_id={}, operator={}, role={:?}",
                                request.correlation_id,
                                request.request.operator_id(),
                                role
                            );

                            // TODO: Route to session controller for execution
                            // For now, echo acknowledgment
                            let response = OperatorResponse::new(
                                Some(request.correlation_id.clone()),
                                OperatorOutcome::rejected("NOT_IMPLEMENTED", "Operator command routing not yet implemented"),
                                now_ns,
                            );

                            command_gate.record_processed(&request.correlation_id, response.clone());
                            let _ = tx_for_commands.send(WebMessage::OperatorResponse { response });
                        }
                        Err(e) => {
                            let response = e.to_response(Some(request.correlation_id.clone()), now_ns);
                            let _ = tx_for_commands.send(WebMessage::OperatorResponse { response });
                        }
                    }
                }
                // ... rest unchanged ...
            }
        }
    });

    // ... select unchanged ...
}
```

**Modify:** `ServerState` to include `CommandGate`:

```rust
pub struct ServerState {
    pub tx: broadcast::Sender<WebMessage>,
    pub control_view: Arc<RwLock<Option<ExecutionControlView>>>,
    pub metrics_snapshot: Arc<RwLock<Option<quantlaxmi_core::MetricsSnapshot>>>,
    pub circuit_breaker_status: Arc<RwLock<Option<CircuitBreakerStatus>>>,
    pub command_gate: Arc<CommandGate>,
}

impl ServerState {
    pub fn new(tx: broadcast::Sender<WebMessage>) -> Self {
        Self {
            tx,
            control_view: Arc::new(RwLock::new(None)),
            metrics_snapshot: Arc::new(RwLock::new(None)),
            circuit_breaker_status: Arc::new(RwLock::new(None)),
            command_gate: Arc::new(CommandGate::new(CommandGateConfig::default(), None)),
        }
    }

    pub fn with_key_store(tx: broadcast::Sender<WebMessage>, key_store: OperatorKeyStore) -> Self {
        Self {
            tx,
            control_view: Arc::new(RwLock::new(None)),
            metrics_snapshot: Arc::new(RwLock::new(None)),
            circuit_breaker_status: Arc::new(RwLock::new(None)),
            command_gate: Arc::new(CommandGate::new(CommandGateConfig::default(), Some(key_store))),
        }
    }
}
```

---

### 17B.4: Rate Limiting + Cooldown at Edge

Already incorporated in 17B.2 via `CommandGate::check_rate_limit()`.

---

### 17B.5: "Dangerous Action" Confirmations

**File:** `crates/quantlaxmi-runner-common/src/cli_confirm.rs`

```rust
//! CLI confirmation helpers for dangerous actions.
//!
//! Provides interactive prompts and --yes flag handling.

use crate::control_view::OperatorRequest;
use std::io::{self, Write};

/// Actions that require confirmation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DangerousAction {
    ForceHalt,
    EmergencyFlatten,
    CancelAllOrders,
    GlobalKillSwitch,
}

impl DangerousAction {
    /// Get warning message for this action.
    pub fn warning_message(&self) -> &'static str {
        match self {
            DangerousAction::ForceHalt =>
                "FORCE HALT will immediately stop all trading and require manual recovery.",
            DangerousAction::EmergencyFlatten =>
                "EMERGENCY FLATTEN will close ALL positions at market price. This cannot be undone.",
            DangerousAction::CancelAllOrders =>
                "CANCEL ALL ORDERS will cancel every open order across all symbols.",
            DangerousAction::GlobalKillSwitch =>
                "GLOBAL KILL-SWITCH will halt ALL strategies and prevent any new orders.",
        }
    }

    /// Get confirmation prompt for this action.
    pub fn confirmation_prompt(&self) -> &'static str {
        match self {
            DangerousAction::ForceHalt => "Type 'HALT' to confirm: ",
            DangerousAction::EmergencyFlatten => "Type the session ID to confirm: ",
            DangerousAction::CancelAllOrders => "Type 'CANCEL' to confirm: ",
            DangerousAction::GlobalKillSwitch => "Type 'GLOBAL' to confirm: ",
        }
    }

    /// Expected confirmation value.
    pub fn expected_confirmation(&self, session_id: Option<&str>) -> String {
        match self {
            DangerousAction::ForceHalt => "HALT".to_string(),
            DangerousAction::EmergencyFlatten => session_id.unwrap_or("unknown").to_string(),
            DangerousAction::CancelAllOrders => "CANCEL".to_string(),
            DangerousAction::GlobalKillSwitch => "GLOBAL".to_string(),
        }
    }
}

/// Classify request as dangerous action (if applicable).
pub fn classify_dangerous(request: &OperatorRequest) -> Option<DangerousAction> {
    match request {
        OperatorRequest::ForceHalt { .. } => Some(DangerousAction::ForceHalt),
        OperatorRequest::EmergencyFlatten { .. } => Some(DangerousAction::EmergencyFlatten),
        OperatorRequest::CancelAllOrders { .. } => Some(DangerousAction::CancelAllOrders),
        OperatorRequest::ActivateKillSwitch { scope, .. } => {
            if matches!(scope, quantlaxmi_gates::KillSwitchScope::Global) {
                Some(DangerousAction::GlobalKillSwitch)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Check if --yes flag bypasses confirmation.
pub fn has_yes_flag(args: &[String]) -> bool {
    args.iter().any(|a| a == "--yes" || a == "-y")
}

/// Prompt for confirmation interactively.
///
/// Returns Ok(true) if confirmed, Ok(false) if declined, Err on I/O error.
pub fn prompt_confirmation(
    action: DangerousAction,
    session_id: Option<&str>,
) -> io::Result<bool> {
    // Print warning
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("WARNING: {}", action.warning_message());
    eprintln!("{}", "=".repeat(60));

    // Get expected value
    let expected = action.expected_confirmation(session_id);

    // Prompt
    eprint!("{}", action.confirmation_prompt());
    io::stderr().flush()?;

    // Read input
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    // Check
    if input == expected {
        eprintln!("Confirmed.");
        Ok(true)
    } else {
        eprintln!("Confirmation failed. Expected '{}', got '{}'", expected, input);
        Ok(false)
    }
}

/// Require confirmation for dangerous action.
///
/// If --yes flag is present, skip interactive prompt.
/// Otherwise, prompt interactively.
///
/// Returns Ok(()) if confirmed, Err if declined or failed.
pub fn require_confirmation(
    request: &OperatorRequest,
    args: &[String],
    session_id: Option<&str>,
) -> Result<(), ConfirmationError> {
    // Check if action requires confirmation
    let action = match classify_dangerous(request) {
        Some(a) => a,
        None => return Ok(()), // Not dangerous, no confirmation needed
    };

    // Check --yes flag
    if has_yes_flag(args) {
        eprintln!("WARNING: {} (--yes flag used, skipping confirmation)", action.warning_message());
        return Ok(());
    }

    // Interactive confirmation
    match prompt_confirmation(action, session_id)? {
        true => Ok(()),
        false => Err(ConfirmationError::Declined { action }),
    }
}

/// Confirmation errors.
#[derive(Debug, thiserror::Error)]
pub enum ConfirmationError {
    #[error("User declined confirmation for {action:?}")]
    Declined { action: DangerousAction },

    #[error("I/O error during confirmation: {0}")]
    IoError(#[from] io::Error),
}
```

---

### 17B.6: Operator Command WAL Audit Linkage

**Modify:** `crates/quantlaxmi-gates/src/execution_session.rs`

Add `correlation_id` to existing event types (required field):

```rust
/// Manual override event with WAL linkage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManualOverrideEvent {
    // ... existing fields ...

    /// Correlation ID linking to operator command (required).
    pub correlation_id: String,
}

/// Kill-switch event with WAL linkage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchEvent {
    // ... existing fields ...

    /// Correlation ID linking to operator command (required).
    pub correlation_id: String,
}

/// Emergency flatten request with WAL linkage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyFlattenRequest {
    // ... existing fields ...

    /// Correlation ID linking to operator command (required).
    pub correlation_id: String,
}
```

---

## Implementation Order

| Step | Task | Location | Depends On |
|------|------|----------|------------|
| 1 | Create `operator_auth.rs` with OperatorAuth, RBAC types | runner-common | - |
| 2 | Create `command_gate.rs` with CommandGate | runner-common | 1 |
| 3 | Create `cli_confirm.rs` with confirmation helpers | runner-common | - |
| 4 | Add `AuthenticatedRequest` wrapper to control_view.rs | runner-common | 1 |
| 5 | Add `correlation_id` to ManualOverrideEvent | quantlaxmi-gates | - |
| 6 | Add `correlation_id` to KillSwitchEvent | quantlaxmi-gates | - |
| 7 | Add `correlation_id` to EmergencyFlattenRequest | quantlaxmi-gates | - |
| 8 | Update WebMessage::OperatorRequest to use AuthenticatedRequest | runner-common | 4 |
| 9 | Add CommandGate to ServerState | runner-common | 2 |
| 10 | Update handle_socket to route through CommandGate | runner-common | 9 |
| 11 | Add module exports to lib.rs | runner-common | 1, 2, 3 |
| 12 | Add hmac, sha2, hex dependencies to Cargo.toml | runner-common | - |
| 13 | Write unit tests for OperatorAuth | runner-common | 1 |
| 14 | Write unit tests for CommandGate | runner-common | 2 |
| 15 | Write unit tests for cli_confirm | runner-common | 3 |
| 16 | Write integration test for WS auth flow | runner-common | 10 |
| 17 | Run clippy + all tests | - | 13-16 |

---

## File Summary

**New files (3):**
```
crates/quantlaxmi-runner-common/src/operator_auth.rs   # Auth + RBAC
crates/quantlaxmi-runner-common/src/command_gate.rs    # Rate limit + idempotency
crates/quantlaxmi-runner-common/src/cli_confirm.rs     # CLI confirmations
```

**Modified files (5):**
```
crates/quantlaxmi-runner-common/src/control_view.rs    # AuthenticatedRequest
crates/quantlaxmi-runner-common/src/web_server.rs      # CommandGate integration
crates/quantlaxmi-runner-common/src/lib.rs             # Module exports
crates/quantlaxmi-runner-common/Cargo.toml             # hmac, sha2, hex deps
crates/quantlaxmi-gates/src/execution_session.rs       # correlation_id fields
```

---

## Verification

1. **HMAC determinism test**: Same inputs → same signature
2. **Signature verification test**: Valid/invalid signatures correctly detected
3. **RBAC test**: Observer can't send Control commands, Operator can't send Dangerous
4. **Idempotency test**: Same correlation_id → cached response returned
5. **Rate limit test**: Exceeding limit → RateLimitExceeded error
6. **Reason length test**: Short reason → ReasonTooShort error
7. **CLI confirmation test**: --yes bypasses, interactive requires correct input
8. **Required fields test**: Missing auth, correlation_id, or session_id fails deserialization
9. **No-keystore security test**: Without keystore and without env override → Observer only
10. **Canonical bytes determinism test**: Same OperatorRequest → same canonical_bytes()
11. **Signing payload determinism test**: Same AuthenticatedRequest → same signing_payload()
12. **Session binding test**: Signature valid for session A is invalid for session B
13. **Secret redaction test**: `format!("{:?}", key_entry)` contains "[REDACTED]", not actual secret
14. **Signature redaction test**: `format!("{:?}", auth)` contains "[REDACTED]", not actual signature

---

## Tests (Minimum 15)

```rust
// operator_auth.rs
#[test] fn test_hmac_signature_deterministic()
#[test] fn test_signature_verification_valid()
#[test] fn test_signature_verification_invalid()
#[test] fn test_auth_expired()
#[test] fn test_rbac_observer_cannot_control()
#[test] fn test_rbac_operator_cannot_dangerous()
#[test] fn test_rbac_admin_can_dangerous()
#[test] fn test_secret_not_in_debug_output()  // Verify [REDACTED]
#[test] fn test_signature_not_in_debug_output()  // Verify [REDACTED]

// command_gate.rs
#[test] fn test_idempotency_returns_cached()
#[test] fn test_rate_limit_exceeded()
#[test] fn test_reason_too_short()
#[test] fn test_no_keystore_returns_observer_by_default()  // SECURITY: No keystore = Observer
#[test] fn test_no_keystore_with_env_override_returns_admin()  // QLX_ALLOW_UNAUTH_OPERATOR=1

// control_view.rs
#[test] fn test_signing_payload_deterministic()  // Same inputs → same bytes
#[test] fn test_signing_payload_includes_session_id()  // Replay protection
#[test] fn test_operator_request_canonical_bytes_deterministic()

// cli_confirm.rs
#[test] fn test_has_yes_flag()
#[test] fn test_classify_dangerous()
```

---

## Success Criteria

- [ ] OperatorAuth with HMAC-SHA256 over canonical bytes (NOT JSON)
- [ ] OperatorKeyStore loaded from file or env var
- [ ] Three RBAC roles: Observer, Operator, Admin
- [ ] CommandGate validates auth, RBAC, rate limits, idempotency
- [ ] AuthenticatedRequest wrapper with required correlation_id, session_id, and auth
- [ ] correlation_id (required) embedded in ManualOverrideEvent, KillSwitchEvent, EmergencyFlattenRequest
- [ ] CLI confirmation for ForceHalt, EmergencyFlatten, CancelAllOrders, Global KillSwitch
- [ ] --yes flag bypasses interactive confirmation
- [ ] EmergencyFlatten requires typing session_id
- [ ] All tests pass, clippy clean
- [ ] No backward compatibility - all fields required, no serde(default)

**Security hardening (from code review):**
- [ ] No keystore configured → Observer-only (no mutations), unless QLX_ALLOW_UNAUTH_OPERATOR=1
- [ ] QLX_ALLOW_UNAUTH_OPERATOR=1 logs LOUD warning (security target)
- [ ] `OperatorRequest::canonical_bytes()` uses deterministic binary encoding (not JSON)
- [ ] `AuthenticatedRequest::signing_payload()` uses deterministic binary encoding (not JSON)
- [ ] session_id included in signed payload (prevents cross-session replay)
- [ ] OperatorKeyEntry.secret_hex redacted in Debug output
- [ ] OperatorAuth.signature redacted in Debug output
- [ ] grep -r "secret_hex" src/ must never appear in info!/warn!/error! macros
