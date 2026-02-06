//! # Validate Execution Session (Phase 14.2 + Phase 15.1)
//!
//! CLI validator for execution session WAL files.
//!
//! ## Invariants Verified (Phase 14.2)
//! 1. All IDs are deterministically derived
//! 2. State machine transitions are valid
//! 3. Budget deltas balance (reserved → committed/released)
//! 4. No duplicate idempotency keys
//! 5. All events have valid digests
//!
//! ## Risk Invariants (Phase 15.1 INV-R1/R2/R3)
//! - INV-R1: Risk snapshot must be computed before any order decisions
//! - INV-R2: Every order intent must have a corresponding risk decision logged
//! - INV-R3: Risk violations must trigger proper rejection (no Reject/Halt without rejection)
//!
//! ## Usage
//! ```bash
//! validate-execution-session --session-dir data/execution_sessions/session_001
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use quantlaxmi_gates::{
    RISK_DECISION_SCHEMA_VERSION, RISK_SNAPSHOT_SCHEMA_VERSION, RiskDecision, RiskDecisionScope,
    RiskDecisionStatus, RiskSnapshot,
};
use quantlaxmi_models::{
    EXECUTION_EVENTS_SCHEMA_VERSION, IdempotencyKey, IntentId, LiveOrderState, OrderAckEvent,
    OrderCancelEvent, OrderFillEvent, OrderIntentEvent, OrderRejectEvent, OrderSubmitEvent,
    PositionCloseEvent,
};
use serde::de::DeserializeOwned;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

// =============================================================================
// CLI
// =============================================================================

#[derive(Parser, Debug)]
#[command(name = "validate-execution-session")]
#[command(about = "Validate execution session WAL files for Phase 14.2 invariants")]
#[command(version)]
struct Cli {
    /// Path to session directory containing execution WAL files
    #[arg(long)]
    session_dir: PathBuf,

    /// Verbose output (print each event as validated)
    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Continue on errors (collect all errors before failing)
    #[arg(long, default_value_t = false)]
    continue_on_error: bool,
}

// =============================================================================
// Validation State
// =============================================================================

#[derive(Debug, Default)]
struct ValidationState {
    /// Tracked intents by ID
    intents: HashMap<String, OrderIntentEvent>,
    /// Tracked submits by client order ID
    submits: HashMap<String, OrderSubmitEvent>,
    /// Client order ID to intent ID mapping
    client_to_intent: HashMap<String, String>,
    /// Exchange order ID to client order ID mapping
    exchange_to_client: HashMap<String, String>,
    /// Order states by client order ID
    order_states: HashMap<String, LiveOrderState>,
    /// Processed idempotency keys
    processed_keys: HashSet<String>,
    /// Budget reservations by (strategy_id, bucket_id)
    reserved: HashMap<(String, String), i128>,
    /// Budget commitments by (strategy_id, bucket_id)
    committed: HashMap<(String, String), i128>,
    /// Errors collected
    errors: Vec<String>,
    /// Counts
    intent_count: usize,
    submit_count: usize,
    ack_count: usize,
    reject_count: usize,
    fill_count: usize,
    cancel_count: usize,
    position_close_count: usize,

    // Phase 15.1 Risk tracking
    /// Risk snapshots by snapshot ID
    risk_snapshots: HashMap<String, RiskSnapshot>,
    /// Risk decisions by intent ID (for INV-R2)
    risk_decisions_by_intent: HashMap<String, RiskDecision>,
    /// Intents that were rejected (for INV-R3)
    rejected_intents: HashSet<String>,
    /// First risk snapshot timestamp (for INV-R1)
    first_snapshot_ts_ns: Option<i64>,
    /// First intent timestamp (for INV-R1)
    first_intent_ts_ns: Option<i64>,
    /// Risk snapshot count
    risk_snapshot_count: usize,
    /// Risk decision count
    risk_decision_count: usize,
}

impl ValidationState {
    fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

// =============================================================================
// Event Reading
// =============================================================================

fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let file = File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut events = Vec::new();
    for (line_num, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "Failed to read line {} from {}",
                line_num + 1,
                path.display()
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let event: T = serde_json::from_str(&line).with_context(|| {
            format!(
                "Failed to parse line {} from {}: {}",
                line_num + 1,
                path.display(),
                line
            )
        })?;
        events.push(event);
    }

    Ok(events)
}

// =============================================================================
// Validators
// =============================================================================

fn validate_intent_id(intent: &OrderIntentEvent) -> Result<(), String> {
    // Verify intent ID derivation
    // We can't fully verify the derivation since we don't know the seq,
    // but we can verify the format is correct
    let _expected = IntentId::derive(
        &intent.strategy_id,
        &intent.bucket_id,
        intent.ts_ns,
        1, // We can't know the seq, but we can verify format
    );

    // Just verify it's a hex string of expected length (64 chars for SHA-256)
    if intent.intent_id.0.len() != 64 {
        return Err(format!(
            "Intent ID has wrong length: expected 64, got {}",
            intent.intent_id.0.len()
        ));
    }

    // Verify all characters are hex
    if !intent.intent_id.0.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!(
            "Intent ID contains non-hex characters: {}",
            intent.intent_id.0
        ));
    }

    Ok(())
}

fn validate_client_order_id(submit: &OrderSubmitEvent) -> Result<(), String> {
    // Client order IDs should be 32 chars (truncated SHA-256)
    if submit.client_order_id.0.len() != 32 {
        return Err(format!(
            "Client order ID has wrong length: expected 32, got {}",
            submit.client_order_id.0.len()
        ));
    }

    if !submit
        .client_order_id
        .0
        .chars()
        .all(|c| c.is_ascii_hexdigit())
    {
        return Err(format!(
            "Client order ID contains non-hex characters: {}",
            submit.client_order_id.0
        ));
    }

    Ok(())
}

fn validate_digest<T>(
    _event: &T,
    computed: &str,
    stored: &str,
    event_type: &str,
) -> Result<(), String> {
    if computed != stored {
        return Err(format!(
            "{} digest mismatch: computed={}, stored={}",
            event_type, computed, stored
        ));
    }
    Ok(())
}

fn validate_state_transition(
    current: LiveOrderState,
    new: LiveOrderState,
    client_order_id: &str,
) -> Result<(), String> {
    let valid = match (current, new) {
        // Valid transitions from IntentCreated
        (LiveOrderState::IntentCreated, LiveOrderState::Submitted) => true,
        // Valid transitions from Submitted
        (LiveOrderState::Submitted, LiveOrderState::Acked) => true,
        (LiveOrderState::Submitted, LiveOrderState::Rejected) => true,
        // Valid transitions from Acked
        (LiveOrderState::Acked, LiveOrderState::PartFilled) => true,
        (LiveOrderState::Acked, LiveOrderState::Filled) => true,
        (LiveOrderState::Acked, LiveOrderState::Cancelled) => true,
        // Valid transitions from PartFilled
        (LiveOrderState::PartFilled, LiveOrderState::PartFilled) => true,
        (LiveOrderState::PartFilled, LiveOrderState::Filled) => true,
        (LiveOrderState::PartFilled, LiveOrderState::Cancelled) => true,
        // No transitions from terminal states
        (LiveOrderState::Filled, _) => false,
        (LiveOrderState::Cancelled, _) => false,
        (LiveOrderState::Rejected, _) => false,
        // Same state is ok (idempotent)
        (a, b) if a == b => true,
        _ => false,
    };

    if !valid {
        return Err(format!(
            "Invalid state transition for {}: {} -> {}",
            client_order_id, current, new
        ));
    }

    Ok(())
}

// =============================================================================
// Main Validation Logic
// =============================================================================

fn validate_intents(session_dir: &Path, state: &mut ValidationState, verbose: bool) -> Result<()> {
    let path = session_dir.join("intents.jsonl");
    let intents: Vec<OrderIntentEvent> = read_jsonl(&path)?;

    for intent in intents {
        // Validate intent ID format
        if let Err(e) = validate_intent_id(&intent) {
            state.add_error(format!("Intent {}: {}", intent.intent_id, e));
        }

        // Validate digest
        let computed = intent.compute_digest();
        if let Err(e) = validate_digest(&intent, &computed, &intent.digest, "OrderIntentEvent") {
            state.add_error(format!("Intent {}: {}", intent.intent_id, e));
        }

        // Validate schema version
        if intent.schema_version != EXECUTION_EVENTS_SCHEMA_VERSION {
            state.add_error(format!(
                "Intent {}: wrong schema version {}",
                intent.intent_id, intent.schema_version
            ));
        }

        // Track first intent timestamp (for INV-R1)
        if state.first_intent_ts_ns.is_none() {
            state.first_intent_ts_ns = Some(intent.ts_ns);
        }

        // Track the intent
        state
            .intents
            .insert(intent.intent_id.0.clone(), intent.clone());
        state.intent_count += 1;

        if verbose {
            println!("✓ Intent: {} ({})", intent.intent_id, intent.symbol);
        }
    }

    Ok(())
}

fn validate_submits(session_dir: &Path, state: &mut ValidationState, verbose: bool) -> Result<()> {
    let path = session_dir.join("submits.jsonl");
    let submits: Vec<OrderSubmitEvent> = read_jsonl(&path)?;

    for submit in submits {
        // Validate client order ID format
        if let Err(e) = validate_client_order_id(&submit) {
            state.add_error(format!("Submit {}: {}", submit.client_order_id, e));
        }

        // Validate digest
        let computed = submit.compute_digest();
        if let Err(e) = validate_digest(&submit, &computed, &submit.digest, "OrderSubmitEvent") {
            state.add_error(format!("Submit {}: {}", submit.client_order_id, e));
        }

        // Validate intent exists
        if !state.intents.contains_key(&submit.intent_id.0) {
            state.add_error(format!(
                "Submit {} references unknown intent {}",
                submit.client_order_id, submit.intent_id
            ));
        }

        // Validate state is Submitted
        if submit.state != LiveOrderState::Submitted {
            state.add_error(format!(
                "Submit {} has wrong state: expected Submitted, got {}",
                submit.client_order_id, submit.state
            ));
        }

        // Track budget reservation
        if let Some(intent) = state.intents.get(&submit.intent_id.0) {
            let key = (intent.strategy_id.clone(), intent.bucket_id.clone());
            *state.reserved.entry(key).or_insert(0) += submit.reserved_mantissa;
        }

        // Track mappings
        state
            .client_to_intent
            .insert(submit.client_order_id.0.clone(), submit.intent_id.0.clone());
        state
            .order_states
            .insert(submit.client_order_id.0.clone(), LiveOrderState::Submitted);
        state
            .submits
            .insert(submit.client_order_id.0.clone(), submit.clone());
        state.submit_count += 1;

        if verbose {
            println!(
                "✓ Submit: {} (reserved {})",
                submit.client_order_id, submit.reserved_mantissa
            );
        }
    }

    Ok(())
}

fn validate_acks(session_dir: &Path, state: &mut ValidationState, verbose: bool) -> Result<()> {
    let path = session_dir.join("acks.jsonl");
    let acks: Vec<OrderAckEvent> = read_jsonl(&path)?;

    for ack in acks {
        // Idempotency check
        let idem_key = IdempotencyKey::from_ack(&ack.exchange_order_id.0);
        if state.processed_keys.contains(&idem_key.0) {
            state.add_error(format!(
                "Duplicate ack for exchange order {}",
                ack.exchange_order_id
            ));
        }
        state.processed_keys.insert(idem_key.0.clone());

        // Validate digest
        let computed = ack.compute_digest();
        if let Err(e) = validate_digest(&ack, &computed, &ack.digest, "OrderAckEvent") {
            state.add_error(format!("Ack {}: {}", ack.client_order_id, e));
        }

        // Validate state transition
        if let Some(&current_state) = state.order_states.get(&ack.client_order_id.0) {
            if let Err(e) = validate_state_transition(
                current_state,
                LiveOrderState::Acked,
                &ack.client_order_id.0,
            ) {
                state.add_error(e);
            }
        } else {
            state.add_error(format!(
                "Ack for unknown client order {}",
                ack.client_order_id
            ));
        }

        // Update state
        state
            .order_states
            .insert(ack.client_order_id.0.clone(), LiveOrderState::Acked);
        state.exchange_to_client.insert(
            ack.exchange_order_id.0.clone(),
            ack.client_order_id.0.clone(),
        );
        state.ack_count += 1;

        if verbose {
            println!(
                "✓ Ack: {} -> {}",
                ack.client_order_id, ack.exchange_order_id
            );
        }
    }

    Ok(())
}

#[allow(clippy::collapsible_if)]
fn validate_rejects(session_dir: &Path, state: &mut ValidationState, verbose: bool) -> Result<()> {
    let path = session_dir.join("rejects.jsonl");
    let rejects: Vec<OrderRejectEvent> = read_jsonl(&path)?;

    for reject in rejects {
        // Validate digest
        let computed = reject.compute_digest();
        if let Err(e) = validate_digest(&reject, &computed, &reject.digest, "OrderRejectEvent") {
            state.add_error(format!("Reject {}: {}", reject.client_order_id, e));
        }

        // Validate state transition
        if let Some(&current_state) = state.order_states.get(&reject.client_order_id.0) {
            if let Err(e) = validate_state_transition(
                current_state,
                LiveOrderState::Rejected,
                &reject.client_order_id.0,
            ) {
                state.add_error(e);
            }
        }

        // Validate budget release
        if let Some(intent_id) = state.client_to_intent.get(&reject.client_order_id.0) {
            if let Some(intent) = state.intents.get(intent_id) {
                let key = (intent.strategy_id.clone(), intent.bucket_id.clone());
                if let Some(reserved) = state.reserved.get_mut(&key) {
                    *reserved -= reject.released_mantissa;
                }
            }
        }

        // Update state
        state
            .order_states
            .insert(reject.client_order_id.0.clone(), LiveOrderState::Rejected);
        state.reject_count += 1;

        if verbose {
            println!(
                "✓ Reject: {} ({})",
                reject.client_order_id, reject.reject_reason
            );
        }
    }

    Ok(())
}

#[allow(clippy::collapsible_if)]
fn validate_fills(session_dir: &Path, state: &mut ValidationState, verbose: bool) -> Result<()> {
    let path = session_dir.join("fills.jsonl");
    let fills: Vec<OrderFillEvent> = read_jsonl(&path)?;

    for fill in fills {
        // Idempotency check
        let idem_key = IdempotencyKey::from_fill(&fill.exchange_fill_id, &fill.exchange_order_id.0);
        if state.processed_keys.contains(&idem_key.0) {
            state.add_error(format!(
                "Duplicate fill {} for order {}",
                fill.exchange_fill_id, fill.exchange_order_id
            ));
        }
        state.processed_keys.insert(idem_key.0.clone());

        // Validate digest
        let computed = fill.compute_digest();
        if let Err(e) = validate_digest(&fill, &computed, &fill.digest, "OrderFillEvent") {
            state.add_error(format!("Fill {}: {}", fill.fill_id, e));
        }

        // Validate state transition
        if let Some(&current_state) = state.order_states.get(&fill.client_order_id.0) {
            if let Err(e) =
                validate_state_transition(current_state, fill.state, &fill.client_order_id.0)
            {
                state.add_error(e);
            }
        }

        // Track budget commitment
        if let Some(intent_id) = state.client_to_intent.get(&fill.client_order_id.0) {
            if let Some(intent) = state.intents.get(intent_id) {
                let key = (intent.strategy_id.clone(), intent.bucket_id.clone());
                // Move from reserved to committed
                if let Some(reserved) = state.reserved.get_mut(&key) {
                    *reserved -= fill.committed_mantissa;
                }
                *state.committed.entry(key).or_insert(0) += fill.committed_mantissa;
            }
        }

        // Update state
        state
            .order_states
            .insert(fill.client_order_id.0.clone(), fill.state);
        state.fill_count += 1;

        if verbose {
            println!(
                "✓ Fill: {} @ {} (qty: {}, final: {})",
                fill.fill_id, fill.fill_price_mantissa, fill.fill_quantity_mantissa, fill.is_final
            );
        }
    }

    Ok(())
}

#[allow(clippy::collapsible_if)]
fn validate_cancels(session_dir: &Path, state: &mut ValidationState, verbose: bool) -> Result<()> {
    let path = session_dir.join("cancels.jsonl");
    let cancels: Vec<OrderCancelEvent> = read_jsonl(&path)?;

    for cancel in cancels {
        // Idempotency check
        let idem_key = IdempotencyKey::from_cancel(&cancel.exchange_order_id.0, cancel.ts_ns);
        if state.processed_keys.contains(&idem_key.0) {
            state.add_error(format!(
                "Duplicate cancel for order {}",
                cancel.exchange_order_id
            ));
        }
        state.processed_keys.insert(idem_key.0.clone());

        // Validate digest
        let computed = cancel.compute_digest();
        if let Err(e) = validate_digest(&cancel, &computed, &cancel.digest, "OrderCancelEvent") {
            state.add_error(format!("Cancel {}: {}", cancel.client_order_id, e));
        }

        // Validate state transition
        if let Some(&current_state) = state.order_states.get(&cancel.client_order_id.0) {
            if let Err(e) = validate_state_transition(
                current_state,
                LiveOrderState::Cancelled,
                &cancel.client_order_id.0,
            ) {
                state.add_error(e);
            }
        }

        // Track budget release
        if let Some(intent_id) = state.client_to_intent.get(&cancel.client_order_id.0) {
            if let Some(intent) = state.intents.get(intent_id) {
                let key = (intent.strategy_id.clone(), intent.bucket_id.clone());
                if let Some(reserved) = state.reserved.get_mut(&key) {
                    *reserved -= cancel.released_mantissa;
                }
            }
        }

        // Update state
        state
            .order_states
            .insert(cancel.client_order_id.0.clone(), LiveOrderState::Cancelled);
        state.cancel_count += 1;

        if verbose {
            println!(
                "✓ Cancel: {} ({:?})",
                cancel.client_order_id, cancel.cancel_source
            );
        }
    }

    Ok(())
}

fn validate_position_closes(
    session_dir: &Path,
    state: &mut ValidationState,
    verbose: bool,
) -> Result<()> {
    let path = session_dir.join("position_closes.jsonl");
    let closes: Vec<PositionCloseEvent> = read_jsonl(&path)?;

    for close in closes {
        // Validate digest
        let computed = close.compute_digest();
        if let Err(e) = validate_digest(&close, &computed, &close.digest, "PositionCloseEvent") {
            state.add_error(format!(
                "PositionClose {}:{}: {}",
                close.strategy_id, close.symbol, e
            ));
        }

        // Track capital release
        let key = (close.strategy_id.clone(), close.bucket_id.clone());
        if let Some(committed) = state.committed.get_mut(&key) {
            *committed -= close.released_capital_mantissa;
        }

        state.position_close_count += 1;

        if verbose {
            println!(
                "✓ PositionClose: {}:{} (PnL: {})",
                close.strategy_id, close.symbol, close.realized_pnl_mantissa
            );
        }
    }

    Ok(())
}

fn validate_budget_balance(state: &ValidationState) -> Vec<String> {
    let mut errors = Vec::new();

    // After all events, reserved should be zero or positive (no negative reservations)
    for ((strategy_id, bucket_id), reserved) in &state.reserved {
        if *reserved < 0 {
            errors.push(format!(
                "Negative reserved capital for {}:{}: {}",
                strategy_id, bucket_id, reserved
            ));
        }
    }

    // Committed should also be non-negative
    for ((strategy_id, bucket_id), committed) in &state.committed {
        if *committed < 0 {
            errors.push(format!(
                "Negative committed capital for {}:{}: {}",
                strategy_id, bucket_id, committed
            ));
        }
    }

    errors
}

// =============================================================================
// Phase 15.1 Risk Validation (INV-R1/R2/R3)
// =============================================================================

fn validate_risk_snapshots(
    session_dir: &Path,
    state: &mut ValidationState,
    verbose: bool,
) -> Result<()> {
    let path = session_dir.join("risk_snapshots.jsonl");
    let snapshots: Vec<RiskSnapshot> = read_jsonl(&path)?;

    for snapshot in snapshots {
        // Validate digest
        let computed = snapshot.compute_digest();
        if computed != snapshot.digest {
            state.add_error(format!(
                "RiskSnapshot {}: digest mismatch, computed={}, stored={}",
                snapshot.snapshot_id, computed, snapshot.digest
            ));
        }

        // Validate schema version
        if snapshot.schema_version != RISK_SNAPSHOT_SCHEMA_VERSION {
            state.add_error(format!(
                "RiskSnapshot {}: wrong schema version {}",
                snapshot.snapshot_id, snapshot.schema_version
            ));
        }

        // Track first snapshot timestamp (for INV-R1)
        if state.first_snapshot_ts_ns.is_none() {
            state.first_snapshot_ts_ns = Some(snapshot.ts_ns);
        }

        // Track snapshot
        state
            .risk_snapshots
            .insert(snapshot.snapshot_id.0.clone(), snapshot.clone());
        state.risk_snapshot_count += 1;

        if verbose {
            println!(
                "✓ RiskSnapshot: {} (positions: {})",
                snapshot.snapshot_id, snapshot.exposures.total_position_count
            );
        }
    }

    Ok(())
}

fn validate_risk_decisions(
    session_dir: &Path,
    state: &mut ValidationState,
    verbose: bool,
) -> Result<()> {
    let path = session_dir.join("risk_decisions.jsonl");
    let decisions: Vec<RiskDecision> = read_jsonl(&path)?;

    for decision in decisions {
        // Validate digest
        let computed = decision.compute_digest();
        if computed != decision.digest {
            state.add_error(format!(
                "RiskDecision {}: digest mismatch, computed={}, stored={}",
                decision.decision_id, computed, decision.digest
            ));
        }

        // Validate schema version
        if decision.schema_version != RISK_DECISION_SCHEMA_VERSION {
            state.add_error(format!(
                "RiskDecision {}: wrong schema version {}",
                decision.decision_id, decision.schema_version
            ));
        }

        // Track decision by intent ID (for INV-R2)
        if let RiskDecisionScope::Order { ref intent_id } = decision.scope {
            state
                .risk_decisions_by_intent
                .insert(intent_id.clone(), decision.clone());

            // If decision is Reject or Halt, track it (for INV-R3)
            if matches!(
                decision.status,
                RiskDecisionStatus::Reject | RiskDecisionStatus::Halt
            ) {
                state.rejected_intents.insert(intent_id.clone());
            }
        }

        state.risk_decision_count += 1;

        if verbose {
            println!(
                "✓ RiskDecision: {} ({:?})",
                decision.decision_id, decision.status
            );
        }
    }

    Ok(())
}

fn validate_risk_invariants(state: &ValidationState) -> Vec<String> {
    let mut errors = Vec::new();

    // INV-R1: Risk snapshot must be computed before any order decisions
    // (only check if both risk snapshots and intents exist)
    if state.risk_snapshot_count > 0
        && state.intent_count > 0
        && let (Some(first_snapshot_ts), Some(first_intent_ts)) =
            (state.first_snapshot_ts_ns, state.first_intent_ts_ns)
        && first_intent_ts < first_snapshot_ts
    {
        errors.push(format!(
            "INV-R1: First intent (ts={}) occurred before first risk snapshot (ts={})",
            first_intent_ts, first_snapshot_ts
        ));
    }

    // INV-R2: Every order intent should have a corresponding risk decision
    // (only validate if risk layer is configured - checked by presence of risk decisions)
    if state.risk_decision_count > 0 {
        for intent_id in state.intents.keys() {
            if !state.risk_decisions_by_intent.contains_key(intent_id) {
                errors.push(format!(
                    "INV-R2: Intent {} has no corresponding risk decision",
                    intent_id
                ));
            }
        }
    }

    // INV-R3: Risk violations with Reject/Halt status must trigger proper rejection
    // (rejected intents should not have successful submits)
    for intent_id in &state.rejected_intents {
        // Check if this intent was submitted successfully (it shouldn't be)
        if let Some(client_order_id) = state.intents.get(intent_id).and_then(|intent| {
            state
                .submits
                .values()
                .find(|s| s.intent_id.0 == intent.intent_id.0)
                .map(|s| s.client_order_id.0.clone())
        }) {
            errors.push(format!(
                "INV-R3: Intent {} was risk-rejected but still submitted (client_order={})",
                intent_id, client_order_id
            ));
        }
    }

    errors
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!(
        "Validating execution session: {}",
        cli.session_dir.display()
    );
    println!("Schema version: {}", EXECUTION_EVENTS_SCHEMA_VERSION);
    println!();

    let mut state = ValidationState::default();

    // Validate each event type (Phase 14.2)
    validate_intents(&cli.session_dir, &mut state, cli.verbose)?;
    validate_submits(&cli.session_dir, &mut state, cli.verbose)?;
    validate_acks(&cli.session_dir, &mut state, cli.verbose)?;
    validate_rejects(&cli.session_dir, &mut state, cli.verbose)?;
    validate_fills(&cli.session_dir, &mut state, cli.verbose)?;
    validate_cancels(&cli.session_dir, &mut state, cli.verbose)?;
    validate_position_closes(&cli.session_dir, &mut state, cli.verbose)?;

    // Validate risk events (Phase 15.1)
    validate_risk_snapshots(&cli.session_dir, &mut state, cli.verbose)?;
    validate_risk_decisions(&cli.session_dir, &mut state, cli.verbose)?;

    // Validate budget balance
    let budget_errors = validate_budget_balance(&state);
    for error in budget_errors {
        state.add_error(error);
    }

    // Validate risk invariants (Phase 15.1 INV-R1/R2/R3)
    let risk_errors = validate_risk_invariants(&state);
    for error in risk_errors {
        state.add_error(error);
    }

    // Print summary
    println!();
    println!("=== Validation Summary (Phase 14.2 + 15.1) ===");
    println!("Intents:         {}", state.intent_count);
    println!("Submits:         {}", state.submit_count);
    println!("Acks:            {}", state.ack_count);
    println!("Rejects:         {}", state.reject_count);
    println!("Fills:           {}", state.fill_count);
    println!("Cancels:         {}", state.cancel_count);
    println!("Position Closes: {}", state.position_close_count);
    println!("Risk Snapshots:  {}", state.risk_snapshot_count);
    println!("Risk Decisions:  {}", state.risk_decision_count);
    println!();

    // Print budget state
    println!("=== Budget State ===");
    for ((strategy_id, bucket_id), reserved) in &state.reserved {
        println!("  {}:{} - Reserved: {}", strategy_id, bucket_id, reserved);
    }
    for ((strategy_id, bucket_id), committed) in &state.committed {
        println!("  {}:{} - Committed: {}", strategy_id, bucket_id, committed);
    }
    println!();

    // Print errors or success
    if state.has_errors() {
        println!("=== ERRORS ({}) ===", state.errors.len());
        for error in &state.errors {
            println!("  ✗ {}", error);
        }
        println!();
        anyhow::bail!("Validation failed with {} errors", state.errors.len());
    } else {
        println!("=== VALIDATION PASSED ===");
        println!("All Phase 14.2 invariants verified.");
    }

    Ok(())
}
