//! # Binance Perp Live Execution (Phase 14.2)
//!
//! Live execution engine for Binance Perpetual Futures.
//!
//! ## Hard Laws
//! 1. No order leaves without budget check + reservation delta
//! 2. Every exchange event reconciles budget ledger with WAL-bound artifacts
//! 3. Rollback on failure is deterministic
//! 4. Idempotent processing of exchange events
//!
//! ## State Machine
//! ```text
//! IntentCreated → BudgetReserved → Submitted → Acked → PartFilled/Filled/Cancelled/Rejected
//! ```
//!
//! ## NOT in Scope (Phase 14.2)
//! - Actual venue connections (stubbed)
//! - WebSocket streaming (stubbed)
//! - Real order submission (stubbed)
//! - PnL accounting (Phase 14.3+)

use quantlaxmi_models::{
    CancelSource, ClientOrderId, EXECUTION_EVENTS_SCHEMA_VERSION, ExchangeOrderId,
    ExecutionOrderType, ExecutionSide, FillId, IdempotencyKey, IntentId, LiveOrderState,
    OrderAckEvent, OrderCancelEvent, OrderFillEvent, OrderIntentEvent, OrderRejectEvent,
    OrderSubmitEvent, PositionCloseEvent,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

// =============================================================================
// Schema Version
// =============================================================================

pub const LIVE_EXECUTION_SCHEMA_VERSION: &str = "live_execution_v1.0";

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("Order not found: {0}")]
    OrderNotFound(String),

    #[error("Invalid state transition: {from} -> {to}")]
    InvalidStateTransition { from: String, to: String },

    #[error("Budget check failed: {0}")]
    BudgetCheckFailed(String),

    #[error("Budget reservation failed: {0}")]
    BudgetReservationFailed(String),

    #[error("Idempotency violation: event already processed")]
    IdempotencyViolation,

    #[error("Venue error: {0}")]
    VenueError(String),

    #[error("WAL write error: {0}")]
    WalError(String),
}

// =============================================================================
// Live Order Tracking
// =============================================================================

/// Tracked live order with full lifecycle state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveOrder {
    /// Intent ID (deterministic).
    pub intent_id: IntentId,
    /// Client order ID sent to exchange.
    pub client_order_id: ClientOrderId,
    /// Exchange-assigned order ID (after ack).
    pub exchange_order_id: Option<ExchangeOrderId>,
    /// Strategy requesting the trade.
    pub strategy_id: String,
    /// Bucket providing capital.
    pub bucket_id: String,
    /// Trading symbol.
    pub symbol: String,
    /// Order side.
    pub side: ExecutionSide,
    /// Order type.
    pub order_type: ExecutionOrderType,
    /// Original quantity (mantissa).
    pub quantity_mantissa: i128,
    /// Quantity exponent.
    pub quantity_exponent: i8,
    /// Limit price (mantissa, None for market).
    pub limit_price_mantissa: Option<i128>,
    /// Price exponent.
    pub price_exponent: i8,
    /// Reference price at decision time.
    pub reference_price_mantissa: i128,
    /// Parent decision ID for correlation.
    pub parent_decision_id: String,
    /// Current state.
    pub state: LiveOrderState,
    /// Reserved capital (mantissa).
    pub reserved_mantissa: i128,
    /// Reserved exponent.
    pub reserved_exponent: i8,
    /// Budget delta ID for reservation.
    pub budget_delta_id: Option<String>,
    /// Cumulative filled quantity (mantissa).
    pub cumulative_filled_mantissa: i128,
    /// Total commission (mantissa).
    pub total_commission_mantissa: i128,
    /// Commission exponent.
    pub commission_exponent: i8,
    /// Creation timestamp (nanoseconds).
    pub created_ts_ns: i64,
    /// Last update timestamp (nanoseconds).
    pub updated_ts_ns: i64,
}

impl LiveOrder {
    /// Check if order is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            LiveOrderState::Filled | LiveOrderState::Cancelled | LiveOrderState::Rejected
        )
    }

    /// Remaining quantity (mantissa).
    pub fn remaining_quantity_mantissa(&self) -> i128 {
        self.quantity_mantissa - self.cumulative_filled_mantissa
    }
}

// =============================================================================
// Delta ID Generation
// =============================================================================

/// Generate deterministic delta ID.
pub fn generate_delta_id(prefix: &str, intent_id: &IntentId, ts_ns: i64, seq: u64) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix.as_bytes());
    hasher.update(b":");
    hasher.update(intent_id.0.as_bytes());
    hasher.update(b":");
    hasher.update(ts_ns.to_le_bytes());
    hasher.update(b":");
    hasher.update(seq.to_le_bytes());
    format!("{:x}", hasher.finalize())
}

// =============================================================================
// Budget Interface (Stub for Phase 14.2)
// =============================================================================

/// Budget manager interface for order capital management.
/// This is a stub for Phase 14.2 - real implementation uses Phase 14.1 BudgetManager.
pub trait BudgetInterface: Send + Sync {
    /// Check if order can be placed within budget constraints.
    fn check_order(
        &self,
        strategy_id: &str,
        bucket_id: &str,
        notional_mantissa: i128,
        notional_exponent: i8,
    ) -> Result<(), ExecutionError>;

    /// Reserve capital for an order.
    fn reserve_for_order(
        &mut self,
        strategy_id: &str,
        bucket_id: &str,
        intent_id: &IntentId,
        notional_mantissa: i128,
        notional_exponent: i8,
        ts_ns: i64,
    ) -> Result<String, ExecutionError>;

    /// Release reserved capital (on reject/cancel).
    fn release_order(
        &mut self,
        strategy_id: &str,
        bucket_id: &str,
        intent_id: &IntentId,
        released_mantissa: i128,
        released_exponent: i8,
        ts_ns: i64,
    ) -> Result<String, ExecutionError>;

    /// Commit capital from reserved to position (on fill).
    fn commit_fill(
        &mut self,
        strategy_id: &str,
        bucket_id: &str,
        intent_id: &IntentId,
        fill_mantissa: i128,
        fill_exponent: i8,
        ts_ns: i64,
    ) -> Result<String, ExecutionError>;

    /// Release position capital (on position close).
    fn release_position(
        &mut self,
        strategy_id: &str,
        bucket_id: &str,
        released_mantissa: i128,
        released_exponent: i8,
        ts_ns: i64,
    ) -> Result<String, ExecutionError>;
}

/// Stub budget manager for Phase 14.2 testing.
#[derive(Debug, Default)]
pub struct StubBudgetManager {
    /// Reserved capital per strategy+bucket.
    reserved: HashMap<(String, String), i128>,
    /// Committed capital per strategy+bucket.
    committed: HashMap<(String, String), i128>,
    /// Delta sequence counter.
    delta_seq: u64,
}

impl StubBudgetManager {
    pub fn new() -> Self {
        Self::default()
    }

    fn next_delta_id(&mut self, prefix: &str, intent_id: &IntentId, ts_ns: i64) -> String {
        self.delta_seq += 1;
        generate_delta_id(prefix, intent_id, ts_ns, self.delta_seq)
    }
}

impl BudgetInterface for StubBudgetManager {
    fn check_order(
        &self,
        _strategy_id: &str,
        _bucket_id: &str,
        _notional_mantissa: i128,
        _notional_exponent: i8,
    ) -> Result<(), ExecutionError> {
        // Stub: always passes
        Ok(())
    }

    fn reserve_for_order(
        &mut self,
        strategy_id: &str,
        bucket_id: &str,
        intent_id: &IntentId,
        notional_mantissa: i128,
        _notional_exponent: i8,
        ts_ns: i64,
    ) -> Result<String, ExecutionError> {
        let key = (strategy_id.to_string(), bucket_id.to_string());
        *self.reserved.entry(key).or_insert(0) += notional_mantissa;
        Ok(self.next_delta_id("reserve", intent_id, ts_ns))
    }

    fn release_order(
        &mut self,
        strategy_id: &str,
        bucket_id: &str,
        intent_id: &IntentId,
        released_mantissa: i128,
        _released_exponent: i8,
        ts_ns: i64,
    ) -> Result<String, ExecutionError> {
        let key = (strategy_id.to_string(), bucket_id.to_string());
        if let Some(reserved) = self.reserved.get_mut(&key) {
            *reserved = (*reserved).saturating_sub(released_mantissa);
        }
        Ok(self.next_delta_id("release", intent_id, ts_ns))
    }

    fn commit_fill(
        &mut self,
        strategy_id: &str,
        bucket_id: &str,
        intent_id: &IntentId,
        fill_mantissa: i128,
        _fill_exponent: i8,
        ts_ns: i64,
    ) -> Result<String, ExecutionError> {
        let key = (strategy_id.to_string(), bucket_id.to_string());
        if let Some(reserved) = self.reserved.get_mut(&key) {
            *reserved = (*reserved).saturating_sub(fill_mantissa);
        }
        *self.committed.entry(key).or_insert(0) += fill_mantissa;
        Ok(self.next_delta_id("commit", intent_id, ts_ns))
    }

    fn release_position(
        &mut self,
        strategy_id: &str,
        bucket_id: &str,
        released_mantissa: i128,
        _released_exponent: i8,
        ts_ns: i64,
    ) -> Result<String, ExecutionError> {
        let key = (strategy_id.to_string(), bucket_id.to_string());
        if let Some(committed) = self.committed.get_mut(&key) {
            *committed = (*committed).saturating_sub(released_mantissa);
        }
        // Use a placeholder IntentId for position release
        let placeholder = IntentId("position_release".to_string());
        Ok(self.next_delta_id("pos_release", &placeholder, ts_ns))
    }
}

// =============================================================================
// WAL Interface (Stub for Phase 14.2)
// =============================================================================

/// WAL writer interface for execution events.
pub trait WalInterface: Send + Sync {
    /// Write order intent event.
    fn write_intent(&mut self, event: &OrderIntentEvent) -> Result<(), ExecutionError>;
    /// Write order submit event.
    fn write_submit(&mut self, event: &OrderSubmitEvent) -> Result<(), ExecutionError>;
    /// Write order ack event.
    fn write_ack(&mut self, event: &OrderAckEvent) -> Result<(), ExecutionError>;
    /// Write order reject event.
    fn write_reject(&mut self, event: &OrderRejectEvent) -> Result<(), ExecutionError>;
    /// Write order fill event.
    fn write_fill(&mut self, event: &OrderFillEvent) -> Result<(), ExecutionError>;
    /// Write order cancel event.
    fn write_cancel(&mut self, event: &OrderCancelEvent) -> Result<(), ExecutionError>;
    /// Write position close event.
    fn write_position_close(&mut self, event: &PositionCloseEvent) -> Result<(), ExecutionError>;
}

/// Stub WAL writer for Phase 14.2 testing.
#[derive(Debug, Default)]
pub struct StubWalWriter {
    pub intents: Vec<OrderIntentEvent>,
    pub submits: Vec<OrderSubmitEvent>,
    pub acks: Vec<OrderAckEvent>,
    pub rejects: Vec<OrderRejectEvent>,
    pub fills: Vec<OrderFillEvent>,
    pub cancels: Vec<OrderCancelEvent>,
    pub position_closes: Vec<PositionCloseEvent>,
}

impl StubWalWriter {
    pub fn new() -> Self {
        Self::default()
    }
}

impl WalInterface for StubWalWriter {
    fn write_intent(&mut self, event: &OrderIntentEvent) -> Result<(), ExecutionError> {
        self.intents.push(event.clone());
        Ok(())
    }

    fn write_submit(&mut self, event: &OrderSubmitEvent) -> Result<(), ExecutionError> {
        self.submits.push(event.clone());
        Ok(())
    }

    fn write_ack(&mut self, event: &OrderAckEvent) -> Result<(), ExecutionError> {
        self.acks.push(event.clone());
        Ok(())
    }

    fn write_reject(&mut self, event: &OrderRejectEvent) -> Result<(), ExecutionError> {
        self.rejects.push(event.clone());
        Ok(())
    }

    fn write_fill(&mut self, event: &OrderFillEvent) -> Result<(), ExecutionError> {
        self.fills.push(event.clone());
        Ok(())
    }

    fn write_cancel(&mut self, event: &OrderCancelEvent) -> Result<(), ExecutionError> {
        self.cancels.push(event.clone());
        Ok(())
    }

    fn write_position_close(&mut self, event: &PositionCloseEvent) -> Result<(), ExecutionError> {
        self.position_closes.push(event.clone());
        Ok(())
    }
}

// =============================================================================
// Live Execution Engine
// =============================================================================

/// Live execution engine for Binance Perp.
/// Manages order lifecycle from intent to completion with deterministic budget tracking.
pub struct LiveExecutionEngine<B: BudgetInterface, W: WalInterface> {
    /// Budget manager for capital tracking.
    budget: B,
    /// WAL writer for event persistence.
    wal: W,
    /// Active orders by client order ID.
    orders_by_client_id: HashMap<ClientOrderId, LiveOrder>,
    /// Intent to client order ID mapping.
    intent_to_client_id: HashMap<IntentId, ClientOrderId>,
    /// Exchange order ID to client order ID mapping.
    exchange_to_client_id: HashMap<ExchangeOrderId, ClientOrderId>,
    /// Processed idempotency keys.
    processed_keys: HashSet<IdempotencyKey>,
    /// Sequence counter for intent generation.
    intent_seq: u64,
}

impl<B: BudgetInterface, W: WalInterface> LiveExecutionEngine<B, W> {
    /// Create a new execution engine.
    pub fn new(budget: B, wal: W) -> Self {
        Self {
            budget,
            wal,
            orders_by_client_id: HashMap::new(),
            intent_to_client_id: HashMap::new(),
            exchange_to_client_id: HashMap::new(),
            processed_keys: HashSet::new(),
            intent_seq: 0,
        }
    }

    /// Create order intent from strategy signal.
    #[allow(clippy::too_many_arguments)]
    pub fn create_intent(
        &mut self,
        strategy_id: &str,
        bucket_id: &str,
        symbol: &str,
        side: ExecutionSide,
        order_type: ExecutionOrderType,
        quantity_mantissa: i128,
        quantity_exponent: i8,
        limit_price_mantissa: Option<i128>,
        price_exponent: i8,
        reference_price_mantissa: i128,
        parent_decision_id: &str,
        ts_ns: i64,
    ) -> Result<OrderIntentEvent, ExecutionError> {
        self.intent_seq += 1;
        let intent_id = IntentId::derive(strategy_id, bucket_id, ts_ns, self.intent_seq);

        let mut event = OrderIntentEvent {
            schema_version: EXECUTION_EVENTS_SCHEMA_VERSION.to_string(),
            ts_ns,
            intent_id: intent_id.clone(),
            strategy_id: strategy_id.to_string(),
            bucket_id: bucket_id.to_string(),
            symbol: symbol.to_string(),
            side,
            order_type,
            quantity_mantissa,
            quantity_exponent,
            limit_price_mantissa,
            price_exponent,
            reference_price_mantissa,
            parent_decision_id: parent_decision_id.to_string(),
            state: LiveOrderState::IntentCreated,
            digest: String::new(),
        };
        event.digest = event.compute_digest();

        // Write to WAL
        self.wal.write_intent(&event)?;

        Ok(event)
    }

    /// Submit order to exchange (budget check → reserve → submit).
    pub fn submit_order(
        &mut self,
        intent: &OrderIntentEvent,
        ts_ns: i64,
    ) -> Result<OrderSubmitEvent, ExecutionError> {
        // Calculate notional for budget check
        let price = intent
            .limit_price_mantissa
            .unwrap_or(intent.reference_price_mantissa);
        let notional_mantissa =
            (price * intent.quantity_mantissa) / 10i128.pow((-intent.quantity_exponent) as u32);

        // Step 1: Budget check
        self.budget.check_order(
            &intent.strategy_id,
            &intent.bucket_id,
            notional_mantissa,
            intent.price_exponent,
        )?;

        // Step 2: Reserve capital
        let delta_id = self.budget.reserve_for_order(
            &intent.strategy_id,
            &intent.bucket_id,
            &intent.intent_id,
            notional_mantissa,
            intent.price_exponent,
            ts_ns,
        )?;

        // Step 3: Generate client order ID
        let client_order_id = ClientOrderId::derive(&intent.intent_id, ts_ns);

        // Step 4: Create tracked order
        let order = LiveOrder {
            intent_id: intent.intent_id.clone(),
            client_order_id: client_order_id.clone(),
            exchange_order_id: None,
            strategy_id: intent.strategy_id.clone(),
            bucket_id: intent.bucket_id.clone(),
            symbol: intent.symbol.clone(),
            side: intent.side,
            order_type: intent.order_type,
            quantity_mantissa: intent.quantity_mantissa,
            quantity_exponent: intent.quantity_exponent,
            limit_price_mantissa: intent.limit_price_mantissa,
            price_exponent: intent.price_exponent,
            reference_price_mantissa: intent.reference_price_mantissa,
            parent_decision_id: intent.parent_decision_id.clone(),
            state: LiveOrderState::Submitted,
            reserved_mantissa: notional_mantissa,
            reserved_exponent: intent.price_exponent,
            budget_delta_id: Some(delta_id.clone()),
            cumulative_filled_mantissa: 0,
            total_commission_mantissa: 0,
            commission_exponent: intent.price_exponent,
            created_ts_ns: intent.ts_ns,
            updated_ts_ns: ts_ns,
        };

        // Track the order
        self.orders_by_client_id
            .insert(client_order_id.clone(), order);
        self.intent_to_client_id
            .insert(intent.intent_id.clone(), client_order_id.clone());

        // Step 5: Create submit event
        let mut event = OrderSubmitEvent {
            schema_version: EXECUTION_EVENTS_SCHEMA_VERSION.to_string(),
            ts_ns,
            intent_id: intent.intent_id.clone(),
            client_order_id: client_order_id.clone(),
            budget_delta_id: delta_id,
            reserved_mantissa: notional_mantissa,
            reserved_exponent: intent.price_exponent,
            state: LiveOrderState::Submitted,
            digest: String::new(),
        };
        event.digest = event.compute_digest();

        // Write to WAL
        self.wal.write_submit(&event)?;

        Ok(event)
    }

    /// Process order acknowledgment from exchange.
    pub fn process_ack(
        &mut self,
        client_order_id: &ClientOrderId,
        exchange_order_id: ExchangeOrderId,
        exchange_ts_ns: i64,
        ts_ns: i64,
    ) -> Result<OrderAckEvent, ExecutionError> {
        // Idempotency check
        let idem_key = IdempotencyKey::from_ack(&exchange_order_id.0);
        if self.processed_keys.contains(&idem_key) {
            return Err(ExecutionError::IdempotencyViolation);
        }

        // Find and update order
        let order = self
            .orders_by_client_id
            .get_mut(client_order_id)
            .ok_or_else(|| ExecutionError::OrderNotFound(client_order_id.0.clone()))?;

        // Validate state transition
        if order.state != LiveOrderState::Submitted {
            return Err(ExecutionError::InvalidStateTransition {
                from: format!("{}", order.state),
                to: "Acked".to_string(),
            });
        }

        // Update order state
        order.state = LiveOrderState::Acked;
        order.exchange_order_id = Some(exchange_order_id.clone());
        order.updated_ts_ns = ts_ns;

        // Track exchange order ID
        self.exchange_to_client_id
            .insert(exchange_order_id.clone(), client_order_id.clone());

        // Mark as processed
        self.processed_keys.insert(idem_key);

        // Create ack event
        let mut event = OrderAckEvent {
            schema_version: EXECUTION_EVENTS_SCHEMA_VERSION.to_string(),
            ts_ns,
            client_order_id: client_order_id.clone(),
            exchange_order_id,
            exchange_ts_ns,
            state: LiveOrderState::Acked,
            digest: String::new(),
        };
        event.digest = event.compute_digest();

        // Write to WAL
        self.wal.write_ack(&event)?;

        Ok(event)
    }

    /// Process order rejection from exchange.
    pub fn process_reject(
        &mut self,
        client_order_id: &ClientOrderId,
        exchange_order_id: Option<ExchangeOrderId>,
        reject_reason: &str,
        reject_code: Option<i32>,
        ts_ns: i64,
    ) -> Result<OrderRejectEvent, ExecutionError> {
        // Idempotency check
        let idem_key = IdempotencyKey::from_reject(&client_order_id.0, ts_ns);
        if self.processed_keys.contains(&idem_key) {
            return Err(ExecutionError::IdempotencyViolation);
        }

        // Find order
        let order = self
            .orders_by_client_id
            .get_mut(client_order_id)
            .ok_or_else(|| ExecutionError::OrderNotFound(client_order_id.0.clone()))?;

        // Release reserved capital
        let rollback_delta_id = self.budget.release_order(
            &order.strategy_id,
            &order.bucket_id,
            &order.intent_id,
            order.reserved_mantissa,
            order.reserved_exponent,
            ts_ns,
        )?;

        let released_mantissa = order.reserved_mantissa;
        let released_exponent = order.reserved_exponent;

        // Update order state
        order.state = LiveOrderState::Rejected;
        order.updated_ts_ns = ts_ns;

        // Mark as processed
        self.processed_keys.insert(idem_key);

        // Create reject event
        let mut event = OrderRejectEvent {
            schema_version: EXECUTION_EVENTS_SCHEMA_VERSION.to_string(),
            ts_ns,
            client_order_id: client_order_id.clone(),
            exchange_order_id,
            reject_reason: reject_reason.to_string(),
            reject_code,
            rollback_delta_id,
            released_mantissa,
            released_exponent,
            state: LiveOrderState::Rejected,
            digest: String::new(),
        };
        event.digest = event.compute_digest();

        // Write to WAL
        self.wal.write_reject(&event)?;

        Ok(event)
    }

    /// Process fill from exchange.
    #[allow(clippy::too_many_arguments)]
    pub fn process_fill(
        &mut self,
        client_order_id: &ClientOrderId,
        exchange_order_id: &ExchangeOrderId,
        exchange_fill_id: &str,
        fill_price_mantissa: i128,
        fill_quantity_mantissa: i128,
        exponent: i8,
        commission_mantissa: i128,
        commission_exponent: i8,
        commission_asset: &str,
        ts_ns: i64,
    ) -> Result<OrderFillEvent, ExecutionError> {
        // Idempotency check
        let idem_key = IdempotencyKey::from_fill(exchange_fill_id, &exchange_order_id.0);
        if self.processed_keys.contains(&idem_key) {
            return Err(ExecutionError::IdempotencyViolation);
        }

        // Find order
        let order = self
            .orders_by_client_id
            .get_mut(client_order_id)
            .ok_or_else(|| ExecutionError::OrderNotFound(client_order_id.0.clone()))?;

        // Calculate fill notional
        let fill_notional =
            (fill_price_mantissa * fill_quantity_mantissa) / 10i128.pow((-exponent) as u32);

        // Commit fill capital
        let commitment_delta_id = self.budget.commit_fill(
            &order.strategy_id,
            &order.bucket_id,
            &order.intent_id,
            fill_notional,
            exponent,
            ts_ns,
        )?;

        // Update order
        order.cumulative_filled_mantissa += fill_quantity_mantissa;
        order.total_commission_mantissa += commission_mantissa;
        order.updated_ts_ns = ts_ns;

        let remaining = order.remaining_quantity_mantissa();
        let is_final = remaining <= 0;

        if is_final {
            order.state = LiveOrderState::Filled;
        } else {
            order.state = LiveOrderState::PartFilled;
        }

        // Mark as processed
        self.processed_keys.insert(idem_key);

        // Create fill event
        let fill_id = FillId::derive(exchange_fill_id, client_order_id);
        let mut event = OrderFillEvent {
            schema_version: EXECUTION_EVENTS_SCHEMA_VERSION.to_string(),
            ts_ns,
            client_order_id: client_order_id.clone(),
            exchange_order_id: exchange_order_id.clone(),
            fill_id,
            exchange_fill_id: exchange_fill_id.to_string(),
            fill_price_mantissa,
            fill_quantity_mantissa,
            exponent,
            commission_mantissa,
            commission_exponent,
            commission_asset: commission_asset.to_string(),
            is_final,
            cumulative_filled_mantissa: order.cumulative_filled_mantissa,
            remaining_quantity_mantissa: remaining.max(0),
            commitment_delta_id,
            committed_mantissa: fill_notional,
            committed_exponent: exponent,
            state: order.state,
            digest: String::new(),
        };
        event.digest = event.compute_digest();

        // Write to WAL
        self.wal.write_fill(&event)?;

        Ok(event)
    }

    /// Process order cancellation.
    pub fn process_cancel(
        &mut self,
        client_order_id: &ClientOrderId,
        exchange_order_id: &ExchangeOrderId,
        cancel_source: CancelSource,
        cancel_reason: &str,
        ts_ns: i64,
    ) -> Result<OrderCancelEvent, ExecutionError> {
        // Idempotency check
        let idem_key = IdempotencyKey::from_cancel(&exchange_order_id.0, ts_ns);
        if self.processed_keys.contains(&idem_key) {
            return Err(ExecutionError::IdempotencyViolation);
        }

        // Find order
        let order = self
            .orders_by_client_id
            .get_mut(client_order_id)
            .ok_or_else(|| ExecutionError::OrderNotFound(client_order_id.0.clone()))?;

        // Calculate remaining reserved capital
        let remaining_reserved = order.reserved_mantissa
            - ((order.cumulative_filled_mantissa * order.reserved_mantissa)
                / order.quantity_mantissa);

        // Release remaining reserved capital
        let release_delta_id = self.budget.release_order(
            &order.strategy_id,
            &order.bucket_id,
            &order.intent_id,
            remaining_reserved,
            order.reserved_exponent,
            ts_ns,
        )?;

        let filled_before = order.cumulative_filled_mantissa;

        // Update order state
        order.state = LiveOrderState::Cancelled;
        order.updated_ts_ns = ts_ns;

        // Mark as processed
        self.processed_keys.insert(idem_key);

        // Create cancel event
        let mut event = OrderCancelEvent {
            schema_version: EXECUTION_EVENTS_SCHEMA_VERSION.to_string(),
            ts_ns,
            client_order_id: client_order_id.clone(),
            exchange_order_id: exchange_order_id.clone(),
            cancel_source,
            cancel_reason: cancel_reason.to_string(),
            release_delta_id,
            released_mantissa: remaining_reserved,
            released_exponent: order.reserved_exponent,
            filled_before_cancel_mantissa: filled_before,
            state: LiveOrderState::Cancelled,
            digest: String::new(),
        };
        event.digest = event.compute_digest();

        // Write to WAL
        self.wal.write_cancel(&event)?;

        Ok(event)
    }

    /// Get order by client order ID.
    pub fn get_order(&self, client_order_id: &ClientOrderId) -> Option<&LiveOrder> {
        self.orders_by_client_id.get(client_order_id)
    }

    /// Get order by intent ID.
    pub fn get_order_by_intent(&self, intent_id: &IntentId) -> Option<&LiveOrder> {
        self.intent_to_client_id
            .get(intent_id)
            .and_then(|coid| self.orders_by_client_id.get(coid))
    }

    /// Get order by exchange order ID.
    pub fn get_order_by_exchange_id(
        &self,
        exchange_order_id: &ExchangeOrderId,
    ) -> Option<&LiveOrder> {
        self.exchange_to_client_id
            .get(exchange_order_id)
            .and_then(|coid| self.orders_by_client_id.get(coid))
    }

    /// Count of active (non-terminal) orders.
    pub fn active_order_count(&self) -> usize {
        self.orders_by_client_id
            .values()
            .filter(|o| !o.is_terminal())
            .count()
    }

    /// Count of all tracked orders.
    pub fn total_order_count(&self) -> usize {
        self.orders_by_client_id.len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_engine() -> LiveExecutionEngine<StubBudgetManager, StubWalWriter> {
        LiveExecutionEngine::new(StubBudgetManager::new(), StubWalWriter::new())
    }

    #[test]
    fn test_create_intent() {
        let mut engine = create_engine();

        let intent = engine
            .create_intent(
                "strategy_001",
                "bucket_001",
                "BTCUSDT",
                ExecutionSide::Buy,
                ExecutionOrderType::Limit,
                100_000_000, // 1 BTC
                -8,
                Some(50000_00000000), // $50,000
                -8,
                50000_00000000,
                "decision_001",
                1234567890_000_000_000,
            )
            .unwrap();

        assert_eq!(intent.strategy_id, "strategy_001");
        assert_eq!(intent.bucket_id, "bucket_001");
        assert_eq!(intent.symbol, "BTCUSDT");
        assert_eq!(intent.state, LiveOrderState::IntentCreated);
        assert!(!intent.digest.is_empty());
    }

    #[test]
    fn test_submit_order() {
        let mut engine = create_engine();

        let intent = engine
            .create_intent(
                "strategy_001",
                "bucket_001",
                "BTCUSDT",
                ExecutionSide::Buy,
                ExecutionOrderType::Limit,
                100_000_000,
                -8,
                Some(50000_00000000),
                -8,
                50000_00000000,
                "decision_001",
                1234567890_000_000_000,
            )
            .unwrap();

        let submit = engine
            .submit_order(&intent, 1234567891_000_000_000)
            .unwrap();

        assert_eq!(submit.intent_id, intent.intent_id);
        assert_eq!(submit.state, LiveOrderState::Submitted);
        assert!(!submit.budget_delta_id.is_empty());
        assert!(submit.reserved_mantissa > 0);

        // Order should be tracked
        let order = engine.get_order(&submit.client_order_id).unwrap();
        assert_eq!(order.state, LiveOrderState::Submitted);
    }

    #[test]
    fn test_process_ack() {
        let mut engine = create_engine();

        let intent = engine
            .create_intent(
                "strategy_001",
                "bucket_001",
                "BTCUSDT",
                ExecutionSide::Buy,
                ExecutionOrderType::Limit,
                100_000_000,
                -8,
                Some(50000_00000000),
                -8,
                50000_00000000,
                "decision_001",
                1234567890_000_000_000,
            )
            .unwrap();

        let submit = engine
            .submit_order(&intent, 1234567891_000_000_000)
            .unwrap();

        let ack = engine
            .process_ack(
                &submit.client_order_id,
                ExchangeOrderId::new("EX_ORDER_001"),
                1234567892_000_000_000,
                1234567892_000_000_000,
            )
            .unwrap();

        assert_eq!(ack.state, LiveOrderState::Acked);
        assert_eq!(ack.exchange_order_id.0, "EX_ORDER_001");

        // Order state updated
        let order = engine.get_order(&submit.client_order_id).unwrap();
        assert_eq!(order.state, LiveOrderState::Acked);
    }

    #[test]
    fn test_process_fill() {
        let mut engine = create_engine();

        let intent = engine
            .create_intent(
                "strategy_001",
                "bucket_001",
                "BTCUSDT",
                ExecutionSide::Buy,
                ExecutionOrderType::Limit,
                100_000_000, // 1 BTC
                -8,
                Some(50000_00000000),
                -8,
                50000_00000000,
                "decision_001",
                1234567890_000_000_000,
            )
            .unwrap();

        let submit = engine
            .submit_order(&intent, 1234567891_000_000_000)
            .unwrap();

        engine
            .process_ack(
                &submit.client_order_id,
                ExchangeOrderId::new("EX_ORDER_001"),
                1234567892_000_000_000,
                1234567892_000_000_000,
            )
            .unwrap();

        // Partial fill
        let fill1 = engine
            .process_fill(
                &submit.client_order_id,
                &ExchangeOrderId::new("EX_ORDER_001"),
                "FILL_001",
                50000_00000000,
                50_000_000, // 0.5 BTC
                -8,
                25_000_000, // Commission
                -8,
                "BNB",
                1234567893_000_000_000,
            )
            .unwrap();

        assert_eq!(fill1.state, LiveOrderState::PartFilled);
        assert!(!fill1.is_final);

        // Final fill
        let fill2 = engine
            .process_fill(
                &submit.client_order_id,
                &ExchangeOrderId::new("EX_ORDER_001"),
                "FILL_002",
                50000_00000000,
                50_000_000, // 0.5 BTC
                -8,
                25_000_000,
                -8,
                "BNB",
                1234567894_000_000_000,
            )
            .unwrap();

        assert_eq!(fill2.state, LiveOrderState::Filled);
        assert!(fill2.is_final);
    }

    #[test]
    fn test_process_reject() {
        let mut engine = create_engine();

        let intent = engine
            .create_intent(
                "strategy_001",
                "bucket_001",
                "BTCUSDT",
                ExecutionSide::Buy,
                ExecutionOrderType::Limit,
                100_000_000,
                -8,
                Some(50000_00000000),
                -8,
                50000_00000000,
                "decision_001",
                1234567890_000_000_000,
            )
            .unwrap();

        let submit = engine
            .submit_order(&intent, 1234567891_000_000_000)
            .unwrap();

        let reject = engine
            .process_reject(
                &submit.client_order_id,
                Some(ExchangeOrderId::new("EX_ORDER_001")),
                "Insufficient margin",
                Some(-2010),
                1234567892_000_000_000,
            )
            .unwrap();

        assert_eq!(reject.state, LiveOrderState::Rejected);
        assert_eq!(reject.reject_reason, "Insufficient margin");
        assert!(reject.released_mantissa > 0);

        // Order state updated
        let order = engine.get_order(&submit.client_order_id).unwrap();
        assert_eq!(order.state, LiveOrderState::Rejected);
    }

    #[test]
    fn test_process_cancel() {
        let mut engine = create_engine();

        let intent = engine
            .create_intent(
                "strategy_001",
                "bucket_001",
                "BTCUSDT",
                ExecutionSide::Buy,
                ExecutionOrderType::Limit,
                100_000_000,
                -8,
                Some(50000_00000000),
                -8,
                50000_00000000,
                "decision_001",
                1234567890_000_000_000,
            )
            .unwrap();

        let submit = engine
            .submit_order(&intent, 1234567891_000_000_000)
            .unwrap();

        engine
            .process_ack(
                &submit.client_order_id,
                ExchangeOrderId::new("EX_ORDER_001"),
                1234567892_000_000_000,
                1234567892_000_000_000,
            )
            .unwrap();

        let cancel = engine
            .process_cancel(
                &submit.client_order_id,
                &ExchangeOrderId::new("EX_ORDER_001"),
                CancelSource::User,
                "User requested cancel",
                1234567893_000_000_000,
            )
            .unwrap();

        assert_eq!(cancel.state, LiveOrderState::Cancelled);
        assert_eq!(cancel.cancel_source, CancelSource::User);
        assert!(cancel.released_mantissa > 0);
    }

    #[test]
    fn test_idempotency() {
        let mut engine = create_engine();

        let intent = engine
            .create_intent(
                "strategy_001",
                "bucket_001",
                "BTCUSDT",
                ExecutionSide::Buy,
                ExecutionOrderType::Limit,
                100_000_000,
                -8,
                Some(50000_00000000),
                -8,
                50000_00000000,
                "decision_001",
                1234567890_000_000_000,
            )
            .unwrap();

        let submit = engine
            .submit_order(&intent, 1234567891_000_000_000)
            .unwrap();

        // First ack succeeds
        engine
            .process_ack(
                &submit.client_order_id,
                ExchangeOrderId::new("EX_ORDER_001"),
                1234567892_000_000_000,
                1234567892_000_000_000,
            )
            .unwrap();

        // Second ack with same exchange order ID fails (idempotency)
        let result = engine.process_ack(
            &submit.client_order_id,
            ExchangeOrderId::new("EX_ORDER_001"),
            1234567893_000_000_000,
            1234567893_000_000_000,
        );

        assert!(matches!(result, Err(ExecutionError::IdempotencyViolation)));
    }

    #[test]
    fn test_intent_id_deterministic() {
        let id1 = IntentId::derive("strategy_001", "bucket_001", 1234567890, 1);
        let id2 = IntentId::derive("strategy_001", "bucket_001", 1234567890, 1);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_order_count() {
        let mut engine = create_engine();

        // Initially empty
        assert_eq!(engine.active_order_count(), 0);
        assert_eq!(engine.total_order_count(), 0);

        // Create and submit an order
        let intent = engine
            .create_intent(
                "strategy_001",
                "bucket_001",
                "BTCUSDT",
                ExecutionSide::Buy,
                ExecutionOrderType::Limit,
                100_000_000,
                -8,
                Some(50000_00000000),
                -8,
                50000_00000000,
                "decision_001",
                1234567890_000_000_000,
            )
            .unwrap();

        engine
            .submit_order(&intent, 1234567891_000_000_000)
            .unwrap();

        assert_eq!(engine.active_order_count(), 1);
        assert_eq!(engine.total_order_count(), 1);
    }
}
