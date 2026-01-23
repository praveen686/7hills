//! # SANOS-Gated Trading Strategies
//!
//! Strategy implementations that use SANOS surfaces as the fair-value engine.
//!
//! ## Strategy v0: Calendar Carry with Skew Regime Filter
//! Trade short-vs-long expiry variance carry only when:
//! - SANOS surface is stable and monotone
//! - Calendar gap exceeds microstructure friction
//! - Skew regime is not indicating tail stress

pub mod calendar_carry;

pub use calendar_carry::{
    CalendarCarryStrategy, StrategyContext, StrategyDecision, GateResult,
    GateCheckResult, QuoteSnapshot, StraddleQuotes, Phase8Features, SessionMeta,
    AuditRecord, EnterIntent, ExitIntent, FROZEN_PARAMS,
};
