// Legacy crate - suppress pre-existing clippy warnings pending migration
#![allow(clippy::collapsible_if)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::unnecessary_sort_by)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::cast_abs_to_unsigned)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(dead_code)]

//! # Advanced Derivatives & Options Engine
//!
//! Comprehensive toolkit for modeling, pricing, and executing option strategies.
//!
//! ## Description
//! QuantKubera's options crate provides a high-performance foundation for
//! derivative trading. It handles the entire lifecycle from contract definition
//! and pricing to multi-leg strategy execution and risk management.
//!
//! ### Core Subsystems
//! - **Pricing & Greeks**: Black-Scholes implementations for theoretical valuation
//!   and sensitivity analysis (Delta, Gamma, Theta, Vega, Rho).
//! - **Strategy Management**: Abstractions for complex multi-leg structures like
//!   Straddles, Iron Condors, and Spreads.
//! - **Analytics**: 3D Implied Volatility (IV) surface modeling and Greeks decay
//!   projections.
//! - **Execution**: Atomic multi-leg order routing with integrated rollback logic.
//!
//! ## References
//! - IEEE Std 1016-2009: Software Design Descriptions
//! - Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.

pub mod analytics;
pub mod backtest;
pub mod chain;
pub mod contract;
pub mod execution;
pub mod greeks;
pub mod kitesim;
pub mod margin;
pub mod nse_specs;
pub mod pricing;
pub mod replay;
pub mod report;
pub mod sanos;
pub mod signals;
pub mod specs;
pub mod strategies;
pub mod strategy;

pub use chain::{IVPoint, IVSurface, OptionChainFetcher};
pub use contract::{OptionChain, OptionContract, OptionType};
pub use greeks::OptionGreeks;
pub use margin::{OptionsMargin, PortfolioGreeks};
pub use nse_specs::{
    LotSizeValidator, NseIndex, NseOrderValidator, NseTradingHours, TickSizeValidator, TradingPhase,
};
pub use pricing::{black_scholes_call, black_scholes_put, implied_volatility};
pub use sanos::{
    ExpirySlice, NormalizedSlice, OptionQuote, SanosCalibrator, SanosDiagnostics, SanosSlice,
};
pub use signals::{OptionsSignal, OptionsSignalGenerator, SignalType};
pub use strategy::{OptionsStrategy, StrategyLeg, StrategyType, build_iron_condor, build_straddle};
