//! QuantLaxmi Executor - shim layer re-exporting kubera-executor
//!
//! This crate provides the migration path from kubera-executor to quantlaxmi-executor.
//! All types are re-exported wholesale; no behavior changes.

pub use kubera_executor::*;
