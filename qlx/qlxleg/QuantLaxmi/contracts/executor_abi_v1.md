# Executor ABI v1 (C++ -> Rust FFI)

See `crates/quantlaxmi-executor/src/lib.rs` for the Rust executor implementation.

## Scope
- L2 book snapshot input (up to DEPTH_MAX=10 levels)
- Market order multi-level fill costing (VWAP)
- Deterministic taker fees + fixed latency

## Non-goals (Phase 0)
- Queue position / matching engine modeling
- Maker (post-only) orders
- Stochastic slippage
