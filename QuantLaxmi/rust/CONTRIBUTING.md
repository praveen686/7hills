# Contributing to QuantLaxmi

## Required Reading

Before contributing, read:
- `docs/DOCTRINE_NO_SILENT_POISONING.md` — **Mandatory** data integrity rules

## Core Rules

### 1. No Silent Poisoning (D1-D4)

**Never default vendor omissions into numeric values.**

```rust
// BANNED
let qty = quote.buy_quantity.unwrap_or(0);

// REQUIRED
let qty = require_u64(quote.buy_quantity, "buy_quantity")?;
// or
let imbalance = book_imbalance_fixed(quote.buy_quantity, quote.sell_quantity)?;
```

See `docs/DOCTRINE_NO_SILENT_POISONING.md` for complete rules.

### 2. Internal Artifacts Are Strict

No `#[serde(default)]` in WAL, manifests, configs, or events.
Missing field = deserialization error = producer bug exposed early.

### 3. Deterministic Rounding

All fixed-point division must:
- State rounding rule in doc comment
- Have at least one test verifying exact behavior

## CI Checks

PRs must pass:
```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
./scripts/lint_no_silent_poisoning.sh
```

## Code Review Checklist

Reviewers verify:
- [ ] New vendor fields are `Option<T>` if vendor may omit
- [ ] Signal computations propagate `Option`/`Result` (no silent defaults)
- [ ] Fixed-point divisions have rounding docs + tests
- [ ] No `unwrap_or(0)` on vendor quantity/volume fields

## Questions?

Open an issue or see `docs/` for architecture and design decisions.
