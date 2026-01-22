# 7hills Migration Policy

## Current State

| Project | Status | Purpose |
|---------|--------|---------|
| **QuantLaxmi** | **Authoritative** | Active development, CI-enforced |
| QuantKubera1 | Legacy Reference | Read-only, migration source |

## Rules

1. **QuantLaxmi is authoritative** — All new features, fixes, and improvements go here.

2. **QuantKubera1 is read-only legacy** — No new commits. Used only as reference during migration.

3. **CI enforces QuantLaxmi only** — The root workflow (`.github/workflows/quantlaxmi-ci.yml`) runs build, clippy, tests, and isolation checks exclusively on QuantLaxmi.

4. **No subproject CI files** — Workflows inside `QuantLaxmi/.github/` or `QuantKubera1/.github/` are deleted to avoid confusion (GitHub only runs root-level workflows).

## Deletion Gate for QuantKubera1

QuantKubera1 will be deleted when **both** of these gates pass:

### Crypto (Binance) Parity Gate
- [ ] Certified depth capture + replay determinism tests remain green
- [ ] Paper/backtest pipeline produces stable hashes
- [ ] VectorBT export runs end-to-end

### India (Zerodha) Parity Gate
- [ ] Capture includes underlying reference (FUT or spot) + options
- [ ] Backtest produces fills for validation strategies
- [ ] HYDRA/AEON runs produce either trades OR explainably gated "no trade" diagnostics
- [ ] VectorBT export runs end-to-end on at least one real session

## Post-Deletion Cleanup

When both gates are met:

1. Remove `QuantKubera1/` directory entirely
2. Evaluate remaining `kubera-*` crates in QuantLaxmi workspace:
   - Keep: Core infrastructure still needed (kubera-core, kubera-options, etc.)
   - Remove: Any crates that are fully superseded
3. Optional: Rename remaining crates for clean branding

## Timeline

No fixed date. Migration completes when quality gates pass, not by calendar.

---

*Last updated: 2026-01-22*
