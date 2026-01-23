# Kubera Decommission Record

## Status: Partially Complete

**Date:** 2026-01-23

## Summary

The QuantKubera project and legacy kubera-* crates are being systematically retired. This document records the decommission status and policies.

---

## Completed Deletions

| Item | Type | Status | Date |
|------|------|--------|------|
| `QuantKubera1/` | External project | **DELETED** | 2026-01-23 |
| `kubera-runner` | Workspace crate | **DELETED** | 2026-01-23 |
| `kubera-connectors` | Workspace crate | **DELETED** | 2026-01-23 |
| `kubera-executor-cpp` | Orphaned directory | **DELETED** | 2026-01-23 |

---

## Remaining Kubera Crates (Foundational)

The following crates remain as foundational infrastructure. They are used throughout the codebase and require careful migration.

| Crate | Used By | Migration Status |
|-------|---------|------------------|
| `kubera-core` | Everything | Foundational, keep for now |
| `kubera-models` | Everything | Foundational, keep for now |
| `kubera-data` | India/Crypto runners | Foundational, keep for now |
| `kubera-executor` | India/Crypto runners | Foundational, keep for now |
| `kubera-risk` | India runner | Foundational, keep for now |
| `kubera-options` | India/Crypto runners | Core options logic (SANOS, KiteSim, pricing) |
| `kubera-sbe` | Crypto runner | Binance SBE codec |
| `kubera-backtest` | Backtest tools | Backtest infrastructure |
| `kubera-mlflow` | Experiment tracking | MLflow integration |
| `kubera-ffi` | FFI bindings | Foreign function interface |
| `kubera-strategy-host` | Strategy hosting | Strategy runtime |

---

## Policy

### DO NOT reintroduce deleted crates

The following crates are **permanently banned**:
- `kubera-runner`
- `kubera-connectors`

CI will fail if these crates reappear in the workspace.

### New shared logic must use quantlaxmi-* naming

All new shared crates must be named `quantlaxmi-*`:
- `quantlaxmi-runner-common`
- `quantlaxmi-runner-india`
- `quantlaxmi-runner-crypto`
- `quantlaxmi-connectors-zerodha`
- `quantlaxmi-connectors-binance`

### Future migration path

When migrating foundational kubera-* crates:
1. Create equivalent `quantlaxmi-*` crate
2. Port required functionality
3. Update all dependents
4. Verify with `cargo tree`
5. Delete old crate
6. Update this document

---

## Verification Commands

```bash
# Verify no banned crates in tree
cargo tree -e normal | grep -E "kubera-runner|kubera-connectors" && echo "FAIL" || echo "OK"

# Verify QuantKubera1 removed
test ! -d ../QuantKubera1 && echo "OK" || echo "FAIL"

# Run full isolation check
bash scripts/check_isolation.sh
```

---

*Last updated: 2026-01-23*
