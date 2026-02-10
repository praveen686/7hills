# 7hills Migration Policy (Post-Decommission)

## 1. Current State

| Project | Status | Policy |
|---------|--------|--------|
| **QuantLaxmi** | **Authoritative** | All active development, CI-enforced |
| ~~QuantKubera1~~ | **DELETED** | Decommissioned 2026-01-23 |

**QuantKubera1 has been permanently deleted.** All development happens in QuantLaxmi.

---

## 2. CI Authority & Enforcement

### Single CI Authority Rule

Only root-level workflows are allowed.

Root CI **must explicitly fail** if:
- Any workflow exists under `QuantLaxmi/.github/`

### Enforcement (required in CI):
```bash
test ! -d QuantLaxmi/.github
```

---

## 3. QuantLaxmi CI Scope

Root CI (`.github/workflows/quantlaxmi-ci.yml`) must:

1. **Build only** QuantLaxmi workspace crates
2. **Enforce strict clippy** on:
   - `quantlaxmi-*` crates
3. **Allow warnings only** on:
   - `kubera-*` crates (until migration complete)
4. **Enforce dependency isolation** (India ↔ Crypto)

---

## 4. Legacy Crate Retirement (In Progress)

The following `kubera-*` crates inside QuantLaxmi are being retired:

| Crate | Status | Replacement |
|-------|--------|-------------|
| `kubera-options` | Pending retirement | `quantlaxmi-options` |
| `kubera-connectors` | Pending retirement | `quantlaxmi-connectors-zerodha` |
| `kubera-runner` | Pending retirement | `quantlaxmi-runner-*` |

**No new code should depend on `kubera-*` crates.**

---

## 5. Final Cleanup Checklist

- [x] Delete `QuantKubera1/` external project
- [ ] Retire `kubera-options` crate (deferred - core functionality)
- [x] Retire `kubera-connectors` crate
- [x] Retire `kubera-runner` crate
- [x] Update `check_isolation.sh` to ban `kubera-*`
- [ ] All `cargo tree | rg kubera` returns empty (foundational crates remain)

---

*Last revised: 2026-01-23*
