# 7hills Migration Policy (Revised & Enforced)

## 1. Authoritative Projects

| Project | Status | Policy |
|---------|--------|--------|
| **QuantLaxmi** | **Authoritative** | All active development, CI-enforced |
| QuantKubera1 | Legacy / Frozen | Read-only migration reference |

**Rules:**
- No new logic, fixes, or refactors are allowed in QuantKubera1
- Any required change must be ported to QuantLaxmi or documented as intentionally abandoned

---

## 2. CI Authority & Enforcement

### Single CI Authority Rule

Only root-level workflows are allowed.

Root CI **must explicitly fail** if:
- Any workflow exists under `QuantKubera1/.github/`
- Any workflow exists under `QuantLaxmi/.github/`

**Rationale:** GitHub executes all workflows recursively. This rule prevents accidental CI re-activation of legacy code.

### Enforcement (required in CI):
```bash
test ! -d QuantKubera1/.github
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

**QuantKubera1 must never be built, tested, or linted by CI.**

---

## 4. Deletion Gate for QuantKubera1 (Hard Gate)

QuantKubera1 may be deleted **only when all of the following are true**.

### Crypto (Binance) Parity Gate

- [ ] SBE + REST capture certified
- [ ] Replay determinism:
  - Identical inputs → identical output hashes
- [ ] Paper + backtest pipelines produce:
  - Stable manifests
  - Stable VectorBT exports
- [ ] Metric parity validated vs QuantKubera:
  - Trade count match
  - PnL tolerance ≤ ε
  - No missing fills

### India (Zerodha) Parity Gate

- [ ] Capture includes:
  - Underlying reference (FUT or spot proxy)
  - Options universe
- [ ] Validation strategies produce fills
- [ ] HYDRA / AEON:
  - Trades OR
  - Explainable gate diagnostics
- [ ] VectorBT export runs end-to-end
- [ ] At least one real session certified

---

## 5. Legacy Crate Handling (Non-Binary)

After QuantKubera deletion, each `kubera-*` crate must be classified as one of:

| State | Meaning |
|-------|---------|
| **Retained** | Still authoritative, used directly |
| **Wrapped** | Used via QuantLaxmi adapter layer |
| **Deprecated** | Marked for deletion, not used |

**No crate may remain in an implicit state.**

---

## 6. Final Cleanup (After Gates Pass)

1. Delete `QuantKubera1/`
2. Remove unused legacy crates
3. Rename retained crates if branding cleanup is desired
4. Freeze migration policy (historical document)

---

## 7. Timeline

**There is no calendar-based deadline.**

Migration completes only when parity and determinism gates pass.

---

*Last revised: 2026-01-22*
