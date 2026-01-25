# Kubera Zero Policy

**Status**: Enforced
**Effective**: January 2026
**Last Updated**: 2026-01-23

## Policy Statement

The "Kubera" brand is permanently retired. No active code in this repository may reference "Kubera" in any form (crate names, module names, imports, or functional comments).

## Allowed Exceptions

References to "Kubera" are permitted ONLY in:

| Path | Reason |
|------|--------|
| `docs/` | Migration history and audit trail |
| `patches/` | Historical patches (archival) |
| `python/quantlaxmi_sdk/` | Deprecation warnings and historical notes only |

## Denied Locations

The following paths must be Kubera-free:

- `crates/**/*.rs`
- `crates/**/Cargo.toml`
- `apps/**/*.rs`
- `apps/**/Cargo.toml`
- `python/**/*.py` (except deprecation warnings)
- `scripts/` (guard logic references are allowed)
- `Cargo.toml` (root)
- `README.md`

## Enforcement

This policy is enforced by CI gates in `.github/workflows/quantlaxmi-ci.yml`:

1. **No Kubera crates**: `crates/kubera-*` directories must not exist
2. **No Kubera deps**: No `kubera-` dependencies in Cargo.toml
3. **No Kubera imports**: No `kubera_*::` imports in Rust code
4. **No kubera_sdk**: Python must use `quantlaxmi_sdk`
5. **Consolidated gate**: No "kubera" in active code paths

## Rationale

- **Audit clarity**: Single brand eliminates confusion
- **Dependency hygiene**: Prevents accidental reintroduction
- **Migration completeness**: Phase 4.8 decommission is final

## History

| Date | Event |
|------|-------|
| 2026-01-23 | Phase 4.8: All kubera-* crates deleted |
| 2026-01-23 | Python SDK renamed: kubera_sdk → quantlaxmi_sdk |
| 2026-01-23 | CI gates hardened |
| 2026-01-23 | Kubera Zero Policy established |

## Related Documents

- [DECOMMISSION_KUBERA.md](./DECOMMISSION_KUBERA.md) - Decommission record
- [MIGRATION_POLICY.md](./MIGRATION_POLICY.md) - Migration history
