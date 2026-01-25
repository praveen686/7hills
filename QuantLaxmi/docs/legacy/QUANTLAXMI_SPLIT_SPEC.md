# QuantLaxmi Split Spec (One Repo, Two Binaries)

This repository is being rebranded from QuantKubera to **QuantLaxmi**.

## Binaries

- **quantlaxmi_india**: India indices & F&O (Zerodha-family venues)
- **quantlaxmi_crypto**: Crypto (Binance-family venues)

## Current State (Scaffold)

The two binaries have been introduced as separate `src/bin/*` targets within `crates/kubera-runner`.
They currently share the same CLI entry (`kubera_runner::run_cli()`), as a safe first cut.

## Next Refactor (Hard Separation)

1. Split `crates/kubera-connectors` into subcrates:
   - `quantlaxmi-connectors-zerodha`
   - `quantlaxmi-connectors-binance`
2. Extract a `quantlaxmi-runner-lib` and move shared orchestration into it.
3. Make `quantlaxmi_india` depend only on India crates + Zerodha connector.
4. Make `quantlaxmi_crypto` depend only on Crypto crates + Binance SBE connector.
5. Add CI `cargo tree` guards to prevent dependency bleed.
