#!/usr/bin/env bash
# quantlaxmi_migration_harness.sh
#
# Purpose:
#   Fast, repeatable migration gating for Phase 4 crate moves (kubera-* -> quantlaxmi-*).
#   Runs:
#     - workspace tests
#     - targeted crate tests (optional)
#     - golden India replay hash
#     - golden Crypto replay hash
#     - golden SANOS hash (India)
#     - direct-import guardrails (kubera_* in quantlaxmi-* code)
#
# Usage:
#   From QuantLaxmi/:
#     chmod +x scripts/quantlaxmi_migration_harness.sh
#     GOLDEN_INDIA_SESSION="data/sessions/INDIA_GOLDEN" \
#     GOLDEN_CRYPTO_SESSION="data/sessions/CRYPTO_GOLDEN" \
#     ./scripts/quantlaxmi_migration_harness.sh
#
# Optional:
#   TARGET_CRATE=quantlaxmi-options ./scripts/quantlaxmi_migration_harness.sh
#   BASELINE_DIR=.golden_hashes ./scripts/quantlaxmi_migration_harness.sh
#
set -euo pipefail

ROOT_DIR="$(pwd)"
if [[ ! -f "${ROOT_DIR}/Cargo.toml" ]]; then
  echo "ERROR: Run this script from QuantLaxmi/ (Cargo.toml not found in $(pwd))" >&2
  exit 1
fi

# ---------- Config ----------
GOLDEN_INDIA_SESSION="${GOLDEN_INDIA_SESSION:-}"
GOLDEN_CRYPTO_SESSION="${GOLDEN_CRYPTO_SESSION:-}"

# Where we store baseline hashes (committable or local; your choice)
BASELINE_DIR="${BASELINE_DIR:-.golden_hashes}"
mkdir -p "${BASELINE_DIR}"

# If set, we run cargo tests for that crate specifically before workspace tests
TARGET_CRATE="${TARGET_CRATE:-}"

# Binaries / commands (adjust flags to match your CLI)
INDIA_REPLAY_CMD=(
  cargo run -p quantlaxmi-runner-india --release -- replay
)
CRYPTO_REPLAY_CMD=(
  cargo run -p quantlaxmi-runner-crypto --release -- replay
)
SANOS_CMD=(
  cargo run -p quantlaxmi-runner-india --release --bin sanos_multi_expiry --
)

# Output capture
OUT_DIR="${OUT_DIR:-/tmp/quantlaxmi_migration_harness}"
mkdir -p "${OUT_DIR}"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

sha_file() {
  local f="$1"
  sha256sum "$f" | awk '{print $1}'
}

save_or_check_hash() {
  local name="$1"
  local actual="$2"
  local baseline_file="${BASELINE_DIR}/${name}.sha256"

  if [[ -f "${baseline_file}" ]]; then
    local expected
    expected="$(cat "${baseline_file}")"
    if [[ "${expected}" != "${actual}" ]]; then
      echo "FAIL: ${name} hash mismatch"
      echo "  expected: ${expected}"
      echo "  actual:   ${actual}"
      exit 2
    else
      log "OK: ${name} hash matches baseline (${actual})"
    fi
  else
    log "BASELINE: writing ${name} hash -> ${baseline_file}"
    echo -n "${actual}" > "${baseline_file}"
    log "NOTE: baseline created. Re-run to enforce stability."
  fi
}

require_env_or_skip() {
  local var_name="$1"
  local var_value="$2"
  local what="$3"
  if [[ -z "${var_value}" ]]; then
    log "SKIP: ${what} (env ${var_name} not set)"
    return 1
  fi
  return 0
}

# ---------- Guardrails ----------
log "Guardrails: ensure quantlaxmi-* code does not directly import kubera_* namespaces"
if rg -n "kubera_(models|core|data|risk|executor|options|sbe)::" crates/quantlaxmi-* 2>/dev/null; then
  echo "FAIL: Found direct imports of kubera_* in quantlaxmi-* code. Fix Phase 2 first." >&2
  exit 3
fi
log "OK: no direct kubera_* imports in quantlaxmi-* code"

# ---------- Build/Test ----------
if [[ -n "${TARGET_CRATE}" ]]; then
  log "Running targeted tests for crate: ${TARGET_CRATE}"
  cargo test -p "${TARGET_CRATE}"
fi

log "Running full workspace tests"
cargo test --workspace
log "OK: workspace tests"

# ---------- Golden Replays ----------
# INDIA replay
if require_env_or_skip "GOLDEN_INDIA_SESSION" "${GOLDEN_INDIA_SESSION}" "India golden replay"; then
  INDIA_OUT="${OUT_DIR}/india_replay.out"
  log "Running India replay on session: ${GOLDEN_INDIA_SESSION}"
  "${INDIA_REPLAY_CMD[@]}" --session "${GOLDEN_INDIA_SESSION}" > "${INDIA_OUT}"
  INDIA_HASH="$(sha_file "${INDIA_OUT}")"
  save_or_check_hash "india_replay" "${INDIA_HASH}"
fi

# CRYPTO replay
if require_env_or_skip "GOLDEN_CRYPTO_SESSION" "${GOLDEN_CRYPTO_SESSION}" "Crypto golden replay"; then
  CRYPTO_OUT="${OUT_DIR}/crypto_replay.out"
  log "Running Crypto replay on session: ${GOLDEN_CRYPTO_SESSION}"
  "${CRYPTO_REPLAY_CMD[@]}" --session "${GOLDEN_CRYPTO_SESSION}" > "${CRYPTO_OUT}"
  CRYPTO_HASH="$(sha_file "${CRYPTO_OUT}")"
  save_or_check_hash "crypto_replay" "${CRYPTO_HASH}"
fi

# SANOS
if require_env_or_skip "GOLDEN_INDIA_SESSION" "${GOLDEN_INDIA_SESSION}" "SANOS golden run"; then
  SANOS_OUT="${OUT_DIR}/sanos_multi_expiry.out"
  log "Running SANOS multi-expiry on session: ${GOLDEN_INDIA_SESSION}"
  "${SANOS_CMD[@]}" --session "${GOLDEN_INDIA_SESSION}" > "${SANOS_OUT}"
  SANOS_HASH="$(sha_file "${SANOS_OUT}")"
  save_or_check_hash "sanos_multi_expiry" "${SANOS_HASH}"
fi

log "SUCCESS: migration harness passed"
log "Outputs in: ${OUT_DIR}"
log "Baselines in: ${BASELINE_DIR}"
