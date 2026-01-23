#!/bin/bash
set -euo pipefail

# Source environment
source /etc/quantlaxmi/india-capture.env

# Create timestamped session dir
ts="$(date +%Y%m%d_%H%M%S)"
out="${QL_SESSIONS_DIR}/phase9_3_${ts}"
mkdir -p "${out}"
echo "SessionDir=${out}"

# Run capture
exec "${QL_HOME}/target/release/quantlaxmi-india" capture-session \
    --underlying "${QL_UNDERLYING}" \
    --strike-band "${QL_STRIKE_BAND}" \
    --expiry-policy "${QL_EXPIRY_POLICY}" \
    --out-dir "${out}" \
    --duration-secs "${QL_DURATION_SECS}" \
    ${QL_EXTRA_FLAGS}
