#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  echo "usage: ./scripts/model_switch_smoke.sh <target-model-or-label>" >&2
  exit 1
fi

echo "[1/4] Refresh census"
"$ROOT/target/release/cto-agent" run-census >/tmp/cto_model_switch_census.json
sed -n '1,220p' /tmp/cto_model_switch_census.json

echo "[2/4] Attempt targeted upgrade"
"$ROOT/target/release/cto-agent" upgrade-kleinhirn "$TARGET"

echo "[3/4] Effective runtime env"
grep '^CTO_AGENT_KLEINHIRN_' "$ROOT/runtime/kleinhirn.env"

echo "[4/4] Probe current model"
CTO_AGENT_KLEINHIRN_STARTUP_WAIT_SECS="${CTO_AGENT_KLEINHIRN_STARTUP_WAIT_SECS:-900}" \
  "$ROOT/target/release/cto-agent" wait-kleinhirn-startup
"$ROOT/target/release/cto-agent" check-kleinhirn
