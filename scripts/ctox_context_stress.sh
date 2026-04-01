#!/usr/bin/env bash
set -euo pipefail

ROOT="${CTOX_ROOT:-$(pwd)}"
DB_PATH="${1:-}"
CONVERSATION_ID="${2:-77}"
ITERATIONS="${3:-24}"
TOKEN_BUDGET="${4:-160}"

cleanup_db=0
if [[ -z "$DB_PATH" ]]; then
  DB_PATH="$(mktemp /tmp/ctox-context-stress-XXXXXX).db"
  cleanup_db=1
fi

cargo build --quiet --manifest-path "$ROOT/Cargo.toml"
CTOX_BIN="$ROOT/target/debug/ctox"

"$CTOX_BIN" context-stress "$DB_PATH" "$CONVERSATION_ID" "$ITERATIONS" "$TOKEN_BUDGET"

if [[ "$cleanup_db" -eq 1 ]]; then
  rm -f "$DB_PATH"
fi
