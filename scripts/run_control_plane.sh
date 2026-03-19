#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

ENV_FILE="$ROOT/runtime/kleinhirn.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

if [ -x "$ROOT/target/release/cto-agent" ]; then
  exec "$ROOT/target/release/cto-agent" "$@"
fi

if [ -x "$ROOT/target/debug/cto-agent" ]; then
  exec "$ROOT/target/debug/cto-agent" "$@"
fi

exec "$ROOT/scripts/start_control_plane.sh" "$@"
