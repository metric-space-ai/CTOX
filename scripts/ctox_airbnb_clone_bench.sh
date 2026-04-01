#!/usr/bin/env bash
set -euo pipefail

ROOT="${CTOX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" "$ROOT/scripts/ctox_airbnb_clone_bench.py" "$@"
