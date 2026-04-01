#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${1:?workdir required}"

while pgrep -af "cargo build --release --features cuda flash-attn cudnn" >/dev/null; do
  sleep 15
done

echo "BUILD_DONE"
stat -c "%y %n" "$WORKDIR/engine/candle/target/release/ctox-engine"
