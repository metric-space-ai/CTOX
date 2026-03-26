#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${1:?workdir required}"

cd "$WORKDIR"
rustup run 1.93.0-x86_64-unknown-linux-gnu cargo build --release
