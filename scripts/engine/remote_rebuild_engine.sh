#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${1:?workdir required}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.6}"

cd "$WORKDIR/engine/candle"
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export CUDA_INCLUDE_DIR="$CUDA_ROOT/targets/x86_64-linux/include"
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$CUDA_ROOT/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
rustup run 1.93.0-x86_64-unknown-linux-gnu cargo build --release --package ctox-engine-cli --bin ctox-engine --features "cuda flash-attn"
