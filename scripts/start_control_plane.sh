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

for TOOLCHAIN_BIN in \
  "$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin" \
  "$HOME/.rustup/toolchains/stable-x86_64-apple-darwin/bin" \
  "$HOME/.rustup/toolchains/stable-aarch64-unknown-linux-gnu/bin" \
  "$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin"
do
  if [ -x "$TOOLCHAIN_BIN/cargo" ] && [ -x "$TOOLCHAIN_BIN/rustc" ]; then
    export PATH="$TOOLCHAIN_BIN:$HOME/.cargo/bin:$PATH"
    exec "$TOOLCHAIN_BIN/cargo" run -- "$@"
  fi
done

if command -v cargo >/dev/null 2>&1; then
  exec cargo run -- "$@"
fi

if command -v rustup >/dev/null 2>&1; then
  exec rustup run stable cargo run -- "$@"
fi

echo "cto-agent launcher could not find a built binary, cargo or rustup." >&2
exit 1
