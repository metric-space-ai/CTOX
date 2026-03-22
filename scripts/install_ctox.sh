#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CARGO_BIN="${CARGO_BIN:-}"
if [[ -z "$CARGO_BIN" ]]; then
  if [[ -x "$HOME/.cargo/bin/cargo" ]]; then
    CARGO_BIN="$HOME/.cargo/bin/cargo"
  else
    CARGO_BIN="$(command -v cargo || true)"
  fi
fi

if [[ -z "$CARGO_BIN" ]]; then
  echo "cargo is required to build CTOX" >&2
  exit 1
fi

cd "$ROOT"
"$CARGO_BIN" build --release

mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/ctox" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CTOX_ROOT="$ROOT"
exec "$ROOT/target/release/ctox" "\$@"
EOF
chmod +x "$HOME/.local/bin/ctox"

"$HOME/.local/bin/ctox" clean-room-bootstrap-deps >/dev/null

cat <<EOF
CTOX installed.

Binary:
  $HOME/.local/bin/ctox

Try:
  ctox clean-room-bootstrap-deps
  ctox clean-room-baseline-plan gpt_oss "Reply with CTOX_OK and nothing else."
EOF
