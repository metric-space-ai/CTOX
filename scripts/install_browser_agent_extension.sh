#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
SOURCE="$ROOT/browser_agent/extension"
TARGET="$ROOT/runtime/browser-agent-extension"

if [ ! -f "$SOURCE/manifest.json" ]; then
  echo "browser-agent manifest missing at $SOURCE/manifest.json" >&2
  exit 1
fi

find "$SOURCE" \( -name '._*' -o -name '.DS_Store' \) -delete 2>/dev/null || true
rm -rf "$TARGET"
mkdir -p "$(dirname "$TARGET")"
ln -s "$SOURCE" "$TARGET"

printf 'Browser-Agent-Extension staged at:\n%s\n\n' "$TARGET"
printf 'Load this unpacked extension in Chrome and keep the CTO-Agent bridge running on http://127.0.0.1:8765.\n'
