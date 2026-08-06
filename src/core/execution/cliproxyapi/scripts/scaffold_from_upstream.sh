#!/bin/bash
set -eu

crate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
upstream_dir=${1:-"$crate_dir/../../../../../runtime/cliproxyapi-upstream"}

if [ ! -d "$upstream_dir/.git" ]; then
    echo "upstream checkout not found: $upstream_dir" >&2
    exit 1
fi

commit=$(git -C "$upstream_dir" rev-parse HEAD)
find "$upstream_dir" -type f -name '*.go' | while IFS= read -r source; do
    relative=${source#"$upstream_dir"/}
    target="$crate_dir/${relative%.go}.rs"
    if [ -e "$target" ]; then
        continue
    fi
    mkdir -p "$(dirname -- "$target")"
    {
        printf '// ref: %s @ %s\n' "$relative" "$commit"
        printf '// Port-Status: scaffold\n'
        printf '// License: MIT (upstream); modifications AGPL-3.0-only\n\n'
        printf '// Intentionally outside the Rust module graph until its signatures are ported.\n'
    } > "$target"
done
