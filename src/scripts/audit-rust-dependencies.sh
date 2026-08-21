#!/usr/bin/env bash
set -euo pipefail

# cargo-audit evaluates every package recorded in Cargo.lock, including optional
# dependencies whose features are disabled. Keep the exceptions below valid
# only while neither package is reachable in any target's active dependency
# graph.
for package in quick-xml@0.39.4 rsa@0.9.10; do
  if [ -n "$(cargo tree --locked --target all -i "$package" 2>/dev/null)" ]; then
    echo "security exception became reachable: $package" >&2
    exit 1
  fi
done

for lockfile in Cargo.lock src/core/harness/Cargo.lock src/tools/sqlserver-mcp/Cargo.lock; do
  cargo audit \
    --file "$lockfile" \
    --deny unsound \
    --ignore RUSTSEC-2026-0194 \
    --ignore RUSTSEC-2026-0195 \
    --ignore RUSTSEC-2023-0071 \
    --ignore RUSTSEC-2026-0002 \
    --ignore RUSTSEC-2026-0097 \
    --ignore RUSTSEC-2026-0183 \
    --ignore RUSTSEC-2026-0184 \
    --ignore RUSTSEC-2026-0186 \
    --ignore RUSTSEC-2026-0190 \
    --ignore RUSTSEC-2026-0221 \
    --ignore RUSTSEC-2026-0253
done
