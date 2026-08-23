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

if [ -n "$(cargo tree --locked --manifest-path src/core/harness/Cargo.toml --target all -i rsa@0.9.10 2>/dev/null)" ]; then
  echo "security exception became reachable in the harness: rsa@0.9.10" >&2
  exit 1
fi

audit_args=(
  --deny unsound
  --ignore RUSTSEC-2026-0002
  --ignore RUSTSEC-2026-0253
)

# The XML and RSA exceptions are root-lockfile-only. Never extend them to a
# standalone component without a separate reachability proof.
cargo audit \
  --file Cargo.lock \
  "${audit_args[@]}" \
  --ignore RUSTSEC-2026-0194 \
  --ignore RUSTSEC-2026-0195 \
  --ignore RUSTSEC-2023-0071

while IFS= read -r lockfile; do
  [ "$lockfile" = "Cargo.lock" ] && continue
  if [ "$lockfile" = "src/core/harness/Cargo.lock" ]; then
    cargo audit \
      --no-fetch \
      --file "$lockfile" \
      "${audit_args[@]}" \
      --ignore RUSTSEC-2023-0071
  else
    cargo audit --no-fetch --file "$lockfile" "${audit_args[@]}"
  fi
done < <(git ls-files | awk '/(^|\/)Cargo[.]lock$/' | sort)
