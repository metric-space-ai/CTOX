#!/bin/bash
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
scratch=$(mktemp -d)
trap 'chmod -R u+w "$scratch" 2>/dev/null || true; find "$scratch" -depth -delete' EXIT
crate="$scratch/crate"
upstream="$scratch/upstream"
mkdir -p "$crate/pkg" "$upstream/pkg"

printf 'package pkg\n' > "$upstream/pkg/promoted.go"
git -C "$upstream" init -q
git -C "$upstream" add pkg/promoted.go
git -C "$upstream" -c user.name=test -c user.email=test@example.invalid commit -qm base

printf '%s\n' '// Candidate-Port-Status: ported' > "$crate/pkg/promoted.rs"
printf '%s\n' '// Candidate-Port-Status: adapted_to_ctox' > "$crate/pkg/staged.rs"
printf '%s\n' '// Port-Status: adapted_to_ctox' > "$crate/support.rs"

CLIPROXYAPI_CRATE_DIR="$crate" \
  "$script_dir/build_port_map.sh" "$upstream" "$scratch/map.json"

jq -e '
  .summary.go_files == 1 and
  .summary.ported_files == 1 and
  .summary.supplemental_files == 1 and
  .summary.candidate_staged_files == 1 and
  .files == [{
    upstream: "pkg/promoted.go",
    rust: "pkg/promoted.rs",
    status: "ported",
    test: false
  }] and
  .supplemental_files == [{rust: "support.rs", status: "adapted_to_ctox"}] and
  .candidate_files == [{rust: "pkg/staged.rs", status: "adapted_to_ctox"}]
' "$scratch/map.json" >/dev/null

printf 'build_port_map candidate overlay tests passed\n'
